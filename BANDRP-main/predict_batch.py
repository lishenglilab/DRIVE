import os
import argparse
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from functools import reduce
import warnings
import codecs
import re

# --- 导入 RDKit 和指纹相关库 ---
try:
    from rdkit import Chem, rdBase
    from rdkit.Chem import AllChem
    from PyBioMed.PyMolecule.PubChemFingerprints import calcPubChemFingerAll
    from subword_nmt.apply_bpe import BPE
except ImportError as e:
    print(f"错误：缺少必要的指紋庫: {e}")
    print("请运行 'pip install rdkit-pypi PyBioMed subword-nmt' 来安装依赖。")
    exit(1)

# 抑制 RDKit 的冗余日志
rdBase.DisableLog('rdApp.*')

# --- 导入项目模块 ---
try:
    from config import get_cfg_defaults
    from model import BANDRP
except ImportError as e:
    print(f"错误：导入项目模块 (config, model) 失败: {e}")
    print("请确保 config.py, model.py, 和 BAN.py 文件与此脚本位于同一目录，或在Python可搜索的路径中。")
    exit(1)


# ==============================================================================
# 模块 1: 药物指纹生成器 (【【【 已修改，增加崩溃防护 】】】)
# ==============================================================================
class DrugFingerprintGenerator:
    """封装 Morgan, PubChem, 和 ESPF 指纹生成逻辑。"""
    def __init__(self, vocab_path='./pre_process/drug_codes_chembl_freq_1500.txt',
                 subword_map_path='./pre_process/subword_units_map_chembl_freq_1500.csv'):
        print("\n--- 步骤 1: 正在初始化药物指纹生成器 ---")
        try:
            bpe_codes_drug = codecs.open(vocab_path, 'r', 'utf-8')
            self.dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
            sub_csv = pd.read_csv(subword_map_path)
            self.idx2word_d = sub_csv['index'].values
            self.words2idx_d = dict(zip(self.idx2word_d, range(0, len(self.idx2word_d))))
            self.morgan_dim = 2048
            self.pubchem_dim = 881
            self.espf_dim = len(self.idx2word_d)
            print(f"  - ESPF生成器初始化成功 (词汇表大小: {self.espf_dim})")
            print(f"  - 指纹维度: Morgan={self.morgan_dim}, PubChem={self.pubchem_dim}, ESPF={self.espf_dim}")
        except FileNotFoundError as e:
            print(f"致命错误：初始化指纹生成器失败，文件未找到: {e}")
            exit(1)

    def smiles_to_espf(self, smiles):
        try:
            t1 = self.dbpe.process_line(smiles).split()
            i1 = np.asarray([self.words2idx_d.get(i) for i in t1 if i in self.words2idx_d])
            if i1.size == 0: return np.zeros(self.espf_dim, dtype=np.float32)
            v1 = np.zeros(self.espf_dim, dtype=np.float32)
            v1[i1] = 1
            return v1
        except Exception:
            return np.zeros(self.espf_dim, dtype=np.float32)

    def smiles_to_morgan(self, mol):
        if mol is None: return np.zeros(self.morgan_dim, dtype=np.float32)
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=self.morgan_dim)
            return np.array(fp, dtype=np.float32)
        except Exception:
            return np.zeros(self.morgan_dim, dtype=np.float32)

    def smiles_to_pubchem(self, mol):
        if mol is None: return np.zeros(self.pubchem_dim, dtype=np.float32)
        try:
            return np.array(calcPubChemFingerAll(mol), dtype=np.float32)
        except Exception:
            return np.zeros(self.pubchem_dim, dtype=np.float32)

    # --- 【【【 核心修改点 】】】 ---
    def generate_all_fingerprints(self, smiles):
        """
        生成所有指纹，并包含一个顶级的异常捕获块来防止程序崩溃。
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                # 这种情况是合法的SMILES解析失败，不算崩溃
                zeros_morgan = np.zeros(self.morgan_dim, dtype=np.float32)
                zeros_pubchem = np.zeros(self.pubchem_dim, dtype=np.float32)
                zeros_espf = np.zeros(self.espf_dim, dtype=np.float32)
                return (zeros_morgan, zeros_pubchem, zeros_espf), False

            morgan_fp = self.smiles_to_morgan(mol)
            pubchem_fp = self.smiles_to_pubchem(mol)
            espf_fp = self.smiles_to_espf(smiles)
            return (morgan_fp, pubchem_fp, espf_fp), True

        except Exception as e:
            # 捕获所有其他异常，包括可能由RDKit底层引起的错误
            # 虽然不能直接捕获Segmentation Fault，但这能捕获其前兆或相关Python层面的错误
            # 这是一个防御性措施，旨在提高程序的鲁棒性
            print(f"\n[严重警告] 处理SMILES时发生严重错误: '{smiles[:100]}...'. 错误: {e}. 将跳过此药物。")
            zeros_morgan = np.zeros(self.morgan_dim, dtype=np.float32)
            zeros_pubchem = np.zeros(self.pubchem_dim, dtype=np.float32)
            zeros_espf = np.zeros(self.espf_dim, dtype=np.float32)
            return (zeros_morgan, zeros_pubchem, zeros_espf), False


# ==============================================================================
# 模块 2: 细胞系特征处理 (无变化)
# ==============================================================================
def load_and_align_multiple_cell_features(cfg, new_exp_path, new_mut_path, new_cnv_path):
    print("\n--- 步骤 2: 正在加载并对齐新的细胞系特征 ---")
    try:
        print("  - 正在从config.py指定的路径中读取参考基因列表以进行对齐...")
        ref_exp_cols = pd.read_csv(cfg.path.expression, index_col=0, nrows=0).columns.tolist()
        ref_mut_cols = pd.read_csv(cfg.path.mutation, index_col=0, nrows=0).columns.tolist()
        ref_cnv_cols = pd.read_csv(cfg.path.cnv, index_col=0, nrows=0).columns.tolist()
        cell_exp_dim = len(ref_exp_cols)
        cell_mut_dim = len(ref_mut_cols)
        cell_cnv_dim = len(ref_cnv_cols)
        print(f"  - 参考特征维度: EXP={cell_exp_dim}, MUT={cell_mut_dim}, CNV={cell_cnv_dim}")
        print("  - 正在加载新细胞系的组学数据...")
        exp_sample_df = pd.read_csv(new_exp_path, index_col=0)
        mut_sample_df = pd.read_csv(new_mut_path, index_col=0)
        cnv_sample_df = pd.read_csv(new_cnv_path, index_col=0)
        print(f"  - 原始样本数量: EXP={len(exp_sample_df)}, MUT={len(mut_sample_df)}, CNV={len(cnv_sample_df)}")
        common_cell_ids = sorted(list(
            reduce(lambda x, y: x.intersection(y),
                   [set(exp_sample_df.index), set(mut_sample_df.index), set(cnv_sample_df.index)])
        ))
        if not common_cell_ids:
            print("\n错误：在提供的三个组学文件中没有找到任何共有的细胞系ID。无法进行预测。")
            exit(1)
        print(f"  - 检测到 {len(common_cell_ids)} 个共有的待预测细胞系: {common_cell_ids[:5]}...")
        exp_common_df = exp_sample_df.loc[common_cell_ids]
        mut_common_df = mut_sample_df.loc[common_cell_ids]
        cnv_common_df = cnv_sample_df.loc[common_cell_ids]
        print("  - 正在将基因特征与参考列表进行对齐...")
        exp_aligned_df = pd.DataFrame(0.0, index=exp_common_df.index, columns=ref_exp_cols)
        mut_aligned_df = pd.DataFrame(0.0, index=mut_common_df.index, columns=ref_mut_cols)
        cnv_aligned_df = pd.DataFrame(0.0, index=cnv_common_df.index, columns=ref_cnv_cols)
        exp_aligned_df.update(exp_common_df)
        mut_aligned_df.update(mut_common_df)
        cnv_aligned_df.update(cnv_common_df)
        print("  - 新的细胞系特征已成功对齐！")
        return exp_aligned_df, mut_aligned_df, cnv_aligned_df, cell_exp_dim, cell_mut_dim, cell_cnv_dim
    except FileNotFoundError as e:
        print(f"\n错误：参考或输入文件未找到: {e}。")
        exit(1)
    except Exception as e:
        print(f"\n错误：处理细胞系特征时发生意外: {e}。请检查文件格式是否正确（行=细胞系ID，列=基因）。")
        exit(1)


# ==============================================================================
# 模块 3: 核心预测函数 (无变化)
# ==============================================================================
def predict_drug_chunk(model, device, cell_data_tuple, drug_chunk_df, fp_generator,
                       small_batch_size, output_csv_base_path):
    model.eval()
    exp_df, mut_df, cnv_df = cell_data_tuple
    cell_lines_to_predict = exp_df.index.tolist()

    print("\n--- 步骤 4a: 正在为当前大块的药物计算指纹并验证SMILES... ---")
    drug_fingerprints_cache = {}
    valid_drugs_for_prediction = []
    unprocessed_drugs = []

    for _, row in tqdm(drug_chunk_df.iterrows(), total=len(drug_chunk_df), desc="验证SMILES并计算指纹"):
        drug_name = row['DrugName']
        smiles = str(row['SMILES'])
        (morgan_fp, pubchem_fp, espf_fp), success = fp_generator.generate_all_fingerprints(smiles)
        
        reason = 'SMILES无效或无法解析'
        if not success:
            # 检查是否是由于严重错误被捕获
            if "严重警告" in locals().get('__warningregistry__', {}):
                 reason = '处理时发生严重错误 (可能导致崩溃)'

            unprocessed_drugs.append({'DrugName': drug_name, 'SMILES': smiles, 'Reason': reason})
            continue

        drug_fingerprints_cache[drug_name] = [morgan_fp, espf_fp, pubchem_fp]
        valid_drugs_for_prediction.append({'DrugName': drug_name})

    if unprocessed_drugs:
        try:
            unprocessed_df = pd.DataFrame(unprocessed_drugs)
            unprocessed_csv_log_path = f"{output_csv_base_path}_unprocessed_drugs_log.csv"
            unprocessed_df.to_csv(unprocessed_csv_log_path, mode='a', header=not os.path.exists(unprocessed_csv_log_path), index=False, encoding='utf-8-sig')
            print(f"\n警告：当前大块中有 {len(unprocessed_df)} 种药物因SMILES无效或处理失败而跳过。")
            print(f"  - 详细列表已追加至: {os.path.abspath(unprocessed_csv_log_path)}")
        except Exception as e:
            print(f"\n错误：保存无法处理的药物列表失败: {e}")

    if not valid_drugs_for_prediction:
        print("\n\n警告：当前大块经过验证后，没有找到任何具有有效SMILES的药物。跳过此大块。")
        return None

    num_valid_drugs = len(valid_drugs_for_prediction)
    print(f"  - 指纹验证完成！将对当前大块的 {num_valid_drugs} 种有效药物进行预测。")

    print(f"\n--- 步骤 4b: 开始小批量预测，每个小批量 {small_batch_size} 种药物 ---")
    
    num_batches = (num_valid_drugs + small_batch_size - 1) // small_batch_size
    batch_iterator = tqdm(range(0, num_valid_drugs, small_batch_size), total=num_batches, desc="处理药物小批量")

    all_results_for_chunk = []

    for i in batch_iterator:
        start_idx = i
        end_idx = min(i + small_batch_size, num_valid_drugs)
        current_drug_batch_info = valid_drugs_for_prediction[start_idx:end_idx]

        drug_fp_batches = [[], [], []]
        drug_names_in_order = []
        for drug_info in current_drug_batch_info:
            drug_name = drug_info['DrugName']
            drug_names_in_order.append(drug_name)
            fps = drug_fingerprints_cache[drug_name]
            for j in range(3):
                drug_fp_batches[j].append(fps[j])

        drug_fp_tensors = [torch.tensor(np.array(batch)).to(device).float() for batch in drug_fp_batches]
        num_drugs_in_batch = len(drug_names_in_order)

        with torch.no_grad():
            for cell_name in cell_lines_to_predict:
                exp_vec = exp_df.loc[cell_name].values.astype(np.float32)
                mut_vec = mut_df.loc[cell_name].values.astype(np.float32)
                cnv_vec = cnv_df.loc[cell_name].values.astype(np.float32)

                exp_batch = torch.from_numpy(exp_vec).unsqueeze(0).repeat(num_drugs_in_batch, 1).to(device)
                mut_batch = torch.from_numpy(mut_vec).unsqueeze(0).repeat(num_drugs_in_batch, 1).to(device)
                cnv_batch = torch.from_numpy(cnv_vec).unsqueeze(0).repeat(num_drugs_in_batch, 1).to(device)
                cell_data_batch = [exp_batch, mut_batch, cnv_batch]

                predictions, _ = model(drug_fp_tensors, cell_data_batch)
                predictions_list = predictions.cpu().numpy().flatten().tolist()

                results_df = pd.DataFrame({
                    'CellLineID': cell_name,
                    'DrugName': drug_names_in_order,
                    'PredictedValue': predictions_list
                })
                all_results_for_chunk.append(results_df)

    if all_results_for_chunk:
        return pd.concat(all_results_for_chunk, ignore_index=True)
    else:
        return None

# ==============================================================================
# 主程序入口 (无变化)
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="【整合版 v9 - 断点续跑】使用BANDRP模型进行分块预测，支持从中断处继续。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--model_path', type=str, default='./github_upload/output_dir/db1/model.pt',
                        help='预训练模型文件 (.pt) 的路径。')
    parser.add_argument('--exp_path', type=str, default='./test/exp_1.csv',
                        help='新细胞系的【基因表达谱】文件路径。')
    parser.add_argument('--mut_path', type=str, default='./test/mu_1.csv',
                        help='新细胞系的【基因突变谱】文件路径。')
    parser.add_argument('--cnv_path', type=str, default='./test/cnv_1.csv',
                        help='新细胞系的【拷贝数变异谱】文件路径。')
    parser.add_argument('--new_drugs_csv', type=str, default='./test/predict_all_np.csv', help='包含所有待预测药物的CSV文件路径。')
    parser.add_argument('--output_csv', type=str, default='./predictions/prediction_results.csv',
                        help='【输出文件基础名】。最终文件名会是 "基础名_1.csv", "基础名_2.csv" 等。')
    parser.add_argument('--drug_chunk_size', type=int, default=100000,
                        help='每个药物大块的大小（即每个输出文件包含约多少种药物）。')
    parser.add_argument('--drug_batch_size', type=int, default=1000,
                        help='GPU一次性预测的小批量药物数量。')
    parser.add_argument('--cuda_id', type=int, default=0, help='要使用的GPU ID (-1为CPU)。')
    parser.add_argument('--start_chunk', type=int, default=0,
                        help='指定从哪个大块编号开始运行。设置为0或不设置，则自动检测断点。')
    
    args = parser.parse_args()

    # --- 主逻辑 ---
    cfg = get_cfg_defaults()
    device = torch.device(f'cuda:{args.cuda_id}' if args.cuda_id >= 0 and torch.cuda.is_available() else 'cpu')
    print(f"--- 使用设备: {device} ---")
    print(f"--- 配置: 大块={args.drug_chunk_size}药物/文件, 小批量={args.drug_batch_size}药物/GPU预测 ---")

    fp_generator = DrugFingerprintGenerator()

    exp_df, mut_df, cnv_df, exp_dim, mut_dim, cnv_dim = load_and_align_multiple_cell_features(
        cfg, args.exp_path, args.mut_path, args.cnv_path
    )

    print("\n--- 步骤 3: 正在加载预训练模型 ---")
    model = BANDRP(cell_exp_dim=exp_dim, cell_mut_dim=mut_dim, cell_cnv_dim=cnv_dim, **cfg).to(device)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"  - 模型状态已从 {args.model_path} 成功加载。")
    except Exception as e:
        print(f"错误: 加载模型状态失败: {e}")
        exit(1)

    print("\n--- 步骤 4: 开始分大块处理药物文件 ---")
    try:
        chunk_iterator = pd.read_csv(
            args.new_drugs_csv, header=None, names=['DrugName', 'SMILES'],
            chunksize=args.drug_chunk_size, low_memory=True, engine='c'
        )
    except FileNotFoundError:
        print(f"错误: 药物文件未找到: {args.new_drugs_csv}")
        exit(1)
    
    output_base, output_ext = os.path.splitext(args.output_csv)
    generated_files = []

    start_from_chunk = args.start_chunk
    if start_from_chunk <= 0:
        output_dir = os.path.dirname(output_base)
        if not output_dir: output_dir = '.'
        
        existing_files = [f for f in os.listdir(output_dir) if f.startswith(os.path.basename(output_base) + "_") and f.endswith(output_ext)]
        
        last_completed_chunk = 0
        for f in existing_files:
            try:
                num_str = re.search(r'_(\d+)\.', f)
                if num_str:
                    chunk_num_found = int(num_str.group(1))
                    if chunk_num_found > last_completed_chunk:
                        last_completed_chunk = chunk_num_found
            except (ValueError, IndexError):
                continue
        
        start_from_chunk = last_completed_chunk + 1
        print(f"\n--- 自动检测到上次已完成到大块 {last_completed_chunk}。将从大块 {start_from_chunk} 开始继续... ---")
    else:
        print(f"\n--- 用户指定从大块 {start_from_chunk} 开始运行... ---")
        
    for chunk_num, drug_chunk_df in enumerate(chunk_iterator, 1):
        if chunk_num < start_from_chunk:
            print(f"快速跳过已处理的大块 {chunk_num}...")
            continue

        print(f"\n==================== 正在处理大块 {chunk_num} ====================")
        
        drug_chunk_df.dropna(subset=['SMILES'], inplace=True)
        drug_chunk_df = drug_chunk_df[drug_chunk_df['SMILES'].apply(lambda x: isinstance(x, str) and len(x.strip()) > 0)]
        if drug_chunk_df.empty:
            print("当前大块在初步过滤后为空，跳过。")
            continue
        
        chunk_result_df = predict_drug_chunk(
            model, device, (exp_df, mut_df, cnv_df), drug_chunk_df,
            fp_generator, args.drug_batch_size, output_base
        )

        if chunk_result_df is not None and not chunk_result_df.empty:
            chunk_output_path = f"{output_base}_{chunk_num}{output_ext}"
            try:
                output_dir = os.path.dirname(chunk_output_path)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                chunk_result_df.to_csv(chunk_output_path, index=False, encoding='utf-8-sig')
                generated_files.append(chunk_output_path)
                print(f"\n--- 大块 {chunk_num} 的预测结果已成功保存至: {os.path.abspath(chunk_output_path)} ---")
            except Exception as e:
                print(f"\n[错误] 保存大块 {chunk_num} 的结果文件失败: {e}")
        else:
            print(f"--- 大块 {chunk_num} 未生成任何有效的预测结果。 ---")

    print("\n\n==================== 所有预测任务完成！ ====================")
    if generated_files:
        print(f"本次运行新生成了 {len(generated_files)} 个结果文件:")
        for f_path in generated_files:
            print(f"  - {os.path.abspath(f_path)}")
    else:
        print("本次运行没有新生成任何预测结果文件。")