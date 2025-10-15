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
# 【【【 新增/修改 】】】: 导入多进程和笛卡尔积工具
from multiprocessing import Pool, cpu_count
from itertools import product

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

# 为多进程工作者定义初始化函数
_fp_generator = None

def _init_worker(vocab_path, subword_map_path):
    """初始化每个工作进程，创建自己的指纹生成器实例。"""
    global _fp_generator
    # 在子进程中静默创建实例
    _fp_generator = DrugFingerprintGenerator(vocab_path=vocab_path, subword_map_path=subword_map_path, verbose=False)

def _process_drug_row(drug_info_tuple):
    """工作进程调用的函数，处理单个药物。"""
    global _fp_generator
    drug_name, smiles = drug_info_tuple
    fingerprints, success = _fp_generator.generate_all_fingerprints(smiles)
    return drug_name, smiles, fingerprints, success


# ==============================================================================
# 模块 1: 药物指纹生成器 (【【【 新增/修改 】】】: 保存路径)
# ==============================================================================
class DrugFingerprintGenerator:
    """封装 Morgan, PubChem, 和 ESPF 指纹生成逻辑。"""
    def __init__(self, vocab_path='./pre_process/drug_codes_chembl_freq_1500.txt',
                 subword_map_path='./pre_process/subword_units_map_chembl_freq_1500.csv', verbose=True):
        
        # 【【【 修正点 1: 保存路径为实例属性 】】】
        self.vocab_path = vocab_path
        self.subword_map_path = subword_map_path
        
        if verbose:
            print("\n--- 步骤 1: 正在初始化药物指纹生成器 ---")
        try:
            bpe_codes_drug = codecs.open(self.vocab_path, 'r', 'utf-8')
            self.dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
            sub_csv = pd.read_csv(self.subword_map_path)
            self.idx2word_d = sub_csv['index'].values
            self.words2idx_d = dict(zip(self.idx2word_d, range(0, len(self.idx2word_d))))
            self.morgan_dim = 2048
            self.pubchem_dim = 881
            self.espf_dim = len(self.idx2word_d)
            if verbose:
                print(f"  - ESPF生成器初始化成功 (词汇表大小: {self.espf_dim})")
                print(f"  - 指纹维度: Morgan={self.morgan_dim}, PubChem={self.pubchem_dim}, ESPF={self.espf_dim}")
        except FileNotFoundError as e:
            if verbose:
                print(f"致命错误：初始化指纹生成器失败，文件未找到: {e}")
            # 向上抛出异常，让主进程处理
            raise e

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

    def generate_all_fingerprints(self, smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                zeros_morgan = np.zeros(self.morgan_dim, dtype=np.float32)
                zeros_pubchem = np.zeros(self.pubchem_dim, dtype=np.float32)
                zeros_espf = np.zeros(self.espf_dim, dtype=np.float32)
                return (zeros_morgan, zeros_pubchem, zeros_espf), False

            morgan_fp = self.smiles_to_morgan(mol)
            pubchem_fp = self.smiles_to_pubchem(mol)
            espf_fp = self.smiles_to_espf(smiles)
            return (morgan_fp, pubchem_fp, espf_fp), True

        except Exception as e:
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
# 模块 3: 核心预测函数 (【【【 新增/修改 】】】: 修正获取路径的方式)
# ==============================================================================
def predict_drug_chunk(model, device, cell_data_tuple, drug_chunk_df, fp_generator,
                       drug_batch_size, cell_batch_size, num_workers, output_csv_base_path):
    model.eval()
    exp_df, mut_df, cnv_df = cell_data_tuple
    cell_lines_to_predict = exp_df.index.tolist()

    # --- 步骤 4a: 使用多进程并行计算指纹 ---
    print(f"\n--- 步骤 4a: 使用 {num_workers} 个CPU核心并行计算指纹...")
    drug_fingerprints_cache = {}
    valid_drugs_for_prediction = []
    unprocessed_drugs = []
    
    drug_info_list = [(row['DrugName'], str(row['SMILES'])) for _, row in drug_chunk_df.iterrows()]

    # 【【【 修正点 2: 从实例属性获取路径，而不是推断 】】】
    vocab_path = fp_generator.vocab_path
    subword_map_path = fp_generator.subword_map_path

    with Pool(processes=num_workers, initializer=_init_worker, initargs=(vocab_path, subword_map_path)) as pool:
        results_iterator = pool.imap_unordered(_process_drug_row, drug_info_list)
        
        for drug_name, smiles, (morgan_fp, pubchem_fp, espf_fp), success in tqdm(results_iterator, total=len(drug_info_list), desc="计算药物指纹"):
            if success:
                drug_fingerprints_cache[drug_name] = [morgan_fp, espf_fp, pubchem_fp]
                valid_drugs_for_prediction.append({'DrugName': drug_name})
            else:
                unprocessed_drugs.append({'DrugName': drug_name, 'SMILES': smiles, 'Reason': 'SMILES无效或处理失败'})

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
    print(f"  - 指纹计算完成！将对当前大块的 {num_valid_drugs} 种有效药物进行预测。")

    # --- 步骤 4b: 药物和细胞系双重分块预测 ---
    print(f"\n--- 步骤 4b: 开始双重批量预测 (药物批大小: {drug_batch_size}, 细胞批大小: {cell_batch_size}) ---")
    
    all_results_for_chunk = []
    
    cell_data_cache = {
        name: [
            exp_df.loc[name].values.astype(np.float32),
            mut_df.loc[name].values.astype(np.float32),
            cnv_df.loc[name].values.astype(np.float32)
        ] for name in cell_lines_to_predict
    }
    
    num_drug_batches = (num_valid_drugs + drug_batch_size - 1) // drug_batch_size
    
    with torch.no_grad():
        for i in tqdm(range(0, num_valid_drugs, drug_batch_size), total=num_drug_batches, desc="处理药物批次"):
            drug_batch_info = valid_drugs_for_prediction[i : i + drug_batch_size]
            drug_names_in_batch = [info['DrugName'] for info in drug_batch_info]
            
            drug_fp_batches = [[], [], []]
            for name in drug_names_in_batch:
                fps = drug_fingerprints_cache[name]
                for j in range(3):
                    drug_fp_batches[j].append(fps[j])
            
            drug_fp_tensors = [torch.tensor(np.array(batch)).to(device).float() for batch in drug_fp_batches]
            
            for j in range(0, len(cell_lines_to_predict), cell_batch_size):
                cell_names_in_batch = cell_lines_to_predict[j : j + cell_batch_size]
                
                cell_data_batches = [[], [], []]
                for name in cell_names_in_batch:
                    omics = cell_data_cache[name]
                    for k in range(3):
                        cell_data_batches[k].append(omics[k])

                cell_data_tensors = [torch.tensor(np.array(batch)).to(device).float() for batch in cell_data_batches]

                num_drugs = len(drug_names_in_batch)
                num_cells = len(cell_names_in_batch)
                
                expanded_drug_fps = [fp.repeat_interleave(num_cells, dim=0) for fp in drug_fp_tensors]
                expanded_cell_data = [data.repeat(num_drugs, 1) for data in cell_data_tensors]
                
                predictions, _ = model(expanded_drug_fps, expanded_cell_data)
                predictions_list = predictions.cpu().numpy().flatten().tolist()
                
                pair_labels = list(product(drug_names_in_batch, cell_names_in_batch))
                
                results_df = pd.DataFrame({
                    'CellLineID': [cell for _, cell in pair_labels],
                    'DrugName': [drug for drug, _ in pair_labels],
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
        description="【整合版 v10.1 - Bug修复】使用BANDRP模型进行分块预测，支持多核CPU指纹生成和细胞系分批。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--model_path', type=str, default='./github_upload/output_dir/db1/model.pt', help='预训练模型文件 (.pt) 的路径。')
    parser.add_argument('--exp_path', type=str, default='./CRC/gene_all.csv', help='新细胞系的【基因表达谱】文件路径。')
    parser.add_argument('--mut_path', type=str, default='./CRC/mu_all.csv', help='新细胞系的【基因突变谱】文件路径。')
    parser.add_argument('--cnv_path', type=str, default='./CRC/cnv_all.csv', help='新细胞系的【拷贝数变异谱】文件路径。')
    parser.add_argument('--new_drugs_csv', type=str, default='./CRC/predict_all_np.csv', help='包含所有待预测药物的CSV文件路径。')
    parser.add_argument('--output_csv', type=str, default='./predictions/prediction_results.csv', help='【输出文件基础名】。最终文件名会是 "基础名_1.csv", "基础名_2.csv" 等。')
    parser.add_argument('--drug_chunk_size', type=int, default=100000, help='每个药物大块的大小（即每个输出文件包含约多少种药物）。')
    parser.add_argument('--drug_batch_size', type=int, default=256, help='GPU一次性预测的小批量药物数量。')
    parser.add_argument('--cuda_id', type=int, default=0, help='要使用的GPU ID (-1为CPU)。')
    parser.add_argument('--start_chunk', type=int, default=0, help='指定从哪个大块编号开始运行。设置为0或不设置，则自动检测断点。')
    parser.add_argument('--cell_batch_size', type=int, default=32, help='GPU一次性预测的小批量细胞系数量。')
    parser.add_argument('--num_workers', type=int, default=max(1, cpu_count() // 2), help='用于生成药物指纹的CPU核心数。')

    args = parser.parse_args()

    cfg = get_cfg_defaults()
    
    if args.cuda_id >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.cuda_id}')
    else:
        if args.cuda_id >= 0:
            print(f"警告: 请求使用 CUDA:{args.cuda_id}，但CUDA不可用。将自动切换到 CPU。")
        device = torch.device('cpu')
        
    print(f"--- 使用设备: {device} ---")
    print(f"--- 配置: 大块={args.drug_chunk_size}药物/文件, 药物小批={args.drug_batch_size}, 细胞小批={args.cell_batch_size}, CPU核心={args.num_workers} ---")

    try:
        fp_generator = DrugFingerprintGenerator()
    except Exception as e:
        exit(1) # 如果主进程初始化失败，直接退出

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
            fp_generator, args.drug_batch_size, args.cell_batch_size, args.num_workers, output_base
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