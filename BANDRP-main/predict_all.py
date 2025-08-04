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
# 模块 1: 药物指纹生成器 (保持不变)
# ==============================================================================
class DrugFingerprintGenerator:
    """封装 Morgan, PubChem, 和 ESPF 指纹生成逻辑。"""

    # ... 此部分代码与之前版本完全相同，此处省略以保持简洁 ...
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

    def generate_all_fingerprints(self, smiles):
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


# ==============================================================================
# 模块 2: 细胞系特征处理 (保持不变)
# ==============================================================================
def load_and_align_multiple_cell_features(cfg, new_exp_path, new_mut_path, new_cnv_path):
    # ... 此部分代码与之前版本完全相同，此处省略以保持简洁 ...
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
# 模块 3: 核心预测函数 (【【【 已修改为分文件保存 】】】)
# ==============================================================================
def predict_matrix(model, device, cell_data_tuple, drugs_df, fp_generator,
                   output_csv_path, drug_batch_size):
    """
    为一组细胞系和药物执行矩阵预测。
    此版本将结果分批保存到带有编号的独立文件中。
    """
    model.eval()
    exp_df, mut_df, cnv_df = cell_data_tuple
    cell_lines_to_predict = exp_df.index.tolist()

    # --- 步骤 4a: 预计算所有药物的指纹并进行验证 ---
    print("\n--- 步骤 4a: 正在为所有药物预计算指纹并进行验证... ---")
    drug_fingerprints_cache = {}
    valid_drugs_for_prediction = []
    unprocessed_drugs = []

    for _, row in tqdm(drugs_df.iterrows(), total=len(drugs_df), desc="验证SMILES并计算指纹"):
        drug_name = row['DrugName']
        smiles = str(row['SMILES'])
        (morgan_fp, pubchem_fp, espf_fp), success = fp_generator.generate_all_fingerprints(smiles)
        if not success:
            unprocessed_drugs.append(
                {'DrugName': drug_name, 'SMILES': smiles, 'Reason': 'Invalid or unparsable SMILES string'})
            continue
        drug_fingerprints_cache[drug_name] = [morgan_fp, espf_fp, pubchem_fp]
        valid_drugs_for_prediction.append({'DrugName': drug_name})

    # 保存无法处理的药物日志
    if unprocessed_drugs:
        try:
            unprocessed_df = pd.DataFrame(unprocessed_drugs)
            base, ext = os.path.splitext(output_csv_path)
            unprocessed_csv_log_path = f"{base}_unprocessed_drugs.csv"
            unprocessed_df.to_csv(unprocessed_csv_log_path, index=False, encoding='utf-8-sig')
            print(f"\n警告：有 {len(unprocessed_df)} 种药物因SMILES无效而无法处理。")
            print(f"  - 详细列表已保存至: {os.path.abspath(unprocessed_csv_log_path)}")
        except Exception as e:
            print(f"\n错误：保存无法处理的药物列表失败: {e}")

    if not valid_drugs_for_prediction:
        print("\n\n致命错误：经过验证后，没有找到任何具有有效SMILES的药物可用于预测。")
        return 0, []  # 返回处理数量和文件列表

    num_valid_drugs = len(valid_drugs_for_prediction)
    print(f"  - 指纹验证完成！将对 {num_valid_drugs} 种有效药物进行预测。")

    # --- 【【【 核心改动：分文件保存逻辑 】】】 ---
    print(f"\n--- 步骤 4b: 开始分批次预测，每批最多 {drug_batch_size} 种药物，结果将分文件保存 ---")

    # 从用户提供的输出路径中提取基本名称和扩展名，用于生成带编号的文件名
    output_base, output_ext = os.path.splitext(output_csv_path)

    num_batches = (num_valid_drugs + drug_batch_size - 1) // drug_batch_size
    batch_iterator = tqdm(range(0, num_valid_drugs, drug_batch_size), total=num_batches, desc="处理药物批次")

    generated_files = []  # 用于存储所有生成的文件名

    for batch_num, i in enumerate(batch_iterator, 1):  # 使用 enumerate 获取批次编号 (从1开始)
        start_idx = i
        end_idx = min(i + drug_batch_size, num_valid_drugs)
        current_drug_batch_info = valid_drugs_for_prediction[start_idx:end_idx]

        # 准备当前药物批次的指纹张量
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

        # 用于存储当前批次所有预测结果的列表
        all_predictions_for_this_batch = []

        cell_iterator = tqdm(cell_lines_to_predict, desc=f"预测细胞系 (批次 {batch_num}/{num_batches})", leave=False)
        with torch.no_grad():
            for cell_name in cell_iterator:
                # ... (预测单个细胞系的代码保持不变) ...
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
                all_predictions_for_this_batch.append(results_df)

        # 在处理完一个批次的所有细胞系后，将该批次的结果合并并保存到独立文件
        if all_predictions_for_this_batch:
            # 合并当前批次的所有结果
            batch_results_df = pd.concat(all_predictions_for_this_batch, ignore_index=True)

            # 生成带编号的输出文件名
            batch_output_path = f"{output_base}_part_{batch_num}{output_ext}"
            generated_files.append(batch_output_path)

            # 将批次结果保存到新文件中
            batch_results_df.to_csv(batch_output_path, index=False, encoding='utf-8-sig')

            batch_iterator.set_postfix_str(
                f"批次 {batch_num}/{num_batches} 已保存至 {os.path.basename(batch_output_path)}")

    # 返回成功处理的药物总数和所有生成的文件列表
    return num_valid_drugs, generated_files


# ==============================================================================
# 主程序入口 (【【【 已修改以处理分文件输出 】】】)
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="【整合版 v7】使用BANDRP模型预测，并将结果分批保存到独立文件中。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # --- 参数定义 (保持不变) ---
    parser.add_argument('--model_path', type=str, default='./github_upload/output_dir/db1/model.pt',
                        help='预训练模型文件 (.pt) 的路径。')
    parser.add_argument('--exp_path', type=str, default='./depmap/gene_depmap.csv',
                        help='新细胞系的【基因表达谱】文件路径。')
    parser.add_argument('--mut_path', type=str, default='./depmap/mu_depmap.csv',
                        help='新细胞系的【基因突变谱】文件路径。')
    parser.add_argument('--cnv_path', type=str, default='./depmap/cnv_depmap.csv',
                        help='新细胞系的【拷贝数变异谱】文件路径。')
    parser.add_argument('--new_drugs_csv', type=str, default='./depmap/drug_results.csv', help='新药物的CSV文件路径。')
    parser.add_argument('--output_csv', type=str, default='./prediction_output_tetst.csv',
                        help='【输出文件基础名】。最终文件名会是 "基础名_part_N.csv"。')
    parser.add_argument('--drug_batch_size', type=int, default=50000,
                        help='每批次处理的药物数量，也是每个输出文件包含的药物量。')
    parser.add_argument('--cuda_id', type=int, default=0, help='要使用的GPU ID (-1为CPU)。')
    args = parser.parse_args()

    # --- 主逻辑 ---
    cfg = get_cfg_defaults()
    device = torch.device(f'cuda:{args.cuda_id}' if args.cuda_id >= 0 and torch.cuda.is_available() else 'cpu')
    print(f"--- 使用设备: {device} ---")

    fp_generator = DrugFingerprintGenerator()

    exp_df, mut_df, cnv_df, exp_dim, mut_dim, cnv_dim = load_and_align_multiple_cell_features(
        cfg, args.exp_path, args.mut_path, args.cnv_path
    )

    print("\n--- 步骤 3: 正在加载并预处理新的药物数据 ---")
    try:
        # ... 数据加载和预处理逻辑不变 ...
        drugs_to_predict_df = pd.read_csv(args.new_drugs_csv, header=None, names=['DrugName', 'SMILES'])
        initial_count = len(drugs_to_predict_df)
        print(f"  - 从 {args.new_drugs_csv} 原始加载了 {initial_count} 行数据。")
        drugs_to_predict_df.dropna(subset=['SMILES'], inplace=True)
        drugs_to_predict_df = drugs_to_predict_df[
            drugs_to_predict_df['SMILES'].apply(lambda x: isinstance(x, str) and len(x.strip()) > 0)]
        invalid_smiles_mask = drugs_to_predict_df['SMILES'].str.startswith('错误')
        drugs_to_predict_df = drugs_to_predict_df[~invalid_smiles_mask]
        filtered_count = len(drugs_to_predict_df)
        if filtered_count < initial_count:
            print(f"  - 注意: 初步过滤掉了 {initial_count - filtered_count} 行含有空值或明显错误文本的SMILES。")
        if drugs_to_predict_df.empty:
            print(f"错误: 在 {args.new_drugs_csv} 中没有找到任何可能有效的药物条目。")
            exit(1)
        print(f"  - 初步筛选后，剩余 {filtered_count} 种药物进入详细验证阶段。")
    except FileNotFoundError:
        print(f"错误: 药物文件未找到: {args.new_drugs_csv}")
        exit(1)

    print("\n--- 步骤 4: 正在加载预训练模型 ---")
    model = BANDRP(cell_exp_dim=exp_dim, cell_mut_dim=mut_dim, cell_cnv_dim=cnv_dim, **cfg).to(device)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"  - 模型状态已从 {args.model_path} 成功加载。")
    except Exception as e:
        print(f"错误: 加载模型状态失败: {e}")
        exit(1)

    # 调用修改后的预测函数，它现在返回处理的药物数和生成的文件列表
    num_processed_drugs, generated_files = predict_matrix(
        model,
        device,
        (exp_df, mut_df, cnv_df),
        drugs_to_predict_df,
        fp_generator,
        args.output_csv,
        args.drug_batch_size
    )

    # 【【【 修改后的最终输出信息 】】】
    if num_processed_drugs > 0:
        print(f"\n--- 所有预测完成！---")
        print(f"对 {len(exp_df)} 个细胞系和 {num_processed_drugs} 种有效药物的交叉预测结果已分批保存。")
        print(f"共生成了 {len(generated_files)} 个文件:")
        for f_path in generated_files:
            print(f"  - {os.path.abspath(f_path)}")
    else:
        print("\n--- 未生成任何预测结果。请检查输入文件和运行过程中的警告/错误信息。 ---")