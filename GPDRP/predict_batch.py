import argparse
import pandas as pd
import torch
import torch.nn as nn
# 【重要】导入IterableDataset
from torch.utils.data import IterableDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import add_self_loops, remove_self_loops
import os
from rdkit import Chem
import numpy as np
import networkx as nx
import csv
import traceback
import math
from tqdm import tqdm
import gc

# --- 模型定义导入 (请确保路径正确) ---
try:
    from GIN.model.gin import GINConvNet as OriginalGINConvNet
    from GAT.model.gat import GATNet
    from GCN.model.gcn import GCNNet
    from GIN_TRANSFORMER.model.gintranformer import GINConvNet2
except ImportError:
    # ... 省略错误处理 ...
    pass

# --- 全局常量 ---
EXPECTED_ATOM_FEATURE_DIM = 78
DRUG_CHUNK_SIZE = 1000 
MODEL_TYPES = ['GIN', 'GAT', 'GCN', 'GINTransformer'] 

# --- 辅助函数 (保持不变) ---
# ... 此处省略所有未变的辅助函数 (one_of_k_encoding, atom_features_from_preprocessing, 等) ...
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        if isinstance(x, int) and allowable_set and isinstance(allowable_set[0], int):
            if x > allowable_set[-1]: x = allowable_set[-1]
            elif x < allowable_set[0]: x = allowable_set[0]
    return list(map(lambda s: x == s, allowable_set))
def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set: x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))
def atom_features_from_preprocessing(atom):
    allowable_symbols = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']
    allowable_degree = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    allowable_total_hs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    allowable_implicit_valence = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    try:
        features = np.array(one_of_k_encoding_unk(atom.GetSymbol(), allowable_symbols) + one_of_k_encoding(atom.GetDegree(), allowable_degree) + one_of_k_encoding_unk(atom.GetTotalNumHs(), allowable_total_hs) + one_of_k_encoding_unk(atom.GetImplicitValence(), allowable_implicit_valence) + [atom.GetIsAromatic()])
    except Exception: return np.zeros(EXPECTED_ATOM_FEATURE_DIM)
    return features

# --- 数据创建函数 (优化) ---
def smiles_to_drug_graph_parts(smiles_string):
    """仅从SMILES创建与药物相关的图部分，用于缓存，避免重复计算"""
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None or mol.GetNumAtoms() == 0: return None, None
    atom_f_list = [atom_features_from_preprocessing(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(np.array(atom_f_list), dtype=torch.float)
    if x.shape[0] > 0 and x.shape[1] != EXPECTED_ATOM_FEATURE_DIM: return None, None
    edges = [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in mol.GetBonds()]
    edge_index = torch.empty((2, 0), dtype=torch.long) if not edges else torch.tensor(np.array(list(nx.Graph(edges).to_directed().edges)).T, dtype=torch.long)
    return x, edge_index

# --- 加载数据和反归一化函数 (保持不变) ---
# ... 省略 load_training_gene_info_and_norm_params, load_and_preprocess_new_cell_lines, unscale_ic50 ...
def load_training_gene_info_and_norm_params(file_path):
    print(f"正在从 '{file_path}' 加载训练时使用的基因列表和归一化参数...")
    try:
        df = pd.read_csv(file_path, sep='\t')
        training_genes = df.iloc[:, 1].tolist()
        numeric_df = df.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')
        all_values = numeric_df.values.flatten()
        all_values = all_values[~np.isnan(all_values)]
        if len(all_values) == 0: raise ValueError("在文件中未找到任何有效的基因表达数值。")
        min_val, max_val = np.min(all_values), 12.0
        return training_genes, min_val, max_val
    except Exception as e: print(f"加载训练基因信息时出错: {e}"); return None, None, None
def load_and_preprocess_new_cell_lines(new_cell_file, training_genes, min_val, max_val):
    print(f"正在加载和预处理新的细胞系文件: '{new_cell_file}'...")
    try:
        df_new = pd.read_csv(new_cell_file, index_col=0)
        df_aligned = df_new.reindex(columns=training_genes, fill_value=0.0)
        X = np.clip(df_aligned.values.astype(np.float32), None, max_val)
        X_normalized = (X - min_val) / (max_val - min_val) if (max_val - min_val) != 0 else np.zeros_like(X)
        X_normalized = np.clip(X_normalized, 0.0, 1.0)
        # 直接返回torch tensor以提高效率
        return {name: torch.tensor(vector, dtype=torch.float).unsqueeze(0) for name, vector in zip(df_aligned.index, X_normalized)}
    except Exception as e: print(f"处理新细胞系文件时出错: {e}"); traceback.print_exc(); return None
def unscale_ic50(scaled_value_pred):
    epsilon = 1e-9
    if not isinstance(scaled_value_pred, (int, float, np.float32, np.float64)): return np.nan
    if np.isnan(scaled_value_pred): return np.nan
    y = np.clip(scaled_value_pred, epsilon, 1.0 - epsilon)
    try: original_value = -10.0 * math.log((1.0 - y) / y)
    except (ValueError, OverflowError): original_value = np.nan
    return original_value
# ========================================================================
#                          【核心性能优化区域】
# ========================================================================

class DrugCellPairIterableDataset(IterableDataset):
    """
    一个流式数据集，它在被请求时动态生成图数据，而不是一次性全部加载到内存。
    这极大地降低了内存消耗，并且能和DataLoader的多进程加载完美配合。
    """
    def __init__(self, drug_chunk, cell_features_map):
        super(DrugCellPairIterableDataset).__init__()
        self.drug_chunk = drug_chunk
        self.cell_features_map = cell_features_map

    def __iter__(self):
        for drug_name, smiles_str in self.drug_chunk:
            # 仅处理有效的SMILES
            if not isinstance(smiles_str, str) or not smiles_str.strip():
                continue
            
            # 【缓存优化】预先计算药物图的部分，避免在细胞循环中重复计算
            drug_x, drug_edge_index = smiles_to_drug_graph_parts(smiles_str)
            if drug_x is None:
                continue
                
            for cell_ge_tensor in self.cell_features_map.values():
                # 动态创建并返回一个完整的图数据对象
                # .clone()确保每个对象独立，防止多进程数据污染
                yield Data(x=drug_x.clone(), 
                           edge_index=drug_edge_index.clone(), 
                           target_ge=cell_ge_tensor.clone())

def predict_optimized(model, device, data_loader, model_type):
    """
    优化的预测函数，配合IterableDataset使用。
    """
    model.eval()
    total_preds = torch.Tensor()
    with torch.no_grad():
        # tqdm现在可以准确地显示批次的预测进度
        for data_batch in tqdm(data_loader, desc=f"  - Predicting Batches for Chunk"):
            data_batch = data_batch.to(device)

            if model_type == 'GAT':
                data_batch.edge_index, _ = remove_self_loops(data_batch.edge_index)
                data_batch.edge_index, _ = add_self_loops(data_batch.edge_index, num_nodes=data_batch.num_nodes)

            try:
                out, _ = model(data_batch)
                output = out
                if output.ndim == 1: output = output.unsqueeze(1)
            except Exception as e:
                print(f"\n  - Error during batch prediction: {e}")
                num_graphs = data_batch.num_graphs
                output = torch.full((num_graphs, 1), float('nan'), device=device)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
    return total_preds.numpy()

# ========================================================================
#                            主函数 MAIN
# ========================================================================
def main():
    parser = argparse.ArgumentParser(description='高性能GNN预测脚本')
    # ... 省略所有参数定义 ...
    parser.add_argument('--smiles_file', type=str, required=True, help='药物SMILES文件路径')
    parser.add_argument('--new_cell_line_file', type=str, required=True, help='新细胞系基因表达文件路径')
    parser.add_argument('--training_gene_expression_file', type=str, required=True, help='原始训练基因表达文件路径(exp.txt)')
    parser.add_argument('--models_dir', type=str, required=True, help='包含所有预训练模型的目录')
    parser.add_argument('--output_dir', type=str, required=True, help='保存预测结果的目录')
    parser.add_argument('--batch_size', type=int, default=256, help='【重要】预测时的批处理大小，可以设得比较大')
    parser.add_argument('--cuda_name', type=str, default="cuda:0", help='CUDA设备名称')
    # num_workers > 0 会启用多进程数据加载，极大提升速度。在Windows上如果出错可以设为0。
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader使用的工作进程数')

    args = parser.parse_args()
    print(f"脚本参数: {args}")

    # --- 1. 数据和模型准备 (一次性) ---
    os.makedirs(args.output_dir, exist_ok=True)
    training_genes, norm_min, norm_max = load_training_gene_info_and_norm_params(args.training_gene_expression_file)
    if not training_genes: return
    cell_features_map = load_and_preprocess_new_cell_lines(args.new_cell_line_file, training_genes, norm_min, norm_max)
    if not cell_features_map: return
    df_smiles = pd.read_csv(args.smiles_file, header=None, names=['drug_name', 'smiles'])
    smiles_input_list = df_smiles.values.tolist()
    num_drugs = len(smiles_input_list)
    
    model_map = {'GIN': OriginalGINConvNet, 'GAT': GATNet, 'GCN': GCNNet, 'GINTransformer': GINConvNet2}
    model_configs = [] # ... 省略模型扫描逻辑 ...
    if not os.path.isdir(args.models_dir): print(f"错误: 模型目录 '{args.models_dir}' 不存在。"); return
    for filename in os.listdir(args.models_dir):
        if filename.endswith(".model"):
            model_type_found = None
            for mtype in MODEL_TYPES:
                if f'_{mtype}_' in filename or filename.startswith(f'{mtype}_') or filename.startswith(f'model_{mtype}_'):
                    model_type_found = mtype; break
            if model_type_found: model_configs.append({'type': model_type_found, 'path': os.path.join(args.models_dir, filename)})
    
    device = torch.device(args.cuda_name if torch.cuda.is_available() and args.cuda_name != "cpu" else "cpu")
    print(f"使用设备: {device}")
    
    num_chunks = math.ceil(num_drugs / DRUG_CHUNK_SIZE)

    # --- 2. 外层模型循环 ---
    for config in model_configs:
        model_type = config['type']
        model_path = config['path']
        
        print(f"\n{'='*25}\n[检查模型]: {model_type}\n{'='*25}")
        
        all_chunks_done = all(os.path.exists(os.path.join(args.output_dir, f"predictions_{model_type}_chunk_{i+1}.csv")) for i in range(num_chunks)) if num_chunks > 0 else True
        if all_chunks_done:
            print(f"模型 {model_type} 的所有区块均已完成。跳过。")
            continue
        
        model_class = model_map.get(model_type)
        if not model_class: continue
        try:
            model_instance = model_class().to(device)
            model_instance.load_state_dict(torch.load(model_path, map_location=device))
        except Exception as e:
            print(f"加载模型 '{model_path}' 时出错: {e}"); continue
        
        # --- 3. 内层药物区块循环 ---
        for i in range(num_chunks):
            chunk_output_file = os.path.join(args.output_dir, f"predictions_{model_type}_chunk_{i+1}.csv")
            if os.path.exists(chunk_output_file):
                print(f"--- [模型: {model_type}] 块 {i+1}/{num_chunks} 已存在，跳过。 ---")
                continue

            chunk_start_idx = i * DRUG_CHUNK_SIZE
            chunk_end_idx = min((i + 1) * DRUG_CHUNK_SIZE, num_drugs)
            current_drug_chunk = smiles_input_list[chunk_start_idx:chunk_end_idx]
            
            print(f"\n--- [模型: {model_type}] 正在处理块 {i+1}/{num_chunks} ---")
            
            # 【性能优化步骤 1】: 创建轻量级的元信息列表，用于结果映射
            print("  - Step 1/3: 准备元数据...")
            all_pairs_info_for_chunk = []
            valid_drug_cell_pairs_count = 0
            for drug_name, smiles_str in current_drug_chunk:
                is_valid_smiles = isinstance(smiles_str, str) and smiles_str.strip() and (Chem.MolFromSmiles(smiles_str) is not None)
                for cell_name in cell_features_map.keys():
                    # 只有当SMILES有效时，我们才期望有预测结果
                    if is_valid_smiles:
                        all_pairs_info_for_chunk.append({'drug_name': drug_name, 'smiles': smiles_str, 'cell_line_name': cell_name, 'is_valid': True})
                        valid_drug_cell_pairs_count += 1
                    else: # 对于无效SMILES，直接记录为无效，后面填NaN
                        all_pairs_info_for_chunk.append({'drug_name': drug_name, 'smiles': smiles_str, 'cell_line_name': cell_name, 'is_valid': False})

            # 【性能优化步骤 2】: 实例化流式数据集和高效的DataLoader
            print(f"  - Step 2/3: 创建流式数据集和DataLoader (Batch size: {args.batch_size}, Workers: {args.num_workers})...")
            dataset = DrugCellPairIterableDataset(current_drug_chunk, cell_features_map)
            # persistent_workers=True (如果torch版本支持) 可以进一步减少开销
            loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True if device.type == 'cuda' else False)

            # 【性能优化步骤 3】: 对整个块进行一次性、高效率的预测
            print(f"  - Step 3/3: 对块内所有 {valid_drug_cell_pairs_count} 个有效样本进行预测...")
            scaled_predictions = np.array([])
            if valid_drug_cell_pairs_count > 0:
                try:
                    scaled_predictions = predict_optimized(model_instance, device, loader, model_type)
                except Exception as e_pred:
                    print(f"\n对块 {i+1} 预测时发生严重错误: {e_pred}")
                    scaled_predictions = np.full((valid_drug_cell_pairs_count, 1), np.nan)

            # --- 4. 整理并保存结果 ---
            print("  - Saving results...")
            chunk_results_list = []
            pred_idx = 0
            for info in all_pairs_info_for_chunk:
                result = {'drug_name': info['drug_name'], 'smiles': info['smiles'], 'cell_line_name': info['cell_line_name']}
                if info['is_valid']:
                    if pred_idx < len(scaled_predictions):
                        scaled_pred_val = float(scaled_predictions[pred_idx][0])
                        result['predicted_ic50_scaled'] = scaled_pred_val
                        result['predicted_ic50_original'] = unscale_ic50(scaled_pred_val)
                        pred_idx += 1
                    else: # 预测数组长度不足
                        result['predicted_ic50_scaled'], result['predicted_ic50_original'] = np.nan, np.nan
                else: # 无效SMILES
                    result['predicted_ic50_scaled'], result['predicted_ic50_original'] = np.nan, np.nan
                chunk_results_list.append(result)

            results_df = pd.DataFrame(chunk_results_list)
            cols_order = ['drug_name', 'smiles', 'cell_line_name', 'predicted_ic50_scaled', 'predicted_ic50_original']
            results_df = results_df.reindex(columns=cols_order)
            results_df.to_csv(chunk_output_file, index=False)
            print(f"模型 {model_type} 的块 {i+1} 结果已保存至 '{chunk_output_file}'")

            del loader, dataset, chunk_results_list, results_df, scaled_predictions, all_pairs_info_for_chunk; gc.collect()

        del model_instance; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    print(f"\n所有模型处理完毕！预测完成！")

if __name__ == "__main__":
    main()