# 文件名: predict_robust.py (最终集成版)
import argparse
import csv
import gc
import math
import os
import pickle
import re
import sys

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# --- 模型定义 (请确保这些文件在您的项目路径中) ---
try:
    from models.gat import GATNet
    from models.gat_gcn import GAT_GCN
    from models.gcn import GCNNet
    from models.ginconv import GINConvNet
except ImportError as e:
    print(f"错误: 无法导入模型定义。请确保 'models' 文件夹及其中的 .py 文件存在。\n{e}", file=sys.stderr)
    sys.exit(1)

# --- 辅助函数和类定义部分 ---

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set: raise Exception(f"Input {x} not in allowable set {allowable_set}")
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set: x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None: return None, None, None
    c_size = mol.GetNumAtoms()
    if c_size == 0: return None, None, None
    features = [atom_features(atom) for atom in mol.GetAtoms()]
    edges = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()]
    g = nx.Graph(edges).to_directed()
    edge_index_list = [[e1, e2] for e1, e2 in g.edges()]
    if edge_index_list:
        return c_size, features, np.array(edge_index_list, dtype=np.int64).T
    else:
        return c_size, features, np.empty((2, 0), dtype=np.int64)

class PredictionDataset(Dataset):
    def __init__(self, drug_cell_pairs, smile_graph_dict, aligned_cell_features_dict):
        super(PredictionDataset, self).__init__()
        self.drug_cell_pairs = drug_cell_pairs
        self.smile_graph_dict = smile_graph_dict
        self.aligned_cell_features_dict = aligned_cell_features_dict

    def len(self):
        return len(self.drug_cell_pairs)

    def get(self, idx):
        smi, cell_name = self.drug_cell_pairs[idx]
        graph_data = self.smile_graph_dict.get(smi)
        cell_feat_np = self.aligned_cell_features_dict.get(cell_name)

        if graph_data is None: raise ValueError(f"Cache miss for SMILES: {smi}")
        if cell_feat_np is None: raise ValueError(f"Cache miss for cell: {cell_name}")
        
        _, atom_feats_list, edge_index_arr = graph_data
        atom_features_tensor = torch.FloatTensor(np.array(atom_feats_list))
        edge_index_tensor = torch.LongTensor(edge_index_arr)
        cell_features_tensor = torch.FloatTensor(cell_feat_np)

        return Data(x=atom_features_tensor,
                    edge_index=edge_index_tensor,
                    y=torch.FloatTensor([0.0]),
                    target=cell_features_tensor)

def get_training_feature_map(data_dir):
    mut_dict_path = os.path.join(data_dir, 'mut_dict.pkl')
    if os.path.exists(mut_dict_path):
        print(f"信息: 正在从 '{mut_dict_path}' 加载预存的特征图谱...")
        with open(mut_dict_path, 'rb') as f: mut_dict = pickle.load(f)
        return mut_dict
    genetic_feature_path = os.path.join(data_dir, "PANCANCER_Genetic_feature.csv")
    print(f"警告: 未找到特征图谱 '{mut_dict_path}'。正在尝试重新生成...")
    if not os.path.exists(genetic_feature_path):
        print(f"严重错误: 无法生成特征图谱，因为 '{genetic_feature_path}' 不存在。", file=sys.stderr); sys.exit(1)
    mut_dict = {}
    with open(genetic_feature_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for item in reader:
            try:
                mut = item[5]
                if mut not in mut_dict: mut_dict[mut] = len(mut_dict)
            except IndexError: continue
    with open(mut_dict_path, 'wb') as f: pickle.dump(mut_dict, f)
    print(f"信息: 已成功生成并保存新的特征图谱到 '{mut_dict_path}'。")
    return mut_dict

def create_cell_feature_vectors(new_cell_data_path, training_feature_map):
    """
    【集成最终版】采用精确匹配逻辑，并打印总交集数作为过程信息。
    """
    if not os.path.exists(new_cell_data_path):
        raise FileNotFoundError(f"新的细胞系数据文件 '{new_cell_data_path}' 未找到。")
        
    print("信息: 正在对齐新的细胞系特征...")
    try:
        new_cells_df = pd.read_csv(new_cell_data_path, index_col=0)
    except Exception as e:
        raise ValueError(f"读取细胞系数据文件 '{new_cell_data_path}' 失败: {e}")

    # --- 打印交集信息 ---
    # 1. 清理并获取模型基因名集合（只包含纯基因名，如 'tp53'）
    model_gene_set = set()
    for feature_name in training_feature_map.keys():
        base_name = re.sub(r'_(mut|cnv|fusion)$', '', str(feature_name)).strip().lower()
        model_gene_set.update(base_name.split('-'))
        
    # 2. 清理并获取用户数据中的基因名集合
    user_gene_set = set(str(col).strip().lower() for col in new_cells_df.columns)
    
    # 3. 计算并打印交集
    intersection_count = len(model_gene_set.intersection(user_gene_set))
    print(f"信息: 您的数据基因与模型基因库的总交集基因数: {intersection_count}")
    # --- 交集信息结束 ---

    # 清理您的数据文件的列名（去空字符、转小写）以进行后续处理
    new_cells_df.columns = [str(col).strip().lower() for col in new_cells_df.columns]
    
    model_feature_to_index = {str(k).strip().lower(): v for k, v in training_feature_map.items()}
    num_features = len(training_feature_map)
    aligned_features_dict = {}
    
    for cell_name, cell_data_row in new_cells_df.iterrows():
        new_feature_vector = np.zeros(num_features, dtype=np.float32)
        for gene_in_data, value in cell_data_row.items():
            if pd.notna(value) and value != 0:
                # 您的数据列是 'tp53' -> 我们查找 'tp53_mut'
                # 您的数据列是 'tp53_cnv' -> 我们查找 'tp53_cnv'
                feature_to_find = gene_in_data if '_' in gene_in_data else f"{gene_in_data}_mut"
                if feature_to_find in model_feature_to_index:
                    feature_index = model_feature_to_index[feature_to_find]
                    new_feature_vector[feature_index] = 1.0
        aligned_features_dict[str(cell_name)] = new_feature_vector # 确保细胞系名称是字符串
        
    print(f"信息: 成功为 {len(aligned_features_dict)} 个新细胞系创建了特征向量。")
    return aligned_features_dict

def unscale_ic50(scaled_ic50_array, epsilon=1e-9):
    scaled_ic50_array = np.array(scaled_ic50_array)
    scaled_clipped = np.clip(scaled_ic50_array, epsilon, 1 - epsilon)
    term = (1 - scaled_clipped) / scaled_clipped
    unscaled_values = -10 * np.log(term)
    return unscaled_values

def perform_prediction(model, device, loader, num_cell_features):
    model.eval()
    total_preds = []
    with torch.no_grad():
        for data in tqdm(loader, desc="  预测中", leave=False):
            data = data.to(device)
            if hasattr(data, 'target') and data.target.dim() == 1:
                num_graphs_in_batch = data.num_graphs
                expected_elements = num_graphs_in_batch * num_cell_features
                if data.target.numel() == expected_elements:
                    data.target = data.target.view(num_graphs_in_batch, num_cell_features)
            output, _ = model(data)
            total_preds.append(output.cpu())
    if not total_preds: return np.array([])
    return torch.cat(total_preds, dim=0).numpy().flatten()

def clear_memory(device):
    """强制清理CPU内存和GPU显存"""
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

# --- 主流程 ---
def main():
    parser = argparse.ArgumentParser(
        description="对新的药物和细胞系进行分块预测，支持内存和显存清理。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # 参数定义
    parser.add_argument('--model_path', required=True, type=str, help="【必需】预训练模型文件的完整路径。")
    parser.add_argument('--model_type', required=True, type=str, choices=['GCNNet', 'GINConvNet', 'GATNet', 'GAT_GCN'], help="【必需】指定模型文件的架构类型。")
    parser.add_argument('--drug_file', required=True, type=str, help="【输入】包含新药物名称和SMILES的CSV文件路径。")
    parser.add_argument('--cell_file', required=True, type=str, help="【输入】包含新细胞系基因数据的CSV文件路径。")
    parser.add_argument('--output_dir', required=True, type=str, help="输出预测结果CSV文件的目录。")
    parser.add_argument('--output_prefix', default='predictions', type=str, help="输出预测结果的文件名前缀。例如：'predictions'。")
    parser.add_argument('--data_dir', type=str, default='mydata/', help="包含预处理数据（如PANCANCER_Genetic_feature.csv）的目录。")
    parser.add_argument('--cuda_name', type=str, default="cuda:0", help="使用的CUDA设备 (例如 'cuda:0') 或 'cpu'。")
    parser.add_argument('--batch_size', type=int, default=1024, help="预测时使用的批处理大小。")
    parser.add_argument('--chunk_size', type=int, default=50000, help="每次从药物文件中读取的行数（块大小）。")
    parser.add_argument('--num_workers', type=int, default=4, help="DataLoader使用的工作进程数。")

    args = parser.parse_args()
    device = torch.device(args.cuda_name if torch.cuda.is_available() and args.cuda_name != 'cpu' else "cpu")

    print(f"--- 开始分块预测流程 ---")
    print(f"模型: {args.model_type} | 药物文件: {args.drug_file} | 块大小: {args.chunk_size}")
    print(f"设备: {device} | DataLoader Workers: {args.num_workers}")

    # 1. 一次性加载和处理细胞系数据和模型
    print("\n--- 步骤 1/3: 准备细胞数据和模型 ---")
    try:
        training_feature_map = get_training_feature_map(args.data_dir)
        num_cell_features = len(training_feature_map)
        # 调用我们最终修正的函数
        aligned_cell_features_dict = create_cell_feature_vectors(args.cell_file, training_feature_map)
        cell_names = list(aligned_cell_features_dict.keys())
        if not cell_names: raise ValueError("未能从细胞文件中解析出任何细胞系。")
    except (FileNotFoundError, ValueError) as e:
        print(f"严重错误: 准备细胞数据失败: {e}", file=sys.stderr); sys.exit(1)

    model_map = {'GCNNet': GCNNet, 'GINConvNet': GINConvNet, 'GATNet': GATNet, 'GAT_GCN': GAT_GCN}
    Model = model_map[args.model_type]
    try:
        model = Model() if args.model_type == 'GINConvNet' else Model(num_features_xt=num_cell_features)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.to(device)
    except Exception as e:
        print(f"严重错误: 加载模型失败: {e}", file=sys.stderr); sys.exit(1)
    
    print("--- 步骤 1/3 完成 ---")

    # 2. 分块读取药物文件并进行预测
    print("\n--- 步骤 2/3: 分块处理药物并预测 ---")
    try:
        chunk_iterator = pd.read_csv(args.drug_file, header=None, names=['drug_name', 'smiles'], chunksize=args.chunk_size, on_bad_lines='skip', low_memory=False)
    except Exception as e:
        print(f"严重错误: 读取药物文件 '{args.drug_file}' 失败: {e}", file=sys.stderr); sys.exit(1)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    for chunk_idx, drug_chunk_df in enumerate(chunk_iterator):
        chunk_num = chunk_idx + 1
        print(f"\n--- 正在处理块 {chunk_num} ---")
        
        clear_memory(device)
        drug_chunk_df.dropna(subset=['smiles'], inplace=True)
        if drug_chunk_df.empty:
            print("块中无有效药物数据，跳过。")
            continue

        smile_graph_dict_chunk = {}
        unique_smiles_chunk = drug_chunk_df['smiles'].unique()
        for smi in tqdm(unique_smiles_chunk, desc=f"  [块 {chunk_num}] 预处理SMILES"):
            try:
                graph_data = smile_to_graph(smi)
                if graph_data[0] is not None: smile_graph_dict_chunk[smi] = graph_data
            except Exception: continue
        
        if not smile_graph_dict_chunk:
            print(f"警告: 块 {chunk_num} 中所有SMILES均无法处理。")
            continue

        drug_cell_pairs_chunk, drug_names_chunk, cell_names_chunk = [], [], []
        valid_drugs_df = drug_chunk_df[drug_chunk_df['smiles'].isin(smile_graph_dict_chunk.keys())]

        for _, row in valid_drugs_df.iterrows():
            drug_name, smi = row['drug_name'], row['smiles']
            for cell_name in cell_names:
                drug_cell_pairs_chunk.append((smi, cell_name))
                drug_names_chunk.append(drug_name)
                cell_names_chunk.append(cell_name)

        if not drug_cell_pairs_chunk:
            print(f"警告: 块 {chunk_num} 中未能构建任何有效的药物-细胞对。")
            continue

        dataset_chunk = PredictionDataset(drug_cell_pairs_chunk, smile_graph_dict_chunk, aligned_cell_features_dict)
        loader_chunk = DataLoader(dataset_chunk, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True if device.type == 'cuda' else False)
        
        print(f"  [块 {chunk_num}] 开始预测 {len(drug_cell_pairs_chunk)} 个数据对...")
        scaled_preds = perform_prediction(model, device, loader_chunk, num_cell_features)
        
        if scaled_preds.size == 0:
            print(f"警告: 块 {chunk_num} 预测结果为空。")
            continue

        original_preds = unscale_ic50(scaled_preds)
        
        results_df = pd.DataFrame({
            'drug_name': drug_names_chunk,
            'cell_line_name': cell_names_chunk,
            'IC50_scaled': scaled_preds,
            'IC50_original': original_preds
        })
        
        output_filename = os.path.join(args.output_dir, f"{args.output_prefix}_{chunk_num}.csv")
        try:
            results_df.to_csv(output_filename, index=False)
            print(f"✅ [块 {chunk_num}] 预测完成！结果已保存到 '{output_filename}'。")
        except Exception as e:
            print(f"错误: 保存块 {chunk_num} 的结果到 '{output_filename}' 失败: {e}")
        
        del drug_chunk_df, smile_graph_dict_chunk, dataset_chunk, loader_chunk, results_df
        clear_memory(device)

    print("\n--- 步骤 2/3 完成 ---")
    print("\n--- 步骤 3/3: 所有块处理完毕 ---")
    print(f"\n🎉🎉🎉 全部预测任务完成！请检查在 '{args.output_dir}' 目录中以 '{args.output_prefix}_*.csv' 命名的所有输出文件。")

if __name__ == '__main__':
    # 在Windows或macOS上，多进程(num_workers>0)通常需要这个设置
    if sys.platform in ["win32", "darwin"]:
        torch.multiprocessing.set_start_method('spawn', force=True)
    main()