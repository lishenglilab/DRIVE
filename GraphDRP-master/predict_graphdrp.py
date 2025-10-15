# 文件名: predict_graphdrp.py (最终集成优化版 - 接受外部参数)
import argparse
import csv
import gc
import math
import os
import pickle
import re
import sys
# 【优化】导入多进程库
from multiprocessing import Pool, cpu_count

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem, rdBase
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

# 抑制 RDKit 的冗余日志
rdBase.DisableLog('rdApp.*')

# --- 辅助函数和类定义部分 (无变化) ---

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
    try:
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
    except Exception:
        return None, None, None

def process_smile_wrapper(smi):
    graph_data = smile_to_graph(smi)
    if graph_data[0] is not None:
        return smi, graph_data
    return smi, None

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
        return Data(x=atom_features_tensor, edge_index=edge_index_tensor, y=torch.FloatTensor([0.0]), target=cell_features_tensor)

def get_training_feature_map(data_dir):
    mut_dict_path = os.path.join(data_dir, 'mut_dict.pkl')
    if os.path.exists(mut_dict_path):
        with open(mut_dict_path, 'rb') as f: mut_dict = pickle.load(f)
        return mut_dict
    genetic_feature_path = os.path.join(data_dir, "PANCANCER_Genetic_feature.csv")
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
    return mut_dict

def create_cell_feature_vectors(new_cell_data_path, training_feature_map):
    if not os.path.exists(new_cell_data_path):
        raise FileNotFoundError(f"新的细胞系数据文件 '{new_cell_data_path}' 未找到。")
    new_cells_df = pd.read_csv(new_cell_data_path, index_col=0)
    model_gene_set = set(re.sub(r'_(mut|cnv|fusion)$', '', str(k)).strip().lower() for k in training_feature_map.keys())
    user_gene_set = set(str(col).strip().lower() for col in new_cells_df.columns)
    print(f"信息: 您的数据基因与模型基因库的总交集基因数: {len(model_gene_set.intersection(user_gene_set))}")
    new_cells_df.columns = [str(col).strip().lower() for col in new_cells_df.columns]
    model_feature_to_index = {str(k).strip().lower(): v for k, v in training_feature_map.items()}
    num_features = len(training_feature_map)
    aligned_features_dict = {}
    for cell_name, cell_data_row in new_cells_df.iterrows():
        new_feature_vector = np.zeros(num_features, dtype=np.float32)
        for gene_in_data, value in cell_data_row.items():
            if pd.notna(value) and value != 0:
                feature_to_find = gene_in_data if '_' in gene_in_data else f"{gene_in_data}_mut"
                if feature_to_find in model_feature_to_index:
                    new_feature_vector[model_feature_to_index[feature_to_find]] = 1.0
        aligned_features_dict[str(cell_name)] = new_feature_vector
    print(f"信息: 成功为 {len(aligned_features_dict)} 个新细胞系创建了特征向量。")
    return aligned_features_dict

def unscale_ic50(scaled_ic50_array, epsilon=1e-9):
    scaled_ic50_array = np.array(scaled_ic50_array)
    scaled_clipped = np.clip(scaled_ic50_array, epsilon, 1 - epsilon)
    term = (1 - scaled_clipped) / scaled_clipped
    return -10 * np.log(term)

def perform_prediction(model, device, loader, num_cell_features):
    model.eval()
    total_preds = []
    with torch.no_grad():
        for data in tqdm(loader, desc="  预测中", leave=False):
            try:
                data = data.to(device)
                if hasattr(data, 'target') and data.target.dim() == 1:
                    data.target = data.target.view(data.num_graphs, num_cell_features)
                output, _ = model(data)
                total_preds.append(output.cpu())
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\n警告: 批处理时触发OOM。自动切换到逐样本预测模式处理此失败批次...")
                    torch.cuda.empty_cache()
                    individual_outputs = []
                    for i in range(data.num_graphs):
                        single_sample = data[i].to(device)
                        try:
                            out, _ = model(single_sample)
                            individual_outputs.append(out.cpu())
                        except Exception:
                            individual_outputs.append(torch.tensor([[float('nan')]]))
                    if individual_outputs: total_preds.append(torch.cat(individual_outputs, dim=0))
                else:
                    total_preds.append(torch.full((data.num_graphs, 1), float('nan')))
    if not total_preds: return np.array([])
    return torch.cat(total_preds, dim=0).numpy().flatten()

def clear_memory(device):
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

# --- 主流程 ---
def main():
    parser = argparse.ArgumentParser(
        description="【集成优化版】对新的药物和细胞系进行分块预测。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # 【【【 修改点 】】】: 添加新的命令行参数
    parser.add_argument('--model_path', required=True, type=str, help="【必需】预训练模型文件的完整路径。")
    parser.add_argument('--model_type', required=True, type=str, choices=['GCNNet', 'GINConvNet', 'GATNet', 'GAT_GCN'], help="【必需】指定模型文件的架构类型。")
    parser.add_argument('--drug_file', required=True, type=str, help="【输入】包含新药物名称和SMILES的CSV文件路径。")
    parser.add_argument('--cell_file', required=True, type=str, help="【输入】包含新细胞系基因数据的CSV文件路径。")
    parser.add_argument('--output_dir', required=True, type=str, help="输出预测结果CSV文件的目录。")
    parser.add_argument('--output_prefix', default='predictions', type=str, help="输出预测结果的文件名前缀。")
    parser.add_argument('--data_dir', type=str, default='mydata/', help="包含预处理数据（如mut_dict.pkl）的目录。")
    parser.add_argument('--cuda_name', type=str, default="cuda:0", help="使用的CUDA设备 (例如 'cuda:0') 或 'cpu'。")
    parser.add_argument('--batch_size', type=int, default=1024, help="预测时使用的批处理大小。")
    parser.add_argument('--chunk_size', type=int, default=50000, help="每次从药物文件中读取的行数（块大小）。")
    parser.add_argument('--num_workers', type=int, default=4, help="DataLoader使用的工作进程数。")
    parser.add_argument('--cell_chunk_size', type=int, default=128, help="一次性加载进行预测的细胞系数量。")
    parser.add_argument('--cpu_workers_smiles', type=int, default=max(1, cpu_count() // 2), help="用于SMILES转图的CPU核心数。")

    args = parser.parse_args()
    device = torch.device(args.cuda_name if torch.cuda.is_available() and args.cuda_name != 'cpu' else "cpu")

    print(f"--- 开始分块预测流程 (集成优化版) ---")
    print(f"模型: {args.model_type} | 药物文件: {args.drug_file} | 药物块大小: {args.chunk_size}")
    print(f"细胞块大小: {args.cell_chunk_size} | SMILES处理核心数: {args.cpu_workers_smiles}")
    print(f"设备: {device} | DataLoader Workers: {args.num_workers}")

    try:
        training_feature_map = get_training_feature_map(args.data_dir)
        num_cell_features = len(training_feature_map)
        all_aligned_cell_features_dict = create_cell_feature_vectors(args.cell_file, training_feature_map)
        all_cell_names = list(all_aligned_cell_features_dict.keys())
        if not all_cell_names: raise ValueError("未能从细胞文件中解析出任何细胞系。")
    except (FileNotFoundError, ValueError) as e:
        print(f"严重错误: 准备细胞数据失败: {e}", file=sys.stderr); sys.exit(1)

    model_map = {'GCNNet': GCNNet, 'GINConvNet': GINConvNet, 'GATNet': GATNet, 'GAT_GCN': GAT_GCN}
    Model = model_map[args.model_type]
    model = Model() if args.model_type == 'GINConvNet' else Model(num_features_xt=num_cell_features)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    
    os.makedirs(args.output_dir, exist_ok=True)
    chunk_iterator = pd.read_csv(args.drug_file, header=None, names=['drug_name', 'smiles'], chunksize=args.chunk_size, on_bad_lines='skip', low_memory=False)

    for chunk_idx, drug_chunk_df in enumerate(chunk_iterator):
        chunk_num = chunk_idx + 1
        print(f"\n--- 正在处理药物块 {chunk_num} ---")
        chunk_results_collector = []
        clear_memory(device)
        drug_chunk_df.dropna(subset=['smiles'], inplace=True)
        if drug_chunk_df.empty: continue

        unique_smiles_chunk = drug_chunk_df['smiles'].unique()
        smile_graph_dict_chunk = {}
        print(f"  [药物块 {chunk_num}] 使用 {args.cpu_workers_smiles} 个CPU核心并行预处理 {len(unique_smiles_chunk)} 个唯一SMILES...")
        with Pool(processes=args.cpu_workers_smiles) as pool:
            for smi, graph_data in tqdm(pool.imap_unordered(process_smile_wrapper, unique_smiles_chunk), total=len(unique_smiles_chunk), desc="  预处理SMILES"):
                if graph_data: smile_graph_dict_chunk[smi] = graph_data
        
        if not smile_graph_dict_chunk: continue
        valid_drugs_df = drug_chunk_df[drug_chunk_df['smiles'].isin(smile_graph_dict_chunk.keys())]

        num_cell_chunks = math.ceil(len(all_cell_names) / args.cell_chunk_size)
        for cell_chunk_idx in range(num_cell_chunks):
            print(f"  -- 正在处理细胞块 {cell_chunk_idx + 1}/{num_cell_chunks} --")
            cell_start_idx, cell_end_idx = cell_chunk_idx * args.cell_chunk_size, (cell_chunk_idx + 1) * args.cell_chunk_size
            current_cell_names = all_cell_names[cell_start_idx:cell_end_idx]
            current_cell_features_dict = {name: all_aligned_cell_features_dict[name] for name in current_cell_names}

            drug_cell_pairs_sub, drug_names_sub, cell_names_sub = [], [], []
            for _, row in valid_drugs_df.iterrows():
                for cell_name in current_cell_names:
                    drug_cell_pairs_sub.append((row['smiles'], cell_name))
                    drug_names_sub.append(row['drug_name'])
                    cell_names_sub.append(cell_name)

            if not drug_cell_pairs_sub: continue
            dataset_sub = PredictionDataset(drug_cell_pairs_sub, smile_graph_dict_chunk, current_cell_features_dict)
            loader_sub = DataLoader(dataset_sub, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == 'cuda')
            
            scaled_preds = perform_prediction(model, device, loader_sub, num_cell_features)
            if scaled_preds.size > 0:
                chunk_results_collector.append(pd.DataFrame({
                    'drug_name': drug_names_sub, 'cell_line_name': cell_names_sub,
                    'IC50_scaled': scaled_preds, 'IC50_original': unscale_ic50(scaled_preds)
                }))
            del dataset_sub, loader_sub, scaled_preds
            clear_memory(device)
        
        if chunk_results_collector:
            chunk_df = pd.concat(chunk_results_collector, ignore_index=True)
            output_filename = os.path.join(args.output_dir, f"{args.output_prefix}_chunk_{chunk_num}.csv")
            chunk_df.to_csv(output_filename, index=False)
            print(f"✅ [药物块 {chunk_num}] 预测完成！结果已保存到 '{output_filename}'。")

    print("\n🎉🎉🎉 全部预测任务完成！")

if __name__ == '__main__':
    if sys.platform in ["win32", "darwin"]:
        torch.multiprocessing.set_start_method('spawn', force=True)
    main()