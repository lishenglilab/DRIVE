# predict_all1.py (最终生产版 - 修正拼写错误)
import argparse
import csv
import os
import pickle
import re
import sys

import networkx as nx
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data
from tqdm import tqdm

# --- 模型定义 ---
try:
    from models.gat import GATNet
    from models.gat_gcn import GAT_GCN
    from models.gcn import GCNNet
    from models.ginconv import GINConvNet
except ImportError as e:
    print(f"错误: 无法导入模型定义。请确保 'models' 文件夹及其中的 .py 文件存在。\n{e}", file=sys.stderr)
    sys.exit(1)

# --- 辅助函数 ---
def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As','Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se','Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr','Pt', 'Hg', 'Pb', 'Unknown']) + one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + [atom.GetIsAromatic()])
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set: raise Exception(f"Input {x} not in allowable set {allowable_set}")
    return list(map(lambda s: x == s, allowable_set))
def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set: x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))
def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None: return None, None, None
    c_size = mol.GetNumAtoms();
    if c_size == 0: return None, None, None
    features = [atom_features(atom) for atom in mol.GetAtoms()]
    edges = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()]
    g = nx.Graph(edges).to_directed()
    edge_index_list = [[e1, e2] for e1, e2 in g.edges()]
    if edge_index_list: return c_size, features, np.array(edge_index_list, dtype=np.int64).T
    else: return c_size, features, np.empty((2, 0), dtype=np.int64)
def get_training_feature_map(data_dir="mydata/"):
    mut_dict_path = os.path.join(data_dir, 'mut_dict.pkl')
    if os.path.exists(mut_dict_path):
        print(f"信息: 正在从 '{mut_dict_path}' 加载预存的特征图谱...")
        with open(mut_dict_path, 'rb') as f: mut_dict = pickle.load(f)
        return mut_dict
    genetic_feature_path = os.path.join(data_dir, "PANCANCER_Genetic_feature.csv")
    print(f"警告: 未找到特征图谱 '{mut_dict_path}'。正在尝试重新生成...")
    if not os.path.exists(genetic_feature_path): print(f"严重错误: '{genetic_feature_path}' 不存在。", file=sys.stderr); sys.exit(1)
    mut_dict = {}
    with open(genetic_feature_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)
        for item in reader:
            try:
                mut = item[5]
                if mut not in mut_dict: mut_dict[mut] = len(mut_dict)
            except IndexError: continue
    with open(mut_dict_path, 'wb') as f: pickle.dump(mut_dict, f);
    return mut_dict

def create_cell_feature_vectors(new_cell_data_path, training_feature_map):
    """
    最终生产版：采用精确匹配逻辑，并打印总交集数作为过程信息。
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
                feature_to_find = gene_in_data if '_' in gene_in_data else f"{gene_in_data}_mut"
                if feature_to_find in model_feature_to_index:
                    feature_index = model_feature_to_index[feature_to_find]
                    new_feature_vector[feature_index] = 1.0
        aligned_features_dict[cell_name] = new_feature_vector
        
    print(f"信息: 成功为 {len(aligned_features_dict)} 个新细胞系创建了特征向量。")
    return aligned_features_dict

def unscale_ic50(scaled_ic50_array, epsilon=1e-9):
    # 【修正】修正了变量名中的非法字符
    scaled_ic50_array = np.array(scaled_ic50_array)
    scaled_clipped = np.clip(scaled_ic50_array, epsilon, 1 - epsilon)
    term = (1 - scaled_clipped) / scaled_clipped
    unscaled_values = -10 * np.log(term)
    return unscaled_values

# --- 主流程 (保持不变) ---
def main():
    parser = argparse.ArgumentParser(description="对新的药物和细胞系进行药物敏感性预测。")
    parser.add_argument('--model_path', type=str, default='./model_GAT_GCN_GDSC_blind_run1.model', help="预训练模型文件的完整路径。")
    parser.add_argument('--model_type', type=str, default='GAT_GCN', choices=['GCNNet', 'GINConvNet', 'GATNet', 'GAT_GCN'], help="指定模型文件的架构类型。")
    parser.add_argument('--drug_file', type=str, default='./test/drug_sample.csv', help="包含新药物名称和SMILES的CSV文件路径。")
    parser.add_argument('--cell_file', type=str, default='./test/mu_sample.csv', help="包含新细胞系基因数据的CSV文件路径。")
    parser.add_argument('--output_file', type=str, default='predictions_combined.csv', help="输出预测结果的CSV文件名。")
    parser.add_argument('--data_dir', type=str, default='mydata/', help="包含预处理数据的目录。")
    parser.add_argument('--cuda_name', type=str, default="cuda:0", help="使用的CUDA设备 (例如 'cuda:0') 或 'cpu'。")
    args = parser.parse_args()

    device = torch.device(args.cuda_name if torch.cuda.is_available() and args.cuda_name != 'cpu' else "cpu")
    print(f"--- 开始合并预测 ---")
    print(f"信息: 使用设备: {device}")

    try:
        new_drugs_df = pd.read_csv(args.drug_file, header=None, names=['drug_name', 'smiles']); new_drugs_df.dropna(subset=['smiles'], inplace=True)
        smile_graph_dict = {smi: smile_to_graph(smi) for smi in tqdm(new_drugs_df['smiles'].unique(), desc="处理SMILES") if smile_to_graph(smi)[0] is not None}
        training_feature_map = get_training_feature_map(args.data_dir); num_cell_features = len(training_feature_map)
        aligned_cell_features_dict = create_cell_feature_vectors(args.cell_file, training_feature_map)
    except (FileNotFoundError, ValueError) as e: print(f"严重错误: 数据准备阶段失败: {e}", file=sys.stderr); return

    print(f"信息: 正在加载模型 {args.model_type} 从 {args.model_path}...")
    Model = {'GCNNet': GCNNet, 'GINConvNet': GINConvNet, 'GATNet': GATNet, 'GAT_GCN': GAT_GCN}[args.model_type]
    try:
        model = Model() if args.model_type == 'GINConvNet' else Model(num_features_xt=num_cell_features); model.load_state_dict(torch.load(args.model_path, map_location=device)); model.to(device); model.eval()
    except Exception as e: print(f"严重错误: 加载模型失败。请确保模型类型和权重文件匹配。\n{e}", file=sys.stderr); return

    all_results = []; valid_drugs_df = new_drugs_df[new_drugs_df['smiles'].isin(smile_graph_dict.keys())]; total_predictions = len(valid_drugs_df) * len(aligned_cell_features_dict)
    with tqdm(total=total_predictions, desc="逐对预测 (药物,细胞)") as pbar:
        for _, drug_row in valid_drugs_df.iterrows():
            drug_name, smi = drug_row['drug_name'], drug_row['smiles']
            _, atom_feats_list, edge_index_arr = smile_graph_dict[smi]; atom_features_tensor = torch.FloatTensor(np.array(atom_feats_list)); edge_index_tensor = torch.LongTensor(edge_index_arr); num_atoms = atom_features_tensor.shape[0]
            for cell_name, cell_vector in aligned_cell_features_dict.items():
                pbar.set_postfix_str(f"药物: {drug_name[:15]}..., 细胞: {cell_name}")
                cell_features_tensor = torch.FloatTensor(cell_vector).unsqueeze(0); batch_tensor = torch.zeros(num_atoms, dtype=torch.long)
                data = Data(x=atom_features_tensor, edge_index=edge_index_tensor, target=cell_features_tensor, batch=batch_tensor).to(device)
                with torch.no_grad(): output, _ = model(data)
                scaled_pred = output.item(); original_pred = unscale_ic50([scaled_pred])[0]
                all_results.append({'drug_name': drug_name, 'cell_line_name': cell_name, 'IC50_scaled': scaled_pred, 'IC50_original': original_pred})
                pbar.update(1)

    if not all_results: print("\n警告: 没有生成任何预测结果。"); return
    final_df = pd.DataFrame(all_results)
    try:
        final_df.to_csv(args.output_file, index=False); print(f"\n\n✅ 预测完成！结果已成功保存到 '{args.output_file}'。")
    except Exception as e: print(f"\n错误: 保存最终结果到 '{args.output_file}' 失败: {e}")

if __name__ == '__main__':
    main()