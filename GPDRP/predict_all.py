import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import os
from rdkit import Chem
import numpy as np
import networkx as nx
import csv
import traceback
import math
from tqdm import tqdm

# --- 模型定义导入 ---
try:
    from GIN.model.gin import GINConvNet as OriginalGINConvNet
except ImportError:
    print("Warning: Could not import OriginalGINConvNet from GIN.model.gin. Predictions with 'GIN' will fail.")
    OriginalGINConvNet = None
try:
    from GAT.model.gat import GATNet
except ImportError:
    print("Warning: Could not import GATNet from GAT.model.gat. Predictions with 'GAT' will fail.")
    GATNet = None
try:
    from GCN.model.gcn import GCNNet
except ImportError:
    print("Warning: Could not import GCNNet from GCN.model.gcn. Predictions with 'GCN' will fail.")
    GCNNet = None
try:
    from GIN_TRANSFORMER.model.gintranformer import GINConvNet2
except ImportError:
    print("Warning: Could not import GINConvNet2 from GIN_TRANSFORMER.model.gintransformer. Predictions with 'GINTransformer' will fail.")
    GINConvNet2 = None

# --- 全局常量 ---
EXPECTED_ATOM_FEATURE_DIM = 78

# ... [所有辅助函数保持不变] ...
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        if isinstance(x, int) and allowable_set and isinstance(allowable_set[0], int):
            if x > allowable_set[-1]:
                x = allowable_set[-1]
            elif x < allowable_set[0]:
                x = allowable_set[0]
    return list(map(lambda s: x == s, allowable_set))
def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))
def atom_features_from_preprocessing(atom):
    allowable_symbols = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                         'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                         'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                         'Pt', 'Hg', 'Pb', 'Unknown']
    allowable_degree = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    allowable_total_hs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    allowable_implicit_valence = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    try:
        features = np.array(
            one_of_k_encoding_unk(atom.GetSymbol(), allowable_symbols) +
            one_of_k_encoding(atom.GetDegree(), allowable_degree) +
            one_of_k_encoding_unk(atom.GetTotalNumHs(), allowable_total_hs) +
            one_of_k_encoding_unk(atom.GetImplicitValence(), allowable_implicit_valence) +
            [atom.GetIsAromatic()]
        )
    except Exception:
        return np.zeros(EXPECTED_ATOM_FEATURE_DIM)
    return features
def smiles_to_graph_data_with_cell(smiles_string, cell_feature_vector):
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    atom_f_list = []
    for atom_idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(atom_idx)
        feature_vec = atom_features_from_preprocessing(atom)
        sum_feature = np.sum(feature_vec)
        if sum_feature == 0:
            normalized_feature = feature_vec
        else:
            normalized_feature = feature_vec / sum_feature
        atom_f_list.append(normalized_feature)

    x = torch.tensor(np.array(atom_f_list), dtype=torch.float)

    if x.shape[0] > 0 and x.shape[1] != EXPECTED_ATOM_FEATURE_DIM:
        print(
            f"FATAL ERROR: Atom features for SMILES {smiles_string} have dim {x.shape[1]}, expected {EXPECTED_ATOM_FEATURE_DIM}.")
        return None

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])

    if not edges:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        g_nx = nx.Graph(edges).to_directed()
        edge_index_list = []
        if g_nx.number_of_edges() > 0:
            for e1, e2 in g_nx.edges:
                edge_index_list.append([e1, e2])
            edge_index = torch.tensor(np.array(edge_index_list).T, dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

    target_ge_tensor = torch.tensor(cell_feature_vector, dtype=torch.float).unsqueeze(0)
    data = Data(x=x, edge_index=edge_index, target_ge=target_ge_tensor)
    return data
def load_training_gene_info_and_norm_params(file_path):
    print(f"正在从 '{file_path}' 加载训练时使用的基因列表和归一化参数...")
    try:
        df = pd.read_csv(file_path, sep='\t')
        training_genes = df.iloc[:, 1].tolist()
        numeric_df = df.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')
        all_values = numeric_df.values.flatten()
        all_values = all_values[~np.isnan(all_values)]

        if len(all_values) == 0:
            raise ValueError("在文件中未找到任何有效的基因表达数值。")

        min_val = np.min(all_values)
        max_val = 12.0
        print(f"加载完成: {len(training_genes)}个基因。归一化参数: min={min_val:.4f}, max={max_val:.4f}")
        return training_genes, min_val, max_val

    except FileNotFoundError:
        print(f"错误: 找不到训练基因表达文件 '{file_path}'。这是进行基因对齐和归一化所必需的。")
        return None, None, None
    except Exception as e:
        print(f"加载训练基因信息时出错: {e}")
        return None, None, None
def load_and_preprocess_new_cell_lines(new_cell_file, training_genes, min_val, max_val):
    print(f"正在加载和预处理新的细胞系文件: '{new_cell_file}'...")
    try:
        df_new = pd.read_csv(new_cell_file, index_col=0)
        print(f"成功加载 {df_new.shape[0]} 个新细胞系，包含 {df_new.shape[1]} 个基因。")

        print("正在对齐基因...")
        df_aligned = df_new.reindex(columns=training_genes, fill_value=0.0)
        print(f"基因对齐完成。特征维度: {df_aligned.shape[1]}")

        print("正在进行归一化...")
        X = df_aligned.values.astype(np.float32)
        X = np.clip(X, None, max_val)

        if (max_val - min_val) == 0:
            X_normalized = np.zeros_like(X)
        else:
            X_normalized = (X - min_val) / (max_val - min_val)

        X_normalized = np.clip(X_normalized, 0.0, 1.0)
        print("归一化完成。")

        cell_features_map = {name: vector for name, vector in zip(df_aligned.index, X_normalized)}
        return cell_features_map

    except FileNotFoundError:
        print(f"错误: 新细胞系文件 '{new_cell_file}' 未找到。")
        return None
    except Exception as e:
        print(f"处理新细胞系文件时出错: {e}")
        traceback.print_exc()
        return None
def unscale_ic50(scaled_value_pred):
    epsilon = 1e-9
    if not isinstance(scaled_value_pred, (int, float, np.float32, np.float64)):
        if isinstance(scaled_value_pred, torch.Tensor):
            scaled_value_pred = scaled_value_pred.item() if scaled_value_pred.numel() == 1 else np.nan
        elif isinstance(scaled_value_pred, np.ndarray):
            scaled_value_pred = scaled_value_pred.item() if scaled_value_pred.size == 1 else np.nan
        else:
            return np.nan
    if np.isnan(scaled_value_pred):
        return np.nan

    y = np.clip(scaled_value_pred, epsilon, 1.0 - epsilon)
    try:
        ratio = y / (1.0 - y)
        if ratio <= 0: return np.nan
        original_value = -10.0 * math.log((1.0 - y) / y)
    except (ValueError, OverflowError):
        original_value = np.nan
    return original_value
def predict_new_drugs(model, device, data_loader):
    model.eval()
    total_preds = torch.Tensor()
    with torch.no_grad():
        for data_batch in data_loader:
            data_batch = data_batch.to(device)
            try:
                out_tuple = model(data_batch)
                output = out_tuple[0] if isinstance(out_tuple, tuple) else out_tuple
                if output.ndim == 1: output = output.unsqueeze(1)
            except Exception:
                num_graphs_in_batch = data_batch.num_graphs if hasattr(data_batch, 'num_graphs') else 1
                output = torch.full((num_graphs_in_batch, 1), float('nan'), device=device)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
    return total_preds.numpy()
def main():
    parser = argparse.ArgumentParser(description='使用预训练GNN模型，对输入的新药物和新细胞系进行IC50预测。')
    # --- MODIFICATION: Make all paths required arguments ---
    parser.add_argument('--smiles_file', type=str, required=True, help='药物输入文件: CSV格式, 无表头, 第1列为药物名, 第2列为SMILES字符串。')
    parser.add_argument('--new_cell_line_file', type=str, required=True, help='新细胞系输入文件: CSV格式, 第1行为基因名(表头), 第1列为细胞系名(索引)。')
    parser.add_argument('--training_gene_expression_file', type=str, required=True, help='用于对齐和归一化的原始训练基因表达文件(exp.txt)路径。')
    parser.add_argument('--model_file', type=str, required=True, help='预训练的模型权重文件路径 (.model)。')
    parser.add_argument('--model_type', type=str, default='GIN', choices=['GIN', 'GAT', 'GCN', 'GINTransformer'], help='要使用的GNN模型类型。')
    parser.add_argument('--output_file', type=str, required=True, help='保存预测结果的CSV文件路径。')
    parser.add_argument('--batch_size', type=int, default=32, help='预测时的批处理大小。')
    parser.add_argument('--cuda_name', type=str, default="cuda:0", help='CUDA设备名称 (例如: cuda:0, cpu)。')
    
    args = parser.parse_args()
    print(f"脚本参数: {args}")

    training_genes, norm_min, norm_max = load_training_gene_info_and_norm_params(args.training_gene_expression_file)
    if not training_genes:
        print("错误: 未能加载训练基因信息，无法继续。"); return

    cell_features_map = load_and_preprocess_new_cell_lines(args.new_cell_line_file, training_genes, norm_min, norm_max)
    if cell_features_map is None or not cell_features_map:
        print("错误: 未能加载或处理新细胞系文件，无法继续。"); return
    print(f"成功为 {len(cell_features_map)} 个新细胞系准备好特征。")

    if not os.path.exists(args.smiles_file):
        print(f"错误: 药物SMILES文件未找到于 '{args.smiles_file}'"); return
    try:
        df_smiles = pd.read_csv(args.smiles_file, header=None, names=['drug_name', 'smiles'])
        smiles_input_list = df_smiles.values.tolist()
        if not smiles_input_list:
            print(f"错误: 在 '{args.smiles_file}' 中未找到药物数据。"); return
    except Exception as e:
        print(f"读取药物SMILES文件 '{args.smiles_file}' 时出错: {e}"); return

    device = torch.device(args.cuda_name if torch.cuda.is_available() and args.cuda_name != "cpu" else "cpu")
    print(f"使用设备: {device}")

    model_map = {'GIN': OriginalGINConvNet, 'GAT': GATNet, 'GCN': GCNNet, 'GINTransformer': GINConvNet2}
    model_class = model_map.get(args.model_type)
    if model_class is None:
        print(f"错误: 模型类 '{args.model_type}' 未被正确导入或不可用。"); return

    model_instance = model_class().to(device)
    print(f"已实例化模型: {args.model_type}")

    if not os.path.exists(args.model_file):
        print(f"错误: 模型文件未找到于 '{args.model_file}'。"); return
    try:
        model_instance.load_state_dict(torch.load(args.model_file, map_location=device))
        print(f"成功从 '{args.model_file}' 加载模型权重。")
    except Exception as e:
        print(f"加载模型权重时出错: {e}"); return

    all_results_list = []
    for drug_name, smiles_str in tqdm(smiles_input_list, desc="处理药物"):
        if not isinstance(smiles_str, str) or not smiles_str.strip() or Chem.MolFromSmiles(smiles_str) is None:
            for cell_name in cell_features_map.keys():
                all_results_list.append({'drug_name': drug_name, 'smiles': smiles_str, 'cell_line_name': cell_name, 'predicted_ic50_scaled': np.nan, 'predicted_ic50_original': np.nan})
            continue

        drug_cell_pairs_data = []
        drug_cell_pairs_info = []
        for cell_name, cell_feat_vector in cell_features_map.items():
            graph_data = smiles_to_graph_data_with_cell(smiles_str, cell_feat_vector)
            if graph_data:
                drug_cell_pairs_data.append(graph_data)
                drug_cell_pairs_info.append({'drug_name': drug_name, 'smiles': smiles_str, 'cell_line_name': cell_name})
            else:
                all_results_list.append({'drug_name': drug_name, 'smiles': smiles_str, 'cell_line_name': cell_name, 'predicted_ic50_scaled': np.nan, 'predicted_ic50_original': np.nan})

        if not drug_cell_pairs_data: continue

        current_drug_loader = DataLoader(drug_cell_pairs_data, batch_size=args.batch_size, shuffle=False)
        try:
            scaled_predictions = predict_new_drugs(model_instance, device, current_drug_loader)
            for i, pred_info in enumerate(drug_cell_pairs_info):
                result = pred_info.copy()
                scaled_pred_val = float(scaled_predictions[i][0]) if i < len(scaled_predictions) else np.nan
                result['predicted_ic50_scaled'] = scaled_pred_val
                result['predicted_ic50_original'] = unscale_ic50(scaled_pred_val)
                all_results_list.append(result)
        except Exception as e_pred:
            print(f"对药物 '{drug_name}' 的预测过程中发生错误: {e_pred}")
            for info_dict in drug_cell_pairs_info:
                result = info_dict.copy()
                result['predicted_ic50_scaled'] = np.nan; result['predicted_ic50_original'] = np.nan
                all_results_list.append(result)
        
        del drug_cell_pairs_data, drug_cell_pairs_info
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    if not all_results_list:
        print("没有生成任何预测结果。"); return

    results_df = pd.DataFrame(all_results_list)
    cols_order = ['drug_name', 'smiles', 'cell_line_name', 'predicted_ic50_scaled', 'predicted_ic50_original']
    results_df = results_df[cols_order]

    results_df.to_csv(args.output_file, index=False)
    print(f"\n预测完成！结果已保存至 '{args.output_file}'")
    print("\n结果预览 (前5行):")
    print(results_df.head())

if __name__ == "__main__":
    main()