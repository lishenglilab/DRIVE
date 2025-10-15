import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from itertools import product
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import add_self_loops, remove_self_loops
import os
from rdkit import Chem
import numpy as np
import networkx as nx
import traceback
import math
from tqdm import tqdm
import gc
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*An output with one or more elements was resized.*")

# --- 模型定义导入 (请确保路径正确) ---
try:
    from GIN.model.gin import GINConvNet as OriginalGINConvNet
    from GAT.model.gat import GATNet
    from GCN.model.gcn import GCNNet
    from GIN_TRANSFORMER.model.gintranformer import GINConvNet2
except ImportError as e:
    print(f"错误：无法导入模型定义。请确保模型文件路径正确且在Python可搜索的路径中。错误信息: {e}")
    OriginalGINConvNet, GATNet, GCNNet, GINConvNet2 = (None, None, None, None)
    pass

# --- 全局常量 ---
EXPECTED_ATOM_FEATURE_DIM = 78
DRUG_CHUNK_SIZE = 5000 # 一次处理的药物数量

# --- 辅助函数 (无变化) ---
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
    except Exception:
        return np.zeros(EXPECTED_ATOM_FEATURE_DIM)
    return features

def smiles_to_drug_graph_parts(smiles_string):
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None or mol.GetNumAtoms() == 0: return None, None
    atom_f_list = [atom_features_from_preprocessing(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(np.array(atom_f_list), dtype=torch.float)
    if x.shape[0] > 0 and x.shape[1] != EXPECTED_ATOM_FEATURE_DIM: return None, None
    edges = [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in mol.GetBonds()]
    edge_index = torch.empty((2, 0), dtype=torch.long) if not edges else torch.tensor(np.array(list(nx.Graph(edges).to_directed().edges)).T, dtype=torch.long)
    return x, edge_index

# --- 加载数据和反归一化函数 (无变化) ---
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
    except Exception as e:
        print(f"加载训练基因信息时出错: {e}");
        return None, None, None

def load_and_preprocess_new_cell_lines(new_cell_file, training_genes, min_val, max_val):
    print(f"正在加载和预处理新的细胞系文件: '{new_cell_file}'...")
    try:
        df_new = pd.read_csv(new_cell_file, index_col=0)
        new_genes = df_new.columns.tolist()
        common_genes = set(new_genes) & set(training_genes)
        print(f"  - 新细胞系文件包含 {len(new_genes)} 个基因。")
        print(f"  - 训练时使用了 {len(training_genes)} 个基因。")
        print(f"  - 两个列表的交集包含 {len(common_genes)} 个基因。")
        if len(common_genes) / len(training_genes) < 0.1:
             print("\n  [!!!] 严重警告：新细胞系文件中的基因名与训练时使用的基因名几乎没有重合！")
             print("      这将导致所有细胞系的特征向量几乎都为0，从而使所有预测结果相同。")
             print("      请仔细检查并统一两个文件中的基因ID格式（例如 'EGFR' vs '7p11.2'）。\n")
        df_aligned = df_new.reindex(columns=training_genes, fill_value=0.0)
        X = np.clip(df_aligned.values.astype(np.float32), None, max_val)
        X_normalized = (X - min_val) / (max_val - min_val) if (max_val - min_val) != 0 else np.zeros_like(X)
        X_normalized = np.clip(X_normalized, 0.0, 1.0)
        return {name: torch.tensor(vector, dtype=torch.float).unsqueeze(0) for name, vector in zip(df_aligned.index, X_normalized)}
    except Exception as e:
        print(f"处理新细胞系文件时出错: {e}");
        traceback.print_exc();
        return None

def unscale_ic50(scaled_value_pred):
    epsilon = 1e-9
    if not isinstance(scaled_value_pred, (int, float, np.float32, np.float64)): return np.nan
    if np.isnan(scaled_value_pred): return np.nan
    y = np.clip(scaled_value_pred, epsilon, 1.0 - epsilon)
    try:
        original_value = -10.0 * math.log((1.0 - y) / y)
    except (ValueError, OverflowError):
        original_value = np.nan
    return original_value

# ========================================================================
#                          【核心性能优化区域】
# ========================================================================

# Dataset (无变化)
class PrecomputedDrugCellDataset(Dataset):
    def __init__(self, drug_graphs, cell_features_map):
        super().__init__()
        self.drug_graphs = drug_graphs
        self.cell_names = list(cell_features_map.keys())
        self.cell_tensors = list(cell_features_map.values())
        self.num_drugs = len(self.drug_graphs)
        self.num_cells = len(self.cell_tensors)

    def __len__(self):
        return self.num_drugs * self.num_cells

    def __getitem__(self, idx):
        drug_idx = idx // self.num_cells
        cell_idx = idx % self.num_cells
        _, _, drug_data = self.drug_graphs[drug_idx]
        cell_tensor = self.cell_tensors[cell_idx]
        final_data = drug_data.clone()
        final_data.target_ge = cell_tensor.clone()
        return final_data


# 【【【 新增/修改 】】】: 预测函数增加OOM回退机制
def predict_optimized(model, device, data_loader, model_type):
    model.eval()
    total_preds = torch.Tensor()
    with torch.no_grad():
        for data_batch in tqdm(data_loader, desc=f"  - Predicting Batches for Chunk"):
            try:
                # 1. 尝试快速的批处理预测
                data_batch = data_batch.to(device)
                if model_type == 'GAT':
                    data_batch.edge_index, _ = remove_self_loops(data_batch.edge_index)
                    data_batch.edge_index, _ = add_self_loops(data_batch.edge_index, num_nodes=data_batch.num_nodes)
                out, _ = model(data_batch)
                output = out
                if output.ndim == 1: output = output.unsqueeze(1)
            
            except RuntimeError as e:
                # 2. 如果发生OOM，则回退到内存安全模式
                if "out of memory" in str(e).lower():
                    print(f"\n警告: 批处理时触发OOM。自动切换到逐样本预测模式处理此失败批次...")
                    torch.cuda.empty_cache() # 清理GPU缓存
                    
                    individual_preds = []
                    for i in range(data_batch.num_graphs):
                        single_data = data_batch[i].to(device)
                        try:
                            if model_type == 'GAT':
                                single_data.edge_index, _ = remove_self_loops(single_data.edge_index)
                                single_data.edge_index, _ = add_self_loops(single_data.edge_index, num_nodes=single_data.num_nodes)
                            out, _ = model(single_data)
                            individual_preds.append(out.cpu())
                        except Exception as single_e:
                             print(f"  - 在回退模式下预测样本时出错: {single_e}")
                             individual_preds.append(torch.tensor([[float('nan')]])) # 失败的样本填充NaN
                    
                    output = torch.cat(individual_preds, dim=0)
                    if output.ndim == 1: output = output.unsqueeze(1)
                
                elif "nvrtc" in str(e) or "CUDA" in str(e).upper():
                     print(f"\n[!!!] 严重CUDA/NVRTC错误: {e}")
                     print("      此批次将填充NaN，但错误可能会在后续批次中持续出现。")
                     output = torch.full((data_batch.num_graphs, 1), float('nan'), device="cpu")
                else:
                    print(f"\n  - 预测期间发生未知运行时错误: {e}")
                    output = torch.full((data_batch.num_graphs, 1), float('nan'), device="cpu")
            
            except Exception as e:
                print(f"\n  - 预测期间发生未知错误: {e}")
                output = torch.full((data_batch.num_graphs, 1), float('nan'), device="cpu")
            
            total_preds = torch.cat((total_preds, output.cpu()), 0)
    return total_preds.numpy()

# ========================================================================
#                            主函数 MAIN
# ========================================================================
def main():
    parser = argparse.ArgumentParser(description='高性能GNN预测脚本')
    parser.add_argument('--smiles_file', type=str, default='./CRC/drug_depmap.csv', help='药物SMILES文件路径')
    parser.add_argument('--new_cell_line_file', type=str, default='./CRC/celllines_gsea.csv', help='新细胞系基因表达文件路径')
    parser.add_argument('--training_gene_expression_file', type=str, default='./mydata/exp.txt', help='原始训练基因表达文件路径(exp.txt)')
    parser.add_argument('--models_dir', type=str, default='./output/models/', help='包含所有预训练模型的目录')
    parser.add_argument('--output_dir', type=str, default='./results/', help='保存预测结果的目录')
    # 【【【 新增/修改 】】】: 增加细胞系分块大小参数
    parser.add_argument('--cell_chunk_size', type=int, default=32, help='【重要】一次性加载到内存/显存中的细胞系数量。如果有很多细胞系且内存/显存不足，请减小此值。')
    parser.add_argument('--batch_size', type=int, default=256, help='预测时的批处理大小')
    parser.add_argument('--cuda_name', type=str, default="cuda:0", help='CUDA设备名称')
    parser.add_argument('--num_workers', type=int, default=16, help='DataLoader使用的工作进程数。在Windows上如果出错可以设为0。')
    parser.add_argument('--models',
                        nargs='+',
                        default=['GAT', 'GCN'],
                        choices=['GIN', 'GAT', 'GCN', 'GINTransformer'],
                        help='选择要运行的模型。可以是一个或多个，用空格分隔。')

    args = parser.parse_args()
    print(f"脚本参数: {args}")

    # --- 1. 数据和模型准备 (一次性) ---
    os.makedirs(args.output_dir, exist_ok=True)
    training_genes, norm_min, norm_max = load_training_gene_info_and_norm_params(args.training_gene_expression_file)
    if not training_genes: return

    # 全量加载细胞系数据到CPU内存
    all_cell_features_map = load_and_preprocess_new_cell_lines(args.new_cell_line_file, training_genes, norm_min, norm_max)
    if not all_cell_features_map: return
    
    # 诊断代码 (无变化)
    print("\n[诊断信息] 检查预处理后的细胞系特征...")
    if all_cell_features_map:
        cell_names = list(all_cell_features_map.keys())
        first_cell_tensor = all_cell_features_map[cell_names[0]]
        print(f"  - 第一个细胞系 '{cell_names[0]}' 的特征向量 (前20个值):")
        print(f"    {first_cell_tensor[0, :20].numpy()}")
        print(f"  - 该向量的统计信息: Min={first_cell_tensor.min().item():.4f}, Max={first_cell_tensor.max().item():.4f}, Mean={first_cell_tensor.mean().item():.4f}")
        all_tensors_are_same = True
        if len(all_cell_features_map) > 1:
            for i in range(1, len(cell_names)):
                if not torch.equal(first_cell_tensor, all_cell_features_map[cell_names[i]]):
                    all_tensors_are_same = False; break
        if all_tensors_are_same and len(all_cell_features_map) > 1:
            print("\n  [!!!] 警告：所有细胞系的特征向量都完全相同！")
        else:
            print("\n  [OK] 细胞系特征向量看起来是不同的，这很好。\n")
    else:
        print("  - 错误：细胞系特征图为空，无法进行诊断。")
    print("======================= 诊断代码结束 =======================\n")
    
    try:
        df_smiles = pd.read_csv(args.smiles_file, header=None, names=['drug_name', 'smiles'])
    except FileNotFoundError:
        print(f"错误: SMILES文件未找到 '{args.smiles_file}'")
        return

    smiles_input_list = df_smiles.values.tolist()
    num_drugs = len(smiles_input_list)

    # 模型扫描和加载逻辑 (无变化)
    model_map = {'GIN': OriginalGINConvNet, 'GAT': GATNet, 'GCN': GCNNet, 'GINTransformer': GINConvNet2}
    model_configs = []
    if not os.path.isdir(args.models_dir):
        print(f"错误: 模型目录 '{args.models_dir}' 不存在。"); return
        
    selected_models_to_run = args.models
    print(f"将要运行以下选择的模型: {', '.join(selected_models_to_run)}")

    for filename in os.listdir(args.models_dir):
        if filename.endswith(".model"):
            model_type_found = None
            for mtype in selected_models_to_run:
                if f'_{mtype}_' in filename or filename.startswith(f'{mtype}_') or filename.startswith(f'model_{mtype}_'):
                    model_type_found = mtype; break
            if model_type_found:
                model_configs.append({'type': model_type_found, 'path': os.path.join(args.models_dir, filename)})
    
    if not model_configs:
        print(f"在目录 '{args.models_dir}' 中没有找到与您选择的模型 {selected_models_to_run} 相关的文件。")
        return

    if not all(model_map.values()):
        print("由于模型导入失败，脚本无法继续。")
        return

    device = torch.device(args.cuda_name if torch.cuda.is_available() and args.cuda_name != "cpu" else "cpu")
    print(f"使用设备: {device}")

    num_drug_chunks = math.ceil(num_drugs / DRUG_CHUNK_SIZE)
    # 【【【 新增/修改 】】】: 计算细胞系区块数量
    cell_names_list = list(all_cell_features_map.keys())
    num_cell_chunks = math.ceil(len(cell_names_list) / args.cell_chunk_size)


    # --- 2. 外层模型循环 (无变化) ---
    for config in model_configs:
        model_type = config['type']
        model_path = config['path']

        print(f"\n{'='*25}\n[处理模型]: {model_type} | 路径: {model_path}\n{'='*25}")

        # 跳过已完成区块的检查逻辑 (无变化)
        output_base_name = f"predictions_{os.path.basename(model_path).replace('.model', '')}"
        final_output_file = os.path.join(args.output_dir, f"{output_base_name}.csv")
        if os.path.exists(final_output_file):
            print(f"最终结果文件 '{final_output_file}' 已存在，跳过此模型。")
            continue
        
        try:
            model_instance = model_map[model_type]().to(device)
            model_instance.load_state_dict(torch.load(model_path, map_location=device))
        except Exception as e:
            print(f"加载模型 '{model_path}' 时出错: {e}"); continue
            
        all_results_for_model = []

        # --- 3. 内层药物区块循环 ---
        for i in range(num_drug_chunks):
            chunk_start_idx = i * DRUG_CHUNK_SIZE
            chunk_end_idx = min((i + 1) * DRUG_CHUNK_SIZE, num_drugs)
            current_drug_chunk = smiles_input_list[chunk_start_idx:chunk_end_idx]
            
            print(f"\n--- [模型: {model_type}] 正在处理药物块 {i+1}/{num_drug_chunks} ({len(current_drug_chunk)} 个药物) ---")

            print("  - 预处理药物图 (一次性)...")
            precomputed_drug_graphs = []
            for drug_name, smiles_str in tqdm(current_drug_chunk, desc="  - Pre-computing drugs"):
                if isinstance(smiles_str, str) and smiles_str.strip():
                    drug_x, drug_edge_index = smiles_to_drug_graph_parts(smiles_str)
                    if drug_x is not None:
                        drug_data = Data(x=drug_x, edge_index=drug_edge_index)
                        precomputed_drug_graphs.append((drug_name, smiles_str, drug_data))
            
            if not precomputed_drug_graphs:
                print("  - 警告: 当前药物块内没有有效的SMILES，跳过预测。")
                continue

            # 【【【 新增/修改 】】】: 增加细胞系区块循环
            for j in range(num_cell_chunks):
                cell_chunk_start = j * args.cell_chunk_size
                cell_chunk_end = (j + 1) * args.cell_chunk_size
                current_cell_names = cell_names_list[cell_chunk_start:cell_chunk_end]
                current_cell_map = {name: all_cell_features_map[name] for name in current_cell_names}
                
                print(f"  -- [细胞块 {j+1}/{num_cell_chunks}] 正在处理 {len(current_cell_map)} 个细胞系...")

                dataset = PrecomputedDrugCellDataset(precomputed_drug_graphs, current_cell_map)
                loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=device.type == 'cuda')
                
                try:
                    scaled_predictions = predict_optimized(model_instance, device, loader, model_type)
                except Exception as e_pred:
                    print(f"\n对块 (药物{i+1}, 细胞{j+1}) 预测时发生严重错误: {e_pred}")
                    scaled_predictions = np.full((len(dataset), 1), np.nan)
                
                # --- 结果处理 (只处理当前细胞块) ---
                valid_drugs_info = [(name, smiles) for name, smiles, _ in precomputed_drug_graphs]
                valid_pairs = product(valid_drugs_info, current_cell_map.keys())
                
                for idx, ((drug_name, smiles), cell_name) in enumerate(valid_pairs):
                    if idx < len(scaled_predictions):
                        scaled_pred_val = float(scaled_predictions[idx][0])
                        original_pred_val = unscale_ic50(scaled_pred_val)
                        all_results_for_model.append({
                            'drug_name': drug_name,
                            'smiles': smiles,
                            'cell_line_name': cell_name,
                            'predicted_ic50_scaled': scaled_pred_val,
                            'predicted_ic50_original': original_pred_val
                        })

                del loader, dataset, scaled_predictions; gc.collect()

            del precomputed_drug_graphs; gc.collect()

        # --- 4. 模型所有块处理完后，统一保存结果 ---
        if all_results_for_model:
            print(f"\n模型 {model_type} 的所有预测已完成，正在合并并保存结果...")
            results_df = pd.DataFrame(all_results_for_model)
            cols_order = ['drug_name', 'smiles', 'cell_line_name', 'predicted_ic50_scaled', 'predicted_ic50_original']
            results_df = results_df.reindex(columns=cols_order)
            results_df.to_csv(final_output_file, index=False)
            print(f"最终结果已保存至 '{final_output_file}'")
        else:
            print(f"模型 {model_type} 未生成任何预测结果。")
            
        del model_instance, all_results_for_model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    print(f"\n所有模型处理完毕！预测完成！")

if __name__ == "__main__":
    if OriginalGINConvNet is None:
        print("\n脚本因模型导入失败而终止。请检查您的项目结构和Python环境。")
    else:
        main()