# predict_all.py (最终完美版)
import torch
import joblib
import numpy as np
import pandas as pd
import argparse
import os
import importlib.util
import time
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.data import Data, Dataset as PyGDataset, Batch
from torch_geometric.utils import add_self_loops
from tqdm import tqdm
import platform
import sys
import re
# --- Dependencies ---
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

try:
    from loader import mol_to_graph_data_obj_complex
except ImportError:
    print("ERROR: loader.py not found."); exit()
try:
    from Model_GNN import MolGNet
except ImportError:
    print("ERROR: Model_GNN.py not found."); exit()
try:
    from Model import Predictor, setup_seed
    print("Successfully imported Predictor, setup_seed from Model.py")
except ImportError as e:
    print(f"ERROR: Failed to import from Model.py. Details: {e}"); exit()

# --- Utility Classes ---
class Self_loop:
    def __call__(self, data):
        device = data.x.device; edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.num_nodes); data.edge_index = edge_index
        self_loop_attr = torch.tensor([0, 5, 8, 10, 12], device=device).long().repeat(data.num_nodes, 1)
        if hasattr(data, 'edge_attr') and data.edge_attr is not None: data.edge_attr = torch.cat((data.edge_attr, self_loop_attr), dim=0)
        else: data.edge_attr = self_loop_attr
        return data

class Add_seg_id:
    def __call__(self, data):
        device = data.x.device; data.edge_seg = torch.zeros(data.num_edges, dtype=torch.long, device=device); data.node_seg = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
        return data

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Predict drug IC50 values.")
parser.add_argument('--input_drugs_csv', type=str, default='./test/drug_sample.csv', help="CSV for new drugs.")
parser.add_argument('--gene_expression_file', type=str, default='./test/gene_sample.csv', help="CSV for new cell lines.")
parser.add_argument('--output_csv', type=str, default='./result/DIPK.csv', help="Output CSV path.")
parser.add_argument('--model_path', type=str, default='./result/Train.pkl', help="Path to the main Predictor model.")
parser.add_argument('--train_config_path', type=str, default='TrainConfig.py', help="TrainConfig.py path.")
parser.add_argument('--data_config_path', type=str, default='DataConfig.py', help="DataConfig.py path.")
parser.add_argument('--molgnet_model_path', type=str, default='./Data/MolGNet.pt', help="Path to MolGNet.pt.")
parser.add_argument('--bionic_dict_path', type=str, default='../Dataset/BIONIC_dict.pkl', help="Path to BIONIC_dict.pkl.")
parser.add_argument('--canonical_gene_list_path', type=str, default='../Dataset/exp.txt', help="Path to canonical gene list.")
parser.add_argument('--batch_size', type=int, default=None, help="Batch size for prediction.")
parser.add_argument('--num_workers', type=int, default=0, help="Number of CPU workers for data loading.")
parser.add_argument('--seed', type=int, default=None, help="Random seed.")
parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'], help="Device.")
args = parser.parse_args()

# --- Dynamic Configuration Loading ---
def import_from_path(module_name, file_path):
    abs_file_path = os.path.abspath(file_path); spec = importlib.util.spec_from_file_location(module_name, abs_file_path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module

try:
    DataConfig = import_from_path("DataConfig", args.data_config_path)
    import Data as data_module
    TrainConfig = import_from_path("TrainConfig", args.train_config_path)
except Exception as e:
    print(f"Error during config/module loading: {e}"); exit()

# --- PyG Dataset Class ---
class DrugCellDataset(PyGDataset):
    def __init__(self, data_list): super(DrugCellDataset, self).__init__(); self.data_list = data_list
    def len(self): return len(self.data_list)
    def get(self, idx): return self.data_list[idx]

def clean_name(name):
    # This function now robustly handles potential non-string inputs from pandas
    if not isinstance(name, str):
        name = str(name)
    # The original cleaning logic, now safer
    cleaned = re.sub(r'\(.*\)', '', name)
    return cleaned.strip().lower()


# --- Cell & Drug Preprocessing Functions ---
def preprocess_new_cell_data(gene_expr_path, canonical_gene_path, bionic_dict_path, gene_add_num, expected_gene_dim):
    print("\n--- Preprocessing New Cell Lines ---")

    # 【关键修正 1】: 正确读取标准基因列表文件
    try:
        # 使用pandas读取制表符分隔的文件，并明确指定第一列为GENE_SYMBOLS
        canonical_df = pd.read_csv(os.path.abspath(canonical_gene_path), sep='\t', header=0)
        # 假设第一列是 'GENE_SYMBOLS'
        canonical_gene_list_raw = canonical_df.iloc[:, 0].tolist()[:expected_gene_dim]
    except Exception as e:
        print(f"错误: 读取标准基因列表 '{canonical_gene_path}' 失败。请确保它是制表符分隔且有表头。错误: {e}")
        # 如果读取失败，尝试原始的行读取方法作为备用
        print("尝试使用备用行读取方法...")
        with open(os.path.abspath(canonical_gene_path), 'r', encoding='gbk', errors='ignore') as f:
            canonical_gene_list_raw = [line.split('\t')[0] for line in f if line.strip()][:expected_gene_dim]
            if canonical_gene_list_raw[0].lower() in ['gene_symbols', 'gene_symbol']:
                canonical_gene_list_raw.pop(0) # 移除表头

    canonical_gene_list_cleaned = [clean_name(name) for name in canonical_gene_list_raw]
    canonical_gene_set = set(canonical_gene_list_cleaned)
    # 创建从清理后的基因名到其在标准列表中原始索引的映射
    gene_to_canonical_index = {name: i for i, name in enumerate(canonical_gene_list_cleaned)}


    bionic_dict = joblib.load(os.path.abspath(bionic_dict_path))
    new_cells_df = pd.read_csv(os.path.abspath(gene_expr_path), index_col=0)
    original_user_genes_cleaned = set(clean_name(col) for col in new_cells_df.columns)
    new_cells_df.columns = [clean_name(col) for col in new_cells_df.columns]

    intersection_count = len(canonical_gene_set.intersection(original_user_genes_cleaned))
    print("\n" + "="*20 + " 【基因对齐诊断】 " + "="*20)
    print(f"标准基因列表中的基因数: {len(canonical_gene_set)}")
    print(f"您的数据文件中的基因数: {len(original_user_genes_cleaned)}")
    print(f"成功对齐（交集）的基因数: {intersection_count}")
    if intersection_count == 0:
        print("警告: 您的基因名与标准列表没有任何交集！将打印示例以供比对：")
        print("--- 标准列表基因名示例 (前10个): ---"); print(canonical_gene_list_cleaned[:10])
        print("--- 您的数据基因名示例 (前10个): ---"); print(list(original_user_genes_cleaned)[:10])
    print("="*58 + "\n")
    
    aligned_df = new_cells_df.reindex(columns=canonical_gene_list_cleaned, fill_value=0.0)
    
    cell_features = {}
    for cell_name, row in tqdm(aligned_df.iterrows(), total=len(aligned_df), desc="Generating Cell Features"):
        gef_vector = torch.tensor(row.values, dtype=torch.float32)
        _, top_gene_indices = torch.sort(gef_vector, descending=True)
        feature_sum = torch.zeros(512, dtype=torch.float32)
        k = 0
        for j in range(gene_add_num):
            gene_canonical_index = int(top_gene_indices[j])
            if gene_canonical_index in bionic_dict:
                k += 1; feature_sum += torch.tensor(bionic_dict[gene_canonical_index], dtype=torch.float32)
        bnf_vector = feature_sum / k if k > 0 else torch.zeros(512, dtype=torch.float32)
        cell_features[str(cell_name)] = {'GEF': gef_vector, 'BNF': bnf_vector}
    return cell_features

drug_graph_cache = {}
def get_drug_graph(drug_id, smiles_str):
    if drug_id in drug_graph_cache: return drug_graph_cache[drug_id]
    if smiles_str:
        try:
            mol = Chem.MolFromSmiles(smiles_str)
            if mol: drug_graph_cache[drug_id] = mol_to_graph_data_obj_complex(mol); return drug_graph_cache[drug_id]
        except Exception: pass
    return None

# --- Setup & Model Loading ---
setup_seed(args.seed if args.seed is not None else TrainConfig.seed)
DEVICE = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Using main device: {DEVICE}")
batch_size = args.batch_size if args.batch_size is not None else getattr(TrainConfig, 'batch_size', 512)
num_workers = args.num_workers if platform.system() == "Linux" else 0
if num_workers > 0 and platform.system() != "Linux":
    print(f"警告: 在 {platform.system()} 系统上将 num_workers 强制设置为0。")

EXPECTED_GENE_FEATURE_DIM = getattr(data_module, 'features_dim_gene', 19221)
EXPECTED_BIONIC_FEATURE_DIM = getattr(data_module, 'features_dim_bionic', 512)

molgnet_featurizer = None; g_self_loop = Self_loop(); g_add_seg_id = Add_seg_id()
if os.path.exists(args.molgnet_model_path):
    try:
        molgnet_featurizer = MolGNet(num_layer=5, emb_dim=768, heads=12, num_message_passing=3, drop_ratio=0.0)
        molgnet_featurizer.load_state_dict(torch.load(args.molgnet_model_path, map_location=DEVICE))
        molgnet_featurizer.to(DEVICE).eval(); print("MolGNet featurizer loaded successfully.")
    except Exception as e: print(f"Error loading MolGNet featurizer: {e}"); molgnet_featurizer = None

print("Loading main predictor model...")
try:
    _, model_state, _ = joblib.load(args.model_path)
    predictor_model = Predictor(TrainConfig.embedding_dim, TrainConfig.heads, TrainConfig.fc_layer_num, TrainConfig.fc_layer_dim, TrainConfig.dropout_rate)
    predictor_model.load_state_dict(model_state.state_dict() if hasattr(model_state, 'state_dict') else model_state)
    predictor_model.to(DEVICE).eval(); print("Main predictor model loaded successfully.")
except Exception as e: print(f"Error loading main predictor model: {e}"); exit()

# --- MAIN PROCESSING LOGIC ---
t_overall_start = time.time()
cell_features_dict = preprocess_new_cell_data(args.gene_expression_file, args.canonical_gene_list_path, args.bionic_dict_path, getattr(DataConfig, 'gene_add_num', 500), EXPECTED_GENE_FEATURE_DIM)
if not cell_features_dict: print("No valid cell lines processed. Exiting."); exit()

print("\n--- Preprocessing New Drugs ---")
input_drugs_df = pd.read_csv(args.input_drugs_csv, header=None, names=['drug_id', 'smiles'])
all_drugs_initial = [{'drug_id': str(r['drug_id']).strip(), 'graph_obj': get_drug_graph(str(r['drug_id']).strip(), str(r['smiles']).strip() if pd.notna(r['smiles']) else None)} for _, r in tqdm(input_drugs_df.iterrows(), total=len(input_drugs_df), desc="Creating Drug Graphs")]
all_drugs_initial = [d for d in all_drugs_initial if d['graph_obj'] is not None]

processed_drugs = []
if molgnet_featurizer and all_drugs_initial:
    print(f"\nBatch-processing {len(all_drugs_initial)} drugs with MolGNet...")
    drug_loader = PyGDataLoader([d['graph_obj'] for d in all_drugs_initial], batch_size=batch_size, shuffle=False, num_workers=num_workers)
    final_features = [];
    with torch.no_grad():
        for batch in tqdm(drug_loader, desc="MolGNet Batches"):
            batch = batch.to(DEVICE); batch = g_self_loop(batch); batch = g_add_seg_id(batch)
            final_features.append(molgnet_featurizer(batch).cpu())
    all_features_tensor = torch.cat(final_features, dim=0)
    slices = Batch.from_data_list([d['graph_obj'] for d in all_drugs_initial]).ptr
    for i, drug_data in enumerate(all_drugs_initial):
        drug_data['node_features'] = all_features_tensor[slices[i]:slices[i + 1]]; processed_drugs.append(drug_data)
else:
    for drug_data in all_drugs_initial:
        drug_data['node_features'] = drug_data['graph_obj'].x; processed_drugs.append(drug_data)
if not processed_drugs: print("No valid drugs processed. Exiting."); exit()

print("\n--- Starting Prediction ---")
all_results = []
for drug_data in tqdm(processed_drugs, desc="Predicting (Drug by Drug)"):
    current_drug_pairs_data, current_drug_identifiers = [], []
    for cell_name, cell_feats in cell_features_dict.items():
        current_drug_pairs_data.append(Data(x=drug_data['node_features'], edge_index=drug_data['graph_obj'].edge_index, edge_attr=getattr(drug_data['graph_obj'], 'edge_attr', None), GEF=cell_feats['GEF'], BNF=cell_feats['BNF']))
        current_drug_identifiers.append({'input_drug_id': drug_data['drug_id'], 'resolved_drug_name': drug_data['drug_id'], 'cell_line_name': cell_name})

    if not current_drug_pairs_data: continue
    current_drug_loader = PyGDataLoader(DrugCellDataset(current_drug_pairs_data), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    predictions_for_this_drug = []
    with torch.no_grad():
        for batch in current_drug_loader:
            pyg_batch_for_model = batch.to(DEVICE)
            node_features_for_model = pyg_batch_for_model.x
            
            # 【关键修正 2】: 确保药物特征总是二维的，以修复IndexError
            if node_features_for_model.dim() == 1:
                node_features_for_model = node_features_for_model.unsqueeze(0)
            
            num_graphs_in_batch = pyg_batch_for_model.num_graphs
            gene_ft_for_model = pyg_batch_for_model.GEF.view(num_graphs_in_batch, EXPECTED_GENE_FEATURE_DIM)
            bionic_ft_for_model = pyg_batch_for_model.BNF.view(num_graphs_in_batch, EXPECTED_BIONIC_FEATURE_DIM)
            predictions = predictor_model(node_features_for_model, pyg_batch_for_model, gene_ft_for_model, bionic_ft_for_model)
            predictions_for_this_drug.extend(torch.squeeze(predictions).cpu().tolist() if predictions.numel() > 1 else [predictions.item()])

    for i, identifier in enumerate(current_drug_identifiers):
        if i < len(predictions_for_this_drug):
            identifier['predicted_ic50'] = predictions_for_this_drug[i]; all_results.append(identifier)
        else: print(f"Warning: Prediction count mismatch for drug {drug_data['drug_id']}.")

t_overall_end = time.time()
print(f"\n--- Overall Summary ---"); print(f"Total predictions generated: {len(all_results)}"); print(f"Total script execution time: {t_overall_end - t_overall_start:.2f}s")
if all_results:
    results_df = pd.DataFrame(all_results); results_df = results_df[['input_drug_id', 'resolved_drug_name', 'cell_line_name', 'predicted_ic50']]
    output_dir = os.path.dirname(args.output_csv);
    if output_dir: os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(args.output_csv, index=False); print(f"\nAll predictions saved to {args.output_csv}")
else: print("\nNo predictions were generated.")
print("\nScript finished.")