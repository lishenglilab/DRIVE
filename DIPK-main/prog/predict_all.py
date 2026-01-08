# predict_all_universal.py
# 自动适配 GPU 和 CPU 环境的 DIPK 预测脚本
import torch
import joblib
import numpy as np
import pandas as pd
import argparse
import os
import importlib.util
import time
import platform
import sys
import re
import random
import io
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.data import Data, Dataset as PyGDataset, Batch
from torch_geometric.utils import add_self_loops
from tqdm import tqdm
from optuna.samplers import TPESampler
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

# --- Imports ---
try:
    from loader import mol_to_graph_data_obj_complex
except ImportError:
    print("ERROR: loader.py not found."); exit(1)
try:
    from Model_GNN import MolGNet
except ImportError:
    print("ERROR: Model_GNN.py not found."); exit(1)
try:
    from Model_MHA import MultiHeadAttentionLayer
except ImportError:
    print("ERROR: Model_MHA.py not found."); exit(1)

# ==========================================
# === 核心工具：Monkey Patch 加载器 (仅 CPU 模式使用) ===
# ==========================================
def joblib_load_cpu(path):
    print(f"DEBUG: CPU Mode detected. Using Monkey-Patching to force load: {path}")
    original_load_from_bytes = torch.storage._load_from_bytes
    def patched_load_from_bytes(b):
        return torch.load(io.BytesIO(b), map_location='cpu')
    try:
        torch.storage._load_from_bytes = patched_load_from_bytes
        return joblib.load(path)
    finally:
        torch.storage._load_from_bytes = original_load_from_bytes

# ==========================================
# === 模型类定义 (内嵌防止 import 导致的全局 GPU 变量问题) ===
# ==========================================
features_dim_gene = 19221
features_dim_bionic = 512

def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    sampler = TPESampler(seed=seed)
    return sampler

class AttentionLayer(nn.Module):
    def __init__(self, heads, device):
        super(AttentionLayer, self).__init__()
        self.fc_layer_0 = nn.Linear(features_dim_gene, 768)
        self.fc_layer_1 = nn.Linear(features_dim_bionic, 768)
        self.attention_0 = MultiHeadAttentionLayer(hid_dim=768, n_heads=1, dropout=0.3, device=device)
        self.attention_1 = MultiHeadAttentionLayer(hid_dim=768, n_heads=1, dropout=0.3, device=device)

    def forward(self, x, g, gene, bionic):
        gene = F.relu(self.fc_layer_0(gene))
        bionic = F.relu(self.fc_layer_1(bionic))
        x = to_dense_batch(x, g.batch)
        query_0 = torch.unsqueeze(gene, 1)
        query_1 = torch.unsqueeze(bionic, 1)
        key = x[0]
        value = x[0]
        mask = torch.unsqueeze(torch.unsqueeze(x[1], 1), 1)
        x_att = self.attention_0(query_0, key, value, mask)
        x = torch.squeeze(x_att[0])
        x_att = self.attention_1(query_1, key, value, mask)
        x += torch.squeeze(x_att[0])
        return x

class DenseLayers(nn.Module):
    def __init__(self, heads, fc_layer_num, fc_layer_dim, dropout_rate):
        super(DenseLayers, self).__init__()
        self.fc_layer_num = fc_layer_num
        self.fc_layer_0 = nn.Linear(features_dim_gene, 512)
        self.fc_layer_1 = nn.Linear(features_dim_bionic, 512)
        self.fc_input = nn.Linear(768 + 512, 768 + 512)
        self.fc_layers = torch.nn.Sequential(
            nn.Linear(768 + 512, 512),
            nn.Linear(512, fc_layer_dim[0]),
            nn.Linear(fc_layer_dim[0], fc_layer_dim[1]),
            nn.Linear(fc_layer_dim[1], fc_layer_dim[2]),
            nn.Linear(fc_layer_dim[2], fc_layer_dim[3]),
            nn.Linear(fc_layer_dim[3], fc_layer_dim[4]),
            nn.Linear(fc_layer_dim[4], fc_layer_dim[5])
        )
        self.dropout_layers = torch.nn.ModuleList(
            [nn.Dropout(p=dropout_rate) for _ in range(fc_layer_num)]
        )
        self.fc_output = nn.Linear(fc_layer_dim[fc_layer_num - 2], 1)

    def forward(self, x, gene, bionic):
        gene = F.relu(self.fc_layer_0(gene))
        bionic = F.relu(self.fc_layer_1(bionic))
        if x.dim() == 1: x = x.unsqueeze(0)
        if gene.dim() == 1: gene = gene.unsqueeze(0)
        if bionic.dim() == 1: bionic = bionic.unsqueeze(0)
        f = torch.cat((x, gene + bionic), 1)
        f = F.relu(self.fc_input(f))
        for layer_index in range(self.fc_layer_num):
            f = F.relu(self.fc_layers[layer_index](f))
            f = self.dropout_layers[layer_index](f)
        f = self.fc_output(f)
        return f

class Predictor(nn.Module):
    def __init__(self, embedding_dim, heads, fc_layer_num, fc_layer_dim, dropout_rate, device):
        super(Predictor, self).__init__()
        self.attention_layer = AttentionLayer(heads, device=device)
        self.dense_layers = DenseLayers(heads, fc_layer_num, fc_layer_dim, dropout_rate)

    def forward(self, x, g, gene, bionic):
        x = self.attention_layer(x, g, gene, bionic)
        f = self.dense_layers(x, gene, bionic)
        return f

# --- Helpers ---
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

class DrugCellDataset(PyGDataset):
    def __init__(self, data_list): super(DrugCellDataset, self).__init__(); self.data_list = data_list
    def len(self): return len(self.data_list)
    def get(self, idx): return self.data_list[idx]

def import_from_path(module_name, file_path):
    abs_file_path = os.path.abspath(file_path); spec = importlib.util.spec_from_file_location(module_name, abs_file_path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module

def clean_name(name):
    if not isinstance(name, str): name = str(name)
    cleaned = re.sub(r'\(.*\)', '', name); return cleaned.strip().lower()

def preprocess_new_cell_data(gene_expr_path, canonical_gene_path, bionic_dict_path, gene_add_num, expected_gene_dim):
    print("\n--- Preprocessing New Cell Lines ---")
    try:
        canonical_df = pd.read_csv(os.path.abspath(canonical_gene_path), sep='\t', header=0)
        canonical_gene_list_raw = canonical_df.iloc[:, 0].tolist()[:expected_gene_dim]
    except Exception:
        with open(os.path.abspath(canonical_gene_path), 'r', encoding='gbk', errors='ignore') as f:
            canonical_gene_list_raw = [line.split('\t')[0] for line in f if line.strip()][:expected_gene_dim]
            if canonical_gene_list_raw and canonical_gene_list_raw[0].lower() in ['gene_symbols', 'gene_symbol']:
                canonical_gene_list_raw.pop(0)
    canonical_gene_list_cleaned = [clean_name(name) for name in canonical_gene_list_raw]
    bionic_dict = joblib.load(os.path.abspath(bionic_dict_path))
    new_cells_df = pd.read_csv(os.path.abspath(gene_expr_path), index_col=0)
    new_cells_df.columns = [clean_name(col) for col in new_cells_df.columns]
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

# --- Main Logic ---
def main(args):
    setup_seed(args.seed)
    
    # === 自动设备检测逻辑 ===
    if args.device:
        device_str = args.device
    else:
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    DEVICE = torch.device(device_str)
    print(f"--- Detected Environment: Using {DEVICE} ---")
    
    num_workers = args.num_workers
    if platform.system() != "Linux" and num_workers > 0:
        num_workers = 0

    EXPECTED_GENE_FEATURE_DIM = getattr(data_module, 'features_dim_gene', 19221)
    EXPECTED_BIONIC_FEATURE_DIM = getattr(data_module, 'features_dim_bionic', 512)

    molgnet_featurizer = None; g_self_loop = Self_loop(); g_add_seg_id = Add_seg_id()
    if os.path.exists(args.molgnet_model_path):
        try:
            molgnet_featurizer = MolGNet(num_layer=5, emb_dim=768, heads=12, num_message_passing=3, drop_ratio=0.0)
            map_loc = 'cpu' if device_str == 'cpu' else None
            molgnet_featurizer.load_state_dict(torch.load(args.molgnet_model_path, map_location=map_loc))
            molgnet_featurizer.to(DEVICE).eval(); print("MolGNet featurizer loaded successfully.")
        except Exception as e: print(f"Error loading MolGNet featurizer: {e}"); molgnet_featurizer = None

    print("Loading main predictor model...")
    try:
        # === 核心加载分支 ===
        if device_str == 'cpu':
            # CPU模式：使用 Monkey Patch 修复 joblib
            loaded_obj = joblib_load_cpu(args.model_path)
        else:
            # GPU模式：使用标准加载
            loaded_obj = joblib.load(args.model_path)
        
        if isinstance(loaded_obj, tuple) and len(loaded_obj) >= 2:
            model_state = loaded_obj[1] 
        else:
            model_state = loaded_obj

        predictor_model = Predictor(TrainConfig.embedding_dim, TrainConfig.heads, TrainConfig.fc_layer_num, TrainConfig.fc_layer_dim, TrainConfig.dropout_rate, device=DEVICE)
        
        if hasattr(model_state, 'state_dict'):
            predictor_model.load_state_dict(model_state.state_dict())
        elif isinstance(model_state, dict):
            predictor_model.load_state_dict(model_state)
        
        predictor_model.to(DEVICE).eval()
        print("Main predictor model loaded successfully.")
    except Exception as e: 
        print(f"CRITICAL ERROR loading main model: {e}")
        exit(1)

    t_overall_start = time.time()
    
    cell_features_dict = preprocess_new_cell_data(args.gene_expression_file, args.canonical_gene_list_path, args.bionic_dict_path, getattr(DataConfig, 'gene_add_num', 500), EXPECTED_GENE_FEATURE_DIM)
    if not cell_features_dict: print("No valid cell lines processed. Exiting."); exit(1)

    print("\n--- Preprocessing New Drugs ---")
    try:
        input_drugs_df = pd.read_csv(args.input_drugs_csv, header=None, names=['drug_id', 'smiles'])
    except FileNotFoundError:
        print(f"ERROR: Drug input file not found: {args.input_drugs_csv}"); exit(1)

    all_drugs_graphs = []
    for _, r in tqdm(input_drugs_df.iterrows(), total=len(input_drugs_df), desc="Creating Drug Graphs"):
        drug_id = str(r['drug_id']).strip()
        smiles = str(r['smiles']).strip() if pd.notna(r['smiles']) else None
        if smiles:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol: all_drugs_graphs.append({'drug_id': drug_id, 'graph_obj': mol_to_graph_data_obj_complex(mol)})
            except Exception: pass
    
    if not all_drugs_graphs: print("No valid drugs to process. Exiting."); exit(1)

    print(f"\n--- Extracting drug features with MolGNet for {len(all_drugs_graphs)} drugs ---")
    if molgnet_featurizer:
        drug_loader = PyGDataLoader([d['graph_obj'] for d in all_drugs_graphs], batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
        final_features = []
        with torch.no_grad():
            for batch in tqdm(drug_loader, desc=f"MolGNet Batches ({DEVICE})"):
                batch = batch.to(DEVICE); batch = g_self_loop(batch); batch = g_add_seg_id(batch)
                final_features.append(molgnet_featurizer(batch).cpu())
        all_features_tensor = torch.cat(final_features, dim=0)
        slices = Batch.from_data_list([d['graph_obj'] for d in all_drugs_graphs]).ptr
        for i, drug_data in enumerate(all_drugs_graphs):
            drug_data['node_features'] = all_features_tensor[slices[i]:slices[i + 1]]
    else:
        print("ERROR: MolGNet featurizer is required for this model but failed to load. Cannot proceed.")
        exit(1)

    print("\n--- Starting Prediction ---")
    all_pairs_data = []
    all_identifiers = []
    for drug_data in tqdm(all_drugs_graphs, desc="Preparing all drug-cell pairs"):
        for cell_name, cell_feats in cell_features_dict.items():
            all_pairs_data.append(Data(
                x=drug_data['node_features'], 
                edge_index=drug_data['graph_obj'].edge_index, 
                edge_attr=getattr(drug_data['graph_obj'], 'edge_attr', None), 
                GEF=cell_feats['GEF'], 
                BNF=cell_feats['BNF']
            ))
            all_identifiers.append({'drug_id': drug_data['drug_id'], 'cell_line_name': cell_name})

    if not all_pairs_data: print("No drug-cell pairs to predict. Exiting."); exit(1)

    all_pairs_loader = PyGDataLoader(DrugCellDataset(all_pairs_data), batch_size=args.batch_size, shuffle=False, num_workers=num_workers)
    
    all_predictions = []
    with torch.no_grad():
        for batch in tqdm(all_pairs_loader, desc=f"Predicting Batches ({DEVICE})"):
            pyg_batch = batch.to(DEVICE)
            node_ft = pyg_batch.x.float()
            if node_ft.dim() == 1: node_ft = node_ft.unsqueeze(0)
            
            num_graphs = pyg_batch.num_graphs
            gene_ft = pyg_batch.GEF.view(num_graphs, EXPECTED_GENE_FEATURE_DIM)
            bionic_ft = pyg_batch.BNF.view(num_graphs, EXPECTED_BIONIC_FEATURE_DIM)
            
            predictions = predictor_model(node_ft, pyg_batch, gene_ft, bionic_ft)
            all_predictions.extend(torch.squeeze(predictions).cpu().tolist() if predictions.numel() > 1 else [predictions.item()])

    results_df = pd.DataFrame(all_identifiers)
    results_df['predicted_ic50'] = all_predictions
    
    t_overall_end = time.time()
    print(f"\n--- Overall Summary ---"); print(f"Total predictions generated: {len(results_df)}"); print(f"Total script execution time: {t_overall_end - t_overall_start:.2f}s")
    
    if not results_df.empty:
        try:
            output_dir = os.path.dirname(args.output_csv)
            if output_dir: os.makedirs(output_dir, exist_ok=True)
            results_df.to_csv(args.output_csv, index=False)
            print(f"\nAll predictions saved to {args.output_csv}")
        except Exception as e:
            print(f"\nERROR saving results: {e}"); exit(1)
    else:
        print("\nNo predictions were generated.")
    
    print("\nScript finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DIPK Universal (CPU/GPU)")
    parser.add_argument('--input_drugs_csv', type=str, required=True)
    parser.add_argument('--gene_expression_file', type=str, required=True)
    parser.add_argument('--output_csv', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--train_config_path', type=str, required=True)
    parser.add_argument('--data_config_path', type=str, required=True)
    parser.add_argument('--molgnet_model_path', type=str, required=True)
    parser.add_argument('--bionic_dict_path', type=str, required=True)
    parser.add_argument('--canonical_gene_list_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default=None, help="Force 'cpu' or 'cuda'. Leave empty to auto-detect.")
    
    args = parser.parse_args()
    
    try:
        DataConfig = import_from_path("DataConfig", args.data_config_path)
        import Data as data_module
        TrainConfig = import_from_path("TrainConfig", args.train_config_path)
    except Exception as e:
        print(f"FATAL ERROR: Failed to load configuration files. Error: {e}")
        exit(1)
        
    main(args)
