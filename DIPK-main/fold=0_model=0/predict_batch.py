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
from functools import partial
# --- Dependencies ---
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

try:
    from joblib import Parallel, delayed
except ImportError:
    print("ERROR: joblib not found. Please run 'pip install joblib'."); exit()
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

# --- Utility Classes (无变化) ---
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

# --- PyG Dataset Class (无变化) ---
class DrugCellDataset(PyGDataset):
    def __init__(self, data_list): super(DrugCellDataset, self).__init__(); self.data_list = data_list
    def len(self): return len(self.data_list)
    def get(self, idx): return self.data_list[idx]

# --- 动态配置加载 (无变化) ---
def import_from_path(module_name, file_path):
    abs_file_path = os.path.abspath(file_path); spec = importlib.util.spec_from_file_location(module_name, abs_file_path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module

def clean_name(name):
    if not isinstance(name, str): name = str(name)
    cleaned = re.sub(r'\(.*\)', '', name); return cleaned.strip().lower()

# --- 【【【 新增：用于多任务并行的药物处理函数 】】】 ---
def process_single_drug(drug_row):
    """
    处理单行药物数据（drug_id, smiles），返回包含图对象的字典。
    这是一个独立的顶层函数，以便joblib可以序列化它。
    """
    drug_id, smiles_str = drug_row
    drug_id = str(drug_id).strip()
    smiles_str = str(smiles_str).strip() if pd.notna(smiles_str) else None
    
    if not smiles_str:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles_str)
        if mol:
            graph_obj = mol_to_graph_data_obj_complex(mol)
            return {'drug_id': drug_id, 'graph_obj': graph_obj}
    except Exception:
        # 捕获 RDKit 或其他处理中的任何错误
        pass
    return None

# --- Cell & Drug Preprocessing Functions (部分修改) ---
def preprocess_new_cell_data(gene_expr_path, canonical_gene_path, bionic_dict_path, gene_add_num, expected_gene_dim):
    # (此函数内部逻辑无变化，仅为保持完整性而保留)
    print("\n--- Preprocessing New Cell Lines ---")
    try:
        canonical_df = pd.read_csv(os.path.abspath(canonical_gene_path), sep='\t', header=0)
        canonical_gene_list_raw = canonical_df.iloc[:, 0].tolist()[:expected_gene_dim]
    except Exception as e:
        print(f"错误: 读取标准基因列表 '{canonical_gene_path}' 失败。错误: {e}")
        with open(os.path.abspath(canonical_gene_path), 'r', encoding='gbk', errors='ignore') as f:
            canonical_gene_list_raw = [line.split('\t')[0] for line in f if line.strip()][:expected_gene_dim]
            if canonical_gene_list_raw[0].lower() in ['gene_symbols', 'gene_symbol']:
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

# --- 【【【 主逻辑已重构为 `main` 函数 】】】 ---
def main(args):
    # --- Setup & Model Loading (与之前类似) ---
    setup_seed(args.seed if args.seed is not None else TrainConfig.seed)
    DEVICE = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"--- Using main device: {DEVICE} ---")
    
    # 根据平台确定 num_workers 的安全值
    num_workers = args.num_workers
    if platform.system() != "Linux" and num_workers > 0:
        print(f"警告: 在 {platform.system()} 系统上将 num_workers 强制设置为0以避免问题。")
        num_workers = 0

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

    t_overall_start = time.time()
    
    # --- 细胞系预处理 (只执行一次) ---
    cell_features_dict = preprocess_new_cell_data(args.gene_expression_file, args.canonical_gene_list_path, args.bionic_dict_path, getattr(DataConfig, 'gene_add_num', 500), EXPECTED_GENE_FEATURE_DIM)
    if not cell_features_dict: print("No valid cell lines processed. Exiting."); exit()
    cell_lines_to_predict = list(cell_features_dict.keys())
    
    # --- 分块预测主循环 ---
    print("\n--- Starting Chunked Prediction ---")
    try:
        chunk_iterator = pd.read_csv(
            args.input_drugs_csv, header=None, names=['drug_id', 'smiles'],
            chunksize=args.drug_chunk_size, low_memory=True, engine='c'
        )
    except FileNotFoundError:
        print(f"错误: 药物文件未找到: {args.input_drugs_csv}"); exit(1)
    
    output_base, output_ext = os.path.splitext(args.output_csv)
    
    # --- 断点续跑逻辑 ---
    start_from_chunk = args.start_chunk
    if start_from_chunk <= 0:  # 自动检测模式
        output_dir = os.path.dirname(output_base) or '.'
        existing_files = [f for f in os.listdir(output_dir) if f.startswith(os.path.basename(output_base) + "_") and f.endswith(output_ext)]
        last_completed_chunk = 0
        for f in existing_files:
            num_str = re.search(r'_(\d+)\.csv$', f)
            if num_str: last_completed_chunk = max(last_completed_chunk, int(num_str.group(1)))
        start_from_chunk = last_completed_chunk + 1
        print(f"\n--- 自动检测到上次已完成到大块 {last_completed_chunk}。将从大块 {start_from_chunk} 开始继续... ---")
    else:
        print(f"\n--- 用户指定从大块 {start_from_chunk} 开始运行... ---")
        
    # --- 循环处理每个大块 ---
    for chunk_num, drug_chunk_df in enumerate(chunk_iterator, 1):
        if chunk_num < start_from_chunk:
            print(f"快速跳过已处理的大块 {chunk_num}...")
            continue

        print(f"\n" + "="*20 + f" Processing Chunk {chunk_num} " + "="*20)
        t_chunk_start = time.time()

        # 1. CPU并行预处理：SMILES -> PyG图对象 (多任务加速)
        print(f"Step 1: Preprocessing {len(drug_chunk_df)} drugs in parallel using {num_workers if num_workers > 0 else '1'} CPU core(s)...")
        # 将DataFrame转换为元组列表以传递给并行函数
        drug_rows = [tuple(x) for x in drug_chunk_df[['drug_id', 'smiles']].to_numpy()]
        # 使用joblib进行并行处理
        processed_drugs_graphs = Parallel(n_jobs=num_workers, backend="multiprocessing")(
            delayed(process_single_drug)(row) for row in tqdm(drug_rows, desc="Creating Drug Graphs (CPU)")
        )
        # 过滤掉处理失败的结果 (返回None的)
        processed_drugs_graphs = [d for d in processed_drugs_graphs if d is not None]
        
        if not processed_drugs_graphs:
            print("当前大块没有有效的药物可供处理，跳过。")
            continue
        
        # 2. GPU批处理：图对象 -> MolGNet特征
        print(f"Step 2: Extracting drug features with MolGNet on GPU for {len(processed_drugs_graphs)} valid drugs...")
        if molgnet_featurizer:
            drug_loader = PyGDataLoader([d['graph_obj'] for d in processed_drugs_graphs], batch_size=args.gpu_batch_size, shuffle=False)
            final_features = []
            with torch.no_grad():
                for batch in tqdm(drug_loader, desc="MolGNet Batches (GPU)"):
                    batch = batch.to(DEVICE); batch = g_self_loop(batch); batch = g_add_seg_id(batch)
                    final_features.append(molgnet_featurizer(batch).cpu())
            all_features_tensor = torch.cat(final_features, dim=0)
            slices = Batch.from_data_list([d['graph_obj'] for d in processed_drugs_graphs]).ptr
            for i, drug_data in enumerate(processed_drugs_graphs):
                drug_data['node_features'] = all_features_tensor[slices[i]:slices[i + 1]]
        else:
            for drug_data in processed_drugs_graphs:
                drug_data['node_features'] = drug_data['graph_obj'].x

        # 3. GPU小批量预测
        print(f"Step 3: Starting prediction in small batches of {args.small_batch_size} drugs...")
        all_results_for_chunk = []
        num_small_batches = (len(processed_drugs_graphs) + args.small_batch_size - 1) // args.small_batch_size
        
        for i in tqdm(range(0, len(processed_drugs_graphs), args.small_batch_size), total=num_small_batches, desc="Prediction Batches (GPU)"):
            small_batch_drugs = processed_drugs_graphs[i : i + args.small_batch_size]
            
            # 为当前小批量的所有“药物-细胞”对创建数据
            batch_pairs_data, batch_identifiers = [], []
            for drug_data in small_batch_drugs:
                for cell_name in cell_lines_to_predict:
                    cell_feats = cell_features_dict[cell_name]
                    batch_pairs_data.append(Data(x=drug_data['node_features'], edge_index=drug_data['graph_obj'].edge_index, edge_attr=getattr(drug_data['graph_obj'], 'edge_attr', None), GEF=cell_feats['GEF'], BNF=cell_feats['BNF']))
                    batch_identifiers.append({'drug_id': drug_data['drug_id'], 'cell_line_name': cell_name})

            if not batch_pairs_data: continue
            
            # 使用PyGDataLoader进行高效的GPU数据加载
            pairs_loader = PyGDataLoader(DrugCellDataset(batch_pairs_data), batch_size=args.gpu_batch_size, shuffle=False)
            predictions_for_batch = []
            with torch.no_grad():
                for batch in pairs_loader:
                    pyg_batch = batch.to(DEVICE)
                    node_ft = pyg_batch.x
                    if node_ft.dim() == 1: node_ft = node_ft.unsqueeze(0)
                    
                    num_graphs = pyg_batch.num_graphs
                    gene_ft = pyg_batch.GEF.view(num_graphs, EXPECTED_GENE_FEATURE_DIM)
                    bionic_ft = pyg_batch.BNF.view(num_graphs, EXPECTED_BIONIC_FEATURE_DIM)
                    
                    predictions = predictor_model(node_ft, pyg_batch, gene_ft, bionic_ft)
                    predictions_for_batch.extend(torch.squeeze(predictions).cpu().tolist() if predictions.numel() > 1 else [predictions.item()])
            
            for j, identifier in enumerate(batch_identifiers):
                if j < len(predictions_for_batch):
                    identifier['predicted_ic50'] = predictions_for_batch[j]
                    all_results_for_chunk.append(identifier)

        # 4. 保存当前大块的结果
        if all_results_for_chunk:
            chunk_results_df = pd.DataFrame(all_results_for_chunk)
            chunk_output_path = f"{output_base}_{chunk_num}.csv"
            try:
                output_dir = os.path.dirname(chunk_output_path)
                if output_dir: os.makedirs(output_dir, exist_ok=True)
                chunk_results_df.to_csv(chunk_output_path, index=False)
                print(f"\nChunk {chunk_num} results saved to {chunk_output_path}")
            except Exception as e:
                print(f"\nError saving chunk {chunk_num} results: {e}")
        
        t_chunk_end = time.time()
        print(f"Chunk {chunk_num} finished in {t_chunk_end - t_chunk_start:.2f}s.")

    t_overall_end = time.time()
    print(f"\n" + "="*20 + " Overall Summary " + "="*20)
    print(f"Total script execution time: {t_overall_end - t_overall_start:.2f}s")
    print("Script finished.")

# --- 脚本入口 ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DIPK 高性能分块预测脚本")
    # --- 输入/输出参数 ---
    parser.add_argument('--input_drugs_csv', type=str, default='./test/predict_all_np.csv', help="包含所有待预测药物的CSV文件路径。")
    parser.add_argument('--gene_expression_file', type=str, default='./test/exp_1.csv', help="新细胞系的基因表达谱文件路径。")
    parser.add_argument('--output_csv', type=str, default='./result/DIPK.csv', help="输出文件基础名，最终会是 '基础名_1.csv' 等。")
    # --- 模型与配置路径 ---
    parser.add_argument('--model_path', type=str, default='./result/Train.pkl', help="主预测器模型路径。")
    parser.add_argument('--train_config_path', type=str, default='TrainConfig.py', help="TrainConfig.py 路径。")
    parser.add_argument('--data_config_path', type=str, default='DataConfig.py', help="DataConfig.py 路径。")
    parser.add_argument('--molgnet_model_path', type=str, default='./Data/MolGNet.pt', help="MolGNet.pt 路径。")
    parser.add_argument('--bionic_dict_path', type=str, default='../Dataset/BIONIC_dict.pkl', help="BIONIC_dict.pkl 路径。")
    parser.add_argument('--canonical_gene_list_path', type=str, default='../Dataset/exp.txt', help="标准基因列表路径。")
    # --- 性能与流程控制参数 ---
    parser.add_argument('--drug_chunk_size', type=int, default=100000, help="每个药物大块的大小。")
    parser.add_argument('--small_batch_size', type=int, default=1000, help="每个预测小批量包含的药物数量。")
    parser.add_argument('--gpu_batch_size', type=int, default=512, help="在GPU上进行特征提取和预测时的DataLoader批次大小。")
    parser.add_argument('--num_workers', type=int, default=-1, help="用于药物预处理的CPU核心数 (-1 表示使用所有可用核心)。")
    parser.add_argument('--start_chunk', type=int, default=0, help='从哪个大块编号开始运行 (0为自动检测)。')
    parser.add_argument('--seed', type=int, default=42, help="随机种子。")
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'], help="指定设备 (cuda 或 cpu)。")
    
    args = parser.parse_args()
    
    # 动态加载配置
    try:
        DataConfig = import_from_path("DataConfig", args.data_config_path)
        import Data as data_module
        TrainConfig = import_from_path("TrainConfig", args.train_config_path)
    except Exception as e:
        print(f"Error during config/module loading: {e}"); exit()
        
    main(args)