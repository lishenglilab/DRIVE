import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, Dataset
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
import torch.cuda.amp as amp
import argparse
import re
from functools import partial
import math
import gc

# --- 【【【 新增依赖 】】】 ---
try:
    from joblib import Parallel, delayed
except ImportError:
    print("ERROR: joblib not found. Please run 'pip install joblib'."); exit(1)

# --- 【【【 关键修改: 导入 Dataset 类, 移除不再需要的 data_process_loader 】】】 ---
try:
    from Step2_DataEncoding import DataEncoding
    from Step3_model import DeepTTC
except ImportError as e:
    print(f"导入自定义模块时出错: {e}"); exit(1)

# --- 【【【 新增: 高性能的PyTorch Dataset类 】】】 ---
class HighPerformanceDataset(Dataset):
    def __init__(self, drug_data_list, cell_data_dict):
        self.drug_data = drug_data_list
        # 将细胞系字典转换为列表和名称映射，以便通过索引访问
        self.cell_names = list(cell_data_dict.keys())
        self.cell_rna_list = [torch.from_numpy(v) for v in cell_data_dict.values()]
        
        self.num_drugs = len(self.drug_data)
        self.num_cells = len(self.cell_names)

    def __len__(self):
        # 数据集的总长度是 药物数 x 细胞系数
        return self.num_drugs * self.num_cells

    def __getitem__(self, index):
        # 根据一维索引动态计算出对应的药物索引和细胞系索引
        drug_idx = index // self.num_cells
        cell_idx = index % self.num_cells
        
        drug_info = self.drug_data[drug_idx]
        
        # 动态组合数据，避免创建大表
        drug_encoding = drug_info['encoding']
        rna_data = self.cell_rna_list[cell_idx]
        
        # 返回一个元组: (药物编码, RNA数据, 假标签)
        return (drug_encoding, rna_data, 0.0)

# --- 辅助函数 ---
def encode_single_smiles(smiles_string, data_encoder):
    if pd.notna(smiles_string):
        try:
            # 直接返回PyTorch张量以减少后续转换开销
            encoding = data_encoder._drug2emb_encoder(smiles_string)
            return (torch.LongTensor(encoding[0]), torch.LongTensor(encoding[1]))
        except Exception: return None
    return None

def load_and_align_new_cell_lines(new_cell_line_path, training_gene_list_path):
    # 修改：返回一个字典 {cell_name: rna_numpy_array} 以便高效查找
    try:
        training_genes = pd.read_csv(training_gene_list_path, sep='\t', index_col=0).index.tolist()
        new_cells_df = pd.read_csv(new_cell_line_path, index_col=0)
        new_cells_df.index = new_cells_df.index.astype(str)
        aligned_cells_df = new_cells_df.reindex(columns=training_genes, fill_value=0.0)
        aligned_cells_df = aligned_cells_df.astype(np.float32)
        print(f"基因对齐完成。最终用于预测的细胞系数据维度为: {aligned_cells_df.shape}")
        return {name: row.values for name, row in aligned_cells_df.iterrows()}
    except Exception as e:
        print(f"加载或对齐细胞系数据时发生错误: {e}"); return {}

def custom_collate_fn(batch):
    # 这个函数现在更高效，因为它处理的是已经准备好的张量
    drug_ids_list, drug_masks_list, rna_data_list, labels_list = [], [], [], []
    for item in batch:
        drug_encoding, rna_data, label = item
        drug_ids_list.append(drug_encoding[0])
        drug_masks_list.append(drug_encoding[1])
        rna_data_list.append(rna_data)
        labels_list.append(label)
        
    # 使用 torch.stack 来高效地将张量列表组合成一个批次
    v_drug_collated = (torch.stack(drug_ids_list), torch.stack(drug_masks_list))
    v_p_collated = torch.stack(rna_data_list)
    y_collated = default_collate(labels_list) # 标签是标量，可以用default_collate
    return v_drug_collated, v_p_collated, y_collated

# ============================================================================
# 主逻辑函数 (已重构)
# ============================================================================
def main(args):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- 使用的设备是: {DEVICE} ---")
    print(f"--- 配置: 大块={args.drug_chunk_size}, 小批量={args.small_batch_size}, CPU核心={args.num_workers} ---")

    print("\n--- Step 1: 初始化和加载共享资源 ---")
    aligned_cell_data_dict = load_and_align_new_cell_lines(args.new_cell_line_file, args.training_gene_list_file)
    if not aligned_cell_data_dict: print("错误: 无法加载细胞系数据，程序终止。"); exit(1)

    data_encoder = DataEncoding(vocab_dir=args.vocab_dir)
    net = DeepTTC(modeldir=args.model_dir)
    try:
        net.load_pretrained(os.path.join(args.model_dir, 'model.pt')); net.model.to(DEVICE); net.model.eval()
        print("模型加载成功并设置为评估模式。")
    except Exception as e: print(f"模型加载失败: {e}"); exit(1)

    print("\n--- Step 2: 开始分大块处理药物文件 ---")
    try:
        chunk_iterator = pd.read_csv(args.new_drug_file, header=None, names=['DrugName', 'SMILES'],
                                     chunksize=args.drug_chunk_size, low_memory=True, sep=',')
    except FileNotFoundError: print(f"错误: 药物文件 {args.new_drug_file} 未找到。"); exit(1)

    output_base, _ = os.path.splitext(args.output_file)
    
    start_from_chunk = args.start_chunk
    if start_from_chunk <= 0:
        output_dir = os.path.dirname(output_base) or '.'; last_completed_chunk = 0
        if os.path.exists(output_dir):
            existing_files = [f for f in os.listdir(output_dir) if f.startswith(os.path.basename(output_base) + "_") and f.endswith(".csv")]
            for f in existing_files:
                num_str = re.search(r'_(\d+)\.csv$', f)
                if num_str: last_completed_chunk = max(last_completed_chunk, int(num_str.group(1)))
        start_from_chunk = last_completed_chunk + 1
        print(f"\n--- 自动检测到上次已完成到大块 {last_completed_chunk}。将从大块 {start_from_chunk} 开始继续... ---")
    else:
        print(f"\n--- 用户指定从大块 {start_from_chunk} 开始运行... ---")

    for chunk_num, drug_chunk_df in enumerate(chunk_iterator, 1):
        if chunk_num < start_from_chunk:
            if (chunk_num == 1) or (chunk_num % 10 == 0): print(f"快速跳过已处理的大块 {chunk_num}...")
            continue
            
        print(f"\n" + "="*20 + f" Processing Chunk {chunk_num} " + "="*20)
        
        print(f"Step A: 并行编码 {len(drug_chunk_df)} 个SMILES...")
        encoding_func = partial(encode_single_smiles, data_encoder=data_encoder)
        encoded_results = Parallel(n_jobs=args.num_workers)(
            delayed(encoding_func)(s) for s in tqdm(drug_chunk_df['SMILES'], desc="并行编码SMILES")
        )
        
        drug_data_list = [{'DrugName': r['DrugName'], 'SMILES': r['SMILES'], 'encoding': enc}
                          for r, enc in zip(drug_chunk_df.to_dict('records'), encoded_results) if enc is not None]
        
        print(f"SMILES编码完成。当前大块有效药物数量: {len(drug_data_list)} / {len(drug_chunk_df)}。")
        if not drug_data_list: print("当前大块没有有效药物，跳过。"); continue
        
        print(f"Step B: 正在为 {len(drug_data_list)} 种药物和 {len(aligned_cell_data_dict)} 个细胞系进行预测...")
        
        all_results_for_chunk = []
        num_small_batches = math.ceil(len(drug_data_list) / args.small_batch_size)
        
        for i in tqdm(range(0, len(drug_data_list), args.small_batch_size), total=num_small_batches, desc="Prediction Batches"):
            small_batch_drug_data = drug_data_list[i : i + args.small_batch_size]
            
            if not small_batch_drug_data: continue

            pred_dataset = HighPerformanceDataset(small_batch_drug_data, aligned_cell_data_dict)
            
            params = {'batch_size': args.gpu_batch_size, 'shuffle': False, 'num_workers': 0, 
                      'drop_last': False, 'sampler': SequentialSampler(pred_dataset),
                      'collate_fn': custom_collate_fn}
            predict_generator = DataLoader(pred_dataset, **params)
            
            predictions_for_small_batch = []
            with torch.no_grad(), amp.autocast():
                for v_drug, v_gene, _ in predict_generator:
                    v_drug = (v_drug[0].to(DEVICE), v_drug[1].to(DEVICE))
                    v_gene = v_gene.to(DEVICE)
                    scores = net.model(v_drug, v_gene)
                    predictions_for_small_batch.extend(scores.squeeze(1).cpu().numpy().tolist())
            
            if predictions_for_small_batch:
                num_cells = len(aligned_cell_data_dict)
                drug_info_df = pd.DataFrame(small_batch_drug_data)[['DrugName', 'SMILES']]
                
                drug_repeated = drug_info_df.loc[drug_info_df.index.repeat(num_cells)].reset_index(drop=True)
                cell_tiled = pd.DataFrame({'COSMIC_ID': list(aligned_cell_data_dict.keys()) * len(drug_info_df)})
                
                small_batch_results_df = pd.concat([drug_repeated, cell_tiled], axis=1)
                small_batch_results_df['Predicted_LN_IC50'] = predictions_for_small_batch
                all_results_for_chunk.append(small_batch_results_df)

        if all_results_for_chunk:
            chunk_results_df = pd.concat(all_results_for_chunk, ignore_index=True)
            chunk_output_path = f"{output_base}_{chunk_num}.csv"
            try:
                output_dir = os.path.dirname(chunk_output_path)
                if output_dir: os.makedirs(output_dir, exist_ok=True)
                chunk_results_df.to_csv(chunk_output_path, index=False)
                print(f"\nChunk {chunk_num} results saved to {chunk_output_path}")
            except Exception as e:
                print(f"\nError saving chunk {chunk_num} results: {e}")
        
        del all_results_for_chunk
        gc.collect()

    print("\n\n" + "="*20 + " All tasks finished " + "="*20)

# --- 【【【 您的原始 argparse 配置，一字不差 】】】 ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DeepTTC 高性能分块预测脚本")
    # 输入/输出
    parser.add_argument('--new_drug_file', type=str, default='./test/predict_all_np.csv', help='包含所有待预测药物的CSV文件路径。')
    parser.add_argument('--new_cell_line_file', type=str, default='./test/exp_1.csv', help='新细胞系基因表达文件路径。')
    parser.add_argument('--output_file', type=str, default='./predictions/DeepTTC.csv', help="输出文件基础名，最终会是 '基础名_1.csv' 等。")
    # 模型与资源
    parser.add_argument('--training_gene_list_file', type=str, default='./mydata/expt.txt', help='用于对齐基因的训练基因列表文件路径。')
    parser.add_argument('--model_dir', type=str, default='./DeepTTC', help='包含模型权重(model.pt)和配置的目录。')
    parser.add_argument('--vocab_dir', type=str, default='./ESPF', help='包含SMILES词汇表文件的目录。')
    # 性能与流程控制
    parser.add_argument('--drug_chunk_size', type=int, default=100000, help="每个药物大块的大小。")
    parser.add_argument('--small_batch_size', type=int, default=1000, help="每个预测小批量包含的药物数量。")
    parser.add_argument('--gpu_batch_size', type=int, default=8192, help="在GPU上进行预测时的DataLoader内部批次大小。")
    parser.add_argument('--num_workers', type=int, default=-1, help="用于SMILES编码的CPU核心数 (-1 表示使用所有可用核心)。")
    parser.add_argument('--start_chunk', type=int, default=0, help='从哪个大块编号开始运行 (0为自动检测)。')
    
    args = parser.parse_args()

    # 移除已废弃的 data_process_loader 导入
    # from Step3_model import data_process_loader 
    
    # 移除多余的打印
    # print("运行新药与新细胞系联合预测脚本...")
    main(args)