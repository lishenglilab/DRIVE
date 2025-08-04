import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
import torch.cuda.amp as amp
import argparse

# 导入自定义模块
try:
    from Step2_DataEncoding import DataEncoding
    from Step3_model import DeepTTC, data_process_loader
except ImportError as e:
    print(f"导入自定义模块时出错: {e}")
    print("请确保 Step2_DataEncoding.py 和 Step3_model.py 在Python的搜索路径中，或者与本脚本在同一目录。")
    exit(1)

# --- 性能配置 ---
GPU_BATCH_SIZE = 4096

def load_new_drugs(filepath):
    """加载新药数据文件。"""
    try:
        df = pd.read_csv(filepath, header=None, names=['DrugName', 'SMILES'], sep=',')
        if df.shape[1] <= 1 or df['SMILES'].isnull().mean() > 0.9:
            df = pd.read_csv(filepath, header=None, names=['DrugName', 'SMILES'], sep='\t')
        print(f"成功加载新药文件: {filepath}, 包含 {df.shape[0]} 条药物数据。")
        return df
    except FileNotFoundError:
        print(f"错误: 新药文件 {filepath} 未找到。")
        return pd.DataFrame()
    except Exception as e:
        print(f"加载新药文件 {filepath} 时发生未知错误: {e}")
        return pd.DataFrame()

def load_and_align_new_cell_lines(new_cell_line_path, training_gene_list_path):
    """加载并对齐新细胞系的基因表达数据。"""
    try:
        print(f"从 {training_gene_list_path} 加载训练基因列表...")
        training_genes = pd.read_csv(training_gene_list_path, sep='\t', index_col=0).index.tolist()
        print(f"模型训练时使用了 {len(training_genes)} 个基因。")

        print(f"从 {new_cell_line_path} 加载新细胞系数据...")
        new_cells_df = pd.read_csv(new_cell_line_path, index_col=0)
        new_cells_df.index = new_cells_df.index.astype(str)
        print(f"加载了 {new_cells_df.shape[0]} 个新细胞系, 每个细胞系有 {new_cells_df.shape[1]} 个基因特征。")

        print("正在对齐新细胞系的基因维度以匹配模型输入...")
        aligned_cells_df = new_cells_df.reindex(columns=training_genes, fill_value=0.0)
        aligned_cells_df = aligned_cells_df.astype(np.float32)

        print(f"基因对齐完成。最终用于预测的细胞系数据维度为: {aligned_cells_df.shape}")
        return aligned_cells_df
    except FileNotFoundError as e:
        print(f"错误: 文件未找到 - {e}。请检查路径。")
        return pd.DataFrame()
    except Exception as e:
        print(f"加载或对齐细胞系数据时发生错误: {e}")
        return pd.DataFrame()

def prepare_data_for_prediction(drugs_df, rna_data):
    """为所有药物和细胞系组合准备数据。"""
    num_drugs = len(drugs_df)
    num_cells = len(rna_data)
    
    # 使用 NumPy 和 Pandas 的广播/重复功能来高效创建配对
    drug_repeated = drugs_df.loc[drugs_df.index.repeat(num_cells)].reset_index(drop=True)
    rna_tiled = pd.concat([rna_data]*num_drugs, ignore_index=False).reset_index()
    rna_tiled.rename(columns={'index': 'COSMIC_ID'}, inplace=True)
    
    # 简单的列绑定，因为顺序是保证的
    combined_df = pd.concat([drug_repeated, rna_tiled], axis=1)

    # 分离出药物和RNA部分
    final_drug_df = combined_df[drugs_df.columns.tolist() + ['COSMIC_ID']]
    final_rna_df = combined_df[rna_data.columns]
    
    return final_drug_df, final_rna_df

def custom_collate_fn(batch):
    """自定义collate函数以处理药物编码元组。"""
    drug_ids_list, drug_masks_list, rna_data_list, labels_list = [], [], [], []
    for item in batch:
        drug_ids_list.append(item[0][0])
        drug_masks_list.append(item[0][1])
        rna_data_list.append(item[1])
        labels_list.append(item[2])
    v_drug_collated = (default_collate(drug_ids_list), default_collate(drug_masks_list))
    v_p_collated = default_collate(rna_data_list)
    y_collated = default_collate(labels_list)
    return v_drug_collated, v_p_collated, y_collated

def main(args):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用的设备是: {DEVICE}")

    print("\n--- Step 1: 加载并准备新药和新细胞系数据 ---")
    new_drugs_df = load_new_drugs(args.new_drug_file)
    if new_drugs_df.empty:
        print("错误: 无法加载药物数据，程序终止。")
        exit(1)

    aligned_new_cells_df = load_and_align_new_cell_lines(args.new_cell_line_file, args.training_gene_list_file)
    if aligned_new_cells_df.empty:
        print("错误: 无法加载细胞系数据，程序终止。")
        exit(1)

    print("\n正在预编码所有新药的SMILES...")
    data_encoder = DataEncoding(vocab_dir=args.vocab_dir)
    encoded_smiles = [
        data_encoder._drug2emb_encoder(s) if pd.notna(s) else None
        for s in tqdm(new_drugs_df['SMILES'], desc="编码SMILES")
    ]
    new_drugs_df['drug_encoding'] = encoded_smiles
    original_drug_count = len(new_drugs_df)
    new_drugs_df.dropna(subset=['drug_encoding'], inplace=True)
    print(f"SMILES预编码完成。有效药物数量: {len(new_drugs_df)} / {original_drug_count}。")
    if new_drugs_df.empty:
        print("错误: 没有药物能够成功编码，程序终止。")
        exit(1)

    print(f"\n--- Step 2: 加载模型 ---")
    model_weights_file = os.path.join(args.model_dir, 'model.pt')
    net = DeepTTC(modeldir=args.model_dir)
    try:
        net.load_pretrained(model_weights_file)
        net.model.to(DEVICE)
        net.model.eval()
        print("模型加载成功并设置为评估模式。")
    except Exception as e:
        print(f"模型加载失败: {e}"); exit(1)

    print(f"\n--- Step 3: 准备组合数据并预测 ---")
    combined_drug_df, combined_rna_df = prepare_data_for_prediction(new_drugs_df, aligned_new_cells_df)
    
    # 增加一个假的 'Label' 列，因为data_process_loader需要它
    combined_drug_df['Label'] = 0.0

    if combined_drug_df.empty:
        print("未能生成用于预测的药物-细胞配对。"); exit(1)

    pred_dataset = data_process_loader(
        list_IDs=combined_drug_df.index.values,
        labels=combined_drug_df['Label'].values,
        drug_df=combined_drug_df,
        rna_df=combined_rna_df
    )
    
    params = {
        'batch_size': GPU_BATCH_SIZE, 'shuffle': False, 'num_workers': 0, 'drop_last': False,
        'sampler': SequentialSampler(pred_dataset), 'collate_fn': custom_collate_fn
    }
    predict_generator = DataLoader(pred_dataset, **params)
    
    all_predictions = []
    with torch.no_grad(), amp.autocast():
        for v_drug_batch, v_gene_batch, _ in tqdm(predict_generator, desc="预测中"):
            v_drug_batch = (v_drug_batch[0].to(DEVICE), v_drug_batch[1].to(DEVICE))
            v_gene_batch = v_gene_batch.to(DEVICE)
            scores = net.model(v_drug_batch, v_gene_batch)
            all_predictions.extend(scores.squeeze(1).cpu().numpy().tolist())

    print("\n--- Step 4: 保存最终结果 ---")
    combined_drug_df['Predicted_LN_IC50'] = all_predictions
    results_df = combined_drug_df[['DrugName', 'COSMIC_ID', 'SMILES', 'Predicted_LN_IC50']]
    
    try:
        results_df.to_csv(args.output_file, index=False)
        print(f"所有预测任务完成。{len(results_df)} 条结果已保存到 {args.output_file}")
    except Exception as e:
        print(f"保存结果到 {args.output_file} 时发生错误: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DeepTTC 药物响应预测脚本")
    parser.add_argument('--new_drug_file', type=str, required=True, help='新药文件路径 (CSV: DrugName,SMILES)')
    parser.add_argument('--new_cell_line_file', type=str, required=True, help='新细胞系基因表达文件路径')
    parser.add_argument('--training_gene_list_file', type=str, required=True, help='用于对齐基因的训练基因列表文件路径')
    parser.add_argument('--model_dir', type=str, required=True, help='包含模型权重(model.pt)和配置的目录路径')
    parser.add_argument('--vocab_dir', type=str, required=True, help='包含SMILES词汇表文件的目录路径')
    parser.add_argument('--output_file', type=str, required=True, help='输出预测结果的CSV文件路径')
    
    args = parser.parse_args()
    
    print("运行新药与新细胞系联合预测脚本...")
    main(args)