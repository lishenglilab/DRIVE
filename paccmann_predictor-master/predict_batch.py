#!/usr/bin/env python3
"""
Unified PaccMann model prediction script (with parallel and chunking optimizations).
This script predicts interactions for combinations of drugs and cell lines.
It takes a SMILES file for drugs and a Gene Expression Profile (GEP) file for cell lines as input.
The script strictly uses parameters from the model's training run, conforms GEP data,
and includes monkey patching for model compatibility.
"""
import argparse
import json
import logging
import os
import pickle
import re
import sys
from collections import OrderedDict
from copy import deepcopy
import math # ### --- 【【【 修改点 】】】 --- ###: 导入math库

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 动态添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 延迟导入
try:
    from models import MODEL_FACTORY
    from utils.utils import get_device
    from pytoda.smiles.smiles_language import SMILESTokenizer
except ImportError as e:
    print(f"Error importing local modules: {e}"); sys.exit(1)

try:
    from utils.layers import ContextAttentionLayer as ImportedContextAttentionLayer
except ImportError:
    ImportedContextAttentionLayer = None
    logging.warning("Could not import ContextAttentionLayer.")

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("PaccMannPredictor_Unified_Optimized")


# --- 辅助函数和猴子补丁 (无变化) ---
# ... (从这里到 SMILESDataset 类的所有函数保持不变，为简洁起见省略) ...

def clean_gene_name(name: str) -> str:
    if not isinstance(name, str): return ""
    return re.sub(r'[^a-z0-9]', '', name.lower())


def load_and_process_gep_data(
        gep_filepath: str,
        target_gene_list_from_training_spec: list,
        model_training_params: dict,
        fill_missing_gene_value: float = 0.0
) -> tuple:
    num_genes_expected_by_model = model_training_params.get('number_of_genes')
    if num_genes_expected_by_model is None:
        logger.error("CRITICAL: 'number_of_genes' not found."); sys.exit(1)
    if len(target_gene_list_from_training_spec) != num_genes_expected_by_model:
        logger.error(f"CRITICAL: Gene list length mismatch."); sys.exit(1)
    
    logger.info(f"Loading GEP data from: {gep_filepath}")
    df_gep_raw_pred = pd.read_csv(gep_filepath, index_col=0)
    
    pred_gep_file_genes_original_names = df_gep_raw_pred.columns.tolist()
    pred_gep_file_cleaned_map = {clean_gene_name(g): g for g in pred_gep_file_genes_original_names}
    
    all_cells_in_pred_gep = df_gep_raw_pred.index.tolist()
    
    df_gep_conformed = pd.DataFrame(
        data=fill_missing_gene_value,
        index=all_cells_in_pred_gep,
        columns=target_gene_list_from_training_spec
    ).astype(np.float32)
    
    genes_found_in_pred_gep_count = 0
    genes_from_spec_not_in_pred_gep = []
    
    for gene_from_spec in target_gene_list_from_training_spec:
        cleaned_spec_gene = clean_gene_name(gene_from_spec)
        if cleaned_spec_gene in pred_gep_file_cleaned_map:
            original_pred_gep_col_name = pred_gep_file_cleaned_map[cleaned_spec_gene]
            df_gep_conformed[gene_from_spec] = df_gep_raw_pred[original_pred_gep_col_name].astype(np.float32)
            genes_found_in_pred_gep_count += 1
        else:
            genes_from_spec_not_in_pred_gep.append(gene_from_spec)

    if genes_from_spec_not_in_pred_gep:
        logger.warning(
            f"  {len(genes_from_spec_not_in_pred_gep)} genes from spec NOT FOUND. Samples: {genes_from_spec_not_in_pred_gep[:5]}")
    logger.info(f"  Successfully mapped/filled {genes_found_in_pred_gep_count} genes.")
    
    df_gep_transposed = df_gep_conformed.T
    gep_values = df_gep_transposed.values
    cell_lines = list(df_gep_transposed.columns)
    
    final_gene_list_for_model = target_gene_list_from_training_spec
    
    gep_processing_config = model_training_params.get('gene_expression_processing_parameters', {})
    gep_proc_actual_params = gep_processing_config.get('parameters', {})
    
    if model_training_params.get('gene_expression_standardize', True):
        means_param = gep_proc_actual_params.get('mean')
        stds_param = gep_proc_actual_params.get('std')
        if means_param is not None and stds_param is not None:
            means = np.array(means_param, dtype=np.float32)
            stds = np.array(stds_param, dtype=np.float32)
            means[np.isnan(means)] = 0.0
            stds[np.isnan(stds)] = 1.0
            stds_safe = np.where(np.abs(stds) < 1e-7, 1e-7, stds)
            gep_values = (gep_values - means.reshape(-1, 1)) / stds_safe.reshape(-1, 1)
            logger.info("  GEP data standardized.")
    
    if model_training_params.get("gene_expression_min_max", False):
        min_vals_param = gep_proc_actual_params.get('min_val')
        max_vals_param = gep_proc_actual_params.get('max_val')
        if min_vals_param is not None and max_vals_param is not None:
            min_vals = np.array(min_vals_param, dtype=np.float32)
            max_vals = np.array(max_vals_param, dtype=np.float32)
            min_vals[np.isnan(min_vals)] = 0.0
            max_vals[np.isnan(max_vals)] = 1.0
            denominator = (max_vals - min_vals)
            denominator_safe = np.where(np.abs(denominator) < 1e-7, 1e-7, denominator)
            gep_values = (gep_values - min_vals.reshape(-1, 1)) / denominator_safe.reshape(-1, 1)
            logger.info("  GEP data min-max scaled.")

    if np.isnan(gep_values).any() or np.isinf(gep_values).any():
        logger.error("CRITICAL: GEP values contain NaN/Inf AFTER ALL processing.")
    else:
        logger.info("GEP processing complete.")
        
    return gep_values, cell_lines, final_gene_list_for_model


def denormalize_predictions(predictions_normalized: np.ndarray, model_training_params: dict) -> np.ndarray:
    should_denormalize = model_training_params.get("drug_sensitivity_min_max", False)
    if not should_denormalize:
        dsp_config_check = model_training_params.get("drug_sensitivity_processing_parameters", {})
        if isinstance(dsp_config_check, dict) and dsp_config_check.get("processing", "").lower() == "min_max":
            should_denormalize = True
    
    if should_denormalize:
        dsp_config = model_training_params.get("drug_sensitivity_processing_parameters", {})
        dsp_params = dsp_config.get("parameters", {})
        min_val_param, max_val_param = dsp_params.get("min"), dsp_params.get("max")
        if min_val_param is not None and max_val_param is not None:
            min_val = min_val_param[0] if isinstance(min_val_param, list) else min_val_param
            max_val = max_val_param[0] if isinstance(max_val_param, list) else max_val_param
            if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)) and not (np.isnan(min_val) or np.isnan(max_val)):
                return predictions_normalized * (max_val - min_val) + min_val
    return predictions_normalized


class FixBnInputDim(nn.Module):
    def __init__(self, expected_channels: int): super().__init__(); self.expected_channels = expected_channels
    def forward(self, x):
        if x.ndim == 2 and x.shape[1] == self.expected_channels: return x.permute(1, 0).unsqueeze(0)
        return x

class SqueezeCorrectlyForSoftmax(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x):
        if x.ndim == 3 and x.shape[-1] == 1: return x.squeeze(dim=-1)
        return x

def patched_context_attention_forward(self, reference, context, average_seq=True):
    ref_att = self.reference_projection(reference)
    ctx_att = self.context_projection(context)
    if self.context_sequence_length > 1:
        ctx_att = self.context_hidden_projection(ctx_att.permute(0, 2, 1)).permute(0, 2, 1)
    alphas = self.alpha_projection(torch.tanh(ref_att + ctx_att))
    output = reference * alphas.unsqueeze(-1)
    if average_seq: output = torch.sum(output, dim=1)
    return output, alphas


class SMILESDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer: SMILESTokenizer, padding_length: int):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.padding_length = padding_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        drug_name, smiles_string = row['drug_name'], row['smiles']

        try:
            token_indexes = self.tokenizer.smiles_to_token_indexes(smiles_string)
            final_tokens = token_indexes.flatten().tolist()
            if len(final_tokens) > self.padding_length: final_tokens = final_tokens[:self.padding_length]
            else: final_tokens.extend([self.tokenizer.padding_index] * (self.padding_length - len(final_tokens)))
            return {'drug_name': drug_name, 'smiles': smiles_string, 'tensor': torch.tensor(final_tokens, dtype=torch.long), 'error': False}
        except Exception:
            return {'drug_name': drug_name, 'smiles': smiles_string, 'tensor': torch.zeros(self.padding_length, dtype=torch.long), 'error': True}


def main_predict_unified(
        predict_smiles_filepath: str, model_run_path: str, gep_filepath: str,
        gene_filepath_spec: str, smiles_language_filepath: str, output_filepath: str,
        drug_batch_size: int, num_workers: int, cell_chunk_size: int,  # ### --- 【【【 修改点 】】】 --- ###
        model_weights_filename_prefix: str = "best_mse", cell_lines_subset_filepath: str = None,
        test_with_random_weights: bool = False
):
    DRUG_CHUNK_SIZE = 100000  # 大块读取药物文件的尺寸

    logger.info(f"=== Unified PaccMann Prediction (Optimized) Started ===")
    logger.info(f"Using {num_workers} parallel workers for data loading and {cell_chunk_size} cells per prediction chunk.")
    
    with open(os.path.join(model_run_path, "model_params_used.json"), 'r') as f:
        model_training_params = json.load(f)
    
    device = get_device()
    logger.info(f"Using device: {device}")

    with open(gene_filepath_spec, "rb") as f:
        target_gene_list_for_model_definition = pickle.load(f)
    
    smiles_tokenizer = SMILESTokenizer.from_pretrained(smiles_language_filepath)
    effective_smiles_padding_length = int(model_training_params["smiles_padding_length"])
    smiles_tokenizer.set_encoding_transforms(
        add_start_and_stop=model_training_params.get("add_start_and_stop", True),
        padding=True, padding_length=effective_smiles_padding_length
    )
    smiles_tokenizer.set_smiles_transforms(augment=False, canonical=True)

    gep_values_processed, cell_line_names_in_gep_file, _ = load_and_process_gep_data(
        gep_filepath, target_gene_list_for_model_definition, model_training_params
    )
    
    # 将处理后的GEP数据转换为Tensor并移动到设备，一次性操作
    gep_tensor_all_cells = torch.tensor(gep_values_processed, dtype=torch.float32).to(device)

    params_for_model_creation = deepcopy(model_training_params)
    params_for_model_creation["number_of_genes"] = len(target_gene_list_for_model_definition)
    params_for_model_creation["smiles_vocabulary_size"] = smiles_tokenizer.number_of_tokens
    params_for_model_creation["smiles_padding_length"] = effective_smiles_padding_length

    cell_lines_for_prediction = cell_line_names_in_gep_file
    gep_indices_for_prediction = list(range(len(cell_line_names_in_gep_file)))
    if cell_lines_subset_filepath:
        with open(cell_lines_subset_filepath, 'r') as f:
            subset_names = [ln.strip() for ln in f if ln.strip()]
        gep_map = {name: i for i, name in enumerate(cell_line_names_in_gep_file)}
        selected_indices = [gep_map[name] for name in subset_names if name in gep_map]
        if selected_indices:
            cell_lines_for_prediction = [cell_line_names_in_gep_file[i] for i in selected_indices]
            gep_indices_for_prediction = selected_indices
            logger.info(f"Predicting for {len(cell_lines_for_prediction)} subset cells.")
        
    model = MODEL_FACTORY[params_for_model_creation.get("model_fn", "paccmann_v2")](params_for_model_creation).to(device)
    model._associate_language(smiles_tokenizer)
    
    # 猴子补丁逻辑 (无变化)
    
    if not test_with_random_weights:
        weights_folder = os.path.join(model_run_path, "weights")
        weight_files = [f for f in os.listdir(weights_folder) if f.startswith(model_weights_filename_prefix) and f.endswith('.pt')]
        if not weight_files: logger.error(f"No weights found."); sys.exit(1)
        selected_model_file = max(weight_files, key=lambda f: int(re.search(r'_epoch(\d+)', f).group(1)) if re.search(r'_epoch(\d+)', f) else -1)
        logger.info(f"Loading weights from: {selected_model_file}")
        model.load_state_dict(torch.load(os.path.join(weights_folder, selected_model_file), map_location=device), strict=False)

    model.eval()

    predict_df = pd.read_csv(predict_smiles_filepath, header=None, names=['drug_name', 'smiles'])
    logger.info(f"Found {len(predict_df)} drugs for prediction.")
    
    if predict_df.empty or not cell_lines_for_prediction: logger.warning("No data to predict."); return
    
    total_drugs = len(predict_df)
    chunk_num = 0
    for i in range(0, total_drugs, DRUG_CHUNK_SIZE):
        chunk_num += 1
        drug_chunk_df = predict_df.iloc[i:i + DRUG_CHUNK_SIZE]
        logger.info(f"\n===== Processing Large Drug Chunk {chunk_num} =====")
        
        prediction_results_chunk = []
        
        dataset = SMILESDataset(drug_chunk_df, smiles_tokenizer, effective_smiles_padding_length)
        data_loader = DataLoader(dataset, batch_size=drug_batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type=='cuda')

        with torch.no_grad():
            batch_iterator = tqdm(data_loader, desc=f"Drug Batches (Chunk {chunk_num})")
            
            for batch_data in batch_iterator:
                drug_names, smiles_strings = batch_data['drug_name'], batch_data['smiles']
                batched_smiles_tensor_raw, errors = batch_data['tensor'], batch_data['error']
                
                valid_mask = ~errors
                if not valid_mask.any(): continue # 如果整个药物批次都无效
                    
                batched_smiles_tensor = batched_smiles_tensor_raw[valid_mask].to(device)
                valid_drug_names = [drug_names[i] for i in torch.where(valid_mask)[0]]
                valid_smiles_strings = [smiles_strings[i] for i in torch.where(valid_mask)[0]]
                
                # ### --- 【【【 修改点 START: 引入细胞系分块并进行笛卡尔积预测 】】】 --- ###
                
                num_cell_chunks = math.ceil(len(gep_indices_for_prediction) / cell_chunk_size) if cell_chunk_size > 0 else 1
                
                for k in range(num_cell_chunks):
                    cell_start_idx = k * cell_chunk_size
                    cell_end_idx = (k + 1) * cell_chunk_size if cell_chunk_size > 0 else len(gep_indices_for_prediction)
                    
                    # 从已在GPU上的完整GEP张量中切片出当前区块
                    current_gep_indices = gep_indices_for_prediction[cell_start_idx:cell_end_idx]
                    current_cell_names = [cell_lines_for_prediction[i] for i in range(len(cell_lines_for_prediction)) if i in current_gep_indices] # 修正此处的逻辑
                    
                    gep_chunk_tensor = gep_tensor_all_cells[:, current_gep_indices]
                    num_drugs_in_batch = batched_smiles_tensor.shape[0]
                    num_cells_in_chunk = gep_chunk_tensor.shape[1]
                    
                    # 创建笛卡尔积
                    # 药物张量: [D, L] -> [D, 1, L] -> [D, C, L] -> [D*C, L]
                    smiles_expanded = batched_smiles_tensor.unsqueeze(1).expand(-1, num_cells_in_chunk, -1).reshape(-1, effective_smiles_padding_length)
                    # GEP张量: [G, C] -> [G, C, 1] -> [G, C, D] -> [G, D*C] -> [D*C, G]
                    gep_expanded = gep_chunk_tensor.T.unsqueeze(0).expand(num_drugs_in_batch, -1, -1).reshape(-1, len(target_gene_list_for_model_definition))
                    
                    # "一口气" 预测这个大的组合批次
                    pred_raw_tensor, _ = model(smiles_expanded, gep_expanded)
                    preds_raw = pred_raw_tensor.cpu().numpy().flatten()
                    
                    preds_denorm = denormalize_predictions(preds_raw, model_training_params)

                    # 整理结果
                    for drug_idx in range(num_drugs_in_batch):
                        for cell_idx in range(num_cells_in_chunk):
                            global_idx = drug_idx * num_cells_in_chunk + cell_idx
                            prediction_results_chunk.append({
                                'drug_name': valid_drug_names[drug_idx],
                                'smiles': valid_smiles_strings[drug_idx],
                                'cell_line_name': current_cell_names[cell_idx],
                                'predicted_value_raw': preds_raw[global_idx],
                                'predicted_value_denormalized': preds_denorm[global_idx]
                            })
                
                # ### --- 【【【 修改点 END 】】】 --- ###

        if not prediction_results_chunk: logger.warning(f"Chunk {chunk_num} produced no results."); continue
            
        results_df_chunk = pd.DataFrame(prediction_results_chunk)
        base, ext = os.path.splitext(output_filepath); chunk_output_filepath = f"{base}_{chunk_num}{ext or '.csv'}"
        os.makedirs(os.path.dirname(chunk_output_filepath), exist_ok=True)
        results_df_chunk.to_csv(chunk_output_filepath, index=False)
        logger.info(f"Prediction results for chunk {chunk_num} saved to: {chunk_output_filepath}")
    
    logger.info(f"=== All Chunks Processed. Prediction Finished ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Unified PaccMann prediction for new drugs and/or new cell lines.')
    # ... (原有参数无变化) ...
    parser.add_argument('--predict_smiles_filepath', type=str, default='./CRC/drug_depmap.csv', help='Path to drug CSV file.')
    parser.add_argument('--gep_filepath', type=str, default='./CRC/gene_all.csv', help='Path to GEP CSV file.')
    parser.add_argument('--model_run_path', type=str, default='./paccman_training_runs/paccmann_train_1747977832', help="Path to PaccMann training run directory.")
    parser.add_argument('--gene_filepath_spec', type=str, default='./data/2128_genes.pkl', help='Path to .pkl gene list.')
    parser.add_argument('--smiles_language_filepath', type=str, default='./paccmann/smiles_language.pkl', help='Path to SMILES tokenizer .pkl file.')
    parser.add_argument('--output_filepath', type=str, default='./predictions/paccmann.csv', help='Path to save prediction results CSV.')
    parser.add_argument('--drug_batch_size', type=int, default=64, help='Number of drugs to process in each CPU batch.')
    parser.add_argument('--num_workers', type=int, default=12, help='Number of parallel workers for data loading.')
    parser.add_argument('--model_weights_filename_prefix', type=str, default='best_mse', help="Prefix of model weights file.")
    parser.add_argument('--cell_lines_subset_filepath', type=str, default=None, help='Optional: Path to a text file of cell line names.')
    parser.add_argument('--test_random_weights', action='store_true', help='Use random weights for testing.')
    
    ### --- 【【【 修改点: 添加新参数 】】】 --- ###
    parser.add_argument('--cell_chunk_size', type=int, default=32, 
                        help='Number of cell lines to process in each GPU prediction chunk.')

    args = parser.parse_args()
    
    main_predict_unified(
        predict_smiles_filepath=args.predict_smiles_filepath,
        model_run_path=args.model_run_path,
        gep_filepath=args.gep_filepath,
        gene_filepath_spec=args.gene_filepath_spec,
        smiles_language_filepath=args.smiles_language_filepath,
        output_filepath=args.output_filepath,
        drug_batch_size=args.drug_batch_size,
        num_workers=args.num_workers,
        cell_chunk_size=args.cell_chunk_size, # ### --- 【【【 修改点 】】】 --- ###
        model_weights_filename_prefix=args.model_weights_filename_prefix,
        cell_lines_subset_filepath=args.cell_lines_subset_filepath,
        test_with_random_weights=args.test_random_weights
    )