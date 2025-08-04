#!/usr/bin/env python3
"""
Unified PaccMann model prediction script.
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

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

# 动态添加路径，确保可以找到项目内的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 延迟导入，以防路径问题
try:
    from models import MODEL_FACTORY
    from utils.utils import get_device
    from pytoda.smiles.smiles_language import SMILESTokenizer
except ImportError as e:
    print(f"Error importing local modules: {e}")
    print("Please ensure the script is run from a directory where 'models', 'utils', and 'pytoda' are accessible.")
    sys.exit(1)

try:
    from utils.layers import ContextAttentionLayer as ImportedContextAttentionLayer
except ImportError:
    ImportedContextAttentionLayer = None
    logging.warning("Could not import ContextAttentionLayer from utils.layers. Some monkey patches might be skipped.")

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("PaccMannPredictor_Unified")


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
        logger.error(
            "CRITICAL: 'number_of_genes' not found in model_training_params. Cannot determine target GEP dimension.")
        sys.exit(1)
    if len(target_gene_list_from_training_spec) != num_genes_expected_by_model:
        logger.error(f"CRITICAL: Length of gene_list_for_param_matching ({len(target_gene_list_from_training_spec)}) "
                     f"does not match 'number_of_genes' in model_training_params ({num_genes_expected_by_model}). "
                     "These must align for correct GEP processing and parameter application.")
        sys.exit(1)

    logger.info(f"Loading GEP data from: {gep_filepath}")
    logger.info(f"Conforming GEP data to target specification of {num_genes_expected_by_model} genes, "
                f"ordered by gene_filepath_spec.")
    try:
        df_gep_raw_pred = pd.read_csv(gep_filepath, index_col=0)
    except Exception as e:
        logger.error(f"Failed to read GEP file {gep_filepath}: {e}");
        raise

    logger.debug(f"  Raw GEP for prediction shape (cells, GEP_file_columns): {df_gep_raw_pred.shape}")

    pred_gep_file_genes_original_names = df_gep_raw_pred.columns.tolist()
    pred_gep_file_cleaned_map = {clean_gene_name(g): g for g in pred_gep_file_genes_original_names}

    all_cells_in_pred_gep = df_gep_raw_pred.index.tolist()

    df_gep_conformed = pd.DataFrame(
        data=fill_missing_gene_value,
        index=all_cells_in_pred_gep,
        columns=target_gene_list_from_training_spec
    ).astype(np.float32)
    logger.debug(f"  Created placeholder GEP DataFrame with shape (cells, target_spec_genes): {df_gep_conformed.shape}")

    genes_found_in_pred_gep_count = 0
    genes_from_spec_not_in_pred_gep = []

    for gene_from_spec in target_gene_list_from_training_spec:
        cleaned_spec_gene = clean_gene_name(gene_from_spec)
        if cleaned_spec_gene in pred_gep_file_cleaned_map:
            original_pred_gep_col_name = pred_gep_file_cleaned_map[cleaned_spec_gene]
            try:
                df_gep_conformed[gene_from_spec] = df_gep_raw_pred[original_pred_gep_col_name].astype(np.float32)
            except Exception as e_astype:
                logger.error(
                    f"Error converting column {original_pred_gep_col_name} for gene '{gene_from_spec}' to float32: {e_astype}. Will remain filled with {fill_missing_gene_value}.")
            genes_found_in_pred_gep_count += 1
        else:
            genes_from_spec_not_in_pred_gep.append(gene_from_spec)

    if genes_from_spec_not_in_pred_gep:
        logger.warning(
            f"  {len(genes_from_spec_not_in_pred_gep)} genes from target spec were NOT FOUND in '{gep_filepath}' "
            f"and were filled with {fill_missing_gene_value}. Samples: {genes_from_spec_not_in_pred_gep[:5]}")
    logger.info(f"  Successfully mapped/filled {genes_found_in_pred_gep_count} genes from '{gep_filepath}' into the "
                f"{num_genes_expected_by_model}-gene structure.")

    df_gep_transposed = df_gep_conformed.T
    gep_values = df_gep_transposed.values
    cell_lines = list(df_gep_transposed.columns)

    final_gene_list_for_model = target_gene_list_from_training_spec
    num_genes_for_processing = num_genes_expected_by_model

    logger.info(f"  Final GEP data matrix for processing, shape (genes, cells): {gep_values.shape}")
    if gep_values.shape[0] != num_genes_for_processing:
        logger.error(
            f"CRITICAL INTERNAL ERROR: GEP values rows ({gep_values.shape[0]}) != num_genes_for_processing ({num_genes_for_processing}) after conforming.")
        raise ValueError("GEP processing shape mismatch after conforming.")

    gep_processing_config = model_training_params.get('gene_expression_processing_parameters', {})
    gep_proc_actual_params = gep_processing_config.get('parameters', {})

    if model_training_params.get('gene_expression_standardize', True):
        means_param = gep_proc_actual_params.get('mean')
        stds_param = gep_proc_actual_params.get('std')
        if means_param is not None and stds_param is not None:
            means = np.array(means_param, dtype=np.float32)
            stds = np.array(stds_param, dtype=np.float32)

            logger.info(
                f"  DEBUG: Standardizing GEP. Data shape: {gep_values.shape}, Loaded Means shape: {means.shape}, Loaded Stds shape: {stds.shape}")

            nan_in_means_indices = np.where(np.isnan(means))[0]
            nan_in_stds_indices = np.where(np.isnan(stds))[0]

            if len(nan_in_means_indices) > 0:
                logger.warning(
                    f"  Loaded 'mean' array from params contains {len(nan_in_means_indices)} NaN values. Replacing these NaNs with 0.0 for standardization. Indices (first 5): {nan_in_means_indices[:5]}")
                means[nan_in_means_indices] = 0.0
            if len(nan_in_stds_indices) > 0:
                logger.warning(
                    f"  Loaded 'std' array from params contains {len(nan_in_stds_indices)} NaN values. Replacing these NaNs with 1.0 for standardization. Indices (first 5): {nan_in_stds_indices[:5]}")
                stds[nan_in_stds_indices] = 1.0

            if means.ndim == 1 and stds.ndim == 1 and \
                    num_genes_for_processing == len(means) and num_genes_for_processing == len(stds):

                stds_safe = np.where(np.abs(stds) < 1e-7, 1e-7, stds)
                original_stds_from_params_for_log = np.array(gep_proc_actual_params.get('std', []),
                                                             dtype=np.float32)
                if len(original_stds_from_params_for_log) == len(stds_safe):
                    zero_std_original_indices = np.where(np.abs(original_stds_from_params_for_log) < 1e-7)[0]
                    if len(zero_std_original_indices) > 0:
                        logger.warning(
                            f"  GEP standardization: {len(zero_std_original_indices)} genes had std approx zero in original training params; "
                            f"using epsilon for division (or 1.0 if original std was NaN). Indices (first 5): {zero_std_original_indices[:5]}")

                gep_values = (gep_values - means.reshape(-1, 1)) / stds_safe.reshape(-1, 1)
                logger.info("  GEP data standardized.")
            else:
                logger.error(
                    f"  CRITICAL: GEP standardization param length mismatch. Data has {num_genes_for_processing} genes, "
                    f"but loaded mean length is {len(means)}, std length is {len(stds)}. Ensure --gene_filepath_spec corresponds to these params.")
        else:
            logger.warning(
                "  GEP standardization specified in model, but mean/std params missing in training config. Skipping.")

    if model_training_params.get("gene_expression_min_max", False):
        min_vals_param = gep_proc_actual_params.get('min_val')
        max_vals_param = gep_proc_actual_params.get('max_val')
        if min_vals_param is not None and max_vals_param is not None:
            min_vals = np.array(min_vals_param, dtype=np.float32)
            max_vals = np.array(max_vals_param, dtype=np.float32)

            nan_in_min_vals = np.isnan(min_vals);
            nan_in_max_vals = np.isnan(max_vals)
            if np.any(nan_in_min_vals): logger.warning(
                f"  Loaded 'min_val' array contains {np.sum(nan_in_min_vals)} NaN. Replacing with 0."); min_vals[
                nan_in_min_vals] = 0.0
            if np.any(nan_in_max_vals): logger.warning(
                f"  Loaded 'max_val' array contains {np.sum(nan_in_max_vals)} NaN. Replacing with 1."); max_vals[
                nan_in_max_vals] = 1.0

            logger.info(
                f"  DEBUG: Min-Max scaling GEP. Data shape: {gep_values.shape}, Min_vals shape: {min_vals.shape}, Max_vals shape: {max_vals.shape}")
            if min_vals.ndim == 1 and max_vals.ndim == 1 and \
                    num_genes_for_processing == len(min_vals) and num_genes_for_processing == len(max_vals):
                denominator = (max_vals - min_vals)
                denominator_safe = np.where(np.abs(denominator) < 1e-7, 1e-7, denominator)
                if np.any(np.abs(denominator) < 1e-7):
                    logger.warning(
                        f"  GEP min-max: {np.sum(np.abs(denominator) < 1e-7)} genes had (max-min) range approx zero in training params; replaced denominator with epsilon.")
                gep_values = (gep_values - min_vals.reshape(-1, 1)) / denominator_safe.reshape(-1, 1)
                logger.info("  GEP data min-max scaled.")
            else:
                logger.error(
                    f"  CRITICAL: GEP min-max param length mismatch. Data has {num_genes_for_processing} genes, "
                    f"but loaded min_val length {len(min_vals)}, max_val length {len(max_vals)}.")
        else:
            logger.warning("  GEP min-max scaling specified, but params missing. Skipping.")

    if np.isnan(gep_values).any() or np.isinf(gep_values).any():
        nan_count = np.sum(np.isnan(gep_values))
        inf_count = np.sum(np.isinf(gep_values))
        logger.error(
            f"CRITICAL FINAL CHECK: GEP values contain {nan_count} NaN(s) or {inf_count} Inf(s) AFTER ALL processing.")
        if nan_count > 0:
            nan_rows_cols = np.argwhere(np.isnan(gep_values))
            problem_gene_indices = np.unique(nan_rows_cols[:, 0])
            logger.error(
                f"    NaNs found in GEP for genes (indices in target_gene_list_from_training_spec): {problem_gene_indices[:5]}")
            for p_idx in problem_gene_indices[:2]:
                original_mean_val = means_param[
                    p_idx] if 'means_param' in locals() and means_param is not None and p_idx < len(
                    means_param) else 'N/A'
                original_std_val = stds_param[
                    p_idx] if 'stds_param' in locals() and stds_param is not None and p_idx < len(stds_param) else 'N/A'
                logger.error(
                    f"      Problem Gene: {final_gene_list_for_model[p_idx]}, Original Param Mean: {original_mean_val}, Original Param Std: {original_std_val}")
                logger.error(
                    f"      Raw values (from GEP file, filled if missing) for this gene (first 5 cells): {df_gep_conformed.T.iloc[p_idx, :5].values}")
                logger.error(
                    f"      Processed values (standardized/scaled) for this gene (first 5 cells): {gep_values[p_idx, :5]}")
    else:
        logger.info("GEP processing complete. No NaN/Inf detected in final GEP values.")

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
        valid_min = min_val_param is not None;
        valid_max = max_val_param is not None
        if valid_min and valid_max:
            min_val_actual = min_val_param[0] if isinstance(min_val_param, (list, np.ndarray)) and len(
                min_val_param) > 0 else min_val_param
            max_val_actual = max_val_param[0] if isinstance(max_val_param, (list, np.ndarray)) and len(
                max_val_param) > 0 else max_val_param
            if isinstance(min_val_actual, (int, float)) and isinstance(max_val_actual, (int, float)):
                if np.isnan(min_val_actual) or np.isnan(max_val_actual):
                    logger.warning("Denormalization skipped: training min/max sensitivity values are NaN.")
                    return predictions_normalized
                if np.abs(max_val_actual - min_val_actual) < 1e-7:
                    logger.warning(
                        f"Denormalization: min ({min_val_actual}) and max ({max_val_actual}) are too close. Returning normalized.")
                    return predictions_normalized
                return predictions_normalized * (max_val_actual - min_val_actual) + min_val_actual
            else:
                logger.warning(
                    f"Denormalization skipped: training min/max values not suitable scalars. Min: {min_val_actual}, Max: {max_val_actual}");
                return predictions_normalized
        else:
            logger.warning(
                f"Denormalization skipped: training min/max values missing/invalid. Min: {min_val_param}, Max: {max_val_param}");
            return predictions_normalized
    else:
        logger.debug("Denormalization not applied based on training parameters.")
    return predictions_normalized


class FixBnInputDim(nn.Module):
    def __init__(self, expected_channels_for_bn: int):
        super().__init__();
        self.expected_channels_for_bn = expected_channels_for_bn
        logger.debug(f"MONKEY_PATCH_INIT (FixBnInputDim): Targeting {expected_channels_for_bn} channels.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            if x.shape[0] == self.expected_channels_for_bn:
                return x.unsqueeze(0)
            elif x.shape[1] == self.expected_channels_for_bn:
                return x.permute(1, 0).unsqueeze(0)
        elif x.ndim == 3 and x.shape[1] != self.expected_channels_for_bn:
            if x.shape[1] > self.expected_channels_for_bn: return x[:, :self.expected_channels_for_bn, :]
        return x


class SqueezeCorrectlyForSoftmax(nn.Module):
    def __init__(self):
        super().__init__();
        logger.debug("MONKEY_PATCH_INIT (SqueezeCorrectlyForSoftmax).")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3 and x.shape[-1] == 1:
            return x.squeeze(dim=-1)
        elif x.ndim == 1:
            return x.unsqueeze(0)
        return x


def patched_context_attention_forward(self_attn_layer, reference: torch.Tensor, context: torch.Tensor,
                                      average_seq: bool = True):
    reference_attention = self_attn_layer.reference_projection(reference)
    projected_context = self_attn_layer.context_projection(context)
    if self_attn_layer.context_sequence_length > 1:
        context_attention = self_attn_layer.context_hidden_projection(projected_context.permute(0, 2, 1)).permute(0, 2,
                                                                                                                  1)
    else:
        context_attention = projected_context
    combined_attention = torch.tanh(reference_attention + context_attention)
    alphas = self_attn_layer.alpha_projection(combined_attention)
    if torch.isnan(alphas).any() or torch.isinf(alphas).any():
        logger.info(
            f"    MONKEY_PATCH_INFO (PatchedContextAttention): ALPHAS may contain NaN/Inf! Shape: {alphas.shape}. Input Ref mean: {reference.mean().item() if not torch.isnan(reference).all() else 'NaN'}, Input Ctx mean: {context.mean().item() if not torch.isnan(context).all() else 'NaN'}")
    output = reference * torch.unsqueeze(alphas, -1)
    if average_seq:
        output = torch.sum(output, dim=1)
    else:
        if self_attn_layer.reference_hidden_size == 1: output = output.squeeze(dim=-1)
    if output.ndim == 1: output = output.unsqueeze(0)
    return output, alphas


def main_predict_unified(
        predict_smiles_filepath: str,
        model_run_path: str,
        gep_filepath: str,
        gene_filepath_spec: str,
        smiles_language_filepath: str,
        output_filepath: str,
        model_weights_filename_prefix: str = "best_mse",
        cell_lines_subset_filepath: str = None,
        test_with_random_weights: bool = False
):
    logger.info(f"=== Unified PaccMann Prediction Started (Logger: {logger.name}) ===")
    if test_with_random_weights: logger.warning("***** RUNNING IN RANDOM WEIGHTS TEST MODE *****")
    logger.info(f"Using model artifacts from run directory: {model_run_path}")

    params_file_path_in_run = os.path.join(model_run_path, "model_params_used.json")
    if not os.path.exists(params_file_path_in_run):
        logger.error(f"CRITICAL: 'model_params_used.json' not found in {model_run_path}.");
        sys.exit(1)
    with open(params_file_path_in_run, 'r') as f:
        model_training_params = json.load(f)
    logger.info(f"Successfully loaded training parameters from: {params_file_path_in_run}")

    device = get_device()
    logger.info(f"Using device: {device} (GPU acceleration is enabled if a CUDA-enabled GPU is available).")

    with open(gene_filepath_spec, "rb") as f:
        target_gene_list_for_model_definition = pickle.load(f)
    num_genes_defined_by_spec = len(target_gene_list_for_model_definition)
    logger.info(
        f"Loaded target gene specification: {gene_filepath_spec} with {num_genes_defined_by_spec} genes. This will be the target gene dimension.")

    smiles_tokenizer = SMILESTokenizer.from_pretrained(smiles_language_filepath)
    if "smiles_padding_length" not in model_training_params:
        logger.error("CRITICAL: 'smiles_padding_length' key is missing from the loaded training parameters JSON file.")
        sys.exit(1)
    tokenizer_padding_length_from_train_params = model_training_params["smiles_padding_length"]

    logger.info(
        f"DEBUG_PREDICT: Value of 'smiles_padding_length' from JSON: '{tokenizer_padding_length_from_train_params}' (type: {type(tokenizer_padding_length_from_train_params)})")
    if tokenizer_padding_length_from_train_params is None:
        logger.error("CRITICAL: 'smiles_padding_length' is null in training params file.");
        sys.exit(1)
    try:
        effective_smiles_padding_length = int(tokenizer_padding_length_from_train_params)
    except (ValueError, TypeError) as e:
        logger.error(
            f"CRITICAL: 'smiles_padding_length' ('{tokenizer_padding_length_from_train_params}') from params is not a valid integer: {e}");
        sys.exit(1)

    smiles_tokenizer.set_encoding_transforms(
        add_start_and_stop=model_training_params.get("add_start_and_stop", True),
        padding=model_training_params.get("padding", True),
        padding_length=effective_smiles_padding_length
    )
    smiles_tokenizer.set_smiles_transforms(augment=False,
                                           canonical=model_training_params.get("test_smiles_canonical", True))

    if smiles_tokenizer.padding != model_training_params.get("padding", True):
        logger.warning(
            f"SMILESTokenizer.padding is {smiles_tokenizer.padding} but param was {model_training_params.get('padding', True)}. Trying direct set.")
        smiles_tokenizer.padding = model_training_params.get("padding", True)
    if smiles_tokenizer.padding_length != effective_smiles_padding_length:
        logger.warning(
            f"SMILESTokenizer.padding_length is {smiles_tokenizer.padding_length} but param was {effective_smiles_padding_length}. Trying direct set.")
        smiles_tokenizer.padding_length = effective_smiles_padding_length

    logger.info(
        f"SMILES Tokenizer configured. Final effective padding: {smiles_tokenizer.padding}, padding_length: {smiles_tokenizer.padding_length}, vocab_size={smiles_tokenizer.number_of_tokens}")
    if smiles_tokenizer.padding and smiles_tokenizer.padding_length is None:
        logger.error(
            f"CRITICAL: SMILES padding is True but effective padding_length is still None after configuration.");
        sys.exit(1)
    if smiles_tokenizer.padding and smiles_tokenizer.padding_length != effective_smiles_padding_length:
        logger.error(
            f"CRITICAL: SMILESTokenizer padding_length ({smiles_tokenizer.padding_length}) != target ({effective_smiles_padding_length}) after all attempts.");
        sys.exit(1)

    num_genes_from_params_json = model_training_params.get('number_of_genes')
    if num_genes_from_params_json is None:
        logger.error("CRITICAL: 'number_of_genes' not found in model_training_params.");
        sys.exit(1)
    num_genes_from_params_json = int(num_genes_from_params_json)

    if num_genes_from_params_json != num_genes_defined_by_spec:
        logger.error(
            f"CRITICAL INCONSISTENCY: 'number_of_genes' in JSON ({num_genes_from_params_json}) "
            f"does NOT match length of --gene_filepath_spec ({num_genes_defined_by_spec}). "
            f"These MUST match.")
        sys.exit(1)

    gep_values_processed, cell_line_names_in_gep_file, final_gene_list_used_for_input = load_and_process_gep_data(
        gep_filepath, target_gene_list_for_model_definition, model_training_params
    )
    logger.info(
        f"GEP data processed for {len(cell_line_names_in_gep_file)} cell lines, conformed to {len(final_gene_list_used_for_input)} genes.")

    params_for_model_creation = deepcopy(model_training_params)
    params_for_model_creation[
        "number_of_genes"] = num_genes_from_params_json
    params_for_model_creation["smiles_vocabulary_size"] = smiles_tokenizer.number_of_tokens
    params_for_model_creation["smiles_padding_length"] = effective_smiles_padding_length

    logger.info(f"Model instantiation: number_of_genes={params_for_model_creation['number_of_genes']}, "
                f"smiles_padding_length={params_for_model_creation['smiles_padding_length']}, "
                f"smiles_vocabulary_size={params_for_model_creation['smiles_vocabulary_size']}")

    cell_lines_for_prediction = cell_line_names_in_gep_file
    gep_indices_for_prediction = list(range(len(cell_line_names_in_gep_file)))
    if cell_lines_subset_filepath:
        if not os.path.exists(cell_lines_subset_filepath):
            logger.warning(
                f"Subset file '{cell_lines_subset_filepath}' not found. Using all {len(cell_lines_for_prediction)} cell lines.")
        else:
            logger.info(f"Loading cell line subset from: {cell_lines_subset_filepath}")
            with open(cell_lines_subset_filepath, 'r') as f:
                subset_names_from_file = [ln.strip() for ln in f if ln.strip()]
            selected_indices_temp, selected_names_temp = [], []
            gep_cell_name_to_idx_map = {name: i for i, name in enumerate(cell_line_names_in_gep_file)}
            for name_in_subset_file in subset_names_from_file:
                if name_in_subset_file in gep_cell_name_to_idx_map:
                    selected_indices_temp.append(gep_cell_name_to_idx_map[name_in_subset_file])
                    selected_names_temp.append(name_in_subset_file)
                else:
                    logger.warning(f"Subset cell '{name_in_subset_file}' not found in GEP data. Skipping.")
            if not selected_names_temp:
                logger.warning("No cells from subset file found in GEP data. Using all.")
            else:
                cell_lines_for_prediction, gep_indices_for_prediction = selected_names_temp, selected_indices_temp
                logger.info(f"Predicting for {len(cell_lines_for_prediction)} subset cells.")
    else:
        logger.info(f"No cell subset file. Predicting for all {len(cell_lines_for_prediction)} GEP cells.")

    model_class_name = params_for_model_creation.get("model_fn", "paccmann_v2")
    model = MODEL_FACTORY[model_class_name](params_for_model_creation).to(device)
    model._associate_language(smiles_tokenizer)

    logger.info("Applying monkey patches...")
    patched_bn_fixer_count, patched_attn_squeeze_count, patched_attn_forward_count = 0, 0, 0
    try:
        if hasattr(model, 'convolutional_layers') and isinstance(model.convolutional_layers, nn.Sequential):
            for conv_block_idx in range(len(model.convolutional_layers)):
                if conv_block_idx < len(model.convolutional_layers) and \
                        isinstance(model.convolutional_layers[conv_block_idx], nn.Sequential) and \
                        hasattr(model.convolutional_layers[conv_block_idx], '_modules'):
                    conv_block_module = model.convolutional_layers[conv_block_idx]
                    original_bn_module = conv_block_module._modules.get('batch_norm')
                    if isinstance(original_bn_module, nn.BatchNorm1d):
                        bn_num_features = original_bn_module.num_features
                        if bn_num_features == 64:
                            new_block_layers = OrderedDict()
                            inserted_fixer = False
                            for name, layer_module in conv_block_module._modules.items():
                                if name == 'batch_norm' and not inserted_fixer:
                                    new_block_layers[f"custom_bn_input_fixer_block_{conv_block_idx}"] = FixBnInputDim(
                                        bn_num_features)
                                    inserted_fixer = True
                                new_block_layers[name] = layer_module
                            if inserted_fixer:
                                model.convolutional_layers[conv_block_idx] = nn.Sequential(new_block_layers)
                                patched_bn_fixer_count += 1
        if ImportedContextAttentionLayer is not None:
            for module_name, child_module in model.named_modules():
                if isinstance(child_module, ImportedContextAttentionLayer):
                    if hasattr(child_module, 'alpha_projection') and \
                            isinstance(child_module.alpha_projection, nn.Sequential) and \
                            hasattr(child_module.alpha_projection, '_modules') and \
                            'squeeze' in child_module.alpha_projection._modules:
                        child_module.alpha_projection._modules['squeeze'] = SqueezeCorrectlyForSoftmax()
                        patched_attn_squeeze_count += 1
                    child_module.forward = patched_context_attention_forward.__get__(child_module,
                                                                                     ImportedContextAttentionLayer)
                    patched_attn_forward_count += 1
        logger.info(
            f"Monkey patching summary: BN Fixers={patched_bn_fixer_count}, AttnSqueezers={patched_attn_squeeze_count}, AttnForwards={patched_attn_forward_count}")
    except Exception as e_patch:
        logger.error(f"Error during monkey patching: {e_patch}", exc_info=True)

    if not test_with_random_weights:
        weights_folder = os.path.join(model_run_path, "weights")
        selected_model_file, max_epoch = None, -1
        if not os.path.isdir(weights_folder): logger.error(f"Weights folder not found: {weights_folder}"); sys.exit(1)
        for fname in os.listdir(weights_folder):
            if fname.startswith(model_weights_filename_prefix) and fname.endswith(f"{model_class_name}.pt"):
                try:
                    epoch_search = re.search(r"_epoch(\d+)", fname);
                    epoch = int(epoch_search.group(1)) if epoch_search else -1
                    if epoch > max_epoch:
                        max_epoch, selected_model_file = epoch, os.path.join(weights_folder, fname)
                    elif epoch == -1 and selected_model_file is None:
                        selected_model_file = os.path.join(weights_folder, fname)
                except Exception:
                    if selected_model_file is None: selected_model_file = os.path.join(weights_folder, fname)
        if not selected_model_file: logger.error(
            f"No model weights found for prefix '{model_weights_filename_prefix}'."); sys.exit(1)

        logger.info(f"Loading model weights from: {selected_model_file}")
        try:
            strict_load_flag = not (patched_bn_fixer_count > 0 or patched_attn_squeeze_count > 0)
            model.load_state_dict(torch.load(selected_model_file, map_location=device), strict=strict_load_flag)
            logger.info(f"Model weights loaded (strict={strict_load_flag}).")
        except RuntimeError as e:
            logger.error(f"RuntimeError loading state_dict (strict={strict_load_flag}): {e}")
            logger.error(
                "This usually indicates a mismatch between the instantiated model architecture and the saved weights.")
            if strict_load_flag:
                logger.info("Attempting to load weights with strict=False as a fallback...")
                try:
                    model.load_state_dict(torch.load(selected_model_file, map_location=device), strict=False)
                    logger.info(
                        "Model weights loaded with strict=False (WARNING: some keys might have been mismatched/ignored).")
                except RuntimeError as e2:
                    logger.error(f"RuntimeError loading state_dict (strict=False) also failed: {e2}", exc_info=True);
                    sys.exit(1)
            else:
                sys.exit(1)
        except Exception as e_load:
            logger.error(f"General error during state_dict load: {e_load}", exc_info=True);
            sys.exit(1)
    else:
        logger.info("Skipping weight loading for random weight test.")
    model.eval()

    if not os.path.exists(predict_smiles_filepath): logger.error(
        f"Predict SMILES file not found: {predict_smiles_filepath}"); sys.exit(1)
    logger.info(f"Reading SMILES for prediction from: {predict_smiles_filepath}")
    try:
        predict_df = pd.read_csv(predict_smiles_filepath, header=None, names=['drug_name', 'smiles'])
    except Exception as e:
        logger.error(f"Could not read SMILES file '{predict_smiles_filepath}': {e}");
        sys.exit(1)
    logger.info(f"Found {len(predict_df)} drugs for prediction. predict_df is empty: {predict_df.empty}")
    if not predict_df.empty: logger.info(f"predict_df head (first 2 rows):\n{predict_df.head(2)}")

    prediction_results = []
    if predict_df.empty or not cell_lines_for_prediction:
        logger.warning("No drugs to predict or no cell lines to predict on. Skipping prediction loop.")
    else:
        logger.info(
            f"Starting prediction for {len(predict_df)} drugs across {len(cell_lines_for_prediction)} cell lines.")
        with torch.no_grad():
            drug_iterator = tqdm(
                predict_df.itertuples(index=False),
                total=len(predict_df),
                desc="Predicting Drugs",
                unit="drug"
            )
            for row in drug_iterator:
                drug_name, smiles_string = row.drug_name, row.smiles
                drug_iterator.set_postfix({'Drug': f"{drug_name[:30]}..."})

                smiles_tensor = None
                try:
                    token_indexes_tensor = smiles_tokenizer.smiles_to_token_indexes(smiles_string)
                    if token_indexes_tensor.ndim > 1: token_indexes_tensor = token_indexes_tensor.flatten()
                    final_token_sequence = token_indexes_tensor

                    if final_token_sequence.shape[0] != effective_smiles_padding_length:
                        logger.debug(f"    SMILES for '{drug_name}' length mismatches padding_length. Adjusting.")
                        token_list = final_token_sequence.tolist();
                        current_len = len(token_list)
                        if current_len > effective_smiles_padding_length:
                            token_list = token_list[:effective_smiles_padding_length]
                        else:
                            pad_idx = getattr(smiles_tokenizer, 'padding_index', 0)
                            token_list.extend([pad_idx] * (effective_smiles_padding_length - current_len))
                        final_token_sequence = torch.tensor(token_list, dtype=torch.long)
                    else:
                        final_token_sequence = final_token_sequence.to(dtype=torch.long)

                    smiles_tensor = final_token_sequence.unsqueeze(0).to(device)

                except Exception as e_smiles:
                    logger.error(f"  Error processing SMILES for '{drug_name}': {e_smiles}", exc_info=False)
                    for cell_name_iter in cell_lines_for_prediction:
                        prediction_results.append(
                            {'drug_name': drug_name, 'smiles': smiles_string, 'cell_line_name': cell_name_iter,
                             'predicted_value_raw': np.nan, 'predicted_value_denormalized': np.nan})
                    continue
                if smiles_tensor is None: continue

                for gep_col_idx, cell_name_for_pred in zip(gep_indices_for_prediction, cell_lines_for_prediction):
                    gep_vector_for_cell = gep_values_processed[:, gep_col_idx]
                    gep_tensor = torch.tensor([gep_vector_for_cell], dtype=torch.float32).to(device)

                    if torch.isnan(gep_tensor).any():
                        logger.error(f"    CRITICAL PRE-CALL: GEP tensor for {drug_name}/{cell_name_for_pred} is NaN!")

                    pred_raw_value, pred_denormalized_value = np.nan, np.nan
                    try:
                        pred_raw_tensor, pred_dict = model(smiles_tensor, gep_tensor, confidence=False)
                        if pred_raw_tensor is not None:
                            is_nan = torch.isnan(pred_raw_tensor).any().item()
                            is_inf = torch.isinf(pred_raw_tensor).any().item()
                            if pred_raw_tensor.numel() == 1 and not (is_nan or is_inf):
                                pred_raw_value = pred_raw_tensor.cpu().item()
                                pred_denormalized_value = \
                                    denormalize_predictions(np.array([pred_raw_value]), model_training_params)[0]
                            elif is_nan or is_inf:
                                logger.warning(
                                    f"        Model output is NaN or Inf for {drug_name}/{cell_name_for_pred}.")
                            else:
                                logger.warning(
                                    f"        Model output tensor is NOT a scalar for {drug_name}/{cell_name_for_pred}. Shape: {pred_raw_tensor.shape}.")
                        else:
                            logger.warning(f"      Model output tensor is None for {drug_name}/{cell_name_for_pred}.")
                    except Exception as e_pred:
                        logger.error(
                            f"    Error during model.forward() for '{drug_name}' on '{cell_name_for_pred}': {e_pred}",
                            exc_info=True)

                    prediction_results.append(
                        {'drug_name': drug_name, 'smiles': smiles_string, 'cell_line_name': cell_name_for_pred,
                         'predicted_value_raw': pred_raw_value,
                         'predicted_value_denormalized': pred_denormalized_value})

    logger.info(f"Total entries in prediction_results: {len(prediction_results)}")
    results_df = pd.DataFrame(prediction_results)
    logger.info(f"Final results_df - Is empty: {results_df.empty}, Shape: {results_df.shape}")
    if not results_df.empty:
        logger.info(f"Final results_df head:\n{results_df.head()}")
        non_nan_df = results_df.dropna(subset=['predicted_value_raw'])
        logger.info(f"Number of rows with non-NaN raw predictions: {len(non_nan_df)}")
        if not non_nan_df.empty: logger.info(f"Sample of non-NaN predictions:\n{non_nan_df.head()}")

    output_dir = os.path.dirname(output_filepath)
    if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(output_filepath, index=False)
    logger.info(f"Prediction results saved to: {output_filepath}")
    logger.info(f"=== Unified PaccMann Prediction Finished ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Unified PaccMann prediction for new drugs and/or new cell lines.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--predict_smiles_filepath', type=str, default='./depmap/drug_results.csv',
                        help='Path to the input CSV file for drug prediction (no header: drug_name,smiles).')
    parser.add_argument('--gep_filepath', type=str, default='./depmap/gene_depmap.csv',
                        help='Path to the gene expression data CSV file for cell line prediction (index=cell_name, columns=genes).')
    parser.add_argument('--model_run_path', type=str, default='./paccman_training_runs/paccmann_train_1747977832',
                        help="Path to the PaccMann training run directory containing model artifacts.")
    parser.add_argument('--gene_filepath_spec', type=str, default='./data/2128_genes.pkl',
                        help='Path to the .pkl gene list that DEFINES the model architecture and parameter order.')
    parser.add_argument('--smiles_language_filepath', type=str, default='./paccmann/smiles_language.pkl',
                        help='Path to the SMILES language tokenizer .pkl file.')
    parser.add_argument('--output_filepath', type=str, default='./predictions/paccmann.csv',
                        help='Path to save the prediction results CSV.')
    parser.add_argument('--model_weights_filename_prefix', type=str, default='best_mse',
                        choices=['best_mse', 'best_pearson', 'training_done', 'periodic_epoch'],
                        help="Prefix of the model weights file to use for prediction.")
    parser.add_argument('--cell_lines_subset_filepath', type=str, default=None,
                        help='Optional: Path to a text file of cell line names for subset prediction.')
    parser.add_argument('--test_random_weights', action='store_true',
                        help='If set, skips loading model weights and uses random initialization for testing.')

    args = parser.parse_args()
    logger.info(f"--- Unified Prediction Script Invoked ---")
    for arg_name_main, value_main in sorted(vars(args).items()):
        logger.info(f"  Argument '{arg_name_main}': {value_main}")
    logger.info(f"-------------------------------------------")

    if not os.path.exists(args.model_run_path) and not args.test_random_weights:
        logger.error(
            f"CRITICAL: '--model_run_path' ({args.model_run_path}) does not exist or is not a directory, and not testing with random weights.")
        sys.exit(1)

    main_predict_unified(
        predict_smiles_filepath=args.predict_smiles_filepath,
        model_run_path=args.model_run_path,
        gep_filepath=args.gep_filepath,
        gene_filepath_spec=args.gene_filepath_spec,
        smiles_language_filepath=args.smiles_language_filepath,
        output_filepath=args.output_filepath,
        model_weights_filename_prefix=args.model_weights_filename_prefix,
        cell_lines_subset_filepath=args.cell_lines_subset_filepath,
        test_with_random_weights=args.test_random_weights
    )