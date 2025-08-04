#!/usr/bin/env python3
"""Train PaccMann predictor."""
import argparse
import json
import logging
import os
import pickle
import sys
from copy import deepcopy
from time import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

from models import MODEL_FACTORY
from utils.hyperparams import OPTIMIZER_FACTORY
from utils.loss_functions import pearsonr  # Assuming this is your custom pearson
from utils.utils import get_device
from pytoda.datasets import DrugSensitivityDataset
from pytoda.smiles.smiles_language import SMILESTokenizer

# setup logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)  # Use INFO for less verbose output generally


# ++++++++++++++++++++ 新增：反向缩放函数 (保持不变) ++++++++++++++++++++
def unscale_values(scaled_values, processing_params):
    """
    将经过最小-最大缩放的值反向转换为原始值。
    Args:
        scaled_values (np.array): 缩放后的值数组。
        processing_params (dict): 包含 'parameters'键，其下有 'min' 和 'max' 的字典。
    Returns:
        np.array: 反向缩放后的原始值。
    """
    if not processing_params or \
            'parameters' not in processing_params or \
            'min' not in processing_params['parameters'] or \
            'max' not in processing_params['parameters']:
        # logging.warning("尝试反向缩放，但药物敏感性处理参数不完整或缺失。将按原样返回值。")
        return scaled_values

    min_val = processing_params['parameters']['min']
    max_val = processing_params['parameters']['max']

    if max_val == min_val:
        # logging.warning(f"反向缩放时，最大值和最小值相同 ({min_val})。所有值将返回为 {min_val}。")
        return np.full_like(scaled_values, min_val, dtype=np.float64)

    unscaled_values = scaled_values * (max_val - min_val) + min_val
    return unscaled_values


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def main(
        train_sensitivity_filepath,
        test_sensitivity_filepath,
        gep_filepath,
        smi_filepath,
        gene_filepath,
        smiles_language_filepath,
        model_path,  # Base directory for all runs, e.g., ./paccmann_runs/
        params_filepath,
        training_name,  # Specific name for this run, e.g., my_first_training
):
    logger = logging.getLogger(f"{training_name}")  # 恢复原始logger名称
    # Process parameter file:
    params = {}
    with open(params_filepath) as fp:
        params.update(json.load(fp))

    # Create model directory for this specific training_name run
    model_dir = os.path.join(model_path, training_name)
    weights_dir = os.path.join(model_dir, "weights")
    results_dir = os.path.join(model_dir, "results")
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(model_dir, "model_params_used.json"), "w") as fp:
        json.dump(params, fp, indent=4)

    logger.info("Start data preprocessing...")
    smiles_language = SMILESTokenizer.from_pretrained(smiles_language_filepath)
    smiles_language.set_encoding_transforms(
        add_start_and_stop=params.get("add_start_and_stop", True),
        padding=params.get("padding", True),
        padding_length=params.get("smiles_padding_length", None),
    )
    test_smiles_language = deepcopy(smiles_language)
    smiles_language.set_smiles_transforms(
        augment=params.get("augment_smiles", False), canonical=params.get("smiles_canonical", False),
        kekulize=params.get("smiles_kekulize", False), all_bonds_explicit=params.get("smiles_bonds_explicit", False),
        all_hs_explicit=params.get("smiles_all_hs_explicit", False),
        remove_bonddir=params.get("smiles_remove_bonddir", False),
        remove_chirality=params.get("smiles_remove_chirality", False), selfies=params.get("selfies", False),
        sanitize=params.get("selfies", False),
    )
    test_smiles_language.set_smiles_transforms(
        augment=False, canonical=params.get("test_smiles_canonical", True),
        kekulize=params.get("smiles_kekulize", False), all_bonds_explicit=params.get("smiles_bonds_explicit", False),
        all_hs_explicit=params.get("smiles_all_hs_explicit", False),
        remove_bonddir=params.get("smiles_remove_bonddir", False),
        remove_chirality=params.get("smiles_remove_chirality", False), selfies=params.get("selfies", False),
        sanitize=params.get("selfies", False),
    )
    with open(gene_filepath, "rb") as f:
        gene_list = pickle.load(f)

    # 确定是否对药物敏感性数据进行min-max缩放
    drug_sensitivity_min_max_scaling_enabled = params.get("drug_sensitivity_min_max", True)

    train_dataset = DrugSensitivityDataset(
        drug_sensitivity_filepath=train_sensitivity_filepath, smi_filepath=smi_filepath,
        gene_expression_filepath=gep_filepath, smiles_language=smiles_language, gene_list=gene_list,
        drug_sensitivity_min_max=drug_sensitivity_min_max_scaling_enabled,  # 使用配置参数
        drug_sensitivity_processing_parameters=params.get("drug_sensitivity_processing_parameters", {}),
        gene_expression_standardize=params.get("gene_expression_standardize", True),
        gene_expression_min_max=params.get("gene_expression_min_max", False),
        gene_expression_processing_parameters=params.get("gene_expression_processing_parameters", {}),
        iterate_dataset=False,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=params.get("batch_size", 256), shuffle=True,
        drop_last=True, num_workers=params.get("num_workers", 0),
    )

    # ++++++++++++++++++++ 获取药物敏感性缩放参数 (保持不变) ++++++++++++++++++++
    actual_drug_sensitivity_processing_params = train_dataset.drug_sensitivity_processing_parameters
    logger.info(f"Drug sensitivity scaling parameters used: {actual_drug_sensitivity_processing_params}")
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    test_dataset = DrugSensitivityDataset(
        drug_sensitivity_filepath=test_sensitivity_filepath, smi_filepath=smi_filepath,
        gene_expression_filepath=gep_filepath, smiles_language=test_smiles_language, gene_list=gene_list,
        drug_sensitivity_min_max=drug_sensitivity_min_max_scaling_enabled,  # 使用配置参数
        drug_sensitivity_processing_parameters=actual_drug_sensitivity_processing_params,
        gene_expression_standardize=params.get("gene_expression_standardize", True),
        gene_expression_min_max=params.get("gene_expression_min_max", False),
        gene_expression_processing_parameters=train_dataset.gene_expression_dataset.processing,
        iterate_dataset=False,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=params.get("batch_size", 256), shuffle=False,
        drop_last=False, num_workers=params.get("num_workers", 0),
    )
    logger.info(f"Training dataset: {len(train_dataset)} samples, Test dataset: {len(test_dataset)} samples.")

    device = get_device()

    params_updates = {
        "number_of_genes": len(gene_list),
        "smiles_vocabulary_size": smiles_language.number_of_tokens,
        "drug_sensitivity_processing_parameters": actual_drug_sensitivity_processing_params,  # 保存实际使用的缩放参数
        "gene_expression_processing_parameters": train_dataset.gene_expression_dataset.processing,
    }
    params.update(params_updates)
    with open(os.path.join(model_dir, "model_params_used.json"), "w") as fp:
        json.dump(params, fp, indent=4)

    model_name = params.get("model_fn", "paccmann_v2")
    model = MODEL_FACTORY[model_name](params).to(device)
    model._associate_language(smiles_language)

    best_mse_overall_scaled = float('inf')  # 修改变量名以示区分
    best_pearson_overall_scaled = -float('inf')  # 修改变量名以示区分
    epoch_for_best_mse_scaled = 0  # 修改变量名以示区分
    epoch_for_best_pearson_scaled = 0  # 修改变量名以示区分

    # ++++++++++++++++++++ 修改DataFrame列名以区分缩放和原始指标 (保持不变) ++++++++++++++++++++
    all_epochs_metrics_df = pd.DataFrame(
        columns=[
            'Epoch', 'Train_Loss_Scaled',
            'Test_MSE_Scaled', 'Test_RMSE_Scaled', 'Test_Pearson_Scaled', 'Test_Spearman_Scaled', 'Test_R2_Scaled',
            'Test_MSE_Original', 'Test_RMSE_Original', 'Test_Pearson_Original', 'Test_Spearman_Original',
            'Test_R2_Original'
        ]
    )
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    optimizer = OPTIMIZER_FACTORY[params.get("optimizer", "Adam")](
        model.parameters(), lr=params.get("lr", 0.01)
    )
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {model_name}, Number of parameters: {num_params}")
    logger.info("Training about to start...\n")

    for epoch in range(params["epochs"]):
        epoch_time_start = time()
        model.train()
        logger.info(f"== Epoch [{epoch + 1}/{params['epochs']}] ({training_name}) ==")

        cumulative_train_loss_scaled = 0  # 训练损失基于缩放值
        for ind, (smiles, gep, y_scaled) in enumerate(train_loader):  # y_scaled 是缩放后的标签
            y_hat_scaled, pred_dict = model(torch.squeeze(smiles.to(device)), gep.to(device))
            loss = model.loss(y_hat_scaled, y_scaled.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cumulative_train_loss_scaled += loss.item()

        avg_epoch_train_loss_scaled = cumulative_train_loss_scaled / len(train_loader)
        logger.info(
            f"\t **** TRAINING **** Epoch [{epoch + 1}/{params['epochs']}], "
            f"Avg Loss (scaled): {avg_epoch_train_loss_scaled:.5f}. "  # 明确指出是scaled loss
            f"Duration: {time() - epoch_time_start:.1f} secs."
        )

        model.eval()
        with torch.no_grad():
            cumulative_test_loss_scaled = 0
            predictions_collector_scaled = []
            labels_collector_scaled = []

            for ind, (smiles, gep, y_true_scaled) in enumerate(test_loader):
                y_pred_scaled, pred_dict = model(torch.squeeze(smiles.to(device)), gep.to(device))
                predictions_collector_scaled.append(y_pred_scaled.cpu())
                labels_collector_scaled.append(y_true_scaled.cpu())
                loss_on_scaled = model.loss(y_pred_scaled, y_true_scaled.to(device))
                cumulative_test_loss_scaled += loss_on_scaled.item()

            predictions_np_scaled = torch.cat(predictions_collector_scaled).numpy().flatten()
            labels_np_scaled = torch.cat(labels_collector_scaled).numpy().flatten()

            # --- 计算基于【缩放后】数据的指标 ---
            current_epoch_mse_scaled = cumulative_test_loss_scaled / len(test_loader)
            current_epoch_rmse_scaled = np.sqrt(current_epoch_mse_scaled)
            try:
                current_epoch_pearson_scaled = pearsonr(torch.from_numpy(predictions_np_scaled),
                                                        torch.from_numpy(labels_np_scaled)).item()
            except Exception:
                current_epoch_pearson_scaled = np.nan
            try:
                current_epoch_spearman_scaled, _ = spearmanr(labels_np_scaled, predictions_np_scaled)
                if np.isnan(current_epoch_spearman_scaled): current_epoch_spearman_scaled = 0.0
            except ValueError:
                current_epoch_spearman_scaled = np.nan
            try:
                current_epoch_r2_scaled = r2_score(labels_np_scaled, predictions_np_scaled)
            except ValueError:
                current_epoch_r2_scaled = np.nan

            logger.info(
                f"\t **** TESTING (SCALED metrics) **** Epoch [{epoch + 1}/{params['epochs']}], "  # 明确指出是scaled metrics
                f"MSE: {current_epoch_mse_scaled:.5f}, RMSE: {current_epoch_rmse_scaled:.3f}, "
                f"Pearson: {current_epoch_pearson_scaled:.3f}, Spearman: {current_epoch_spearman_scaled:.3f}, R2: {current_epoch_r2_scaled:.3f}"
            )

            # --- 将预测值和真实标签【反向缩放】到原始范围 ---
            predictions_np_original = unscale_values(predictions_np_scaled, actual_drug_sensitivity_processing_params)
            labels_np_original = unscale_values(labels_np_scaled, actual_drug_sensitivity_processing_params)

            # --- 计算基于【原始】数据的指标 ---
            current_epoch_mse_original = np.mean((predictions_np_original - labels_np_original) ** 2)
            current_epoch_rmse_original = np.sqrt(current_epoch_mse_original)
            try:
                current_epoch_pearson_original = pearsonr(torch.from_numpy(predictions_np_original),
                                                          torch.from_numpy(labels_np_original)).item()
            except Exception:
                current_epoch_pearson_original = np.nan
            try:
                current_epoch_spearman_original, _ = spearmanr(labels_np_original, predictions_np_original)
                if np.isnan(current_epoch_spearman_original): current_epoch_spearman_original = 0.0
            except ValueError:
                current_epoch_spearman_original = np.nan
            try:
                current_epoch_r2_original = r2_score(labels_np_original, predictions_np_original)
            except ValueError:
                current_epoch_r2_original = np.nan

            logger.info(
                f"\t **** TESTING (ORIGINAL metrics) **** Epoch [{epoch + 1}/{params['epochs']}], "  # 明确指出是original metrics
                f"MSE: {current_epoch_mse_original:.5f}, RMSE: {current_epoch_rmse_original:.3f}, "
                f"Pearson: {current_epoch_pearson_original:.3f}, Spearman: {current_epoch_spearman_original:.3f}, R2: {current_epoch_r2_original:.3f}"
            )

            # ++++++++++++++++++++ 更新 current_metrics_data 以包含两套指标 (保持不变) ++++++++++++++++++++
            current_metrics_data = {
                'Epoch': epoch + 1, 'Train_Loss_Scaled': avg_epoch_train_loss_scaled,
                'Test_MSE_Scaled': current_epoch_mse_scaled, 'Test_RMSE_Scaled': current_epoch_rmse_scaled,
                'Test_Pearson_Scaled': current_epoch_pearson_scaled,
                'Test_Spearman_Scaled': current_epoch_spearman_scaled,
                'Test_R2_Scaled': current_epoch_r2_scaled,
                'Test_MSE_Original': current_epoch_mse_original, 'Test_RMSE_Original': current_epoch_rmse_original,
                'Test_Pearson_Original': current_epoch_pearson_original,
                'Test_Spearman_Original': current_epoch_spearman_original,
                'Test_R2_Original': current_epoch_r2_original
            }
            all_epochs_metrics_df = pd.concat([all_epochs_metrics_df, pd.DataFrame([current_metrics_data])],
                                              ignore_index=True)

            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

            # ++++++++++++++++++++ 更新 save_model_and_metrics_package 以保存两套预测 (保持不变) ++++++++++++++++++++
            def save_model_and_metrics_package(save_type_prefix, current_epoch_num, metrics_dict,
                                               preds_scaled_np, lbls_scaled_np,
                                               preds_original_np, lbls_original_np):
                model_filename = f"{save_type_prefix}_epoch{current_epoch_num + 1}_{model_name}.pt"
                model_save_path = os.path.join(weights_dir, model_filename)
                model.save(model_save_path)

                metrics_filename = f"{save_type_prefix}_epoch{current_epoch_num + 1}_{model_name}_metrics.json"
                metrics_save_path = os.path.join(results_dir, metrics_filename)
                serializable_metrics = {
                    k: (float(v) if isinstance(v, (np.floating, float, int, np.float32, np.float64)) else v) for k, v in
                    metrics_dict.items()}
                serializable_metrics['epoch'] = current_epoch_num + 1
                with open(metrics_save_path, "w") as f:
                    json.dump(serializable_metrics, f, indent=4)

                preds_labels_scaled_filename = f"{save_type_prefix}_epoch{current_epoch_num + 1}_{model_name}_preds_labels_scaled.npy"
                preds_labels_scaled_save_path = os.path.join(results_dir, preds_labels_scaled_filename)
                np.save(preds_labels_scaled_save_path, np.vstack([preds_scaled_np, lbls_scaled_np]))

                preds_labels_original_filename = f"{save_type_prefix}_epoch{current_epoch_num + 1}_{model_name}_preds_labels_original.npy"
                preds_labels_original_save_path = os.path.join(results_dir, preds_labels_original_filename)
                np.save(preds_labels_original_save_path, np.vstack([preds_original_np, lbls_original_np]))

                logger.info(f"Saved package for '{save_type_prefix}' at epoch {current_epoch_num + 1}:")
                logger.info(f"  Model: {model_save_path}")
                logger.info(f"  Metrics: {metrics_save_path}")
                logger.info(f"  Scaled Preds/Labels: {preds_labels_scaled_save_path}")
                logger.info(f"  Original Preds/Labels: {preds_labels_original_save_path}")

            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

            # 基于【缩放后】的指标来判断最佳模型
            if current_epoch_mse_scaled < best_mse_overall_scaled:
                best_mse_overall_scaled = current_epoch_mse_scaled
                epoch_for_best_mse_scaled = epoch
                save_model_and_metrics_package(
                    "best_mse_scaled", epoch, current_metrics_data,
                    predictions_np_scaled, labels_np_scaled,
                    predictions_np_original, labels_np_original
                )

            if not np.isnan(
                    current_epoch_pearson_scaled) and current_epoch_pearson_scaled > best_pearson_overall_scaled:
                best_pearson_overall_scaled = current_epoch_pearson_scaled
                epoch_for_best_pearson_scaled = epoch
                save_model_and_metrics_package(
                    "best_pearson_scaled", epoch, current_metrics_data,
                    predictions_np_scaled, labels_np_scaled,
                    predictions_np_original, labels_np_original
                )

        if (epoch + 1) % params.get("save_every_n_epochs", 1000) == 0:
            save_model_and_metrics_package(
                f"periodic_epoch", epoch, current_metrics_data,
                predictions_np_scaled, labels_np_scaled,
                predictions_np_original, labels_np_original
            )

    save_model_and_metrics_package(
        "training_done", params["epochs"] - 1, current_metrics_data,
        predictions_np_scaled, labels_np_scaled,
        predictions_np_original, labels_np_original
    )

    all_epochs_csv_path = os.path.join(results_dir, f"all_epochs_metrics_{training_name}.csv")
    all_epochs_metrics_df.to_csv(all_epochs_csv_path, index=False)
    logger.info(f"Metrics for all epochs saved to: {all_epochs_csv_path}")

    # ++++++++++++++++++++ 更新最终日志总结 (保持不变) ++++++++++++++++++++
    # 确保从DataFrame中正确提取原始指标
    best_mse_epoch_original_mse = np.nan
    if epoch_for_best_mse_scaled + 1 in all_epochs_metrics_df['Epoch'].values:  # 检查epoch是否存在
        best_mse_epoch_original_mse = all_epochs_metrics_df.loc[
            all_epochs_metrics_df['Epoch'] == epoch_for_best_mse_scaled + 1, 'Test_MSE_Original'].iloc[0]

    best_pearson_epoch_original_pearson = np.nan
    if epoch_for_best_pearson_scaled + 1 in all_epochs_metrics_df['Epoch'].values:  # 检查epoch是否存在
        best_pearson_epoch_original_pearson = all_epochs_metrics_df.loc[
            all_epochs_metrics_df['Epoch'] == epoch_for_best_pearson_scaled + 1, 'Test_Pearson_Original'].iloc[0]

    logger.info(
        f"--- Training Summary ({training_name}) ---\n"
        f"Best SCALED MSE achieved: {best_mse_overall_scaled:.5f} at epoch {epoch_for_best_mse_scaled + 1}\n"
        f"  (Associated ORIGINAL MSE at this epoch: {best_mse_epoch_original_mse:.5f})\n"
        f"Best SCALED Pearson achieved: {best_pearson_overall_scaled:.3f} at epoch {epoch_for_best_pearson_scaled + 1}\n"
        f"  (Associated ORIGINAL Pearson at this epoch: {best_pearson_epoch_original_pearson:.3f})\n"
        f"Final epoch ({params['epochs']}) SCALED MSE: {current_metrics_data['Test_MSE_Scaled']:.5f}, SCALED Pearson: {current_metrics_data['Test_Pearson_Scaled']:.3f}\n"
        f"Final epoch ({params['epochs']}) ORIGINAL MSE: {current_metrics_data['Test_MSE_Original']:.5f}, ORIGINAL Pearson: {current_metrics_data['Test_Pearson_Original']:.3f}"
    )
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    logger.info(f"Done with training run: {training_name}")  # 修改日志，更清晰


if __name__ == "__main__":
    # --- 保持您原始的 __main__ 块不变 ---
    parser = argparse.ArgumentParser(description='Train PaccMann model.')
    # Keeping args simple for this focused modification example
    # In a real scenario, you'd use argparse as in your original more complete script
    # For now, using direct path definitions for simplicity of testing this specific change

    # --- Define paths directly for testing this modification ---
    # PLEASE REPLACE THESE WITH YOUR ACTUAL PATHS
    base_data_dir = './'  # Or wherever your 'mydata', 'data', 'paccmann' folders are relative to

    train_sensitivity_filepath = os.path.join(base_data_dir, 'mydata/train_cell_line.csv')
    test_sensitivity_filepath = os.path.join(base_data_dir, 'mydata/test_cell_line.csv')
    gep_filepath = os.path.join(base_data_dir, 'mydata/exp0.csv')
    smi_filepath = os.path.join(base_data_dir, 'mydata/smile.smi')
    gene_filepath = os.path.join(base_data_dir, 'data/2128_genes.pkl')  # This was in 'data/'
    smiles_language_filepath = os.path.join(base_data_dir, 'paccmann/smiles_language.pkl')
    params_filepath = os.path.join(base_data_dir, 'paccmann/model_params.json')

    model_path_base = './paccman_training_runs/'
    os.makedirs(model_path_base, exist_ok=True)

    # ++++++++++++++++++++ 新增的循环部分 ++++++++++++++++++++
    # 定义要运行的次数
    number_of_runs = 1

    # 记录初始时间戳，用于生成一系列相关的训练名称
    base_timestamp = f'{time():.0f}'

    for i in range(number_of_runs):
        # 打印当前是第几次运行，方便跟踪
        print(f"\n{'=' * 20} STARTING TRAINING RUN {i + 1}/{number_of_runs} {'=' * 20}\n")

        # 为每次运行创建一个唯一的名称，确保结果不会相互覆盖
        # 格式：paccmann_train_{时间戳}_run_{序号}
        current_training_name = f'paccmann_train_{base_timestamp}_run_{i + 1}'

        # 调用主训练函数
        main(
            train_sensitivity_filepath,
            test_sensitivity_filepath,
            gep_filepath,
            smi_filepath,
            gene_filepath,
            smiles_language_filepath,
            model_path_base,
            params_filepath,
            current_training_name
        )

        print(f"\n{'=' * 20} FINISHED TRAINING RUN {i + 1}/{number_of_runs} {'=' * 20}\n")
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++