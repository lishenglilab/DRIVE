import datetime
import os
import random

import torch
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
from torch import nn
import torch.utils.data as Data
import torch.optim as optim
import numpy as np

# Assuming you have a GADRP.py file with GADRP_Net defined
from GADRP import GADRP_Net  # Make sure GADRP.py is in the same directory

# drug
drug_fingerprint_file = "../mydata/drug/drug_with_conditions.csv"
# cell
cell_index_file = "../mydata/cell_line/cell_index.csv"

cell_RNAseq_ae = "../mydata/cell_line/cell_RNAseq400_ae.pt"
cell_copynumber_ae = "../mydata/cell_line/cell_copynumber400_ae.pt"

# drug_cell pair
edge_idx_file = "../mydata/pair/edge_idx_file.pt"

# label
drug_cell_label_index_file = "../mydata/pair/drug_cell_label.pt"

lr = 0.0001
num_epoch = 150  # Reduced for quicker testing, you can set it back
batch_size = 1024

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

# --- 新增: 定义全局缩放范围 ---
# ---!!! 关键: 用户必须修改这里 !!!---
# 请根据您在预处理数据时使用的原始药物敏感性标签的全局最小值和最大值来更新这两个值。
# 这是计算正确RMSE的唯一前提。
ORIGINAL_MIN_VAL = 0.0  # 示例值，请替换为您的真实最小值
ORIGINAL_MAX_VAL = 15.0  # 示例值，请替换为您的真实最大值

# --- Define a directory for saved models ---
MODEL_SAVE_DIR = "saved_models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)  # Create directory if it doesn't exist


def data_process():
    # load molecular substructure fingerprints of drugs
    fingerprint = pd.read_csv(drug_fingerprint_file, sep=',', header=0, index_col=[0])
    # Check for NaN or Inf in fingerprint
    if fingerprint.isnull().values.any() or not np.isfinite(fingerprint.values).all():
        raise ValueError("Drug fingerprint data contains NaN or Inf values.")
    fingerprint = torch.from_numpy(fingerprint.values).float().to(device)

    # load gene expression data and DNA copy number data of cell lines
    cell_index = pd.read_csv(cell_index_file, sep=',', header=None, index_col=[0])
    RNAseq_feature = torch.load(cell_RNAseq_ae).to(device)
    copynumber_feature = torch.load(cell_copynumber_ae).to(device)
    # Check for NaN or Inf
    if torch.isnan(RNAseq_feature).any() or torch.isinf(RNAseq_feature).any():
        raise ValueError("RNAseq_feature contains NaN or Inf values.")
    if torch.isnan(copynumber_feature).any() or torch.isinf(copynumber_feature).any():
        raise ValueError("copynumber_feature contains NaN or Inf values.")

    # load drug-cell line similarity matrix
    edge_idx = torch.load(edge_idx_file).to(device)
    drug_cell_label = torch.load(drug_cell_label_index_file).to(device)
    if torch.isnan(drug_cell_label).any() or torch.isinf(drug_cell_label).any():
        raise ValueError("drug_cell_label contains NaN or Inf values.")

    return fingerprint, \
        list(cell_index.index), RNAseq_feature, copynumber_feature, \
        edge_idx, drug_cell_label


def split_data(drug_cell_label, mode='drug_blind', fold=5):
    """Splits data based on the specified mode."""

    drug_ids = torch.unique(drug_cell_label[:, 1]).cpu().numpy()
    cell_ids = torch.unique(drug_cell_label[:, 0]).cpu().numpy()
    num_drugs = len(drug_ids)
    num_cells = len(cell_ids)
    all_indices_np = torch.arange(len(drug_cell_label)).cpu().numpy()  # Use numpy for shuffling

    train_indices_list = []
    test_indices_list = []

    if mode == 'mix':
        for _ in range(fold):
            # Ensure all_indices_np is shuffled fresh for each fold
            current_indices = all_indices_np.copy()
            random.shuffle(current_indices)
            train_size = int(0.8 * len(current_indices))
            train_indices = torch.tensor(current_indices[:train_size], device=device)
            test_indices = torch.tensor(current_indices[train_size:], device=device)
            train_indices_list.append(train_indices)
            test_indices_list.append(test_indices)

    elif mode == 'drug_blind':
        for _ in range(fold):
            current_drug_ids = drug_ids.copy()
            random.shuffle(current_drug_ids)
            train_drugs_size = int(0.8 * num_drugs)
            train_drugs = current_drug_ids[:train_drugs_size]
            test_drugs = current_drug_ids[train_drugs_size:]

            # Use torch.isin on the original all_indices tensor derived on device
            all_indices_device = torch.arange(len(drug_cell_label)).to(device)
            train_indices = all_indices_device[torch.isin(drug_cell_label[:, 1], torch.tensor(train_drugs).to(device))]
            test_indices = all_indices_device[torch.isin(drug_cell_label[:, 1], torch.tensor(test_drugs).to(device))]

            train_indices_list.append(train_indices)
            test_indices_list.append(test_indices)
    elif mode == 'cell_blind':
        for _ in range(fold):
            current_cell_ids = cell_ids.copy()
            random.shuffle(current_cell_ids)
            train_cells_size = int(0.8 * num_cells)
            train_cells = current_cell_ids[:train_cells_size]
            test_cells = current_cell_ids[train_cells_size:]

            all_indices_device = torch.arange(len(drug_cell_label)).to(device)
            train_indices = all_indices_device[torch.isin(drug_cell_label[:, 0], torch.tensor(train_cells).to(device))]
            test_indices = all_indices_device[torch.isin(drug_cell_label[:, 0], torch.tensor(test_cells).to(device))]
            train_indices_list.append(train_indices)
            test_indices_list.append(test_indices)
    else:
        raise ValueError("Invalid mode. Choose from 'mix', 'drug_blind', 'cell_blind'.")

    return train_indices_list, test_indices_list


# --- MODIFIED training function to accept model_save_path ---
def training(model, drug_feature, cell_feature1, cell_feature2,
             edge_idx, data_iter, features_test, labels_test, optimizer, scheduler, model_save_path):
    """
    Trains the model and evaluates it on the test set. Saves the best model to model_save_path.
    """
    loss_fn = nn.MSELoss()

    best_test_loss = float('inf')
    best_pearson = -1.0  # Initialize with worst possible values
    best_spearman = -1.0
    best_r2 = -1.0
    best_epoch = -1

    for epoch in range(1, num_epoch + 1):
        model.train()
        epoch_train_loss = 0.0
        num_batches = 0

        for X, y in data_iter:
            y_pre = model(drug_feature, cell_feature1, cell_feature2, edge_idx, X)
            l = loss_fn(y_pre, y.view(-1, 1))

            optimizer.zero_grad()
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_train_loss += l.item()
            num_batches += 1

        avg_train_loss = epoch_train_loss / num_batches if num_batches > 0 else 0

        model.eval()
        with torch.no_grad():
            y_pre = model(drug_feature, cell_feature1, cell_feature2, edge_idx, features_test)
            l_test = loss_fn(y_pre, labels_test.view(-1, 1)).mean()

            # --- 新增代码: 计算反向缩放后的RMSE ---
            original_range = ORIGINAL_MAX_VAL - ORIGINAL_MIN_VAL
            y_pre_unscaled = y_pre * original_range + ORIGINAL_MIN_VAL
            labels_test_unscaled = labels_test.view(-1, 1) * original_range + ORIGINAL_MIN_VAL
            # 使用 loss_fn 计算 unscaled MSE, 然后开方得到 RMSE
            unscaled_rmse = torch.sqrt(loss_fn(y_pre_unscaled, labels_test_unscaled)).item()
            # --- 新增代码结束 ---

            try:
                p = pearsonr(y_pre.view(-1).cpu().numpy(), labels_test.view(-1).cpu().numpy())[0]
                s = spearmanr(y_pre.view(-1).cpu().numpy(), labels_test.view(-1).cpu().numpy())[0]
                r2 = r2_score(labels_test.view(-1).cpu().numpy(),
                              y_pre.view(-1).cpu().numpy())  # Corrected order for r2_score
            except ValueError as e:
                print(
                    f"Warning: Error calculating metrics: {e}. y_pre shape: {y_pre.shape}, labels_test shape: {labels_test.shape}")
                p, s, r2 = 0.0, 0.0, -1.0  # R2 can be negative

            # --- 修改: 在打印信息中加入新的 unscaled_RMSE ---
            print(
                f"e: {epoch}, tr_loss: {avg_train_loss:.4f}, te_loss: {l_test.item():.4f}, unscaled_RMSE: {unscaled_rmse:.4f}, "
                f"te_pearson: {p:.4f}, test_spearman: {s:.4f}, test_r2: {r2:.4f}")

            if l_test.item() < best_test_loss:
                best_test_loss = l_test.item()
                best_pearson = p
                best_spearman = s
                best_r2 = r2
                best_epoch = epoch
                # --- Save to the specific path ---
                torch.save(model.state_dict(), model_save_path)
                print(f"    Best model for current fold saved to {model_save_path} (Epoch {epoch})")

        scheduler.step()

    return best_test_loss, best_pearson, best_spearman, best_r2, best_epoch


def main():
    random.seed(88)
    np.random.seed(88)  # Also seed numpy for shuffles
    torch.manual_seed(88)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(88)

    fingerprint, cell_index, RNAseq_feature, copynumber_feature, edge_idx, drug_cell_label = data_process()

    modes = ['mix', 'drug_blind', 'cell_blind']
    results = {}  # Store results for all modes and folds

    for mode in modes:
        print(f"\n--- Training with mode: {mode} ---")
        results[mode] = []
        train_indices_list, test_indices_list = split_data(drug_cell_label, mode=mode, fold=5)  # set fold number

        for fold, (train_index, test_index) in enumerate(zip(train_indices_list, test_indices_list)):
            fold_num = fold + 1
            print(f"\n  Fold {fold_num}:")
            print("  Train set size:", len(train_index))
            print("  Test set size:", len(test_index))

            if len(train_index) == 0 or len(test_index) == 0:
                print(f"  Skipping fold {fold_num} due to empty train/test set.")
                results[mode].append({
                    'fold': fold_num,
                    'best_loss': float('inf'),
                    'best_pearson': -1.0,
                    'best_spearman': -1.0,
                    'best_r2': -1.0,
                    'best_epoch': -1,
                    'notes': 'Skipped due to empty data split'
                })
                continue

            features = drug_cell_label[train_index, :2]
            labels = drug_cell_label[train_index, 2]
            features_test = drug_cell_label[test_index, :2]
            labels_test = drug_cell_label[test_index, 2]
            dataset = Data.TensorDataset(features, labels)
            data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

            model = GADRP_Net(device).to(device)
            optimizer = optim.Adam(params=model.parameters(), lr=lr)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

            # --- Define unique model save path for this mode and fold ---
            model_save_filename = f"best_model_{mode}_fold{fold_num}.pth"
            current_model_save_path = os.path.join(MODEL_SAVE_DIR, model_save_filename)
            print(f"  Best model for this fold will be saved to: {current_model_save_path}")

            best_loss, best_pearson, best_spearman, best_r2, best_epoch = training(
                model, fingerprint, copynumber_feature, RNAseq_feature,
                edge_idx, data_iter, features_test, labels_test, optimizer, scheduler,
                model_save_path=current_model_save_path  # Pass the unique path
            )

            results[mode].append({
                'fold': fold_num,
                'best_loss': best_loss,
                'best_pearson': best_pearson,
                'best_spearman': best_spearman,
                'best_r2': best_r2,
                'best_epoch': best_epoch,
                'model_path': current_model_save_path  # Store path in results too
            })

    # Save results to CSV
    all_results_list = []
    for mode, fold_results in results.items():
        for res in fold_results:
            res['mode'] = mode  # Add mode to each result dictionary
            all_results_list.append(res)

    if all_results_list:
        df_all_results = pd.DataFrame(all_results_list)
        df_all_results.to_csv(f'all_training_results_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                              index=False)
        print(f"\nAll results saved to all_training_results_....csv")

    # Print overall best results for each mode
    print("\nOverall Best Results per Mode (based on Pearson correlation):")
    for mode, fold_results_list in results.items():
        # Filter out skipped folds if any
        valid_fold_results = [r for r in fold_results_list if 'notes' not in r]
        if not valid_fold_results:
            print(f"Mode: {mode} - No valid folds completed.")
            continue
        best_fold_data = max(valid_fold_results, key=lambda x: x['best_pearson'])
        print(f"Mode: {mode}, Best Fold: {best_fold_data['fold']}, "
              f"Best Pearson: {best_fold_data['best_pearson']:.4f}, "
              f"Best R2: {best_fold_data['best_r2']:.4f}, "
              f"Model: {best_fold_data['model_path']}")


if __name__ == '__main__':
    # Make sure GADRP.py and GADRP_Net exist
    # For testing, you might need to create a dummy GADRP.py:
    # class GADRP_Net(nn.Module):
    #     def __init__(self, device):
    #         super().__init__()
    #         self.device = device
    #         self.fc = nn.Linear(2,1) # Dummy layer, adjust based on X input in training
    #     def forward(self, drug_feature, cell_feature1, cell_feature2, edge_idx, X):
    #         # This is a very DUMMY forward pass.
    #         # Your actual GADRP_Net will use the other features.
    #         # For this dummy version, let's assume X has 2 features that map to 1 output.
    #         # If X are indices, this dummy won't work directly.
    #         # Assuming X is [batch_size, num_features_for_fc_layer]
    #         # print("Dummy model input X shape:", X.shape) # to debug
    #         # Dummy output based on the first two columns of edge_idx (pretend they are features of X)
    #         # This is highly dependent on what X actually is (indices or features)
    #         # If X are indices for drug_cell_label, then this simple fc needs adjustment
    #         # Or a dummy output:
    #         return torch.rand(X.shape[0], 1, device=self.device) * 0.1 # dummy output
    main()