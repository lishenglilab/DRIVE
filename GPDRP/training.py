import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
import json
import datetime
import argparse
import csv
import math

# --- 假设模型和 utils 已正确导入 ---
# 为了脚本能独立运行，这里提供Mocks
# 在您的环境中，请确保这些都已正确导入
try:
    from GAT.model.gat import GATNet
    from GCN.model.gcn import GCNNet
    from GIN.model.gin import GINConvNet
    from GIN_TRANSFORMER.model.gintranformer import GINConvNet2
    from GPDRP.utils import *
except ImportError:
    print("Warning: Could not import models or utils. Using mock versions for demonstration.")
    from torch_geometric.nn import global_mean_pool


    class PlaceholderGNN(nn.Module):
        def __init__(self):
            super(PlaceholderGNN, self).__init__()
            self.lin1 = nn.Linear(78, 128)
            self.lin2 = nn.Linear(128, 1)

        def forward(self, data):
            x = torch.relu(self.lin1(data.x))
            graph_embedding = global_mean_pool(x, data.batch if hasattr(data, 'batch') else torch.zeros(data.x.size(0),
                                                                                                        dtype=torch.long,
                                                                                                        device=data.x.device))
            output = self.lin2(graph_embedding)
            return output, graph_embedding


    GATNet, GCNNet, GINConvNet, GINConvNet2 = PlaceholderGNN, PlaceholderGNN, PlaceholderGNN, PlaceholderGNN

    from math import sqrt
    from scipy import stats
    import matplotlib.pyplot as plt
    from torch_geometric.data import InMemoryDataset, DataLoader


    def rmse(y, f):
        return sqrt(((y - f) ** 2).mean(axis=0))


    def mse(y, f):
        return ((y - f) ** 2).mean(axis=0)


    def pearson(y, f):
        if np.std(y) == 0 or np.std(f) == 0: return 0.0
        return np.corrcoef(y, f)[0, 1]


    def spearman(y, f):
        return stats.spearmanr(y, f)[0]


    def draw_loss(train_losses, test_losses, title):
        plt.figure();
        plt.plot(train_losses, label='train loss');
        plt.plot(test_losses, label='test loss');
        plt.title(title);
        plt.xlabel('Epoch');
        plt.ylabel('Loss');
        plt.legend();
        plt.savefig(title + ".png");
        plt.close()


    def draw_pearson(pearsons, title):
        plt.figure();
        plt.plot(pearsons, label='test pearson');
        plt.title(title);
        plt.xlabel('Epoch');
        plt.ylabel('Pearson');
        plt.legend();
        plt.savefig(title + ".png");
        plt.close()


    # TestbedDataset mock updated to handle root correctly
    class TestbedDataset(InMemoryDataset):
        def __init__(self, root, dataset, **kwargs):
            self.dataset_name = dataset
            super(TestbedDataset, self).__init__(root, **kwargs)
            self.data, self.slices = torch.load(self.processed_paths[0])

        @property
        def processed_file_names(self):
            return [f'{self.dataset_name}.pt']
# --- End of Mocks ---


# --- Seed setting ---
seed_value = 88
np.random.seed(seed_value)
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --- 还原函数 (来自上次修改，保持不变) ---
def unscale_ic50(y):
    y = np.clip(y, 1e-10, 1.0 - 1e-10)
    return -10 * np.log((1 / y) - 1)


# --- Training and Predicting functions (您的代码，完全不变) ---
def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    loss_fn = nn.MSELoss()
    avg_loss = []
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output, _ = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        avg_loss.append(loss.item())
        if batch_idx % log_interval == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * data.num_graphs,
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
    return sum(avg_loss) / len(avg_loss)


def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output, _ = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


# --- Main experiment execution logic (只修改路径部分) ---
def run_experiment(model_class, model_name_str, dataset_name,
                   train_batch, val_batch, test_batch, lr, num_epoch, log_interval, cuda_name,
                   num_runs=1):
    print(f"\n===== Processing Model: {model_name_str} on Dataset: {dataset_name} =====")
    print(f"Learning rate: {lr}, Epochs: {num_epoch}, Batch Size (Train): {train_batch}")

    # 动态生成数据文件路径
    processed_data_file_train = f'mydata/processed/{dataset_name}_train.pt'
    processed_data_file_val = f'mydata/processed/{dataset_name}_val.pt'
    processed_data_file_test = f'mydata/processed/{dataset_name}_test.pt'

    if not all(
            os.path.isfile(f) for f in [processed_data_file_train, processed_data_file_val, processed_data_file_test]):
        print(f'Data files not found for {dataset_name}. Please run preparation script.')
        print(
            f'Missing: \n - {processed_data_file_train if not os.path.isfile(processed_data_file_train) else "OK"}\n - {processed_data_file_val if not os.path.isfile(processed_data_file_val) else "OK"}\n - {processed_data_file_test if not os.path.isfile(processed_data_file_test) else "OK"}')
        return None

    ### 路径修正 ###
    # root 应该是 'mydata', dataset 应该是 'GDSC_mix_train' 这样的形式
    train_data = TestbedDataset(root='mydata', dataset=f'{dataset_name}_train')
    val_data = TestbedDataset(root='mydata', dataset=f'{dataset_name}_val')
    test_data = TestbedDataset(root='mydata', dataset=f'{dataset_name}_test')

    train_loader = DataLoader(train_data, batch_size=train_batch, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=val_batch, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=test_batch, shuffle=False)

    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    overall_best_val_mse_for_model = float('inf')
    best_run_summary_for_model = {}

    for run_idx in range(num_runs):
        print(f"\n--- Model: {model_name_str}, Run: {run_idx + 1}/{num_runs}, Dataset: {dataset_name} ---")

        model = model_class().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best_val_mse_this_run = float('inf')
        best_val_pearson_this_run = -1
        best_epoch_this_run = -1

        run_specific_tag = f"{model_name_str}_{dataset_name}_run{run_idx + 1}"
        model_file_name = f'output/models/model_{run_specific_tag}.model'
        result_file_name = f'output/results/result_epoch_history_{run_specific_tag}.csv'
        predictions_file_name = f'output/predictions/predictions_test_{run_specific_tag}_best_val.csv'
        loss_fig_name = f'output/plots/loss_{run_specific_tag}'
        pearson_fig_name = f'output/plots/pearson_{run_specific_tag}'

        for d in ['output/models', 'output/results', 'output/predictions', 'output/plots']:
            os.makedirs(d, exist_ok=True)

        train_losses_epoch = []
        val_losses_epoch = []
        val_pearsons_epoch = []
        all_epoch_results_this_run = []

        test_metrics_at_best_val = None
        test_rmse_orig_at_best_val = None

        for epoch in range(num_epoch):
            print(f"Epoch {epoch + 1}/{num_epoch}")
            train_loss = train(model, device, train_loader, optimizer, epoch + 1, log_interval)

            G_val, P_val = predicting(model, device, val_loader)
            val_metrics = [rmse(G_val, P_val), mse(G_val, P_val), pearson(G_val, P_val), spearman(G_val, P_val),
                           r2_score(G_val, P_val)]

            G_test, P_test = predicting(model, device, test_loader)
            current_test_metrics = [rmse(G_test, P_test), mse(G_test, P_test), pearson(G_test, P_test),
                                    spearman(G_test, P_test), r2_score(G_test, P_test)]

            # 还原RMSE计算 (来自上次修改，保持不变)
            G_val_orig, P_val_orig = unscale_ic50(G_val), unscale_ic50(P_val)
            G_test_orig, P_test_orig = unscale_ic50(G_test), unscale_ic50(P_test)
            val_rmse_orig, test_rmse_orig = rmse(G_val_orig, P_val_orig), rmse(G_test_orig, P_test_orig)

            train_losses_epoch.append(train_loss)
            val_losses_epoch.append(val_metrics[1])
            val_pearsons_epoch.append(val_metrics[2])

            all_epoch_results_this_run.append(
                [epoch + 1, train_loss] + val_metrics + current_test_metrics + [val_rmse_orig, test_rmse_orig])

            if val_metrics[1] < best_val_mse_this_run:
                best_val_mse_this_run = val_metrics[1]
                best_val_pearson_this_run = val_metrics[2]
                best_epoch_this_run = epoch + 1
                test_metrics_at_best_val = current_test_metrics
                test_rmse_orig_at_best_val = test_rmse_orig

                torch.save(model.state_dict(), model_file_name)
                with open(predictions_file_name, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(
                        ['Ground_Truth_Scaled', 'Prediction_Scaled', 'Ground_Truth_Orig', 'Prediction_Orig'])
                    for i in range(len(G_test)):
                        writer.writerow([G_test[i], P_test[i], G_test_orig[i], P_test_orig[i]])
                print(
                    f'Run {run_idx + 1}, Epoch {epoch + 1}: Val MSE improved to {best_val_mse_this_run:.4f}, Val Pearson: {best_val_pearson_this_run:.4f}. Model saved.')
            else:
                if epoch % 10 == 0:
                    print(
                        f'Run {run_idx + 1}, Epoch {epoch + 1}: No improvement. Best Val MSE: {best_val_mse_this_run:.4f} at epoch {best_epoch_this_run}.')

        with open(result_file_name, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train_Loss',
                             'Val_RMSE_Scaled', 'Val_MSE_Scaled', 'Val_Pearson', 'Val_Spearman', 'Val_R2',
                             'Test_RMSE_Scaled', 'Test_MSE_Scaled', 'Test_Pearson', 'Test_Spearman', 'Test_R2',
                             'Val_RMSE_Orig', 'Test_RMSE_Orig'])
            writer.writerows(all_epoch_results_this_run)
        print(f"Full epoch history for run {run_idx + 1} saved to {result_file_name}")

        draw_loss(train_losses_epoch, val_losses_epoch, loss_fig_name)
        draw_pearson(val_pearsons_epoch, pearson_fig_name)
        print(f"Loss and Pearson plots for run {run_idx + 1} saved.")

        if best_val_mse_this_run < overall_best_val_mse_for_model:
            overall_best_val_mse_for_model = best_val_mse_this_run
            best_run_summary_for_model = {
                'model_name': model_name_str,
                'dataset': dataset_name,
                'run_id': run_idx + 1,
                'best_epoch': best_epoch_this_run,
                'val_mse_scaled': best_val_mse_this_run,
                'val_pearson': best_val_pearson_this_run,
                'test_rmse_scaled_at_best_val': test_metrics_at_best_val[0] if test_metrics_at_best_val else float(
                    'nan'),
                'test_mse_scaled_at_best_val': test_metrics_at_best_val[1] if test_metrics_at_best_val else float(
                    'nan'),
                'test_pearson_at_best_val': test_metrics_at_best_val[2] if test_metrics_at_best_val else float('nan'),
                'test_spearman_at_best_val': test_metrics_at_best_val[3] if test_metrics_at_best_val else float('nan'),
                'test_r2_at_best_val': test_metrics_at_best_val[4] if test_metrics_at_best_val else float('nan'),
                'test_rmse_orig_at_best_val': test_rmse_orig_at_best_val if test_rmse_orig_at_best_val is not None else float(
                    'nan'),
                'model_file_path': model_file_name,
                'hyperparameters': {
                    'lr': lr, 'train_batch': train_batch, 'num_epoch_completed': best_epoch_this_run
                }
            }
    return best_run_summary_for_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train multiple GNN models')
    parser.add_argument('--dataset', type=str, required=False, default='GDSC',
                        help='Base dataset name (e.g., GDSC, CCLE)')
    parser.add_argument('--train_batch', type=int, required=False, default=64, help='Batch size training set')
    parser.add_argument('--val_batch', type=int, required=False, default=64, help='Batch size validation set')
    parser.add_argument('--test_batch', type=int, required=False, default=64, help='Batch size test set')
    parser.add_argument('--lr', type=float, required=False, default=1e-4, help='Learning rate')
    parser.add_argument('--num_epoch', type=int, required=False, default=200, help='Number of epoch')
    parser.add_argument('--log_interval', type=int, required=False, default=20, help='Log interval within epoch')
    parser.add_argument('--cuda_name', type=str, required=False, default="cuda:0",
                        help='Cuda device name (e.g., cuda:0, cuda:1)')
    parser.add_argument('--num_runs', type=int, required=False, default=3,
                        help='Number of independent runs for each model and dataset split combination.')

    args = parser.parse_args()

    MODELS_TO_RUN = [
        {"class": GINConvNet, "name": "GIN"},
        {"class": GATNet, "name": "GAT"},
        {"class": GCNNet, "name": "GCN"},
        {"class": GINConvNet2, "name": "GINTransformer"},
    ]

    split_types = ['mix', 'cell_blind', 'drug_blind']

    all_experiments_best_summary = []

    for model_config in MODELS_TO_RUN:
        for split_type in split_types:
            full_dataset_name = f"{args.dataset}_{split_type}"

            best_summary_for_this_combo = run_experiment(
                model_class=model_config["class"],
                model_name_str=model_config["name"],
                dataset_name=full_dataset_name,
                train_batch=args.train_batch,
                val_batch=args.val_batch,
                test_batch=args.test_batch,
                lr=args.lr,
                num_epoch=args.num_epoch,
                log_interval=args.log_interval,
                cuda_name=args.cuda_name,
                num_runs=args.num_runs
            )
            if best_summary_for_this_combo:
                all_experiments_best_summary.append(best_summary_for_this_combo)

    summary_file_path = f'output/overall_best_runs_summary_{args.dataset}_all_splits.json'
    os.makedirs('output', exist_ok=True)

    print(f"\n===== All Experiments Complete =====")
    print(f"Summary of best runs for all models and splits saved to: {summary_file_path}")

    if all_experiments_best_summary:
        summary_df = pd.DataFrame(all_experiments_best_summary)
        if 'hyperparameters' in summary_df.columns:
            try:
                hyperparams_df = summary_df['hyperparameters'].apply(pd.Series)
                summary_df = pd.concat([summary_df.drop('hyperparameters', axis=1), hyperparams_df], axis=1)
            except Exception as e:
                print(f"Could not expand hyperparameters column for CSV: {e}")

        summary_csv_path = f'output/overall_best_runs_summary_{args.dataset}_all_splits.csv'
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"CSV summary saved to: {summary_csv_path}")