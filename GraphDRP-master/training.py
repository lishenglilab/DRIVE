import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from utils import *
import datetime
import argparse
import math  # 确保 math 库已导入


# 新增: 还原IC50值的函数
def unscale_ic50(y, epsilon=1e-8):
    """
    根据公式 y_scaled = 1 / (1 + exp(-0.1 * y_original))
    反向计算 y_original = -10 * log((1 - y_scaled) / y_scaled)
    """
    y_clipped = np.clip(y, epsilon, 1 - epsilon)
    return -10 * np.log((1 - y_clipped) / y_clipped)


# training function at each epoch
def train(model, device, train_loader, optimizer, epoch, log_interval):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
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
                                                                           batch_idx * len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
    return sum(avg_loss) / len(avg_loss)


def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output, _ = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


# 'main' 函数现在接受一个额外的 'repetition' 参数来区分每次运行
def main(modeling, dataset_suffix, repetition, train_batch, val_batch, test_batch, lr, num_epoch, log_interval,
         cuda_name):
    print('Learning rate: ', lr)
    print('Epochs: ', num_epoch)

    model_st = modeling.__name__
    dataset = 'GDSC'
    train_losses, val_losses, val_pearsons = [], [], []

    print(f'\nRunning on {model_st} for dataset GDSC_{dataset_suffix}, Repetition: {repetition + 1}')
    processed_data_file_train = f'mydata/processed/{dataset}_train_{dataset_suffix}.pt'
    processed_data_file_val = f'mydata/processed/{dataset}_val_{dataset_suffix}.pt'
    processed_data_file_test = f'mydata/processed/{dataset}_test_{dataset_suffix}.pt'

    if not all(
            os.path.isfile(p) for p in [processed_data_file_train, processed_data_file_val, processed_data_file_test]):
        print(f'Data files for {dataset_suffix} not found! Please run create_data.py to prepare data.')
        return

    train_data = TestbedDataset(root='mydata', dataset=f'{dataset}_train_{dataset_suffix}')
    val_data = TestbedDataset(root='mydata', dataset=f'{dataset}_val_{dataset_suffix}')
    test_data = TestbedDataset(root='mydata', dataset=f'{dataset}_test_{dataset_suffix}')

    train_loader = DataLoader(train_data, batch_size=train_batch, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=val_batch, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=test_batch, shuffle=False)

    print("CPU/GPU: ", torch.cuda.is_available())
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = modeling().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_mse, best_pearson, best_epoch = 1000, -1, -1

    # 将 'repetition' 编号加入文件名，以防止覆盖
    model_file_name = f'model_{model_st}_{dataset}_{dataset_suffix}_run{repetition + 1}.model'
    result_file_name = f'result_{model_st}_{dataset}_{dataset_suffix}_run{repetition + 1}.csv'
    loss_fig_name = f'model_{model_st}_{dataset}_{dataset_suffix}_run{repetition + 1}_loss'
    pearson_fig_name = f'model_{model_st}_{dataset}_{dataset_suffix}_run{repetition + 1}_pearson'

    with open(result_file_name, 'w') as f:
        f.write(
            'epoch,val_rmse,val_mse,val_pearson,val_spearman,val_r2,val_rmse_unscaled,test_rmse,test_mse,test_person,test_spearman,test_r2,test_rmse_unscaled\n')

    for epoch in range(num_epoch):
        train_loss = train(model, device, train_loader, optimizer, epoch + 1, log_interval)

        G_val, P_val = predicting(model, device, val_loader)
        G_val_unscaled, P_val_unscaled = unscale_ic50(G_val), unscale_ic50(P_val)
        val_ret = [rmse(G_val, P_val), mse(G_val, P_val), pearson(G_val, P_val), spearman(G_val, P_val),
                   r2(G_val, P_val), rmse(G_val_unscaled, P_val_unscaled)]

        G_test, P_test = predicting(model, device, test_loader)
        G_test_unscaled, P_test_unscaled = unscale_ic50(G_test), unscale_ic50(P_test)
        test_ret = [rmse(G_test, P_test), mse(G_test, P_test), pearson(G_test, P_test), spearman(G_test, P_test),
                    r2(G_test, P_test), rmse(G_test_unscaled, P_test_unscaled)]

        train_losses.append(train_loss)
        val_losses.append(val_ret[1])
        val_pearsons.append(val_ret[2])

        with open(result_file_name, 'a') as f:
            epoch_results = [epoch + 1] + val_ret + test_ret
            f.write(','.join(map(str, epoch_results)) + '\n')

        if val_ret[1] < best_mse:
            torch.save(model.state_dict(), model_file_name)
            best_epoch, best_mse, best_pearson = epoch + 1, val_ret[1], val_ret[2]
            print(
                f'Val_MSE improved at epoch {best_epoch}; best_mse: {best_mse:.4f}, best_pearson: {best_pearson:.4f}. Model saved.')
        else:
            print(
                f'No improvement since epoch {best_epoch}; best_mse: {best_mse:.4f}, best_pearson: {best_pearson:.4f}')

    draw_loss(train_losses, val_losses, loss_fig_name)
    draw_pearson(val_pearsons, pearson_fig_name)
    print(
        f"Finished training for {model_st} on {dataset_suffix} (Repetition {repetition + 1}). Best model from epoch {best_epoch} saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train GraphDTA models on various datasets with multiple repetitions')
    parser.add_argument('--train_batch', type=int, required=False, default=1024, help='Batch size training set')
    parser.add_argument('--val_batch', type=int, required=False, default=1024, help='Batch size validation set')
    parser.add_argument('--test_batch', type=int, required=False, default=1024, help='Batch size test set')
    parser.add_argument('--lr', type=float, required=False, default=1e-4, help='Learning rate')
    parser.add_argument('--num_epoch', type=int, required=False, default=300, help='Number of epoch')
    parser.add_argument('--log_interval', type=int, required=False, default=20, help='Log interval')
    parser.add_argument('--cuda_name', type=str, required=False, default="cuda:0", help='Cuda')
    parser.add_argument('--repetitions', type=int, required=False, default=3,
                        help='Number of times to repeat each experiment')

    args = parser.parse_args()

    models = [GINConvNet, GATNet, GAT_GCN, GCNNet]
    dataset_types = ['mix', 'blind', 'cell_blind']

    # --- 这里是关键：实现了“再乘以3”的逻辑 ---
    # 1. 最外层的循环，用于控制重复次数
    for repetition in range(args.repetitions):
        print(f"\n{'#' * 70}")
        print(f"### STARTING REPETITION {repetition + 1} / {args.repetitions} ###")
        print(f"{'#' * 70}\n")

        # 2. 内层循环遍历数据集类型
        for dataset_suffix in dataset_types:
            # 3. 最内层循环遍历模型
            for model_class in models:
                print(f"\n{'=' * 60}")
                print(
                    f"STARTING TRAINING: Model: {model_class.__name__}, Dataset: GDSC_{dataset_suffix}, Repetition: {repetition + 1}")
                print(f"{'=' * 60}\n")

                # 4. 调用 main 函数，将 'repetition' 编号传入
                main(model_class,
                     dataset_suffix,
                     repetition,  # 传递当前是第几次重复
                     args.train_batch,
                     args.val_batch,
                     args.test_batch,
                     args.lr,
                     args.num_epoch,
                     args.log_interval,
                     args.cuda_name)