# predict_single_chunk.py (生产级最终版 - 内存清洗 & 设备修复)
# 版本说明：
# - 保持原有的基于文件大小的断点续跑逻辑。
# - 在处理每个 small_chunk 时，在内存中检测并丢弃特征全为空或NaN的行。
# - 修复了数据张量未被正确移动到GPU，导致 "cuda:0 and cpu" 的设备不匹配错误。

import argparse
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import init
import math
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr
import scipy.sparse as sp
from scipy.sparse import coo_matrix
import torch.utils.data as Data
import random
import datetime
import gc
import traceback

# 导入tqdm等库...
try:
    from tqdm import tqdm
except ImportError:
    print("错误: 'tqdm' 库未找到。")
    print("请在您的环境中运行 'pip install tqdm' 来安装它。")
    def tqdm(iterable, *args, **kwargs):
        return iterable
try:
    import mygene
except ImportError:
    print("错误: 'mygene' 库未找到。")
    print("请在您的环境中运行 'pip install mygene' 来安装它。")
    exit()

# ==============================================================================
# 1. 模型定义 (无变化)
# ==============================================================================
class Auto_Encoder(nn.Module):
    def __init__(self, device, indim, outdim=400):
        super(Auto_Encoder, self).__init__();
        self.encoder = Encoder(device=device, indim=indim, outdim=outdim);
        self.decoder = Decoder(device=device, outdim=indim, indim=outdim)
    def forward(self, x):
        encoded = self.encoder(x);
        decoded = self.decoder(encoded);
        return encoded, decoded
    def output(self, x): return self.encoder(x)
class Encoder(nn.Module):
    def __init__(self, device, indim, outdim=400):
        super(Encoder, self).__init__();
        self.linear1 = nn.Linear(indim, 2048, device=device);
        self.linear2 = nn.Linear(2048, 1024, device=device);
        self.linear3 = nn.Linear(1024, outdim, device=device)
    def forward(self, x):
        x = nn.SELU()(self.linear1(x));
        x = nn.SELU()(self.linear2(x));
        x = nn.Sigmoid()(self.linear3(x));
        return x
class Decoder(nn.Module):
    def __init__(self, device, outdim, indim=400):
        super(Decoder, self).__init__();
        self.linear3 = nn.Linear(indim, 1024, device=device);
        self.linear2 = nn.Linear(1024, 2048, device=device);
        self.linear1 = nn.Linear(2048, outdim, device=device)
    def forward(self, x):
        x = nn.SELU()(self.linear3(x));
        x = nn.SELU()(self.linear2(x));
        x = nn.Sigmoid()(self.linear1(x));
        return x
class GraphConvolution(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype};
        super(GraphConvolution, self).__init__();
        self.in_features = in_features;
        self.out_features = out_features;
        self.weight = Parameter(torch.empty((in_features, out_features), **factory_kwargs));
        self.a = Parameter(torch.zeros((1), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty((out_features), **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None: fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight); bound = 1 / math.sqrt(
            fan_in) if fan_in > 0 else 0; init.uniform_(self.bias, -bound, bound)
    def forward(self, H0, input, adj, a):
        H = (1 - a) * torch.matmul(adj, input) + a * H0;
        output = torch.matmul(H, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
class Drug_cell_encoder(nn.Module):
    def __init__(self, indim, device):
        super(Drug_cell_encoder, self).__init__();
        self.gcn = GraphConvolution(indim, indim, device=device);
        self.relu = nn.ReLU(inplace=True);
        self.dropout = nn.Dropout(0.2)
    def forward(self, drug_cell_pair_feature, edge_idx):
        output1 = self.relu(self.gcn(drug_cell_pair_feature, drug_cell_pair_feature, edge_idx, 0.1));
        output1 = self.dropout(output1);
        output2 = self.relu(self.gcn(drug_cell_pair_feature, output1, edge_idx, 0.1));
        output3 = self.relu(self.gcn(drug_cell_pair_feature, output2, edge_idx, 0.1));
        output3 = self.dropout(output3);
        output4 = self.relu(self.gcn(drug_cell_pair_feature, output3, edge_idx, 0.1));
        output5 = self.relu(self.gcn(drug_cell_pair_feature, output4, edge_idx, 0.1));
        output5 = self.dropout(output5);
        return output1, output2, output3, output4, output5
class GADRP_Net(nn.Module):
    def __init__(self, device):
        super(GADRP_Net, self).__init__();
        self.device = device;
        self.drugfc1 = nn.Linear(881, 200);
        self.cellfc1 = nn.Linear(800, 200);
        self.embedding = Drug_cell_encoder(400, device=device);
        self.att = Parameter(torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], device=device, dtype=torch.float) / 5.0);
        self.fc1 = nn.Linear(400, 256);
        self.fc2 = nn.Linear(256, 128);
        self.fc3 = nn.Linear(128, 1);
        self.relu = nn.ReLU(inplace=True);
        self.sigmoid = nn.Sigmoid();
        self.dropout = nn.Dropout(0.2)
    def forward(self, drug_feature_input, cell_feature1_input, cell_feature2_input, edge_idx_input,
                drug_cell_indices_to_select):
        drug_feature_processed = self.relu(self.drugfc1(drug_feature_input));
        drug_feature_processed = self.dropout(drug_feature_processed);
        cell_feature_combined = torch.cat((cell_feature1_input, cell_feature2_input), dim=1);
        cell_feature_processed = self.relu(self.cellfc1(cell_feature_combined));
        cell_feature_processed = self.dropout(cell_feature_processed);
        gcn_batch_drug_num = drug_feature_processed.shape[0];
        gcn_batch_cell_num = cell_feature_processed.shape[0];
        list_drug_gcn_indices = torch.arange(gcn_batch_drug_num, device=self.device).view(-1, 1).repeat(1,
                                                                                                        gcn_batch_cell_num).view(-1);
        list_cell_gcn_indices = torch.arange(gcn_batch_cell_num, device=self.device).repeat(gcn_batch_drug_num);
        drug_cell_pair_feature_for_gcn = torch.cat(
            (drug_feature_processed[list_drug_gcn_indices], cell_feature_processed[list_cell_gcn_indices]), dim=1)
        if drug_cell_pair_feature_for_gcn.shape[0] != edge_idx_input.shape[0]:
            raise ValueError(f"GCN input dimension mismatch: Features for {drug_cell_pair_feature_for_gcn.shape[0]} nodes, but adjacency matrix is for {edge_idx_input.shape[0]} nodes.")
        emb_out1, emb_out2, emb_out3, emb_out4, emb_out5 = self.embedding(drug_cell_pair_feature_for_gcn, edge_idx_input);
        selection_indices = (drug_cell_indices_to_select[:, 0] * gcn_batch_cell_num + drug_cell_indices_to_select[:, 1]).long();
        feature1 = emb_out1[selection_indices];
        feature2 = emb_out2[selection_indices];
        feature3 = emb_out3[selection_indices];
        feature4 = emb_out4[selection_indices];
        feature5 = emb_out5[selection_indices];
        feature = self.att[0] * feature1 + self.att[1] * feature2 + self.att[2] * feature3 + self.att[3] * feature4 + self.att[4] * feature5;
        feature = self.dropout(feature);
        output = self.fc1(feature);
        output = self.relu(output);
        output = self.dropout(output);
        output = self.fc2(output);
        output = self.relu(output);
        output = self.dropout(output);
        output = self.fc3(output);
        output = self.sigmoid(output);
        return output

# ==============================================================================
# 2. 辅助函数
# ==============================================================================
def sym_adj(adj):
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj.tocoo()
    rowsum = np.array(adj.sum(1))
    rowsum[np.isnan(rowsum)] = 0.
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_inv_sqrt[np.isnan(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).tocoo()

def get_ic50_scaling_parameters(args):
    print("  > 正在动态计算IC50缩放参数...")
    try:
        cell_index_list = pd.read_csv(args.cell_index_file, sep=',', header=None, index_col=0).index.tolist()
        drug_cell_label = pd.read_csv(args.drug_response_file, sep=',', header=0, usecols=["ccle_name", "pubchem_cid", "ic50"])
        drug_cell_label = drug_cell_label.dropna(axis=0)
        drug_cell_label = drug_cell_label[drug_cell_label.ccle_name.isin(cell_index_list)]
        drug_cell_label = drug_cell_label.loc[drug_cell_label["ic50"] > 0]
        drug_cell_label_sorted = drug_cell_label.sort_values(by=["ic50"])
        Q1_index = math.ceil(len(drug_cell_label_sorted) / 4) - 1
        Q3_index = math.ceil(len(drug_cell_label_sorted) / 4 * 3) - 1
        Q1 = drug_cell_label_sorted.iloc[Q1_index]["ic50"]
        Q3 = drug_cell_label_sorted.iloc[Q3_index]["ic50"]
        IQR = Q3 - Q1
        LOW = Q1 - 1.5 * IQR
        HIGH = Q3 + 1.5 * IQR
        drug_cell_label_filtered = drug_cell_label_sorted[(drug_cell_label_sorted["ic50"] >= LOW) & (drug_cell_label_sorted["ic50"] <= HIGH)]
        ic50_values_for_scaling = drug_cell_label_filtered["ic50"].values.astype(float)
        min_val, max_val = np.min(ic50_values_for_scaling), np.max(ic50_values_for_scaling)
        print(f"    > 计算完成: min_ic50 = {min_val}, max_ic50 = {max_val}")
        return min_val, max_val
    except Exception as e:
        print(f"    错误: 动态计算IC50缩放参数失败: {e}")
        return None, None

def dynamic_train_ae(data, name, args, device):
    if len(data) == 0:
        raise ValueError(f"无法训练 {name.upper()} AE，因为没有不重叠的旧数据样本。")
    random.seed(4); torch.manual_seed(4)
    in_dim = data.shape[1]
    model = Auto_Encoder(device, in_dim, 400).to(device)
    print(f"      > 开始动态训练 {name.upper()} AE (输入维度: {in_dim}, 样本数: {len(data)})...")
    data_loader = Data.DataLoader(data, batch_size=min(428, len(data)), shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_func = nn.MSELoss()
    best_model_state, best_loss = model.state_dict(), float('inf')
    epoch_iterator = tqdm(range(1, args.ae_epochs + 1), desc=f"训练 {name.upper()} AE", leave=False)
    for epoch in epoch_iterator:
        epoch_loss = 0.0
        model.train()
        for x in data_loader:
            x = x.to(device)
            _, decoded = model(x)
            train_loss = loss_func(decoded, x)
            optimizer.zero_grad(); train_loss.backward(); optimizer.step()
            epoch_loss += train_loss.item()
        avg_epoch_loss = epoch_loss / len(data_loader)
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_model_state = model.state_dict()
        epoch_iterator.set_postfix(loss=f"{avg_epoch_loss:.6f}", best_loss=f"{best_loss:.6f}")
    print(f"      > {name.upper()} AE 训练完成，最终最佳损失: {best_loss:.6f}")
    best_model = Auto_Encoder(device, in_dim, 400).to(device)
    best_model.load_state_dict(best_model_state)
    best_model.eval()
    return best_model

def prepare_drug_chunks(args):
    output_dir = args.temp_chunk_dir
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
        print(f"--- ✓ 检测到已存在的临时大块文件目录: {output_dir}")
        print("--- ✓ 将跳过分割步骤，直接使用这些文件进行预测。")
        return
    print(f"--- 警告: 未找到临时大块文件目录，现在开始生成...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"    > 创建临时目录: {output_dir}")
    print(f"    > 开始分割文件: {args.new_drug_feature_file}")
    print(f"    > 每个大块包含 {args.big_chunk_size} 行。")
    try:
        chunk_iter = pd.read_csv(args.new_drug_feature_file, index_col='ID', chunksize=args.big_chunk_size)
        for i, chunk in enumerate(chunk_iter):
            if 'Unnamed: 0' in chunk.columns:
                chunk = chunk.drop(columns=['Unnamed: 0'])
            chunk_num = i + 1
            output_file_path = os.path.join(output_dir, f"drug_big_chunk_{chunk_num}.csv")
            chunk.to_csv(output_file_path, index=True, index_label='ID')
            print(f"      > 已生成大块临时文件: {output_file_path}")
        print("\n--- ✓ 临时文件生成完成！ ---")
    except FileNotFoundError:
        print(f"致命错误：找不到原始新药文件 {args.new_drug_feature_file}。请检查路径。")
        exit()
    except KeyError:
        print(f"致命错误：在文件 {args.new_drug_feature_file} 中找不到名为 'ID' 的列。请检查文件内容。")
        exit()
    except Exception as e:
        print(f"分割过程中发生致命错误: {e}")
        exit()

def preprocess_all_cells(args, device):
    print("--- 1. 开始全局细胞系预处理 ---")
    if not (args.new_exp_file and os.path.exists(args.new_exp_file)):
        raise FileNotFoundError("新细胞系表达文件未找到，无法继续。")
    new_exp_df_raw = pd.read_csv(args.new_exp_file, index_col=0)
    new_cell_names = new_exp_df_raw.index.tolist()
    new_cell_names_set = set(new_cell_names)
    print(f"    > 识别到 {len(new_cell_names)} 个新细胞系。")
    all_train_cell_names = pd.read_csv(args.cell_index_file, header=None, index_col=0).index.tolist()
    unique_train_cell_names = [name for name in all_train_cell_names if name not in new_cell_names_set]
    print(f"    > 从 {len(all_train_cell_names)} 个旧细胞系中筛选出 {len(unique_train_cell_names)} 个不重叠的用于背景建模。")
    print("    > 处理基因表达数据...")
    train_exp_df = pd.read_csv(args.train_exp_file, sep=',', header=None, index_col=0, skiprows=1).loc[unique_train_cell_names]
    train_exp_df.columns = train_exp_df.columns.astype(str)
    exp_scaler = MinMaxScaler().fit(train_exp_df.values)
    exp_ae = dynamic_train_ae(torch.from_numpy(exp_scaler.transform(train_exp_df.values)).float(), 'exp', args, device)
    with torch.no_grad():
        old_exp_ae = exp_ae.output(torch.from_numpy(exp_scaler.transform(train_exp_df.values)).float().to(device))
    print("    > 处理拷贝数数据...")
    train_cn_df = pd.read_csv(args.train_cn_file, sep=',', header=None, index_col=0, skiprows=1).loc[unique_train_cell_names]
    train_cn_df.columns = train_cn_df.columns.astype(str)
    cn_scaler = MinMaxScaler().fit(train_cn_df.values)
    cn_ae = dynamic_train_ae(torch.from_numpy(cn_scaler.transform(train_cn_df.values)).float(), 'cn', args, device)
    with torch.no_grad():
        old_cn_ae = cn_ae.output(torch.from_numpy(cn_scaler.transform(train_cn_df.values)).float().to(device))
    print("    > 查询基因ID...")
    new_cn_df_raw = pd.read_csv(args.new_cn_file, index_col=0).reindex(new_cell_names).fillna(0)
    all_gene_symbols = set(new_exp_df_raw.columns.tolist() + new_cn_df_raw.columns.tolist())
    gene_symbol_to_id_map = {}
    if all_gene_symbols:
        mg = mygene.MyGeneInfo()
        gene_list = list(all_gene_symbols)
        for i in tqdm(range(0, len(gene_list), 1000), desc="  > 基因ID查询中", leave=False):
            batch = gene_list[i:i + 1000]
            try:
                query_result = mg.querymany(batch, scopes='symbol', fields='entrezgene', species='human', verbose=False)
                for query in query_result:
                    if 'entrezgene' in query and 'query' in query:
                        gene_symbol_to_id_map[query['query']] = str(query['entrezgene'])
            except Exception as e: print(f"查询批次失败: {e}")
    del all_gene_symbols, gene_list, mg; gc.collect()
    def rename_cols(df):
        rename_map = {s: g for s, g in gene_symbol_to_id_map.items() if s in df.columns}
        df_renamed = df.rename(columns=rename_map)
        valid_cols = [c for c in df_renamed.columns if c in gene_symbol_to_id_map.values()]
        return df_renamed[valid_cols].T.groupby(level=0).mean().T
    print("    > 处理新细胞系特征...")
    new_exp_df = rename_cols(new_exp_df_raw).reindex(columns=train_exp_df.columns, fill_value=0)
    new_cn_df = rename_cols(new_cn_df_raw).reindex(columns=train_cn_df.columns, fill_value=0)
    del new_exp_df_raw, new_cn_df_raw, gene_symbol_to_id_map; gc.collect()
    with torch.no_grad():
        new_exp_ae = exp_ae.output(torch.from_numpy(exp_scaler.transform(new_exp_df.values)).float().to(device))
        new_cn_ae = cn_ae.output(torch.from_numpy(cn_scaler.transform(new_cn_df.values)).float().to(device))
    all_exp_features = torch.cat([old_exp_ae, new_exp_ae], dim=0)
    all_cn_features = torch.cat([old_cn_ae, new_cn_ae], dim=0)
    del old_exp_ae, new_exp_ae, old_cn_ae, new_cn_ae, new_exp_df, new_cn_df, exp_ae, cn_ae; gc.collect()
    print("    > 计算细胞相似度...")
    train_meth_df = pd.read_csv(args.train_meth_file, index_col=0, header=None, skiprows=1).loc[unique_train_cell_names]
    meth_scaler = MinMaxScaler().fit(train_meth_df.values)
    old_meth_scaled = meth_scaler.transform(train_meth_df.values)
    new_meth_df_raw = pd.read_csv(args.new_meth_file, index_col=0).reindex(new_cell_names).fillna(0)
    new_meth_df = new_meth_df_raw.reindex(columns=train_meth_df.columns, fill_value=0)
    new_meth_scaled = meth_scaler.transform(new_meth_df.values)
    del train_meth_df, new_meth_df_raw, new_meth_df, meth_scaler; gc.collect()
    all_meth_scaled = np.vstack([old_meth_scaled, new_meth_scaled])
    all_cell_names = unique_train_cell_names + new_cell_names
    cell_sim = torch.zeros(len(all_cell_names), len(all_cell_names), device=device)
    for i in range(len(all_meth_scaled)):
        for j in range(i, len(all_meth_scaled)):
            p_val = 0.0 # Default value
            if np.std(all_meth_scaled[i]) > 0 and np.std(all_meth_scaled[j]) > 0:
                try:
                    p_val = pearsonr(all_meth_scaled[i], all_meth_scaled[j])[0]
                    if np.isnan(p_val): p_val = 0.0 # Handle NaN from pearsonr
                except ValueError: # Handle cases where input contains NaN
                    p_val = 0.0
            cell_sim[i, j] = cell_sim[j, i] = abs(p_val)
    _, cell_sim_top10 = torch.topk(cell_sim, k=min(10, len(all_cell_names)), dim=1)
    del all_meth_scaled, cell_sim; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("--- ✓ 全局细胞系预处理完成 ---")
    return {
        "all_cell_names": all_cell_names,
        "train_cell_names": unique_train_cell_names,
        "new_cell_names": new_cell_names,
        "all_exp_features": all_exp_features,
        "all_cn_features": all_cn_features,
        "cell_sim_top10": cell_sim_top10
    }

def prepare_chunk_data(new_drug_df, old_drug_phys_df, old_drug_finger_df):
    phys_cols = old_drug_phys_df.columns
    for col in phys_cols:
        if col in new_drug_df.columns:
            new_drug_df[col] = pd.to_numeric(new_drug_df[col], errors='coerce').fillna(0)
    new_drug_phys_df = new_drug_df.reindex(columns=phys_cols, fill_value=0)
    new_drug_finger_df = new_drug_df.reindex(columns=old_drug_finger_df.columns, fill_value=0)
    chunk_phys_df = pd.concat([old_drug_phys_df, new_drug_phys_df])
    chunk_finger_df = pd.concat([old_drug_finger_df, new_drug_finger_df])
    
    # 【健壮性增强】在转换为numpy前，填充所有可能的NaN值
    chunk_phys_df.fillna(0, inplace=True)
    chunk_finger_df.fillna(0, inplace=True)

    phys_features = chunk_phys_df.values.astype(np.float32)
    drug_sim = torch.zeros(len(phys_features), len(phys_features))
    for i in range(len(phys_features)):
        for j in range(i, len(phys_features)):
            p_val = 0.0
            if np.std(phys_features[i]) > 0 and np.std(phys_features[j]) > 0:
                try:
                    p_val = pearsonr(phys_features[i], phys_features[j])[0]
                    if np.isnan(p_val): p_val = 0.0
                except (ValueError, RuntimeWarning):
                    p_val = 0.0
            drug_sim[i,j] = drug_sim[j,i] = abs(p_val)
    _, drug_sim_top10 = torch.topk(drug_sim, k=min(10, len(phys_features)), dim=1)
    return {
        "drug_names": chunk_finger_df.index.tolist(),
        "drug_features": torch.from_numpy(chunk_finger_df.values.astype(np.float32)),
        "drug_sim_top10": drug_sim_top10
    }

def build_chunk_edge_idx(chunk_drug_data, cell_data, device):
    drug_sim_top10_np = chunk_drug_data['drug_sim_top10'].numpy()
    cell_sim_top10_np = cell_data['cell_sim_top10'].cpu().numpy()
    N_drugs = len(chunk_drug_data['drug_names'])
    M_cells = len(cell_data['all_cell_names'])
    k_drug = drug_sim_top10_np.shape[1]
    k_cell = cell_sim_top10_np.shape[1]
    source_drug_idx = np.repeat(np.arange(N_drugs), k_drug)
    target_drug_idx = drug_sim_top10_np.flatten()
    cell_indices_for_drug_edges = np.tile(np.arange(M_cells), len(source_drug_idx))
    rows_drug_edges = np.repeat(source_drug_idx, M_cells) * M_cells + cell_indices_for_drug_edges
    cols_drug_edges = np.repeat(target_drug_idx, M_cells) * M_cells + cell_indices_for_drug_edges
    source_cell_idx = np.repeat(np.arange(M_cells), k_cell)
    target_cell_idx = cell_sim_top10_np.flatten()
    drug_indices_for_cell_edges = np.tile(np.arange(N_drugs) * M_cells, len(source_cell_idx))
    rows_cell_edges = np.repeat(source_cell_idx, N_drugs) + drug_indices_for_cell_edges
    cols_cell_edges = np.repeat(target_cell_idx, N_drugs) + drug_indices_for_cell_edges
    final_rows = np.concatenate([rows_drug_edges, rows_cell_edges])
    final_cols = np.concatenate([cols_drug_edges, cols_cell_edges])
    mask = final_rows != final_cols
    final_rows = final_rows[mask]
    final_cols = final_cols[mask]
    total_nodes = N_drugs * M_cells
    if len(final_rows) == 0:
        return torch.sparse_coo_tensor(torch.empty((2, 0), dtype=torch.long), torch.empty(0), (total_nodes, total_nodes), device=device)
    adj = coo_matrix((np.ones(len(final_rows)), (final_rows, final_cols)), shape=(total_nodes, total_nodes))
    adj_normalized = sym_adj(adj)
    indices = torch.from_numpy(np.vstack((adj_normalized.row, adj_normalized.col))).long().to(device)
    values = torch.from_numpy(adj_normalized.data).float().to(device)
    return torch.sparse_coo_tensor(indices, values, (total_nodes, total_nodes))


# ==============================================================================
# 3. 主工作流 (关键修改在这里)
# ==============================================================================
def run_prediction_for_single_chunk(args):
    big_chunk_num = args.chunk_num
    
    start_time = datetime.datetime.now()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"================== 开始处理大块 {big_chunk_num} on device {device} ==================")

    big_chunk_filename = f"drug_big_chunk_{big_chunk_num}.csv"
    big_chunk_file_path = os.path.join(args.temp_chunk_dir, big_chunk_filename)

    if not os.path.exists(big_chunk_file_path):
        print(f"错误: 找不到大块文件 {big_chunk_file_path}。")
        exit(1)

    min_ic50, max_ic50 = get_ic50_scaling_parameters(args)
    cell_data = preprocess_all_cells(args, device)
    
    model = GADRP_Net(device=device).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    try:
        print(f"  > 正在读取旧药背景文件...")
        old_drug_finger_df_full = pd.read_csv(args.all_drugs_feature_file, index_col=0)
        old_drug_phys_df_full = pd.read_csv(args.drug_physicochemical_file, index_col='pubchem_cid')
        old_drug_finger_df_full.index = old_drug_finger_df_full.index.astype(str)
        old_drug_phys_df_full.index = old_drug_phys_df_full.index.astype(str)
        all_old_drug_names = old_drug_finger_df_full.index.tolist()
    except Exception as e:
        print(f"读取旧药背景文件时发生致命错误: {e}")
        exit(1)

    try:
        big_chunk_df = pd.read_csv(big_chunk_file_path, index_col='ID')
    except Exception as e:
        print(f"读取临时分块文件 {big_chunk_filename} 失败: {e}")
        exit(1)

    new_drug_names_in_big_chunk = big_chunk_df.index.astype(str).tolist()
    if 'SMILES' in big_chunk_df.columns:
        big_chunk_df = big_chunk_df.drop(columns=['SMILES'])
    if 'Unnamed: 0' in big_chunk_df.columns:
        big_chunk_df = big_chunk_df.drop(columns=['Unnamed: 0'])
    big_chunk_df.index = big_chunk_df.index.astype(str)
    
    new_drug_names_set = set(new_drug_names_in_big_chunk)
    old_drug_names_unique = [name for name in all_old_drug_names if name not in new_drug_names_set]
    
    old_drug_phys_df = old_drug_phys_df_full.loc[old_drug_names_unique]
    old_drug_finger_df = old_drug_finger_df_full.loc[old_drug_names_unique]
    
    total_small_chunks = math.ceil(len(new_drug_names_in_big_chunk) / args.small_chunk_size)
    
    for i in range(total_small_chunks):
        small_chunk_num = i + 1
        
        base, ext = os.path.splitext(args.output_file)
        chunk_output_file = f"{base}_big_chunk_{big_chunk_num}_small_chunk_{small_chunk_num}{ext}"
        
        if os.path.exists(chunk_output_file) and os.path.getsize(chunk_output_file) > 1024:
            print(f"\n--- 跳过: 大块 {big_chunk_num} 的小块 {small_chunk_num}，因为输出文件 {os.path.basename(chunk_output_file)} 已存在且有效。")
            continue

        chunk_start_time = datetime.datetime.now()
        print(f"\n--- 处理大块 {big_chunk_num} 的小块 {small_chunk_num}/{total_small_chunks} ---")
        
        start_idx = i * args.small_chunk_size
        end_idx = (i + 1) * args.small_chunk_size
        small_chunk_drug_names = new_drug_names_in_big_chunk[start_idx:end_idx]
        
        if not small_chunk_drug_names:
            continue
            
        small_chunk_new_drug_df = big_chunk_df.loc[small_chunk_drug_names]
        
        # 【内存清洗步骤】
        feature_cols = small_chunk_new_drug_df.select_dtypes(include=np.number).columns
        original_count = len(small_chunk_new_drug_df)
        small_chunk_new_drug_df_cleaned = small_chunk_new_drug_df.dropna(subset=feature_cols, how='all')
        cleaned_count = len(small_chunk_new_drug_df_cleaned)
        
        if original_count > cleaned_count:
            print(f"  > 警告: 在小块 {small_chunk_num} 中，有 {original_count - cleaned_count} 行因特征全为NaN被丢弃。")

        if small_chunk_new_drug_df_cleaned.empty:
            print(f"  > 信息: 小块 {small_chunk_num} 在清洗后为空，将生成空结果文件并跳过预测。")
            empty_df = pd.DataFrame(columns=['DrugName', 'CellLineName', 'Predicted_Sensitivity_Scaled', 'Predicted_IC50'])
            empty_df.to_csv(chunk_output_file, index=False)
            continue
        
        print(f"  > a. 为 {len(small_chunk_new_drug_df_cleaned)} 个有效新药准备数据...")
        chunk_drug_data = prepare_chunk_data(small_chunk_new_drug_df_cleaned, old_drug_phys_df, old_drug_finger_df)
        
        print("  > b. 为当前小块构建GCN图 (向量化)...")
        chunk_edge_idx = build_chunk_edge_idx(chunk_drug_data, cell_data, device)

        print("  > c. 在小块内执行预测...")
        
        # 【设备匹配修复】
        drug_features_input = chunk_drug_data['drug_features'].to(device)
        cell_cn_input = cell_data['all_cn_features'].to(device)
        cell_exp_input = cell_data['all_exp_features'].to(device)
        
        num_old_drugs = len(old_drug_names_unique)
        num_old_cells = len(cell_data['train_cell_names'])
        
        local_new_drug_indices = range(num_old_drugs, len(chunk_drug_data['drug_names']))
        local_new_cell_indices = range(num_old_cells, len(cell_data['all_cell_names']))
        
        indices_to_predict = [[d_idx, c_idx] for d_idx in local_new_drug_indices for c_idx in local_new_cell_indices]
        
        if not indices_to_predict: 
            print("   > 无需预测的药物-细胞对，跳过。")
            continue
            
        indices_to_select_tensor = torch.tensor(indices_to_predict, device=device, dtype=torch.long)
        
        all_predictions = []
        pair_batch_size = 256
        
        with torch.no_grad():
            for j in tqdm(range(0, len(indices_to_select_tensor), pair_batch_size), desc="    > 预测中", leave=False):
                batch_indices = indices_to_select_tensor[j : j + pair_batch_size]
                
                try:
                    batch_preds = model(drug_features_input, cell_cn_input, cell_exp_input, chunk_edge_idx, batch_indices)
                except Exception as e:
                    print(f"\n  > 警告: 对一个批次的预测失败: {e}")
                    traceback.print_exc()
                    num_pairs_in_batch = len(batch_indices)
                    batch_preds = torch.full((num_pairs_in_batch, 1), float('nan'), device='cpu')
                
                all_predictions.append(batch_preds.cpu())
        
        predictions = torch.cat(all_predictions, dim=0)

        print("  > d. 保存结果...")
        predicted_drug_indices_local = indices_to_select_tensor[:, 0].cpu().numpy()
        predicted_cell_indices_global = indices_to_select_tensor[:, 1].cpu().numpy()
        result_df_chunk = pd.DataFrame({
            'DrugName': np.array(chunk_drug_data['drug_names'])[predicted_drug_indices_local],
            'CellLineName': np.array(cell_data['all_cell_names'])[predicted_cell_indices_global],
            'Predicted_Sensitivity_Scaled': predictions.numpy().flatten()
        })

        if min_ic50 is not None and max_ic50 is not None:
            scaled_preds = result_df_chunk['Predicted_Sensitivity_Scaled']
            result_df_chunk['Predicted_IC50'] = np.where(
                np.isnan(scaled_preds), 
                np.nan, 
                scaled_preds * (max_ic50 - min_ic50) + min_ic50
            )
        else:
            result_df_chunk['Predicted_IC50'] = np.nan
        
        result_df_chunk.to_csv(chunk_output_file, index=False)
        
        chunk_duration = datetime.datetime.now() - chunk_start_time
        print(f"--- ✓ 小块完成, 耗时: {chunk_duration}. 结果已保存到 {chunk_output_file} ---")

    total_duration = datetime.datetime.now() - start_time
    print(f"\n===== 大块 {big_chunk_num} 处理完成, 总耗时: {total_duration} =====")

# ==============================================================================
# 4. 主程序入口
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GADRP单一块预测脚本 (由外部bash调用，支持断点续跑)")
    
    parser.add_argument('--chunk_num', type=int, required=True, help='要处理的大块文件的编号 (例如: 5)。')
    
    parser.add_argument('--new_drug_feature_file', type=str, default='./test/drug_with_conditions.csv', help='包含所有新药特征的巨大CSV文件。')
    parser.add_argument('--new_exp_file', type=str, default='./test/exp_1.csv', help='新细胞系基因表达谱文件。')
    parser.add_argument('--new_cn_file', type=str, default='./test/cnv_1.csv', help='新细胞系拷贝数变异文件。')
    parser.add_argument('--new_meth_file', type=str, default='./test/mu_1.csv', help='新细胞系甲基化文件。')
    parser.add_argument('--new_mirna_file', type=str, default='./test/mi_1.csv', help='新细胞系microRNA表达谱文件。')
    parser.add_argument('--temp_chunk_dir', type=str, default='./drug_temp_chunks', help='用于存放自动生成的药物大块临时文件的目录。')
    parser.add_argument('--big_chunk_size', type=int, default=100000, help='从巨大药物文件中一次性生成/处理的行数（大块）。')
    parser.add_argument('--small_chunk_size', type=int, default=1000, help='在每个大块内部，用于GPU处理的小块的大小。')
    parser.add_argument('--model_path', type=str, default='./model/saved_models/best_model_drug_blind_fold2.pth')
    parser.add_argument('--output_file', type=str, default='./predictions/GADRP_predictions.csv', help='输出文件名的前缀和扩展名。')
    parser.add_argument('--ae_epochs', type=int, default=2500, help="动态训练AE的轮数。")
    data_group = parser.add_argument_group('原始背景数据文件 (用于背景建模)')
    data_group.add_argument('--cell_index_file', type=str, default='./mydata/cell_line/cell_index.csv')
    data_group.add_argument('--train_exp_file', type=str, default='./mydata/cell_line/exp_process.csv')
    data_group.add_argument('--train_cn_file', type=str, default='./mydata/cell_line/cn_process.csv')
    data_group.add_argument('--train_meth_file', type=str, default='./mydata/cell_line/meth_process.csv')
    data_group.add_argument('--train_mirna_file', type=str, default='./mydata/cell_line/mi_process.csv')
    data_group.add_argument('--drug_physicochemical_file', type=str, default='./mydata/drug/269dim.csv')
    data_group.add_argument('--all_drugs_feature_file', type=str, default='./mydata/drug/drug_with_conditions.csv')
    data_group.add_argument('--drug_response_file', type=str, default='./mydata/pair/drug_response.csv')
    
    args = parser.parse_args()
    
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 准备分割文件（如果需要的话，这步可以单独提前运行一次）
    prepare_drug_chunks(args)

    # 调用处理单个 chunk 的主函数
    run_prediction_for_single_chunk(args)