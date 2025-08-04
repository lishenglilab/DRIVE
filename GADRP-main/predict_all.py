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

# 导入tqdm，如果失败则定义一个替代函数
try:
    from tqdm import tqdm
except ImportError:
    print("错误: 'tqdm' 库未找到。")
    print("请在您的环境中运行 'pip install tqdm' 来安装它。")


    def tqdm(iterable, *args, **kwargs):
        return iterable

# 尝试导入mygene
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
                                                                                                        gcn_batch_cell_num).view(
            -1);
        list_cell_gcn_indices = torch.arange(gcn_batch_cell_num, device=self.device).repeat(gcn_batch_drug_num);

        drug_cell_pair_feature_for_gcn = torch.cat(
            (drug_feature_processed[list_drug_gcn_indices], cell_feature_processed[list_cell_gcn_indices]), dim=1)

        if drug_cell_pair_feature_for_gcn.shape[0] != edge_idx_input.shape[0]:
            raise ValueError(
                f"GCN input dimension mismatch: Features for {drug_cell_pair_feature_for_gcn.shape[0]} nodes, but adjacency matrix is for {edge_idx_input.shape[0]} nodes.")

        emb_out1, emb_out2, emb_out3, emb_out4, emb_out5 = self.embedding(drug_cell_pair_feature_for_gcn,
                                                                          edge_idx_input);

        selection_indices = (
                    drug_cell_indices_to_select[:, 0] * gcn_batch_cell_num + drug_cell_indices_to_select[:, 1]).long();

        feature1 = emb_out1[selection_indices];
        feature2 = emb_out2[selection_indices];
        feature3 = emb_out3[selection_indices];
        feature4 = emb_out4[selection_indices];
        feature5 = emb_out5[selection_indices];

        feature = self.att[0] * feature1 + self.att[1] * feature2 + self.att[2] * feature3 + self.att[3] * feature4 + \
                  self.att[4] * feature5;
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
# 2. 动态预处理模块 (无变化)
# ==============================================================================
def sym_adj(adj):
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj);
    adj = adj.tocoo();
    rowsum = np.array(adj.sum(1));
    d_inv_sqrt = np.power(rowsum, -0.5).flatten();
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.;
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt);
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).tocoo()


def get_ic50_scaling_parameters(drug_response_file, cell_index_file):
    print("  > 正在动态计算IC50缩放参数...")
    try:
        cell_index_list = pd.read_csv(cell_index_file, sep=',', header=None, index_col=0).index.tolist()
        drug_cell_label = pd.read_csv(drug_response_file, sep=',', header=0,
                                      usecols=["ccle_name", "pubchem_cid", "ic50"])
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
        drug_cell_label_filtered = drug_cell_label_sorted[
            (drug_cell_label_sorted["ic50"] >= LOW) & (drug_cell_label_sorted["ic50"] <= HIGH)]
        ic50_values_for_scaling = drug_cell_label_filtered["ic50"].values.astype(float)
        min_val = np.min(ic50_values_for_scaling)
        max_val = np.max(ic50_values_for_scaling)
        print(f"    > 计算完成: min_ic50 = {min_val}, max_ic50 = {max_val}")
        return min_val, max_val
    except Exception as e:
        print(f"    错误: 动态计算IC50缩放参数失败: {e}")
        return None, None


class DynamicPreprocessor:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.data_cache = {}
        self.gene_symbol_to_id_map = {}

    def run(self):
        print("--- 动态预处理开始 ---")
        self.data_cache['min_ic50'], self.data_cache['max_ic50'] = get_ic50_scaling_parameters(
            self.args.drug_response_file, self.args.cell_index_file)

        if self.args.new_exp_file:
            self._convert_gene_symbols_to_ids()

        self._preprocess_drugs()
        self._preprocess_cells()
        self._build_edge_idx()
        print("--- 动态预处理完成 ---\n")
        return self.data_cache

    def _convert_gene_symbols_to_ids(self):
        print("  > 从新细胞系文件收集基因名并查询ID...")
        gene_name_files = [self.args.new_exp_file, self.args.new_cn_file]
        all_gene_symbols = set()
        for f in gene_name_files:
            if not f or not os.path.exists(f): continue
            try:
                df = pd.read_csv(f, index_col=0)
                all_gene_symbols.update(df.columns.tolist())
            except Exception as e:
                print(f"    警告: 无法读取文件 {f} 的列名: {e}")

        if not all_gene_symbols:
            print("    > 未找到任何新基因名，跳过查询。")
            return

        print(f"    > 共找到 {len(all_gene_symbols)} 个唯一的基因名，开始分批查询...")
        mg = mygene.MyGeneInfo()
        gene_list = list(all_gene_symbols)
        batch_size = 1000

        for i in range(0, len(gene_list), batch_size):
            batch = gene_list[i:i + batch_size]
            print(f"      > 正在查询批次 {i // batch_size + 1}/{(len(gene_list) + batch_size - 1) // batch_size}...",
                  end=" ")
            try:
                query_result = mg.querymany(batch, scopes='symbol', fields='entrezgene', species='human', verbose=False)
                for query in query_result:
                    if 'entrezgene' in query and 'query' in query:
                        self.gene_symbol_to_id_map[query['query']] = str(query['entrezgene'])
                print("成功。")
            except Exception as e:
                print(f"失败: {e}. 跳过此批次。")
                continue

        print(f"    > 查询完成。成功映射 {len(self.gene_symbol_to_id_map)} 个基因名到Gene ID。")

    def _rename_new_sample_cols(self, df):
        rename_map = {symbol: gene_id for symbol, gene_id in self.gene_symbol_to_id_map.items() if symbol in df.columns}
        df_renamed = df.rename(columns=rename_map)
        valid_cols = [col for col in df_renamed.columns if col in self.gene_symbol_to_id_map.values()]
        df_filtered = df_renamed[valid_cols]
        return df_filtered.T.groupby(level=0).mean().T

    def _preprocess_drugs(self):
        print("  > 正在处理药物数据...")
        known_drug_physchem_df = pd.read_csv(self.args.drug_physicochemical_file, index_col=0)
        known_drug_fingerprint_df = pd.read_csv(self.args.all_drugs_feature_file, index_col=0)

        all_physchem_df_list = [known_drug_physchem_df]
        all_fingerprint_df_list = [known_drug_fingerprint_df]

        self.data_cache['new_drug_names'] = []
        if self.args.new_drug_feature_file and os.path.exists(self.args.new_drug_feature_file):
            print("    > 发现新药文件，正在加载...")
            new_drug_df = pd.read_csv(self.args.new_drug_feature_file, index_col=0)
            self.data_cache['new_drug_names'] = new_drug_df.index.tolist()
            new_physchem_df = new_drug_df.reindex(columns=known_drug_physchem_df.columns, fill_value=0)
            new_fingerprint_df = new_drug_df.reindex(columns=known_drug_fingerprint_df.columns, fill_value=0)
            all_physchem_df_list.append(new_physchem_df)
            all_fingerprint_df_list.append(new_fingerprint_df)
            print(f"    > 成功加载 {len(self.data_cache['new_drug_names'])} 种新药。")

        all_physchem_df = pd.concat(all_physchem_df_list)
        all_fingerprint_df = pd.concat(all_fingerprint_df_list)

        drug_sim_features = all_physchem_df.values
        drug_sim = torch.zeros(len(drug_sim_features), len(drug_sim_features), device=self.device)
        for i in range(len(drug_sim_features)):
            for j in range(len(drug_sim_features)):
                if np.std(drug_sim_features[i]) == 0 or np.std(drug_sim_features[j]) == 0:
                    drug_sim[i, j] = 1.0 if np.array_equal(drug_sim_features[i], drug_sim_features[j]) else 0.0
                else:
                    drug_sim[i, j] = pearsonr(drug_sim_features[i], drug_sim_features[j])[0]
        _, drug_sim_top10 = torch.topk(drug_sim, k=10, dim=1)

        self.data_cache['all_drug_names'] = all_fingerprint_df.index.tolist()
        self.data_cache['known_drug_names'] = known_drug_fingerprint_df.index.tolist()
        self.data_cache['drug_sim'] = drug_sim
        self.data_cache['drug_sim_top10'] = drug_sim_top10
        self.data_cache['all_drug_features_fingerprints'] = torch.from_numpy(
            all_fingerprint_df.values.astype(np.float32)).to(self.device)
        print("    > 药物数据处理完毕。")

    def _preprocess_cells(self):
        print("  > 正在处理细胞系数据...")
        train_cell_names = pd.read_csv(self.args.cell_index_file, header=None, index_col=0).index.tolist()
        self.data_cache['train_cell_names'] = train_cell_names

        print("    > 正在处理训练集数据...")
        train_exp_df = pd.read_csv(self.args.train_exp_file, sep=',', header=None, index_col=0, skiprows=1);
        train_exp_df.columns = train_exp_df.columns.astype(str)
        train_cn_df = pd.read_csv(self.args.train_cn_file, sep=',', header=None, index_col=0, skiprows=1);
        train_cn_df.columns = train_cn_df.columns.astype(str)
        exp_scaler = MinMaxScaler().fit(train_exp_df.values);
        train_exp_raw_scaled = exp_scaler.transform(train_exp_df.values)
        cn_scaler = MinMaxScaler().fit(train_cn_df.values);
        train_cn_raw_scaled = cn_scaler.transform(train_cn_df.values)
        train_meth_raw, train_meth_dim, meth_scaler = self._load_and_scale_raw(self.args.train_meth_file)
        train_mirna_raw, train_mirna_dim, mirna_scaler = self._load_and_scale_raw(self.args.train_mirna_file)
        self.data_cache['train_meth_scaled'] = train_meth_raw;
        self.data_cache['train_mirna_scaled'] = train_mirna_raw

        print("    > 动态训练AE并降维训练集数据...")
        exp_ae = self._dynamic_train_ae(torch.from_numpy(train_exp_raw_scaled).float(), 'exp')
        cn_ae = self._dynamic_train_ae(torch.from_numpy(train_cn_raw_scaled).float(), 'cn')
        with torch.no_grad():
            self.data_cache['old_exp_ae'] = exp_ae.output(
                torch.from_numpy(train_exp_raw_scaled).float().to(self.device))
            self.data_cache['old_cn_ae'] = cn_ae.output(torch.from_numpy(train_cn_raw_scaled).float().to(self.device))

        print("    > 计算旧细胞系间相似度...")
        cell_sim = torch.zeros(len(train_cell_names), len(train_cell_names), device=self.device)
        for i in range(len(train_cell_names)):
            for j in range(len(train_cell_names)):
                meth_p = pearsonr(train_meth_raw[i], train_meth_raw[j])[0] if np.std(train_meth_raw[i]) > 0 and np.std(
                    train_meth_raw[j]) > 0 else 1.0;
                mirna_p = pearsonr(train_mirna_raw[i], train_mirna_raw[j])[0] if np.std(
                    train_mirna_raw[i]) > 0 and np.std(train_mirna_raw[j]) > 0 else 1.0;
                cell_sim[i, j] = (abs(meth_p) + abs(mirna_p)) / 2
        _, cell_sim_top10 = torch.topk(cell_sim, k=10, dim=1)
        self.data_cache['cell_sim'] = cell_sim;
        self.data_cache['cell_sim_top10'] = cell_sim_top10

        self.data_cache['new_cell_names'] = []
        if self.args.new_exp_file and os.path.exists(self.args.new_exp_file):
            print("    > 发现新细胞系文件，正在对齐并处理...")
            new_exp_df_raw = pd.read_csv(self.args.new_exp_file, index_col=0)
            final_new_cell_list = new_exp_df_raw.index.tolist()

            new_cn_df_raw = pd.read_csv(self.args.new_cn_file, index_col=0).reindex(final_new_cell_list).fillna(0)
            new_meth_df_raw = pd.read_csv(self.args.new_meth_file, index_col=0).reindex(final_new_cell_list).fillna(0)
            new_mirna_df_raw = pd.read_csv(self.args.new_mirna_file, index_col=0).reindex(final_new_cell_list).fillna(0)

            self.data_cache['new_cell_names'] = final_new_cell_list

            new_exp_df_renamed = self._rename_new_sample_cols(new_exp_df_raw)
            new_cn_df_renamed = self._rename_new_sample_cols(new_cn_df_raw)

            new_exp_aligned = new_exp_df_renamed.reindex(columns=train_exp_df.columns, fill_value=0)
            new_cn_aligned = new_cn_df_renamed.reindex(columns=train_cn_df.columns, fill_value=0)

            new_exp_scaled = exp_scaler.transform(new_exp_aligned.values)
            new_cn_scaled = cn_scaler.transform(new_cn_aligned.values)

            new_meth_scaled = self._align_and_transform_df(new_meth_df_raw, self.args.train_meth_file, meth_scaler)
            new_mirna_scaled = self._align_and_transform_df(new_mirna_df_raw, self.args.train_mirna_file, mirna_scaler)

            self.data_cache['new_meth_scaled'] = new_meth_scaled;
            self.data_cache['new_mirna_scaled'] = new_mirna_scaled

            with torch.no_grad():
                self.data_cache['new_exp_ae'] = exp_ae.output(torch.from_numpy(new_exp_scaled).float().to(self.device))
                self.data_cache['new_cn_ae'] = cn_ae.output(torch.from_numpy(new_cn_scaled).float().to(self.device))

            print(f"    > 成功处理 {len(self.data_cache['new_cell_names'])} 个新细胞系样本。所有组学数据已对齐。")
        else:
            self.data_cache['new_exp_ae'] = torch.empty(0, 400, device=self.device)
            self.data_cache['new_cn_ae'] = torch.empty(0, 400, device=self.device)
            self.data_cache['new_meth_scaled'] = np.empty((0, train_meth_dim))
            self.data_cache['new_mirna_scaled'] = np.empty((0, train_mirna_dim))

        print("    > 细胞系数据处理完毕。")

    def _build_edge_idx(self):
        print("  > 正在动态构建全局 GCN edge_idx...")
        cache = self.data_cache;
        drug_sim, cell_sim = cache['drug_sim'], cache['cell_sim']
        drug_sim_top10 = cache['drug_sim_top10']

        N_drugs = drug_sim.shape[0]
        M_old_cells = cell_sim.shape[0]
        M_new_cells = cache['new_exp_ae'].shape[0]
        M_total_cells = M_old_cells + M_new_cells

        list_drug_old_cell_old = torch.arange(N_drugs, device=self.device).view(-1, 1).repeat(1, M_old_cells).view(-1)
        list_cell_old_drug_old = torch.arange(M_old_cells, device=self.device).repeat(N_drugs)
        cand_drug_old = drug_sim_top10[list_drug_old_cell_old].unsqueeze(2).repeat(1, 1, 10).view(-1, 100)
        cand_cell_old = cache['cell_sim_top10'][list_cell_old_drug_old].unsqueeze(1).repeat(1, 10, 1).view(-1, 100)
        sim_drug_part_old = drug_sim[list_drug_old_cell_old.unsqueeze(1), cand_drug_old]
        sim_cell_part_old = cell_sim[list_cell_old_drug_old.unsqueeze(1), cand_cell_old]
        sim_old = (sim_drug_part_old + sim_cell_part_old) / 2
        vals_old, top_indices_old = torch.topk(sim_old, k=10, dim=1)

        neighbor_indices_old = (
                    torch.gather(cand_drug_old, 1, top_indices_old) * M_total_cells + torch.gather(cand_cell_old, 1,
                                                                                                   top_indices_old)).view(
            -1)
        row_old = (list_drug_old_cell_old * M_total_cells + list_cell_old_drug_old).view(-1, 1).repeat(1, 10).view(-1)

        all_rows = [row_old.cpu()]
        all_cols = [neighbor_indices_old.cpu()]
        all_vals = [vals_old.cpu().view(-1)]

        if M_new_cells > 0:
            print("    > 正在计算新旧细胞系之间的连接...")
            new_meth, new_mirna = cache['new_meth_scaled'], cache['new_mirna_scaled']
            train_meth, train_mirna = cache['train_meth_scaled'], cache['train_mirna_scaled']

            new_old_cell_sim = torch.zeros(M_new_cells, M_old_cells, device=self.device)
            for i in range(M_new_cells):
                for j in range(M_old_cells):
                    meth_p = pearsonr(new_meth[i], train_meth[j])[0] if np.std(new_meth[i]) > 0 and np.std(
                        train_meth[j]) > 0 else 1.0;
                    mirna_p = pearsonr(new_mirna[i], train_mirna[j])[0] if np.std(new_mirna[i]) > 0 and np.std(
                        train_mirna[j]) > 0 else 1.0;
                    new_old_cell_sim[i, j] = (abs(meth_p) + abs(mirna_p)) / 2

            _, new_cell_sim_top10_old = torch.topk(new_old_cell_sim, k=10, dim=1)

            list_drug_new_cell = torch.arange(N_drugs, device=self.device).view(-1, 1).repeat(1, M_new_cells).view(-1)
            list_cell_new_drug = torch.arange(M_new_cells, device=self.device).repeat(N_drugs)

            cand_drug_new = drug_sim_top10[list_drug_new_cell].unsqueeze(2).repeat(1, 1, 10).view(-1, 100)
            cand_cell_new_old = new_cell_sim_top10_old[list_cell_new_drug].unsqueeze(1).repeat(1, 10, 1).view(-1, 100)

            sim_drug_part_new = drug_sim[list_drug_new_cell.unsqueeze(1), cand_drug_new]
            sim_cell_part_new = new_old_cell_sim[list_cell_new_drug.unsqueeze(1), cand_cell_new_old]
            sim_new_vs_old = (sim_drug_part_new + sim_cell_part_new) / 2

            vals_new, top_indices_new = torch.topk(sim_new_vs_old, k=10, dim=1)
            neighbor_indices_new_old = (torch.gather(cand_drug_new, 1, top_indices_new) * M_total_cells + torch.gather(
                cand_cell_new_old, 1, top_indices_new)).view(-1)
            row_new = (list_drug_new_cell * M_total_cells + (list_cell_new_drug + M_old_cells)).view(-1, 1).repeat(1,
                                                                                                                   10).view(
                -1)

            all_rows.append(row_new.cpu())
            all_cols.append(neighbor_indices_new_old.cpu())
            all_vals.append(vals_new.cpu().view(-1))

        total_nodes = N_drugs * M_total_cells
        rows_tensor = torch.cat(all_rows)
        cols_tensor = torch.cat(all_cols)
        vals_tensor = torch.cat(all_vals)

        adj = coo_matrix((vals_tensor.numpy(), (rows_tensor.numpy(), cols_tensor.numpy())),
                         shape=(total_nodes, total_nodes))
        adj_normalized = sym_adj(adj)
        indices = torch.from_numpy(np.vstack((adj_normalized.row, adj_normalized.col))).long().to(self.device)
        values = torch.from_numpy(adj_normalized.data).float().to(self.device)
        self.data_cache['final_edge_idx'] = torch.sparse_coo_tensor(indices, values, (total_nodes, total_nodes),
                                                                    requires_grad=False)
        print(f"    > 全局 GCN edge_idx 构建完毕, 共 {total_nodes} 个节点。")

    def _load_and_scale_raw(self, data_file):
        df = pd.read_csv(data_file, index_col=0, header=None, skiprows=1)
        scaler = MinMaxScaler().fit(df.values)
        return scaler.transform(df.values), df.shape[1], scaler

    def _align_and_transform_df(self, target_df, reference_file, scaler):
        ref_df_data = pd.read_csv(reference_file, index_col=0, header=None, skiprows=1)
        ref_header = pd.read_csv(reference_file, nrows=1, header=None).iloc[0, 1:].astype(str).tolist()
        ref_df_data.columns = ref_header
        aligned_df = target_df.reindex(columns=ref_df_data.columns, fill_value=0)
        return scaler.transform(aligned_df.values)

    def _align_and_transform(self, target_file, reference_file, scaler):
        try:
            target_df = pd.read_csv(target_file, index_col=0)
            return self._align_and_transform_df(target_df, reference_file, scaler)
        except FileNotFoundError:
            raise FileNotFoundError(f"新样本文件 {target_file} 未找到，无法继续。")

    def _dynamic_train_ae(self, data, name):
        random.seed(4);
        torch.manual_seed(4)
        in_dim = data.shape[1];
        model = Auto_Encoder(self.device, in_dim, 400).to(self.device);
        print(f"      > 开始动态训练 {name.upper()} AE (输入维度: {in_dim})...");

        data_loader = Data.DataLoader(data, batch_size=min(428, len(data)), shuffle=True);
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001);
        loss_func = nn.MSELoss();
        best_model_state = model.state_dict();
        best_loss = float('inf')

        epoch_iterator = tqdm(range(1, self.args.ae_epochs + 1), desc=f"训练 {name.upper()} AE")
        for epoch in epoch_iterator:
            epoch_loss = 0.0
            model.train()
            for x in data_loader:
                x = x.to(self.device)
                _, decoded = model(x);
                train_loss = loss_func(decoded, x);
                optimizer.zero_grad();
                train_loss.backward();
                optimizer.step()
                epoch_loss += train_loss.item()

            avg_epoch_loss = epoch_loss / len(data_loader)
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                best_model_state = model.state_dict()

            epoch_iterator.set_postfix(loss=f"{avg_epoch_loss:.6f}", best_loss=f"{best_loss:.6f}")

        print(f"      > {name.upper()} AE 训练完成，最终最佳损失: {best_loss:.6f}");
        best_model = Auto_Encoder(self.device, in_dim, 400).to(self.device)
        best_model.load_state_dict(best_model_state)
        best_model.eval()
        return best_model


# ==============================================================================
# 3. 主预测函数 (【【【 最终精确修改版 】】】)
# ==============================================================================
def main(args):
    start_time = datetime.datetime.now()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");
    print(f"使用设备: {device}\n");

    # 检查是否同时提供了新药和新细胞系文件
    if not args.new_drug_feature_file or not args.new_exp_file:
        print("错误：必须同时提供新药文件(--new_drug_feature_file)和新细胞系文件(--new_exp_file等)才能进行预测。")
        return

    preprocessor = DynamicPreprocessor(args, device);
    data_cache = preprocessor.run()

    print("--- 准备最终预测输入 ---")
    all_drug_features = data_cache['all_drug_features_fingerprints'];
    all_exp_features = torch.cat([data_cache['old_exp_ae'], data_cache['new_exp_ae']], dim=0);
    all_cn_features = torch.cat([data_cache['old_cn_ae'], data_cache['new_cn_ae']], dim=0);
    print(f"    > 总药物数 (旧+新): {all_drug_features.shape[0]}")
    print(f"    > 总细胞系特征数 (旧+新): {all_exp_features.shape[0]}")

    print("\n--- 加载主模型并执行预测 ---")
    model = GADRP_Net(device=device).to(device);
    model.load_state_dict(torch.load(args.model_path, map_location=device));
    model.eval()

    all_drug_names = data_cache['all_drug_names']
    new_drug_names = data_cache['new_drug_names']

    all_cell_names = data_cache['train_cell_names'] + data_cache['new_cell_names']
    new_cell_names = data_cache['new_cell_names']

    # ================== 核心修改在这里 ==================
    # 只生成 "新药 vs. 新细胞系" 的预测任务
    indices_to_predict = []
    
    if new_drug_names and new_cell_names:
        print("  > 正在准备 [新药 vs. 新细胞] 的预测任务...")
        new_drug_indices = [all_drug_names.index(name) for name in new_drug_names]
        new_cell_indices = [all_cell_names.index(name) for name in new_cell_names]
        for d_idx in new_drug_indices:
            for c_idx in new_cell_indices:
                indices_to_predict.append([d_idx, c_idx])

    if not indices_to_predict:
        print("\n没有需要执行的 [新药 vs. 新细胞] 预测任务。请确保输入的新药和新细胞系文件非空。")
        return

    # 去重并转换为tensor
    # 【【【注意】】】这里的 all_indices_to_select 就是唯一需要模型计算的部分
    indices_df = pd.DataFrame(indices_to_predict, columns=['drug_idx', 'cell_idx']).drop_duplicates()
    all_indices_to_select = torch.tensor(indices_df.values, device=device, dtype=torch.long)
    print(f"  > 总共需要模型计算 {len(all_indices_to_select)} 个独特的 [新药-新细胞] 对。")

    # 为16GB显存设置优化的批大小
    batch_size = 65536
    all_predictions = []

    prediction_iterator = tqdm(range(0, len(all_indices_to_select), batch_size), desc="分批预测中")

    with torch.no_grad():
        for i in prediction_iterator:
            batch_indices = all_indices_to_select[i: i + batch_size]

            # 模型只会计算 batch_indices 中指定的配对
            batch_predictions = model(drug_feature_input=all_drug_features,
                                      cell_feature1_input=all_cn_features,
                                      cell_feature2_input=all_exp_features,
                                      edge_idx_input=data_cache['final_edge_idx'],
                                      drug_cell_indices_to_select=batch_indices)

            all_predictions.append(batch_predictions.cpu())

    predictions = torch.cat(all_predictions, dim=0)

    print("\n--- 保存预测结果 ---")
    scaled_predictions = predictions.numpy().flatten()

    # 【【【注意】】】这里的索引直接对应了我们请求预测的索引，所以结果也是精确的
    predicted_drug_indices = all_indices_to_select[:, 0].cpu().numpy()
    predicted_cell_indices = all_indices_to_select[:, 1].cpu().numpy()

    result_df = pd.DataFrame({
        'DrugName': np.array(all_drug_names)[predicted_drug_indices],
        'CellLineName': np.array(all_cell_names)[predicted_cell_indices],
        'Predicted_Sensitivity_Scaled': scaled_predictions
    })

    min_ic50, max_ic50 = data_cache.get('min_ic50'), data_cache.get('max_ic50')
    if min_ic50 is not None and max_ic50 is not None:
        result_df['Predicted_IC50'] = result_df['Predicted_Sensitivity_Scaled'] * (max_ic50 - min_ic50) + min_ic50
        print("    > 已将预测值反向缩放为IC50。")

    result_df.to_csv(args.output_file, index=False);
    end_time = datetime.datetime.now()
    print(f"\n预测完成！结果已保存到: {args.output_file}")
    print(f"总耗时: {end_time - start_time}")


# ==============================================================================
# 4. 主程序入口 (已修正笔误)
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GADRP全功能预测脚本 (动态AE训练, 支持新药/新细胞)。")

    # --- 输入: 新数据 (可选) ---
    parser.add_argument('--new_drug_feature_file', type=str, default='./depmap/drug_with_conditions_predict.csv',
                        help='新药特征文件路径 (可选)')
    parser.add_argument('--new_exp_file', type=str, default='./depmap/gene_depmap.csv',
                        help='新细胞系基因表达谱文件路径 (可选, 需与下面3个文件同时提供)')
    parser.add_argument('--new_cn_file', type=str, default='./depmap/cnv_depmap.csv',
                        help='新细胞系拷贝数变异文件路径 (可选)')
    parser.add_argument('--new_meth_file', type=str, default='./depmap/mu_depmap.csv',
                        help='新细胞系甲基化文件路径 (可选)')
    parser.add_argument('--new_mirna_file', type=str, default='./depmap/mi_depmap.csv',
                        help='新细胞系microRNA表达谱文件路径 (可选)')

    # --- 输入: 训练时用的所有原始数据 (必需) ---
    parser.add_argument('--cell_index_file', type=str, default='./mydata/cell_line/cell_index.csv')
    parser.add_argument('--train_exp_file', type=str, default='./mydata/cell_line/exp_process.csv')
    parser.add_argument('--train_cn_file', type=str, default='./mydata/cell_line/cn_process.csv')
    parser.add_argument('--train_meth_file', type=str, default='./mydata/cell_line/meth_process.csv')
    parser.add_argument('--train_mirna_file', type=str, default='./mydata/cell_line/mi_process.csv')
    parser.add_argument('--drug_physicochemical_file', type=str, default='./mydata/drug/269dim.csv')
    parser.add_argument('--all_drugs_feature_file', type=str, default='./mydata/drug/drug_with_conditions.csv')
    parser.add_argument('--drug_response_file', type=str, default='./mydata/pair/drug_response.csv')

    # --- 输入: 预训练主模型 (必需) ---
    parser.add_argument('--model_path', type=str, default='./model/saved_models/best_model_drug_blind_fold2.pth',
                        help='预训练的GADRP_Net主模型路径 (必需)')

    # --- 动态AE训练参数 ---
    parser.add_argument('--ae_epochs', type=int, default=2500, help="动态训练AE的轮数。")

    # --- 输出 ---
    parser.add_argument('--output_file', type=str, default='./predictions/GADRP_predictions.csv')

    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    main(args)