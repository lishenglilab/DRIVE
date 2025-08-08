import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_max_pool as gmp
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.preprocessing import MinMaxScaler
import os
import csv
import json
import itertools
from tqdm import tqdm
import argparse # 导入 argparse
import gc # 导入垃圾回收模块

# --- Autoencoder 定义 (保持不变) ---
class AutoEncoder(nn.Module):
    def __init__(self, input_dim=25272):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2048), nn.BatchNorm1d(2048), nn.ReLU(),
            nn.Linear(2048, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.BatchNorm1d(512),
        )
        self.decoder = nn.Sequential(
            nn.Linear(512, 1024), nn.BatchNorm1d(1024), nn.ReLU(),
            nn.Linear(1024, 2048), nn.BatchNorm1d(2048), nn.ReLU(),
            nn.Linear(2048, input_dim), nn.Sigmoid(),
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


# --- NeRD_Net 定义 (保持不变) ---
class NeRD_Net(torch.nn.Module):
    def __init__(self, n_filters=4, num_features_xd=78, output_dim=128, dropout=0.5):
        super(NeRD_Net, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.ds_conv1 = GCNConv(num_features_xd, num_features_xd)
        self.ds_conv2 = GCNConv(num_features_xd, num_features_xd * 2)
        self.ds_conv3 = GCNConv(num_features_xd * 2, num_features_xd * 4)
        self.ds_fc1 = torch.nn.Linear(num_features_xd * 4, 1024)
        self.ds_bn4 = nn.BatchNorm1d(1024)
        self.ds_fc2 = torch.nn.Linear(1024, output_dim)
        self.ds_bn5 = nn.BatchNorm1d(output_dim)
        self.df_conv1 = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=8)
        self.df_bn1 = nn.BatchNorm1d(n_filters)
        self.df_pool1 = nn.MaxPool1d(3)
        self.df_conv2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters * 2, kernel_size=8)
        self.df_bn2 = nn.BatchNorm1d(n_filters * 2)
        self.df_pool2 = nn.MaxPool1d(3)
        self.df_conv3 = nn.Conv1d(in_channels=n_filters * 2, out_channels=n_filters * 4, kernel_size=8)
        self.df_bn3 = nn.BatchNorm1d(n_filters * 4)
        self.df_pool3 = nn.MaxPool1d(3)
        self.df_fc1 = nn.Linear(464, 512)
        self.df_bn4 = nn.BatchNorm1d(512)
        self.df_fc2 = nn.Linear(512, output_dim)
        self.df_bn5 = nn.BatchNorm1d(output_dim)
        self.cm_conv1 = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=8)
        self.cm_bn1 = nn.BatchNorm1d(n_filters)
        self.cm_pool1 = nn.MaxPool1d(3)
        self.cm_conv2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters * 2, kernel_size=8)
        self.cm_bn2 = nn.BatchNorm1d(n_filters * 2)
        self.cm_pool2 = nn.MaxPool1d(3)
        self.cm_conv3 = nn.Conv1d(in_channels=n_filters * 2, out_channels=n_filters * 4, kernel_size=8)
        self.cm_bn3 = nn.BatchNorm1d(n_filters * 4)
        self.cm_pool3 = nn.MaxPool1d(3)
        self.cm_fc1 = nn.Linear(368, 512)
        self.cm_bn4 = nn.BatchNorm1d(512)
        self.cm_fc2 = nn.Linear(512, output_dim)
        self.cm_bn5 = nn.BatchNorm1d(output_dim)
        self.cc_fc1 = nn.Linear(512, 1024)
        self.cc_bn1 = nn.BatchNorm1d(1024)
        self.cc_fc2 = nn.Linear(1024, 256)
        self.cc_bn2 = nn.BatchNorm1d(256)
        self.cc_fc3 = nn.Linear(256, output_dim)
        self.cc_bn3 = nn.BatchNorm1d(output_dim)
        self.comb_fc1 = nn.Linear(4 * output_dim, 1024)
        self.comb_bn1 = nn.BatchNorm1d(1024)
        self.comb_fc2 = nn.Linear(1024, 128)
        self.comb_bn2 = nn.BatchNorm1d(128)
        self.comb_out = nn.Linear(128, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        miRNA = data.miRNA[:, None, :]
        copynumber = data.copynumber
        finger = data.finger[:, None, :]
        x = self.ds_conv1(x, edge_index);
        x = self.relu(x)
        x = self.ds_conv2(x, edge_index);
        x = self.relu(x)
        x = self.ds_conv3(x, edge_index);
        x = self.relu(x)
        x = gmp(x, batch)
        x = self.ds_fc1(x);
        x = self.ds_bn4(x);
        x = self.relu(x);
        x = self.dropout(x)
        x = self.ds_fc2(x);
        x = self.ds_bn5(x);
        x = self.dropout(x)
        xdf = self.df_conv1(finger);
        xdf = self.df_bn1(xdf);
        xdf = self.relu(xdf);
        xdf = self.df_pool1(xdf)
        xdf = self.df_conv2(xdf);
        xdf = self.df_bn2(xdf);
        xdf = self.relu(xdf);
        xdf = self.df_pool2(xdf)
        xdf = self.df_conv3(xdf);
        xdf = self.df_bn3(xdf);
        xdf = self.relu(xdf);
        xdf = self.df_pool3(xdf)
        xdf = xdf.view(-1, xdf.shape[1] * xdf.shape[2])
        xdf = self.df_fc1(xdf);
        xdf = self.df_bn4(xdf);
        xdf = self.relu(xdf);
        xdf = self.dropout(xdf)
        xdf = self.df_fc2(xdf);
        xdf = self.df_bn5(xdf)
        xcm = self.cm_conv1(miRNA);
        xcm = self.cm_bn1(xcm);
        xcm = self.relu(xcm);
        xcm = self.cm_pool1(xcm)
        xcm = self.cm_conv2(xcm);
        xcm = self.cm_bn2(xcm);
        xcm = self.relu(xcm);
        xcm = self.cm_pool2(xcm)
        xcm = self.cm_conv3(xcm);
        xcm = self.cm_bn3(xcm);
        xcm = self.relu(xcm);
        xcm = self.cm_pool3(xcm)
        xcm = xcm.view(-1, xcm.shape[1] * xcm.shape[2])
        xcm = self.cm_fc1(xcm);
        xcm = self.cm_bn4(xcm);
        xcm = self.cm_fc2(xcm);
        xcm = self.cm_bn5(xcm)
        xcc = self.cc_fc1(copynumber);
        xcc = self.cc_bn1(xcc);
        xcc = self.relu(xcc)
        xcc = self.cc_fc2(xcc);
        xcc = self.cc_bn2(xcc);
        xcc = self.relu(xcc)
        xcc = self.cc_fc3(xcc);
        xcc = self.cc_bn3(xcc)
        xfusion = torch.cat((x, xdf, xcm, xcc), 1)
        xfusion = self.comb_fc1(xfusion);
        xfusion = self.comb_bn1(xfusion);
        xfusion = self.relu(xfusion);
        xfusion = self.dropout(xfusion)
        xfusion = self.comb_fc2(xfusion);
        xfusion = self.comb_bn2(xfusion);
        xfusion = self.relu(xfusion);
        xfusion = self.dropout(xfusion)
        out = self.comb_out(xfusion)
        out = self.sigmoid(out)
        return out


# --- 辅助函数 (保持不变) ---
def smile_to_graph_custom(smile_string, num_node_features=78):
    mol = Chem.MolFromSmiles(smile_string)
    if mol is None:
        mol = Chem.MolFromSmiles(smile_string, sanitize=True)
        if mol is None:
            raise ValueError(f"Could not parse SMILES (even after sanitization): {smile_string}")
        else:
            print(f"  Note: SMILES '{smile_string}' required sanitization.")
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feat = np.zeros(num_node_features)
        try:
            atom_feat[atom.GetAtomicNum() % num_node_features] = 1
            atom_feat[atom.GetDegree() % num_node_features] = 1
            if atom.GetIsAromatic(): atom_feat[num_node_features - 1] = 1
        except Exception as e:
            print(
                f"    Warning: Error extracting basic features for an atom in {smile_string}. Using zeros. Error: {e}")
        atom_features_list.append(atom_feat)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.float)
    edge_list = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_list.append((i, j));
        edge_list.append((j, i))
    if not edge_list and mol.GetNumAtoms() > 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    elif not mol.GetNumAtoms():
        raise ValueError(f"SMILES '{smile_string}' resulted in a molecule with no atoms.")
    else:
        edge_index = torch.tensor(np.array(edge_list).T, dtype=torch.long)
    return x, edge_index

def load_drug_fingerprints_from_file(fingerprint_file_path, drug_name_col=0, expected_length=None):
    fingerprint_dict = {}
    print(f"Loading drug fingerprints from {fingerprint_file_path}...")
    try:
        df = pd.read_csv(fingerprint_file_path)
        for index, row in df.iterrows():
            drug_identifier = str(row.iloc[drug_name_col]).strip().lower()
            fp_values = pd.to_numeric(row.iloc[2:], errors='coerce').fillna(0).values.astype(np.float32)

            if expected_length is not None and len(fp_values) != expected_length:
                if len(fp_values) < expected_length:
                    fp_values = np.pad(fp_values, (0, expected_length - len(fp_values)), 'constant')
                else:
                    fp_values = fp_values[:expected_length]
            fingerprint_dict[drug_identifier] = fp_values
        print(f"  Loaded {len(fingerprint_dict)} drug fingerprints.")
    except FileNotFoundError:
        print(f"  ERROR: Fingerprint file '{fingerprint_file_path}' not found.")
    except Exception as e:
        print(f"  ERROR: Could not load fingerprints from '{fingerprint_file_path}': {e}")
    return fingerprint_dict

def read_drugs_for_prediction_from_file(file_path):
    drugs_to_predict = []
    try:
        with open(file_path, 'r', newline='', encoding='utf-8-sig') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader, None)
            if header and 'name' in header[0].lower() and 'smiles' in header[1].lower():
                print("  Header detected and skipped in drug input file.")
            else:
                if header and len(header) >= 2:
                    drug_name = header[0].strip()
                    smiles = header[1].strip()
                    if drug_name and smiles:
                        drugs_to_predict.append({"name": drug_name, "smiles": smiles})
            for row in reader:
                if len(row) >= 2:
                    drug_name = row[0].strip()
                    smiles = row[1].strip()
                    if drug_name and smiles:
                        drugs_to_predict.append({"name": drug_name, "smiles": smiles})
    except FileNotFoundError:
        print(f"ERROR: Drug prediction input file not found at {file_path}");
        return []
    except Exception as e:
        print(f"ERROR reading drug prediction input file {file_path}: {e}");
        return []
    return drugs_to_predict

def get_cnv_autoencoder(train_cnv_path, device, ae_epochs=50, batch_size=64):
    print("正在加载并训练CNV降维模型 (Autoencoder)...")
    try:
        train_cnv_df = pd.read_csv(train_cnv_path, index_col=0)
        train_cnv_features = train_cnv_df.values.astype(np.float32)
        original_dim = train_cnv_features.shape[1]
    except FileNotFoundError:
        print(f"错误: 训练用CNV数据文件 '{train_cnv_path}' 未找到。");
        return None, None, None
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_cnv_scaled = scaler.fit_transform(train_cnv_features)
    train_tensor = torch.tensor(train_cnv_scaled, dtype=torch.float).to(device)
    autoencoder = AutoEncoder(input_dim=original_dim).to(device)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-4)
    loss_func = nn.MSELoss()
    print(f"  开始训练AE ({ae_epochs}个轮次)...")
    progress_bar = tqdm(range(ae_epochs), desc="    AE训练中")
    for epoch in progress_bar:
        autoencoder.train()
        _, decoded = autoencoder(train_tensor)
        loss = loss_func(decoded, train_tensor)
        optimizer.zero_grad();
        loss.backward();
        optimizer.step()
        progress_bar.set_postfix(loss=f"{loss.item():.6f}")
    print("  AE训练完成。")
    autoencoder.eval()
    return autoencoder, scaler, train_cnv_df.columns

def predict_response(model, drug_graph_x, drug_graph_edge_index, drug_fingerprint_tensor,
                     cell_miRNA_tensor, cell_copynumber_tensor, device):
    model.eval()
    data = Data(x=drug_graph_x, edge_index=drug_graph_edge_index, finger=drug_fingerprint_tensor.unsqueeze(0),
                miRNA=cell_miRNA_tensor, copynumber=cell_copynumber_tensor)
    data.batch = torch.zeros(drug_graph_x.size(0), dtype=torch.long)
    data = data.to(device)
    with torch.no_grad():
        normalized_prediction = model(data).cpu().item()
    return normalized_prediction

g_min_ic50, g_max_ic50 = None, None

def load_ic50_scaling_params(params_file_path):
    global g_min_ic50, g_max_ic50
    try:
        with open(params_file_path, 'r') as f:
            params = json.load(f)
        g_min_ic50, g_max_ic50 = params.get('min_ic50'), params.get('max_ic50')
        if g_min_ic50 is None or g_max_ic50 is None:
            print(f"  警告: 'min_ic50' 或 'max_ic50' 未在文件 '{params_file_path}' 中找到。");
            return False
        print(f"  IC50反归一化参数已加载: min={g_min_ic50}, max={g_max_ic50}");
        return True
    except FileNotFoundError:
        print(f"  错误: IC50缩放参数文件 '{params_file_path}' 未找到。"); return False
    except Exception as e:
        print(f"  错误: 加载IC50缩放参数失败: {e}"); return False

def unscale_ic50_value(scaled_value):
    if g_min_ic50 is None or g_max_ic50 is None or pd.isna(scaled_value): return np.nan
    ic50_range = g_max_ic50 - g_min_ic50
    return scaled_value * ic50_range + g_min_ic50 if ic50_range != 0 else g_min_ic50

def prepare_new_cell_line_data(new_mirna_path, new_cnv_path, train_mirna_path, train_cnv_path_raw, device):
    print("正在准备新的细胞系数据...")
    autoencoder, cnv_scaler, train_cnv_cols = get_cnv_autoencoder(train_cnv_path_raw, device)
    if autoencoder is None: return {}
    try:
        train_mirna_df = pd.read_csv(train_mirna_path)
        train_mirna_cols = train_mirna_df.columns[1:]
        train_mirna_features = train_mirna_df.iloc[:, 1:].values.astype(np.float32)
        miRNA_scaler = MinMaxScaler(feature_range=(0, 1)).fit(train_mirna_features)
    except FileNotFoundError as e:
        print(f"错误: 训练miRNA文件未找到: {e}。");
        return {}
    try:
        new_mirna_df = pd.read_csv(new_mirna_path, index_col=0)
        new_cnv_df = pd.read_csv(new_cnv_path, index_col=0)
    except FileNotFoundError as e:
        print(f"错误: 新细胞系样本文件未找到: {e}。");
        return {}
    
    new_cell_line_names = new_mirna_df.index.tolist()
    if set(new_cell_line_names) != set(new_cnv_df.index.tolist()):
        print("警告: miRNA和CNV文件的细胞系列表不匹配。将使用二者的交集。")
        shared_cells = list(set(new_cell_line_names) & set(new_cnv_df.index.tolist()))
        new_mirna_df, new_cnv_df = new_mirna_df.loc[shared_cells], new_cnv_df.loc[shared_cells]
        new_cell_line_names = shared_cells
    
    print("  正在对齐并归一化miRNA数据...")
    new_mirna_aligned_df = new_mirna_df.reindex(columns=train_mirna_cols, fill_value=0.0)
    new_mirna_normalized = miRNA_scaler.transform(new_mirna_aligned_df.values.astype(np.float32))
    
    print("  正在对齐、归一化并使用AE降维CNV数据...")
    new_cnv_aligned_df = new_cnv_df.reindex(columns=train_cnv_cols, fill_value=0.0)
    new_cnv_scaled = cnv_scaler.transform(new_cnv_aligned_df.values.astype(np.float32))
    new_cnv_tensor = torch.tensor(new_cnv_scaled, dtype=torch.float).to(device)
    with torch.no_grad():
        new_cnv_latent, _ = autoencoder(new_cnv_tensor)
    
    final_cnv_scaler = MinMaxScaler(feature_range=(0, 1))
    with torch.no_grad():
        train_cnv_raw_df = pd.read_csv(train_cnv_path_raw, index_col=0)
        train_cnv_raw_scaled = cnv_scaler.transform(train_cnv_raw_df.values.astype(np.float32))
        train_cnv_raw_tensor = torch.tensor(train_cnv_raw_scaled, dtype=torch.float).to(device)
        train_cnv_latent, _ = autoencoder(train_cnv_raw_tensor)
        final_cnv_scaler.fit(train_cnv_latent.cpu().numpy())
    
    new_cnv_normalized = final_cnv_scaler.transform(new_cnv_latent.cpu().numpy())
    
    prepared_data = {}
    for i, cell_name in enumerate(new_cell_line_names):
        prepared_data[cell_name] = {
            'miRNA': torch.tensor(new_mirna_normalized[i:i + 1], dtype=torch.float),
            'copynumber': torch.tensor(new_cnv_normalized[i:i + 1], dtype=torch.float)
        }
    
    print(f"  成功准备了 {len(prepared_data)} 个新细胞系的数据。")
    return prepared_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeRD 药物响应预测脚本")
    parser.add_argument('--model_path', type=str, required=True, help='预训练模型文件路径')
    parser.add_argument('--output_file', type=str, required=True, help='输出预测结果的CSV文件基础路径 (例如: results.csv)')
    parser.add_argument('--drugs_input_file', type=str, required=True, help='待预测的药物SMILES文件路径')
    parser.add_argument('--new_mirna_file', type=str, required=True, help='新细胞系的miRNA表达数据文件路径')
    parser.add_argument('--new_cnv_file', type=str, required=True, help='新细胞系的CNV数据文件路径')
    parser.add_argument('--precomputed_fingerprint_file', type=str, required=True, help='预计算的药物指纹文件路径')
    parser.add_argument('--train_mirna_file', type=str, required=True, help='用于对齐的训练集miRNA文件路径')
    parser.add_argument('--train_cnv_raw_file', type=str, required=True, help='用于AE训练和对齐的原始CNV文件路径')
    parser.add_argument('--ic50_scaling_params_file', type=str, required=True, help='IC50归一化参数的JSON文件路径')
    args = parser.parse_args()

    N_FILTERS, NUM_FEATURES_XD, OUTPUT_DIM, DRUG_FINGERPRINT_LENGTH = 4, 78, 128, 881
    
    # 【【【 MODIFICATION: 定义批处理大小 】】】
    CHUNK_SIZE = 50000

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {DEVICE}")

    print(f"\n正在从 {args.model_path} 加载主预测模型...")
    model = NeRD_Net(n_filters=N_FILTERS, num_features_xd=NUM_FEATURES_XD, output_dim=OUTPUT_DIM, dropout=0.0)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE));
        model.to(DEVICE);
        model.eval()
        print("  主模型加载成功。")
    except Exception as e:
        print(f"错误: 加载主模型失败: {e}");
        exit()
    
    print(f"\n正在从 {args.ic50_scaling_params_file} 加载IC50归一化参数...")
    scaling_params_loaded_successfully = load_ic50_scaling_params(args.ic50_scaling_params_file)

    new_cell_line_data = prepare_new_cell_line_data(args.new_mirna_file, args.new_cnv_file, args.train_mirna_file, args.train_cnv_raw_file, DEVICE)
    if not new_cell_line_data:
        print("错误: 未能加载任何新的细胞系数据，无法进行预测。");
        exit()
    
    print(f"\n正在从 {args.drugs_input_file} 读取待预测的药物...")
    drugs_for_prediction = read_drugs_for_prediction_from_file(args.drugs_input_file)
    if not drugs_for_prediction:
        print("错误: 未能加载任何待预测的药物，无法进行预测。");
        exit()
    print(f"  找到 {len(drugs_for_prediction)} 种药物用于预测。")
    drug_fingerprints_store = load_drug_fingerprints_from_file(args.precomputed_fingerprint_file, drug_name_col=0,
                                                               expected_length=DRUG_FINGERPRINT_LENGTH)

    print("\n正在准备所有药物数据 (生成图结构和查找指纹)...")
    prepared_drugs_list = []
    for drug_info in tqdm(drugs_for_prediction, desc="  准备药物中"):
        drug_name, drug_smiles = drug_info["name"], drug_info["smiles"]
        prepared_drug_data = {'name': drug_name, 'smiles': drug_smiles, 'status': 'Pending',
                              'graph_x': None, 'graph_edge_index': None, 'fingerprint': None}
        clean_drug_name = drug_name.strip().lower()
        fp_numpy_arr = drug_fingerprints_store.get(clean_drug_name)
        if fp_numpy_arr is None:
            tqdm.write(
                f"  信息: 在指纹库中未找到药物 '{drug_name}' (查找时使用: '{clean_drug_name}')。该药物的所有预测将被跳过。")
            prepared_drug_data['status'] = 'Error_FingerprintMissing'
            prepared_drugs_list.append(prepared_drug_data)
            continue
        try:
            prepared_drug_data['graph_x'], prepared_drug_data['graph_edge_index'] = smile_to_graph_custom(drug_smiles,
                                                                                                          NUM_FEATURES_XD)
            prepared_drug_data['fingerprint'] = torch.tensor(fp_numpy_arr, dtype=torch.float)
            prepared_drug_data['status'] = 'Success'
        except ValueError as e:
            tqdm.write(f"  错误: 无法为药物 '{drug_name}' 生成图结构。该药物的所有预测将被跳过。错误: {e}")
            prepared_drug_data['status'] = f'Error_GraphGen: {e}'

        prepared_drugs_list.append(prepared_drug_data)

    print(f"\n准备完成。{sum(1 for d in prepared_drugs_list if d['status'] == 'Success')} / {len(prepared_drugs_list)} 种药物可以用于预测。")
    
    total_predictions = len(prepared_drugs_list) * len(new_cell_line_data)
    print(f"开始预测药物与 {len(new_cell_line_data)} 个细胞系的药物反应... 总计 {total_predictions} 个预测任务。")

    # 【【【 MODIFICATION: 分块处理和保存的核心逻辑 】】】
    
    # 1. 创建迭代器，而不是完整的列表，以节省内存
    prediction_tasks_iterator = itertools.product(prepared_drugs_list, new_cell_line_data.items())
    
    # 2. 初始化用于分块的变量
    chunk_results = []
    chunk_index = 1
    output_base, output_ext = os.path.splitext(args.output_file)

    # 3. 使用 tqdm 手动管理进度条，并迭代处理任务
    with tqdm(total=total_predictions, desc="药物-细胞系响应预测") as pbar:
        for task_index, (prepared_drug, (cell_name, omics_data)) in enumerate(prediction_tasks_iterator):
            drug_name, drug_smiles = prepared_drug["name"], prepared_drug["smiles"]
            pbar.set_description(f"预测: {drug_name[:15]:<15} vs {cell_name[:15]:<15}")

            # 如果药物数据准备失败，则记录错误并跳过
            if prepared_drug['status'] != 'Success':
                result = {'drug_name': drug_name, 'drug_smiles': drug_smiles, 'cell_line_name': cell_name,
                          'predicted_ic50_normalized': np.nan, 'predicted_ic50_unscaled': np.nan,
                          'status': prepared_drug['status']}
            else:
                # 正常执行预测
                drug_graph_x = prepared_drug['graph_x']
                drug_graph_edge_index = prepared_drug['graph_edge_index']
                drug_fingerprint_tensor = prepared_drug['fingerprint']
                cell_miRNA_tensor = omics_data['miRNA']
                cell_copynumber_tensor = omics_data['copynumber']

                scaled_prediction, original_prediction, status = np.nan, np.nan, 'Success'
                try:
                    scaled_prediction = predict_response(model, drug_graph_x, drug_graph_edge_index,
                                                         drug_fingerprint_tensor,
                                                         cell_miRNA_tensor, cell_copynumber_tensor, DEVICE)
                    if scaling_params_loaded_successfully:
                        original_prediction = unscale_ic50_value(scaled_prediction)
                    else:
                        original_prediction = "ScalingParamsMissing"
                    status = 'Success' if not pd.isna(scaled_prediction) else 'PredIsNaN'
                except Exception as e_pred:
                    status = f'Error_Prediction: {str(e_pred)[:100]}'
                
                result = {'drug_name': drug_name, 'drug_smiles': drug_smiles, 'cell_line_name': cell_name,
                          'predicted_ic50_normalized': scaled_prediction,
                          'predicted_ic50_unscaled': original_prediction,
                          'status': status}
            
            chunk_results.append(result)
            pbar.update(1)

            # 4. 检查是否达到批次大小，或者是否为最后一个任务
            if (task_index + 1) % CHUNK_SIZE == 0 or (task_index + 1) == total_predictions:
                if not chunk_results:
                    continue
                
                # 生成文件名并保存
                output_filename = f"{output_base}_{chunk_index}{output_ext}"
                print(f"\n处理了 {len(chunk_results)} 个预测，正在保存到文件: {output_filename}")
                results_df = pd.DataFrame(chunk_results)
                results_df.to_csv(output_filename, index=False, float_format='%.8g')
                print(f"  文件已保存。")

                # 5. 清理内存和显存
                print("  正在清理内存和显存...")
                del chunk_results
                del results_df
                chunk_results = []  # 为下一个批次重新初始化
                gc.collect()
                if DEVICE.type == 'cuda':
                    torch.cuda.empty_cache()
                print("  清理完毕。")
                
                chunk_index += 1

    if not total_predictions:
        print("\n未能生成任何预测结果。")
    else:
        print(f"\n所有预测任务已完成并分批保存。文件名前缀为 '{output_base}'。")

    print("\n脚本执行完毕。")