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
import argparse


class AutoEncoder(nn.Module):
    """Defines the AutoEncoder for dimensionality reduction of CNV data."""

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


class NeRD_Net(torch.nn.Module):
    """Defines the main NeRD (NEtwork-based anlaysis for Rational Drug discovery) model."""

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


# --- Helper Functions ---
def smile_to_graph_custom(smile_string, num_node_features=78):
    """Converts a SMILES string to a graph representation (node features and edge index)."""
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
    """Loads pre-computed drug fingerprints from a CSV file."""
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
    """Reads a list of drugs (name, SMILES) to predict from a CSV file."""
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
    """Loads training CNV data and dynamically trains an AutoEncoder for feature reduction."""
    print("Loading and training the CNV dimensionality reduction model (Autoencoder)...")
    try:
        train_cnv_df = pd.read_csv(train_cnv_path, index_col=0)
        train_cnv_features = train_cnv_df.values.astype(np.float32)
        original_dim = train_cnv_features.shape[1]
    except FileNotFoundError:
        print(f"Error: Training CNV data file '{train_cnv_path}' not found.");
        return None, None, None
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_cnv_scaled = scaler.fit_transform(train_cnv_features)
    train_tensor = torch.tensor(train_cnv_scaled, dtype=torch.float).to(device)
    autoencoder = AutoEncoder(input_dim=original_dim).to(device)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-4)
    loss_func = nn.MSELoss()
    print(f"  Starting AE training ({ae_epochs} epochs)...")
    progress_bar = tqdm(range(ae_epochs), desc="    AE Training")
    for epoch in progress_bar:
        autoencoder.train()
        _, decoded = autoencoder(train_tensor)
        loss = loss_func(decoded, train_tensor)
        optimizer.zero_grad();
        loss.backward();
        optimizer.step()
        progress_bar.set_postfix(loss=f"{loss.item():.6f}")
    print("  AE training complete.")
    autoencoder.eval()
    return autoencoder, scaler, train_cnv_df.columns


def predict_response(model, drug_graph_x, drug_graph_edge_index, drug_fingerprint_tensor,
                     cell_miRNA_tensor, cell_copynumber_tensor, device):
    """Performs prediction for a single drug-cell pair."""
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
    """Loads parameters for IC50 value scaling from a JSON file."""
    global g_min_ic50, g_max_ic50
    try:
        with open(params_file_path, 'r') as f:
            params = json.load(f)
        g_min_ic50, g_max_ic50 = params.get('min_ic50'), params.get('max_ic50')
        if g_min_ic50 is None or g_max_ic50 is None:
            print(f"  Warning: 'min_ic50' or 'max_ic50' not found in file '{params_file_path}'.");
            return False
        print(f"  IC50 un-scaling parameters loaded: min={g_min_ic50}, max={g_max_ic50}");
        return True
    except FileNotFoundError:
        print(f"  Error: IC50 scaling parameters file '{params_file_path}' not found.");
        return False
    except Exception as e:
        print(f"  Error: Failed to load IC50 scaling parameters: {e}");
        return False


def unscale_ic50_value(scaled_value):
    """Unscales a normalized prediction back to the original IC50 value."""
    if g_min_ic50 is None or g_max_ic50 is None or pd.isna(scaled_value): return np.nan
    ic50_range = g_max_ic50 - g_min_ic50
    return scaled_value * ic50_range + g_min_ic50 if ic50_range != 0 else g_min_ic50


# --- Main Execution Logic ---
def prepare_new_cell_line_data(new_mirna_path, new_cnv_path, train_mirna_path, train_cnv_path_raw, device):
    """Prepares new cell line data by aligning features and applying dimensionality reduction."""
    print("Preparing new cell line data...")
    autoencoder, cnv_scaler, train_cnv_cols = get_cnv_autoencoder(train_cnv_path_raw, device)
    if autoencoder is None: return {}
    try:
        train_mirna_df = pd.read_csv(train_mirna_path)
        train_mirna_cols = train_mirna_df.columns[1:]
        train_mirna_features = train_mirna_df.iloc[:, 1:].values.astype(np.float32)
        miRNA_scaler = MinMaxScaler(feature_range=(0, 1)).fit(train_mirna_features)
    except FileNotFoundError as e:
        print(f"Error: Training miRNA file not found: {e}.");
        return {}
    try:
        new_mirna_df = pd.read_csv(new_mirna_path, index_col=0)
        new_cnv_df = pd.read_csv(new_cnv_path, index_col=0)
    except FileNotFoundError as e:
        print(f"Error: New cell line sample file not found: {e}.");
        return {}

    new_cell_line_names = new_mirna_df.index.tolist()
    if set(new_cell_line_names) != set(new_cnv_df.index.tolist()):
        print("Warning: miRNA and CNV files have mismatched cell line lists. Using the intersection.")
        shared_cells = list(set(new_cell_line_names) & set(new_cnv_df.index.tolist()))
        new_mirna_df, new_cnv_df = new_mirna_df.loc[shared_cells], new_cnv_df.loc[shared_cells]
        new_cell_line_names = shared_cells
    print("  Aligning and normalizing miRNA data...")
    new_mirna_aligned_df = new_mirna_df.reindex(columns=train_mirna_cols, fill_value=0.0)
    new_mirna_normalized = miRNA_scaler.transform(new_mirna_aligned_df.values.astype(np.float32))
    print("  Aligning, normalizing, and using AE to reduce CNV data dimensionality...")
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
    print(f"  Successfully prepared data for {len(prepared_data)} new cell lines.")
    return prepared_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeRD drug response prediction script")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pretrained model file')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Path to the output CSV file for prediction results')
    parser.add_argument('--drugs_input_file', type=str, required=True,
                        help='Path to the drug SMILES file to be predicted')
    parser.add_argument('--new_mirna_file', type=str, required=True,
                        help='Path to the miRNA expression data file for new cell lines')
    parser.add_argument('--new_cnv_file', type=str, required=True, help='Path to the CNV data file for new cell lines')
    parser.add_argument('--precomputed_fingerprint_file', type=str, required=True,
                        help='Path to the precomputed drug fingerprint file')
    parser.add_argument('--train_mirna_file', type=str, required=True,
                        help='Path to the training set miRNA file for alignment')
    parser.add_argument('--train_cnv_raw_file', type=str, required=True,
                        help='Path to the raw CNV file for AE training and alignment')
    parser.add_argument('--ic50_scaling_params_file', type=str, required=True,
                        help='Path to the JSON file with IC50 normalization parameters')
    args = parser.parse_args()

    N_FILTERS, NUM_FEATURES_XD, OUTPUT_DIM, DRUG_FINGERPRINT_LENGTH = 4, 78, 128, 881

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    print(f"\nLoading main prediction model from {args.model_path}...")
    model = NeRD_Net(n_filters=N_FILTERS, num_features_xd=NUM_FEATURES_XD, output_dim=OUTPUT_DIM, dropout=0.0)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE));
        model.to(DEVICE);
        model.eval()
        print("  Main model loaded successfully.")
    except Exception as e:
        print(f"Error: Failed to load main model: {e}");
        exit()

    print(f"\nLoading IC50 normalization parameters from {args.ic50_scaling_params_file}...")
    scaling_params_loaded_successfully = load_ic50_scaling_params(args.ic50_scaling_params_file)

    new_cell_line_data = prepare_new_cell_line_data(args.new_mirna_file, args.new_cnv_file, args.train_mirna_file,
                                                    args.train_cnv_raw_file, DEVICE)
    if not new_cell_line_data:
        print("Error: No new cell line data could be loaded. Cannot proceed with prediction.");
        exit()

    print(f"\nReading drugs to predict from {args.drugs_input_file}...")
    drugs_for_prediction = read_drugs_for_prediction_from_file(args.drugs_input_file)
    if not drugs_for_prediction:
        print("Error: No drugs to predict could be loaded. Cannot proceed with prediction.");
        exit()
    print(f"  Found {len(drugs_for_prediction)} drugs for prediction.")
    drug_fingerprints_store = load_drug_fingerprints_from_file(args.precomputed_fingerprint_file, drug_name_col=0,
                                                               expected_length=DRUG_FINGERPRINT_LENGTH)

    print("\nPreparing all drug data (generating graph structures and looking up fingerprints)...")
    prepared_drugs_list = []
    for drug_info in tqdm(drugs_for_prediction, desc="  Preparing drugs"):
        drug_name, drug_smiles = drug_info["name"], drug_info["smiles"]
        prepared_drug_data = {'name': drug_name, 'smiles': drug_smiles, 'status': 'Pending',
                              'graph_x': None, 'graph_edge_index': None, 'fingerprint': None}
        clean_drug_name = drug_name.strip().lower()
        fp_numpy_arr = drug_fingerprints_store.get(clean_drug_name)
        if fp_numpy_arr is None:
            tqdm.write(
                f"  Info: Drug '{drug_name}' (looked up as: '{clean_drug_name}') not found in fingerprint library. All predictions for this drug will be skipped.")
            prepared_drug_data['status'] = 'Error_FingerprintMissing'
            prepared_drugs_list.append(prepared_drug_data)
            continue
        try:
            prepared_drug_data['graph_x'], prepared_drug_data['graph_edge_index'] = smile_to_graph_custom(drug_smiles,
                                                                                                          NUM_FEATURES_XD)
            prepared_drug_data['fingerprint'] = torch.tensor(fp_numpy_arr, dtype=torch.float)
            prepared_drug_data['status'] = 'Success'
        except ValueError as e:
            tqdm.write(
                f"  Error: Could not generate graph structure for drug '{drug_name}'. All predictions for this drug will be skipped. Error: {e}")
            prepared_drug_data['status'] = f'Error_GraphGen: {e}'

        prepared_drugs_list.append(prepared_drug_data)

    print(
        f"\nPreparation complete. {sum(1 for d in prepared_drugs_list if d['status'] == 'Success')} / {len(prepared_drugs_list)} drugs can be used for prediction.")
    print(f"Starting prediction of drug response for {len(new_cell_line_data)} cell lines...")

    all_results = []
    prediction_tasks = list(itertools.product(prepared_drugs_list, new_cell_line_data.items()))

    with tqdm(prediction_tasks, desc="Drug-Cell Response Prediction") as pbar:
        for prepared_drug, (cell_name, omics_data) in pbar:
            drug_name, drug_smiles = prepared_drug["name"], prepared_drug["smiles"]
            pbar.set_description(f"Predicting: {drug_name[:15]:<15} vs {cell_name[:15]:<15}")

            if prepared_drug['status'] != 'Success':
                all_results.append({'drug_name': drug_name, 'drug_smiles': drug_smiles, 'cell_line_name': cell_name,
                                    'predicted_ic50_normalized': np.nan, 'predicted_ic50_unscaled': np.nan,
                                    'status': prepared_drug['status']})
                continue

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

            all_results.append({'drug_name': drug_name, 'drug_smiles': drug_smiles, 'cell_line_name': cell_name,
                                'predicted_ic50_normalized': scaled_prediction,
                                'predicted_ic50_unscaled': original_prediction,
                                'status': status})

    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(args.output_file, index=False, float_format='%.8g')
        print(f"\nAll prediction results have been saved to '{args.output_file}'")
    else:
        print("\nNo prediction results were generated.")

    print("\nScript execution finished.")