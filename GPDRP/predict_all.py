import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import os
from rdkit import Chem
import numpy as np
import networkx as nx
import csv
import traceback
import math
from tqdm import tqdm

# --- Model Definition Imports ---
try:
    from GIN.model.gin import GINConvNet as OriginalGINConvNet
except ImportError:
    print("Warning: Could not import OriginalGINConvNet from GIN.model.gin. Predictions with 'GIN' will fail.")
    OriginalGINConvNet = None
try:
    from GAT.model.gat import GATNet
except ImportError:
    print("Warning: Could not import GATNet from GAT.model.gat. Predictions with 'GAT' will fail.")
    GATNet = None
try:
    from GCN.model.gcn import GCNNet
except ImportError:
    print("Warning: Could not import GCNNet from GCN.model.gcn. Predictions with 'GCN' will fail.")
    GCNNet = None
try:
    from GIN_TRANSFORMER.model.gintranformer import GINConvNet2
except ImportError:
    print(
        "Warning: Could not import GINConvNet2 from GIN_TRANSFORMER.model.gintransformer. Predictions with 'GINTransformer' will fail.")
    GINConvNet2 = None

# --- Global Constants ---
EXPECTED_ATOM_FEATURE_DIM = 78


# --- Helper Functions ---
def one_of_k_encoding(x, allowable_set):
    # Encodes a value into a one-hot vector.
    if x not in allowable_set:
        if isinstance(x, int) and allowable_set and isinstance(allowable_set[0], int):
            if x > allowable_set[-1]:
                x = allowable_set[-1]
            elif x < allowable_set[0]:
                x = allowable_set[0]
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    # Encodes a value into a one-hot vector, mapping unknown values to the last element.
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features_from_preprocessing(atom):
    # Generates a feature vector for a given atom based on its properties.
    allowable_symbols = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                         'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                         'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                         'Pt', 'Hg', 'Pb', 'Unknown']
    allowable_degree = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    allowable_total_hs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    allowable_implicit_valence = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    try:
        features = np.array(
            one_of_k_encoding_unk(atom.GetSymbol(), allowable_symbols) +
            one_of_k_encoding(atom.GetDegree(), allowable_degree) +
            one_of_k_encoding_unk(atom.GetTotalNumHs(), allowable_total_hs) +
            one_of_k_encoding_unk(atom.GetImplicitValence(), allowable_implicit_valence) +
            [atom.GetIsAromatic()]
        )
    except Exception:
        return np.zeros(EXPECTED_ATOM_FEATURE_DIM)
    return features


def smiles_to_graph_data_with_cell(smiles_string, cell_feature_vector):
    # Converts a SMILES string and a cell line feature vector into a PyG Data object.
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    atom_f_list = []
    for atom_idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(atom_idx)
        feature_vec = atom_features_from_preprocessing(atom)
        sum_feature = np.sum(feature_vec)
        if sum_feature == 0:
            normalized_feature = feature_vec
        else:
            normalized_feature = feature_vec / sum_feature
        atom_f_list.append(normalized_feature)

    x = torch.tensor(np.array(atom_f_list), dtype=torch.float)

    if x.shape[0] > 0 and x.shape[1] != EXPECTED_ATOM_FEATURE_DIM:
        print(
            f"FATAL ERROR: Atom features for SMILES {smiles_string} have dim {x.shape[1]}, expected {EXPECTED_ATOM_FEATURE_DIM}.")
        return None

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])

    if not edges:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        g_nx = nx.Graph(edges).to_directed()
        edge_index_list = []
        if g_nx.number_of_edges() > 0:
            for e1, e2 in g_nx.edges:
                edge_index_list.append([e1, e2])
            edge_index = torch.tensor(np.array(edge_index_list).T, dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

    target_ge_tensor = torch.tensor(cell_feature_vector, dtype=torch.float).unsqueeze(0)
    data = Data(x=x, edge_index=edge_index, target_ge=target_ge_tensor)
    return data


def load_training_gene_info_and_norm_params(file_path):
    # Loads the gene list and normalization parameters used during model training.
    print(f"Loading training gene list and normalization parameters from '{file_path}'...")
    try:
        df = pd.read_csv(file_path, sep='\t')
        training_genes = df.iloc[:, 1].tolist()
        numeric_df = df.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')
        all_values = numeric_df.values.flatten()
        all_values = all_values[~np.isnan(all_values)]

        if len(all_values) == 0:
            raise ValueError("No valid gene expression values found in the file.")

        min_val = np.min(all_values)
        max_val = 12.0
        print(
            f"Loading complete: {len(training_genes)} genes. Normalization parameters: min={min_val:.4f}, max={max_val:.4f}")
        return training_genes, min_val, max_val

    except FileNotFoundError:
        print(
            f"Error: Training gene expression file '{file_path}' not found. This is necessary for gene alignment and normalization.")
        return None, None, None
    except Exception as e:
        print(f"Error loading training gene information: {e}")
        return None, None, None


def load_and_preprocess_new_cell_lines(new_cell_file, training_genes, min_val, max_val):
    # Loads and preprocesses new cell line data by aligning genes and normalizing values.
    print(f"Loading and preprocessing new cell line file: '{new_cell_file}'...")
    try:
        df_new = pd.read_csv(new_cell_file, index_col=0)
        print(f"Successfully loaded {df_new.shape[0]} new cell lines with {df_new.shape[1]} genes.")

        print("Aligning genes...")
        df_aligned = df_new.reindex(columns=training_genes, fill_value=0.0)
        print(f"Gene alignment complete. Feature dimension: {df_aligned.shape[1]}")

        print("Performing normalization...")
        X = df_aligned.values.astype(np.float32)
        X = np.clip(X, None, max_val)

        if (max_val - min_val) == 0:
            X_normalized = np.zeros_like(X)
        else:
            X_normalized = (X - min_val) / (max_val - min_val)

        X_normalized = np.clip(X_normalized, 0.0, 1.0)
        print("Normalization complete.")

        cell_features_map = {name: vector for name, vector in zip(df_aligned.index, X_normalized)}
        return cell_features_map

    except FileNotFoundError:
        print(f"Error: New cell line file '{new_cell_file}' not found.")
        return None
    except Exception as e:
        print(f"Error processing new cell line file: {e}")
        traceback.print_exc()
        return None


def unscale_ic50(scaled_value_pred):
    # Reverses the scaling transformation to get the original IC50 value.
    epsilon = 1e-9
    if not isinstance(scaled_value_pred, (int, float, np.float32, np.float64)):
        if isinstance(scaled_value_pred, torch.Tensor):
            scaled_value_pred = scaled_value_pred.item() if scaled_value_pred.numel() == 1 else np.nan
        elif isinstance(scaled_value_pred, np.ndarray):
            scaled_value_pred = scaled_value_pred.item() if scaled_value_pred.size == 1 else np.nan
        else:
            return np.nan
    if np.isnan(scaled_value_pred):
        return np.nan

    y = np.clip(scaled_value_pred, epsilon, 1.0 - epsilon)
    try:
        ratio = y / (1.0 - y)
        if ratio <= 0: return np.nan
        original_value = -10.0 * math.log((1.0 - y) / y)
    except (ValueError, OverflowError):
        original_value = np.nan
    return original_value


def predict_new_drugs(model, device, data_loader):
    # Performs prediction for a given model and data loader.
    model.eval()
    total_preds = torch.Tensor()
    with torch.no_grad():
        for data_batch in data_loader:
            data_batch = data_batch.to(device)
            try:
                out_tuple = model(data_batch)
                output = out_tuple[0] if isinstance(out_tuple, tuple) else out_tuple
                if output.ndim == 1: output = output.unsqueeze(1)
            except Exception:
                num_graphs_in_batch = data_batch.num_graphs if hasattr(data_batch, 'num_graphs') else 1
                output = torch.full((num_graphs_in_batch, 1), float('nan'), device=device)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
    return total_preds.numpy()


def main():
    parser = argparse.ArgumentParser(
        description='Predict IC50 for new drugs and new cell lines using a pre-trained GNN model.')
    # Define command-line arguments.
    parser.add_argument('--smiles_file', type=str, required=True,
                        help='Drug input file: CSV format, no header, 1st column is drug name, 2nd column is SMILES string.')
    parser.add_argument('--new_cell_line_file', type=str, required=True,
                        help='New cell line input file: CSV format, 1st row is gene names (header), 1st column is cell line names (index).')
    parser.add_argument('--training_gene_expression_file', type=str, required=True,
                        help='Path to the original training gene expression file (exp.txt) for alignment and normalization.')
    parser.add_argument('--model_file', type=str, required=True,
                        help='Path to the pre-trained model weights file (.model).')
    parser.add_argument('--model_type', type=str, default='GIN', choices=['GIN', 'GAT', 'GCN', 'GINTransformer'],
                        help='The GNN model type to use.')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Path to the CSV file to save prediction results.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for prediction.')
    parser.add_argument('--cuda_name', type=str, default="cuda:0", help='CUDA device name (e.g., cuda:0, cpu).')

    args = parser.parse_args()
    print(f"Script arguments: {args}")

    # Load gene info from the training dataset for alignment and normalization
    training_genes, norm_min, norm_max = load_training_gene_info_and_norm_params(args.training_gene_expression_file)
    if not training_genes:
        print("Error: Failed to load training gene information, cannot continue.");
        return

    # Load and preprocess new cell lines
    cell_features_map = load_and_preprocess_new_cell_lines(args.new_cell_line_file, training_genes, norm_min, norm_max)
    if cell_features_map is None or not cell_features_map:
        print("Error: Failed to load or process the new cell line file, cannot continue.");
        return
    print(f"Successfully prepared features for {len(cell_features_map)} new cell lines.")

    # Load drug SMILES data
    if not os.path.exists(args.smiles_file):
        print(f"Error: Drug SMILES file not found at '{args.smiles_file}'");
        return
    try:
        df_smiles = pd.read_csv(args.smiles_file, header=None, names=['drug_name', 'smiles'])
        smiles_input_list = df_smiles.values.tolist()
        if not smiles_input_list:
            print(f"Error: No drug data found in '{args.smiles_file}'.");
            return
    except Exception as e:
        print(f"Error reading drug SMILES file '{args.smiles_file}': {e}");
        return

    # Set up device and model
    device = torch.device(args.cuda_name if torch.cuda.is_available() and args.cuda_name != "cpu" else "cpu")
    print(f"Using device: {device}")

    model_map = {'GIN': OriginalGINConvNet, 'GAT': GATNet, 'GCN': GCNNet, 'GINTransformer': GINConvNet2}
    model_class = model_map.get(args.model_type)
    if model_class is None:
        print(f"Error: Model class '{args.model_type}' was not correctly imported or is unavailable.");
        return

    model_instance = model_class().to(device)
    print(f"Instantiated model: {args.model_type}")

    if not os.path.exists(args.model_file):
        print(f"Error: Model file not found at '{args.model_file}'.");
        return
    try:
        model_instance.load_state_dict(torch.load(args.model_file, map_location=device))
        print(f"Successfully loaded model weights from '{args.model_file}'.")
    except Exception as e:
        print(f"Error loading model weights: {e}");
        return

    # Main prediction loop
    all_results_list = []
    for drug_name, smiles_str in tqdm(smiles_input_list, desc="Processing drugs"):
        if not isinstance(smiles_str, str) or not smiles_str.strip() or Chem.MolFromSmiles(smiles_str) is None:
            # Handle invalid SMILES by recording NaN for all cell lines
            for cell_name in cell_features_map.keys():
                all_results_list.append({'drug_name': drug_name, 'smiles': smiles_str, 'cell_line_name': cell_name,
                                         'predicted_ic50_scaled': np.nan, 'predicted_ic50_original': np.nan})
            continue

        drug_cell_pairs_data = []
        drug_cell_pairs_info = []
        # Create a graph for each drug-cell pair
        for cell_name, cell_feat_vector in cell_features_map.items():
            graph_data = smiles_to_graph_data_with_cell(smiles_str, cell_feat_vector)
            if graph_data:
                drug_cell_pairs_data.append(graph_data)
                drug_cell_pairs_info.append({'drug_name': drug_name, 'smiles': smiles_str, 'cell_line_name': cell_name})
            else:
                # Handle cases where graph creation fails for a specific pair
                all_results_list.append({'drug_name': drug_name, 'smiles': smiles_str, 'cell_line_name': cell_name,
                                         'predicted_ic50_scaled': np.nan, 'predicted_ic50_original': np.nan})

        if not drug_cell_pairs_data: continue

        # Predict in batches for the current drug against all cell lines
        current_drug_loader = DataLoader(drug_cell_pairs_data, batch_size=args.batch_size, shuffle=False)
        try:
            scaled_predictions = predict_new_drugs(model_instance, device, current_drug_loader)
            for i, pred_info in enumerate(drug_cell_pairs_info):
                result = pred_info.copy()
                scaled_pred_val = float(scaled_predictions[i][0]) if i < len(scaled_predictions) else np.nan
                result['predicted_ic50_scaled'] = scaled_pred_val
                result['predicted_ic50_original'] = unscale_ic50(scaled_pred_val)
                all_results_list.append(result)
        except Exception as e_pred:
            print(f"An error occurred during prediction for drug '{drug_name}': {e_pred}")
            for info_dict in drug_cell_pairs_info:
                result = info_dict.copy()
                result['predicted_ic50_scaled'] = np.nan;
                result['predicted_ic50_original'] = np.nan
                all_results_list.append(result)

        # Clean up memory
        del drug_cell_pairs_data, drug_cell_pairs_info
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Save results
    if not all_results_list:
        print("No predictions were generated.");
        return

    results_df = pd.DataFrame(all_results_list)
    cols_order = ['drug_name', 'smiles', 'cell_line_name', 'predicted_ic50_scaled', 'predicted_ic50_original']
    results_df = results_df[cols_order]

    results_df.to_csv(args.output_file, index=False)
    print(f"\nPrediction complete! Results saved to '{args.output_file}'")
    print("\nResults preview (first 5 rows):")
    print(results_df.head())


if __name__ == "__main__":
    main()