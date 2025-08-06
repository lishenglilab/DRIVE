# predict_all1.py
import argparse
import csv
import os
import pickle
import re
import sys

import networkx as nx
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data
from tqdm import tqdm

# --- Model Definition Imports ---
try:
    from models.gat import GATNet
    from models.gat_gcn import GAT_GCN
    from models.gcn import GCNNet
    from models.ginconv import GINConvNet
except ImportError as e:
    print(f"Error: Could not import model definitions. Please ensure the 'models' folder and its .py files exist.\n{e}",
          file=sys.stderr)
    sys.exit(1)


# --- Helper Functions ---
def atom_features(atom):
    # Generates a feature vector for a single atom.
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) + one_of_k_encoding(atom.GetDegree(),
                                                                                             [0, 1, 2, 3, 4, 5, 6, 7, 8,
                                                                                              9,
                                                                                              10]) + one_of_k_encoding_unk(
        atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + one_of_k_encoding_unk(atom.GetImplicitValence(),
                                                                                          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                                                           10]) + [
                        atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    # Encodes a value into a one-hot vector.
    if x not in allowable_set: raise Exception(f"Input {x} not in allowable set {allowable_set}")
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    # Encodes a value into a one-hot vector, mapping unknown values to the last element.
    if x not in allowable_set: x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
    # Converts a SMILES string to a graph representation (node features and edge index).
    mol = Chem.MolFromSmiles(smile)
    if mol is None: return None, None, None
    c_size = mol.GetNumAtoms();
    if c_size == 0: return None, None, None
    features = [atom_features(atom) for atom in mol.GetAtoms()]
    edges = [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()]
    g = nx.Graph(edges).to_directed()
    edge_index_list = [[e1, e2] for e1, e2 in g.edges()]
    if edge_index_list:
        return c_size, features, np.array(edge_index_list, dtype=np.int64).T
    else:
        return c_size, features, np.empty((2, 0), dtype=np.int64)


def get_training_feature_map(data_dir="mydata/"):
    # Loads or generates a mapping of genetic features to integer indices.
    mut_dict_path = os.path.join(data_dir, 'mut_dict.pkl')
    if os.path.exists(mut_dict_path):
        print(f"Info: Loading pre-saved feature map from '{mut_dict_path}'...")
        with open(mut_dict_path, 'rb') as f: mut_dict = pickle.load(f)
        return mut_dict
    genetic_feature_path = os.path.join(data_dir, "PANCANCER_Genetic_feature.csv")
    print(f"Warning: Feature map '{mut_dict_path}' not found. Attempting to regenerate...")
    if not os.path.exists(genetic_feature_path): print(f"Fatal Error: '{genetic_feature_path}' does not exist.",
                                                       file=sys.stderr); sys.exit(1)
    mut_dict = {}
    with open(genetic_feature_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader, None)
        for item in reader:
            try:
                mut = item[5]
                if mut not in mut_dict: mut_dict[mut] = len(mut_dict)
            except IndexError:
                continue
    with open(mut_dict_path, 'wb') as f:
        pickle.dump(mut_dict, f);
    return mut_dict


def create_cell_feature_vectors(new_cell_data_path, training_feature_map):
    """
    Uses precise matching logic to create feature vectors for new cell lines
    and prints the total intersection count for informational purposes.
    """
    if not os.path.exists(new_cell_data_path):
        raise FileNotFoundError(f"New cell line data file '{new_cell_data_path}' not found.")

    print("Info: Aligning new cell line features...")
    try:
        new_cells_df = pd.read_csv(new_cell_data_path, index_col=0)
    except Exception as e:
        raise ValueError(f"Failed to read cell line data file '{new_cell_data_path}': {e}")

    # --- Print intersection information ---
    # 1. Clean and get the set of model gene names (containing only pure gene names, e.g., 'tp53')
    model_gene_set = set()
    for feature_name in training_feature_map.keys():
        base_name = re.sub(r'_(mut|cnv|fusion)$', '', str(feature_name)).strip().lower()
        model_gene_set.update(base_name.split('-'))

    # 2. Clean and get the set of gene names from user data
    user_gene_set = set(str(col).strip().lower() for col in new_cells_df.columns)

    # 3. Calculate and print the intersection
    intersection_count = len(model_gene_set.intersection(user_gene_set))
    print(
        f"Info: Total number of intersecting genes between your data and the model's gene library: {intersection_count}")
    # --- End of intersection information ---

    # Clean the column names of your data file for subsequent processing
    new_cells_df.columns = [str(col).strip().lower() for col in new_cells_df.columns]

    model_feature_to_index = {str(k).strip().lower(): v for k, v in training_feature_map.items()}
    num_features = len(training_feature_map)
    aligned_features_dict = {}

    for cell_name, cell_data_row in new_cells_df.iterrows():
        new_feature_vector = np.zeros(num_features, dtype=np.float32)
        for gene_in_data, value in cell_data_row.items():
            if pd.notna(value) and value != 0:
                # Assume mutation type if not specified
                feature_to_find = gene_in_data if '_' in gene_in_data else f"{gene_in_data}_mut"
                if feature_to_find in model_feature_to_index:
                    feature_index = model_feature_to_index[feature_to_find]
                    new_feature_vector[feature_index] = 1.0
        aligned_features_dict[cell_name] = new_feature_vector

    print(f"Info: Successfully created feature vectors for {len(aligned_features_dict)} new cell lines.")
    return aligned_features_dict


def unscale_ic50(scaled_ic50_array, epsilon=1e-9):
    # Reverses the sigmoid scaling to get the original IC50 value.
    scaled_ic50_array = np.array(scaled_ic50_array)
    scaled_clipped = np.clip(scaled_ic50_array, epsilon, 1 - epsilon)
    term = (1 - scaled_clipped) / scaled_clipped
    unscaled_values = -10 * np.log(term)
    return unscaled_values


# --- Main Workflow ---
def main():
    parser = argparse.ArgumentParser(description="Predict drug sensitivity for new drugs and new cell lines.")
    parser.add_argument('--model_path', type=str, default='./model_GAT_GCN_GDSC_blind_run1.model',
                        help="Full path to the pretrained model file.")
    parser.add_argument('--model_type', type=str, default='GAT_GCN',
                        choices=['GCNNet', 'GINConvNet', 'GATNet', 'GAT_GCN'],
                        help="Specify the architecture type of the model file.")
    parser.add_argument('--drug_file', type=str, default='./test/drug_sample.csv',
                        help="Path to the CSV file containing new drug names and SMILES.")
    parser.add_argument('--cell_file', type=str, default='./test/mu_sample.csv',
                        help="Path to the CSV file containing new cell line gene data.")
    parser.add_argument('--output_file', type=str, default='predictions_combined.csv',
                        help="Output CSV filename for prediction results.")
    parser.add_argument('--data_dir', type=str, default='mydata/', help="Directory containing preprocessed data.")
    parser.add_argument('--cuda_name', type=str, default="cuda:0", help="CUDA device to use (e.g., 'cuda:0') or 'cpu'.")
    args = parser.parse_args()

    device = torch.device(args.cuda_name if torch.cuda.is_available() and args.cuda_name != 'cpu' else "cpu")
    print(f"--- Starting Combined Prediction ---")
    print(f"Info: Using device: {device}")

    try:
        new_drugs_df = pd.read_csv(args.drug_file, header=None, names=['drug_name', 'smiles']);
        new_drugs_df.dropna(subset=['smiles'], inplace=True)
        smile_graph_dict = {smi: smile_to_graph(smi) for smi in
                            tqdm(new_drugs_df['smiles'].unique(), desc="Processing SMILES") if
                            smile_to_graph(smi)[0] is not None}
        training_feature_map = get_training_feature_map(args.data_dir);
        num_cell_features = len(training_feature_map)
        aligned_cell_features_dict = create_cell_feature_vectors(args.cell_file, training_feature_map)
    except (FileNotFoundError, ValueError) as e:
        print(f"Fatal Error: Data preparation stage failed: {e}", file=sys.stderr); return

    print(f"Info: Loading model {args.model_type} from {args.model_path}...")
    Model = {'GCNNet': GCNNet, 'GINConvNet': GINConvNet, 'GATNet': GATNet, 'GAT_GCN': GAT_GCN}[args.model_type]
    try:
        model = Model() if args.model_type == 'GINConvNet' else Model(num_features_xt=num_cell_features);
        model.load_state_dict(torch.load(args.model_path, map_location=device));
        model.to(device);
        model.eval()
    except Exception as e:
        print(f"Fatal Error: Failed to load model. Please ensure model type and weights file match.\n{e}",
              file=sys.stderr); return

    all_results = [];
    valid_drugs_df = new_drugs_df[new_drugs_df['smiles'].isin(smile_graph_dict.keys())];
    total_predictions = len(valid_drugs_df) * len(aligned_cell_features_dict)
    with tqdm(total=total_predictions, desc="Predicting pairs (drug,cell)") as pbar:
        for _, drug_row in valid_drugs_df.iterrows():
            drug_name, smi = drug_row['drug_name'], drug_row['smiles']
            _, atom_feats_list, edge_index_arr = smile_graph_dict[smi];
            atom_features_tensor = torch.FloatTensor(np.array(atom_feats_list));
            edge_index_tensor = torch.LongTensor(edge_index_arr);
            num_atoms = atom_features_tensor.shape[0]
            for cell_name, cell_vector in aligned_cell_features_dict.items():
                pbar.set_postfix_str(f"Drug: {drug_name[:15]}..., Cell: {cell_name}")
                cell_features_tensor = torch.FloatTensor(cell_vector).unsqueeze(0);
                batch_tensor = torch.zeros(num_atoms, dtype=torch.long)
                data = Data(x=atom_features_tensor, edge_index=edge_index_tensor, target=cell_features_tensor,
                            batch=batch_tensor).to(device)
                with torch.no_grad(): output, _ = model(data)
                scaled_pred = output.item();
                original_pred = unscale_ic50([scaled_pred])[0]
                all_results.append({'drug_name': drug_name, 'cell_line_name': cell_name, 'IC50_scaled': scaled_pred,
                                    'IC50_original': original_pred})
                pbar.update(1)

    if not all_results: print("\nWarning: No predictions were generated."); return
    final_df = pd.DataFrame(all_results)
    try:
        final_df.to_csv(args.output_file, index=False);
        print(f"\n\n✅ Prediction complete! Results have been successfully saved to '{args.output_file}'.")
    except Exception as e:
        print(f"\nError: Failed to save final results to '{args.output_file}': {e}")


if __name__ == '__main__':
    main()