import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
import torch.cuda.amp as amp
import argparse

# Import custom modules
try:
    from Step2_DataEncoding import DataEncoding
    from Step3_model import DeepTTC, data_process_loader
except ImportError as e:
    print(f"Error when importing custom modules: {e}")
    print(
        "Please ensure Step2_DataEncoding.py and Step3_model.py are in the Python search path or the same directory as this script.")
    exit(1)

# --- Performance Configuration ---
GPU_BATCH_SIZE = 4096


def load_new_drugs(filepath):
    """Load new drug data file."""
    try:
        df = pd.read_csv(filepath, header=None, names=['DrugName', 'SMILES'], sep=',')
        # Fallback to tab separator if comma-separated parsing is likely incorrect
        if df.shape[1] <= 1 or df['SMILES'].isnull().mean() > 0.9:
            df = pd.read_csv(filepath, header=None, names=['DrugName', 'SMILES'], sep='\t')
        print(f"Successfully loaded new drug file: {filepath}, containing {df.shape[0]} drug records.")
        return df
    except FileNotFoundError:
        print(f"Error: New drug file {filepath} not found.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An unknown error occurred while loading new drug file {filepath}: {e}")
        return pd.DataFrame()


def load_and_align_new_cell_lines(new_cell_line_path, training_gene_list_path):
    """Load and align gene expression data for new cell lines."""
    try:
        print(f"Loading training gene list from {training_gene_list_path}...")
        training_genes = pd.read_csv(training_gene_list_path, sep='\t', index_col=0).index.tolist()
        print(f"The model was trained using {len(training_genes)} genes.")

        print(f"Loading new cell line data from {new_cell_line_path}...")
        new_cells_df = pd.read_csv(new_cell_line_path, index_col=0)
        new_cells_df.index = new_cells_df.index.astype(str)
        print(f"Loaded {new_cells_df.shape[0]} new cell lines, each with {new_cells_df.shape[1]} gene features.")

        print("Aligning gene dimensions of new cell lines to match model input...")
        aligned_cells_df = new_cells_df.reindex(columns=training_genes, fill_value=0.0)
        aligned_cells_df = aligned_cells_df.astype(np.float32)

        print(f"Gene alignment complete. Final dimension of cell line data for prediction is: {aligned_cells_df.shape}")
        return aligned_cells_df
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}. Please check the path.")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred while loading or aligning cell line data: {e}")
        return pd.DataFrame()


def prepare_data_for_prediction(drugs_df, rna_data):
    """Prepare data for all drug-cell line combinations."""
    num_drugs = len(drugs_df)
    num_cells = len(rna_data)

    # Use NumPy and Pandas broadcasting/repeating for efficient pair creation
    drug_repeated = drugs_df.loc[drugs_df.index.repeat(num_cells)].reset_index(drop=True)
    rna_tiled = pd.concat([rna_data] * num_drugs, ignore_index=False).reset_index()
    rna_tiled.rename(columns={'index': 'COSMIC_ID'}, inplace=True)

    # Simple column binding, as the order is guaranteed
    combined_df = pd.concat([drug_repeated, rna_tiled], axis=1)

    # Separate the drug and RNA parts
    final_drug_df = combined_df[drugs_df.columns.tolist() + ['COSMIC_ID']]
    final_rna_df = combined_df[rna_data.columns]

    return final_drug_df, final_rna_df


def custom_collate_fn(batch):
    """Custom collate function to handle drug encoding tuples."""
    drug_ids_list, drug_masks_list, rna_data_list, labels_list = [], [], [], []
    for item in batch:
        drug_ids_list.append(item[0][0])
        drug_masks_list.append(item[0][1])
        rna_data_list.append(item[1])
        labels_list.append(item[2])
    v_drug_collated = (default_collate(drug_ids_list), default_collate(drug_masks_list))
    v_p_collated = default_collate(rna_data_list)
    y_collated = default_collate(labels_list)
    return v_drug_collated, v_p_collated, y_collated


def main(args):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    print("\n--- Step 1: Loading and preparing new drug and cell line data ---")
    new_drugs_df = load_new_drugs(args.new_drug_file)
    if new_drugs_df.empty:
        print("Error: Failed to load drug data, terminating program.")
        exit(1)

    aligned_new_cells_df = load_and_align_new_cell_lines(args.new_cell_line_file, args.training_gene_list_file)
    if aligned_new_cells_df.empty:
        print("Error: Failed to load cell line data, terminating program.")
        exit(1)

    print("\nPre-encoding SMILES for all new drugs...")
    data_encoder = DataEncoding(vocab_dir=args.vocab_dir)
    encoded_smiles = [
        data_encoder._drug2emb_encoder(s) if pd.notna(s) else None
        for s in tqdm(new_drugs_df['SMILES'], desc="Encoding SMILES")
    ]
    new_drugs_df['drug_encoding'] = encoded_smiles
    original_drug_count = len(new_drugs_df)
    new_drugs_df.dropna(subset=['drug_encoding'], inplace=True)
    print(f"SMILES pre-encoding complete. Number of valid drugs: {len(new_drugs_df)} / {original_drug_count}.")
    if new_drugs_df.empty:
        print("Error: No drugs could be successfully encoded, terminating program.")
        exit(1)

    print(f"\n--- Step 2: Loading the model ---")
    model_weights_file = os.path.join(args.model_dir, 'model.pt')
    net = DeepTTC(modeldir=args.model_dir)
    try:
        net.load_pretrained(model_weights_file)
        net.model.to(DEVICE)
        net.model.eval()
        print("Model loaded successfully and set to evaluation mode.")
    except Exception as e:
        print(f"Failed to load model: {e}");
        exit(1)

    print(f"\n--- Step 3: Preparing combined data and predicting ---")
    combined_drug_df, combined_rna_df = prepare_data_for_prediction(new_drugs_df, aligned_new_cells_df)

    # Add a dummy 'Label' column, as it's required by data_process_loader
    combined_drug_df['Label'] = 0.0

    if combined_drug_df.empty:
        print("Failed to generate drug-cell pairs for prediction.");
        exit(1)

    pred_dataset = data_process_loader(
        list_IDs=combined_drug_df.index.values,
        labels=combined_drug_df['Label'].values,
        drug_df=combined_drug_df,
        rna_df=combined_rna_df
    )

    params = {
        'batch_size': GPU_BATCH_SIZE, 'shuffle': False, 'num_workers': 0, 'drop_last': False,
        'sampler': SequentialSampler(pred_dataset), 'collate_fn': custom_collate_fn
    }
    predict_generator = DataLoader(pred_dataset, **params)

    all_predictions = []
    with torch.no_grad(), amp.autocast():
        for v_drug_batch, v_gene_batch, _ in tqdm(predict_generator, desc="Predicting"):
            v_drug_batch = (v_drug_batch[0].to(DEVICE), v_drug_batch[1].to(DEVICE))
            v_gene_batch = v_gene_batch.to(DEVICE)
            scores = net.model(v_drug_batch, v_gene_batch)
            all_predictions.extend(scores.squeeze(1).cpu().numpy().tolist())

    print("\n--- Step 4: Saving final results ---")
    combined_drug_df['Predicted_LN_IC50'] = all_predictions
    results_df = combined_drug_df[['DrugName', 'COSMIC_ID', 'SMILES', 'Predicted_LN_IC50']]

    try:
        results_df.to_csv(args.output_file, index=False)
        print(f"All prediction tasks complete. {len(results_df)} results saved to {args.output_file}")
    except Exception as e:
        print(f"An error occurred while saving results to {args.output_file}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DeepTTC drug response prediction script")
    parser.add_argument('--new_drug_file', type=str, required=True,
                        help='Path to the new drug file (CSV: DrugName,SMILES)')
    parser.add_argument('--new_cell_line_file', type=str, required=True,
                        help='Path to the new cell line gene expression file')
    parser.add_argument('--training_gene_list_file', type=str, required=True,
                        help='Path to the training gene list file for gene alignment')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory path containing model weights (model.pt) and configuration')
    parser.add_argument('--vocab_dir', type=str, required=True,
                        help='Directory path containing SMILES vocabulary files')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Path to the output CSV file for prediction results')

    args = parser.parse_args()

    print("Running script for combined prediction on new drugs and new cell lines...")
    main(args)