import os
import argparse
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from functools import reduce
import warnings
import codecs
import re

# --- Import RDKit and fingerprint-related libraries ---
try:
    from rdkit import Chem, rdBase
    from rdkit.Chem import AllChem
    from PyBioMed.PyMolecule.PubChemFingerprints import calcPubChemFingerAll
    from subword_nmt.apply_bpe import BPE
except ImportError as e:
    print(f"Error: Missing necessary fingerprint libraries: {e}")
    print("Please run 'pip install rdkit-pypi PyBioMed subword-nmt' to install dependencies.")
    exit(1)

# Suppress redundant RDKit logs
rdBase.DisableLog('rdApp.*')

# --- Import project modules ---
try:
    from config import get_cfg_defaults
    from model import BANDRP
except ImportError as e:
    print(f"Error: Failed to import project modules (config, model): {e}")
    print("Please ensure config.py, model.py, and BAN.py are in the same directory as this script or in a Python-searchable path.")
    exit(1)


# ==============================================================================
# Module 1: Drug Fingerprint Generator
# ==============================================================================
class DrugFingerprintGenerator:
    """Encapsulates Morgan, PubChem, and ESPF fingerprint generation logic."""
    def __init__(self, vocab_path='./pre_process/drug_codes_chembl_freq_1500.txt',
                 subword_map_path='./pre_process/subword_units_map_chembl_freq_1500.csv'):
        print("\n--- Step 1: Initializing Drug Fingerprint Generator ---")
        try:
            bpe_codes_drug = codecs.open(vocab_path, 'r', 'utf-8')
            self.dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
            sub_csv = pd.read_csv(subword_map_path)
            self.idx2word_d = sub_csv['index'].values
            self.words2idx_d = dict(zip(self.idx2word_d, range(0, len(self.idx2word_d))))
            self.morgan_dim = 2048
            self.pubchem_dim = 881
            self.espf_dim = len(self.idx2word_d)
            print(f"  - ESPF generator initialized successfully (vocabulary size: {self.espf_dim})")
            print(f"  - Fingerprint dimensions: Morgan={self.morgan_dim}, PubChem={self.pubchem_dim}, ESPF={self.espf_dim}")
        except FileNotFoundError as e:
            print(f"Fatal Error: Failed to initialize fingerprint generator, file not found: {e}")
            exit(1)

    def smiles_to_espf(self, smiles):
        try:
            t1 = self.dbpe.process_line(smiles).split()
            i1 = np.asarray([self.words2idx_d.get(i) for i in t1 if i in self.words2idx_d])
            if i1.size == 0: return np.zeros(self.espf_dim, dtype=np.float32)
            v1 = np.zeros(self.espf_dim, dtype=np.float32)
            v1[i1] = 1
            return v1
        except Exception:
            return np.zeros(self.espf_dim, dtype=np.float32)

    def smiles_to_morgan(self, mol):
        if mol is None: return np.zeros(self.morgan_dim, dtype=np.float32)
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=self.morgan_dim)
            return np.array(fp, dtype=np.float32)
        except Exception:
            return np.zeros(self.morgan_dim, dtype=np.float32)

    def smiles_to_pubchem(self, mol):
        if mol is None: return np.zeros(self.pubchem_dim, dtype=np.float32)
        try:
            return np.array(calcPubChemFingerAll(mol), dtype=np.float32)
        except Exception:
            return np.zeros(self.pubchem_dim, dtype=np.float32)

    def generate_all_fingerprints(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            zeros_morgan = np.zeros(self.morgan_dim, dtype=np.float32)
            zeros_pubchem = np.zeros(self.pubchem_dim, dtype=np.float32)
            zeros_espf = np.zeros(self.espf_dim, dtype=np.float32)
            return (zeros_morgan, zeros_pubchem, zeros_espf), False

        morgan_fp = self.smiles_to_morgan(mol)
        pubchem_fp = self.smiles_to_pubchem(mol)
        espf_fp = self.smiles_to_espf(smiles)
        return (morgan_fp, pubchem_fp, espf_fp), True


# ==============================================================================
# Module 2: Cell Line Feature Processing
# ==============================================================================
def load_and_align_multiple_cell_features(cfg, new_exp_path, new_mut_path, new_cnv_path):
    print("\n--- Step 2: Loading and aligning new cell line features ---")
    try:
        print("  - Reading reference gene lists from paths specified in config.py for alignment...")
        ref_exp_cols = pd.read_csv(cfg.path.expression, index_col=0, nrows=0).columns.tolist()
        ref_mut_cols = pd.read_csv(cfg.path.mutation, index_col=0, nrows=0).columns.tolist()
        ref_cnv_cols = pd.read_csv(cfg.path.cnv, index_col=0, nrows=0).columns.tolist()
        cell_exp_dim = len(ref_exp_cols)
        cell_mut_dim = len(ref_mut_cols)
        cell_cnv_dim = len(ref_cnv_cols)
        print(f"  - Reference feature dimensions: EXP={cell_exp_dim}, MUT={cell_mut_dim}, CNV={cell_cnv_dim}")
        print("  - Loading omics data for new cell lines...")
        exp_sample_df = pd.read_csv(new_exp_path, index_col=0)
        mut_sample_df = pd.read_csv(new_mut_path, index_col=0)
        cnv_sample_df = pd.read_csv(new_cnv_path, index_col=0)
        print(f"  - Original sample counts: EXP={len(exp_sample_df)}, MUT={len(mut_sample_df)}, CNV={len(cnv_sample_df)}")
        common_cell_ids = sorted(list(
            reduce(lambda x, y: x.intersection(y),
                   [set(exp_sample_df.index), set(mut_sample_df.index), set(cnv_sample_df.index)])
        ))
        if not common_cell_ids:
            print("\nError: No common cell line IDs found across the three provided omics files. Cannot proceed with prediction.")
            exit(1)
        print(f"  - Detected {len(common_cell_ids)} common cell lines for prediction: {common_cell_ids[:5]}...")
        exp_common_df = exp_sample_df.loc[common_cell_ids]
        mut_common_df = mut_sample_df.loc[common_cell_ids]
        cnv_common_df = cnv_sample_df.loc[common_cell_ids]
        print("  - Aligning gene features with the reference lists...")
        exp_aligned_df = pd.DataFrame(0.0, index=exp_common_df.index, columns=ref_exp_cols)
        mut_aligned_df = pd.DataFrame(0.0, index=mut_common_df.index, columns=ref_mut_cols)
        cnv_aligned_df = pd.DataFrame(0.0, index=cnv_common_df.index, columns=ref_cnv_cols)
        exp_aligned_df.update(exp_common_df)
        mut_aligned_df.update(mut_common_df)
        cnv_aligned_df.update(cnv_common_df)
        print("  - New cell line features aligned successfully!")
        return exp_aligned_df, mut_aligned_df, cnv_aligned_df, cell_exp_dim, cell_mut_dim, cell_cnv_dim
    except FileNotFoundError as e:
        print(f"\nError: Reference or input file not found: {e}.")
        exit(1)
    except Exception as e:
        print(f"\nError: An unexpected error occurred while processing cell line features: {e}. Please check if the file format is correct (rows=CellLineID, columns=Gene).")
        exit(1)


# ==============================================================================
# Module 3: Core Prediction Function
# ==============================================================================
def predict_matrix(model, device, cell_data_tuple, drugs_df, fp_generator,
                   output_csv_path, drug_batch_size):
    """
    Performs matrix prediction for a set of cell lines and drugs.
    This version saves the results in batches to separate, numbered files.
    """
    model.eval()
    exp_df, mut_df, cnv_df = cell_data_tuple
    cell_lines_to_predict = exp_df.index.tolist()

    # --- Step 4a: Pre-calculating and validating fingerprints for all drugs ---
    print("\n--- Step 4a: Pre-calculating and validating fingerprints for all drugs... ---")
    drug_fingerprints_cache = {}
    valid_drugs_for_prediction = []
    unprocessed_drugs = []

    for _, row in tqdm(drugs_df.iterrows(), total=len(drugs_df), desc="Validating SMILES and calculating fingerprints"):
        drug_name = row['DrugName']
        smiles = str(row['SMILES'])
        (morgan_fp, pubchem_fp, espf_fp), success = fp_generator.generate_all_fingerprints(smiles)
        if not success:
            unprocessed_drugs.append(
                {'DrugName': drug_name, 'SMILES': smiles, 'Reason': 'Invalid or unparsable SMILES string'})
            continue
        drug_fingerprints_cache[drug_name] = [morgan_fp, espf_fp, pubchem_fp]
        valid_drugs_for_prediction.append({'DrugName': drug_name})

    # Save log for unprocessed drugs
    if unprocessed_drugs:
        try:
            unprocessed_df = pd.DataFrame(unprocessed_drugs)
            base, ext = os.path.splitext(output_csv_path)
            unprocessed_csv_log_path = f"{base}_unprocessed_drugs.csv"
            unprocessed_df.to_csv(unprocessed_csv_log_path, index=False, encoding='utf-8-sig')
            print(f"\nWarning: {len(unprocessed_df)} drugs could not be processed due to invalid SMILES.")
            print(f"  - A detailed list has been saved to: {os.path.abspath(unprocessed_csv_log_path)}")
        except Exception as e:
            print(f"\nError: Failed to save the list of unprocessed drugs: {e}")

    if not valid_drugs_for_prediction:
        print("\n\nFatal Error: After validation, no drugs with valid SMILES were found for prediction.")
        return 0, []  # Return processed count and file list

    num_valid_drugs = len(valid_drugs_for_prediction)
    print(f"  - Fingerprint validation complete! Predictions will be made for {num_valid_drugs} valid drugs.")

    # --- Batch prediction logic ---
    print(f"\n--- Step 4b: Starting batch prediction, max {drug_batch_size} drugs per batch, results will be saved to separate files ---")

    # Extract base name and extension from the user-provided output path to generate numbered filenames
    output_base, output_ext = os.path.splitext(output_csv_path)

    num_batches = (num_valid_drugs + drug_batch_size - 1) // drug_batch_size
    batch_iterator = tqdm(range(0, num_valid_drugs, drug_batch_size), total=num_batches, desc="Processing drug batches")

    generated_files = []  # Stores all generated filenames

    # Use enumerate to get batch number (starting from 1)
    for batch_num, i in enumerate(batch_iterator, 1):
        start_idx = i
        end_idx = min(i + drug_batch_size, num_valid_drugs)
        current_drug_batch_info = valid_drugs_for_prediction[start_idx:end_idx]

        # Prepare fingerprint tensors for the current drug batch
        drug_fp_batches = [[], [], []]
        drug_names_in_order = []
        for drug_info in current_drug_batch_info:
            drug_name = drug_info['DrugName']
            drug_names_in_order.append(drug_name)
            fps = drug_fingerprints_cache[drug_name]
            for j in range(3):
                drug_fp_batches[j].append(fps[j])

        drug_fp_tensors = [torch.tensor(np.array(batch)).to(device).float() for batch in drug_fp_batches]
        num_drugs_in_batch = len(drug_names_in_order)

        # List to store all prediction results for the current batch
        all_predictions_for_this_batch = []

        cell_iterator = tqdm(cell_lines_to_predict, desc=f"Predicting cell lines (Batch {batch_num}/{num_batches})", leave=False)
        with torch.no_grad():
            for cell_name in cell_iterator:
                exp_vec = exp_df.loc[cell_name].values.astype(np.float32)
                mut_vec = mut_df.loc[cell_name].values.astype(np.float32)
                cnv_vec = cnv_df.loc[cell_name].values.astype(np.float32)

                exp_batch = torch.from_numpy(exp_vec).unsqueeze(0).repeat(num_drugs_in_batch, 1).to(device)
                mut_batch = torch.from_numpy(mut_vec).unsqueeze(0).repeat(num_drugs_in_batch, 1).to(device)
                cnv_batch = torch.from_numpy(cnv_vec).unsqueeze(0).repeat(num_drugs_in_batch, 1).to(device)
                cell_data_batch = [exp_batch, mut_batch, cnv_batch]

                predictions, _ = model(drug_fp_tensors, cell_data_batch)
                predictions_list = predictions.cpu().numpy().flatten().tolist()

                results_df = pd.DataFrame({
                    'CellLineID': cell_name,
                    'DrugName': drug_names_in_order,
                    'PredictedValue': predictions_list
                })
                all_predictions_for_this_batch.append(results_df)

        # After processing all cell lines for a batch, merge and save the results to a separate file
        if all_predictions_for_this_batch:
            # Merge all results from the current batch
            batch_results_df = pd.concat(all_predictions_for_this_batch, ignore_index=True)

            # Generate the numbered output filename
            batch_output_path = f"{output_base}_part_{batch_num}{output_ext}"
            generated_files.append(batch_output_path)

            # Save the batch results to the new file
            batch_results_df.to_csv(batch_output_path, index=False, encoding='utf-8-sig')

            batch_iterator.set_postfix_str(
                f"Batch {batch_num}/{num_batches} saved to {os.path.basename(batch_output_path)}")

    # Return the total number of successfully processed drugs and the list of all generated files
    return num_valid_drugs, generated_files


# ==============================================================================
# Main execution block
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Use the BANDRP model for prediction and save results in batches to separate files.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # --- Argument definitions ---
    parser.add_argument('--model_path', type=str, default='./github_upload/output_dir/db1/model.pt',
                        help='Path to the pre-trained model file (.pt).')
    parser.add_argument('--exp_path', type=str, default='./depmap/gene_depmap.csv',
                        help='Path to the [Gene Expression Profile] file for new cell lines.')
    parser.add_argument('--mut_path', type=str, default='./depmap/mu_depmap.csv',
                        help='Path to the [Gene Mutation Profile] file for new cell lines.')
    parser.add_argument('--cnv_path', type=str, default='./depmap/cnv_depmap.csv',
                        help='Path to the [Copy Number Variation Profile] file for new cell lines.')
    parser.add_argument('--new_drugs_csv', type=str, default='./depmap/drug_results.csv', help='Path to the CSV file for new drugs.')
    parser.add_argument('--output_csv', type=str, default='./prediction_output_test.csv',
                        help="[Output file base name]. The final filenames will be 'basename_part_N.csv'.")
    parser.add_argument('--drug_batch_size', type=int, default=50000,
                        help='Number of drugs to process per batch, which is also the number of drugs per output file.')
    parser.add_argument('--cuda_id', type=int, default=0, help='GPU ID to use (-1 for CPU).')
    args = parser.parse_args()

    # --- Main logic ---
    cfg = get_cfg_defaults()
    device = torch.device(f'cuda:{args.cuda_id}' if args.cuda_id >= 0 and torch.cuda.is_available() else 'cpu')
    print(f"--- Using device: {device} ---")

    fp_generator = DrugFingerprintGenerator()

    exp_df, mut_df, cnv_df, exp_dim, mut_dim, cnv_dim = load_and_align_multiple_cell_features(
        cfg, args.exp_path, args.mut_path, args.cnv_path
    )

    print("\n--- Step 3: Loading and preprocessing new drug data ---")
    try:
        drugs_to_predict_df = pd.read_csv(args.new_drugs_csv, header=None, names=['DrugName', 'SMILES'])
        initial_count = len(drugs_to_predict_df)
        print(f"  - Initially loaded {initial_count} rows from {args.new_drugs_csv}.")
        drugs_to_predict_df.dropna(subset=['SMILES'], inplace=True)
        drugs_to_predict_df = drugs_to_predict_df[
            drugs_to_predict_df['SMILES'].apply(lambda x: isinstance(x, str) and len(x.strip()) > 0)]
        invalid_smiles_mask = drugs_to_predict_df['SMILES'].str.startswith('Error')
        drugs_to_predict_df = drugs_to_predict_df[~invalid_smiles_mask]
        filtered_count = len(drugs_to_predict_df)
        if filtered_count < initial_count:
            print(f"  - Note: Initially filtered out {initial_count - filtered_count} rows with null or obviously incorrect SMILES text.")
        if drugs_to_predict_df.empty:
            print(f"Error: No potentially valid drug entries found in {args.new_drugs_csv}.")
            exit(1)
        print(f"  - After initial filtering, {filtered_count} drugs remain for the detailed validation phase.")
    except FileNotFoundError:
        print(f"Error: Drug file not found: {args.new_drugs_csv}")
        exit(1)

    print("\n--- Step 4: Loading pre-trained model ---")
    model = BANDRP(cell_exp_dim=exp_dim, cell_mut_dim=mut_dim, cell_cnv_dim=cnv_dim, **cfg).to(device)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"  - Model state successfully loaded from {args.model_path}.")
    except Exception as e:
        print(f"Error: Failed to load model state: {e}")
        exit(1)

    # Call the prediction function, which now returns the number of processed drugs and a list of generated files
    num_processed_drugs, generated_files = predict_matrix(
        model,
        device,
        (exp_df, mut_df, cnv_df),
        drugs_to_predict_df,
        fp_generator,
        args.output_csv,
        args.drug_batch_size
    )

    # Final output message
    if num_processed_drugs > 0:
        print(f"\n--- All predictions complete! ---")
        print(f"Cross-predictions for {len(exp_df)} cell lines and {num_processed_drugs} valid drugs have been saved in batches.")
        print(f"A total of {len(generated_files)} files were generated:")
        for f_path in generated_files:
            print(f"  - {os.path.abspath(f_path)}")
    else:
        print("\n--- No prediction results were generated. Please check the input files and any warnings/errors during execution. ---")