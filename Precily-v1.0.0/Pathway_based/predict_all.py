import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import csv
import re
from tqdm import tqdm

# ==============================================================================
# --- SMILES Processing Functions from wordextract.py and cmethods.py ---
# (These functions are directly from the "predict new drug" script and remain unchanged)
# ==============================================================================
_WORDEXTRACT_LETTERS = ["D", "E", "J", "R", "L", "M", "T", "Z", "X", "d", "e", "j", "r", "m", "t", "z", "x"]
_WORDEXTRACT_ELEMENTS = None


def _load_elements_once(elements_file_path="../mydata/utils/elements.txt"):
    """Loads the list of chemical elements once and caches it."""
    global _WORDEXTRACT_ELEMENTS
    if _WORDEXTRACT_ELEMENTS is None:
        try:
            with open(elements_file_path) as f:
                _WORDEXTRACT_ELEMENTS = f.read().splitlines()
            if not _WORDEXTRACT_ELEMENTS:
                print(f"Warning: The element list loaded from {elements_file_path} is empty.")
        except FileNotFoundError:
            print(f"Error: Element file '{elements_file_path}' not found. SMILES modification functionality may be affected.")
            _WORDEXTRACT_ELEMENTS = []
        except Exception as e:
            print(f"An error occurred while loading the element file '{elements_file_path}': {e}")
            _WORDEXTRACT_ELEMENTS = []
    return _WORDEXTRACT_ELEMENTS


def _modify_smiles_internal(smiles_str, elements_list, letters_list):
    """Internal function to replace chemical elements in a SMILES string with temporary letters."""
    replacements = {}
    current_smiles = str(smiles_str)
    matched_count = 0
    for el in elements_list:
        if not el: continue
        if el in current_smiles:
            if matched_count < len(letters_list):
                replacement_char = letters_list[matched_count]
                current_smiles = current_smiles.replace(el, replacement_char)
                replacements[matched_count] = el + "," + replacement_char
                matched_count += 1
            else:
                break
    return replacements, current_smiles


def _contains_from_list_internal(smi_text, check_list):
    """Checks if any item from a list is present in a string."""
    for item in check_list:
        if item in smi_text:
            return True
    return False


def _create_lingos_internal(smiles_str, q_val, elements_list, letters_list):
    """Internal function to create LINGOs (substrings) from a SMILES string."""
    lingo_list_internal = []
    current_smiles = str(smiles_str)
    if not current_smiles:
        current_smiles = "_" * q_val
    if len(current_smiles) < q_val:
        current_smiles = current_smiles + "_" * (q_val - len(current_smiles))
    reps, upsmi = _modify_smiles_internal(current_smiles, elements_list, letters_list)
    if len(upsmi) >= q_val:
        for index in range(len(upsmi) - (q_val - 1)):
            lingo = upsmi[index: index + q_val]
            if _contains_from_list_internal(lingo, letters_list):
                temp_lingo = str(lingo)
                for rep_idx_key in reps:
                    original_el, replacement_char = reps[rep_idx_key].split(",")
                    temp_lingo = temp_lingo.replace(replacement_char, original_el)
                lingo_list_internal.append(temp_lingo)
            else:
                lingo_list_internal.append(lingo)
    if not lingo_list_internal:
        final_lingo_for_empty_case = upsmi
        if _contains_from_list_internal(final_lingo_for_empty_case, letters_list):
            for rep_idx_key in reps:
                original_el, replacement_char = reps[rep_idx_key].split(",")
                final_lingo_for_empty_case = final_lingo_for_empty_case.replace(replacement_char, original_el)
        lingo_list_internal.append(final_lingo_for_empty_case)
    return lingo_list_internal


def _vector_add_internal(lingo_embeddings, lingo_list_from_smiles):
    """Internal function to sum the embedding vectors of LINGOs."""
    if not lingo_embeddings:
        return []
    first_key = next(iter(lingo_embeddings), None)
    if first_key is None:
        return []
    vsize = len(lingo_embeddings[first_key])
    sum_vec = [float(0) for _ in range(vsize)]
    for lingo in lingo_list_from_smiles:
        lingo_vec_float = [float(0) for _ in range(vsize)]
        if lingo in lingo_embeddings:
            lingo_embedding = lingo_embeddings[lingo]
            if len(lingo_embedding) == vsize:
                lingo_vec_float = [float(val) for val in lingo_embedding]
        sum_vec = [sum_val + lvf_val for sum_val, lvf_val in zip(sum_vec, lingo_vec_float)]
    return sum_vec


def _vector_add_avg_internal(lingo_embeddings, lingo_list_from_smiles):
    """Calculates the average embedding vector for a list of LINGOs."""
    sum_vec = _vector_add_internal(lingo_embeddings, lingo_list_from_smiles)
    num_lingos = len(lingo_list_from_smiles)
    if num_lingos == 0:
        if not lingo_embeddings: return []
        first_key = next(iter(lingo_embeddings), None)
        if first_key is None: return []
        vsize = len(lingo_embeddings[first_key])
        return [float(0) for _ in range(vsize)]
    if not sum_vec:
        return []
    avg_vec = [val / num_lingos for val in sum_vec]
    return avg_vec


# ==============================================================================
# --- Main Script Constants and Functions ---
# ==============================================================================
N_DRUG_FEATURES = 100
N_CELL_LINE_FEATURES = 1329
DRUG_EMBEDDING_FILE_PATH = '../mydata/utils/drug.pubchem.canon.l8.ws20.txt'
ELEMENTS_FILE_PATH = '../mydata/utils/elements.txt'
# This file is now used as a "template" to get the standard feature column names
CANONICAL_CELL_FEATURES_TEMPLATE_PATH = '../mydata/mycell_gsva2.csv'

_DRUG_EMBEDDINGS_INDEX = None
_DRUG_EMBEDDING_VSIZE = None
_CANONICAL_CELL_FEATURE_NAMES = None


def _load_drug_embeddings_once(embedding_file_path):
    """Loads the drug SMILES word embeddings once and caches them."""
    global _DRUG_EMBEDDINGS_INDEX, _DRUG_EMBEDDING_VSIZE
    if _DRUG_EMBEDDINGS_INDEX is None:
        print(f"Loading drug SMILES word embedding file: {embedding_file_path} ...")
        embeddings_index = {}
        vsize = 0
        try:
            with open(os.path.join(embedding_file_path)) as f:
                total_lines = None
                try:
                    header = next(f).split()
                    if len(header) == 2 and header[0].isdigit() and header[1].isdigit():
                        total_lines = int(header[0])
                    else:
                        values = header
                        word = values[0]
                        coefs = np.asarray(values[1:], dtype='float32')
                        embeddings_index[word] = coefs
                        if vsize == 0: vsize = len(coefs)
                except StopIteration:
                    pass

                for line in tqdm(f, total=total_lines, desc="  Loading word embeddings", unit=" vecs"):
                    values = line.split()
                    if not values: continue
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype='float32')
                    if vsize == 0:
                        vsize = len(coefs)
                    elif len(coefs) != vsize and vsize > 0:
                        continue
                    embeddings_index[word] = coefs

            _DRUG_EMBEDDINGS_INDEX = embeddings_index
            _DRUG_EMBEDDING_VSIZE = vsize
            if not _DRUG_EMBEDDINGS_INDEX:
                raise SystemExit(f"Error: Failed to load any valid word embeddings from {embedding_file_path}.")
            if _DRUG_EMBEDDING_VSIZE != N_DRUG_FEATURES:
                print(
                    f"CRITICAL WARNING: The vector dimension loaded from the embedding file ({_DRUG_EMBEDDING_VSIZE}) does not match the expected N_DRUG_FEATURES ({N_DRUG_FEATURES}).")
            print(
                f"Drug SMILES word embeddings loaded. Effective vocabulary size: {len(_DRUG_EMBEDDINGS_INDEX)}, actual vector dimension: {_DRUG_EMBEDDING_VSIZE}")
        except FileNotFoundError:
            raise SystemExit(f"Error: Drug SMILES word embedding file not found at '{embedding_file_path}'.")
        except Exception as e:
            raise SystemExit(f"An error occurred while loading the drug SMILES word embedding file: {e}")
    return _DRUG_EMBEDDINGS_INDEX, _DRUG_EMBEDDING_VSIZE


def generate_drug_features_from_smiles(smiles_str: str, q_val: int = 8) -> np.ndarray:
    """Generates a drug feature vector from a SMILES string using the LINGO method."""
    global _WORDEXTRACT_ELEMENTS, _WORDEXTRACT_LETTERS
    elements = _load_elements_once(ELEMENTS_FILE_PATH)
    embeddings, vec_size = _load_drug_embeddings_once(DRUG_EMBEDDING_FILE_PATH)
    if embeddings is None or not elements:
        return np.zeros(N_DRUG_FEATURES)
    lingo_list = _create_lingos_internal(smiles_str, q_val, elements, _WORDEXTRACT_LETTERS)
    smiles_vec = _vector_add_avg_internal(embeddings, lingo_list)
    if not smiles_vec:
        return np.zeros(N_DRUG_FEATURES)
    if len(smiles_vec) != N_DRUG_FEATURES:
        final_vec = np.zeros(N_DRUG_FEATURES)
        if len(smiles_vec) > N_DRUG_FEATURES:
            final_vec = np.array(smiles_vec[:N_DRUG_FEATURES], dtype=float)
        else:
            final_vec[:len(smiles_vec)] = smiles_vec
        return final_vec
    return np.array(smiles_vec, dtype=float)


def _load_canonical_cell_feature_names_once(template_file_path: str):
    """Loads the canonical cell line feature column names once from a template file."""
    global _CANONICAL_CELL_FEATURE_NAMES
    if _CANONICAL_CELL_FEATURE_NAMES is None:
        print(f"Loading canonical cell line feature names from template: {template_file_path}")
        try:
            # Only need to read the column names, not the data
            df_template = pd.read_csv(template_file_path, index_col=0, nrows=0)
            _CANONICAL_CELL_FEATURE_NAMES = df_template.columns.tolist()

            if not _CANONICAL_CELL_FEATURE_NAMES:
                raise SystemExit(f"Error: Failed to load any feature names from template file '{template_file_path}'.")
            if len(_CANONICAL_CELL_FEATURE_NAMES) != N_CELL_LINE_FEATURES:
                raise SystemExit(
                    f"Error: The number of features in the template file ({len(_CANONICAL_CELL_FEATURE_NAMES)}) does not match the preset N_CELL_LINE_FEATURES ({N_CELL_LINE_FEATURES}).")
            print(f"Successfully loaded {len(_CANONICAL_CELL_FEATURE_NAMES)} canonical cell line feature names.")
        except FileNotFoundError:
            raise SystemExit(f"Error: Cell line feature template file not found at '{template_file_path}'.")
        except Exception as e:
            raise SystemExit(f"An error occurred while reading the cell line feature template file: {e}")
    return _CANONICAL_CELL_FEATURE_NAMES


# (This function comes from the "predict new cell line" script and is used to align input GSVA data)
def align_gsva_data(new_gsva_df: pd.DataFrame, training_feature_names: list) -> pd.DataFrame:
    """Aligns new GSVA data with the feature columns used during training."""
    print("Starting alignment of input GSVA data...")
    # Create a template DataFrame with training features as columns, new cell lines as rows, filled with 0
    aligned_df = pd.DataFrame(0.0, index=new_gsva_df.index, columns=training_feature_names)

    # Find common columns between the new and old data
    common_cols = list(set(new_gsva_df.columns) & set(training_feature_names))

    if common_cols:
        print(f"  - Found {len(common_cols)} common feature columns.")
        aligned_df[common_cols] = new_gsva_df[common_cols]
    else:
        print("  - Warning: Input GSVA data has no common columns with training features. All cell line features will be zero.")

    missing_cols_count = len(training_feature_names) - len(common_cols)
    if missing_cols_count > 0:
        print(f"  - {missing_cols_count} features used in training were missing from the input file and will be filled with 0.")

    return aligned_df


# (This function exists in both scripts with consistent logic, ensuring the correct concatenation order)
def combine_features(cell_feats: np.ndarray, drug_feats: np.ndarray) -> np.ndarray:
    """
    Strictly concatenates features in the order [cell line features, drug features].
    This is the order used during model training.
    """
    if drug_feats is None or cell_feats is None: return None
    combined = np.concatenate((cell_feats, drug_feats))
    if len(combined) != (N_DRUG_FEATURES + N_CELL_LINE_FEATURES):
        return None
    return combined


def load_keras_models(models_base_path: str, num_models: int = 5) -> list:
    """Loads an ensemble of Keras models from a directory."""
    loaded_models = []
    print(f"Loading {num_models} models from '{models_base_path}'...")
    for i in tqdm(range(1, num_models + 1), desc="Loading models", unit="model"):
        model_filename = f'precily_cv_{i}.hdf5'
        model_path = os.path.join(models_base_path, model_filename)
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            loaded_models.append(model)
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
    if not loaded_models:
        print("Critical Error: No models were loaded.")
    elif len(loaded_models) != num_models:
        print(f"Warning: Expected to load {num_models} models, but actually loaded {len(loaded_models)}.")
    return loaded_models


def predict_with_ensemble(models_list: list, features_array: np.ndarray) -> tuple:
    """Performs prediction using an ensemble of models and returns the mean and individual predictions."""
    if not models_list or features_array.shape[0] == 0: return np.array([]), np.array([])
    all_predictions_list = []
    for model in tqdm(models_list, desc="Ensemble Prediction", unit="model"):
        try:
            preds = model.predict(features_array, verbose=0)
            all_predictions_list.append(preds.flatten())
        except Exception as e:
            print(f"An error occurred during model prediction: {e}")
            all_predictions_list.append(np.full(features_array.shape[0], np.nan))
    if not all_predictions_list: return np.array([]), np.array([])
    individual_predictions_array = np.array(all_predictions_list)
    mean_predictions = np.nanmean(individual_predictions_array, axis=0)
    return mean_predictions, individual_predictions_array


def main():
    """Main function to drive the prediction workflow."""
    global ELEMENTS_FILE_PATH, DRUG_EMBEDDING_FILE_PATH, CANONICAL_CELL_FEATURES_TEMPLATE_PATH
    global _WORDEXTRACT_ELEMENTS, _DRUG_EMBEDDINGS_INDEX, _DRUG_EMBEDDING_VSIZE, _CANONICAL_CELL_FEATURE_NAMES

    parser = argparse.ArgumentParser(description="Predict IC50 values for all combinations of new drugs and new cell lines.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # --- Unified Input Arguments ---
    parser.add_argument('--input_drugs_file', type=str, default='../depmap/drug_results.csv',
                        help='[Required] Path to the input CSV file for new drugs. Format: no header, col1=drug_name, col2=SMILES.')
    parser.add_argument('--input_cell_lines_file', type=str, default='../depmap/gsva_depmap.csv',
                        help='[Required] Path to the input CSV file for new cell line GSVA data. Format: col1=cell_line_name_index, subsequent cols=GSVA pathway scores.')
    parser.add_argument('--models_dir', type=str, default='.',
                        help="Directory where the trained .hdf5 model files are stored.")
    parser.add_argument('--output_file', type=str, default='Precily.csv',
                        help='Path to save the prediction results CSV file.')
    parser.add_argument('--num_models', type=int, default=5,
                        help='Number of cross-validation models to load and ensemble.')
    parser.add_argument('--lingo_q', type=int, default=8,
                        help='The q parameter (SMILES substring length) for the LINGO algorithm.')
    # --- Optional Dependency File Paths ---
    parser.add_argument('--elements_file', type=str, default=None,
                        help=f"Custom element list file path (default: '{ELEMENTS_FILE_PATH}').")
    parser.add_argument('--drug_embedding_file', type=str, default=None,
                        help=f"Drug SMILES word embedding file path (default: '{DRUG_EMBEDDING_FILE_PATH}').")
    parser.add_argument('--cell_features_template_file', type=str, default=None,
                        help=f"Cell line GSVA feature template file path (default: '{CANONICAL_CELL_FEATURES_TEMPLATE_PATH}').")
    args = parser.parse_args()

    # --- Determine final file paths to use ---
    _current_elements_file = args.elements_file if args.elements_file is not None else ELEMENTS_FILE_PATH
    _current_drug_embedding_file = args.drug_embedding_file if args.drug_embedding_file is not None else DRUG_EMBEDDING_FILE_PATH
    _current_cell_template_file = args.cell_features_template_file if args.cell_features_template_file is not None else CANONICAL_CELL_FEATURES_TEMPLATE_PATH

    # --- Update global path variables (if overridden by command line) ---
    if _current_elements_file != ELEMENTS_FILE_PATH:
        ELEMENTS_FILE_PATH = _current_elements_file
        _WORDEXTRACT_ELEMENTS = None
    if _current_drug_embedding_file != DRUG_EMBEDDING_FILE_PATH:
        DRUG_EMBEDDING_FILE_PATH = _current_drug_embedding_file
        _DRUG_EMBEDDINGS_INDEX = None
        _DRUG_EMBEDDING_VSIZE = None
    if _current_cell_template_file != CANONICAL_CELL_FEATURES_TEMPLATE_PATH:
        CANONICAL_CELL_FEATURES_TEMPLATE_PATH = _current_cell_template_file
        _CANONICAL_CELL_FEATURE_NAMES = None

    print("\n" + "=" * 80)
    print("Prediction Script Configuration:")
    print(f"1. Drug Feature Dimension (N_DRUG_FEATURES): {N_DRUG_FEATURES}")
    print(f"2. Cell Line Feature Dimension (N_CELL_LINE_FEATURES): {N_CELL_LINE_FEATURES}")
    print(f"3. Drug SMILES Word Embedding File: '{DRUG_EMBEDDING_FILE_PATH}'")
    print(f"4. Cell Line Feature Template File: '{CANONICAL_CELL_FEATURES_TEMPLATE_PATH}'")
    print(f"5. Element List File: '{ELEMENTS_FILE_PATH}'")
    print(f"6. Feature Concatenation Order: [Cell Line Features, Drug Features]")
    print("=" * 80 + "\n")

    # --- 1. Load all necessary models and data ---
    _load_elements_once(ELEMENTS_FILE_PATH)
    _load_drug_embeddings_once(DRUG_EMBEDDING_FILE_PATH)
    canonical_cell_feature_names = _load_canonical_cell_feature_names_once(CANONICAL_CELL_FEATURES_TEMPLATE_PATH)
    models = load_keras_models(args.models_dir, args.num_models)

    if not models:
        raise SystemExit("No models were loaded. Cannot proceed with prediction. Exiting.")

    # --- 2. Load and process input drug and cell line data ---
    try:
        new_drugs_df = pd.read_csv(args.input_drugs_file, header=None, names=['drug_name', 'smiles_string'])
        if new_drugs_df.empty: raise SystemExit(f"The input drug file '{args.input_drugs_file}' is empty.")
        print(f"Successfully loaded {len(new_drugs_df)} new drugs from '{args.input_drugs_file}'.")
    except FileNotFoundError:
        raise SystemExit(f"Error: Input drug file not found at '{args.input_drugs_file}'.")
    except Exception as e:
        raise SystemExit(f"An error occurred while reading the drug file '{args.input_drugs_file}': {e}.")

    try:
        new_gsva_df = pd.read_csv(args.input_cell_lines_file, index_col=0)
        if new_gsva_df.empty: raise SystemExit(f"The input cell line file '{args.input_cell_lines_file}' is empty.")
        print(f"Successfully loaded {len(new_gsva_df)} new cell lines from '{args.input_cell_lines_file}'.")
        aligned_cell_features_df = align_gsva_data(new_gsva_df, canonical_cell_feature_names)
    except FileNotFoundError:
        raise SystemExit(f"Error: Input cell line file not found at '{args.input_cell_lines_file}'.")
    except Exception as e:
        raise SystemExit(f"An error occurred while reading or processing the new cell line file: {e}")

    # --- 3. Generate feature vectors for all (new drug, new cell line) combinations ---
    feature_vectors_for_prediction = []
    prediction_identifiers = []

    print("\nGenerating feature vectors for all drug-cell line combinations...")
    # Pre-calculate features for all drugs to avoid redundant computations in the inner loop
    drug_feature_cache = {}
    for _, drug_row in tqdm(new_drugs_df.iterrows(), total=len(new_drugs_df), desc="Calculating drug features"):
        drug_feature_cache[drug_row['drug_name']] = {
            'features': generate_drug_features_from_smiles(drug_row['smiles_string'], q_val=args.lingo_q),
            'smiles': drug_row['smiles_string']
        }

    # Iterate through all cell lines and pre-calculated drug features to create combinations
    for cell_line_name, cell_series in tqdm(aligned_cell_features_df.iterrows(), total=len(aligned_cell_features_df),
                                            desc="Combining features"):
        cell_features_np = cell_series.values
        for drug_name, drug_data in drug_feature_cache.items():
            drug_features_np = drug_data['features']

            combined_input_features = combine_features(cell_features_np, drug_features_np)

            if combined_input_features is not None:
                feature_vectors_for_prediction.append(combined_input_features)
                prediction_identifiers.append({
                    'drug_name': drug_name,
                    'smiles': drug_data['smiles'],
                    'cell_line_name': cell_line_name
                })

    if not feature_vectors_for_prediction:
        raise SystemExit("No valid feature vectors were generated for prediction. Please check the input files.")

    # --- 4. Perform batch prediction ---
    X_to_predict_combined = np.array(feature_vectors_for_prediction)
    print(f"\nGenerated {X_to_predict_combined.shape[0]} feature vectors, starting ensemble prediction...")
    mean_predictions, individual_predictions = predict_with_ensemble(models, X_to_predict_combined)

    # --- 5. Organize and save results ---
    results_list = []
    print("\nOrganizing prediction results...")
    for i, identifier in enumerate(tqdm(prediction_identifiers, desc="Formatting results")):
        row_data = {
            'drug_name': identifier['drug_name'],
            'smiles': identifier['smiles'],
            'cell_line_name': identifier['cell_line_name'],
            'predicted_ic50_mean': mean_predictions[i] if mean_predictions.size > i else np.nan
        }
        # Add individual predictions for each model
        for model_idx in range(individual_predictions.shape[0]):
            row_data[f'predicted_ic50_model_{model_idx + 1}'] = individual_predictions[
                model_idx, i] if individual_predictions.size > (
                        model_idx * X_to_predict_combined.shape[0] + i) else np.nan
        results_list.append(row_data)

    results_df = pd.DataFrame(results_list)

    try:
        results_df.to_csv(args.output_file, index=False)
        print(f"\nPrediction results successfully saved to: {args.output_file}")
    except Exception as e:
        print(f"\nError: Failed to save prediction results to '{args.output_file}': {e}")


if __name__ == '__main__':
    # Set TensorFlow threads, which may improve performance and stability in some CPU environments
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    main()