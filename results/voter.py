import pandas as pd
import os
import numpy as np
import argparse
import warnings
import pickle
from sklearn.ensemble import RandomForestRegressor
import gc
from tqdm import tqdm

# --- Global Warning Filters ---
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# --- Global Configuration ---
# ==============================================================================
INPUT_DIR = '.'  # Assume all raw CSV files are in the current directory
OUTPUT_DIR = 'ensemble_outputs'

# --- Default Filenames ---
DEFAULT_MODEL_PKL = 'best_model_RandomForest_rmse.pkl'
DEFAULT_WEIGHT_FILE = 'weight.csv'

# --- CORE CONFIG 1: Map raw prediction filenames to concise model names ---
FILENAME_TO_METHOD_MAP = {
    'bandrp_predictions_part_1': 'BANDRP',
    'DeepTTC_predictions': 'DeepTTA',  # Note: DeepTTC filename is mapped to the DeepTTA model name
    'DIPK_predictions': 'DIPK',
    'GPDRP_predictions_GAT': 'GPDRP_GAT',
    'GPDRP_predictions_GCN': 'GPDRP_GCN',
    'GraphDRP_predictions_GAT_GCN': 'GraphDRP_GAT_GCN',
    'GraphDRP_predictions_GATNet': 'GraphDRP_GATNet',
    'paccmann_predictions': 'paccmann',
    'Precily_predictions': 'Precily'
}

# --- CORE CONFIG 2: Parsing rules (column names) for each model's prediction file ---
PARSING_RULES = {
    'BANDRP': {'drug': 'DrugName', 'cell': 'CellLineID', 'pred': 'PredictedValue'},
    'DeepTTA': {'drug': 'DrugName', 'cell': 'COSMIC_ID', 'pred': 'Predicted_LN_IC50'},
    'DIPK': {'drug': 'drug_id', 'cell': 'cell_line_name', 'pred': 'predicted_ic50'},
    'paccmann': {'drug': 'drug_name', 'cell': 'cell_line_name', 'pred': 'predicted_value_denormalized'},
    'Precily': {'drug': 'drug_name', 'cell': 'cell_line_name', 'pred': 'predicted_ic50'},
    'GPDRP_GAT': {'drug': 'drug_name', 'cell': 'cell_line_name', 'pred': 'predicted_ic50_original'},
    'GPDRP_GCN': {'drug': 'drug_name', 'cell': 'cell_line_name', 'pred': 'predicted_ic50_original'},
    'GraphDRP_GATNet': {'drug': 'drug_name', 'cell': 'cell_line_name', 'pred': 'IC50_original'},
    'GraphDRP_GAT_GCN': {'drug': 'drug_name', 'cell': 'cell_line_name', 'pred': 'IC50_original'},
}

# ==============================================================================
# --- Data Loading and Preprocessing ---
# ==============================================================================

def load_and_preprocess_all_data(input_dir, file_map, rules):
    """
    Loads all raw CSV prediction files and consolidates them into a single, 
    standardized long-format DataFrame.
    """
    print("\n[STEP 1] Loading and consolidating all raw prediction files...")
    all_dfs = []
    for filename_base, method_name in tqdm(file_map.items(), desc="Loading raw files"):
        filepath = os.path.join(input_dir, f"{filename_base}.csv")
        if not os.path.exists(filepath):
            tqdm.write(f"  - WARNING: File {filepath} not found, skipping.")
            continue
        
        rule = rules.get(method_name)
        if not rule:
            tqdm.write(f"  - WARNING: No parsing rule found for model {method_name}, skipping.")
            continue
        
        try:
            df = pd.read_csv(filepath, low_memory=False)
            required_cols = [rule['drug'], rule['cell'], rule['pred']]
            
            if not all(c in df.columns for c in required_cols):
                tqdm.write(f"  - WARNING: File {filepath} is missing required columns, skipping. Required: {required_cols}")
                continue
                
            # Standardize columns
            temp_df = df[required_cols].copy()
            temp_df.columns = ['drug_name', 'cell_line', 'prediction']
            temp_df['method'] = method_name
            
            # Clean data
            temp_df.dropna(subset=['drug_name', 'cell_line', 'prediction'], inplace=True)
            temp_df['drug_name'] = temp_df['drug_name'].astype(str).str.strip()
            temp_df['cell_line'] = temp_df['cell_line'].astype(str).str.strip()
            
            all_dfs.append(temp_df)
        except Exception as e:
            tqdm.write(f"  - ERROR: Failed to process file {filepath}: {e}")

    if not all_dfs:
        print("\nFATAL: Failed to load any data. Terminating.")
        return None
        
    print(f"\n[STEP 1] Loading complete! Consolidated data from {len(all_dfs)} models.")
    full_data = pd.concat(all_dfs, ignore_index=True)
    print(f"  -> Total of {len(full_data)} prediction records.")
    return full_data

# ==============================================================================
# --- Mode 0: Machine Learning Model Prediction ---
# ==============================================================================

def run_mode_0_ml_prediction(full_data, model_pkl_path, output_filename):
    """
    Executes Mode 0: Generates a single large prediction file using a pre-trained ML model.
    """
    print("\n--- MODE 0: Generating Ensemble Predictions with ML Model ---")
    
    # 1. Load the pre-trained model
    print(f"[STEP 2] Loading pre-trained model '{model_pkl_path}'...")
    try:
        with open(model_pkl_path, 'rb') as f:
            pkl_data = pickle.load(f)
        model = pkl_data['model']
        authoritative_features = [str(f).replace('-', '_') for f in pkl_data['features']]
    except Exception as e:
        print(f"FATAL: Error loading or parsing PKL file: {e}")
        return

    # 2. Reshape data from long to wide format
    print("[STEP 3] Reshaping data to match model input format...")
    # Filter for features the model was trained on
    data_for_pivot = full_data[full_data['method'].isin(authoritative_features)]
    
    # Use pivot_table with a multi-index for uniqueness
    preds_wide_df = data_for_pivot.pivot_table(
        index=['cell_line', 'drug_name'], 
        columns='method', 
        values='prediction'
    )
    
    # 3. Align features and impute missing values
    print("[STEP 4] Aligning features and imputing missing values...")
    X_input = preds_wide_df.reindex(columns=authoritative_features, fill_value=np.nan)
    
    # Impute with global medians
    global_medians = X_input.median()
    X_input.fillna(global_medians, inplace=True)
    X_input.fillna(0, inplace=True) # Fill any remaining NaNs (e.g., if a column was all NaN) with 0

    # 4. Make predictions
    print("[STEP 5] Making predictions...")
    try:
        predictions = model.predict(X_input)
        X_input['Ensemble_Score'] = predictions
    except Exception as e:
        print(f"FATAL: Model prediction failed: {e}")
        return

    # 5. Save the results
    final_predictions = X_input[['Ensemble_Score']].reset_index()
    final_predictions.to_csv(output_filename, index=False)
    print(f"\nSuccess! Global prediction file saved to: '{output_filename}'")
    print(f"  -> Contains {len(final_predictions)} prediction records.")


# ==============================================================================
# --- Mode 1: Weighted Average Report Generation ---
# ==============================================================================
def load_weights(filepath):
    """Loads the weight file for Mode 1."""
    try:
        weights_df = pd.read_csv(filepath)
        weights_df = weights_df.iloc[:, [1, 2]]
        weights_df.columns = ['method', 'rmse']
        return {row['method']: row['rmse'] for _, row in weights_df.iterrows()}
    except Exception as e:
        print(f"ERROR: Failed to load weight file '{filepath}': {e}")
        return None

def run_mode_1_weighted_reports(full_data, weight_file, top_n):
    """
    Executes Mode 1: Generates a separate top-N report for each cell line using a weighted average.
    """
    print("\n--- MODE 1: Generating Individual Reports with Weighted Average ---")
    
    # 1. Load model weights
    print(f"[STEP 2] Loading weight file '{weight_file}'...")
    global_weights = load_weights(weight_file)
    if not global_weights: return

    reports_output_dir = os.path.join(OUTPUT_DIR, 'cell_line_reports')
    os.makedirs(reports_output_dir, exist_ok=True)
    
    print(f"[STEP 3] Grouping by cell line to generate {len(full_data['cell_line'].unique())} reports...")
    
    # 2. Process cell lines one by one using groupby for memory efficiency
    grouped = full_data.groupby('cell_line')
    for cell_name, cell_df_long in tqdm(grouped, desc="Generating reports"):
        
        # --- Core calculation ---
        models_present = cell_df_long['method'].unique()
        local_weights = {m: global_weights[m] for m in models_present if m in global_weights}
        if not local_weights: continue

        inverse_rmse = {m: 1 / (r if r > 1e-9 else 1e-9) for m, r in local_weights.items()}
        sum_inverse_rmse = sum(inverse_rmse.values())
        if sum_inverse_rmse == 0: continue
        
        normalized_weights = {m: inv_r / sum_inverse_rmse for m, inv_r in inverse_rmse.items()}
        
        cell_df_long['norm_weight'] = cell_df_long['method'].map(normalized_weights)
        cell_df_long['score_comp'] = cell_df_long['prediction'] * cell_df_long['norm_weight']
        final_scores = cell_df_long.groupby('drug_name')['score_comp'].sum().reset_index(name='Ensemble_Score')
        
        # --- Report generation ---
        top_n_drugs = final_scores.sort_values('Ensemble_Score', ascending=True).head(top_n)
        if top_n_drugs.empty: continue
        
        top_n_raw_preds = cell_df_long[cell_df_long['drug_name'].isin(top_n_drugs['drug_name'])]
        top_n_wide = top_n_raw_preds.pivot_table(index='drug_name', columns='method', values='prediction')
        
        report_df = pd.merge(top_n_drugs.set_index('drug_name'), top_n_wide, on='drug_name', how='left').reset_index()
        report_df = report_df.sort_values('Ensemble_Score', ascending=True)
        report_df.insert(0, 'Rank', range(1, len(report_df) + 1))
        report_df['cell_line'] = cell_name

        # --- Format and save ---
        safe_cell_name = "".join(c if c.isalnum() else "_" for c in cell_name)
        output_path = os.path.join(reports_output_dir, f'top{top_n}_{safe_cell_name}_report.csv')
        report_df.to_csv(output_path, index=False, float_format='%.4f')
        
    print(f"\nSuccess! Reports saved to: '{reports_output_dir}'")


# ==============================================================================
# --- Script Execution Entry Point ---
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
A script to perform ensemble analysis directly from raw model prediction files.
This version does not require a pre-splitting step and is suitable for small to medium-sized datasets.

It supports two distinct modes of operation:
  Mode 0: Uses a pre-trained ML model to generate a single, large global prediction file.
  Mode 1: Uses a weighted-average method to generate a separate, detailed Top-N report for each cell line.
""",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument('--mode', type=int, required=True, choices=[0, 1],
                        help='[REQUIRED] Select the operating mode.\n'
                             '  0: ML Model Prediction (generates one large file).\n'
                             '  1: Weighted Average Reports (generates a report per cell line).')
    
    # --- Mode 0 Specific Arguments ---
    parser.add_argument('--model_pkl', type=str, default=DEFAULT_MODEL_PKL,
                        help=f'[Mode 0] Path to the pre-trained ML model .pkl file (default: {DEFAULT_MODEL_PKL})')

    # --- Mode 1 Specific Arguments ---
    parser.add_argument('--weight_file', type=str, default=DEFAULT_WEIGHT_FILE,
                        help=f'[Mode 1] Path to the .csv file containing model weights (default: {DEFAULT_WEIGHT_FILE})')
    parser.add_argument('--top_n', type=int, default=30,
                        help='[Mode 1] The number of top-ranked drugs to include in each report (default: 30)')

    args = parser.parse_args()
    
    # Step 1 is common to both modes
    master_df = load_and_preprocess_all_data(INPUT_DIR, FILENAME_TO_METHOD_MAP, PARSING_RULES)
    
    if master_df is not None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        # Execute the selected mode
        if args.mode == 0:
            run_mode_0_ml_prediction(
                full_data=master_df,
                model_pkl_path=args.model_pkl,
                output_filename=os.path.join(OUTPUT_DIR, 'ml_ensemble_predictions.csv')
            )
        elif args.mode == 1:
            run_mode_1_weighted_reports(
                full_data=master_df,
                weight_file=args.weight_file,
                top_n=args.top_n
            )
        
    print("\nAll tasks completed!")