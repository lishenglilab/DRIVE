import pandas as pd
import os
import numpy as np
import argparse
import warnings
import pickle
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

# --- Global Warning Filters ---
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# --- Global Configuration ---
# ==============================================================================
INPUT_DIR = '.'
OUTPUT_DIR = 'ensemble_outputs'
DEFAULT_MODEL_PKL = 'best_model_RandomForest_rmse.pkl'

# --- CORE CONFIG 1: Map raw prediction filenames to concise model names ---
FILENAME_TO_METHOD_MAP = {
    'bandrp_predictions_part_1': 'BANDRP',
    'DeepTTC_predictions': 'DeepTTA',
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
            tqdm.write(f"  - WARNING: No parsing rule for {method_name}, skipping.")
            continue
        
        try:
            df = pd.read_csv(filepath, low_memory=False)
            required_cols = [rule['drug'], rule['cell'], rule['pred']]
            
            if not all(c in df.columns for c in required_cols):
                tqdm.write(f"  - WARNING: File {filepath} missing columns, skipping.")
                continue
                
            temp_df = df[required_cols].copy()
            temp_df.columns = ['drug_name', 'cell_line', 'prediction']
            temp_df['method'] = method_name
            
            temp_df.dropna(inplace=True)
            temp_df['drug_name'] = temp_df['drug_name'].astype(str).str.strip()
            temp_df['cell_line'] = temp_df['cell_line'].astype(str).str.strip()
            
            all_dfs.append(temp_df)
        except Exception as e:
            tqdm.write(f"  - ERROR: Failed to process {filepath}: {e}")

    if not all_dfs:
        print("\nFATAL: No data loaded. Terminating.")
        return None
        
    print(f"\n[STEP 1] Loaded data from {len(all_dfs)} models.")
    full_data = pd.concat(all_dfs, ignore_index=True)
    return full_data

# ==============================================================================
# --- Core Prediction Logic (Used by both modes) ---
# ==============================================================================
def generate_ml_predictions(full_data, model_pkl_path):
    """
    Takes long-format data and returns a DataFrame with ML-based ensemble predictions.
    This is the core prediction engine for both Mode 0 and Mode 1.
    """
    print("\n[STEP 2] Preparing data and running ML model for predictions...")
    
    # 1. Load the pre-trained model and its features
    print(f"  -> Loading pre-trained model '{model_pkl_path}'...")
    try:
        with open(model_pkl_path, 'rb') as f:
            pkl_data = pickle.load(f)
        model = pkl_data['model']
        authoritative_features = [str(f).replace('-', '_') for f in pkl_data['features']]
    except Exception as e:
        print(f"FATAL: Error loading PKL file: {e}")
        return None

    # 2. Reshape data from long to wide format
    print("  -> Reshaping data to model input format...")
    data_for_pivot = full_data[full_data['method'].isin(authoritative_features)]
    preds_wide_df = data_for_pivot.pivot_table(
        index=['cell_line', 'drug_name'],
        columns='method',
        values='prediction'
    )
    
    # 3. Align features and impute missing values
    print("  -> Aligning features and imputing missing values...")
    X_input = preds_wide_df.reindex(columns=authoritative_features, fill_value=np.nan)
    global_medians = X_input.median()
    X_input.fillna(global_medians, inplace=True)
    X_input.fillna(0, inplace=True)

    # 4. Make predictions
    print("  -> Making predictions...")
    try:
        predictions = model.predict(X_input)
        X_input['Ensemble_Score'] = predictions
    except Exception as e:
        print(f"FATAL: Model prediction failed: {e}")
        return None

    final_predictions = X_input[['Ensemble_Score']].reset_index()
    print(f"[STEP 2] ML prediction complete. Generated {len(final_predictions)} scores.")
    return final_predictions

# ==============================================================================
# --- Mode 0: Output a Single Large Prediction File ---
# ==============================================================================
def run_mode_0_global_file(prediction_df, output_filename):
    """
    Executes Mode 0: Saves the ML prediction results to a single large CSV file.
    """
    print("\n--- MODE 0: Saving Global Prediction File ---")
    prediction_df.to_csv(output_filename, index=False, float_format='%.4f')
    print(f"\nSuccess! Global prediction file saved to: '{output_filename}'")

# ==============================================================================
# --- Mode 1: Generate Top-K Reports for Each Cell Line ---
# ==============================================================================
def run_mode_1_top_k_reports(prediction_df, top_k):
    """
    Executes Mode 1: Generates a separate top-K report for each cell line
    based on the ML model's predictions.
    """
    print(f"\n--- MODE 1: Generating Top-{top_k} Reports for Each Cell Line ---")
    
    reports_output_dir = os.path.join(OUTPUT_DIR, 'cell_line_reports')
    os.makedirs(reports_output_dir, exist_ok=True)
    
    print(f"  -> Grouping by cell line to generate {len(prediction_df['cell_line'].unique())} reports...")
    
    grouped = prediction_df.groupby('cell_line')
    for cell_name, cell_df in tqdm(grouped, desc="Generating reports"):
        
        # Sort by the ensemble score (lower is better) and get the top K
        top_k_drugs = cell_df.sort_values('Ensemble_Score', ascending=True).head(top_k)
        
        if top_k_drugs.empty:
            continue
            
        # Add a rank column
        top_k_drugs.insert(0, 'Rank', range(1, len(top_k_drugs) + 1))
        
        # Save the report
        safe_cell_name = "".join(c if c.isalnum() else "_" for c in cell_name)
        output_path = os.path.join(reports_output_dir, f'top{top_k}_{safe_cell_name}_report.csv')
        top_k_drugs[['Rank', 'drug_name', 'Ensemble_Score']].to_csv(output_path, index=False, float_format='%.4f')
        
    print(f"\nSuccess! Reports saved to: '{reports_output_dir}'")

# ==============================================================================
# --- Script Execution Entry Point ---
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
A script to perform ensemble analysis using a pre-trained RandomForest model.

It supports two distinct output modes:
  Mode 0: Generates a single, large global prediction file with all scores.
  Mode 1: Generates a separate, concise Top-K sensitive drug report for each cell line.
""",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument('--mode', type=int, required=True, choices=[0, 1],
                        help='[REQUIRED] Select the operating mode.\n'
                             '  0: Global Prediction File.\n'
                             '  1: Top-K Reports per Cell Line.')
    
    parser.add_argument('--model_pkl', type=str, default=DEFAULT_MODEL_PKL,
                        help=f'Path to the pre-trained RandomForest model .pkl file (default: {DEFAULT_MODEL_PKL})')

    parser.add_argument('--top_k', type=int, default=5,
                        help='[Mode 1 only] The number of top-ranked drugs to include in each report (default: 5)')

    args = parser.parse_args()
    
    # Step 1: Load and standardize raw data from all models
    master_df = load_and_preprocess_all_data(INPUT_DIR, FILENAME_TO_METHOD_MAP, PARSING_RULES)
    
    if master_df is not None:
        # Step 2: Generate predictions using the ML model (core logic for both modes)
        ml_predictions = generate_ml_predictions(master_df, args.model_pkl)
        
        if ml_predictions is not None:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            # Step 3: Execute the selected output mode
            if args.mode == 0:
                run_mode_0_global_file(
                    prediction_df=ml_predictions,
                    output_filename=os.path.join(OUTPUT_DIR, 'ml_ensemble_predictions.csv')
                )
            elif args.mode == 1:
                run_mode_1_top_k_reports(
                    prediction_df=ml_predictions,
                    top_k=args.top_k
                )
        
    print("\nAll tasks completed!")