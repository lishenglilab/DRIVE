import pandas as pd
import os
import numpy as np
import argparse

# --- 基础配置 ---
INPUT_DIR = './'
CELL_MAP_FILE = 'cellline2.csv'
WEIGHT_FILE = 'weight.csv'

# --- 【【【 核心修正: 将 DeepTTC 文件名映射到 DeepTTA 方法名 】】】 ---
FILENAME_TO_METHOD_MAP = {
    'bandrp_predictions_part_1': 'BANDRP',
    'DeepAEG_predictions': 'DeepAEG',
    'DeepCDR_predictions': 'DeepCDR',
    'DeepTTC_predictions': 'DeepTTA',  # <--- 修正于此: 文件名是TTC, 但方法名是TTA
    'DIPK_predictions': 'DIPK',
    'GADRP_predictions': 'GADRP',
    'GPDRP_predictions_GAT': 'GPDRP_GAT',
    'GPDRP_predictions_GCN': 'GPDRP_GCN',
    'GPDRP_predictions_GIN': 'GPDRP_GIN',
    'GPDRP_predictions_GINTransformer': 'GPDRP_GINTransformer',
    'GraphDRP_predictions_GATNet': 'GraphDRP_GATNet',
    'GraphDRP_predictions_GAT_GCN': 'GraphDRP_GAT_GCN',
    'GraphDRP_predictions_GCNNet': 'GraphDRP_GCNNet',
    'GraphDRP_predictions_GINConvNet': 'GraphDRP_GINConvNet',
    'NeRD_predictions': 'NERD',
    'paccmann_predictions': 'paccmann',
    'Precily_predictions': 'Precily'
}

# --- 解析规则 ---
# 注意：规则的键现在也应该是权重文件中的标准名称 (DeepTTA)
PARSING_RULES = {
    'BANDRP': {'drug': 'DrugName', 'cell': 'CellLineID', 'pred': 'PredictedValue', 'format': 'long'},
    'DeepAEG': {'drug': 'Drug_ID', 'cell': None, 'pred': None, 'format': 'wide'},
    'DeepCDR': {'drug': 'drug_name', 'cell': 'cell_line_id', 'pred': 'predicted_ln_IC50', 'format': 'long'},
    'DeepTTA': {'drug': 'DrugName', 'cell': 'COSMIC_ID', 'pred': 'Predicted_LN_IC50', 'format': 'long'}, # <-- 修正于此
    'DIPK': {'drug': 'resolved_drug_name', 'cell': 'cell_line_name', 'pred': 'predicted_ic50', 'format': 'long'},
    'GADRP': {'drug': 'DrugName', 'cell': 'CellLineName', 'pred': 'Predicted_IC50', 'format': 'long'},
    'GPDRP_GAT': {'drug': 'drug_name', 'cell': 'cell_line_name', 'pred': 'predicted_ic50_original', 'format': 'long'},
    'GPDRP_GCN': {'drug': 'drug_name', 'cell': 'cell_line_name', 'pred': 'predicted_ic50_original', 'format': 'long'},
    'GPDRP_GIN': {'drug': 'drug_name', 'cell': 'cell_line_name', 'pred': 'predicted_ic50_original', 'format': 'long'},
    'GPDRP_GINTransformer': {'drug': 'drug_name', 'cell': 'cell_line_name', 'pred': 'predicted_ic50_original', 'format': 'long'},
    'GraphDRP_GATNet': {'drug': 'drug_name', 'cell': 'cell_line_name', 'pred': 'IC50_original', 'format': 'long'},
    'GraphDRP_GAT_GCN': {'drug': 'drug_name', 'cell': 'cell_line_name', 'pred': 'IC50_original', 'format': 'long'},
    'GraphDRP_GCNNet': {'drug': 'drug_name', 'cell': 'cell_line_name', 'pred': 'IC50_original', 'format': 'long'},
    'GraphDRP_GINConvNet': {'drug': 'drug_name', 'cell': 'cell_line_name', 'pred': 'IC50_original', 'format': 'long'},
    'NERD': {'drug': 'drug_name', 'cell': 'cell_line_name', 'pred': 'predicted_ic50_unscaled', 'format': 'long'},
    'paccmann': {'drug': 'drug_name', 'cell': 'cell_line_name', 'pred': 'predicted_value_denormalized', 'format': 'long'},
    'Precily': {'drug': 'drug_name', 'cell': 'cell_line_name', 'pred': 'predicted_ic50_mean', 'format': 'long'}
}


# --- 辅助函数 (无变化) ---
def create_cell_map(filepath):
    # ... (代码不变)
    try:
        cell_df = pd.read_csv(filepath, index_col=0)
        cell_df.columns = [col.strip() for col in cell_df.columns]
        id_col = 'ID'
        name_cols = ['ID', 'cell.names', 'DepmapModelType', 'cosmic.id', 'cell.names_nerd', 'cell.names.1']
        if id_col not in cell_df.columns: return None
    except: return None
    cell_map = {}
    for _, row in cell_df.iterrows():
        canonical_id = str(row[id_col]).strip()
        if not canonical_id or pd.isna(row[id_col]): continue
        for col in name_cols:
            if col in cell_df.columns and pd.notna(row[col]):
                synonyms = str(row[col]).split('|')
                for syn in synonyms:
                    cleaned_syn = syn.strip().lower()
                    if cleaned_syn: cell_map[cleaned_syn] = canonical_id
    print(f"成功创建细胞系映射，包含 {len(cell_map)} 个条目。")
    return cell_map

def load_weights(filepath):
    # ... (代码不变)
    try:
        weights_df = pd.read_csv(filepath)
    except: return None
    weights_df = weights_df.iloc[:, [1, 2]]
    weights_df.columns = ['method', 'rmse']
    if 'method' not in weights_df.columns or 'rmse' not in weights_df.columns: return None
    weights = {row['method']: row['rmse'] for _, row in weights_df.iterrows()}
    print(f"成功从权重文件加载了 {len(weights)} 个模型的权重。")
    return weights

# --- 主逻辑 (无变化) ---
def run_ensemble(selected_methods: list, predict_lnic50_mode: bool):
    # ... (代码不变)
    full_weights_dict = load_weights(os.path.join(INPUT_DIR, WEIGHT_FILE))
    if not full_weights_dict: return
    model_weights_rmse = {method: rmse for method, rmse in full_weights_dict.items() if method in selected_methods}
    if not model_weights_rmse:
        print("错误：没有模型被选中参与预测。程序终止。")
        return
    if predict_lnic50_mode:
        OUTPUT_FILE = 'final_ensemble_LNIC50.csv'
        print("\n--- 模式: 预测最终 LNIC50 值 ---")
    else:
        OUTPUT_FILE = 'final_ensemble_scores.csv'
        print("\n--- 模式: 计算敏感性排序得分 ---")
    print(f"\n--- 投票系统启动 ---")
    print(f"将使用以下 {len(model_weights_rmse)} 个模型进行投票: {list(model_weights_rmse.keys())}")
    cell_map = create_cell_map(os.path.join(INPUT_DIR, CELL_MAP_FILE))
    if cell_map is None: return
    all_predictions = []
    method_to_filename_map = {v: k for k, v in FILENAME_TO_METHOD_MAP.items()}
    for method_name, rmse in model_weights_rmse.items():
        filename_base = method_to_filename_map.get(method_name)
        if not filename_base:
            print(f"警告: 在文件名映射中找不到方法 '{method_name}' 的对应文件，跳过。")
            continue
        rule = PARSING_RULES.get(method_name)
        if not rule:
            print(f"警告: 在解析规则中找不到方法 '{method_name}' 的规则，跳过。")
            continue
        filename = f"{filename_base}.csv"
        filepath = os.path.join(INPUT_DIR, filename)
        if not os.path.exists(filepath):
            print(f"警告: 预测文件 '{filename}' 不存在，跳过模型 '{method_name}'。")
            continue
        try:
            pred_df = pd.read_csv(filepath)
            if rule['format'] == 'wide':
                id_vars = [rule['drug']]
                value_vars = [col for col in pred_df.columns if col not in id_vars]
                pred_df_long = pd.melt(pred_df, id_vars=id_vars, value_vars=value_vars, var_name='cell_identifier', value_name='prediction')
                pred_df_long.rename(columns={rule['drug']: 'drug_name'}, inplace=True)
            else: 
                if not all(c in pred_df.columns for c in [rule['drug'], rule['cell'], rule['pred']]):
                     print(f"警告: 文件 '{filename}' 缺少必要的列 '{rule['drug']}', '{rule['cell']}', or '{rule['pred']}'，跳过。")
                     continue
                pred_df_long = pred_df[[rule['drug'], rule['cell'], rule['pred']]].copy()
                pred_df_long.columns = ['drug_name', 'cell_identifier', 'prediction']
            pred_df_long['prediction'] = pd.to_numeric(pred_df_long['prediction'], errors='coerce')
            pred_df_long.dropna(subset=['prediction'], inplace=True)
            if pred_df_long.empty: 
                print(f"信息: 文件 '{filename}' 在转换和清洗后没有有效数据。")
                continue
            pred_df_long['cell_identifier_lower'] = pred_df_long['cell_identifier'].astype(str).str.lower().str.strip()
            pred_df_long['cell_id'] = pred_df_long['cell_identifier_lower'].map(cell_map)
            pred_df_long.dropna(subset=['cell_id'], inplace=True)
            if not pred_df_long.empty:
                pred_df_long['drug_name'] = pred_df_long['drug_name'].astype(str).str.strip()
                pred_df_long['method'] = method_name
                all_predictions.append(pred_df_long[['drug_name', 'cell_id', 'prediction', 'method']])
                print(f"成功处理文件 '{filename}'。")
        except Exception as e:
            print(f"处理文件 '{filename}' 时出错: {e}")
    if not all_predictions:
        print("错误: 没有可供合并的预测数据。")
        return
    combined_df = pd.concat(all_predictions, ignore_index=True)
    print(f"\n成功合并所有预测，总共有 {len(combined_df)} 条有效记录。")
    if predict_lnic50_mode:
        print("执行直接加权平均法...")
        inverse_rmse = {method: 1 / rmse for method, rmse in model_weights_rmse.items()}
        sum_inverse_rmse = sum(inverse_rmse.values())
        if sum_inverse_rmse == 0: return
        normalized_weights = {method: inv_rmse / sum_inverse_rmse for method, inv_rmse in inverse_rmse.items()}
        combined_df['normalized_weight'] = combined_df['method'].map(normalized_weights)
        combined_df['weighted_prediction'] = combined_df['prediction'] * combined_df['normalized_weight']
        final_df = combined_df.groupby(['drug_name', 'cell_id'])['weighted_prediction'].sum().reset_index()
        final_pivot_table = final_df.pivot_table(index='drug_name', columns='cell_id', values='weighted_prediction')
    else:
        print("执行 Min-Max 缩放和加权法...")
        drug_min_max = combined_df.groupby('drug_name')['prediction'].agg(['min', 'max']).reset_index()
        combined_df = pd.merge(combined_df, drug_min_max, on='drug_name', how='left')
        denominator = combined_df['max'] - combined_df['min']
        combined_df['normalized_score'] = np.where(denominator > 1e-9, (combined_df['prediction'] - combined_df['min']) / denominator, 0)
        inverse_rmse = {method: 1 / rmse for method, rmse in model_weights_rmse.items()}
        sum_inverse_rmse = sum(inverse_rmse.values())
        if sum_inverse_rmse == 0: return
        normalized_weights = {method: inv_rmse / sum_inverse_rmse for method, inv_rmse in inverse_rmse.items()}
        combined_df['normalized_weight'] = combined_df['method'].map(normalized_weights)
        combined_df['weighted_score'] = combined_df['normalized_score'] * combined_df['normalized_weight']
        final_df = combined_df.groupby(['drug_name', 'cell_id'])['weighted_score'].sum().reset_index()
        final_pivot_table = final_df.pivot_table(index='drug_name', columns='cell_id', values='weighted_score')
    try:
        final_pivot_table.to_csv(OUTPUT_FILE)
        print(f"\n投票系统完成！结果已成功保存到 '{OUTPUT_FILE}'。")
        if not final_pivot_table.empty:
            print(f"输出表格的维度: {final_pivot_table.shape[0]} 行 (药物) x {final_pivot_table.shape[1]} 列 (细胞系)。")
    except Exception as e:
        print(f"保存最终结果时出错: {e}")

# --- 运行主程序 (无变化) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="为每个模型提供独立的开关来控制是否参与集成预测。", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--predict_lnic50', type=int, default=0, choices=[0, 1], help='设置预测模式:\n  0: (默认) 计算用于排序的相对敏感性得分。\n  1: 计算最终的集成 LNIC50 预测值。')
    all_weights = load_weights(os.path.join(INPUT_DIR, WEIGHT_FILE))
    if all_weights:
        all_methods_in_weights = list(all_weights.keys())
        for method in all_methods_in_weights:
            if method in PARSING_RULES:
                parser.add_argument(f'--{method}', type=int, default=1, choices=[0, 1], help=f'设置 {method} 模型是否参与预测 (0=否, 1=是)')
        args = parser.parse_args()
        selected_methods = []
        for method_name in all_methods_in_weights:
             if method_name in PARSING_RULES:
                if hasattr(args, method_name) and getattr(args, method_name) == 1:
                    selected_methods.append(method_name)
        run_ensemble(selected_methods, predict_lnic50_mode=bool(args.predict_lnic50))
    else:
        print("无法加载权重文件，程序终止。")