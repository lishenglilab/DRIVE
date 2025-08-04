import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import csv
import re
from tqdm import tqdm

# ==============================================================================
# --- 从 wordextract.py 和 cmethods.py 导入/定义的SMILES处理函数 ---
# (这些函数直接来自“预测新药”的代码，保持不变)
# ==============================================================================
_WORDEXTRACT_LETTERS = ["D", "E", "J", "R", "L", "M", "T", "Z", "X", "d", "e", "j", "r", "m", "t", "z", "x"]
_WORDEXTRACT_ELEMENTS = None


def _load_elements_once(elements_file_path="../mydata/utils/elements.txt"):
    global _WORDEXTRACT_ELEMENTS
    if _WORDEXTRACT_ELEMENTS is None:
        try:
            with open(elements_file_path) as f:
                _WORDEXTRACT_ELEMENTS = f.read().splitlines()
            if not _WORDEXTRACT_ELEMENTS:
                print(f"警告：从 {elements_file_path} 加载的元素列表为空。")
        except FileNotFoundError:
            print(f"错误：未找到元素文件 '{elements_file_path}'。SMILES修改功能可能受影响。")
            _WORDEXTRACT_ELEMENTS = []
        except Exception as e:
            print(f"加载元素文件 '{elements_file_path}' 时发生错误: {e}")
            _WORDEXTRACT_ELEMENTS = []
    return _WORDEXTRACT_ELEMENTS


def _modify_smiles_internal(smiles_str, elements_list, letters_list):
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
    for item in check_list:
        if item in smi_text:
            return True
    return False


def _create_lingos_internal(smiles_str, q_val, elements_list, letters_list):
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
# --- 主要脚本常量和函数 ---
# ==============================================================================
N_DRUG_FEATURES = 100
N_CELL_LINE_FEATURES = 1329
DRUG_EMBEDDING_FILE_PATH = '../mydata/utils/drug.pubchem.canon.l8.ws20.txt'
ELEMENTS_FILE_PATH = '../mydata/utils/elements.txt'
# 此文件现在用作获取标准特征列名的“模板”
CANONICAL_CELL_FEATURES_TEMPLATE_PATH = '../mydata/mycell_gsva2.csv'

_DRUG_EMBEDDINGS_INDEX = None
_DRUG_EMBEDDING_VSIZE = None
_CANONICAL_CELL_FEATURE_NAMES = None


def _load_drug_embeddings_once(embedding_file_path):
    global _DRUG_EMBEDDINGS_INDEX, _DRUG_EMBEDDING_VSIZE
    if _DRUG_EMBEDDINGS_INDEX is None:
        print(f"正在加载药物SMILES词嵌入文件: {embedding_file_path} ...")
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

                for line in tqdm(f, total=total_lines, desc="  加载词嵌入", unit=" vecs"):
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
                raise SystemExit(f"错误：未能从 {embedding_file_path} 加载任何有效的词嵌入。")
            if _DRUG_EMBEDDING_VSIZE != N_DRUG_FEATURES:
                print(
                    f"严重警告：从嵌入文件加载的向量维度 ({_DRUG_EMBEDDING_VSIZE}) 与预期的 N_DRUG_FEATURES ({N_DRUG_FEATURES}) 不匹配。")
            print(
                f"药物SMILES词嵌入加载完成。有效词汇量: {len(_DRUG_EMBEDDINGS_INDEX)}, 实际向量维度: {_DRUG_EMBEDDING_VSIZE}")
        except FileNotFoundError:
            raise SystemExit(f"错误：未找到药物SMILES词嵌入文件 '{embedding_file_path}'。")
        except Exception as e:
            raise SystemExit(f"加载药物SMILES词嵌入文件时发生错误: {e}")
    return _DRUG_EMBEDDINGS_INDEX, _DRUG_EMBEDDING_VSIZE


def generate_drug_features_from_smiles(smiles_str: str, q_val: int = 8) -> np.ndarray:
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
    """从模板文件加载一次标准的细胞系特征列名。"""
    global _CANONICAL_CELL_FEATURE_NAMES
    if _CANONICAL_CELL_FEATURE_NAMES is None:
        print(f"正在从模板文件加载标准的细胞系特征名称: {template_file_path}")
        try:
            # 只需读取列名，不需要数据
            df_template = pd.read_csv(template_file_path, index_col=0, nrows=0)
            _CANONICAL_CELL_FEATURE_NAMES = df_template.columns.tolist()

            if not _CANONICAL_CELL_FEATURE_NAMES:
                raise SystemExit(f"错误：从模板文件'{template_file_path}'中未能加载任何特征名称。")
            if len(_CANONICAL_CELL_FEATURE_NAMES) != N_CELL_LINE_FEATURES:
                raise SystemExit(
                    f"错误：模板文件的特征数量 ({len(_CANONICAL_CELL_FEATURE_NAMES)}) 与预设值 N_CELL_LINE_FEATURES ({N_CELL_LINE_FEATURES}) 不匹配。")
            print(f"成功加载 {len(_CANONICAL_CELL_FEATURE_NAMES)} 个标准的细胞系特征名称。")
        except FileNotFoundError:
            raise SystemExit(f"错误：未找到细胞系特征模板文件 '{template_file_path}'。")
        except Exception as e:
            raise SystemExit(f"读取细胞系特征模板文件时发生错误: {e}")
    return _CANONICAL_CELL_FEATURE_NAMES


# (此函数来自“预测新细胞系”的代码，用于对齐输入的GSVA数据)
def align_gsva_data(new_gsva_df: pd.DataFrame, training_feature_names: list) -> pd.DataFrame:
    """将新的GSVA数据与训练时使用的特征列对齐。"""
    print("开始对齐输入的GSVA数据...")
    # 创建一个以训练特征为列、新细胞系为行，并用0填充的DataFrame作为模板
    aligned_df = pd.DataFrame(0.0, index=new_gsva_df.index, columns=training_feature_names)

    # 找出新旧数据中共有的列
    common_cols = list(set(new_gsva_df.columns) & set(training_feature_names))

    if common_cols:
        print(f"  - 找到 {len(common_cols)} 个共有的特征列。")
        aligned_df[common_cols] = new_gsva_df[common_cols]
    else:
        print("  - 警告: 输入的GSVA数据与训练特征没有共同列。所有细胞系特征将为0。")

    missing_cols_count = len(training_feature_names) - len(common_cols)
    if missing_cols_count > 0:
        print(f"  - {missing_cols_count} 个在训练中使用的特征在输入文件中缺失，将用0填充。")

    return aligned_df


# (此函数在两个脚本中都存在，逻辑一致，确保拼接顺序正确)
def combine_features(cell_feats: np.ndarray, drug_feats: np.ndarray) -> np.ndarray:
    """
    严格按照 [细胞系特征, 药物特征] 的顺序拼接。
    这是模型训练时使用的顺序。
    """
    if drug_feats is None or cell_feats is None: return None
    combined = np.concatenate((cell_feats, drug_feats))
    if len(combined) != (N_DRUG_FEATURES + N_CELL_LINE_FEATURES):
        return None
    return combined


def load_keras_models(models_base_path: str, num_models: int = 5) -> list:
    loaded_models = []
    print(f"正在从 '{models_base_path}' 加载 {num_models} 个模型...")
    for i in tqdm(range(1, num_models + 1), desc="加载模型", unit="model"):
        model_filename = f'precily_cv_{i}.hdf5'
        model_path = os.path.join(models_base_path, model_filename)
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            loaded_models.append(model)
        except Exception as e:
            print(f"加载模型 {model_path} 时出错: {e}")
    if not loaded_models:
        print("关键错误：没有模型被加载。")
    elif len(loaded_models) != num_models:
        print(f"警告：期望加载 {num_models} 个模型, 但实际加载了 {len(loaded_models)} 个。")
    return loaded_models


def predict_with_ensemble(models_list: list, features_array: np.ndarray) -> tuple:
    if not models_list or features_array.shape[0] == 0: return np.array([]), np.array([])
    all_predictions_list = []
    for model in tqdm(models_list, desc="集成预测", unit="model"):
        try:
            preds = model.predict(features_array, verbose=0)
            all_predictions_list.append(preds.flatten())
        except Exception as e:
            print(f"模型预测时出错: {e}")
            all_predictions_list.append(np.full(features_array.shape[0], np.nan))
    if not all_predictions_list: return np.array([]), np.array([])
    individual_predictions_array = np.array(all_predictions_list)
    mean_predictions = np.nanmean(individual_predictions_array, axis=0)
    return mean_predictions, individual_predictions_array


def main():
    global ELEMENTS_FILE_PATH, DRUG_EMBEDDING_FILE_PATH, CANONICAL_CELL_FEATURES_TEMPLATE_PATH
    global _WORDEXTRACT_ELEMENTS, _DRUG_EMBEDDINGS_INDEX, _DRUG_EMBEDDING_VSIZE, _CANONICAL_CELL_FEATURE_NAMES

    parser = argparse.ArgumentParser(description="为新的药物和新的细胞系预测所有组合的IC50值。",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # --- 合并后的输入参数 ---
    parser.add_argument('--input_drugs_file', type=str, default='../depmap/drug_results.csv',
                        help='【必需】输入新药CSV文件的路径。格式: 无表头, 第1列=药物名, 第2列=SMILES。')
    parser.add_argument('--input_cell_lines_file', type=str, default='../depmap/gsva_depmap.csv',
                        help='【必需】输入新细胞系GSVA数据的CSV文件路径。格式: 第1列为细胞系名称索引, 后续列为GSVA通路得分。')
    parser.add_argument('--models_dir', type=str, default='.',
                        help="存储训练好的.hdf5模型文件的目录。")
    parser.add_argument('--output_file', type=str, default='Precily.csv',
                        help='保存预测结果的CSV文件路径。')
    parser.add_argument('--num_models', type=int, default=5,
                        help='要加载和集成的交叉验证模型的数量。')
    parser.add_argument('--lingo_q', type=int, default=8,
                        help='LINGO算法中的q参数 (SMILES子串长度)。')
    # --- 可选的依赖文件路径 ---
    parser.add_argument('--elements_file', type=str, default=None,
                        help=f"自定义元素列表文件路径 (默认: '{ELEMENTS_FILE_PATH}')。")
    parser.add_argument('--drug_embedding_file', type=str, default=None,
                        help=f"药物SMILES词嵌入文件路径 (默认: '{DRUG_EMBEDDING_FILE_PATH}')。")
    parser.add_argument('--cell_features_template_file', type=str, default=None,
                        help=f"细胞系GSVA特征模板文件路径 (默认: '{CANONICAL_CELL_FEATURES_TEMPLATE_PATH}')。")
    args = parser.parse_args()

    # --- 确定最终使用的文件路径 ---
    _current_elements_file = args.elements_file if args.elements_file is not None else ELEMENTS_FILE_PATH
    _current_drug_embedding_file = args.drug_embedding_file if args.drug_embedding_file is not None else DRUG_EMBEDDING_FILE_PATH
    _current_cell_template_file = args.cell_features_template_file if args.cell_features_template_file is not None else CANONICAL_CELL_FEATURES_TEMPLATE_PATH

    # --- 更新全局路径变量 (如果被命令行覆盖) ---
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
    print("预测脚本配置:")
    print(f"1. 药物特征维度 (N_DRUG_FEATURES): {N_DRUG_FEATURES}")
    print(f"2. 细胞系特征维度 (N_CELL_LINE_FEATURES): {N_CELL_LINE_FEATURES}")
    print(f"3. 药物SMILES词嵌入文件: '{DRUG_EMBEDDING_FILE_PATH}'")
    print(f"4. 细胞系特征模板文件: '{CANONICAL_CELL_FEATURES_TEMPLATE_PATH}'")
    print(f"5. 元素列表文件: '{ELEMENTS_FILE_PATH}'")
    print(f"6. 特征拼接顺序: [细胞系特征, 药物特征]")
    print("=" * 80 + "\n")

    # --- 1. 加载所有必要的模型和数据 ---
    _load_elements_once(ELEMENTS_FILE_PATH)
    _load_drug_embeddings_once(DRUG_EMBEDDING_FILE_PATH)
    canonical_cell_feature_names = _load_canonical_cell_feature_names_once(CANONICAL_CELL_FEATURES_TEMPLATE_PATH)
    models = load_keras_models(args.models_dir, args.num_models)

    if not models:
        raise SystemExit("没有模型被加载。无法进行预测。程序退出。")

    # --- 2. 加载和处理输入的药物和细胞系数据 ---
    try:
        new_drugs_df = pd.read_csv(args.input_drugs_file, header=None, names=['drug_name', 'smiles_string'])
        if new_drugs_df.empty: raise SystemExit(f"输入的药物文件 '{args.input_drugs_file}' 为空。")
        print(f"成功从 '{args.input_drugs_file}' 加载 {len(new_drugs_df)} 个新药。")
    except FileNotFoundError:
        raise SystemExit(f"错误：未找到输入的药物文件 '{args.input_drugs_file}'。")
    except Exception as e:
        raise SystemExit(f"读取药物文件 '{args.input_drugs_file}' 时发生错误: {e}。")

    try:
        new_gsva_df = pd.read_csv(args.input_cell_lines_file, index_col=0)
        if new_gsva_df.empty: raise SystemExit(f"输入的细胞系文件 '{args.input_cell_lines_file}' 为空。")
        print(f"成功从 '{args.input_cell_lines_file}' 加载 {len(new_gsva_df)} 个新细胞系。")
        aligned_cell_features_df = align_gsva_data(new_gsva_df, canonical_cell_feature_names)
    except FileNotFoundError:
        raise SystemExit(f"错误: 找不到输入的细胞系文件 '{args.input_cell_lines_file}'。")
    except Exception as e:
        raise SystemExit(f"读取或处理新细胞系文件时出错: {e}")

    # --- 3. 为所有 (新药, 新细胞系) 组合生成特征向量 ---
    feature_vectors_for_prediction = []
    prediction_identifiers = []

    print("\n正在为所有药物-细胞系组合生成特征...")
    # 预先计算所有药物的特征，避免在内层循环中重复计算
    drug_feature_cache = {}
    for _, drug_row in tqdm(new_drugs_df.iterrows(), total=len(new_drugs_df), desc="计算药物特征"):
        drug_feature_cache[drug_row['drug_name']] = {
            'features': generate_drug_features_from_smiles(drug_row['smiles_string'], q_val=args.lingo_q),
            'smiles': drug_row['smiles_string']
        }

    # 遍历所有细胞系和预计算的药物特征，生成组合
    for cell_line_name, cell_series in tqdm(aligned_cell_features_df.iterrows(), total=len(aligned_cell_features_df),
                                            desc="组合特征"):
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
        raise SystemExit("没有生成任何有效的特征向量用于预测。请检查输入文件。")

    # --- 4. 执行批量预测 ---
    X_to_predict_combined = np.array(feature_vectors_for_prediction)
    print(f"\n已生成 {X_to_predict_combined.shape[0]} 个特征向量，开始进行集成预测...")
    mean_predictions, individual_predictions = predict_with_ensemble(models, X_to_predict_combined)

    # --- 5. 整理并保存结果 ---
    results_list = []
    print("\n正在整理预测结果...")
    for i, identifier in enumerate(tqdm(prediction_identifiers, desc="格式化结果")):
        row_data = {
            'drug_name': identifier['drug_name'],
            'smiles': identifier['smiles'],
            'cell_line_name': identifier['cell_line_name'],
            'predicted_ic50_mean': mean_predictions[i] if mean_predictions.size > i else np.nan
        }
        # 添加每个模型的单独预测值
        for model_idx in range(individual_predictions.shape[0]):
            row_data[f'predicted_ic50_model_{model_idx + 1}'] = individual_predictions[
                model_idx, i] if individual_predictions.size > (
                        model_idx * X_to_predict_combined.shape[0] + i) else np.nan
        results_list.append(row_data)

    results_df = pd.DataFrame(results_list)

    try:
        results_df.to_csv(args.output_file, index=False)
        print(f"\n预测结果已成功保存至: {args.output_file}")
    except Exception as e:
        print(f"\n错误: 保存预测结果至 '{args.output_file}' 时发生: {e}")


if __name__ == '__main__':
    # 设置TensorFlow线程，可能有助于提高在某些CPU环境下的性能和稳定性
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    main()