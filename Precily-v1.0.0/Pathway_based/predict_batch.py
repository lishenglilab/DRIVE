import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import re
from tqdm import tqdm
import gc  # 导入垃圾回收模块
import math  # 导入数学模块用于计算分块数量

# ==============================================================================
# --- 从 wordextract.py 和 cmethods.py 导入/定义的SMILES处理函数 ---
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
                # 尝试读取并解析 Word2Vec/GloVe 的头部信息
                total_lines = None
                try:
                    header = next(f).split()
                    if len(header) == 2 and header[0].isdigit() and header[1].isdigit():
                        total_lines = int(header[0])
                    else:  # 如果没有头部信息，把第一行当作数据处理
                        values = header
                        word = values[0]
                        coefs = np.asarray(values[1:], dtype='float32')
                        embeddings_index[word] = coefs
                        if vsize == 0: vsize = len(coefs)
                except StopIteration:
                    pass  # 文件为空

                # 使用 tqdm 显示加载进度
                for line in tqdm(f, total=total_lines, desc="  加载词嵌入", unit=" vecs"):
                    values = line.split()
                    if not values: continue
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype='float32')
                    if vsize == 0:
                        vsize = len(coefs)
                    elif len(coefs) != vsize and vsize > 0:
                        # 忽略维度不匹配的向量
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
    # 确保返回的向量维度正确
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


# ==============================================================================
# --- 主要执行逻辑 ---
# ==============================================================================
def main():
    global ELEMENTS_FILE_PATH, DRUG_EMBEDDING_FILE_PATH, CANONICAL_CELL_FEATURES_TEMPLATE_PATH
    global _WORDEXTRACT_ELEMENTS, _DRUG_EMBEDDINGS_INDEX, _DRUG_EMBEDDING_VSIZE, _CANONICAL_CELL_FEATURE_NAMES

    parser = argparse.ArgumentParser(description="为新的药物和新的细胞系预测所有组合的IC50值（双重分块处理）。",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # --- 输入参数 ---
    parser.add_argument('--input_drugs_file', type=str, default='../depmap/predict_all_np.csv',
                        help='【必需】输入新药CSV文件的路径。格式: 无表头, 第1列=药物名, 第2列=SMILES。')
    parser.add_argument('--input_cell_lines_file', type=str, default='../depmap/gsva_1.csv',
                        help='【必需】输入新细胞系GSVA数据的CSV文件路径。格式: 第1列为细胞系名称索引, 后续列为GSVA通路得分。')
    parser.add_argument('--model_file', type=str, default='./db/precily_cv_1.hdf5',
                        help="【必需】要使用的单个.hdf5模型文件路径，例如 'precily_cv_1.hdf5'。")
    parser.add_argument('--output_file', type=str, default='./result/Precily_prediction.csv',
                        help='保存预测结果的CSV文件基础路径。最终文件名将是 "基础路径_cell_分块号_drug_分块号.csv"。')
    parser.add_argument('--lingo_q', type=int, default=8,
                        help='LINGO算法中的q参数 (SMILES子串长度)。')
    parser.add_argument('--cell_chunk_size', type=int, default=1,
                        help='每个处理分块中包含的细胞系数量，以控制内存使用。')
    parser.add_argument('--drug_chunk_size', type=int, default=1000,
                        help='每个处理分块中包含的药物数量，以控制内存使用。')
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

    # --- 更新全局路径变量 ---
    ELEMENTS_FILE_PATH = _current_elements_file
    DRUG_EMBEDDING_FILE_PATH = _current_drug_embedding_file
    CANONICAL_CELL_FEATURES_TEMPLATE_PATH = _current_cell_template_file

    print("\n" + "=" * 80)
    print("预测脚本配置 (双重分块模式):")
    print(f"  - 药物文件: '{args.input_drugs_file}'")
    print(f"  - 细胞系文件: '{args.input_cell_lines_file}'")
    print(f"  - 模型文件: '{args.model_file}'")
    print(f"  - 细胞系分块大小: {args.cell_chunk_size}")
    print(f"  - 药物分块大小: {args.drug_chunk_size}")
    print(f"  - 输出文件基础名: '{args.output_file}'")
    print("=" * 80 + "\n")

    # --- 1. 加载所有必要的依赖数据和单个模型 ---
    _load_elements_once(ELEMENTS_FILE_PATH)
    _load_drug_embeddings_once(DRUG_EMBEDDING_FILE_PATH)
    canonical_cell_feature_names = _load_canonical_cell_feature_names_once(CANONICAL_CELL_FEATURES_TEMPLATE_PATH)

    print(f"正在加载指定的模型: {args.model_file}...")
    try:
        model = tf.keras.models.load_model(args.model_file, compile=False)
        print("模型加载成功。")
    except Exception as e:
        raise SystemExit(f"加载模型 {args.model_file} 时出错: {e}。程序退出。")

    # --- 2. 加载输入的药物和细胞系数据 ---
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

    # --- 3. 双重分块循环处理 ---
    num_cell_lines = len(aligned_cell_features_df)
    num_drugs = len(new_drugs_df)
    cell_chunk_size = args.cell_chunk_size
    drug_chunk_size = args.drug_chunk_size

    num_cell_chunks = math.ceil(num_cell_lines / cell_chunk_size)
    num_drug_chunks = math.ceil(num_drugs / drug_chunk_size)

    output_base, output_ext = os.path.splitext(args.output_file)
    if not output_ext: output_ext = ".csv"

    print(f"\n总共有 {num_cell_lines} 个细胞系，将分为 {num_cell_chunks} 个分块。")
    print(f"总共有 {num_drugs} 个药物，将分为 {num_drug_chunks} 个分块。")
    print(f"总计将执行 {num_cell_chunks * num_drug_chunks} 次预测和保存操作。")

    # 外层循环：细胞系分块
    for i in range(num_cell_chunks):
        cell_start_index = i * cell_chunk_size
        cell_end_index = min((i + 1) * cell_chunk_size, num_cell_lines)
        cell_chunk_df = aligned_cell_features_df.iloc[cell_start_index:cell_end_index]

        # 内层循环：药物分块
        for j in range(num_drug_chunks):
            print("\n" + "-" * 80)
            print(f"--- 开始处理 细胞系分块 {i + 1}/{num_cell_chunks} | 药物分块 {j + 1}/{num_drug_chunks} ---")

            drug_start_index = j * drug_chunk_size
            drug_end_index = min((j + 1) * drug_chunk_size, num_drugs)
            drug_chunk_df = new_drugs_df.iloc[drug_start_index:drug_end_index]

            print(f"此批次包含 {len(cell_chunk_df)} 个细胞系 和 {len(drug_chunk_df)} 个药物。")

            # --- 为当前双重分块生成特征向量 ---
            feature_vectors_for_batch = []
            prediction_identifiers_for_batch = []

            # 预计算当前药物分块的特征
            drug_feature_cache_batch = {}
            for _, drug_row in drug_chunk_df.iterrows():
                drug_feature_cache_batch[drug_row['drug_name']] = {
                    'features': generate_drug_features_from_smiles(drug_row['smiles_string'], q_val=args.lingo_q),
                    'smiles': drug_row['smiles_string']
                }

            # 组合特征
            for cell_line_name, cell_series in cell_chunk_df.iterrows():
                cell_features_np = cell_series.values
                for drug_name, drug_data in drug_feature_cache_batch.items():
                    drug_features_np = drug_data['features']
                    combined_input_features = combine_features(cell_features_np, drug_features_np)
                    if combined_input_features is not None:
                        feature_vectors_for_batch.append(combined_input_features)
                        prediction_identifiers_for_batch.append({
                            'drug_name': drug_name,
                            'smiles': drug_data['smiles'],
                            'cell_line_name': cell_line_name
                        })

            if not feature_vectors_for_batch:
                print(f"警告：批次 (细胞系 {i + 1}, 药物 {j + 1}) 未生成任何有效的特征向量。跳过此批次。")
                continue

            # --- 执行批量预测 ---
            X_to_predict_batch = np.array(feature_vectors_for_batch, dtype=np.float32)  # 使用float32节省一半内存
            print(f"已生成 {X_to_predict_batch.shape[0]} 个特征向量，开始为当前批次进行预测...")

            try:
                predictions = model.predict(X_to_predict_batch, verbose=0).flatten()
            except Exception as e:
                print(f"模型在批次 (细胞系 {i + 1}, 药物 {j + 1}) 预测时出错: {e}。跳过此批次。")
                continue

            # --- 整理并保存当前批次的结果 ---
            results_list = []
            for k, identifier in enumerate(prediction_identifiers_for_batch):
                row_data = {
                    'drug_name': identifier['drug_name'],
                    'smiles': identifier['smiles'],
                    'cell_line_name': identifier['cell_line_name'],
                    'predicted_ic50': predictions[k] if predictions.size > k else np.nan
                }
                results_list.append(row_data)

            results_df = pd.DataFrame(results_list)

            batch_output_filename = f"{output_base}_cell_{i + 1}_drug_{j + 1}{output_ext}"

            try:
                results_df.to_csv(batch_output_filename, index=False)
                print(f"批次 (细胞系 {i + 1}, 药物 {j + 1}) 的预测结果已成功保存至: {batch_output_filename}")
            except Exception as e:
                print(f"错误: 保存批次结果至 '{batch_output_filename}' 时发生: {e}")

            # --- 清理内存 ---
            print("清理当前批次的内存...")
            del drug_chunk_df
            del drug_feature_cache_batch
            del feature_vectors_for_batch
            del prediction_identifiers_for_batch
            del X_to_predict_batch
            del predictions
            del results_df
            del results_list
            gc.collect()
            tf.keras.backend.clear_session()
            print("内存清理完成。")

    print("\n" + "=" * 80)
    print("所有分块处理完毕。程序正常结束。")
    print("=" * 80)


if __name__ == '__main__':
    # 设置TensorFlow线程，可能有助于提高在某些CPU环境下的性能和稳定性
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    main()