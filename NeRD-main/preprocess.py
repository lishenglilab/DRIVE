import csv
from smiles2graph import smile_to_graph  # 确保这个模块可用
import pickle
from sklearn import preprocessing
import random
import numpy as np
from functions import TestbedDataset  # 确保这个模块和类定义可用
import json  # <-- 新增导入
import os  # <-- 新增导入 (如果 mydata 目录不存在，可能需要创建)


def read_drug_list_and_properties(filename):  # load drugs and their physicochemical properties from files
    f = open(filename, 'r')
    reader = csv.reader(f)
    reader.__next__()
    drug_properties = []
    drug_dict = {}
    index = 0
    for line in reader:
        drug_properties.append(line[1:])
        drug_dict[line[0]] = index  # build a dictionary to save the index of samples
        index += 1
    f.close()  # <-- 建议添加: 关闭文件
    return drug_dict, drug_properties


def read_drug_finger(filename, drug_dict):  # load drugs' molecular fingerprints
    f = open(filename, 'r')
    reader = csv.reader(f)
    reader.__next__()

    drug_finger = [list() for i in range(len(drug_dict))]
    for line in reader:
        drug_name = line[0]
        if drug_name not in drug_dict:
            # print(f"Warning: Drug {drug_name} not found in drug_dict.") #  保持原有注释状态
            continue
        index = drug_dict[drug_name]
        # if index >= len(drug_finger): # 这个检查在 drug_dict 键存在时通常多余
        # print(f"Error: Index {index} is out of range for drug {drug_name}.")
        # continue
        drug_finger[index] = list(map(int, line[1:]))  # use the index in dictionary
    f.close()  # <-- 建议添加: 关闭文件
    return drug_finger


def read_drug_smiles(filename, drug_dict):  # load drugs' SMILES
    f = open(filename, 'r')
    reader = csv.reader(f)
    reader.__next__()
    drug_smiles = [list() for i in range(len(drug_dict))]
    # print(f"Length of drug_smiles list: {len(drug_smiles)}") # 保持原有注释状态
    for line in reader:
        if line[0] in drug_dict:  # <-- 建议添加: 检查 drug 是否在 dict 中
            drug_smiles[drug_dict[line[0]]] = line[1]  # use the index in dictionary
    f.close()  # <-- 建议添加: 关闭文件
    return drug_smiles


def read_cell_line_list(filename):  # load cell lines and build a dictionary
    f = open(filename, 'r')
    reader = csv.reader(f)
    reader.__next__()
    cell_line_dict = {}
    index = 0
    for line in reader:
        cell_line_dict[line[0]] = index
        index += 1
    f.close()  # <-- 建议添加: 关闭文件
    return cell_line_dict


def read_cell_line_miRNA(filename, cell_line_dict):  # load one of the features of cell line - miRNA
    f = open(filename, 'r')
    reader = csv.reader(f)
    reader.__next__()
    miRNA = [list() for i in range(len(cell_line_dict))]
    for line in reader:
        if line[0] in cell_line_dict:
            miRNA[cell_line_dict[line[0]]] = line[1:]
    f.close()  # <-- 建议添加: 关闭文件
    return miRNA


def read_cell_line_copynumber(filename, cell_line_dict):  # load one of the features of cell line - copynumber
    # 此函数在原始代码中未使用，因为copynumber是从pickle加载的
    f = open(filename, 'r')
    reader = csv.reader(f)
    reader.__next__()
    copynumber = [list() for i in range(len(cell_line_dict))]
    for line in reader:
        if line[0] in cell_line_dict:
            copynumber[cell_line_dict[line[0]]] = line[1:]
    f.close()  # <-- 建议添加: 关闭文件
    return copynumber


def min_max_nomalization(data_list, min_val, max_val):  # 参数名修改为 data_list, min_val, max_val
    res = []
    if max_val - min_val == 0:  # 避免除以零
        return [0.0 for _ in data_list]  # 如果所有值相同，标准化为0
    for item in data_list:
        temp = (item - min_val) / (max_val - min_val)
        res.append(temp)
    return res


def get_all_graph(drug_smiles):
    smile_graph = {}
    for smile in drug_smiles:
        # 确保 smile 是字符串且非空
        if isinstance(smile, str) and len(smile) > 0:  # <-- 修改：检查类型和长度
            graph = smile_to_graph(smile)
            smile_graph[smile] = graph
        # else: # 保持原有注释状态
        # print(f"Warning: Invalid SMILES string encountered: {smile}")
    return smile_graph


def read_response_data_and_process(filename):
    # load features
    drug_dict, drug_properties = read_drug_list_and_properties('./mydata/drug/graph.csv')
    finger = read_drug_finger('./mydata/drug/drug_with_conditions.csv', drug_dict)
    smile = read_drug_smiles('./mydata/drug/drug.csv', drug_dict)
    smile_graph = get_all_graph(smile)
    cell_line_dict = read_cell_line_list('mydata/cell_line/cell_line_list.csv')
    miRNA_raw = read_cell_line_miRNA('./mydata/cell_line/mirna.csv', cell_line_dict)  # 重命名以区分
    copynumber_raw = pickle.load(
        open('./mydata/cell_line/512dim_copynumber.pkl', 'rb'))  # Copy number pre-reduced by AE

    # feature normalization
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)  # copy=True更安全
    # 确保miRNA_raw中的空列表被处理
    miRNA_processed = [m if m else [0.0] * (len(miRNA_raw[0]) if miRNA_raw and miRNA_raw[0] else 1) for m in miRNA_raw]
    if not miRNA_processed:
        print("Warning: miRNA data is empty after processing empty lists.")
        miRNA = np.array([])
    else:
        try:
            miRNA = min_max_scaler.fit_transform(np.array(miRNA_processed, dtype=float))
        except ValueError as e:
            print(f"Error during miRNA scaling: {e}. miRNA data might be inconsistent.")
            # Fallback or re-raise based on how critical this is.
            # For now, let's assume if scaling fails, we might have to skip this feature or use zeros.
            # This part needs careful consideration based on your data.
            # Example: miRNA = np.zeros((len(miRNA_processed), len(miRNA_processed[0]) if miRNA_processed else 0))

    # 确保copynumber_raw是numpy array
    if not isinstance(copynumber_raw, np.ndarray):
        copynumber_raw_np = np.array(copynumber_raw, dtype=float)
    else:
        copynumber_raw_np = copynumber_raw.astype(float)

    if copynumber_raw_np.size == 0:
        print("Warning: Copynumber data is empty.")
        copynumber = np.array([])
    else:
        copynumber = min_max_scaler.fit_transform(copynumber_raw_np)

    # read response data
    f = open(filename, 'r')
    reader = csv.reader(f)
    reader.__next__()
    data = []
    for line in reader:
        drug = line[1]
        cell_line = line[0]
        try:  # <-- 建议添加: 转换错误处理
            ic50 = float(line[2])
            data.append((drug, cell_line, ic50))
        except ValueError:
            # print(f"Warning: Could not convert IC50 value '{line[2]}' to float. Skipping entry: {line}") # 保持原有注释状态
            pass
    f.close()  # <-- 建议添加: 关闭文件
    random.shuffle(data)

    # match features and labels
    drug_smile_list = []  # <-- 修改变量名以示区分
    drug_finger_list = []  # <-- 修改变量名以示区分
    cell_miRNA_list = []  # <-- 修改变量名以示区分
    cell_copy_list = []  # <-- 修改变量名以示区分
    label_original_ic50 = []  # <--- 新增：用于存储原始IC50值

    for item in data:
        drug, cell_line, ic50 = item
        if drug in drug_dict and cell_line in cell_line_dict:
            # 确保 SMILES 存在于 smile_graph 中
            current_smile = smile[drug_dict[drug]]
            if not (isinstance(current_smile, str) and len(current_smile) > 0 and current_smile in smile_graph):
                # print(f"Warning: SMILES '{current_smile}' for drug '{drug}' not found in smile_graph or invalid. Skipping.") # 保持原有注释状态
                continue

            # 确保特征存在且维度正确 (简单检查)
            if not finger[drug_dict[drug]]:
                # print(f"Warning: Fingerprint for drug '{drug}' is empty. Skipping.") # 保持原有注释状态
                continue
            if cell_line_dict[cell_line] >= len(miRNA) or not miRNA[cell_line_dict[cell_line]].any():
                # print(f"Warning: miRNA data for cell line '{cell_line}' is missing or invalid. Skipping.") # 保持原有注释状态
                continue
            if cell_line_dict[cell_line] >= len(copynumber) or not copynumber[cell_line_dict[cell_line]].any():
                # print(f"Warning: Copynumber data for cell line '{cell_line}' is missing or invalid. Skipping.") # 保持原有注释状态
                continue

            drug_smile_list.append(current_smile)
            drug_finger_list.append(finger[drug_dict[drug]])
            cell_miRNA_list.append(miRNA[cell_line_dict[cell_line]])
            cell_copy_list.append(copynumber[cell_line_dict[cell_line]])
            label_original_ic50.append(ic50)  # <--- 收集原始IC50值

    # --- 新增代码：计算并保存全局IC50的min和max ---
    if not label_original_ic50:
        print(
            "Error: No valid IC50 values found after matching features. Cannot proceed to normalize or save scaling parameters.")
        return  # 或者抛出异常

    global_min_ic50 = min(label_original_ic50)
    global_max_ic50 = max(label_original_ic50)

    # 确保 mydata 目录存在
    if not os.path.exists('./mydata'):
        os.makedirs('./mydata')
        print("Created directory ./mydata")

    scaling_params_path = './mydata/ic50_scaling_parameters.json'
    try:
        with open(scaling_params_path, 'w') as f_params:
            json.dump({'min_ic50': global_min_ic50, 'max_ic50': global_max_ic50}, f_params, indent=4)
        print(
            f"IC50 scaling parameters (min: {global_min_ic50}, max: {global_max_ic50}) saved to {scaling_params_path}")
    except IOError as e:
        print(f"Error saving IC50 scaling parameters to {scaling_params_path}: {e}")
    # --- 新增代码结束 ---

    # 使用计算出的全局min/max进行标准化
    label_normalized = min_max_nomalization(label_original_ic50, global_min_ic50, global_max_ic50)

    # split data
    # 将 list 转换为 numpy array
    drug_smile_arr = np.asarray(drug_smile_list, dtype=object)  # dtype=object因为SMILES是字符串
    drug_finger_arr = np.asarray(drug_finger_list, dtype=float)  # 假设fingerprint是数值
    cell_miRNA_arr = np.asarray(cell_miRNA_list, dtype=float)
    cell_copy_arr = np.asarray(cell_copy_list, dtype=float)
    label_arr = np.asarray(label_normalized, dtype=float)  # 使用标准化后的标签

    if drug_smile_arr.shape[0] == 0:
        print("Error: No data points remaining after feature matching. Cannot create datasets.")
        return

    for i in range(5):  # 5-fold cross-validation split
        total_size = drug_smile_arr.shape[0]
        if total_size < 5:  # Not enough data for 5 folds with 10% test/10% val
            print(
                f"Warning: Total data size ({total_size}) is too small for 5-fold CV with 10% test/val. Adjusting or skipping.")
            # อาจจะต้องปรับ logic การแบ่งข้อมูลที่นี่ หรือยกเลิกการทำ CV ถ้าข้อมูลน้อยไป
            if total_size == 0: return
            # Simple single split if too small for CV
            size_test = int(total_size * 0.1)
            size_val = int(total_size * 0.1)
            if size_test == 0 and total_size > 0: size_test = 1
            if size_val == 0 and total_size > 1: size_val = 1

            size_0 = 0  # Start of test
            size_1 = size_test  # Start of val
            size_2 = size_test + size_val  # End of val / Start of train part 2

            if size_2 >= total_size and total_size > 0:  # If test+val is almost all data
                if total_size == 1:  # Only one sample
                    size_0, size_1, size_2 = 0, 0, 0  # train with this one sample
                elif total_size == 2:  # Two samples
                    size_0, size_1, size_2 = 0, 1, 1  # test=0, val=1, train=empty / or test=0, val=0, train=1,2
                # Add more sophisticated handling if needed

            if i > 0:  # Only do one split if data is too small
                print(f"Skipping CV fold {i + 1} due to small dataset size.")
                continue
        else:
            size_0 = int(total_size * 0.2 * i)
            size_1 = size_0 + int(total_size * 0.1)  # Test set is 10%
            size_2 = int(total_size * 0.2 * (i + 1))  # End of Val set (Val set is 10%)

        # Ensure indices are within bounds, especially for the last fold
        size_1 = min(size_1, total_size)
        size_2 = min(size_2, total_size)

        # features of drug fingers
        drugfinger_test = drug_finger_arr[size_0:size_1]
        drugfinger_val = drug_finger_arr[size_1:size_2]
        drugfinger_train = np.concatenate((drug_finger_arr[:size_0], drug_finger_arr[size_2:]), axis=0)
        # features of drug smiles
        drugsmile_test = drug_smile_arr[size_0:size_1]
        drugsmile_val = drug_smile_arr[size_1:size_2]
        drugsmile_train = np.concatenate((drug_smile_arr[:size_0], drug_smile_arr[size_2:]), axis=0)
        # features of cell miRNA
        cellmiRNA_test = cell_miRNA_arr[size_0:size_1]
        cellmiRNA_val = cell_miRNA_arr[size_1:size_2]
        cellmiRNA_train = np.concatenate((cell_miRNA_arr[:size_0], cell_miRNA_arr[size_2:]), axis=0)
        # features of cell copynumber
        cellcopy_test = cell_copy_arr[size_0:size_1]
        cellcopy_val = cell_copy_arr[size_1:size_2]
        cellcopy_train = np.concatenate((cell_copy_arr[:size_0], cell_copy_arr[size_2:]), axis=0)
        # label
        label_test = label_arr[size_0:size_1]
        label_val = label_arr[size_1:size_2]
        label_train = np.concatenate((label_arr[:size_0], label_arr[size_2:]), axis=0)

        # Check if any split is empty, which can happen if total_size is small
        if drugfinger_train.shape[0] == 0 or drugsmile_train.shape[0] == 0 or \
                cellmiRNA_train.shape[0] == 0 or cellcopy_train.shape[0] == 0 or \
                label_train.shape[0] == 0:
            print(f"Warning: Training set for fold {i} is empty. Skipping dataset creation for this fold.")
            continue
        # Similar checks for val and test if you want to be very robust,
        # though TestbedDataset might handle empty inputs internally or error out.

        print(f"Fold {i}: Train size: {len(label_train)}, Val size: {len(label_val)}, Test size: {len(label_test)}")

        TestbedDataset(root='mydata', dataset='train_set{num}'.format(num=i), xdf=drugfinger_train,
                       xds=drugsmile_train,
                       xcm=cellmiRNA_train, xcc=cellcopy_train,
                       y=label_train, smile_graph=smile_graph)
        if len(label_val) > 0:  # Only create val set if it's not empty
            TestbedDataset(root='mydata', dataset='val_set{num}'.format(num=i), xdf=drugfinger_val, xds=drugsmile_val,
                           xcm=cellmiRNA_val, xcc=cellcopy_val,
                           y=label_val, smile_graph=smile_graph)
        if len(label_test) > 0:  # Only create test set if it's not empty
            TestbedDataset(root='mydata', dataset='test_set{num}'.format(num=i), xdf=drugfinger_test,
                           xds=drugsmile_test,
                           xcm=cellmiRNA_test, xcc=cellcopy_test,
                           y=label_test, smile_graph=smile_graph)
    return


def process_blind_cell(filename):
    # load features
    drug_dict, drug_properties = read_drug_list_and_properties('./mydata/drug/graph.csv')
    finger = read_drug_finger('./mydata/drug/drug_with_conditions.csv', drug_dict)
    smile = read_drug_smiles('./mydata/drug/drug.csv', drug_dict)
    smile_graph = get_all_graph(smile)
    cell_line_dict = read_cell_line_list('mydata/cell_line/cell_line_list.csv')
    miRNA_raw = read_cell_line_miRNA('./mydata/cell_line/mirna.csv', cell_line_dict)
    copynumber_raw = pickle.load(
        open('./mydata/cell_line/512dim_copynumber.pkl', 'rb'))  # Copy number pre-reduced by AE

    # feature normalization
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)
    miRNA_processed = [m if m else [0.0] * (len(miRNA_raw[0]) if miRNA_raw and miRNA_raw[0] else 1) for m in miRNA_raw]
    if not miRNA_processed:
        miRNA = np.array([])
    else:
        miRNA = min_max_scaler.fit_transform(np.array(miRNA_processed, dtype=float))

    if not isinstance(copynumber_raw, np.ndarray):
        copynumber_raw_np = np.array(copynumber_raw, dtype=float)
    else:
        copynumber_raw_np = copynumber_raw.astype(float)
    if copynumber_raw_np.size == 0:
        copynumber = np.array([])
    else:
        copynumber = min_max_scaler.fit_transform(copynumber_raw_np)

    # read response data
    f = open(filename, 'r')
    reader = csv.reader(f)
    reader.__next__()
    data = []
    label_original_ic50_blind_cell = []  # Collect original IC50s for this function's scope

    for line in reader:
        drug = line[1]
        cell_line = line[0]
        try:
            ic50 = float(line[2])
            data.append((drug, cell_line, ic50))
            label_original_ic50_blind_cell.append(ic50)  # Collect for potential local normalization scaling params
        except ValueError:
            pass
    f.close()
    random.shuffle(data)

    # --- 如果这里的标签也需要使用全局的min/max进行标准化，需要从文件加载 ---
    # global_min_ic50, global_max_ic50 = None, None
    # scaling_params_path = './mydata/ic50_scaling_parameters.json'
    # try:
    #     with open(scaling_params_path, 'r') as f_params:
    #         params = json.load(f_params)
    #         global_min_ic50 = params['min_ic50']
    #         global_max_ic50 = params['max_ic50']
    # except Exception as e:
    #     print(f"Warning: Could not load global IC50 scaling parameters for process_blind_cell: {e}")
    #     # Fallback: use local min/max for normalization if global params are not available
    #     if label_original_ic50_blind_cell:
    #         global_min_ic50 = min(label_original_ic50_blind_cell)
    #         global_max_ic50 = max(label_original_ic50_blind_cell)
    #     else: # No labels, cannot normalize
    #         print("Error: No IC50 labels for process_blind_cell, cannot normalize.")
    #         return
    # --- 上述代码块用于加载全局参数，如果每个process_blind_*函数都应使用独立的min/max，则不需要它 ---
    # 当前代码的逻辑是 TestbedDataset 接收到的 y 已经是原始值，它内部不做标准化。
    # 如果要让 process_blind_cell/drug 的输出也标准化，需要在这里进行。
    # 为了与 read_response_data_and_process 的行为一致（它对label进行了标准化），
    # 我们应该也对这里的 label_train, label_val, label_test 进行标准化。
    # 最好的做法是使用全局的 min/max。

    dict_drug_cell = {}
    for item in data:  # data already contains (drug, cell_line, original_ic50)
        drug, cell_line, ic50_val = item  # ic50_val is original
        if drug in drug_dict and cell_line in cell_line_dict:
            if cell_line in dict_drug_cell:
                dict_drug_cell[cell_line].append((drug, ic50_val))
            else:
                dict_drug_cell[cell_line] = [(drug, ic50_val)]

    for i in range(5):
        total_size = len(dict_drug_cell)  # Number of unique cell lines
        if total_size < 5:
            if i > 0: continue  # Simplified handling for small datasets
            size = 0;
            size1 = int(total_size * 0.5);
            size2 = total_size  # Example: 50% test, 50% train for non-CV
        else:
            size = int(total_size * i * 0.2)
            size1 = size + int(total_size * 0.1)  # Test based on cell lines
            size2 = int(total_size * (i + 1) * 0.2)  # Val based on cell lines

        size1 = min(size1, total_size)
        size2 = min(size2, total_size)

        drugsmile_train_l, drugfinger_train_l, cellmiRNA_train_l, cellcopy_train_l, label_train_original_l = [], [], [], [], []
        drugsmile_val_l, drugfinger_val_l, cellmiRNA_val_l, cellcopy_val_l, label_val_original_l = [], [], [], [], []
        drugsmile_test_l, drugfinger_test_l, cellmiRNA_test_l, cellcopy_test_l, label_test_original_l = [], [], [], [], []

        pos = 0
        # Sort dict_drug_cell items to ensure consistent splits if data order changes
        sorted_dict_drug_cell_items = sorted(dict_drug_cell.items())

        for cell, values in sorted_dict_drug_cell_items:
            pos += 1
            for v_drug, v_ic50_orig in values:  # v_ic50_orig is original
                current_smile = smile[drug_dict[v_drug]]
                if not (isinstance(current_smile, str) and len(
                    current_smile) > 0 and current_smile in smile_graph): continue
                if not finger[drug_dict[v_drug]]: continue
                if cell_line_dict[cell] >= len(miRNA) or (
                        isinstance(miRNA, np.ndarray) and not miRNA[cell_line_dict[cell]].any()): continue
                if cell_line_dict[cell] >= len(copynumber) or (
                        isinstance(copynumber, np.ndarray) and not copynumber[cell_line_dict[cell]].any()): continue

                ds = current_smile
                df = finger[drug_dict[v_drug]]
                cm = miRNA[cell_line_dict[cell]]
                cc = copynumber[cell_line_dict[cell]]
                lbl_orig = v_ic50_orig

                if pos > size and pos <= size1:  # Test set cells
                    drugsmile_test_l.append(ds);
                    drugfinger_test_l.append(df);
                    cellmiRNA_test_l.append(cm);
                    cellcopy_test_l.append(cc);
                    label_test_original_l.append(lbl_orig)
                elif pos > size1 and pos <= size2:  # Val set cells
                    drugsmile_val_l.append(ds);
                    drugfinger_val_l.append(df);
                    cellmiRNA_val_l.append(cm);
                    cellcopy_val_l.append(cc);
                    label_val_original_l.append(lbl_orig)
                else:  # Train set cells
                    drugsmile_train_l.append(ds);
                    drugfinger_train_l.append(df);
                    cellmiRNA_train_l.append(cm);
                    cellcopy_train_l.append(cc);
                    label_train_original_l.append(lbl_orig)

        # Normalize labels using global min/max if available and loaded
        # For consistency, let's assume process_blind_cell/drug also use the global scaling parameters
        # Load global scaling parameters if not already loaded (or pass them as arguments)
        current_global_min_ic50, current_global_max_ic50 = None, None
        scaling_params_path = './mydata/ic50_scaling_parameters.json'  # Redundant if loaded at script start
        if os.path.exists(scaling_params_path):
            with open(scaling_params_path, 'r') as f_params:
                params = json.load(f_params)
                current_global_min_ic50 = params['min_ic50']
                current_global_max_ic50 = params['max_ic50']

        if current_global_min_ic50 is None or current_global_max_ic50 is None:
            print(
                f"Warning (process_blind_cell fold {i}): Global IC50 scaling params not found. Using local min/max or skipping normalization if no labels.")
            if label_train_original_l: current_global_min_ic50 = min(
                label_train_original_l + label_val_original_l + label_test_original_l)
            if label_train_original_l: current_global_max_ic50 = max(
                label_train_original_l + label_val_original_l + label_test_original_l)

        if current_global_min_ic50 is not None and current_global_max_ic50 is not None:
            label_train_norm = np.asarray(
                min_max_nomalization(label_train_original_l, current_global_min_ic50, current_global_max_ic50),
                dtype=float)
            label_val_norm = np.asarray(
                min_max_nomalization(label_val_original_l, current_global_min_ic50, current_global_max_ic50),
                dtype=float)
            label_test_norm = np.asarray(
                min_max_nomalization(label_test_original_l, current_global_min_ic50, current_global_max_ic50),
                dtype=float)
        else:  # Should not happen if there's data, but as a fallback
            label_train_norm = np.asarray(label_train_original_l, dtype=float)
            label_val_norm = np.asarray(label_val_original_l, dtype=float)
            label_test_norm = np.asarray(label_test_original_l, dtype=float)

        drugsmile_train, drugfinger_train = np.asarray(drugsmile_train_l, dtype=object), np.asarray(drugfinger_train_l,
                                                                                                    dtype=float)
        cellmiRNA_train, cellcopy_train = np.asarray(cellmiRNA_train_l, dtype=float), np.asarray(cellcopy_train_l,
                                                                                                 dtype=float)
        # label_train = np.asarray(label_train_original_l, dtype=float) # Original code passes original IC50s
        label_train = label_train_norm

        drugsmile_val, drugfinger_val = np.asarray(drugsmile_val_l, dtype=object), np.asarray(drugfinger_val_l,
                                                                                              dtype=float)
        cellmiRNA_val, cellcopy_val = np.asarray(cellmiRNA_val_l, dtype=float), np.asarray(cellcopy_val_l, dtype=float)
        # label_val = np.asarray(label_val_original_l, dtype=float)
        label_val = label_val_norm

        drugsmile_test, drugfinger_test = np.asarray(drugsmile_test_l, dtype=object), np.asarray(drugfinger_test_l,
                                                                                                 dtype=float)
        cellmiRNA_test, cellcopy_test = np.asarray(cellmiRNA_test_l, dtype=float), np.asarray(cellcopy_test_l,
                                                                                              dtype=float)
        # label_test = np.asarray(label_test_original_l, dtype=float)
        label_test = label_test_norm

        if len(label_train) > 0:
            TestbedDataset(root='mydata', dataset='train_blind_cell{num}'.format(num=i), xdf=drugfinger_train,
                           xds=drugsmile_train, xcm=cellmiRNA_train, xcc=cellcopy_train,
                           y=label_train, smile_graph=smile_graph)
        if len(label_val) > 0:
            TestbedDataset(root='mydata', dataset='val_blind_cell{num}'.format(num=i), xdf=drugfinger_val,
                           xds=drugsmile_val,
                           xcm=cellmiRNA_val, xcc=cellcopy_val,
                           y=label_val, smile_graph=smile_graph)
        if len(label_test) > 0:
            TestbedDataset(root='mydata', dataset='test_blind_cell{num}'.format(num=i), xdf=drugfinger_test,
                           xds=drugsmile_test,
                           xcm=cellmiRNA_test, xcc=cellcopy_test,
                           y=label_test, smile_graph=smile_graph)
    return


def process_blind_drug(filename):
    # (Similar structure to process_blind_cell, applying normalization to labels consistently)
    # load features
    drug_dict, drug_properties = read_drug_list_and_properties('./mydata/drug/graph.csv')
    finger = read_drug_finger('./mydata/drug/drug_with_conditions.csv', drug_dict)
    smile = read_drug_smiles('./mydata/drug/drug.csv', drug_dict)
    smile_graph = get_all_graph(smile)
    cell_line_dict = read_cell_line_list('mydata/cell_line/cell_line_list.csv')
    miRNA_raw = read_cell_line_miRNA('./mydata/cell_line/mirna.csv', cell_line_dict)
    copynumber_raw = pickle.load(open('./mydata/cell_line/512dim_copynumber.pkl', 'rb'))

    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)
    miRNA_processed = [m if m else [0.0] * (len(miRNA_raw[0]) if miRNA_raw and miRNA_raw[0] else 1) for m in miRNA_raw]
    if not miRNA_processed:
        miRNA = np.array([])
    else:
        miRNA = min_max_scaler.fit_transform(np.array(miRNA_processed, dtype=float))

    if not isinstance(copynumber_raw, np.ndarray):
        copynumber_raw_np = np.array(copynumber_raw, dtype=float)
    else:
        copynumber_raw_np = copynumber_raw.astype(float)
    if copynumber_raw_np.size == 0:
        copynumber = np.array([])
    else:
        copynumber = min_max_scaler.fit_transform(copynumber_raw_np)

    f = open(filename, 'r')
    reader = csv.reader(f)
    reader.__next__()
    data = []
    for line in reader:
        drug = line[1]
        cell_line = line[0]
        try:
            ic50 = float(line[2])
            data.append((drug, cell_line, ic50))
        except ValueError:
            pass
    f.close()
    random.shuffle(data)

    dict_drug_cell = {}
    for item in data:
        drug, cell_line, ic50_val = item
        if drug in drug_dict and cell_line in cell_line_dict:
            if drug in dict_drug_cell:  # Group by drug for blind drug split
                dict_drug_cell[drug].append((cell_line, ic50_val))
            else:
                dict_drug_cell[drug] = [(cell_line, ic50_val)]

    for i in range(5):
        total_size = len(dict_drug_cell)  # Number of unique drugs
        if total_size < 5:
            if i > 0: continue
            size = 0;
            size1 = int(total_size * 0.5);
            size2 = total_size
        else:
            size = int(total_size * i * 0.2)
            size1 = size + int(total_size * 0.1)  # Test based on drugs
            size2 = int(total_size * (i + 1) * 0.2)  # Val based on drugs

        size1 = min(size1, total_size)
        size2 = min(size2, total_size)

        drugsmile_train_l, drugfinger_train_l, cellmiRNA_train_l, cellcopy_train_l, label_train_original_l = [], [], [], [], []
        drugsmile_val_l, drugfinger_val_l, cellmiRNA_val_l, cellcopy_val_l, label_val_original_l = [], [], [], [], []
        drugsmile_test_l, drugfinger_test_l, cellmiRNA_test_l, cellcopy_test_l, label_test_original_l = [], [], [], [], []

        pos = 0
        sorted_dict_drug_cell_items = sorted(dict_drug_cell.items())

        for drug, values in sorted_dict_drug_cell_items:
            pos += 1
            current_smile = smile[drug_dict[drug]]
            if not (isinstance(current_smile, str) and len(
                current_smile) > 0 and current_smile in smile_graph): continue
            if not finger[drug_dict[drug]]: continue

            for v_cell, v_ic50_orig in values:
                if cell_line_dict[v_cell] >= len(miRNA) or (
                        isinstance(miRNA, np.ndarray) and not miRNA[cell_line_dict[v_cell]].any()): continue
                if cell_line_dict[v_cell] >= len(copynumber) or (
                        isinstance(copynumber, np.ndarray) and not copynumber[cell_line_dict[v_cell]].any()): continue

                ds = current_smile
                df = finger[drug_dict[drug]]
                cm = miRNA[cell_line_dict[v_cell]]
                cc = copynumber[cell_line_dict[v_cell]]
                lbl_orig = v_ic50_orig

                if pos > size and pos <= size1:  # Test set drugs
                    drugsmile_test_l.append(ds);
                    drugfinger_test_l.append(df);
                    cellmiRNA_test_l.append(cm);
                    cellcopy_test_l.append(cc);
                    label_test_original_l.append(lbl_orig)
                elif pos > size1 and pos <= size2:  # Val set drugs
                    drugsmile_val_l.append(ds);
                    drugfinger_val_l.append(df);
                    cellmiRNA_val_l.append(cm);
                    cellcopy_val_l.append(cc);
                    label_val_original_l.append(lbl_orig)
                else:  # Train set drugs
                    drugsmile_train_l.append(ds);
                    drugfinger_train_l.append(df);
                    cellmiRNA_train_l.append(cm);
                    cellcopy_train_l.append(cc);
                    label_train_original_l.append(lbl_orig)

        current_global_min_ic50, current_global_max_ic50 = None, None
        scaling_params_path = './mydata/ic50_scaling_parameters.json'
        if os.path.exists(scaling_params_path):
            with open(scaling_params_path, 'r') as f_params:
                params = json.load(f_params)
                current_global_min_ic50 = params['min_ic50']
                current_global_max_ic50 = params['max_ic50']

        if current_global_min_ic50 is None or current_global_max_ic50 is None:
            print(
                f"Warning (process_blind_drug fold {i}): Global IC50 scaling params not found. Using local min/max or skipping normalization if no labels.")
            if label_train_original_l: current_global_min_ic50 = min(
                label_train_original_l + label_val_original_l + label_test_original_l)
            if label_train_original_l: current_global_max_ic50 = max(
                label_train_original_l + label_val_original_l + label_test_original_l)

        if current_global_min_ic50 is not None and current_global_max_ic50 is not None:
            label_train_norm = np.asarray(
                min_max_nomalization(label_train_original_l, current_global_min_ic50, current_global_max_ic50),
                dtype=float)
            label_val_norm = np.asarray(
                min_max_nomalization(label_val_original_l, current_global_min_ic50, current_global_max_ic50),
                dtype=float)
            label_test_norm = np.asarray(
                min_max_nomalization(label_test_original_l, current_global_min_ic50, current_global_max_ic50),
                dtype=float)
        else:
            label_train_norm = np.asarray(label_train_original_l, dtype=float)
            label_val_norm = np.asarray(label_val_original_l, dtype=float)
            label_test_norm = np.asarray(label_test_original_l, dtype=float)

        drugsmile_train, drugfinger_train = np.asarray(drugsmile_train_l, dtype=object), np.asarray(drugfinger_train_l,
                                                                                                    dtype=float)
        cellmiRNA_train, cellcopy_train = np.asarray(cellmiRNA_train_l, dtype=float), np.asarray(cellcopy_train_l,
                                                                                                 dtype=float)
        label_train = label_train_norm

        drugsmile_val, drugfinger_val = np.asarray(drugsmile_val_l, dtype=object), np.asarray(drugfinger_val_l,
                                                                                              dtype=float)
        cellmiRNA_val, cellcopy_val = np.asarray(cellmiRNA_val_l, dtype=float), np.asarray(cellcopy_val_l, dtype=float)
        label_val = label_val_norm

        drugsmile_test, drugfinger_test = np.asarray(drugsmile_test_l, dtype=object), np.asarray(drugfinger_test_l,
                                                                                                 dtype=float)
        cellmiRNA_test, cellcopy_test = np.asarray(cellmiRNA_test_l, dtype=float), np.asarray(cellcopy_test_l,
                                                                                              dtype=float)
        label_test = label_test_norm

        if len(label_train) > 0:
            TestbedDataset(root='mydata', dataset='train_blind_drug{num}'.format(num=i), xdf=drugfinger_train,
                           xds=drugsmile_train, xcm=cellmiRNA_train, xcc=cellcopy_train,
                           y=label_train, smile_graph=smile_graph)
        if len(label_val) > 0:
            TestbedDataset(root='mydata', dataset='val_blind_drug{num}'.format(num=i), xdf=drugfinger_val,
                           xds=drugsmile_val,
                           xcm=cellmiRNA_val, xcc=cellcopy_val,
                           y=label_val, smile_graph=smile_graph)
        if len(label_test) > 0:
            TestbedDataset(root='mydata', dataset='test_blind_drug{num}'.format(num=i), xdf=drugfinger_test,
                           xds=drugsmile_test,
                           xcm=cellmiRNA_test, xcc=cellcopy_test,
                           y=label_test, smile_graph=smile_graph)
    return


if __name__ == "__main__":
    # 确保 functions.py 中的 TestbedDataset 和 smiles2graph.py 中的 smile_to_graph 可用
    # 并且它们与这个脚本的期望输入/输出一致。
    # 例如，TestbedDataset 应该期望接收已经标准化的 y 值（如果模型预测的是标准化值）。
    # 如果 TestbedDataset 内部有自己的标准化逻辑，那么这里传递的 y 应该是原始值。
    # 根据之前的讨论，模型输出的是 0-1 范围的值，所以 TestbedDataset 应该接收标准化的 y。

    print("Starting data preprocessing...")
    read_response_data_and_process('./mydata/label/drug_response.csv')
    print("\nProcessing blind cell line splits...")
    process_blind_cell('./mydata/label/drug_response.csv')
    print("\nProcessing blind drug splits...")
    process_blind_drug('./mydata/label/drug_response.csv')
    print("\nData preprocessing finished.")