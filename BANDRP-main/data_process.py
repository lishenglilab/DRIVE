import numpy as np
import pandas as pd
import torch.utils.data as Data
import torch
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


class my_dataloader(Data.Dataset):
    def __init__(self, drug_data, expression, cnv, mutation, pair, position):
        'Initialization'
        self.pair = pair
        self.drug_data = drug_data
        self.expression = expression
        self.mutation = mutation
        self.cnv = cnv
        self.position = position

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.position)

    def __getitem__(self, index):
        'Generates one sample of data'
        index = self.pair[self.position[index]]
        drug_data = [self.drug_data[i][index[1]] for i in range(len(self.drug_data))]
        expression = self.expression[index[0]]
        cnv = self.cnv[index[0]]
        mutation = self.mutation[index[0]]
        label = index[2]
        return drug_data, expression, mutation, cnv, label


def collate_fn(batch):
    fp1 = torch.stack([i[0][0] for i in batch], 0)
    fp2 = torch.stack([i[0][1] for i in batch], 0)
    fp3 = torch.stack([i[0][2] for i in batch], 0)
    drug_data = [fp1, fp2, fp3]

    exp = torch.stack([i[1] for i in batch], 0)
    mut = torch.stack([i[2] for i in batch], 0)
    cnv = torch.stack([i[3] for i in batch], 0)
    label = torch.tensor([i[4] for i in batch], dtype=torch.float32)
    return [drug_data, exp, mut, cnv, label]


def data_process(drug_feature, mut_feature, exp_feature, cnv_feature, pair, cellline_id, drug_id,
                 method='MIX'):
    # cell_line drug ic50 pairs
    cellline_id.sort()
    drug_id.sort()
    cell_map = list(zip(cellline_id, list(range(len(cellline_id)))))
    drug_map = list(zip(drug_id, list(range(len(drug_id)))))
    cell_dict = {i[0]: i[1] for i in cell_map}
    drug_dict = {i[0]: i[1] for i in drug_map}
    all_pairs = []
    for i in pair:
        all_pairs.append([cell_dict[i[0]], drug_dict[i[1]], i[2]])

    # drug_feature
    drug_feature_num = len(drug_feature[drug_id[0]])
    drug_feature_df = pd.DataFrame(index=drug_id, columns=list(range(drug_feature_num)))
    for index in drug_id:
        for j in range(drug_feature_num):
            drug_feature_df.loc[index, j] = drug_feature[index][j]
    drug_data = [torch.from_numpy(np.array(list(drug_feature_df.iloc[:, i]), dtype='float32')) for i in
                 range(drug_feature_num)]

    # cell lines feature
    # mutation expression methylation
    mutation = mut_feature.loc[cellline_id]
    expression = exp_feature.loc[cellline_id]
    cnv = cnv_feature.loc[cellline_id]

    mutation = torch.from_numpy(np.array(mutation, dtype='float32'))
    expression = torch.from_numpy(np.array(expression, dtype='float32'))
    cnv = torch.from_numpy(np.array(cnv, dtype='float32'))

    # compile train and test
    params = {'batch_size': 128,
              'shuffle': True,
              'num_workers': 4,
              'drop_last': False,
              'collate_fn': collate_fn}

    if method == 'MIX':
        train_index, temp_index = train_test_split(range(len(pair)), test_size=0.2, random_state=42)
        val_index, test_index = train_test_split(temp_index, test_size=0.5, random_state=42)
    elif method == 'DB':
        drug_train, drug_temp = train_test_split(drug_id, test_size=0.2, random_state=42)
        drug_val, drug_test = train_test_split(drug_temp, test_size=0.5, random_state=42)
        train_index = [i for i, (cell, drug, ic50) in enumerate(all_pairs) if drug_id[drug] in drug_train]
        val_index = [i for i, (cell, drug, ic50) in enumerate(all_pairs) if drug_id[drug] in drug_val]
        test_index = [i for i, (cell, drug, ic50) in enumerate(all_pairs) if drug_id[drug] in drug_test]
    elif method == 'CB':
        cell_train, cell_temp = train_test_split(cellline_id, test_size=0.2, random_state=42)
        cell_val, cell_test = train_test_split(cell_temp, test_size=0.5, random_state=42)
        train_index = [i for i, (cell, drug, ic50) in enumerate(all_pairs) if cellline_id[cell] in cell_train]
        val_index = [i for i, (cell, drug, ic50) in enumerate(all_pairs) if cellline_id[cell] in cell_val]
        test_index = [i for i, (cell, drug, ic50) in enumerate(all_pairs) if cellline_id[cell] in cell_test]
    else:
        raise ValueError("Invalid method. Choose from 'MIX', 'DB', 'CB'.")

    train_set = Data.DataLoader(
        my_dataloader(drug_data, expression, mutation, cnv, all_pairs, train_index),
        **params)
    test_set = Data.DataLoader(
        my_dataloader(drug_data, expression, mutation, cnv, all_pairs, test_index),
        **params)
    val_set = Data.DataLoader(
        my_dataloader(drug_data, expression, mutation, cnv, all_pairs, val_index),
        **params)
    return train_set, test_set, val_set