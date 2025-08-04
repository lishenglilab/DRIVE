import os
import csv
import numpy as np
import numbers
import h5py
import math
import pandas as pd
import json, pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from torch._C import device
# from gin_utils import *
import random
import pickle
import sys
import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from GPDRP.utils import *

seed_value =88
np.random.seed(seed_value)
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def is_not_float(string_list):
    try:
        for string in string_list:
            float(string)
        return False
    except:
        return True
folder = "mydata/"
"""
The following code will convert the SMILES format into onehot format
"""
def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))
def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))
def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))
    edges = []

    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index
def load_drug_smile():
    reader = csv.reader(open(folder + "drugsmiles.csv"))
    next(reader, None)
    drug_dict = {}
    drug_smile = []

    for item in reader:
        name = item[0]
        smile = item[1]

        if name in drug_dict:
            pos = drug_dict[name]
        else:
            pos = len(drug_dict)
            drug_dict[name] = pos
        drug_smile.append(smile)

    smile_graph = {}
    for smile in drug_smile:
        g = smile_to_graph(smile)
        smile_graph[smile] = g

    return drug_dict, drug_smile, smile_graph
"""
This part is used to read Cell line features
"""
def save_cell_ge_matrix():
    f = open(folder + "exp.csv")
    reader = csv.reader(f)
    firstRow = next(reader)
    numberCol = len(firstRow) - 1
    features = {}
    cell_dict = {}
    matrix_list = []
    for item in reader:
        cell_id = item[0]
        ge = []
        for i in range(1, len(item)):
            ge.append(int(item[i]))
        cell_dict[cell_id] = np.asarray(ge)
    return cell_dict
def save_cell_oge_matrix():
    f = open(folder + "exp.txt")
    line = f.readline()
    elements = line.split()
    cell_names = []
    feature_names = []
    cell_dict = {}
    i = 0
    for cell in range(2, len(elements)):
        if i < 500:
            cell_name = elements[cell]
            cell_names.append(cell_name)
            cell_dict[cell_name] = []

    min = 0
    max = 12
    for line in f.readlines():
        elements = line.split("\t")
        if len(elements) < 2:
            print(line)
            continue
        feature_names.append(elements[1])

        for i in range(2, len(elements)):
            cell_name = cell_names[i - 2]
            value = float(elements[i])
            if min == 0:
                min = value
            if value < min:
                min = value
            if max < value:
                value = max
            cell_dict[cell_name].append(value)
    # print(min)
    # print(max)
    cell_feature = []
    for cell_name in cell_names:
        for i in range(0, len(cell_dict[cell_name])):
            cell_dict[cell_name][i] = (cell_dict[cell_name][i] - min) / (max - min)
        cell_dict[cell_name] = np.asarray(cell_dict[cell_name])
        cell_feature.append(np.asarray(cell_dict[cell_name]))

    cell_feature = np.asarray(cell_feature)
    i = 0
    for cell in list(cell_dict.keys()):
        cell_dict[cell] = i
        i += 1
    print(len(list(cell_dict.values())))
    # exit()
    return cell_dict, cell_feature
"""
This part is used to extract the ic50 of drug - cell line
"""
class DataBuilder(Dataset):
    def __init__(self, cell_feature_ge):
        self.cell_feature_ge = cell_feature_ge
        self.cell_feature_ge = torch.FloatTensor(self.cell_feature_ge)
        self.len = self.cell_feature_ge[0]

    def __getitem__(self, index):
        return self.cell_feature_ge[index]

    def __len__(self):
        return self.len
def save_blind_cell_matrix():
    f = open(folder + "ic50.csv")
    reader = csv.reader(f)
    next(reader)

    cell_dict_ge, cell_feature_ge = save_cell_oge_matrix()
    drug_dict, drug_smile, smile_graph = load_drug_smile()
    xd_all = np.asarray(drug_smile)
    matrix_list = []
    temp_data = []
    xd_train = []
    xc_ge_train = []
    y_train = []
    xd_val = []
    xc_ge_val = []
    y_val = []
    xd_test = []
    xc_ge_test = []
    y_test = []
    dict_drug_cell = {}
    # bExist = np.zeros((len(drug_dict), len(cell_dict)))
    for item in reader:
        drug = item[0]
        cell = item[1]
        ic50 = item[2]
        ic50 = 1 / (1 + pow(math.exp(float(ic50)), -0.1))
        temp_data.append((drug, cell, ic50))
    # random.shuffle(temp_data)
    for data in temp_data:
        drug, cell, ic50 = data
        if drug in drug_dict and cell in cell_dict_ge:
            if cell in dict_drug_cell:
                dict_drug_cell[cell].append((drug, ic50))
            else:
                dict_drug_cell[cell] = [(drug, ic50)]
            # bExist[drug_dict[drug], cell_dict[cell]] = 1
    lstCellTest = []

    size = int(len(dict_drug_cell) * 0.8)
    size1 = int(len(dict_drug_cell) * 0.9)
    pos = 0
    temp_test_cell = None
    temp_val_cell = None
    test_cell_list = []
    val_cell_list = []
    value = 1

    for cell, values in dict_drug_cell.items():
        pos += 1
        for v in values:
            drug, ic50 = v
            if pos < size:
                xd_train.append(drug_smile[drug_dict[drug]])
                xc_ge_train.append(cell_feature_ge[cell_dict_ge[cell]])
                y_train.append(ic50)
            elif pos < size1:
                xd_val.append(drug_smile[drug_dict[drug]])
                xc_ge_val.append(cell_feature_ge[cell_dict_ge[cell]])
                y_val.append(ic50)
                if temp_val_cell != cell:
                    temp_val_cell = cell
                    value = 1
                    val_cell_list.append([cell, value])
                else:
                    value += 1
                    val_cell_list.append([cell, value])
            else:
                xd_test.append(drug_smile[drug_dict[drug]])
                xc_ge_test.append(cell_feature_ge[cell_dict_ge[cell]])
                y_test.append(ic50)
                lstCellTest.append(cell)
                if temp_test_cell != cell:
                    temp_test_cell = cell
                    value = 1
                    test_cell_list.append([cell,value])
                else:
                    value += 1
                    test_cell_list.append([cell,value])

    with open('cell_blind_test', 'wb') as fp:
        pickle.dump(lstCellTest, fp)

    print(len(y_train), len(y_val), len(y_test))

    xd_train, xc_train, y_train = np.asarray(xd_train), np.asarray(xc_ge_train), np.asarray(y_train)
    xd_val, xc_val, y_val = np.asarray(xd_val), np.asarray(xc_ge_val), np.asarray(y_val)
    xd_test, xc_test, y_test = np.asarray(xd_test), np.asarray(xc_ge_test), np.asarray(y_test)
    np.save('test_cell', test_cell_list)
    dataset = 'GDSC'
    print('preparing ', dataset + '_train.pt in pytorch format!')
    train_data = TestbedDataset(root='mydata', dataset=dataset + '_train_cell_blind', xd=xd_train, xt_ge=xc_ge_train, y=y_train,
                                smile_graph=smile_graph)
    val_data = TestbedDataset(root='mydata', dataset=dataset + '_val_cell_blind', xd=xd_val, xt_ge=xc_ge_val, y=y_val,
                              smile_graph=smile_graph)
    # test_all = TestbedDataset(root='data', dataset=dataset + '_all_mix', xd=xd_all, xt_ge=xc_ge_test[:173],
    #                           y=y_test[:173], smile_graph=smile_graph)
    test_data = TestbedDataset(root='mydata', dataset=dataset + '_test_cell_blind', xd=xd_test, xt_ge=xc_ge_test, y=y_test,
                               smile_graph=smile_graph)
    print(train_data)
    print(val_data)
    print(test_data)

def random_split_data():
    # 打开包含IC50值的CSV文件
    with open(folder + "ic50.csv") as f:
        reader = csv.reader(f)
        next(reader)  # 跳过标题行

        cell_dict_ge, cell_feature_ge = save_cell_oge_matrix()  # 加载细胞特征矩阵
        drug_dict, drug_smile, smile_graph = load_drug_smile()  # 加载药物特征及其图形
        xd_all = np.asarray(drug_smile)

        temp_data = []
        for item in reader:
            drug = item[0]
            cell = item[1]
            ic50 = item[2]
            ic50 = 1 / (1 + pow(math.exp(float(ic50)), -0.1))  # 标准化IC50值
            temp_data.append((drug, cell, ic50))

        random.shuffle(temp_data)  # 随机打乱数据顺序

        dict_drug_cell = {}
        # 生成药物-细胞字典
        for data in temp_data:
            drug, cell, ic50 = data
            if drug in drug_dict and cell in cell_dict_ge:
                dict_drug_cell.setdefault(cell, []).append((drug, ic50))

        # 按 80% 训练、10% 验证、10% 测试分割数据
        size_train = int(len(dict_drug_cell) * 0.8)
        size_val = int(len(dict_drug_cell) * 0.9)

        xd_train, xc_ge_train, y_train = [], [], []
        xd_val, xc_ge_val, y_val = [], [], []
        xd_test, xc_ge_test, y_test = [], [], []
        lstCellTest = []

        pos = 0
        for cell, values in dict_drug_cell.items():
            pos += 1
            for v in values:
                drug, ic50 = v
                if pos <= size_train:  # 训练集
                    xd_train.append(drug_smile[drug_dict[drug]])
                    xc_ge_train.append(cell_feature_ge[cell_dict_ge[cell]])
                    y_train.append(ic50)
                elif pos <= size_val:  # 验证集
                    xd_val.append(drug_smile[drug_dict[drug]])
                    xc_ge_val.append(cell_feature_ge[cell_dict_ge[cell]])
                    y_val.append(ic50)
                else:  # 测试集
                    xd_test.append(drug_smile[drug_dict[drug]])
                    xc_ge_test.append(cell_feature_ge[cell_dict_ge[cell]])
                    y_test.append(ic50)
                    lstCellTest.append(cell)

        # 数据保存
        with open('cell_blind_test', 'wb') as fp:
            pickle.dump(lstCellTest, fp)

        print('Training samples:', len(y_train), 'Validation samples:', len(y_val), 'Testing samples:', len(y_test))

        # 转换为Numpy数组
        xd_train, xc_train, y_train = map(np.asarray, [xd_train, xc_ge_train, y_train])
        xd_val, xc_val, y_val = map(np.asarray, [xd_val, xc_ge_val, y_val])
        xd_test, xc_test, y_test = map(np.asarray, [xd_test, xc_ge_test, y_test])

        # 创建数据集
        dataset = 'GDSC'
        print('Preparing ', dataset + '_train.pt in PyTorch format!')
        train_data = TestbedDataset(root='mydata', dataset=dataset + '_train_cell_blind', xd=xd_train, xt_ge=xc_train, y=y_train, smile_graph=smile_graph)
        val_data = TestbedDataset(root='mydata', dataset=dataset + '_val_cell_blind', xd=xd_val, xt_ge=xc_val, y=y_val, smile_graph=smile_graph)
        test_data = TestbedDataset(root='mydata', dataset=dataset + '_test_cell_blind', xd=xd_test, xt_ge=xc_test, y=y_test, smile_graph=smile_graph)

        print('Train data:', train_data)
        print('Validation data:', val_data)
        print('Test data:', test_data)


def drug_blind_split_data():
    # 打开包含IC50值的CSV文件
    with open(folder + "ic50.csv") as f:
        reader = csv.reader(f)
        next(reader)  # 跳过标题行

        cell_dict_ge, cell_feature_ge = save_cell_oge_matrix()  # 加载细胞特征矩阵
        drug_dict, drug_smile, smile_graph = load_drug_smile()  # 加载药物特征及其图形
        xd_all = np.asarray(drug_smile)

        # 收集所有数据
        temp_data = []
        for item in reader:
            drug = item[0]
            cell = item[1]
            ic50 = item[2]
            ic50 = 1 / (1 + pow(math.exp(float(ic50)), -0.1))  # 标准化IC50值
            temp_data.append((drug, cell, ic50))

        random.shuffle(temp_data)  # 随机打乱数据顺序

        dict_drug_cell = {}
        # 生成药物-细胞字典
        for data in temp_data:
            drug, cell, ic50 = data
            if drug in drug_dict and cell in cell_dict_ge:
                dict_drug_cell.setdefault(cell, []).append((drug, ic50))

        # 将药物分为测试集和训练+验证集
        all_drugs = list(drug_dict.keys())
        random.shuffle(all_drugs)

        # 选择测试集药物，这里假设选择20%药物作为测试集
        num_test_drugs = int(len(all_drugs) * 0.2)
        test_drugs = set(all_drugs[:num_test_drugs])

        xd_train, xc_ge_train, y_train = [], [], []
        xd_val, xc_ge_val, y_val = [], [], []
        xd_test, xc_ge_test, y_test = [], [], []
        lstCellTest = []

        # 遍历药物-细胞字典
        for cell, values in dict_drug_cell.items():
            for v in values:
                drug, ic50 = v
                if drug in test_drugs:  # 测试集
                    xd_test.append(drug_smile[drug_dict[drug]])
                    xc_ge_test.append(cell_feature_ge[cell_dict_ge[cell]])
                    y_test.append(ic50)
                    lstCellTest.append(cell)
                else:  # 训练集或验证集
                    if random.random() < 0.8:  # 80%放入训练集
                        xd_train.append(drug_smile[drug_dict[drug]])
                        xc_ge_train.append(cell_feature_ge[cell_dict_ge[cell]])
                        y_train.append(ic50)
                    else:  # 20%放入验证集
                        xd_val.append(drug_smile[drug_dict[drug]])
                        xc_ge_val.append(cell_feature_ge[cell_dict_ge[cell]])
                        y_val.append(ic50)

        # 数据保存
        with open('cell_blind_test', 'wb') as fp:
            pickle.dump(lstCellTest, fp)

        print('Training samples:', len(y_train), 'Validation samples:', len(y_val), 'Testing samples:', len(y_test))

        # 转换为Numpy数组
        xd_train, xc_train, y_train = map(np.asarray, [xd_train, xc_ge_train, y_train])
        xd_val, xc_val, y_val = map(np.asarray, [xd_val, xc_ge_val, y_val])
        xd_test, xc_test, y_test = map(np.asarray, [xd_test, xc_ge_test, y_test])

        # 创建数据集
        dataset = 'GDSC'
        print('Preparing ', dataset + '_train.pt in PyTorch format!')
        train_data = TestbedDataset(root='mydata', dataset=dataset + '_train_cell_blind', xd=xd_train, xt_ge=xc_train,
                                    y=y_train, smile_graph=smile_graph)
        val_data = TestbedDataset(root='mydata', dataset=dataset + '_val_cell_blind', xd=xd_val, xt_ge=xc_val, y=y_val,
                                  smile_graph=smile_graph)
        test_data = TestbedDataset(root='mydata', dataset=dataset + '_test_cell_blind', xd=xd_test, xt_ge=xc_test,
                                   y=y_test, smile_graph=smile_graph)

        print('Train data:', train_data)
        print('Validation data:', val_data)
        print('Test data:', test_data)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='prepare dataset to train model')
    args = parser.parse_args()
    # choice = args.choice
    # drug_blind_split_data()
    # random_split_data()
    save_blind_cell_matrix()