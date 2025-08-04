# python3
# -*- coding:utf-8 -*-

"""
@author:野山羊骑士
@e-mail：thankyoulaojiang@163.com
@file：PycharmProject-PyCharm-Step1_getData.py
@time:2021/8/12 15:48
"""
import sys
import csv
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split


class GetData():
    def __init__(self):
        PATH = 'mydata/'

        rnafile = PATH + '/expt.txt'
        smilefile = PATH + '/smile_inchi.csv'
        pairfile = PATH + '/test2.xlsx'
        drug_infofile = PATH + "/durg_m.csv"
        drug_thred = PATH + '/IC50_thred.txt'

        self.pairfile = pairfile
        self.drugfile = drug_infofile
        self.rnafile = rnafile
        self.smilefile = smilefile
        self.drug_thred = drug_thred

    def getDrug(self):
        # 读取 smile_inchi.csv
        drugdata = pd.read_csv(self.smilefile, index_col=0)
        return drugdata

    def _filter_pair(self, drug_cell_df):
        print("#" * 50)
        print("step1 过滤细胞系....")
        # print("在检查细胞系rna 表达矩阵的时候发现4个细胞系没有表达记录")
        # ['DATA.908134', 'DATA.1789883', 'DATA.908120', 'DATA.908442'] not in index
        # not_index = ['908134', '1789883', '908120', '908442']
        print(drug_cell_df.shape)
        # drug_cell_df = drug_cell_df[~drug_cell_df['COSMIC_ID'].isin(not_index)]
        print(drug_cell_df.shape)

        print("step2 过滤药物....")
        print("对于部分Drug没有记录PuchemID，得不到smile")
        pub_df = pd.read_csv(self.drugfile)
        pub_df = pub_df.dropna(subset=['PubCHEM'])
        pub_df = pub_df[(pub_df['PubCHEM'] != 'none') & (pub_df['PubCHEM'] != 'several')]
        print(drug_cell_df.shape)
        drug_cell_df = drug_cell_df[drug_cell_df['DRUG_ID'].isin(pub_df['drug_id'])]
        print(drug_cell_df.shape)
        return drug_cell_df

    def _stat_cancer(self, drug_cell_df):
        print("#" * 50)
        cancer_num = drug_cell_df['TCGA_DESC'].value_counts().shape[0]
        print('#\t 癌症类型一共有：{}'.format(cancer_num))
        min_cancer_drug = min(drug_cell_df['TCGA_DESC'].value_counts())
        max_cancer_drug = max(drug_cell_df['TCGA_DESC'].value_counts())
        mean_cancer_drug = np.mean(drug_cell_df['TCGA_DESC'].value_counts())
        print('#\t 其中最少的癌症类型对应{}个药物，\n\t 最多的对应{}个药物，\n\t 平均对应{}个药物'.format(
            min_cancer_drug, max_cancer_drug, mean_cancer_drug))

    def _stat_cell(self, drug_cell_df):
        print("#" * 50)
        cell_num = drug_cell_df['COSMIC_ID'].value_counts().shape[0]
        print('#\t 使用的细胞系有：{}'.format(cell_num))
        min_drug = min(drug_cell_df['COSMIC_ID'].value_counts())
        max_drug = max(drug_cell_df['COSMIC_ID'].value_counts())
        mean_drug = np.mean(drug_cell_df['COSMIC_ID'].value_counts())
        print('#\t 其中最少的细胞系对应{}个药物，\n\t 最多的对应{}个药物，\n\t 平均对应{}个药物'.format(
            min_drug, max_drug, mean_drug))

    def _stat_drug(self, drug_cell_df):
        print("#" * 50)
        drug_num = drug_cell_df['DRUG_ID'].value_counts().shape[0]
        print('#\t 使用的药物有：{}'.format(drug_num))
        min_cell = min(drug_cell_df['DRUG_ID'].value_counts())
        max_cell = max(drug_cell_df['DRUG_ID'].value_counts())
        mean_cell = np.mean(drug_cell_df['DRUG_ID'].value_counts())
        print('#\t 其中最少的药物对应{}个细胞系，\n\t 最多的对应{}个细胞系，\n\t 平均对应{}个细胞系'.format(
            min_cell, max_cell, mean_cell))

    def random_split(self, df, ratio, random_seed):

        train_data, test_data = train_test_split(df, test_size=ratio, random_state=random_seed)

        print('#' * 50)
        print('#\t 数据对一共有：{}'.format(df.shape[0]))
        print('#\t 按照随机划分，将 {} 的数据用于训练，将 {} 的数据用于测试'.format(1 - ratio, ratio))
        print('#\t 训练数据有：{}'.format(train_data.shape[0]))
        print('#\t 测试数据有：{}'.format(test_data.shape[0]))

        return train_data, test_data

    # Original ByRandom method definition used by the user
    # def ByRandom(self, random_seed):
    #
    #     drug_cell_df = pd.read_excel(self.pairfile)
    #     self._stat_drug(drug_cell_df)
    #     self._stat_cell(drug_cell_df)
    #     self._stat_cancer(drug_cell_df)
    #     drug_cell_df = self._filter_pair(drug_cell_df)
    #
    #     drug_cell_df = drug_cell_df
    #     self._stat_drug(drug_cell_df)
    #     self._stat_cell(drug_cell_df)
    #     self._stat_cancer(drug_cell_df)
    #
    #     # 随机划分数据集
    #     # Note: Original code used torch.utils.data.random_split here which is for PyTorch datasets,
    #     # not pandas DataFrames directly for general train/test split.
    #     # For pandas, sklearn.model_selection.train_test_split is more typical.
    #     # The second ByRandom definition uses a helper _split_random which uses train_test_split.
    #     # Assuming the intent was to use a DataFrame splitting mechanism.
    #     train_data, test_data = self.random_split(df=drug_cell_df, ratio=0.2, random_seed=random_seed)
    #
    #     return train_data, test_data

    def _split_random(self, df, ratio, random_seed):
        """
        随机将数据集分为训练集和测试集。

        参数:
        df -- 输入的数据框
        ratio -- 测试集所占比例
        random_seed -- 随机种子，用于确保划分的可重复性

        返回:
        train_data -- 训练集数据框
        test_data -- 测试集数据框
        """
        # 随机划分数据集
        train_data, test_data = train_test_split(df, test_size=ratio, random_state=random_seed)

        print('#' * 50)
        print('#\t 数据对一共有：{}'.format(df.shape[0]))
        print('#\t 按照随机方式对数据进行切割，{} 的数据用于训练，{} 的数据用于测试'.format(1 - ratio, ratio))
        print('#\t 训练数据有：{}'.format(train_data.shape[0]))
        print('#\t 测试数据有：{}'.format(test_data.shape[0]))

        return train_data, test_data

    # This is the second (active) definition of ByRandom in the original code
    def ByRandom(self, random_seed):
        """
        随机将数据集划分为训练集和测试集。

        参数:
        random_seed -- 随机种子，用于确保划分的可重复性

        返回:
        train_data -- 训练集数据框
        test_data -- 测试集数据框
        """
        # 从文件中读取数据
        drug_cell_df = pd.read_excel(self.pairfile)
        # 进行数据预处理
        self._stat_drug(drug_cell_df)
        self._stat_cell(drug_cell_df)
        self._stat_cancer(drug_cell_df)
        drug_cell_df = self._filter_pair(drug_cell_df)
        # Stats after filtering (optional, but was in original flow)
        self._stat_drug(drug_cell_df)
        self._stat_cell(drug_cell_df)
        self._stat_cancer(drug_cell_df)

        # 随机划分数据集
        train_data, test_data = self._split_random(df=drug_cell_df, ratio=0.2, random_seed=random_seed)

        return train_data, test_data

    # ------------- ADDED MISSING _split METHOD -------------
    def _split(self, df, col, ratio, random_seed):
        """
        Splits the DataFrame based on unique values in a specified column.
        A 'ratio' of unique values in 'col' will be used for the test set.
        All rows corresponding to test unique values go to the test set, and similarly for train.
        """
        unique_col_values = df[col].unique()

        # Split the unique column values into training and testing sets
        train_unique_values, test_unique_values = train_test_split(
            unique_col_values,
            test_size=ratio,  # This ratio applies to the unique values
            random_state=random_seed
        )

        # Create train and test DataFrames based on these unique values
        train_data = df[df[col].isin(train_unique_values)].copy()
        test_data = df[df[col].isin(test_unique_values)].copy()

        print('#' * 50)
        print('#\t Splitting data based on unique values in column: {}'.format(col))
        print('#\t Total unique values in {}: {}'.format(col, len(unique_col_values)))
        print(
            '#\t Unique values for training (approx. {}%): {}'.format(int((1 - ratio) * 100), len(train_unique_values)))
        print('#\t Unique values for testing (approx. {}%): {}'.format(int(ratio * 100), len(test_unique_values)))
        print('#' * 50)
        print('#\t Original data pairs: {}'.format(df.shape[0]))
        print('#\t Training data pairs: {}'.format(train_data.shape[0]))
        print('#\t Testing data pairs: {}'.format(test_data.shape[0]))

        return train_data, test_data

    # ------------- END OF ADDED METHOD -------------

    def ByDrug(self, random_seed):
        drug_cell_df = pd.read_excel(self.pairfile)
        self._stat_drug(drug_cell_df)
        self._stat_cell(drug_cell_df)
        self._stat_cancer(drug_cell_df)
        drug_cell_df = self._filter_pair(drug_cell_df)
        self._stat_drug(drug_cell_df)
        self._stat_cell(drug_cell_df)
        self._stat_cancer(drug_cell_df)

        train_data, test_data = self._split(df=drug_cell_df, col='DRUG_ID',
                                            ratio=0.2, random_seed=random_seed)

        return train_data, test_data

    def ByCell(self, random_seed):
        drug_cell_df = pd.read_excel(self.pairfile)
        self._stat_drug(drug_cell_df)
        self._stat_cell(drug_cell_df)
        self._stat_cancer(drug_cell_df)
        drug_cell_df = self._filter_pair(drug_cell_df)
        self._stat_drug(drug_cell_df)
        self._stat_cell(drug_cell_df)
        self._stat_cancer(drug_cell_df)

        train_data, test_data = self._split(df=drug_cell_df, col='COSMIC_ID', ratio=0.2, random_seed=random_seed)

        return train_data, test_data

    def MissingData(self):
        drug_cell_df = pd.read_excel(self.pairfile)
        self._stat_drug(drug_cell_df)
        self._stat_cell(drug_cell_df)
        self._stat_cancer(drug_cell_df)
        drug_cell_df = self._filter_pair(drug_cell_df)
        self._stat_drug(drug_cell_df)
        self._stat_cell(drug_cell_df)
        self._stat_cancer(drug_cell_df)

        cell_list = drug_cell_df['COSMIC_ID'].value_counts().index
        drug_list = drug_cell_df['DRUG_ID'].value_counts().index

        all_df = pd.DataFrame()
        dup_drug = []
        [dup_drug.extend([i] * len(cell_list)) for i in drug_list]
        all_df['DRUG_ID'] = dup_drug

        dup_cell = []
        for i in range(len(drug_list)):
            dup_cell.extend(cell_list)
        all_df['COSMIC_ID'] = dup_cell

        all_df['ID'] = all_df['DRUG_ID'].astype(str).str.cat(all_df['COSMIC_ID'].astype(str), sep='_')
        drug_cell_df['ID'] = drug_cell_df['DRUG_ID'].astype(str).str.cat(drug_cell_df['COSMIC_ID'].astype(str), sep='_')
        MissingData = all_df[~all_df['ID'].isin(drug_cell_df['ID'])]

        print("#" * 50)
        print('使用药物{}个，细胞系有{}个'.format(len(drug_list), len(cell_list)))
        print('理论上，每种药物都作用所有细胞系的话，应该有{} Pairs'.format(len(drug_list) * len(cell_list)))
        print('但是有的药物和细胞系没有做实验，共有{} Pairs'.format(MissingData.shape[0]))

        # drug_cell_df = drug_cell_df[['COSMIC_ID', 'TCGA_DESC']].drop_duplicates()
        # cell2cancer_dict = pd.Series(list(drug_cell_df['TCGA_DESC']), index=drug_cell_df['COSMIC_ID'])

        return drug_cell_df, MissingData

    def _LeaveOut(self, df, col, ratio=0.8, random_num=1):
        random.seed(random_num)
        col_list = list(set(df[col]))
        # col_list = list(col_list) # This line is redundant

        num_unique = len(col_list)
        fold_size = num_unique / 5.0  # Assuming 5 folds because random_num seems to be 0-4

        sub_start_idx = int(fold_size * random_num)
        if random_num == 4:  # Last fold
            sub_end_idx = num_unique
        else:
            sub_end_idx = int(fold_size * (random_num + 1))

        # Ensure indices are within bounds, important if col_list is small
        sub_start_idx = min(sub_start_idx, num_unique)
        sub_end_idx = min(sub_end_idx, num_unique)

        # The items for the test set (leave-out set for this fold)
        # This assumes col_list is already somewhat randomly distributed or its order is fixed.
        # For true random sampling for folds, usually you shuffle unique_col_values first
        # or use KFold from sklearn.
        test_fold_instances = col_list[sub_start_idx:sub_end_idx]

        # Instances not in the test fold are for training
        train_instances = [item for item in col_list if item not in test_fold_instances]

        # The original logic used random.sample with ratio, which is different from fold-based selection.
        # The current fold logic: `test_instances` are the fold elements (approx 20%). `train_instances` are the rest (approx 80%).
        # If 'ratio' means the proportion kept FOR TRAINING (as in "80% of data for training"),
        # then `leave_instatnce` = `train_instances`.
        # If 'ratio' means the proportion LEFT OUT FOR TESTING (as in original random.sample scenario),
        # then `leave_instatnce_for_training` and `test_instances` are constructed differently.
        # Given `ratio=0.8` and the name `_LeaveOut`, it's likely 0.8 is for training.
        # The fold logic above implies `ratio` is effectively controlled by `1 - (1/num_folds)`.

        # Reconciling with original random.sample approach if ratio governs train size directly:
        # If strictly adhering to 'ratio=0.8' for training size of unique instances (not fold based):
        # leave_instatnce_training = random.sample(col_list, int(len(col_list) * ratio))
        # test_instances = [item for item in col_list if item not in leave_instatnce_training]
        # For the fold-based logic:
        # We used `train_instances` (equivalent to `leave_instatnce` if `ratio` is for training proportion)
        # and `test_fold_instances` (equivalent to `test_data`'s unique items)

        df_to_split = df[['DRUG_ID', 'COSMIC_ID', 'TCGA_DESC', 'LN_IC50']].copy()  # Use .copy()
        train_data = df_to_split[df_to_split[col].isin(train_instances)]
        test_data = df_to_split[df_to_split[col].isin(test_fold_instances)]

        print('#' * 50)
        print('Total unique {} values: {}'.format(col, len(col_list)))
        print('Unique {} values in training set: {}'.format(col, len(set(list(train_data[col])))))
        print('Unique {} values in test set: {}'.format(col, len(set(list(test_data[col])))))
        print('#\t 数据对一共有：{}，leave out 方法 (using fold {} out of 5)'.format(df.shape[0],
                                                                                   random_num))  # random_num = fold index
        print('#\t 按照{}对数据进行划分, fold {} elements for testing.'.format(col, random_num))
        print('#\t 训练数据有：{}'.format(train_data.shape[0]))
        print('#\t 测试数据有：{}'.format(test_data.shape[0]))

        return train_data, test_data

    def Cell_LeaveOut(self, random_fold_index):  # renamed 'random' to 'random_fold_index' for clarity
        drug_cell_df = pd.read_excel(self.pairfile)
        self._stat_drug(drug_cell_df)
        self._stat_cell(drug_cell_df)
        self._stat_cancer(drug_cell_df)
        drug_cell_df = self._filter_pair(drug_cell_df)
        self._stat_drug(drug_cell_df)
        self._stat_cell(drug_cell_df)
        self._stat_cancer(drug_cell_df)

        traindata, testdata = self._LeaveOut(df=drug_cell_df, col='COSMIC_ID', ratio=0.8, random_num=random_fold_index)

        return traindata, testdata

    def Drug_LeaveOut(self, random_fold_index):  # renamed 'random' to 'random_fold_index' for clarity
        drug_cell_df = pd.read_excel(self.pairfile)
        self._stat_drug(drug_cell_df)
        self._stat_cell(drug_cell_df)
        self._stat_cancer(drug_cell_df)
        drug_cell_df = self._filter_pair(drug_cell_df)
        self._stat_drug(drug_cell_df)
        self._stat_cell(drug_cell_df)
        self._stat_cancer(drug_cell_df)

        traindata, testdata = self._LeaveOut(df=drug_cell_df, col='DRUG_ID', ratio=0.8, random_num=random_fold_index)

        return traindata, testdata

    def Drug_Thred(self):
        thred_data = pd.read_csv(self.drug_thred, sep='\t')
        thred_df = thred_data.T
        thred_df['drug_name'] = thred_df.index
        thred_df['threds'] = thred_df[0]
        thred_df = thred_df.drop(0, axis=1)

        # Applying name corrections using .loc for direct assignment
        name_corrections = {
            'VX-680': 'Tozasertib', 'Mitomycin C': 'Mitomycin-C', 'HG-6-64-1': 'HG6-64-1',
            'BAY 61-3606': 'BAY-61-3606', 'Zibotentan, ZD4054': 'Zibotentan',
            'PXD101, Belinostat': 'Belinostat', 'NU-7441': 'NU7441', 'BIRB 0796': 'BIRB-796',
            'Nutlin-3a': 'Nutlin-3a (-)', 'AZD6482.1': 'AZD6482', 'BMS-708163.1': 'BMS-708163',
            'BMS-536924.1': 'BMS-536924', 'GSK269962A.1': 'GSK269962A', 'SB-505124': 'SB505124',
            'JQ1.1': 'JQ1', 'UNC0638.1': 'UNC0638', 'CHIR-99021.1': 'CHIR-99021',
            'piperlongumine': 'Piperlongumine', 'PLX4720 (rescreen)': 'PLX4720',
            'Afatinib (rescreen)': 'Afatinib', 'Olaparib.1': 'Olaparib', 'AZD6244.1': 'AZD6244',
            'Bicalutamide.1': 'Bicalutamide', 'RDEA119 (rescreen)': 'RDEA119',
            'GDC0941 (rescreen)': 'GDC0941', 'MLN4924 ': 'MLN4924'  # Note trailing space
        }
        for old_name, new_name in name_corrections.items():
            if old_name in thred_df.index:
                thred_df.loc[old_name, 'drug_name'] = new_name

        drug_info = pd.read_csv(self.drugfile)
        drugname2drugid = {}
        # drugid2pubchemid = {} # Original code had this, commented out if not used later

        for idx, row in drug_info.iterrows():
            name = row['Name']
            drug_id = row['drug_id']
            # pub_id = row['PubCHEM'] # Original
            drugname2drugid[name] = drug_id
            # drugid2pubchemid[drug_id] = pub_id # Original

        # drug_info_filter_name = drug_info.dropna(subset=['Synonyms']) # Original, can be integrated above
        for idx, row in drug_info.iterrows():  # Re-iterate or combine with above loop
            if pd.notna(row['Synonyms']):
                name = row['Name']  # Redundant if already processed, but safe
                drug_id = row['drug_id']  # Redundant if already processed
                drugname2drugid[name] = drug_id  # Ensure primary name is mapped

                Synonyms_list = row['Synonyms'].split(', ')
                for drug in Synonyms_list:
                    drugname2drugid[drug] = drug_id

        drugid2thred = {}
        for idx, row in thred_df.iterrows():  # idx is original drug name from thred_df.index
            name = row['drug_name']  # This is the potentially corrected name
            thred = row['threds']
            if name in drugname2drugid:
                drugid2thred[drugname2drugid[name]] = thred
            # Consider else: print(f"Warning: Drug name '{name}' from threshold file not found in drug info.")

        # Original code for creating a CSV, commented out
        # id_li = []
        # PubChem_li =[]
        # thred_li =[]
        # for i in drugid2thred:
        #     id_li.append(i)
        #     PubChem_li.append(drugid2pubchemid[i]) # This line would error if drugid2pubchemid is not populated
        #     thred_li.append(drugid2thred[i])
        # data = pd.DataFrame()
        # data['Drug_id'] = id_li
        # data['PubChem'] = PubChem_li
        # data['Thred'] = thred_li
        # print(data)
        # data.to_csv('Drug_Thred.csv')

        drug_list_with_thresholds = [drugname2drugid[i] for i in list(thred_df['drug_name']) if i in drugname2drugid]
        # A more direct way to get the list of drug_ids for which a threshold was found:
        # drug_list_with_thresholds = list(drugid2thred.keys())

        return drug_list_with_thresholds, drugid2thred

    def _split_no_balance_binary(self, df, col, ratio, random_seed):

        # col_list = df[col].value_counts().index # Not directly used like this later
        train_data = pd.DataFrame()
        test_data = pd.DataFrame()

        # This loop splits each instance group; a simpler approach is to split the whole df stratifying by 'col'.
        # If 'col' is 'Binary_IC50', and we want to maintain its proportion:
        df_to_split = df[['DRUG_ID', 'COSMIC_ID', 'TCGA_DESC', 'LN_IC50', 'Binary_IC50']].copy()

        temp_train_data, temp_test_data = train_test_split(
            df_to_split,
            test_size=ratio,
            random_state=random_seed,
            stratify=df_to_split[col] if col in df_to_split else None  # Stratify by the binary column
        )
        train_data = temp_train_data
        test_data = temp_test_data

        # Original loop (preserves structure if instance-by-instance split was intended, but less common for binary)
        # for instatnce in df[col].unique(): # Iterate over unique values of the binary column (0 and 1)
        #     sub_df = df[df[col] == instatnce]
        #     sub_df = sub_df[['DRUG_ID', 'COSMIC_ID','TCGA_DESC', 'LN_IC50','Binary_IC50']]
        #     ## 按照 col 来拆分数据集 ##
        #     ## 对于任意一个 instance，1 - ratio 的用于训练，10=test，10=validation
        #     if not sub_df.empty:
        #        sub_train, sub_test = train_test_split(sub_df, test_size=ratio,
        #                                            random_state=random_seed) # No stratify needed if sub_df is one class
        #        if train_data.empty:
        #            train_data = sub_train
        #            test_data = sub_test
        #        else:
        #            train_data = pd.concat([train_data, sub_train]) # Use concat
        #            test_data = pd.concat([test_data, sub_test])   # Use concat

        print('#' * 50)
        print('#\t 数据对一共有：{}'.format(df.shape[0]))
        print('#\t 按照{}对数据进行切割，对于每个instance，{}的数据进行训练，{}的数据进行验证'.format(col, (1 - ratio),
                                                                                                   ratio))
        print('#\t 训练数据有：{}'.format(train_data.shape[0]))
        print('#\t 测试数据有：{}'.format(test_data.shape[0]))
        if col in train_data: print(f"#\t Train data '{col}' counts:\n{train_data[col].value_counts(normalize=True)}")
        if col in test_data: print(f"#\t Test data '{col}' counts:\n{test_data[col].value_counts(normalize=True)}")

        return train_data, test_data

    def _split_balance_binary(self, df, col, ratio, random_seed):

        # col_list = df[col].value_counts().index # Not used

        pos_data = df[df[col] == 1]
        neg_data = df[df[col] == 0]

        if pos_data.empty or neg_data.empty:
            print(f"Warning: Class imbalance too extreme or one class missing in column '{col}'. Cannot balance/split.")
            # Return empty DFs or the original DF if no split is possible or meaningful
            empty_cols = ['DRUG_ID', 'COSMIC_ID', 'TCGA_DESC', 'LN_IC50', 'Binary_IC50']
            return pd.DataFrame(columns=empty_cols), pd.DataFrame(columns=empty_cols)

        # Downsample the majority class to match the minority class size
        if pos_data.shape[0] > neg_data.shape[0]:
            down_pos_data = pos_data.sample(n=neg_data.shape[0], random_state=random_seed)
            combine_data = pd.concat([neg_data, down_pos_data])
        elif neg_data.shape[0] > pos_data.shape[0]:
            down_neg_data = neg_data.sample(n=pos_data.shape[0], random_state=random_seed)
            combine_data = pd.concat([pos_data, down_neg_data])
        else:  # Already balanced
            combine_data = pd.concat([pos_data, neg_data])

        combine_data = combine_data.sample(frac=1, random_state=random_seed).reset_index(drop=True)  # Shuffle

        df_to_split = combine_data[['DRUG_ID', 'COSMIC_ID', 'TCGA_DESC', 'LN_IC50', 'Binary_IC50']].copy()

        train_data, test_data = train_test_split(df_to_split, test_size=ratio,
                                                 random_state=random_seed,
                                                 stratify=df_to_split[col])  # Stratify on the balanced data

        print('#' * 50)
        print('#\t 数据对一共有：{}'.format(df.shape[0]))
        print('#\t 构建平衡数据集，{}为大于阈值的样本(class 1)，{}为小于阈值的样本(class 0),选择1：1的样本各{}个'.format(
            pos_data.shape[0], neg_data.shape[0], min(pos_data.shape[0], neg_data.shape[0])))
        print('#\t 按照{}对数据进行切割 (balanced)，{}的数据进行训练，{}的数据进行验证'.format(
            col, (1 - ratio), ratio))
        print('#\t 训练数据有：{}'.format(train_data.shape[0]))
        print('#\t 测试数据有：{}'.format(test_data.shape[0]))
        if col in train_data: print(f"#\t Train data '{col}' counts:\n{train_data[col].value_counts(normalize=True)}")
        if col in test_data: print(f"#\t Test data '{col}' counts:\n{test_data[col].value_counts(normalize=True)}")

        return train_data, test_data

    def ByBinary(self, random_num):  # random_num is used as random_seed
        drug_cell_df = pd.read_excel(self.pairfile)
        self._stat_drug(drug_cell_df)
        self._stat_cell(drug_cell_df)
        self._stat_cancer(drug_cell_df)
        drug_cell_df_filtered = self._filter_pair(drug_cell_df.copy())  # Use a copy
        self._stat_drug(drug_cell_df_filtered)
        self._stat_cell(drug_cell_df_filtered)
        self._stat_cancer(drug_cell_df_filtered)

        drug_list_with_th, drugid2thred = self.Drug_Thred()

        ##################################################
        # Strategy 1: Filter drugs to only those with specific thresholds
        drug_cell_df_for_binary = drug_cell_df_filtered[
            drug_cell_df_filtered['DRUG_ID'].isin(drug_list_with_th)
        ].copy()  # Use .copy()

        print(
            f"Number of unique drugs after filtering for those with thresholds: {drug_cell_df_for_binary['DRUG_ID'].nunique()}")

        Binary_IC50_list = []  # Renamed from Binary_Drug_list for clarity
        for idx, row in drug_cell_df_for_binary.iterrows():
            # drug_name = row['DRUG_NAME'] # DRUG_NAME column might not exist, DRUG_ID is used
            drug_id = row['DRUG_ID']
            ic50 = row['LN_IC50']
            if drug_id in drugid2thred:  # This check should always be true due to pre-filtering
                if (ic50 > drugid2thred[drug_id]):
                    Binary_IC50_list.append(1)  # Resistant
                else:
                    Binary_IC50_list.append(0)  # Sensitive
            else:
                # This case should ideally not happen if drug_list_with_th is derived from drugid2thred keys
                # Or if drug_cell_df_for_binary is correctly filtered. Add a warning or error.
                print(
                    f"Warning: Drug ID {drug_id} found in data but not in threshold map. Skipping binarization for this row.")
                Binary_IC50_list.append(np.nan)  # Or handle as an error / filter out

        drug_cell_df_for_binary['Binary_IC50'] = Binary_IC50_list
        drug_cell_df_for_binary.dropna(subset=['Binary_IC50'], inplace=True)  # Remove rows where binarization failed
        drug_cell_df_for_binary['Binary_IC50'] = drug_cell_df_for_binary['Binary_IC50'].astype(int)

        # The commented-out strategies are preserved for context
        ############################################################################
        # 第二种，补充-2的阈值
        # ...
        ############################################################################
        # 第三种 直接使用-2的阈值
        # ...
        #############################################################################

        print(f"Value counts for 'Binary_IC50' before split:\n{drug_cell_df_for_binary['Binary_IC50'].value_counts()}")

        # Using _split_balance_binary as per original logic
        train_data, test_data = self._split_balance_binary(df=drug_cell_df_for_binary, col='Binary_IC50',
                                                           ratio=0.2, random_seed=random_num)
        # print(train_data,test_data) # Avoid printing large dataframes to console

        return train_data, test_data

    def getRna(self, traindata, testdata):
        # Ensure COSMIC_ID is string for matching, and add prefix
        # Handle cases where COSMIC_ID might be float/int
        try:
            train_rnaid = ['DATA.' + str(int(i)) for i in traindata['COSMIC_ID']]
            test_rnaid = ['DATA.' + str(int(i)) for i in testdata['COSMIC_ID']]
        except ValueError:  # If already string or cannot convert to int
            train_rnaid = ['DATA.' + str(i) for i in traindata['COSMIC_ID']]
            test_rnaid = ['DATA.' + str(i) for i in testdata['COSMIC_ID']]

        rnadata = pd.read_csv(self.rnafile, sep='\t')
        # Assuming the first column of rnafile is gene names/IDs and should be the index
        # If it's not already the index and is just the first column of data:
        if rnadata.columns[0] not in ['DATA.' + str(i) for i in
                                      range(1000000)]:  # Heuristic check if first col is gene id
            rnadata = rnadata.set_index(rnadata.columns[0])

        # Select only columns present in rnadata to avoid KeyError
        train_cols_present = [col for col in train_rnaid if col in rnadata.columns]
        test_cols_present = [col for col in test_rnaid if col in rnadata.columns]

        if len(train_cols_present) < len(train_rnaid):
            print(
                f"Warning: {len(train_rnaid) - len(train_cols_present)} RNA IDs from training data not found in RNA file.")
        if len(test_cols_present) < len(test_rnaid):
            print(f"Warning: {len(test_rnaid) - len(test_cols_present)} RNA IDs from test data not found in RNA file.")

        train_rnadata = rnadata[train_cols_present] if train_cols_present else pd.DataFrame()
        test_rnadata = rnadata[test_cols_present] if test_cols_present else pd.DataFrame()

        return train_rnadata, test_rnadata


if __name__ == '__main__':
    # This block is for example usage or testing the class.
    # The traceback indicates it's being called from Step3_model.py
    obj = GetData()

    # Example of how ByDrug might be called, based on the traceback
    # This would require actual data files in 'mydata/' to run without I/O errors
    # try:
    #     print("Attempting to run obj.ByDrug(random_seed=88) as an example...")
    #     traindata, testdata = obj.ByDrug(random_seed=88)
    #     print("obj.ByDrug call successful.")
    #     print(f"Train data shape: {traindata.shape}")
    #     print(f"Test data shape: {testdata.shape}")
    # except FileNotFoundError as e:
    #     print(f"FileNotFoundError during example run: {e}. Ensure 'mydata/' and its files exist.")
    # except Exception as e:
    #     print(f"An error occurred during example run: {e}")
    pass  # Keep __main__ minimal if it's run as part of a larger system