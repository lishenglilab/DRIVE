import argparse
import random, os, sys
from collections import defaultdict

import numpy as np
import csv
from scipy import stats
import time
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error
from sklearn import preprocessing
import pandas as pd
import keras.backend as K
from keras.models import Model, Sequential
from keras.models import load_model  # ****** ADDED for loading model ******
from keras.layers import Input, InputLayer, Multiply, ZeroPadding2D
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Activation, Dropout, Flatten, Concatenate
from keras.layers import BatchNormalization
from keras.layers import Lambda
from keras import optimizers, utils
from keras.constraints import max_norm
from keras import regularizers
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, History
from keras.utils import multi_gpu_model, plot_model
from keras.optimizers import Adam, SGD
from keras.models import model_from_json
import tensorflow as tf
from sklearn.metrics import average_precision_score
from scipy.stats import pearsonr, spearmanr
from model import KerasMultiSourceGCNModel  # Assuming model.py is in the same directory or accessible
import hickle as hkl
import scipy.sparse as sp

# import argparse # Already imported

####################################Settings#################################
parser = argparse.ArgumentParser(description='Drug_response_pre')
parser.add_argument('-gpu_id', dest='gpu_id', type=str, default='cpu', help='GPU devices')
parser.add_argument('-use_mut', dest='use_mut', type=bool, default=True, help='use gene mutation or not')
parser.add_argument('-use_gexp', dest='use_gexp', type=bool, default=True, help='use gene expression or not')
parser.add_argument('-use_methy', dest='use_methy', type=bool, default=True, help='use methylation or not')

parser.add_argument('-israndom', dest='israndom', type=bool, default=False, help='randomlize X and A')
# hyparameters for GCN
parser.add_argument('-unit_list', dest='unit_list', nargs='+', type=int, default=[256, 256, 256],
                    help='unit list for GCN')
parser.add_argument('-use_bn', dest='use_bn', type=bool, default=True, help='use batchnormalization for GCN')
parser.add_argument('-use_relu', dest='use_relu', type=bool, default=True, help='use relu for GCN')
parser.add_argument('-use_GMP', dest='use_GMP', type=bool, default=False,
                    help='use GlobalMaxPooling for GCN')  # Default to False if GAP is the other option

# ****** ADDED ARGUMENTS for model saving and loading ******
parser.add_argument('--save_model_path', dest='save_model_path', type=str, default='./saved_models/bd2.h5',
                    help='Path to save the best trained model (e.g., ./saved_models/my_model.h5). Uses default if None.')
parser.add_argument('--load_model_path', dest='load_model_path', type=str, default=None,
                    help='Path to a pre-trained model file (.h5) to load for prediction/evaluation.')
# ****** END OF ADDED ARGUMENTS ******

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
use_mut, use_gexp, use_methy = args.use_mut, args.use_gexp, args.use_methy
israndom = args.israndom
model_suffix = ('with_mut' if use_mut else 'without_mut') + '_' + (
    'with_gexp' if use_gexp else 'without_gexp') + '_' + ('with_methy' if use_methy else 'without_methy')

GCN_deploy = '_'.join(map(str, args.unit_list)) + '_' + ('bn' if args.use_bn else 'no_bn') + '_' + (
    'relu' if args.use_relu else 'tanh') + '_' + ('GMP' if args.use_GMP else 'GAP')
model_suffix = model_suffix + '_' + GCN_deploy

# Ensure checkpoint directory exists for default save path
if not os.path.exists('../checkpoint'):
    try:
        os.makedirs('../checkpoint')
        print("Created directory ../checkpoint")
    except OSError as e:
        print(f"Error creating directory ../checkpoint: {e}")
        # Decide how to handle this: exit, or proceed without default saving
        # For now, let's assume it's critical for the default save path
        # sys.exit(1)

####################################Constants Settings###########################
DPATH = '../mydata'  # Make sure this path is correct relative to where you run the script
Drug_info_file = '%s/drug_m.csv' % DPATH
Cell_line_info_file = '%s/Cell_lines_annotations_20181226.txt' % DPATH
Drug_feature_file = '%s/GDSC/drug_graph_feat' % DPATH
Genomic_mutation_file = '%s/CCLE/mu.csv' % DPATH
Cancer_response_exp_file = '%s/CCLE/GDSC_IC50.csv' % DPATH
Gene_expression_file = '%s/CCLE/exp.csv' % DPATH
Methylation_file = '%s/CCLE/mu.csv' % DPATH  # Note: you used mu.csv for Methylation, same as mutation. Is this correct?
Max_atoms = 100


def MetadataGenerate(Drug_info_file, Cell_line_info_file, Genomic_mutation_file, Drug_feature_file,
                     Gene_expression_file, Methylation_file, filtered):
    # drug_id --> pubchem_id
    reader = csv.reader(open(Drug_info_file, 'r'))
    rows = [item for item in reader]
    drugid2pubchemid = {item[0]: item[5] for item in rows if item[5].isdigit()}

    # map cellline --> cancer type
    cellline2cancertype = {}
    for line in open(Cell_line_info_file).readlines()[1:]:
        cellline_id = line.split('\t')[1]
        TCGA_label = line.strip().split('\t')[-1]
        # if TCGA_label in TCGA_label_set:
        cellline2cancertype[cellline_id] = TCGA_label

    # load demap cell lines genomic mutation features
    mutation_feature = pd.read_csv(Genomic_mutation_file, sep=',', header=0, index_col=[0])
    cell_line_id_set = list(mutation_feature.index)

    # load drug features
    drug_pubchem_id_set = []
    drug_feature = {}
    for each in os.listdir(Drug_feature_file):
        drug_pubchem_id_set.append(each.split('.')[0])
        feat_mat, adj_list, degree_list = hkl.load('%s/%s' % (Drug_feature_file, each))
        drug_feature[each.split('.')[0]] = [feat_mat, adj_list, degree_list]
    assert len(drug_pubchem_id_set) == len(drug_feature.values())

    # load gene expression faetures
    gexpr_feature = pd.read_csv(Gene_expression_file, sep=',', header=0, index_col=[0])

    # only keep overlapped cell lines
    common_indices = list(set(gexpr_feature.index).intersection(set(mutation_feature.index)))
    gexpr_feature = gexpr_feature.loc[common_indices]
    mutation_feature = mutation_feature.loc[common_indices]

    # load methylation
    methylation_feature = pd.read_csv(Methylation_file, sep=',', header=0, index_col=[0])
    methylation_feature = methylation_feature.loc[common_indices]  # Also filter methylation by common cell lines

    assert methylation_feature.shape[0] == gexpr_feature.shape[0] == mutation_feature.shape[0]
    experiment_data = pd.read_csv(Cancer_response_exp_file, sep=',', header=0, index_col=[0])
    # filter experiment data
    drug_match_list = [item for item in experiment_data.index if item.split(':')[1] in drugid2pubchemid.keys()]
    experiment_data_filtered = experiment_data.loc[drug_match_list]

    data_idx = []
    for each_drug in experiment_data_filtered.index:
        for each_cellline in experiment_data_filtered.columns:
            pubchem_id = drugid2pubchemid[each_drug.split(':')[-1]]
            if str(pubchem_id) in drug_pubchem_id_set and each_cellline in mutation_feature.index:  # ensure cell line in filtered features
                if not np.isnan(experiment_data_filtered.loc[
                                    each_drug, each_cellline]) and each_cellline in cellline2cancertype.keys():
                    ln_IC50 = float(experiment_data_filtered.loc[each_drug, each_cellline])
                    data_idx.append((each_cellline, pubchem_id, ln_IC50, cellline2cancertype[each_cellline]))
    nb_celllines = len(set([item[0] for item in data_idx]))
    nb_drugs = len(set([item[1] for item in data_idx]))
    print('%d instances across %d cell lines and %d drugs were generated.' % (len(data_idx), nb_celllines, nb_drugs))
    return mutation_feature, drug_feature, gexpr_feature, methylation_feature, data_idx


# split into training and test set
TCGA_label_set = ["ALL", "BLCA", "BRCA", "CESC", "DLBC", "LIHC", "LUAD",
                  "ESCA", "GBM", "HNSC", "KIRC", "LAML", "LCML", "LGG",
                  "LUSC", "MESO", "MM", "NB", "OV", "PAAD", "SCLC", "SKCM",
                  "STAD", "THCA", 'COAD/READ']  # Define this earlier or pass it


def DataSplit(data_idx, ratio=0.8):
    data_train_idx, data_test_idx = [], []
    # Filter TCGA_label_set to include only labels present in data_idx
    present_TCGA_labels = set(item[-1] for item in data_idx)
    filtered_TCGA_label_set = [label for label in TCGA_label_set if label in present_TCGA_labels]

    for each_type in filtered_TCGA_label_set:  # Use filtered set
        data_subtype_idx = [item for item in data_idx if item[-1] == each_type]
        if not data_subtype_idx:  # Skip if no data for this subtype
            continue
        # Ensure sample size is not less than 1 for random.sample
        sample_size = int(ratio * len(data_subtype_idx))
        if sample_size < 1 and len(
                data_subtype_idx) > 0:  # If ratio is too small for small groups, take at least one if possible
            sample_size = 1 if len(data_subtype_idx) > 0 else 0
        if sample_size == 0 and len(
                data_subtype_idx) > 0:  # if still 0, add all to test to avoid error, or handle differently
            test_list = data_subtype_idx
            train_list = []
        elif sample_size == len(data_subtype_idx):  # if sample size is total, all go to train
            train_list = data_subtype_idx
            test_list = []
        else:
            train_list = random.sample(data_subtype_idx, sample_size)
            test_list = [item for item in data_subtype_idx if item not in train_list]

        data_train_idx += train_list
        data_test_idx += test_list
    if not data_train_idx or not data_test_idx:
        print(
            "Warning: Data splitting resulted in an empty training or testing set. Check data distribution and ratios.")
        # Fallback to mixed split if stratified split fails badly
        print("Falling back to mixed data split.")
        return DataSplit_mix(data_idx, ratio)
    return data_train_idx, data_test_idx


def DataSplit_mix(data_idx, ratio=0.8):
    random.shuffle(data_idx)
    split_index = int(ratio * len(data_idx))
    data_train_idx = data_idx[:split_index]
    data_test_idx = data_idx[split_index:]
    return data_train_idx, data_test_idx


def DataSplit_drug(data_idx, ratio=0.8):
    drug_to_samples = defaultdict(list)
    for sample in data_idx:
        _, pubchem_id, _, _ = sample
        drug_to_samples[pubchem_id].append(sample)

    drugs = list(drug_to_samples.keys())
    random.shuffle(drugs)
    num_train_drugs = int(ratio * len(drugs))

    train_drugs = set(drugs[:num_train_drugs])

    data_train_idx = []
    data_test_idx = []
    for drug, samples in drug_to_samples.items():
        if drug in train_drugs:
            data_train_idx.extend(samples)
        else:
            data_test_idx.extend(samples)
    return data_train_idx, data_test_idx


def DataSplit_cell(data_idx, ratio=0.8):
    cellline_to_samples = defaultdict(list)
    for sample in data_idx:
        cellline, _, _, _ = sample
        cellline_to_samples[cellline].append(sample)

    celllines = list(cellline_to_samples.keys())
    random.shuffle(celllines)
    num_train_celllines = int(ratio * len(celllines))
    train_celllines = set(celllines[:num_train_celllines])

    data_train_idx = []
    data_test_idx = []
    for cellline, samples in cellline_to_samples.items():
        if cellline in train_celllines:
            data_train_idx.extend(samples)
        else:
            data_test_idx.extend(samples)
    return data_train_idx, data_test_idx


def NormalizeAdj(adj):
    adj = adj + np.eye(adj.shape[0])
    d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0).toarray()
    a_norm = adj.dot(d).transpose().dot(d)
    return a_norm


def random_adjacency_matrix(n):
    matrix = [[random.randint(0, 1) for i in range(n)] for j in range(n)]
    for i in range(n):
        matrix[i][i] = 0
    for i in range(n):
        for j in range(n):
            matrix[j][i] = matrix[i][j]
    return matrix


def CalculateGraphFeat(feat_mat, adj_list):
    assert feat_mat.shape[0] == len(adj_list)
    feat = np.zeros((Max_atoms, feat_mat.shape[-1]), dtype='float32')
    adj_mat = np.zeros((Max_atoms, Max_atoms), dtype='float32')
    if israndom:  # This global `israndom` might be better passed as an argument
        feat = np.random.rand(Max_atoms, feat_mat.shape[-1])
        adj_mat[feat_mat.shape[0]:, feat_mat.shape[0]:] = random_adjacency_matrix(Max_atoms - feat_mat.shape[0])
    feat[:feat_mat.shape[0], :] = feat_mat
    for i in range(len(adj_list)):
        nodes = adj_list[i]
        for each in nodes:
            adj_mat[i, int(each)] = 1
    assert np.allclose(adj_mat,
                       adj_mat.T)  # This might fail if random_adjacency_matrix introduces non-symmetric parts before symmetrization

    # Normalize connections within existing molecule and within padded part separately
    if len(adj_list) > 0:  # handle case with no atoms in adj_list (empty molecule graph)
        adj_ = adj_mat[:len(adj_list), :len(adj_list)]
        norm_adj_ = NormalizeAdj(adj_)
        adj_mat[:len(adj_list), :len(adj_list)] = norm_adj_

    if Max_atoms - len(adj_list) > 0:  # handle padding
        adj_2 = adj_mat[len(adj_list):, len(adj_list):]
        # Only normalize if there are actual random connections, otherwise NormalizeAdj might fail on all zeros
        if np.any(adj_2):  # if adj_2 is not all zeros
            norm_adj_2 = NormalizeAdj(adj_2)
            adj_mat[len(adj_list):, len(adj_list):] = norm_adj_2
        # else: adj_mat for padding remains zeros, which is fine

    return [feat, adj_mat]


def FeatureExtract(data_idx, drug_feature, mutation_feature, gexpr_feature, methylation_feature):
    cancer_type_list = []
    nb_instance = len(data_idx)
    nb_mutation_feature = mutation_feature.shape[1]
    nb_gexpr_features = gexpr_feature.shape[1]
    nb_methylation_features = methylation_feature.shape[1]
    drug_data = [[] for item in range(nb_instance)]
    mutation_data = np.zeros((nb_instance, 1, nb_mutation_feature, 1), dtype='float32')
    gexpr_data = np.zeros((nb_instance, nb_gexpr_features), dtype='float32')
    methylation_data = np.zeros((nb_instance, nb_methylation_features), dtype='float32')
    target = np.zeros(nb_instance, dtype='float32')
    for idx in range(nb_instance):
        cell_line_id, pubchem_id, ln_IC50, cancer_type = data_idx[idx]
        feat_mat, adj_list, _ = drug_feature[str(pubchem_id)]
        drug_data[idx] = CalculateGraphFeat(feat_mat, adj_list)
        mutation_data[idx, 0, :, 0] = mutation_feature.loc[cell_line_id].values
        gexpr_data[idx, :] = gexpr_feature.loc[cell_line_id].values
        methylation_data[idx, :] = methylation_feature.loc[cell_line_id].values
        target[idx] = ln_IC50
        cancer_type_list.append([cancer_type, cell_line_id, pubchem_id])
    return drug_data, mutation_data, gexpr_data, methylation_data, target, cancer_type_list


class MyCallback(Callback):
    # ****** MODIFIED __init__ and on_train_end ******
    def __init__(self, validation_data, patience, save_model_path=None, model_suffix_for_default=""):
        super(MyCallback, self).__init__()
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        self.best_weight = None
        self.patience = patience
        self.custom_save_path = save_model_path
        self.model_suffix = model_suffix_for_default  # Used for default path

    def on_train_begin(self, logs={}):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = -np.Inf
        return

    def on_train_end(self, logs={}):
        if self.best_weight is not None:  # Ensure best_weight was actually set
            self.model.set_weights(self.best_weight)

            if self.custom_save_path:
                save_path = self.custom_save_path
            else:
                # Default path, ensure model_suffix is available
                save_path = f'../checkpoint/MyBestDeepCDR_{self.model_suffix}.h5'

            # Ensure directory for save_path exists
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
                print(f"Created directory: {save_dir}")

            self.model.save(save_path)
            print(f"Best model saved to {save_path}")
        else:
            print(
                "Warning: No best weights found to save the model (perhaps training was too short or an issue occurred).")

        if self.stopped_epoch > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred_val = self.model.predict(self.x_val)
        if self.y_val.shape[0] > 1:  # Pearsonr needs at least 2 samples
            pcc_val = pearsonr(self.y_val, y_pred_val[:, 0])[0]
            print('pcc-val: %s' % str(round(pcc_val, 4)))
            if pcc_val > self.best:
                self.best = pcc_val
                self.wait = 0
                self.best_weight = self.model.get_weights()
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
        else:
            print("Skipping PCC calculation for validation: not enough samples.")
        return
    # ****** END OF MODIFICATIONS ******


# ****** MODIFIED ModelTraining ******
def ModelTraining(model, X_drug_data_train, X_mutation_data_train, X_gexpr_data_train, X_methylation_data_train,
                  Y_train, validation_data, nb_epoch=100, save_model_path_arg=None, current_model_suffix=""):
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])

    # We'll use MyCallback to save the best model, so ModelCheckpoint can be removed if it's redundant
    # If you want ModelCheckpoint for other reasons (e.g. saving every N epochs), keep it.
    # For now, assuming MyCallback handles the "best model" saving.

    # callbacks = [ModelCheckpoint(f'../checkpoint/best_DeepCDR_{current_model_suffix}.h5', monitor='val_loss', save_best_only=True, save_weights_only=False), # Default Keras Checkpoint
    #             MyCallback(validation_data=validation_data, patience=10, save_model_path=save_model_path_arg, model_suffix_for_default=current_model_suffix)]

    callbacks = [MyCallback(validation_data=validation_data,
                            patience=10,  # Early stopping patience
                            save_model_path=save_model_path_arg,
                            model_suffix_for_default=current_model_suffix)
                 ]

    X_drug_feat_data_train = [item[0] for item in X_drug_data_train]
    X_drug_adj_data_train = [item[1] for item in X_drug_data_train]
    X_drug_feat_data_train = np.array(X_drug_feat_data_train)
    X_drug_adj_data_train = np.array(X_drug_adj_data_train)

    model.fit(x=[X_drug_feat_data_train, X_drug_adj_data_train, X_mutation_data_train, X_gexpr_data_train,
                 X_methylation_data_train],
              y=Y_train,
              batch_size=64,  # Consider making batch_size an argparse argument
              epochs=nb_epoch,
              validation_data=validation_data,
              # Provide validation data directly to fit for Keras internal val_loss/val_mse
              callbacks=callbacks,
              verbose=1)  # verbose=1 for progress bar, 2 for one line per epoch
    return model


# ****** END OF MODIFICATIONS ******


def ModelEvaluate(model, X_drug_data_test, X_mutation_data_test, X_gexpr_data_test, X_methylation_data_test, Y_test,
                  cancer_type_test_list):  # cancer_type_test_list is not used in this function
    X_drug_feat_data_test = [item[0] for item in X_drug_data_test]
    X_drug_adj_data_test = [item[1] for item in X_drug_data_test]
    X_drug_feat_data_test = np.array(X_drug_feat_data_test)
    X_drug_adj_data_test = np.array(X_drug_adj_data_test)

    Y_pred = model.predict(
        [X_drug_feat_data_test, X_drug_adj_data_test, X_mutation_data_test, X_gexpr_data_test, X_methylation_data_test])

    if Y_test.shape[0] > 1:  # Ensure enough samples for correlation
        overall_pcc = pearsonr(Y_pred[:, 0], Y_test)[0]
        overall_spearman, _ = spearmanr(Y_pred[:, 0], Y_test)
    else:
        overall_pcc = np.nan
        overall_spearman = np.nan
        print("Not enough test samples to calculate Pearson/Spearman correlation.")

    r2 = r2_score(Y_test, Y_pred[:, 0])
    rmse = np.sqrt(mean_squared_error(Y_test, Y_pred[:, 0]))

    print(f"The overall Pearson's correlation is {overall_pcc:.4f}.")
    print(f"The overall Spearman's correlation is {overall_spearman:.4f}.")
    print(f"The overall R^2 score is {r2:.4f}.")
    print(f"The overall RMSE is {rmse:.4f}.")

    results_df_path = "evaluation_results.csv"
    results = {
        'Pearson Correlation': [overall_pcc],
        'Spearman Correlation': [overall_spearman],
        'R^2 Score': [r2],
        'RMSE': [rmse]
    }
    df = pd.DataFrame(results)
    df.to_csv(results_df_path, index=False)
    print(f"Evaluation results have been saved to {results_df_path}.")


def main():
    random.seed(42)
    np.random.seed(42)  # Also good to seed numpy
    tf.set_random_seed(42)  # And TensorFlow

    global model_suffix  # Make sure it's accessible or passed around
    global israndom  # Make sure it's accessible or passed around
    israndom = args.israndom  # Set global from args

    print("Loading metadata...")
    mutation_feature, drug_feature, gexpr_feature, methylation_feature, data_idx = \
        MetadataGenerate(Drug_info_file, Cell_line_info_file, Genomic_mutation_file,
                         Drug_feature_file, Gene_expression_file, Methylation_file, False)

    if not data_idx:
        print("No data instances generated. Exiting.")
        sys.exit(1)

    print("Splitting data...")
    # Consider making data split strategy an argument too
    data_train_idx, data_test_idx = DataSplit_drug(data_idx, ratio=0.8)  # Using drug-wise split

    if not data_train_idx or not data_test_idx:
        print("Error: Data splitting resulted in empty train or test set. Check your data and splitting strategy.")
        sys.exit(1)

    print("Extracting features for training and test sets...")
    X_drug_data_train, X_mutation_data_train, X_gexpr_data_train, X_methylation_data_train, Y_train, cancer_type_train_list = \
        FeatureExtract(data_train_idx, drug_feature, mutation_feature, gexpr_feature, methylation_feature)
    X_drug_data_test, X_mutation_data_test, X_gexpr_data_test, X_methylation_data_test, Y_test, cancer_type_test_list = \
        FeatureExtract(data_test_idx, drug_feature, mutation_feature, gexpr_feature, methylation_feature)

    # Prepare validation data for MyCallback, regardless of whether we train or load
    # This ensures MyCallback can function if, for example, you load a model and then fine-tune it.
    # For pure evaluation after loading, this specific validation_data structure is used by MyCallback's on_epoch_end.
    # If you're only evaluating a loaded model without any training epochs, MyCallback isn't strictly called in that phase.
    X_drug_feat_data_test_for_val = np.array([item[0] for item in X_drug_data_test])
    X_drug_adj_data_test_for_val = np.array([item[1] for item in X_drug_data_test])
    validation_data_for_callback = [
        [X_drug_feat_data_test_for_val, X_drug_adj_data_test_for_val, X_mutation_data_test, X_gexpr_data_test,
         X_methylation_data_test], Y_test]

    model = None  # Initialize model variable

    # ****** LOGIC FOR LOADING OR TRAINING MODEL ******
    if args.load_model_path and os.path.exists(args.load_model_path):
        print(f"Loading pre-trained model from: {args.load_model_path}")
        try:
            # If your KerasMultiSourceGCNModel uses custom layers/objects not part of standard Keras,
            # you might need to provide custom_objects dictionary to load_model.
            # e.g., custom_objects={'CustomLayer': CustomLayer}
            # For now, assuming it's standard or the class itself handles it.
            model = load_model(args.load_model_path)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}. Training a new model instead.")
            model = None  # Ensure model is None if loading fails

    if model is None:  # If not loaded or loading failed
        if args.load_model_path and not os.path.exists(args.load_model_path):
            print(
                f"Warning: Pre-trained model path specified ({args.load_model_path}) but file not found. Training a new model.")

        print('Initializing a new model...')
        # Ensure KerasMultiSourceGCNModel is correctly defined and imported
        # The input shapes to createMaster must be correct.
        # X_drug_data_train[0][0].shape[-1] is feature dim of drug
        # X_mutation_data_train.shape[-2] is num_mutation_features
        # X_gexpr_data_train.shape[-1] is num_gexpr_features
        # X_methylation_data_train.shape[-1] is num_methylation_features
        # Ensure these are not empty or zero.
        if not X_drug_data_train:
            print("Error: Training data for drug is empty. Cannot determine feature dimensions.")
            sys.exit(1)

        drug_feat_dim = X_drug_data_train[0][0].shape[-1] if X_drug_data_train[0][
                                                                 0].size > 0 else 0  # Handle empty features
        mut_feat_dim = X_mutation_data_train.shape[-2] if X_mutation_data_train.size > 0 else 0
        gexp_feat_dim = X_gexpr_data_train.shape[-1] if X_gexpr_data_train.size > 0 else 0
        meth_feat_dim = X_methylation_data_train.shape[-1] if X_methylation_data_train.size > 0 else 0

        # Check if any essential features are missing, if they are supposed to be used
        if args.use_mut and mut_feat_dim == 0:
            print("Warning: use_mut is True, but mutation feature dimension is 0.")
        if args.use_gexp and gexp_feat_dim == 0:
            print("Warning: use_gexp is True, but gene expression feature dimension is 0.")
        if args.use_methy and meth_feat_dim == 0:
            print("Warning: use_methy is True, but methylation feature dimension is 0.")

        gcn_model_builder = KerasMultiSourceGCNModel(use_mut, use_gexp, use_methy)
        model = gcn_model_builder.createMaster(
            drug_feat_dim,
            mut_feat_dim,
            gexp_feat_dim,
            meth_feat_dim,
            args.unit_list, args.use_relu, args.use_bn, args.use_GMP
        )

        print('Begin training...')
        model = ModelTraining(model, X_drug_data_train, X_mutation_data_train, X_gexpr_data_train,
                              X_methylation_data_train, Y_train,
                              validation_data_for_callback,  # Use the prepped validation data for the callback
                              nb_epoch=50,  # Reduced for quick testing, default was 500
                              save_model_path_arg=args.save_model_path,
                              current_model_suffix=model_suffix)
    # ****** END OF LOGIC ******

    if model:
        print("Evaluating model...")
        ModelEvaluate(model, X_drug_data_test, X_mutation_data_test, X_gexpr_data_test, X_methylation_data_test, Y_test,
                      cancer_type_test_list)
    else:
        print("No model available for evaluation.")


if __name__ == '__main__':
    main()