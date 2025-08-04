import pickle
import numpy as np
import pandas as pd


def dataload(**cfg):
    response = cfg['path']['response']
    mutation = cfg['path']['mutation']
    cnv = cfg['path']['cnv']
    expression = cfg['path']['expression']
    drug_fpFile_morgan = cfg['path']['morgan']
    drug_fpFile_espf = cfg['path']['espf']
    drug_fpFile_psfp = cfg['path']['psfp']

    # response load cell_line-drug pairs
    response = pd.read_csv(response, index_col=0)
    drug_key = response.columns.values
    # pair [depmap_id pubchem_id Ln_ic50]
    pair = []
    for index, row in response.iterrows():
        for i in drug_key:
            if np.isnan(row[i]) == False:
                pair.append([index, i, row[i]])

    # cell_lines load
    mut_feature = pd.read_csv(mutation, index_col=0)
    exp_feature = pd.read_csv(expression, index_col=0)
    cnv_feature = pd.read_csv(cnv, index_col=0)

    # drug
    with open(drug_fpFile_morgan, 'rb') as f:
        morgan_fp = pickle.load(f)
    with open(drug_fpFile_espf, 'rb') as f:
        espf_fp = pickle.load(f)
    with open(drug_fpFile_psfp, 'rb') as f:
        pubchem_fp = pickle.load(f)

    drug_feature = {}
    for i in drug_key:
        drug_feature[i] = [morgan_fp[i], espf_fp[i], pubchem_fp[i]]
    return drug_feature, mut_feature, exp_feature, cnv_feature, pair, response.index.values, response.columns.values

