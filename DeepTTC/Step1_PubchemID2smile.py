# python3
# -*- coding:utf-8 -*-

"""
@author:野山羊骑士
@e-mail：thankyoulaojiang@163.com
@file：PycharmProject-PyCharm-step1_drugGet.py
@time:2021/8/10 15:48 
@从 pubchem数据库中，根据id查找smile
"""

import os
import sys
import pandas as pd
import pubchempy as pcp
import pickle
pub_file = sys.argv[1]
pub_df = pd.read_csv(pub_file)
pub_df = pub_df.dropna(subset=['PubCHEM'])
pub_df = pub_df[(pub_df['PubCHEM'] != 'none') & (pub_df['PubCHEM'] != 'several')]
# pub_df = pub_df.head(20)

smile_list = []
inchi_list = []

for idx, row in pub_df.iterrows():
    print(f"Processing index: {idx}")
    pubid = row['PubCHEM'].split(',')[0]
    print(f"PubChem ID: {pubid}")

    try:
        compound = pcp.Compound.from_cid(pubid)
        smile = compound.isomeric_smiles
        inchi = compound.inchi
        smile_list.append(smile)
        inchi_list.append(inchi)
        print(f"SMILES: {smile}")
        print(f"InChI: {inchi}")
    except pcp.BadRequestError as e:
        print(f"Error for PubChem ID {pubid}: {e}")
        smile_list.append(None)
        inchi_list.append(None)
    except Exception as e:
        print(f"Unexpected error for PubChem ID {pubid}: {e}")
        smile_list.append(None)
        inchi_list.append(None)

pub_df['smiles'] = smile_list
pub_df['inchi'] = inchi_list

pub_df.to_pickle('smile_inchi.pkl', protocol=4)
pub_df.to_csv('smile_inchi.csv', index=False)