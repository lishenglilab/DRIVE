# predict_all.py

# =============================================================================
# 0. 参数解析
# =============================================================================
import os
import argparse
import shutil
import tempfile

# --- 定义命令行参数 ---
parser = argparse.ArgumentParser(
    description="使用预训练的DeepCDR模型，为新药物和新细胞系预测药物反应(IC50)。支持对大型药物文件分块处理。",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument("--model_file", type=str, default = './saved_models/bd1.h5', help="预训练的Keras模型文件路径(.h5)")
parser.add_argument("--drugs_file", type=str, default = './test/predict_all_np.csv', help="包含新药信息(name,smiles)的CSV文件路径")
parser.add_argument("--output_file", type=str, default = './results/', help="保存最终预测结果的CSV文件路径")
parser.add_argument("--mut_file", type=str, default = './test/mu_1.csv', help="新细胞系的突变数据CSV文件路径")
parser.add_argument("--gexp_file", type=str, default = './test/exp_1.csv', help="新细胞系的基因表达数据CSV文件路径")
parser.add_argument("--methy_file", type=str, default = './test/mu_1.csv', help="新细胞系的甲基化数据CSV文件路径")
parser.add_argument("--align_gexp_file", type=str, default = '../depmap/CCLE/exp.csv', help="原始训练集的基因表达文件路径")
parser.add_argument("--align_mut_file", type=str, default = '../depmap/CCLE/mu.csv', help="原始训练集的突变文件路径")
parser.add_argument("--chunk_size", type=int, default=1000, help="处理药物文件时每个区块的大小（行数）")
parser.add_argument("--seed", type=int, default=42, help="随机种子")

args = parser.parse_args()


# -*- coding: UTF-8 -*-

# =============================================================================
# 1. 库导入
# =============================================================================
try:
    import mygene
except ImportError:
    print("错误: mygene库未安装。请运行 'pip install mygene' 来安装。")
    exit(1)

import keras
import keras.backend as K
from keras.models import Model, load_model
from keras.layers import (Input, Dense, Activation, Dropout, Flatten, Concatenate, BatchNormalization, Lambda, GlobalMaxPooling1D, GlobalAveragePooling1D)
from keras.layers import Conv2D, MaxPooling2D

import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from typing import List, Tuple, Dict

from rdkit import Chem
from rdkit.Chem import AllChem

from deepchem.utils.typing import RDKitAtom, RDKitBond, RDKitMol
from deepchem.feat.graph_data import GraphData
from deepchem.feat.base_classes import MolecularFeaturizer
from deepchem.utils.molecule_feature_utils import (
    one_hot_encode, get_atom_type_one_hot, construct_hydrogen_bonding_info,
    get_atom_hydrogen_bonding_one_hot, get_atom_hybridization_one_hot,
    get_atom_total_num_Hs_one_hot, get_atom_is_in_aromatic_one_hot,
    get_atom_chirality_one_hot, get_atom_formal_charge, get_atom_partial_charge,
    get_atom_total_degree_one_hot, get_bond_type_one_hot,
    get_bond_is_in_same_ring_one_hot, get_bond_is_conjugated_one_hot,
    get_bond_stereo_one_hot
)


# =============================================================================
# 2. 药物特征生成模块 (代码不变)
# =============================================================================
# (此处省略，与你提供的版本完全相同)
DEFAULT_ATOM_IMPLICIT_VALENCE_SET = [0, 1, 2, 3, 4, 5, 6]
DEFAULT_ATOM_EXPLICIT_VALENCE_SET = [1, 2, 3, 4, 5, 6]
_USER_ATOM_TYPE_SET = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
                       'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag',
                       'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni',
                       'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb']
_USER_HYBRIDIZATION_SET = ["SP", "SP2", "SP3", 'SP3D', 'SP3D2']
_USER_TOTAL_DEGREE_SET = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
_DEFAULT_TOTAL_NUM_Hs_SET = [0, 1, 2, 3, 4]

def local_get_atom_implicit_valence_one_hot(atom: RDKitAtom, allowable_set: List[int] = DEFAULT_ATOM_IMPLICIT_VALENCE_SET, include_unknown_set: bool = True) -> List[float]: return one_hot_encode(atom.GetImplicitValence(), allowable_set, include_unknown_set)
def local_get_atom_explicit_valence_one_hot(atom: RDKitAtom, allowable_set: List[int] = DEFAULT_ATOM_EXPLICIT_VALENCE_SET, include_unknown_set: bool = True) -> List[float]: return one_hot_encode(atom.GetExplicitValence(), allowable_set, include_unknown_set)
def _construct_atom_feature(atom: RDKitAtom, h_bond_infos: List[Tuple[int, str]], use_chirality: bool, use_partial_charge: bool) -> np.ndarray:
    atom_type = get_atom_type_one_hot(atom, allowable_set=_USER_ATOM_TYPE_SET, include_unknown_set=True)
    formal_charge = get_atom_formal_charge(atom)
    hybridization = get_atom_hybridization_one_hot(atom, allowable_set=_USER_HYBRIDIZATION_SET, include_unknown_set=False)
    acceptor_donor = get_atom_hydrogen_bonding_one_hot(atom, h_bond_infos)
    aromatic = get_atom_is_in_aromatic_one_hot(atom)
    degree = get_atom_total_degree_one_hot(atom, allowable_set=_USER_TOTAL_DEGREE_SET, include_unknown_set=True)
    total_num_Hs = get_atom_total_num_Hs_one_hot(atom, allowable_set=_DEFAULT_TOTAL_NUM_Hs_SET, include_unknown_set=True)
    atom_feat = np.concatenate([atom_type, formal_charge, hybridization, acceptor_donor, aromatic, degree, total_num_Hs])
    if True:
        imp_valence = local_get_atom_implicit_valence_one_hot(atom, DEFAULT_ATOM_IMPLICIT_VALENCE_SET, include_unknown_set=True)
        exp_valence = local_get_atom_explicit_valence_one_hot(atom, DEFAULT_ATOM_EXPLICIT_VALENCE_SET, include_unknown_set=True)
        atom_feat = np.concatenate([atom_feat, imp_valence, exp_valence, [float(atom.HasProp('_ChiralityPossible')), float(atom.GetNumRadicalElectrons())], ])
    if use_chirality: atom_feat = np.concatenate([atom_feat, np.array(get_atom_chirality_one_hot(atom))])
    if use_partial_charge: atom_feat = np.concatenate([atom_feat, np.array(get_atom_partial_charge(atom))])
    return atom_feat

class MolGraphConvFeaturizer(MolecularFeaturizer):
    def __init__(self, use_edges: bool = False, use_chirality: bool = False, use_partial_charge: bool = False):
        self.use_edges, self.use_partial_charge, self.use_chirality = use_edges, use_partial_charge, use_chirality
        self._atom_feature_dim_cache = None
    def _prepare_mol_for_featurization(self, mol: RDKitMol):
        if self.use_partial_charge:
            try: mol.GetAtomWithIdx(0).GetProp('_GasteigerCharge')
            except (KeyError, AttributeError, IndexError):
                try:
                    if mol.GetNumAtoms() > 0: AllChem.ComputeGasteigerCharges(mol)
                except (RuntimeError, Exception): pass
        return mol
    def _get_atom_feature_dim(self):
        if self._atom_feature_dim_cache is not None: return self._atom_feature_dim_cache
        try:
            dummy_mol = Chem.MolFromSmiles("CC")
            dummy_mol = self._prepare_mol_for_featurization(dummy_mol)
            dummy_atom = dummy_mol.GetAtomWithIdx(0)
            h_bond_infos = construct_hydrogen_bonding_info(dummy_mol)
            feat = _construct_atom_feature(dummy_atom, h_bond_infos, self.use_chirality, self.use_partial_charge)
            self._atom_feature_dim_cache = feat.shape[0]
        except Exception as e:
            raise RuntimeError(f"无法动态确定原子特征维度: {e}")
        return self._atom_feature_dim_cache
    def _featurize(self, datapoint: RDKitMol, **kwargs) -> GraphData:
        mol = self._prepare_mol_for_featurization(datapoint)
        h_bond_infos = construct_hydrogen_bonding_info(mol)
        atom_features = np.asarray([_construct_atom_feature(atom, h_bond_infos, self.use_chirality, self.use_partial_charge) for atom in mol.GetAtoms()], dtype=float)
        src, dest = [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            src.extend([start, end]); dest.extend([end, start])
        edge_index = np.asarray([src, dest], dtype=int)
        return GraphData(node_features=atom_features, edge_index=edge_index)

# =============================================================================
# 3. Keras模型和图层定义 (代码不变)
# =============================================================================
# (此处省略，与你提供的版本完全相同)
class GraphLayer(keras.layers.Layer):
    def __init__(self, step_num=1, activation=None, **kwargs):
        self.supports_masking = True; self.step_num = step_num
        self.activation = keras.activations.get(activation); super(GraphLayer, self).__init__(**kwargs)
    def get_config(self):
        config = {'step_num': self.step_num, 'activation': keras.activations.serialize(self.activation)}
        base_config = super(GraphLayer, self).get_config(); return dict(list(base_config.items()) + list(config.items()))
    def _get_walked_edges(self, edges, step_num):
        if step_num <= 1: return edges
        deeper = self._get_walked_edges(K.batch_dot(edges, edges), step_num // 2)
        if step_num % 2 == 1: deeper = K.batch_dot(deeper, edges)
        return K.cast(K.greater(deeper, 0.0), K.floatx())
    def call(self, inputs, **kwargs):
        features, edges_in = inputs; edges_float = K.cast(edges_in, K.floatx())
        walked_edges = self._get_walked_edges(edges_float, self.step_num) if self.step_num > 1 else edges_float
        return self.activation(self._call(features, walked_edges))
    def _call(self, features, edges): raise NotImplementedError('The class is not intended to be used directly.')

class GraphConv(GraphLayer):
    def __init__(self, units, kernel_initializer='glorot_uniform', kernel_regularizer=None, kernel_constraint=None,
                 use_bias=True, bias_initializer='zeros', bias_regularizer=None, bias_constraint=None, **kwargs):
        self.units = units
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.use_bias = use_bias
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.bias_constraint = keras.constraints.get(bias_constraint)
        super(GraphConv, self).__init__(**kwargs)
    def get_config(self):
        config = {'units': self.units, 'kernel_initializer': keras.initializers.serialize(self.kernel_initializer), 'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer), 'kernel_constraint': keras.constraints.serialize(self.kernel_constraint), 'use_bias': self.use_bias, 'bias_initializer': keras.initializers.serialize(self.bias_initializer), 'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer), 'bias_constraint': keras.constraints.serialize(self.bias_constraint)}
        base_config = super(GraphConv, self).get_config(); return dict(list(base_config.items()) + list(config.items()))
    def build(self, input_shape):
        feature_dim_in = input_shape[0][-1]
        self.W = self.add_weight(shape=(feature_dim_in, self.units), initializer=self.kernel_initializer, regularizer=self.kernel_regularizer, constraint=self.kernel_constraint, name=f'{self.name}_W')
        if self.use_bias: self.b = self.add_weight(shape=(self.units,), initializer=self.bias_initializer, regularizer=self.bias_regularizer, constraint=self.bias_constraint, name=f'{self.name}_b')
        super(GraphConv, self).build(input_shape)
    def compute_output_shape(self, input_shape): return input_shape[0][:2] + (self.units,)
    def compute_mask(self, inputs, mask=None): return mask[0] if mask else None
    def _call(self, features, edges):
        transformed = K.dot(features, self.W)
        if self.use_bias: transformed = transformed + self.b
        return K.batch_dot(edges, transformed)

# =============================================================================
# 4. 辅助函数 (代码不变)
# =============================================================================
# (此处省略，与你提供的版本完全相同)
Max_atoms = 100
israndom = False
def NormalizeAdj(adj):
    adj = adj + np.eye(adj.shape[0]); d = adj.sum(1); d_inv_sqrt = np.power(d, -0.5, where=d!=0); d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
def CalculateGraphFeat(node_features_raw, edge_index_raw, max_atoms_val, model_expected_atom_feat_dim):
    num_nodes = node_features_raw.shape[0]
    feat_dim = node_features_raw.shape[1] if num_nodes > 0 else 0
    if feat_dim > model_expected_atom_feat_dim: node_features = node_features_raw[:, :model_expected_atom_feat_dim]
    elif feat_dim < model_expected_atom_feat_dim:
        node_features = np.zeros((num_nodes, model_expected_atom_feat_dim));
        if num_nodes > 0: node_features[:, :feat_dim] = node_features_raw
    else: node_features = node_features_raw
    adj = np.zeros((num_nodes, num_nodes));
    if num_nodes > 0 and edge_index_raw.shape[1] > 0: adj[edge_index_raw[0], edge_index_raw[1]] = 1
    padded_adj = np.zeros((max_atoms_val, max_atoms_val)); padded_feat = np.zeros((max_atoms_val, model_expected_atom_feat_dim))
    if num_nodes > 0:
        num_to_pad = min(num_nodes, max_atoms_val)
        padded_adj[:num_to_pad, :num_to_pad] = NormalizeAdj(adj[:num_to_pad, :num_to_pad])
        padded_feat[:num_to_pad] = node_features[:num_to_pad]
    return [padded_feat, padded_adj]
def build_gene_map_with_mygene(gene_symbols: list):
    print(f"使用 mygene.info 查询 {len(gene_symbols)} 个基因名的Entrez ID..."); mg = mygene.MyGeneInfo()
    results = mg.querymany(gene_symbols, scopes='symbol', species='human', fields='entrezgene', as_dataframe=True, verbose=False)
    results.dropna(subset=['entrezgene'], inplace=True); results = results[~results.index.duplicated(keep='first')]
    gene_map = results['entrezgene'].astype(int).astype(str).to_dict()
    print(f"查询完成。成功将 {len(gene_map)} 个基因名映射到Entrez ID。"); return gene_map
def load_and_align_new_cell_data(new_omics_files: Dict[str, str], train_gene_id_lists: Dict[str, List[str]]):
    print("加载并对齐新的细胞系组学数据..."); all_gene_symbols, raw_dfs = set(), {}
    for omic, path in new_omics_files.items():
        if path and os.path.exists(path):
            try: df = pd.read_csv(path, index_col=0); df.index = df.index.str.strip(); all_gene_symbols.update(df.columns); raw_dfs[omic] = df
            except Exception as e: print(f"读取文件 {path} 失败: {e}")
        elif path: print(f"警告: 文件 {path} 未找到")
    if not all_gene_symbols: print("错误: 未能读取任何基因名"); return {}
    gene_map = build_gene_map_with_mygene(list(all_gene_symbols))
    if not gene_map: print("错误: 基因名映射失败"); return {}
    aligned_data = {}
    for omic, df in raw_dfs.items():
        df.rename(columns=gene_map, inplace=True)
        train_ids = train_gene_id_lists.get(omic)
        if not train_ids: print(f"警告: 找不到 {omic} 的训练基因列表"); continue
        aligned_df = df.reindex(columns=train_ids, fill_value=0); aligned_df.index = aligned_df.index.astype(str)
        print(f"已将 {omic} 数据对齐到 {aligned_df.shape[1]} 个训练基因ID"); aligned_data[omic] = aligned_df
    return aligned_data


# =============================================================================
# 5. 分块预测函数 (修改自原有的主流程)
# =============================================================================
def predict_drug_chunk(
    drug_chunk_file: str,
    output_chunk_file: str,
    trained_model: Model,
    aligned_cell_data: Dict[str, pd.DataFrame],
    new_cell_line_ids: List[str],
    graph_featurizer: MolGraphConvFeaturizer,
    model_expected_drug_dim: int,
    active_omic_types: List[str],
    chunk_name: str
):
    """
    对单个药物文件块执行预测，并将结果保存到对应的块输出文件。
    这是一个被主循环调用的核心工作函数。
    """
    print(f"\n--- 开始处理药物块: {chunk_name} ---")

    try:
        drug_chunk_df = pd.read_csv(drug_chunk_file, header=None, names=['drug_name', 'smiles'])
    except FileNotFoundError:
        print(f"错误: 药物块文件 {drug_chunk_file} 未找到。")
        return

    predictions = []
    # 为当前块设置进度条
    with tqdm(total=len(drug_chunk_df) * len(new_cell_line_ids), desc=f"预测 {chunk_name}") as pbar:
        for _, drug_row in drug_chunk_df.iterrows():
            drug_name, smiles = str(drug_row['drug_name']), str(drug_row['smiles'])
            if not smiles or pd.isna(smiles) or smiles.lower() == 'nan':
                pbar.update(len(new_cell_line_ids))
                continue
            
            mol = Chem.MolFromSmiles(smiles)
            if not mol or mol.GetNumAtoms() == 0:
                pbar.update(len(new_cell_line_ids))
                continue

            try:
                graph_data = graph_featurizer.featurize([mol])[0]
                feat, adj = CalculateGraphFeat(graph_data.node_features, graph_data.edge_index, Max_atoms, model_expected_drug_dim)
                drug_input = [np.array([feat]), np.array([adj])]
            except Exception as e:
                print(f"\n警告: 药物 {drug_name} 特征化失败: {e}")
                pbar.update(len(new_cell_line_ids))
                continue
            
            for cell_id in new_cell_line_ids:
                model_inputs = list(drug_input)
                try:
                    model_input_names = [inp.name for inp in trained_model.inputs]
                    
                    if 'mutation_feat_input' in model_input_names and 'mut' in active_omic_types:
                        model_inputs.append(aligned_cell_data['mut'].loc[cell_id].values.reshape(1, 1, -1, 1))
                    if 'gexpr_feat_input' in model_input_names and 'gexp' in active_omic_types:
                        model_inputs.append(aligned_cell_data['gexp'].loc[cell_id].values.reshape(1, -1))
                    if 'methy_feat_input' in model_input_names and 'methy' in active_omic_types:
                        model_inputs.append(aligned_cell_data['methy'].loc[cell_id].values.reshape(1, -1))
                    
                    pred_val = trained_model.predict(model_inputs, verbose=0)[0][0]
                    predictions.append({'drug_name': drug_name, 'smiles': smiles, 'cell_line_id': cell_id, 'predicted_ln_IC50': float(pred_val)})
                except Exception as e:
                    print(f"\n错误 (药物 {drug_name}, 细胞系 {cell_id}): {e}")
                pbar.update(1)

    if predictions:
        results_df = pd.DataFrame(predictions)
        results_df.to_csv(output_chunk_file, index=False, encoding='utf-8-sig')
        print(f"块 {chunk_name} 的预测结果已保存到 {output_chunk_file}")
    else:
        print(f"块 {chunk_name} 没有生成任何预测结果。")


# =============================================================================
# 6. 主程序入口 (重构为总调度)
# =============================================================================
def main():
    """主函数，组织整个分块预测流程"""
    random.seed(args.seed)
    np.random.seed(args.seed)

    # --- 步骤 1: 一次性加载和准备所有不-变的数据 (模型, 细胞系数据) ---
    print("="*50+"\n开始预测流程初始化...\n"+"="*50)
    print(f"加载预训练模型从 {args.model_file}...")
    try:
        trained_model = load_model(args.model_file, custom_objects={'GraphLayer': GraphLayer, 'GraphConv': GraphConv})
        print("模型加载成功。")
        MODEL_EXPECTED_DRUG_DIM = 75 # 此处可根据模型进行调整
        print(f"根据预训练模型，设定药物原子特征维度为: {MODEL_EXPECTED_DRUG_DIM}")
    except Exception as e:
        print(f"错误: 加载模型失败: {e}")
        return

    graph_featurizer = MolGraphConvFeaturizer(use_edges=True, use_chirality=True, use_partial_charge=True)
    print(f"药物特征提取器已初始化，将生成 {graph_featurizer._get_atom_feature_dim()} 维的原子特征。")

    print("\n--- 准备细胞系数据 ---")
    new_omics_files = {'mut': args.mut_file, 'gexp': args.gexp_file, 'methy': args.methy_file}
    data_paths_for_alignment = {'train_gexp': args.align_gexp_file, 'train_mut': args.align_mut_file}
    try:
        train_mut_df_cols = pd.read_csv(data_paths_for_alignment['train_mut'], index_col=0, nrows=0).columns.astype(str).tolist()
        train_gexp_df_cols = pd.read_csv(data_paths_for_alignment['train_gexp'], index_col=0, nrows=0).columns.astype(str).tolist()
        train_gene_id_lists = {'mut': train_mut_df_cols, 'gexp': train_gexp_df_cols, 'methy': train_mut_df_cols}
    except Exception as e:
        print(f"错误: 加载对齐用基因列表失败: {e}")
        return

    aligned_cell_data = load_and_align_new_cell_data(new_omics_files, train_gene_id_lists)
    active_omic_types = [omic for omic, path in new_omics_files.items() if path is not None and omic in aligned_cell_data]
    if not active_omic_types:
        print("错误：未提供任何有效的组学数据文件或数据加载失败。")
        return

    cell_line_id_sets = {omic: set(df.index) for omic, df in aligned_cell_data.items()}
    if not cell_line_id_sets:
        print("错误: 未能从任何提供的文件中加载细胞系数据")
        return
        
    intersected_ids = set.intersection(*cell_line_id_sets.values())
    new_cell_line_ids = sorted(list(intersected_ids))
    if not new_cell_line_ids:
        print("\n错误: 所有提供的组学数据文件间无共同细胞系ID。")
        return
    print(f"\n细胞系数据准备完毕。将对 {len(new_cell_line_ids)} 个共同细胞系进行预测。")

    # --- 步骤 2: 分割药物文件到临时目录 ---
    print("\n--- 步骤 2: 分割药物文件 ---")
    temp_drug_dir = tempfile.mkdtemp(prefix="drug_chunks_")
    chunk_files = []
    try:
        print(f"创建临时目录: {temp_drug_dir}")
        reader = pd.read_csv(args.drugs_file, chunksize=args.chunk_size, header=None)
        for i, chunk_df in enumerate(reader):
            chunk_file_path = os.path.join(temp_drug_dir, f"drug_chunk_{i+1}.csv")
            chunk_df.to_csv(chunk_file_path, index=False, header=False)
            chunk_files.append(chunk_file_path)
        if not chunk_files:
            print("错误：药物文件为空或读取失败。")
            return
        print(f"已将药物文件分割成 {len(chunk_files)} 个块，每个块最多 {args.chunk_size} 行。")
    except FileNotFoundError:
        print(f"错误: 原始药物文件 {args.drugs_file} 未找到。")
        shutil.rmtree(temp_drug_dir)
        return
    except Exception as e:
        print(f"错误: 分割药物文件时出错: {e}")
        shutil.rmtree(temp_drug_dir)
        return

    # --- 步骤 3: 循环处理每个药物块 ---
    print("\n--- 步骤 3: 开始分块预测 ---")
    output_chunk_files = []
    base_output, ext = os.path.splitext(args.output_file)

    for i, drug_chunk_file in enumerate(chunk_files):
        chunk_name = f"块_{i+1}_共_{len(chunk_files)}"
        output_chunk_file = f"{base_output}_part_{i+1}{ext}"
        output_chunk_files.append(output_chunk_file)

        predict_drug_chunk(
            drug_chunk_file=drug_chunk_file,
            output_chunk_file=output_chunk_file,
            trained_model=trained_model,
            aligned_cell_data=aligned_cell_data,
            new_cell_line_ids=new_cell_line_ids,
            graph_featurizer=graph_featurizer,
            model_expected_drug_dim=MODEL_EXPECTED_DRUG_DIM,
            active_omic_types=active_omic_types,
            chunk_name=chunk_name
        )

    # --- 步骤 4: 合并所有结果并清理 ---
    print("\n--- 步骤 4: 合并预测结果并进行清理 ---")
    all_results_df = []
    for f in output_chunk_files:
        if os.path.exists(f):
            try:
                df = pd.read_csv(f)
                all_results_df.append(df)
            except Exception as e:
                print(f"警告: 读取结果块文件 {f} 失败: {e}")
    
    if all_results_df:
        final_df = pd.concat(all_results_df, ignore_index=True)
        final_df.to_csv(args.output_file, index=False, encoding='utf-8-sig')
        print(f"\n所有预测结果已成功合并并保存到最终文件: {args.output_file}")
        print(f"总共预测了 {len(final_df)} 条记录。")
    else:
        print("警告：没有生成任何有效的预测结果，无法创建最终输出文件。")

    # 清理临时文件
    print("正在清理临时文件...")
    try:
        shutil.rmtree(temp_drug_dir)
        print(f"已删除临时药物文件目录: {temp_drug_dir}")
        for f in output_chunk_files:
            if os.path.exists(f):
                os.remove(f)
        print("已删除临时的分块结果文件。")
    except Exception as e:
        print(f"清理临时文件时出错: {e}")

    print("\n预测流程全部结束。")


if __name__ == '__main__':
    main()