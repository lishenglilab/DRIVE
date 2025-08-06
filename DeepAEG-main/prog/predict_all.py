import random, os, sys
from collections import defaultdict
import numpy as np
import csv
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import \
    load_model
from tensorflow.keras import backend as K
from tensorflow.keras import layers
import argparse
import codecs
from subword_nmt.apply_bpe import BPE
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
from typing import List, Tuple, Dict, Any
import math
import gc

# ==============================================================================
# SECTION 1: FEATURE GENERATION CODE
# ==============================================================================
RDKitAtom = Chem.rdchem.Atom
RDKitBond = Chem.rdchem.Bond
RDKitMol = Chem.rdchem.Mol

try:
    from deepchem.feat.graph_data import GraphData
    from deepchem.feat.base_classes import MolecularFeaturizer
    from deepchem.utils.molecule_feature_utils import one_hot_encode, get_atom_type_one_hot, \
        construct_hydrogen_bonding_info, get_atom_hydrogen_bonding_one_hot, get_atom_hybridization_one_hot, \
        get_atom_total_num_Hs_one_hot, get_atom_is_in_aromatic_one_hot, get_atom_chirality_one_hot, \
        get_atom_formal_charge, get_atom_partial_charge, get_atom_total_degree_one_hot, \
        get_bond_type_one_hot, get_bond_is_in_same_ring_one_hot, get_bond_is_conjugated_one_hot, \
        get_bond_stereo_one_hot
except ImportError as e:
    print(f"ERROR: DeepChem import failed: {e}. Ensure DeepChem is installed.")
    sys.exit(1)

DEFAULT_ATOM_IMPLICIT_VALENCE_SET = [0, 1, 2, 3, 4, 5, 6]
DEFAULT_ATOM_EXPLICIT_VALENCE_SET = [1, 2, 3, 4, 5, 6]
DEFAULT_TOTAL_NUM_Hs_SET = [0, 1, 2, 3, 4]

def get_atom_implicit_valence_one_hot_custom(atom: RDKitAtom, allowable_set: List[int] = DEFAULT_ATOM_IMPLICIT_VALENCE_SET, include_unknown_set: bool = True) -> List[float]:
    return one_hot_encode(atom.GetImplicitValence(), allowable_set, include_unknown_set)

def get_atom_explicit_valence_one_hot_custom(atom: RDKitAtom, allowable_set: List[int] = DEFAULT_ATOM_EXPLICIT_VALENCE_SET, include_unknown_set: bool = True) -> List[float]:
    return one_hot_encode(atom.GetExplicitValence(), allowable_set, include_unknown_set)

def _construct_atom_feature_from_script(atom: RDKitAtom, h_bond_infos: List[Tuple[int, str]], use_chirality: bool, use_partial_charge: bool) -> np.ndarray:
    atom_type = get_atom_type_one_hot(atom, include_unknown_set=True)
    formal_charge = get_atom_formal_charge(atom)
    hybridization = get_atom_hybridization_one_hot(atom, include_unknown_set=False)
    acceptor_donor = get_atom_hydrogen_bonding_one_hot(atom, h_bond_infos)
    aromatic = get_atom_is_in_aromatic_one_hot(atom)
    degree = get_atom_total_degree_one_hot(atom, include_unknown_set=True)
    total_num_Hs = get_atom_total_num_Hs_one_hot(atom, DEFAULT_TOTAL_NUM_Hs_SET, include_unknown_set=True)
    atom_feat = np.concatenate([atom_type, formal_charge, hybridization, acceptor_donor, aromatic, degree, total_num_Hs])
    if True:
        imp_valence = get_atom_implicit_valence_one_hot_custom(atom)
        exp_valence = get_atom_explicit_valence_one_hot_custom(atom)
        chirality_possible_val = float(atom.HasProp('_ChiralityPossible')) if atom.HasProp('_ChiralityPossible') else 0.0
        num_radical_electrons_val = float(atom.GetNumRadicalElectrons())
        atom_feat = np.concatenate([atom_feat, imp_valence, exp_valence, [chirality_possible_val, num_radical_electrons_val]])
    if use_chirality:
        chirality = get_atom_chirality_one_hot(atom)
        atom_feat = np.concatenate([atom_feat, np.array(chirality)])
    if use_partial_charge:
        partial_charge_val = [0.0]
        try:
            if atom.HasProp('_GasteigerCharge'):
                pc = atom.GetProp('_GasteigerCharge')
                if pc is not None and not np.isnan(pc) and not np.isinf(pc): partial_charge_val = [float(pc)]
        except Exception:
            pass
        atom_feat = np.concatenate([atom_feat, np.array(partial_charge_val)])
    return atom_feat

def _construct_bond_feature_from_script(bond: RDKitBond) -> np.ndarray:
    bond_type = get_bond_type_one_hot(bond)
    same_ring = get_bond_is_in_same_ring_one_hot(bond)
    conjugated = get_bond_is_conjugated_one_hot(bond)
    stereo = get_bond_stereo_one_hot(bond)
    return np.concatenate([bond_type, same_ring, conjugated, stereo])

class MolGraphConvFeaturizerForPredict(MolecularFeaturizer):
    def __init__(self, use_edges: bool = True, use_chirality: bool = True, use_partial_charge: bool = True):
        self.use_edges = use_edges; self.use_chirality = use_chirality; self.use_partial_charge = use_partial_charge
        mol_dummy = Chem.MolFromSmiles("CC"); mol_dummy = Chem.AddHs(mol_dummy)
        if self.use_partial_charge: AllChem.ComputeGasteigerCharges(mol_dummy)
        dummy_h_bond_infos = construct_hydrogen_bonding_info(mol_dummy)
        dummy_atom = mol_dummy.GetAtomWithIdx(0); dummy_bond = mol_dummy.GetBondWithIdx(0)
        self.actual_atom_feat_dim = _construct_atom_feature_from_script(dummy_atom, dummy_h_bond_infos, self.use_chirality, self.use_partial_charge).shape[0]
        self.actual_bond_feat_dim = _construct_bond_feature_from_script(dummy_bond).shape[0]
        print(f"Featurizer Initialized: Atom Dim = {self.actual_atom_feat_dim}, Bond Dim = {self.actual_bond_feat_dim}")

    def _featurize(self, datapoint: RDKitMol, **kwargs) -> GraphData:
        if datapoint.GetNumAtoms() == 0: return GraphData(node_features=np.empty((0, self.actual_atom_feat_dim)), edge_index=np.array([[], []], dtype=int), edge_features=np.empty((0, self.actual_bond_feat_dim)))
        if self.use_partial_charge:
            try:
                if not datapoint.GetAtomWithIdx(0).HasProp('_GasteigerCharge'): AllChem.ComputeGasteigerCharges(datapoint)
            except Exception: pass
        h_bond_infos = construct_hydrogen_bonding_info(datapoint)
        atom_features_list = [_construct_atom_feature_from_script(atom, h_bond_infos, self.use_chirality, self.use_partial_charge) for atom in datapoint.GetAtoms()]
        node_features_final = np.empty((0, self.actual_atom_feat_dim))
        if atom_features_list: node_features_final = np.asarray(atom_features_list, dtype=float)
        src, dest = [], []; bond_features_list = []
        if datapoint.GetNumBonds() > 0:
            for bond in datapoint.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                src += [start, end]; dest += [end, start]
                if self.use_edges: bond_features_list += 2 * [_construct_bond_feature_from_script(bond)]
        edge_features_final = np.empty((0, self.actual_bond_feat_dim))
        if self.use_edges and bond_features_list: edge_features_final = np.asarray(bond_features_list, dtype=float)
        return GraphData(node_features=node_features_final, edge_index=np.asarray([src, dest], dtype=int), edge_features=edge_features_final)

# ============================================================================
# SECTION 2: MODEL DEFINITION
# ============================================================================
class GraphConvTest(tf.keras.layers.Layer):
    def __init__(self, units, units_edge, step, activation='tanh', use_bias=False, update_edge=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', **kwargs):
        super(GraphConvTest, self).__init__(**kwargs)
        self.units, self.units_edge, self.step, self.activation_fn, self.use_bias, self.update_edge, self.kernel_initializer, self.bias_initializer = units, units_edge, step, tf.keras.activations.get(activation), use_bias, update_edge, kernel_initializer, bias_initializer
    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units, 'units_edge': self.units_edge, 'step': self.step, 'activation': tf.keras.activations.serialize(self.activation_fn), 'use_bias': self.use_bias, 'update_edge': self.update_edge, 'kernel_initializer': self.kernel_initializer, 'bias_initializer': self.bias_initializer})
        return config
    def build(self, input_shape):
        F_in, E_in = input_shape[0][-1], input_shape[1][-1]
        self.W_node_transform = self.add_weight(name=f's{self.step}_w_node_transform', shape=(F_in, F_in), initializer=self.kernel_initializer, trainable=True)
        self.W_out_linear = self.add_weight(name=f's{self.step}_w_out_linear', shape=(F_in * E_in, self.units), initializer=self.kernel_initializer, trainable=True)
        if self.use_bias: self.b_out_linear = self.add_weight(name=f's{self.step}_b_out_linear', shape=(self.units,), initializer=self.bias_initializer, trainable=True)
        if self.update_edge:
            self.W_edge_hidden = self.add_weight(name=f's{self.step}_w_edge_hidden', shape=(self.units * 2, self.units_edge), initializer=self.kernel_initializer, trainable=True)
            self.W_edge_out = self.add_weight(name=f's{self.step}_w_edge_out', shape=(self.units_edge + E_in, self.units_edge), initializer=self.kernel_initializer, trainable=True)
            if self.use_bias:
                self.b_edge_hidden = self.add_weight(name=f's{self.step}_b_edge_hidden', shape=(self.units_edge,), initializer=self.bias_initializer, trainable=True)
                self.b_edge_out = self.add_weight(name=f's{self.step}_b_edge_out', shape=(self.units_edge,), initializer=self.bias_initializer, trainable=True)
        super(GraphConvTest, self).build(input_shape)
    def call(self, inputs, **kwargs):
        node_feature, lcq_adj = inputs
        transformed_nodes = K.dot(node_feature, self.W_node_transform)
        lcq_adj_perm = tf.transpose(lcq_adj, (0, 3, 1, 2))
        B, E, N, _ = tf.unstack(tf.shape(lcq_adj_perm))
        _, _, F = tf.unstack(tf.shape(transformed_nodes))
        aggregated_messages = K.batch_dot(K.reshape(lcq_adj_perm, (B, E * N, N)), transformed_nodes)
        aggregated_messages = K.reshape(aggregated_messages, (B, E, N, F))
        node_self_contribution = K.expand_dims(transformed_nodes, axis=1)
        aggregated_nodes = aggregated_messages + node_self_contribution
        aggregated_nodes = K.permute_dimensions(aggregated_nodes, (0, 2, 3, 1))
        aggregated_nodes = K.reshape(aggregated_nodes, (B, N, F * E))
        node_embeddings = K.dot(aggregated_nodes, self.W_out_linear)
        if self.use_bias: node_embeddings += self.b_out_linear
        node_embeddings = self.activation_fn(node_embeddings)
        if self.update_edge:
            node_embed_expanded_src = tf.expand_dims(node_embeddings, axis=2)
            node_embed_expanded_dest = tf.expand_dims(node_embeddings, axis=1)
            tiled_src_embeds = tf.tile(node_embed_expanded_src, [1, 1, N, 1])
            tiled_dest_embeds = tf.tile(node_embed_expanded_dest, [1, N, 1, 1])
            edge_candidate_feats = K.concatenate([tiled_src_embeds, tiled_dest_embeds], axis=3)
            edge_hidden = K.dot(edge_candidate_feats, self.W_edge_hidden)
            if self.use_bias: edge_hidden += self.b_edge_hidden
            edge_hidden = tf.nn.relu(edge_hidden)
            edge_update_input = K.concatenate([edge_hidden, lcq_adj], axis=3)
            updated_lcq_adj = K.dot(edge_update_input, self.W_edge_out)
            if self.use_bias: updated_lcq_adj += self.b_edge_out
            updated_lcq_adj = tf.nn.relu(updated_lcq_adj)
            return [node_embeddings, updated_lcq_adj]
        else:
            return [node_embeddings, lcq_adj]

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.2, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([layers.Dense(ff_dim, activation="gelu"), layers.Dense(embed_dim)])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.embed_dim, self.num_heads, self.ff_dim, self.rate = embed_dim, num_heads, ff_dim, rate
    def call(self, inputs, attention_mask=None, training=None):
        attn_output = self.att(inputs, inputs, attention_mask=attention_mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "num_heads": self.num_heads, "ff_dim": self.ff_dim, "rate": self.rate})
        return config

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super(TokenAndPositionEmbedding, self).__init__(**kwargs)
        self.maxlen, self.vocab_size, self.embed_dim = maxlen, vocab_size, embed_dim
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, name="token_embedding")
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim, name="position_embedding")
    def call(self, inputs):
        maxlen_dynamic = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=maxlen_dynamic, delta=1)
        return self.token_emb(inputs) + self.pos_emb(positions)
    def get_config(self):
        config = super().get_config()
        config.update({"maxlen": self.maxlen, "vocab_size": self.vocab_size, "embed_dim": self.embed_dim})
        return config

class SmilesTransformerEncoder(layers.Layer):
    def __init__(self, maxlen=50, vocab_size=2586, embed_dim=128, num_heads=8, ff_dim=512, num_blocks=1, rate=0.2, **kwargs):
        super(SmilesTransformerEncoder, self).__init__(**kwargs)
        self.embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
        self.encoder_blocks = [TransformerBlock(embed_dim, num_heads, ff_dim, rate) for _ in range(num_blocks)]
        self.maxlen, self.vocab_size, self.embed_dim, self.num_heads, self.ff_dim, self.num_blocks, self.rate = maxlen, vocab_size, embed_dim, num_heads, ff_dim, num_blocks, rate
    def call(self, inputs, training=None):
        x_tokens, padding_mask = inputs
        attention_mask_additive = tf.cast(tf.expand_dims(tf.expand_dims(padding_mask, 1), 1), tf.float32)
        attention_mask_additive = (1.0 - attention_mask_additive) * -1e9
        x = self.embedding_layer(x_tokens)
        for block in self.encoder_blocks: x = block(x, attention_mask=attention_mask_additive, training=training)
        return x[:, 0, :]
    def get_config(self):
        config = super().get_config()
        config.update({"maxlen": self.maxlen, "vocab_size": self.vocab_size, "embed_dim": self.embed_dim, "num_heads": self.num_heads, "ff_dim": self.ff_dim, "num_blocks": self.num_blocks, 'rate': self.rate})
        return config

class KerasMultiSourceGCNModel_new(object):
    def __init__(self, use_mut, use_gexp, use_methy, use_copy, regr=True):
        self.use_mut, self.use_gexp, self.use_methy, self.use_copy, self.regr = use_mut, use_gexp, use_methy, use_copy, regr
    def createMaster(self, drug_dim, edge_dim, smiles_dim, mask_dim, cellline_dim_tuple, units_list, unit_edge_list, batch_size_for_input, dropout_rate, activation_name, use_relu_in_gcn_unused, use_bn_in_gcn, use_gmp_in_gcn):
        fixed_batch_size = None
        node_feature_in = layers.Input(batch_shape=(fixed_batch_size, 100, drug_dim), name='node_feature')
        lcq_adj_in = layers.Input(batch_shape=(fixed_batch_size, 100, 100, edge_dim), name='lcq_adj')
        smiles_tokens_in = layers.Input(batch_shape=(fixed_batch_size, smiles_dim), name="smiles_input")
        smiles_mask_in = layers.Input(batch_shape=(fixed_batch_size, mask_dim), name="mask_input")
        num_genes = cellline_dim_tuple[1]
        copy_feat_in = layers.Input(batch_shape=(fixed_batch_size, num_genes,), name='copy_input')
        mutation_feat_in = layers.Input(batch_shape=(fixed_batch_size, num_genes,), name='mutation_feat_input')
        gexpr_feat_in = layers.Input(batch_shape=(fixed_batch_size, num_genes,), name='gexpr_feat_input')
        methy_feat_in = layers.Input(batch_shape=(fixed_batch_size, num_genes,), name='methy_feat_input')
        gcn_out = [node_feature_in, lcq_adj_in]
        for i, (units_gcn, units_edge_gcn) in enumerate(zip(units_list, unit_edge_list)):
            gcn_out = GraphConvTest(units=units_gcn, units_edge=units_edge_gcn, step=i, activation=activation_name, use_bias=True)([gcn_out[0], gcn_out[1]])
            temp_gcn_nodes, temp_gcn_edges = layers.Activation(activation_name)(gcn_out[0]), layers.Activation(activation_name)(gcn_out[1])
            if use_bn_in_gcn: temp_gcn_nodes, temp_gcn_edges = layers.BatchNormalization()(temp_gcn_nodes), layers.BatchNormalization()(temp_gcn_edges)
            temp_gcn_nodes, temp_gcn_edges = layers.Dropout(dropout_rate)(temp_gcn_nodes), layers.Dropout(dropout_rate)(temp_gcn_edges)
            gcn_out = [temp_gcn_nodes, temp_gcn_edges]
        pool_layer_node, pool_layer_edge = (layers.GlobalMaxPooling1D, layers.GlobalMaxPooling2D) if use_gmp_in_gcn else (layers.GlobalAveragePooling1D, layers.GlobalAveragePooling2D)
        x_drug_nodes, x_drug_edges = pool_layer_node()(gcn_out[0]), pool_layer_edge()(gcn_out[1])
        x_drug_gcn_features = layers.Concatenate()([x_drug_nodes, x_drug_edges])
        smiles_transformer = SmilesTransformerEncoder(maxlen=smiles_dim, vocab_size=2586, embed_dim=128, num_heads=8, ff_dim=512, num_blocks=1, rate=dropout_rate)
        x_smiles_encoded_features = smiles_transformer([smiles_tokens_in, smiles_mask_in])
        x_copy = layers.Dense(100, activation='relu')(layers.Dropout(0.2)(layers.BatchNormalization()(layers.Dense(256, activation='tanh')(copy_feat_in))))
        x_mutation = layers.Dense(100, activation='relu')(layers.Dropout(0.2)(layers.BatchNormalization()(layers.Dense(256, activation='tanh')(mutation_feat_in))))
        x_gexpr = layers.Dense(100, activation='relu')(layers.Dropout(0.2)(layers.BatchNormalization()(layers.Dense(256, activation='tanh')(gexpr_feat_in))))
        x_methy = layers.Dense(100, activation='relu')(layers.Dropout(0.2)(layers.BatchNormalization()(layers.Dense(256, activation='tanh')(methy_feat_in))))
        x_gene_combined = layers.Concatenate()([x_copy, x_mutation, x_gexpr, x_methy])
        final_concat_features = layers.Concatenate()([x_gene_combined, x_drug_gcn_features])
        x = layers.Dropout(0.2)(layers.Dense(300, activation='tanh')(final_concat_features))
        x = layers.Lambda(lambda t: K.expand_dims(t, axis=-1))(x)
        x = layers.Lambda(lambda t: K.expand_dims(t, axis=1))(x)
        x = layers.MaxPooling2D(pool_size=(1, 2))(layers.Conv2D(filters=30, kernel_size=(1, 150), activation="relu", padding='valid')(x))
        x = layers.MaxPooling2D(pool_size=(1, 3))(layers.Conv2D(filters=10, kernel_size=(1, 5), activation="relu", padding='valid')(x))
        x = layers.MaxPooling2D(pool_size=(1, 3))(layers.Conv2D(filters=5, kernel_size=(1, 5), activation="relu", padding='valid')(x))
        x = layers.Flatten()(layers.Dropout(0.2)(x))
        x = layers.Dropout(0.2)(x)
        output = layers.Dense(1, name='output')(x) if self.regr else layers.Dense(1, activation='sigmoid', name='output')(x)
        return tf.keras.Model(inputs=[node_feature_in, lcq_adj_in, smiles_tokens_in, smiles_mask_in, copy_feat_in, mutation_feat_in, gexpr_feat_in, methy_feat_in], outputs=output)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
_dbpe_instance = None
_words2idx_d_instance = None

def _preload_bpe_resources(vocab_path_bpe: str, subword_csv_path_bpe: str):
    global _dbpe_instance, _words2idx_d_instance
    if _dbpe_instance is not None and _words2idx_d_instance is not None:
        return
    if not (os.path.exists(vocab_path_bpe) and os.path.exists(subword_csv_path_bpe)):
        raise FileNotFoundError(f"BPE files not found: {vocab_path_bpe}, {subword_csv_path_bpe}")
    
    print(f"Preloading BPE vocabulary from: {vocab_path_bpe}")
    with codecs.open(vocab_path_bpe, encoding='utf-8') as bpe_codes_file:
        _dbpe_instance = BPE(bpe_codes_file, merges=-1, separator='')
    
    print(f"Preloading subword units map from: {subword_csv_path_bpe}")
    try:
        sub_csv = pd.read_csv(subword_csv_path_bpe)
        idx2word_d = sub_csv['index'].values
        _words2idx_d_instance = dict(zip(idx2word_d, range(0, len(idx2word_d))))
        print("BPE resources preloaded successfully.")
    except Exception as e:
        print(f"ERROR: Failed to preload BPE subword map '{subword_csv_path_bpe}': {e}")
        sys.exit(1)

def bpe_drug2emb_encoder_from_train_optimized(smile: str, max_bpe_len: int):
    if _dbpe_instance is None or _words2idx_d_instance is None:
        raise RuntimeError("BPE resources not preloaded.")
    t1 = _dbpe_instance.process_line(str(smile)).split()
    processed_t1 = [_words2idx_d_instance.get(token, 0) for token in t1]
    i1 = np.asarray(processed_t1)
    l = len(i1)
    if l < max_bpe_len:
        i = np.pad(i1, (0, max_bpe_len - l), 'constant', constant_values=0)
        input_mask = ([1] * l) + ([0] * (max_bpe_len - l))
    else:
        i = i1[:max_bpe_len]
        input_mask = [1] * max_bpe_len
    return i, np.asarray(input_mask)

def generate_drug_inputs_for_model_optimized(smiles_string: str, featurizer_instance: MolGraphConvFeaturizerForPredict, max_atoms_padding: int, bpe_max_len: int):
    if not smiles_string or pd.isna(smiles_string) or str(smiles_string).strip().lower() == 'nan':
        return None
    mol = None
    try:
        mol = Chem.MolFromSmiles(smiles_string)
    except:
        mol = None
    if not mol:
        return None
    if mol.GetNumAtoms() == 0:
        return None
    try:
        mol = Chem.AddHs(mol)
    except:
        return None
    try:
        graph_data = featurizer_instance.featurize([mol])[0]
        node_features_raw = graph_data.node_features
        edge_indices = graph_data.edge_index
        edge_features_raw = graph_data.edge_features
        num_atoms_in_mol = node_features_raw.shape[0]
        if num_atoms_in_mol == 0:
            return None
        adj_matrix_temp = np.zeros((num_atoms_in_mol, num_atoms_in_mol, featurizer_instance.actual_bond_feat_dim), dtype=np.float32)
        if edge_features_raw is not None and edge_indices.shape[1] > 0:
            for j in range(edge_indices.shape[1]):
                src, dest = edge_indices[0, j], edge_indices[1, j]
                adj_matrix_temp[src, dest] = edge_features_raw[j]
        padded_node_features = np.zeros((max_atoms_padding, featurizer_instance.actual_atom_feat_dim), dtype=np.float32)
        num_atoms_to_copy = min(num_atoms_in_mol, max_atoms_padding)
        padded_node_features[:num_atoms_to_copy] = node_features_raw[:num_atoms_to_copy]
        padded_adj_matrix = np.zeros((max_atoms_padding, max_atoms_padding, featurizer_instance.actual_bond_feat_dim), dtype=np.float32)
        padded_adj_matrix[:num_atoms_to_copy, :num_atoms_to_copy] = adj_matrix_temp[:num_atoms_to_copy, :num_atoms_to_copy]
        node_features_final, adj_matrix_final = padded_node_features, padded_adj_matrix
    except:
        return None
    try:
        smiles_tokens_final, smiles_mask_final = bpe_drug2emb_encoder_from_train_optimized(smiles_string, bpe_max_len)
    except:
        return None
    return node_features_final, adj_matrix_final, smiles_tokens_final, smiles_mask_final

def load_new_drugs_dataframe(filepath: str) -> pd.DataFrame:
    print(f"Loading new drug SMILES from: {filepath}")
    try:
        new_drug_smiles_df = pd.read_csv(filepath, header=0, names=['drug_id', 'smiles'])
    except FileNotFoundError:
        print(f"ERROR: New drug file '{filepath}' not found.")
        sys.exit(1)
    new_drug_smiles_df['drug_id'] = new_drug_smiles_df['drug_id'].astype(str)
    new_drug_smiles_df['smiles'] = new_drug_smiles_df['smiles'].astype(str)
    print(f"Successfully loaded {len(new_drug_smiles_df)} drug SMILES strings.")
    return new_drug_smiles_df

def load_cell_lines_for_model(gene_info_filepath: str):
    print(f"Loading cell line data from: {gene_info_filepath}")
    try:
        gene_df = pd.read_csv(gene_info_filepath, sep=',', index_col=[0])
    except FileNotFoundError:
        print(f"ERROR: Gene info file '{gene_info_filepath}' not found.")
        sys.exit(1)
    processed_cell_lines_dict = {}
    problematic_cell_lines_count = 0
    for cell_line_id in tqdm(gene_df.columns, desc="Processing Cell Lines"):
        cell_data_raw_list = gene_df[cell_line_id].values
        processed_gene_data_for_cell = []
        is_cell_line_problematic = False
        for gene_entry_str in cell_data_raw_list:
            try:
                gene_features_list = eval(str(gene_entry_str))
            except:
                is_cell_line_problematic = True
                break
            if isinstance(gene_features_list, list) and len(gene_features_list) == 4:
                processed_gene_data_for_cell.append(gene_features_list)
            else:
                is_cell_line_problematic = True
                break
        if is_cell_line_problematic or not processed_gene_data_for_cell:
            problematic_cell_lines_count += 1
            continue
        processed_data_np = np.array(processed_gene_data_for_cell, dtype=np.float32)
        if processed_data_np.ndim != 2 or processed_data_np.shape[1] != 4:
            problematic_cell_lines_count += 1
            continue
        processed_cell_lines_dict[cell_line_id] = {
            "copy": processed_data_np[:, 0],
            "gexpr": processed_data_np[:, 1],
            "mutation": processed_data_np[:, 2],
            "methy": processed_data_np[:, 3],
            "num_genes": processed_data_np.shape[0]
        }
    print(f"Successfully loaded and processed {len(processed_cell_lines_dict)} cell lines. Skipped {problematic_cell_lines_count} problematic cell lines.")
    return processed_cell_lines_dict

# ============================================================================
# Main Prediction Logic
# ============================================================================
def main_prediction_driver(config):
    print("Starting prediction process...")
    drug_featurizer = MolGraphConvFeaturizerForPredict(use_edges=True, use_chirality=True, use_partial_charge=True)
    BPE_MAX_SEQUENCE_LENGTH = 50

    _preload_bpe_resources(config.vocab_path_bpe, config.subword_csv_path_bpe)
    new_drug_smiles_df = load_new_drugs_dataframe(config.new_drug_file)
    cell_line_inputs_map = load_cell_lines_for_model(config.gene_info_file)
    all_cell_line_ids = sorted(list(cell_line_inputs_map.keys()))

    if new_drug_smiles_df.empty or not all_cell_line_ids:
        print("No valid drug or cell line data. Exiting.")
        return

    print(f"Loading trained model from: {config.model_path}")
    keras_custom_objects = {"GraphConvTest": GraphConvTest, "TokenAndPositionEmbedding": TokenAndPositionEmbedding,
                            "SmilesTransformerEncoder": SmilesTransformerEncoder, "TransformerBlock": TransformerBlock, "K": K}
    try:
        loaded_model = load_model(config.model_path, custom_objects=keras_custom_objects, compile=False)
        print("Model loaded successfully using load_model().")
    except Exception as e_load1:
        print(f"Direct model loading failed: {e_load1}.\nAttempting to load by creating structure and loading weights...")
        try:
            sample_cell_key = next(iter(cell_line_inputs_map))
            num_genes_for_model = cell_line_inputs_map[sample_cell_key]['num_genes']
            model_constructor = KerasMultiSourceGCNModel_new(config.use_mut, config.use_gexp, config.use_methy, config.use_copy)
            loaded_model = model_constructor.createMaster(drug_featurizer.actual_atom_feat_dim, drug_featurizer.actual_bond_feat_dim,
                                                          BPE_MAX_SEQUENCE_LENGTH, BPE_MAX_SEQUENCE_LENGTH,
                                                          (None, num_genes_for_model), config.unit_list,
                                                          config.unit_edge_list, None, config.Dropout_rate,
                                                          config.activation, config.use_relu, config.use_bn, config.use_GMP)
            loaded_model.load_weights(config.model_path)
            print("Model loaded (structure creation + weights).")
        except Exception as e_load2:
            print(f"ERROR: Failed to load model by structure + weights: {e_load2}. Exiting.")
            return

    # ############################ START OF MODIFICATION ############################
    
    # Set the model to run eagerly, which can be more robust for complex inputs.
    loaded_model.run_eagerly = True
    print("\n[INFO] Model configured to run eagerly. This may be slightly slower but is more robust against shape-related errors.\n")

    total_predictions_to_make = len(new_drug_smiles_df) * len(all_cell_line_ids)
    print(f"Total drug-cell pairs to predict: {total_predictions_to_make}")

    all_predictions = []
    
    ## Pre-featurize all drugs to avoid redundant computation in the main loop.
    featurized_drugs = {}
    for drug_row in tqdm(new_drug_smiles_df.itertuples(index=False), total=len(new_drug_smiles_df), desc="Pre-featurizing all drugs"):
        features = generate_drug_inputs_for_model_optimized(drug_row.smiles, drug_featurizer, config.Max_atoms, BPE_MAX_SEQUENCE_LENGTH)
        if features:
            featurized_drugs[drug_row.drug_id] = features

    # Use a nested loop to predict for each drug-cell pair.
    pbar = tqdm(total=len(featurized_drugs) * len(all_cell_line_ids), desc="Predicting drug-cell pairs")
    for drug_id, drug_features in featurized_drugs.items():
        for cell_id, cell_features in cell_line_inputs_map.items():
            # Prepare input data for a single prediction.
            # Add a batch dimension (batch_size=1) using np.expand_dims.
            model_inputs = [
                np.expand_dims(drug_features[0], axis=0), # node_features
                np.expand_dims(drug_features[1], axis=0), # adj_matrix
                np.expand_dims(drug_features[2], axis=0), # smiles_tokens
                np.expand_dims(drug_features[3], axis=0), # smiles_mask
                np.expand_dims(cell_features["copy"], axis=0),
                np.expand_dims(cell_features["mutation"], axis=0),
                np.expand_dims(cell_features["gexpr"], axis=0),
                np.expand_dims(cell_features["methy"], axis=0)
            ]
            
            try:
                prediction = loaded_model.predict(model_inputs, batch_size=1, verbose=0)
                pred_val = prediction[0, 0]
                all_predictions.append([drug_id, cell_id, pred_val])
            except Exception as e:
                tqdm.write(f"ERROR during prediction for drug '{drug_id}' and cell '{cell_id}': {e}")
            
            pbar.update(1)
    
    pbar.close()

    # Save results.
    if all_predictions:
        results_df = pd.DataFrame(all_predictions, columns=['Drug_ID', 'Cell_Line_ID', 'Predicted_LN_IC50'])
        try:
            pivot_df = results_df.pivot(index='Drug_ID', columns='Cell_Line_ID', values='Predicted_LN_IC50')
            pivot_df = pivot_df.reindex(columns=all_cell_line_ids)
            pivot_df.to_csv(config.output_file, index=True)
            print(f"\nPrediction complete. Results for {len(pivot_df)} drugs saved to {config.output_file}")
        except Exception as e_pivot:
            print(f"\nPivoting failed: {e_pivot}. Saving results in long format instead.")
            results_df.to_csv(config.output_file, index=False)
            print(f"\nPrediction complete. {len(results_df)} predictions saved in long format to {config.output_file}")
    else:
        print("\nNo predictions were generated.")

    # ############################ END OF MODIFICATIONS ############################

if __name__ == '__main__':
    from rdkit import rdBase
    rdBase.DisableLog('rdApp.error')
    cli_parser = argparse.ArgumentParser(description='Unified Drug Response Prediction for New Drugs and New Cell Lines')
    cli_parser.add_argument('-gpu_id', type=str, default='0', help="GPU ID to use, or 'cpu' to disable GPU.")
    cli_parser.add_argument('-model_path', type=str, default='../prog/MyBestDeepAEG_0.7789226722858869.h5', help="Path to the trained .h5 model file.")
    cli_parser.add_argument('-new_drug_file', type=str, default='../depmap/drug_results.csv', help="Path to the input CSV file for new drugs (format: drug_id,smiles).")
    cli_parser.add_argument('-gene_info_file', type=str, default='../depmap/al_depmap.csv', help="Path to the input CSV file for new cell lines (genomic features).")
    cli_parser.add_argument('-output_file', type=str, default='predictions.csv', help="Path to save the output prediction CSV file.")
    cli_parser.add_argument('-vocab_path_bpe', type=str, default='../prog/ESPF/drug_codes_chembl_freq_1500.txt', help="Path to the BPE vocabulary file.")
    cli_parser.add_argument('-subword_csv_path_bpe', type=str, default='../prog/ESPF/subword_units_map_chembl_freq_1500.csv', help="Path to the BPE subword map CSV file.")

    # These batching parameters are no longer critical but kept for argument parsing compatibility
    cli_parser.add_argument('-prediction_batch_size', type=int, default=1)
    cli_parser.add_argument('-featurization_batch_size', type=int, default=500)
    
    # These parameters are crucial for reconstructing the model if direct loading fails.
    cli_parser.add_argument('-unit_list', nargs='+', type=int, default=[128, 128, 128])
    cli_parser.add_argument('-unit_edge_list', nargs='+', type=int, default=[32, 32, 32])
    cli_parser.add_argument('-Max_atoms', type=int, default=100)
    cli_parser.add_argument('-Dropout_rate', type=float, default=0.2)
    cli_parser.add_argument('-activation', type=str, default='gelu')
    cli_parser.add_argument('-use_bn', type=lambda x: (str(x).lower() == 'true'), default=True)
    cli_parser.add_argument('-use_relu', type=lambda x: (str(x).lower() == 'true'), default=True)
    cli_parser.add_argument('-use_GMP', type=lambda x: (str(x).lower() == 'true'), default=True)
    cli_parser.add_argument('-batch_size_set', type=int, default=1024, help="Deprecated.")
    cli_parser.add_argument('-use_mut', type=lambda x: (str(x).lower() == 'true'), default=True)
    cli_parser.add_argument('-use_gexp', type=lambda x: (str(x).lower() == 'true'), default=True)
    cli_parser.add_argument('-use_methy', type=lambda x: (str(x).lower() == 'true'), default=True)
    cli_parser.add_argument('-use_copy', type=lambda x: (str(x).lower() == 'true'), default=True)

    parsed_config_args = cli_parser.parse_args()

    if parsed_config_args.gpu_id.lower() == 'cpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("Running on CPU.")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = parsed_config_args.gpu_id
        print(f"Running on GPU: {parsed_config_args.gpu_id}")

    try:
        import deepchem
        print(f"DeepChem version: {deepchem.__version__}")
    except ImportError:
        print("ERROR: DeepChem not installed.")
        sys.exit(1)

    main_prediction_driver(parsed_config_args)