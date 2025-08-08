import random, os, sys
from collections import defaultdict
import numpy as np
import csv
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras import layers
import argparse
import codecs
from subword_nmt.apply_bpe import BPE
from rdkit import RDLogger
from tqdm import tqdm
from typing import List, Tuple
import math
import gc
import re
from functools import partial
from rdkit import Chem
from rdkit.Chem import AllChem
# --- 【【【 新增依赖 】】】 ---
try:
    from joblib import Parallel, delayed
except ImportError:
    print("ERROR: joblib not found. Please run 'pip install joblib'."); sys.exit(1)

# ==============================================================================
# SECTION 1: FEATURE GENERATION CODE (无变化, 保持完整性)
# ==============================================================================
RDLogger.DisableLog('rdApp.*')
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
    atom_type = get_atom_type_one_hot(atom, include_unknown_set=True); formal_charge = get_atom_formal_charge(atom); hybridization = get_atom_hybridization_one_hot(atom, include_unknown_set=False); acceptor_donor = get_atom_hydrogen_bonding_one_hot(atom, h_bond_infos); aromatic = get_atom_is_in_aromatic_one_hot(atom); degree = get_atom_total_degree_one_hot(atom, include_unknown_set=True); total_num_Hs = get_atom_total_num_Hs_one_hot(atom, DEFAULT_TOTAL_NUM_Hs_SET, include_unknown_set=True)
    atom_feat = np.concatenate([atom_type, formal_charge, hybridization, acceptor_donor, aromatic, degree, total_num_Hs])
    if True:
        imp_valence = get_atom_implicit_valence_one_hot_custom(atom); exp_valence = get_atom_explicit_valence_one_hot_custom(atom)
        chirality_possible_val = float(atom.HasProp('_ChiralityPossible')) if atom.HasProp('_ChiralityPossible') else 0.0; num_radical_electrons_val = float(atom.GetNumRadicalElectrons())
        atom_feat = np.concatenate([atom_feat, imp_valence, exp_valence, [chirality_possible_val, num_radical_electrons_val]])
    if use_chirality: atom_feat = np.concatenate([atom_feat, np.array(get_atom_chirality_one_hot(atom))])
    if use_partial_charge:
        partial_charge_val = [0.0]
        try:
            if atom.HasProp('_GasteigerCharge'):
                pc = atom.GetProp('_GasteigerCharge')
                if pc is not None and not np.isnan(pc) and not np.isinf(pc): partial_charge_val = [float(pc)]
        except Exception: pass
        atom_feat = np.concatenate([atom_feat, np.array(partial_charge_val)])
    return atom_feat
def _construct_bond_feature_from_script(bond: RDKitBond) -> np.ndarray:
    return np.concatenate([get_bond_type_one_hot(bond), get_bond_is_in_same_ring_one_hot(bond), get_bond_is_conjugated_one_hot(bond), get_bond_stereo_one_hot(bond)])
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
        node_features_final = np.asarray(atom_features_list, dtype=float) if atom_features_list else np.empty((0, self.actual_atom_feat_dim))
        src, dest = [], []; bond_features_list = []
        if datapoint.GetNumBonds() > 0:
            for bond in datapoint.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(); src += [start, end]; dest += [end, start]
                if self.use_edges: bond_features_list += 2 * [_construct_bond_feature_from_script(bond)]
        edge_features_final = np.asarray(bond_features_list, dtype=float) if self.use_edges and bond_features_list else np.empty((0, self.actual_bond_feat_dim))
        return GraphData(node_features=node_features_final, edge_index=np.asarray([src, dest], dtype=int), edge_features=edge_features_final)

# ============================================================================
# SECTION 2: MODEL DEFINITION (已修正语法)
# ============================================================================
class GraphConvTest(tf.keras.layers.Layer):
    def __init__(self, units, units_edge, step, activation='tanh', use_bias=False, update_edge=True, **kwargs):
        super(GraphConvTest, self).__init__(**kwargs); self.units, self.units_edge, self.step, self.activation_fn, self.use_bias, self.update_edge = units, units_edge, step, tf.keras.activations.get(activation), use_bias, update_edge
    def build(self, input_shape):
        F_in, E_in = input_shape[0][-1], input_shape[1][-1]; self.W_node_transform = self.add_weight(name=f's{self.step}_w_node_transform', shape=(F_in, F_in), trainable=True); self.W_out_linear = self.add_weight(name=f's{self.step}_w_out_linear', shape=(F_in * E_in, self.units), trainable=True)
        if self.use_bias: self.b_out_linear = self.add_weight(name=f's{self.step}_b_out_linear', shape=(self.units,), trainable=True)
        if self.update_edge: self.W_edge_hidden = self.add_weight(name=f's{self.step}_w_edge_hidden', shape=(self.units * 2, self.units_edge), trainable=True); self.W_edge_out = self.add_weight(name=f's{self.step}_w_edge_out', shape=(self.units_edge + E_in, self.units_edge), trainable=True)
        super(GraphConvTest, self).build(input_shape)
    def call(self, inputs, **kwargs):
        node_feature, lcq_adj = inputs; transformed_nodes = K.dot(node_feature, self.W_node_transform); lcq_adj_perm = tf.transpose(lcq_adj, (0, 3, 1, 2)); B, E, N, _ = tf.unstack(tf.shape(lcq_adj_perm)); _, _, F = tf.unstack(tf.shape(transformed_nodes))
        aggregated_messages = K.batch_dot(K.reshape(lcq_adj_perm, (B, E * N, N)), transformed_nodes); aggregated_messages = K.reshape(aggregated_messages, (B, E, N, F)); node_self_contribution = K.expand_dims(transformed_nodes, axis=1); aggregated_nodes = aggregated_messages + node_self_contribution
        aggregated_nodes = K.permute_dimensions(aggregated_nodes, (0, 2, 3, 1)); aggregated_nodes = K.reshape(aggregated_nodes, (B, N, F * E)); node_embeddings = K.dot(aggregated_nodes, self.W_out_linear)
        if self.use_bias: node_embeddings += self.b_out_linear; node_embeddings = self.activation_fn(node_embeddings)
        if self.update_edge:
            tiled_src = tf.tile(tf.expand_dims(node_embeddings, 2), [1, 1, N, 1]); tiled_dest = tf.tile(tf.expand_dims(node_embeddings, 1), [1, N, 1, 1]); edge_candidate_feats = K.concatenate([tiled_src, tiled_dest], axis=3)
            edge_hidden = K.dot(edge_candidate_feats, self.W_edge_hidden); edge_update_input = K.concatenate([tf.nn.relu(edge_hidden), lcq_adj], axis=3); updated_lcq_adj = K.dot(edge_update_input, self.W_edge_out); return [node_embeddings, tf.nn.relu(updated_lcq_adj)]
        return [node_embeddings, lcq_adj]
    def get_config(self): config = super().get_config(); config.update({'units': self.units, 'units_edge': self.units_edge}); return config
class TransformerBlock(layers.Layer): # ... (内容不变)
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.2, **kwargs): super().__init__(**kwargs); self.att, self.ffn, self.layernorm1, self.layernorm2, self.dropout1, self.dropout2 = layers.MultiHeadAttention(num_heads, embed_dim), tf.keras.Sequential([layers.Dense(ff_dim, "gelu"), layers.Dense(embed_dim)]), layers.LayerNormalization(1e-6), layers.LayerNormalization(1e-6), layers.Dropout(rate), layers.Dropout(rate)
    def call(self, inputs, attention_mask=None, training=None): attn_output = self.att(inputs, inputs, attention_mask=attention_mask); out1 = self.layernorm1(inputs + self.dropout1(attn_output, training=training)); ffn_output = self.ffn(out1); return self.layernorm2(out1 + self.dropout2(ffn_output, training=training))
class TokenAndPositionEmbedding(layers.Layer): # ... (内容不变)
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs): super().__init__(**kwargs); self.maxlen, self.vocab_size, self.embed_dim, self.token_emb, self.pos_emb = maxlen, vocab_size, embed_dim, layers.Embedding(vocab_size, embed_dim), layers.Embedding(maxlen, embed_dim)
    def call(self, inputs): return self.token_emb(inputs) + self.pos_emb(tf.range(tf.shape(inputs)[-1]))
class SmilesTransformerEncoder(layers.Layer): # ... (内容不变)
    def __init__(self, maxlen=50, vocab_size=2586, embed_dim=128, num_heads=8, ff_dim=512, **kwargs): super().__init__(**kwargs); self.embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim); self.encoder_blocks = [TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(1)]
    def call(self, inputs, training=None): x_tokens, padding_mask = inputs; attention_mask = (1.0 - tf.cast(tf.expand_dims(tf.expand_dims(padding_mask, 1), 1), tf.float32)) * -1e9; x = self.embedding_layer(x_tokens); x = self.encoder_blocks[0](x, attention_mask=attention_mask, training=training); return x[:, 0, :]
class KerasMultiSourceGCNModel_new(object):
    def createMaster(self, drug_dim, edge_dim, smiles_dim, mask_dim, cellline_dim_tuple, units_list, unit_edge_list, **kwargs):
        node_in, lcq_in, smiles_in, mask_in = layers.Input((100, drug_dim)), layers.Input((100, 100, edge_dim)), layers.Input((smiles_dim,)), layers.Input((mask_dim,))
        copy_in, mut_in, gexp_in, meth_in = layers.Input((cellline_dim_tuple[1],)), layers.Input((cellline_dim_tuple[1],)), layers.Input((cellline_dim_tuple[1],)), layers.Input((cellline_dim_tuple[1],))
        
        # --- 【【【 核心修改点: 兼容 Python < 3.8 】】】 ---
        # 将使用海象运算符的列表推导式，改写为标准的 for 循环
        gcn_out = [node_in, lcq_in]
        for i, (u, ue) in enumerate(zip(units_list, unit_edge_list)):
            gcn_out = GraphConvTest(u, ue, i, 'gelu')(gcn_out)
            # 使用临时列表来存储 Dropout 后的结果
            temp_gcn_out_list = []
            for l in gcn_out:
                temp_gcn_out_list.append(layers.Dropout(0.2)(l))
            gcn_out = temp_gcn_out_list
        # ----------------------------------------------------

        gcn_feat = layers.Concatenate()([layers.GlobalMaxPooling1D()(gcn_out[0]), layers.GlobalMaxPooling2D()(gcn_out[1])])
        smiles_feat = SmilesTransformerEncoder(smiles_dim, 2586, 128)( [smiles_in, mask_in] )
        cell_feat = layers.Concatenate()([layers.Dense(100, 'relu')(layers.Dropout(0.2)(layers.Dense(256, 'tanh')(feat))) for feat in [copy_in, mut_in, gexp_in, meth_in]])
        x = layers.Concatenate()([cell_feat, gcn_feat]); x = layers.Dropout(0.2)(layers.Dense(300, 'tanh')(x)); x = layers.Lambda(lambda t: K.expand_dims(t, axis=-1))(x); x = layers.Lambda(lambda t: K.expand_dims(t, axis=1))(x)
        x = layers.MaxPooling2D((1, 2))(layers.Conv2D(30, (1, 150), "relu")(x)); x = layers.MaxPooling2D((1, 3))(layers.Conv2D(10, (1, 5), "relu")(x)); x = layers.MaxPooling2D((1, 3))(layers.Conv2D(5, (1, 5), "relu")(x))
        x = layers.Flatten()(layers.Dropout(0.2)(x)); output = layers.Dense(1)(layers.Dropout(0.2)(x))
        return tf.keras.Model([node_in, lcq_in, smiles_in, mask_in, copy_in, mut_in, gexp_in, meth_in], output)

# ============================================================================
# SECTION 3: HELPER FUNCTIONS (无变化)
# ============================================================================
_dbpe_instance, _words2idx_d_instance = None, None
def _preload_bpe_resources(vocab_path_bpe: str, subword_csv_path_bpe: str):
    global _dbpe_instance, _words2idx_d_instance
    if _dbpe_instance and _words2idx_d_instance: return
    with codecs.open(vocab_path_bpe, encoding='utf-8') as f: _dbpe_instance = BPE(f, -1, '')
    _words2idx_d_instance = dict(zip(pd.read_csv(subword_csv_path_bpe)['index'].values, range(2586)))

def process_single_drug_tf(drug_row, featurizer_instance, max_atoms, bpe_max_len):
    drug_id, smiles_string = drug_row
    drug_id, smiles_string = str(drug_id).strip(), str(smiles_string).strip() if pd.notna(smiles_string) else None
    if not smiles_string: return None
    try:
        t1 = _dbpe_instance.process_line(smiles_string).split(); i1 = np.asarray([_words2idx_d_instance.get(t, 0) for t in t1])
        l = len(i1); smiles_tokens, smiles_mask = (np.pad(i1, (0, bpe_max_len-l)), np.array([1]*l+[0]*(bpe_max_len-l))) if l < bpe_max_len else (i1[:bpe_max_len], np.ones(bpe_max_len))
        mol = Chem.MolFromSmiles(smiles_string)
        if not mol or mol.GetNumAtoms() == 0: return None
        mol = Chem.AddHs(mol); graph_data = featurizer_instance.featurize([mol])[0]
        node_features_raw, edge_indices, edge_features_raw = graph_data.node_features, graph_data.edge_index, graph_data.edge_features
        num_atoms = node_features_raw.shape[0]
        adj_matrix = np.zeros((max_atoms, max_atoms, featurizer_instance.actual_bond_feat_dim), dtype=np.float32)
        if edge_features_raw is not None and edge_indices.shape[1] > 0:
            for j in range(edge_indices.shape[1]):
                src, dest = edge_indices[0, j], edge_indices[1, j]
                if src < max_atoms and dest < max_atoms: adj_matrix[src, dest] = edge_features_raw[j]
        node_features = np.zeros((max_atoms, featurizer_instance.actual_atom_feat_dim), dtype=np.float32)
        num_atoms_to_copy = min(num_atoms, max_atoms); node_features[:num_atoms_to_copy] = node_features_raw[:num_atoms_to_copy]
        return {'drug_id': drug_id, 'features': [node_features, adj_matrix, smiles_tokens, smiles_mask]}
    except Exception: return None

def load_cell_lines_for_model(gene_info_filepath: str):
    print(f"Loading cell line data from: {gene_info_filepath}")
    gene_df = pd.read_csv(gene_info_filepath, sep=',', index_col=[0])
    processed_cell_lines_dict = {}
    for cell_line_id in tqdm(gene_df.columns, desc="Processing Cell Lines"):
        try:
            cell_data_raw_list = [eval(str(g)) for g in gene_df[cell_line_id].values]
            processed_data_np = np.array(cell_data_raw_list, dtype=np.float32)
            if processed_data_np.ndim == 2 and processed_data_np.shape[1] == 4:
                processed_cell_lines_dict[cell_line_id] = {"copy": processed_data_np[:, 0], "gexpr": processed_data_np[:, 1], "mutation": processed_data_np[:, 2], "methy": processed_data_np[:, 3]}
        except: continue
    print(f"Successfully processed {len(processed_cell_lines_dict)} cell lines.")
    return processed_cell_lines_dict

# ============================================================================
# SECTION 4: MAIN PREDICTION LOGIC (无变化)
# ============================================================================
def main(config):
    # --- 步骤 1: 初始化和加载 ---
    print("--- Starting High-Performance Chunked Prediction for DeepAEG ---")
    drug_featurizer = MolGraphConvFeaturizerForPredict(use_edges=True, use_chirality=True, use_partial_charge=True)
    BPE_MAX_LEN = 50
    _preload_bpe_resources(config.vocab_path_bpe, config.subword_csv_path_bpe)
    
    cell_line_inputs_map = load_cell_lines_for_model(config.gene_info_file)
    if not cell_line_inputs_map: print("No valid cell lines. Exiting."); return
    
    print(f"Loading trained model from: {config.model_path}")
    keras_objects = {"GraphConvTest": GraphConvTest, "TokenAndPositionEmbedding": TokenAndPositionEmbedding, "SmilesTransformerEncoder": SmilesTransformerEncoder, "TransformerBlock": TransformerBlock, "K": K}
    try: loaded_model = load_model(config.model_path, custom_objects=keras_objects, compile=False)
    except Exception:
        num_genes = next(iter(cell_line_inputs_map.values()))['copy'].shape[0]
        model_builder = KerasMultiSourceGCNModel_new(True, True, True, True)
        loaded_model = model_builder.createMaster(drug_featurizer.actual_atom_feat_dim, drug_featurizer.actual_bond_feat_dim,
                                                  BPE_MAX_LEN, BPE_MAX_LEN, (None, num_genes),
                                                  config.unit_list, config.unit_edge_list)
        loaded_model.load_weights(config.model_path)
    print("Model loaded successfully.")
    
    # --- 步骤 2: 分块预测主循环 ---
    print("\n--- Starting Chunked Drug File Processing ---")
    try:
        chunk_iterator = pd.read_csv(config.new_drug_file, header=None, names=['drug_id', 'smiles'],
                                     chunksize=config.drug_chunk_size, low_memory=True)
    except FileNotFoundError: print(f"ERROR: Drug file not found: {config.new_drug_file}"); return
    
    output_base, _ = os.path.splitext(config.output_file)
    
    start_from_chunk = config.start_chunk
    if start_from_chunk <= 0:
        output_dir = os.path.dirname(output_base) or '.'
        existing_files = [f for f in os.listdir(output_dir) if f.startswith(os.path.basename(output_base) + "_") and f.endswith(".csv")]
        last_completed_chunk = 0
        for f in existing_files:
            num_str = re.search(r'_(\d+)\.csv$', f)
            if num_str: last_completed_chunk = max(last_completed_chunk, int(num_str.group(1)))
        start_from_chunk = last_completed_chunk + 1
        print(f"\n--- 自动检测到上次已完成到大块 {last_completed_chunk}。将从大块 {start_from_chunk} 开始继续... ---")
    else:
        print(f"\n--- 用户指定从大块 {start_from_chunk} 开始运行... ---")
        
    for chunk_num, drug_chunk_df in enumerate(chunk_iterator, 1):
        if chunk_num < start_from_chunk:
            print(f"快速跳过已处理的大块 {chunk_num}...")
            continue
            
        print(f"\n" + "="*20 + f" Processing Chunk {chunk_num} " + "="*20)
        t_chunk_start = time.time()
        
        print(f"Step 1: Featurizing {len(drug_chunk_df)} drugs in parallel...")
        drug_rows = [tuple(x) for x in drug_chunk_df[['drug_id', 'smiles']].to_numpy()]
        processing_func = partial(process_single_drug_tf, featurizer_instance=drug_featurizer, max_atoms=config.Max_atoms, bpe_max_len=BPE_MAX_LEN)
        featurized_drugs = Parallel(n_jobs=config.num_workers, backend="multiprocessing")(
            delayed(processing_func)(row) for row in tqdm(drug_rows, desc="Featurizing Drugs (CPU)")
        )
        featurized_drugs = [d for d in featurized_drugs if d is not None]
        
        if not featurized_drugs:
            print("当前大块没有有效的药物可供处理，跳过。"); continue
            
        print(f"Step 2: Starting prediction for {len(featurized_drugs)} valid drugs...")
        all_results_for_chunk = []
        num_small_batches = math.ceil(len(featurized_drugs) / config.small_batch_size)
        
        for i in tqdm(range(0, len(featurized_drugs), config.small_batch_size), total=num_small_batches, desc="Prediction Batches (GPU)"):
            small_batch_drugs = featurized_drugs[i : i + config.small_batch_size]
            batch_node_features = np.array([d['features'][0] for d in small_batch_drugs]); batch_adj_matrices = np.array([d['features'][1] for d in small_batch_drugs])
            batch_smiles_tokens = np.array([d['features'][2] for d in small_batch_drugs]); batch_smiles_masks = np.array([d['features'][3] for d in small_batch_drugs])
            batch_drug_ids = [d['drug_id'] for d in small_batch_drugs]
            
            for cell_id, cell_features in cell_line_inputs_map.items():
                num_drugs_in_batch = len(small_batch_drugs)
                batch_cell_copy = np.tile(cell_features["copy"], (num_drugs_in_batch, 1)); batch_cell_gexpr = np.tile(cell_features["gexpr"], (num_drugs_in_batch, 1))
                batch_cell_mutation = np.tile(cell_features["mutation"], (num_drugs_in_batch, 1)); batch_cell_methy = np.tile(cell_features["methy"], (num_drugs_in_batch, 1))
                
                model_inputs = [batch_node_features, batch_adj_matrices, batch_smiles_tokens, batch_smiles_masks, batch_cell_copy, batch_cell_mutation, batch_cell_gexpr, batch_cell_methy]
                
                try:
                    predictions = loaded_model.predict(model_inputs, batch_size=config.gpu_batch_size, verbose=0)
                    for j in range(num_drugs_in_batch):
                        all_results_for_chunk.append([batch_drug_ids[j], cell_id, predictions[j, 0]])
                except Exception as e:
                    tqdm.write(f"ERROR during prediction for a batch with cell '{cell_id}': {e}")
        
        if all_results_for_chunk:
            chunk_results_df = pd.DataFrame(all_results_for_chunk, columns=['Drug_ID', 'Cell_Line_ID', 'Predicted_LN_IC50'])
            chunk_output_path = f"{output_base}_{chunk_num}.csv"
            try:
                output_dir = os.path.dirname(chunk_output_path)
                if output_dir: os.makedirs(output_dir, exist_ok=True)
                pivot_df = chunk_results_df.pivot(index='Drug_ID', columns='Cell_Line_ID', values='Predicted_LN_IC50')
                pivot_df.to_csv(chunk_output_path, index=True)
                print(f"\nChunk {chunk_num} results (wide format) saved to {chunk_output_path}")
            except Exception as e:
                print(f"\nError saving chunk {chunk_num} results: {e}")
        
        t_chunk_end = time.time()
        print(f"Chunk {chunk_num} finished in {t_chunk_end - t_chunk_start:.2f}s.")

    print(f"\n" + "="*20 + " Script Finished " + "="*20)

if __name__ == '__main__':
    # --- 命令行参数解析 ---
    parser = argparse.ArgumentParser(description='DeepAEG 高性能分块预测脚本')
    # ... (参数定义与之前版本相同)
    parser.add_argument('-new_drug_file', type=str, default='../test/predict_all_np.csv', help="包含所有待预测药物的CSV文件路径。")
    parser.add_argument('-gene_info_file', type=str, default='../test/al_1.csv', help="新细胞系的基因组特征文件路径。")
    parser.add_argument('-output_file', type=str, default='DeepAEG_predictions.csv', help="输出文件基础名，最终会是 '基础名_1.csv' 等。")
    parser.add_argument('-model_path', type=str, default='../prog/MyBestDeepAEG_0.7789226722858869.h5', help="训练好的 .h5 模型文件路径。")
    parser.add_argument('-vocab_path_bpe', type=str, default='../prog/ESPF/drug_codes_chembl_freq_1500.txt', help="BPE 词汇表文件路径。")
    parser.add_argument('-subword_csv_path_bpe', type=str, default='../prog/ESPF/subword_units_map_chembl_freq_1500.csv', help="BPE subword map CSV 路径。")
    parser.add_argument('-drug_chunk_size', type=int, default=10000, help="每个药物大块的大小。")
    parser.add_argument('-small_batch_size', type=int, default=1000, help="每个预测小批量包含的药物数量。")
    parser.add_argument('-gpu_batch_size', type=int, default=512, help="在GPU上进行预测时的内部批次大小。")
    parser.add_argument('-num_workers', type=int, default=-1, help="用于药物特征化的CPU核心数 (-1 表示使用所有可用核心)。")
    parser.add_argument('-start_chunk', type=int, default=0, help='从哪个大块编号开始运行 (0为自动检测)。')
    parser.add_argument('-gpu_id', type=str, default='0', help="要使用的GPU ID，或 'cpu'。")
    parser.add_argument('-unit_list', nargs='+', type=int, default=[128, 128, 128])
    parser.add_argument('-unit_edge_list', nargs='+', type=int, default=[32, 32, 32])
    parser.add_argument('-Max_atoms', type=int, default=100)
    
    config = parser.parse_args()

    if config.gpu_id.lower() == 'cpu': os.environ["CUDA_VISIBLE_DEVICES"] = "-1"; print("Running on CPU.")
    else: os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id; print(f"Running on GPU: {config.gpu_id}")

    main(config)