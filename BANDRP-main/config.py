from yacs.config import CfgNode as CN

_C = CN()

# data path
_C.path = CN()
rootpath = ""
_C.path.savedir = rootpath + "github_upload/output_dir"
_C.path.response = rootpath + "TEST_DATA/GDSC/ic50.csv"
_C.path.mutation = rootpath + "TEST_DATA/CCLE/mu.csv"
_C.path.cnv = rootpath + "TEST_DATA/CCLE/cn.csv"
_C.path.expression = rootpath + "TEST_DATA/CCLE/exp.csv"
_C.path.morgan = rootpath + "TEST_DATA/Drug/morgan_encoding.pkl"
_C.path.espf = rootpath + "TEST_DATA/Drug/espf_encoding.pkl"
_C.path.psfp = rootpath + "TEST_DATA/Drug/pubchem_encoding.pkl"

# model params
_C.model = CN()
_C.model.lr = 0.001
_C.model.weight_decay = 0
_C.model.epoch = 150
_C.model.cuda_id = 0

# Drug
_C.drug = CN()
_C.drug.drug_out_dim = 128

# Cell
_C.cell = CN()
_C.cell.cell_out_dim = 128

# Ban
_C.ban = CN()
_C.ban.ban_heads = 3
_C.ban.dropout_rate = 0.5

# Mlp
_C.mlp = CN()
_C.mlp.mlp_in_dim = 256
_C.mlp.mlp_hidden_dim = [512, 128]


def get_cfg_defaults():
    return _C.clone()
