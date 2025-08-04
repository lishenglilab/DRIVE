import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm

from BAN import BANLayer


class BANDRP(nn.Module):
    def __init__(self, cell_exp_dim, cell_mut_dim, cell_cnv_dim, **config):
        super(BANDRP, self).__init__()
        drug_out_dim = config['drug']['drug_out_dim']
        cell_out_dim = config['cell']['cell_out_dim']
        ban_heads = config['ban']['ban_heads']
        mlp_in_dim = config['mlp']['mlp_in_dim']
        mlp_hidden_dim = config['mlp']['mlp_hidden_dim']
        self.drug_embedding = DrugEmbedding(drug_out_dim, use_morgan=True, use_espf=True, use_pubchem=True)
        self.cell_embedding = CellEmbedding(cell_exp_dim, cell_mut_dim, cell_cnv_dim, cell_out_dim,
                                            use_exp=True, use_mut=True, use_cnv=True)
        self.ban = weight_norm(
            BANLayer(v_dim=drug_out_dim, q_dim=cell_out_dim, h_dim=mlp_in_dim, h_out=ban_heads,
                     dropout=config['ban']['dropout_rate']), name='h_mat', dim=None)
        self.mlp = MLP(mlp_in_dim, mlp_hidden_dim, out_dim=1)

    def forward(self, drug_data, cell_data):
        v_d = self.drug_embedding(drug_data)
        v_c = self.cell_embedding(cell_data[0], cell_data[1], cell_data[2])
        f, att = self.ban(v_d, v_c)
        predict = self.mlp(f)
        predict = torch.squeeze(predict)
        return predict, att


class DrugEmbedding(nn.Module):
    def __init__(self, out_dim, use_morgan=True, use_espf=True, use_pubchem=True):
        super(DrugEmbedding, self).__init__()
        self.use_morgan = use_morgan
        self.use_espf = use_espf
        self.use_pubchem = use_pubchem
        # morgan finger print
        morgan_fp_layers = [
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, out_dim)
        ]
        self.morgan_fp = nn.Sequential(*morgan_fp_layers)

        # espf finger print
        espf_fp_layers = [
            nn.Linear(2586, 1024),
            nn.ReLU(),
            nn.Linear(1024, out_dim)
        ]
        self.espf_fp = nn.Sequential(*espf_fp_layers)

        # pubchem finger print
        pubchem_fp_layers = [
            nn.Linear(881, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        ]
        self.pubchem_fp = nn.Sequential(*pubchem_fp_layers)

    def forward(self, finger_print):
        x_drug = []

        if self.use_morgan:
            morgan_f = self.morgan_fp(finger_print[0])
            x_drug.append(morgan_f)

        if self.use_espf:
            espf_f = self.espf_fp(finger_print[1])
            x_drug.append(espf_f)

        if self.use_pubchem:
            puchem_f = self.pubchem_fp(finger_print[2])
            x_drug.append(puchem_f)

        out = torch.stack(x_drug, dim=1)
        return out


class CellEmbedding(nn.Module):
    def __init__(self, exp_in_dim, mut_in_dim, cnv_in_dim, out_dim, use_exp=True, use_mut=True,
                 use_cnv=True):
        super(CellEmbedding, self).__init__()
        self.use_exp = use_exp
        self.use_mut = use_mut
        self.use_cnv = use_cnv

        # exp_layer
        self.gexp_fc1 = nn.Linear(exp_in_dim, 256)
        self.gexp_bn = nn.BatchNorm1d(256)
        self.gexp_fc2 = nn.Linear(256, out_dim)

        # mut_layer
        self.mut_fc1 = nn.Linear(mut_in_dim, 256)
        self.mut_bn = nn.BatchNorm1d(256)
        self.mut_fc2 = nn.Linear(256, out_dim)

        # cnv_layer
        self.cnv_fc1 = nn.Linear(cnv_in_dim, 256)
        self.cnv_bn = nn.BatchNorm1d(256)
        self.cnv_fc2 = nn.Linear(256, out_dim)


    def forward(self, expression_data, mutation_data, cnv_data):
        x_cell = []
        #  expression representation
        if self.use_exp:
            x_exp = self.gexp_fc1(expression_data)
            x_exp = F.relu(self.gexp_bn(x_exp))
            x_exp = F.relu(self.gexp_fc2(x_exp))
            x_cell.append(x_exp)

        # mutation representation
        if self.use_mut:
            x_mut = self.mut_fc1(mutation_data)
            x_mut = F.relu(self.mut_bn(x_mut))
            x_mut = F.relu(self.mut_fc2(x_mut))
            x_cell.append(x_mut)

        # methylation representation
        if self.use_cnv:
            x_cnv = self.cnv_fc1(cnv_data)
            x_cnv = F.relu(self.cnv_bn(x_cnv))
            x_cnv = F.relu(self.cnv_fc2(x_cnv))
            x_cell.append(x_cnv)

        x_cell = torch.stack(x_cell, dim=1)
        return x_cell


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(hidden_dim)):
            if i == len(hidden_dim) - 1:
                break
            layers.append(nn.Linear(hidden_dim[i], hidden_dim[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim[i + 1]))
        self.fc1 = nn.Linear(in_dim, hidden_dim[0])
        self.bn1 = nn.BatchNorm1d(hidden_dim[0])
        self.hidden = nn.Sequential(*layers)
        self.fc2 = nn.Linear(hidden_dim[-1], out_dim)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.hidden(x)
        x = self.fc2(x)
        return x
