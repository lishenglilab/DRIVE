import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp

seed_value = 1
np.random.seed(seed_value)
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# GCN based model
class GCNNet(torch.nn.Module):
    def __init__(self, n_output=1, n_filters=32, num_features_xd=78, num_features_xc=1329, output_dim=128, dropout=0.2):

        super(GCNNet, self).__init__()

        # SMILES graph branch
        self.n_output = n_output
        self.conv1 = GCNConv(num_features_xd, num_features_xd)
        self.conv2 = GCNConv(num_features_xd, num_features_xd*2)
        self.conv3 = GCNConv(num_features_xd*2, num_features_xd * 4)
        self.fc_g1 = torch.nn.Linear(num_features_xd*4, 1024)
        self.fc_g2 = torch.nn.Linear(1024, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # cell line feature
        self.fc1_xt_ge = nn.Linear(num_features_xc, 512)
        self.fc2_xt_ge = nn.Linear(512, 1024)
        self.fc3_xt_ge = nn.Linear(1024, output_dim)

        # combined layers
        self.fc1 = nn.Linear(2*output_dim, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.out = nn.Linear(128, self.n_output)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, data):
        # get graph input
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # protein input feed-forward:
        target_ge = data.target_ge
        # print(len(data.target_ge))
        target_ge = target_ge[:, None, :]

        # get protein input
        x = self.conv1(x, edge_index)
        x = self.relu(x)

        x = self.conv2(x, edge_index)
        x = self.relu(x)

        x = self.conv3(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)       # global max pooling

        # flatten
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)
        x = self.dropout(x)

        # DNN layers
        fc_xt_ge = self.fc1_xt_ge(target_ge)
        fc_xt_ge = F.relu(fc_xt_ge)
        fc_xt_ge = self.fc2_xt_ge(fc_xt_ge)
        fc_xt_ge = F.relu(fc_xt_ge)
        fc_xt_ge = F.dropout(fc_xt_ge, p=0.2, training=self.training)

        # flatten ge
        xt_ge = fc_xt_ge.view(-1, fc_xt_ge.shape[1] * fc_xt_ge.shape[2])
        xt_ge = self.fc3_xt_ge(xt_ge)

        # concat
        xc = torch.cat((x, xt_ge), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out, x



