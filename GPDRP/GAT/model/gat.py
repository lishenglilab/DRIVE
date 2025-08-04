import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_max_pool as gmp
import numpy as np

seed_value = 1
np.random.seed(seed_value)
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# GAT  model
class GATNet(torch.nn.Module):
    def __init__(self, num_features_xd=78, n_output=1, num_features_xc=1329,
                     n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):
        super(GATNet, self).__init__()

        # graph layers
        self.gcn1 = GATConv(num_features_xd, num_features_xd, heads=10, dropout=dropout)
        self.gcn2 = GATConv(num_features_xd * 10, output_dim, dropout=dropout)
        self.gcn3 = GATConv(output_dim, output_dim, dropout=dropout)
        self.fc_g1 = nn.Linear(output_dim, output_dim)

        # cell line feature
        self.fc1_xt_ge = nn.Linear(num_features_xc, 512)
        self.fc2_xt_ge = nn.Linear(512, 1024)
        self.fc3_xt_ge = nn.Linear(1024, output_dim)

        # combined layers
        self.fc1 = nn.Linear(2*output_dim, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.out = nn.Linear(128, n_output)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, data):
        # graph input feed-forward
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.gcn1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gcn2(x, edge_index)
        x = self.relu(x)
        x = self.gcn3(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)          # global max pooling
        x = self.fc_g1(x)
        x = self.relu(x)

        # protein input feed-forward:
        target_ge = data.target_ge
        target_ge = target_ge[:, None, :]
        # 1d conv layers
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
        out = nn.Sigmoid()(out)
        return out, x

#