import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import numpy as np

seed_value = 1
np.random.seed(seed_value)
torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# GINConv model
class GINConvNet(torch.nn.Module):
    def __init__(self, n_output=1,num_features_xd=78, num_features_xc=1329,
                 n_filters=32, output_dim=128, dropout=0.2):

        super(GINConvNet, self).__init__()

        dim = 32
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output
        # convolution layers
        nn1 = Sequential(Linear(num_features_xd, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        self.fc1_xd = Linear(dim, output_dim)


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
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # print(x)
        # print(data.target)
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)

        x = global_add_pool(x, batch)
        x = F.relu(self.fc1_xd(x))
        x = F.dropout(x, p=0.2, training=self.training)

        # protein input feed-forward:
        target_ge = data.target_ge
        # print(len(data.target_ge))
        target_ge = target_ge[:, None, :]

        # 1d conv layers
        fc_xt_ge = self.fc1_xt_ge(target_ge)
        fc_xt_ge = F.relu(fc_xt_ge)
        fc_xt_ge = self.fc2_xt_ge(fc_xt_ge)
        fc_xt_ge = F.relu(fc_xt_ge)
        fc_xt_ge = F.dropout(fc_xt_ge, p=0.2, training=self.training)


        # flatten ge
        xt_ge = fc_xt_ge.view(-1,fc_xt_ge.shape[1] * fc_xt_ge.shape[2])
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
