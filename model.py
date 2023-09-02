import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv, SGConv, global_mean_pool, global_max_pool, global_add_pool, SAGPooling

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, encoder_type='GCN'):
        super(Encoder, self).__init__()
        self.hidden_channels = hidden_channels
        if encoder_type=='GCN':
            self.conv1 = GCNConv(in_channels, self.hidden_channels)
        elif encoder_type=='GAT':
            self.conv1 = GATConv(in_channels, self.hidden_channels)
        elif encoder_type=='GraphSAGE':
            self.conv1 = SAGEConv(in_channels, self.hidden_channels)
        elif encoder_type=='SGC':
            self.conv1 = SGConv(in_channels, self.hidden_channels)
        elif encoder_type=='GIN':
            self.mlp = nn.Linear(in_channels, self.hidden_channels)
            self.conv1 = GINConv(self.mlp)

        self.prelu1 = nn.PReLU(self.hidden_channels)

    def forward(self, x, edge_index, edge_attr=None):

        if edge_attr!=None:
            x1 = self.conv1(x, edge_index, edge_attr)
        else:
            x1 = self.conv1(x, edge_index)
        x1 = self.prelu1(x1)
        x1 = F.normalize(x1)
        return x1 


        
class Pool(nn.Module):
    def __init__(self, in_channels, ratio=1.0):
        super(Pool, self).__init__()
        self.sag_pool = SAGPooling(in_channels, ratio)
        self.lin1 = torch.nn.Linear(in_channels * 2, in_channels)
        
    def forward(self, x, edge, batch, type='mean_pool'):
        if type == 'mean_pool':
            return global_mean_pool(x, batch)
        elif type == 'max_pool':
            return global_max_pool(x, batch)
        elif type == 'sum_pool':
            return global_add_pool(x, batch)
        elif type == 'sag_pool':
            x1, _, _, batch, _, _ = self.sag_pool(x, edge, batch=batch)
            return global_mean_pool(x1, batch)
            
        


class Scorer(nn.Module):
    def __init__(self, hidden_size):
        super(Scorer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_size, hidden_size))

    def forward(self, input1, input2):
        output = torch.sigmoid(torch.sum(input1 * torch.matmul(input2, self.weight), dim = -1))
        return output


