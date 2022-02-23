# Idea: increase the size of the hidden vector as the number of nodes are removed
import pdb

import torch
import torch.nn.functional as F

from torch.nn import Linear
from torch_sparse import spspmm
from torch_geometric.nn import TopKPooling, GCNConv, global_mean_pool, GraphConv, GATv2Conv, ASAPooling
from torch_geometric.utils import (add_self_loops, sort_edge_index,
                                   remove_self_loops)
from torch_scatter import scatter_mean, scatter_max

def readout(x, batch):
    x_mean = scatter_mean(x, batch, dim=0)
    x_max, _ = scatter_max(x, batch, dim=0)
    return torch.cat((x_mean, x_max), dim=-1)

class GraphReduction(torch.nn.Module):

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 n_layers: int,
                 ratio: float = 0.8,
                 dropout: float = 0.1,
                 ):
        super(GraphReduction, self).__init__()
        if type(ratio) != list:
            ratio = [ratio for _ in range(n_layers)]

        self.n_layers = n_layers

        self.conv1 = GCNConv(in_channels=in_channels,
                             out_channels=hidden_channels)
        self.pool1 = ASAPooling(in_channels=hidden_channels,
                                ratio=ratio[0],
                                dropout=dropout)

        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()

        for i in range(n_layers - 1):
            self.convs.append(GCNConv(in_channels=hidden_channels,
                                      out_channels=hidden_channels))
            self.pools.append(ASAPooling(in_channels=hidden_channels,
                                         ratio=ratio[i],
                                         dropout=dropout))

        self.lin1 = Linear(2*hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)

        self.reset_layers()

    def reset_layers(self):
        self.conv1.reset_parameters()
        self.pool1.reset_parameters()
        for conv, pool in zip(self.convs, self.pools):
            conv.reset_parameters()
            pool.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, edge_weight, batch, perm = self.pool1(x=x,
                                                             edge_index=edge_index,
                                                             edge_weight=None,
                                                             batch=batch
                                                             )
        xs = readout(x, batch)

        for conv, pool in zip(self.convs, self.pools):
            x = F.relu(conv(x, edge_index))
            x, edge_index, edge_weight, batch, perm = pool(x=x,
                                                           edge_index=edge_index,
                                                           edge_weight=edge_weight,
                                                           batch=batch
                                                           )
            xs += readout(x, batch)

        embedding = global_mean_pool(x, batch)
        prediction = F.relu(self.lin1(xs))
        prediction = F.dropout(prediction, p=0.5, training=self.training)
        prediction = self.lin2(prediction)

        return prediction, x, edge_index, embedding
