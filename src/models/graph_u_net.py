# Idea: increase the size of the hidden vector as the number of nodes are removed
import pdb

import torch
import torch.nn.functional as F

from torch.nn import Linear
from torch_sparse import spspmm
from torch_geometric.nn import TopKPooling, GCNConv, global_mean_pool
from torch_geometric.utils import (add_self_loops, sort_edge_index,
                                   remove_self_loops)
from torch_geometric.utils.repeat import repeat


class GraphUNet(torch.nn.Module):
    r"""The Graph U-Net model from the `"Graph U-Nets"
    <https://arxiv.org/abs/1905.05178>`_ paper which implements a U-Net like
    architecture with graph pooling and unpooling operations.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        dim_gr_embdding (int): Size of the graph embedding
        out_channels (int): Size of each output sample.
        depth (int): The depth of the U-Net architecture.
        pool_ratios (float or [float], optional): Graph pooling ratio for each
            depth. (default: :obj:`0.5`)
        sum_res (bool, optional): If set to :obj:`False`, will use
            concatenation for integration of skip connections instead
            summation. (default: :obj:`True`)
        act (torch.nn.functional, optional): The nonlinearity to use.
            (default: :obj:`torch.nn.functional.relu`)
    """

    def __init__(self, in_channels: int, hidden_channels: int,
                 dim_gr_embedding: int, out_channels: int, depth: int,
                 pool_ratios=0.5, sum_res=True, act=F.relu, layer=GCNConv):
        super(GraphUNet, self).__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = depth
        self.pool_ratios = repeat(pool_ratios, depth)
        self.act = act
        self.sum_res = sum_res
        self.layer = layer

        channels = hidden_channels

        self.down_convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()

        self.down_convs.append(self.layer(in_channels,
                                          channels))
        self.down_convs.append(self.layer(channels,
                                          channels))

        for i in range(depth):
            self.pools.append(TopKPooling(channels, self.pool_ratios[i]))
            self.down_convs.append(self.layer(channels,
                                              channels))
            self.down_convs.append(self.layer(channels,
                                              channels))

        self.conv_to_gr_embedding = self.layer(channels,
                                               dim_gr_embedding)
        # self.lin_to_gr_embedding = Linear(channels//2, dim_gr_embedding)

        self.lin = Linear(dim_gr_embedding, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.down_convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()

    def augment_depth(self, pool_ratio=0.5):
        self.pools.append(TopKPooling(in_channels=self.hidden_channels,
                                      ratio=pool_ratio))
        self.down_convs.append(self.layer(self.hidden_channels,
                                          self.hidden_channels,
                                          )
                               )


    def forward(self, x, edge_index, batch=None):
        """"""
        # pdb.set_trace()
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        edge_weight = x.new_ones(edge_index.size(1))
        x = self.down_convs[0](x, edge_index, edge_weight)
        x = self.down_convs[1](x, edge_index, edge_weight)
        x = self.act(x)

        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        perms = []

        for i in range(1, self.depth + 1):
            edge_index, edge_weight = self.augment_adj(edge_index,
                                                       edge_weight,
                                                       x.size(0))
            x, edge_index, edge_weight, batch, perm, _ = self.pools[i - 1](
                x, edge_index, edge_weight, batch)

            idx_layer = i * 2
            x = self.down_convs[idx_layer](x, edge_index, edge_weight)
            x = self.down_convs[idx_layer+1](x, edge_index, edge_weight)
            x = self.act(x)

            if i < self.depth:
                xs += [x]
                edge_indices += [edge_index]
                edge_weights += [edge_weight]
            perms += [perm]

        x = self.act(self.conv_to_gr_embedding(x, edge_index, edge_weight))
        graph_embeddings = global_mean_pool(x, batch)
        # graph_embeddings = self.act(self.lin_to_gr_embedding(graph_embeddings))

        preds = F.dropout(graph_embeddings, p=0.5, training=self.training)
        preds = self.lin(preds)

        return preds, x, edge_index, graph_embeddings

    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = remove_self_loops(edge_index,
                                                    edge_weight)
        edge_index, edge_weight = add_self_loops(edge_index,
                                                 edge_weight,
                                                 num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index,
                                                  edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight,
                                         edge_index,
                                         edge_weight, num_nodes,
                                         num_nodes,
                                         num_nodes)
        edge_index, edge_weight = remove_self_loops(edge_index,
                                                    edge_weight)
        return edge_index, edge_weight

    # def __repr__(self):
    #     return '{}({}, {}, {}, depth={}, pool_ratios={})'.format(
    #         self.__class__.__name__, self.in_channels, self.hidden_channels,
    #         self.out_channels, self.depth, self.pool_ratios)