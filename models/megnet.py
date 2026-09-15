from __future__ import print_function, division

import torch
import torch.nn as nn
from torch_geometric.nn.aggr import Set2Set

from models.model import MEGConv, MLP
from data.dataloader import GaussianDistance

class MEGNet(nn.Module):
    '''
    dim_node_embedding: Dimension of node embedding.
    dim_edge_embedding: Dimension of edge embedding.
    dim_state_embedding: Dimension of state embedding.
    ntypes_state: Number of state types.
    nblocks: Number of blocks.
    hidden_layer_sizes_input: Architecture of dense layers before the graph convolution
    hidden_layer_sizes_conv: Architecture of dense layers for message and update functions. Notice that the output of conv must equal the input of ff due to residual connection
    hidden_layer_sizes_output: Architecture of dense layers for concatenated features after graph convolution
    nlayers_set2set: Number of layers in Set2Set layer
    niters_set2set: Number of iterations in Set2Set layer

    x: atomic number, shape (num_atom, 1)
    edge_attr: distance between atoms, shape (num_edge)
    state: global state vector, shape (num_graph, state_fea_len)
    '''
    def __init__(self, dim_node_embedding: int = 16,  dim_edge_embedding: int = 100, dim_state_embedding: int = 2,
                 hidden_layer_sizes_input: tuple[int, ...] = (64, 32),
                 hidden_layer_sizes_conv: tuple[int, ...] = (64, 64, 32),
                 hidden_layer_sizes_output: tuple[int, ...] = (32, 16),
                 nblocks: int = 3,
                 nlayers_set2set: int = 1,
                 niters_set2set: int = 2,
                 **kwargs,
                 ):
        super(MEGNet, self).__init__()
        self.dim_state_embedding = dim_state_embedding
        self.node_embedding = nn.Embedding(94, dim_node_embedding)  #The dataset used in Paper only has 94 atom type
        self.edge_init = GaussianDistance(var = 0.5, centers=torch.linspace(0, 5.0, dim_edge_embedding))
        self.node_ff = MLP(dim_node_embedding, hidden_layer_sizes_input[:-1], hidden_layer_sizes_input[-1], activation='Softplus2')
        self.edge_ff = MLP(dim_edge_embedding, hidden_layer_sizes_input[:-1], hidden_layer_sizes_input[-1], activation='Softplus2')
        self.state_ff = MLP(dim_state_embedding, hidden_layer_sizes_input[:-1], hidden_layer_sizes_input[-1], activation='Softplus2')

        self.convs = nn.ModuleList([MEGConv(hidden_layer_sizes_input, hidden_layer_sizes_conv, has_ff = False) if i==0
                                    else MEGConv(hidden_layer_sizes_input, hidden_layer_sizes_conv, has_ff = True)
                                    for i in range(nblocks)])

        self.node_s2s = Set2Set(in_channels=hidden_layer_sizes_conv[-1], processing_steps=niters_set2set, num_layers=nlayers_set2set)
        self.edge_s2s = Set2Set(in_channels=hidden_layer_sizes_conv[-1], processing_steps=niters_set2set, num_layers=nlayers_set2set)

        self.output = MLP(5 * hidden_layer_sizes_conv[-1], hidden_layer_sizes_output, 1, activation='Softplus2')

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.distance, data.batch
        if hasattr(data, 'state'):
            state = data.state
        else:
            state = torch.zeros([int(batch.max() + 1), self.dim_state_embedding], dtype=data.distance.dtype, device=data.x.device)

        x = self.node_embedding(x.view(-1) - 1)
        edge_attr = self.edge_init(edge_attr.view(-1))
        x = self.node_ff(x)
        edge_attr = self.edge_ff(edge_attr)
        state = self.state_ff(state)

        for conv_func in self.convs:
            x, edge_attr, state = conv_func(x, edge_index, edge_attr, batch, state)

        x = self.node_s2s(x, batch)
        edge_attr = self.edge_s2s(edge_attr, batch[edge_index[0]])

        output = self.output(torch.cat([x, edge_attr, state], dim=-1))
        return output





