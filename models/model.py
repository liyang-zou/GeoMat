from __future__ import print_function, division

import os

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, Set2Set, global_add_pool
from typing import Tuple, Union
import torch.nn.functional as F
from torch import Tensor
from torch.nn import BatchNorm1d, Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairTensor
from data.dataloader import AtomInitializer, GaussianDistance
from models.PaiNN import PainnGate

class Softplus2(nn.Module):
    """
    modified Softplus activation function used in MEGNET:
    out = log(exp(x)+1) - log(2)
        = relu(x) + log(0.5 * exp(-|x|) + 0.5)
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x) + torch.log(0.5 * torch.exp(-torch.abs(x)) + 0.5)

class CGConv(MessagePassing):
    """
    Initialize CGConv by PyG
    Parameters
    ----------

    channels: int
      Number of atom hidden features
    dim: int
      Number of edge features.
    """
    def __init__(self, channels: Union[int, Tuple[int, int]], dim: int = 0,
                 aggr: str = 'add', batch_norm: bool = True,
                 bias: bool = True, **kwargs):
        super().__init__(aggr=aggr, **kwargs)
        self.channels = channels
        self.dim = dim
        self.batch_norm = batch_norm

        if isinstance(channels, int):
            channels = (channels, channels)

        self.lin_f = Linear(sum(channels) + dim, channels[1], bias=bias)
        self.lin_s = Linear(sum(channels) + dim, channels[1], bias=bias)
        if batch_norm:
            self.bn1 = BatchNorm1d(2*channels[1])
            self.bn2 = BatchNorm1d(channels[1])
        else:
            self.bn1 = None
            self.bn2 = None

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_f.reset_parameters()
        self.lin_s.reset_parameters()
        if self.batch_norm:
            self.bn1.reset_parameters()
            self.bn2.reset_parameters()

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None) -> Tensor:
        if isinstance(x, Tensor):
            x = (x, x)
        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.bn2(out) if self.batch_norm else out
        out = out + x[1]
        out = F.softplus(out)
        return out

    def message(self, x_i, x_j, edge_attr: OptTensor) -> Tensor:
        if edge_attr is None:
            z = torch.cat([x_i, x_j], dim=-1)
        else:
            z = torch.cat([x_i, x_j, edge_attr], dim=-1)
        if self.batch_norm:
            total = self.bn1(torch.cat([self.lin_f(z), self.lin_s(z)], dim=-1))
            filter, core = total.chunk(2, dim=-1)
            return filter.sigmoid() * F.softplus(core)
        else:
            return self.lin_f(z).sigmoid() * F.softplus(self.lin_s(z))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.channels}, dim={self.dim})'

class MEGConv(MessagePassing):
    '''
    hidden_layer_sizes_input: Architecture of dense layers before the graph convolution
    hidden_layer_sizes_conv: Architecture of dense layers for message and update functions. Notice that the output of conv is the input of ff due to residual connection
    x: node feature, shape (num_atom, atom_fea_len)
    edge_attr: edge feature, shape (num_edge, edge_fea_len)
    state: global state vector, shape (num_graph, state_fea_len)
    '''
    def __init__(self,
                 hidden_layer_sizes_input: tuple[int, ...] = (64, 32),
                 hidden_layer_sizes_conv: tuple[int, ...] = (64, 32, 32),
                 has_ff: bool = True, aggr: str = 'mean', norm = None, **kwargs):
        super().__init__(aggr=aggr, **kwargs)
        self.hidden_layer_sizes_input = hidden_layer_sizes_input
        self.hidden_layer_sizes_conv = hidden_layer_sizes_conv
        self.has_ff = has_ff

        self.node_ff = MLP(hidden_layer_sizes_conv[-1], hidden_layer_sizes_input[:-1], hidden_layer_sizes_input[-1], activation='Softplus2', norm=norm)
        self.edge_ff = MLP(hidden_layer_sizes_conv[-1], hidden_layer_sizes_input[:-1], hidden_layer_sizes_input[-1], activation='Softplus2', norm=norm)
        self.state_ff = MLP(hidden_layer_sizes_conv[-1], hidden_layer_sizes_input[:-1], hidden_layer_sizes_input[-1], activation='Softplus2', norm=norm)

        self.node_conv = MLP(2 * hidden_layer_sizes_input[-1] + hidden_layer_sizes_conv[-1], hidden_layer_sizes_conv[:-1], hidden_layer_sizes_conv[-1], activation='Softplus2', norm=norm)
        self.edge_conv = MLP(4 * hidden_layer_sizes_input[-1], hidden_layer_sizes_conv[:-1], hidden_layer_sizes_conv[-1], activation='Softplus2', norm=norm)
        self.state_conv = MLP(hidden_layer_sizes_input[-1] + 2 * hidden_layer_sizes_conv[-1], hidden_layer_sizes_conv[:-1], hidden_layer_sizes_conv[-1], activation='Softplus2', norm=norm)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None, batch = None, state: OptTensor = None):
        x0, edge_attr_0, state_0 = x, edge_attr, state
        if self.has_ff:
            x = self.node_ff(x)
            edge_attr = self.edge_ff(edge_attr)
            state = self.state_ff(state)

        edge_batch = batch[edge_index[0]]
        edge_attr_updated = self.edge_conv(torch.cat(
            [x[edge_index[0, :], :], x[edge_index[1, :], :],
             edge_attr, state[edge_batch]], dim=-1))

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr_updated, state=state)
        x_updated = self.node_conv(torch.cat([out, x, state[batch]], dim=-1))

        state_e = global_mean_pool(edge_attr_updated, edge_batch)
        state_x = global_mean_pool(x_updated, batch)
        state_updated = self.state_conv(torch.cat([state_e, state_x, state], dim=-1))

        return x_updated + x0, edge_attr_updated + edge_attr_0, state_updated + state_0

    def message(self, x_i, x_j, edge_attr: OptTensor, state) -> Tensor:
        return edge_attr

class encoder(nn.Module):
    '''
    For models without clear architecture, we use the following parameters to control the embedding vector dimension ————
        node_dim: the dimension of output node features
        edge_dim: the dimension of output edge features
    '''
    def __init__(self, num_layer, node_dim, edge_dim, gnn_type='CGConv', atom_init=None, norm=None, num_atom_types=94, **kwargs):
        super().__init__()
        self.num_layer = num_layer
        self.gnn_type = gnn_type
        self.readout = MLP(node_dim, [int(node_dim/2)], out_dim=1, activation='Softplus2', norm=None)
        if self.gnn_type == 'CGConv':
            assert os.path.exists(atom_init), 'atom_init.json does not exist!'
            self.atom_init = AtomInitializer(atom_init)
            self.edge_init = GaussianDistance(0.0, 8.0, 0.2)
            self.embedding = nn.Linear(self.atom_init.node_dim, node_dim)
            self.edge_embedding = nn.Linear(self.edge_init.edge_dim, edge_dim)
            self.convs = nn.ModuleList([CGConv(channels=node_dim,
                                               dim=edge_dim)
                                        for _ in range(num_layer)])
        elif self.gnn_type == 'MEGConv':   #The arhitecture is just copied from matbench
            self.atom_init = nn.Embedding(num_atom_types + 1, 64)                            # +1 for additional masked atom type
            self.edge_init = GaussianDistance(var = 0.4 * 2**0.5, centers=torch.arange(0,25,1)/25*5)       #The gdf implemented here 's factor is '1/var**2' rather than '1/2*var**2'
            self.node_ff = MLP(64, [64], 32, activation='Softplus2', norm=norm)
            self.edge_ff = MLP(25, [64], 32, activation='Softplus2', norm=norm)
            self.state_ff = MLP(1, [64], 32, activation='Softplus2', norm=norm)

            self.convs = nn.ModuleList(
                [MEGConv((64, 32), (64, 32, 32), has_ff=False, norm=norm) if i == 0
                 else MEGConv((64, 32), (64, 32, 32), has_ff=True, norm=norm)
                 for i in range(num_layer)])
        else:
            raise NotImplementedError

    def forward(self, data, denoise=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # Embed node features.
        x = self.atom_init(x.view(-1)) if self.atom_init is not None else x
        if hasattr(data, 'masked_atom_indices'):
            x[data.masked_atom_indices] = torch.zeros_like(x[data.masked_atom_indices])
        if self.gnn_type == 'CGConv':
            x = self.embedding(x)
        elif self.gnn_type == 'MEGConv':
            x = self.node_ff(x)
            if hasattr(data, 'state'):  #Embedding global state vector
                state = data.state
            else:
                state = torch.zeros([int(batch.max() + 1), 1], dtype=data.displacement.dtype,
                                    device=data.displacement.device)
            state = self.state_ff(state)
        else:
            raise NotImplementedError

        # Embed edge features.
        if denoise:
            src_pos = data.pos[data.edge_index[0, :], :] + data.lattice_displacement
            dst_pos = data.pos[data.edge_index[1, :], :]
            distance = torch.norm(src_pos - dst_pos, p=2, dim=-1)
        else:
            distance = torch.norm(data.displacement, p=2, dim=-1)

        edge_attr = self.edge_init(distance)
        if self.gnn_type == 'CGConv':
            edge_attr = self.edge_embedding(edge_attr)
        elif self.gnn_type == 'MEGConv':
            edge_attr = self.edge_ff(edge_attr)
        else:
            raise NotImplementedError

        for conv_func in self.convs:
            if self.gnn_type == 'MEGConv':
                x, edge_attr, state = conv_func(x, edge_index, edge_attr, batch, state)
            else:
                x = conv_func(x=x, edge_index=edge_index, edge_attr=edge_attr)

        if denoise:
            graph_fea = global_add_pool(x, batch)
            E = self.readout(graph_fea)
            pred_noise = -torch.autograd.grad(outputs=E, inputs=data.pos, grad_outputs=torch.ones_like(E), create_graph=True, retain_graph=True)[0]
            if self.gnn_type == 'MEGConv':
                return x, edge_attr, graph_fea, pred_noise, state
            else:
                return x, edge_attr, graph_fea, pred_noise
        else:
            graph_fea = global_mean_pool(x, batch)
            if self.gnn_type == 'MEGConv':
                return x, edge_attr, graph_fea, state
            else:
                return x, edge_attr, graph_fea

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, p=0, activation='ReLU', norm='batchnorm'):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.p = p
        prev_dim = in_dim
        layers = []
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))

            if norm == 'batchnorm':
                layers.append(nn.BatchNorm1d(h))

            if activation == 'ReLU':
                layers.append(nn.ReLU())
            elif activation == 'Softplus':
                layers.append(nn.Softplus())
            elif activation == 'Softplus2':
                layers.append(Softplus2())
            elif activation == 'SiLU':
                layers.append(nn.SiLU())
            else:
                raise NotImplementedError

            if norm == 'layernorm':
                layers.append(nn.LayerNorm(h))

            layers.append(nn.Dropout(p=self.p))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, out_dim))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

class decoder(nn.Module):
    def __init__(self, hidden_dim, out_dim, decoder_type='CGConv', MLP_hidden_dims=None, edge_dim=None, p=0, norm='batchnorm'):
        super().__init__()

        self.activation = nn.PReLU()
        self.enc_to_dec = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.decoder_type = decoder_type
        if decoder_type == 'Linear':
            self.dec = nn.Linear(hidden_dim, out_dim)
        elif decoder_type == 'MLP' and MLP_hidden_dims is not None:
            self.dec = MLP(hidden_dim, MLP_hidden_dims, out_dim, p, norm=norm)
        elif decoder_type == 'CGConv':
            self.edge_init = GaussianDistance(var=0.4 * 2 ** 0.5, centers=torch.arange(0, 25, 1) / 25 * 5)
            self.edge_embedding = nn.Linear(self.edge_init.edge_dim, edge_dim)
            self.dec = CGConv(hidden_dim, edge_dim, aggr='add')
            self.mlp = MLP(hidden_dim, MLP_hidden_dims, out_dim, p, norm=norm)
        else:
            raise NotImplementedError

    def forward(self, x, edge_index = None, edge_dist = None, mask_atom_indices = None):
        if mask_atom_indices is not None:
            x = self.activation(x)
            x = self.enc_to_dec(x)
            x[mask_atom_indices] = 0.0

        if self.decoder_type == 'Linear':
            x = self.dec(x)
        if self.decoder_type == 'MLP':
            x = self.dec(x)
        if self.decoder_type == 'CGConv':
            edge_attr = self.edge_embedding(self.edge_init(edge_dist))
            x = self.dec(x, edge_index, edge_attr)
            x = self.mlp(x)

        return x

class distance_decoder(nn.Module):
    def __init__(self, atom_dim, edge_dim, out_dim, MLP_hidden_dims, p=0, norm='layernorm'):
        super().__init__()
        self.linear = nn.Linear(3, edge_dim)
        self.mlp = MLP(2*atom_dim+edge_dim, MLP_hidden_dims, out_dim, p, norm=norm)
    def forward(self, edge_rep, nbr_image):
        nbr_image = self.linear(nbr_image)
        out = self.mlp(torch.cat([edge_rep, nbr_image], dim=-1))
        return out

class displacement_decoder(nn.Module):
    def __init__(self, atom_dim, out_dim):
        super().__init__()
        self.out = PainnGate(node_in_size=atom_dim, node_out_size=atom_dim)
        self.linear = nn.Linear(atom_dim, out_dim, bias=False)
    def forward(self, node_scalar, node_vector):
        _, node_vector = self.out(node_scalar, node_vector)
        node_vector = self.linear(node_vector).squeeze(-1)
        return node_vector

class readout(nn.Module):
    '''
    The default readout function used in MatBench or original paper for every GNN
    '''
    def __init__(self, hidden_dim, out_dim, readout_type='CGConv', MLP_hidden_dims=None, p=0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.enc_to_dec = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.readout_type = readout_type
        self.MLP_hidden_dims = MLP_hidden_dims
        if readout_type == 'CGConv' and MLP_hidden_dims is not None:
            self.out = MLP(hidden_dim, MLP_hidden_dims, out_dim, p,
                           activation='Softplus')
        elif readout_type == 'MEGConv':
            self.node_ff = nn.Linear(hidden_dim, 16, bias=True)
            self.edge_ff = nn.Linear(hidden_dim, 16, bias=True)
            self.node_s2s = Set2Set(in_channels=16, processing_steps=2, num_layers=1)
            self.edge_s2s = Set2Set(in_channels=16, processing_steps=2, num_layers=1)
            self.out = MLP(3 * 32, MLP_hidden_dims, 16,
                           activation='Softplus2', norm=None)
            self.Softplus2 = Softplus2()
            self.linear = nn.Linear(16, out_dim)
        elif readout_type == 'Painn':
            self.out = MLP(hidden_dim, MLP_hidden_dims, out_dim, p, activation='SiLU', norm=None)
        else:
            raise NotImplementedError

    def forward(self, x, batch, edge_index = None, edge_attr = None, state = None, v=None):
        if self.readout_type == 'CGConv':
            x = global_mean_pool(x, batch)
            out = self.out(x)
        elif self.readout_type == 'MEGConv':
            #Notice that this doesn't mean use MEGConv to decode node information, but means use MEGNet readout function in paper
            assert edge_index is not None, 'edge_index vector which is necessary in MEGNet readout function is None '
            assert edge_attr is not None, 'edge_attr vector which is necessary in MEGNet readout function is None '
            assert state is not None, 'state vector which is necessary in MEGNet readout function is None '

            x = self.node_ff(x)
            edge_attr = self.edge_ff(edge_attr)
            x = self.node_s2s(x, batch)
            edge_attr = self.edge_s2s(edge_attr, batch[edge_index[0]])
            out = self.out(torch.cat([x, edge_attr, state], dim=-1))
            out = self.Softplus2(out)
            out = self.linear(out)
        elif self.readout_type == 'Painn':
            x = global_mean_pool(x, batch)
            out = self.out(x)
        return out

if __name__ == '__main__':
    edge_init = GaussianDistance(var = 0.4 * 2**0.5, centers=torch.arange(0,25,1)/25*5)
    print(edge_init(torch.tensor(1.0)).shape)
