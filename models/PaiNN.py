import torch
from torch import nn
from torch_geometric.nn.conv import MessagePassing
import torch.nn.functional as F

def sinc_expansion(edge_dist: torch.Tensor, edge_size: int, cutoff: float, eps: float = 1e-8):
    """
    calculate sinc radial basis function:

    sin(n *pi*d/d_cut)/d
    """
    n = torch.arange(edge_size, device=edge_dist.device) + 1
    return torch.sin(edge_dist.unsqueeze(-1) * n * torch.pi / cutoff) / (edge_dist.unsqueeze(-1) + eps)


def gaussian_expansion(edge_dist: torch.Tensor, edge_size: int, cutoff: float):
    '''
    Gaussian: exp(-(x - mu)^2 / (2 * sigma^2))
    '''
    centers = torch.linspace(0, cutoff, edge_size, device=edge_dist.device)
    width = cutoff / edge_size

    diff = edge_dist.unsqueeze(-1) - centers.unsqueeze(0)
    return torch.exp(-0.5 * (diff / width) ** 2)

def cosine_cutoff(edge_dist: torch.Tensor, cutoff: float):
    """
    Calculate cutoff value based on distance.
    This uses the cosine Behler-Parinello cutoff function:

    f(d) = 0.5*(cos(pi*d/d_cut)+1) for d < d_cut and 0 otherwise
    """

    return torch.where(
        edge_dist < cutoff,
        0.5 * (torch.cos(torch.pi * edge_dist / cutoff) + 1),
        torch.tensor(0.0, device=edge_dist.device, dtype=edge_dist.dtype),
    )

class EquivariantLayerNorm(nn.Module):
    def __init__(self, hidden_dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.norm_s = nn.LayerNorm(hidden_dim)
        self.scale_v = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, s, v):
        """
        s: [N, hidden_dim] scalar
        v: [N, 3, hidden_dim] vector
        """
        s_out = self.norm_s(s)

        v_squared_norm = (v ** 2).sum(dim=1)
        v_rms = torch.sqrt(v_squared_norm.mean(dim=-1, keepdim=True) + self.eps)
        v_out = v / v_rms.unsqueeze(1)           # [N, 3, F] / [N, 1, 1]
        v_out = v_out * self.scale_v.reshape(1, 1, -1)   # [N, 3, F] * [1, 1, F]

        return s_out, v_out

class PainnConv(MessagePassing):
    def __init__(self, node_size: int, edge_size: int, cutoff: float, eps: float = 1e-8):
        super().__init__(node_dim=0)

        self.edge_size = edge_size
        self.node_size = node_size
        self.cutoff = cutoff

        self.scalar_message_mlp = nn.Sequential(
            nn.Linear(node_size, node_size),
            nn.SiLU(),
            nn.Linear(node_size, node_size * 3),
        )

        self.filter_layer = nn.Linear(edge_size, node_size * 3)

        self.update_U = nn.Linear(node_size, node_size, bias=False)
        self.update_V = nn.Linear(node_size, node_size, bias=False)

        self.update_mlp = nn.Sequential(
            nn.Linear(node_size * 2, node_size),
            nn.SiLU(),
            nn.Linear(node_size, node_size * 3),
        )
        self.layer_norm = EquivariantLayerNorm(node_size)
        self.eps = eps

    def forward(self, node_scalar, node_vector, edge_index, edge_diff, edge_dist):
        filter_weight = self.filter_layer(sinc_expansion(edge_dist, self.edge_size, self.cutoff))
        filter_weight = filter_weight * cosine_cutoff(edge_dist, self.cutoff).unsqueeze(-1)
        scalar_out = self.scalar_message_mlp(node_scalar)
        '''
        res_scalar = node_scalar
        res_vector = node_vector
        node_scalar_norm, node_vector_norm = self.layer_norm(node_scalar, node_vector)
        '''
        combined_node = self.propagate(edge_index, x=node_scalar, v=node_vector, scalar=scalar_out,
                                       edge_diff=edge_diff, edge_dist=edge_dist, filter_weight=filter_weight)
        node_scalar = combined_node[:, 0, :].squeeze(1)
        node_vector = combined_node[:, 1:, :]
        return node_scalar, node_vector

    def message(self, scalar_j, v_j, edge_diff, edge_dist, filter_weight):
        filter_out = filter_weight * scalar_j
        gate_state_vector, gate_edge_vector, message_scalar = torch.split(
            filter_out,
            self.node_size,
            dim = 1,
        )
        message_vector =  v_j * gate_state_vector.unsqueeze(1)      # shape: (num_edges, 3, node_size) * (num_edges, 1, node_size)
        edge_vector = gate_edge_vector.unsqueeze(1) * (edge_diff / (edge_dist.unsqueeze(-1) + self.eps)).unsqueeze(-1)     # shape: (num_edges, 1, node_size) * (num_edges, 3, 1)
        message_vector = message_vector + edge_vector

        combined_msg = torch.cat([message_scalar.unsqueeze(1), message_vector], dim=1)
        return combined_msg

    def update(self, aggr_out, x, v):
        new_node_scalar = aggr_out[:, 0, :] + x
        new_node_vector = aggr_out[:, 1:, :] + v

        Uv = self.update_U(new_node_vector)
        Vv = self.update_V(new_node_vector)

        Vv_norm = torch.sqrt(torch.sum(Vv ** 2, dim=1) + self.eps)
        mlp_input = torch.cat((Vv_norm, new_node_scalar), dim=1)
        mlp_output = self.update_mlp(mlp_input)

        a_vv, a_sv, a_ss = torch.split(
            mlp_output,
            self.node_size,
            dim = 1,
        )

        delta_v = a_vv.unsqueeze(1) * Uv
        inner_prod = torch.sum(Uv * Vv, dim=1)
        delta_s = a_sv * inner_prod + a_ss

        node_scalar_updated, node_vector_updated = self.layer_norm(new_node_scalar + delta_s, new_node_vector + delta_v)
        return torch.cat([node_scalar_updated.unsqueeze(1), node_vector_updated], dim=1)


class PainnEncoder(nn.Module):
    """PainnModel without edge updating"""

    def __init__(
            self,
            num_interactions,
            hidden_state_size,
            cutoff=5.0,
            **kwargs,
    ):
        super().__init__()

        num_embedding = 94 + 1  # number of all elements in MP and OQMD, one for mask token
        self.cutoff = cutoff
        self.num_interactions = num_interactions
        self.hidden_state_size = hidden_state_size
        self.edge_embedding_size = 20

        # Setup atom embeddings
        self.atom_embedding = nn.Embedding(num_embedding, hidden_state_size)
        # Setup Conv laters
        self.convs = nn.ModuleList(
            [
                PainnConv(self.hidden_state_size, self.edge_embedding_size, self.cutoff)
                for _ in range(self.num_interactions)
            ]
        )
        # Setup readout function
        self.readout_mlp = nn.Sequential(
            nn.Linear(self.hidden_state_size, self.hidden_state_size),
            nn.SiLU(),
            nn.Linear(self.hidden_state_size, 1),
        )

    def forward(self, data):
        x, edge_index, edge_diff, batch = data.x, data.edge_index, data.displacement, data.batch
        edge_dist = torch.norm(data.displacement, p=2, dim=-1)
        node_scalar = self.atom_embedding(x.view(-1))
        if hasattr(data, 'masked_atom_indices'):
            node_scalar[data.masked_atom_indices] = torch.zeros_like(node_scalar[data.masked_atom_indices])
        node_vector = torch.zeros((x.shape[0], 3, self.hidden_state_size),
                                  device=edge_diff.device,
                                  dtype=edge_diff.dtype,
                                  )  # shape: (num_nodes, 3, num_features)

        for conv in self.convs:
            node_scalar, node_vector = conv(node_scalar, node_vector, edge_index, edge_diff, edge_dist)
        return node_scalar, node_vector


class PainnGate(nn.Module):
    """Gated equivariant function in PaiNN"""

    def __init__(self, node_in_size: int, node_out_size: int, eps: float = 1e-8):
        super().__init__()

        self.update_U = nn.Linear(node_in_size, node_out_size, bias=False)
        self.update_V = nn.Linear(node_in_size, node_out_size, bias=False)

        self.update_mlp = nn.Sequential(
            nn.Linear(node_in_size + node_out_size, node_in_size),
            nn.SiLU(),
            nn.Linear(node_in_size, node_out_size * 2),
        )

        self.eps = eps

    def forward(self, node_scalar, node_vector):
        Uv = self.update_U(node_vector)
        Vv = self.update_V(node_vector)

        Vv_norm = torch.sqrt(torch.sum(Vv ** 2, dim=1) + self.eps)
        mlp_input = torch.cat((Vv_norm, node_scalar), dim=1)
        mlp_output = self.update_mlp(mlp_input)

        a_vv, new_node_scalar = torch.split(
            mlp_output,
            Uv.shape[-1],
            dim=1,
        )
        a_vv = torch.sigmoid(a_vv)
        new_node_vector = a_vv.unsqueeze(1) * Uv

        return new_node_scalar, new_node_vector


