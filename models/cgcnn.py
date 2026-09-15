from __future__ import print_function, division

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from models.model import CGConv
from data.dataloader import AtomInitializer, GaussianDistance

class CrystalGraphConvNet(nn.Module):
    """
    CGCNN baseline implemented with PyTorch Geometric.
    Create a crystal graph convolutional neural network for predicting total
    material properties. By PyG.
    """
    def __init__(self, atom_init_file, dmin=0.0, dmax=8.0, step=0.2,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                 classification=False):
        """
        Initialize CrystalGraphConvNet.

        Parameters
        ----------

        atom_init_file: str
          Path to atom_init.json containing initial atom features.
        dmin, dmax, step: float
          Distance range and spacing for the Gaussian basis.
        atom_fea_len: int
          Number of hidden atom features in the convolutional layers
        n_conv: int
          Number of convolutional layers
        h_fea_len: int
          Number of hidden features after pooling
        n_h: int
          Number of hidden layers after pooling
        """
        super(CrystalGraphConvNet, self).__init__()
        self.classification = classification
        self.ari = AtomInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=dmax, step=step)
        self.embedding = nn.Linear(self.ari.node_dim, atom_fea_len)   # Project initial atom features into the hidden embedding.
        self.convs = nn.ModuleList([CGConv(channels=atom_fea_len,
                                    dim=self.gdf.edge_dim)
                                    for _ in range(n_conv)])
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h-1)])
            self.softpluses = nn.ModuleList([nn.Softplus()
                                             for _ in range(n_h-1)])
        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, 2)
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)
        if self.classification:
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()

    def forward(self, data):
        """
        Forward pass

        Parameters
        ----------
        data: torch_geometric.data.Data or Batch
          Crystal graph with x (atomic numbers), edge_index, distance
          (edge distances), and batch (graph membership).

        Returns
        -------
        torch.Tensor
          Shape (number of graphs, 1) for regression, or
          (number of graphs, 2) with log probabilities for classification.

        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.distance, data.batch
        x = self.ari(x.view(-1))
        edge_attr = self.gdf(edge_attr.view(-1))

        x = self.embedding(x)
        for conv_func in self.convs:
            x = conv_func(x, edge_index, edge_attr)

        crys_fea = global_mean_pool(x, batch)
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        if self.classification:
            crys_fea = self.dropout(crys_fea)
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))
        out = self.fc_out(crys_fea)
        if self.classification:
            out = self.logsoftmax(out)
        return out
