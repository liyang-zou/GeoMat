"""JARVIS-DFT crystal-property graph dataset."""

import json
import os

import numpy as np
import torch
from pymatgen.core import Structure
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

from data.dataloader import get_idx


class JARVISData(InMemoryDataset):
    """Build graphs from a JARVIS JSON release for one numeric target."""

    def __init__(self, root, target, json_file='jdft_3d-8-18-2021.json',
                 max_num_nbr=12, radius=8.0, neighbor_strategy='k-nearest',
                 transform=None, pre_transform=None, pre_filter=None):
        self.target = target
        self.json_file = json_file
        self.max_num_nbr = max_num_nbr
        self.radius = radius
        self.neighbor_strategy = neighbor_strategy
        source = os.path.join(root, json_file)
        if not os.path.isfile(source):
            raise FileNotFoundError(source)
        with open(source, encoding='utf-8') as stream:
            records = json.load(stream)
        self.records = []
        for record in records:
            try:
                value = float(record.get(target))
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                self.records.append(record)
        if not self.records:
            raise ValueError(f'No finite values found for JARVIS target {target!r}')
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    def download(self):
        pass

    @property
    def processed_file_names(self):
        safe_target = ''.join(c if c.isalnum() else '_' for c in self.target)
        return [f'data_{safe_target}_{self.neighbor_strategy}_{self.radius:g}.pt']

    def process(self):
        graphs = []
        for record in tqdm(self.records, desc=f'Processing {self.target}'):
            atoms = record['atoms']
            structure = Structure(
                lattice=atoms['lattice_mat'], species=atoms['elements'],
                coords=atoms['coords'],
                coords_are_cartesian=bool(atoms.get('cartesian', True)))
            lattice = torch.tensor(structure.lattice.matrix, dtype=torch.float)
            atom_idx = torch.tensor(
                [[site.specie.number] for site in structure], dtype=torch.long)
            src, images, dst = get_idx(
                structure, self.neighbor_strategy, self.radius, record['jid'],
                self.max_num_nbr)
            edge_index = torch.tensor([src, dst], dtype=torch.long)
            images = torch.tensor(np.asarray(images), dtype=torch.float)
            pos = torch.tensor(structure.cart_coords, dtype=torch.float)
            displacement = (pos[edge_index[0]] + images @ lattice
                            - pos[edge_index[1]])
            graphs.append(Data(
                x=atom_idx, edge_index=edge_index,
                y=torch.tensor([float(record[self.target])], dtype=torch.float),
                pos=pos, displacement=displacement,
                distance=torch.linalg.vector_norm(displacement, dim=-1),
                id=record['jid']))
        if self.pre_filter is not None:
            graphs = [graph for graph in graphs if self.pre_filter(graph)]
        if self.pre_transform is not None:
            graphs = [self.pre_transform(graph) for graph in graphs]
        self.save(graphs, self.processed_paths[0])
