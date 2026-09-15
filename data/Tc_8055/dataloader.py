"""Crystal graph dataset for Tc_8055."""

import os
import warnings
import pandas as pd
import numpy as np
import torch
from pymatgen.core.structure import Structure
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm


def get_material_id(compound_id):
    """Return the dataset identifier without its storage-batch prefix."""
    return str(compound_id).split('/', 1)[-1]


def normalize_edge_order(src_idx, src_image, dst_idx, distance):
    '''
    normalize the edge order of graph data by node and edge distance
    Parameters
    ----------
    src_idx: list of src node idx
    src_image: list of src node image
    dst_idx: list of dst node idx
    distance: list of edge distance
    Returns
    ----------
    the normalized Parameters
    '''

    combined = list(zip(src_idx, src_image, dst_idx, distance))
    combined_sorted = sorted(combined, key=lambda x: (x[2], x[3]))
    src_idx, src_image, dst_idx, distance = zip(*combined_sorted)

    return src_idx, src_image, dst_idx


def get_idx(structure, neighbor_stratage, radius, id, max_num_nbr = None):
    """
    get neighbor indices to construct edge_index according to neighbor_stratage
    structure: return value or pymatgen.core.structure.Structure
    """
    all_nbrs = structure.get_all_neighbors(radius, include_index=True)
    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]  # Sort neighbors by distance.
    src_idx, src_image, dst_idx, distance = [], [], [], []

    if neighbor_stratage == 'k-nearest':
        # Note that according to the cgcnn's method, it cannot guarantee the construction of an undirected graph!
        for i, nbrs in enumerate(all_nbrs):
            if 0 < len(nbrs) < max_num_nbr:
                warnings.warn('{} not find enough neighbors to build graph. '
                              'If it happens frequently, consider increase '
                              'radius.'.format(id))
                src_idx += list(map(lambda x: x[2], nbrs))
                src_image += list(map(lambda x: x.image, nbrs))
                dst_idx += [i] * len(nbrs)
                distance += list(map(lambda x: x[1], nbrs))
            elif len(nbrs) == 0:
                warnings.warn('{} crystal structure has atom whoes neightbors are zero, use the nearest atom as a neighbor '.format(id))
                nbrs = structure.get_neighbors(structure[i], 20)  # get neightbors at higher radius
                nbrs = sorted(nbrs, key=lambda x: x[1])
                src_idx += [nbrs[0][2]]
                src_image += [nbrs[0].image]
                dst_idx += [i]
                distance += [nbrs[0][1]]
            else:
                src_idx += list(map(lambda x: x[2],
                                    nbrs[:max_num_nbr]))  # Source atom indices.
                src_image += list(map(lambda x: x.image,
                                      nbrs[:max_num_nbr])) # Periodic cell images of source atoms.
                dst_idx += [i] * max_num_nbr
                distance += list(map(lambda x: x[1],
                                     nbrs[:max_num_nbr]))
    elif neighbor_stratage == 'radius_graph':   #
        for i, nbrs in enumerate(all_nbrs):
            if len(nbrs) == 0:
                warnings.warn('{} crystal structure has atom whoes neightbors are zero, use the nearest atom as a neighbor '.format(id))
                nbrs = structure.get_neighbors(structure[i], 20)  # get neightbors at higher radius
                nbrs = sorted(nbrs, key=lambda x: x[1])
                src_idx += [nbrs[0][2], i]
                src_image += [nbrs[0].image, tuple([-x for x in nbrs[0].image])]
                dst_idx += [i, nbrs[0][2]]
                distance += [nbrs[0][1], nbrs[0][1]]
            else:
                src_idx += list(map(lambda x: x[2], nbrs))  # Source atom indices.
                src_image += list(map(lambda x: x.image, nbrs))  # Periodic cell images of source atoms.
                dst_idx += [i] * len(nbrs)
                distance += list(map(lambda x: x[1], nbrs))
    else:
        raise ValueError('Error neighbor_stratage parameter！')

    src_idx, src_image, dst_idx = normalize_edge_order(src_idx, src_image, dst_idx, distance)

    return src_idx, src_image, dst_idx


class CIFData(InMemoryDataset):
    """Load task inputs and cache periodic crystal graphs with PyG.

    Input: ``root/superconductors_clean_dataset.pkl``.
    A pickled pandas DataFrame containing compound_id, target_Tc_AD
    (K), and cif_string.

    Parameters
    ----------
    root : str
        Directory containing the task input file.
    max_num_nbr : int
        Maximum neighbors per atom for the k-nearest strategy.
    radius : float
        Neighbor search radius in angstroms.
    neighbor_strategy : str
        Either "k-nearest" or "radius_graph".
    random_seed : int
        Accepted for compatibility; this loader does not shuffle the input.
    transform, pre_transform, pre_filter : callable, optional
        Standard PyG access-time transform and processing hooks.

    Graph fields: x, edge_index, y, pos, displacement, distance, nbr_image, lattice,
    lattice_lengths, lattice_angles, and id.
    ``id`` stores the portion of ``compound_id`` after the optional
    ``batch-*/`` storage prefix.
    Graphs are cached in ``root/processed/data.pt``. An existing cache
    is reused; use a separate root when changing inputs or graph settings.
    """
    def __init__(self, root, max_num_nbr=12, radius=8, neighbor_strategy='k-nearest',
                 random_seed=123,transform=None, pre_transform=None, pre_filter=None):
        assert os.path.exists(root), 'root_dir does not exist!'

        id_prop_file = os.path.join(root, 'superconductors_clean_dataset.pkl')
        assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'

        self.id_prop_data = pd.read_pickle(id_prop_file)

        self.max_num_nbr, self.radius = max_num_nbr, radius
        self.neighbor_strategy = neighbor_strategy

        super(CIFData,self).__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    def download(self):
        pass

    @property
    def raw_file_names(self) :

        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        data_list = []
        for row in tqdm(self.id_prop_data.itertuples(), desc='Processing...'):
            cif_id = get_material_id(row.compound_id)
            target = row.target_Tc_AD
            crystal = Structure.from_str(row.cif_string, fmt="cif")

            lattice = crystal.lattice.matrix
            lattice_lengths = np.array(crystal.lattice.abc)
            lattice_angles = np.array(crystal.lattice.angles)
            lattice_angles = lattice_angles * np.pi / 180

            atom_idx = np.vstack([crystal[i].specie.number for i in range(len(crystal))])
            src_idx, src_image, dst_idx = get_idx(crystal, self.neighbor_strategy, self.radius, cif_id, self.max_num_nbr)
            edge_index = torch.tensor(np.array([
                np.array(src_idx),
                np.array(dst_idx),]), dtype=torch.long)

            pos = torch.tensor(crystal.cart_coords, dtype=torch.float)
            lattice = torch.tensor(lattice, dtype=torch.float)
            lattice_lengths = torch.tensor(lattice_lengths, dtype=torch.float).view(1,-1)
            lattice_angles = torch.tensor(lattice_angles, dtype=torch.float).view(1,-1)
            atom_idx = torch.tensor(atom_idx, dtype=torch.long)      # Atomic numbers for graph nodes.
            src_image = torch.tensor(src_image, dtype=torch.float)   # Periodic cell images of source atoms.
            target = torch.tensor([float(target)], dtype=torch.float)

            src_pos = pos[edge_index[0, :], :] + torch.matmul(src_image, lattice)
            dst_pos = pos[edge_index[1, :], :]
            displacement = src_pos - dst_pos
            distance = torch.norm(displacement, p=2, dim=-1)  # One distance per edge.

            data=Data(x=atom_idx,
                      edge_index=edge_index,
                      y=target,
                      pos=pos,
                      displacement=displacement,
                      distance=distance,
                      nbr_image=src_image,
                      lattice=lattice,
                      lattice_lengths=lattice_lengths,
                      lattice_angles=lattice_angles,
                      id=cif_id
                      )
            # Model-specific edge embeddings are computed inside the model.

            data_list.append(data)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        self.save(data_list,self.processed_paths[0])
