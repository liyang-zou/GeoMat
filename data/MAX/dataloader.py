"""Crystal graphs and manuscript motif IDs for the M2AX dataset."""

import csv
import os
import random
import warnings
import numpy as np
import torch
from pymatgen.core.structure import Structure
from pymatgen.core import Composition
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm


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


M_ELEMENTS = frozenset(('Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni',
                        'Cu', 'Y', 'Zr', 'Nb', 'Mo', 'Hf', 'Ta', 'W'))


def get_max_type(formula, struct_type):
    """Map source motifs to the four mutually exclusive manuscript types."""
    if struct_type == 'type4':
        return 'Type4'
    if struct_type == 'type1':
        metals = {element.symbol for element in Composition(formula).elements
                  if element.symbol in M_ELEMENTS}
        if len(metals) == 1:
            return 'Type1'
        if len(metals) == 2:
            return 'Type4'
        raise ValueError(f'Expected one or two M elements in {formula}')
    if struct_type == 'type2':
        return 'Type2'
    if struct_type == 'type3':
        return 'Type3'
    raise ValueError(f'Unknown MAX struct_type: {struct_type}')


def get_material_id(formula, struct_type):
    """Return the unique TypeN-formula identifier without changing the formula."""
    return f'{get_max_type(formula, struct_type)}-{formula}'


class CIFData(InMemoryDataset):
    """Load root/M2AX_data_3681.csv and cache periodic crystal graphs.

    Required columns: formula, struct_type, eform, cif. ``struct_type`` may
    contain either the source labels or the released type1-type4 labels. Each cif string is
    used as supplied, with no lattice/coordinate replacement or relaxation.
    y is the eform column, in meV/atom as reported by the manuscript.
    id is the full TypeN-formula string, not a numeric tensor.

    Graph fields: x, edge_index, y, displacement, distance, and id.
    Input rows are shuffled with random_seed before graph construction.
    train.csv/test.csv select motifs by full ID, independently of this order.
    Graphs are cached in root/processed/data.pt. Use a fresh root or remove
    this cache when changing the data, processing logic or graph settings.
    transform, pre_transform and pre_filter are standard PyG hooks.
    """
    def __init__(self, root, max_num_nbr=12, radius=8, neighbor_strategy='k-nearest',
                 random_seed=123,transform=None, pre_transform=None, pre_filter=None):
        assert os.path.exists(root), 'root_dir does not exist!'

        id_prop_file = os.path.join(root, 'M2AX_data_3681.csv')
        assert os.path.exists(id_prop_file), 'M2AX_data_3681.csv does not exist!'

        with open(id_prop_file, encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            missing = {'formula', 'struct_type', 'eform', 'cif'}.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f'Missing CSV columns: {sorted(missing)}')
            self.id_prop_data = [row for row in reader]
        ids = [get_material_id(row['formula'], row['struct_type'])
               for row in self.id_prop_data]
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate TypeN-formula identifiers in MAX input')


        random.seed(random_seed)
        random.shuffle(self.id_prop_data)
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
        for row in tqdm(self.id_prop_data, desc='Processing...'):
            cif_id = get_material_id(row['formula'], row['struct_type'])
            target = float(row['eform'])
            if not np.isfinite(target):
                raise ValueError(f'{cif_id} has a non-finite eform')
            crystal = Structure.from_str(row['cif'], fmt='cif')

            lattice = crystal.lattice.matrix

            atom_idx = np.vstack([crystal[i].specie.number for i in range(len(crystal))])
            src_idx, src_image, dst_idx = get_idx(crystal, self.neighbor_strategy, self.radius, cif_id, self.max_num_nbr)
            edge_index = torch.tensor(np.array([
                np.array(src_idx),
                np.array(dst_idx),]), dtype=torch.long)

            pos = torch.tensor(crystal.cart_coords, dtype=torch.float)
            lattice = torch.tensor(lattice, dtype=torch.float)
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
                      displacement=displacement,
                      distance=distance,
                      id=cif_id
                      )
            # Model-specific edge embeddings are computed inside the model.

            data_list.append(data)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        self.save(data_list,self.processed_paths[0])
