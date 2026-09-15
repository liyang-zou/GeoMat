"""Crystal graph datasets, atom/distance features, and SSL augmentations."""

import ast
import csv
import json
import math
import os
import random
import re
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from pymatgen.core.structure import Structure
from torch import nn
from torch_geometric.data import Data, InMemoryDataset, Batch
from torch_geometric.utils import k_hop_subgraph
from tqdm import tqdm


class GaussianDistance(nn.Module):
    """
    Expands the distance by Gaussian basis (PyTorch version).
    Supports autograd and GPU acceleration.
    """
    def __init__(self, dmin = 0.0, dmax = 8.0, step = 0.2, var = None, centers = None):
        """
        Parameters
        ----------
        dmin: float
            Minimum interatomic distance
        dmax: float
            Maximum interatomic distance
        step: float
            Step size for the Gaussian filter
        var: float or None
            Variance of the Gaussian filter (default = step)
        """
        super().__init__()
        assert dmin < dmax and (dmax - dmin) > step
        if centers is None:
            filter_centers = torch.arange(dmin, dmax + step, step)
        else:
            filter_centers = centers
        self.register_buffer('filter', filter_centers.detach())
        self.register_buffer('var', torch.tensor(var if var is not None else step).detach())
        self.edge_dim = filter_centers.shape[0]

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Apply Gaussian distance filter.

        Parameters
        ----------
        distances: torch.Tensor, shape (...), in angstrom
            Distance matrix of any shape

        Returns
        -------
        expanded: torch.Tensor, shape (..., num_filters)
            Gaussian expanded features for each distance
        """
        # distances: shape (...), filter: shape (num_filters,)
        # Result: shape (..., num_filters)

        return torch.exp(-((distances.unsqueeze(-1) - self.filter)**2) / self.var**2)


class AtomInitializer(nn.Module):
    """
    Fixed atom embedding layer using pre-defined element-wise features.
    """
    def __init__(self, embedding_path):
        super().__init__()
        assert os.path.exists(embedding_path), 'atom_init.json for CGCNN does not exist!'
        with open(embedding_path) as f:
            embedding_dict = json.load(f)
        embedding_dict = {int(k): np.array(v, dtype=np.float32) for k, v in embedding_dict.items()}

        # Build index-to-vector tensor
        self.max_atom_idx = max(embedding_dict.keys())
        self.node_dim = len(embedding_dict[1])
        embedding_tensor = torch.zeros(self.max_atom_idx + 1, self.node_dim, dtype=torch.float32)   #The zeroth embedding is full zero vector which means masked embedding

        for z, vec in embedding_dict.items():
            embedding_tensor[z] = torch.tensor(vec, dtype=torch.float32)

        # Register as buffer (non-trainable) or Parameter (trainable)
        self.register_buffer("embedding", embedding_tensor.detach())

    def forward(self, atomic_numbers):
        """
        atomic_numbers: LongTensor of shape (N,)
        return: Tensor of shape (N+1, embedding_dim), index 0 for mask_atom
        """
        return self.embedding[atomic_numbers.squeeze()]


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
    all_nbrs = structure.get_all_neighbors(radius, include_index=True)  # Find periodic neighbors, including atom indices and cell images.
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
    """Build pre-training graphs from a merged crystal-structure CSV.

    root contains OQMD_MP_tot.csv by default; csv_file can select another file.
    Required columns are id, cell, numbers, and pos_cart.
    cell, numbers, and pos_cart contain Python list literals. Coordinates are
    Cartesian in angstroms. The optional dataset column records provenance.
    Graphs include x, edge_index, pos, displacement, distance, nbr_image,
    lattice, lattice_lengths, lattice_angles, and the original numeric id;
    cif_id preserves the full identifier and dataset preserves its source.
    By default no property target is loaded or stored. For supervised use,
    load_target=True requires energy_per_atom and stores it in y.

    Graphs are cached in root/processed/data.pt. Remove this cache or use a
    new root when changing the input, graph parameters or processing logic.
    """

    def __init__(self, root, max_num_nbr=12, radius=8, neighbor_strategy='k-nearest',
                 random_seed=123,transform=None, pre_transform=None, pre_filter=None,
                 csv_file='OQMD_MP_tot.csv', load_target=False):
        assert os.path.exists(root), 'root_dir does not exist!'
        self.load_target = load_target

        id_prop_file = os.path.join(root, csv_file)
        if not os.path.isfile(id_prop_file):
            raise FileNotFoundError(id_prop_file)
        with open(id_prop_file, encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            required = {'id', 'cell', 'numbers', 'pos_cart'}
            if load_target:
                required.add('energy_per_atom')
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f'Missing CSV columns: {sorted(missing)}')
            self.id_prop_data = [row for row in reader]

        random.Random(random_seed).shuffle(self.id_prop_data)
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
            cif_id = row['id']

            crystal = Structure(lattice=ast.literal_eval(row['cell']),
                                species=ast.literal_eval(row['numbers']),
                                coords=ast.literal_eval(row['pos_cart']),
                                coords_are_cartesian=True)
            id = int(re.findall(r'\d+', cif_id)[0])
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
            id = torch.tensor(id, dtype=torch.long)

            src_pos = pos[edge_index[0, :], :] + torch.matmul(src_image, lattice)
            dst_pos = pos[edge_index[1, :], :]
            displacement = src_pos - dst_pos
            distance = torch.norm(displacement, p=2, dim=-1)  # One distance per edge.

            data=Data(x=atom_idx,
                      edge_index=edge_index,
                      pos=pos,
                      displacement=displacement,
                      distance=distance,
                      nbr_image=src_image,
                      lattice=lattice,
                      lattice_lengths=lattice_lengths,
                      lattice_angles=lattice_angles,
                      id=id,
                      cif_id=cif_id,
                      dataset=row.get('dataset', '')
                      )
            if self.load_target:
                data.y = torch.tensor([float(row['energy_per_atom'])], dtype=torch.float)
            # Model-specific edge embeddings are computed inside the model.

            data_list.append(data)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        self.save(data_list,self.processed_paths[0])


class MatBench_Data(InMemoryDataset):
    """Build and cache graph data from a MatBench task.

    MatbenchBenchmark supplies the selected task's fold-0 train/validation
    and test data, including labels. This class assembles both pools; callers
    apply the official outer-fold membership separately.

    root is the local cache directory and subset_data is the MatBench task
    name. task_type selects regression or classification target encoding.
    max_num_nbr, radius, and neighbor_strategy control periodic neighbors;
    transform, pre_transform, and pre_filter are standard PyG hooks.

    Graphs contain x, edge_index, y, pos, displacement, distance, and id.
    Regression labels use float64; classification labels use int64.
    Processed graphs are cached in root/processed/data.pt.
    """
    def __init__(self, root, subset_data, task_type, max_num_nbr=12, radius=8, neighbor_strategy='k-nearest',
                 transform=None, pre_transform=None, pre_filter=None):
        from matbench import MatbenchBenchmark

        assert os.path.exists(root), 'root_dir does not exist!'

        self.subset_data = subset_data
        self.task_type = task_type
        self.mb_test = MatbenchBenchmark(autoload=False, subset=[subset_data])
        for task in self.mb_test.tasks:
            task.load()
            self.train_df = task.get_train_and_val_data(0, as_type="df")
            self.test_df = task.get_test_data(0, include_target=True, as_type="df")

        self.max_num_nbr, self.radius = max_num_nbr, radius
        self.neighbor_strategy = neighbor_strategy

        super(MatBench_Data,self).__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    def download(self):
        pass

    @property
    def raw_file_names(self) :
        return [f'{self.subset_data}.csv']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        data_list = []
        for df in [self.train_df, self.test_df]:
            for mbid, row in df.iterrows():
                crystal = row.structure
                id = int(re.findall(r'-(\d+)', mbid)[0])
                target = row.iloc[-1]
                lattice = crystal.lattice.matrix
                atom_idx = np.vstack([crystal[i].specie.number for i in range(len(crystal))])

                src_idx, src_image, dst_idx = get_idx(crystal, self.neighbor_strategy, self.radius, id, self.max_num_nbr)
                edge_index = torch.tensor(np.array([
                    np.array(src_idx),
                    np.array(dst_idx), ]), dtype=torch.long)

                pos = torch.tensor(crystal.cart_coords, dtype=torch.float)
                atom_idx = torch.tensor(atom_idx, dtype=torch.long)      # Atomic numbers for graph nodes.
                src_image = torch.tensor(src_image, dtype=torch.float)   # Periodic cell images of source atoms.
                id = torch.tensor(id, dtype=torch.long)
                if self.task_type == 'regression':
                    target = torch.tensor([float(target)], dtype=torch.float64)
                else:
                    if target == 'False' or target == 0:
                        label = 0
                    elif target == 'True' or target == 1:
                        label = 1
                    target = torch.LongTensor([(label)])

                lattice = torch.tensor(lattice, dtype=torch.float)
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
                          id=id
                          )
                # Model-specific edge embeddings are computed inside the model.
                print(mbid," loaded successfully")
                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        self.save(data_list,self.processed_paths[0])


class MaskTransform(object):
    # Mask atom types and record reconstruction labels and indices.
    def __init__(self, num_atom_types, mask_rate, mask_subgraph = None, delete_rate=1.0, replace = False, k=1):
        self.num_atom_types = num_atom_types
        self.mask_rate = mask_rate
        self.mask_subgraph = mask_subgraph
        self.delete_rate = delete_rate
        self.k =k
        self.replace = replace

    def __call__(self, data: Data) -> Data:
        data = data.clone()
        if self.mask_subgraph == 'k_hop':
            # mask subgraph by k_hop
            node_num = data.x.size(0)
            edge_num = data.edge_index.size(1)
            sub_num = math.ceil(node_num * self.mask_rate)
            edge_index = data.edge_index.numpy()

            all_nodes = list(range(node_num))
            seed = random.choice(all_nodes)
            idx_sub = torch.tensor([seed], dtype=torch.long)
            k = 1
            while len(idx_sub) < sub_num:
                try:
                    subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(seed, k, data.edge_index, relabel_nodes=False)
                except Exception as e:
                    print("Error occurred!")
                    print("seed:", seed)
                    print("k:", k)
                    print("data.edge_index:", data.edge_index)
                    print("id:", data.id)
                if len(idx_sub) == len(subset):  # Stop when expanding the neighborhood adds no nodes.
                    break
                else:
                    idx_sub = subset
                    k += 1

            if len(idx_sub) < sub_num:   # Adjust the requested mask size for a smaller connected component.
                sub_num = int(len(idx_sub) * self.mask_rate + 1)
                idx_mask = torch.tensor(sorted(idx_sub.tolist()), dtype=torch.long)[:sub_num]    # Select nodes up to the requested mask size.
                edge_mask = torch.tensor(
                    [n for n in range(edge_num) if (edge_index[0, n] in idx_mask and edge_index[1, n] in idx_mask)], dtype=torch.long)
            else:
                idx_mask = torch.tensor(sorted(idx_sub.tolist()), dtype=torch.long)[:sub_num]
                edge_mask = torch.tensor(
                    [n for n in range(edge_num) if (edge_index[0, n] in idx_mask and edge_index[1, n] in idx_mask)], dtype=torch.long)

            # Record reconstruction labels.
            node_labels = data.x[idx_mask.view(-1)].view(-1)
            data.masked_atom_labels = F.one_hot(
                node_labels - 1,
                num_classes=self.num_atom_types
            ).float()
            data.masked_atom_indices = idx_mask
            data.masked_edge_labels = data.distance[edge_mask]    # Store distance labels for masked edges.
            data.masked_edge_indices = edge_mask

            # Apply atom masking.
            data.x[idx_mask,:] = torch.zeros(data.x.shape[1], dtype=torch.long)
        elif self.mask_subgraph == 'random_walk':
            #mask_subgraph by random walk
            node_num = data.x.size(0)
            edge_num = data.edge_index.size(1)
            sub_num = math.ceil(node_num * self.mask_rate)

            edge_index = data.edge_index.numpy()
            idx_sub = [np.random.randint(node_num, size=1)[0]]
            idx_neigh = set([n for n in edge_index[1][edge_index[0] == idx_sub[0]]])  # Collect distinct destinations of outgoing edges.

            count = 0
            while len(idx_sub) <= sub_num:
                count = count + 1
                if count > node_num:  # Bound the number of random-walk attempts.
                    break
                if len(idx_neigh) == 0:
                    break
                sample_node = np.random.choice(list(idx_neigh))
                if sample_node in idx_sub:
                    continue
                idx_sub.append(sample_node)     # Record sampled subgraph nodes.
                idx_neigh.union(set([n for n in edge_index[1][edge_index[0] == idx_sub[-1]]]))  # Consider destinations from the newly sampled node.

            idx_mask = torch.tensor(sorted(idx_sub), dtype=torch.long)
            src_in_sub = torch.isin(data.edge_index[0], idx_mask)
            dst_in_sub = torch.isin(data.edge_index[1], idx_mask)
            edge_mask = src_in_sub & dst_in_sub

            rand_tensor = torch.rand(data.edge_index.size(1))
            random_delete_mask = (rand_tensor < self.delete_rate)
            edge_mask = edge_mask & random_delete_mask
            edge_retrained = ~edge_mask
            if node_num == len(idx_sub):
                edge_mask = torch.zeros_like(edge_mask, dtype=torch.bool)
                edge_retrained = ~edge_mask

            # Record reconstruction labels.
            node_labels = data.x[idx_mask.view(-1)].view(-1)
            data.masked_atom_labels = F.one_hot(
                node_labels - 1,
                num_classes=self.num_atom_types
            ).float()
            data.masked_atom_indices = idx_mask

            # Apply atom masking.
            if self.replace:
                #BERT style
                mask_prob = 0.8  # 80% MASK ID (0)
                random_prob = 0.1  # 10% Random ID (1-94)
                rand_tensor = torch.rand(idx_mask.size(), device=data.x.device)
                idx_mask_mask = idx_mask[rand_tensor < mask_prob]
                idx_mask_random = idx_mask[(rand_tensor >= mask_prob) & (rand_tensor < mask_prob + random_prob)]
                if idx_mask_mask.numel() > 0:
                    data.x[idx_mask_mask] = 0
                if idx_mask_random.numel() > 0:
                    random_atom_ids = torch.randint(1, self.num_atom_types + 1, size=(idx_mask_random.numel(),))
                    data.x[idx_mask_random] = random_atom_ids
            else:
                data.x[idx_mask] = 0
            # Remove selected edges and retain their reconstruction labels.
            data.deleted_edge_index = data.edge_index[:, edge_mask]
            data.deleted_distance = torch.norm(data.displacement, p=2, dim=-1)[edge_mask]
            data.deleted_displacement = data.displacement[edge_mask, :]
            data.deleted_nbr_image = data.nbr_image[edge_mask, :]

            data.edge_index = data.edge_index[:, edge_retrained]

            data.displacement = data.displacement[edge_retrained, :]
            data.nbr_image = data.nbr_image[edge_retrained, :]

        else:
            # Randomly select atom indices to mask.
            num_atoms = data.num_nodes
            sample_size = math.ceil(num_atoms * self.mask_rate)
            masked_atom_indices = torch.tensor(sorted(random.sample(range(num_atoms), sample_size)),dtype=torch.long)

            # Store one-hot atom reconstruction labels.
            node_labels = data.x[masked_atom_indices].view(-1)
            data.masked_atom_labels = F.one_hot(
                node_labels - 1,
                num_classes=self.num_atom_types
            ).float()
            data.masked_atom_indices = masked_atom_indices

            # Apply atom masking.
            data.x[masked_atom_indices] = 0

        if hasattr(data, 'lattice'):
            del data.lattice

        if hasattr(data, 'pos'):
            del data.pos

        if hasattr(data, 'lattice_lengths'):
            del data.lattice_lengths

        if hasattr(data, 'lattice_angles'):
            del data.lattice_angles

        return data

    def __repr__(self):
        return '{}(num_atom_type={},mask_rate={},mask_subgraph={})'.format(
            self.__class__.__name__, self.num_atom_types, self.mask_rate, self.mask_subgraph)


class BatchMasking(Data):
    r"""A plain old python object modeling a batch of graphs as one big
    (dicconnected) graph. With :class:`torch_geometric.data.Data` being the
    base class, all its methods can also be used here.
    In addition, single graphs can be reconstructed via the assignment vector
    :obj:`batch`, which maps each node to its respective graph identifier.
    """

    def __init__(self, batch=None, **kwargs):
        super(BatchMasking, self).__init__(**kwargs)
        self.batch = batch

    @staticmethod
    def from_data_list(data_list):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        keys = [set(data.keys()) for data in data_list]
        keys = list(set.union(*keys))
        assert 'batch' not in keys

        batch = BatchMasking()

        for key in keys:
            batch[key] = []
        batch.batch = []

        cumsum_node = 0
        cumsum_edge = 0
        for i, data in enumerate(data_list):
            num_nodes = data.x.shape[0]
            num_edges = data.edge_index.shape[1]
            batch.batch.append(torch.full((num_nodes, ), i, dtype=torch.long))
            for key in data.keys():
                item = data[key]
                if key in ['edge_index', 'deleted_edge_index', 'masked_atom_indices']:
                    item = item + cumsum_node
                if key in ['masked_edge_indices']:
                    item = item + cumsum_edge
                batch[key].append(item)

            cumsum_node += num_nodes
            cumsum_edge += num_edges

        for key in keys:
            if isinstance(batch[key][0], torch.Tensor):
                batch[key] = torch.cat(
                    batch[key], dim=data_list[0].__cat_dim__(key, batch[key][0]))
        batch.batch = torch.cat(batch.batch, dim=-1)
        return batch.contiguous()

    def cumsum(self, key, item):
        r"""If :obj:`True`, the attribute :obj:`key` with content :obj:`item`
        should be added up cumulatively before concatenated together.
        .. note::
            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        return key in ['edge_index', 'masked_atom_indices', 'masked_edge_indices']

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1


class PerturbTransform(object):
    # Perturb positions and update edge geometry while retaining graph topology.
    def __init__(self, mu=0.0, sigma=0.4, Pertlatt = False, mu2 = 0.0, sigma2 = 0.2):
        self.mu = mu
        self.sigma = sigma
        self.Pertlatt = Pertlatt
        self.mu2 = mu2
        self.sigma2 = sigma2

    def __call__(self, data: Data) -> Data:
        data = data.clone()
        device = data.x.device
        if self.Pertlatt:
            data.lattice_lengths_labels = data.lattice_lengths.reshape(1,-1)
            data.lattice_angles_labels = torch.cat([torch.cos(data.lattice_angles), torch.sin(data.lattice_angles)], dim=-1)
            data.lattice += torch.normal(self.mu2, self.sigma2, size=data.lattice.shape).to(device)   # Add Gaussian lattice noise.


        # Recompute edge geometry from perturbed positions.
        pos_noise = torch.normal(self.mu, self.sigma, size=data.pos.shape).to(device)

        pert_pos = data.pos + pos_noise
        data.pos = pert_pos
        src_pos = data.pos[data.edge_index[0,:],:] + torch.matmul(data.nbr_image, data.lattice)
        dst_pos = data.pos[data.edge_index[1,:],:]
        data.distance = torch.norm(src_pos - dst_pos, p=2, dim=-1)   # One distance per edge.
        data.displacement = (src_pos - dst_pos)

        data.lattice_displacement = torch.matmul(data.nbr_image, data.lattice)
        data.noise_labels = pos_noise

        if hasattr(data, 'lattice'):
            del data.lattice

        if hasattr(data, 'nbr_image'):
            del data.nbr_image

        if hasattr(data, 'lattice_lengths'):
            del data.lattice_lengths

        if hasattr(data, 'lattice_angles'):
            del data.lattice_angles

        return data

    def __repr__(self):
        return '{}(mu={},sigma={},Pertlatt={},mu2={},sigma2={})'.format(
            self.__class__.__name__, self.mu, self.sigma, self.Pertlatt, self.mu2, self.sigma2)


class Subgraph(object):
    def __init__(self, ratio, method):
        self.ratio = ratio
        self.method = method

    def __call__(self, data):
        data = data.clone()
        if self.method == 'k_hop':
            # get subgraph by k_hop
            node_num = data.x.size(0)
            edge_num = data.edge_index.size(1)
            sub_num = math.ceil(node_num * self.ratio)
            edge_index = data.edge_index.numpy()

            idx_sub = torch.tensor([])
            all_nodes = list(range(node_num))
            seed = random.choice(all_nodes)
            k = 1
            while len(idx_sub) < sub_num:
                try:
                    subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(seed, k, data.edge_index, relabel_nodes=False)
                except Exception as e:
                    print("Error occurred!")
                    print("seed:", seed)
                    print("k:", k)
                    print("data.edge_index:", data.edge_index)
                    print("id:", data.id)
                if len(idx_sub) == len(subset):    # Stop when neighborhood expansion adds no nodes.
                    break
                else:
                    idx_sub = subset
                    k += 1

            idx_sub = idx_sub[:sub_num]
            idx_dict = {int(idx_sub[n]): n for n in list(range(len(idx_sub)))}
            edge_mask = np.array(
                [n for n in range(edge_num) if (edge_index[0, n] in idx_sub and edge_index[1, n] in idx_sub)])

            edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in edge_mask]    # Remap retained edges to the subgraph node indices.

            try:
                data.edge_index = torch.tensor(edge_index).transpose_(0, 1)
                data.x = data.x[idx_sub]

                data.displacement = data.displacement[edge_mask]
            except:
                data = data
        elif self.method == 'random_walk':
            node_num = data.x.size(0)
            sub_num = math.ceil(node_num * self.ratio)
            edge_index = data.edge_index.numpy()

            idx_sub = [np.random.randint(node_num, size=1)[0]]
            idx_neigh = set([n for n in edge_index[1][edge_index[0] == idx_sub[0]]])  # Collect distinct destinations of outgoing edges.

            count = 0
            while len(idx_sub) <= sub_num:
                count = count + 1
                if count > node_num:     # Bound the number of random-walk attempts.
                    break
                if len(idx_neigh) == 0:
                    break
                sample_node = np.random.choice(list(idx_neigh))
                if sample_node in idx_sub:
                    continue
                idx_sub.append(sample_node)
                idx_neigh.union(set([n for n in edge_index[1][edge_index[0] == idx_sub[-1]]]))  # Consider destinations from the newly sampled node.

            idx_dict = {idx_sub[n]: n for n in list(range(len(idx_sub)))}                 # Map retained nodes to consecutive subgraph indices.
            idx_nondrop = torch.tensor(sorted(idx_sub), dtype=torch.long)
            src_in_sub = torch.isin(data.edge_index[0], idx_nondrop)
            dst_in_sub = torch.isin(data.edge_index[1], idx_nondrop)
            edge_retained = torch.nonzero(src_in_sub & dst_in_sub, as_tuple=True)[0]
            edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in edge_retained]    # Remap retained edges to the subgraph node indices.

            try:
                data.edge_index = torch.tensor(edge_index).transpose_(0, 1)
                data.x = data.x[idx_nondrop]

                data.displacement = data.displacement[edge_retained]
            except:
                data = data
        else:
            print('not correct method! only accept random walk or k_hop method!')
            raise NotImplementedError

        if hasattr(data, 'lattice'):
            del data.lattice

        if hasattr(data, 'pos'):
            del data.pos

        if hasattr(data, 'nbr_image'):
            del data.nbr_image

        if hasattr(data, 'lattice_lengths'):
            del data.lattice_lengths

        if hasattr(data, 'lattice_angles'):
            del data.lattice_angles

        return data

    def __repr__(self):
        return '{}(ratio={})'.format(
            self.__class__.__name__, self.ratio)


class Dropedge(object):
    def __init__(self, ratio, random=False):
        self.ratio = ratio
        self.random = random

    def __call__(self, data):
        data = data.clone()
        if self.random:
            # Randomly remove edges.
            edge_num = data.edge_index.size(-1)
            drop_num = math.ceil(edge_num * self.ratio)
            save_num = edge_num - drop_num
            edge_mask = torch.tensor(sorted(random.sample(range(edge_num), save_num)))
        else:
            # Remove the farthest incoming edges of each node.
            _, counts = torch.unique(data.edge_index[1,:].view(-1), return_counts=True)
            drop_num = torch.ceil(counts * self.ratio).int()
            cumsum = torch.cumsum(counts, dim=0)
            edge_mask = []
            for idx in range(counts.size(0)):
                if counts[idx] == 1:           # Retain an incoming edge when it is the only one.
                    continue
                edge_mask.append(torch.arange(cumsum[idx] - drop_num[idx], cumsum[idx]))
            if len(edge_mask) == 0:
                edge_mask = torch.empty(0, dtype=torch.long)
            else:
                edge_mask = torch.hstack(edge_mask)
            edge_mask = ~torch.isin(torch.arange(data.edge_index.size(1)), edge_mask)

        data.edge_index = data.edge_index[:,edge_mask]

        data.displacement = data.displacement[edge_mask,:]

        if hasattr(data, 'lattice'):
            del data.lattice

        if hasattr(data, 'pos'):
            del data.pos

        if hasattr(data, 'nbr_image'):
            del data.nbr_image

        if hasattr(data, 'lattice_lengths'):
            del data.lattice_lengths

        if hasattr(data, 'lattice_angles'):
            del data.lattice_angles

        return data

    def __repr__(self):
        return '{}(ratio={}, max_nbr_num={}, random]{})'.format(
            self.__class__.__name__, self.ratio, self.max_nbr_num, self.random)


class Replace(object):
    def __init__(self, ratio):
        self.ratio = ratio
        self.group2element = {
            1: [1, 3, 11, 19, 37, 55, 87],                  # Group 1, including hydrogen.
            2: [4, 12, 20, 38, 56, 88],                      # Group 2.
            3: [21, 39, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103], # Group 3 plus lanthanides and actinides.
            4: [22, 40, 72, 104],
            5: [23, 41, 73, 105],
            6: [24, 42, 74, 106],
            7: [25, 43, 75, 107],
            8: [26, 44, 76, 108],
            9: [27, 45, 77, 109],
            10: [28, 46, 78, 110],
            11: [29, 47, 79, 111],
            12: [30, 48, 80, 112],
            13: [5, 13, 31, 49, 81, 113],                    # Group 13.
            14: [6, 14, 32, 50, 82, 114],                    # Group 14.
            15: [7, 15, 33, 51, 83, 115],                    # Group 15.
            16: [8, 16, 34, 52, 84, 116],                    # Group 16.
            17: [9, 17, 35, 53, 85, 117],                    # Halogens.
            18: [2, 10, 18, 36, 54, 86, 118]                 # Noble gases.
            }
        self.element2group = {1: 1, 3: 1, 11: 1, 19: 1, 37: 1, 55: 1, 87: 1,
                              4: 2, 12: 2, 20: 2, 38: 2, 56: 2, 88: 2,
                              21: 3, 39: 3, 57: 3, 58: 3, 59: 3, 60: 3, 61: 3, 62: 3, 63: 3, 64: 3, 65: 3, 66: 3, 67: 3, 68: 3, 69: 3, 70: 3, 71: 3, 89: 3, 90: 3, 91: 3, 92: 3, 93: 3, 94: 3, 95: 3, 96: 3, 97: 3, 98: 3, 99: 3, 100: 3, 101: 3, 102: 3, 103: 3,
                              22: 4, 40: 4, 72: 4, 104: 4,
                              23: 5, 41: 5, 73: 5, 105: 5,
                              24: 6, 42: 6, 74: 6, 106: 6,
                              25: 7, 43: 7, 75: 7, 107: 7,
                              26: 8, 44: 8, 76: 8, 108: 8,
                              27: 9, 45: 9, 77: 9, 109: 9,
                              28: 10, 46: 10, 78: 10, 110: 10,
                              29: 11, 47: 11, 79: 11, 111: 11,
                              30: 12, 48: 12, 80: 12, 112: 12,
                              5: 13, 13: 13, 31: 13, 49: 13, 81: 13, 113: 13,
                              6: 14, 14: 14, 32: 14, 50: 14, 82: 14, 114: 14,
                              7: 15, 15: 15, 33: 15, 51: 15, 83: 15, 115: 15,
                              8: 16, 16: 16, 34: 16, 52: 16, 84: 16, 116: 16,
                              9: 17, 17: 17, 35: 17, 53: 17, 85: 17, 117: 17,
                              2: 18, 10: 18, 18: 18, 36: 18, 54: 18, 86: 18, 118: 18
                              }
    def __call__(self, data):
        data = data.clone()
        num_nodes = data.x.shape[0]
        replace_num = math.ceil(num_nodes * self.ratio)
        replace_node_index = torch.tensor(sorted(random.sample(range(num_nodes), replace_num)))
        replace_x = data.x[replace_node_index]
        for i in range(len(replace_node_index)):
            new_idx= random.choice([idx for idx in self.group2element[self.element2group[int(replace_x[i])]] if idx <= 100])  # Restrict replacements to atomic numbers supported by the embedding.
            data.x[replace_node_index[i]] = new_idx

        if hasattr(data, 'lattice'):
            del data.lattice

        if hasattr(data, 'pos'):
            del data.pos

        if hasattr(data, 'nbr_image'):
            del data.nbr_image

        if hasattr(data, 'lattice_lengths'):
            del data.lattice_lengths

        if hasattr(data, 'lattice_angles'):
            del data.lattice_angles

        return data

    def __repr__(self):
        return '{}(ratio={})'.format(
            self.__class__.__name__, self.ratio)


class MultiViewWrapper(torch.utils.data.Dataset):
    def __init__(self, base_dataset, mask=None, perturb=None, transform1=None, transform2=None):
        self.base_dataset = base_dataset
        self.mask = mask
        self.perturb = perturb
        self.transform1 = transform1
        self.transform2 = transform2

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        data = self.base_dataset[idx]
        mask_data = self.mask(data) if self.mask is not None else data.clone()
        pert_data = self.perturb(data) if self.perturb is not None else data.clone()
        view1 = self.transform1(data) if self.transform1 is not None else data.clone()
        view2 = self.transform2(data) if self.transform2 is not None else data.clone()
        return mask_data, pert_data, view1, view2


def contrastive_collate_fn(batch):
    mask_data_list, pert_data_list, view1_list, view2_list = zip(*batch)
    batch1 = BatchMasking.from_data_list(mask_data_list)
    batch2 = Batch.from_data_list(pert_data_list)
    batch3 = Batch.from_data_list(view1_list)
    batch4 = Batch.from_data_list(view2_list)

    return batch1, batch2, batch3, batch4


class DataLoaderMultiView(torch.utils.data.DataLoader):
    # Collate masked, perturbed, and contrastive views.
    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderMultiView, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=contrastive_collate_fn,
            **kwargs)
