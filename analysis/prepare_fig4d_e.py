"""Extract pretrained PaiNN features and prepare t-SNE Source Data for Fig. 4d-e."""

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from data.dataloader import MatBench_Data
from models.PaiNN import PainnEncoder


TASKS = {
    'mp_e_form': {
        'matbench_name': 'matbench_mp_e_form',
        'task_type': 'regression',
        'panel': 'd',
    },
    'mp_is_metal': {
        'matbench_name': 'matbench_mp_is_metal',
        'task_type': 'classification',
        'panel': 'e',
    },
}


def numeric_mbid(mbid):
    match = re.search(r'-(\d+)$', str(mbid))
    if match is None:
        raise ValueError(f'Cannot parse MatBench ID: {mbid}')
    return int(match.group(1))


def map_test_indices(dataset, test_ids):
    stored_ids = dataset._data.id
    if not torch.is_tensor(stored_ids) or stored_ids.numel() != len(dataset):
        raise ValueError('Processed dataset does not expose one tensor ID per graph')
    id_to_index = {
        int(material_id): index
        for index, material_id in enumerate(stored_ids.view(-1).tolist())
    }
    if len(id_to_index) != len(dataset):
        raise ValueError('Processed dataset contains duplicate material IDs')
    missing = [mbid for mbid in test_ids if numeric_mbid(mbid) not in id_to_index]
    if missing:
        raise ValueError(f'Test IDs missing from processed dataset: {missing[:5]}')
    return [id_to_index[numeric_mbid(mbid)] for mbid in test_ids]


@torch.no_grad()
def extract_features(model, loader, device):
    model.eval()
    features, labels = [], []
    for batch in tqdm(loader, desc='Extracting graph features'):
        batch = batch.to(device)
        node_scalar, _ = model(batch)
        graph_features = global_mean_pool(node_scalar, batch.batch)
        features.append(graph_features.cpu().numpy())
        labels.append(batch.y.view(-1).cpu().numpy())
    return np.concatenate(features), np.concatenate(labels)


def write_source_data(path, material_ids, coordinates, labels):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['id', 'tsne_1', 'tsne_2', 'label'])
        writer.writerows(
            (material_id, float(point[0]), float(point[1]), float(label))
            for material_id, point, label in zip(material_ids, coordinates, labels)
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True, choices=tuple(TASKS))
    parser.add_argument('--data-root', required=True, type=Path,
                        help='MatBench graph-cache directory containing processed/data.pt')
    parser.add_argument('--checkpoint', type=Path,
                        default=REPO / 'checkpoint/geomat_painn_pretrained.pth.tar',
                        help='pretrained PaiNN checkpoint')
    parser.add_argument('--fold', type=int, default=4, choices=range(5),
                        help='MatBench outer fold used for visualization (default: 4)')
    parser.add_argument('--output', type=Path,
                        help='Source Data CSV; defaults to the panel-specific path')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--perplexity', type=float, default=30.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--disable-cuda', action='store_true')
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error('--batch-size must be positive and --workers cannot be negative')
    if args.perplexity <= 0:
        parser.error('--perplexity must be positive')
    if not args.data_root.is_dir():
        parser.error(f'--data-root does not exist: {args.data_root}')
    if not args.checkpoint.is_file():
        parser.error(f'--checkpoint does not exist: {args.checkpoint}')

    config = TASKS[args.task]
    dataset = MatBench_Data(
        str(args.data_root), subset_data=config['matbench_name'],
        task_type=config['task_type'], radius=5.0,
        neighbor_strategy='radius_graph')
    task = next(iter(dataset.mb_test.tasks))
    test_frame = task.get_test_data(
        args.fold, include_target=True, as_type='df')
    material_ids = [str(mbid) for mbid in test_frame.index]
    test_indices = map_test_indices(dataset, material_ids)
    loader = DataLoader(
        Subset(dataset, test_indices), batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers,
        pin_memory=not args.disable_cuda and torch.cuda.is_available())

    device = torch.device(
        'cuda' if not args.disable_cuda and torch.cuda.is_available() else 'cpu')
    model = PainnEncoder(num_interactions=3, hidden_state_size=64).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    features, labels = extract_features(model, loader, device)
    if len(features) != len(material_ids):
        raise RuntimeError('Feature count does not match the MatBench test split')

    print(f'Running t-SNE on {features.shape[0]} x {features.shape[1]} features...',
          flush=True)
    coordinates = TSNE(
        n_components=2, perplexity=args.perplexity,
        random_state=args.seed, init='pca', learning_rate='auto',
    ).fit_transform(features)
    output = args.output or (
        REPO / 'source_data' / f"Source_Data_Fig4{config['panel']}.csv")
    write_source_data(output, material_ids, coordinates, labels)
    print(f'Saved {len(material_ids)} rows to {output}')


if __name__ == '__main__':
    main()
