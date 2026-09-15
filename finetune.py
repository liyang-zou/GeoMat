"""Fine-tune GeoMat on released non-MatBench regression datasets."""

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader

from models.PaiNN import PainnEncoder
from models.model import encoder, readout
from utils.metrics import AverageMeter, mae
from utils.normalizer import Normalizer
from utils.save_checkpoint import save_checkpoint
from utils.set_seeds import set_all_seeds


TASKS = ('afm', 'exciton', 'tc8055', 'max', 'jarvis')
PRETRAINED_CHECKPOINTS = {
    'CGConv': 'geomat_cgcnn_pretrained.pth.tar',
    'MEGConv': 'geomat_megnet_pretrained.pth.tar',
    'Painn': 'geomat_painn_pretrained.pth.tar',
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data_root', type=Path)
    parser.add_argument('--dataset', required=True, choices=TASKS)
    parser.add_argument('--jarvis-target',
                        help='JARVIS property key; required for dataset=jarvis')
    parser.add_argument('--jarvis-json', default='jdft_3d-8-18-2021.json')
    parser.add_argument('--pretrain-model', type=Path,
                        help='pretrained checkpoint; defaults to the selected backbone')
    parser.add_argument('--finetune-config', default='./finetune_config.yaml')
    parser.add_argument('--save-dir', type=Path, default=Path('./runs/finetune'))
    parser.add_argument('--resume', choices=('pretrain', 'scratch'), default='pretrain')
    parser.add_argument('--gnn', choices=('CGConv', 'MEGConv', 'Painn'),
                        default='MEGConv')
    parser.add_argument('--n-conv', default=3, type=int)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--workers', default=0, type=int)
    parser.add_argument('--disable-cuda', action='store_true')
    parser.add_argument('--runs', default=3, type=int)
    parser.add_argument('--seed', default=42, type=int,
                        help='base training seed; run r uses seed + r')
    parser.add_argument('--split-seed', default=42, type=int,
                        help='seed used to divide data (default: 42)')
    parser.add_argument(
        '--split-mode', choices=('auto', 'fixed-files', 'fixed-random', 'resampled'),
        default='auto', help='auto uses released IDs, or fixed-random for JARVIS')
    parser.add_argument('--train-fraction', type=float,
                        help='fraction of all samples; protocol default depends on task/model')
    parser.add_argument('--val-fraction', type=float,
                        help='fraction of all samples; protocol default depends on task/model')
    parser.add_argument('--loss', choices=('auto', 'mse', 'mae'), default='auto')
    parser.add_argument('--atom-init', type=Path,
                        default=Path(__file__).resolve().parent / 'data' / 'atom_init.json',
                        help='CGCNN elemental feature file')
    args = parser.parse_args()
    if args.pretrain_model is None:
        args.pretrain_model = (Path(__file__).resolve().parent / 'checkpoint' /
                               PRETRAINED_CHECKPOINTS[args.gnn])
    return args


def load_hparams(path, gnn):
    defaults = {
        'CGConv': dict(batch_size_finetune=128, p_finetune=0,
                       lr_finetune=0.01, weight_decay_finetune=1e-6),
        'MEGConv': dict(batch_size_finetune=32, p_finetune=0,
                        lr_finetune=5e-4, weight_decay_finetune=0.0),
        'Painn': dict(batch_size_finetune=128, p_finetune=0,
                      lr_finetune=1e-4, weight_decay_finetune=0.01),
    }[gnn]
    with open(path, encoding='utf-8') as stream:
        configured = yaml.safe_load(stream) or {}
    result = {**defaults, **configured}
    result.setdefault('ph_lr', result['lr_finetune'])
    result.setdefault('ph_wd', result['weight_decay_finetune'])
    return result


def load_dataset(args):
    kwargs = dict(radius=5.0, neighbor_strategy='radius_graph')
    if args.dataset == 'afm':
        from data.Tc_AFM.dataloader import CIFData
        dataset = CIFData(str(args.data_root), **kwargs)
    elif args.dataset == 'exciton':
        from data.exciton_c2db.dataloader import CIFData
        dataset = CIFData(str(args.data_root), **kwargs)
    elif args.dataset == 'tc8055':
        from data.Tc_8055.dataloader import CIFData
        dataset = CIFData(str(args.data_root), **kwargs)
    elif args.dataset == 'max':
        from data.MAX.dataloader import CIFData
        dataset = CIFData(str(args.data_root), **kwargs)
    else:
        if not args.jarvis_target:
            raise ValueError('--jarvis-target is required for dataset=jarvis')
        from data.jarvis import JARVISData
        dataset = JARVISData(str(args.data_root), target=args.jarvis_target,
                             json_file=args.jarvis_json, **kwargs)
    ids = [str(dataset[index].id) for index in range(len(dataset))]
    if len(ids) != len(dataset) or len(ids) != len(set(ids)):
        raise ValueError('Material IDs are missing, duplicated, or misaligned')
    return dataset, ids


def read_id_table(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['id']:
            raise ValueError(f'{path} must contain only an id column')
        return [row['id'] for row in reader]


def map_ids(all_ids, selected, source):
    lookup = {material_id: index for index, material_id in enumerate(all_ids)}
    missing = [material_id for material_id in selected if material_id not in lookup]
    if missing:
        raise ValueError(f'{source} contains absent IDs: {missing[:3]}')
    return [lookup[material_id] for material_id in selected]


def protocol_fractions(args):
    if args.dataset == 'jarvis':
        default_train, default_val = 0.8, 0.1
    elif args.gnn == 'MEGConv':
        default_train, default_val = 0.8, 0.0
    else:
        default_train, default_val = 0.6, 0.2
    return (default_train if args.train_fraction is None else args.train_fraction,
            default_val if args.val_fraction is None else args.val_fraction)


def split_indices(args, ids, run_index):
    mode = args.split_mode
    if mode == 'auto':
        if args.dataset in ('afm', 'exciton', 'tc8055', 'max'):
            mode = 'fixed-files'
        else:
            mode = 'fixed-random'
    split_seed_base = args.split_seed
    train_fraction, val_fraction = protocol_fractions(args)
    if train_fraction <= 0 or val_fraction < 0:
        raise ValueError('Split fractions must be non-negative and train must be positive')
    if mode == 'fixed-files':
        train_file = args.data_root / 'train.csv'
        test_file = args.data_root / 'test.csv'
        train = map_ids(ids, read_id_table(train_file), train_file)
        test = map_ids(ids, read_id_table(test_file), test_file)
        val = []
        if val_fraction:
            relative_val = val_fraction / (train_fraction + val_fraction)
            train, val = train_test_split(
                train, test_size=relative_val, random_state=split_seed_base,
                shuffle=True)
    else:
        split_seed = (split_seed_base + run_index
                      if mode == 'resampled' else split_seed_base)
        test_fraction = 1.0 - train_fraction - val_fraction
        if test_fraction <= 0:
            raise ValueError('train_fraction + val_fraction must be less than 1')
        train, remainder = train_test_split(
            range(len(ids)), train_size=train_fraction,
            random_state=split_seed, shuffle=True)
        if val_fraction:
            relative_test = test_fraction / (test_fraction + val_fraction)
            val, test = train_test_split(
                remainder, test_size=relative_test,
                random_state=split_seed, shuffle=True)
        else:
            val, test = [], remainder
    groups = [set(train), set(val), set(test)]
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise AssertionError('Data split overlap detected')
    return list(train), list(val), list(test), mode


def write_ids(path, ids, indices):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['id'])
        writer.writerows([ids[index]] for index in indices)


def build_model(args, hparams, device):
    node_dim, edge_dim, hidden = {
        'CGConv': (64, 32, [128]),
        'MEGConv': (32, 32, [32]),
        'Painn': (64, 32, [64]),
    }[args.gnn]
    atom_init = args.atom_init
    if args.gnn == 'Painn':
        model = PainnEncoder(num_interactions=args.n_conv,
                             hidden_state_size=node_dim).to(device)
    else:
        model = encoder(num_layer=args.n_conv, node_dim=node_dim,
                        edge_dim=edge_dim, gnn_type=args.gnn,
                        atom_init=str(atom_init), norm='layernorm').to(device)
    head = readout(hidden_dim=node_dim, out_dim=1, readout_type=args.gnn,
                   MLP_hidden_dims=hidden, p=hparams['p_finetune']).to(device)
    model.apply(init_weights)
    head.apply(init_weights)
    return model, head


def predict(model, head, batch, gnn):
    if gnn == 'MEGConv':
        x, edge_attr, _, state = model(batch)
        return head(x, batch.batch, edge_index=batch.edge_index,
                    edge_attr=edge_attr, state=state)
    if gnn == 'Painn':
        x, _ = model(batch)
        return head(x, batch.batch)
    if gnn == 'CGConv':
        x, _, _ = model(batch)
        return head(x, batch.batch)


def train_epoch(loader, model, head, criterion, model_optimizer,
                head_optimizer, normalizer, device, gnn):
    model.train()
    head.train()
    losses, errors = AverageMeter(), AverageMeter()
    for batch in loader:
        batch = batch.to(device)
        prediction = predict(model, head, batch, gnn)
        loss = criterion(prediction, normalizer.norm(batch.y.float()).view(-1, 1))
        model_optimizer.zero_grad()
        head_optimizer.zero_grad()
        loss.backward()
        model_optimizer.step()
        head_optimizer.step()
        count = batch.y.numel()
        losses.update(loss.item(), count)
        errors.update(mae(normalizer.denorm(prediction),
                          batch.y.view(-1, 1)).item(), count)
    return losses.avg, errors.avg


@torch.no_grad()
def evaluate(loader, model, head, criterion, normalizer, device, gnn,
             return_values=False):
    model.eval()
    head.eval()
    losses, errors = AverageMeter(), AverageMeter()
    true_values, predicted_values = [], []
    for batch in loader:
        batch = batch.to(device)
        prediction = predict(model, head, batch, gnn)
        loss = criterion(prediction, normalizer.norm(batch.y.float()).view(-1, 1))
        count = batch.y.numel()
        losses.update(loss.item(), count)
        errors.update(mae(normalizer.denorm(prediction),
                          batch.y.view(-1, 1)).item(), count)
        if return_values:
            true_values.extend(batch.y.view(-1).cpu().tolist())
            predicted_values.extend(
                normalizer.denorm(prediction).view(-1).cpu().tolist())
    if return_values:
        return losses.avg, errors.avg, true_values, predicted_values
    return losses.avg, errors.avg


def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=0.1)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def build_schedulers(gnn, model_optimizer, head_optimizer):
    if gnn == 'CGConv':
        scheduler_model = optim.lr_scheduler.ReduceLROnPlateau(
            model_optimizer, mode='min', factor=0.1, patience=100,
            min_lr=1e-6)
        scheduler_head = optim.lr_scheduler.ReduceLROnPlateau(
            head_optimizer, mode='min', factor=0.1, patience=100,
            min_lr=1e-6)
    elif gnn == 'Painn':
        scheduler_model = optim.lr_scheduler.ReduceLROnPlateau(
            model_optimizer, mode='min', factor=0.5, patience=50,
            min_lr=1e-6)
        scheduler_head = optim.lr_scheduler.ReduceLROnPlateau(
            head_optimizer, mode='min', factor=0.5, patience=50,
            min_lr=1e-6)
    elif gnn == 'MEGConv':
        scheduler_model = optim.lr_scheduler.LambdaLR(
            model_optimizer, lr_lambda=megnet_decay_multiplier)
        scheduler_head = optim.lr_scheduler.LambdaLR(
            head_optimizer, lr_lambda=megnet_decay_multiplier)
    return [scheduler_model, scheduler_head]


def megnet_decay_multiplier(epoch, epo_min=100, epo=1000,
                             lr_start=5e-4, lr_stop=5e-6):
    if epoch < epo_min:
        return 1.0
    if epoch < epo:
        progress = (epoch - epo_min) / (epo - epo_min)
        current_lr = lr_start - (lr_start - lr_stop) * progress
        return current_lr / lr_start
    return lr_stop / lr_start


def run_once(args, hparams, dataset, ids, run_index, device, cuda):
    run_number = run_index + 1
    run_seed = args.seed + run_index
    set_all_seeds(run_seed)
    train_idx, val_idx, test_idx, mode = split_indices(args, ids, run_index)
    run_dir = args.save_dir / f'run_{run_number}'
    run_dir.mkdir(parents=True, exist_ok=True)
    for name, indices in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        write_ids(run_dir / f'{name}.csv', ids, indices)
    print(f'Run {run_number}/{args.runs}: training_seed={run_seed}, mode={mode}, '
          f'train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}')

    generator = torch.Generator().manual_seed(run_seed)
    loader_kwargs = dict(batch_size=hparams['batch_size_finetune'],
                         num_workers=args.workers, pin_memory=cuda)
    train_loader = DataLoader(Subset(dataset, train_idx), shuffle=True,
                              generator=generator, **loader_kwargs)
    val_loader = (DataLoader(Subset(dataset, val_idx), shuffle=False, **loader_kwargs)
                  if val_idx else None)
    test_loader = DataLoader(Subset(dataset, test_idx), shuffle=False, **loader_kwargs)
    normalizer = Normalizer(dataset[train_idx].y).to(device)
    model, head = build_model(args, hparams, device)
    loss_name = args.loss
    if loss_name == 'auto':
        loss_name = ('mae' if args.dataset in ('afm', 'exciton', 'tc8055')
                     or args.gnn == 'MEGConv' else 'mse')
    criterion = (nn.L1Loss() if loss_name == 'mae' else nn.MSELoss()).to(device)
    model_optimizer = optim.AdamW(
        model.parameters(), lr=hparams['lr_finetune'],
        weight_decay=hparams['weight_decay_finetune'])
    head_optimizer = optim.AdamW(
        head.parameters(), lr=hparams['ph_lr'], weight_decay=hparams['ph_wd'])
    schedulers = build_schedulers(args.gnn, model_optimizer, head_optimizer)
    if args.resume == 'pretrain':
        if not os.path.isfile(args.pretrain_model):
            raise FileNotFoundError(args.pretrain_model)
        checkpoint = torch.load(args.pretrain_model, map_location=device)
        model.load_state_dict(checkpoint['model'])

    best_score = float('inf')
    best_path = run_dir / 'model_best.pth.tar'
    writer = SummaryWriter(log_dir=str(run_dir / 'tensorboard'))
    for epoch in range(args.epochs):
        train_loss, train_mae = train_epoch(
            train_loader, model, head, criterion, model_optimizer,
            head_optimizer, normalizer, device, args.gnn)
        if not np.isfinite(train_loss):
            raise FloatingPointError('Non-finite training loss')
        if val_loader is not None:
            score, val_mae = evaluate(
                val_loader, model, head, criterion, normalizer, device, args.gnn)
            writer.add_scalar('mae/validation', val_mae, epoch)
        else:
            score = train_loss
        for scheduler in schedulers:
            if args.gnn == 'MEGConv':
                scheduler.step()
            else:
                scheduler.step(score)
        is_best = args.gnn != 'MEGConv' and score < best_score
        if args.gnn != 'MEGConv':
            best_score = min(best_score, score)
        state = {
            'epoch': epoch + 1, 'model': model.state_dict(),
            'pred_head': head.state_dict(),
            'optimizer_model': model_optimizer.state_dict(),
            'optimizer_pred_head': head_optimizer.state_dict(),
            'schedulers': [scheduler.state_dict() for scheduler in schedulers],
            'best_loss': best_score, 'run_seed': run_seed,
            'train_indices': train_idx, 'val_indices': val_idx,
            'test_indices': test_idx, 'args': vars(args),
        }
        save_checkpoint(state, is_best, str(run_dir / 'checkpoint.pth.tar'),
                        str(best_path))
        writer.add_scalar('loss/train', train_loss, epoch)
        writer.add_scalar('mae/train', train_mae, epoch)
    if args.gnn != 'MEGConv':
        selected = torch.load(best_path, map_location=device)
        model.load_state_dict(selected['model'])
        head.load_state_dict(selected['pred_head'])
    test_loss, test_mae, true_values, predicted_values = evaluate(
        test_loader, model, head, criterion, normalizer, device, args.gnn,
        return_values=True)
    test_r2 = r2_score(true_values, predicted_values)
    with (run_dir / 'predictions.csv').open(
            'w', encoding='utf-8', newline='') as stream:
        writer_csv = csv.writer(stream)
        writer_csv.writerow(['id', 'y_true', 'y_pred'])
        writer_csv.writerows(zip(
            [ids[index] for index in test_idx], true_values, predicted_values))
    writer.add_hparams(hparams, {'final/test_mae': test_mae})
    writer.close()
    print(f'Run {run_number}: test loss={test_loss:.6f}, '
          f'MAE={test_mae:.6f}, R2={test_r2:.6f}')
    return {'mae': test_mae, 'r2': test_r2, 'split_mode': mode}


def main():
    args = parse_args()
    if args.runs < 1:
        raise ValueError('--runs must be at least 1')
    if args.epochs < 1:
        raise ValueError('--epochs must be at least 1')
    args.save_dir.mkdir(parents=True, exist_ok=True)
    cuda = not args.disable_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    hparams = load_hparams(args.finetune_config, args.gnn)
    dataset, ids = load_dataset(args)
    start = time.time()
    results = [run_once(args, hparams, dataset, ids, run_index, device, cuda)
               for run_index in range(args.runs)]
    run_maes = [result['mae'] for result in results]
    run_r2 = [result['r2'] for result in results]
    np.save(args.save_dir / 'metric_list.npy', np.asarray(run_maes))
    summary = {
        'dataset': args.dataset, 'gnn': args.gnn, 'runs': args.runs,
        'training_seeds': [args.seed + i for i in range(args.runs)],
        'split_mode': args.split_mode, 'split_seed': args.split_seed,
        'resolved_split_modes': [result['split_mode'] for result in results],
        'mae_values': [float(value) for value in run_maes],
        'mae_mean': float(np.mean(run_maes)),
        'mae_std': float(np.std(run_maes)),
        'r2_values': [float(value) for value in run_r2],
        'r2_mean': float(np.mean(run_r2)),
        'r2_std': float(np.std(run_r2)),
        'elapsed_seconds': time.time() - start,
    }
    with (args.save_dir / 'summary.yaml').open('w', encoding='utf-8') as stream:
        yaml.safe_dump(summary, stream, sort_keys=False)
    print(f"MAE: {summary['mae_mean']:.6f} +/- {summary['mae_std']:.6f}")


if __name__ == '__main__':
    main()
