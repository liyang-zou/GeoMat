"""Fine-tune GeoMat backbones on the official MatBench outer folds."""

import argparse
import os
import re
import time

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from matbench import MatbenchBenchmark
from torch.utils.tensorboard import SummaryWriter

from data.dataloader import MatBench_Data
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from models.model import encoder, readout
from models.PaiNN import PainnEncoder
from utils.normalizer import Normalizer
from utils.save_checkpoint import save_checkpoint, EarlyStopping
from utils.metrics import AverageMeter, mae
from utils.set_seeds import set_all_seeds

PRETRAINED_CHECKPOINTS = {
    'CGConv': 'geomat_cgcnn_pretrained.pth.tar',
    'MEGConv': 'geomat_megnet_pretrained.pth.tar',
    'Painn': 'geomat_painn_pretrained.pth.tar',
}

parser = argparse.ArgumentParser(description='GeoMat MatBench fine-tuning')
parser.add_argument('data_root', help='directory used for the MatBench graph cache')
parser.add_argument('--pretrain-model',
                    help='pretrained checkpoint; defaults to the selected backbone')
parser.add_argument('--finetune-config', default='./finetune_config.yaml',
                    help='path to finetune config yaml file')
parser.add_argument('--atom-init', default='./data/atom_init.json',
                    help='CGCNN elemental feature file (default: ./data/atom_init.json)')
parser.add_argument('--save-dir', default='./',
                    help='directory to save checkpoints and log')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--seed', default=42, type=int,
                    help='global seed for splits, initialization, and data order (default: 42)')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run (default: 30)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--resume', default='pretrain',
                    choices=('pretrain', 'finetune', 'scratch'),
                    help='initialization or checkpoint mode (default: pretrain)')

parser.add_argument('--n-conv', default=3, type=int, metavar='N',
                    help='number of conv layers')
parser.add_argument('--gnn', default='CGConv', choices=('CGConv', 'MEGConv', 'Painn'),
                    help='type of gnn')

args = parser.parse_args()
if args.pretrain_model is None:
    args.pretrain_model = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'checkpoint',
        PRETRAINED_CHECKPOINTS[args.gnn])

args.cuda = not args.disable_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

def main():
    set_all_seeds(args.seed)
    with open(args.finetune_config, 'r') as f:
        hparams_finetune = yaml.safe_load(f)
    if args.gnn == 'MEGConv':
        hparams_finetune2 = {'batch_size_finetune': 32, 'p_finetune': 0, 'lr_finetune': 0.0005, 'weight_decay_finetune': 0.0}
    elif args.gnn == 'CGConv':
        hparams_finetune2 = {'batch_size_finetune': 128, 'p_finetune': 0, 'lr_finetune': 0.01, 'weight_decay_finetune': 0.000001}
    elif args.gnn == 'Painn':
        hparams_finetune2 = {'batch_size_finetune': 128, 'p_finetune': 0, 'lr_finetune': 0.0001, 'weight_decay_finetune': 0.01}
    hparams = {'train_ratio': 1.0, **hparams_finetune2, **hparams_finetune}
    hparams.setdefault('ph_lr', hparams['lr_finetune'])
    hparams.setdefault('ph_wd', hparams['weight_decay_finetune'])
    if not 0 < hparams['train_ratio'] <= 1:
        raise ValueError('train_ratio in the fine-tuning YAML must be in (0, 1]')

    start_time = time.time()

    # load data
    if 'jdft2d' in hparams['data_name']:
        task_type = 'regression'
        task_name = "matbench_jdft2d"
    elif 'phonons' in hparams['data_name']:
        task_type = 'regression'
        task_name = "matbench_phonons"
    elif 'mp_gap' in hparams['data_name']:
        task_type = 'regression'
        task_name = "matbench_mp_gap"
    elif 'dielectric' in hparams['data_name']:
        task_type = 'regression'
        task_name = "matbench_dielectric"
    elif 'log_gvrh' in hparams['data_name']:
        task_type = 'regression'
        task_name = "matbench_log_gvrh"
    elif 'log_kvrh' in hparams['data_name']:
        task_type = 'regression'
        task_name = "matbench_log_kvrh"
    elif 'perovskites' in hparams['data_name']:
        task_type = 'regression'
        task_name = "matbench_perovskites"
    elif 'mp_e_form' in hparams['data_name']:
        task_type = 'regression'
        task_name = "matbench_mp_e_form"
    else:
        raise ValueError(f"Unsupported MatBench regression task: {hparams['data_name']}")

    mb = MatbenchBenchmark(subset=[task_name], autoload=False)
    train_dfs = []
    test_dfs = []

    for task in mb.tasks:
        task.load()
        for k in task.folds:
            train_dfs.append(task.get_train_and_val_data(k, as_type="df"))
            test_dfs.append(task.get_test_data(k, as_type="df"))

    dataset = MatBench_Data(args.data_root, subset_data=task_name, task_type=task_type, radius=5.0, neighbor_strategy='radius_graph')
    metric_list = []
    for k in range(len(train_dfs)):
        set_all_seeds(args.seed)
        print(f"====== Fold {k + 1}/{len(train_dfs)} start training ======")
        fold_start_time = time.time()
        best_loss = 1e10
        fold_start_epoch = args.start_epoch
        checkpoint_path = os.path.join(args.save_dir, f'checkpoint{k}.pth.tar')
        best_model_path = os.path.join(args.save_dir, f'model_best{k}.pth.tar')
        fold_checkpoint = None
        if args.resume == 'finetune':
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(checkpoint_path)
            fold_checkpoint = torch.load(checkpoint_path, map_location=device)

        train_idx = mbid2idx(dataset, train_dfs[k])
        test_idx = mbid2idx(dataset, test_dfs[k])
        test_dataset = Subset(dataset, test_idx)

        if fold_checkpoint is not None and 'train_indices' in fold_checkpoint:
            train_dataset = Subset(dataset, fold_checkpoint['train_indices'])
            val_dataset = Subset(dataset, fold_checkpoint['val_indices'])
        else:
            train_dataset, val_dataset = make_data_subsets(
                dataset, train_idx, args.gnn, hparams['train_ratio'],
                args.seed)

        train_generator = torch.Generator().manual_seed(args.seed)
        train_loader = DataLoader(train_dataset,
                                  batch_size=hparams['batch_size_finetune'], shuffle=True, num_workers=args.workers,
                                  pin_memory=args.cuda, generator=train_generator)
        if args.gnn != 'MEGConv':
            val_loader = DataLoader(val_dataset,
                                batch_size=hparams['batch_size_finetune'], shuffle=False, num_workers=args.workers,
                                pin_memory=args.cuda)
        test_loader = DataLoader(test_dataset,
                                 batch_size=hparams['batch_size_finetune'], shuffle=False, num_workers=args.workers,
                                 pin_memory=args.cuda)

        print('train_dataset: ', len(train_dataset))
        if args.gnn != 'MEGConv':
            print('val_dataset: ', len(val_dataset))
        print('test_dataset: ', len(test_dataset))

        # obtain target value normalizer
        target_normalizer = Normalizer(
            dataset[base_dataset_indices(train_dataset)].y)
        print('train std: {}'.format(target_normalizer.std))
        print('train mean: {}'.format(target_normalizer.mean))
        target_normalizer.to(device)

        # build model
        if args.gnn == 'CGConv':
            target_MLP_hidden = [128]
            atom_out_fea_len = 64
            edge_out_fea_len = 32
        elif args.gnn == 'MEGConv':
            target_MLP_hidden = [32]
            atom_out_fea_len = 32
            edge_out_fea_len = 32  # always use default config
        elif args.gnn == 'Painn':
            target_MLP_hidden = [64]
            atom_out_fea_len = 64
            edge_out_fea_len = 32
        atom_init = args.atom_init
        if args.gnn == 'Painn':
            model = PainnEncoder(num_interactions=args.n_conv, hidden_state_size=atom_out_fea_len).to(device)
        else:
            model = encoder(num_layer = args.n_conv, node_dim = atom_out_fea_len, edge_dim = edge_out_fea_len, gnn_type=args.gnn, atom_init=atom_init, norm='layernorm').to(device)
            
        pred_head = readout(hidden_dim = atom_out_fea_len, out_dim = 1, readout_type=args.gnn, MLP_hidden_dims=target_MLP_hidden, p=hparams['p_finetune']).to(device)
        model.apply(init_weights)
        pred_head.apply(init_weights)

        # define loss func and optimizer
        criterion = (nn.L1Loss() if args.gnn == 'MEGConv' else nn.MSELoss()).to(device)

        optimizer_model = optim.AdamW(model.parameters(), hparams['lr_finetune'], weight_decay=hparams['weight_decay_finetune'])
        optimizer_pred_head = optim.AdamW(pred_head.parameters(), hparams['ph_lr'], weight_decay=hparams['ph_wd'])

        if args.gnn == 'CGConv':
            print('plateau decay is used')
            scheduler_model = optim.lr_scheduler.ReduceLROnPlateau(optimizer_model, mode='min', factor=0.1, patience=100, verbose=True, min_lr=1e-6)
            scheduler_pred_head = optim.lr_scheduler.ReduceLROnPlateau(optimizer_pred_head, mode='min', factor=0.1, patience=100, verbose=True, min_lr=1e-6)
        elif args.gnn == 'Painn':
            print('plateau decay is used')
            scheduler_model = optim.lr_scheduler.ReduceLROnPlateau(optimizer_model, mode='min', factor=0.5, patience=50, verbose=True, min_lr=1e-6)
            scheduler_pred_head = optim.lr_scheduler.ReduceLROnPlateau(optimizer_pred_head, mode='min', factor=0.5, patience=50, verbose=True, min_lr=1e-6)
        elif args.gnn == 'MEGConv':
            print('linear decay is used')
            scheduler_model = torch.optim.lr_scheduler.LambdaLR(optimizer_model, lr_lambda=megnet_decay_multiplier)
            scheduler_pred_head = torch.optim.lr_scheduler.LambdaLR(optimizer_pred_head, lr_lambda=megnet_decay_multiplier)

        scheduler_list = [scheduler_model, scheduler_pred_head]

        # Initialize from pre-training, resume a fold, or train from scratch.
        if args.resume == 'pretrain':
            resume_path  = args.pretrain_model
            if os.path.isfile(resume_path):
                print("=> loading pretrain checkpoint '{}'".format(resume_path))
                checkpoint = torch.load(resume_path, map_location=device)
                model.load_state_dict(checkpoint['model'])
                pred_head.apply(init_weights)
            else:
                raise FileNotFoundError(resume_path)
        elif args.resume == 'finetune':
            resume_path = checkpoint_path
            if os.path.isfile(resume_path):
                print("=> loading finetune checkpoint '{}'".format(args.resume))
                checkpoint = fold_checkpoint
                fold_start_epoch = checkpoint['epoch']
                best_loss = checkpoint['best_loss']
                model.load_state_dict(checkpoint['model'])
                pred_head.load_state_dict(checkpoint['pred_head'])
                optimizer_model.load_state_dict(checkpoint['optimizer_model'])
                optimizer_pred_head.load_state_dict(checkpoint['optimizer_pred_head'])
                if 'schedulers' in checkpoint:
                    for scheduler, state in zip(scheduler_list, checkpoint['schedulers']):
                        scheduler.load_state_dict(state)
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                raise FileNotFoundError(resume_path)
        else:
            model.apply(init_weights)
            pred_head.apply(init_weights)
            print('=> training from scratch')

        log_dir = os.path.join(args.save_dir, f'finetune{k}_log')
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        if args.gnn == 'CGConv':
            early_stop = EarlyStopping(patience=500)
        elif args.gnn == 'Painn':
            early_stop = EarlyStopping(patience=200)

        for epoch in range(fold_start_epoch, args.epochs):
            epoch_time = time.time()
            # train for one epoch
            train_loss, train_mae = train(
                train_loader, model, pred_head, criterion, optimizer_model,
                optimizer_pred_head, target_normalizer, device)
            writer.add_scalar(' Loss/train', train_loss, epoch)
            writer.add_scalar(' MAE/train', train_mae, epoch)
            # evaluate on validation set
            if args.gnn != 'MEGConv':
                val_loss, val_mae = validate(
                    val_loader, model, pred_head, criterion, target_normalizer, device)
                print(' Loss/train: ', round(train_loss, 5), '     Loss/val: ', round(val_loss, 5))
                print(' MAE/train: ', round(train_mae, 5), '     MAE/val: ', round(val_mae, 5))
                writer.add_scalar(' MAE/val', val_mae, epoch)
                writer.add_scalar(' Loss/val', val_loss, epoch)
                is_best = val_loss < best_loss
            else:
                print(' Loss/train: ', round(train_loss, 5))
                print(' MAE/train: ', round(train_mae, 5))
                is_best = False

            if not np.isfinite(train_loss):
                raise FloatingPointError('Non-finite training loss')
            if args.gnn != 'MEGConv' and not np.isfinite(val_loss):
                raise FloatingPointError('Non-finite validation loss')

            for sche in scheduler_list:
                if sche is not None:
                    if args.gnn != 'MEGConv':
                        sche.step(val_loss)
                    else:
                        sche.step()

            # Save the latest state and the best validation checkpoint.
            if is_best:
                best_loss = val_loss
            save_checkpoint({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'pred_head': pred_head.state_dict(),
                'optimizer_model': optimizer_model.state_dict(),
                'optimizer_pred_head': optimizer_pred_head.state_dict(),
                'schedulers': [scheduler.state_dict() for scheduler in scheduler_list],
                'train_indices': base_dataset_indices(train_dataset),
                'val_indices': base_dataset_indices(val_dataset),
                'best_loss': best_loss,
                'args': vars(args)
            }, is_best,checkpoint_path,best_model_path)

            print(f'Epoch {epoch} time used (sec): ', round(time.time() - epoch_time, 3))

            if args.gnn != 'MEGConv':
                if early_stop(val_loss):
                    print('Exit training loop due to early stopping!')
                    break

        if args.gnn != 'MEGConv':
            print('****** best_loss: ', round(best_loss, 6))

        # evaluation in test dataset
        if args.gnn != 'MEGConv':
            best_model = torch.load(best_model_path, map_location=device)
            model.load_state_dict(best_model['model'])
            pred_head.load_state_dict(best_model['pred_head'])
        test_loss, test_mae = validate(
            test_loader, model, pred_head, criterion, target_normalizer, device)
        print(f'****** test in Fold {k + 1}/{len(train_dfs)} ****** ')
        print('Loss/test: ', round(test_loss, 6))
        print('MAE/test: ', round(test_mae, 6))
        print('one fold time used (sec) : ', round(time.time() - fold_start_time, 3))
        writer.add_hparams(hparams, metric_dict={"final/test_mae": test_mae})
        writer.close()
        metric_list.append(test_mae)

    print("****** full test ******")
    print(f"training in {len(train_dataset)} samples")
    print(metric_list)
    print(' mae average: ', round(np.mean(metric_list), 6))
    print(' mae std: ', round(np.std(metric_list), 6))
    print('total times used (sec) : ', round(time.time() - start_time, 3))

def predict_batch(model, pred_head, batch, gnn):
    """Apply the backbone-specific regression readout."""
    if gnn == 'MEGConv':
        x, edge_attr, _, state = model(batch)
        return pred_head(x, batch.batch, edge_index=batch.edge_index,
                         edge_attr=edge_attr, state=state)
    if gnn == 'Painn':
        x, _ = model(batch)
        return pred_head(x, batch.batch)
    x, _, _ = model(batch)
    return pred_head(x, batch.batch)


def train(train_loader, model, pred_head, criterion, optimizer_model,
          optimizer_pred_head, target_normalizer, device):

    model.train()
    pred_head.train()

    loss_accum = AverageMeter()
    mae_accum = AverageMeter()

    for step, batch in enumerate(train_loader):
        batch = batch.to(device)

        prediction = predict_batch(model, pred_head, batch, args.gnn)
        loss = criterion(prediction, target_normalizer.norm(batch.y.float()).view(-1, 1))
        loss_accum.update(loss.item(), n=batch.y.size(0))
        mae_accum.update(mae(target_normalizer.denorm(prediction), batch.y.view(-1, 1)).item(), n=batch.y.size(0))
        optimizer_model.zero_grad()
        optimizer_pred_head.zero_grad()

        loss.backward()
        optimizer_model.step()
        optimizer_pred_head.step()

        if step % 20 == 0:
            print('train Step {0}/{1}\n'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'MAE {mae.val:.4f} ({mae.avg:.4f})\t'.format(
                step, len(train_loader),
                loss=loss_accum,mae=mae_accum)
                )

    return loss_accum.avg, mae_accum.avg

@torch.no_grad()
def validate(val_loader, model, pred_head, criterion, target_normalizer, device):

    # switch to eval mode
    model.eval()
    pred_head.eval()

    loss_accum = AverageMeter()
    mae_accum = AverageMeter()

    for step, batch in enumerate(val_loader):
        batch = batch.to(device)

        prediction = predict_batch(model, pred_head, batch, args.gnn)
        loss = criterion(prediction, target_normalizer.norm(batch.y.float()).view(-1, 1))
        loss_accum.update(loss.item(), n=batch.y.size(0))
        mae_accum.update(mae(target_normalizer.denorm(prediction), batch.y.view(-1, 1)).item(), n=batch.y.size(0))

        if step % 2 == 0:
            print('val Step {0}/{1}\n'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'MAE {mae.val:.4f} ({mae.avg:.4f})\t'.format(
                step, len(val_loader),
                loss=loss_accum,mae=mae_accum)
                )

    return loss_accum.avg, mae_accum.avg

def megnet_decay_multiplier(epoch, epo_min=100, epo=1000, lr_start=5e-4, lr_stop=5e-6):
    """Return the MEGNET linear-decay LR multiplier after the initial plateau."""
    lr_range = lr_start - lr_stop
    if epoch < epo_min:
        return 1.0
    if epoch < epo:
        progress = (epoch - epo_min) / (epo - epo_min)
        current_lr = lr_start - lr_range * progress
        return current_lr / lr_start
    return lr_stop / lr_start

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=0.1)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

def base_dataset_indices(dataset):
    """Resolve nested Subset indices against the graph dataset."""
    if isinstance(dataset, Subset):
        parent = base_dataset_indices(dataset.dataset)
        return [parent[i] for i in dataset.indices]
    return list(range(len(dataset)))


def make_data_subsets(dataset, outer_train_indices, gnn, train_ratio, split_seed):
    """Create a fixed train/validation split and nested data-fraction subset."""
    split_generator = torch.Generator().manual_seed(split_seed)
    split_order = torch.randperm(
        len(outer_train_indices), generator=split_generator).tolist()
    ordered_indices = [outer_train_indices[i] for i in split_order]

    if gnn == 'MEGConv':
        train_pool = ordered_indices
        val_indices = []
    else:
        n_train = int(0.75 * len(ordered_indices))
        train_pool = ordered_indices[:n_train]
        val_indices = ordered_indices[n_train:]

    fraction_generator = torch.Generator().manual_seed(split_seed)
    fraction_order = torch.randperm(
        len(train_pool), generator=fraction_generator).tolist()
    ordered_train_pool = [train_pool[i] for i in fraction_order]
    n_subset = int(train_ratio * len(ordered_train_pool))
    if n_subset == 0:
        raise ValueError('train_ratio selects an empty training subset')

    return (Subset(dataset, ordered_train_pool[:n_subset]),
            Subset(dataset, val_indices))

# map mat_id to idx
def mbid2idx(dataset, df):
    id_to_idx = {int(data.id): i for i, data in enumerate(dataset)}
    idx = [id_to_idx[int(re.findall(r'-(\d+)', mbid)[0])] for mbid in df.index]
    return idx

if __name__ == '__main__':
    main()
