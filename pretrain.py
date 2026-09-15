"""Pre-train GeoMat with fixed loss weights or run the gradient pilot."""

import argparse
import json
import os
import time
from functools import partial

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch_scatter import scatter

from data.dataloader import CIFData
from data.dataloader import MaskTransform, PerturbTransform, Replace, Subgraph, Dropedge
from data.dataloader import MultiViewWrapper, DataLoaderMultiView
from models.PaiNN import PainnEncoder
from models.model import encoder, decoder, displacement_decoder
from utils.normalizer import Normalizer
from utils.save_checkpoint import save_checkpoint
from utils.metrics import AverageMeter, sce_loss, info_nce_loss, mae, accuracy_torch
from utils.set_seeds import set_all_seeds

parser = argparse.ArgumentParser(description='GeoMat self-supervised pretraining')
parser.add_argument('data_root',
                    help='directory containing the pre-training CSV')
parser.add_argument('--pretrain-config', default='./pretrain_config.yaml',
                    help='path to pretrain config yaml file')
parser.add_argument('--atom-init', default='./data/atom_init.json',
                    help='CGCNN elemental feature file (default: ./data/atom_init.json)')
parser.add_argument('--save-dir', default='./',
                    help='directory to save checkpoints and log')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='INT',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--seed', default=123, type=int,
                    help='random seed (default: 123)')
parser.add_argument('--epochs', default=25, type=int, metavar='INT',
                    help='number of total epochs to run (default: 25)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='INT',
                    help='manual epoch number (useful on restarts)')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--n-conv', default=3, type=int, metavar='INT',
                    help='number of conv layers')
parser.add_argument('--gnn', default='CGConv', choices=('CGConv', 'MEGConv', 'Painn'),
                    help='type of gnn')

parser.add_argument('--mask-subgraph', default=None, type=str, metavar='NAME',
                    help='enable subgraph masking and deleted-edge reconstruction')
parser.add_argument('--pertlatt', action='store_true',
                    help='Perturb lattice when perturbing atom position')
parser.add_argument('--drop-random', action='store_true',
                    help='Drop random edges during Dropping edge')
parser.add_argument('--task', default='union',
                    choices=('mask', 'denoise', 'contrastive', 'union'),
                    help='SSL tasks for pretrain (default: union)')
parser.add_argument('--gradient-pilot', action='store_true',
                    help='run the two-epoch unit-weight gradient pilot and exit')
parser.add_argument('--reduction', default='mean', choices=('mean', 'none'),
                    help='Average by node or edge (default: mean)')
args = parser.parse_args()

args.cuda = not args.disable_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
best_loss = 1e10

def main():
    global args, best_loss
    num_atom_types = 94
    with open(args.pretrain_config, 'r') as f:
        hparams = yaml.safe_load(f)
    print(hparams)
    print(args)
    if args.gradient_pilot and (args.resume or args.start_epoch != 0):
        raise ValueError('The gradient pilot must start from a newly initialized model')
    if args.gradient_pilot and args.task != 'union':
        raise ValueError('The gradient pilot requires --task union')

    set_all_seeds(args.seed)
    start_time = time.time()
    # load data
    mask = MaskTransform(num_atom_types, hparams['ratio'], args.mask_subgraph, hparams['delete'])
    perturb = PerturbTransform(hparams['mu'], hparams['sigma'], args.pertlatt, hparams['mu2'], hparams['sigma2'])
    replace = Replace(hparams['ratio'])
    subgraph = Subgraph(0.35, method = hparams['subgraph_method'])
    dropedge = Dropedge(0.35, random = args.drop_random)

    if hparams['aug1'] == 'perturb' :
        transform1 = perturb
    elif hparams['aug1'] == 'replace':
        transform1 = replace
    elif hparams['aug1'] == 'dropedge':
        transform1 = dropedge
    elif hparams['aug1'] == 'subgraph':
        transform1 = subgraph
    elif hparams['aug1'] == None:
        transform1 = None
    else:
        print('augmentation error')
        assert False

    if hparams['aug2'] == 'perturb' :
        transform2 = perturb
    elif hparams['aug2'] == 'replace':
        transform2 = replace
    elif hparams['aug2'] == 'dropedge':
        transform2 = dropedge
    elif hparams['aug2'] == 'subgraph':
        transform2 = subgraph
    elif hparams['aug2'] == None:
        transform2 = None
    else:
        print('augmentation error')
        assert False

    dataset = CIFData(args.data_root)
    train_size = int(0.95 * len(dataset))
    if train_size == 0 or train_size == len(dataset):
        raise ValueError('The 95/5 pre-training split requires at least 20 structures')
    train_data = MultiViewWrapper(dataset[:train_size], mask, perturb, transform1, transform2)
    val_data = MultiViewWrapper(dataset[train_size:], mask, perturb, transform1, transform2)
    train_loader = DataLoaderMultiView(train_data, batch_size=hparams['batch_size'], shuffle=True, num_workers=args.workers, pin_memory=args.cuda)
    val_loader = DataLoaderMultiView(val_data, batch_size=hparams['batch_size'], shuffle=False, num_workers=args.workers, pin_memory=args.cuda)
    print(f'Dataset split: {train_size} train / {len(dataset) - train_size} validation')
    # obtain target value normalizer
    edge_normalizer = Normalizer(torch.norm(dataset.displacement, p=2, dim=-1))   # Normalize distance targets for the MSE loss.
    edge_normalizer.to(device)
    if args.pertlatt:
        latt_normalizer = Normalizer(dataset.lattice_lengths)
        latt_normalizer.to(device)
    else:
        latt_normalizer = None
    normalizer_list = [edge_normalizer, latt_normalizer]

    # build model
    node_MLP_hidden = [hparams['node_hidden_MLP1'], hparams['node_hidden_MLP2']]
    edge_MLP_hidden = [hparams['edge_hidden_MLP1'], hparams['edge_hidden_MLP2']]
    graph_MLP_hidden = [hparams['graph_hidden_MLP1'], hparams['graph_hidden_MLP2']]
    atom_init = args.atom_init
    backbone_dims = {
        'CGConv': (64, 32),
        'MEGConv': (32, 32),
        'Painn': (64, 64),
    }
    atom_out_fea_len, edge_out_fea_len = backbone_dims[args.gnn]
    print('atom_out_fea_len: {}'.format(atom_out_fea_len))
    print('edge_out_fea_len: {}'.format(edge_out_fea_len))
    if args.gnn == 'Painn':
        model = PainnEncoder(num_interactions=args.n_conv, hidden_state_size=atom_out_fea_len).to(device)
    else:
        model = encoder(num_layer=args.n_conv, node_dim=atom_out_fea_len, edge_dim=edge_out_fea_len, gnn_type=args.gnn, atom_init=atom_init, norm='layernorm').to(device)
    node_decoder = decoder(hidden_dim=atom_out_fea_len, out_dim=num_atom_types,
                           decoder_type='CGConv', MLP_hidden_dims=node_MLP_hidden, edge_dim=edge_out_fea_len, p=hparams['p']).to(device)
    projector = decoder(hidden_dim=atom_out_fea_len, out_dim=hparams['z_dim'],
                        decoder_type='MLP', MLP_hidden_dims=graph_MLP_hidden, p=hparams['p']).to(device)
    if args.mask_subgraph is not None:
        if args.gnn == 'Painn':
            edge_decoder = displacement_decoder(atom_dim=atom_out_fea_len, out_dim=1).to(device)
        else:
            edge_decoder = decoder(hidden_dim=atom_out_fea_len, out_dim=1,
                                decoder_type='MLP', MLP_hidden_dims=edge_MLP_hidden, p=hparams['p']).to(device)
    else:
        edge_decoder = None

    if args.gnn == 'Painn':
        noise_decoder = displacement_decoder(atom_dim=atom_out_fea_len, out_dim=1).to(device)
    else:
        noise_decoder = None

    if args.pertlatt:
        print('prediction lattice parameter!')
        lattice_decoder = decoder(hidden_dim = atom_out_fea_len, out_dim = 9,
                                  decoder_type='MLP', MLP_hidden_dims=graph_MLP_hidden, p=hparams['p']).to(device)
    else:
        lattice_decoder = None

    model.apply(init_weights)
    node_decoder.apply(init_weights)
    projector.apply(init_weights)
    if args.mask_subgraph is not None :
        edge_decoder.apply(init_weights)
    if args.pertlatt:
        lattice_decoder.apply(init_weights)

    model_list = [model, node_decoder, edge_decoder, projector, lattice_decoder, noise_decoder]

    # define loss func and optimizer
    if hparams['classification_loss'] == 'sce':
        criterion_class = partial(sce_loss, alpha = hparams['alpha'], reduction = args.reduction)
    else:
        criterion_class = nn.CrossEntropyLoss().to(device)
    criterion_mse = nn.MSELoss(reduction = args.reduction).to(device)
    criterion_contrast = partial(info_nce_loss, temperature = hparams['T'], normalize = True)
    criterion_list = [criterion_class, criterion_mse, criterion_contrast]

    optimizer_model = optim.AdamW(model.parameters(), hparams['lr'],weight_decay=hparams['weight_decay'])
    optimizer_node_dec = optim.AdamW(node_decoder.parameters(), hparams['lr'],weight_decay=hparams['weight_decay'])
    optimizer_projector = optim.AdamW(projector.parameters(), hparams['lr'],weight_decay=hparams['weight_decay'])
    if args.mask_subgraph is not None:
        optimizer_edge_dec = optim.AdamW(edge_decoder.parameters(), hparams['lr'],weight_decay=hparams['weight_decay'])
    else:
        optimizer_edge_dec = None
    if noise_decoder is not None:
        optimizer_noise_dec = optim.AdamW(noise_decoder.parameters(), hparams['lr'],weight_decay=hparams['weight_decay'])
    else:
        optimizer_noise_dec = None
    if args.pertlatt:
        optimizer_lattice_dec = optim.AdamW(lattice_decoder.parameters(), hparams['lr'],weight_decay=hparams['weight_decay'])
    else:
        optimizer_lattice_dec = None
    optimizer_list = [optimizer_model, optimizer_node_dec, optimizer_edge_dec,
                      optimizer_projector, optimizer_lattice_dec, optimizer_noise_dec]

    print('Cosine learning-rate decay')
    scheduler = lambda epoch: (1 + np.cos(epoch * np.pi / args.epochs)) * 0.5
    scheduler_model = torch.optim.lr_scheduler.LambdaLR(optimizer_model, lr_lambda=scheduler)
    scheduler_node_dec = torch.optim.lr_scheduler.LambdaLR(optimizer_node_dec, lr_lambda=scheduler)
    if args.mask_subgraph is not None:
        scheduler_edge_dec = torch.optim.lr_scheduler.LambdaLR(optimizer_edge_dec, lr_lambda=scheduler)
    else:
        scheduler_edge_dec = None
    if noise_decoder is not None:
        scheduler_noise_dec = torch.optim.lr_scheduler.LambdaLR(optimizer_noise_dec, lr_lambda=scheduler)
    else:
        scheduler_noise_dec = None
    if args.pertlatt:
        scheduler_lattice_dec = torch.optim.lr_scheduler.LambdaLR(optimizer_lattice_dec, lr_lambda=scheduler)
    else:
        scheduler_lattice_dec = None
    scheduler_proj_dec = torch.optim.lr_scheduler.LambdaLR(optimizer_projector, lr_lambda=scheduler)
    scheduler_list = [scheduler_model]
    if args.task in ('mask', 'union'):
        scheduler_list.extend([scheduler_node_dec, scheduler_edge_dec])
    if args.task in ('contrastive', 'union'):
        scheduler_list.append(scheduler_proj_dec)
    if args.task in ('denoise', 'union'):
        scheduler_list.extend([scheduler_lattice_dec, scheduler_noise_dec])
    # optionally resume from a checkpoint
    if args.resume:
        resume_path = os.path.join(args.save_dir, args.resume)
        if os.path.isfile(resume_path):
            print("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path, map_location=device)
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['model'])
            node_decoder.load_state_dict(checkpoint['node_decoder'])
            projector.load_state_dict(checkpoint['projector'])
            if edge_decoder is not None:
                edge_decoder.load_state_dict(checkpoint['edge_decoder'])
            if noise_decoder is not None:
                noise_decoder.load_state_dict(checkpoint['noise_decoder'])
            if lattice_decoder is not None:
                lattice_decoder.load_state_dict(checkpoint['lattice_decoder'])
            optimizer_model.load_state_dict(checkpoint['optimizer_model'])
            optimizer_node_dec.load_state_dict(checkpoint['optimizer_node_dec'])
            optimizer_projector.load_state_dict(checkpoint['optimizer_projector'])
            if optimizer_edge_dec is not None:
                optimizer_edge_dec.load_state_dict(checkpoint['optimizer_edge_dec'])
            if optimizer_noise_dec is not None:
                optimizer_noise_dec.load_state_dict(checkpoint['optimizer_noise_dec'])
            if optimizer_lattice_dec is not None:
                optimizer_lattice_dec.load_state_dict(checkpoint['optimizer_lattice_dec'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    log_dir = os.path.join(args.save_dir, 'tensorboard_log')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    final_epoch = 2 if args.gradient_pilot else args.epochs
    pilot_grad_stats = {
        'mask': [], 'mask2': [], 'denoise': [], 'lattice': [], 'contrast': []
    }
    for epoch in range(args.start_epoch, final_epoch):
        epoch_time = time.time()
        # train for one epoch

        train_total_loss, train_mask_loss, train_denoise_loss, train_contrast_loss, train_mask_loss2, train_latt_loss, grad_stats = train(
            train_loader, model_list, criterion_list, optimizer_list, normalizer_list,
            hparams, args.gradient_pilot, device)

        for name, values in grad_stats.items():
            pilot_grad_stats[name].extend(values)

        if args.gradient_pilot:
            if not np.isfinite(train_total_loss):
                raise FloatingPointError('Non-finite loss during the gradient pilot')
            print(' Loss/train_total: ', round(train_total_loss, 5))
            print(' Loss/train_mask: ', round(train_mask_loss, 5))
            print(' Loss/train_denoise: ', round(train_denoise_loss, 5))
            print(' Loss/train_contrast: ', round(train_contrast_loss, 5))
            print(' Loss/train_mask2: ', round(train_mask_loss2, 5))
            print(' Loss/train_lattice: ', round(train_latt_loss, 5))
            for sche in scheduler_list:
                if sche is not None:
                    sche.step()
            print(f'Epoch {epoch} gradient pilot time (sec): ',
                  round(time.time() - epoch_time, 3))
            continue

        # evaluate on validation set
        val_total_loss, val_mask_loss, val_denoise_loss, val_contrast_loss, val_mask_loss2, val_latt_loss = validate(
            val_loader, model_list, criterion_list, normalizer_list,
            hparams, device)
        torch.cuda.empty_cache()

        print(' Loss/train_total: ', round(train_total_loss, 5), '     Loss/val_total: ', round(val_total_loss, 5))
        print(' Loss/train_mask: ', round(train_mask_loss, 5), '     Loss/val_mask: ', round(val_mask_loss, 5))
        print(' Loss/train_denoise: ', round(train_denoise_loss, 5), '     Loss/val_denoise: ', round(val_denoise_loss, 5))
        print(' Loss/train_contrast: ', round(train_contrast_loss, 5), '     Loss/val_contrast: ', round(val_contrast_loss, 5))
        print(' Loss/train_mask2: ', round(train_mask_loss2, 5), '     Loss/val_mask2: ', round(val_mask_loss2, 5))
        print(' Loss/train_lattice: ', round(train_latt_loss, 5), '     Loss/val_lattice: ', round(val_latt_loss, 5))
        #tensorboard visualization
        writer.add_scalar(' Loss/train_total', train_total_loss, epoch)
        writer.add_scalar(' Loss/train_mask', train_mask_loss, epoch)
        writer.add_scalar(' Loss/train_denoise', train_denoise_loss, epoch)
        writer.add_scalar(' Loss/train_contrast', train_contrast_loss, epoch)
        writer.add_scalar(' Loss/train_mask2', train_mask_loss2, epoch)
        writer.add_scalar(' Loss/train_lattice', train_latt_loss, epoch)

        writer.add_scalar(' Loss/val_total', val_total_loss, epoch)
        writer.add_scalar(' Loss/val_mask', val_mask_loss, epoch)
        writer.add_scalar(' Loss/val_denoise', val_denoise_loss, epoch)
        writer.add_scalar(' Loss/val_contrast', val_contrast_loss, epoch)
        writer.add_scalar(' Loss/val_mask2', val_mask_loss2, epoch)
        writer.add_scalar(' Loss/val_lattice', val_latt_loss, epoch)

        if not np.isfinite(train_total_loss) or not np.isfinite(val_total_loss):
            raise FloatingPointError('Non-finite training or validation loss')

        for sche in scheduler_list:
            if sche is not None:
                sche.step()

        # remember the best mae_eror and save checkpoint
        is_best = val_total_loss < best_loss
        if is_best:
            best_loss = val_total_loss
        checkpoint_path = os.path.join(args.save_dir, f'checkpoint.pth_{epoch}.tar')   # Save a checkpoint for each epoch.
        best_model_path = os.path.join(args.save_dir, 'model_best.pth.tar')
        save_checkpoint({  # Save model and optimizer states with the best validation loss.
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'node_decoder': node_decoder.state_dict(),
            'edge_decoder': edge_decoder.state_dict() if edge_decoder is not None else None,
            'projector': projector.state_dict(),
            'noise_decoder': noise_decoder.state_dict() if noise_decoder is not None else None,
            'lattice_decoder': lattice_decoder.state_dict() if lattice_decoder is not None else None,
            'optimizer_model': optimizer_model.state_dict(),
            'optimizer_node_dec': optimizer_node_dec.state_dict(),
            'optimizer_edge_dec': optimizer_edge_dec.state_dict() if optimizer_edge_dec is not None else None,
            'optimizer_projector': optimizer_projector.state_dict(),
            'optimizer_noise_dec': optimizer_noise_dec.state_dict() if optimizer_noise_dec is not None else None,
            'optimizer_lattice_dec': optimizer_lattice_dec.state_dict() if optimizer_lattice_dec is not None else None,
            'best_loss': best_loss,
            'args': vars(args)                          # Store command-line arguments as a dictionary.
        }, is_best,checkpoint_path,best_model_path)
        print(f'Epoch {epoch} time for pretrain used (sec): ', round(time.time() - epoch_time, 3) )
    if args.gradient_pilot:
        write_gradient_pilot_summary(pilot_grad_stats, args.save_dir, args.seed)
    else:
        print('****** best_loss: ', best_loss)
    print('pretrain time used (sec) : ', round(time.time() - start_time, 3))

    writer.close()

def combine_task_losses(losses, hparams, task, include_masked_edges,
                        include_lattice, unit_weights=False):
    """Combine active loss terms with unit or fixed configuration weights."""
    if task == 'mask':
        active = ['mask']
        if include_masked_edges:
            active.append('mask2')
    elif task == 'denoise':
        active = ['denoise']
        if include_lattice:
            active.append('lattice')
    elif task == 'contrastive':
        active = ['contrast']
    elif task == 'union':
        active = ['mask', 'denoise', 'contrast']
        if include_masked_edges:
            active.append('mask2')
        if include_lattice:
            active.append('lattice')
    else:
        raise ValueError(f'Unknown pre-training task: {task}')

    terms = []
    for name in active:
        weight = 1.0 if unit_weights else hparams[f'{name}_weight']
        terms.append(weight * losses[name])
    return sum(terms[1:], terms[0])


def train(train_loader, model_list, criterion_list, optimizer_list, normalizer_list,
          hparams, gradient_pilot, device):
    model, node_decoder, edge_decoder, projector, lattice_decoder, noise_decoder = model_list
    criterion_class, criterion_mse, criterion_contrast = criterion_list
    optimizer_model, optimizer_node_dec, optimizer_edge_dec, optimizer_projector, optimizer_lattice_dec, optimizer_noise_dec = optimizer_list
    edge_normalizer, latt_normalizer = normalizer_list

    # switch to train mode
    model.train()
    node_decoder.train()
    projector.train()
    if edge_decoder is not None:
        edge_decoder.train()
    if noise_decoder is not None:
        noise_decoder.train()
    if lattice_decoder is not None:
        lattice_decoder.train()

    # Accumulate batch metrics.
    total_loss_accum = AverageMeter()
    mask_loss_accum = AverageMeter()
    denoise_loss_accum = AverageMeter()
    contrast_loss_accum = AverageMeter()
    mask2_loss_accum = AverageMeter()
    latt_loss_accum = AverageMeter()
    type_accu_accum = AverageMeter()
    distance_mae_accum = AverageMeter()

    grad_stats = {"mask": [], "mask2": [], "denoise": [], "lattice": [], "contrast": []}
    for step, batch in enumerate(train_loader):
        mask_batch, pert_batch, view1_batch, view2_batch = batch

        mask_loss = torch.tensor(0.).to(device)
        mask2_loss = torch.tensor(0.).to(device)
        denoise_loss = torch.tensor(0.).to(device)
        latt_loss = torch.tensor(0.).to(device)
        contrast_loss = torch.tensor(0.).to(device)
        with torch.enable_grad():
            if args.task == 'mask' or args.task == 'union':
                # Reconstruct masked atom types.
                mask_batch = mask_batch.to(device)
                if args.gnn == 'Painn':
                    mask_x, mask_x_vector = model(mask_batch)
                else:
                    mask_x, mask_edge_attr, *_ = model(mask_batch)
                if hparams['remask']:
                    pred_node = node_decoder(mask_x, mask_batch.edge_index, torch.norm(mask_batch.displacement, p=2, dim=-1), mask_atom_indices = mask_batch.masked_atom_indices)
                else:
                    pred_node = node_decoder(mask_x, mask_batch.edge_index, mask_edge_attr)

                if args.reduction == 'mean':
                    mask_loss = criterion_class(pred_node[mask_batch.masked_atom_indices], mask_batch.masked_atom_labels)
                elif args.reduction == 'none':
                    mask_loss_per_node = criterion_class(pred_node[mask_batch.masked_atom_indices], mask_batch.masked_atom_labels)
                    mask_loss = torch.mean(scatter(mask_loss_per_node, mask_batch.batch[mask_batch.masked_atom_indices], dim=0, reduce='mean'), dim=0)
                type_accu_accum.update(accuracy_torch(pred_node[mask_batch.masked_atom_indices], mask_batch.masked_atom_labels).item())
                mask_loss_accum.update(mask_loss.item())

                # Reconstruct masked edge distances.
                if edge_decoder is not None:
                    if args.gnn == 'Painn':
                        pred_edge = edge_decoder(mask_x[mask_batch.deleted_edge_index[0]] + mask_x[mask_batch.deleted_edge_index[1]],
                                                 mask_x_vector[mask_batch.deleted_edge_index[0]] - mask_x_vector[mask_batch.deleted_edge_index[1]])
                        mask2_loss = criterion_mse(pred_edge, mask_batch.deleted_displacement / edge_normalizer.mean)
                    else:
                        pred_edge = edge_decoder(mask_x[mask_batch.deleted_edge_index[0]] + mask_x[mask_batch.deleted_edge_index[1]])
                        if args.reduction == 'mean':
                            mask2_loss = criterion_mse(pred_edge, edge_normalizer.norm(mask_batch.deleted_distance).view(-1,1))
                        elif args.reduction == 'none':
                            edge_batch = mask_batch.batch[mask_batch.edge_index[0]]
                            mask2_loss_per_edge = criterion_mse(pred_edge, edge_normalizer.norm(mask_batch.deleted_distance).view(-1,1))
                            mask2_loss = torch.mean(scatter(mask2_loss_per_edge, edge_batch[mask_batch.masked_edge_indices], dim=0,reduce='mean'))
                    mask2_loss_accum.update(mask2_loss.item())

            if args.task == 'denoise' or args.task == 'union':
                # Reconstruct distances from perturbed structures.
                pert_batch = pert_batch.to(device)
                if args.gnn == 'Painn':
                    pert_x, pert_x_vector = model(pert_batch)
                    pred_noise = noise_decoder(pert_x, pert_x_vector)
                    pert_graph_fea = scatter(pert_x, pert_batch.batch, dim=0, reduce='mean')
                else:
                    pert_batch.pos.requires_grad_(True)
                    _, _, pert_graph_fea, pred_noise, *_ = model(pert_batch, True)
                if args.reduction == 'mean':
                    denoise_loss = criterion_mse(pred_noise, pert_batch.noise_labels / hparams['sigma'])
                elif args.reduction == 'none':
                    denoise_loss_per_node = criterion_mse(pred_noise, pert_batch.noise_labels)
                    denoise_loss = torch.mean(scatter(denoise_loss_per_node, pert_batch.batch, dim=0, reduce='mean'))
                denoise_loss_accum.update(denoise_loss.item())
                distance_mae_accum.update(mae(pred_noise * hparams['sigma'], pert_batch.noise_labels).item())
                # Reconstruct lattice parameters.
                if lattice_decoder is not None:
                    pred_lattice = lattice_decoder(pert_graph_fea)
                    pert_latt_labels = torch.cat([latt_normalizer.norm(pert_batch.lattice_lengths_labels),pert_batch.lattice_angles_labels], dim=-1)
                    latt_loss = criterion_mse(pred_lattice, pert_latt_labels).mean()
                    latt_loss_accum.update(latt_loss.item())
                    del pert_latt_labels
                else:
                    latt_loss = torch.tensor(0.0).to(device)

            if args.task == 'contrastive' or args.task == 'union':

                view1_batch = view1_batch.to(device)
                view2_batch = view2_batch.to(device)
                if args.gnn == 'Painn':
                    view1_x, _ = model(view1_batch)
                    view2_x, _ = model(view2_batch)
                    view1_graph_fea = scatter(view1_x, view1_batch.batch, dim=0, reduce='mean')
                    view2_graph_fea = scatter(view2_x, view2_batch.batch, dim=0, reduce='mean')
                else:
                    _, _, view1_graph_fea, *_ = model(view1_batch)
                    _, _, view2_graph_fea, *_ = model(view2_batch)
                z1 = projector(view1_graph_fea)
                z2 = projector(view2_graph_fea)
                contrast_loss = criterion_contrast(z1, z2)
                contrast_loss_accum.update(contrast_loss.item())

            losses = {
                'mask': mask_loss,
                'mask2': mask2_loss,
                'denoise': denoise_loss,
                'lattice': latt_loss,
                'contrast': contrast_loss,
            }
            loss = combine_task_losses(
                losses, hparams, args.task,
                include_masked_edges=edge_decoder is not None,
                include_lattice=lattice_decoder is not None,
                unit_weights=gradient_pilot,
            )

        total_loss_accum.update(loss.item())

        # Measure each unweighted loss on the shared backbone during the pilot.
        if gradient_pilot:
            for name, task_loss in losses.items():
                if not task_loss.requires_grad:
                    continue
                optimizer_model.zero_grad()
                task_loss.backward(retain_graph=True)
                grad_stats[name].append(get_grad_norm(model))
                optimizer_model.zero_grad()

        optimizer_model.zero_grad()
        optimizer_node_dec.zero_grad()
        optimizer_projector.zero_grad()
        if edge_decoder is not None:
            optimizer_edge_dec.zero_grad()
        if lattice_decoder is not None:
            optimizer_lattice_dec.zero_grad()
        if noise_decoder is not None:
            optimizer_noise_dec.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer_model.step()
        if args.task == 'mask' or args.task == 'union':
            optimizer_node_dec.step()
        if args.task == 'contrastive' or args.task == 'union':
            optimizer_projector.step()
        if edge_decoder is not None:
            optimizer_edge_dec.step()
        if noise_decoder is not None:
            optimizer_noise_dec.step()
        if lattice_decoder is not None:
            optimizer_lattice_dec.step()
        if step % 200 == 0:
            print('train Step {0}/{1}\n'
                  'Mask Loss {mask_loss.val:.4f} ({mask_loss.avg:.4f})\t'
                  'Mask Loss2 {mask_loss2.val:.4f} ({mask_loss2.avg:.4f})\t'
                  'Denoise Loss {denoise_loss.val:.4f} ({denoise_loss.avg:.4f})\t'
                  'Denoise Loss2 {denoise_loss2.val:.4f} ({denoise_loss2.avg:.4f})\t'
                  'Contrast Loss {contrast_loss.val:.4f} ({contrast_loss.avg:.4f})\n'
                  'Distance MAE {distance_mae.val:.3f} ({distance_mae.avg:.3f})\t'
                  'Mask Accu {type_accu.val:.3f} ({type_accu.avg:.3f})'.format(
                step, len(train_loader),
                mask_loss=mask_loss_accum,mask_loss2=mask2_loss_accum,
                denoise_loss=denoise_loss_accum,denoise_loss2=latt_loss_accum,
                contrast_loss=contrast_loss_accum,
                distance_mae=distance_mae_accum,
                type_accu=type_accu_accum)
                )

    return total_loss_accum.avg, mask_loss_accum.avg, denoise_loss_accum.avg, contrast_loss_accum.avg, mask2_loss_accum.avg, latt_loss_accum.avg, grad_stats

def validate(val_loader, model_list, criterion_list, normalizer_list, hparams, device):
    model, node_decoder, edge_decoder, projector, lattice_decoder, noise_decoder = model_list
    criterion_class, criterion_mse, criterion_contrast = criterion_list
    edge_normalizer, latt_normalizer = normalizer_list

    # switch to eval mode
    model.eval()
    node_decoder.eval()
    projector.eval()
    if edge_decoder is not None:
        edge_decoder.eval()
    if lattice_decoder is not None:
        lattice_decoder.eval()

    # Accumulate batch metrics.
    total_loss_accum = AverageMeter()
    mask_loss_accum = AverageMeter()
    denoise_loss_accum = AverageMeter()
    contrast_loss_accum = AverageMeter()
    mask2_loss_accum = AverageMeter()
    latt_loss_accum = AverageMeter()
    type_accu_accum = AverageMeter()
    distance_mae_accum = AverageMeter()

    for step, batch in enumerate(val_loader):

        mask_batch, pert_batch, view1_batch, view2_batch = batch
        mask_loss = torch.tensor(0.).to(device)
        mask2_loss = torch.tensor(0.).to(device)
        denoise_loss = torch.tensor(0.).to(device)
        latt_loss = torch.tensor(0.).to(device)
        contrast_loss = torch.tensor(0.).to(device)
        mask_batch = mask_batch.to(device)
        pert_batch = pert_batch.to(device)
        view1_batch = view1_batch.to(device)
        view2_batch = view2_batch.to(device)
        with torch.set_grad_enabled(True):
            if args.task == 'mask' or args.task == 'union':
                # Reconstruct masked atom types.
                mask_batch = mask_batch.to(device)
                if args.gnn == 'Painn':
                    mask_x, mask_x_vector= model(mask_batch)
                else:
                    mask_x, mask_edge_attr, *_ = model(mask_batch)
                if hparams['remask']:
                    pred_node = node_decoder(mask_x, mask_batch.edge_index, torch.norm(mask_batch.displacement, p=2, dim=-1), mask_atom_indices = mask_batch.masked_atom_indices)
                else:
                    pred_node = node_decoder(mask_x, mask_batch.edge_index, mask_edge_attr)

                if args.reduction == 'mean':
                    mask_loss = criterion_class(pred_node[mask_batch.masked_atom_indices], mask_batch.masked_atom_labels)
                elif args.reduction == 'none':
                    mask_loss_per_node = criterion_class(pred_node[mask_batch.masked_atom_indices], mask_batch.masked_atom_labels)
                    mask_loss = torch.mean(scatter(mask_loss_per_node, mask_batch.batch[mask_batch.masked_atom_indices], dim=0, reduce='mean'), dim=0)
                type_accu_accum.update(accuracy_torch(pred_node[mask_batch.masked_atom_indices], mask_batch.masked_atom_labels).item())
                mask_loss_accum.update(mask_loss.item())

                # Reconstruct masked edge distances.
                if edge_decoder is not None:
                    if args.gnn == 'Painn':
                        pred_edge = edge_decoder(mask_x[mask_batch.deleted_edge_index[0]] + mask_x[mask_batch.deleted_edge_index[1]],
                                                 mask_x_vector[mask_batch.deleted_edge_index[0]] - mask_x_vector[mask_batch.deleted_edge_index[1]])
                        mask2_loss = criterion_mse(pred_edge, mask_batch.deleted_displacement / edge_normalizer.mean)
                    else:
                        pred_edge = edge_decoder(mask_x[mask_batch.deleted_edge_index[0]] + mask_x[mask_batch.deleted_edge_index[1]])
                        if args.reduction == 'mean':
                            mask2_loss = criterion_mse(pred_edge, edge_normalizer.norm(mask_batch.deleted_distance).view(-1,1))
                        elif args.reduction == 'none':
                            edge_batch = mask_batch.batch[mask_batch.edge_index[0]]
                            mask2_loss_per_edge = criterion_mse(pred_edge, edge_normalizer.norm(mask_batch.deleted_distance).view(-1,1))
                            mask2_loss = torch.mean(scatter(mask2_loss_per_edge, edge_batch[mask_batch.masked_edge_indices], dim=0,reduce='mean'))
                    mask2_loss_accum.update(mask2_loss.item())

            if args.task == 'denoise' or args.task == 'union':
                # Reconstruct distances from perturbed structures.
                pert_batch = pert_batch.to(device)
                if args.gnn == 'Painn':
                    pert_x, pert_x_vector = model(pert_batch)
                    pred_noise = noise_decoder(pert_x, pert_x_vector)
                    pert_graph_fea = scatter(pert_x, pert_batch.batch, dim=0, reduce='mean')
                else:
                    pert_batch.pos.requires_grad_(True)
                    _, _, pert_graph_fea, pred_noise, *_ = model(pert_batch, True)
                if args.reduction == 'mean':
                    denoise_loss = criterion_mse(pred_noise, pert_batch.noise_labels / hparams['sigma'])
                elif args.reduction == 'none':
                    denoise_loss_per_node = criterion_mse(pred_noise, pert_batch.noise_labels)
                    denoise_loss = torch.mean(scatter(denoise_loss_per_node, pert_batch.batch, dim=0, reduce='mean'))
                denoise_loss_accum.update(denoise_loss.item())
                distance_mae_accum.update(mae(pred_noise * hparams['sigma'], pert_batch.noise_labels).item())
                # Reconstruct lattice parameters.
                if lattice_decoder is not None:
                    pred_lattice = lattice_decoder(pert_graph_fea)
                    pert_latt_labels = torch.cat([latt_normalizer.norm(pert_batch.lattice_lengths_labels),pert_batch.lattice_angles_labels], dim=-1)
                    latt_loss = criterion_mse(pred_lattice, pert_latt_labels).mean()
                    latt_loss_accum.update(latt_loss.item())
                    del pert_latt_labels
                else:
                    latt_loss = torch.tensor(0.0).to(device)

            if args.task == 'contrastive' or args.task == 'union':

                view1_batch = view1_batch.to(device)
                view2_batch = view2_batch.to(device)
                if args.gnn == 'Painn':
                    view1_x, _ = model(view1_batch)
                    view2_x, _ = model(view2_batch)
                    view1_graph_fea = scatter(view1_x, view1_batch.batch, dim=0, reduce='mean')
                    view2_graph_fea = scatter(view2_x, view2_batch.batch, dim=0, reduce='mean')
                else:
                    _, _, view1_graph_fea, *_ = model(view1_batch)
                    _, _, view2_graph_fea, *_ = model(view2_batch)
                z1 = projector(view1_graph_fea)
                z2 = projector(view2_graph_fea)
                contrast_loss = criterion_contrast(z1, z2)
                contrast_loss_accum.update(contrast_loss.item())

            losses = {
                'mask': mask_loss,
                'mask2': mask2_loss,
                'denoise': denoise_loss,
                'lattice': latt_loss,
                'contrast': contrast_loss,
            }
            loss = combine_task_losses(
                losses, hparams, args.task,
                include_masked_edges=edge_decoder is not None,
                include_lattice=lattice_decoder is not None,
            )

        total_loss_accum.update(loss.item())
        if step % 10 == 0:
            print('validate Step {0}/{1}\n'
                  'Mask Loss {mask_loss.val:.4f} ({mask_loss.avg:.4f})\t'
                  'Mask Loss2 {mask_loss2.val:.4f} ({mask_loss2.avg:.4f})\t'
                  'Denoise Loss {denoise_loss.val:.4f} ({denoise_loss.avg:.4f})\t'
                  'Denoise Loss2 {denoise_loss2.val:.4f} ({denoise_loss2.avg:.4f})\t'
                  'Contrast Loss {contrast_loss.val:.4f} ({contrast_loss.avg:.4f})\n'
                  'Distance MAE {distance_mae.val:.3f} ({distance_mae.avg:.3f})\t'
                  'Mask Accu {type_accu.val:.3f} ({type_accu.avg:.3f})'.format(
                step, len(val_loader),
                mask_loss=mask_loss_accum,mask_loss2=mask2_loss_accum,
                denoise_loss=denoise_loss_accum,denoise_loss2=latt_loss_accum,
                contrast_loss=contrast_loss_accum,
                distance_mae=distance_mae_accum,
                type_accu=type_accu_accum)
                )

    return total_loss_accum.avg, mask_loss_accum.avg, denoise_loss_accum.avg, contrast_loss_accum.avg, mask2_loss_accum.avg, latt_loss_accum.avg

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

def get_grad_norm(model):
    """Return the total L2 norm of model parameter gradients."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def write_gradient_pilot_summary(grad_stats, save_dir, seed):
    """Write average FP32 gradient norms and inverse-gradient weights."""
    summary = {}
    inverse_means = {}
    for name, values in grad_stats.items():
        if not values:
            continue
        values = np.asarray(values, dtype=float)
        mean = float(values.mean())
        if not np.isfinite(mean) or mean <= 0:
            raise ValueError(f'Invalid mean gradient norm for {name}: {mean}')
        summary[name] = {
            'num_batches': int(values.size),
            'mean_gradient_norm': mean,
            'std_gradient_norm': float(values.std()),
        }
        inverse_means[name] = 1.0 / mean

    if not summary:
        raise ValueError('The gradient pilot did not collect any gradients')

    mean_inverse = float(np.mean(list(inverse_means.values())))
    suggested_weights = {
        f'{name}_weight': value / mean_inverse
        for name, value in inverse_means.items()
    }
    result = {
        'pilot_epochs': 2,
        'seed': seed,
        'pilot_loss_weights': 'all active losses set to 1.0',
        'gradient_norms': summary,
        'suggested_inverse_gradient_weights': suggested_weights,
        'normalization': 'weights are normalized to have arithmetic mean 1',
    }
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, 'gradient_pilot_summary.json')
    with open(output_path, 'w', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(f'Gradient pilot summary written to {output_path}')
    print('Restart pre-training from a newly initialized model after copying '
          'the selected fixed weights into the configuration file.')

if __name__ == '__main__':
    main()
