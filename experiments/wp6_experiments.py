import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE, LEARNING_RATE, NUM_CLASSES, TR_MAX_EPOCH, WEIGHT_DECAY,
)
from data.datasets import CheXpertDataset, MixedDataset, SyntheticDataset
from models.densenet import DenseNet121
from training.trainer import evaluate, set_seed, train_one_epoch
from training.transforms import (
    get_heavy_train_transforms, get_train_transforms, get_val_transforms,
)


def _build_optimizer(model):
    """Adam optimiser matching the WP1/WP3/WP4 training recipe."""
    return torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE,
        betas=(0.9, 0.999), eps=1e-8, weight_decay=WEIGHT_DECAY,
    )


def _train_and_evaluate(model, train_loader, val_loader, device, ckpt_prefix, ckpt_dir):
    """Run TR_MAX_EPOCH epochs of training and return validation metrics."""
    criterion = nn.BCELoss()
    optimizer = _build_optimizer(model)
    for epoch in range(1, TR_MAX_EPOCH + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        print(f'[{ckpt_prefix}] Epoch {epoch} | train loss {train_loss:.4f}')
        torch.save({'epoch': epoch, 'state_dict': model.state_dict()},
                   os.path.join(ckpt_dir, f'{ckpt_prefix}_epoch{epoch}.pth.tar'))
    return evaluate(model, val_loader, device)


def _save_results(name, preds, gt, per_class_auc, macro_auc, output_dir, extra=None):
    """Persist predictions and a results JSON for an experiment."""
    pred_dir = os.path.join(output_dir, 'predictions')
    os.makedirs(pred_dir, exist_ok=True)
    np.save(os.path.join(pred_dir, f'{name}_pred.npy'), preds)
    np.save(os.path.join(pred_dir, f'{name}_gt.npy'), gt)
    payload = {
        'macro_auc_5': macro_auc,
        'per_class_auc': {k: v for k, v in per_class_auc.items() if v is not None},
    }
    if extra:
        payload.update(extra)
    with open(os.path.join(output_dir, f'{name}_results.json'), 'w') as f:
        json.dump(payload, f, indent=2)
    print(f'[{name}] Macro AUC (5): {macro_auc:.4f}')


def run_cell_f(args, device):
    """Cell F: naive augmentation re-run with a different seed (stability check)."""
    set_seed(args.cell_f_seed)
    train_tf, val_tf = get_train_transforms(), get_val_transforms()
    real = CheXpertDataset(args.train_csv, args.data_root, transform=train_tf)
    syn = SyntheticDataset(args.synthetic_dir, args.synthetic_csv, transform=train_tf)
    mixed = MixedDataset(real, syn)
    val_ds = CheXpertDataset(args.val_csv, args.data_root, transform=val_tf)

    train_loader = DataLoader(mixed, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = DenseNet121(NUM_CLASSES).to(device)
    preds, gt, per_class_auc, macro_auc = _train_and_evaluate(
        model, train_loader, val_loader, device, 'F_naive_aug_seed2', args.checkpoint_dir,
    )
    _save_results('F_naive_aug_seed2', preds, gt, per_class_auc, macro_auc, args.output_dir,
                  extra={'seed': args.cell_f_seed})
    del model
    torch.cuda.empty_cache()


def run_cell_g(args, device):
    """Cell G: real-only training with heavy traditional augmentation."""
    set_seed(args.cell_g_seed)
    train_tf, val_tf = get_heavy_train_transforms(), get_val_transforms()
    real = CheXpertDataset(args.train_csv, args.data_root, transform=train_tf)
    val_ds = CheXpertDataset(args.val_csv, args.data_root, transform=val_tf)

    train_loader = DataLoader(real, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = DenseNet121(NUM_CLASSES).to(device)
    preds, gt, per_class_auc, macro_auc = _train_and_evaluate(
        model, train_loader, val_loader, device, 'G_traditional_aug', args.checkpoint_dir,
    )
    _save_results('G_traditional_aug', preds, gt, per_class_auc, macro_auc, args.output_dir,
                  extra={'seed': args.cell_g_seed})
    del model
    torch.cuda.empty_cache()


def run_cell_h(args, device):
    """Cell H: filtered augmentation sensitivity sweep across alternative thresholds."""
    filter_df = pd.read_csv(args.filter_analysis_csv)
    train_tf, val_tf = get_train_transforms(), get_val_transforms()
    val_ds = CheXpertDataset(args.val_csv, args.data_root, transform=val_tf)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    for tau in args.cell_h_thresholds:
        set_seed(args.cell_h_seed)
        mask = filter_df['label_consistent'] & (filter_df['confidence_on_intended'] > tau)
        kept = filter_df[mask]['filename'].tolist()

        real = CheXpertDataset(args.train_csv, args.data_root, transform=train_tf)
        syn = SyntheticDataset(args.synthetic_dir, args.synthetic_csv,
                               transform=train_tf, allowed_filenames=kept)
        mixed = MixedDataset(real, syn)
        train_loader = DataLoader(mixed, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)

        tag = f'H_filtered_tau{int(tau * 100):02d}'
        model = DenseNet121(NUM_CLASSES).to(device)
        preds, gt, per_class_auc, macro_auc = _train_and_evaluate(
            model, train_loader, val_loader, device, tag, args.checkpoint_dir,
        )
        _save_results(tag, preds, gt, per_class_auc, macro_auc, args.output_dir,
                      extra={'seed': args.cell_h_seed, 'threshold': tau, 'kept_synthetic': len(kept)})
        del model
        torch.cuda.empty_cache()


def main(args):
    """Run the requested subset of WP6 experiments (F, G, H)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    if 'F' in args.cells:
        run_cell_f(args, device)
    if 'G' in args.cells:
        run_cell_g(args, device)
    if 'H' in args.cells:
        run_cell_h(args, device)


def parse_args():
    """Parse CLI arguments for the WP6 experiment driver."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--cells', nargs='+', default=['F', 'G', 'H'])
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--train_csv', required=True)
    parser.add_argument('--val_csv', required=True)
    parser.add_argument('--synthetic_dir', required=True)
    parser.add_argument('--synthetic_csv', required=True)
    parser.add_argument('--filter_analysis_csv', help='Required when running cell H')
    parser.add_argument('--checkpoint_dir', default='outputs/wp6_experiments/checkpoints')
    parser.add_argument('--output_dir', default='outputs/wp6_experiments')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--cell_f_seed', type=int, default=123)
    parser.add_argument('--cell_g_seed', type=int, default=42)
    parser.add_argument('--cell_h_seed', type=int, default=42)
    parser.add_argument('--cell_h_thresholds', type=float, nargs='+', default=[0.5, 0.9])
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
