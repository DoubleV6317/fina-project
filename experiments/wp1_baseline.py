import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE, LEARNING_RATE, NUM_CLASSES, TR_MAX_EPOCH, WEIGHT_DECAY,
)
from data.datasets import CheXpertDataset
from models.densenet import DenseNet121
from training.trainer import compute_val_loss, evaluate, set_seed, train_one_epoch
from training.transforms import get_train_transforms, get_val_transforms


def main(args):
    """Train the WP1 real-data-only DenseNet121 baseline and save artefacts."""
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds = CheXpertDataset(args.train_csv, args.data_root, transform=get_train_transforms())
    val_ds = CheXpertDataset(args.val_csv, args.data_root, transform=get_val_transforms())
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = DenseNet121(NUM_CLASSES).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE,
        betas=(0.9, 0.999), eps=1e-8, weight_decay=WEIGHT_DECAY,
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, 'wp1_baseline_densenet121.pth.tar')

    best_val_loss = float('inf')
    for epoch in range(1, TR_MAX_EPOCH + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss = compute_val_loss(model, val_loader, criterion, device)
        print(f'Epoch {epoch} | train {train_loss:.4f} | val {val_loss:.4f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'epoch': epoch, 'state_dict': model.state_dict()}, ckpt_path)

    model.load_state_dict(torch.load(ckpt_path, map_location=device)['state_dict'])
    preds, gt, per_class_auc, macro_auc = evaluate(model, val_loader, device)
    np.save(os.path.join(args.output_dir, 'wp1_val_pred.npy'), preds)
    np.save(os.path.join(args.output_dir, 'wp1_val_gt.npy'), gt)
    results = {
        'macro_auc_5': macro_auc,
        'per_class_auc': {k: v for k, v in per_class_auc.items() if v is not None},
        'seed': args.seed,
    }
    with open(os.path.join(args.output_dir, 'wp1_baseline_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f'WP1 Macro AUC (5 classes): {macro_auc:.4f}')


def parse_args():
    """Parse CLI arguments for the WP1 baseline experiment."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--train_csv', required=True)
    parser.add_argument('--val_csv', required=True)
    parser.add_argument('--checkpoint_dir', default='outputs/checkpoints')
    parser.add_argument('--output_dir', default='outputs/results')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
