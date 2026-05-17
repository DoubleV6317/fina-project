import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE, FILTER_THRESHOLD, LEARNING_RATE, NUM_CLASSES, TR_MAX_EPOCH, WEIGHT_DECAY,
)
from data.datasets import CheXpertDataset, MixedDataset, SyntheticDataset
from filtering.quality_filter import apply_filter, score_synthetic_images, split_kept_rejected
from models.densenet import DenseNet121
from training.trainer import evaluate, set_seed, train_one_epoch
from training.transforms import get_train_transforms, get_val_transforms


def load_scorer(checkpoint_path, device):
    """Restore the WP1 baseline model used to score synthetic image quality."""
    scorer = DenseNet121(NUM_CLASSES).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    scorer.load_state_dict(state['state_dict'])
    scorer.eval()
    return scorer


def main(args):
    """Run the WP4 quality-aware filtering and downstream training pipeline."""
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    val_tf = get_val_transforms()
    train_tf = get_train_transforms()

    scoring_ds = SyntheticDataset(args.synthetic_dir, args.synthetic_csv, transform=val_tf)
    scorer = load_scorer(args.wp1_checkpoint, device)
    predictions = score_synthetic_images(scorer, scoring_ds, device, batch_size=32,
                                         num_workers=args.num_workers)

    metadata_df = pd.read_csv(args.synthetic_csv)
    filter_df = apply_filter(metadata_df, predictions, threshold=args.threshold)
    kept_df, rejected_df = split_kept_rejected(filter_df)

    os.makedirs(args.filter_output_dir, exist_ok=True)
    filter_df.to_csv(os.path.join(args.filter_output_dir, 'filter_analysis.csv'), index=False)
    kept_df.to_csv(os.path.join(args.filter_output_dir, 'kept_synthetic.csv'), index=False)
    rejected_df.to_csv(os.path.join(args.filter_output_dir, 'rejected_synthetic.csv'), index=False)
    np.save(os.path.join(args.filter_output_dir, 'synthetic_predictions.npy'), predictions)

    print(f'Filter @ tau={args.threshold}: kept {len(kept_df)}/{len(filter_df)} '
          f'({100 * len(kept_df) / max(len(filter_df), 1):.1f}%)')

    del scorer
    torch.cuda.empty_cache()

    real_train = CheXpertDataset(args.train_csv, args.data_root, transform=train_tf)
    synthetic_train = SyntheticDataset(
        args.synthetic_dir, args.synthetic_csv, transform=train_tf,
        allowed_filenames=kept_df['filename'].tolist(),
    )
    mixed_train = MixedDataset(real_train, synthetic_train)
    val_ds = CheXpertDataset(args.val_csv, args.data_root, transform=val_tf)

    train_loader = DataLoader(mixed_train, batch_size=BATCH_SIZE, shuffle=True,
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

    for epoch in range(1, TR_MAX_EPOCH + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        print(f'Epoch {epoch} | train loss {train_loss:.4f}')
        torch.save({'epoch': epoch, 'state_dict': model.state_dict()},
                   os.path.join(args.checkpoint_dir, f'wp4_filtered_aug_epoch{epoch}.pth.tar'))

    preds, gt, per_class_auc, macro_auc = evaluate(model, val_loader, device)
    np.save(os.path.join(args.output_dir, 'wp4_val_pred.npy'), preds)
    np.save(os.path.join(args.output_dir, 'wp4_val_gt.npy'), gt)
    results = {
        'macro_auc_5': macro_auc,
        'per_class_auc': {k: v for k, v in per_class_auc.items() if v is not None},
        'seed': args.seed,
        'threshold': args.threshold,
        'kept_synthetic': int(len(kept_df)),
        'rejected_synthetic': int(len(rejected_df)),
    }
    with open(os.path.join(args.output_dir, 'wp4_filtered_aug_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f'WP4 Macro AUC (5 classes): {macro_auc:.4f}')


def parse_args():
    """Parse CLI arguments for the WP4 filtered-augmentation experiment."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--train_csv', required=True)
    parser.add_argument('--val_csv', required=True)
    parser.add_argument('--synthetic_dir', required=True)
    parser.add_argument('--synthetic_csv', required=True)
    parser.add_argument('--wp1_checkpoint', required=True)
    parser.add_argument('--threshold', type=float, default=FILTER_THRESHOLD)
    parser.add_argument('--checkpoint_dir', default='outputs/checkpoints')
    parser.add_argument('--output_dir', default='outputs/results')
    parser.add_argument('--filter_output_dir', default='outputs/wp4_filter_outputs')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
