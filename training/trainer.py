import random
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from config import CHEXPERT_CLASSES, COMPETITION_5


def set_seed(seed):
    """Seed Python, NumPy and PyTorch RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, log_interval=50):
    """Train the model for a single epoch and return the mean batch loss."""
    model.train()
    total_loss = 0.0
    start = time.time()
    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (batch_idx + 1) % log_interval == 0:
            elapsed = time.time() - start
            print(f'Epoch {epoch} | Batch {batch_idx + 1}/{len(loader)} '
                  f'| Loss {loss.item():.4f} | Elapsed {elapsed:.1f}s')
    return total_loss / max(len(loader), 1)


def evaluate(model, loader, device):
    """Run inference on a loader and return predictions, labels, per-class AUC and macro AUC."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            preds = model(inputs).cpu()
            all_preds.append(preds)
            all_labels.append(labels)
    preds = torch.cat(all_preds).numpy()
    gt = torch.cat(all_labels).numpy()

    per_class_auc = {}
    for i, cls in enumerate(CHEXPERT_CLASSES):
        try:
            per_class_auc[cls] = float(roc_auc_score(gt[:, i], preds[:, i]))
        except ValueError:
            per_class_auc[cls] = None

    comp_aucs = [per_class_auc[c] for c in COMPETITION_5 if per_class_auc[c] is not None]
    macro_auc = float(np.mean(comp_aucs)) if comp_aucs else 0.0
    return preds, gt, per_class_auc, macro_auc


def compute_val_loss(model, loader, criterion, device):
    """Compute the mean BCE loss over a validation loader."""
    model.eval()
    total = 0.0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            total += criterion(model(inputs), labels).item()
    return total / max(len(loader), 1)
