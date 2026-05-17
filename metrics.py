import numpy as np
from sklearn.metrics import roc_auc_score

from config import CHEXPERT_CLASSES, COMPETITION_5


def _competition_indices():
    """Return CheXpert column indices for the 5 competition classes."""
    return [CHEXPERT_CLASSES.index(c) for c in COMPETITION_5]


def macro_auc_5(gt, pred):
    """Macro AUC averaged over the 5 CheXpert competition classes."""
    aucs = []
    for idx in _competition_indices():
        try:
            aucs.append(roc_auc_score(gt[:, idx], pred[:, idx]))
        except ValueError:
            continue
    return float(np.mean(aucs)) if aucs else 0.0


def bootstrap_macro_auc_diff(gt, pred_dict, n_bootstrap=2000, seed=0, alpha=0.05):
    """Paired bootstrap 95% CI for macro AUC differences between models."""
    rng = np.random.default_rng(seed)
    n = len(gt)
    indices = _competition_indices()
    keys = list(pred_dict.keys())
    boot = {k: np.zeros(n_bootstrap, dtype=np.float64) for k in keys}

    for b in range(n_bootstrap):
        sample = rng.integers(0, n, size=n)
        gt_b = gt[sample]
        for k, pred in pred_dict.items():
            pred_b = pred[sample]
            aucs = []
            for ci in indices:
                try:
                    aucs.append(roc_auc_score(gt_b[:, ci], pred_b[:, ci]))
                except ValueError:
                    continue
            boot[k][b] = float(np.mean(aucs)) if aucs else 0.0

    results = {}
    for i, k1 in enumerate(keys):
        for k2 in keys[i + 1:]:
            diffs = boot[k1] - boot[k2]
            lo = float(np.percentile(diffs, 100 * alpha / 2))
            hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
            results[f'{k1}_vs_{k2}'] = {
                'mean_diff': float(diffs.mean()),
                'ci_low': lo,
                'ci_high': hi,
                'significant': (lo > 0) or (hi < 0),
            }
    return results


def per_sample_error_analysis(gt, pred_base, pred_new):
    """Per-class summary of how a new model shifts predictions compared to a baseline."""
    results = {}
    for idx in _competition_indices():
        cls = CHEXPERT_CLASSES[idx]
        pos = gt[:, idx] == 1
        neg = gt[:, idx] == 0
        diff = pred_new[:, idx] - pred_base[:, idx]
        results[cls] = {
            'helped_pos': int((diff[pos] > 0).sum()),
            'hurt_pos': int((diff[pos] < 0).sum()),
            'helped_neg': int((diff[neg] < 0).sum()),
            'hurt_neg': int((diff[neg] > 0).sum()),
            'mean_prob_shift_pos': float(diff[pos].mean()) if pos.sum() else 0.0,
            'mean_prob_shift_neg': float(diff[neg].mean()) if neg.sum() else 0.0,
        }
    return results


def filter_ablation(filter_df):
    """Count kept samples under the four filter variants used in the WP5 analysis."""
    label = filter_df['label_consistent']
    conf = filter_df['confidence_pass']
    variants = {
        'both (current)': label & conf,
        'label_only': label,
        'confidence_only': conf,
        'neither (no filter)': [True] * len(filter_df),
    }
    summary = {}
    for name, mask in variants.items():
        subset = filter_df[mask]
        summary[name] = {
            'total_kept': int(len(subset)),
            'mean_confidence': float(subset['confidence_on_intended'].mean()) if len(subset) else 0.0,
            'median_confidence': float(subset['confidence_on_intended'].median()) if len(subset) else 0.0,
            'per_class_kept': subset['intended_class'].value_counts().to_dict(),
        }
    return summary


def threshold_sweep(filter_df, thresholds=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)):
    """Sweep the confidence threshold and report kept-sample statistics."""
    summary = {}
    for tau in thresholds:
        mask = filter_df['label_consistent'] & (filter_df['confidence_on_intended'] > tau)
        kept = filter_df[mask]
        counts = kept['intended_class'].value_counts()
        total = int(len(kept))
        summary[str(tau)] = {
            'total_kept': total,
            'mean_confidence': float(kept['confidence_on_intended'].mean()) if total else 0.0,
            'per_class_kept': counts.to_dict(),
            'max_class_share': float(counts.max() / total) if total else 0.0,
            'empty_classes': int((counts == 0).sum()),
        }
    return summary
