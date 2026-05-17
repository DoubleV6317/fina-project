import ast

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import CHEXPERT_CLASSES, CXR_IRGEN_CLASSES, FILTER_THRESHOLD


def score_synthetic_images(model, dataset, device, batch_size=32, num_workers=4):
    """Run the WP1 scorer over a synthetic dataset and return a (N, 14) prediction array."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    all_preds = []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device, non_blocking=True)
            preds = model(inputs).cpu().numpy()
            all_preds.append(preds)
    return np.vstack(all_preds)


def apply_filter(metadata_df, predictions, threshold=FILTER_THRESHOLD):
    """Apply the quality-aware filter (label consistency + confidence > threshold)."""
    rows = []
    for i, row in metadata_df.reset_index(drop=True).iterrows():
        irgen_labels = ast.literal_eval(row['labels'])
        intended_irgen_idx = int(np.argmax(irgen_labels))
        intended_name = CXR_IRGEN_CLASSES[intended_irgen_idx]
        intended_chexpert_idx = next(
            (j for j, c in enumerate(CHEXPERT_CLASSES) if c.lower() == intended_name),
            None,
        )

        pred = predictions[i]
        top1_idx = int(np.argmax(pred))
        confidence = float(pred[intended_chexpert_idx]) if intended_chexpert_idx is not None else 0.0
        label_consistent = (top1_idx == intended_chexpert_idx)
        confidence_pass = (confidence > threshold)

        rows.append({
            'filename': row['filename'],
            'intended_class': intended_name,
            'intended_chexpert_idx': intended_chexpert_idx,
            'top1_idx': top1_idx,
            'confidence_on_intended': confidence,
            'label_consistent': bool(label_consistent),
            'confidence_pass': bool(confidence_pass),
            'kept': bool(label_consistent and confidence_pass),
        })
    return pd.DataFrame(rows)


def split_kept_rejected(filter_df):
    """Split a filter analysis dataframe into kept and rejected views."""
    kept = filter_df[filter_df['kept']].reset_index(drop=True)
    rejected = filter_df[~filter_df['kept']].reset_index(drop=True)
    return kept, rejected
