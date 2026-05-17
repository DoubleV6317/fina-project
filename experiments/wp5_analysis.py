import argparse
import json
import os

import numpy as np
import pandas as pd

from analysis.metrics import (
    bootstrap_macro_auc_diff, filter_ablation, macro_auc_5,
    per_sample_error_analysis, threshold_sweep,
)


def _load_npy(path):
    """Load a NumPy array if the file exists, else return None."""
    return np.load(path) if os.path.exists(path) else None


def main(args):
    """Run the zero-training WP5 post-hoc analyses on previously saved predictions."""
    os.makedirs(args.output_dir, exist_ok=True)

    filter_df = pd.read_csv(args.filter_analysis_csv)
    ablation = filter_ablation(filter_df)
    sweep = threshold_sweep(filter_df, thresholds=tuple(args.thresholds))
    with open(os.path.join(args.output_dir, 'A_filter_ablation.json'), 'w') as f:
        json.dump(ablation, f, indent=2, default=str)
    with open(os.path.join(args.output_dir, 'B_threshold_sweep.json'), 'w') as f:
        json.dump(sweep, f, indent=2, default=str)

    gt = _load_npy(args.gt)
    pred_wp1 = _load_npy(args.pred_wp1)
    pred_wp3 = _load_npy(args.pred_wp3)
    pred_wp4 = _load_npy(args.pred_wp4)

    if gt is not None and pred_wp1 is not None and pred_wp4 is not None:
        pred_dict = {'WP1': pred_wp1}
        if pred_wp3 is not None:
            pred_dict['WP3'] = pred_wp3
        pred_dict['WP4'] = pred_wp4
        ci = bootstrap_macro_auc_diff(gt, pred_dict, n_bootstrap=args.n_bootstrap, seed=args.seed)
        ci['point_estimates'] = {k: macro_auc_5(gt, v) for k, v in pred_dict.items()}
        with open(os.path.join(args.output_dir, 'C_bootstrap_ci.json'), 'w') as f:
            json.dump(ci, f, indent=2)

        errors = per_sample_error_analysis(gt, pred_wp1, pred_wp4)
        with open(os.path.join(args.output_dir, 'D_error_analysis.json'), 'w') as f:
            json.dump(errors, f, indent=2)

    if args.seed_results_a and args.seed_results_b:
        with open(args.seed_results_a) as f:
            seed_a = json.load(f)
        with open(args.seed_results_b) as f:
            seed_b = json.load(f)
        comparison = {
            'seed_a': seed_a,
            'seed_b': seed_b,
            'per_class_abs_diff': {
                k: abs(seed_a['per_class_auc'].get(k, 0.0) - seed_b['per_class_auc'].get(k, 0.0))
                for k in seed_a.get('per_class_auc', {})
            },
        }
        with open(os.path.join(args.output_dir, 'E_seed_comparison.json'), 'w') as f:
            json.dump(comparison, f, indent=2)

    print(f'WP5 analyses written to {args.output_dir}')


def parse_args():
    """Parse CLI arguments for the WP5 analysis driver."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter_analysis_csv', required=True)
    parser.add_argument('--gt')
    parser.add_argument('--pred_wp1')
    parser.add_argument('--pred_wp3')
    parser.add_argument('--pred_wp4')
    parser.add_argument('--seed_results_a')
    parser.add_argument('--seed_results_b')
    parser.add_argument('--output_dir', default='outputs/wp5_analysis')
    parser.add_argument('--n_bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--thresholds', type=float, nargs='+',
                        default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
