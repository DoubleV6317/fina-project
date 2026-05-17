# CXR Synthetic Augmentation

A study of whether **CXR-IRGen synthetic chest X-rays** can improve a DenseNet121 multi-label disease classifier trained on **CheXpert**, and whether a **quality-aware filter** that uses the baseline classifier as a critic can rescue gains lost to noisy synthetic data.

The code is a refactor of five Colab notebooks (`cheXpert_final`, `wp3_naive_augmentation`, `wp4_filtered_augmentation`, `wp5_analysis`, `wp6_experiments`) into a clean, reusable Python package.

---

## Repository layout

```
cxr_synthetic_augmentation/
├── config.py                # Global constants, class names, hyperparameters
├── data/
│   ├── datasets.py          # CheXpertDataset, SyntheticDataset, MixedDataset
│   └── label_mapping.py     # CXR-IRGen → CheXpert label remapping
├── models/
│   └── densenet.py          # DenseNet121 classifier + CAM HeatmapGenerator
├── training/
│   ├── transforms.py        # Train / val / heavy augmentation pipelines
│   └── trainer.py           # set_seed, train_one_epoch, evaluate
├── filtering/
│   └── quality_filter.py    # WP4 quality-aware filter (label + confidence)
├── analysis/
│   └── metrics.py           # Macro AUC, bootstrap CI, error analysis, sweeps
└── experiments/
    ├── wp1_baseline.py      # WP1: real-only baseline
    ├── wp3_naive_aug.py     # WP3: real + all synthetic (no filter)
    ├── wp4_filtered_aug.py  # WP4: real + filtered synthetic
    ├── wp5_analysis.py      # WP5: post-hoc analyses (no training)
    └── wp6_experiments.py   # WP6: cells F (seed), G (heavy aug), H (τ sweep)
```

---

## Experimental design

| WP  | Training data                                  | Purpose                                       |
|-----|------------------------------------------------|-----------------------------------------------|
| WP1 | CheXpert real only (223,414 images)            | Baseline classifier and synthetic-image scorer |
| WP3 | Real + **all** CXR-IRGen synthetic (1,000)     | Naive synthetic augmentation                  |
| WP4 | Real + **filtered** synthetic (≈249 @ τ=0.7)   | Quality-aware augmentation (core contribution) |
| WP5 | (no training)                                  | Ablation, threshold sweep, bootstrap CI, error analysis |
| WP6 | F: WP3 with second seed; G: heavy traditional aug; H: WP4 at τ∈{0.5, 0.9} | Robustness / sensitivity checks |

All training runs use the same recipe: **DenseNet121 (ImageNet)**, **BCELoss**, **Adam (lr=1e-4, wd=1e-5)**, **3 epochs**, **batch size 64**, **224×224**.

Performance is reported as **macro AUC over the 5 CheXpert competition classes**: Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion.

---

## Datasets

- **CheXpert-v1.0-small** ([Stanford ML Group](https://stanfordmlgroup.github.io/competitions/chexpert/)) — 223,414 train / 234 valid images, 14 labels. Uncertainty handled with the **U-Ones** policy (`−1 → 1`, `NaN → 0`).
- **CXR-IRGen synthetic set** — 1,000 PNGs generated from a CXR-IRGen text-conditioned diffusion model, with a `metadata.csv` that stores each image's 14-dim intended-label vector.

Class ordering differs between the two sources; `data/label_mapping.py` re-maps CXR-IRGen labels into the CheXpert column order by name.

---

## Installation

```bash
git clone <your-repo-url>
cd cxr_synthetic_augmentation
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA-enabled GPU is strongly recommended.

---

## Running the experiments

All entry points read `config.py` as a top-level module, so run them from the **project root**:

```bash
cd cxr_synthetic_augmentation
export PYTHONPATH=.
```

### WP1 — baseline

```bash
python experiments/wp1_baseline.py \
  --data_root /path/to/data \
  --train_csv /path/to/CheXpert-v1.0-small/train.csv \
  --val_csv   /path/to/CheXpert-v1.0-small/valid.csv \
  --checkpoint_dir outputs/checkpoints \
  --output_dir     outputs/results
```

Artefacts: `wp1_baseline_densenet121.pth.tar`, `wp1_val_pred.npy`, `wp1_val_gt.npy`, `wp1_baseline_results.json`.

### WP3 — naive augmentation

```bash
python experiments/wp3_naive_aug.py \
  --data_root /path/to/data \
  --train_csv /path/to/CheXpert-v1.0-small/train.csv \
  --val_csv   /path/to/CheXpert-v1.0-small/valid.csv \
  --synthetic_dir /path/to/synthetic_output \
  --synthetic_csv /path/to/synthetic_output/metadata.csv
```

### WP4 — filtered augmentation

```bash
python experiments/wp4_filtered_aug.py \
  --data_root /path/to/data \
  --train_csv /path/to/CheXpert-v1.0-small/train.csv \
  --val_csv   /path/to/CheXpert-v1.0-small/valid.csv \
  --synthetic_dir /path/to/synthetic_output \
  --synthetic_csv /path/to/synthetic_output/metadata.csv \
  --wp1_checkpoint outputs/checkpoints/wp1_baseline_densenet121.pth.tar \
  --threshold 0.7
```

Saves the filter report (`filter_analysis.csv`, `kept_synthetic.csv`, `rejected_synthetic.csv`, `synthetic_predictions.npy`) alongside the trained model and validation metrics.

### WP5 — post-hoc analysis

```bash
python experiments/wp5_analysis.py \
  --filter_analysis_csv outputs/wp4_filter_outputs/filter_analysis.csv \
  --gt       outputs/results/wp1_val_gt.npy \
  --pred_wp1 outputs/results/wp1_val_pred.npy \
  --pred_wp3 outputs/results/wp3_val_pred.npy \
  --pred_wp4 outputs/results/wp4_val_pred.npy \
  --output_dir outputs/wp5_analysis
```

Writes `A_filter_ablation.json`, `B_threshold_sweep.json`, `C_bootstrap_ci.json`, `D_error_analysis.json`, and (if seed results are supplied) `E_seed_comparison.json`.

### WP6 — additional experiments

```bash
python experiments/wp6_experiments.py \
  --cells F G H \
  --data_root /path/to/data \
  --train_csv /path/to/CheXpert-v1.0-small/train.csv \
  --val_csv   /path/to/CheXpert-v1.0-small/valid.csv \
  --synthetic_dir /path/to/synthetic_output \
  --synthetic_csv /path/to/synthetic_output/metadata.csv \
  --filter_analysis_csv outputs/wp4_filter_outputs/filter_analysis.csv
```

Each cell can be run individually with `--cells F`, `--cells G`, or `--cells H`. Cell H also takes `--cell_h_thresholds 0.5 0.9` to sweep alternative thresholds.

---

## Results (notebook reference)

| Experiment                        | Macro AUC (5) | Δ vs WP1 |
|-----------------------------------|---------------|----------|
| WP1: real only                    | 0.8782        | 0.0000   |
| WP3: real + naive synthetic (s=42)| 0.8650        | −0.0132  |
| WP4: real + filtered synthetic (s=42)  | 0.8526   | −0.0256  |
| WP4: real + filtered synthetic (s=123) | 0.8742   | −0.0040  |

Filter retention at τ=0.7 was **249/1000 (24.9%)**, dominated by Pleural Effusion (176) and Cardiomegaly (62); Consolidation contributed 0.

The bootstrap CI in WP5 indicates the differences between the three runs are **not statistically significant** for this small validation set — the quality filter recovers most of the loss from naive augmentation but does not exceed the real-only baseline.

---

## Module reference

- `data.CheXpertDataset(csv_path, data_root, transform)` — real CheXpert images with U-Ones labels.
- `data.SyntheticDataset(synthetic_dir, metadata_csv, transform, allowed_filenames=None)` — CXR-IRGen images, label-remapped to CheXpert order; pass `allowed_filenames` to filter.
- `data.MixedDataset(real, synthetic)` — concatenation used for joint training.
- `models.DenseNet121(out_size)` — ImageNet-pretrained backbone with sigmoid multi-label head.
- `models.HeatmapGenerator(model)` — class activation maps for qualitative inspection.
- `training.train_one_epoch / evaluate / set_seed` — training utilities.
- `filtering.score_synthetic_images / apply_filter` — score synthetic samples with the WP1 model and apply the label-consistency + confidence filter.
- `analysis.macro_auc_5 / bootstrap_macro_auc_diff / per_sample_error_analysis / filter_ablation / threshold_sweep` — evaluation and analysis helpers.

---

## Reproducing the Colab pipeline

1. Train WP1 → produces the scorer checkpoint.
2. Run WP3 to record the naive-augmentation baseline.
3. Run WP4 to score, filter, and re-train.
4. Run WP5 over the four prediction arrays + filter CSV.
5. Run WP6 cells F/G/H for additional comparisons.

Seeds, thresholds, and paths are exposed as CLI flags — no source edits required.

---

## License

The code is released for research and educational use. Underlying datasets (CheXpert, CXR-IRGen outputs) keep their own licenses and access requirements.
