# PET/CT Research Code

Official repository of the paper **“Radiogenomic Signatures Derived from Pretreatment
PET/CT Predict Pathological Complete Response to Neoadjuvant Chemoimmunotherapy in
Non-Small Cell Lung Cancer.”**

This repository contains the reproducible training and preprocessing code for a
multimodal PET/CT binary-classification study. Python 3.10 or newer is required.
The public code deliberately excludes label-table contents, images, patient-level
results, and legacy checkpoints. Server data paths are configured in YAML.

Chinese documentation: [README.zh-CN.md](README.zh-CN.md)

## Installation

```bash
python -m venv .venv
python -m pip install -e ".[dev,hpo,two-stage]"
```

Core training installs PyTorch, TorchVision, TorchIO, SimpleITK, and the
scientific Python stack. Optuna and boosted-tree libraries are optional extras.

## Commands

```bash
python -m petct split --config configs/split.yaml
python -m petct train --config configs/train.yaml
python -m petct hpo --config configs/hpo.yaml
python -m petct crossval --config configs/compare.yaml
python -m petct two-stage --config configs/two_stage.yaml
python -m petct preprocess all --config configs/preprocess.yaml
python -m petct evaluate --config configs/evaluate.yaml
```

Splitting and cross-validation default to five stratified folds (approximately
80% training / 20% validation per fold). Run `split` first to generate
`dataset/generated_splits_5fold`, then `crossval` to train all five folds.
The single-run `train`, `hpo`, and `two-stage` configurations use fold 1 by
default. See the configuration guide for five-fold training of the main model.

Any YAML value can be overridden without editing the file:

```bash
python -m petct train --config configs/train.yaml \
  --set training.optimizer.learning_rate=0.0002 --set device=cuda:0
```

The historical scripts remain as thin wrappers. For example,
`python train_petct.py --config configs/train.yaml` calls the same public API.
An HPO best configuration should keep `evaluation.external_after_training: false`;
use `configs/evaluate.yaml` for the separate final checkpoint evaluation.

## Data contract

The `metadata` setting can point directly to an existing CSV or XLSX label table;
no additional metadata file or conversion to CSV is required. Server YAML files
reuse the original radiomics XLSX tables, with `影像组学序列号` as the ID column
and `pCR` as the outcome. Clinical and radiomics feature columns are not used.
Manifests contain one case ID per line. Each configured CT and PET root must have
matching case directories with exactly 64 naturally sorted grayscale slices of
64 x 64 pixels. Missing modalities, unknown labels, duplicate IDs, invalid image
sizes, and invalid slice counts are errors.

Training-only spatial augmentation is applied to CT and PET in a shared TorchIO
`Subject`. Validation and external cohorts are never augmented. The external
cohort is not constructed during HPO and is evaluated only after model selection.

## Models and outputs

The explicit registry provides `fusion3d`, `densenet`, and `feature_extractor`.
TorchVision's weights API is used, and pretrained first-layer weights are expanded
to the required input channels.

By default, a run writes `resolved_config.yaml`, `metrics.json`, `history.csv`,
`run.log`, and a pure `best.pt` state dict. Patient IDs and per-case probabilities
are not written. Set `output.save_predictions: true` only when anonymous paired
prediction rows are required for statistical comparison.

## Reproducibility checks

```bash
pytest
ruff check .
python -m petct --help
```

See [preprocess/README.md](preprocess/README.md) for image processing and
[configs/README.md](configs/README.md) for the configuration schema.
