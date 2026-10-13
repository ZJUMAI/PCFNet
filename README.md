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
enable `output.save_checkpoint: true` during final training if you plan to use
`configs/evaluate.yaml` for separate final checkpoint evaluation.

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
cohort is not constructed during HPO and, by default, is evaluated only after
validation-based model selection.

Default final evaluation uses three groups: `multi_hospital` (pooled wuhan,
ruijin, and FUSCC), `france`, and `SPH_test`. The pooled group retains each
hospital's image roots and manifest; metrics are recomputed over all its cases,
not averaged across hospitals. It produces one combined prediction CSV.

For exploratory `train` / `crossval` runs, the opt-in setting
`evaluation.external_selection: per_cohort_best_auc` selects each test cohort's
highest-AUC epoch separately, keeping its metrics and probabilities together.
Output schemas stay unchanged. This uses test labels for selection and is not
unbiased final evaluation. The default is `validation_best`; validation-only
checkpoint selection and early stopping remain unchanged. See the configuration
guide for limitations.

## Models and outputs

The explicit registry provides `fusion3d`, `fusion3d_spatial`, `densenet`, and
`feature_extractor`.
TorchVision's weights API is used, and pretrained first-layer weights are expanded
to the required input channels.

`fusion3d` retains the original global-feature fusion for baseline comparisons.
The experimental `fusion3d_spatial` retains both R3D-18 backbones but takes their
spatial maps before global pooling. Each modality is pooled to 4 x 4 x 4 regions,
projected to 64 tokens of dimension 128, and processed by one bidirectional,
four-head cross-attention block with pre-normalization and residual connections.
The updated tokens are mean-pooled, concatenated, and passed to the classification
head. There are no gates, auxiliary heads, extra Transformer feed-forward blocks,
or positional encodings in this first experimental version.

```bash
python -m petct train --config configs/train_spatial.yaml
```

This uses the same data, fold-1 manifests, optimizer, and training settings as
`train.yaml`, with a separate `runs/fusion3d_spatial` output directory. Seed 42,
three final-test cohorts, validation-only default selection, prediction export
format, and opt-in checkpoint export are unchanged. Use the same folds for a fair
comparison; the existing preprocessing and crop geometry are not modified.
New spatial-model checkpoints are not interchangeable with global-fusion
checkpoints. See the configuration guide for five-fold and checkpoint commands.

By default, a run writes `resolved_config.yaml`, `metrics.json`, `history.csv`,
and `run.log`, but does not write `best.pt`. The validation-best weights are
copied to CPU memory and restored before final evaluation or second-stage
feature extraction. Set `output.save_checkpoint: true` (or
`--set output.save_checkpoint=true`) to also export a pure `best.pt` state dict.
Standalone checkpoint evaluation requires an existing saved checkpoint.
Previously saved checkpoints are not deleted or used to restore a new training run.
The training and two-stage server
configurations explicitly enable final prediction CSVs with case IDs; the
checkpoint evaluation configuration does the same. These private files are
excluded from Git. The API remains opt-in: `save_predictions` enables export,
and `save_prediction_ids` adds identified, five-column tables. With the latter
disabled, exports contain only anonymous `label,probability` rows. HPO keeps
prediction export disabled and does not access external cohorts.

Identified CSV headers are `影像组学序列号`, `是否预测成功`, `预测概率`, `预测结果`,
and `ground truth`. Probability is P(pCR=1), not confidence in the predicted
class. Predicted class uses a fixed 0.5 threshold; correctness, prediction, and
ground truth are numeric 0/1. Probabilities are not rounded during export.

## Reproducibility checks

```bash
pytest
ruff check .
python -m petct --help
```

See [preprocess/README.md](preprocess/README.md) for image processing and
[configs/README.md](configs/README.md) for the configuration schema.
