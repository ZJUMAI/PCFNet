# Configuration Guide

All relative paths are resolved from the YAML file's directory. Copy an example
before use and point it at local, non-versioned data. Do not commit metadata.

- `data`: metadata schema, volume shape, augmentation, and split-specific
  manifests/CT/PET roots. `external` is a named cohort mapping.
- `model`: registry name and constructor parameters.
- `training`: optimizer, scheduler, loss, loader, epoch, and early-stop settings.
- `output`: run directory and the opt-in anonymous prediction switch.
- `hpo`: persistent Optuna study and typed dotted-key search space.
- `preprocess`: cohort paths and parameters for all four image-processing steps.
- `evaluate`: aggregate metrics, Youden threshold, fold summary, paired DeLong, or
  checkpoint-based final external evaluation.

Use YAML-native values in overrides. For example, `--set model.params.pretrained=false`
is a Boolean while `--set training.batch_size=4` is an integer.

`split.yaml` defaults to a stratified 80% training / 20% validation holdout with
seed 513. Its metadata must contain only the internal cohort to be split, with
the same metadata path and column names used for training. Keep external test
cohorts separate. Run `python -m petct split --config configs/split.yaml` before
`python -m petct train --config configs/train.yaml`.

The training configuration uses the resulting `train.txt` and `validation.txt`
in `dataset/generated_split_80_20`. An anonymous `summary.json` records each
subset's size and class counts. Integer rounding can slightly change the ratio.
Validation is used for checkpoint selection and early stopping.

For the existing cross-validation workflow, use `split.mode: kfold`,
`split.folds: 10`, `split.output_directory: ../dataset/generated_splits`, and
`split.filename_pattern: 'fold_{fold}_{split}.txt'`, and point the cross-validation
configuration at those manifests. Configurations without a mode retain k-fold
behavior.
