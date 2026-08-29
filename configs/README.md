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
