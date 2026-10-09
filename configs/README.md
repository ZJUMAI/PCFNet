# Configuration Guide

All relative paths are resolved from the YAML file's directory. The server
configurations use the existing radiomics XLSX label tables; `metadata` is the
path to that existing table, not an additional file you need to create. The ID
column is `影像组学序列号` for every cohort, including France, and the outcome
column is `pCR`. Only the ID and outcome columns
are used. Do not commit the label tables.

The server label tables are:

- Internal SPH: `/data4/zhenglujie/petct/PET_SPH_training_2025-4-8/PET_SPH_radiomics-2025-8-14.xlsx`.
- External ruijin, wuhan, FUSCC, and SPH_test: `/data4/zhenglujie/petct/PET_SPH_training_2025-4-8/PET_External_radiomics-2025-8-14.xlsx`.
- External France: `/data4/zhenglujie/petct/PET_France/PETCT.xlsx`, with `id_column: 影像组学序列号` and `label_column: pCR`.

`dataset/france.txt` retains the 40-case France cohort, enabled in `train.yaml`,
`two_stage.yaml`, and `evaluate.yaml`. Its paired image roots remain
`/data4/zhenglujie/petct/robust_match_64_0.2_20/france/ct` and
`/data4/zhenglujie/petct/robust_match_64_0.2_20/france/pet`, as in the old project.
France uses its own label table, not the table shared by the other four cohorts.
All label tables must contain `影像组学序列号` values matching the manifests and
case-directory names. A legacy France table using `number` must be updated
before running; no fallback ID column is used.
All external cohorts are evaluated only after model selection; HPO does not
access them.

中文说明：内部 SPH 使用上述内部标签表，四个外部队列共用上述外部标签表。
France 的 40 例名单已启用，单独使用 `PET_France/PETCT.xlsx`，编号列为
`影像组学序列号`、标签列为 `pCR`；CT/PET 保留旧项目的 `robust_match_64_0.2_20` 路径。
标签表中的编号须与名单及病例目录一致。旧 France 表若只有 `number` 列，
须先更新为统一列名；代码不自动回退到其他编号列。
外部队列仅在模型选择结束后评估，不参与 HPO。

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
seed 513. `source_manifests` combines the original `dataset/train.txt` and
`dataset/valid.txt` into the eligible case list; labels are read directly from
the original SPH radiomics XLSX. Table rows outside this list are excluded.
The same label-table path and column names are used for training. Keep external
test cohorts separate. Run `python -m petct split --config configs/split.yaml` before
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
