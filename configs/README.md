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
`two_stage.yaml`, and `evaluate.yaml`. Its paired image roots are
`/data4/zhenglujie/petct/clahed_64_0.2_20/france/ct` and
`/data4/zhenglujie/petct/clahed_64_0.2_20/france/pet`, using the same processed
data root as the other cohorts.
France uses its own label table, not the table shared by the three hospitals and SPH_test.
All label tables must contain `影像组学序列号` values matching the manifests and
case-directory names. A legacy France table using `number` must be updated
before running; no fallback ID column is used.
By default, external cohorts are evaluated only after model selection; HPO does not
access them.

中文说明：内部 SPH 使用上述内部标签表，四个外部队列共用上述外部标签表。
France 的 40 例名单已启用，单独使用 `PET_France/PETCT.xlsx`，编号列为
`影像组学序列号`、标签列为 `pCR`；CT/PET 与其他队列统一使用 `clahed_64_0.2_20` 路径。
标签表中的编号须与名单及病例目录一致。旧 France 表若只有 `number` 列，
须先更新为统一列名；代码不自动回退到其他编号列。
默认外部队列仅在模型选择结束后评估，不参与 HPO。

- `data`: metadata schema, volume shape, augmentation, and split-specific
  manifests/CT/PET roots. `external` is a named cohort mapping; an evaluation
  group uses `members` to map hospital names to individual paired data sources.
- `model`: registry name and constructor parameters.
- `training`: optimizer, scheduler, loss, loader, epoch, and early-stop settings.
- `output`: run directory, optional checkpoint export, prediction export, and
  the separate case-ID switch.
- `hpo`: persistent Optuna study and typed dotted-key search space.
- `preprocess`: cohort paths and parameters for all four image-processing steps.
- `evaluate`: aggregate metrics, Youden threshold, fold summary, paired DeLong, or
  checkpoint-based final external evaluation.

## Spatial-token fusion experiment

`train_spatial.yaml` selects the new registry entry `fusion3d_spatial`. The
original `train.yaml` / `fusion3d` remains unchanged for comparison. Both use the
same two R3D-18 backbones and preprocessing; the new version takes the last
spatial maps before global pooling, then uses:

- Exactly 64 tokens per modality, from adaptive 4 x 4 x 4 pooling.
- Separate projections to `embedding_dim: 128` (must be divisible by `num_heads`).
- One bidirectional Cross-Attention block with `num_heads: 4` and
  `attention_dropout: 0.1`; both directions use the original normalized tokens.
- Residual additions to each modality's own tokens, followed by separate token
  means, concatenation, and the existing style of classification head.

There are no gates, auxiliary heads, additional feed-forward blocks, or positional
encodings. Data paths, metadata, manifests, loss, optimizer, scheduler, early
stopping, and prediction schemas match `train.yaml`. Output is separated as
`runs/fusion3d_spatial`. Default external selection remains `validation_best`,
and `output.save_checkpoint` remains `false`. No performance improvement is
assumed; use internal validation to compare architectures on the same folds.

Run one fold or all five folds on the server:

```bash
CUDA_VISIBLE_DEVICES=5 python -m petct train --config configs/train_spatial.yaml \
  --set device=cuda:0

CUDA_VISIBLE_DEVICES=5 python -m petct crossval --config configs/train_spatial.yaml \
  --set device=cuda:0 \
  --set 'data.train.manifest=../dataset/generated_splits_5fold/fold_{fold}_train.txt' \
  --set 'data.validation.manifest=../dataset/generated_splits_5fold/fold_{fold}_validation.txt' \
  --set output.run_name=pcfnet_spatial_crossval
```

CUDA device indices inside Python are relative to `CUDA_VISIBLE_DEVICES`.
For CPU runs, omit the environment assignment and set `device=cpu` instead.
Enable `--set output.save_checkpoint=true` during training if later checkpoint
evaluation is needed. Spatial-model checkpoints require the spatial model name
and the same embedding dimension and head count; old global-fusion checkpoints
are not compatible. For a saved single-fold spatial checkpoint:

```bash
python -m petct evaluate --config configs/evaluate.yaml \
  --set model.name=fusion3d_spatial \
  --set model.params.embedding_dim=128 \
  --set model.params.num_heads=4 \
  --set evaluate.checkpoint=../runs/fusion3d_spatial/best.pt \
  --set evaluate.output=../runs/fusion3d_spatial/final_evaluation.json \
  --set evaluate.prediction_directory=../runs/fusion3d_spatial/final_predictions
```

中文说明：新配置 `train_spatial.yaml` 每模态固定 64 个 token，128 维、4 个头、
一层双向注意力和残差，不加其他融合模块。保留原模型和配置；数据路径及训练
设置不变，独立输出到 `runs/fusion3d_spatial`。比较须使用同一划分。若显式导出
模型，独立测试时须选择 `fusion3d_spatial` 和匹配的维度、头数，不能加载旧权重。

## Five repeated stratified 70:30 holdouts

`split_70_30.yaml` uses `split.mode: repeated_holdout`, `split.repeats: 5`,
`split.validation_fraction: 0.3`, and seed 42. It uses StratifiedShuffleSplit
to draw five reproducible stratified train/validation pairs from the eligible
internal SPH cases. The RNG is initialized once, not reset before each draw.
Only `dataset/train.txt` and `dataset/valid.txt` determine eligibility; the
existing SPH label table supplies stratification labels. External cohorts are
not included. [StratifiedShuffleSplit documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html)

Every repeat has approximately 70% training and 30% validation, with disjoint
sets within that repeat. Validation sets can overlap between repeats, and some
cases may never be selected for validation. This is **five repetitions of
stratified random holdout**, not disjoint five-fold cross-validation. Integer
rounding is unavoidable: for 208 eligible cases, each repeat has 145 training
and 63 validation cases.

The new output directory is `dataset/generated_splits_70_30_5repeats`, containing
`repeat_1_train.txt` / `repeat_1_validation.txt` through `repeat_5_train.txt` /
`repeat_5_validation.txt`. An anonymous summary records repeat counts, ratio,
seed, and per-repeat class counts. The directory is excluded from Git; original
TXT files and the standard `generated_splits_5fold` directory are not changed.

`train_spatial_70_30.yaml` keeps the spatial model and all existing training,
external-evaluation, and export defaults, but selects these manifest templates
and writes to `runs/fusion3d_spatial_70_30`. Use the existing `crossval` runner:

```bash
python -m petct split --config configs/split_70_30.yaml
CUDA_VISIBLE_DEVICES=5 python -m petct crossval --config configs/train_spatial_70_30.yaml \
  --set device=cuda:0
```

The runner's `cross_validation.folds: 5`, `{fold}` placeholder, `fold_1` through
`fold_5` output directories, and `cross_validation.json` name are retained for
compatibility; here their indices represent **repeats**, not disjoint folds.
Use `crossval`, not `train`, with this templated configuration. Change both
`split.repeats` and `cross_validation.folds` if changing the repetition count.
For one repeat, use `train_spatial.yaml` with both paths overridden to concrete
`repeat_1_train.txt` and `repeat_1_validation.txt` paths.

Other registered models can use the same split files. For the original PCFNet:

```bash
CUDA_VISIBLE_DEVICES=5 python -m petct crossval --config configs/train.yaml \
  --set device=cuda:0 --set cross_validation.folds=5 \
  --set 'data.train.manifest=../dataset/generated_splits_70_30_5repeats/repeat_{fold}_train.txt' \
  --set 'data.validation.manifest=../dataset/generated_splits_70_30_5repeats/repeat_{fold}_validation.txt' \
  --set output.run_name=pcfnet_70_30
```

Report this as "five repeated stratified 70:30 holdouts". Do not concatenate
validation predictions across repeats and treat repeated cases as distinct
patients. Report per-repeat results and their mean/SD; the runner's existing
normal-approximation CI fields are not a formal patient-level confidence interval
for these dependent repeated estimates. External test data remain independent
of training and epoch selection under the default policy.

中文说明：新增 `repeated_holdout` 代码与 `split_70_30.yaml`，生成五次分层
7:3 划分，seed=42，每次训练和验证无交集，但跨次验证可重叠，不保证每例恰好
验证一次。空间模型配套 `train_spatial_70_30.yaml` 可直接用 `crossval` 运行。
208 例时每次为 145 例训练、63 例验证。旧标准五折不变，新增名单目录不提交到 Git。
论文写“五次重复分层随机 7:3 验证”，不能将重复病例当作额外独立样本。

## Checkpoint export

`output.save_checkpoint` is a Boolean and defaults to `false` in the shared
engine and all training configurations (`train`, `train_spatial`,
`train_spatial_70_30`, `compare`, `hpo`, and
`two_stage`). Validation-best weights are copied to CPU memory and restored at
the end of training, so metrics, external predictions, and two-stage feature
extraction still use the correct model without reading or writing `best.pt`.

To keep a model for later independent evaluation, enable export before training:

```bash
python -m petct train --config configs/train.yaml --set output.save_checkpoint=true
```

Cross-validation applies the same switch to every fold. Exported `best.pt`
contains only the validation-selected state dict. Disabling export does not
delete or overwrite existing checkpoints, and new training never restores its
best weights from a stale checkpoint in the output directory. The two-stage
workflow still saves its existing `tree_model.joblib`; this switch controls only
the CNN checkpoint. Without export, no new checkpoint is available for the
standalone checkpoint `evaluate` task after the training process exits.

中文说明：所有训练默认 `output.save_checkpoint: false`，不落盘 `best.pt`。
最佳权重在 CPU 内存中独立保存并恢复，测试和两阶段特征提取不变。
需要之后单独测试时，训练前设置 `--set output.save_checkpoint=true`；已有模型
文件不自动删除，关闭导出时也不覆盖旧文件。树模型文件的保存行为不受影响。

## External evaluation groups

`train.yaml`, `two_stage.yaml`, and `evaluate.yaml` evaluate exactly three groups:

- `multi_hospital`: pooled wuhan, ruijin, and FUSCC (70 cases in the current manifests).
- `france`: the France cohort.
- `SPH_test`: the independent SPH test cohort.

Each `multi_hospital.members` entry retains its own manifest and CT/PET roots.
No images are moved and no original TXT files are changed. Cases are concatenated
in wuhan, ruijin, FUSCC order, preserving each manifest's order and original IDs.
Duplicate IDs across members are rejected to prevent double-counting or ambiguous
identified exports. Metrics are computed once from all pooled labels and
probabilities, not by averaging hospital metrics. The combined group gets one
prediction CSV and one metrics entry per fold; individual hospital results are
not generated by these configurations. Grouping is evaluation-only.

中文说明：默认外部评估仅分为 `multi_hospital`（wuhan、ruijin、FUSCC 合并，
当前共 70 例）、`france`、`SPH_test` 三组。不移动影像、不改原 TXT，三家按配置
顺序拼接，保留原编号；跨医院重号直接报错。合并组从所有病例重新计算指标，
不是三家指标取平均，每折仅保存一份合并组预测表。

## Optional per-cohort best-epoch test selection

`evaluation.external_selection` defaults to `validation_best`: select the best model
using validation AUC, then evaluate each external cohort once. Existing output
schemas and the default training behavior remain unchanged.

For an explicitly exploratory `train` or `crossval` run, set:

```yaml
evaluation:
  external_after_training: true
  external_selection: per_cohort_best_auc
```

Or add `--set evaluation.external_selection=per_cohort_best_auc` to the existing
training command. Use a separate `output.run_name` to keep exploratory outputs
separate from unbiased final-test results.

This evaluates the three cohorts at every completed epoch. Each cohort retains
the epoch with its highest AUC; ties keep the first occurrence. All its final
metrics and exported probabilities come from that same epoch, not independently
maximized metrics. The three cohorts may select three different epochs. Output
filenames, metric JSON structures, history columns, and CSV headers stay the
same; no selected-epoch fields are written. Prediction export still respects
the existing `save_predictions` and `save_prediction_ids` switches.

**This uses external test labels for model selection and produces optimistically
biased test estimates. Do not present them as unbiased final test performance.**
The mode increases inference time and requires both classes in every test cohort.
Checkpoint selection, scheduler, early stopping, and HPO remain validation-only.
Only epochs actually completed before validation-based early stopping are
considered; set `training.early_stopping_patience: 0` explicitly if all configured
epochs are needed. When checkpoint export is enabled, `best.pt` remains the
validation-selected checkpoint; no
additional external-selected checkpoints are saved. A later single-checkpoint
`evaluate` cannot reconstruct these per-epoch predictions. Enable this option
before training; it is not supported by HPO or the two-stage tree workflow.

中文说明：`external_selection` 默认 `validation_best`，维持现有验证集选模方案。
显式设为 `per_cohort_best_auc` 后，每个实际完成的 epoch 都评估三组，各自按
AUC 保留最佳轮次（并列保留最早一轮），最终指标与预测概率来自同一轮。
不增加选中 epoch 字段，指标和 CSV 格式不变。该模式用于 `train` / `crossval`，
会使用测试集选模并高估测试表现，不应作为论文无偏终评；早停、调度及最佳模型
仍只使用验证集。须在训练前开启，已有 `best.pt` 无法还原其他轮次的预测。

## Opt-in test-leakage simulation

This is a deliberately contaminated sensitivity experiment, not a valid way to
report final generalization. It is disabled by default and supported only by
`train` / `crossval`, including repeated 70:30 holdouts:

```yaml
training:
  leakage_simulation:
    enabled: false
    samples_per_cohort: 3
    cohort_samples: {}  # Optional per-group count overrides; 0 skips a group.
    # Optional; omit to use all configured data.external groups.
    cohorts: [multi_hospital, france, SPH_test]
```

For example, append these options to the existing training command:

```bash
python -m petct crossval --config configs/train_spatial_70_30.yaml \
  --set training.leakage_simulation.enabled=true \
  --set training.leakage_simulation.samples_per_cohort=3
```

This adds **3 cases per group**, hence 9 additional training cases per epoch for
the three default groups. `multi_hospital` is one pooled group: three cases are
drawn from its combined members, not three per hospital. To use only one group,
add `--set 'training.leakage_simulation.cohorts=[SPH_test]'`.

To set independent counts, use `cohort_samples`. For example:

```yaml
training:
  leakage_simulation:
    enabled: true
    samples_per_cohort: 3
    cohort_samples:
      multi_hospital: 10
      france: 5
      SPH_test: 0
```

Equivalent options to append to your existing command:

```bash
--set training.leakage_simulation.enabled=true \
--set training.leakage_simulation.cohort_samples.multi_hospital=10 \
--set training.leakage_simulation.cohort_samples.france=5 \
--set training.leakage_simulation.cohort_samples.SPH_test=0
```

This draws 15 cases per epoch in total. Each named count overrides the shared
`samples_per_cohort` value; omitted counts fall back to that shared value, not
zero. A zero count skips constructing that group's training copy, but **does not
remove it from evaluation**. `cohorts`, if supplied, still restricts eligible
groups; overriding an unselected or unknown group is an error. At least one
selected group must have a positive count when simulation is enabled. The empty
override mapping in example configs preserves all previous uniform-count commands
and their sampling sequence. Sampling seeds, resampling policy, and output schemas
are unchanged. Logs now report the resolved anonymous counts for each group.

Every epoch samples afresh without replacement within each selected group;
cases can recur across epochs, and a new draw need not differ from the previous
one. Sampling is uniform, not class-stratified. Draws use epoch-derived seeds
from the configured seed (42 by default), independently of augmentation/shuffle
RNG state. Each cross-validation run restarts this reproducible sequence.
The original training cases plus only that epoch's draws are shuffled together.
Draws do not accumulate in the training dataset, but model weights retain prior
exposure: contamination accumulates over epochs. Drawn training copies use the
same paired augmentation as the original training set; evaluation copies remain
unaugmented and include **all** original test cases, including those drawn.

Manifests, image folders, split files, existing output paths/names, metrics JSON,
history columns, and prediction CSV schemas remain unchanged. No extra output
files or case IDs are written for the simulation. The existing
`resolved_config.yaml` and `run.log` record the setting and anonymous counts.
Use a distinct `output.run_name` yourself if you need to preserve an earlier run;
this option does not rename output directories automatically.

It can be combined with `evaluation.external_selection=per_cohort_best_auc`,
which introduces additional test-based selection bias. Validation still controls
checkpoint selection, scheduling, and early stopping, but the resulting metrics
are not unbiased. Disabled simulation never constructs training views of external
data. Enabled simulation requires external evaluation/access, existing named
groups, and integer counts no larger than their corresponding group sizes
(per-group overrides may be zero; the shared fallback remains positive).
HPO and two-stage experiments reject this option rather than silently ignoring it.

中文说明：默认关闭。开启后每轮从每个指定测试组不放回随机抽取若干例并加入原
训练集，下一轮重新抽，不永久追加名单；跨轮可重复，模型也会保留以前的暴露。
三家合并医院作为一组抽样。测试名单保持完整，输出格式及目录规则不变，仅在原有
配置快照和日志记录开关。结果有数据污染，仅用于模拟，不能作为论文无偏终评。

## Final test prediction files

### Validation probabilities for offline reevaluation

Training and cross-validation default to `output.save_validation_predictions: true`,
even for older custom configurations that omit this field. To disable export, append
`--set output.save_validation_predictions=false` to your original training command.
After `fit` restores the validation-AUC-best weights, an additional validation
prediction pass writes `predictions_validation.csv` in each run/fold directory.
It uses the existing identified five-column schema when `save_prediction_ids`
is true, otherwise anonymous `label,probability`. This flag is independent
of test prediction export and also defaults to true for library training callers
and the comparison YAML. HPO trials keep export off by default; HPO-derived
configurations explicitly containing false need an override to true for final training.
These private prediction files remain gitignored.

Only export is added: training, model selection, test metrics, prediction
probabilities, and the fixed 0.5 decision rule are unchanged. No automatic Youden
selection, threshold-analysis file, or rewriting of existing results is performed.
Offline Youden selection needs this validation file's labels and probabilities,
then applies the selected threshold to test probabilities from the **same model**.
With `per_cohort_best_auc`, test files may come from different epochs and cannot
blindly reuse the validation-best model's threshold. Old runs without saved
validation probabilities or model weights cannot be reconstructed from aggregate
metrics or external probabilities alone. New exports apply to future runs only.

### Original training scores

`output.save_training_predictions` defaults to true for train/crossval and
two-stage workflows, including custom configurations that omit it. Each run/fold
writes `predictions_train.csv`. After the validation-AUC-best CNN weights are
restored, train/crossval reconstructs `data.train` with `training=False`: original
manifest order, no spatial augmentation, no shuffle, and no dropped final batch.
It never reuses the epoch-specific leakage loader, so temporarily sampled test
cases are excluded. This does not remove any influence of leakage on learned weights.
Two-stage runs export the final tree's class-1 probabilities on original training features.

The file uses the existing five identified columns when `save_prediction_ids`
is true, otherwise anonymous `label,probability`. Score means the unrounded
probability of pCR=1; decisions still use 0.5. This switch is independent of
`save_predictions` and `save_validation_predictions`. Disable with
`--set output.save_training_predictions=false`. HPO trials default to false;
an explicitly disabled HPO-derived configuration needs an override for final training.
No training metrics are added to `metrics.json` or cross-validation summaries.

### External prediction files

The server `train.yaml` and `two_stage.yaml` configurations explicitly set
`output.save_predictions: true` and `output.save_prediction_ids: true`.
`evaluate.yaml` sets the same flags under `evaluate`. Library defaults and HPO
remain disabled. Turn off `save_prediction_ids` for anonymous two-column CSVs,
or turn off `save_predictions` to write no external prediction files.

Final external tests write one UTF-8-with-BOM CSV per cohort, using these columns
in this order: `影像组学序列号`, `是否预测成功`, `预测概率`, `预测结果`, `ground truth`.
Probabilities are P(pCR=1) and retain their original numerical precision.
`预测结果` is 1 when probability >= 0.5, otherwise 0. `是否预测成功` is 1 when
prediction equals ground truth, otherwise 0. Ground truth is normalized to 0/1.
The fixed threshold is not selected from external test data.

Training outputs are `runs/<run_name>/predictions_<cohort>.csv`; cross-validation
outputs are `runs/<run_name>/fold_<n>/predictions_<cohort>.csv`.
Two-stage runs also export the final tree's validation predictions.
Checkpoint evaluation writes `<cohort>.csv` in `evaluate.prediction_directory`.
IDs follow the complete manifest order (member order, then manifest order for
grouped cohorts). Identified exports reject shuffled or
drop-last loaders. These private results must not be committed or published.

Already-saved checkpoints can be evaluated without retraining. New training runs
must explicitly set `output.save_checkpoint: true` beforehand to produce such a
file. For an existing checkpoint:

```bash
python -m petct evaluate --config configs/evaluate.yaml \
  --set evaluate.checkpoint=../runs/pcfnet_crossval/fold_1/best.pt \
  --set evaluate.prediction_directory=../runs/pcfnet_crossval/fold_1/final_predictions \
  --set evaluate.output=../runs/pcfnet_crossval/fold_1/final_evaluation.json
```

The metrics, Youden, and paired DeLong evaluation tasks automatically recognize
both the five-column identified schema and legacy anonymous `label,probability`
files. For paired DeLong, identified inputs must have the same IDs and labels
in the same row order. Do not combine different folds as independent patients.

Use YAML-native values in overrides. For example, `--set model.params.pretrained=false`
is a Boolean while `--set training.batch_size=4` is an integer.

`split.yaml` defaults to stratified five-fold cross-validation with seed 42.
Each fold uses approximately 80% of cases for training and 20% for validation;
every case appears in exactly one validation fold. `source_manifests` combines
the original `dataset/train.txt` and `dataset/valid.txt` into the eligible case
list. Labels are read directly from the SPH radiomics XLSX; table rows outside
this list and all external test cohorts are excluded.

Generate the manifests, then run all five folds:

```bash
python -m petct split --config configs/split.yaml
python -m petct crossval --config configs/compare.yaml
```

The generated directory is `dataset/generated_splits_5fold`, containing
`fold_1_train.txt` / `fold_1_validation.txt` through
`fold_5_train.txt` / `fold_5_validation.txt`, plus an anonymous `summary.json`.
`compare.yaml` reads these templates and runs five folds. Both the splitter and
cross-validation runner also default to five folds when no fold count is given.
Integer rounding can slightly change each fold's 80:20 ratio.

All configured seeds and code defaults are 42, including training, data loading,
Optuna, tree classifiers, splitting, and bootstrap intervals. Existing generated
manifests are not changed automatically; rerun `split` to regenerate the folds
with seed 42 before starting a new experiment. Original cohort TXT files are
preserved. DataLoader workers use distinct seeds deterministically derived
from the base seed rather than identical random streams.

The single-run commands `train` (including `train_spatial.yaml`), `hpo`, and
`two-stage` read fold 1 by default;
they do not automatically run all five folds. Override both manifest paths to
select another fold. HPO remains validation-only and does not access external
cohorts. Validation is used for checkpoint selection and early stopping.

To run the main model across all five folds, use its training configuration with
fold-template overrides:

```bash
python -m petct crossval --config configs/train.yaml \
  --set 'data.train.manifest=../dataset/generated_splits_5fold/fold_{fold}_train.txt' \
  --set 'data.validation.manifest=../dataset/generated_splits_5fold/fold_{fold}_validation.txt' \
  --set output.run_name=fusion3d_crossval
```

A single holdout remains available only by explicitly setting `split.mode:
holdout`, `split.validation_fraction: 0.2`, a separate output directory, and
`split.filename_pattern: '{split}.txt'`. Point single-run training configurations
to those holdout files when using this optional mode.
