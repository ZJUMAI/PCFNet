# Clinical late fusion / 临床后融合

Combine **nine clinical variables and one image probability**. `Cycles` and `PDL1`
are excluded from loading, preprocessing, fitting, and generated fusion inputs.
The retained variables are gender, age, smoking, drug, pathology, tumor size,
T stage, N stage, and overall stage.

整合模型使用九项临床变量和一个影像概率，不使用治疗周期数或 PD-L1。
原始 Excel 和预测文件不会被修改。

## Prepare independent experiment inputs

Run from the repository root. Example in PowerShell:

```powershell
python -m fusion.prepare_inputs `
  --runs F:\study\sph\data\pcfnet_crossval_seed_739330856 F:\study\sph\data\pcfnet_crossval_1023 `
  --clinical fusion/PET_all_radiomics-2026-1-22.xlsx `
  --output-root F:\study\sph\data\fusion_no_cycles_pdl1 `
  --label-source predictions
```

- Internal image scores: concatenate the five `predictions_validation.csv`
  files. Each case must appear exactly once. Never read `predictions_train.csv`.
- External image scores: for each experiment and each of the three cohorts,
  align identifiers and take the arithmetic mean of five probabilities.
  **The two experiments are never averaged together.**
- Match clinical rows using cohort + case identifier. The old clinical table's
  `检查编号` matches the prediction files' `影像组学序列号`; IDs may repeat across hospitals.
- Retain the clinical table's original `Cohort_final` Training/Test assignments
  for fitting/validation. In the provided table these contain 189/19 cases.
  The remaining cohorts map to multi_hospital/France/SPH_test.
- Refuse missing cases, duplicate cases within a cohort, label disagreements,
  non-finite scores, and inconsistent case sets or labels between external folds.
- `--label-source predictions` explicitly resolves clinical/CSV disagreements in
  favor of CSV ground truth. `--label-source clinical` uses clinical pCR instead.
  Default `strict` stops on disagreements. The original clinical label is retained
  in `clinical_pCR_original`, excluded from model features, and disagreement counts
  are recorded in the summary. No original labels are overwritten. For the
  supplied data, three SPH_test labels disagree; the user selected prediction
  CSV ground truth as the authority. The example makes that decision explicit.

每个实验输出独立目录，包含三组测试概率均值、内部折外概率、`fusion_input.csv`
和处理摘要。保留原临床表的 Training/Test 划分，不重新随机划分。
旧 `1023` 的部分 validation CSV 与 metrics JSON 不一致，摘要会报告该问题；
输入概率以用户指定的 validation CSV 为准，不能因此认定其 checkpoint 来源已核实。

## Fit the fusion models

```powershell
python -m pip install -e ".[two-stage]"
python -m fusion.run_fusion `
  --input F:\study\sph\data\fusion_no_cycles_pdl1\pcfnet_crossval_seed_739330856\fusion_input.csv `
  --output F:\study\sph\data\fusion_no_cycles_pdl1\pcfnet_crossval_seed_739330856\models
```

Repeat with `pcfnet_crossval_1023` for the other experiment. Supported models:
`--models lgbm logistic rf xgb` (default: all four). Seed: 42.
The original 40 random-search trials and Top-10/minimum train-validation AUC gap
selection are retained. `--top-k 1` selects strictly by validation AUC.
Every trial gets an independent cloned pipeline. Preprocessing and model fitting
use Training only. External labels/scores are not used for parameter selection.

Outputs are per-model CSV probabilities and JSON metrics (threshold 0.5), not Excel.
Default search does not generate SHAP or retrain any CNN. Use new output directories;
existing results are never overwritten. Optional LightGBM/XGBoost imports are lazy.

新入口使用独立模型对象，避免旧随机搜索“最佳模型被后续 trial 覆盖”的问题。
旧 `train_*.py`、`grid_search.py` 仅作历史参考，尤其不要用按外部测试 AUC 调参的
`grid_search.py` 作为正式评估流程。

## Linux server: use the prepared inputs

Git transfers code only, never the private inputs. Copy the locally prepared
`fusion_no_cycles_pdl1` directory to `~/PCFNet/data/`, preserving both experiment
subdirectories and their `input_summary.json` provenance records. The prepared
CSV inputs already contain the required clinical variables, OOF internal scores,
and averaged external scores, so the original XLSX is not needed for this route.

```bash
conda activate pcfnet
cd ~/PCFNet
git pull
python -m pip install -e ".[two-stage]"

for run in pcfnet_crossval_seed_739330856 pcfnet_crossval_1023; do
  python -m fusion.run_fusion \
    --input "data/fusion_no_cycles_pdl1/$run/fusion_input.csv" \
    --output "runs/fusion_no_cycles_pdl1/$run" \
    --models lgbm logistic rf xgb || break
done
```

This is CPU-only tabular training; no CUDA device selection is required. Each
experiment is fitted separately. Outputs contain per-model CSV probabilities
for train, validation, and the three test cohorts, plus JSON metrics and search
records. Default: 40 trials/model, Top-10/minimum-gap selection, seed 42.
Choose a new output directory for a rerun: existing results are not overwritten.

服务器只需已生成的输入目录，不需要重新运行 CNN，也不需要再次准备原 Excel。
Git 不传输患者数据，请自行通过安全渠道复制。训练默认使用 CPU，两个实验分别运行。

## Interpretation and privacy

OOF construction and averaging do **not** repair upstream leakage simulation,
test-driven epoch/seed selection, or validation-driven CNN checkpoint selection.
Fusion validation is a tuning diagnostic, not a nested, independent performance
estimate. Preserve experiment provenance and disclose upstream limitations.
Clinical availability of every retained variable (including drug) must be checked
against the intended prediction time point.

折外概率仅避免直接使用 CNN 训练内分数，不会消除原实验已有的数据泄露和选模偏差。
此流程没有执行嵌套交叉验证，不应将调参验证集成绩视为独立泛化估计。
源 Excel、患者级 CSV、SHAP 和输出目录均不得提交到公共仓库；忽略规则只放行代码和说明。
