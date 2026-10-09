# 配置说明

所有相对路径均以 YAML 文件所在目录为基准。使用前请复制示例并指向本地未纳入版本
控制的数据，切勿提交标签表。服务器配置已直接指向旧项目原有的 radiomics Excel；
`metadata` 只是这张已有标签表的路径，不需要另建 `metadata.csv`。
ID 列为“影像组学序列号”，标签列为 `pCR`，程序只使用这两列。

- `data`：metadata 列、体数据形状、增强参数及各数据划分路径；`external` 是命名队列。
- `model`：模型注册名和构造参数。
- `training`：优化器、调度器、损失、DataLoader、epoch 和早停参数。
- `output`：实验目录、逐例预测开关及独立的编号导出开关。
- `hpo`：可恢复的 Optuna study 和带类型的点号键搜索空间。
- `preprocess`：各队列路径及四步预处理参数。
- `evaluate`：指标、Youden、五折汇总、配对 DeLong 或 checkpoint 外部终评。

`--set` 使用 YAML 类型解析，例如 `--set model.params.pretrained=false` 是布尔值。

## 最终测试预测文件

当前 `train.yaml`、`two_stage.yaml` 的 `output.save_predictions` 与
`output.save_prediction_ids` 均已显式设为 `true`；`evaluate.yaml` 在 `evaluate`
段启用相同开关。公共 API 与 HPO 仍默认关闭导出。关闭 `save_prediction_ids`
可仅导出匿名两列，关闭 `save_predictions` 则完全不保存逐例文件。

每个外部队列分别保存 UTF-8 BOM 编码的 CSV，五列表头依次为：
“影像组学序列号”“是否预测成功”“预测概率”“预测结果”“ground truth”。
概率为 pCR=1 的概率，不进行四舍五入；概率 >= 0.5 时预测结果为 1，否则为 0。
预测结果等于真实标签时，预测成功为 1，否则为 0；真实标签也统一为 0/1。
阈值固定为 0.5，不使用外部测试数据选阈值。

单次训练输出 `runs/<run_name>/predictions_<cohort>.csv`，五折输出
`runs/<run_name>/fold_<n>/predictions_<cohort>.csv`。两阶段额外保存最终树模型的
验证集预测。独立 checkpoint 测试输出到 `evaluate.prediction_directory`。
编号与名单顺序严格一致，带编号导出禁止打乱病例或丢弃尾批次。结果含隐私编号，
不得提交到 Git 或随论文公开。

已有 checkpoint 不需要重新训练，例如第 1 折可运行：

```bash
python -m petct evaluate --config configs/evaluate.yaml \
  --set evaluate.checkpoint=../runs/pcfnet_crossval/fold_1/best.pt \
  --set evaluate.prediction_directory=../runs/pcfnet_crossval/fold_1/final_predictions \
  --set evaluate.output=../runs/pcfnet_crossval/fold_1/final_evaluation.json
```

metrics、Youden 与配对 DeLong 命令可自动识别新表头和旧匿名表头。
配对 DeLong 要求两模型的编号、标签和行顺序完全相同。
不同折对同一病例的预测不能当成独立病例拼接后计算统计量。

`split.yaml` 默认生成分层五折名单，随机种子为 513。每折约 80% 训练、20% 验证，
每例恰好在一折中作为验证病例，不是重复随机划分五次。
`split.metadata` 直接读取原 SPH radiomics Excel。`source_manifests` 合并已有
`dataset/train.txt` 和 `dataset/valid.txt`，限定本次重新划分的病例范围，表内其他
病例不参与划分。分层划分需要 `pCR` 标签，不需要新建 CSV，也不要加入外部队列。
先运行：

```bash
python -m petct split --config configs/split.yaml
python -m petct crossval --config configs/compare.yaml
```

名单输出到 `dataset/generated_splits_5fold/`，文件为 `fold_1_train.txt`、
`fold_1_validation.txt` 至 `fold_5_train.txt`、`fold_5_validation.txt`，共 10 个 TXT。
`compare.yaml` 已同步读取这些文件并执行五折。未指定折数时，划分和交叉验证代码
也都默认五折。`summary.json` 仅记录匿名病例总数及类别统计。
人数不能整除时每折比例略有取整；验证集用于早停与模型选择，外部测试队列保持独立。

`train`、`hpo`、`two-stage` 仍是单次运行，默认读取五折名单中的第 1 折；
切换其他折时须同时覆盖训练和验证 manifest。HPO 只使用验证集，不访问外部队列。
划分配置和训练配置的 metadata 路径及列名应保持一致。

主模型完整五折训练可使用：

```bash
python -m petct crossval --config configs/train.yaml \
  --set 'data.train.manifest=../dataset/generated_splits_5fold/fold_{fold}_train.txt' \
  --set 'data.validation.manifest=../dataset/generated_splits_5fold/fold_{fold}_validation.txt' \
  --set output.run_name=fusion3d_crossval
```

一次 8:2 划分仍可显式启用：设置 `split.mode: holdout`、
`split.validation_fraction: 0.2`、独立输出目录和 `split.filename_pattern: '{split}.txt'`，
再将单次训练配置指向对应文件。这不再是默认模式。
