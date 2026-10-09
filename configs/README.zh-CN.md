# 配置说明

所有相对路径均以 YAML 文件所在目录为基准。使用前请复制示例并指向本地未纳入版本
控制的数据，切勿提交标签表。服务器配置已直接指向旧项目原有的 radiomics Excel；
`metadata` 只是这张已有标签表的路径，不需要另建 `metadata.csv`。
ID 列为“影像组学序列号”，标签列为 `pCR`，程序只使用这两列。

- `data`：metadata 列、体数据形状、增强参数及各数据划分路径；`external` 是命名队列。
- `model`：模型注册名和构造参数。
- `training`：优化器、调度器、损失、DataLoader、epoch 和早停参数。
- `output`：实验目录及默认关闭的匿名预测保存开关。
- `hpo`：可恢复的 Optuna study 和带类型的点号键搜索空间。
- `preprocess`：各队列路径及四步预处理参数。
- `evaluate`：指标、Youden、五折汇总、配对 DeLong 或 checkpoint 外部终评。

`--set` 使用 YAML 类型解析，例如 `--set model.params.pretrained=false` 是布尔值。

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
