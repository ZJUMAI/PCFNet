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
- `evaluate`：指标、Youden、十折汇总、配对 DeLong 或 checkpoint 外部终评。

`--set` 使用 YAML 类型解析，例如 `--set model.params.pretrained=false` 是布尔值。

`split.yaml` 默认按标签分层生成 80% 训练、20% 验证名单，随机种子为 513。
`split.metadata` 直接读取原 SPH radiomics Excel。`source_manifests` 合并已有
`dataset/train.txt` 和 `dataset/valid.txt`，限定本次重新划分的病例范围，表内其他
病例不参与划分。分层划分需要 `pCR` 标签，不需要新建 CSV，也不要加入外部队列。
先运行：

```bash
python -m petct split --config configs/split.yaml
python -m petct train --config configs/train.yaml
```

名单输出到 `dataset/generated_split_80_20/train.txt` 和 `validation.txt`，
`train.yaml` 已指向这两个文件。`summary.json` 记录各组总数及阳性/阴性数，
人数不能整除时比例会略有取整。验证集用于早停与模型选择，外部测试队列保持独立。
划分配置和训练配置的 metadata 路径及列名应保持一致。

如需十折划分，设置 `split.mode: kfold`、`split.folds: 10`，
并将输出目录改为 `../dataset/generated_splits`、文件名模板改为
`fold_{fold}_{split}.txt`，再将交叉验证配置指向生成的名单。
