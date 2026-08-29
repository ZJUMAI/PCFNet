# 配置说明

所有相对路径均以 YAML 文件所在目录为基准。使用前请复制示例并指向本地未纳入版本
控制的数据，切勿提交 metadata。

- `data`：metadata 列、体数据形状、增强参数及各数据划分路径；`external` 是命名队列。
- `model`：模型注册名和构造参数。
- `training`：优化器、调度器、损失、DataLoader、epoch 和早停参数。
- `output`：实验目录及默认关闭的匿名预测保存开关。
- `hpo`：可恢复的 Optuna study 和带类型的点号键搜索空间。
- `preprocess`：各队列路径及四步预处理参数。
- `evaluate`：指标、Youden、十折汇总、配对 DeLong 或 checkpoint 外部终评。

`--set` 使用 YAML 类型解析，例如 `--set model.params.pretrained=false` 是布尔值。
