# 配置说明

所有相对路径均以 YAML 文件所在目录为基准。使用前请复制示例并指向本地未纳入版本
控制的数据，切勿提交标签表。服务器配置已直接指向旧项目原有的 radiomics Excel；
`metadata` 只是这张已有标签表的路径，不需要另建 `metadata.csv`。
ID 列为“影像组学序列号”，标签列为 `pCR`，程序只使用这两列。

- `data`：metadata 列、体数据形状、增强参数及各数据划分路径；`external` 是命名队列。
- `model`：模型注册名和构造参数。
- `training`：优化器、调度器、损失、DataLoader、epoch 和早停参数。
- `output`：实验目录、可选模型权重导出、逐例预测开关及独立的编号导出开关。
- `hpo`：可恢复的 Optuna study 和带类型的点号键搜索空间。
- `preprocess`：各队列路径及四步预处理参数。
- `evaluate`：指标、Youden、五折汇总、配对 DeLong 或 checkpoint 外部终评。

`--set` 使用 YAML 类型解析，例如 `--set model.params.pretrained=false` 是布尔值。

## 空间 token 融合实验

`train_spatial.yaml` 使用新增模型 `fusion3d_spatial`；原 `train.yaml` 和
`fusion3d` 保留作对照。两条 R3D-18 backbone 不变，读取全局池化前的空间
特征图，通过自适应 4×4×4 池化得到每模态 64 个 token，再投影到 128 维。
使用一层双向、4 头 Cross-Attention，注意力前做 LayerNorm，并将交互输出
残差加回各自原 token。两路更新后的 token 分别平均、拼接，再输入分类头。
不加门控、辅助头、额外 FFN 或位置编码。

新配置仅调整模型及输出目录，数据路径、标签表、名单、损失、优化器、调度
和早停均沿用原配置。独立输出到 `runs/fusion3d_spatial`，seed=42，默认仍
按验证集选模、默认不保存 `best.pt`，三组测试及预测表格式不变。模型是否
改善须通过内部验证判断，不根据测试集选择架构；比较时使用同一划分。

服务器单折与完整五折命令：

```bash
CUDA_VISIBLE_DEVICES=5 python -m petct train --config configs/train_spatial.yaml \
  --set device=cuda:0

CUDA_VISIBLE_DEVICES=5 python -m petct crossval --config configs/train_spatial.yaml \
  --set device=cuda:0 \
  --set 'data.train.manifest=../dataset/generated_splits_5fold/fold_{fold}_train.txt' \
  --set 'data.validation.manifest=../dataset/generated_splits_5fold/fold_{fold}_validation.txt' \
  --set output.run_name=pcfnet_spatial_crossval
```

GPU 编号是可见设备的相对编号，物理 GPU 5 在该进程内是 `cuda:0`。CPU 测试
可去掉环境变量并设置 `--set device=cpu`。未修改现有配准、裁剪或强度处理。
如需之后单独测试，训练时显式加 `--set output.save_checkpoint=true`。
新旧融合模型 checkpoint 不兼容。已有空间模型 checkpoint 的单独测试命令：

```bash
python -m petct evaluate --config configs/evaluate.yaml \
  --set model.name=fusion3d_spatial \
  --set model.params.embedding_dim=128 \
  --set model.params.num_heads=4 \
  --set evaluate.checkpoint=../runs/fusion3d_spatial/best.pt \
  --set evaluate.output=../runs/fusion3d_spatial/final_evaluation.json \
  --set evaluate.prediction_directory=../runs/fusion3d_spatial/final_predictions
```

## 五次重复分层 7:3 划分

仅新增 YAML 不能实现五次不同的 7:3 划分；代码已补充 `repeated_holdout` 模式。
`split_70_30.yaml` 设置 `split.repeats: 5`、`validation_fraction: 0.3`、seed=42，
使用 StratifiedShuffleSplit 一次初始化随机数生成器，再生成五次分层随机划分，
不是每次重置同一种子、重复生成同一份名单。

每次约 70% 训练、30% 验证，保留正负类比例。单次训练与验证无交集，但不同次
的验证集可重叠，有些病例也可能一次都未进入验证。因此不属于标准五折 K-fold，
论文应称为“五次重复分层随机 7:3 验证”。整数取整不可避免，208 例时每次是
145 例训练、63 例验证。[官方说明](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html)

仍只使用原 `dataset/train.txt` 与 `dataset/valid.txt` 指定的内部病例，标签来自
原 SPH Excel，三组外部队列不参与划分。新名单保存到独立目录
`dataset/generated_splits_70_30_5repeats`，文件为 `repeat_1_train.txt`、
`repeat_1_validation.txt` 至第 5 次，共 10 个 TXT；摘要仅记录匿名人数及类别数。
新目录已加入 Git 忽略，原 TXT 与标准五折生成目录均保持不变。

空间 token 模型的配套训练 YAML 为 `train_spatial_70_30.yaml`。模型参数、优化器、
三组测试、预测导出、默认不存 `best.pt` 均不变，结果独立保存到
`runs/fusion3d_spatial_70_30`。服务器运行：

```bash
python -m petct split --config configs/split_70_30.yaml
CUDA_VISIBLE_DEVICES=5 python -m petct crossval --config configs/train_spatial_70_30.yaml \
  --set device=cuda:0
```

复用原 `crossval` 执行器，`cross_validation.folds: 5`、`{fold}` 占位符、
`fold_1`–`fold_5` 结果目录及 `cross_validation.json` 文件名沿用，但在本方案中
它们表示五次重复实验，不表示验证病例互斥的五折。配套模板配置使用 `crossval`
运行，不能直接 `train`；改变重复次数时同时调整 `split.repeats` 和
`cross_validation.folds`。只跑一次时，用原 `train_spatial.yaml` 并覆盖两条 manifest
为具体的 `repeat_1_train.txt` 和 `repeat_1_validation.txt`。

原 PCFNet 同样可读取这批名单：

```bash
CUDA_VISIBLE_DEVICES=5 python -m petct crossval --config configs/train.yaml \
  --set device=cuda:0 --set cross_validation.folds=5 \
  --set 'data.train.manifest=../dataset/generated_splits_70_30_5repeats/repeat_{fold}_train.txt' \
  --set 'data.validation.manifest=../dataset/generated_splits_70_30_5repeats/repeat_{fold}_validation.txt' \
  --set output.run_name=pcfnet_70_30
```

建议报告各次结果及均值、标准差；不要把跨次重复病例的预测拼成额外独立患者。
现有汇总文件里的正态近似 CI 字段，不能作为这些相关重复估计的正式患者级
置信区间。外部测试默认仍不参与模型或 epoch 选择。

## 默认不保存 best.pt

公共训练引擎及所有训练 YAML（含 `train_spatial_70_30.yaml`）均默认
`output.save_checkpoint: false`。每次验证集 AUC 改善时，在 CPU 内存中复制
最佳权重（包括模型 buffer）；结束后从内存恢复最佳模型，再测试或提取特征。
不生成 `best.pt`，不从旧 checkpoint 读取权重；指标和预测文件格式保持不变。

需要训练后单独加载模型测试时，在训练前显式启用：

```bash
python -m petct train --config configs/train.yaml --set output.save_checkpoint=true
```

五折训练时该开关对每折生效。启用后导出的 `best.pt` 仍为验证集最佳模型的
纯 state dict。关闭导出不会删除或覆盖旧 `best.pt`。两阶段的 `tree_model.joblib`
仍按现有逻辑保存，此开关仅控制 CNN 权重。未导出时，进程退出后无法从本次
训练加载 checkpoint；单独 `evaluate` 的 checkpoint 模式必须使用已有模型文件。

## 外部评估三组队列

`train.yaml`、`two_stage.yaml` 和 `evaluate.yaml` 默认评估三组：

- `multi_hospital`：wuhan、ruijin、FUSCC 三家医院合并，当前名单共 70 例。
- `france`：France 队列。
- `SPH_test`：独立 SPH 测试队列。

`multi_hospital.members` 下保留各医院原有 manifest、Excel 与 CT/PET 路径，
不移动影像目录、不修改原 TXT。按 wuhan、ruijin、FUSCC 顺序拼接，医院内部
保留名单顺序和原始编号；跨医院编号重复会报错，避免重复计数或预测错配。
合并组使用全部病例的真实标签与概率统一计算指标，不取三家指标的平均值。
每折输出一份 `predictions_multi_hospital.csv` 和一个合并组指标条目；
默认不再单独输出三家医院指标。合并仅用于评估，训练和验证划分不变。

## 可选：三组各自选择最高 AUC 的 epoch

默认配置不变：`evaluation.external_selection: validation_best`，仅按验证集
AUC 选择最佳模型，训练结束后对三组分别测试一次。

在 `train` 或 `crossval` 中可显式开启：

```yaml
evaluation:
  external_after_training: true
  external_selection: per_cohort_best_auc
```

也可在现有命令后追加 `--set evaluation.external_selection=per_cohort_best_auc`。
建议使用单独的 `output.run_name`，避免覆盖原终评结果。

开启后，每个实际完成的 epoch 都评估三组测试集；三组各自保留最高 AUC 的
一轮，因此可以来自三个不同 epoch。并列时保留最早一轮。每组最终保存的
全部 metrics 与预测概率都来自该组同一轮，不分别对每项指标取最大值。
不额外保存选中 epoch：指标 JSON 结构、history 列、CSV 表头及文件名均保持原样。
逐例导出仍由 `save_predictions` 和 `save_prediction_ids` 控制。

**该模式用测试集标签选模，会高估测试表现，不应作为论文无偏最终测试结果。**
每轮额外推理三组队列，耗时会增加；各组需同时包含正负类才能计算 AUC。
验证集仍独立控制最佳模型、学习率调度和早停；只考虑早停前已完成的轮次，
需要完整训练所有 epoch 时须自行显式设置 `training.early_stopping_patience: 0`。
不额外保存三组测试选出的模型权重；若显式导出，`best.pt` 仍是验证集最佳模型，重新
运行单 checkpoint `evaluate` 不会还原这些跨轮次的概率。须在训练前开启该模式，
旧 checkpoint 无法补算；HPO 和两阶段树模型流程不支持此模式。

## 可选的数据泄露模拟

默认关闭，仅支持 `train` / `crossval`（包含五次重复 7:3），用于对照模拟，
不能作为论文无偏泛化性能。配置为：

```yaml
training:
  leakage_simulation:
    enabled: false
    samples_per_cohort: 3
    cohort_samples: {}  # 可分别覆盖各组数量，0 表示不从该组抽样。
    # 可省略 cohorts，默认使用 data.external 中的全部组。
    cohorts: [multi_hospital, france, SPH_test]
```

运行示例：

```bash
python -m petct crossval --config configs/train_spatial_70_30.yaml \
  --set training.leakage_simulation.enabled=true \
  --set training.leakage_simulation.samples_per_cohort=3
```

这里是每组每轮抽 3 例，默认三组共加入 9 例。三家合并医院是一个整体，
从合并的病例池抽 3 例，不是每家医院各抽 3 例。只用某组可额外设置
`--set 'training.leakage_simulation.cohorts=[SPH_test]'`。

三组单独设置数量使用 `cohort_samples`，例如：

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

也可在原运行命令后追加：

```bash
--set training.leakage_simulation.enabled=true \
--set training.leakage_simulation.cohort_samples.multi_hospital=10 \
--set training.leakage_simulation.cohort_samples.france=5 \
--set training.leakage_simulation.cohort_samples.SPH_test=0
```

此例每轮共加入 15 例。单组设置优先于统一的 `samples_per_cohort`；未设置的组沿用
统一数量，不会自动当作 0。设为 0 时不构建该组训练副本，但该组依然完整参与测试。
若还设置了 `cohorts`，单组覆盖只能指定该列表中选中的组；未知或未选中组会报错。
开启模拟时至少一组数量必须大于 0。示例 YAML 默认空映射，兼容旧命令和原有均匀
数量下的抽样序列；随机种子、每轮重抽及输出格式不变，日志记录各组实际数量。

每个 epoch 重新均匀随机抽取，组内不放回、不按标签分层；跨轮允许重复，也不保证
新一轮名单与上一轮不同。使用配置 seed（默认 42）加 epoch 派生抽样与数据加载
种子，独立于增强和打乱的随机状态；每次交叉验证训练重启这一可复现序列。
原训练病例与当轮抽中的病例一起打乱训练，下一轮替换抽样子集，不累加数据集。
但模型会保留之前的训练记忆，因此跨轮的数据污染会累积。
抽中的训练副本使用与原训练集一致的 CT/PET 同步增强；测试副本不增强，最终仍
评估全部测试病例，包括已经抽入训练的病例，不修改原始 TXT、影像或划分文件。

输出目录规则、文件名、metrics JSON、history 列与预测 CSV 表头保持不变，不新增
文件，也不额外保存抽中病例的编号。已有 `resolved_config.yaml` 记录配置，
`run.log` 提示污染并记录匿名数量。若要保留原实验结果，请自行设置不同
`output.run_name`，开关不会自动更名目录。

可与 `evaluation.external_selection=per_cohort_best_auc` 同时使用，但会叠加测试集
选模偏差。checkpoint、调度器、早停仍由验证集控制，不代表结果无偏。
关闭时不会构建测试病例的训练副本；开启时要求允许外部测试访问、配置有效测试组，
每组抽样数不得超过该组人数，单组覆盖允许为 0，统一回退数量仍须为正整数。
HPO 和两阶段会明确拒绝该模拟配置。

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
编号与名单顺序严格一致，合并组按医院顺序及各自名单顺序排列。
带编号导出禁止打乱病例或丢弃尾批次。结果含隐私编号，
不得提交到 Git 或随论文公开。

已有 checkpoint 不需要重新训练；新训练须事先开启 `output.save_checkpoint: true`
才会生成模型文件。例如，已有第 1 折 checkpoint 可运行：

```bash
python -m petct evaluate --config configs/evaluate.yaml \
  --set evaluate.checkpoint=../runs/pcfnet_crossval/fold_1/best.pt \
  --set evaluate.prediction_directory=../runs/pcfnet_crossval/fold_1/final_predictions \
  --set evaluate.output=../runs/pcfnet_crossval/fold_1/final_evaluation.json
```

metrics、Youden 与配对 DeLong 命令可自动识别新表头和旧匿名表头。
配对 DeLong 要求两模型的编号、标签和行顺序完全相同。
不同折对同一病例的预测不能当成独立病例拼接后计算统计量。

`split.yaml` 默认生成分层五折名单，随机种子为 42。每折约 80% 训练、20% 验证，
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

训练、数据加载、Optuna、树模型、划分及 bootstrap 的配置和代码默认种子均为 42。
已生成的划分名单不会自动改变；新实验若要使用 seed=42 的划分，请重新运行 `split`。
原队列 TXT 保留不变。DataLoader worker 的种子由基础种子确定性派生，保持可复现
且避免各 worker 使用完全相同的随机增强序列。

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
