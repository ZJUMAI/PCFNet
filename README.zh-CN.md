# PET/CT 论文代码

本仓库是论文 **“Radiogenomic Signatures Derived from Pretreatment PET/CT Predict
Pathological Complete Response to Neoadjuvant Chemoimmunotherapy in Non-Small Cell
Lung Cancer”** 的官方代码仓库。

本仓库提供多模态 PET/CT 二分类研究的可复现实验代码，要求 Python 3.10+。
公开仓库不包含标签表内容、医学影像、患者级结果或旧 checkpoint，服务器数据路径
由 YAML 配置。

## 安装与运行

```bash
python -m venv .venv
python -m pip install -e ".[dev,hpo,two-stage]"
python -m petct split --config configs/split.yaml
python -m petct train --config configs/train.yaml
python -m petct hpo --config configs/hpo.yaml
python -m petct crossval --config configs/compare.yaml
python -m petct two-stage --config configs/two_stage.yaml
python -m petct preprocess all --config configs/preprocess.yaml
python -m petct evaluate --config configs/evaluate.yaml
```

默认划分与交叉验证均为分层五折，每折约 80% 训练、20% 验证。先运行 `split`
生成 `dataset/generated_splits_5fold` 名单，再运行 `crossval` 完成五折训练。
`train`、`hpo`、`two-stage` 仍为单次运行，默认读取第 1 折。
主模型五折运行方式见配置说明。

另有五次重复分层 7:3 划分（不是标准五折 K-fold），不同次的验证病例可重叠。
原五折默认方案和名单不改动。新增空间 token 模型配套配置可运行：

```bash
python -m petct split --config configs/split_70_30.yaml
python -m petct crossval --config configs/train_spatial_70_30.yaml
```

可重复使用 `--set key=value` 覆盖 YAML，例如
`--set training.optimizer.learning_rate=0.0002`。根目录原训练脚本仅为统一 API 的薄入口。

## 随机划分 seed 批量实验

`python run_seed_sweep.py --runs 6 --sampler-seed 42 --gpus 3 5 --parallel 2`
自动生成 6 个不同的划分 seed，每组先划分再训练五折，最多两组同时运行。
每组独立保存名单、日志和结果，不覆盖原名单。训练 seed 不变，继续使用 YAML
中的 seed（默认 42）。`--set` 传训练参数，`--dry-run` 仅预览、不启动训练。
完整参数及泄露模拟示例见[批量实验说明](configs/README.zh-CN.md#随机划分-seed-批量实验)。

## 数据与隐私约束

`metadata` 直接指向已有 CSV/XLSX 标签表，不要求另建文件或转成 CSV。服务器配置
已复用旧项目 radiomics Excel，ID 列为“影像组学序列号”，标签列为 `pCR`；临床和
radiomics 特征列不参与训练。manifest 每行一个病例
ID；CT/PET 根目录下必须存在同名病例目录，每个目录严格包含 64 张 64×64 灰度图。
缺失模态、未知标签、重复 ID、尺寸或层数错误都会显式报错。

训练集 CT/PET 通过同一个 TorchIO `Subject` 做同步空间增强；验证集和外部队列不增强。
HPO 不会构建或访问外部队列，也不保存逐例预测。当前服务器训练、两阶段和单独
checkpoint 测试配置已显式开启逐例 CSV，包含病例编号，结果文件不提交到 Git。
公共 API 仍默认关闭外部测试导出；`save_predictions` 控制外部测试导出，`save_prediction_ids`
控制是否包含编号，关闭后者时只保存匿名 `label,probability` 两列。

带编号 CSV 的五列表头为“影像组学序列号”“是否预测成功”“预测概率”“预测结果”
和 `ground truth`。“预测概率”是 pCR=1 的概率，不是预测类别的置信度；
预测结果以固定阈值 0.5 判定，预测成功表示结果等于真实标签。这三类标志均为数值
0/1，概率导出不进行四舍五入。

默认最终测试分三组：`multi_hospital`（wuhan、ruijin、FUSCC 三家合并）、
`france`、`SPH_test`。保留各医院原数据目录和 TXT，合并组统一计算所有病例
指标并输出一份预测表，不对三家医院指标取平均。

`train` / `crossval` 可显式设置
`evaluation.external_selection: per_cohort_best_auc`，让三组各自选择 AUC
最高的 epoch，并从同一轮保存指标和概率；输出格式不变。
该模式用测试集选模，会高估测试表现，不应作为论文无偏终评。
默认仍为 `validation_best`，验证集选 checkpoint 和早停的逻辑均不改变。

另提供默认关闭的 `training.leakage_simulation.enabled`，仅供数据泄露敏感性实验。
开启后，每个 epoch 从选定测试组重新随机抽取带标签病例加入训练，原 TXT 和完整
测试集不变。这些结果存在数据污染，不能作为无偏泛化性能。输出目录规则、文件名、
metrics 与 CSV 格式均不变，不新增文件；已有配置快照和日志记录模拟开关。
`training.leakage_simulation.cohort_samples` 可单独设置三组数量（0 表示不抽取），
未覆盖的组沿用 `samples_per_cohort`。抽样数量及队列选择方式见配置说明。

训练和交叉验证默认开启 `output.save_validation_predictions: true`，
未包含此字段的旧自定义配置同样生效。恢复验证集 AUC 最佳模型后，在每折输出
目录增加 `predictions_validation.csv`，保存其真实标签和预测概率，方便离线选择
阈值并重评对应测试概率。训练、选模、测试指标及 0.5 阈值均不改变。
可用 `--set output.save_validation_predictions=false` 关闭。HPO trial 默认仍不导出；
若 HPO 最佳配置显式包含 false，最终训练时需覆盖为 true。

原始训练集 score 也默认保存为 `predictions_train.csv`，开关为
`output.save_training_predictions: true`。使用验证集最佳模型，无增强、按名单顺序
预测原始训练 manifest；即使开启 leakage 模拟，也不会导出临时加入的外部病例。
表头和编号开关与现有预测文件一致，训练、选模和指标 JSON 不变。两阶段流程保存
最终树模型的训练概率；HPO trial 默认不导出。可将该开关设为 false 关闭。

新增实验模型 `fusion3d_spatial`，原 `fusion3d` 保留作对照。保留两条 R3D-18
backbone，取全局池化前的三维特征图，池化为 4×4×4 个区域，再投影为每模态
64 个 128 维 token。使用一层双向、4 头 Cross-Attention 与残差连接；更新后的
token 分别取均值、拼接，再由分类头预测。第一版不加门控、辅助头、额外 FFN
或位置编码，也不修改预处理。

```bash
python -m petct train --config configs/train_spatial.yaml
```

新配置沿用原数据、第 1 折名单、优化器和训练参数，独立输出到
`runs/fusion3d_spatial`。seed=42、三组测试、默认验证集选模、预测格式及模型
导出开关均保持不变。旧全局融合模型的 checkpoint 不能直接加载到新模型。
完整五折命令见配置说明；比较时使用相同划分。

默认 `output.save_checkpoint: false`，不保存 `best.pt`；验证集最佳权重保留在
CPU 内存中，训练结束后恢复，用于测试或两阶段特征提取。指标与预测输出不受影响。
需要保留模型供之后单独测试时，训练命令加 `--set output.save_checkpoint=true`。
已有 `best.pt` 不会自动删除，新训练也不会读取旧文件来恢复最佳权重。

预处理说明见 [preprocess/README.zh-CN.md](preprocess/README.zh-CN.md)，配置说明见
[configs/README.zh-CN.md](configs/README.zh-CN.md)。
