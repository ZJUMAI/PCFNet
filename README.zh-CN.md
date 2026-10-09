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

可重复使用 `--set key=value` 覆盖 YAML，例如
`--set training.optimizer.learning_rate=0.0002`。根目录原训练脚本仅为统一 API 的薄入口。

## 数据与隐私约束

`metadata` 直接指向已有 CSV/XLSX 标签表，不要求另建文件或转成 CSV。服务器配置
已复用旧项目 radiomics Excel，ID 列为“影像组学序列号”，标签列为 `pCR`；临床和
radiomics 特征列不参与训练。manifest 每行一个病例
ID；CT/PET 根目录下必须存在同名病例目录，每个目录严格包含 64 张 64×64 灰度图。
缺失模态、未知标签、重复 ID、尺寸或层数错误都会显式报错。

训练集 CT/PET 通过同一个 TorchIO `Subject` 做同步空间增强；验证集和外部队列不增强。
HPO 不会构建或访问外部队列，也不保存逐例预测。当前服务器训练、两阶段和单独
checkpoint 测试配置已显式开启逐例 CSV，包含病例编号，结果文件不提交到 Git。
公共 API 仍默认关闭导出；`save_predictions` 控制是否导出，`save_prediction_ids`
控制是否包含编号，关闭后者时只保存匿名 `label,probability` 两列。

带编号 CSV 的五列表头为“影像组学序列号”“是否预测成功”“预测概率”“预测结果”
和 `ground truth`。“预测概率”是 pCR=1 的概率，不是预测类别的置信度；
预测结果以固定阈值 0.5 判定，预测成功表示结果等于真实标签。这三类标志均为数值
0/1，概率导出不进行四舍五入。

预处理说明见 [preprocess/README.zh-CN.md](preprocess/README.zh-CN.md)，配置说明见
[configs/README.zh-CN.md](configs/README.zh-CN.md)。
