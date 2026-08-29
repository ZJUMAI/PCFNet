# PET/CT 论文代码

本仓库是论文 **“Radiogenomic Signatures Derived from Pretreatment PET/CT Predict
Pathological Complete Response to Neoadjuvant Chemoimmunotherapy in Non-Small Cell
Lung Cancer”** 的官方代码仓库。

本仓库提供多模态 PET/CT 二分类研究的可复现实验代码，要求 Python 3.10+。
公开仓库不包含 metadata、医学影像、患者级结果、旧 checkpoint 或私有绝对路径。

## 安装与运行

```bash
python -m venv .venv
python -m pip install -e ".[dev,hpo,two-stage]"
python -m petct train --config configs/train.yaml
python -m petct hpo --config configs/hpo.yaml
python -m petct crossval --config configs/compare.yaml
python -m petct two-stage --config configs/two_stage.yaml
python -m petct preprocess all --config configs/preprocess.yaml
python -m petct evaluate --config configs/evaluate.yaml
```

可重复使用 `--set key=value` 覆盖 YAML，例如
`--set training.optimizer.learning_rate=0.0002`。根目录原训练脚本仅为统一 API 的薄入口。

## 数据与隐私约束

metadata 支持 CSV/XLSX，ID 列和二分类标签列由 YAML 指定。manifest 每行一个病例
ID；CT/PET 根目录下必须存在同名病例目录，每个目录严格包含 64 张 64×64 灰度图。
缺失模态、未知标签、重复 ID、尺寸或层数错误都会显式报错。

训练集 CT/PET 通过同一个 TorchIO `Subject` 做同步空间增强；验证集和外部队列不增强。
HPO 不会构建或访问外部队列。默认输出不含病例 ID 和逐例概率；只有显式设置
`output.save_predictions: true` 时，才保存不带 ID 的配对预测。

预处理说明见 [preprocess/README.zh-CN.md](preprocess/README.zh-CN.md)，配置说明见
[configs/README.zh-CN.md](configs/README.zh-CN.md)。
