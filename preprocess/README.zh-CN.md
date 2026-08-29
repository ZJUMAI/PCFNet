# PET/CT 预处理

预处理顺序固定为：

```text
resize_separate -> CT 窗位窗宽 -> CT/PET 配对切片 -> CLAHE
```

运行完整流程：

```bash
python -m petct preprocess all --config configs/preprocess.yaml
```

将 `all` 换成 `resize`、`window`、`slice` 或 `clahe` 可单独执行一步。目录中的旧脚本
现在只是薄入口，实际实现集中在 `petct.preprocessing`。

- 重采样会检查 CT、PET 及各自 mask 的配对和几何信息，统一为 RAI 方向；影像使用
  线性插值，mask 使用最近邻插值。
- CT 使用窗宽 1500 HU、窗位 -600 HU；PET 不套用 CT 窗口。
- 配对切片固定输出 CT/PET 各 64×64×64，病例目录直接使用 ID，不带 `_center`，
  文件编号为 `000.png`–`063.png`。
- 每一步生成不含病例 ID 的汇总；失败病例只按错误类型计数，不用伪造数据替代。
