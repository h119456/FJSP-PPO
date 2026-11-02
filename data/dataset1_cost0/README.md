# dataset1_cost0

该数据集由`data/dataset`派生，仅进行以下修改：

- `stress_relief.cost = 0.0`；
- `compensation.cost = 0.0`。

所有加工时间、M3/M4占用时间、变形参数、补偿阈值和目标函数权重保持不变。原数据集不会被修改。

重新生成：

```powershell
python scripts/make_zero_cost_dataset.py
```
