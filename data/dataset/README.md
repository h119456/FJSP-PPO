# PPO-RS 合成数据集

本数据集由 `scripts/generate_dataset.py` 使用固定随机种子 `20261028` 生成，主要用于验证多实例 PPO 训练流程，不代表任何具体材料的真实物理参数。

## 数据划分

- `train/`：100 个训练实例；
- `validation/`：20 个验证实例，用于Checkpoint选择；
- `test/`：20 个测试实例，只用于最终测试；
- `manifest.json`：数据文件清单。

每个实例包含6～10个工件，每个工件包含4～7道有序工序。M1、M2、M5、M6承担普通加工，M3承担去应力，M4承担变形补偿。每道普通工序随机具有2～4台可选加工设备，因此数据描述的是FJSP而不是JSP；每道工序还具有独立的 `deformation_scale` 和 `deformation_rate`。

自然放置时，完成工序 `k` 后的变形按照 `D(t)=A_k(1-exp(-B_k t))` 增长。M3去应力是自然停留的主动替代：处理完成后直接令 `D=A_k`，并免除下一工序的剩余最小停留时间。M4补偿只能在自然停留完成或M3完成后执行，负责降低已经形成的变形；补偿后的变形保持稳定。

当前合成数据只有一个补偿阈值 `trigger_deformation=0.30`。当变形严格大于0.30时，下一道普通工序被禁止，必须先在M4或原工序设备上完成补偿；工序饱和值A在0.18～0.45之间，因此M3只应选择在等待收益能够覆盖潜在强制补偿代价的场景。

普通加工时间为1.5～5.0。相对上一版，M3去应力时间统一增加30%，由0.2～0.6调整为0.26～0.78；独立成本统一增加20%，由0.1～0.4调整为0.12～0.48。M4或原机补偿时间为0.6～1.2、成本为0.1～0.4。目标函数中M3与补偿的成本权重均为0.5，因此单次M3的直接加权成本为0.06～0.24，单次补偿为0.05～0.20。

补偿既可以在专用M4上执行，也可以回到产生当前变形的上一道普通工序实际使用的机器上执行。默认原机补偿与M4使用相同的处理时间和成本；`original_machine_duration_factor`可用于表达原机补偿相对M4的时间差异。两类资源都进入统一机器账本并受不可重叠约束。

最小停留时间不再作为工序输入。环境统一使用 `minimum_deformation_ratio=0.5`，根据上一道已完成工序的增长率动态计算 `t_min=ln(2)/B`；因此达到最小停留时间时严格满足 `D(t_min)=0.5A`。当前数据的计算结果为2.781～8.654个时间单位。

环境语义版本为7。版本6删除了“补偿候选阈值+逐工序硬上限”的双阈值设计，改用单一强制补偿阈值；版本7进一步把全局State从21维精简到14维、候选动作特征从23维精简到21维，并修复已完工工件继续污染停留和变形统计的问题。旧Checkpoint不能用于当前语义，必须重新训练。

## 直接训练

在项目根目录运行：

```powershell
$env:PYTHONPATH = "src"
python -m ppo_rs.train --config configs/dataset.json
```

配置默认训练2000个episode，相当于每个训练实例平均被调度约20次。当前参数为：

- 学习率从 `3e-4` 线性下降到 `5e-5`；
- entropy系数从 `0.02` 线性下降到 `0.005`；
- 连续采集20个完整episode后执行一次PPO更新；
- mini-batch大小为64，PPO epochs为4；
- 每100个episode在20个验证实例上进行确定性评估；
- 每20个episode覆盖保存 `latest.pt`；
- 验证目标改善时覆盖保存 `best.pt`；
- 每500个episode保存一个永久快照；
- 训练结束保存 `final.pt`，同时保留兼容入口 `model.pt`。

Checkpoint位于 `outputs/dataset_run5/checkpoints/`。可以修改 `configs/dataset.json` 中的相应参数。

## 最终测试

训练完成后在20个未参与训练和验证的实例上评估。下面给出单个测试实例的命令示例：

```powershell
python -m ppo_rs.evaluate --instance data/dataset/test/instance_01.json --checkpoint outputs/dataset_run5/model.pt --output-dir outputs/test_01
```

## 重新生成

```powershell
python scripts/generate_dataset.py
```

固定随机种子保证重新生成的140个JSON文件保持一致。
