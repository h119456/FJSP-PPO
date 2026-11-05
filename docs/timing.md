# 训练与验证耗时统计

所有 `*_seconds` 字段的单位都是实际秒，由 `time.perf_counter()` 计量。它们表示程序完成调度所需的计算时间；`makespan` 仍表示模拟车间中的加工完工时间，两者含义不同。

## 统计字段与范围

| 字段 | 含义 |
| --- | --- |
| `episode_wall_seconds` | 一个实例从 `env.reset()` 开始，到正常结束或截断、最终指标与排程约束检查完成的总耗时。包含状态与候选动作生成、GNN 构图、策略决策、环境推进；训练采样还包含轨迹写入 buffer。 |
| `policy_inference_seconds` | 本回合所有 `model.act()` 的累计耗时，包括输入张量转换、GNN 图批处理、Actor／Critic 前向及动作选择。规则方法对应规则选择动作的耗时。 |
| `graph_construction_seconds` | 本回合所有 `env.get_graph_state()` 的累计耗时。MLP 和规则方法为 0。已经包含在 `episode_wall_seconds` 中。 |
| `ppo_update_wall_seconds` | 一个 rollout 块对应的一次 `algorithm.update()` 总耗时，包含所配置的全部 PPO epoch／mini-batch，单独记录在该块最后一个训练回合的行中。多卡取各 rank 的最大值。 |
| `validation_wall_seconds` | 完整验证集的一轮验证耗时，包含环境构造、完整回合、排程提取及结果聚合。多卡按实例分片，包含结果汇集，各 rank 耗时取最大值。 |

单回合计时不含环境对象构造、实例／checkpoint 文件读取、回合结束后的 `get_schedule()`、跨 rank 的轨迹汇集与等待、PPO 更新、文件保存和绘图。完整验证集计时不含结果文件保存和绘图，包含并行验证内部结果通信与同步；并行回合可能重叠，不能将它们相加当作整轮耗时。

CUDA 模型在回合及 PPO 更新边界同步，避免未完成的设备计算使计时偏小。现有 `act()` 返回 Python 标量，返回前已等待相关 GPU 结果，因此决策内部不再额外逐步同步。CPU 与规则方法不执行 CUDA 同步。

## 结果保存位置

路径均相对于本次运行的输出目录：

- `training_history.json`：每个训练实例的三项回合计时，以及 `terminated`／`truncated`；每个 rollout 块最后一行另有 `ppo_update_wall_seconds`，其余行不重复记录更新耗时。
- `validation_history.json`：每次验证的三项**每实例平均**回合计时，以及整轮 `validation_wall_seconds`。
- `evaluation.json`、`best_evaluation.json`：顶层三项回合计时是验证集每实例平均值，整轮时长单独保存；`instances[].metrics` 保留各实例计时。`best_evaluation.json` 在验证目标改善时更新。
- `scripts/evaluate_dataset.py --output ...`：`instances[].policies` 保存各实例／策略计时，`aggregates` 中的 `mean_episode_wall_seconds` 等字段表示平均值，`total_episode_wall_seconds` 是逐实例回合耗时之和。
- `scripts/compare_observation_runs.py --output ...`：各 split／动态场景中的 `episode_wall_seconds` 等字段表示每实例平均值，`total_episode_wall_seconds` 表示回合耗时之和。

控制台在原有训练进度行中增加 `episode_seconds` 和 `ppo_update_seconds`；执行验证时另显示 `validation_seconds`。

## 使用与比较

正常完成的加工过程应筛选 `terminated=true` 且 `truncated=false`；验证／评估还应检查 `schedule_violation_count=0`。截断回合仍会记录其实际耗时，不能当作一次成功完工的速度结果。

训练采样使用探索策略、写入轨迹，验证默认使用确定性策略，两者耗时可能不同。跨 MLP／GNN 比较时，应统一数据、硬件、CPU 线程、设备、是否并行及其他计算负载，并报告平均值和离散程度。首次推理、CUDA 初始化或缓存预热可能影响结果；当前记录保留实际观测时间，不做预热扣除。

多卡收集的各回合可能同时进行，其耗时之和不代表训练实际经过的时间。评估脚本的逐回合总和也不包含外围开销，应与完整验证集的 `validation_wall_seconds` 区分。

旧 checkpoint 无需迁移，旧历史行保持原样。断点续训后仅新回合增加计时字段，无法补算历史耗时。代码修改不会热更新到已经运行的 Python 训练进程；新启动／恢复的训练及新执行的验证／评估会使用这些统计。
