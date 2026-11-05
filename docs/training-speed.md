# 缩短单组训练时间

本轮优化面向一组 MLP 或 GNN 同时使用多卡完成训练，保留现有网络、目标函数、数据、全局 batch64、每10回合更新、PPO4轮和20k回合预算。没有自动启动新训练或重启当前 MLP。

## 已实现

- **多卡验证**：每个 rank 验证全局实例索引 `rank, rank+world_size, ...`，100实例在8卡上约为每卡12～13个。保留原始实例顺序、`eval_seed+index` 和有效解判定；结果汇集后才选择最佳模型。验证异常同步传播，空分片可以参与。
- **PPO 日志通信减少**：各 mini-batch 的梯度同步照常；仅日志指标在设备上累计，每个 rollout 结束归约一次，避免每个 mini-batch 的额外归约与 CPU 读取。日志均值的浮点末位可能变化。
- **候选动作批量传输**：rollout 的可变长动作特征先在 CPU 填充，再整块传入设备，避免按每个决策单独传输。
- **GNN 图批处理**：先在 CPU 拼接图，再按字段传输；在 CPU 准备节点／边类型索引，正常前向使用索引，减少 GPU 上按类型判断的同步和布尔索引。无该类型时仍跳过对应模块，保持缺关系参数的 `grad=None` 行为。
- **动作结果一次返回**：MLP／GNN 的动作、log_prob、value 合并为一次 CPU 读取。仍然需要等待动作结果才能推进环境，未宣称彻底消除 CUDA 同步。
- **GNN 更新索引**：每个 PPO epoch 复制一次随机排列到 CPU，替代每个 mini-batch 单独复制图索引，保持原随机排列和样本选择。

模型参数键、checkpoint 结构和输入维度保持兼容；工程变更仍可能引入浮点末位差异。未关闭 `find_unused_parameters`，因为合法图可能缺少某些关系，现有回归测试明确覆盖该情形。

## 短基准

入口 `scripts/benchmark_training_speed.py` 创建独立唯一输出目录，保留输入配置文件与原训练结果。支持 CPU、单卡或 `torchrun` 多卡；默认2个更新块、前16个验证实例，只在末尾验证一次。学习率和熵系数固定为配置起始值，因此仅用于吞吐诊断，不能解释为短预算收敛实验。

参数：`--config`、`--output` 必填，`--updates` 默认2，`--validation-instances` 默认16；`--device` 覆盖设备，`--checkpoint` 可显式加载模型权重。不传 checkpoint 时从随机初始化开始，不会继承配置中的微调来源。

待当前训练完成、8卡可用后，可执行 GNN 示例：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/root/miniconda3/envs/datascience/bin/python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=8 \
  scripts/benchmark_training_speed.py \
  --config configs/dataset2_gnn.json \
  --checkpoint outputs/20261104_gnn_attention_jingtai_data2/checkpoints/best.pt \
  --updates 5 --validation-instances 16 \
  --output outputs/training_speed_benchmarks --device cuda
```

MLP 改用 `configs/dataset2.json` 和对应 MLP checkpoint；两种模型分别测量。基准按实际进程数覆盖副本中的 world size，不修改源配置。更换卡数的短基准使用权重初始化，不恢复原训练优化器和分布式随机数状态。

`benchmark.json` 保存总墙钟、环境 steps/s、回合/s、PPO更新总秒、末轮验证秒、验证目标、GPU小时及硬件／线程信息；配置和训练历史保存在同一运行目录。

总墙钟包含输入／模型初始化、文件写入及最终验证，不含启动 Python 之前的外部开销。默认不做独立预热，应多次重复并区分冷启动；验证实例数与更新次数必须一致。不同设备可能采到不同动作，从而出现不同决策数，应同时比较回合吞吐、steps/s 和实际采样量。对照优化前后版本时使用相同硬件、checkpoint、数据、随机种子和线程数，避免与正在运行的任务争抢 GPU。

## 当前验证与限制

- 完整 CPU 测试141项通过，包括两rank训练／续训、并行验证与串行结果对照、静态／动态、空分片和异常传播。
- GNN 前向和梯度与原计算对照通过，缺节点类型／缺关系的参数参与情况一致；模型状态字典不变。
- 短基准脚本已通过CPU单进程与双进程冒烟检查。
- 尚未在空闲8卡上实测整体加速比，不能把局部减少通信或8卡验证直接宣称为整轮训练8倍加速。
- 采样仍然是每rank串行环境、每步单图推理；完整图轨迹仍使用 `all_gather_object` 汇集。这两项是后续进一步优化方向，本轮未引入新的多环境或异步学习语义。
- 不需要改变原参数即可在后续训练中使用上述优化。已运行的Python进程不会热更新；当前MLP持续使用旧实现完成，避免把不同实现混进同一次实验。
