# PPO-RS：时间相关变形约束下的柔性作业车间调度

数据目录与容量见 [data/README.md](data/README.md)：旧小集归档为 dataset_test，新版 dataset1／dataset2 均含静态和动态两部分。

这是一个可运行、可测试、可替换变形模型的首版工程骨架。系统使用事件驱动的 FJSP 环境统一调度普通加工、主动去应力和变形补偿，通过最小停留时间与变形硬阈值阻止不合格工件进入后续工序，并使用 PPO 学习时间—独立工艺成本之间的权衡。当前模型不单独计算残余应力：去应力将变形推进到工序饱和值并免除剩余自然停留，补偿负责修正已经形成的变形。

> `data/examples/` 和默认配置中的全部物理参数都是**合成测试数据**，只用于验证软件机制，不代表任何材料的真实变形规律。

## 已实现能力

- 有序普通工序、候选机器、机器相关加工时长、不可抢占加工。
- 同一资源账本管理普通加工、去应力和补偿工序，所有占用区间可追溯。
- 事件驱动自然等待；只有在“自然演化”和“主动工艺”需要策略取舍时，才提供事件级 `WAIT` 候选，不使用固定时间步。
- 每道普通工序具有独立的指数变形函数参数；去应力和补偿转移可通过构造环境时注入真实模型。
- 变形超限且要求补偿时，`NORMAL` 动作会从合法动作集中被硬性移除。
- 可变数量候选动作：Actor 对每个“状态 + 候选动作特征”共享打分，不依赖固定工件数、机器数或动作数。
- 可配置的 MLP Actor-Critic 与两层异构 Attention GNN Actor-Critic；共享 GAE、PPO clipped loss、value loss、entropy bonus、mini-batch、多 epoch、梯度裁剪和随机种子。
- 增量目标奖励、四种规则基线、模型保存/回载、JSON 指标、调度时间线、甘特图和训练曲线。
- 死端、最大决策数和最大仿真时间保护，不会在无合法动作时无限循环。

## 项目结构

```text
configs/                    训练配置（JSON 与 YAML）
data/examples/              独立的合成训练/测试实例
src/ppo_rs/
  algorithms/               GAE、rollout buffer、PPO
  baselines/                四种规则策略
  environment/              事件环境、机器资源账本、约束验证
  evaluation/               PPO 评估与可视化
  models/                   MLP 与 GNN Actor-Critic、图批处理和模型工厂
  physics/                  指数变形、去应力和补偿模型协议
  config.py                 实例配置与校验
  train.py / evaluate.py    命令行入口
tests/                      约束、物理、环境、奖励、PPO 测试
outputs/                    运行产物（默认不提交）
```

## 安装

建议使用 Python 3.10 或更高版本和独立虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

Linux/macOS 只需把激活命令替换为 `source .venv/bin/activate`。

CPU 可直接运行。若 PyTorch 检测到 CUDA，配置中的 `device: auto` 会自动选择 GPU；也可显式指定 `cpu` 或 `cuda`。

## 运行测试

安装测试依赖后：

```powershell
pytest -q
```

只依赖 Python 标准库测试运行器也可以执行（PyTorch 仍为必需）：

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

测试覆盖：工序先后与停留时间、机器冲突、机器资格、逐工序指数变形、去应力饱和稳定与停留豁免、补偿状态转移、质量硬约束、奖励一致性、动作掩码、完整 PPO 更新和死端终止。

## 训练

```powershell
python -m ppo_rs.train --config configs/default.yaml
```

不希望安装 PyYAML 时可使用等价 JSON 配置：

```powershell
python -m ppo_rs.train --config configs/default.json
```

默认输出到 `outputs/default_run/`：

- `model.pt`：Actor-Critic 权重和网络维度；
- `config.json`：实际训练配置；
- `training_history.json`：每个 episode 的目标、奖励与损失；
- `evaluation.json`、`schedule.json`：更新后策略在独立测试实例上的结果；
- `training_curve.png`、`gantt.png`：训练曲线和机器甘特图。

### MLP 与两层 Attention GNN

训练入口由配置项 `model_type` 选择模型，原 MLP 路径保持不变：

```json
{"model_type": "mlp", "hidden_dim": 128}
```

两层异构 Attention GNN 使用：

```json
{
  "model_type": "gnn_attention",
  "hidden_dim": 128,
  "gnn_layers": 2,
  "attention_heads": 4
}
```

图包含工件、工序、设备三类节点，以及工件归属、工序先后、候选设备、M3 和 M4 十类有向关系。各类型节点先由独立 MLP 编码到统一隐藏维度，再经过两层边特征感知的多头 Attention。Actor 联合使用工件/工序/设备节点嵌入、21 维动作特征和图级嵌入评分候选动作；Critic 使用三类节点池化、默认23维全局状态和共享图编码预测价值。实现只依赖 PyTorch，不要求 PyG。

小数据集 Pilot：

```powershell
python -m ppo_rs.train --config configs/dataset_gnn_pilot.json
```

dataset2 正式配置：

```powershell
python -m ppo_rs.train --config configs/dataset2_gnn.json
```

GNN 的图批次显存/内存开销高于 MLP，因此配置使用 `mini_batch_size=32` 和 `episodes_per_update=10`。Checkpoint 会保存 `model_type` 与完整 GNN 结构参数；历史 MLP Checkpoint 没有该字段时仍按 `mlp` 加载。

## 评估与基线

评估四种规则基线：

```powershell
python -m ppo_rs.evaluate `
  --instance data/examples/synthetic_test.json `
  --output-dir outputs/evaluation
```

同时评估训练后的 PPO：

```powershell
python -m ppo_rs.evaluate `
  --instance data/examples/synthetic_test.json `
  --checkpoint outputs/default_run/model.pt `
  --output-dir outputs/evaluation
```

基线包括：

- `ordinary_fjsp`：按传统作业/机器顺序生成派工优先级，不主动选择额外工艺；为保证公平和安全，仍经过同一个物理环境及硬质量约束安全层。
- `natural_wait`：受约束时优先自然等待。
- `stress_relief_rule`：受约束时优先主动去应力。
- `compensation_rule`：达到触发阈值时优先补偿。
- `ppo`：联合学习普通派工、自然等待、主动去应力和补偿。

## 配置与数据

实例文件支持 JSON/YAML，核心字段如下：

```json
{
  "machines": ["M1", "M2", "M3", "M4"],
  "jobs": [{
    "id": "J1",
    "operations": [{
      "id": "O1",
      "machines": {"M1": 2.0, "M2": 3.0},
      "deformation_scale": 1.0,
      "deformation_rate": 0.04
    }]
  }],
  "physics": {"minimum_deformation_ratio": 0.5},
  "stress_relief": {
    "enabled": true,
    "machines": {"M3": 0.65},
    "cost": 0.36,
    "max_per_job": 1,
    "allowed_after_operations": ["O1"]
  },
  "compensation": {
    "enabled": true,
    "machines": {"M4": 1.0},
    "cost": 0.3,
    "max_per_job": 2,
    "trigger_deformation": 0.20,
    "allow_original_machine": true,
    "original_machine_duration_factor": 1.0
  }
}
```

时间单位由业务统一指定（示例把它视为抽象时间单位）；同一实例内所有加工时长、停留时间和 `max_time` 必须一致。`cost` 是独立货币/资源成本，不包含已由完工时间体现的加工时长成本。

## 目标函数与奖励

环境使用：

```text
J = makespan_weight × C_max
  + relief_cost_weight × C_relief
  + compensation_cost_weight × C_comp
  + waiting_cost_weight × (inter_operation_idle_time × waiting_unit_cost)
```

每步奖励是部分目标增量的相反数：

```text
r_t = -(J_partial_after - J_partial_before)
```

因此在默认 `gamma = 1.0` 的有限时域中，episode 累计奖励严格等于 `-J_final`（初始目标为零），测试会验证这一点。默认等待成本权重为零，因为等待已经影响 `C_max`；只有存在独立在制品持有或延误成本时，才应同时启用 `waiting_cost_weight` 与 `waiting_unit_cost`，避免重复计费。

## 接入真实物理模型

环境构造函数接受三类物理依赖：

```python
env = DeformationFJSPEnv(
    instance,
    deformation_model=my_deformation_model,
    stress_relief_model=my_relief_transition,
    compensation_model=my_compensation_transition,
)
```

协议定义位置：

- `src/ppo_rs/physics/deformation.py`
- `src/ppo_rs/physics/stress_relief.py`
- `src/ppo_rs/physics/compensation.py`

真实模型至少需要明确：每道工序的指数尺度与增长率、变形单位、最小停留时间、去应力后的稳定饱和值、补偿触发阈值与处理后的残余变形、适用工艺位置、M3/M4 加工时长、独立成本及次数限制。如果变形依赖多道历史，应把足够的历史摘要纳入自定义变形模型，使转移保持马尔可夫性。

当前参考模型对完成工序 `k` 使用 `D(τ)=A_k(1-exp(-B_kτ))`。其中 `A_k>0` 是稳定饱和值，`B_k>0` 是增长速率，因此自然放置时变形随时间单调增加并趋近 `A_k`。主动去应力完成后直接令 `D=A_k`、免除下一工序的剩余最小停留，并保持稳定；补偿只能在自然停留完成或去应力完成后执行，补偿后的变形也保持稳定。配置中的 `deformation_scale` 对应 `A_k`，`deformation_rate` 对应 `B_k`，每道工序独立配置。

最小停留时间由 `physics.minimum_deformation_ratio` 统一确定，当前取0.5。对上一道工序有 `D/A=0.5`，因此环境动态计算 `t_min=-ln(1-0.5)/B=ln(2)/B`，实例文件不再接受人工填写的 `min_dwell_before`。

## 主要评估指标

结果包含 `makespan`、普通加工总时长、自然等待总时长、去应力次数/成本、补偿次数/成本、独立成本总和、综合目标值、变形超限事件、停留时间违规、逐机利用率、策略推理耗时，以及 episode 是否正常终止或因死端/上限而截断。只有 `terminated: true` 的调度才是完整可接受解；`truncated: true` 表示配置或策略进入了不可恢复状态或达到保护上限。
#   F J S P - P P O  
 
