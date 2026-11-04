Doc-ID: PARAM-2026-11-04-001
Doc-Type: params
Owner: user
Date: 2026-11-04
Version: v1.1
Status: final
Source: 检查 dataset 上现有 MLP 与 GNN 完整方案对比的 state、训练参数与公平性
Activated-Skills: code-agent-core, parameter-sanity

## Summary

- 现有配置的全局 state/action 已统一为23/21维，但 MLP batch64/每20回合更新，GNN batch32/每10回合更新，训练条件不一致。
- 新增 `configs/dataset_compare_mlp.json`、`configs/dataset_compare_gnn.json`，统一为全局batch64/每10回合更新；每模型使用GPU 0–7进行8卡DDP训练，其余训练设置一致。
- 数据固定为100训练、20验证、20测试；验证集选 `best.pt`，测试集用于最终评估。
- 本实验比较现有完整方案：GNN 额外使用图信息且参数量约10.93倍，不能解释为等信息、等参数量的架构消融。

## Assumptions

- 用户选择比较现有 MLP 与 GNN 完整方案，保持各自模型结构。
- 使用 `data/dataset`，不重新生成或修改数据。
- 主对比采用相同 episode 预算；实际决策数、梯度步数与耗时可以因策略不同而变化，需单独报告。

## Open Questions

None

## Context

旧配置 `configs/dataset.json` 与 `configs/dataset_gnn_pilot.json` 均为2000 episodes、seed7、state23/action21；MLP 更新100次，GNN 更新200次。`dataset1_*state23_action21.json` 等已有配对配置也保留相同差异。batch32使相同决策数下的 Adam 步数约为 batch64 的两倍；因此原配置不能直接用于“相同训练参数”的比较。

### 输入与模型容量

| 项目 | MLP | GNN |
| --- | --- | --- |
| 全局 state | unified23，23维 | 同左 |
| 每候选动作特征 | compact21，21维 | 同左 |
| 完整图输入 | 无 | job/operation/machine，3类节点、10类有向关系 |
| 节点/边特征 | 无 | 每节点13维、每边2维，数量随实例变化 |
| Actor/Critic | 独立MLP，Critic只看全局state | 共享图编码，Critic看图池化和全局state |
| hidden_dim | 128 | 128 |
| 参数量 | 91,650 | 1,001,762 |

23维state由14维基础统计、5维负载/派工统计和4维动态故障统计组成。完整名称由 `src/ppo_rs/environment/fjsp_env.py` 的 `OBSERVATION_PROFILES` 定义，检查报告也会导出名称。静态dataset的后4维恒为零；GNN节点后2维故障特征同样为零。

GNN 图输入包含所有已知工件的全部工序、候选机器、加工时长、变形参数、顺序和运行状态，包括尚未加工的工序。这些是已知实例信息；MLP 接收聚合统计及当前候选动作，信息粒度不同。两边共享相同合法动作集、约束、物理模型、目标函数及奖励。

同hidden_dim不代表同容量。本次保留默认结构，以效果比较为主；如以后增加等容量辅助实验，MLP hidden_dim=440有1,001,442个参数，与GNN128相差约0.032%，但仍有输入信息和编码共享的差异。

## Parameter Table

以下默认值均指新增的配对配置，单位未注明时为无量纲。

| 参数 | 类型 | 统一值 | 有效范围/单位/约束 |
| --- | --- | --- | --- |
| train_instances | 路径 | data/dataset/train | 100个实例；两边相同 |
| eval_instances | 路径 | data/dataset/validation | 20个实例；只用于验证与选模型 |
| test_instances | 路径 | data/dataset/test | 20个实例；训练入口不读取，训练后显式评估 |
| observation_profile | 字符串 | unified23 | compact14/enhanced19/unified23；两边相同 |
| action_feature_profile | 字符串 | compact21 | compact21/enhanced25；两边相同 |
| hidden_dim | 整数 | 128 | 正数；GNN须能被attention_heads整除 |
| episodes | 整数 | 2000 | 正数，回合；本配置每训练实例恰好20次 |
| episodes_per_update | 整数 | 10 | 正数，回合；正式对比每边200次PPO更新 |
| mini_batch_size | 整数 | 64（全局） | 满批每卡8个决策；尾批以零权重填充，避免重复计权 |
| ppo_epochs | 整数 | 4 | 正数；每次rollout复用次数 |
| learning_rate / final_learning_rate | 浮点 | 3e-4 / 5e-5 | 正数；随episode线性下降 |
| entropy_coef / final_entropy_coef | 浮点 | 0.02 / 0.005 | 非负；随episode线性下降 |
| gamma | 浮点 | 1.0 | [0,1]；有限时域增量奖励累计对应最终目标 |
| gae_lambda | 浮点 | 0.95 | [0,1] |
| clip_ratio | 浮点 | 0.2 | (0,1) |
| value_coef | 浮点 | 0.5 | 非负 |
| max_grad_norm | 浮点 | 0.5 | 正数 |
| seed / eval_seed | 整数 | 7 / 1007 | 配对训练seed；评估seed保持固定 |
| validation_interval | 整数 | 100 | 回合；能被episodes_per_update整除 |
| latest_checkpoint_interval | 整数 | 10 | 回合；能被episodes_per_update整除 |
| checkpoint_interval | 整数 | 500 | 回合；能被episodes_per_update整除 |
| enable_plots | 布尔 | false | 两边相同，评测后可另绘图 |
| device | 字符串 | cuda | 每进程由LOCAL_RANK绑定GPU；CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 |
| distributed_world_size | 整数 | 8 | 实际进程数必须为8；全局batch64不乘8 |
| output_dir_mode | 字符串 | unique | 新训练在output_dir根下创建时间戳目录；resume复用原目录 |
| model_type | 字符串 | mlp / gnn_attention | 允许的架构差异 |
| gnn_layers / attention_heads | 整数 | GNN为2 / 4 | 当前实现要求恰好2层；MLP不使用 |
| output_dir | 路径 | /root/VLA/FJSP-PPO-cloud/outputs | 根目录；运行子目录名包含上海时间、模型、seed与随机后缀 |

优化器均为相同实现的 Adam。PPO minibatch每轮随机重排，优势在整次rollout内标准化。设某次rollout含S个决策，Adam步数为 `ppo_epochs * ceil(S / mini_batch_size)`；同episode不保证同S，因两种策略会选择不同的WAIT及主动工艺。

## Constraint Rules

1. 两边共享所有非架构训练设置；只有模型类型、GNN层数/heads及运行输出路径允许不同。mini_batch_size是跨8卡的全局值64，完整批次每卡8个样本。
2. 相同seed下，训练入口用独立Python随机发生器洗牌，实例顺序与环境seed一致；Torch初始化及动作采样流因架构不同而不同。相同seed不意味着逐步相同轨迹。
3. 每100回合在全部验证实例上确定性评估。仅当全部正常终止、未截断、约束违规为零，验证目标改善才保存 `best.pt`。两边均比较best，不能一边best另一边final。
4. `model.pt` 与 `checkpoints/final.pt` 均为最后一轮模型。测试集不参与调参、早停或Checkpoint选择。
5. 汇总测试结果先检查两边 `valid_instance_count == 20` 且 `total_schedule_violations == 0`，再比较目标值；失效调度的低目标不能当作更优结果。
6. 对比时保持相同的8卡数量、软件版本和运行方式。CUDA图操作可能存在浮点归约非确定性；断点续训不能据此承诺逐位一致。
7. 当前 `policy_inference_seconds` 不包含GNN图构造时间；比较端到端推理开销时需额外统一计时，不能将该字段解释为完整决策耗时。

## Preset Profiles

| 配置档 | 设置 | 用途 |
| --- | --- | --- |
| 冒烟检查 | 两边同10 episodes、同10回合更新、1 epoch、全部验证集 | 检查真实dataset链路，不解释效果 |
| 标准配对 | 两边同2000 episodes、4 epochs、seed7、8卡DDP、global batch64 | 首轮实验，两边各200次更新；不能据单seed定论 |
| 重复实验 | 标准配对，训练seed为7/17/27，eval_seed固定1007 | 建议效果结论至少使用3个配对seed，计算均值与样本标准差 |

所有档位均只使用已有参数。改变预算时必须同时改变两边，输出到独立目录；若继续调参，应给予两边相同搜索次数并只查看验证集。

### 标准配对命令

```bash
conda activate datascience
cd ~/VLA/FJSP-PPO-cloud
bash scripts/train_dataset_comparison.sh
```

脚本先校验两份配置，再顺序执行MLP和GNN，每个模型均同时使用GPU 0–7。等价的单模型命令为：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
  -m ppo_rs.train --config configs/dataset_compare_mlp.json
```

GNN只需替换为 `configs/dataset_compare_gnn.json`。每次新运行创建独立目录，终端打印 `run_directory=...`。在该目录保存 `config.json`、训练/验证历史、`checkpoints/best.pt`、`latest.pt`、`final.pt` 和每500回合快照；`model.pt` 为最后模型。

全局2000个episode只采集一次，按全局episode索引分配到8卡，并在每10个episode完成后汇总更新；预算不乘8。仅rank0进行验证和保存，所有rank同步模型梯度。

最终评估需把实际运行目录填入命令：

```bash
python scripts/evaluate_dataset.py \
  --instances data/dataset/test \
  --checkpoint mlp=/root/VLA/FJSP-PPO-cloud/outputs/<MLP运行目录>/checkpoints/best.pt \
  --checkpoint gnn=/root/VLA/FJSP-PPO-cloud/outputs/<GNN运行目录>/checkpoints/best.pt \
  --device cuda:0 --seed 1007 \
  --output /root/VLA/FJSP-PPO-cloud/outputs/<MLP运行目录>/test_comparison.json
```

### 衔接动态事件微调

两份动态配置 `configs/dataset_compare_mlp_dynamic.json` 和 `configs/dataset_compare_gnn_dynamic.json` 保持state23/action21、GNN节点13维、hidden128、global batch64和8卡设置。静态阶段学习率与原MLP多阶段配置相同，为3e-4到5e-5；动态阶段沿用原多阶段设置，为1e-4到2e-5，entropy为0.01到0.003。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
  -m ppo_rs.train --config configs/dataset_compare_mlp_dynamic.json
```

两份动态配置已通过 `initialize_from` 分别绑定正式静态目录 `20261104_mlp_seed7_jingtai` 和 `20261104_gnn_attention_seed7_jingtai` 的 `checkpoints/best.pt`。CLI 的 `--initialize-from` 可以覆盖配置中的路径；只加载模型权重，重新开始动态阶段的Adam、计数与学习率计划，并创建新目录；`--resume`则在原运行目录中恢复中断训练。静态阶段故障预留维度保持0，动态阶段由事件实际状态自然填值，无需改变网络输入尺寸。

统一入口为 `bash scripts/train_dynamic_comparison.sh check`，只执行数据集、参数配对、Checkpoint严格加载和8卡可见性检查。改为 `mlp`、`gnn` 可单独启动，改为 `both` 依次执行MLP/GNN，每个模型均使用全部8卡。默认Python为 `/root/miniconda3/envs/datascience/bin/python`，可用 `PYTHON` 环境变量覆盖。检查报告保存在 `outputs/dynamic_training/preflight.json`，实际启动时另外保存配置快照与日志。训练结果在 `outputs/dynamic_training/` 下按时间戳创建，静态目录保持可用。

### 结果汇报

以测试集每个实例的 objective 为主指标，越低越好；同时报告 makespan、独立成本、去应力/补偿次数、有效解数与违规数。每个seed先求20实例均值，再对3个seed求均值与样本标准差，勿将60个结果视作60次独立训练。记录Checkpoint中的 `total_environment_steps`、参数量、训练实际耗时与软硬件版本。两模型对同一实例的目标值可进一步做配对比较。

## Validation Notes

### 前次单卡检查（batch32，历史记录）

- `scripts/check_comparison.py` 对新配对配置检查通过，所有非架构训练设置无差异，两边计划各200次PPO更新；报告见 `outputs/dataset_comparison/fairness_check.json`。
- 按去除name后排序序列化的实例内容计算SHA256，100/20/20数据划分两两无重复内容；报告同时记录各集合摘要，便于运行后确认数据未变。
- `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q`：`45 passed in 3.64s`。覆盖旧配置不一致、新配对通过、数据改名重复识别、更新间隔错误、错误预算与clip范围、不同state/action维度Checkpoint的真实CLI评估。
- 两边均基于新配置在内存中统一覆盖 `episodes=10`、`ppo_epochs=1`、独立冒烟输出目录，其余保留state23/action21、hidden128、batch32和每10回合更新。在真实dataset训练10回合并验证全部20个实例，两边均正常终止且违规为零，`best.pt` 均生成。
- MLP冒烟产生1431个环境决策，GNN为1392，直观说明相同episodes不等于相同决策数。两边训练实例顺序相同，不能强行要求不同策略产生相同轨迹。
- 冒烟汇总为 `outputs/dataset_comparison/smoke/summary.json`；此轮仅1次PPO更新，不能据此判断哪种方法效果更好。正式2000回合和多seed效果实验尚未执行。

### 本次8卡配置验证

- 新静态配对配置均使用global mini-batch64、8卡DDP、学习率3e-4到5e-5；动态配对配置保留相同schema与batch，使用原多阶段动态学习率1e-4到2e-5。
- MLP和GNN均通过GPU 0–7的真实dataset10回合、1个PPO epoch短训练；各评估20个验证实例，正常终止且零违规，均生成best/latest/final Checkpoint。
- 两边实际Checkpoint确认world_size=8、global_mini_batch_size=64、per_rank_mini_batch_size=8；训练历史仅10条全局episode，collection_rank覆盖0–7，未将episode预算乘8。
- 静态dataset全部140实例检查了预留维度；state后4维、GNN节点后2维均为0。回归测试还覆盖静态实例的完整轨迹。
- 双进程CPU/Gloo验证了MLP和GNN全局梯度与单卡更新等价，覆盖无法均分的尾批次和未使用GNN关系；完整训练测试验证同步参数、学习率首尾值、唯一输出目录和resume。
- Checkpoint保存所有rank的CPU和CUDA随机状态；resume要求原world_size，模型参数名不带DDP包装前缀，普通单卡评估可加载。
- 两种模型均由各自静态best Checkpoint在8卡上完成2个动态微调episode；验证实例实际包含工件释放与设备故障，正常终止且零违规，学习率首尾为1e-4与2e-5，未改变state/action/GNN节点维度，输出到新目录。
- 全量回归测试 `54 passed in 12.57s`，包括rank0错误传播、迁移时语义/profile/GNN结构不匹配拒绝、2rank同步与恢复；shell语法检查和 `git diff --check` 通过。
- 实际运行目录及验证汇总：`outputs/ddp_setup_summary.json`；短训练仅验证工程链路，未执行正式2000回合实验。

---

Doc-ID: PARAM-2026-11-04-002
Doc-Type: params
Owner: user
Date: 2026-11-04
Version: v1.0
Status: final
Source: 基于两份正式静态Checkpoint准备MLP与GNN动态训练
Activated-Skills: code-agent-core, parameter-sanity

## Summary

- 动态配置分别绑定现存正式MLP/GNN的best权重，校验语义版本、输入结构、完整权重和来源SHA256。
- 两模型各2000全局episode、200次PPO更新、8卡、全局mini-batch64，每卡8。
- 已完成准备与检查，尚未启动正式动态训练。

## Assumptions

- 保持现有动态数据及其事件分布；沿用seed7，不增加随机种子。
- 静态模型目录已由用户重命名为带`jingtai`后缀的当前目录。

## Open Questions

None

## Context

原版 `joint_training_dynamic_finetune.json` 已同步为配对MLP动态配置，避免旧初始化路径失效和更新预算不一致。MLP来源为1900回合best，GNN来源为1800回合best；动态阶段重置优化器、学习率计划与计数。

## Parameter Table

| 参数 | 类型 | 当前值 | 范围、单位及依赖 |
| --- | --- | --- | --- |
| initialize_from | 路径字符串 | 各自`outputs/20261104_*_seed7_jingtai/checkpoints/best.pt` | 必须存在，模型类型及语义/profile/权重匹配 |
| output_dir / output_dir_mode | 字符串 | `outputs/dynamic_training` / `unique` | 每次新建目录；不复用静态结果 |
| episodes / episodes_per_update | 正整数 | 2000 / 10 | 全局episode；计划200次更新 |
| distributed_world_size / mini_batch_size | 正整数 | 8 / 64 | 全局batch须被world size整除 |
| learning_rate / final_learning_rate | 正浮点数 | `1e-4` / `2e-5` | 随更新次数线性衰减 |
| entropy_coef / final_entropy_coef | 非负浮点数 | 0.01 / 0.003 | 随更新次数线性衰减 |
| ppo_epochs | 正整数 | 4 | 每批收集数据的优化轮数 |
| validation_interval | 正整数 | 100 | 全局episode；应落在更新边界 |
| latest_checkpoint_interval / checkpoint_interval | 正整数 | 10 / 500 | 全局episode；应落在更新边界 |
| observation_profile / action_feature_profile | 字符串 | unified23 / compact21 | 必须与源Checkpoint一致 |

## Constraint Rules

- 两个模型均初始化或均从头训练；不允许一方初始化而另一方随机初始化。
- 允许来源路径和网络结构不同，其他训练设置必须匹配。
- 微调使用`initialize_from`，不能同时指定`resume`；恢复中断任务另用原运行的latest权重及原world size。
- 未到达的事件和工件仍对策略隐藏，动态训练不修改输入维度或事件规则。

## Preset Profiles

- 校验：`bash scripts/train_dynamic_comparison.sh check`，加载并检查源权重，不更新模型。
- 正式：`bash scripts/train_dynamic_comparison.sh both`，依次MLP、GNN，每个模型独占8卡；也可指定`mlp`或`gnn`。

## Validation Notes

- 实际`best.pt`严格加载成功，来源episode分别为1900与1800；预检结果位于`outputs/dynamic_training/preflight.json`。
- 140个动态实例校验通过，训练/验证/测试无重复内容，非架构训练参数无差异，8张CUDA GPU可见。
- 完整回归测试62项通过；shell语法与`git diff --check`通过。尚未运行正式动态训练或报告动态效果。

---

Doc-ID: PARAM-2026-11-04-004
Doc-Type: params
Owner: user
Date: 2026-11-04
Version: v1.0
Status: final
Source: 推荐dataset2统一训练参数，并澄清上一轮是否已统一
Activated-Skills: parameter-sanity

## Summary

- 上一轮dataset1正式静态配对与动态微调已统一各自阶段内的训练参数，历史结果不受dataset2旧配置差异影响。
- 推荐dataset2首先复用已验证的静态优化方案，保持全局batch64、每10回合更新、8卡，训练扩为20000回合。
- 本条仅推荐并在内存中检查，未修改训练配置、未启动训练。

## Assumptions

- 本推荐为dataset2静态训练的起始方案，不假设为性能最优参数。
- 主对比从头训练；若研究迁移，需单独定义双方一致的初始化协议。

## Open Questions

- dataset2上的实际训练吞吐、峰值显存和收敛预算需短跑验证，不能按dataset1耗时简单线性推算。

## Context

`dataset_compare_mlp.json`与`dataset_compare_gnn.json`已统一为2000回合、全局batch64、每10回合更新、8卡；动态配对同样统一，学习率与熵系数按动态阶段另设。
尚未统一的是较早的`dataset2.json`和`dataset2_gnn.json`，其batch分别128/32、每20/10回合更新。这与上一轮实际运行使用的配置不同。

## Parameter Table

| 参数 | 类型 | 推荐值 | 单位、范围及说明 |
| --- | --- | --- | --- |
| episodes | 正整数 | 20000 | 全局回合，1000训练实例平均约20遍；需根据验证曲线判断是否充分 |
| episodes_per_update | 正整数 | 10 | 两模型各2000次rollout/PPO更新 |
| mini_batch_size | 正整数 | 64 | 全局样本数；8卡满批每卡8 |
| ppo_epochs | 正整数 | 4 | 每批收集数据优化4遍 |
| learning_rate / final_learning_rate | 正浮点数 | 3e-4 / 5e-5 | 与上一轮静态阶段相同，线性衰减 |
| entropy_coef / final_entropy_coef | 非负浮点数 | 0.02 / 0.005 | 与上一轮静态阶段相同，线性衰减 |
| gamma / gae_lambda | [0,1]浮点数 | 1.0 / 0.95 | 与现有累计目标和GAE方案一致 |
| clip_ratio / value_coef / max_grad_norm | 正浮点数 | 0.2 / 0.5 / 0.5 | 保留已验证参数 |
| hidden_dim | 正整数 | 128 | GNN额外保持2层、4头；容量不相等 |
| observation_profile / action_feature_profile | 字符串 | unified23 / compact21 | 不变；GNN额外读取图 |
| validation_interval | 正整数 | 500 | 每次评估全部100验证实例，共40次 |
| latest_checkpoint_interval | 正整数 | 100 | 减少checkpoint/历史保存I/O；最多回退99个完成回合 |
| checkpoint_interval | 正整数 | 5000 | 固定快照；验证改善另存best |
| distributed_world_size / device | 整数/字符串 | 8 / cuda | 两模型顺序各使用8卡 |
| output_dir_mode / enable_plots | 字符串/布尔 | unique / false | 独立目录，训练结束后另生成对比曲线 |
| seed / eval_seed | 整数 | 7 / 1007 | 首次配对；之后增加训练seed17/27，固定评估协议 |

## Constraint Rules

- batch须被world size整除；验证和保存间隔须落在每10回合的更新边界，以上组合均满足。
- 相同episode和PPO更新次数不等于相同环境决策数、Adam步数或计算成本，仍需报告实际值。
- 若更改batch或收集回合数，双方同步更改；不根据最终测试集选择超参数。
- 保留100/200验证与测试划分，按验证集选择best后测试；训练seed需配对。

## Preset Profiles

- 推荐基线：上述batch64、每10回合更新，最便于与上一轮方法衔接。
- 吞吐候选：双方batch128、每20回合更新，其余同表；这是不同优化设定，先通过短跑和验证集比较，不能预设优于推荐基线。

## Validation Notes

- 已读取上一轮静态、动态配对与两份dataset2配置，确认差异属于旧dataset2配置。
- 以推荐值在内存中覆盖上一轮配对配置，运行audit_comparison检查实际dataset2：通过约束、数据分离、模型构建及两模型参数配对检查。
- 此检查不包含dataset2正式训练或吞吐测试；源配置未修改。

### dataset2正式启动参数落实（2026-11-04）

用户已授权创建固定目录并启动训练。`dataset2.json`与`dataset2_gnn.json`已落实上述推荐值；输出分别为`outputs/20261104_mlp_jingtai_data2`和`outputs/20261104_gnn_attention_jingtai_data2`，为直接写入指定目录，`output_dir_mode`改为`existing`。监督入口检查已有训练日志与checkpoint，避免重复启动覆盖。两模型分别20000全局回合、2000次PPO更新、global batch64/per-rank8、8卡，顺序为GNN后MLP。启动前真实配置配对校验通过。
