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

---

Doc-ID: PARAM-2026-11-05-001
Doc-Type: params
Owner: user
Date: 2026-11-05
Version: v1.0
Status: final
Source: 统一数据目录并扩充dataset1，补齐dataset2动态数据；用户确认两套容量均为1000／100／200
Activated-Skills: parameter-sanity, code-agent-core

## Summary

- dataset_test保存原100／20／20的静态和动态数据；dataset1、dataset2的静态和动态均为1000／100／200。
- 保留原物理／时间／成本分布；新增动态数据自动适配生产机器池与参考时长。
- 原划分成员和旧文件字节不变，训练参数和正在运行的dataset2训练不变。

## Assumptions

- 容量指每个模式各自的实例数量；static和dynamic是配对场景，不是两份互不相干的数据。
- 原dataset1小集作为可追溯子集保留在扩容版对应split中。

## Open Questions

None

## Context (system/module)

`scripts/generate_dataset.py`、`generate_dataset2.py`、`generate_dynamic_dataset.py`与`organize_datasets.py`。

## Parameter Table

| 参数／字段 | 类型 | 默认／当前值 | 范围／单位 | 依赖 |
|---|---|---|---|---|
| 新版split_sizes | dict[str,int] | train1000、validation100、test200 | 正整数，实例 | 两模式同名文件配对 |
| dataset_test容量 | dict[str,int] | 100／20／20 | 历史固定 | 字节保留 |
| dataset1扩容seed | int | 20261105 | 随机种子 | split另加独立偏移 |
| dataset1动态seed | int | 20261103 | 随机种子 | 旧轨迹原样保留，新增按同协议 |
| dataset2静态seed | int | 20261030 | 随机种子 | 既有数据不重新生成 |
| dataset2动态seed | int | 20261105 | 随机种子 | 固定评估轨迹按split偏移 |
| scenario_probabilities | dict[str,float] | 无事件.65／插单.20／故障.10／混合.05 | 非负、总和1 | 固定评估按比例分层 |
| delayed_job_count_range | list[int] | 1～2 | 工件 | 最少保留1件初始可见 |
| 插单count_probabilities | dict | 1:.90／2:.10 | 概率 | 有插单的场景内条件分布 |
| event_count_range | list[int] | 1～2 | 次故障 | 故障机器不重复、限实际生产池 |
| 故障count_probabilities | dict | 1:.95／2:.05 | 概率 | 有故障的场景内条件分布 |
| release_time_fraction | list[float] | .05～.30 | 参考时长比例 | 每实例参考时长 |
| breakdown_time_fraction | list[float] | .15～.70 | 参考时长比例 | 每实例参考时长 |
| repair_duration_factor | list[float] | .50～1.50 | 平均加工时长倍数 | 每实例加工时间均值 |
| machine_pool | list[str] | 实例公开普通设备 | 4台或6～10台 | 排除M3／M4 |
| target_root | Path | dataset1/static或dynamic | 输出目录 | 非空时拒绝覆盖 |

## Constraint Rules

- 新版validation场景65／20／10／5，test场景130／40／20／10；旧集保持12／5／2／1。
- 隐藏事件不可见；维修采用pause_resume；训练重新采样、评估固定轨迹。
- 不跨split移动旧样本；去名称和事件后的基础实例内容在各split间查重。
- 不把dataset_test作为新版dataset1之外的独立泛化集；两者有明确保留关系。
- 训练episode、学习率、batch、保存间隔均不因本次数据扩容自动改变。

## Preset Profiles

- safe：dataset_test，100／20／20，用于代码检查和短实验。
- balanced：dataset1，1000／100／200，保持6～10工件的小规模问题分布。
- high：dataset2，1000／100／200，12～20工件及6～10台生产设备。

## Validation Notes

- 六个静态／动态子集容量与配对校验通过，旧数据SHA256核对一致。
- 固定与随机动态抽样共276次规则排程无违规；100项pytest通过。
- 同样1～2个事件在dataset2中相对扰动较低；事件强度未随规模暗中增加。

---

Doc-ID: PARAM-2026-11-05-002
Doc-Type: params
Owner: user
Date: 2026-11-05
Version: v1.0
Status: final
Source: 分别生成新版dataset1和dataset2的cost0静态／动态数据，额外费用为0、时间占用不变
Activated-Skills: parameter-sanity, code-agent-core

## Summary

- dataset1_cost0与dataset2_cost0各包含static/dynamic，每个模式1000／100／200。
- 仅实例stress_relief.cost及compensation.cost置0.0，时长、事件、目标权重不变。
- 原始有成本数据只读；逐实例源哈希与严格字段对照可验证。

## Assumptions

- cost0表示额外费用为零，操作仍占用设备且执行物理状态变换。

## Open Questions

None

## Context (system/module)

`scripts/make_zero_cost_dataset.py`从规范静态／动态family派生，不重新随机生成基础问题。

## Parameter Table

| 字段／参数 | 类型 | 当前值 | 单位／约束 |
|---|---|---|---|
| stress_relief.cost | float | 0.0 | 每次去应力额外费用 |
| compensation.cost | float | 0.0 | 每次补偿额外费用，包含原机／专用机 |
| source-root | Path | data/dataset1或dataset2 | 只读，含static/dynamic |
| target-root | Path | data/dataset1_cost0或dataset2_cost0 | 与源分离且非空时拒绝覆盖 |
| split_sizes | dict | 1000／100／200 | 每模式，与源逐文件配对 |
| 去应力时长 | float | 原值0.26～0.78 | 时间单位，不修改 |
| 补偿时长 | float | 原值0.6～1.2 | 时间单位，不修改 |
| 目标权重 | dict | makespan1.0、两辅助费用各0.5 | 不修改 |

## Constraint Rules

- 只允许两费用字段变化；所有实例名称、文件名、split、工序、物理参数及动态事件完全相同。
- manifest的费用范围对应更新为[0,0]，保留源manifest哈希和实例哈希。
- 当前等待费用为0，正常完成时目标等于makespan；不能据此删除去应力或补偿的时间占用。
- 不改变正在运行的有成本训练，不自动启动cost0训练。

## Preset Profiles

- safe：先用临时目标和源样本验证，只读源，不覆盖已有数据。
- balanced：dataset1_cost0，小规模1300静态＋1300动态。
- high：dataset2_cost0，大规模1300静态＋1300动态。

## Validation Notes

全部5200派生实例严格字段对照通过；静态／动态相同行为序列的时钟和排程除费用外一致。

---

Doc-ID: PARAM-2026-11-05-003
Doc-Type: params
Owner: user
Date: 2026-11-05
Version: v1.0
Status: final
Source: 生成dataset3与dataset3_cost0，18～30工件、6～13工序、12～18设备，其余保持一致
Activated-Skills: parameter-sanity, code-agent-core

## Summary

- dataset3及cost0各含static/dynamic，各模式1000／100／200。
- 总设备12～18台，含M3／M4，普通设备10～16台。
- 除规模及相应辅助操作允许位置／次数外，沿用dataset2工艺、费用、目标及环境参数。

## Assumptions

- “设备数”沿用前文口径，指包含两台辅助设备的总数。
- 工序数为每工件普通加工工序数量。

## Open Questions

None

## Context (system/module)

`scripts/generate_dataset3.py`复用dataset2实例生成逻辑，动态和cost0复用既有通用生成器。

## Parameter Table

| 参数 | 类型 | 默认／范围 | 单位／依赖 |
|---|---|---|---|
| split_sizes | dict | 1000／100／200 | 每个static或dynamic模式 |
| job_count_range | list[int] | 18～30 | 工件，边际均衡 |
| operation_count_range | list[int] | 6～13 | 每工件普通工序 |
| total_machine_count_range | list[int] | 12～18 | 含M3、M4 |
| normal_machine_count_range | list[int] | 10～16 | 普通加工设备 |
| candidate_machine_count_range | list[int] | 2～5 | 单工序候选，沿用dataset2 |
| 辅助max_per_job | int | 12 | 13道工序对应最多12个工序间隙 |
| allowed_after_operations | list[str] | O1～O12 | 去应力及补偿允许位置 |
| 静态seed | int | 20261106 | 独立split偏移 |
| 动态seed | int | 20261107 | 固定评估轨迹；train按reset种子 |
| 普通时长／变形A／B | float | 1.5～5.0／.18～.45／.080～.250 | 原分布 |
| 去应力费用／时长 | float | .12～.48／.26～.78 | cost0费用为0，时长不变 |
| 补偿费用／时长 | float | .1～.4／.6～1.2 | cost0费用为0，时长不变 |
| 环境max_decisions／max_time | int／float | 5000／1000 | 沿用dataset2 |
| 动态场景概率 | dict | .65／.20／.10／.05 | 无事件／插单／故障／混合 |

## Constraint Rules

- 同split和文件名的static／dynamic以及cost0逐实例配对，不交叉混入train／test。
- 动态事件仍每类1～2次，参考时长与机器池按实际规模计算。
- cost0仅修改两费用，不改变任何实例时长、设备占用、事件或物理参数。
- 设备清单中的M3／M4不加入普通工序候选，故障只针对普通设备。
- 新增生成器拒绝覆盖已有非空目录；共享dataset2默认生成结果保持完全一致。

## Preset Profiles

- safe：利用split_sizes参数生成临时小容量验证数据，规模范围不变。
- balanced：dataset3/static或dynamic，各1300实例。
- high：dataset3和dataset3_cost0配对比较，包含两模式共5200实例。

## Validation Notes

完整容量、范围、配对、内容查重、固定场景及cost0逐字段校验通过；全量dataset2原实例生成复现确认兼容。

---

Doc-ID: PARAM-2026-11-05-004
Doc-Type: params
Owner: user
Date: 2026-11-05
Version: v1.0
Status: final
Source: 参考本次GNN训练评估早停／加速，以及GPU相对CPU提升有限的原因
Activated-Skills: parameter-sanity

## Summary

- 本次GNN共20000回合、6928907步、21小时11分钟，最佳验证目标60.5911出现在19500回合。
- 截取原曲线至10000／16000回合，最佳目标比全程最佳分别差3.44%／0.95%，不能认定早期已充分收敛。
- 8卡全局batch64使每卡仅8样本；单图推理、GPU标量同步、CPU构图、Python轨迹汇集及rank0串行验证限制吞吐。
- 建议优先做保持学习语义的工程优化，再用同条件CPU／1／2／4／8卡短基准选择资源；本轮不改配置或当前训练。

## Assumptions

- 结论来自dataset2静态、seed7这一次训练，验证集100实例，目标越低越好。
- 旧运行无完整episode／更新／整轮验证计时字段，亦无同设置CPU／GPU基准；代码可定位开销来源，不能精确归因耗时百分比。

## Open Questions

- 批量推理、缓存、减少同步和不同卡数的实测收益，需使用新增计时与短基准确认。

## Context (system/module)

读取`outputs/20261104_gnn_attention_jingtai_data2`的固定配置、训练／验证历史和状态，以及训练、图网络与图批处理代码。本轮仅分析和记录建议，不实现早停、不启动训练。

## Parameter Table

| 现有参数 | 类型 | 本次值 | 候选／单位／依赖 |
|---|---|---|---|
| episodes | int | 20000 | 正式参考保持20000；开发可考虑16000回合，需重新评估解质量 |
| validation_interval | int | 500 | 保持500回合；须与episodes_per_update整除关系兼容 |
| distributed_world_size | int | 8 | 短基准比较1、2、4、8，与实际进程数一致 |
| mini_batch_size | int | 64 | 全局样本数；1／2／4／8卡分别64／32／16／8样本每卡 |
| episodes_per_update | int | 10 | 全局回合数；8卡分配2、2、1、1、1、1、1、1，各卡回合大小也不同 |
| ppo_epochs | int | 4 | 每个rollout的优化轮数，先保持，减少会改变学习过程 |
| hidden_dim | int | 128 | 先保持；缩小改变容量，不属于保持架构的工程加速 |
| gnn_layers／attention_heads | int | 2／4 | 代码要求两层；hidden_dim必须能被heads整除 |
| learning_rate／final_learning_rate | float | 3e-4／5e-5 | 随总回合预算线性变化，改episodes也会改变途中学习率 |
| entropy_coef／final_entropy_coef | float | .02／.005 | 随总回合预算线性变化，改episodes也会改变途中探索强度 |

## Constraint Rules

- 当前未实现early-stop配置字段；不得仅向JSON加入不存在的参数并声称已生效。
- 可供后续实现的保守停止规则：至少10000回合、每500回合验证，显著改善定义为相对0.1%，连续8次验证无显著改善再停，并始终保留真正最佳的有效checkpoint。该规则回放本次曲线不会提前停止。
- 激进规则“至少5000回合、连续4次验证无任何改善”会在9000回合误停，当时最佳64.0376，比全程最佳差5.69%。
- 停止判断仅使用固定验证集，不使用测试集；MLP／GNN正式对比应统一预算或使用预先规定的同一停止规则，并报告实际计算量。
- 本次16000回合截断点的0.95%差距是单seed事后观察。直接把episodes改为16000会同时压缩学习率／熵系数日程，不保证复现原曲线同一回合的表现。
- 参数更新、GPU卡数变化可改变浮点归约和训练轨迹；扩大batch、改变episodes_per_update或减少ppo_epochs会进一步改变优化过程，应独立验证。
- 实际launch进程数与distributed_world_size必须一致；速度比较统一数据、线程设置、预热方式和外部计算负载。

## Preset Profiles

- safe／正式参考：原20000回合、batch64、10回合更新、PPO4轮、验证500；使用best.pt。用于保持本次预算口径。
- balanced／开发预算：可考虑16000回合，其余相同，接受重新验证质量及日程变化。减少20%回合不等于已测得减少20%墙钟时间。
- throughput-study／资源选择：先保持20000回合日程和其他优化参数，以短跑比较CPU及1／2／4／8卡的相同工作量；不得假定8卡最快。短跑只用于吞吐诊断，不作为收敛质量结论。

## Validation Notes

- Python读取完整40次验证记录并回放两种停止规则；截取10000／15000／16000／19000回合的最佳值为62.6761／61.4391／61.1688／60.6769，相对最终最佳差3.44%／1.40%／0.95%／0.14%。
- 由2000个更新块的rollout_steps计算，4轮PPO合计437032个全局mini-batch，每批触发DDP梯度同步和额外指标归约。
- 每500回合验证100实例，共40轮4000实例；最后一轮policy_inference_seconds均值5.0257秒，仅策略部分100实例合计502.57秒，不能据此准确回推全程占比。
- 未运行新的CPU／GPU基准、未改训练源代码／配置、未中断当前MLP；无需为本次分析重跑实现测试。

---

Doc-ID: PARAM-2026-11-05-005
Doc-Type: params
Owner: user
Date: 2026-11-05
Version: v1.0
Status: final
Source: 如何充分利用8张GPU加快MLP与GNN训练
Activated-Skills: parameter-sanity

## Summary

- 8卡加速单次训练与8卡提高整组实验吞吐是不同目标；小模型和全局batch64下，应先比较多任务单卡／双卡方案。
- 保持现有优化参数时，优先并行验证、减少GPU标量同步、缓存固定图信息和合并CPU到GPU的数据搬运。
- 多环境批量采样需要改代码，并正确维护独立轨迹和on-policy边界；当前全局10回合无法支撑每卡多个完整回合的大批量供数。
- 8卡batch256／每40回合更新可作为新吞吐候选，不能保证解质量不变；本轮未更改配置或当前运行。

## Assumptions

- 当前机器有8张A800 80GB，正在运行的MLP仍使用8卡，建议实施于后续实验。
- 训练目标、网络、state与动作特征保持当前口径；新参数仅为待测候选。

## Open Questions

- CPU／1／2／4／8卡同工作量短基准，以及多任务总吞吐尚未实测，不能预报加速倍数。

## Context (system/module)

当前采样每rank串行运行分配到的完整回合，每步单图推理；每10个全局回合进行一次PPO更新，完整轨迹用all_gather_object复制到所有rank；验证仅rank0顺序执行100实例。MLP约9.2万参数，GNN约100万参数。

## Parameter Table

| 现有字段 | 类型 | 当前值 | 推荐比较值／约束 |
|---|---|---|---|
| distributed_world_size | int | 8 | 单实验1、2、4、8；必须与实际进程数一致 |
| mini_batch_size | int | 64 | 全局值；新8卡吞吐候选256，对应每卡32 |
| episodes_per_update | int | 10 | 保持口径先不改；新8卡候选40，对应每卡5完整回合 |
| episodes | int | 20000 | 保持，40回合块下为500个更新块 |
| ppo_epochs | int | 4 | 先保持，减少轮数会改变学习过程 |
| validation_interval | int | 500 | 原参数方案保持500；40回合块候选改1000 |
| latest_checkpoint_interval | int | 100 | 原方案100；40回合块候选200 |
| checkpoint_interval | int | 5000 | 两方案均兼容 |
| seed | int | 7 | 独立任务使用预先指定的多个seed，两种模型配对 |
| output_dir | Path | 两固定运行目录 | 新任务使用独立新目录，禁止覆盖当前结果 |

## Constraint Rules

- 每卡64样本等于8卡全局512，不能将其当作当前全局64的无损并行化。
- 每40回合更新时，验证1000、latest200和snapshot5000均为40的倍数；原验证500和latest100不兼容此更新边界。
- 同轨迹量下batch256相比64减少约四分之三mini-batch优化步数；rollout40相比10降低策略更新频率，因此必须重新比较验证质量，双方采用一致训练协议。
- 先保留学习率／熵系数及其日程，不根据卡数机械放大学习率；20k预算保持不变。
- 同步多环境采样需在整个采样块固定策略版本，分别保存各轨迹并计算GAE，再组成全局更新；不能将交错环境步当作连续轨迹，也不能将8卡预算乘8。
- 并行验证固定实例与seed，汇总保持原顺序、均值和可行性判定。验证前后模型／随机数状态的影响应明确处理；设备变化可能产生数值差异，不承诺逐位一致。
- 现有checkpoint恢复要求原world size；改卡数应启动独立新运行，不能直接以不同world size恢复当前分布式checkpoint并假定等价。
- 环境多进程要限制每进程CPU线程并实测供数速度；不可只提高OMP线程数并期待Python循环同比加速。
- nvidia-smi利用率含内核运行／通信等待，NCCL等待也可能显示高值，不能只据此判断有效算力利用。

## Preset Profiles

- safe／实验吞吐：8个独立单卡任务，distributed_world_size=1，每任务batch64、rollout10、PPO4轮、验证500。可安排MLP4seed＋GNN4seed；先短跑确认单卡吞吐和CPU／内存承载，未保证整组最快。
- balanced／实验吞吐：4个独立双卡任务，每任务world_size=2，其余优化参数同safe；与单卡组对比单位时间完成的独立实验数量，而非只看单任务延迟。
- high／单任务8卡新方案：world_size=8、batch256、rollout40、PPO4轮、验证1000、latest200、snapshot5000、episodes20000；须配合并行验证及批量采样工程实现，并作为新的优化设定验证。

## Validation Notes

- 代码审阅确认全局batch64的满批在8卡各8样本；10回合静态轮转分配为2、2、1、1、1、1、1、1，存在负载不均。
- 代码确认网络内部torch.any的Python分支、mask.sum().item及逐步act返回标量导致CUDA同步；图批处理和rollout张量化多次逐项传输。
- Python算术检查：20000／1000／200／5000均能被40整除，256能被8整除。
- 推荐短基准报告采样steps/s、PPO更新秒、整轮验证秒、总墙钟、GPU小时和验证目标；CPU/GPU及不同卡数组使用相同数据和工作量，计入预热策略与共同负载。
- 本轮只检查代码、配置及瞬时资源信息，未运行新的基准或更改训练代码，无需重跑实现测试。

---

Doc-ID: PARAM-2026-11-05-006
Doc-Type: params
Owner: user
Date: 2026-11-05
Version: v1.0
Status: final
Source: 实施保持训练超参数的单组多GPU工程加速及独立短基准
Activated-Skills: parameter-sanity, code-agent-core

## Summary

- 正式训练的batch64、rollout10、PPO4轮、20k预算、学习率和熵系数不变。
- 多卡验证及同步／搬运优化默认生效，不引入新的训练调参项。
- 新短基准仅使用配置副本，创建独立输出，用于计量吞吐，不用于判断收敛。

## Assumptions

- 基准数据、checkpoint、线程和外部负载统一后才比较速度；不与当前8卡MLP争抢设备。

## Open Questions

空闲8卡整体加速比待测。

## Context (system/module)

`scripts/benchmark_training_speed.py`复用train，支持CPU、单GPU及torchrun；详细说明见docs/training-speed.md。

## Parameter Table

| CLI参数 | 类型 | 默认／范围 | 单位与依赖 |
|---|---|---|---|
| config | Path | 必填 | 只读源配置 |
| output | Path | 必填 | 新建唯一基准目录的父路径 |
| updates | int | 2，正整数 | 更新块；episodes=updates×原episodes_per_update |
| validation-instances | int | 16，正整数 | 实例；固定验证路径前缀，不足则用全部 |
| device | str | 默认源配置 | cpu／cuda等，必须与可用硬件一致 |
| checkpoint | Path或None | None | 只加载权重，不恢复optimizer或计数 |

## Constraint Rules

- 基准world size取实际进程数，配置副本中的distributed_world_size同步，源文件不变。
- 基准validation／latest／snapshot间隔等于短跑总回合，只在结尾验证和保存；PPO epoch、全局batch、rollout与架构保持源配置。
- 基准学习率与熵系数固定在源配置的起始值，不将原20k衰减日程压入几回合；不能据此推断正式训练效果。
- 原配置initialize_from移除，只有显式--checkpoint生效；输出独立，不能覆盖原结果。
- 总耗时包括train内部初始化／最终验证／文件写入，默认无独立预热，比较时重复并一致处理冷启动。
- 正式训练恢复仍要求原world size；基准不同卡数使用新运行和可选权重初始化。

## Preset Profiles

- safe／入口检查：CPU、updates1、validation-instances1，使用小例子，验证端到端与报告。
- balanced／快速诊断：实际8卡、updates5、validation-instances16，两模型分别测试。
- high／稳定性复核：相同设置updates10、validation-instances100，重复测试并报告离散程度，避免外部GPU负载。

## Validation Notes

完整141项CPU测试通过；脚本单／双CPU进程冒烟通过；未测空闲8卡性能，当前MLP正常运行。

---

Doc-ID: PARAM-2026-11-05-007
Doc-Type: params
Owner: user
Date: 2026-11-05
Version: v1.0
Status: final
Source: 完整全局19维／局部25维的state对照试验如何简化
Activated-Skills: parameter-sanity

## Summary

- 推荐现有14／21作为基础，围绕19／25做全局与候选动作增强的2×2消融。
- 全局去掉5项负载／机器可用性／立即派工统计，局部去掉4项相对速度／剩余工作量／候选压力／剩余放置时间摘要。
- 保留工艺、变形、放置、辅助操作和合法动作机制；四组均从头训练、统一数据和预算。
- 当前dataset2运行是23／21，静态后4维恒零，仍不是19／25完整输入组。

## Assumptions

- “局部25维”指每个候选动作的action_features，GNN另有节点／边输入。
- 首先在dataset2静态有成本条件下隔离输入增强贡献，不同时改变成本或事件分布。

## Open Questions

None

## Context (system/module)

环境现有compact14／enhanced19全局profile、compact21／enhanced25候选profile；无需新增维度定义即可构造四组，但14／25暂没有独立现成配置文件。

## Parameter Table

| 组别 | observation_profile | action_feature_profile | 维度／解释 |
|---|---|---|---|
| 基础 | compact14 | compact21 | 14／21，两类增强均去掉 |
| 仅全局增强 | enhanced19 | compact21 | 19／21，基础＋全局5项 |
| 仅局部增强 | compact14 | enhanced25 | 14／25，基础＋候选4项 |
| 完整 | enhanced19 | enhanced25 | 19／25，两类增强均保留 |

全局删除：remaining_processing_load_ratio、max_job_remaining_load_ratio、max_machine_availability_gap、machine_availability_gap_std、dispatchable_job_ratio。候选删除：relative_fastest_duration_score、job_remaining_workload_ratio、machine_candidate_pressure、normalized_remaining_dwell_time。

## Constraint Rules

- 保留14维中已有进度、忙闲、就绪、放置、变形、去应力／补偿比例；21维中已有动作类型、时长、成本、工件进度、物理参数及辅助动作结果特征。此实验检验额外摘要贡献，不是完整去掉物理信息。
- 相同模型内只改变选定输入组，其余数据、种子、网络宽度／层数、PPO参数、训练预算、checkpoint选择和测试集不变。各条件从头训练，至少3个配对训练seed。
- 全局14／19的GNN节点维度均11，图输入通道应保持相同。GNN可从图中恢复部分被删摘要，不能将结果解释为彻底剥夺对应信息。
- 直接删维会改变首层参数量，应报告参数量。严格等容量可保持19／25形状、将对应5／4维固定置零，并重新训练四组；该mask方案目前尚未实现为配置项。
- 不修改合法动作、约束、安全筛选、物理模型或奖励；mask仍可能隐式透露约束信息，各组必须一致。
- 本次dataset2两模型unified23／compact21，静态后4全局及2节点维为零；其信息类似19／21，但输入形状／参数量不同，不能直接充当19／25或精确19／21消融。
- 后续动态对比应给基础与增强组同时保留同样4个动态全局字段（信息维度18／23）及同样图动态节点字段；需实现相应profile，现代码没有compact18。不能只给完整组事件信息。

## Preset Profiles

- safe／最低预算：14／21对19／25，只能判断联合增强收益，无法分离全局与局部贡献。
- balanced／推荐：四组2×2，每个MLP／GNN至少3seed，固定dataset2静态有成本；共8种条件、24次训练。
- high／容量控制：保持19／25形状，四组按信息组置零，至少3seed；固定GNN图输入，明确为增强特征可用性消融。

## Validation Notes

只读核对环境字段定义、get_observation／get_action_features和现有配置；两类增强分别是末5维与末4维。未修改代码／配置或启动实验。

---

Doc-ID: PARAM-2026-11-05-008
Doc-Type: params
Owner: user
Date: 2026-11-05
Version: v1.0
Status: final
Source: 基础state训练代码与动态预留、完整组统一训练配置
Activated-Skills: parameter-sanity, code-agent-core

## Summary

- compact18动态接口现已实现，基础实际18／21、完整23／25，静态有效14／21和19／25。
- 两组GNN节点同13维；静态后4全局和后2节点为零，动态阶段填入相同公开事件信息。
- 8份统一配置与launcher可用于dataset_test、dataset1／2／3及cost0，默认dataset2有成本。

## Assumptions

- 不修改当前实验，新增训练独立输出；现有训练预算20k作为新静态对照默认。

## Open Questions

None

## Context (system/module)

`configs/state_ablation/`与`scripts/train_state_comparison.py`，细节见docs/state-ablation.md。

## Parameter Table

| 参数 | 类型 | 静态／动态默认 | 约束与依赖 |
|---|---|---|---|
| observation_profile | str | base compact18；full unified23 | 同组两阶段不变 |
| action_feature_profile | str | base compact21；full enhanced25 | 同组两阶段不变 |
| episodes | int | 20000／10000 | 全局回合，不乘卡数 |
| learning_rate→final | float | 3e-4→5e-5／1e-4→2e-5 | 同阶段两组两模型一致 |
| entropy_coef→final | float | .02→.005／.01→.003 | 同上 |
| mini_batch_size | int | 64 | 全局，8卡每卡8 |
| episodes_per_update／ppo_epochs | int | 10／4 | 两阶段一致 |
| validation／latest／snapshot间隔 | int | 500／100／5000 | 均可被10整除 |
| hidden_dim；gnn_layers／heads | int | 128；2／4 | GNN宽度可被heads整除 |
| world-size／device | int／str | 8／cuda | --check仅CPU预检，启动时才检查设备 |
| seed／eval_seed | int | 7／1007 | --seed只改训练seed，评估固定 |
| group／phase／model | str | 必填base或full／必填static或dynamic／both | 单模型mlp或gnn；both依次各用全部卡 |
| dataset | str | dataset2 | 可选dataset_test及dataset1／2／3、对应cost0 |
| output-root | Path | outputs/state_ablation | 数据／组／阶段／模型／seed下唯一运行目录 |
| initialize-from／static-run | MODEL=Path | 动态显式提供或唯一发现 | 静态不接受，单模型只需本模型来源 |

## Constraint Rules

- 旧14、19维profile仍保留；不将compact14自动变成compact18，不静默修改旧checkpoint。
- 静态与动态输入维度／profile／架构一致；动态仅继承权重，不继承optimizer、计数或历史。
- 自动来源限定同dataset／group／model／seed；多个候选要求显式指定，缺来源拒绝启动。显式外部来源仅校验schema，需人工核对数据与实验条件。
- 旧23／21 checkpoint不兼容full23／25；完整组必须使用新完整静态来源，禁止隐式替换旧训练结果。
- 新group参数默认是直接删维消融，参数量base/full分别MLP90370／92162、GNN1001122／1002274；不是严格等容量。
- 所有模板动态initialize_from均空缺，由launcher绑定，不使用不存在的占位路径。直接调用公共train未传初始化则从头训练，推荐使用新入口。
- 更新／验证／保存间隔检查、数据分离、输入维度及来源兼容检查在启动任何训练进程前完成。

## Preset Profiles

- safe／检查：--group base或full --phase static --check，默认dataset2，CPU检查不启动训练。
- balanced／基础：base static，MLP／GNN各20k，8卡；完成后base dynamic各10k，显式相应best.pt。
- full／完整：full static与dynamic同预算／优化参数，输入23／25；用于与基础配对比较，至少3seed。

## Validation Notes

166项完整测试、真实dataset2两组预检及四个组／模型的CPU静态到动态训练通过；未执行新的正式GPU实验。

---

Doc-ID: PARAM-2026-11-05-009
Doc-Type: params
Owner: user
Date: 2026-11-05
Version: v1.0
Status: final
Source: 启动dataset1零额外成本、完整state、静态MLP/GNN训练并核对上次参数
Activated-Skills: code-agent-core, parameter-sanity

## Summary

- 两组从头训练，使用dataset1_cost0/static，各20,000回合，依次各占用8张GPU。
- 核心PPO参数与上次dataset2有成本实验一致；局部输入由21改为完整25维，全局仍23维。
- 全部1300实例逐字段核对，仅去应力／补偿成本归零，时间占用不变。

## Assumptions

- 用户“去盈利”按上下文解释为“去应力”；成本为0不取消操作或设备占用。
- 不加载上次23/21权重，使用本次23/25结构从头训练。

## Open Questions

None

## Context

正式输出为`outputs/fullstate_jingtai_set1/MLP`与`GNN`；GPU0～7；先MLP后GNN。

## Parameter Table

| 参数 | 类型 | 本次值 | 单位／约束 |
| --- | --- | --- | --- |
| episodes | int | 20000 | 全局回合，正整数 |
| seed / eval_seed | int | 7 / 1007 | 与上次相同 |
| mini_batch_size | int | 64 | 全局步数，须整除world size |
| distributed_world_size | int | 8 | 每组8个进程，单rank batch8 |
| episodes_per_update / ppo_epochs | int | 10 / 4 | 全局回合／更新轮数 |
| learning_rate / final_learning_rate | float | 3e-4 / 5e-5 | 正数，线性下降 |
| entropy_coef / final_entropy_coef | float | 0.02 / 0.005 | 非负，线性下降 |
| gamma / gae_lambda | float | 1.0 / 0.95 | [0,1] |
| clip_ratio / value_coef / max_grad_norm | float | 0.2 / 0.5 / 0.5 | 与上次相同 |
| hidden_dim | int | 128 | GNN须整除attention_heads |
| gnn_layers / attention_heads | int | 2 / 4 | 仅GNN |
| validation_interval | int | 500 | 全局回合，为更新回合数倍数 |
| latest_checkpoint_interval / checkpoint_interval | int | 100 / 5000 | 全局回合 |
| observation_profile / action_feature_profile | str | unified23 / enhanced25 | 全局23／局部25；GNN节点13 |
| output_dir_mode | str | existing | 已预检新空模型目录，精确输出路径 |
| train / validation / test | int | 1000 / 100 / 200 | 静态配对cost0实例 |

## Constraint Rules

- 两组除架构与输出路径外无配置差异，所有核心PPO超参数与上次相同。
- 当前8卡依次运行，不同时启动两组争用GPU；正式预算保持20k，未添加自动早停。
- 正常完成时目标等于makespan；去应力及补偿仍有持续时间与设备竞争。
- 测试集参与数据分离检查，不参与训练或best选择；本轮不自动生成最终测试报告。

## Preset Profiles

- 检查：使用本次输出目录内`preflight.json`、`parameter_comparison.json`及`data_audit.json`核对，不启动训练。
- 本次正式：上述固定20k、8卡、完整23/25配置；不启用其他预算预设以免偏离用户指定对照条件。

## Validation Notes

实际两组预检通过，参数差异为空，原始有成本与cost0的1300实例逐字段比较通过；针对静态／动态训练、分布式验证、cost0的19项回归通过（9.21秒）。

---

Doc-ID: PARAM-2026-11-05-010
Doc-Type: params
Owner: user
Date: 2026-11-05
Version: v1.0
Status: final
Source: 在dataset1完整state零费用静态最佳权重基础上检查参数并启动动态训练
Activated-Skills: code-agent-core, parameter-sanity

## Summary

- 两模型分别从对应静态best初始化，各微调10,000回合，Adam、计数和学习率日程重新开始。
- 输入和物理条件保持完整23／25、cost0；动态训练事件每次reset采样，验证／测试事件固定。
- GPU0～7依次运行MLP、GNN，输出为outputs/fullstate_dongtai_set1下的MLP与GNN。

## Assumptions

- “在此基础上”沿用dataset1_cost0、seed7、完整输入和各模型静态best；早停后的GNN使用第9500回合best，不使用latest。
- 本次按已有动态模板预算执行，不将上一次临时早停决定转成自动早停规则。

## Open Questions

None

## Context

MLP静态来源为fullstate_jingtai_set1/MLP/checkpoints/best.pt，第20,000回合，验证目标36.0909。GNN来源为同静态目录GNN/checkpoints/best.pt，第9,500回合，验证目标34.1139。精确来源、SHA256与来源配置保存在本次initialization_sources.json。

## Parameter Table

| 参数 | 类型 | 本次值 | 范围／单位／依赖 |
| --- | --- | --- | --- |
| train／validation／test | 路径 | data/dataset1_cost0/dynamic下对应split | 1000／100／200，保持分离 |
| episodes | int | 10000 | 正整数，全局回合，静态原预算20000 |
| seed／eval_seed | int | 7／1007 | 训练随机过程／固定验证种子 |
| learning_rate／final_learning_rate | float | 1e-4／2e-5 | 正数，线性下降，静态为3e-4／5e-5 |
| entropy_coef／final_entropy_coef | float | 0.01／0.003 | 非负，线性下降，静态为0.02／0.005 |
| gamma／gae_lambda | float | 1.0／0.95 | [0,1] |
| mini_batch_size | int | 64 | 全局样本数，可被8整除 |
| distributed_world_size | int | 8 | GPU进程数，各rank batch8 |
| episodes_per_update／ppo_epochs | int | 10／4 | 全局回合／每块优化轮数 |
| clip_ratio／value_coef／max_grad_norm | float | 0.2／0.5／0.5 | clip在(0,1)，value非负，梯度界正数 |
| observation_profile／action_feature_profile | str | unified23／enhanced25 | 与静态来源相同；全局23、动作25 |
| hidden_dim | int | 128 | GNN须整除attention_heads |
| gnn_layers／attention_heads | int | 2／4 | 仅GNN，节点13、边2维 |
| validation_interval | int | 500 | 全局回合，为10的倍数 |
| latest_checkpoint_interval／checkpoint_interval | int | 100／5000 | 全局回合，为10的倍数 |
| initialize_from | 路径 | 各自静态best.pt | 只加载权重，不恢复优化器和旧历史 |
| output_dir_mode | str | existing | 精确新建模型目录；监督程序拒绝已有产物 |
| enable_plots | bool | false | 训练与验证原始曲线数据持续保存 |

## Constraint Rules

- 同阶段两模型非架构优化参数差异为空；来源checkpoint与配置在每模型启动前校验SHA256。
- 环境语义版本7、动态输入版本2；静态与动态全局23／动作25／GNN节点13完全兼容。
- 静态预训练实际预算不同：MLP训练20k、GNN早停保存历史13270，所选权重分别来自20k／9500；动态同预算不代表整个训练链等计算投入。
- 两种辅助费用为0，时长和设备占用保持；等待成本为0，当前目标为makespan。
- 事件比例无事件65%、插单20%、故障10%、混合5%；未来事件和未释放工件对策略隐藏。
- 测试集只做配置与数据完整性检查，不参与微调或best选择；后续测试评估须另行执行。

## Preset Profiles

- 预检：preflight.json、data_audit.json和initialization_smoke.json；CPU核查，不更新静态权重。
- 本次微调：固定上述10k、8卡和低学习率／熵系数，依次两模型；无需额外预算预设。

## Validation Notes

两模型严格加载静态best及配对参数预检通过；cost0源逐字段检查和动态数据校验通过。两模型各运行四种固定验证场景，共8回合正常完成、零违规、额外成本0。定向pytest 19项通过，9.69秒。未改变通用算法、网络或输入模板。


---

Doc-ID: PARAM-2026-11-06-001
Doc-Type: params
Owner: Codex
Date: 2026-11-06
Version: v1.0
Status: final
Source: dataset2和dataset3完整state静态到动态训练，分别固定GPU0–3和4–7
Activated-Skills: code-agent-core, parameter-sanity

## Summary

- 两数据集并行、每数据集各模型依次运行，静态20k与动态10k回合保持原预算。
- 使用dataset2_cost0与dataset3_cost0，完整全局23／局部25，seed7。
- 四卡保持全局batch64，每rank batch16；动态来源仅为各自数据集和模型的静态best。

## Assumptions

- 用户新的分卡要求适用于静态及后续动态阶段。
- 延续零额外费用与原物理时长、原固定划分、验证总体均值选best，无自动早停。

## Open Questions

None

## Context

两个数据集各有静态／动态阶段与MLP／GNN配置，共8个模型任务。dataset2使用outputs/fullstate_jingtai_set2及fullstate_dongtai_set2；dataset3使用对应set3目录。数据路径为data/dataset{2,3}_cost0/{static,dynamic}/{train,validation,test}。

## Parameter Table

| 参数 | 类型 | 静态值 | 动态值 | 范围／单位／依赖 |
| --- | --- | --- | --- | --- |
| CUDA_VISIBLE_DEVICES | str | dataset2:0,1,2,3；dataset3:4,5,6,7 | 同静态 | 两组物理GPU不重叠 |
| distributed_world_size | int | 4 | 4 | 正整数，与torchrun及可见GPU一致 |
| episodes | int | 20000 | 10000 | 全局回合 |
| seed／eval_seed | int | 7／1007 | 7／1007 | 训练／验证随机源 |
| mini_batch_size | int | 64 | 64 | 全局样本数，可被4整除；每卡16 |
| episodes_per_update／ppo_epochs | int | 10／4 | 10／4 | 全局回合／每块优化轮数 |
| learning_rate→final_learning_rate | float | 3e-4→5e-5 | 1e-4→2e-5 | 正数，线性下降 |
| entropy_coef→final_entropy_coef | float | 0.02→0.005 | 0.01→0.003 | 非负，线性下降 |
| gamma／gae_lambda | float | 1.0／0.95 | 同静态 | [0,1] |
| clip_ratio／value_coef／max_grad_norm | float | 0.2／0.5／0.5 | 同静态 | clip在(0,1)，value非负，梯度界正数 |
| observation_profile／action_feature_profile | str | unified23／enhanced25 | 同静态 | 全局19+4动态预留、局部25 |
| hidden_dim | int | 128 | 128 | 正整数，可被4头整除 |
| gnn_layers／attention_heads | int | 2／4 | 2／4 | GNN节点13、边2维 |
| validation_interval | int | 500 | 500 | 全局回合，为10的倍数 |
| latest_checkpoint_interval／checkpoint_interval | int | 100／5000 | 同静态 | 全局回合，为10的倍数 |
| initialize_from | path/null | 无，从头训练 | 对应本数据集静态best.pt | 不使用归档八卡权重，不恢复优化器 |
| output_dir_mode／enable_plots | str／bool | existing／false | 同静态 | 精确新目录，原始曲线JSON保存 |
| train／validation／test | path | 各1000／100／200 | 同静态 | 固定划分且内容不交叉 |

## Constraint Rules

- 同阶段两模型仅架构、输出及对应来源不同；四卡配置不改全局batch、学习率或更新回合数。
- 每数据集两种静态模型均完成且有best／latest／final、完整历史后，才检查与启动动态阶段。
- 代码和配置哈希固定，动态来源严格匹配相同数据集、相同模型、相同state与网络结构；Adam和历史重置。
- 环境语义7、动态输入语义2；未来事件与未释放工件隐藏。
- 无事件／插单／故障／混合训练概率65%／20%／10%／5%；验证为65／20／10／5，测试为130／40／20／10。
- 两种辅助额外费用0，处理时长和设备占用不变；测试集不用于训练或best选择。

## Preset Profiles

- CPU预检：全数据来源、划分与参数检查，不更新模型。
- 本次四卡：dataset2绑定0–3、dataset3绑定4–7，独立监督链并行，为本次固定执行档。
- 先前八卡：仅归档溯源，已停止，不作为本次执行档。

## Validation Notes

两个数据集与各阶段参数审核通过；19项既有定向测试及GPU绑定／接续／失败隔离模拟检查通过。实际4进程训练与模型进度由运行目录记录，静态best生成后的严格迁移预检将自动执行。


---

Doc-ID: PARAM-2026-11-06-002
Doc-Type: params
Owner: Codex
Date: 2026-11-06
Version: v1.0
Status: final
Source: 简化state dataset1使用0–1卡、dataset2使用2–4卡、dataset3使用5–7卡的静态后动态训练准备
Activated-Skills: code-agent-core, parameter-sanity

## Summary

- 简化state实际全局18维（14基础＋4动态预留）、动作21维，MLP／GNN对应参数90370／1001122。
- 各数据集静态20k后动态10k，seed7；动态来源严格为本次对应简化静态best。
- 保持全局batch64，2进程有效32／32；3进程有效22／22／20、补齐2槽位且零loss权重。

## Assumptions

- 延续cost0、原数据划分、base模板超参、原验证选择规则；本次仅准备，未启动训练。
- 命名simplestate_jingtai_setN及simplestate_dongtai_setN，与fullstate结果分离。

## Open Questions

None

## Context

入口scripts/train_simplestate_pipeline.py以base模板派生12份任务配置，启动时保存到目标模型目录，不预写动态权重文件。三个数据集并行，每数据集四项任务顺序执行。

## Parameter Table

| 参数 | 类型 | 静态 | 动态 | 范围／单位／依赖 |
| --- | --- | --- | --- | --- |
| CUDA_VISIBLE_DEVICES | str | set1:0,1；set2:2,3,4；set3:5,6,7 | 相同 | 物理GPU不重叠 |
| distributed_world_size | int | 2／3／3 | 相同 | 与GPU数量及torchrun一致 |
| mini_batch_size | int | 64 | 64 | 正整数，全局有效样本数，支持非整除 |
| 每rank槽位 | 推导int | set1:32；set2/3:22 | 相同 | ceil(batch/world_size)，补齐不参与loss |
| observation_profile／action_feature_profile | str | compact18／compact21 | 相同 | 基础14＋动态4，动作21 |
| hidden_dim／gnn_layers／attention_heads | int | 128／2／4 | 相同 | hidden可被heads整除，GNN图13／2维 |
| episodes | int | 20000 | 10000 | 全局回合 |
| seed／eval_seed | int | 7／1007 | 相同 | 可用--seed改变训练seed，验证种子固定 |
| learning_rate→final_learning_rate | float | 3e-4→5e-5 | 1e-4→2e-5 | 正数，线性衰减 |
| entropy_coef→final_entropy_coef | float | .02→.005 | .01→.003 | 非负，线性衰减 |
| gamma／gae_lambda | float | 1／.95 | 相同 | [0,1] |
| clip_ratio／value_coef／max_grad_norm | float | .2／.5／.5 | 相同 | clip在(0,1)，value非负，梯度界正数 |
| episodes_per_update／ppo_epochs | int | 10／4 | 相同 | 全局回合／每块优化轮数 |
| validation／latest／snapshot间隔 | int | 500／100／5000 | 相同 | 回合数，10的倍数 |
| initialize_from | path/null | 无 | 对应静态best.pt | 只迁移权重，Adam与计数重置 |
| output_dir_mode／enable_plots | str／bool | existing／false | 相同 | 新建精确目录，曲线原始JSON保留 |
| train／validation／test | path | datasetN_cost0/static:1000／100／200 | dynamic:1000／100／200 | 内容分离，测试不选best |

## Constraint Rules

- 三卡不把batch改为63或66，PPO现有屏蔽和DDP缩放确保loss按有效64样本均值更新；尾批同理。
- 新入口兼容非整除batch；原公共checker仍保留整除约束，不影响正在运行的完整state监督链。
- 两种静态模型完整完成后，严格校验动态来源模型／schema／来源train路径／seed与SHA256，随后逐模型动态微调。
- 未来事件和未释放工件隐藏；静态预留字段0，动态填可观测事件；GNN图通道在两state组保持相同。
- 新训练真实启动前要求既有fullstate set2与set3均complete；不覆盖已有简化输出，文件锁防止同数据集重复运行。

## Preset Profiles

- CPU预检：--check，可在当前训练期间运行，不更新模型，不创建模型输出目录。
- 三数据集并行：--dataset all，固定2／3／3进程和原batch64。
- 单数据集：--dataset 1/2/3，保持同样分卡和四任务顺序，适合单独终端管理。

## Validation Notes

三个数据集全数据审核、输入／配对预检均通过；21项组合测试通过，随后新增的总启动扇出检查也通过。三进程MLP与GNN全局batch64梯度和单进程一致性在浮点容差内通过。手动命令见docs/simplestate-training.md。
