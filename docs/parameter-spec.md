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

---

Doc-ID: PARAM-2026-11-09-001
Doc-Type: params
Owner: Codex
Date: 2026-11-09
Version: v1.0
Status: final
Source: 六维当前物理信号驱动三类四头门控，保留原模型默认行为
Activated-Skills: code-agent-core, parameter-sanity

## Summary

- physical_gating默认false；开启只适用于gnn_attention，示例初始门值约0.9。
- 六维信号以当前工序间释放阶段定义，普通边和辅助边分配到不同工序端点，结构和不适用边门值1。
- 共享6→16→16、三组16→8→4网络两层共用，共900参数；完整GNN1,003,174参数。

## Assumptions

- 本次不引入FiLM、共享K/V、Dropout、奖励变更或新正式训练。
- 负补偿裕量与负时间收益保留，不截为0；六维仅由当前可观测状态计算。

## Open Questions

None

## Context

图节点与边原始13／2维保持；物理信号单独保存在physical_node_features[N,6]，适用性保存在physical_edge_mask[E]。门控产生[E,heads]，在原attention softmax后控制完整Value。

## Parameter Table

| 参数／常量 | 类型 | 默认／本次值 | 范围／单位／依赖 |
| --- | --- | --- | --- |
| physical_gating | bool | false；新示例true | 仅GNN，默认schema与旧权重保持 |
| physical_gate_initial | float | .9 | 严格(0,1)，有限值；控制bias，非精确门值 |
| physical_gate_schema_version | int | checkpoint开启时1 | 模型自动保存，回载不接受其它版本 |
| 物理输入／编码宽度 | int | 6／16／16 | 架构固定，Tanh |
| 三组门控头宽度 | int | 16／8／attention_heads | 架构固定，默认四头，Tanh＋Sigmoid |
| 门控共享 | 架构 | 层间共享、同组正反共享 | 三类普通／去应力／补偿，不是十个独立头 |
| 初始末层weight标准差 | float | .001 | 小非零，使物理编码器首步可获梯度 |
| 时间尺度S | float | max(1,duration_scale) | 原环境可见实例时间尺度 |
| 变形比例 | float | D/A | A来自当前已完成工序，避免用下一工序参数 |
| 当前／M3后裕量 | float | (θ−D)/θ、(θ−D_after_relief)/θ | 有符号，θ缺失／非正则门控不适用 |
| 净时间收益 | float | (剩余自然等待−已知最快M3类等待与处理)/S | 有符号，不含未来补偿和最终排程收益 |
| 已用M3标志 | bool转float | 当前普通工序间阶段是否完成去应力 | 以最近NORMAL之后history判定 |
| 静态／动态episodes | int | 20000／10000 | 新示例，全局回合，不修改旧模板 |
| 新示例GPU进程数 | int | 1 | 多卡须匹配distributed_world_size与torchrun |
| 新示例seed／batch | int | 7／64 | 延续原协议，动态使用新门控静态best |

## Constraint Rules

- first／busy／finished／远期工序间隔及结构关系的适用mask为false，门值严格1；普通关系4/5仅当前待加工工序，辅助6–9仅上一已完成工序对应当前间隔。
- M3按stress_relief角色读取，不写死机器ID；阈值有效且启用补偿才启用六维门控。缺失去应力时第3、4项置0。
- 门控后消息不再次归一化，节点和边Value合并后整体缩放；安全约束仍由环境合法动作实现。
- 开关关闭不实例化额外模块、不消耗新增初始化随机数，checkpoint_kwargs保持旧结构；开启时要求10种FJSP关系。
- 新旧有无门控checkpoint不能静默混载；新的门控static/dynamic兼容，模型kwargs持久化固定架构和语义版本。

## Preset Profiles

- 原模型：原配置，不添加physical_gating，保持默认false与原图字段。
- 门控：新configs/physical_gate示例，900参数增量、初始约0.9，独立unique输出；尚未启动。
- 消融诊断：get_edge_gates导出每边每头门值，归一化前后语义明确；单元测试可注入全1／全0验证路径。

## Validation Notes

124项定向测试通过；独立完整state示例预检和真实旧checkpoint回载通过。三组物理头、四个门控维度与四个32维Value head一致（hidden128时）。真实六维边映射、负收益／裕量、M3后状态、跨图拼接、单入边、PPO概率一致性、两／三进程缺关系梯度与保存回载均有测试。

---

Doc-ID: PARAM-2026-11-09-002
Doc-Type: params
Owner: Codex
Date: 2026-11-09
Version: v1.0
Status: final
Source: 独立共享 K/V＋类型 FiLM 实现的默认参数及静态动态配置
Activated-Skills: code-agent-core, parameter-sanity

## Summary

- shared_kv_film 默认 false，独立示例 true；physical_gating 明确 false，两者暂时互斥。
- 每层 K/V 各一个128×128矩阵，三张10×128类型表，层间独立；缩放初始1、偏置初始0。
- 完整模型420130参数；沿用原静态／动态训练超参数，不预设性能收益。

## Assumptions

- 示例 dataset1_cost0、unified23／enhanced25、两层四头；未更改现有训练配置或启动正式训练。

## Open Questions

- None

## Context

FiLM 根据十种有向关系逐维调制共享 K/V 的节点投影，不读取六维物理信号；节点 Value 调制后再加原边 Value，按原 softmax 权重聚合。

## Parameter Table

| 参数／属性 | 类型 | 默认／示例值 | 范围／依赖 |
| --- | --- | --- | --- |
| shared_kv_film | bool | false／true | 仅 gnn_attention；与 physical_gating 互斥 |
| shared_kv_film_schema_version | int | 开启时 checkpoint 自动保存1 | 工厂不接受其它版本 |
| shared_key／shared_value | Linear | hidden_dim→hidden_dim，无偏置 | 两矩阵不同、跨关系共享、层间独立 |
| film_key_scale | Parameter | ones(edge_type_count, hidden_dim) | 每维任意实数，不使用 Sigmoid |
| film_value_scale | Parameter | ones(edge_type_count, hidden_dim) | 每维任意实数 |
| film_value_bias | Parameter | zeros(edge_type_count, hidden_dim) | 每维任意实数，仅加在节点 V |
| 示例 hidden／layers／heads | int | 128／2／4 | 每 head32维；hidden 必须整除 heads |
| 示例关系数量 | int | 10 | 正反向分别学习，不采用门控的三组分类 |
| 示例物理门控 | bool | false | 不生成物理输入，不增加门控900参数 |
| 静态／动态episodes | int | 20000／10000 | 独立输出，batch64、seed7／eval1007 |
| 静态学习率／熵系数 | float | 3e-4→5e-5／.02→.005 | 保持原训练计划 |
| 动态学习率／熵系数 | float | 1e-4→2e-5／.01→.003 | 保持原微调计划 |
| 示例设备／进程数 | str／int | cuda／1 | 手动选择空闲GPU；未自动占用 |
| 动态 initialize_from | 路径 | 示例不预填 | 微调必须显式指定新 FiLM 静态 checkpoints/best.pt |

## Constraint Rules

- 不设置静态类型 K 偏置：当前按关系和目标节点归一化时相同加数抵消；若未来更改归一化或 K 后处理，应重新判断。
- 类型缩放初始恒等仅对应共享投影基线，不保持原版独立 K/V 的任意策略；旧架构 checkpoint 严格迁移拒绝。
- Query、原边分数／Value、关系内 softmax、Actor/Critic 与环境输入定义保持原逻辑。
- 新配置 unique 输出根为 outputs/shared_kv_film/dataset1/{static,dynamic}/GNN；动态 source 由实际静态运行子目录确定，不能硬编码缺失路径。
- 新增开关不进入旧 checkpoint kwargs，关闭路径不消耗新增随机初始化；两个扩展默认均关闭。

## Preset Profiles

- 原版基线：两个开关均 false 或省略，完整 state 参数1002274。
- 类型 FiLM：shared_kv_film=true、physical_gating=false，完整 state 参数420130。
- 原物理门控独立消融：shared_kv_film=false、physical_gating=true，完整 state 参数1003174。

## Validation Notes

82项定向测试通过。完整参数量由实际构建核对，独立静态配置 CPU 数据／超参数审查通过；两／三进程梯度、静态动态 checkpoint 迁移、真实旧模型严格回载与两个原分支回归通过。正式训练收敛效果和墙钟速度尚未测量。

---

Doc-ID: PARAM-2026-11-10-001
Doc-Type: params
Owner: Codex
Date: 2026-11-10
Version: v1.0
Status: final
Source: 用户指定GPU0–1启动dataset1共享K/V＋类型FiLM静态动态训练
Activated-Skills: code-agent-core, parameter-sanity

## Summary

- 完整state、dataset1_cost0、两卡、seed7，静态20k→动态10k，物理门控关闭。
- 全局batch64，两rank各32；独立结果GNN+FiLM_set1，动态严格加载本次静态best。
- 沿用原学习率、熵和PPO设置，不按参数减少调整训练预算。

## Assumptions

- 按前文已确认实验顺序先静态后动态；不启用门控或改变现有简化训练配置。

## Open Questions

- None

## Context

scripts/train_gnn_film_set1.py从configs/shared_kv_film模板生成运行配置，保留原模板默认单卡unique示例。

## Parameter Table

| 参数 | 本次值 | 约束 |
| --- | --- | --- |
| CUDA_VISIBLE_DEVICES／distributed_world_size | 0,1／2 | 仅两卡，启动前无计算进程 |
| shared_kv_film／physical_gating | true／false | 独立FiLM实验 |
| state／action profile | unified23／enhanced25 | 完整state静态动态一致 |
| hidden／layers／heads | 128／2／4 | 参数420130 |
| episodes | 静态20000／动态10000 | 全局预算，无自动早停 |
| batch／episodes_per_update／PPO epochs | 64／10／4 | 每rank32有效样本 |
| seed／eval_seed | 7／1007 | 相同划分与固定验证事件 |
| LR | 静态3e-4→5e-5；动态1e-4→2e-5 | 沿用原计划 |
| entropy | 静态.02→.005；动态.01→.003 | 沿用原计划 |
| gamma／GAE lambda／clip | 1／.95／.2 | 原PPO设置 |
| validation／latest／snapshot interval | 500／100／5000 | 均落在update边界 |
| output | outputs/GNN+FiLM_set1/{static,dynamic} | 父目录存在拒绝覆盖 |
| initialize_from | 静态无；动态static/checkpoints/best.pt | 动态重置优化器与历史 |

## Constraint Rules

- 静态完整历史及三种checkpoint齐全后才启动动态；迁移schema必须一致，失败不继续。
- 监督状态与阶段配置独立保存；不修改原模板、原版或门控结果。

## Preset Profiles

- 预检：--check，CPU审核静态完整数据与模型参数，不创建结果。
- 正式：后台监督，两卡静态→动态，本次已启动。

## Validation Notes

5项启动链测试通过，实际两卡静态数据和模型预检passed=true；已有FiLM梯度／DDP／迁移测试在实现阶段通过。本次正式性能与耗时以新日志测量为准。

---

Doc-ID: PARAM-2026-11-10-002
Doc-Type: params
Owner: Codex
Date: 2026-11-10
Version: v1.0
Status: final
Source: 最新要求GPU2–4排队训练set2 GNN＋物理门控
Activated-Skills: code-agent-core, parameter-sanity

## Summary

- GPU2、3、4，world_size3，完整state，dataset2_cost0，门控开启FiLM关闭。
- batch64真实有效样本22／22／20，两个填充屏蔽，保持原训练预算与优化设置。
- 静态20k→动态10k，本次静态best迁移，独立结果GNN+gate_set2。

## Assumptions

- 未改变物理门控模型、原训练代码或已有模板；原set2简化链完成且三卡空闲才启动。

## Open Questions

- None

## Context

queue_gnn_gate_set2.py从现有physical_gate模板构造set2运行配置，排队期保存独立配置和预检。

## Parameter Table

| 参数 | 本次值 | 约束 |
| --- | --- | --- |
| GPU／world_size | 2,3,4／3 | 三卡均无计算进程 |
| poll interval／idle confirmations | 60秒／2 | 前序四阶段均complete |
| state／action profile | unified23／enhanced25 | 完整state两阶段一致 |
| physical_gating／shared_kv_film | true／false | 原门控结构，初始.9 |
| hidden／layers／heads | 128／2／4 | 总参数1003174 |
| static／dynamic episodes | 20000／10000 | 无自动早停 |
| batch／per-rank valid samples | 64／22,22,20 | 补齐至66但两槽不计loss |
| seed／eval_seed | 7／1007 | dataset2_cost0固定划分 |
| LR／entropy | 静态3e-4→5e-5／.02→.005；动态1e-4→2e-5／.01→.003 | 原计划 |
| update／epochs／validation | 10回合／4／500回合 | checkpoint latest100、snapshot5000 |
| output／dynamic source | GNN+gate_set2/{static,dynamic}／本次static/checkpoints/best.pt | 严格schema迁移，动态重置优化器 |

## Constraint Rules

- 不改batch为63或66；审查副本暂调整为66只用于绕过旧checker整除限制，保存配置与实际模型仍64。
- 目标父目录存在则拒绝覆盖；前序失败不自动接续，排队与训练信号可终止本链。

## Preset Profiles

- --check：CPU检查静态动态设置、数据及当前就绪状态，不创建输出。
- 后台队列：本次已启动，条件满足自动先静态后动态。

## Validation Notes

21项入口及回归测试通过，实际两阶段数据模型预检passed=true；batch64三进程梯度等价此前已验证。当前排队不分配GPU。

---

Doc-ID: PARAM-2026-11-10-003
Doc-Type: params
Owner: Codex
Date: 2026-11-10
Version: v1.0
Status: final
Source: set3完整state GNN＋门控在GPU5–7排队
Activated-Skills: code-agent-core, parameter-sanity

## Summary

- 完整state、dataset3_cost0、GPU5／6／7、三进程，静态20k→动态10k。
- 全局batch64、有效22／22／20，两补齐槽位屏蔽；门控true、FiLM false。
- 静态best严格迁移到动态，独立输出GNN+gate_set3，排队PID253017。

## Assumptions

- 与set2门控沿用相同模型和优化协议，仅数据、GPU、结果路径不同。

## Open Questions

- None

## Context

queue_gnn_gate_set3.py从physical_gate模板生成独立配置，保留原模板及当前训练。

## Parameter Table

| 参数 | 值 | 约束 |
| --- | --- | --- |
| GPU／world_size | 5,6,7／3 | 三卡均无计算进程 |
| 检查间隔／连续就绪 | 60秒／2次 | 原simplestate set3四阶段完成 |
| 数据／state | dataset3_cost0／unified23＋enhanced25 | 静态动态相同维度 |
| 门控／FiLM／初始门值 | true／false／.9 | 原900参数共享门控 |
| hidden／layers／heads | 128／2／4 | 总参数1003174 |
| 回合／batch／seed | 静态20000、动态10000／64／7 | eval1007，有效22／22／20 |
| LR／熵计划 | 静态3e-4→5e-5、.02→.005；动态1e-4→2e-5、.01→.003 | 原配置 |
| update／PPO epochs／验证 | 10／4／500 | latest100、snapshot5000 |
| 结果／动态初始化 | GNN+gate_set3/{static,dynamic}／本次static/checkpoints/best.pt | 严格迁移、优化器重置 |

## Constraint Rules

- 三卡实际batch仍64，预检约束副本绕过旧整除规则；padding不计loss。
- 静态完整训练且checkpoint齐全后启动动态，任何阶段失败不继续；父目录防覆盖、锁防重复。

## Preset Profiles

- --check：CPU预检，无训练输出。
- 后台队列：本次已启动，自动等待并串行训练。

## Validation Notes

23项新旧队列测试通过，实际set3两阶段CPU预检passed=true；原三进程PPO梯度等价测试已有覆盖。


---

Doc-ID: PARAM-2026-11-11-001
Doc-Type: params
Owner: Codex
Date: 2026-11-11
Version: v1.0
Status: final
Source: 全实验独立测试评估，仅使用GPU0–1
Activated-Skills: parameter-sanity, code-agent-core

## Summary

- 只使用GPU0、1，每卡一个模型评估进程，CPU线程每进程2。
- 固定200例对应测试集、seed1007加实例序号、确定性动作，验证best选择保持不变。
- 成功实例平均目标和完成率共同报告，另保留全量目标及失败明细。

## Assumptions

测试集与训练／验证路径分开；原环境物理和目标定义保持一致。

## Open Questions

未结束／未启动实验无最终测试值，留空等待。

## Context

scripts/evaluate_experiment_suite.py为独立推理入口，不修改训练超参数。

## Parameter Table

| 参数 | 类型 | 固定值／默认值 | 约束 |
|---|---|---|---|
| GPU | tuple | 0,1 | 每卡最多一个worker，已有计算进程不抢占 |
| CPU线程 | int | 2 | OMP/MKL/torch一致 |
| seed | int | 1007 | 每实例1007+排序序号 |
| deterministic | bool | true | 使用argmax |
| 实例数 | int | 200 | data/datasetN_cost0/{static,dynamic}/test |
| checkpoint | path | checkpoints/best.pt | 与有效验证历史最佳episode、objective匹配 |
| 轮询间隔 | 秒 | 15 | 等待新完成模型，无训练占卡 |
| 成功定义 | bool | terminated且非truncated且两类违规均0 | 不按目标大小筛选 |

## Constraint Rules

- 评估前后核对checkpoint和测试数据SHA256，保存各自manifest与profile。
- 已结束完整／早停／失败实验可评估best；运行中best不作最终测试。
- 复用旧测试需checkpoint哈希、seed、确定性、profile、测试manifest和全部实例名一致。
- 成功实例均值无成功样本时为null，失败实例保留在全量均值及完成率分母。

## Preset Profiles

- --check：列举32项及训练状态，无GPU分配。
- --output：GPU0/1后台监督两条串行评估队列，等待当前训练结束自动补测。

## Validation Notes

新统计与状态测试4项、旧多profile测试6项通过；实际启动GPU0/1成功，GPU2–7继续原训练。


---

Doc-ID: PARAM-2026-11-11-002
Doc-Type: params
Owner: user
Date: 2026-11-11
Version: v1.0
Status: final
Source: GPU0–1上set2 GNN共享K/V＋类型FiLM静态到动态训练
Activated-Skills: code-agent-core, parameter-sanity

## Summary

- 沿用FiLM set1的完整state模型和训练参数，仅改变数据集与输出目录。
- 每阶段使用dataset2_cost0对应静态或动态划分：1000训练、100验证、200测试。
- 两卡全局batch64，对应每rank32，无需padding；动态继承本次静态最佳权重。

## Assumptions

- 验证集选择best，不使用测试集选模型。

## Open Questions

None

## Context (system/module)

`scripts/train_gnn_film_set1.py --dataset 2 --pause-evaluator 310832`，后台PID336150，输出 `outputs/GNN+FiLM_set2`。

## Parameter Table

| 参数 | 类型 | 静态值 | 动态值 | 单位／约束 |
|---|---|---|---|---|
| CUDA_VISIBLE_DEVICES | str | 0,1 | 0,1 | 物理GPU编号 |
| distributed_world_size | int | 2 | 2 | 进程数，等于可见GPU数 |
| shared_kv_film | bool | true | true | 共享K/V与关系类型FiLM |
| physical_gating | bool | false | false | 与FiLM互斥 |
| observation_profile | str | unified23 | unified23 | 完整state |
| action_feature_profile | str | enhanced25 | enhanced25 | 动作特征 |
| hidden_dim / gnn_layers / attention_heads | int | 128 / 2 / 4 | 128 / 2 / 4 | hidden_dim能被heads整除 |
| episodes | int | 20000 | 10000 | 全局回合数，不乘GPU数 |
| mini_batch_size | int | 64 | 64 | 全局有效样本数 |
| episodes_per_update / ppo_epochs | int | 10 / 4 | 10 / 4 | 更新间隔／优化轮次 |
| learning_rate → final_learning_rate | float | 0.0003 → 0.00005 | 0.0001 → 0.00002 | 正数、末值不高于初值 |
| entropy_coef → final_entropy_coef | float | 0.02 → 0.005 | 0.01 → 0.003 | 非负递减 |
| validation_interval | int | 500 | 500 | 回合 |
| latest_checkpoint_interval / checkpoint_interval | int | 100 / 5000 | 100 / 5000 | 回合 |
| seed / eval_seed | int | 7 / 1007 | 7 / 1007 | 随机种子 |
| initialize_from | path | 无，从头初始化 | static/checkpoints/best.pt | 指向本次set2静态阶段 |

## Constraint Rules

- 两阶段均使用独立目录，已有目标目录或GPU计算进程存在时拒绝启动。
- 静态进程退出码为0、历史达到目标且best/latest/final齐全，才进入动态阶段。
- 动态启动前严格校验静态checkpoint架构匹配；预先的动态CPU配置检查暂不读取尚未生成的checkpoint。
- 空闲评估调度器暂停期间不参与GPU调度；整条训练链退出后恢复。

## Preset Profiles

- 静态从头训练：20,000回合，学习率3e-4→5e-5，熵系数0.02→0.005。
- 动态迁移微调：10,000回合，学习率1e-4→2e-5，熵系数0.01→0.003，重置优化器和历史。

## Validation Notes

20项相关测试通过；两阶段CPU预检通过；参数全部来自现有模板或用户明确指定的GPU和数据集，未改变全局batch或模型结构。


---

Doc-ID: PARAM-2026-11-12-001
Doc-Type: params
Owner: user
Date: 2026-11-12
Version: v1.0
Status: final
Source: 在对应训练卡组上补充独立测试集评估
Activated-Skills: code-agent-core, parameter-sanity

## Summary

- 不改变训练参数，复用已有测试集、确定性动作、种子与统计口径。
- 每项独立使用单GPU；gate set2仅在2–4卡组，simplestate set3动态仅在5–7卡组，FiLM set2仅在0–1卡组。
- 五项各200例；测试模型来自验证集选出的best，不用测试目标值改选模型。

## Assumptions

- 原卡组内无其它计算进程后才分配测试任务。

## Open Questions

None

## Context (system/module)

`scripts/test_completed_training.py` 复用 `scripts/evaluate_experiment_suite.py --worker`，新队列PID418642。

## Parameter Table

| 参数 | 类型 | 值 | 单位／约束 |
|---|---|---|---|
| gpu_ids | list[int] | gate=[2,3,4]；simplestate=[5,6,7]；FiLM=[0,1] | 原训练物理卡组 |
| CUDA_VISIBLE_DEVICES | str | 每个worker单卡 | 单项测试独立运行 |
| seed | int | 1007 | 第i例使用1007+i，保持原测试顺序 |
| deterministic | bool | true | 与既有评估口径一致 |
| instance_count | int | 200 | 各dataset/phase独立test，不使用训练或验证集 |
| checkpoint | path | 当前run/checkpoints/best.pt | 验证历史选取的最佳checkpoint |
| OMP_NUM_THREADS / MKL_NUM_THREADS | str | 2 / 2 | 每worker CPU线程上限 |
| polling interval | int | 5 | 秒 |
| wait_pipeline | bool | FiLM=true | 静态与动态均成功完成后才测试 |

## Constraint Rules

- 训练未完整结束时不评估中途best；训练失败记录测试未执行原因。
- 测试前后检查checkpoint和数据hash，结果必须包含有序200条实例和对应manifest。
- 报告成功实例目标均值、有效完成数、完成率及包含失败惩罚的全量目标均值，不删除大目标样本。
- 结果保存在模型运行目录下标准 `test_evaluation_*/results.json`，供原队列按hash、seed和数据manifest校验复用。

## Preset Profiles

- 已完成训练：gate静态、gate动态、simplestate动态立即并行补测，每项单卡。
- 待结束训练：FiLM静态与动态等待整条训练链成功结束，随后在0、1卡各评估一项。

## Validation Notes

14项相关测试通过，五项GPU分配与训练完成条件预检通过；无新增或调整模型超参数。


---

Doc-ID: PARAM-2026-11-12-002
Doc-Type: params
Owner: user
Date: 2026-11-12
Version: v1.0
Status: final
Source: set3三卡FiLM与物理门控静态到动态训练
Activated-Skills: code-agent-core, parameter-sanity

## Summary

- 两组均为完整state、hidden128、两层四头，使用独立的set3静态与动态数据划分。
- 静态20,000回合，动态10,000回合；每组动态只加载自己的静态best权重。
- 不改变有效batch64、更新间隔10回合、seed7和验证seed1007。

## Assumptions

- 沿用既有set1／set2模板的学习率和熵系数计划。

## Open Questions

None

## Context (system/module)

FiLM：`train_gnn_film_set1.py --dataset 3 --gpus 2,3,4`；门控：`queue_gnn_gate_set3.py --start-now --poll-seconds 5`。

## Parameter Table

| 参数 | 类型 | FiLM | 物理门控 | 约束／单位 |
|---|---|---|---|---|
| CUDA_VISIBLE_DEVICES | str | 2,3,4 | 5,6,7 | 用户指定物理卡组 |
| distributed_world_size | int | 3 | 3 | 每卡一个DDP进程 |
| shared_kv_film | bool | true | false | 两变体互斥 |
| physical_gating | bool | false | true | 两变体互斥 |
| physical_gate_initial | float | 不启用 | 0.9 | 现有门控模板 |
| observation_profile | str | unified23 | unified23 | 完整state |
| action_feature_profile | str | enhanced25 | enhanced25 | 相同动作特征 |
| hidden_dim / gnn_layers / attention_heads | int | 128 / 2 / 4 | 128 / 2 / 4 | head维度32 |
| episodes | int | 静态20000；动态10000 | 静态20000；动态10000 | 全局回合 |
| mini_batch_size | int | 64 | 64 | 全局有效样本 |
| episodes_per_update / ppo_epochs | int | 10 / 4 | 10 / 4 | 采样更新间隔／优化轮次 |
| learning_rate → final_learning_rate | float | 静态3e-4→5e-5；动态1e-4→2e-5 | 同左 | 正数、递减 |
| entropy_coef → final_entropy_coef | float | 静态0.02→0.005；动态0.01→0.003 | 同左 | 非负、递减 |
| validation_interval | int | 500 | 500 | 回合 |
| latest_checkpoint_interval / checkpoint_interval | int | 100 / 5000 | 100 / 5000 | 回合 |
| seed / eval_seed | int | 7 / 1007 | 7 / 1007 | 随机种子 |
| OMP_NUM_THREADS / MKL_NUM_THREADS | str | 2 / 2 | 2 / 2 | CPU线程 |

## Constraint Rules

- dataset3_cost0每阶段1000训练、100验证、200测试；三种split路径独立，静态与动态路径分别匹配。
- batch64在三个rank的完整mini-batch有效样本分配为22、22、20，两个padding位置在损失中屏蔽；预检仅绕过旧的整除限制，不改变训练有效batch。
- 两组均从头静态训练；动态initialize_from分别指向 `GNN+FiLM_set3/static/checkpoints/best.pt` 与 `GNN+gate_set3/static/checkpoints/best.pt`。
- 静态正常退出、历史达到20000且best/latest/final齐全才进入动态；迁移前严格校验架构和输入profile。
- GPU有计算进程或目标目录已存在时不覆盖；本次门控明确启动模式仅解除旧前置链依赖。

## Preset Profiles

- FiLM静态／动态：上述静态／动态计划，GPU2–4，shared_kv_film=true。
- 物理门控静态／动态：相同训练计划，GPU5–7，physical_gating=true。

## Validation Notes

两组两阶段CPU预检通过；37项launcher／queue检查及2项三卡有效batch64的分布式数值一致性检查通过。无新增模型结构或未验证的批量缩放规则。


---

Doc-ID: PARAM-2026-11-16-001
Doc-Type: params
Owner: user
Date: 2026-11-16
Version: v1.0
Status: final
Source: 最佳checkpoint对应测试集测评，补齐FiLM set3
Activated-Skills: code-agent-core, parameter-sanity

## Summary

- 训练已完整结束，best来自对应验证集，不根据测试结果改选checkpoint。
- 缺少的FiLM set3静态、动态各200例，分别在GPU2、3执行；原训练卡组范围仍为2、3、4。
- 已有四项匹配结果通过数据和模型hash、种子及实例完整性复核后保留。

## Assumptions

- 本次只补齐缺项，沿用此前独立测试的所有推理和统计参数。

## Open Questions

None

## Context (system/module)

`test_completed_training.py --scope film-set3`；监督PID782967，输出 `outputs/test_evaluation_film_set3_20261116_110636`。

## Parameter Table

| 参数 | 类型 | 值 | 约束／单位 |
|---|---|---|---|
| scope | str | film-set3 | 本次仅补测FiLM set3静态与动态 |
| gpu_ids | list[int] | [2,3,4] | 原训练卡组，静态／动态分别使用空闲GPU2、3 |
| seed | int | 1007 | 第i例使用1007+i |
| deterministic | bool | true | 确定性动作 |
| test count | int | 200/阶段 | 独立的dataset3_cost0/static/test或dynamic/test |
| checkpoint | path | 各run/checkpoints/best.pt | 与验证历史最优episode及objective严格一致 |
| OMP_NUM_THREADS / MKL_NUM_THREADS | str | 2 / 2 | 每worker CPU线程数 |
| poll interval | int | 5 | 秒 |

## Constraint Rules

- 测试前后checkpoint和数据hash不变，训练链全部成功完成后才测试。
- 实例名与实际JSON内部name比较；manifest按原始文件名及sha256比较。
- 报告成功实例均值、有效完成数、完成率和含失败惩罚的全量均值；不删除大目标或失败实例。

## Preset Profiles

- 静态：dataset3_cost0/static/test，当前FiLM set3静态best，GPU2。
- 动态：dataset3_cost0/dynamic/test，当前FiLM set3动态best，GPU3。

## Validation Notes

19项相关测试通过，四项已有结果完整性复核通过，本次不改变模型或训练超参数。


---

Doc-ID: PARAM-2026-11-16-002
Doc-Type: params
Owner: user
Date: 2026-11-16
Version: v1.0
Status: final
Source: set1完整state静态MLP到动态测试集的迁移基线
Activated-Skills: code-agent-core, parameter-sanity

## Summary

- 静态checkpoint在动态test评估，两个phase分别保存，不改原静态run的test_instances字段。
- 保持此前逐例seed1007+i与确定性动作，便于与已有动态MLP结果配对。
- 使用指定0–1卡组内GPU0，保留全部200例及失败惩罚。

## Assumptions

- 当前MLP checkpoint使用相同动态输入语义和完整state，未启用门控或FiLM。

## Open Questions

None

## Context (system/module)

`test_completed_training.py --scope fullstate-mlp-set1-static-to-dynamic`，源run为fullstate_jingtai_set1/MLP。

## Parameter Table

| 参数 | 类型 | 值 | 约束／单位 |
|---|---|---|---|
| checkpoint | path | 静态run/checkpoints/best.pt | 静态验证集选出的第20000回合 |
| checkpoint_training_phase | str | static | 保持模型来源准确 |
| evaluation_phase | str | dynamic | 仅显式允许static→dynamic跨阶段方向 |
| test_instances | path | data/dataset1_cost0/dynamic/test | 独立200例test |
| model_type | str | mlp | 无图注意力、物理门控或FiLM |
| observation_profile / action_feature_profile | str | unified23 / enhanced25 | 完整state及匹配动作输入 |
| CUDA_VISIBLE_DEVICES | str | 0 | 指定0–1卡组内单GPU |
| seed | int | 1007 | 每例seed=1007+i |
| deterministic | bool | true | 确定性动作 |
| OMP_NUM_THREADS / MKL_NUM_THREADS | str | 2 / 2 | 每worker CPU线程数 |

## Constraint Rules

- 测试目标必须是匹配dataset1的dynamic/test，不能指向train或validation；原训练配置不变。
- checkpoint episode与静态验证历史best一致；测试前后checkpoint和全部数据hash不变。
- 对照时匹配完整manifest、实例顺序和逐例seed，全量平均目标与有效完成率共同报告。

## Preset Profiles

- 静态到动态基线：本次静态best，在动态test重新评估200例。
- 动态到动态参考：复用已有动态best、同一动态test、相同seed与动作设置的200例结果。

## Validation Notes

25项相关回归通过；静态模型schema与动态语义版本预检通过；已有动态MLP参考结果完整性复核通过。


---

Doc-ID: PARAM-2026-11-16-003
Doc-Type: params
Owner: user
Date: 2026-11-16
Version: v1.0
Status: final
Source: 全部静态／动态最佳checkpoint的30例故障对照
Activated-Skills: code-agent-core, parameter-sanity

## Summary

- 三个动态test各固定200例中的30故障例，原索引和逐例seed保持不变。
- 18组模型配对、36个checkpoint任务，最多8卡并发，每卡一个，释放后自动领取下一任务。
- 缓存校验包含checkpoint与测试文件hash、实例名、scenario、逐例seed和输入profile；全量30例目标用于主表均值。

## Assumptions

- 使用现有数据、现有验证集best及现有环境语义，不调整训练超参数。

## Open Questions

None

## Context (system/module)

`compare_fault_checkpoints.py --output outputs/fault_checkpoint_comparison_20261116_123353`，监督PID790200。

## Parameter Table

| 参数 | 类型 | 值 | 约束／单位 |
|---|---|---|---|
| gpu_ids | list[int] | 0–7 | 每卡最多一个本队列worker，已有计算进程则等待 |
| fault_only | bool | true | 每数据集必须恰好30例 |
| 场景 | str | breakdown_only＋both | 20例仅故障＋10例故障与新作业 |
| source | path | dataset1／2／3_cost0/dynamic/test | 同一配对双方使用同一测试集合 |
| seed | int | 1007 | 逐例1007＋原200例中的索引，不在30例内重新编号 |
| deterministic | bool | true | 既有评估动作方式 |
| checkpoint | path | 各run/checkpoints/best.pt | 按对应验证历史选best |
| OMP_NUM_THREADS / MKL_NUM_THREADS | str | 2 / 2 | 每worker CPU线程数 |
| queue polling interval | int | 2 | 秒 |
| mean_objective | float | 全30例均值 | 保留失败惩罚，不改为成功子集均值 |
| 改善百分比 | float | (静态均值−动态均值)/静态均值×100 | 正值动态更好；静态为0时不定义 |

## Constraint Rules

- 两模型的full test manifest、故障实例顺序与逐例seed严格匹配；不得把故障子集重新从seed1007编号。
- 静态模型在对应动态测试故障子集评估，原静态训练配置不变；训练中模型不参与。
- 已结束但failed／early_stopped的实验允许使用已保存的验证best，必须在结果附注保留状态。
- 缓存与新推理结果均重新汇总30例，报告中保留有效完成数及逐例信息。

## Preset Profiles

- 缓存复用：当前checkpoint与动态test完整匹配时提取30例，CPU验证后标为完成。
- GPU补测：无匹配缓存时，在空闲GPU独立推理30例，每卡一个，结束后自动接续。

## Validation Notes

35项相关回归通过；三个数据集均验证20例breakdown_only与10例both；初始19项缓存通过审计，17项需GPU补测。


---

Doc-ID: PARAM-2026-11-16-004
Doc-Type: params
Owner: user
Date: 2026-11-16
Version: v1.0
Status: final
Source: 共享K/V＋类型FiLM固定工作负载速度基准
Activated-Skills: code-agent-core, parameter-sanity

## Summary

- 仅比较计算耗时，不重训生产模型、不修改checkpoint。
- 输入为原版策略在各数据集首个静态测试实例上生成的相同64步快照。
- GPU4独占本次模型任务，保持CPU线程2、float32，交替3次测量取中位数。

## Assumptions

- 模型均完整state、hidden128、两层四头，无物理门控；仅shared_kv_film开关不同。

## Open Questions

None

## Context (system/module)

`benchmark_film_fixed_workload.py`，结果 `outputs/film_fixed_speed_20261116_1440`。

## Parameter Table

| 参数 | 类型 | 值 | 单位／约束 |
|---|---|---|---|
| CUDA_VISIBLE_DEVICES | str | 4 | 空闲A800 80GB；占用时拒绝运行 |
| dtype | str | float32 | 两架构一致 |
| torch_num_threads / OMP / MKL | int | 2 / 2 / 2 | CPU线程 |
| snapshots | int | 64 | 同观测、动作候选和图输入 |
| inference warmup | int | 16 | 每模型、每动作模式的预热调用 |
| repeats | int | 3 | 原版／FiLM交替顺序，取中位数 |
| ppo_samples / mini_batch_size | int | 64 / 64 | 单卡，无DDP |
| ppo_epochs | int | 4 | 每次更新 |
| ppo warmup | int | 1 | 每架构预热一次完整更新 |
| validation_act deterministic | bool | true | 验证动作选择 |
| sampling_act deterministic | bool | false | 采样动作选择 |

## Constraint Rules

- 动作计时前后CUDA同步；包含原生act整理、传输和返回，不包含重新生成环境图及环境推进。
- PPO只用于固定输入计算基准；同一历史rollout复用不解释为合法收敛实验。
- checkpoint仅加载到内存，PPO更新只修改内存模型，不保存训练权重。

## Preset Profiles

- 动作推理：预热16次，再交替3轮，每轮64次，报告ms/动作中位数。
- PPO更新：相同64条数据、batch64、4epochs，预热1次后交替3轮，报告秒/更新中位数。

## Validation Notes

set1–3所有短基准完成，9种指标的全部重复值均为有限正数。历史验证耗时另列GPU数量，不将跨卡数差异当作架构加速比。

---

Doc-ID: PARAM-2026-11-16-005
Doc-Type: params
Owner: user
Date: 2026-11-16
Version: v1.0
Status: final
Source: set1–3仅共享K/V、不加FiLM的静态＋动态训练
Activated-Skills: code-agent-core, parameter-sanity

## Summary

- 仅改变关系K/V投影共享方式，关闭FiLM与物理门控；其余沿用完整state既有超参。
- set1／2单卡，set3两卡，全局mini_batch_size64保持一致。
- 各组静态20,000回合，动态10,000回合从对应静态best初始化。

## Assumptions

延用已有 seed7／eval_seed1007，使用 train1000／validation100／test200 的既有划分；本任务只训练和验证，测试集不参与best选择。

## Open Questions

None

## Context (system/module)

`scripts/train_shared_kv_only.py` 与 `outputs/GNN+sharedKV_set{1,2,3}/{static,dynamic}/training_config.json`。

## Parameter Table

| 参数 | 类型 | 值 | 范围／单位／依赖 |
|---|---|---|---|
| shared_kv_only | bool | true | 与FiLM、物理门控互斥 |
| shared_kv_film / physical_gating | bool | false / false | 不创建调制或门控参数 |
| shared_kv_only_schema_version | int | 1 | checkpoint架构版本，拒绝未知版本 |
| observation_profile / action_feature_profile | str | unified23 / enhanced25 | 完整state，迁移须匹配 |
| hidden_dim / gnn_layers / attention_heads | int | 128 / 2 / 4 | hidden_dim可被heads整除 |
| GPU / world_size | int列表／int | set1:4/1；set2:5/1；set3:6,7/2 | 固定用户指定卡 |
| episodes | int | 静态20000，动态10000 | 全局回合，不乘world_size |
| learning_rate / final_learning_rate | float | 静态3e-4→5e-5，动态1e-4→2e-5 | 线性调度，阶段内计数 |
| entropy_coef / final_entropy_coef | float | 静态.02→.005，动态.01→.003 | 沿用原预设 |
| mini_batch_size / ppo_epochs | int | 64 / 4 | global batch，单／双卡可整除 |
| episodes_per_update | int | 10 | 全局采样；两卡各5回合 |
| gamma / gae_lambda / clip_ratio | float | 1 / .95 / .2 | 沿用原预设 |
| value_coef / max_grad_norm | float | .5 / .5 | 沿用原预设 |
| validation_interval | int | 500 | 每次评估全部100例验证集 |
| latest / periodic checkpoint interval | int | 100 / 5000 | best另按验证目标改善保存 |
| seed / eval_seed | int | 7 / 1007 | 固定随机种子 |
| OMP_NUM_THREADS / MKL_NUM_THREADS | int | 2 / 2 | 每训练worker限制CPU线程 |
| initialize_from | path | 动态使用本组static/checkpoints/best.pt | 静态从头训练；动态重置优化器 |

## Constraint Rules

- 不允许同时打开shared_kv_only与FiLM或物理门控；MLP不接受新开关。
- 实际动态启动前严格校验环境语义、输入profile和model_kwargs；不兼容则停止。
- GPU已占用或输出目录已存在时拒绝启动，不覆盖旧结果。
- 每阶段退出成功后校验history完整长度、最后回合及best／latest／final文件，再进入下一阶段。

## Preset Profiles

- set1：GPU4单卡，完整state，静态20k→动态10k。
- set2：GPU5单卡，相同全局训练参数，采用set2数据。
- set3：GPU6–7双卡DDP，全局batch64／每卡32，采用set3数据。

## Validation Notes

六阶段CPU预检均通过，静态／动态模型均为412,450参数。55项相关回归通过，涵盖前向、梯度、checkpoint、迁移、profile和双卡PPO更新。

---

Doc-ID: PARAM-2026-11-16-006
Doc-Type: params
Owner: user
Date: 2026-11-16
Version: v1.0
Status: final
Source: 独立故障动态测试集set1／set2／set3，cost0，各100例
Activated-Skills: code-agent-core, parameter-sanity

## Summary

- 仅保留breakdown_only与both，按原动态分布条件化为2/3与1/3；每组100例取67／33。
- 复用现有基础实例生成函数与事件生成函数，采用独立种子，不改变工艺及事件参数。
- 固定轨迹、零辅助费用、隐藏未来事件，所有300例只用于测试。

## Assumptions

插单仍指原生成器的延迟释放既有工件；cost0只将去应力和补偿费用设为0，保留原时长、物理参数和目标权重。

## Open Questions

None

## Context (system/module)

`scripts/generate_fault_test100.py`；目标 `data/dynamic_fault_test100_cost0/`。

## Parameter Table

| 参数 | 类型 | 值 | 单位／范围／依赖 |
|---|---|---|---|
| SIZE | int | 100 | 每个set的测试实例数，共300 |
| SEED | int | 2026111601 | 本次独立生成种子 |
| base_seed | int | SEED＋set编号×1000003 | 每例再加1～100 |
| event_seed | int | base_seed＋400000 | 每例再加1～100，固定轨迹 |
| scenario shuffle seed | int | base_seed＋200000 | 随机化两类场景排列 |
| breakdown_only / both | int | 67 / 33 | 最大余数法，源概率.10／.05 |
| mode | str | fixed | reset seed不改变事件轨迹 |
| job release count | int | 1～2 | 条件概率.90／.10，仅both启用 |
| release time fraction | float | .05～.30 | 相对原参考时长 |
| breakdown count | int | 1～2 | 条件概率.95／.05，普通生产设备 |
| breakdown time fraction | float | .15～.70 | 相对原参考时长 |
| repair duration factor | float | .50～1.50 | 相对平均加工时长 |
| interruption_policy | str | pause_resume | 故障暂停、修复后恢复 |
| stress_relief.cost / compensation.cost | float | 0 / 0 | 仅两项辅助费用清零 |
| future / unreleased visibility | bool | false / false | 沿用动态输入语义2 |

## Constraint Rules

- 不含无事件或仅插单场景；每例必须有机器故障事件，both还须含1～2个释放事件。
- 每个set保留原工件、工序、机器规模及所有非成本参数，不根据模型分数筛选。
- 与三规模既有cost0静态／动态所有split的基础实例不得重复；跨新集合也不得重复。
- 源数据完整、原故障场景2:1配比未改变、所有文件校验和及种子重建均通过后才发布；不覆盖已存在目标。

## Preset Profiles

- set1：6～10工件，每件4～7道普通工序，4普通机器＋2辅助机器。
- set2：12～20工件，每件5～10道普通工序，6～10普通机器＋2辅助机器。
- set3：18～30工件，每件6～13道普通工序，10～16普通机器＋2辅助机器。

## Validation Notes

4项测试通过。生产300例均成功加载，固定轨迹不依赖reset seed；事件合法性、cost0、SHA256、去重、配额和全参数重建均通过。原始来源和逐例种子记录在manifest，规模统计见validation_report。

---

Doc-ID: PARAM-2026-11-16-007
Doc-Type: params
Owner: user
Date: 2026-11-16
Version: v1.0
Status: final
Source: MLP完整state set2静态／动态best在新100例故障动态测试集上的配对测评
Activated-Skills: code-agent-core, parameter-sanity

## Summary

- 使用GPU5共享评估，CPU线程2、nice10，保留既有训练。
- 两模型同100例、同固定事件、同seed和确定性动作。
- 全部样本及双侧有效口径分开记录，不按目标值剔除。

## Assumptions

只评估现有验证best，不重新训练、不更新模型参数；case文件名排序与生成manifest一致。

## Open Questions

None

## Context (system/module)

`scripts/evaluate_fault_test100_pair.py`；本次dataset2，输出 `outputs/fault100_fullstate_set2_MLP_20261116_172151`。

## Parameter Table

| 参数 | 类型 | 值 | 范围／依赖 |
|---|---|---|---|
| CUDA_VISIBLE_DEVICES | str | 5 | 与现有训练共享，额外模型显存约0.5GiB |
| OMP / MKL / torch threads | int | 2 / 2 / 2 | 限制评估CPU资源 |
| nice | int | 10 | 低于默认CPU调度优先级 |
| instances | path | data/dynamic_fault_test100_cost0/set2/test | 恰好100例，67仅故障＋33混合 |
| static checkpoint | path | outputs/fullstate_jingtai_set2/MLP/checkpoints/best.pt | 验证最优，episode20000 |
| dynamic checkpoint | path | outputs/fullstate_dongtai_set2/MLP/checkpoints/best.pt | 验证最优，episode500 |
| observation / action profile | str | unified23 / enhanced25 | 完整state，各92,162参数 |
| seed | int | 1007＋文件排序索引 | 两模型相同 |
| deterministic | bool | true | 固定策略动作选择 |

## Constraint Rules

- 新数据文件与manifest逐一校验；原checkpoint与验证最优记录一致。
- 两模型必须都是完整state MLP且输入语义兼容。
- 测试前后文件和权重SHA256保持不变；输出目录必须是新目录。
- 有效样本须正常结束、无截断、无调度或就绪违规；删除时两侧共同删除，保留全部实例主统计。

## Preset Profiles

- 全部故障：100例，主评估。
- 仅故障：67例，分组统计。
- 故障＋插单：33例，分组统计。

## Validation Notes

预检已确认数据100例、两模型输入profile一致、checkpoint语义7／2、验证best回合20000／500。最终结果与有效完成情况见同目录results.json。

实际200次调度全部正常完成，无截断、无调度／就绪违规；逐例seed为1007～1106。测试前后权重及数据哈希一致，三组均值通过独立重算校验；结果于17:25:21保存完成。

---

Doc-ID: PARAM-2026-11-16-008
Doc-Type: params
Owner: user
Date: 2026-11-16
Version: v1.0
Status: final
Source: 新100例故障测试集18组对比的GPU5–7共享评估队列
Activated-Skills: code-agent-core, parameter-sanity

## Summary

- 每卡一个评估进程，三卡最多三个worker，较大set优先，完成自动补位。
- 共36个验证best、每个100例；200条可复用，新增3400次调度。
- 主表为双侧共同有效配对；全量原始值、场景子集、异常记录另存。

## Assumptions

沿用此前18组及其验证best，加入的是新的独立故障测试集，不包含正在训练的仅共享K/V新实验。

## Open Questions

None

## Context (system/module)

`scripts/compare_fault_test100.py`，输出 `outputs/fault_test100_comparison_20261116_173132`。

## Parameter Table

| 参数 | 类型 | 值 | 范围／依赖 |
|---|---|---|---|
| GPUS | tuple[int] | 5,6,7 | 固定指定卡；允许与已授权训练共享 |
| workers_per_gpu | 调度约束 | 1 | 本队列并发总数3 |
| model pairs / checkpoints | int | 18 / 36 | 每组static与dynamic各一个验证best |
| instances_per_checkpoint | int | 100 | 对应set，67仅故障＋33混合 |
| seed | int | 1007+i | i为新集合固定文件排序0～99，两侧相同 |
| deterministic | bool | true | 固定动作选择 |
| torch / OMP / MKL threads | int | 2 / 2 / 2 | 每评估worker |
| nice | int | 10 | 低优先级CPU调度 |
| scheduling poll interval | int | 2 | 秒 |
| dispatch priority | tuple | dataset降序、GNN优先 | 减少大任务尾部等待 |
| context | path | data/dynamic_fault_test100_cost0/set{1,2,3}/test | SHA256与生成manifest一致 |
| improvement_percent | float | 100×(static−dynamic)/static | 保留正负号，不截断 |

## Constraint Rules

- 逐项核对模型类型、完整／简化profile、门控／FiLM标记、环境语义、验证best回合和目标。
- 缓存只允许此前新set2完整MLP结果，要求checkpoint、测试文件、seed、profile与逐例顺序完全一致；不复用旧30例分数。
- 任一侧无效时，主表从两侧共同剔除同一实例；合法高值不剔除，全100例原始均值另存。
- 单项失败后继续其余项；不将失败或待完成项填写为0%。

## Preset Profiles

- set3：先调度，100例／checkpoint，较大规模。
- set2：其次调度，完整MLP静态／动态已缓存，其余照常测评。
- set1：最后调度，较小规模；原静态GNN早停状态保留。

## Validation Notes

36个checkpoint预检通过，2项缓存通过逐例校验，21项新旧回归通过；首批3个worker准确映射GPU5／6／7且均已开始评估。

---

Doc-ID: PARAM-2026-11-16-009
Doc-Type: params
Owner: user
Date: 2026-11-16
Version: v1.0
Status: final
Source: 新100例测试队列扩展到GPU4–7
Activated-Skills: code-agent-core, parameter-sanity

## Summary

- 本次显式设置`--gpus 4,5,6,7 --resume`，并发从3增加到4。
- 原worker身份和进度保留，数据及统计规则不变。
- 原训练继续运行，评估线程和CPU优先级沿用既有设置。

## Assumptions

用户授权GPU4与原训练共享。

## Open Questions

None

## Context (system/module)

`compare_fault_test100.py`，原输出目录保持不变，新监督PID826376。

## Parameter Table

| 参数 | 类型 | 本次值 | 约束 |
|---|---|---|---|
| --gpus | tuple[int] | 4,5,6,7 | 唯一GPU索引，每卡一个worker |
| --resume | bool | true | 已存在队列，原调度器已退出 |
| process_start_ticks | str | /proc/PID/stat第22项 | 防止将复用PID误作原worker |
| threads / nice | int | 2 / 10 | 沿用原评估设置 |

## Constraint Rules

- 已运行worker所在GPU必须包含在新GPU列表内，不允许一个GPU接管两个worker。
- 旧调度器仍存活时拒绝接管；进程身份变化或已退出后必须通过结果校验才可记完成。
- 已完成结果复核，新旧测试manifest、checkpoint SHA256保持一致。

## Preset Profiles

- 原三卡：5,6,7。
- 用户本次扩容：4,5,6,7。

## Validation Notes

25项回归通过；生产接管后三个旧PID保持不变，新GPU4任务已启动，四卡运行正常。

---

Doc-ID: PARAM-2026-11-17-001
Doc-Type: params
Owner: user
Date: 2026-11-17
Version: v1.0
Status: final
Source: set1仅共享K/V结构、训练、数据审计及静态／动态checkpoint补测
Activated-Skills: code-agent-core, parameter-sanity

## Summary

- 仅补测既有验证best；运行中训练超参和架构保持原定义。
- 同GPU4、CPU线程2、确定性动作、seed1007+i，对应模型使用完全相同的事件与实例。
- 两种测试分布分开汇总，不混淆全动态与故障条件均值。

## Assumptions

所有checkpoint语义7／2、完整state23、动作25；新集合只含两类故障，原测试包含四类场景。

## Open Questions

等预算多seed的因果比较尚未执行；不根据本轮测试结果修改best选择。

## Context (system/module)

`evaluate_fault_test100_pair.py --dataset 1 --model GNN+sharedKV`；`--test-source new_fault100`或`original_dynamic200`。

## Parameter Table

| 参数 | 类型 | 值 | 约束／依赖 |
|---|---|---|---|
| shared_kv_only / FiLM / gate | bool | true / false / false | 412450参数，层间独立K与V |
| GPU / threads | int | 4 / 2 | 使用已完成set1训练后空闲卡 |
| seed / deterministic | int／bool | 1007+i / true | 静态动态同实例同seed |
| static best / dynamic best | int | 20000 / 7000 | 各自原验证选择结果 |
| new_fault100 | counts | 67仅故障＋33混合 | 全新100基础实例 |
| original_dynamic200 | counts | 130无事件＋40仅插单＋20仅故障＋10混合 | 原独立test，不涉及训练或验证 |
| dynamic_train_mix | probabilities | .65 / .20 / .10 / .05 | 无事件／插单／故障／混合 |
| evaluation filtering | validity | 正常结束、无截断／违规 | 另记双侧有效配对；不按目标高低筛选 |

## Constraint Rules

- 架构、profile、验证best与数据manifest逐项匹配；前后哈希保持一致。
- 完全共享K/V不是类型无损的等价改写；singleton零特征前驱／后继消息存在方向标签混同。
- 静态和动态验证集不同，不能直接将33.669314与34.079902计算微调改善；必须在相同动态集合配对。
- epoch0动态验证保底是未来协议建议，本次既有实验不追溯改选。

## Preset Profiles

- 新100例：故障条件对照，与此前18组直接对齐。
- 原200例：混合动态分布，可单独报告仅插单／仅故障／混合。

## Validation Notes

98项回归通过，原三个规模cost0与动态数据校验、新300例去重与轨迹检查均通过。新100例共享K/V静态／动态全部有效，均值34.896686／34.851915，改善0.1283%。

原200例也全部有效，静态／动态均值34.227197／34.143576，改善0.2443%；仅插单＋0.8069%、仅故障−0.7392%、混合＋1.9612%。共600次调度的checkpoint／数据哈希、逐例seed、CSV行数及分组均值复核通过。GPU4评估进程已结束，显存释放。

---

Doc-ID: PARAM-2026-11-17-002
Doc-Type: params
Owner: user
Date: 2026-11-17
Version: v1.0
Status: final
Source: 新100例测试19组的合法高值／大退化筛选
Activated-Skills: code-agent-core, parameter-sanity

## Summary

- 固定联合IQR阈值，原始100例上单次计算。
- 同一实例从静态和动态两侧共同删除；正收益和负收益均保留原符号。
- 四列表格第一列注明n，另存全部阈值与逐例清单。

## Assumptions

统计高值是筛选候选，不代表无效数据；原主表不覆盖。

## Open Questions

None

## Context (system/module)

`filter_fault_test100_results.py`；输出 `outputs/fault_test100_filtered_sensitivity_20261117_104520`。

## Parameter Table

| 参数 | 类型 | 值 | 范围／依赖 |
|---|---|---|---|
| --multiplier | float | 1.5 | 有限正数，本次固定 |
| quantile method | str | linear | NumPy线性插值，原100例计算 |
| dynamic high condition | bool | D＞Q3(D)＋1.5×IQR(D) | 严格大于 |
| deterioration condition | bool | D−S＞0且D−S＞Q3(D−S)＋1.5×IQR(D−S) | 仅正退化尾部 |
| combined condition | bool | 两条件取并集 | 重叠只删除一次 |
| iteration | int | 1 | 不以改善为目标反复删样本 |
| paired deletion | bool | true | 静态动态共同删除同一实例 |
| input pairs / rows | int | 19 / 1900 | 原18组＋仅共享K/V set1 |

## Constraint Rules

- 原结果须对应同一新100例manifest、相同实例顺序／seed／场景，两侧均合法且目标有限。
- 不重新计算筛选后阈值；不删除低目标或改善较大的样本；不将负改善改为0。
- 各组n不同且保留实例可能不同，跨架构比较应以原同100例表为依据。

## Preset Profiles

- 原100例：主结果。
- 本次联合1.5×IQR筛选：敏感性分析，不替换主结果。

## Validation Notes

7项测试通过，生产每组剔除0～4例、共30条模型配对记录，保留1870条；四列输出、来源哈希和独立重算均通过。
