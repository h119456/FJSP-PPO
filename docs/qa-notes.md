Doc-ID: QA-2026-11-04-001
Doc-Type: qa
Owner: user
Date: 2026-11-04
Version: v1.0
Status: final
Source: 独立成本降低是什么意思
Activated-Skills: qa-basic

## Summary

- 独立成本是应力消除成本与补偿加工成本之和。
- 正式测试中，MLP 平均为 16.890，GNN 平均为 9.686，降低 42.65%。
- 本次目标值为完工时间加上独立成本的 0.5 倍。

## Assumptions

- 数值来自本次正式对比的 20 个测试实例，为每实例均值。

## Open Questions

None

## Question

独立成本降低是什么意思？

## Answer

独立成本指调度方案额外执行应力消除、补偿加工所产生的成本，与加工耗时分别计量，不是模型训练费用。
MLP 的应力消除成本为 11.430、补偿加工成本为 5.460，合计 16.890；GNN 分别为 6.996、2.690，合计 9.686。
GNN 的应力消除次数均值从 35.9 降至 22.3，补偿次数均值从 19.95 降至 9.5，表明本次结果中额外处理更少。
本次测试的目标为 $J=C_{\max}+0.5(C_{\mathrm{relief}}+C_{\mathrm{compensation}})$，等待成本权重为零。
因此，目标均值降低的约 4.285 中，约 3.602 来自成本项，约 0.683 来自完工时间。
这些数值是仿真配置中的成本单位，不能直接解释为人民币金额。

## References

- `src/ppo_rs/environment/fjsp_env.py`：`metrics` 与 `_partial_objective`。
- `data/dataset/test/instance_20.json`：目标函数权重。
- `outputs/comparison_20261104_105537_5bbd5efc/summary.json`：正式测试结果。

---

Doc-ID: QA-2026-11-04-002
Doc-Type: qa
Owner: user
Date: 2026-11-04
Version: v1.0
Status: final
Source: 为何GNN可以降低应力消除和补偿次数？
Activated-Skills: qa-basic

## Summary

- 自然放置达到就绪条件可以替代主动应力消除。
- 当前仿真中主动应力消除使变形达到饱和值，可能进一步触发补偿。
- GNN 的关系信息有助于协调等待与机器使用；本次统计符合这一解释，但尚无因果消融验证。

## Assumptions

- 按当前代码及正式测试统计解释，未新增训练或逐决策反事实实验。

## Open Questions

- 优势中图结构、额外信息、参数量和随机种子的各自贡献尚未分离。

## Question

为何 GNN 可以降低应力消除和补偿次数？

## Answer

两种模型优化同一目标，差异在于学到的调度策略及输入信息。环境允许自然放置满足工艺就绪条件，也允许付费主动应力消除提前就绪。
当前自然变形按 $D(t)=A(1-e^{-Bt})$ 增长，主动应力消除直接稳定到 $A$；若此时超过补偿阈值，就必须补偿后才能继续加工。
因此，合理利用自然放置，并在就绪且未超限时衔接下一工序，可能同时省去主动应力消除和后续补偿；等待过久仍可能超限。
GNN 额外读取工件、工序、机器及其关系信息，便于结合就绪进度、变形参数和资源占用协调调度；MLP 也读取候选动作中的变形和成本信息，因此并非只有 GNN 能学会此策略。
本次应力释放等待均值从 MLP 的 11.646 增至 GNN 的 59.702，应力消除和补偿次数分别降至 22.3 和 9.5，与更多利用自然放置的解释一致。
这些均为每实例、跨工件累计量的均值，等待可以与其他工件加工并行，不能将其直接加到 makespan。
目前不能把下降完全归因于 GNN 架构，仍需逐决策轨迹和控制输入、参数量的多 seed 消融验证。

## References

- `src/ppo_rs/environment/fjsp_env.py`：`check_process_readiness`、`_normal_actions`、`_stress_relief_actions`、`_compensation_actions`、`get_graph_state`。
- `src/ppo_rs/physics/deformation.py`：自然变形曲线。
- `src/ppo_rs/physics/stress_relief.py`：主动应力消除后的饱和变形。
- `outputs/comparison_20261104_105537_5bbd5efc/summary.json`：正式测试统计。

---

Doc-ID: QA-2026-11-04-003
Doc-Type: qa
Owner: user
Date: 2026-11-04
Version: v1.0
Status: final
Source: 熵系数是什么？训练进程
Activated-Skills: qa-basic

## Summary

- 熵系数控制PPO对动作分布多样性的奖励强度，影响探索程度。
- 本次动态微调从0.01线性衰减到0.003，并非随机动作概率。
- 上海时间12:48检查：MLP完成2000回合，GNN最新checkpoint为1800回合。

## Assumptions

- 训练状态为查询时快照，后续会继续更新。

## Open Questions

None

## Question

熵系数是什么，动态训练进行到哪里？

## Answer

策略熵衡量动作概率分布的分散程度，熵系数决定训练损失中奖励这种分散程度的权重。
当前损失为 $L=L_{\mathrm{policy}}+0.5L_{\mathrm{value}}-\beta H(\pi)$，较大的 $\beta$ 更鼓励探索，较小的值允许策略更集中。
例如两种动作概率0.5/0.5比0.99/0.01的熵更高；0.01不是1%的随机动作概率。
本次动态阶段熵系数从0.01降至0.003，前期保留探索，后期逐渐集中；确定性评估仍选择概率最大的合法动作。
MLP已完成，最佳动态验证目标为43.8774，位于第1300回合；GNN最新checkpoint为1800/2000，当前最佳动态验证目标39.9906，位于第1700回合。

## References

- `src/ppo_rs/algorithms/ppo.py`：PPO熵项。
- `src/ppo_rs/train.py`：学习率与熵系数的线性计划。
- `outputs/dynamic_training/`：本次运行日志与Checkpoint。

---

Doc-ID: QA-2026-11-04-004
Doc-Type: qa
Owner: user
Date: 2026-11-04
Version: v1.0
Status: final
Source: 熵系数作用在哪里，如何鼓励尝试的
Activated-Skills: qa-basic

## Summary

- 熵系数乘在PPO损失的负熵项上，通过反向传播影响Actor的动作概率。
- 熵项倾向于避免分布过早集中，训练时按分布采样使其他合法动作有机会被选择。
- 实际更新同时受奖励优势、价值损失和共享参数等因素影响，不保证每次熵都增大。

## Assumptions

- 概率例子仅作说明，并非本次轨迹的实际测量值。

## Open Questions

None

## Question

熵系数作用在哪里，如何鼓励尝试？

## Answer

当前代码在PPO更新时计算 $L=L_{\mathrm{policy}}+c_vL_{\mathrm{value}}-\beta H(\pi_\theta(\cdot\mid s))$，其中 $H=-\sum_a p_a\log p_a$，$\beta$ 为熵系数。
最小化损失时，负熵项产生的梯度倾向于提高熵，经Actor网络参数改变logits和softmax概率，使动作分布不易过早集中在少数动作上。
训练收集回合时使用`Categorical(logits=logits).sample()`，概率得到保留的其他合法动作便有机会被采样，其回报再进入下一轮PPO更新。
例如0.99/0.01的分布熵约0.056，0.7/0.3的熵约0.611；在其他损失项相同且熵系数为0.01时，后者使损失低约0.00555。
熵系数仅控制此项梯度强度，不直接设置随机动作比例或softmax温度，亦不保证每步都提高熵。
确定性评估采用argmax；动作候选集和mask仍限制其只能选择合法动作。

## References

- `src/ppo_rs/algorithms/ppo.py:171`：损失、反向传播与优化器更新。
- `src/ppo_rs/models/actor_critic.py`：分类分布、采样和熵计算。
- `src/ppo_rs/models/graph_actor_critic.py`：GNN采用相同采样与熵机制。

---

Doc-ID: QA-2026-11-04-005
Doc-Type: qa
Owner: user
Date: 2026-11-04
Version: v1.0
Status: final
Source: 现在Actor和critic的loss分别是什么
Activated-Skills: qa-basic

## Summary

- Actor采用PPO clipped surrogate损失，并加上负熵正则。
- Critic采用预测状态价值与GAE回报目标的均方误差。
- 两模型使用同一损失公式，当前value_coef为0.5、clip_ratio为0.2。

## Assumptions

- 数值为读取时训练历史中最近一次PPO更新的记录，是该次更新内多个mini-batch和epoch的平均。

## Open Questions

None

## Question

现在Actor和Critic的loss分别是什么？

## Answer

Actor的策略项为 $L_{\mathrm{policy}}=-\mathbb{E}[\min(r_t\hat A_t,\operatorname{clip}(r_t,0.8,1.2)\hat A_t)]$，其中 $r_t=\pi_\theta(a_t\mid s_t)/\pi_{\mathrm{old}}(a_t\mid s_t)$，$\hat A_t$ 为标准化后的GAE优势。
计入熵正则后，Actor部分为 $L_{\mathrm{actor}}=L_{\mathrm{policy}}-\beta\mathbb{E}[H(\pi_\theta)]$；日志中的`policy_loss`未包含熵项。
Critic原始损失为 $L_{\mathrm{value}}=\mathbb{E}[(V_\phi(s_t)-R_t^{\mathrm{GAE}})^2]$，目标由标准化之前的GAE优势加收集轨迹时的价值预测构成。
总损失为 $L=L_{\mathrm{policy}}+0.5L_{\mathrm{value}}-\beta\mathbb{E}[H]$；MLP的Actor和Critic独立，GNN共享图编码器、分别输出动作评分和价值。
本次读取MLP第2000回合记录：policy_loss=-0.00173656、value_loss=1.27158514、entropy=0.25409609、beta=0.003。
本次读取GNN第1930回合记录：policy_loss=-0.00637944、value_loss=0.18455353、entropy=0.32877142、beta=0.00324623；训练尚在继续。

## References

- `src/ppo_rs/algorithms/ppo.py:150`：策略clip、价值MSE和组合损失。
- `src/ppo_rs/algorithms/gae.py`：GAE与回报目标。
- `outputs/dynamic_training/`：两模型训练历史。

---

Doc-ID: QA-2026-11-04-006
Doc-Type: qa
Owner: user
Date: 2026-11-04
Version: v1.0
Status: final
Source: 可否认为GNN的效果更好，现在数据集的设定是否合理
Activated-Skills: qa-basic

## Summary

- 本次相同预算下，GNN完整方案的静态和动态测试解质量均优于MLP；尚不能归因于架构本身或泛化到其他规模。
- 现有数据适合初步实验，静态及动态基础实例跨train/validation/test均无重复。
- 动态测试仅8个有事件实例、故障仅3个，无法充分支持动态鲁棒性结论。

## Assumptions

- 当前分析基于已完成seed7结果及当前数据生成代码，不修改数据或重训。

## Open Questions

- 真实生产中的事件分布及物理参数是否匹配当前合成设定尚未校准。

## Question

能否认为GNN更好，以及当前数据集设定是否合理？

## Answer

在本次合成分布、目标函数和配对seed7下，GNN完整方案具有更好的解质量：静态测试目标降低9.35%，动态测试降低8.85%，动态20实例全胜。
GNN额外读取图信息、参数量约10.93倍，静态训练耗时约10.8倍；这不是等参数、等信息、等计算成本的架构消融。
100/20/20划分、按验证集选best、训练事件随机重采样、固定评估事件及隐藏未来信息的原则合理；剔除动态事件后复核，两套数据的基础调度实例跨划分均无重复。
训练无事件65%、仅释放20%、仅故障10%、混合5%；评估为12/5/2/1，即60%/25%/10%/5%，仅为近似匹配。
在2000回合下期望约700回合有事件、约100回合为混合场景；事件比例是否合理取决于应用分布。
单条固定轨迹和极少的故障/混合实例限制稳健性；延迟释放为预先存储但策略不可见的既有工件，不等同于持续到达、数量和属性变化的新订单流。
当前6至10工件、固定6台机器、相同参数分布只衡量同分布泛化。物理与成本设定为合成，需改变等待、阈值和成本权重检验策略偏好。
推荐新增独立基础实例，对同一实例分别构造四种场景、多个固定事件种子，按场景报告，再按目标事件概率加权；多seed训练从静态到动态完整配对。
只看8个有事件实例，MLP微调前后43.2450到43.5596，GNN39.8660到39.6860，当前不足以说明动态微调显著提升应急能力。

## References

- `scripts/generate_dataset.py`、`scripts/generate_dynamic_dataset.py`。
- `data/dataset/README.md`、`data/dataset_dynamic/manifest.json`。
- `outputs/20261104_dynamic_training_data1/summary.json`。

---

Doc-ID: QA-2026-11-04-007
Doc-Type: qa
Owner: user
Date: 2026-11-04
Version: v1.0
Status: final
Source: 已经预留了动态学习的接口了吗
Activated-Skills: qa-basic

## Summary

- 动态微调加载接口和固定输入维度已具备，dataset2静态配置也使用unified23/compact21。
- MLP预留4维故障状态；GNN节点预留2维事件状态，静态时为零。
- dataset2专用动态数据及微调配置尚未准备，现有动态生成器仍绑定dataset1规模与机器池。

## Assumptions

- 沿用当前环境语义版本及模型结构进行静态到动态迁移。

## Open Questions

None

## Question

是否已预留动态学习接口？

## Answer

已提供配置initialize_from和CLI --initialize-from，仅载入权重并重置优化器、计数及学习率计划，适用于两种模型的动态微调。
unified23的后4维为故障机器比例、平均及最大剩余维修时间、中断工件比例；GNN节点13维中的后2维表达事件影响及剩余恢复时间。
环境已支持工件释放和故障暂停维修恢复；工件释放后更新可见工件及图结构，动作特征保持compact21。
因此dataset2静态模型可以为后续动态训练保留相同输入结构，但这不表示dataset2动态数据和专用配置已经存在。
目前动态生成器仍读取data/dataset并固定四台普通机器的事件池、100/20/20规模，需适配dataset2的实际设备和划分后再生成动态版本。

## References

- `src/ppo_rs/train.py`：initialize_from与迁移校验。
- `src/ppo_rs/environment/fjsp_env.py`：动态状态、图和事件执行。
- `scripts/generate_dynamic_dataset.py`：当前dataset1生成设置。

---

Doc-ID: QA-2026-11-05-001
Doc-Type: qa
Owner: user
Date: 2026-11-05
Version: v1.0
Status: final
Source: 总结MLP与GNN训练结果
Activated-Skills: qa-basic

## Summary

- 本次 dataset2 静态、非零辅助成本实验中，最佳 GNN 在相同 100 个验证实例上的综合目标比最佳 MLP 低 13.63%，makespan 低 11.53%。
- GNN 总额外成本低 25.48%，主要来自去应力减少；补偿次数和成本分别高 15.42% 和 22.46%。
- GNN 完成 20,000 回合；MLP 按用户要求在记录到 18,660 回合时停止；两组最佳权重及记录已保存在各自 `_cost` 目录。
- 结果来自 seed7 和验证集选出的最佳 checkpoint，尚未进行本轮固定 200 实例测试集评估与多训练种子复核。

## Assumptions

- 以两组 `best_evaluation.json` 比较最佳模型；其中数值为每个完整加工实例的均值。
- 使用本轮实际配置 unified23/compact21，即全局 23 维、局部动作 21 维；不与新建完整组 23/25 实验混用。

## Open Questions

- 尚需在固定 200 实例测试集、多训练种子及其他规模上检验泛化与稳定性。
- 旧日志缺少完整 episode 墙钟计时，推理速度需按新增统一计时方式重新评估。

## Question

总结 MLP 与 GNN 本轮训练结果。

## Answer

两组均使用 dataset2 静态非零成本数据（训练 1000、验证 100、测试 200），相同 seed7/eval_seed1007、8 个 DDP 进程及核心 PPO 超参数。全局 batch64、每次更新 10 回合、PPO4 轮、学习率 3e-4 到 5e-5、熵系数 0.02 到 0.005，计划预算 20,000 回合。综合目标为 $J=C_{\max}+0.5C_{\mathrm{relief}}+0.5C_{\mathrm{compensation}}$。

| 最佳模型验证指标 | MLP | GNN | GNN 相对变化 |
| --- | ---: | ---: | ---: |
| 综合目标 | 70.1549 | 60.5911 | -13.63% |
| Makespan | 59.6012 | 52.7263 | -11.53% |
| 总额外成本 | 21.1074 | 15.7296 | -25.48% |
| 去应力次数 | 55.16 | 38.32 | -30.53% |
| 去应力成本 | 17.8344 | 11.7216 | -34.28% |
| 补偿次数 | 14.20 | 16.39 | +15.42% |
| 补偿成本 | 3.2730 | 4.0080 | +22.46% |

100 个验证实例名称一一对应；GNN 综合目标胜 100/100，makespan 胜 97/100，总额外成本胜 79/100。两组均全部完成，未截断，工艺就绪违约与调度约束违约均为零。makespan 使用仿真时间单位，次数是跨实例均值，成本为数据集设定的成本单位。

GNN 完成 20,000 回合、6,928,907 环境步，耗时约 21 小时 11 分；最佳在第 19,500 回合，末次验证目标 61.1110。MLP 记录到 18,660 回合、6,010,693 环境步，耗时约 2 小时 12 分；最佳在第 7,500 回合，末次验证第 18,500 回合目标 71.4402。MLP 最近保存权重为第 18,600 回合，不能将第 18,660 回合记录视为存在对应权重。

GNN 在本次验证集上获得更好的解质量，其较少的去应力操作抵消了较高的补偿成本。MLP 在最佳点后未出现新的验证最优；GNN 后期仍有改进。相同核心超参数不等于相同参数量、输入信息或计算成本；两组最终实际回合数也不同，故不能将此作为严格等计算预算的架构优劣定论。

旧日志中策略推理累计时间的单实例均值为 MLP 0.3081 秒、GNN 5.3669 秒，未覆盖完整仿真过程，也不是优化后统一基准。训练耗时为本次实际运行记录，不作为后续优化版本的固定速度结论。

## References

- `outputs/20261104_mlp_jingtai_data2_cost/best_evaluation.json`、`validation_history.json`、`config.json`、`checkpoint_summary.json`、`status.json`。
- `outputs/20261104_gnn_attention_jingtai_data2_cost/best_evaluation.json`、`validation_history.json`、`config.json`、`status.json`。
- `outputs/20261104_mlp_jingtai_data2_cost/artifact_merge_report.json`。
- `data/dataset2/static/train/` 实例的 `objective` 字段。

---

Doc-ID: QA-2026-11-05-002
Doc-Type: qa
Owner: user
Date: 2026-11-05
Version: v1.0
Status: final
Source: 已划分训练集、验证集、测试集，训练seed的意义是什么
Activated-Skills: qa-basic

## Summary

- 数据划分用于检查未见实例上的泛化，训练seed用于检查训练随机性下结果是否稳定。
- 固定划分后，只更换训练seed并从头独立训练，无需重新生成或重新划分数据。
- seed影响初始权重、训练实例顺序、动作探索、PPO打乱以及动态训练事件采样。

## Assumptions

- 使用当前PPO训练入口，所有方法沿用相同数据划分、固定验证／测试实例和事件轨迹。

## Open Questions

None

## Question

已经划分训练集、验证集、测试集，为什么还需要多个seed？

## Answer

训练seed是控制伪随机过程的整数。即使三个集合不变，网络初始权重、实例访问顺序、强化学习探索动作和PPO mini-batch顺序仍影响学到的策略；动态训练每次reset采样事件也受seed影响。当前代码将训练seed用于这些过程，eval_seed独立用于验证。

可以固定现有划分，使用seed7、17、27分别从头训练同一配置，每次由同一validation规则选择best，再在相同test及固定事件轨迹上评估，报告三次成绩的均值与标准差。静态→动态训练时每条链绑定对应seed的静态来源，保留独立初始化来源与历史。这里不需要重新划分数据，也不是从同一checkpoint续训三次。

多个测试实例回答一份训练所得模型能否泛化；多个训练seed回答重新训练是否仍获得相近表现。单seed适合初步实验，正式方法比较建议至少3个seed；这是实验可靠性建议，并非程序运行的必要条件。固定seed有助于复现，但跨硬件、库版本和非确定性算子不保证逐位相同。

## References

- `src/ppo_rs/train.py`：set_seed、_episode_order、逐回合环境seed及eval_seed。
- `src/ppo_rs/cli_utils.py`：Python与PyTorch随机数初始化。
- `src/ppo_rs/models/actor_critic.py`：Categorical动作采样。
- `src/ppo_rs/algorithms/ppo.py`：PPO随机排列。

---

Doc-ID: QA-2026-11-05-003
Doc-Type: qa
Owner: user
Date: 2026-11-05
Version: v1.0
Status: final
Source: 当前训练是否保留目标函数下降曲线
Activated-Skills: qa-basic

## Summary

- MLP和GNN均保存逐回合训练目标及每500回合的验证目标。
- 本轮enable_plots=false，尚未生成曲线图片，原始历史可用于后续绘图。
- 收敛判断优先使用固定验证集曲线；训练实例不同，逐回合目标不保证单调下降。

## Assumptions

- 当前训练指outputs/fullstate_jingtai_set1。

## Open Questions

None

## Question

当前训练是否保留了目标函数下降曲线？

## Answer

保留了绘图所需原始数据，但关闭了自动出图。检查时MLP的training_history.json有20,000条记录，validation_history.json有40个点；GNN分别有7,630条记录和15个验证点，最新验证在7,500回合。训练与验证历史包含episode、total_environment_steps和objective，可按回合数或环境步数绘图；当前零额外成本条件下目标等于makespan。GNN继续训练时记录会增加。

## References

- `outputs/fullstate_jingtai_set1/MLP/training_history.json`、`validation_history.json`。
- `outputs/fullstate_jingtai_set1/GNN/training_history.json`、`validation_history.json`。
- `src/ppo_rs/train.py`：历史保存与enable_plots控制。

---

Doc-ID: QA-2026-11-05-004
Doc-Type: qa
Owner: user
Date: 2026-11-05
Version: v1.0
Status: final
Source: GNN比MLP多的参数在哪里
Activated-Skills: qa-basic

## Summary

- 本轮MLP共92,162参数，GNN共1,002,274参数，相差910,112，约10.88倍。
- GNN两层异构Attention共759,968参数，占GNN总参数75.8%，是额外参数的主要来源。
- 参数按节点／关系类型共享，实例节点数量增加会增加计算量，不增加模型参数量。

## Assumptions

- 当前完整输入为全局23、动作25，hidden_dim128，GNN节点13、边2、3类节点、10类有向关系、两层4头Attention。

## Open Questions

None

## Question

GNN比MLP多出的参数集中在哪里？

## Answer

按当前网络定义中的Linear权重／bias及LayerNorm参数逐模块计算如下；Tanh、池化、softmax与图连接本身没有可训练参数。

| 模块 | MLP参数 | GNN参数 |
| --- | ---: | ---: |
| 三类节点编码器 | 0 | 54,912 |
| 两层异构Attention消息传递 | 0 | 759,968 |
| 状态／全局编码器 | 19,584 | 68,736 |
| 动作特征编码器 | 19,840 | 19,840 |
| Actor评分网络 | 33,025 | 82,177 |
| Critic，GNN共享编码器另计于上方 | 19,713 | 16,641 |
| 总计 | 92,162 | 1,002,274 |

每层Attention为10类关系分别设置128×128的Key和Value矩阵，仅两层的这些矩阵就有2×10×2×128×128＝655,360参数。每层另外包含query16,384、edge_biases80、edge_values2,560、self_projection16,512、message_projection16,512、LayerNorm256，共379,984，乘两层为759,968。

GNN全局编码器将三类节点池化结果与23维状态拼接，首层输入为3×128＋23＝407；Actor将工件、工序、机器、动作及全局嵌入拼接，首层输入5×128＝640。MLP Actor评分首层输入为状态与动作的2×128＝256，因此GNN在这两处也更大。4个注意力头各分得32维，不能将这些矩阵再乘4。相同hidden_dim不代表相同容量。

## References

- `src/ppo_rs/models/graph_attention.py`：关系专属Key／Value与Attention层定义。
- `src/ppo_rs/models/graph_actor_critic.py`：节点编码、全局编码和Actor／Critic。
- `src/ppo_rs/models/actor.py`、`critic.py`：MLP各模块。
- `outputs/fullstate_jingtai_set1/preflight.json`：模型总参数量。

---

Doc-ID: QA-2026-11-05-005
Doc-Type: qa
Owner: user
Date: 2026-11-05
Version: v1.0
Status: final
Source: Actor是否同时完成动作选择和打分
Activated-Skills: qa-basic

## Summary

- Actor给所有合法候选动作输出logits，后续选择逻辑使用这些分数选动作。
- 训练通常按Categorical概率采样，当前确定性验证采用argmax。
- Critic估计状态价值，不负责给候选动作排序。

## Assumptions

- 指当前MLP／GNN策略的标准训练及确定性评估路径。

## Open Questions

None

## Question

Actor同时完成动作选择和打分吗？

## Answer

从完整策略流程看，Actor打分服务于动作选择。具体实现中，网络先为每个合法候选动作输出一个logit；Categorical将logits归一化为概率。训练时按该概率分布采样以探索；确定性验证／测试选择分数最高的候选动作。动作选择逻辑位于ActorCritic.act／GraphActorCritic.act，评分由Actor网络执行，因此是依次完成的两个步骤。候选动作由环境生成，Actor不负责枚举合法性。Critic另行预测状态价值供PPO训练；Actor的logit不是动作的预计makespan或Q值。

## References

- `src/ppo_rs/models/actor.py`：候选动作logits与mask。
- `src/ppo_rs/models/actor_critic.py`：Categorical采样与argmax选择。
- `src/ppo_rs/models/graph_actor_critic.py`：图条件动作评分、状态价值及选择。

---

Doc-ID: QA-2026-11-05-006
Doc-Type: qa
Owner: user
Date: 2026-11-05
Version: v1.0
Status: final
Source: 核对当前Actor、奖励、Critic、TD、GAE与PPO损失的训练逻辑
Activated-Skills: qa-basic

## Summary

- Actor输出动作logits并选择动作，奖励由环境目标增量产生，Actor分数不是奖励。
- 采样期Critic预测当前状态价值；下一决策点预测的价值用于非终止TD残差。
- Actor使用标准化优势和PPO裁剪损失，Critic使用原始优势构造固定目标的均方误差。

## Assumptions

- 对应当前dataset1_cost0完整输入MLP／GNN；gamma=1、lambda=0.95、clip_ratio=0.2。

## Open Questions

None

## Question

当前训练是否为Actor选择动作并评分R，TD为R加下一状态价值减当前价值，优势为TD折扣和，Actor使用概率比乘优势，Critic使用优势加旧价值减新价值？

## Answer

整体链路接近，但奖励来源、PPO裁剪以及Critic损失平方需纠正。Actor对合法候选动作输出logits，经Categorical采样；环境执行动作后返回奖励$r_t=-(J_{t+1}-J_t)$与下一观测。本轮零额外费用时$J$为当前累计完工时间目标，正常完成回合累计奖励等于负makespan。Critic采样时预测$V_{\mathrm{old}}(s_t)$，并保存对应已选动作的旧log_prob与价值。

TD残差为$\delta_t=r_t+\gamma m_tV_{\mathrm{old}}(s_{t+1})-V_{\mathrm{old}}(s_t)$，其中$m_t$非终止为1、结束为0。代码下一状态价值从下一决策的保存价值取得，完整回合结束后不bootstrap；当前实现对terminated与truncated均置结束mask。

GAE倒序计算$\hat A_t=\delta_t+\gamma\lambda m_t\hat A_{t+1}$；本轮$\gamma\lambda=0.95$。原始优势生成价值目标$y_t=\hat A_t+V_{\mathrm{old}}(s_t)$，Actor使用整批标准化后的$\tilde A_t$，不得用标准化优势重新构造Critic目标。

同一已采样动作概率比为$\rho_t=\pi_\theta(a_t\mid s_t)/\pi_{\mathrm{old}}(a_t\mid s_t)$。Actor损失为$L_\pi=-\operatorname{mean}_t[\min(\rho_t\tilde A_t,\operatorname{clip}(\rho_t,0.8,1.2)\tilde A_t)]$。Critic损失为$L_V=\operatorname{mean}_t[(y_t-V_\phi(s_t))^2]$，平方与均值不可省略。

合并损失为$L=L_\pi+0.5L_V-\beta H(\pi_\theta)$，熵系数从0.02下降至0.005。每次采集全局10回合后，使用全局batch64更新4个epoch；该块的旧概率、原始优势和价值目标在这些更新期间固定，不对其反向传播。GNN共享图编码器同时接收策略和价值两部分损失的梯度。

## References

- `src/ppo_rs/environment/fjsp_env.py`：step奖励与_partial_objective。
- `src/ppo_rs/algorithms/ppo.py`：collect_episode、优势标准化、PPO裁剪、MSE与合并损失。
- `src/ppo_rs/algorithms/gae.py`：终止mask、TD残差、GAE与returns。
- `src/ppo_rs/algorithms/rollout_buffer.py`：保存采样期标量及固定价值目标。
- `outputs/fullstate_jingtai_set1/GNN/training_config.json`：当前超参数。

---

Doc-ID: QA-2026-11-05-007
Doc-Type: qa
Owner: user
Date: 2026-11-05
Version: v1.0
Status: final
Source: 熵奖励是否用于鼓励探索不同动作
Activated-Skills: qa-basic

## Summary

- 熵项鼓励策略保留多个合法动作的选择概率，减少过早集中到单一动作。
- 当前实现将负熵项加入优化损失，不修改环境奖励或GAE中的奖励。
- 熵系数从0.02降至0.005，逐步减弱探索约束。

## Assumptions

- 对应当前PPO训练损失与确定性评估配置。

## Open Questions

None

## Question

熵奖励是为了鼓励探索不同动作吗？

## Answer

是。动作分布熵为$H(\pi)=-\sum_a\pi(a\mid s)\log\pi(a\mid s)$；概率分布越分散，熵越高。最小化损失中的$-\beta H$会鼓励保留多个合法候选动作的概率，降低策略过早变得近乎确定的倾向。它不保证所有动作都会被探索，也不强制最终均匀选择；策略收益目标与熵项共同决定更新。当前系数从0.02降至0.005，训练后期探索倾向减弱。熵只作为损失正则项，环境奖励及目标函数不变；当前确定性验证／测试仍取概率最高动作。

## References

- `src/ppo_rs/algorithms/ppo.py`：entropy_mean与合并损失。
- `src/ppo_rs/train.py`：熵系数日程。
- `src/ppo_rs/models/actor_critic.py`：Categorical与确定性选择。

---

Doc-ID: QA-2026-11-05-008
Doc-Type: qa
Owner: user
Date: 2026-11-05
Version: v1.0
Status: final
Source: 当前训练是否需要Dropout以及应放在MLP还是Attention层
Activated-Skills: qa-basic

## Summary

- 当前没有充分过拟合证据，不建议直接向正在运行的PPO加入Dropout。
- Dropout会给动作概率重算引入随机掩码噪声，需要处理PPO采样与更新的一致性。
- 若进行新实验，先检查特征层或独立Critic分支的小比例Dropout，Attention权重Dropout作为后续消融。

## Assumptions

- 用户在询问建议，本轮不修改模型、训练配置或当前进程。
- 当前MLP／GNN均无Dropout，GNN包含LayerNorm；已有表现差异不单独证明过拟合。

## Open Questions

- 尚无训练／固定验证实例的成对泛化差距及多seed证据，无法确认Dropout是否改善最终性能。

## Question

当前训练需要加Dropout吗；若加，应放在MLP层还是Attention层？

## Answer

不建议仅因GNN参数更多或MLP进入平台就增加Dropout。应先判断相同训练分布下训练成绩是否持续改善而固定验证成绩持续恶化；单次验证波动不足以判断。熵项鼓励动作探索，Dropout用于正则化，两者作用不同。

PPO使用同一已采样动作的新旧概率比。若采样与更新采用不一致的Dropout启用方式，或重算时使用独立随机掩码，概率比可能混入额外噪声，不能把普通监督学习中直接插入Dropout的流程照搬到Actor。若对Actor或共享图编码器增加Dropout，应在新实验中明确随机掩码处理及采样／更新／评估模式，并验证未更新参数时概率比的一致性。

如确需测试，可先从0.05～0.1的候选比例做小范围消融。MLP可放在隐藏层激活之后；GNN优先考虑节点编码器或消息输出的特征Dropout，再考虑Attention系数Dropout，后者直接随机削弱图上消息路径。独立Critic输出分支隐藏层的Dropout对Actor概率比影响更小，但只直接正则化价值分支；GNN共享编码器中的Dropout会同时影响Actor与Critic，不能视为Critic专属改动。最终动作logits和标量价值输出不作为优先放置位置。

## References

- `src/ppo_rs/algorithms/ppo.py`：概率比、采样与更新。
- `src/ppo_rs/models/actor.py`、`critic.py`：MLP隐藏层。
- `src/ppo_rs/models/graph_actor_critic.py`：共享图编码器与独立输出分支。
- `src/ppo_rs/models/graph_attention.py`：Attention、消息聚合和LayerNorm。

---

Doc-ID: QA-2026-11-05-009
Doc-Type: qa
Owner: user
Date: 2026-11-05
Version: v1.0
Status: final
Source: GNN是否稳定在34左右还是持续下降
Activated-Skills: qa-basic

## Summary

- 固定验证集目标已进入34附近的波动区间，近期区间均值仍有缓慢改善。
- 第9500回合最佳34.1139；最新第12000回合34.4100，尚未连续创新低。
- 不能据此确认完全收敛，继续按既定20k预算观察。

## Assumptions

- 分析当前fullstate_jingtai_set1/GNN的固定100实例验证记录，而非不同训练实例的逐回合目标。

## Open Questions

None

## Question

GNN已经稳定在34左右，还是持续下降？

## Answer

北京时间2026-11-05 17:55检查时，训练保存到12350回合，最新验证在12000回合。第3500～6000回合的6次验证均值为34.7473，第6500～9000为34.6689，第9500～12000为34.3385。后一区间相对前一区间均值改善约0.95%，提示仍有缓慢下降；但最新几个验证点为34.1139、34.6614、34.2903、34.1674、34.3883、34.4100，属于波动而非持续下降。当前最佳保存在9500回合，之后2500回合没有新最优，不能单凭这些点证明完全收敛或之后不再改善。

## References

- `outputs/fullstate_jingtai_set1/GNN/validation_history.json`。
- `outputs/fullstate_jingtai_set1/GNN/training_history.json`。

---

Doc-ID: QA-2026-11-05-010
Doc-Type: qa
Owner: user
Date: 2026-11-05
Version: v1.0
Status: final
Source: 动态训练参数lambda的含义
Activated-Skills: qa-basic

## Summary

- lambda是GAE优势估计参数，本次值为0.95。
- 它决定后续TD残差对当前优势估计的影响衰减速度。
- 它与奖励折扣gamma、优化器学习率不同。

## Assumptions

- 当前动态PPO使用gamma=1.0、gae_lambda=0.95，并采集完整回合。

## Open Questions

None

## Question

lambda是什么参数？

## Answer

lambda是广义优势估计GAE的衰减参数，通常取值[0,1]。在非终止步骤，$\hat A_t=\delta_t+\gamma\lambda\hat A_{t+1}$，因此未来第k步TD残差的权重为$(\gamma\lambda)^k$。本次gamma为1、lambda为0.95，下一步权重0.95，向后10步约0.60。lambda为0时只使用当前一步TD；lambda接近1时纳入更多后续残差，一般降低对Critic自举误差的依赖但增大估计方差。lambda不是学习率，也不是奖励本身的折扣率。

## References

- `src/ppo_rs/algorithms/gae.py`：TD残差与GAE递推。
- `configs/state_ablation/full_mlp_dynamic.json`、`full_gnn_dynamic.json`：gamma与gae_lambda。

---

Doc-ID: QA-2026-11-05-011
Doc-Type: qa
Owner: user
Date: 2026-11-05
Version: v1.0
Status: final
Source: 查询本轮dataset1动态MLP训练结果
Activated-Skills: qa-basic

## Summary

- 动态MLP完成10,000回合、1,184,166环境步，耗时16分54秒。
- 最佳验证目标36.60831，对应第500回合、累计58,863环境步。
- 末轮验证37.11640，后续训练未刷新最优；测试集尚未评估。

## Assumptions

- 本轮指outputs/fullstate_dongtai_set1/MLP，目标为零额外费用条件下的makespan。

## Open Questions

None

## Question

MLP训练结果如何？

## Answer

MLP于北京时间2026-11-05 18:30:50完成本轮动态微调，累计10,000回合、1,184,166环境步，耗时1,014秒；全部训练回合正常结束，无截断。100个固定动态验证实例的最佳目标为36.60831，出现在第500回合；该checkpoint验证100/100正常完成、零调度违规、总额外成本0。最终第10,000回合验证为37.11640338591979，后续验证围绕约37波动，未超过第500回合的最佳结果。应使用checkpoints/best.pt作为验证选模结果，latest和final也已保存。当前没有该轮测试报告；静态best在同一动态验证集的完整初始评估尚未生成，不能用静态验证成绩直接判断微调收益。GNN已于MLP结束后自动启动。

## References

- `outputs/fullstate_dongtai_set1/status.json`。

---

Doc-ID: QA-2026-11-09-001
Doc-Type: qa
Owner: user
Date: 2026-11-09
Version: v1.0
Status: final
Source: 为什么K不需要偏置；FiLM是否学习参数与K、V逐项相乘
Activated-Skills: qa-basic

## Summary

- 当前按关系类型和目标节点分组的点积注意力中，静态类型K偏置产生组内常数，softmax会消去它。
- FiLM学习每类关系的逐维缩放向量，以及V的平移向量；不是额外的稠密矩阵乘法。
- 共享K/V投影矩阵和FiLM参数均通过PPO反向传播更新。

## Assumptions

- 讨论尚未实现的共享K/V＋类型FiLM方案，沿用当前关系内softmax，K调制后不再经过非线性或归一化。

## Open Questions

- None

## Question

为什么K不需要偏置？是否学习矩阵与K、V逐项相乘？

## Answer

设同一目标节点$i$、同一关系$r$的邻居为$j$。如果$\widetilde{k}_{j,r}=\gamma^K_r\odot W_Kh_j+\beta^K_r$，则注意力分数增加$q_i^T\beta^K_r/\sqrt{d}$。该项对组内所有$j$相同，因此被softmax抵消；省略的是此处静态类型K偏置，不是所有注意力架构的K偏置。若跨关系统一softmax，或K调制后增加非线性、归一化，则结论需重新判断。

每层学习共享$W_K,W_V$和按关系索引的$\gamma^K,\gamma^V,\beta^V$表。每张表为$10\times128$，每条边取出对应关系的128维向量，与投影后的K/V逐项相乘，V再逐项加偏置：$\widetilde{k}=\gamma^K_r\odot W_Kh_j$，$\widetilde{v}=\gamma^V_r\odot W_Vh_j+\beta^V_r$。初始化缩放为1、偏置为0，训练通过PPO梯度共同更新；FiLM仍是待实现方案。

---

Doc-ID: QA-2026-11-09-002
Doc-Type: qa
Owner: Codex
Date: 2026-11-09
Version: v1.0
Status: final
Source: 原不共享KV代码是否保留；Q为什么共享；FiLM参数量变化
Activated-Skills: qa-basic

## Summary

- 原关系独立K/V分支保留，shared_kv_film默认false，新结构为显式开启的可选分支。
- 原版每层Query已跨关系共享；关系K仍使同一Query对不同类型消息产生不同匹配。
- 完整state无物理门控模型参数从1002274降至420130，减少582144，约58.1%。

## Assumptions

- hidden128，两层四头，十种有向关系，完整state；不包含物理门控。

## Open Questions

- None

## Question

原有不共享KV代码保留了吗；为什么Q可以共享；使用FiLM后参数量如何变化？

## Answer

关闭shared_kv_film时，每层保留十套独立keys和values；开启时使用各一套共享K/V和三张类型表。原默认参数名、初始化顺序及checkpoint schema保持兼容。

Q表达目标节点当前的信息需求，K表达邻居提供的信息；同一目标节点可用共同的查询空间，通过类型化K学习不同关系的匹配方式。原版K使用各关系独立矩阵，新版K使用共享投影加类型缩放；因此Q共享不等于所有关系的注意力相同。每层的Q仍独立，跨关系共享属于架构取舍，不保证与任意关系独立Q同等表达能力。

两层原K/V共655360参数，新共享K/V共65536，类型FiLM表共7680；其余346914不变，合计1002274→420130。收益来自共享矩阵后用轻量类型参数表达差异，而不是额外添加FiLM本身减少参数。
- `outputs/fullstate_dongtai_set1/MLP/validation_history.json`、`best_evaluation.json`、`checkpoint_summary.json`。
- `outputs/fullstate_dongtai_set1/MLP/training_history.json`。

---

Doc-ID: QA-2026-11-05-012
Doc-Type: qa
Owner: user
Date: 2026-11-05
Version: v1.0
Status: final
Source: 对比动态微调前后验证目标并查询GNN进度与预计时间
Activated-Skills: qa-basic

## Summary

- 同一动态100实例、eval_seed1007、CPU确定性复评：MLP从36.6880改善至36.5920，约0.26%。
- GNN静态权重直接用于动态验证的目标为34.7100；检查时尚未完成首次动态训练验证。
- 18:36检查GNN保存460／10000回合，剩余约2小时，初估20:40左右完成。

## Assumptions

- 前后比较使用同一动态验证集合与CPU设备，不混用静态验证集成绩。
- 预计时间来自训练初期吞吐，尚未充分覆盖验证与后期保存开销。

## Open Questions

- GNN动态微调验证结果需等待首个500回合验证点；两模型测试成绩尚未生成。

## Question

接入动态训练前后验证目标怎么变化，GNN训练进度和预计时间如何？

## Answer

本轮补评两份静态best在完整动态验证集的目标，并复评动态MLP best。MLP微调前36.6880153914417，微调best第500回合为36.591950000000004，约改善0.2618%；逐实例49胜、51负，平均收益较小。复评后的动态MLP best与训练时GPU记录36.60831略有不同，共3个实例目标不同，设备间复评结果不能混合作精确归因。相同设备复评的GNN微调前目标34.70997030652208，100实例正常结束、无截断。使用CPU单线程评估，未占用训练GPU或改变权重，结果保存在validation_transfer_comparison.json。

北京时间18:36:29，GNN动态训练460／10000回合，已运行339.6秒，尚无验证点。按初期平均速度剩余7042秒，考虑后续验证及速度波动，约2小时、20:40左右完成，建议留约20分钟浮动。监督进程与8个训练rank正常。

## References

- `outputs/fullstate_dongtai_set1/validation_transfer_comparison.json`。
- `outputs/fullstate_dongtai_set1/MLP/best_evaluation.json`。
- `outputs/fullstate_dongtai_set1/GNN/training_history.json`、`validation_history.json`、`status.json`。

---

Doc-ID: QA-2026-11-05-013
Doc-Type: qa
Owner: user
Date: 2026-11-05
Version: v1.0
Status: final
Source: 第二阶段动态训练的意义以及如何判断达成目标
Activated-Skills: qa-basic

## Summary

- 第二阶段在动态状态分布下继续优化调度策略与价值估计，目标是比静态模型直接部署更有效应对事件。
- 事件模拟与合法性由环境提供，不是模型需要从零学会的功能；当前奖励仍优化makespan。
- 验收依赖固定动态测试集上的微调前后成对比较、分场景可靠性和静态能力保持，不依赖训练预算是否完成。

## Assumptions

- 当前dataset1_cost0两阶段方案：静态best初始化，动态全量PPO微调。
- 用户询问目的与评价标准，本轮不修改训练、选模规则或启动额外测试。

## Open Questions

- 可接受的事件子集改善幅度、静态退化和计算开销阈值需结合业务目标预先确定；当前未设定这些业务阈值。

## Question

第二阶段训练的意义是什么，如何知道是否达成目标？

## Answer

静态阶段学习固定工件集合与正常设备下的调度；动态阶段将训练分布扩展到可观测工件释放、设备故障、维修及中断状态，沿用负目标增量奖励，使策略与Critic适应事件后的负载和剩余加工状态。其核心目标应表述为：在相同动态事件下，相比静态best直接部署，获得更好的有效调度，并保持可接受的无事件表现。自动释放、故障暂停恢复、非法动作过滤由环境实现，零违规不能单独证明微调有效。当前训练并未显式优化交期或最短插单等待，若作为硬业务目标需另行定义和对齐奖励／约束。

验收在冻结评价协议后，用同一基础静态best和微调best、相同测试实例与事件轨迹、相同推理设备及动作选择方式进行成对比较。主要看有事件子集及插单／故障／混合的makespan改善与尾部风险；辅助看插单释放至开工／完工周期、完成率、违规／截断、无事件／静态质量退化及推理成本。事件子集改善不足或单类明显退化时，不能因总体均值改善就宣称全面成功。按业务预定最小有用改善、关键场景容许退化、静态保持及延迟阈值；没有预定阈值时可报告观察到的收益与不确定性，不能事后选择有利阈值。

训练seed重复及配对实例统计用于区分稳定收益和偶然波动。validation选模型并做早停，test用于最终验收；若根据test反复修改方案，需要另留最终测试集。当前代码按动态总体均值选best，65%样本无事件，该选模口径与强调事件适应的目标不完全一致；后续实验应预先指定事件子集或场景平衡验证指标，并保留静态能力约束，不自动修改现有运行。

已有MLP同CPU动态验证复评显示，总体改善约0.26%、事件子集约0.52%、故障子集变差约0.80%。这些结果仅说明收益有限且场景间有取舍，尚无测试与多seed证据支持其已达成稳定动态适应目标。GNN需在其训练／选模后按同协议检验，不能因模型较大或训练完成而直接判定成功。

## References

- `src/ppo_rs/environment/fjsp_env.py`：奖励、事件转移、合法性和指标。
- `src/ppo_rs/train.py`：动态微调初始化与验证选模。
- `outputs/fullstate_dongtai_set1/validation_transfer_comparison.json`：已有同设备微调前后验证复评。
- `docs/feasibility.md`：动态评估与LoRA方案分析。

---

Doc-ID: QA-2026-11-05-014
Doc-Type: qa
Owner: user
Date: 2026-11-05
Version: v1.0
Status: final
Source: 静态模型能否直接运行动态验证集，以及与微调后比较的指标
Activated-Skills: qa-basic

## Summary

- 静态训练模型具备兼容的动态接口，可在动态环境直接推理；缺少动态训练经验不等于无法运行。
- 环境负责释放、故障暂停恢复和合法候选生成，策略在当前可观测状态下持续选择动作。
- 同动态实例／轨迹／设备的直接部署与微调后结果可用事件条件目标改善率等指标比较。

## Assumptions

- 指本轮unified23／enhanced25、GNN节点13接口，沿用当前环境语义。
- 数值采用已有同CPU确定性评估的MLP静态best和动态best，不混用GPU训练时成绩。

## Open Questions

None

## Question

静态模型不具备处理动态事件的能力吗？直接给动态验证集与动态微调后的表现是否有比较指标？

## Answer

本轮静态模型可以直接在动态环境运行：静态阶段保留相同4维动态全局接口和GNN的2维节点接口，环境在事件发生后更新可见工件、设备和合法候选，策略重新选择动作。它未从静态训练学习这些事件字段的有效使用，事件下策略可能依赖已有调度经验泛化，不能保证最优；自动事件执行与安全过滤由环境承担，因而可行率高不单独证明动态学习。

可定义微调改善率$G_E=100(\bar J_{\mathrm{direct},E}-\bar J_{\mathrm{finetuned},E})/\bar J_{\mathrm{direct},E}$，其中E为同一有事件实例集合、相同轨迹和推理设置。逐类场景也分别计算，并补充逐实例胜平负、完成率、违规、尾部完工时间、插单等待和完工周期。后两类服务时长需从事件及排程派生，当前汇总未直接保存。

已有MLP同CPU验证对照：全部100实例36.6880→36.5920，事件35实例37.5611→37.3664；插单20例37.2200→36.8345，故障10例37.6140→37.9164，混合5例38.8194→38.3942。前后均100/100有效、零调度违规。事件子集改善约0.52%，故障退化约0.80%，说明当前有小幅平均收益和场景取舍；单seed验证不能证明稳定泛化。

## References

- `outputs/fullstate_dongtai_set1/validation_transfer_comparison.json`。
- `src/ppo_rs/environment/fjsp_env.py`：事件状态更新、合法候选及动态预留维度。
- `src/ppo_rs/models/actor_critic.py`、`graph_actor_critic.py`：根据当前状态逐步推理。

---

Doc-ID: QA-2026-11-05-015
Doc-Type: qa
Owner: user
Date: 2026-11-05
Version: v1.0
Status: final
Source: 动态MLP是否降低故障处理能力及第二阶段是否收敛
Activated-Skills: qa-basic

## Summary

- 在10个纯故障验证实例上，微调best相对静态直接部署3胜7负，平均makespan增加0.3024，约0.80%。
- 混合事件5例则3胜2负、均值改善，不能据纯故障子集断言所有含故障场景全面退化。
- 二阶段验证在37附近长期波动，无持续下降，表现为平台；无法证明参数或策略已严格收敛。

## Assumptions

- 纯故障前后比较使用既有相同CPU、实例及固定事件轨迹的复评；收敛趋势采用训练期间固定GPU验证记录。
- 当前best按全体100个动态验证实例目标选取，并非按故障子集目标选取。

## Open Questions

- 退化是否稳定存在，尚需多seed及独立测试集验证；数据比例与共享参数取舍只是可能原因，尚未因果验证。

## Question

MLP动态训练是否降低故障处理能力，二阶段训练收敛了吗？

## Answer

限定本次纯故障验证子集，微调后调度效率确实变差：37.6140→37.9164，10例中3例改善、7例变差，平均增加0.3024，约0.80%，中位差增加0.2255。前后都正常完成且无违规，下降的是该子集的调度效率，不是运行或合法性能力。混合5例均值38.8194→38.3942、3胜2负，因此不能扩展为所有故障条件都更差，也不能以10例单seed结果认定稳定退化。

MLP二阶段第500回合验证36.60831为记录最优，之后9500回合未刷新；最终37.1164。验证区间500～2500、3000～5000、5500～7500、8000～10000各5点均值依次37.0893、37.1156、37.0216、37.0515，缺少持续改善趋势。可判断当前超参数与评价协议下进入平台，但不等于到达最优或严格证明收敛。仅延长同样训练没有明确收益依据；更值得先分析事件分场景选模、训练事件覆盖和优化策略。现有best针对含65%无事件的总体目标，故障效益未作为独立选择目标。

## References

- `outputs/fullstate_dongtai_set1/validation_transfer_comparison.json`：逐实例微调前后CPU复评。
- `outputs/fullstate_dongtai_set1/MLP/validation_history.json`：完整20点验证曲线。
- `data/dataset1_cost0/dynamic/validation/`：场景标签。

---

Doc-ID: QA-2026-11-05-016
Doc-Type: qa
Owner: user
Date: 2026-11-05
Version: v1.0
Status: final
Source: MLP二阶段平台是否由参数量太小引起以及GNN动态训练进展
Activated-Skills: qa-basic

## Summary

- MLP容量较小可能限制适应能力，但当前没有容量对照，不能据平台或GNN领先确认。
- MLP摘要输入、动态场景权重、优化设置及随机波动同样可能影响结果。
- 北京时间18:50，GNN动态1530／10000回合，最佳验证34.4878在1000回合，预计20:40左右完成。

## Assumptions

- 当前模型hidden_dim128，MLP92,162参数，GNN1,002,274参数。
- 本轮只回答与核查，不改变运行中训练。

## Open Questions

- 需要宽度消融及多seed以判断容量是否为主要瓶颈。

## Question

是否可能MLP参数量太小？GNN训练进展如何？

## Answer

存在这种可能，但GNN同时拥有图信息、关系结构和更多参数，无法用它的领先单独证明容量不足。增加MLP宽度能增强函数表示，却无法还原输入摘要已丢失的具体关联信息。当前65%无事件混合训练及整体均值选模不专门保护故障子集，优化取舍也可能造成纯故障退化。可固定输入和其他条件，对比hidden_dim128、256及约等参数的448；对应参数92,162、348,162、1,039,362，均按现有结构公式$5h^2+80h+2$计算。更宽模型需要新的配套静态与动态训练，不能直接严格加载当前128维权重。

北京时间18:50:11，GNN动态训练1530／10000回合、190,567环境步，8卡正常，全部1530回合正常完成，无截断。GPU验证500／1000／1500回合分别34.5207／34.4878／34.6203；最佳在1000。已运行19分21秒，按累计速度估算剩余约1小时47分，预计20:40左右，速度变化可能带来约20分钟浮动。微调前GNN基线为CPU复评，不能直接和GPU验证记录混合计算精确改善率。

## References

- `src/ppo_rs/models/actor.py`、`critic.py`：MLP层定义。
- `outputs/fullstate_dongtai_set1/GNN/training_history.json`、`validation_history.json`。
- `outputs/fullstate_dongtai_set1/status.json`。
