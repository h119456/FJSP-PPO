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
