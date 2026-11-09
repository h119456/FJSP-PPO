Doc-ID: IMPL-2026-11-04-001
Doc-Type: implementation
Owner: user
Date: 2026-11-04
Version: v1.0
Status: final
Source: 阅读 FJSP-PPO 项目，创建 datascience 环境并安装所需库
Activated-Skills: code-agent-core

## Summary

- 项目位于 `/root/VLA/FJSP-PPO-cloud`，安装时提交为 `4a53207`。
- 项目通过 PPO 求解带时间相关变形约束的柔性作业车间调度，支持 MLP 和 Attention GNN。
- 创建独立 Conda 环境 `datascience`，使用 Python 3.11。
- 根据 `pyproject.toml` 安装运行与测试依赖，并安装项目的 editable 包。
- 已有 36 项测试全部通过，MLP 和 GNN 均通过 GPU 短训练、Checkpoint 回载和绘图检查。

## Assumptions

- 使用当前仓库内容配置环境。
- 当前 NVIDIA 驱动为 535.129.03，使用 CUDA 12.1 版 PyTorch 2.5.1；无需安装独立 CUDA Toolkit。
- 训练仅用于环境检查，正式训练配置仍由用户选择。

## Open Questions

None

## Plan Summary

1. 阅读 README、依赖声明、训练与评估入口、模型代码和测试。
2. 创建 Conda 环境，安装 GPU 版 PyTorch 及项目依赖。
3. 检查依赖一致性、运行已有测试，验证 GPU 上的短训练、模型回载和绘图。

## Changes

- 新建环境路径：`/root/miniconda3/envs/datascience`。
- 本文记录安装方式和验证结果。
- `.gitignore` 增加 `*.egg-info/`，忽略 editable 安装生成的包元数据。
- 项目 GNN 使用原生 PyTorch 实现，无需 PyG、torchvision 或 torchaudio。

### 安装命令

```bash
conda create --name datascience python=3.11 pip -y
conda activate datascience
cd ~/VLA/FJSP-PPO-cloud
python -m pip install 'torch==2.5.1' --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e '.[test]' --index-url https://pypi.org/simple
```

系统默认 pip 镜像在本次安装中返回 502，安装时通过命令参数指定官方源，未修改全局镜像配置。

### 实际版本

| 组件 | 版本 |
| --- | --- |
| Python | 3.11.18 |
| PyTorch | 2.5.1+cu121 |
| NumPy | 2.4.6 |
| PyYAML | 6.0.3 |
| Matplotlib | 3.11.2 |
| pytest | 9.2.0 |
| ppo-rs | 0.1.0，editable 安装 |

完整环境快照保存在 `outputs/environment_setup/pip-freeze.txt` 和 `outputs/environment_setup/conda-environment.yml`。

### 使用方式

```bash
conda activate datascience
cd ~/VLA/FJSP-PPO-cloud
python -m ppo_rs.train --config configs/default.yaml
```

## Tests Run

- `python -m pip check`：通过，未发现依赖冲突。
- 核心包导入、项目包路径检查：通过，项目加载自当前仓库 `src/ppo_rs`。
- NumPy 数组与 PyTorch Tensor 转换：通过。
- CUDA 可用性和 GPU 矩阵乘法：通过，识别 8 张 NVIDIA A800-SXM4-80GB；PyTorch CUDA runtime 为 12.1。
- `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q`：`36 passed in 3.10s`。
- 使用 `configs/default.yaml` 加载配置，在内存中为环境检查覆盖为每种模型 2 个 episode、hidden_dim=32、ppo_epochs=1、mini_batch_size=8、device=cuda:0；MLP 与两层 Attention GNN 均完成 GPU 训练和评估。
- 两种模型均成功保存并以 `weights_only=True` 回载 Checkpoint，回载后评估正常终止且调度约束违规数为 0。
- 两种模型的 `training_curve.png`、`gantt.png`、`model.pt`、`evaluation.json` 均成功生成；结果位于 `outputs/environment_setup/mlp/` 和 `outputs/environment_setup/gnn_attention/`。
- 验证汇总：`outputs/environment_setup/smoke_summary.json`。

## Risks and Rollback Notes

- 冒烟训练只确认运行链路，不用于判断策略性能或训练收敛。
- 环境独立于已有 Conda 环境；如需移除，可在退出该环境后使用 `conda env remove -n datascience`。

## Follow-ups

- 根据实验目标选择 `configs/` 中的正式训练配置。

---

Doc-ID: IMPL-2026-11-04-002
Doc-Type: implementation
Owner: user
Date: 2026-11-04
Version: v1.0
Status: final
Source: 检查并统一dataset上的MLP与GNN完整方案对比条件
Activated-Skills: code-agent-core, parameter-sanity

## Summary

- 用户确认目标是比较现有MLP与GNN完整方案，保留图输入和模型容量差异。
- 现有配置都是state23/action21，但batch64/32、每20/10回合更新不一致；新增配对配置统一训练设置。
- 修复dataset评估脚本忽略Checkpoint输入profile的问题，并输出模型参数量等元数据。
- 自动检查、45项测试与真实dataset两模型10回合冒烟全部通过。

## Assumptions

- 正式对比保持相同episode预算与训练设置，使用验证集best模型评估完整测试集。
- 原实验配置保留，避免改写历史实验定义。

## Open Questions

None

## Plan Summary

1. 交叉检查环境特征、网络结构、训练代码、数据划分和Checkpoint评估规则。
2. 创建同训练设置的独立配置，补充自动检查与评估回归测试。
3. 在datascience环境验证配置、运行全量测试与真实dataset短训练，形成实验协议。

## Changes

- `configs/dataset_compare_mlp.json`、`configs/dataset_compare_gnn.json`：state23/action21、hidden128、2000回合、batch32、每10回合更新、相同LR/entropy退火、seed和验证保存规则。只允许架构参数和输出路径不同。
- `scripts/check_comparison.py`：解析默认值，比较有效配置，检查约束与更新边界，测量实际输入维度和参数量，校验跨split内容重复并记录数据摘要；失败返回非零状态。
- `scripts/evaluate_dataset.py`：根据各Checkpoint记录的state/action profile构造环境；汇总中记录profile、维度、模型类型和参数量。
- `tests/test_comparison.py`：旧配置差异、统一配置、数据改名泄漏、参数约束回归。
- `tests/test_evaluate_dataset_profiles.py`：实际CLI评估MLP14/action25、GNN19/action25以及旧默认profile回退。
- `README.md`、`src/ppo_rs/models/README.md`、`data/dataset/README.md`：更正默认state维度并解释新对比入口。
- `docs/parameter-spec.md`：完整参数表、输入/容量区别、Checkpoint规则、多seed协议与运行命令。

## Tests Run

- 全量测试：`45 passed in 3.64s`；`git diff --check`通过。
- 新配对配置自动检查通过；旧配置检查失败并准确指出batch及更新频率差异。
- 实际数据100/20/20，跨split未发现去除name后的重复实例内容。
- MLP与GNN各在真实dataset上训练10个episode、执行1个PPO epoch并验证20个实例；正常终止、零违规、best Checkpoint均生成。冒烟产物位于 `outputs/dataset_comparison/smoke/`。

## Risks and Rollback Notes

- GNN额外接收完整图输入，参数量1,001,762，MLP为91,650。不能将方案差异全部归因于消息传递。
- 等episode不是等环境决策数、等Adam步数或等计算时间；正式结果需同时披露实际预算。
- 现有推理计时不包含GNN图构造，GPU断点续训未保存CUDA RNG状态；这些限制已记录，未扩展修改训练算法。
- 新配置不覆盖旧配置，冒烟产物与正式训练目录独立。

## Follow-ups

- 按 `docs/parameter-spec.md` 的命令执行2000回合正式配对训练；至少使用3个配对seed，统一用验证集best模型测试。

---

Doc-ID: IMPL-2026-11-04-003
Doc-Type: implementation
Owner: user
Date: 2026-11-04
Version: v1.0
Status: final
Source: 全局mini-batch统一64，静态学习率3e-4到5e-5，预留动态维度，每模型同时GPU0–7，每次训练新建outputs子目录
Activated-Skills: code-agent-core, parameter-sanity

## Summary

- 两份静态对比配置统一为global batch64、静态学习率3e-4到5e-5，其余训练预算保持一致。
- 用户明确单模型训练同时使用8卡，已增加DDP更新与跨rank并行采样，全局episode预算不乘8。
- 保留state23中的4维动态特征、GNN节点13维中的2维动态特征，静态运行时为0；提供同结构动态微调配置。
- 每次新训练在 `/root/VLA/FJSP-PPO-cloud/outputs` 下创建独立时间戳目录，Checkpoint与记录随运行隔离；resume继续原目录。
- 54项测试、真实8卡MLP/GNN静态短训练与静态到动态微调验证全部通过。

## Assumptions

- 两种模型顺序训练，各自一次训练占用GPU0–7。
- mini_batch_size=64为跨卡全局批次，满批每卡8；每10个全局episode执行一次PPO更新。
- 本次只执行短训练验证，未开始正式2000回合实验。

## Open Questions

None

## Plan Summary

1. 对照原MLP多阶段学习率与动态输入结构。
2. 实现DDP全局批次、全局episode采样预算和rank0唯一保存。
3. 增加独立运行目录、Checkpoint迁移校验、动态配置与启动脚本。
4. 先CPU双进程验证，再使用GPU0–7验证静态及动态运行。

## Changes

- `configs/dataset_compare_{mlp,gnn}.json`：batch64、device=cuda、distributed_world_size=8、output_dir为指定绝对根目录、output_dir_mode=unique。
- `configs/dataset_compare_{mlp,gnn}_dynamic.json`：保持相同输入/模型结构及8卡设置，沿用原MLP动态阶段学习率1e-4到2e-5、entropy0.01到0.003，通过 `--initialize-from` 指定实际静态Checkpoint。
- `src/ppo_rs/algorithms/ppo.py`：DDP包装训练forward，rank0广播全局排列，按rank切片global batch；尾批填充以零权重排除，并补偿DDP均值归约，保证全局均值梯度。GNN启用未使用参数检测。
- `src/ppo_rs/train.py`：torchrun初始化NCCL/LOCAL_RANK；按全局episode分配采样、汇总后同步更新；rank0验证/保存/日志；rank0错误广播；保存各rank CPU/CUDA RNG，resume校验world_size。
- 新运行目录名包含上海时间、模型类型、seed和随机后缀，只有rank0分配并广播；CLI默认unique，直接Python API保留existing默认以兼容已有调用。新静态与动态配置显式指定unique。
- 动态迁移校验Checkpoint语义版本、profile、维度与完整GNN结构；动态初始化只载入模型权重，新建Adam/计数/学习率计划，记录来源路径。
- `scripts/train_dataset_comparison.sh`：固定可见卡0–7，先检查一致性，再分别以8进程执行MLP和GNN。
- `scripts/check_comparison.py`：检查global batch与world size，允许unique模式共享输出根，报告每卡批次与动态预留维度。
- 新增 `tests/test_distributed_ppo.py`、`tests/test_distributed_training.py`，更新配置检查及实验文档。

## Tests Run

- 完整测试：`54 passed in 12.57s`；`git diff --check`与启动脚本shell语法检查通过。
- 2rank CPU/Gloo：MLP与GNN的DDP更新和单卡global batch更新接近，涵盖不能整除的批次与小于world_size的尾批、GNN未使用关系。
- 2rank完整训练：参数同步、全局episode计数、rank0单份输出、LR首尾值、唯一目录、恢复训练和rank0错误传播通过。
- 静态dataset全部140个实例检查预留动态维度为0；测试覆盖完整静态轨迹。
- 真实8卡静态MLP与GNN各10个episode、1个PPO epoch，验证完整20实例；均正常终止且零违规，best/latest/final均生成。
- 实际Checkpoint核验：world_size=8、global batch64、per-rank batch8、8份CUDA RNG、无DDP参数名前缀；历史仅10个全局episode且rank覆盖0–7。
- 两模型各从静态best迁移到包含工件释放与故障的动态实例，8卡各运行2个episode；验证正常终止、零违规，输入结构未变，动态学习率端点正确，并创建新的运行目录。
- 静态与动态两套配对配置自动审查均通过；结果位于 `outputs/ddp_fairness_check.json`、`outputs/ddp_dynamic_fairness_check.json`。
- 真实运行及动态迁移目录汇总位于 `outputs/ddp_setup_summary.json`。

## Risks and Rollback Notes

- DDP保持预算与梯度语义，不保证8倍加速；对象汇总及rank0验证存在通信和串行开销。
- GNN CUDA归约可能存在浮点非确定性；同seed不能承诺逐位相同。
- 恢复训练应保持原world_size；动态微调使用initialize-from并新建运行，勿与resume混用。
- 原历史配置仍可单卡使用；新8卡配置要求实际8进程，不会默默回退为单卡。

## Follow-ups

```bash
conda activate datascience
cd ~/VLA/FJSP-PPO-cloud
bash scripts/train_dataset_comparison.sh
```

启动后按终端打印的 `run_directory=...` 定位每次运行，最终使用各运行的 `checkpoints/best.pt` 评估测试集。动态阶段命令见 `docs/parameter-spec.md`。

---

Doc-ID: IMPL-2026-11-04-004
Doc-Type: implementation
Owner: user
Date: 2026-11-04
Version: v1.0
Status: final
Source: 进行MLP和GNN的训练
Activated-Skills: code-agent-core

## Summary

- 已于2026-11-04 10:56（Asia/Shanghai）启动并完成正式静态训练，MLP后自动运行GNN，各2000个全局episode；统一测试也已完成。
- 两模型均固定GPU0–7、global batch64、每10回合更新、静态学习率3e-4到5e-5、state23/action21、seed7。
- 后台监督进程PID为3805759，退出对话不会中断进程；任一阶段失败会写status并停止后续阶段。
- 两模型训练成功后自动用各自验证集best模型评估全部20个测试实例，保存测试指标和配对胜负汇总。

## Assumptions

- 本次执行一个配对训练seed7；动态微调配置不属于本次正式静态训练任务。
- 训练启动时记录了健康检查，最终结果已追加于本文；批次status.json为complete。

## Open Questions

None

## Plan Summary

1. 检查GPU空闲、配置公平性、依赖一致性和磁盘空间。
2. 固定本次配置与代码快照，启动持久后台监督进程依次执行两次8卡训练。
3. 检查实际采样进度、验证结果、Checkpoint写入和所有GPU参与状态。

## Changes

- 实验批次目录：`outputs/comparison_20261104_105537_5bbd5efc/`。
- 目录含固定配置副本、公平性报告、代码快照、pip依赖快照、launch_metadata.json、supervisor.pid、status.json、监督脚本及模型日志。
- MLP实际运行目录：`outputs/20261104_105638_mlp_seed7_4e2a13dc/`，training.log链接至本次MLP日志。
- GNN实际运行目录：`outputs/20261104_110027_gnn_attention_seed7_eefeada5/`。
- 自动测试输出为批次目录下的 `test_comparison.json`、`evaluation.log` 与 `summary.json`。

## Tests Run

- 公平性检查passed=true；pip check未发现依赖冲突；所有GPU启动前空闲，磁盘约124GB可用。
- 监督进程已脱离终端，PPID为1，持久运行；8张GPU均显示训练负载。
- 健康检查时MLP已完成370/2000回合；episode300验证目标为46.099，正常终止且未截断；latest与best均已保存。
- 学习率按计划下降，训练历史collection_rank覆盖0–7，全局episode数没有乘8。

## Risks and Rollback Notes

- 两模型均已完成训练，监督状态complete；没有遗留训练进程。
- 任务失败详情写入status.json和对应日志；成功完成后status会变为complete。

## Follow-ups

- 查看批次status.json了解当前模型和运行路径，查看对应training_history.json可获每10回合更新的精确进度。
- 训练与测试均完成，test_comparison.json和summary.json已生成；单seed结果尚不足以代表多seed稳定性。

---

Doc-ID: IMPL-2026-11-04-005
Doc-Type: implementation
Owner: user
Date: 2026-11-04
Version: v1.0
Status: final
Source: 进行MLP和GNN的正式训练及统一测试
Activated-Skills: code-agent-core

## Summary

- 两种模型均完成2000个全局episode、200次PPO更新；每次训练同时使用GPU0–7，global batch64。
- 统一使用验证集best Checkpoint，在20个测试实例上评估：MLP目标均值45.811000，GNN为41.526094，降低9.3534%。
- 逐实例比较：GNN胜17，MLP胜3，平局0；两模型20/20有效解、零调度违规。
- 本次为seed7单次配对结果，不代表多seed统计结论。

## Assumptions

- MLP/GNN完整方案比较，GNN额外使用图信息且参数量更大；相同episode与优化设置不代表相同计算成本。
- 静态state23中的动态4维、GNN节点13维中的动态2维按原schema保留；本次未执行正式动态微调。

## Open Questions

None

## Plan Summary

- 固定配置与代码快照，8卡依次训练MLP/GNN，按验证集选择best，最终只评估一次测试集。

## Changes

- 批次目录：`/root/VLA/FJSP-PPO-cloud/outputs/comparison_20261104_105537_5bbd5efc`。
- 已保存status、源码/依赖/配置快照、模型日志、测试逐实例结果、汇总和Checkpoint核验报告。
- 两模型运行目录分别保存training_curve.png与gantt.png，批次目录保存comparison_curves.png。

## Tests Run

- 启动前公平性检查与pip check通过；训练与测试监督状态为complete。
- 两模型均验证完整回合序列1..2000、200次更新、20次验证、8rank随机状态、global batch64、per-rank batch8、最终LR为5e-5。
- best按验证集目标选取，测试未参与训练/Checkpoint选择；best/latest/final与500/1000/1500/2000快照均存在。

### 指标

| 指标 | MLP | GNN |
| --- | ---: | ---: |
| 验证集best episode | 1900 | 1800 |
| best验证目标 | 44.750000 | 39.983056 |
| 测试目标均值 | 45.811000 | 41.526094 |
| 测试makespan均值 | 37.366000 | 36.683094 |
| 测试独立成本均值 | 16.890000 | 9.686000 |
| 有效解 / 测试实例 | 20 / 20 | 20 / 20 |
| 调度违规数 | 0 | 0 |
| 训练实际决策数 | 214034 | 244397 |
| 训练用时（分钟，含验证与启动） | 3.82 | 41.22 |
| 参数量 | 91,650 | 1,001,762 |

### 模型位置

- MLP best：`/root/VLA/FJSP-PPO-cloud/outputs/20261104_105638_mlp_seed7_4e2a13dc/checkpoints/best.pt`。
- GNN best：`/root/VLA/FJSP-PPO-cloud/outputs/20261104_110027_gnn_attention_seed7_eefeada5/checkpoints/best.pt`。

### 结果文件

- `test_comparison.json`：全部方法的汇总与逐实例结果。
- `summary.json`：MLP/GNN逐实例胜负与均值差。
- `checkpoint_audit.json`：训练预算、LR与best选择核验。
- `comparison_curves.png`：训练与验证曲线。

## Risks and Rollback Notes

- 单seed、20个合成测试实例，不能据此承诺真实生产场景或其他数据集泛化。
- 当前推理耗时字段不包含GNN图构造，未用它作端到端速度结论。

## Follow-ups

- 若需统计稳健结论，沿用相同协议增加配对训练seed17与27。

---

Doc-ID: IMPL-2026-11-04-006
Doc-Type: implementation
Owner: user
Date: 2026-11-04
Version: v1.0
Status: final
Source: 修改并准备好MLP和GNN的动态训练
Activated-Skills: code-agent-core, parameter-sanity

## Summary

- 两份动态配对配置分别绑定当前已重命名的正式静态best；原版MLP配置同步为同一8卡动态设置。
- 提供只检查、单模型训练、顺序训练两模型的入口，保存预检、来源摘要、配置快照和训练日志。
- 已完成实际Checkpoint与数据检查及62项回归测试，尚未启动正式动态训练。

## Assumptions

- 保留动态阶段2000全局episode、学习率1e-4到2e-5、熵系数0.01到0.003以及现有事件分布。
- 使用现存的`20261104_mlp_seed7_jingtai`和`20261104_gnn_attention_seed7_jingtai`目录。

## Open Questions

None

## Plan Summary

绑定真实Checkpoint，扩展预检以验证初始化来源，准备8卡入口，再完成数据与回归检查。

## Changes

- `configs/dataset_compare_{mlp,gnn}_dynamic.json`：固定来源best，输出到`outputs/dynamic_training/`的新目录。
- `configs/joint_training_dynamic_finetune.json`：同步MLP路径、8卡、每10回合更新及唯一目录设置。
- `scripts/check_comparison.py`：允许两模型采用不同来源路径，但要求两边都初始化；检查模型类型、维度、语义版本、profile、GNN结构和严格权重加载，记录SHA256和来源episode。
- `scripts/train_dynamic_comparison.sh`：默认check；支持mlp/gnn/both，both按MLP再GNN顺序各使用8卡，错误立即停止；支持PYTHON覆盖解释器。
- `tests/test_comparison.py`：补充正确配对、来源不匹配/缺失/权重错误、语义与profile错误、单边初始化的检查。
- `docs/parameter-spec.md`：更新可执行命令及参数约束。

## Tests Run

- `bash scripts/train_dynamic_comparison.sh check`通过：140个动态实例、两份真实源Checkpoint严格加载、训练设置配对、8卡可见性。
- `/root/miniconda3/envs/datascience/bin/python -m pytest -q`：62 passed in 13.07s。
- `bash -n scripts/train_dynamic_comparison.sh`与`git diff --check`通过。
- 原版MLP动态配置与配对MLP动态配置逐字段一致。

## Risks and Rollback Notes

- 当前绑定本机绝对路径；移动模型目录需同步initialize_from，预检会拒绝失效路径。
- 仅准备训练，尚无正式动态微调效果结论；静态模型权重未更新。

## Follow-ups

- 正式训练入口：`bash scripts/train_dynamic_comparison.sh both`；两模型各2000回合，验证集选best后再统一测试。

---

Doc-ID: IMPL-2026-11-04-007
Doc-Type: implementation
Owner: user
Date: 2026-11-04
Version: v1.0
Status: final
Source: 创建指定dataset2结果目录并启动GNN和MLP训练
Activated-Skills: code-agent-core, parameter-sanity

## Summary

- 创建指定`20261104_gnn_attention_jingtai_data2`与`20261104_mlp_jingtai_data2`结果目录。
- 统一两份dataset2配置，从头各训练20000回合，每模型使用8卡，先GNN后MLP。
- 后台监督进程记录两目录的status与配置快照，失败时停止后续训练。

## Assumptions

- 沿用之前推荐的batch64、每10回合更新、学习率3e-4到5e-5、熵0.02到0.005。
- 用户指定直接结果目录，因此output_dir_mode使用existing；监督入口拒绝覆盖已有checkpoint或训练日志。

## Open Questions

None

## Plan Summary

更新配置并配对检查，创建目录与后台监督入口，核对首个模型实际运行。

## Changes

- `configs/dataset2.json`、`configs/dataset2_gnn.json`：统一优化参数、8卡、保存间隔和关闭绘图，直接写入指定结果目录。
- `scripts/train_dataset2_comparison.py`：后台顺序执行GNN/MLP，保存启动状态、模型配置、预检和日志，正常完成后检查回合数与checkpoint。

## Tests Run

- 两份真实dataset2配置audit_comparison通过，训练参数无差异，各计划2000次更新。
- `pytest -q tests/test_comparison.py`：14 passed in 2.72s。
- 监督脚本py_compile与git diff --check通过。

## Risks and Rollback Notes

- 训练持续后台运行，dataset2大图训练耗时需以真实进度估算。
- 静态输入动态预留维度继续保留，未改变环境、模型结构或数据。

## Follow-ups

- 使用两目录`status.json`和`training.log`查看运行；checkpoint位于各目录`checkpoints/`。

---

Doc-ID: IMPL-2026-11-04-008
Doc-Type: implementation
Owner: user
Date: 2026-11-04
Version: v1.0
Status: final
Source: 修复此前审计发现的两处动态信息泄露，保持正在执行的dataset2静态训练
Activated-Skills: code-agent-core

## Summary

- 归一化尺度只读取已释放工件；释放和reset同步更新尺度及图边特征。
- 动态WAIT的候选存在性与时长不再查询隐藏事件队列；等待可由真实事件提前中断，等待必有正时长。
- 一并修复故障比例分母读取隐藏路线的问题，保留兼任辅助角色的生产机器故障支持。
- 静态输入、候选、图与最终排程的6条完整轨迹哈希和修复前完全一致；不重启或改动dataset2训练参数。
- 增加独立动态输入版本，避免静默混用旧动态checkpoint进行断点续训。

## Assumptions

- 全部机器清单和辅助工艺角色为公开信息；尚未释放的工件属性及未来事件为隐藏信息。
- 已发生故障的维修剩余时间沿用既有可观测设定。
- 动态WAIT允许主动等待，但无公开未来事件且仅剩必须补偿时不提供WAIT；不可决策时仍由内部事件推进。

## Open Questions

None

## Plan Summary

隔离公开与隐藏事件时间，修复尺度与机器比例依赖，验证可见历史一致时全部策略输入一致，检查静态兼容性并添加checkpoint协议标记。

## Changes (by file/module)

- `src/ppo_rs/environment/fjsp_env.py`：可见工件尺度刷新；公开WAIT期限与隐藏事件中断；故障比例使用已知机器集合；新增动态输入版本2。静态环境版本7和全部输入维度保持不变。
- `src/ppo_rs/environment/checkpoint_semantics.py`：旧动态输入协议的权重初始化／评价发出提示，断点续训拒绝混用；静态使用不受影响。
- `src/ppo_rs/train.py`：保存动态输入版本；在动态初始化和续训时检查，允许静态权重迁移并重置优化器。
- `src/ppo_rs/evaluate.py`、`scripts/evaluate_dataset.py`、`scripts/compare_observation_runs.py`：动态评价提示协议变化；数据集评价及对比摘要同时记录来源版本与评价版本。
- `tests/test_dynamic_environment.py`：释放测试适配公开WAIT期限。
- `tests/test_dynamic_input_visibility.py`：覆盖三种state和两种action profile下隐藏工件参数／数量／路线变化，事件时间／目标／维修长度／有无变化，运行、放置、维修状态，尺度释放/reset、等待推进与补偿、四种事件场景，CPU双模型训练／续训／旧权重迁移。
- `data/dataset_dynamic/README.md`：记录公开WAIT与动态输入协议。
- `docs/project-improvement-review.md`：标记相关问题已修复，保留原审计记录。
- `docs/dynamic-input-fix-validation.json`：保存静态轨迹前后哈希及固定动态场景验证数量。

## Tests Run

- 完整pytest回归：94项通过（CPU，OMP/MKL各2线程）。
- `scripts/validate_dynamic_dataset.py`通过：100 train、20 validation、20 test。
- 固定动态validation+test共40实例，4种规则策略共160次排程全部正常结束、无截断且无排程违规。
- 真实data1正式MLP/GNN静态best分别在动态实例上完成CPU推理，无违规；此项为兼容性检查，不代表新正式动态实验成绩。
- 两种模型分别完成CPU微型动态训练、新版checkpoint续训、拒绝旧版动态断点续训，以及旧权重初始化新运行。
- 静态examples、dataset1、dataset2各1实例，compact21/enhanced25两种动作profile共6条完整轨迹：全部状态、候选动作及次序、动作特征、图数组和最终排程SHA256与修复前完全一致。
- `git diff --check`通过；dataset2监督进程、GNN训练进程仍运行，MLP仍排队。

## Risks and Rollback Notes

- 修复前动态成绩包含未来信息影响，不能直接用于无泄露正式对比；需要在新协议下重新微调并统一评价。旧checkpoint文件未覆盖。
- 动态WAIT候选规则发生必要改变，历史动态策略可以加载，但可能需要重新学习；新旧动态回合不可混入同一续训记录。
- 静态轨迹验证覆盖3种规模的抽样实例，不构成所有实例的形式化证明；实现保留独立静态分支，完整回归也通过。
- 当前dataset2的运行中GNN继续使用启动时载入代码；后续MLP会载入修复后代码，静态输入／行为等价已经抽样全轨迹验证。
- 本次未修改配置、数据文件或模型架构，未启动新的正式动态训练。

## Follow-ups

- 待需要正式动态对比时，使用静态MLP/GNN来源权重创建新的动态运行，并按相同修复协议重新评价。
- 先前审计中的截断奖励／bootstrap属于独立问题，本次没有扩展修改。

---

Doc-ID: IMPL-2026-11-05-001
Doc-Type: implementation
Owner: user
Date: 2026-11-05
Version: v1.0
Status: final
Source: 旧dataset1静态和动态归dataset_test，扩充dataset1并补充dataset2动态；用户最终确认容量1000／100／200
Activated-Skills: code-agent-core, parameter-sanity

## Summary

- 统一data/dataset_test、data/dataset1、data/dataset2下的static/dynamic目录，每个模式各自含train、validation、test。
- 旧小集静态／动态各100／20／20原样保存；dataset1、dataset2静态／动态各1000／100／200。
- dataset1保留旧成员并按split追加；dataset2现有静态文件字节不变，补齐配对动态1300实例。
- 保留旧路径兼容链接，配置改为规范路径；正在运行的dataset2 MLP继续正常运行。
- 完整pytest100项和276次动态规则排程通过，生成器和迁移记录可复查。

## Assumptions

- 最终容量以用户明确回复“统一为1000训练＋100验证＋200测试”为准。
- 保留同等物理／时间／成本分布及每类1～2次事件，不增加动态扰动强度。
- dataset_test是历史小数据集名称，不是新增独立holdout；其test始终留在新版dataset1/test。

## Open Questions

None

## Plan Summary

先归档原数据并记录全部SHA256，在临时目录扩容与生成，校验静态配对、split互斥、事件合法性后发布新布局，保留原路径兼容入口，再验证当前训练和全部回归。

## Changes (by file/module)

- `data/dataset_test/{static,dynamic}`：完整旧小集。原README、manifest和全部实例字节保持一致。
- `data/dataset1/{static,dynamic}`：各1300实例；每split保留旧文件，分别追加900训练、80验证、180测试。静态扩容使用20261105与独立split偏移，动态保留原轨迹并补齐65／20／10／5%固定场景。
- `data/dataset2/static`：迁入原1300静态实例及元数据；`dynamic`新增1300配对实例，故障池与reference_horizon按实际生产机数生成。
- `data/dataset`、`data/dataset_dynamic`及dataset2旧split/manifest：相对兼容链接，历史配置／checkpoint仍定位相同数据。
- `data/dataset-layout-migration.json`：迁移前全部旧文件哈希和兼容路径；`dataset-layout-validation.json`：容量、范围、配对及固定事件校验摘要。
- `scripts/organize_datasets.py`：临时生成、校验、迁移发布及错误回滚；支持`--check`复验，拒绝重复覆盖已整理目录。
- `scripts/generate_dataset.py`：扩容时保留原split和字节，独立split RNG避免train容量改变导致eval样本变化；拒绝覆盖非空目标。
- `scripts/generate_dataset2.py`：支持独立目标路径与seed，默认指向static并拒绝覆盖；原seed可精确复现既有1300静态实例与manifest。
- `scripts/generate_dynamic_dataset.py`：容量读取源manifest，支持旧动态保留、固定场景差额补足，以及不同机器规模。
- `scripts/validate_dynamic_dataset.py`：参数化目录，检查文件与manifest、基础实例配对、内容去重、事件隐藏／合法／固定轨迹／随机重采样。
- `scripts/validate_dataset2.py`：支持规范static路径及CLI覆盖；`make_zero_cost_dataset.py`明确继续派生旧小集。
- 18份配置：更新规范数据路径，修复3份动态配置的失效初始化路径；所有训练参数和输出目录保持不变。现有checkpoint仍来自旧数据训练，本次未重新训练。
- `tests/test_dataset_layout.py`、`test_dynamic_generation.py`：迁移哈希／配对／划分／生成稳定性、可变规模动态生成及错误拒绝测试；旧数据回归改指dataset_test，对比预检容量改为1000。
- `data/README.md`及各组README：当前布局、容量、兼容、原数据关系与复现命令；根README增加入口，保留其原编码字节。

## Tests Run

- `python -m pytest -q`：100 passed in 31.33s。
- 整理脚本及`--check`：六个子集文件数与manifest完全一致；全部静态／动态基础内容配对；每套去名称／事件后无跨split或split内重复。
- 迁移前SHA256核对：旧140静态、140动态及dataset2的1300静态实例和原元数据完全一致。
- 生成器复现：dataset1静态、dataset1动态、dataset2动态的全部实例字节可复现；dataset2原seed复现1300实例和原manifest完全一致。
- `validate_dataset2.py`通过，包括完整范围检查与规则排程抽样。
- dataset1静态、动态两对真实配置的`check_comparison.py`预检通过；初始化checkpoint可严格加载。
- 新动态数据按validation/test场景与生产机器数抽样，dataset1固定32次＋随机48次、dataset2固定148次＋随机48次，共276次规则排程均正常完成且无违规。
- `git diff --check`通过；MLP进程持续运行，整理后已推进至3290回合以上，旧训练数据路径兼容。

## Risks and Rollback Notes

- 文件重定位会让Git工作区显示大量删除与未跟踪新增；旧文件实际保存在规范目录，未执行git add、commit或push。
- 兼容使用相对symlink，在当前Linux环境有效；迁移到不支持symlink的平台应使用规范路径。
- dataset_test与新版dataset1有重合，不能当独立泛化集；静态与动态配对也不能跨split混用。
- 相同事件数量在更大实例上相对扰动更低；本次保留协议，后续强度实验应另设受控条件。
- 配置路径更新仅作用于后续启动；已运行MLP与既有outputs配置快照未改写。历史报告的旧容量和旧路径保留为实验记录。

## Follow-ups

- 后续按data/README.md的规范路径训练／评价；旧训练结果不自动变成扩容数据集上的重新训练结果。

---

Doc-ID: IMPL-2026-11-05-002
Doc-Type: implementation
Owner: user
Date: 2026-11-05
Version: v1.0
Status: final
Source: 确认并删除data/dataset及data/dataset_dynamic兼容入口
Activated-Skills: code-agent-core

## Summary

- 删除两个旧小集相对symlink，仅unlink链接，真实dataset_test数据不变。
- 目录校验允许旧链接缺失；从旧布局迁移时不再创建这两个链接。
- 当前MLP使用dataset2，保留其兼容链接，训练持续正常。

## Assumptions

- 新配置已全部使用dataset1或dataset_test的规范路径。

## Open Questions

None

## Plan Summary

核对链接目标、活动配置与训练来源后删除链接，更新校验及说明，再验证真实数据哈希与历史数据回归。

## Changes (by file/module)

- `data/dataset`、`data/dataset_dynamic`：移除链接，保留其目标dataset_test/static及dynamic。
- `scripts/organize_datasets.py`：旧小集链接变为可选，不再在迁移时创建。
- `data/README.md`：更新当前路径和历史checkpoint／命令的显式路径替换说明。
- `data/dataset-layout-migration.json`：追加链接删除记录，原始迁移哈希保留。

## Tests Run

- `pytest -q tests/test_dataset_layout.py tests/test_dataset.py tests/test_dynamic_dataset.py`：8 passed in 7.27s。
- 删除前后dataset_test全部文件SHA256完全一致，静态／动态各100／20／20。
- `git diff --check`通过；MLP launcher及rank进程仍正常运行。

## Risks and Rollback Notes

- 历史命令若仍使用两个旧路径，应显式改用dataset_test/static或dynamic。
- 如需恢复，可创建相同相对symlink；数据文件无需恢复。

## Follow-ups

None

---

Doc-ID: IMPL-2026-11-05-003
Doc-Type: implementation
Owner: user
Date: 2026-11-05
Version: v1.0
Status: final
Source: 为dataset1、dataset2生成cost0版本，去应力及补偿费用归零、时间占用不变
Activated-Skills: code-agent-core, parameter-sanity

## Summary

- 新增dataset1_cost0和dataset2_cost0，各含static/dynamic，四个子集均为1000／100／200，共5200实例。
- 两额外费用置0.0，全部时长、设备、物理参数、目标权重与动态事件保持不变。
- 保留实例名称与对应文件名，manifest记录来源、源哈希及变更字段。
- 未修改原始数据或当前训练，未启动新的训练。

## Assumptions

- 本轮开始时旧dataset1_cost0目录已不存在，因此无需覆盖或归档旧数据。
- cost0同时覆盖静态及动态模式，沿用此前统一目录结构。

## Open Questions

None

## Plan Summary

核对目标目录和旧配置，参数化派生脚本，以临时目录生成并验证，再发布两个family，校验动态协议、源数据只读以及物理时间一致性。

## Changes (by file/module)

- `scripts/make_zero_cost_dataset.py`：新增family级generate_dataset和validate_dataset，默认生成两组；CLI支持--source-root、--target-root、--check。严格检查manifest文件数和静态／动态配对，拒绝覆盖非空目标，生成期间源文件哈希不变。
- `data/dataset1_cost0`、`data/dataset2_cost0`：四个派生子集，仅实例中的stress_relief.cost与compensation.cost改变。manifest去除对旧有成本字节的保留断言，记录派生源哈希，动态base_dataset指向cost0/static；费用范围元数据同步为[0,0]。
- `configs/dataset1_cost0.json`：数据路径改为新版static目录，原训练参数及输出目录不变。
- `tests/test_zero_cost_dataset.py`：6项覆盖只改费用、源只读、拒绝覆盖／不完整源、篡改时长和事件拒绝；静态／动态执行包含去应力和补偿的相同动作序列，比较时钟和排程。
- `data/README.md`及两组README：更新容量、费用语义及验证／复现命令。`docs/parameter-spec.md`记录参数约束。

## Tests Run

- 完整pytest：106 passed in 31.19s；后续增加源manifest检查及修正费用范围元数据后，6项相关测试再次通过。
- 全部5200实例逐字段对照，除两项费用外与来源相同；source manifest及实例SHA256一致。
- 两组cost0/dynamic通过完整动态验证，保留固定场景配额、原事件轨迹及随机采样配置。
- 真实cost0数据按两规模、静态及四动态场景抽样共10次规则排程，均正常完成、无违规、额外费用为0且objective=makespan。
- `git diff --check`通过，MLP训练进程持续正常。

## Risks and Rollback Notes

- 成本特征和奖励因费用归零发生预期变化；即使物理过程不变，重新训练的策略仍可能选择不同动作，这是成本消融的目的。
- 保留实例名便于配对，不应将cost0与原始实例不加区分合并为独立测试样本。
- 当前等待额外费用为0，因此正常结束目标等于makespan；故障、去应力、补偿仍可通过时长和资源占用影响目标。

## Follow-ups

- 需要训练时显式选择cost0数据路径；本轮仅生成并验证数据。

---

Doc-ID: IMPL-2026-11-05-004
Doc-Type: implementation
Owner: user
Date: 2026-11-05
Version: v1.0
Status: final
Source: 生成dataset3和dataset3_cost0，工件18～30、每件工序6～13、设备12～18，其余设置与dataset1／2一致
Activated-Skills: code-agent-core, parameter-sanity

## Summary

- dataset3和dataset3_cost0均生成static/dynamic，每模式1000训练、100验证、200测试，共5200实例。
- 每实例18～30个工件，每工件6～13道普通工序，总设备12～18台，含M3／M4各1台。
- 复用dataset2工艺和成本分布、通用动态事件生成器和cost0派生器；动态概率和原参数保持一致。
- 辅助允许位置及次数随最大13工序扩到12个工序间隙；未改变训练配置或启动新训练。

## Assumptions

- 沿用前文设备口径：12～18指总设备数，普通设备为10～16台。
- “其余一致”包括1000／100／200容量、每工序2～5候选设备、物理／时间／费用分布、目标权重、事件概率和环境限制。

## Open Questions

None

## Plan Summary

扩展共享实例生成器的显式可选规模参数，保持dataset2默认行为不变；生成dataset3静态，再生成配对动态和cost0，验证全量范围、事件、配对和物理排程。

## Changes (by file/module)

- `scripts/generate_dataset2.py`：_instance新增可选工序范围、机器池、名称前缀参数，默认值与旧行为一致；辅助次数及允许位置随指定最大工序数生成。
- `scripts/generate_dataset3.py`：固定种子20261106，独立split随机种子；工件数和设备数边际均衡，91种工件／设备组合可覆盖；目标非空时拒绝覆盖，临时生成后发布。
- `scripts/validate_dataset3.py`：容量、完整manifest、去名称内容查重、规模分布、工艺与目标范围、辅助允许位置，以及动态配对／事件与可选cost0逐字段检查。
- `data/dataset3/static`：1300实例，工件18～30、工序6～13、普通设备10～16，总设备12～18；去应力和补偿最多12次，允许O1～O12后执行。
- `data/dataset3/dynamic`：基于同名静态实例，seed20261107；train每reset采样，validation场景65／20／10／5，test130／40／20／10。
- `data/dataset3_cost0/{static,dynamic}`：2600派生实例，仅两项辅助费用为0，时长、事件和物理配置不变。
- `tests/test_dataset3.py`：4项覆盖可复现／拒绝覆盖、split稳定性、规模及原参数一致性、dataset2默认实例兼容。
- `data/README.md`、dataset3 README、`docs/parameter-spec.md`：记录总设备口径、容量、规则与复现／校验命令；dataset3/validation.json保存全量校验结果。

## Tests Run

- 完整pytest：110 passed in 32.67s。
- `validate_dataset3.py --cost0-root data/dataset3_cost0`通过：全部5200实例容量、范围、配对、查重、动态规则和cost0字段差异正确。
- 独立重新生成dataset2的1300个原实例，与现存静态文件内容全部一致；共享生成器默认行为未改变。
- 大规模静态／动态按场景及总设备数量分层，另选工序总量最大实例，共67实例、268次规则排程均正常完成且无违规；另18次有成本／cost0配对动作序列验证通过。结果见data/dataset3/scheduling-smoke.json。
- cost0与有成本版本相同动作序列的时钟和排程（除费用）一致；cost0正常完成目标等于makespan。
- `git diff --check`通过；当前dataset2 MLP进程继续正常运行。

## Risks and Rollback Notes

- 12～18台设备包括辅助设备；普通加工设备是10～16，文档与manifest已显式记录。
- 动态每类事件仍1～2次，规模增大后相对扰动更低；并未静默提高事件强度。
- 固定2～5台候选意味着随总机器增多，相对柔性比例下降，沿用dataset2现规则；跨规模结果包含该分布变化。
- cost0与有成本数据共享基础实例，静态与动态也配对，不得跨版本混入train／test。
- 新增数据、脚本及文档保持未提交状态；没有改写现有数据实例和训练参数。

## Follow-ups

- 如需训练，显式使用data/dataset3或dataset3_cost0下的static/dynamic split；本轮未启动训练。

---

Doc-ID: IMPL-2026-11-05-005
Doc-Type: implementation
Owner: user
Date: 2026-11-05
Version: v1.0
Status: final
Source: 确认data是否被Git忽略；统计训练和验证中完成一次完整加工过程的实际耗时
Activated-Skills: code-agent-core

## Summary

- data未被.gitignore忽略；旧位置已有1730个受跟踪路径，新数据多数尚未跟踪，本轮未暂存或提交。
- 训练采样、模型验证及规则评估新增完整回合实际耗时，并分别记录策略决策和GNN构图耗时。
- PPO更新独立计时，多卡取各rank最大值；完整验证集另记录整轮耗时。
- 保存训练／验证历史及评估汇总，兼容旧checkpoint和缺少计时字段的旧日志。
- 当前运行中的dataset2 MLP保持运行，新增统计从后续启动／恢复的训练及新评估开始生效。

## Assumptions

- “完成一次加工过程需要多久”指程序生成完整排程的实际计算秒数；车间模拟完工时间仍使用makespan。
- 一个完整加工回合对应一个数据实例内全部工件的调度过程，而非单道工序。

## Open Questions

None

## Plan Summary

核对Git忽略规则，统一回合计时范围，接入训练／验证日志及评估汇总，再验证时钟范围、静动态结果、多进程和旧日志续训。

## Changes (by file/module)

- `src/ppo_rs/timing.py`：统一EpisodeTimer，使用perf_counter，CUDA回合边界同步；输出episode_wall_seconds、policy_inference_seconds、graph_construction_seconds。
- `algorithms/ppo.py`、`evaluation/runner.py`、`baselines/policies.py`：覆盖reset、状态处理、动作选择、环境推进、最终指标／约束检查；训练还含轨迹buffer写入。回合计时排除环境对象构造及结束后的排程提取。
- `src/ppo_rs/train.py`：training_history逐回合保存三项耗时及terminated／truncated；每个rollout块仅最后一行保存ppo_update_wall_seconds。多卡更新耗时以MAX归约。validation_history保存每实例平均耗时和validation_wall_seconds，后者包含环境构造、排程提取和聚合；控制台显示耗时。
- `scripts/evaluate_dataset.py`、`scripts/compare_observation_runs.py`：汇总逐实例平均及总回合耗时，历史数据缺字段时不补造计时。
- `tests/test_timing.py`、`tests/test_timing_reports.py`、`tests/test_evaluate_dataset_profiles.py`、`tests/test_distributed_training.py`：确定性模拟时钟、实际MLP／GNN、静态／动态、规则、CUDA边界、多进程日志保存、历史缺字段续训及CLI汇总检查。
- `docs/timing.md`：说明字段、范围、保存位置、正常完工筛选及比较方法。未修改训练超参数、动作／奖励或现有训练进程。

## Tests Run

- 完整pytest：130 passed in 33.44s，使用datascience Python，OMP_NUM_THREADS=2、MKL_NUM_THREADS=2。
- 模拟时钟验证reset、构图、决策、环境推进、buffer和最终指标均计入回合时长；验证排程提取在回合计时外、在整轮验证计时内。
- CPU真实MLP／GNN在静态及动态实例中，训练采样和确定性验证排程与目标一致、无约束违规。
- 两rank CPU短训练及续训通过：计时保存到JSON、更新时长各rank一致；移除旧行计时字段后仍可续训，历史不变，新行具备计时。
- 三种输入profile的评估CLI通过，模型和规则方法的逐实例、平均与总耗时字段均保存。
- git diff --check通过；当前dataset2 MLP launcher和8个rank仍正常运行。

## Risks and Rollback Notes

- 截断回合也记录耗时，统计成功完工应筛选terminated=true、truncated=false并核对约束违规。
- policy_inference_seconds包含act内部输入转换／GNN批处理及Actor／Critic计算；GNN环境构图另计，完整回合总耗时包含两者。
- 首次运行／预热、硬件、CPU线程和共同运行负载都会影响实际耗时，本轮保留实际观测值，不做扣除。
- 多卡并行采样的回合耗时之和不代表训练经过的总时间；PPO更新耗时另计，外围通信、保存和绘图不计入回合。
- 旧历史无法追溯补算计时，当前运行Python不会热更新。本轮未重启或恢复现有训练。
- 回滚只需移除本轮计时接入与文档；新JSON字段为附加指标，checkpoint格式未要求迁移。

## Follow-ups

None

---

Doc-ID: IMPL-2026-11-05-006
Doc-Type: implementation
Owner: user
Date: 2026-11-05
Version: v1.0
Status: final
Source: 缩短每组试验的训练时间，例如本次GNN和MLP训练
Activated-Skills: code-agent-core

## Summary

- 实现保持现有网络、数据、全局batch64、rollout10、PPO4轮和20k预算的单组训练优化。
- 验证按实例分配到所有rank，保留seed与原顺序，统一汇集指标、排程和异常。
- PPO日志指标每rollout归约一次，候选动作和图批量传输；GNN使用CPU预生成的类型索引减少GPU同步。
- 新增CPU／torchrun多卡短基准入口，独立输出并记录速度、耗时和硬件信息。
- 完整141项测试及基准脚本CPU单／双进程冒烟通过；当前MLP未被中断、未宣称8卡整体加速倍数。

## Assumptions

- 本轮优先减少工程开销，保持原优化参数和数据协议，不通过减少训练预算获取速度收益。
- 当前运行MLP使用启动时代码，自动热更新不成立；新代码用于后续启动／同world size恢复的运行。

## Open Questions

- 8卡全部空闲时的优化前后整体训练加速比尚待基准测量；采样和完整轨迹通信仍可能限制收益。

## Plan Summary

先并行实现多卡验证、PPO数据准备和日志通信、GNN索引前向，再对照旧前向／梯度和多进程行为；补充可重复短基准和使用说明，保持当前训练运行。

## Changes (by file/module)

- `src/ppo_rs/train.py`：验证全rank按全局索引分片，恢复原顺序，eval_seed按原索引计算；支持空分片，错误随结果一次汇集；验证墙钟取rank最大值，所有rank参与调用。
- `algorithms/ppo.py`：mini-batch的梯度同步不变，仅将日志指标延后至rollout结束归约／读取；GNN图索引每epoch复制一次CPU，保留原GPU随机排列、填充和分片语义。
- `algorithms/rollout_buffer.py`：CPU一次填充可变候选数组／mask，再整块传入设备，保持原dtype、GAE和内容。
- `models/graph_batch.py`：先CPU拼接、按字段传输；CPU按类型分组生成节点／边索引，打包后搬到设备，避免逐图复制和前向逐类型GPU布尔判断。
- `models/graph_actor_critic.py`、`graph_attention.py`：按索引执行，缺类型仍跳过模块；pool计数利用索引长度，不再读取mask.sum().item。兼容无预生成索引的手工GraphBatch。
- `models/actor_critic.py`及GNN act：三个标量合并一次CPU读取，保持返回类型和网络参数结构。
- `scripts/benchmark_training_speed.py`：固定更新块数和验证前缀的独立诊断入口；保留batch／rollout／架构，起始学习率和熵系数固定，支持checkpoint权重初始化；报告墙钟、steps/s、更新／验证耗时和GPU小时。
- `tests/test_parallel_validation.py`、`test_ppo_performance.py`、`test_graph_performance.py`：验证串行与多进程一致、异常与空分片、缺关系梯度、原前向对照、buffer内容和归约次数。
- `docs/training-speed.md`、`docs/timing.md`：记录使用方式、并行计时语义及现有限制。

## Tests Run

- 完整pytest：141 passed in 38.10s，OMP_NUM_THREADS=2、MKL_NUM_THREADS=2，CPU执行。
- 对照旧GNN前向及反向：完整／缺关系／缺节点类型／空边均一致，state_dict键及checkpoint_kwargs相同，缺参数grad=None行为保持。
- 现有MLP／GNN双rank PPO与单进程全局batch等价测试通过，含不均匀batch和填充；日志仅一次显式all_reduce。
- 并行验证与串行结果对照通过，含MLP／GNN、静态／释放／故障、空rank、seed与RNG、异常传播；双rank训练／旧日志续训测试通过。
- benchmark脚本CPU单rank和双rank各1更新块、2回合、1验证实例正常结束，报告落在outputs/speed_smoke_checks；验证正常完成、无截断。该极小样本只检验入口，不用于性能结论。
- git diff --check通过；当前8卡MLP launcher及所有rank持续运行。

## Risks and Rollback Notes

- 未实测空闲8卡整体加速比，不承诺几倍或将21小时缩至某固定时长；局部通信减少不等于整体同比加速。
- GPU动作读取和合法动作检查等仍可同步；本轮减少同步，未消除全部同步。
- 使用不同索引原语或汇总顺序可能引入浮点末位差异；梯度和更新协议保持一致，不承诺训练轨迹逐位重现。
- 保留find_unused_parameters=True，合法图可缺少关系，不能仅凭某次DDP警告关闭。
- 当前采样仍单环境逐步运行，完整Python图轨迹仍all_gather_object复制；未自动扩大rollout、增大batch或引入异步策略。
- checkpoint兼容但恢复仍要求原world size；本轮未重启当前MLP或改写历史训练结果。
- 可逐模块恢复旧实现；训练超参数／数据／checkpoint无需迁移。

## Follow-ups

8卡空闲后使用独立短基准比较实际耗时，再决定是否实施多环境批量采样和轨迹通信重构。

---

Doc-ID: IMPL-2026-11-05-007
Doc-Type: implementation
Owner: user
Date: 2026-11-05
Version: v1.0
Status: final
Source: 额外补齐基础state训练代码、预留动态接口并检查完整组代码
Activated-Skills: code-agent-core, parameter-sanity

## Summary

- 新增compact18全局profile，含基础14项＋共享动态4项；局部compact21，GNN节点13维，两阶段尺寸相同。
- 新增基础／完整×MLP／GNN×静态／动态8份配置，非架构训练参数统一。
- 新完整组采用unified23／enhanced25，即有效19全局＋4动态预留／25局部，静态到动态可直接载入各自权重。
- 新启动器支持检查、数据规模／cost0选择、单模型或双模型、seed、卡数及安全独立输出；动态必须解析到静态来源。
- 完整166项测试及实际dataset2两组静态预检通过；未启动正式训练或中断当前MLP。

## Assumptions

- 默认dataset2静态有成本，训练20k回合；动态独立10k回合。两组均预留相同事件状态，静态预留值为0。
- “基础14／21、完整19／25”指静态有效信息，实际输入18／21与23／25用于保持静态到动态的结构一致。

## Open Questions

None

## Plan Summary

核对已有profile和历史配置，补基础动态保留profile及统一配置，以新入口安全绑定静态checkpoint，再验证可观测性、配对参数、完整静态到动态训练链路和评估兼容。

## Changes (by file/module)

- `src/ppo_rs/environment/fjsp_env.py`：新增compact18，基础直接拼接共享_dynamic_observation，跳过5项全局增强计算；旧compact14／enhanced19／unified23输出不变，图节点compact18为13维。
- `configs/state_ablation/`：8份新配置，base_mlp_static等；base compact18／compact21，full unified23／enhanced25。默认batch64、rollout10、PPO4轮、8卡、unique输出，统一验证500、latest100、snapshot5000。
- `scripts/train_state_comparison.py`：--group、--phase、--model、--dataset、--seed、--world-size、--device、--check、--output-root；动态显式--initialize-from／--static-run或唯一匹配同数据／组／模型／seed的静态best。单模型不要求另一来源，多个来源拒绝猜测；先CPU预检再训练。
- `scripts/check_comparison.py`：按profile实际字段统计动态预留维数，兼容compact18。
- 新tests：base_state_profile、state_ablation_configs、state_training、state_launcher；扩展动态可观测性和评估CLI测试以覆盖新profile和完整组。
- `docs/state-ablation.md`、`models/README.md`及参数说明：启动命令、字段语义、checkpoint限制和旧完整组审计。

## Tests Run

- 完整pytest：166 passed in 40.28s；CPU，OMP_NUM_THREADS=2、MKL_NUM_THREADS=2。
- 实际dataset2 base/full静态--check均通过，数据1000／100／200，组内MLP／GNN训练设置无差异；base18／21，full23／25，动态全局预留4、GNN节点预留2。
- CPU四个组／模型组合静态训练、best保存、回载、故障动态评估、initialize_from微调均通过，正常完成且无违规，微调计数和optimizer重新开始。
- compact18与compact14基础14项、unified23动态4项完全一致，静动态物理排程保持；隐藏工件／未来事件变化不改变可见输入。
- launcher检查模式无CUDA检查和训练进程，支持唯一来源发现、缺来源／多来源拒绝、错误profile拒绝和单模型独立预检。
- 旧dataset1完整配置审计确认batch64／32、rollout20／10、latest20／10不一致；旧配置保持历史，不作为新配对入口。
- git diff --check及脚本语法检查通过，当前训练进程未被停止。

## Risks and Rollback Notes

- 未采用固定尺寸mask等容量消融，基础／完整输入参数量不同，文档列明；两组GNN保持相同图通道，能恢复部分被删摘要。
- 旧dataset2为23／21，不能直接载入完整23／25；旧19／25和节点11维同样不兼容新动态预留输入，不做隐式扩展权重。
- 动态JSON模板没有伪造initialize_from；必须经新launcher绑定来源，直接交给公共train且不加初始化参数会从头训练动态策略。
- 显式外部来源只校验模型schema／权重，不保证训练数据／成本条件／seed来源匹配；正式实验应检查来源并等待静态结束。
- 当前正在运行训练不会热加载新代码，未覆盖现有数据、配置快照、checkpoint和输出。
- 可移除新templates／launcher并删除compact18分支回滚；旧profile和checkpoint无需迁移。

## Follow-ups

用户可使用--check核对后按文档启动新组；本轮未自动启动。

---

Doc-ID: IMPL-2026-11-05-008
Doc-Type: implementation
Owner: user
Date: 2026-11-05
Version: v1.0
Status: final
Source: 启动dataset1零额外成本、完整state、静态MLP/GNN训练
Activated-Skills: code-agent-core, parameter-sanity

## Summary

- 创建指定的`outputs/fullstate_jingtai_set1/MLP`与`GNN`，保存独立配置、日志、历史和checkpoint。
- 核心PPO参数与上一组一致；本次完整state为23/25，上次为23/21；数据改为dataset1_cost0/static。
- 8卡按MLP、GNN顺序运行，MLP已开始采样更新，GNN排队自动接续。

## Assumptions

- 去应力与补偿只取消额外成本，仍保留时长；两组均从头训练。

## Open Questions

None

## Plan Summary

核对旧快照、完整组模板及零成本数据，精确分配新输出目录，通过定向检查后后台启动监督脚本。

## Changes (by file/module)

- 输出目录内`MLP/training_config.json`与`GNN/training_config.json`：从完整静态模板派生，不改通用训练算法及原模板。
- `preflight.json`、`parameter_comparison.json`、`data_audit.json`：配对配置、上次参数及全部实例成本／时长检查。
- `run_training.py`、`status.json`、`launch_metadata.json`、`supervisor.log`：依次8卡训练，拒绝已有训练产物，进程退出检查和信号停止，逐组保留日志。
- 记录Git提交和训练源码SHA256，避免后续代码变化混淆实际运行版本。
- `docs/parameter-spec.md`：PARAM-2026-11-05-009记录本轮精确参数和差异。

## Tests Run

- 实际两组CPU预检通过，非架构配置差异为空，全局23／局部25，GNN节点13。
- 1300个cost0静态实例与有成本源逐字段核对通过，1000／100／200，无动态事件，仅两个成本字段改变。
- 定向pytest：state_training、parallel_validation、distributed_training、zero_cost_dataset共19项通过，9.21秒。
- 监督脚本语法检查及模型目录检查通过；正式8rank MLP已写出训练历史、正常完成标志及完整计时字段。

## Risks and Rollback Notes

- 本次改变数据规模、辅助成本和局部输入，不能把与上次的解质量差异单独归因于某一项。
- 使用加速后的代码，训练速度不沿用上次未优化版本的耗时结论；GPU端到端加速比尚无单独基准。
- 指定模型目录直接保存checkpoint，训练期间不要重命名输出目录。
- 输出均受.gitignore忽略，不自动暂存、提交或推送本轮文档改动。

## Follow-ups

MLP完成后自动运行GNN；后续可按需在固定200测试实例上评估最佳模型。

---

Doc-ID: IMPL-2026-11-05-009
Doc-Type: implementation
Owner: user
Date: 2026-11-05
Version: v1.0
Status: final
Source: 核查参数并在fullstate_dongtai_set1启动dataset1完整state MLP/GNN动态微调
Activated-Skills: code-agent-core, parameter-sanity

## Summary

- 本次输出目录为outputs/fullstate_dongtai_set1，MLP与GNN分别保留配置、日志、历史与checkpoint。
- 由对应静态best初始化，保持cost0与完整23／25输入；各动态微调10,000回合，低学习率和熵系数使用现有动态模板。
- 8卡后台监督进程依次运行MLP与GNN，模型启动前检查配置和来源checkpoint哈希。

## Assumptions

- 用户已授权参数检查后启动两组训练，沿用静态阶段GPU0～7及MLP→GNN顺序。
- 不添加Dropout、不重新划分数据、不继承静态Adam和历史，不自动复用上次人工早停规则。

## Open Questions

None

## Plan Summary

核对动态模板及静态来源，验证cost0／动态数据和权重运行兼容性，创建精确新目录，完成定向测试后启动持久监督程序。

## Changes (by file/module)

- 本次目录的MLP/training_config.json与GNN/training_config.json：从完整动态模板派生，绑定对应静态best，更新数据和输出绝对路径。
- preflight.json、data_audit.json、parameter_comparison.json：记录配对优化参数、数据校验及静动态差异。
- initialization_sources.json、initialization_smoke.json：来源权重／配置／SHA256，以及两模型四场景推理检查。
- run_training.py：复用静态顺序监督实现，增加启动前配置与来源权重哈希验证；保留状态、退出检查和信号停止逻辑。
- status.json、launch_metadata.json、supervisor.log：保存状态与监督进程信息，启动时间为北京时间2026-11-05 18:13:56。
- docs/parameter-spec.md：追加PARAM-2026-11-05-010；本报告只追加新条目，保留既有文档改动。

## Tests Run

- 真实配置两模型CPU预检通过：非架构优化设置无差异、严格加载静态best，全局23／动作25，GNN节点13。
- dataset1静态与动态cost0逐字段来源检查通过；动态1300实例的配对、数据分离、隐藏未来信息、事件采样与固定轨迹校验通过。
- 实际静态MLP／GNN各运行无事件、插单、故障、混合验证场景，共8个完整回合，零违规、额外成本0。
- pytest：test_state_training、test_parallel_validation、test_distributed_training、test_zero_cost_dataset，共19项通过，9.69秒。
- 监督脚本AST语法检查、新目录防覆盖、8GPU空闲及127GB磁盘余量检查通过。

## Risks and Rollback Notes

- 本次两模型静态来源的训练回合预算不同，结果应报告完整预训练来源；不能只凭动态预算相同声称全链路等预算。
- 动态集合65%无事件，最终应同时按场景报告，避免仅看总体均值。
- 新目录产物受.gitignore忽略，未改通用训练代码或静态产物；无自动提交／推送。
- 需要停止时向本次监督进程发送SIGTERM，保留已保存best和latest；不要对同目录重复启动。

## Follow-ups

MLP完成后自动接续GNN；训练过程中检查验证趋势，最终在固定测试集进行各场景评估。原始training_history与validation_history持续保存，可后续绘制曲线。


---

Doc-ID: IMPL-2026-11-06-001
Doc-Type: implementation
Owner: Codex
Date: 2026-11-06
Version: v1.0
Status: final
Source: dataset2使用0–3卡、dataset3使用4–7卡，并行执行完整state静态MLP/GNN及后续动态训练
Activated-Skills: code-agent-core, parameter-sanity

## Summary

- dataset1动态MLP与GNN均已完成10,000回合，最佳验证目标分别36.60831与34.1991009133。
- dataset2与dataset3分别绑定物理GPU0–3和4–7，两条独立监督链并行，每条链按静态MLP、静态GNN、动态MLP、动态GNN顺序运行。
- 静态各20,000回合从头训练，动态各10,000回合从本数据集对应静态best仅迁移权重；保持完整23／25输入、cost0、seed7与全局batch64。

## Assumptions

- 沿用前序实验的去应力与补偿额外费用0、物理时间不变；不调整网络、PPO算法或best选择指标。
- 新的分卡要求替代刚启动的dataset2八卡运行；停止后归档旧产物，四卡运行重新开始。

## Open Questions

None

## Plan Summary

核实并停止旧八卡监督链，归档产物，校验两个数据集和四卡配置，检查自动接续与故障隔离，启动两个持久后台进程并验证GPU绑定与训练日志。

## Changes (by file/module)

- outputs/fullstate_jingtai_set2与outputs/fullstate_dongtai_set2：精确保存dataset2静／动态结果，每目录含MLP与GNN；新GPU绑定为0–3。
- outputs/fullstate_jingtai_set3与outputs/fullstate_dongtai_set3：同结构保存dataset3静／动态结果，GPU绑定为4–7。
- 两个静态目录的run_training.py：独立文件锁、四卡环境与world_size一致性检查、阶段及模型状态、配置与代码哈希、错误停止后续任务、信号转发和完成检查。
- 动态阶段等待本链两种静态模型完成后严格校验对应best权重；自动记录initialization_sources.json，重新初始化Adam和日程。
- 各目录保存data_audit.json、配置、预检及launch_metadata.json；动态来源预检在静态完成后执行，当前记录为preflight_pending_sources.json。
- 两个静态目录的pipeline_status.json记录各自四项任务；supervisor.log记录启动和切换，模型training.log与原始训练／验证历史持续保存。
- 旧dataset2八卡目录归档为fullstate_jingtai_set2_archived_8gpu_20261106_101537及fullstate_dongtai_set2_archived_8gpu_20261106_101537，不用于新模型初始化。
- 本报告与docs/parameter-spec.md只追加条目，保留现有工作区修改；训练算法和模板没有改动。

## Tests Run

- 两数据集static／dynamic共5200个实例的cost0来源逐字段校验通过，物理时长与事件保持，划分各1000／100／200。
- 动态数据的静态配对、隐藏未来信息、随机训练事件、固定验证测试轨迹、场景比例校验通过。
- 两数据集各阶段MLP／GNN优化设置配对检查通过，参数量92162／1002274，world_size4、每卡batch16。
- 19项定向pytest通过，9.61秒：state_training、parallel_validation、distributed_training、zero_cost_dataset；之后仅调整运行监督脚本及配置，未更改训练代码。
- 四卡监督脚本模拟完整接续两种GPU分配均通过；检查torchrun四进程、CUDA_VISIBLE_DEVICES、静态到动态来源、顺序与完成状态；模拟训练失败后后续任务不启动通过。
- 旧八卡进程已退出且GPU释放；新监督脚本语法检查通过。实际训练启动和进度另见各training.log与pipeline_status.json。

## Risks and Rollback Notes

- 四卡与此前八卡的rank划分改变，固定seed仍不保证与八卡逐步权重相同；本轮dataset2／3统一使用四卡与相同全局参数。
- 两链共用CPU和磁盘，吞吐可能受环境模拟影响；训练时长应以实际进展估算。
- 动态验证65%无事件，事件处理效果仍需分场景报告；当前保持既有总体目标选best的协议。
- 每条链独立失败停止；停止时核实命令行后向对应supervisor发送SIGTERM，保留已保存checkpoint。不得在同目录重复启动。

## Follow-ups

观察两条链的训练日志与验证历史；静态完成后由监督程序自动检查并接续动态阶段，无需再次手动启动。


---

Doc-ID: IMPL-2026-11-06-002
Doc-Type: implementation
Owner: Codex
Date: 2026-11-06
Version: v1.0
Status: final
Source: 为dataset1/2/3准备简化state静态后动态的MLP/GNN代码和手动命令，GPU分别0–1、2–4、5–7
Activated-Skills: code-agent-core, parameter-sanity

## Summary

- 新增可手动启动三个并行数据集训练链的入口；每条链静态MLP→静态GNN→动态MLP→动态GNN。
- 使用compact18／compact21、cost0和原预算，输出按simplestate_jingtai_set1/2/3与simplestate_dongtai_set1/2/3命名。
- 三卡仍保持全局batch64，使用现有PPO补齐屏蔽；未修改正在运行的完整state代码或公共checker。

## Assumptions

- 用户本次要求准备代码和命令；不立即运行或自动排队简化训练。
- “命名同上”解释为保留状态组、静动态、set序号及MLP/GNN/checkpoints层级，用simplestate区分fullstate避免覆盖。
- 简化全局14项加4个动态预留字段，实际网络18维，动作21维，GNN图输入保持原13／2维。

## Open Questions

None

## Plan Summary

复用base模板与数据校验，增加手动前台监督入口和三卡参数预检，编写梯度与链路测试，提供CPU检查、前台／后台及单数据集命令。

## Changes (by file/module)

- scripts/train_simplestate_pipeline.py：动态生成并保存配置；固定GPU映射、独立数据集锁、防覆盖、完整state完成检查、状态保存、日志、信号转发、阶段完成检查与对应best权重严格迁移。
- 新入口支持--dataset all/1/2/3、--check、--output-root和--seed；all同时启动三条链。CPU check不创建输出目录，也不要求GPU。
- 新入口独立处理已有checker的整除限制，约束检查用可整除大小核对其它参数，实际模型、数据、训练配置与batch64保持不变；报告每rank真实有效样本与补齐槽位。
- tests/test_simplestate_pipeline.py：配置映射、三卡参数、待生成来源、真实checkpoint迁移审核、完整四任务顺序、失败隔离、防覆盖、check只读与总入口扇出。
- tests/test_distributed_ppo.py：扩展MLP和GNN到3个CPU进程、全局batch64及67样本，与单进程比较权重和损失，保留原2进程覆盖。
- docs/simplestate-training.md：操作说明、全部／分别启动命令、后台运行、输出和checkpoint示例、停止与恢复范围。
- docs/parameter-spec.md追加本次参数记录；保留已有文档修改。

## Tests Run

- 3个数据集共7800个静态／动态cost0实例逐字段来源检查及动态事件／配对／划分审核通过；预检JSON位于/tmp/simplestate_preflight_20261106.json。
- 预检模型输入18／21、GNN13／2，参数量MLP90370、GNN1001122，12个任务配置来源无误；动态best校验明确延迟至静态完成。
- 21项组合测试通过，16.08秒：新pipeline首8项、distributed_ppo四项、state_launcher五项、state_training四项；追加总入口测试后新pipeline独立9项全部通过，1.52秒。
- 三进程MLP/GNN的完整64样本非整除batch与尾批梯度等价测试通过；补齐样本不计入loss。
- --help与git diff --check通过；本次没有启动正式简化state训练，outputs无simplestate模型目录。

## Risks and Rollback Notes

- 启动时若当前fullstate两条链未complete会明确退出，不自动等待；已有目标目录则拒绝覆盖。本入口不自动恢复中断任务。
- 三数据集使用不同rank数，seed一致不保证同一浮点轨迹，需报告GPU与实际环境步数。
- 运行日志和曲线JSON持续保存，动态best仍按既有总体均值选择；事件处理研究应另行报告各场景结果。
- 公共训练代码、base模板和check_comparison.py保持不变，因此不触发当前fullstate监督程序哈希校验失败。

## Follow-ups

当前完整state训练完成后，用户按docs/simplestate-training.md手动执行--dataset all，或在三个终端分别启动--dataset 1/2/3。

---

Doc-ID: IMPL-2026-11-09-001
Doc-Type: implementation
Owner: Codex
Date: 2026-11-09
Version: v1.0
Status: final
Source: 保留原GNN，加入六维物理信号、共享6→16→16编码器及三类16→8→4边门控，并提供数据处理框图
Activated-Skills: code-agent-core, parameter-sanity

## Summary

- 新增默认关闭的物理门控；原关系K/V、Actor/Critic、奖励、输入profile与合法动作机制保持原协议。
- 六维物理信号在当前释放间隔生成，经图快照、Rollout、GraphBatch传至共享编码器和三组门控头；每边四个门值在softmax后缩放完整Value。
- 模块两层共享，普通／去应力／补偿正反边分三组，新增900参数；结构边与不适用阶段门值固定1。

## Assumptions

- 本次只实现已确认的物理门控，未同时实现共享K/V或FiLM，不启动新门控正式训练。
- “保留原有代码”采用开关和兼容旧schema保留原运行路径；已有训练配置不加开关，后续模型仍是原GNN。
- 多去应力设备时净收益取当前已知最早释放完成者，信号为工序阶段上下文，不逐机器独立生成。

## Open Questions

None

## Plan Summary

核对现有图与物理模型，加入按需物理字段、门控模块及消息缩放，打通采样／验证／checkpoint，执行梯度和兼容测试，提供独立配置及Mermaid框图。

## Changes (by file/module)

- environment/fjsp_env.py：新增六维字段名、当前工序间隔映射、适用mask、独立物理快照；get_graph_state新增include_physics可选参数，默认保持原返回字段。
- models/physical_gate.py：共享6→16→16 Tanh编码器、三组16→8→heads Tanh/Sigmoid头；初值约0.9，初始化末层小非零权重，结构和不适用边输出1。
- models/graph_actor_critic.py：开关、两层共享门控、get_edge_gates诊断接口及兼容旧schema的checkpoint kwargs。
- models/graph_attention.py：关系内softmax后乘门控，整体控制节点投影加边投影Value，无再次归一化。
- models/graph_batch.py：按需校验拼接物理节点信号和边mask；rollout原图快照保留新增字段。
- algorithms/ppo.py、evaluation/runner.py：只在门控模型采样与评估时请求include_physics图；train.py和models/factory.py读取开关并支持保存回载。
- scripts/check_comparison.py：识别门控架构选项、实际构建门控网络校验；原配置检查逻辑不变。
- configs/physical_gate/full_gnn_static.json、full_gnn_dynamic.json：独立dataset1_cost0完整state单卡示例，静态20k／动态10k，unique输出；动态需要显式指定新门控静态best。
- tests/test_physical_gate.py：物理信号、mask／方向／阶段、旧路径、概率与checkpoint全链测试；tests/test_distributed_ppo.py增加门控GNN两／三进程缺关系梯度测试。
- docs/physical-gating.md：六维定义、代码位置、配置、诊断、边类型表及完整数据处理Mermaid框图；本报告和parameter-spec只追加记录，保留feasibility既有改动。

## Tests Run

- 43项通过，16.37秒：physical_gate、graph_performance、state_training、comparison、simplestate_pipeline。
- 54项通过，22.50秒：distributed_ppo、dynamic_input_visibility、environment，包含两／三进程门控GNN梯度等价和缺关系DDP。
- 27项通过，6.61秒：state_launcher、state_ablation_configs、parallel_validation、distributed_training、base_state_profile、dynamic_environment。
- 合计124项定向测试通过；同样覆盖旧图forward／grad兼容、完整Value门控和单入边行为、快照独立、隐藏信息不泄漏、真实静态到动态训练与新checkpoint回载。
- 独立静态配置CPU预检通过，参数总量1,003,174；真实fullstate dataset1 GNN旧best严格回载通过，未启用门控。
- git diff --check通过；北京时间11:15核对三条simplestate监督状态均training、实际进程存活、原配置physical_gating为false，无错误。

## Risks and Rollback Notes

- 当前运行进程不热加载已import模块；之后启动的原配置默认关闭门控，仍为旧架构。未停止、重启或修改现有训练输出。
- 新开关模型改变schema，旧无门控checkpoint严格迁移会拒绝；新门控静态checkpoint可以按原initialize_from语义仅迁移权重到动态。
- 初始约0.9门值不等于完全保持旧策略，不宣称无损迁移；性能和额外墙钟开销尚需正式消融。
- 默认门控分支不计算物理字段；仅启用时增加图构造与推理成本。没有新增未来信息或修改调度约束。
- 停用门控只需使用原配置；门控结果放独立outputs/physical_gate目录，不覆盖fullstate／simplestate产物。

## Follow-ups

按docs/physical-gating.md查看完整框图与门控导出接口。正式训练需另行选择数据集／state和GPU；先基线与门控对照，不同时引入共享K/V和FiLM。

---

Doc-ID: IMPL-2026-11-09-002
Doc-Type: implementation
Owner: Codex
Date: 2026-11-09
Version: v1.0
Status: final
Source: 在原版代码中加入共享 K/V＋类型 FiLM，先不与物理门控合并
Activated-Skills: code-agent-core, parameter-sanity

## Summary

- 新增默认关闭的 shared_kv_film 开关，每层共享两个不同的 K/V 投影，以十种有向关系的三张参数表调制输出。
- K/V 逐维缩放、V 加偏置；FiLM 作用于节点 Value，随后加原边 Value；不设置会在当前 softmax 中抵消的类型 K 偏置。
- 完整 state 参数量从 1,002,274 降为 420,130，静态／动态训练与 checkpoint 链路接通；物理门控独立并互斥。

## Assumptions

- 沿用两层四头、隐藏维度128、原关系内目标节点 softmax；两层间不共享投影或 FiLM 表。
- 本次实现和验证，不启动新的正式训练；保留已在运行的简化 state 实验及其后续阶段配置。

## Open Questions

- None

## Plan Summary

核对原注意力与迁移协议，加入兼容旧结构的可选分支，接入训练／工厂／审查器，新增独立配置及说明，检查公式梯度、PPO、静态动态迁移和多进程一致性。

## Changes (by file/module)

- src/ppo_rs/models/graph_attention.py：新增共享 K/V 线性投影、三张关系参数表；按节点预计算投影一次，再索引并调制；旧关闭路径保留参数名及初始化顺序。
- src/ppo_rs/models/graph_actor_critic.py：开关、与物理门控互斥校验、两层独立构建、checkpoint schema v1。
- src/ppo_rs/models/factory.py、src/ppo_rs/train.py：构建／回载与训练参数接线；MLP 启用 FiLM 时明确拒绝。
- scripts/check_comparison.py：识别 shared_kv_film 架构差异并实际构建新模型审查。
- configs/shared_kv_film/full_gnn_static.json、full_gnn_dynamic.json：dataset1_cost0 完整 state，静态20k／动态10k、单卡、独立 unique 输出、门控显式关闭。
- tests/test_shared_kv_film.py：独立逐边参考公式、全部梯度、恒等初始化、节点／边 Value 顺序、类型偏置抵消、schema、概率与 PPO 参数更新、空边及参数量检查。
- tests/test_distributed_ppo.py：新增 FiLM 两／三进程及缺失关系测试；tests/test_state_training.py：新增简化／完整 state 的 FiLM 静态至含故障动态迁移。
- docs/shared-kv-film.md：公式、框图、配置、代码位置、手动启动及 checkpoint 边界；parameter-spec 和本报告追加记录。已有 feasibility.md、qa-notes.md 修改保留。

## Tests Run

- 45项通过，17.54秒：shared_kv_film、graph_performance、state_training、comparison、physical_gate。
- 37项通过，31.90秒：distributed_ppo、simplestate_pipeline、state_ablation_configs、distributed_training、parallel_validation、state_launcher。
- 合计82项通过，包含 FiLM 两／三进程非整除全局 batch 的梯度与损失等价、缺失关系、旧模型回归、物理门控回归、真实小规模静态／动态训练及 checkpoint 迁移。
- 新静态配置与匹配 MLP 的 CPU 全数据审查通过；训练／验证／测试1000／100／200，模型参数量420130／92162。
- outputs/fullstate_jingtai_set1/GNN/checkpoints/best.pt 真实旧 checkpoint 严格回载通过，参数1002274，两开关均关闭。
- 三条 simplestate 监督状态仍为 training，已生成静态／动态配置的两个扩展开关均关闭；未修改其输出或启动新正式训练。
- git diff --check 通过。

## Risks and Rollback Notes

- 新结构不支持无损吸收十套任意独立 K/V；旧架构 checkpoint 不能严格迁移到 FiLM 模型。新静态 FiLM checkpoint 可严格迁移到同架构动态训练。
- 参数减少58.1%不保证相同比例墙钟加速；尚未评估正式训练目标、收敛或速度。
- 缩放向量可取任意实数，不使用 Sigmoid；独立保留物理门控模型，目前模型入口拒绝两开关同时 true。
- 原配置默认关闭，不改变正在运行及后续原版实验；停用本变体只需不启用 shared_kv_film。

## Follow-ups

正式静态／动态启动示例见 docs/shared-kv-film.md。动态配置需手动传入本次 FiLM 静态最佳 checkpoint；未提供自动训练串行入口，也未启动正式训练。
