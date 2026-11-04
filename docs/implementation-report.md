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
