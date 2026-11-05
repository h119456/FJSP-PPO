# 基础／完整 state 训练

统一入口为 `scripts/train_state_comparison.py`，配置在 `configs/state_ablation/`，包含基础／完整 × MLP／GNN × 静态／动态共8份模板。默认使用dataset2、有辅助费用、8卡，每个模型依次使用全部8卡；本次仅准备代码并进行CPU测试，未启动新的正式训练。

## 输入与动态接口

| 组别 | 静态有效全局信息 | 动态预留 | 实际全局输入 | 每候选动作输入 | GNN节点／边维度 |
|---|---:|---:|---:|---:|---:|
| base基础组 | 14 | 4 | 18，compact18 | 21，compact21 | 13／2 |
| full完整组 | 19 | 4 | 23，unified23 | 25，enhanced25 | 13／2 |

静态全局最后4维及图节点最后2维为0，动态时填入可观测故障／维修／中断信息。静态和动态阶段使用同一profile与网络尺寸，所以无需改第一层即可微调各自的checkpoint。延迟工件释放仍通过可见工件集合和图更新表示，未来事件及未释放工件保持隐藏。

基础组全局去掉5项增强：剩余总加工负载、最大工件剩余负载、最大机器可用时间差、机器可用时间差标准差、可立即派工工件比例。候选动作去掉4项增强：相对最快时长评分、工件剩余工作量、机器候选压力、归一化剩余放置时间。合法动作、安全约束、奖励和物理模型不变。

GNN图通道在两组保持相同，仍可能从图中提取对应信息。本实验衡量增强全局／候选摘要的价值；不是剥夺GNN所有相关信息。直接缩小输入维数会改变少量参数，未采用固定尺寸零mask消融。

## 训练与检查命令

在仓库根目录 `/root/VLA/FJSP-PPO-cloud` 执行。以下命令使用已安装依赖的datascience解释器。`--check` 仅执行CPU预检，检查数据分离、输入尺寸、训练参数、动态checkpoint兼容，不占用训练GPU或启动训练。

基础组静态预检：

```bash
/root/miniconda3/envs/datascience/bin/python scripts/train_state_comparison.py \
  --group base --phase static --dataset dataset2 --check
```

基础组静态训练（依次MLP、GNN，每次8卡）：

```bash
/root/miniconda3/envs/datascience/bin/python scripts/train_state_comparison.py \
  --group base --phase static --dataset dataset2 --model both
```

只训练其中一个时使用 `--model mlp` 或 `--model gnn`。完整组将 `--group base` 改为 `--group full`，相同命令均可使用。选择物理设备可在启动前设置 `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`，默认不覆盖已有可见设备设置。

数据集可选择 dataset_test、dataset1、dataset1_cost0、dataset2、dataset2_cost0、dataset3、dataset3_cost0。`--seed 17` 改训练seed，eval_seed保持1007。`--world-size` 默认8，`--device` 默认cuda；CPU测试可指定 `--world-size 1 --device cpu`，但不会自动缩短正式训练预算。

## 静态到动态

建议静态训练完成后，从该次运行的 `checkpoints/best.pt` 开始动态微调：

```bash
/root/miniconda3/envs/datascience/bin/python scripts/train_state_comparison.py \
  --group base --phase dynamic --dataset dataset2 --model both \
  --initialize-from mlp=/绝对路径/基础MLP静态运行/checkpoints/best.pt \
  --initialize-from gnn=/绝对路径/基础GNN静态运行/checkpoints/best.pt \
  --check
```

确认预检通过后去掉 `--check` 执行。完整组同样使用 `--group full` 和完整组来源。也可使用 `--static-run mlp=/运行目录`／`--static-run gnn=/运行目录`，启动器自动补上 `checkpoints/best.pt`。

不显式提供来源时，仅在当前output-root下匹配相同数据集、组别、模型、seed的静态运行；只有唯一候选才自动绑定。没有候选会提示先训练静态，多于一个候选要求显式指定。单模型不要求另一模型checkpoint；both会先检查两者，再启动任何训练。

初始化严格检查profile、输入维数、hidden_dim、模型类型、GNN结构和完整权重。显式外部checkpoint可以来自其他实验目录，只保证schema兼容，不自动保证训练数据、seed、阶段或成本条件相同；使用者应核对来源。已有best.pt但静态仍在运行时技术上可加载，正式实验应等待静态完成。

动态阶段采用 `initialize_from`：仅继承权重，Adam、计数、学习率日程、训练历史从新运行开始。动态模板不包含假定的checkpoint路径，应经新launcher启动；若直接调用 `ppo_rs.train` 且未加 `--initialize-from`，则会从头训练动态策略。

## 默认参数及输出

| 参数 | 静态 | 动态微调 |
|---|---:|---:|
| episodes | 20000 | 10000 |
| learning_rate → final | 3e-4 → 5e-5 | 1e-4 → 2e-5 |
| entropy_coef → final | .02 → .005 | .01 → .003 |
| 全局mini_batch_size | 64 | 64 |
| episodes_per_update／ppo_epochs | 10／4 | 10／4 |
| validation／latest／snapshot间隔 | 500／100／5000 | 500／100／5000 |
| hidden_dim | 128 | 128 |
| GNN层数／heads | 2／4 | 2／4 |
| GPU进程数 | 8 | 8 |

两组、两模型在同一阶段使用相同优化参数，动态10k是独立阶段预算，不自动继承旧数据集的2000回合设置。所有保存间隔均落在更新边界。

默认结果父目录：`outputs/state_ablation/<dataset>/<base或full>/<static或dynamic>/<mlp或gnn>/seed7/`，其中每次训练创建唯一时间戳子目录，不覆盖历史结果。`--output-root` 可改父目录。启动配置快照、预检报告和stdout日志位于output-root下的 `launches/`。checkpoint、训练／验证历史及计时字段由公共train入口保存，已使用多卡并行验证等工程优化。

只检查时输出JSON到终端、不创建训练运行；如需保留，可重定向到自选文件。训练完成后应显式用两个best.pt在相同test集评估；启动器不自动使用测试集选模型。

## 完整组检查结果

- 原有 `dataset1_*state19_action25.json` 的环境与模型代码可运行，但MLP batch64／每20回合更新、GNN batch32／每10回合更新，不满足配对参数一致，缺少该输入组完整动态链路。旧配置为历史复现保留。
- 新完整组 `unified23/enhanced25` 已补齐静态／动态与MLP／GNN入口。静态有效信息仍是19／25，预留字段使其可直接转动态。
- 本次正在／已经进行的dataset2训练使用23／21，局部不是25，不能称为新完整组，也不能直接加载到23／25输入。旧精确19／25且GNN节点11维的checkpoint同样不能直接加载到新23／25、节点13维；需使用对应新静态训练的checkpoint。
- base和full的参数量分别为：MLP90370／92162，GNN1001122／1002274；同hidden不等容量。正式训练至少3个配对seed，数据和预算一致。
- 实际dataset2静态两组预检通过：1000／100／200，MLP／GNN非架构训练设置无差异。CPU已验证四个组／模型组合的静态训练、checkpoint回载、动态故障评估和动态微调；未执行新的正式GPU训练。
