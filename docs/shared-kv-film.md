# 共享 K/V＋类型 FiLM

这是独立于物理门控的 GNN 变体。配置 `shared_kv_film: true`、`physical_gating: false` 开启；原配置默认使用原版关系独立 K/V。新结构已接入训练、采样、验证、保存、回载和静态权重迁移，不自动启动正式训练。

## 每层计算

节点编码 $h_j\in\mathbb R^{128}$，关系 $r\in\{0,\ldots,9\}$。K、V 各用一个不同的共享投影矩阵，十种关系共用；两层的投影矩阵及 FiLM 表各自独立。

$$
k_j=W_Kh_j,\qquad v_j=W_Vh_j
$$

$$
\widetilde k_{j,r}=\gamma^K_r\odot k_j,
\qquad
\widetilde v_{j,r}=\gamma^V_r\odot v_j+\beta^V_r+U_rx_{ij}
$$

$$
\alpha_{ij,r}=\operatorname{softmax}_{j\in\mathcal N_r(i)}
\left(\frac{q_i^\top\widetilde k_{j,r}}{\sqrt{32}}+b_r(x_{ij})\right),
\qquad
m_{ij,r}=\alpha_{ij,r}\widetilde v_{j,r}
$$

上式点积及消息按四个 head 分别计算。FiLM 只调制节点 Value，原关系边 Value $U_rx_{ij}$ 随后相加。保留原 Query、边分数、关系内目标节点 softmax、残差、LayerNorm 和 Actor/Critic；没有物理信号或物理门控。

每层三张可训练表 `film_key_scale`、`film_value_scale`、`film_value_bias` 均为 `[10,128]`；每条边取对应关系一行后逐维运算。缩放初始化为 1，V 偏置初始化为 0，缩放不受 Sigmoid 限制。静态类型 K 偏置在当前分组 softmax 中抵消，因此不设置。参数全部由 PPO 反向传播更新，不添加辅助监督目标。

```mermaid
flowchart TD
    H[节点编码 h] --> K[共享 K 投影]
    H --> V[共享 V 投影]
    R[十种有向边类型 r] --> GK[类型 K 缩放向量]
    R --> GV[类型 V 缩放及偏置]
    K --> KF[K 逐维缩放]
    GK --> KF
    V --> VF[V 逐维缩放再加偏置]
    GV --> VF
    H --> Q[原 Query 投影]
    Q --> S[分 head 点积及原边分数]
    KF --> S
    S --> A[按关系和目标节点 softmax]
    E[原边特征投影] --> SUM[节点 V 加边 V]
    VF --> SUM
    SUM --> M[attention 乘完整 Value]
    A --> M
    M --> O[原聚合、残差和 LayerNorm]
```

## 代码位置与参数量

- `src/ppo_rs/models/graph_attention.py`：共享投影、三张类型参数表与逐关系消息计算。共享 K/V 每节点计算一次，再索引到边。
- `src/ppo_rs/models/graph_actor_critic.py`：开关、互斥校验及 checkpoint 架构信息。
- `src/ppo_rs/models/factory.py`、`src/ppo_rs/train.py`：构建、保存回载与训练入口。
- `scripts/check_comparison.py`：模型审查支持；`tests/test_shared_kv_film.py`：公式、梯度、概率及兼容性测试。

完整 state、hidden128、两层四头时，原模型 1,002,274 参数，新模型 **420,130** 参数，减少 582,144，约 58.1%。这不直接代表训练墙钟时间缩短相同比例。门控参数不计入本变体。

## 静态／动态训练示例

独立配置在 `configs/shared_kv_film/full_gnn_static.json` 和 `full_gnn_dynamic.json`，默认 dataset1_cost0、完整 state、seed7、单 GPU；静态 20,000 回合、动态 10,000 回合，batch64，保持原学习率及熵系数计划。输出根目录分别为 `outputs/shared_kv_film/dataset1/static/GNN`、`dynamic/GNN`，每次使用独立子目录。

手动训练前先选择空闲 GPU；以下 `<空闲GPU编号>` 和 `<本次FiLM静态运行目录>` 需替换为实际值：

```bash
cd /root/VLA/FJSP-PPO-cloud
CUDA_VISIBLE_DEVICES=<空闲GPU编号> PYTHONPATH=src \
  /root/miniconda3/envs/datascience/bin/python -u -m ppo_rs.train \
  --config configs/shared_kv_film/full_gnn_static.json
```

静态完成后，使用终端打印的 `run_directory` 找到新结构的最佳 checkpoint，再运行：

```bash
CUDA_VISIBLE_DEVICES=<空闲GPU编号> PYTHONPATH=src \
  /root/miniconda3/envs/datascience/bin/python -u -m ppo_rs.train \
  --config configs/shared_kv_film/full_gnn_dynamic.json \
  --initialize-from <本次FiLM静态运行目录>/checkpoints/best.pt
```

动态配置不预填不存在的 checkpoint 路径；需要显式传入 `--initialize-from` 才是静态后微调，不传则按训练入口原语义从头训练。动态仅迁移匹配架构的权重，重新初始化优化器与训练历史。这两个命令不构成自动串行脚本。

## 兼容与实验边界

`shared_kv_film` 默认 false，不增加旧模型参数、不改变旧初始化随机数顺序；旧 checkpoint kwargs 原样保留。开启时 checkpoint 自动加入 `shared_kv_film: true` 和 `shared_kv_film_schema_version: 1`，评估时据此还原结构。

新结构不能无损表示原版十套任意独立 K/V，不支持把旧独立 K/V checkpoint 严格加载为 FiLM 模型。应从新静态训练开始，再迁移至新动态训练。FiLM 初始恒等只对应共享 K/V 基线，不等同于原版独立 K/V 模型。

当前模型入口拒绝同时开启 `shared_kv_film` 和 `physical_gating`。已有 fullstate、simplestate 配置及监督脚本未改动；本次未启动新正式训练。

## dataset1 两卡自动串行训练（2026-11-10）

用户已授权在GPU0–1执行完整state训练。新增 `scripts/train_gnn_film_set1.py`，以原模板生成两卡独立配置，静态20,000回合完成且best/latest/final齐全后，通过严格迁移预检进入动态10,000回合。`shared_kv_film=true`、`physical_gating=false`，其余训练超参数沿用模板。

结果路径为 `outputs/GNN+FiLM_set1/static`、`dynamic`；监督状态为父目录的 `pipeline_status.json`，总日志为 `outputs/gnn_film_set1_launcher.log`，阶段日志为各阶段 `training.log`。静态从头训练，动态加载本次 `static/checkpoints/best.pt`；不使用原版或门控checkpoint。

2026-11-10已启动后台监督进程PID223977。入口有文件锁、目标目录防覆盖和GPU占用检查，失败停止不进入下一阶段，SIGTERM转发到训练进程组，不自动恢复中断或机器重启。

只读配置预检：

```bash
/root/miniconda3/envs/datascience/bin/python scripts/train_gnn_film_set1.py --check
```

首次手动后台启动（本次已经启动，请勿重复执行）：

```bash
nohup /root/miniconda3/envs/datascience/bin/python -u scripts/train_gnn_film_set1.py \
  > outputs/gnn_film_set1_launcher.log 2>&1 < /dev/null &
```


## dataset2 两卡自动串行训练（2026-11-11）

用户授权使用GPU0、1训练set2静态，再接动态。原入口现支持 `--dataset 2`，其余set1默认行为不变。实际后台命令为：

```bash
/root/miniconda3/envs/datascience/bin/python -u scripts/train_gnn_film_set1.py \
  --dataset 2 --pause-evaluator 310832
```

监督进程PID336150，结果目录 `outputs/GNN+FiLM_set2/static`、`dynamic`；总日志 `outputs/gnn_film_set2_launcher.log`，状态 `outputs/GNN+FiLM_set2/pipeline_status.json`。静态20,000回合，动态10,000回合；动态只加载本次静态best权重并重置优化器。两卡batch64，完整state，`shared_kv_film=true`、`physical_gating=false`。

本次暂时停止空闲评估调度器PID310832，整条训练链退出后自动SIGCONT恢复；活动评估worker存在时不会暂停或启动训练。SIGKILL／主机重启不能保证恢复。只读配置预检可使用 `--dataset 2 --check`，动态来源严格兼容性检查在静态完成后执行。

## 仅共享 K/V 消融（2026-11-16）

`shared_kv_only=true` 使用每层独立的一对 K、V 投影，在关系类型之间共享，不创建类型 FiLM 参数。完整 state、hidden128、两层四头模型为 412,450 参数；共享 K/V＋FiLM 为 420,130 参数。关系专属边特征投影、按关系／目标节点／头的 softmax 与跨关系求和均保留。

该开关与 `shared_kv_film`、`physical_gating` 互斥；checkpoint 保存 `shared_kv_only=true` 和 `shared_kv_only_schema_version=1`，静态至动态迁移要求架构完全匹配。原版及 FiLM 的默认构造、参数名称和 checkpoint 元数据保持兼容。

`scripts/train_shared_kv_only.py --dataset {1,2,3}` 分别使用 GPU4、GPU5、GPU6–7；输出到 `outputs/GNN+sharedKV_set{1,2,3}`。每组静态 20,000 回合，验证集 best 接续动态 10,000 回合，优化器重新初始化。`--check` 不占用 GPU，静态权重尚未生成时推迟实际迁移检查。监督器在静态完成、历史与 checkpoint 检查通过后才启动动态阶段；失败则停止后续阶段。
