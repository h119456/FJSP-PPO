# 模型策略结构

## MLP策略（兼容原实现）

- `actor.py`：默认23维全局状态与21维候选动作的双分支编码和动作评分；
- `critic.py`：默认23维全局状态价值网络；
- `actor_critic.py`：MLP Actor/Critic统一接口，`model_type = "mlp"`。

## 两层Attention GNN策略

- `graph_attention.py`：关系专属K/V变换、边特征偏置和多头邻居Attention；
- `graph_actor_critic.py`：三类节点编码、两层消息传递、候选动作Actor和图级Critic；
- `graph_batch.py`：变长图和变长候选动作的mini-batch拼接；
- 环境中的`get_graph_state()`：生成工件、工序、设备节点及十类有向边；
- `model_type = "gnn_attention"`。

两类模型使用相同的全局状态与候选动作 schema，但 GNN 还输入完整的工件、工序、设备图。默认 `unified23` 对应13维节点特征和2维边特征；`compact14`、`enhanced19` 对应11维节点特征。静态实例中的动态故障特征为零。

默认 hidden_dim=128、state23/action21 时，MLP 有91,650个参数，GNN 有1,001,762个参数；相同 hidden_dim 不表示参数量相同。公平训练的配对配置见 `configs/dataset_compare_mlp.json` 与 `configs/dataset_compare_gnn.json`，协议见 `docs/parameter-spec.md`。

## 公共入口

- `factory.py`：根据配置或Checkpoint创建MLP/GNN；
- `algorithms/ppo.py`：两种模型共享同一套PPO、GAE、Clip和Value Loss；
- 旧Checkpoint没有`model_type`时自动按MLP加载。
