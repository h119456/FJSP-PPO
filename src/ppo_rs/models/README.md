# 模型策略结构

## MLP策略（兼容原实现）

- `actor.py`：14维全局状态与21维候选动作的双分支编码和动作评分；
- `critic.py`：14维全局状态价值网络；
- `actor_critic.py`：MLP Actor/Critic统一接口，`model_type = "mlp"`。

## 两层Attention GNN策略

- `graph_attention.py`：关系专属K/V变换、边特征偏置和多头邻居Attention；
- `graph_actor_critic.py`：三类节点编码、两层消息传递、候选动作Actor和图级Critic；
- `graph_batch.py`：变长图和变长候选动作的mini-batch拼接；
- 环境中的`get_graph_state()`：生成工件、工序、设备节点及十类有向边；
- `model_type = "gnn_attention"`。

## 公共入口

- `factory.py`：根据配置或Checkpoint创建MLP/GNN；
- `algorithms/ppo.py`：两种模型共享同一套PPO、GAE、Clip和Value Loss；
- 旧Checkpoint没有`model_type`时自动按MLP加载。
