# 简化 state：三个数据集静态到动态训练

入口：[train_simplestate_pipeline.py](../scripts/train_simplestate_pipeline.py)。本次仅准备和检查代码，没有启动简化 state 训练。等待当前完整 state 两条训练链全部完成后，手动执行以下命令。

## 输入、数据和预算

- 全局输入 `compact18`：14个基础特征＋4个动态预留特征；静态阶段后4项为0。局部输入 `compact21`，每候选动作21维。GNN保留原13维节点、2维边。
- 数据使用 `dataset1_cost0`、`dataset2_cost0`、`dataset3_cost0`，延续去应力与补偿额外费用0，物理时长与设备占用保持。
- 每个数据集训练／验证／测试1000／100／200；动态训练事件随机重采样，验证测试轨迹固定，未来事件与未释放工件隐藏。
- 每条链顺序：静态MLP 20,000回合 → 静态GNN 20,000回合 → 动态MLP 10,000回合 → 动态GNN 10,000回合。
- 动态仅继承本数据集、本模型、本次简化state静态 `best.pt` 的权重；Adam、回合计数和日程重置。
- seed=7，eval_seed=1007；hidden128，GNN两层四头；PPO全局batch64、每10回合更新、每块4轮；验证每500回合，latest每100回合，快照每5000回合；不自动早停。
- 静态学习率3e-4→5e-5、熵系数0.02→0.005；动态学习率1e-4→2e-5、熵系数0.01→0.003；其余与完整组协议保持。

| 数据集 | 物理GPU | 每rank实际计算槽位 | 完整batch有效样本数 |
| --- | --- | --- | --- |
| dataset1 | 0,1 | 32 | 32／32 |
| dataset2 | 2,3,4 | 22 | 22／22／20 |
| dataset3 | 5,6,7 | 22 | 22／22／20 |

三卡完整batch填充2个槽位，这两个槽位损失被屏蔽，全局有效batch仍为64。已有PPO负责屏蔽与DDP梯度缩放，无需调整训练算法或batch。最终不足64的batch也按有效样本计权。

## 预检

从任意目录执行绝对路径入口也可以；以下命令在项目目录执行。

```bash
cd /root/VLA/FJSP-PPO-cloud
/root/miniconda3/envs/datascience/bin/python scripts/train_simplestate_pipeline.py --check
```

预检仅使用CPU，逐实例检查cost0来源、动态数据、划分与输入；无须等待当前训练结束，不创建训练目录。静态权重尚未生成时，动态来源检查标记为待执行；真实启动动态前会严格检查权重与来源数据集、seed、schema。

## 一条命令启动全部数据集

```bash
cd /root/VLA/FJSP-PPO-cloud
/root/miniconda3/envs/datascience/bin/python -u scripts/train_simplestate_pipeline.py --dataset all
```

这是前台监督程序，会同时启动三个数据集子监督程序，每条链依次运行四个模型阶段，并自动设置各自GPU、PYTHONPATH与线程数。建议在tmux中执行，或使用后台命令：

```bash
cd /root/VLA/FJSP-PPO-cloud
nohup /root/miniconda3/envs/datascience/bin/python -u scripts/train_simplestate_pipeline.py --dataset all > outputs/simplestate_launcher.log 2>&1 < /dev/null &
```

`$!` 为刚启动的总监督进程PID，可执行 `echo $!` 查看；不要重复执行启动命令。查看总日志：`tail -f outputs/simplestate_launcher.log`。

## 分别启动

如果希望三个终端分别管理，每个终端只执行下面对应的一条；不与 `--dataset all` 同时使用。

```bash
/root/miniconda3/envs/datascience/bin/python -u scripts/train_simplestate_pipeline.py --dataset 1
```

```bash
/root/miniconda3/envs/datascience/bin/python -u scripts/train_simplestate_pipeline.py --dataset 2
```

```bash
/root/miniconda3/envs/datascience/bin/python -u scripts/train_simplestate_pipeline.py --dataset 3
```

## 输出与检查点

| 数据集 | 静态目录 | 动态目录 |
| --- | --- | --- |
| 1 | outputs/simplestate_jingtai_set1 | outputs/simplestate_dongtai_set1 |
| 2 | outputs/simplestate_jingtai_set2 | outputs/simplestate_dongtai_set2 |
| 3 | outputs/simplestate_jingtai_set3 | outputs/simplestate_dongtai_set3 |

每个目录内均有 `MLP/` 与 `GNN/`。例如：

```text
outputs/simplestate_jingtai_set1/MLP/checkpoints/best.pt
outputs/simplestate_jingtai_set1/MLP/checkpoints/latest.pt
outputs/simplestate_jingtai_set1/MLP/checkpoints/final.pt
outputs/simplestate_dongtai_set1/GNN/checkpoints/best.pt
```

`MLP/training_config.json`、`MLP/training.log`、`training_history.json`、`validation_history.json` 保存配置、日志和曲线原始数据；GNN同理。每个静／动态父目录的 `pipeline_status.json` 记录该数据集全部四项任务状态；动态父目录还保存 `transfer_preflight.json`、`initialization_sources.json`。

## 防覆盖、停止与恢复

真实启动会检查现有 `fullstate_jingtai_set2` 与 `fullstate_jingtai_set3` 的 `pipeline_status.json`，两条完整训练链状态都为 `complete` 才允许训练；提前执行会报错退出，不会排队等待或停止当前训练。目录不存在时无需该项检查。

已有任一目标简化state目录时拒绝重新启动；文件锁防止同一数据集并发启动。该入口为从头训练和自动接续设计，不会自动跳过旧结果或恢复中断；若要重新实验可通过 `--output-root /新的父目录` 保存，若要续训则应单独使用checkpoint恢复流程。

向总监督进程发送SIGTERM会转发停止请求至三个数据集监督程序，再转发到torchrun进程组；只停止一个数据集时，核实该数据集 `pipeline_status.json` 中supervisor_pid后对其发SIGTERM。训练失败时该数据集后续任务不启动，其余数据集可继续。

GPU数量／rank分配改变时，即使seed相同，采样和浮点计算仍可能产生不同权重；报告实验应同时注明GPU数、全局budget与实际环境步数。
