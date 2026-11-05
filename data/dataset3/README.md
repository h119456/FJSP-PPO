# dataset3

static与dynamic每部分各1000训练、100验证、200测试实例，按相同split和文件名配对。

- 每实例18～30个工件，每件6～13道普通加工工序。
- 总设备12～18台：10～16台普通加工设备，另有去应力机M3、补偿机M4各1台。
- 每工序2～5台候选设备，普通加工时间1.5～5.0；物理、辅助时间、费用、目标权重及环境限制与dataset2一致。
- 最大工序数升至13，因此去应力和补偿max_per_job设12，允许在O1～O12后执行，保留每工序间隙可执行一次的原规则。
- dynamic训练每reset采样，场景概率无事件65%／插单20%／故障10%／混合5%；验证固定65／20／10／5例，测试固定130／40／20／10例。
- 插单与故障每类仍1～2次，参考时长按实际生产机器数量计算；大规模下相对扰动比例较低。
- cost0对应data/dataset3_cost0，仅两种辅助费用归零，时间与事件不变。

静态seed=20261106，动态seed=20261107；独立split随机种子，工件数与设备数边际均衡。对有成本和cost0共享基础实例的划分保持一致，不交叉混用训练／测试。

## 校验

```bash
python scripts/validate_dataset3.py --cost0-root data/dataset3_cost0
```

## 复现到新目录

```bash
python scripts/generate_dataset3.py --target-root /tmp/fjsp_dataset3/static --seed 20261106
python scripts/generate_dynamic_dataset.py --source-root /tmp/fjsp_dataset3/static --target-root /tmp/fjsp_dataset3/dynamic --seed 20261107
python scripts/make_zero_cost_dataset.py --source-root /tmp/fjsp_dataset3 --target-root /tmp/fjsp_dataset3_cost0
```

生成器拒绝覆盖已有非空目录。本次未自动启动训练。
