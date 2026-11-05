# 合成调度数据集

所有容量均指每个静态或动态子集的实例数量。工序数为每件普通加工工序数，不包含调度时按需插入的应力消除和补偿。

| 数据集 | train / validation / test | 工件数 | 每件工序数 | 普通加工设备 | 设备总数 |
|---|---|---|---|---|---|
| dataset_test | 100 / 20 / 20 | 6～10 | 4～7 | 4 | 6 |
| dataset1 | 1000 / 100 / 200 | 6～10 | 4～7 | 4 | 6 |
| dataset2 | 1000 / 100 / 200 | 12～20 | 5～10 | 6～10 | 8～12 |
| dataset3 | 1000 / 100 / 200 | 18～30 | 6～13 | 10～16 | 12～18 |

每组均采用 `static/{train,validation,test}` 和 `dynamic/{train,validation,test}`，应力消除机M3、补偿机M4各1台。以dataset2动态训练集为例：`data/dataset2/dynamic/train`。

`dataset_test` 原样归档了旧 `data/dataset` 和 `data/dataset_dynamic`。其原README和manifest保留历史描述；当前路径以本文件和父目录README为准。dataset1在每个原split上分别追加到目标容量，原140个静态及140个动态文件字节不变。dataset2静态1300个实例及原manifest全部原样保留，新增配对动态1300实例。`dataset1_cost0`、`dataset2_cost0`、`dataset3_cost0`是三组数据的零辅助费用派生版，均包含static/dynamic，各为1000／100／200；只将去应力和补偿费用设为0，时长、物理参数、目标权重及动态事件不变。

## 动态事件

- train每次reset重新采样，场景概率为无事件65%、插单20%、故障10%、混合5%；场景是延迟释放既有工件及生产机器故障，采用pause-repair-resume。
- 新版validation的固定场景数量依次为65／20／10／5；test为130／40／20／10。旧dataset_test保留12／5／2／1的历史配额。
- 每类事件数量仍为1～2；时间采用参考调度时长比例，维修采用平均加工时长比例。dataset2机器池和参考时长自动适配实例规模，同样事件数在大实例中相对扰动较低。
- 静态和动态按同一split的文件名配对。它们共享基础调度实例，不能把某静态实例放训练集而把对应动态实例放测试集。
- dataset_test是dataset1的历史子集，不能当作独立于新版dataset1的泛化测试；dataset_test/test仍归dataset1/test，未混入训练集。
- 沿用修复后的动态输入版本2：隐藏事件和未释放工件不会提前进入策略输入。

## 兼容与已有训练

`data/dataset`、`data/dataset_dynamic` 两个临时兼容链接已删除。旧小集使用 `data/dataset_test/static` 和 `data/dataset_test/dynamic`；新版配置已使用规范路径。历史checkpoint配置或旧命令若仍含这两个旧路径，应在续训／评价时显式改用dataset_test对应路径，原checkpoint文件无需改写。

保留dataset2原 `train`、`validation`、`test`及`manifest.json`到其static子目录的链接，当前dataset2训练及其配置快照继续有效。

新配置中的dataset1训练预算未随容量自动增加。旧dataset1的checkpoint仍来自旧100实例训练，现有初始化路径只保证权重可加载，不代表在新版1000实例上重新训练过。dataset2当前静态训练的样本、超参数和输出目录均未改变。

## 验证与复现

```bash
python scripts/organize_datasets.py --check
python scripts/validate_dynamic_dataset.py --dataset-root data/dataset1/dynamic
python scripts/validate_dynamic_dataset.py --dataset-root data/dataset2/dynamic
python scripts/validate_dataset2.py
```

`organize_datasets.py`的迁移操作仅用于旧布局，已有规范目录时会拒绝覆盖。`dataset-layout-migration.json`保存迁移前全部旧文件SHA256，`dataset-layout-validation.json`保存容量、范围、静态配对和固定场景校验结果。

生成器拒绝覆盖非空目录。需要复现时输出到新目录：

```bash
python scripts/generate_dataset.py --target-root /tmp/fjsp_dataset1/static --preserved-root data/dataset_test/static --seed 20261105
python scripts/generate_dynamic_dataset.py --source-root /tmp/fjsp_dataset1/static --target-root /tmp/fjsp_dataset1/dynamic --preserved-root data/dataset_test/dynamic --seed 20261103
python scripts/generate_dataset2.py --target-root /tmp/fjsp_dataset2/static --seed 20261030
python scripts/generate_dynamic_dataset.py --source-root /tmp/fjsp_dataset2/static --target-root /tmp/fjsp_dataset2/dynamic --seed 20261105
```

## 零额外费用版本 cost0

- `data/dataset1_cost0/{static,dynamic}/{train,validation,test}`
- `data/dataset2_cost0/{static,dynamic}/{train,validation,test}`
- `data/dataset3_cost0/{static,dynamic}/{train,validation,test}`

每个模式均为1000训练、100验证、200测试。相同相对文件名对应原始有成本实例；名称保持不变，只有`stress_relief.cost`和`compensation.cost`置0.0。原机与专用机补偿费用均为0，仍占用原设备与原时长。所有其他字段（含固定事件轨迹、随机事件分布、目标权重）保持一致。当前等待额外费用为0，因此正常完成时目标等于makespan；去应力、补偿仍会通过时间和资源竞争影响目标。

每个子集manifest记录源manifest及逐实例SHA256，父目录`validation.json`保存全部实例逐字段对照结果。验证命令：

```bash
python scripts/make_zero_cost_dataset.py --check
python scripts/validate_dynamic_dataset.py --dataset-root data/dataset1_cost0/dynamic
python scripts/validate_dynamic_dataset.py --dataset-root data/dataset2_cost0/dynamic
```

重新生成请指定新的空目标目录，已有非空目标会拒绝覆盖：

```bash
python scripts/make_zero_cost_dataset.py --source-root data/dataset1 --target-root /tmp/fjsp_dataset1_cost0
```

cost0和有成本版共享基础样本，应保持对应split成员一致；不跨版本混用训练与测试。只生成数据，不会自动开始训练或改变当前有成本训练。

## dataset3 大规模扩展

18～30工件、每件6～13道工序、12～18台总设备（包含M3／M4）。static与dynamic各1000／100／200，去应力／补偿仍保持原时长和费用分布，允许位置随最大13工序扩至O1～O12。动态每类事件数量仍为1～2，未随规模加强扰动。详见[dataset3说明](dataset3/README.md)。

```bash
python scripts/validate_dataset3.py --cost0-root data/dataset3_cost0
python scripts/make_zero_cost_dataset.py --source-root data/dataset3 --target-root data/dataset3_cost0 --check
```
