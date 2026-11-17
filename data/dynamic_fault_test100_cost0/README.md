# 独立故障动态测试集（cost0）

set1/test、set2/test、set3/test各100例，共300例。每组仅故障67例、故障＋插单33例，由原动态场景10%／5%的条件比例2:1按最大余数法取整。

基础调度实例全新生成，与现有三组cost0的全部训练、验证、测试基础实例逐一去重；不复用旧测试实例，也不补入无事件或仅插单场景。

沿用各规模原生成器、物理参数和事件规则；仅辅助费用为0，目标权重与时长不变。插单指延迟释放既有工件，故障采用pause-repair-resume；每类事件1～2个。事件轨迹固定，未来事件和未释放工件隐藏。

manifest.json记录种子、配额、源manifest和每例SHA256；validation_report.json记录完整性校验。评估时直接传入各set的test子目录，避免将元数据JSON当作实例。

复现：`python scripts/generate_fault_test100.py --target-root /tmp/fault_test100_copy`。复检：`python scripts/generate_fault_test100.py --check`。

这是故障条件下的独立测试集，均值不代表原含65%无事件的混合动态分布。
