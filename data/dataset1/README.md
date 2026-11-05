# dataset1

static和dynamic各含1000训练、100验证、200测试实例。

同一split内按文件名静态／动态配对；dynamic训练事件每次reset重新采样，验证和测试事件固定。
详细参数与生成来源见两个子目录的README和manifest。

规模：6～10工件，每件4～7道工序，4台普通设备＋M3／M4各1台。旧dataset_test按原split保留并追加；详细迁移与复现见[data/README](../README.md)。
