# Dataset Dynamic

This dataset reuses dataset1's static FJSP instances and adds exogenous dynamic-event scenarios.

- Split sizes: 100 train, 20 validation, 20 test.
- Static scale: 6-10 jobs, 4-7 operations per job, and the same six machines and parameter ranges as dataset1.
- Training instances contain stochastic event distributions that must be resampled from the environment reset seed on every episode.
- Validation and test instances contain fixed event traces for fair, reproducible policy comparison.
- Scenario mix: 65% no event, 20% insertion only, 10% breakdown only, and 5% both.
- Each 20-instance validation/test split is stratified as 12/5/2/1 across those four scenarios.
- Dynamic events are delayed job releases and normal-machine breakdowns.
- Breakdowns use pause-repair-resume semantics and are not visible to the policy before occurrence.
- Job definitions are stored in the file for reproducibility, but an unreleased job and its operations must not be exposed in the policy observation or graph. The active GNN topology grows when `job_release` occurs.

The environment executes these traces as external events. Future events and unreleased jobs remain hidden from the policy.

## 动态输入可见性（修复版本 2）

- 时间、变形幅值与变形速率的归一化尺度仅使用当前已释放工件及公开辅助工艺；工件释放时更新尺度并重建图，reset 时重新计算。
- 动态 WAIT 的候选存在性和 `duration` 只根据公开状态计算：优先等待最近的运行结束、已发生故障的维修结束或可见工件自然就绪；没有上述事件且存在 NORMAL 候选时，使用最短可见 NORMAL 时长作为等待期限。
- WAIT 的 `duration` 表示请求的公开等待期限；真实释放／故障可提前中断等待，环境内部仍在真实事件时刻推进。仅剩必须补偿且无公开未来事件时，不因隐藏事件额外增加 WAIT。不可决策期间仍由环境内部推进事件。
- 故障比例的机器集合使用公开机器清单扣除辅助角色，加上已释放路线和已发生故障揭示的生产机器；不会用未释放工件的路线推断分母。兼任辅助与生产的机器仍支持故障。
- 静态环境语义保持版本 7；新 checkpoint 额外记录 `dynamic_input_semantics_version=2`。旧静态 checkpoint 可继续用于静态任务，也可初始化新的动态训练。
- 缺少该版本或版本不同的 checkpoint 在动态评估／权重初始化时提示协议变化；动态断点续训会拒绝混用，需要用 `initialize_from` 创建新运行。
- 修复前的动态训练和评价使用过未来信息，其历史成绩需在修复后重新训练、统一评价才能用于正式结论。输入维度未改变，但维度一致不代表新旧动态成绩可直接比较。
