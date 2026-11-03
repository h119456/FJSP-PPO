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
