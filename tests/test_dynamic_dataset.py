from __future__ import annotations

import unittest

from ppo_rs import load_instances
from ppo_rs.environment import materialize_dynamic_events


class DynamicDatasetTests(unittest.TestCase):
    def test_split_sizes_scales_and_event_modes(self) -> None:
        expected = {"train": 100, "validation": 20, "test": 20}
        for split, size in expected.items():
            instances = load_instances(f"data/dataset_dynamic/{split}")
            self.assertEqual(len(instances), size)
            for instance in instances:
                self.assertGreaterEqual(len(instance.jobs), 6)
                self.assertLessEqual(len(instance.jobs), 10)
                self.assertTrue(instance.dynamic_events)
                self.assertFalse(
                    instance.dynamic_events["unreleased_jobs_visible_to_agent"]
                )
                expected_mode = "stochastic" if split == "train" else "fixed"
                self.assertEqual(instance.dynamic_events["mode"], expected_mode)

    def test_training_resamples_and_evaluation_traces_are_fixed(self) -> None:
        train = load_instances("data/dataset_dynamic/train")
        for instance in train:
            events = instance.dynamic_events
            self.assertTrue(events["resample_each_reset"])
            self.assertNotIn("events", events)

        for split in ("validation", "test"):
            instances = load_instances(f"data/dataset_dynamic/{split}")
            seeds = set()
            scenarios: dict[str, int] = {}
            for instance in instances:
                events = instance.dynamic_events
                seeds.add(events["scenario_seed"])
                trace = events["events"]
                scenario = events["scenario"]
                scenarios[scenario] = scenarios.get(scenario, 0) + 1
                self.assertEqual(bool(trace), scenario != "none")
                self.assertEqual(
                    trace,
                    sorted(trace, key=lambda row: (row["time"], row["type"])),
                )
                self.assertTrue(
                    all(
                        not row["known_to_agent_before_occurrence"]
                        for row in trace
                    )
                )
            self.assertEqual(len(seeds), len(instances))
            self.assertEqual(
                scenarios,
                {
                    "none": 12,
                    "job_release_only": 5,
                    "breakdown_only": 2,
                    "both": 1,
                },
            )

    def test_trace_materialization_is_random_for_train_and_fixed_for_eval(self) -> None:
        train = load_instances("data/dataset_dynamic/train")[0]
        first = materialize_dynamic_events(train, seed=7)
        repeated = materialize_dynamic_events(train, seed=7)
        self.assertEqual(first, repeated)
        traces = {
            repr(materialize_dynamic_events(train, seed=seed))
            for seed in range(40)
        }
        self.assertGreater(len(traces), 1)

        validation = load_instances("data/dataset_dynamic/validation")[0]
        fixed_a = materialize_dynamic_events(validation, seed=7)
        fixed_b = materialize_dynamic_events(validation, seed=999)
        self.assertEqual(fixed_a, fixed_b)
        self.assertEqual(
            bool(fixed_a), validation.dynamic_events["scenario"] != "none"
        )


if __name__ == "__main__":
    unittest.main()
