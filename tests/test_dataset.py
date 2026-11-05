from __future__ import annotations

import unittest

from ppo_rs import load_instances


class DatasetTests(unittest.TestCase):
    def test_dataset_split_sizes_and_names(self) -> None:
        train = load_instances("data/dataset_test/static/train")
        validation = load_instances("data/dataset_test/static/validation")
        test = load_instances("data/dataset_test/static/test")

        self.assertEqual(len(train), 100)
        self.assertEqual(len(validation), 20)
        self.assertEqual(len(test), 20)
        self.assertEqual(len({instance.name for instance in train + validation + test}), 140)

    def test_every_instance_has_all_resource_types(self) -> None:
        for split in ("train", "validation", "test"):
            for instance in load_instances(f"data/dataset_test/static/{split}"):
                self.assertIn("M3", instance.stress_relief.machines)
                self.assertIn("M4", instance.compensation.machines)
                self.assertTrue(instance.compensation.allow_original_machine)
                self.assertAlmostEqual(instance.compensation.trigger_deformation, 0.30)
                self.assertEqual(
                    set(instance.machines),
                    {"M1", "M2", "M3", "M4", "M5", "M6"},
                )
                self.assertGreaterEqual(len(instance.jobs), 6)
                self.assertLessEqual(len(instance.jobs), 10)
                for job in instance.jobs:
                    self.assertGreaterEqual(len(job.operations), 4)
                    self.assertLessEqual(len(job.operations), 7)
                    for operation in job.operations:
                        self.assertGreaterEqual(len(operation.machines), 2)
                        self.assertLessEqual(len(operation.machines), 4)
                        self.assertTrue(
                            set(operation.machines).issubset({"M1", "M2", "M5", "M6"})
                        )

    def test_auxiliary_process_time_and_cost_ranges(self) -> None:
        for split in ("train", "validation", "test"):
            for instance in load_instances(f"data/dataset_test/static/{split}"):
                shortest_normal = min(
                    duration
                    for job in instance.jobs
                    for operation in job.operations
                    for duration in operation.machines.values()
                )
                longest_relief = max(instance.stress_relief.machines.values())
                self.assertLess(longest_relief, shortest_normal)
                self.assertGreaterEqual(instance.stress_relief.cost, 0.12)
                self.assertLessEqual(instance.stress_relief.cost, 0.48)
                self.assertGreaterEqual(instance.compensation.cost, 0.1)
                self.assertLessEqual(instance.compensation.cost, 0.4)
                self.assertAlmostEqual(instance.objective.relief_cost_weight, 0.5)
                self.assertAlmostEqual(instance.objective.compensation_cost_weight, 0.5)


if __name__ == "__main__":
    unittest.main()
