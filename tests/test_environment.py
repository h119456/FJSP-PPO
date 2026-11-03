from __future__ import annotations

import unittest

from ppo_rs import load_instance
from ppo_rs.environment import ResidualStressFJSPEnv

from tests.helpers import choose, tiny_instance


class EnvironmentTests(unittest.TestCase):
    def test_enhanced_action_features_add_relative_workload_and_pressure(self) -> None:
        env = ResidualStressFJSPEnv(
            tiny_instance(jobs=2), action_feature_profile="enhanced25"
        )
        _, info = env.reset()
        actions = info["valid_actions"]
        rows = info["action_features"]

        self.assertEqual(env.ACTION_FEATURE_DIM, 25)
        self.assertTrue(all(len(row) == 25 for row in rows))
        m1_index = next(
            index
            for index, action in enumerate(actions)
            if action.job_id == "J1" and action.machine_id == "M1"
        )
        m2_index = next(
            index
            for index, action in enumerate(actions)
            if action.job_id == "J1" and action.machine_id == "M2"
        )
        m1 = dict(zip(env.ACTION_FEATURE_NAMES, rows[m1_index]))
        m2 = dict(zip(env.ACTION_FEATURE_NAMES, rows[m2_index]))

        self.assertAlmostEqual(m1["relative_fastest_duration_score"], 1.0)
        self.assertAlmostEqual(
            m2["relative_fastest_duration_score"], 1.0 / 1.5
        )
        self.assertAlmostEqual(m1["job_remaining_workload_ratio"], 1.0)
        self.assertAlmostEqual(m1["machine_candidate_pressure"], 1.0)
        self.assertAlmostEqual(m1["normalized_remaining_dwell_time"], 0.0)

    def test_enhanced_observation_adds_workload_bottleneck_and_dispatchability(self) -> None:
        env = ResidualStressFJSPEnv(
            tiny_instance(jobs=2), observation_profile="enhanced19"
        )
        observation, _ = env.reset()
        state = dict(zip(env.OBSERVATION_NAMES, observation))

        self.assertEqual(env.OBSERVATION_DIM, 19)
        self.assertEqual(len(observation), 19)
        self.assertAlmostEqual(state["remaining_processing_load_ratio"], 1.0)
        self.assertAlmostEqual(state["max_job_remaining_load_ratio"], 1.0)
        self.assertAlmostEqual(state["max_machine_availability_gap"], 0.0)
        self.assertAlmostEqual(state["machine_availability_gap_std"], 0.0)
        self.assertAlmostEqual(state["dispatchable_job_ratio"], 1.0)

        observation, *_ = env.step(choose(env, "NORMAL", job_id="J1", machine="M1"))
        state = dict(zip(env.OBSERVATION_NAMES, observation))
        self.assertGreater(state["max_machine_availability_gap"], 0.0)
        self.assertGreater(state["machine_availability_gap_std"], 0.0)
        self.assertAlmostEqual(state["dispatchable_job_ratio"], 0.5)

    def test_observation_profile_rejects_unknown_name(self) -> None:
        with self.assertRaises(ValueError):
            ResidualStressFJSPEnv(
                tiny_instance(), observation_profile="unknown"
            )

        with self.assertRaises(ValueError):
            ResidualStressFJSPEnv(
                tiny_instance(), action_feature_profile="unknown"
            )

    def test_graph_state_matches_nodes_edges_and_candidates(self) -> None:
        env = ResidualStressFJSPEnv(load_instance("data/examples/synthetic_train.json"))
        _, info = env.reset(seed=3)

        graph = env.get_graph_state(info["valid_actions"])

        expected_nodes = (
            len(env.instance.jobs)
            + sum(len(job.operations) for job in env.instance.jobs)
            + len(env.instance.machines)
        )
        self.assertEqual(graph["node_features"].shape, (expected_nodes, 11))
        self.assertEqual(graph["node_types"].shape, (expected_nodes,))
        self.assertEqual(graph["edge_index"].shape[0], 2)
        self.assertEqual(graph["edge_features"].shape[1], 2)
        self.assertEqual(
            graph["action_nodes"].shape,
            (len(info["valid_actions"]), 3),
        )

    def test_minimum_dwell_blocks_next_operation(self) -> None:
        env = ResidualStressFJSPEnv(
            tiny_instance(min_dwell=3.0, compensation_threshold=10.0)
        )
        env.reset()
        env.step(choose(env, "NORMAL", machine="M1"))
        self.assertAlmostEqual(env.current_time, 1.0)
        self.assertFalse(env.check_process_readiness("J1"))
        self.assertFalse(any(a.kind.value == "NORMAL" for a in env.get_valid_actions()))
        env.step(choose(env, "WAIT"))
        self.assertAlmostEqual(env.current_time, 4.0)
        self.assertAlmostEqual(env.required_dwell("J1"), 3.0)
        self.assertAlmostEqual(env.compute_deformation("J1"), 0.5)
        self.assertTrue(any(a.kind.value == "NORMAL" for a in env.get_valid_actions()))
        self.assertFalse(any(a.kind.value == "STRESS_RELIEF" for a in env.get_valid_actions()))

    def test_stress_relief_occupies_machine_and_updates_deformation(self) -> None:
        env = ResidualStressFJSPEnv(
            tiny_instance(min_dwell=5.0, compensation_threshold=10.0)
        )
        env.reset()
        env.step(choose(env, "NORMAL", machine="M1"))
        before = env.compute_deformation("J1")
        self.assertAlmostEqual(before, 0.0)
        self.assertFalse(env.check_process_readiness("J1"))
        self.assertIn("STRESS_RELIEF", {a.kind.value for a in env.get_valid_actions()})
        env.step(choose(env, "STRESS_RELIEF"))
        relief = [row for row in env.get_schedule() if row["kind"] == "STRESS_RELIEF"]
        self.assertEqual(len(relief), 1)
        self.assertEqual(relief[0]["machine_id"], "R")
        self.assertAlmostEqual(relief[0]["end"] - relief[0]["start"], 0.65)
        self.assertAlmostEqual(env.relief_cost, 0.36)
        self.assertAlmostEqual(env.jobs["J1"].deformation_anchor, 1.0)
        self.assertAlmostEqual(env.compute_deformation("J1"), 1.0)
        self.assertAlmostEqual(env.compute_deformation("J1", env.current_time + 10.0), 1.0)
        self.assertTrue(env.jobs["J1"].release_completed)
        self.assertTrue(env.check_process_readiness("J1"))
        self.assertAlmostEqual(env.remaining_dwell("J1"), 0.0)
        self.assertLess(env.current_time, 1.0 + 5.0)
        self.assertIn("NORMAL", {a.kind.value for a in env.get_valid_actions()})

    def test_stress_relief_features_expose_delayed_cost_and_benefit(self) -> None:
        env = ResidualStressFJSPEnv(
            tiny_instance(min_dwell=5.0, compensation_threshold=0.2)
        )
        env.reset()
        env.step(choose(env, "NORMAL", machine="M1"))
        actions = env.get_valid_actions()
        rows = env.get_action_features(actions)
        relief_index = next(
            index
            for index, action in enumerate(actions)
            if action.kind.value == "STRESS_RELIEF"
        )
        feature = dict(zip(env.ACTION_FEATURE_NAMES, rows[relief_index]))

        self.assertEqual(len(rows[relief_index]), env.ACTION_FEATURE_DIM)
        self.assertEqual(env.ACTION_FEATURE_DIM, 21)
        self.assertAlmostEqual(feature["normalized_deformation_scale"], 1.0)
        self.assertGreater(feature["natural_wait_saved_ratio"], 0.0)
        self.assertGreater(
            feature["normalized_mandatory_compensation_duration"], 0.0
        )
        self.assertGreater(feature["normalized_mandatory_compensation_cost"], 0.0)
        self.assertNotIn("machine_available", feature)
        self.assertNotIn("estimated_net_objective_gain", feature)

    def test_compensation_transition_and_mandatory_threshold(self) -> None:
        env = ResidualStressFJSPEnv(
            tiny_instance(
                min_dwell=3.0,
                compensation_threshold=0.2,
                deformation_rate=0.1,
            )
        )
        env.reset()
        env.step(choose(env, "NORMAL", machine="M1"))
        self.assertNotIn(
            "COMPENSATION", {action.kind.value for action in env.get_valid_actions()}
        )
        env.step(choose(env, "WAIT"))
        self.assertGreater(env.compute_deformation("J1"), 0.2)
        kinds = {action.kind.value for action in env.get_valid_actions()}
        self.assertNotIn("NORMAL", kinds)
        self.assertIn("COMPENSATION", kinds)
        compensation_machines = {
            action.machine_id
            for action in env.get_valid_actions()
            if action.kind.value == "COMPENSATION"
        }
        self.assertEqual(compensation_machines, {"C", "M1"})
        before = env.compute_deformation("J1")
        env.step(choose(env, "COMPENSATION", machine="M1"))
        self.assertLess(env.jobs["J1"].deformation_anchor, before)
        self.assertAlmostEqual(
            env.compute_deformation("J1", env.current_time + 10.0),
            env.jobs["J1"].deformation_anchor,
        )
        self.assertAlmostEqual(env.compensation_cost, 0.3)
        compensation = [row for row in env.get_schedule() if row["kind"] == "COMPENSATION"]
        self.assertEqual(compensation[0]["machine_id"], "M1")
        self.assertEqual(env.get_metrics()["original_machine_compensation_count"], 1)
        self.assertEqual(env.get_metrics()["dedicated_compensation_count"], 0)
        self.assertFalse(
            any(
                "unqualified compensation" in violation
                for violation in env.get_metrics()["schedule_violations"]
            )
        )

    def test_compensation_can_follow_completed_stress_relief_without_natural_dwell(self) -> None:
        env = ResidualStressFJSPEnv(
            tiny_instance(min_dwell=5.0, compensation_threshold=0.2)
        )
        env.reset()
        env.step(choose(env, "NORMAL", machine="M1"))
        self.assertNotIn(
            "COMPENSATION", {action.kind.value for action in env.get_valid_actions()}
        )

        env.step(choose(env, "STRESS_RELIEF"))

        self.assertTrue(env.jobs["J1"].release_completed)
        self.assertLess(env.current_time, 1.0 + env.required_dwell("J1"))
        self.assertIn(
            "COMPENSATION", {action.kind.value for action in env.get_valid_actions()}
        )

    def test_dead_end_truncates_instead_of_looping(self) -> None:
        env = ResidualStressFJSPEnv(
            tiny_instance(
                min_dwell=1.0,
                compensation_threshold=0.01,
                deformation_rate=1.0,
                relief_max=0,
                compensation_max=0,
            )
        )
        env.reset()
        _, _, terminated, truncated, _ = env.step(choose(env, "NORMAL", machine="M1"))
        self.assertFalse(terminated)
        self.assertTrue(truncated)

    def test_wait_is_available_while_normal_work_is_legal(self) -> None:
        env = ResidualStressFJSPEnv(
            tiny_instance(jobs=2, min_dwell=3.0, compensation_threshold=10.0)
        )
        env.reset()
        env.step(choose(env, "NORMAL", job_id="J1", machine="M1"))
        kinds = {action.kind.value for action in env.get_valid_actions()}

        self.assertIn("NORMAL", kinds)
        self.assertIn("WAIT", kinds)

    def test_inter_operation_idle_time_is_split(self) -> None:
        env = ResidualStressFJSPEnv(
            tiny_instance(min_dwell=3.0, compensation_threshold=10.0)
        )
        env.reset()
        env.step(choose(env, "NORMAL", machine="M1"))
        env.step(choose(env, "WAIT"))
        metrics = env.get_metrics()

        self.assertAlmostEqual(metrics["stress_release_wait_time"], 3.0)
        self.assertAlmostEqual(metrics["machine_queue_wait_time"], 0.0)
        self.assertAlmostEqual(metrics["post_release_queue_wait_time"], 0.0)
        self.assertAlmostEqual(metrics["inter_operation_idle_time"], 3.0)

    def test_compact_observation_excludes_completed_job_dwell_and_deformation(self) -> None:
        env = ResidualStressFJSPEnv(
            tiny_instance(min_dwell=3.0, compensation_threshold=10.0)
        )
        env.reset()
        runtime = env.jobs["J1"]
        runtime.next_operation = len(env._jobs_by_id["J1"].operations)
        runtime.anchor_time = 0.0
        runtime.after_operation = "O2"
        env.current_time = 20.0
        observation = dict(zip(env.OBSERVATION_NAMES, env.get_observation()))

        self.assertEqual(env.OBSERVATION_DIM, 14)
        self.assertNotIn("remaining_operation_ratio", observation)
        self.assertNotIn("running_event_ratio", observation)
        self.assertNotIn("synthetic_data_flag", observation)
        self.assertNotIn("bias", observation)
        self.assertAlmostEqual(observation["mean_dwell_progress_ratio"], 0.0)
        self.assertAlmostEqual(
            observation["mean_deformation_to_compensation_threshold"], 0.0
        )
        self.assertAlmostEqual(
            observation["max_deformation_to_compensation_threshold"], 0.0
        )


if __name__ == "__main__":
    unittest.main()

