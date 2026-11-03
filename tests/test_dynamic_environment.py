from __future__ import annotations

import unittest
from dataclasses import replace

from ppo_rs.baselines import run_baseline
from ppo_rs.environment import ResidualStressFJSPEnv

from tests.helpers import choose, tiny_instance


def fixed_dynamic_instance(events: list[dict]) -> object:
    return replace(
        tiny_instance(jobs=2, compensation_threshold=10.0),
        dynamic_events={
            "mode": "fixed",
            "future_events_visible_to_agent": False,
            "unreleased_jobs_visible_to_agent": False,
            "events": events,
        },
    )


class DynamicEnvironmentTests(unittest.TestCase):
    def test_unreleased_job_is_hidden_until_release_event(self) -> None:
        instance = fixed_dynamic_instance(
            [
                {
                    "time": 2.0,
                    "type": "job_release",
                    "job_id": "J2",
                    "known_to_agent_before_occurrence": False,
                }
            ]
        )
        env = ResidualStressFJSPEnv(instance)
        observation, info = env.reset(seed=7)
        state = dict(zip(env.OBSERVATION_NAMES, observation))

        self.assertEqual(set(env.jobs), {"J1"})
        self.assertNotIn("J2", env._graph_job_nodes)
        self.assertTrue(all(action.job_id != "J2" for action in info["valid_actions"]))
        self.assertAlmostEqual(state["remaining_processing_load_ratio"], 1.0)

        observation, *_ = env.step(choose(env, "WAIT"))
        state = dict(zip(env.OBSERVATION_NAMES, observation))
        self.assertEqual(env.current_time, 2.0)
        self.assertEqual(set(env.jobs), {"J1", "J2"})
        self.assertIn("J2", env._graph_job_nodes)
        self.assertGreater(state["remaining_processing_load_ratio"], 1.0)
        self.assertTrue(
            any(action.job_id == "J2" for action in env.get_valid_actions())
        )

    def test_breakdown_pauses_repairs_and_resumes_running_operation(self) -> None:
        instance = fixed_dynamic_instance(
            [
                {
                    "time": 0.5,
                    "type": "machine_breakdown",
                    "machine_id": "M1",
                    "repair_duration": 2.0,
                    "interruption_policy": "pause_resume",
                    "known_to_agent_before_occurrence": False,
                }
            ]
        )
        env = ResidualStressFJSPEnv(instance)
        env.reset(seed=7)
        env.step(choose(env, "NORMAL", job_id="J1", machine="M1"))
        observation, *_ = env.step(choose(env, "WAIT"))
        state = dict(zip(env.OBSERVATION_NAMES, observation))

        self.assertEqual(env.current_time, 0.5)
        self.assertAlmostEqual(state["failed_production_machine_ratio"], 0.5)
        self.assertAlmostEqual(state["interrupted_job_ratio"], 0.5)
        self.assertFalse(
            any(action.machine_id == "M1" for action in env.get_valid_actions())
        )

        metrics, schedule = run_baseline(
            ResidualStressFJSPEnv(instance), "ordinary_fjsp", seed=7
        )
        interrupted = next(
            entry
            for entry in schedule
            if entry["job_id"] == "J1" and entry["operation_id"] == "O1"
        )
        self.assertAlmostEqual(interrupted["end"] - interrupted["start"], 3.0)
        self.assertAlmostEqual(interrupted["interruption_time"], 2.0)
        self.assertEqual(metrics["machine_breakdown_event_count"], 1)
        self.assertEqual(metrics["interrupted_operation_count"], 1)
        self.assertEqual(metrics["schedule_violation_count"], 0)
        self.assertTrue(metrics["terminated"])
        self.assertFalse(metrics["truncated"])


if __name__ == "__main__":
    unittest.main()
