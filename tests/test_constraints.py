from __future__ import annotations

import unittest

from ppo_rs.baselines import run_baseline
from ppo_rs.environment import ResidualStressFJSPEnv
from ppo_rs.environment.constraints import schedule_violations
from ppo_rs.types import ActionType, ScheduleEntry

from tests.helpers import tiny_instance


class ConstraintTests(unittest.TestCase):
    def test_precedence_and_required_dwell(self) -> None:
        instance = tiny_instance(min_dwell=2.0, compensation_threshold=10.0)
        env = ResidualStressFJSPEnv(instance)
        metrics, schedule = run_baseline(env, "natural_wait")
        normal = [row for row in schedule if row["kind"] == "NORMAL"]
        self.assertTrue(metrics["terminated"])
        self.assertGreaterEqual(normal[1]["start"], normal[0]["end"] + 2.0 - 1e-9)

    def test_machine_conflicts_and_qualifications(self) -> None:
        instance = tiny_instance(jobs=2, min_dwell=1.0, compensation_threshold=10.0)
        env = ResidualStressFJSPEnv(instance)
        _, _ = run_baseline(env, "natural_wait")
        self.assertEqual(schedule_violations(instance, env.ledger.entries), [])
        invalid = [ScheduleEntry(ActionType.NORMAL, "J1", "O2", "M2", 2.0, 3.0)]
        violations = schedule_violations(instance, invalid)
        self.assertTrue(any("unqualified" in item for item in violations))

    def test_validator_detects_readiness_and_mandatory_compensation(self) -> None:
        readiness_instance = tiny_instance(
            min_dwell=3.0, compensation_threshold=10.0
        )
        early = [
            ScheduleEntry(ActionType.NORMAL, "J1", "O1", "M1", 0.0, 1.0),
            ScheduleEntry(ActionType.NORMAL, "J1", "O2", "M1", 2.0, 3.0),
        ]
        readiness = schedule_violations(readiness_instance, early)
        self.assertTrue(any(item.startswith("process readiness:") for item in readiness))

        deformation_instance = tiny_instance(
            min_dwell=3.0, compensation_threshold=0.2
        )
        uncompensated = [
            ScheduleEntry(ActionType.NORMAL, "J1", "O1", "M1", 0.0, 1.0),
            ScheduleEntry(ActionType.NORMAL, "J1", "O2", "M1", 4.0, 5.0),
        ]
        deformation = schedule_violations(deformation_instance, uncompensated)
        self.assertTrue(any(item.startswith("deformation threshold:") for item in deformation))

    def test_validator_detects_job_overlap_with_auxiliary_process(self) -> None:
        instance = tiny_instance(min_dwell=3.0, compensation_threshold=10.0)
        overlapping = [
            ScheduleEntry(ActionType.NORMAL, "J1", "O1", "M1", 0.0, 1.0),
            ScheduleEntry(ActionType.STRESS_RELIEF, "J1", "relief_1", "R", 1.0, 1.5),
            ScheduleEntry(ActionType.NORMAL, "J1", "O2", "M1", 1.25, 2.25),
        ]
        violations = schedule_violations(instance, overlapping)
        self.assertTrue(any("job conflict" in item for item in violations))


if __name__ == "__main__":
    unittest.main()

