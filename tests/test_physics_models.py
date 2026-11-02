from __future__ import annotations

import unittest
import math

from ppo_rs.physics import (
    SyntheticCompensationModel,
    SyntheticIncreasingDeformation,
    SyntheticStressReliefModel,
)


class PhysicsModelTests(unittest.TestCase):
    def test_synthetic_deformation_increases_with_dwell(self) -> None:
        model = SyntheticIncreasingDeformation()
        early = model.compute(
            job_id="J", after_operation="O1", anchor_deformation=0.1,
            dwell_time=1.0, deformation_scale=0.5, deformation_rate=0.2, history=[]
        )
        late = model.compute(
            job_id="J", after_operation="O1", anchor_deformation=0.1,
            dwell_time=3.0, deformation_scale=0.5, deformation_rate=0.2, history=[]
        )
        self.assertAlmostEqual(early, 0.1 + 0.5 * (1.0 - math.exp(-0.2)))
        self.assertGreater(late, early)
        self.assertLess(late, 0.6)

    def test_each_operation_can_have_independent_parameters(self) -> None:
        model = SyntheticIncreasingDeformation()
        slow = model.compute(
            job_id="J", after_operation="O1", anchor_deformation=0.0,
            dwell_time=2.0, deformation_scale=0.5, deformation_rate=0.1, history=[]
        )
        fast = model.compute(
            job_id="J", after_operation="O2", anchor_deformation=0.0,
            dwell_time=2.0, deformation_scale=0.8, deformation_rate=0.2, history=[]
        )
        self.assertGreater(fast, slow)

    def test_stress_relief_reaches_saturation_and_compensation_corrects(self) -> None:
        post_deformation = SyntheticStressReliefModel().apply(
            deformation=0.2, maximum_deformation=1.0
        )
        self.assertAlmostEqual(post_deformation, 1.0)
        residual = SyntheticCompensationModel(0.1).apply(deformation=1.0)
        self.assertAlmostEqual(residual, 0.1)


if __name__ == "__main__":
    unittest.main()

