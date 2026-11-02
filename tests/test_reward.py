from __future__ import annotations

import unittest

from ppo_rs import load_instance
from ppo_rs.baselines.policies import natural_wait
from ppo_rs.environment import ResidualStressFJSPEnv


class RewardTests(unittest.TestCase):
    def test_undiscounted_rewards_telescope_to_negative_objective(self) -> None:
        env = ResidualStressFJSPEnv(load_instance("data/examples/synthetic_train.json"))
        _, info = env.reset(seed=4)
        total = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            choice = natural_wait(info["valid_actions"])
            _, reward, terminated, truncated, info = env.step(choice)
            total += reward
        self.assertTrue(terminated)
        self.assertAlmostEqual(total, -env.get_metrics()["objective"], places=8)


if __name__ == "__main__":
    unittest.main()

