from __future__ import annotations

import unittest

import torch
from torch.distributions import Categorical

from ppo_rs import load_instance
from ppo_rs.algorithms import (
    PPO,
    PPOConfig,
    RolloutBuffer,
    collect_episode,
    generalized_advantage_estimate,
)
from ppo_rs.environment import ResidualStressFJSPEnv
from ppo_rs.models import ActorCritic, CandidateActor, build_actor_critic


class PPOTests(unittest.TestCase):
    def test_rollout_buffers_can_combine_complete_episodes(self) -> None:
        first = RolloutBuffer()
        second = RolloutBuffer()
        first.add([0.0], [[1.0]], 0, -0.1, 0.2, -1.0, True)
        second.add([1.0], [[0.0]], 0, -0.2, 0.3, -2.0, True)

        first.extend(second)

        self.assertEqual(len(first), 2)
        self.assertEqual(first.dones, [True, True])
        self.assertEqual(first.rewards, [-1.0, -2.0])

    def test_gae_known_values(self) -> None:
        advantages, returns = generalized_advantage_estimate(
            [1.0, 1.0], [0.5, 0.25], [False, True],
            last_value=0.0, gamma=1.0, gae_lambda=1.0
        )
        self.assertEqual([round(value, 6) for value in advantages], [1.5, 0.75])
        self.assertEqual([round(value, 6) for value in returns], [2.0, 1.0])

    def test_action_mask_never_samples_illegal_candidate(self) -> None:
        actor = CandidateActor(3, 2, hidden_dim=8)
        states = torch.zeros(3)
        features = torch.zeros((3, 2))
        mask = torch.tensor([True, False, True])
        logits = actor(states, features, mask)
        distribution = Categorical(logits=logits)
        samples = [int(distribution.sample().item()) for _ in range(100)]
        self.assertNotIn(1, samples)

    def test_complete_rollout_gae_update_and_evaluation(self) -> None:
        env = ResidualStressFJSPEnv(load_instance("data/examples/synthetic_train.json"))
        model = ActorCritic(env.OBSERVATION_DIM, env.ACTION_FEATURE_DIM, hidden_dim=32)
        buffer, metrics = collect_episode(env, model, seed=12)
        self.assertGreater(len(buffer), 0)
        self.assertTrue(metrics["terminated"])
        before = [parameter.detach().clone() for parameter in model.parameters()]
        losses = PPO(
            model,
            PPOConfig(ppo_epochs=1, mini_batch_size=8, learning_rate=1e-3),
        ).update(buffer)
        self.assertIn("policy_loss", losses)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, model.parameters())))
        eval_buffer, eval_metrics = collect_episode(env, model, seed=13, deterministic=True)
        self.assertGreater(len(eval_buffer), 0)
        self.assertTrue(eval_metrics["terminated"])

    def test_ppo_learning_rate_and_entropy_can_be_scheduled(self) -> None:
        model = ActorCritic(3, 2, hidden_dim=8)
        algorithm = PPO(
            model,
            PPOConfig(learning_rate=3e-4, entropy_coef=0.02),
        )

        algorithm.set_learning_rate(5e-5)
        algorithm.set_entropy_coef(0.005)

        self.assertAlmostEqual(algorithm.learning_rate, 5e-5)
        self.assertAlmostEqual(algorithm.entropy_coef, 0.005)

    def test_two_layer_attention_gnn_collects_and_updates(self) -> None:
        env = ResidualStressFJSPEnv(load_instance("data/examples/synthetic_train.json"))
        model = build_actor_critic(
            "gnn_attention",
            env.OBSERVATION_DIM,
            env.ACTION_FEATURE_DIM,
            32,
            node_feature_dim=env.GRAPH_NODE_FEATURE_DIM,
            edge_feature_dim=env.GRAPH_EDGE_FEATURE_DIM,
            node_type_count=env.GRAPH_NODE_TYPE_COUNT,
            edge_type_count=env.GRAPH_EDGE_TYPE_COUNT,
            gnn_layers=2,
            attention_heads=4,
        )

        buffer, metrics = collect_episode(env, model, seed=21)

        self.assertTrue(metrics["terminated"])
        self.assertTrue(all(graph is not None for graph in buffer.graph_states))
        before = [parameter.detach().clone() for parameter in model.parameters()]
        losses = PPO(
            model,
            PPOConfig(ppo_epochs=1, mini_batch_size=8, learning_rate=1e-3),
        ).update(buffer)
        self.assertTrue(torch.isfinite(torch.tensor(losses["value_loss"])))
        self.assertTrue(
            any(
                not torch.equal(old, new)
                for old, new in zip(before, model.parameters())
            )
        )


if __name__ == "__main__":
    unittest.main()
