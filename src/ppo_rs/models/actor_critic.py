"""Combined actor/critic API used by PPO and deterministic evaluation."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.distributions import Categorical

from .actor import CandidateActor
from .critic import StateCritic


class ActorCritic(nn.Module):
    model_type = "mlp"
    requires_graph = False

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.actor = CandidateActor(state_dim, action_dim, hidden_dim)
        self.critic = StateCritic(state_dim, hidden_dim)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.no_grad()
    def act(
        self,
        observation: Sequence[float],
        action_features: Sequence[Sequence[float]],
        *,
        graph_state=None,
        deterministic: bool = False,
    ) -> tuple[int, float, float]:
        del graph_state
        if not action_features:
            raise ValueError("cannot act on an empty candidate set")
        state = torch.tensor(observation, dtype=torch.float32, device=self.device)
        features = torch.tensor(action_features, dtype=torch.float32, device=self.device)
        logits = self.actor(state, features)
        distribution = Categorical(logits=logits)
        choice = torch.argmax(logits) if deterministic else distribution.sample()
        value = self.critic(state)
        # Return the three scalars with one device-to-host transfer rather than
        # synchronizing separately for action, log probability and value.
        result = torch.stack((choice.to(logits.dtype), distribution.log_prob(choice), value)).cpu().tolist()
        return int(result[0]), float(result[1]), float(result[2])

    def evaluate_actions(
        self,
        observations: torch.Tensor,
        action_features: torch.Tensor,
        action_masks: torch.Tensor,
        choices: torch.Tensor,
        *,
        graph_batch=None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del graph_batch
        logits = self.actor(observations, action_features, action_masks)
        distribution = Categorical(logits=logits)
        return distribution.log_prob(choices), distribution.entropy(), self.critic(observations)

