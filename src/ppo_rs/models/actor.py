"""Candidate-scoring actor for variable-size action sets."""

from __future__ import annotations

import torch
from torch import nn


class CandidateActor(nn.Module):
    """Scores each state/action pair independently with shared parameters."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh()
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh()
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        states: torch.Tensor,
        action_features: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return masked logits shaped ``[batch, candidates]`` or ``[candidates]``."""

        single = states.ndim == 1
        if single:
            states = states.unsqueeze(0)
            action_features = action_features.unsqueeze(0)
            if action_mask is not None:
                action_mask = action_mask.unsqueeze(0)
        state_embeddings = self.state_encoder(states).unsqueeze(1)
        action_embeddings = self.action_encoder(action_features)
        expanded_states = state_embeddings.expand(-1, action_embeddings.shape[1], -1)
        logits = self.scorer(torch.cat([expanded_states, action_embeddings], dim=-1)).squeeze(-1)
        if action_mask is not None:
            if not torch.all(action_mask.any(dim=-1)):
                raise ValueError("every policy row needs at least one legal action")
            logits = logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)
        return logits.squeeze(0) if single else logits

