from __future__ import annotations

import torch
from torch import nn


class QNetwork(nn.Module):
    """Small MLP that maps one observation to three action values."""

    def __init__(
        self,
        observation_dim: int = 4,
        action_dim: int = 3,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # DQN outputs raw Q-values, not probabilities.
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.model(observations)
