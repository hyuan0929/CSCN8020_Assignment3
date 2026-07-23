from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import numpy as np
import torch


@dataclass(frozen=True)
class TransitionBatch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    terminated: torch.Tensor


class ReplayBuffer:
    """Fixed-size memory for random DQN mini-batches."""

    def __init__(
        self,
        capacity: int = 50_000,
        observation_dim: int = 4,
        seed: int | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")

        self.capacity = capacity
        self.observation_dim = observation_dim
        self._memory: deque[
            tuple[np.ndarray, int, float, np.ndarray, bool]
        ] = deque(maxlen=capacity)
        self._random = random.Random(seed)

    def __len__(self) -> int:
        return len(self._memory)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
    ) -> None:
        state_array = np.asarray(state, dtype=np.float32).reshape(
            self.observation_dim
        )
        next_state_array = np.asarray(next_state, dtype=np.float32).reshape(
            self.observation_dim
        )

        self._memory.append(
            (
                state_array,
                int(action),
                float(reward),
                next_state_array,
                bool(terminated),
            )
        )

    def sample(
        self,
        batch_size: int,
        device: torch.device,
    ) -> TransitionBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if len(self._memory) < batch_size:
            raise ValueError("not enough transitions to sample")

        transitions = self._random.sample(list(self._memory), batch_size)
        states, actions, rewards, next_states, terminated = zip(*transitions)

        return TransitionBatch(
            states=torch.as_tensor(
                np.asarray(states, dtype=np.float32),
                dtype=torch.float32,
                device=device,
            ),
            actions=torch.as_tensor(
                np.asarray(actions, dtype=np.int64).reshape(-1, 1),
                dtype=torch.long,
                device=device,
            ),
            rewards=torch.as_tensor(
                np.asarray(rewards, dtype=np.float32).reshape(-1, 1),
                dtype=torch.float32,
                device=device,
            ),
            next_states=torch.as_tensor(
                np.asarray(next_states, dtype=np.float32),
                dtype=torch.float32,
                device=device,
            ),
            terminated=torch.as_tensor(
                np.asarray(terminated, dtype=np.float32).reshape(-1, 1),
                dtype=torch.float32,
                device=device,
            ),
        )
