from __future__ import annotations

from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn

from .q_network import QNetwork
from .replay_buffer import ReplayBuffer


class DQNAgent:
    """DQN agent with online network, target network, and replay memory."""

    def __init__(
        self,
        observation_dim: int = 4,
        action_dim: int = 3,
        gamma: float = 0.95,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        replay_capacity: int = 50_000,
        warmup_transitions: int = 500,
        target_update_steps: int = 250,
        gradient_clip_norm: float | None = 10.0,
        device: str | torch.device | None = None,
        seed: int = 0,
    ) -> None:
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup_transitions = warmup_transitions
        self.target_update_steps = target_update_steps
        self.gradient_clip_norm = gradient_clip_norm
        self.optimization_steps = 0

        self.device = torch.device(
            device if device is not None else "cpu"
        )

        self._random = random.Random(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.online_network = QNetwork(
            observation_dim=observation_dim,
            action_dim=action_dim,
        ).to(self.device)
        self.target_network = QNetwork(
            observation_dim=observation_dim,
            action_dim=action_dim,
        ).to(self.device)

        # Start both networks with the same weights.
        self.sync_target_network()

        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(),
            lr=learning_rate,
        )
        self.loss_function = nn.SmoothL1Loss()
        self.replay_buffer = ReplayBuffer(
            capacity=replay_capacity,
            observation_dim=observation_dim,
            seed=seed,
        )

    def select_action(
        self,
        observation: np.ndarray,
        epsilon: float,
    ) -> int:
        if self._random.random() < epsilon:
            return self._random.randrange(self.action_dim)

        state = torch.as_tensor(
            np.asarray(observation, dtype=np.float32).reshape(1, -1),
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():
            q_values = self.online_network(state)

        return int(torch.argmax(q_values, dim=1).item())

    def optimize_model(self) -> float | None:
        if len(self.replay_buffer) < self.warmup_transitions:
            return None

        batch = self.replay_buffer.sample(
            batch_size=self.batch_size,
            device=self.device,
        )

        current_q_values = self.online_network(batch.states).gather(
            dim=1,
            index=batch.actions,
        )

        with torch.no_grad():
            next_q_values = self.target_network(
                batch.next_states
            ).max(dim=1, keepdim=True).values
            # Do not bootstrap from true terminal states.
            target_q_values = batch.rewards + (
                1.0 - batch.terminated
            ) * self.gamma * next_q_values

        loss = self.loss_function(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()

        if self.gradient_clip_norm is not None:
            nn.utils.clip_grad_norm_(
                self.online_network.parameters(),
                self.gradient_clip_norm,
            )

        self.optimizer.step()
        self.optimization_steps += 1

        if self.optimization_steps % self.target_update_steps == 0:
            self.sync_target_network()

        return float(loss.detach().cpu().item())

    def sync_target_network(self) -> None:
        # Use the target network for stable Q targets.
        self.target_network.load_state_dict(
            self.online_network.state_dict()
        )
        self.target_network.eval()

    def save_checkpoint(
        self,
        path: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "online_network": self.online_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "optimization_steps": self.optimization_steps,
                "metadata": metadata or {},
            },
            checkpoint_path,
        )

    def load_checkpoint(
        self,
        path: str | Path,
        load_optimizer: bool = True,
    ) -> dict[str, Any]:
        checkpoint = torch.load(path, map_location=self.device)
        self.online_network.load_state_dict(checkpoint["online_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])

        if load_optimizer and "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.optimization_steps = int(
            checkpoint.get("optimization_steps", 0)
        )

        return dict(checkpoint.get("metadata", {}))
