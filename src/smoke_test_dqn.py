from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from dqn import DQNAgent
from g1_rl import G1ElbowTargetEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a short headless DQN smoke test."
    )
    parser.add_argument(
        "--scene",
        default="assets/g1_fixed_base/scene_29dof_fixed_base.xml",
        help="Path to the fixed-base G1 scene XML.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=8020,
        help="Random seed used by NumPy, PyTorch, and the environment.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = G1ElbowTargetEnv(
        scene_path=Path(args.scene),
        goal_range=(-0.8, 0.8),
    )
    agent = DQNAgent(
        observation_dim=4,
        action_dim=3,
        batch_size=4,
        warmup_transitions=4,
        seed=args.seed,
    )

    observation, _ = env.reset(seed=args.seed)
    last_loss = None

    for step_index in range(8):
        action = agent.select_action(observation, epsilon=1.0)
        next_observation, reward, terminated, truncated, info = env.step(
            action
        )

        # Time-limit truncation ends the episode, but it is not true success.
        agent.replay_buffer.push(
            state=observation,
            action=action,
            reward=reward,
            next_state=next_observation,
            terminated=terminated,
        )

        loss = agent.optimize_model()
        if loss is not None:
            last_loss = loss

        observation = next_observation

        if terminated or truncated:
            break

    env.close()

    if last_loss is None:
        raise RuntimeError("DQN smoke test did not run optimization.")

    print("DQN smoke test passed.")
    print(f"Replay size: {len(agent.replay_buffer)}")
    print(f"Optimization steps: {agent.optimization_steps}")
    print(f"Last loss: {last_loss:.6f}")
    print(f"Final absolute error: {info['absolute_error']:.6f}")


if __name__ == "__main__":
    main()
