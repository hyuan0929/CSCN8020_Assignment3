from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random
import time

import numpy as np
import torch

from dqn import DQNAgent
from g1_rl import G1ElbowTargetEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a DQN for the Unitree G1 left elbow task."
    )
    parser.add_argument("--config-name", default="config_a")
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=8020)
    parser.add_argument("--max-wall-clock-seconds", type=float, default=18000.0)
    parser.add_argument(
        "--scene",
        default="assets/g1_fixed_base/scene_29dof_fixed_base.xml",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model-path", default=None)
    return parser.parse_args()


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)

    output_dir = Path(args.output_dir or f"results/{args.config_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model_path or f"models/{args.config_name}_dqn.pt")
    metrics_path = output_dir / "training_metrics.csv"

    env = G1ElbowTargetEnv(
        scene_path=Path(args.scene),
        goal_range=(-0.8, 0.8),
    )
    agent = DQNAgent(seed=args.seed)

    epsilon = 1.0
    epsilon_min = 0.05
    start_time = time.perf_counter()
    best_recent_success_rate = -1.0
    recent_successes: list[int] = []

    fieldnames = [
        "episode",
        "goal_angle",
        "cumulative_reward",
        "success",
        "episode_length",
        "final_absolute_error",
        "epsilon",
        "loss_mean",
        "terminated",
        "truncated",
        "wall_clock_seconds",
    ]

    with metrics_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for episode in range(1, args.episodes + 1):
            elapsed = time.perf_counter() - start_time
            if elapsed >= args.max_wall_clock_seconds:
                print("Stopped because wall-clock limit was reached.")
                break

            observation, info = env.reset(seed=args.seed + episode)
            cumulative_reward = 0.0
            losses: list[float] = []
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action = agent.select_action(observation, epsilon=epsilon)
                next_observation, reward, terminated, truncated, info = env.step(
                    action
                )

                # Truncation ends data collection, but only terminated is success.
                agent.replay_buffer.push(
                    state=observation,
                    action=action,
                    reward=reward,
                    next_state=next_observation,
                    terminated=terminated,
                )

                loss = agent.optimize_model()
                if loss is not None:
                    losses.append(loss)

                cumulative_reward += reward
                observation = next_observation

            success = bool(info.get("is_success", False))
            recent_successes.append(int(success))
            recent_successes = recent_successes[-50:]
            recent_success_rate = sum(recent_successes) / len(recent_successes)

            # Save the best recent success checkpoint during training.
            if recent_success_rate >= best_recent_success_rate:
                best_recent_success_rate = recent_success_rate
                agent.save_checkpoint(
                    model_path,
                    metadata={
                        "config_name": args.config_name,
                        "epsilon_decay": args.epsilon_decay,
                        "episode": episode,
                        "seed": args.seed,
                    },
                )

            writer.writerow(
                {
                    "episode": episode,
                    "goal_angle": info["goal_angle"],
                    "cumulative_reward": cumulative_reward,
                    "success": int(success),
                    "episode_length": info["episode_step"],
                    "final_absolute_error": info["absolute_error"],
                    "epsilon": epsilon,
                    "loss_mean": (
                        float(np.mean(losses)) if losses else ""
                    ),
                    "terminated": int(terminated),
                    "truncated": int(truncated),
                    "wall_clock_seconds": time.perf_counter() - start_time,
                }
            )
            csv_file.flush()

            epsilon = max(epsilon_min, epsilon * args.epsilon_decay)

            print(
                f"episode={episode:4d} "
                f"reward={cumulative_reward:8.3f} "
                f"success={int(success)} "
                f"epsilon={epsilon:.3f} "
                f"error={info['absolute_error']:.4f}"
            )

    env.close()
    agent.save_checkpoint(
        model_path,
        metadata={
            "config_name": args.config_name,
            "epsilon_decay": args.epsilon_decay,
            "seed": args.seed,
            "final_epsilon": epsilon,
        },
    )

    print(f"Saved metrics to {metrics_path}")
    print(f"Saved checkpoint to {model_path}")


if __name__ == "__main__":
    main()
