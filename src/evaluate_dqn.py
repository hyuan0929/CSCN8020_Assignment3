from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random

import numpy as np
import torch

from dqn import DQNAgent
from g1_rl import G1ElbowTargetEnv
from test_g1_elbow_env import choose_rule_based_action


BENCHMARK_GOALS = [-0.8, -0.4, 0.4, 0.8]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate DQN or rule-based policy on benchmark goals."
    )
    parser.add_argument("--policy", choices=["dqn", "rule_based"], default="dqn")
    parser.add_argument("--checkpoint", default="models/selected_dqn.pt")
    parser.add_argument("--episodes-per-goal", type=int, default=5)
    parser.add_argument("--seed", type=int, default=9020)
    parser.add_argument(
        "--scene",
        default="assets/g1_fixed_base/scene_29dof_fixed_base.xml",
    )
    parser.add_argument("--output-csv", default="results/evaluation.csv")
    return parser.parse_args()


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_policy_action(
    policy: str,
    agent: DQNAgent | None,
    observation: np.ndarray,
    info: dict,
    env: G1ElbowTargetEnv,
) -> int:
    if policy == "rule_based":
        return choose_rule_based_action(
            observation,
            info["controller_target"],
            env.action_increment,
        )

    if agent is None:
        raise ValueError("DQN agent is required for dqn policy.")

    # Final evaluation is greedy: epsilon is exactly zero.
    return agent.select_action(observation, epsilon=0.0)


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)

    env = G1ElbowTargetEnv(
        scene_path=Path(args.scene),
        goal_range=(-0.8, 0.8),
    )

    agent = None
    if args.policy == "dqn":
        agent = DQNAgent(seed=args.seed)
        agent.load_checkpoint(args.checkpoint, load_optimizer=False)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    for goal in BENCHMARK_GOALS:
        for episode_index in range(args.episodes_per_goal):
            seed = args.seed + len(rows)
            observation, info = env.reset(
                seed=seed,
                options={"goal_angle": goal},
            )
            cumulative_reward = 0.0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action = select_policy_action(
                    args.policy,
                    agent,
                    observation,
                    info,
                    env,
                )
                observation, reward, terminated, truncated, info = env.step(
                    action
                )
                cumulative_reward += reward

            rows.append(
                {
                    "policy": args.policy,
                    "goal_angle": goal,
                    "episode": episode_index + 1,
                    "success": int(info.get("is_success", False)),
                    "cumulative_reward": cumulative_reward,
                    "episode_length": info["episode_step"],
                    "final_absolute_error": info["absolute_error"],
                    "terminated": int(terminated),
                    "truncated": int(truncated),
                }
            )

    env.close()

    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    total_successes = sum(row["success"] for row in rows)
    print(f"Saved evaluation to {output_csv}")
    print(f"Successes: {total_successes}/{len(rows)}")

    for goal in BENCHMARK_GOALS:
        goal_rows = [row for row in rows if row["goal_angle"] == goal]
        successes = sum(row["success"] for row in goal_rows)
        mean_reward = np.mean([row["cumulative_reward"] for row in goal_rows])
        print(
            f"goal={goal:+.1f} "
            f"success={successes}/{len(goal_rows)} "
            f"mean_reward={mean_reward:.3f}"
        )


if __name__ == "__main__":
    main()
