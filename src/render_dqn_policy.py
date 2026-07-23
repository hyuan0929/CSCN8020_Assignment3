from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import torch

from dqn import DQNAgent
from g1_rl import G1ElbowTargetEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a trained DQN policy in the MuJoCo viewer."
    )
    parser.add_argument("--checkpoint", default="models/selected_dqn.pt")
    parser.add_argument("--goals", nargs="+", type=float, default=[-0.8, 0.8])
    parser.add_argument("--seed", type=int, default=10020)
    parser.add_argument(
        "--pause-between-goals",
        type=float,
        default=5.0,
        help="Seconds to pause after each rendered goal.",
    )
    parser.add_argument(
        "--step-delay",
        type=float,
        default=0.15,
        help="Extra seconds to wait after each policy step for recording.",
    )
    parser.add_argument(
        "--scene",
        default="assets/g1_fixed_base/scene_29dof_fixed_base.xml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = G1ElbowTargetEnv(
        scene_path=Path(args.scene),
        render_mode="human",
        goal_range=(-0.8, 0.8),
    )
    agent = DQNAgent(seed=args.seed)
    agent.load_checkpoint(args.checkpoint, load_optimizer=False)

    print()
    print("Loaded selected trained DQN checkpoint.")
    print("Policy mode: greedy evaluation, epsilon = 0.0")
    print("The MuJoCo viewer will stay open until you close it here.")
    print()

    for goal_index, goal in enumerate(args.goals):
        observation, info = env.reset(
            seed=args.seed + goal_index,
            options={"goal_angle": goal},
        )
        env.render()

        if goal_index == 0:
            input("Adjust the viewer/camera, start recording, then press Enter...")
        else:
            input(f"Next goal is {goal:+.2f}. Press Enter to continue...")

        cumulative_reward = 0.0
        terminated = False
        truncated = False

        print(f"\n=== DQN RENDER EPISODE goal={goal:+.2f} ===")

        while not (terminated or truncated):
            # Rendered demos must use the learned greedy DQN policy.
            action = agent.select_action(observation, epsilon=0.0)
            observation, reward, terminated, truncated, info = env.step(action)
            cumulative_reward += reward

            print(
                f"step={info['episode_step']:3d} "
                f"action={action} "
                f"angle={info['elbow_angle']:+.4f} "
                f"goal={info['goal_angle']:+.4f} "
                f"error={info['absolute_error']:.4f}"
            )
            time.sleep(args.step_delay)

        print(
            f"success={bool(info.get('is_success', False))} "
            f"reward={cumulative_reward:.3f} "
            f"steps={info['episode_step']}"
        )
        print(f"Episode complete for goal={goal:+.2f}")

        if goal_index < len(args.goals) - 1:
            time.sleep(args.pause_between_goals)

    input("Demo complete. Press Enter to close the MuJoCo viewer...")
    env.close()


if __name__ == "__main__":
    main()
