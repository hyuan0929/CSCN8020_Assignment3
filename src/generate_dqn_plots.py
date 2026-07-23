from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt


CONFIGS = {
    "config_a": {
        "label": "Config A: decay 0.995",
        "training": Path("results/config_a/training_metrics.csv"),
        "evaluation": Path("results/config_a/evaluation_dqn.csv"),
    },
    "config_b": {
        "label": "Config B: decay 0.985",
        "training": Path("results/config_b/training_metrics.csv"),
        "evaluation": Path("results/config_b/evaluation_dqn.csv"),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Assignment 3 DQN plots and summary tables."
    )
    parser.add_argument(
        "--output-dir",
        default="results/plots",
        help="Directory for generated PNG plots and summary CSV files.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def moving_average(values: list[float], window: int) -> list[float]:
    result: list[float] = []
    for index in range(len(values)):
        start = max(0, index + 1 - window)
        result.append(mean(values[start : index + 1]))
    return result


def rolling_rate(values: list[int], window: int) -> list[float]:
    result: list[float] = []
    for index in range(len(values)):
        start = max(0, index + 1 - window)
        result.append(mean(values[start : index + 1]))
    return result


def save_training_reward_plot(
    training_data: dict[str, list[dict[str, str]]],
    output_dir: Path,
) -> None:
    plt.figure(figsize=(10, 6))

    for config_name, rows in training_data.items():
        episodes = [int(row["episode"]) for row in rows]
        rewards = [float(row["cumulative_reward"]) for row in rows]
        label = CONFIGS[config_name]["label"]

        # Raw rewards show noise; moving average shows trend.
        plt.plot(episodes, rewards, alpha=0.18, linewidth=0.8)
        plt.plot(
            episodes,
            moving_average(rewards, window=20),
            linewidth=2,
            label=f"{label} moving average",
        )

    plt.title("Training Reward")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_reward.png", dpi=160)
    plt.close()


def save_success_plot(
    training_data: dict[str, list[dict[str, str]]],
    output_dir: Path,
) -> None:
    plt.figure(figsize=(10, 6))

    for config_name, rows in training_data.items():
        episodes = [int(row["episode"]) for row in rows]
        successes = [int(row["success"]) for row in rows]
        plt.plot(
            episodes,
            rolling_rate(successes, window=50),
            linewidth=2,
            label=CONFIGS[config_name]["label"],
        )

    plt.title("Rolling Training Success Rate")
    plt.xlabel("Episode")
    plt.ylabel("Success rate, 50-episode window")
    plt.ylim(-0.02, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "rolling_success_rate.png", dpi=160)
    plt.close()


def save_epsilon_plot(
    training_data: dict[str, list[dict[str, str]]],
    output_dir: Path,
) -> None:
    plt.figure(figsize=(10, 6))

    for config_name, rows in training_data.items():
        episodes = [int(row["episode"]) for row in rows]
        epsilons = [float(row["epsilon"]) for row in rows]
        plt.plot(
            episodes,
            epsilons,
            linewidth=2,
            label=CONFIGS[config_name]["label"],
        )

    plt.title("Epsilon During Training")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "epsilon_over_episodes.png", dpi=160)
    plt.close()


def save_loss_plot(
    training_data: dict[str, list[dict[str, str]]],
    output_dir: Path,
) -> None:
    plt.figure(figsize=(10, 6))

    for config_name, rows in training_data.items():
        filtered = [row for row in rows if row["loss_mean"]]
        episodes = [int(row["episode"]) for row in filtered]
        losses = [float(row["loss_mean"]) for row in filtered]
        plt.plot(
            episodes,
            moving_average(losses, window=20),
            linewidth=2,
            label=CONFIGS[config_name]["label"],
        )

    plt.title("Training Loss")
    plt.xlabel("Episode")
    plt.ylabel("Mean Huber loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_over_episodes.png", dpi=160)
    plt.close()


def summarize_config(
    config_name: str,
    training_rows: list[dict[str, str]],
    evaluation_rows: list[dict[str, str]],
) -> dict[str, float | int | str]:
    final_20 = training_rows[-20:]
    final_50 = training_rows[-50:]

    return {
        "config": config_name,
        "episodes": len(training_rows),
        "wall_clock_seconds": float(training_rows[-1]["wall_clock_seconds"]),
        "final_epsilon": float(training_rows[-1]["epsilon"]),
        "final_20_mean_training_reward": mean(
            float(row["cumulative_reward"]) for row in final_20
        ),
        "final_50_training_success_rate": mean(
            int(row["success"]) for row in final_50
        ),
        "evaluation_successes": sum(
            int(row["success"]) for row in evaluation_rows
        ),
        "evaluation_episodes": len(evaluation_rows),
        "evaluation_success_rate": mean(
            int(row["success"]) for row in evaluation_rows
        ),
        "mean_evaluation_reward": mean(
            float(row["cumulative_reward"]) for row in evaluation_rows
        ),
        "mean_evaluation_length": mean(
            float(row["episode_length"]) for row in evaluation_rows
        ),
        "mean_final_absolute_error": mean(
            float(row["final_absolute_error"]) for row in evaluation_rows
        ),
    }


def save_comparison_outputs(
    summaries: list[dict[str, float | int | str]],
    output_dir: Path,
) -> None:
    summary_path = output_dir / "config_comparison_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    labels = [str(row["config"]) for row in summaries]
    rewards = [float(row["mean_evaluation_reward"]) for row in summaries]
    errors = [float(row["mean_final_absolute_error"]) for row in summaries]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].bar(labels, rewards, color=["#2b6cb0", "#2f855a"])
    axes[0].set_title("Mean Evaluation Reward")
    axes[0].set_ylabel("Reward")

    axes[1].bar(labels, errors, color=["#2b6cb0", "#2f855a"])
    axes[1].set_title("Mean Final Absolute Error")
    axes[1].set_ylabel("Radians")

    fig.tight_layout()
    fig.savefig(output_dir / "config_comparison.png", dpi=160)
    plt.close(fig)


def save_evaluation_by_target_plot(
    evaluation_data: dict[str, list[dict[str, str]]],
    output_dir: Path,
) -> None:
    goals = [-0.8, -0.4, 0.4, 0.8]
    width = 0.35
    x_values = list(range(len(goals)))

    plt.figure(figsize=(10, 6))

    for offset, (config_name, rows) in enumerate(evaluation_data.items()):
        rates = []
        for goal in goals:
            goal_rows = [
                row for row in rows if float(row["goal_angle"]) == goal
            ]
            rates.append(mean(int(row["success"]) for row in goal_rows))

        positions = [x + (offset - 0.5) * width for x in x_values]
        plt.bar(
            positions,
            rates,
            width=width,
            label=CONFIGS[config_name]["label"],
        )

    plt.title("Evaluation Success Rate by Target Angle")
    plt.xlabel("Target angle (rad)")
    plt.ylabel("Success rate")
    plt.xticks(x_values, [f"{goal:+.1f}" for goal in goals])
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "evaluation_success_by_target.png", dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_data = {
        name: read_csv(paths["training"])
        for name, paths in CONFIGS.items()
    }
    evaluation_data = {
        name: read_csv(paths["evaluation"])
        for name, paths in CONFIGS.items()
    }

    save_training_reward_plot(training_data, output_dir)
    save_success_plot(training_data, output_dir)
    save_epsilon_plot(training_data, output_dir)
    save_loss_plot(training_data, output_dir)
    save_evaluation_by_target_plot(evaluation_data, output_dir)

    summaries = [
        summarize_config(
            name,
            training_data[name],
            evaluation_data[name],
        )
        for name in CONFIGS
    ]
    save_comparison_outputs(summaries, output_dir)

    print(f"Saved DQN plots and summary files to {output_dir}")


if __name__ == "__main__":
    main()
