from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


REPORT_PATH = Path("report/DQN_Assignment_Report.pdf")
PLOTS_DIR = Path("results/plots")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def paragraph(text: str, style_name: str = "BodyText") -> Paragraph:
    styles = getSampleStyleSheet()
    return Paragraph(text, styles[style_name])


def heading(text: str, level: int = 1) -> Paragraph:
    styles = getSampleStyleSheet()
    return Paragraph(text, styles[f"Heading{level}"])


def bullet_list(items: list[str]) -> ListFlowable:
    styles = getSampleStyleSheet()
    return ListFlowable(
        [ListItem(Paragraph(item, styles["BodyText"])) for item in items],
        bulletType="bullet",
        leftIndent=18,
    )


def styled_table(data: list[list[object]], col_widths: list[float]) -> Table:
    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9e7f7")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def add_plot(story: list, filename: str, caption: str) -> None:
    path = PLOTS_DIR / filename
    # Fixed plot size keeps the report layout predictable.
    story.append(Image(str(path), width=6.4 * inch, height=3.7 * inch))
    story.append(paragraph(f"<b>Figure.</b> {caption}"))
    story.append(Spacer(1, 0.12 * inch))


def evaluation_summary(path: Path) -> dict[str, float]:
    rows = read_csv(path)
    return {
        "successes": sum(int(row["success"]) for row in rows),
        "episodes": len(rows),
        "success_rate": mean(int(row["success"]) for row in rows),
        "mean_reward": mean(float(row["cumulative_reward"]) for row in rows),
        "mean_length": mean(float(row["episode_length"]) for row in rows),
        "mean_error": mean(float(row["final_absolute_error"]) for row in rows),
    }


def build_report() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    config_summary = read_csv(PLOTS_DIR / "config_comparison_summary.csv")
    config_a_eval = read_csv(Path("results/config_a/evaluation_dqn.csv"))
    config_b_eval = read_csv(Path("results/config_b/evaluation_dqn.csv"))

    rule_summary = evaluation_summary(Path("results/rule_based_evaluation.csv"))
    config_a_summary = evaluation_summary(
        Path("results/config_a/evaluation_dqn.csv")
    )
    config_b_summary = evaluation_summary(
        Path("results/config_b/evaluation_dqn.csv")
    )

    doc = SimpleDocTemplate(
        str(REPORT_PATH),
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
    )

    story: list = []

    story.append(heading("DQN Control of the Unitree G1 Left Elbow", 1))
    story.append(
        paragraph(
            "CSCN8020 Reinforcement Learning Assignment 3. "
            "This report documents a student-written PyTorch Deep Q-Network "
            "for the approved Unitree G1 left elbow Gymnasium environment."
        )
    )
    story.append(Spacer(1, 0.2 * inch))

    story.append(heading("1. Introduction", 2))
    story.append(
        paragraph(
            "The Unitree G1 Primer Workshop built a fixed-base MuJoCo model, "
            "a low-level proportional-derivative controller with MuJoCo bias "
            "compensation, and a Gymnasium environment for discrete high-level "
            "left elbow control. This assignment starts from that platform and "
            "replaces the transparent rule-based high-level action policy with "
            "a learned DQN policy."
        )
    )
    story.append(
        paragraph(
            "The learning objective is not to redesign the robot model. The "
            "DQN only chooses among three discrete changes to the internal "
            "controller target. The approved low-level controller then applies "
            "torque and produces the physical elbow movement in simulation."
        )
    )

    story.append(heading("2. Environment Definition", 2))
    story.append(
        bullet_list(
            [
                "Environment class: G1ElbowTargetEnv.",
                "Controlled joint: left_elbow_joint.",
                "Controlled actuator: left_elbow.",
                "Observation: current angle, velocity, goal angle, and angle error.",
                "Actions: decrease target, hold target, or increase target.",
                "Training goal range: -0.8 rad to +0.8 rad.",
                "Episode limit: 150 environment steps.",
                "Success: maintain the elbow inside the tolerance for the required streak.",
            ]
        )
    )
    story.append(
        paragraph(
            "The reward is shaped by the absolute angle error, includes a small "
            "bonus inside the success region, and gives a terminal bonus after "
            "the required success streak. The report treats true termination as "
            "success. Time-limit truncation stops the episode but is not treated "
            "as a terminal success signal for bootstrapping."
        )
    )

    story.append(heading("3. DQN Method", 2))
    story.append(
        paragraph(
            "DQN is suitable because the action space is discrete and small. "
            "The network maps a four-value continuous observation vector to "
            "three unconstrained Q-values, one for each action. The selected "
            "action is the action with the largest Q-value during greedy "
            "evaluation."
        )
    )
    story.append(
        styled_table(
            [
                ["Component", "Implementation"],
                ["Input", "4 observation values"],
                ["Hidden layer 1", "64 units, ReLU"],
                ["Hidden layer 2", "64 units, ReLU"],
                ["Output", "3 raw Q-values, no softmax"],
                ["Loss", "Huber loss"],
                ["Optimizer", "Adam, learning rate 0.001"],
                ["Discount factor", "gamma = 0.95"],
            ],
            [1.8 * inch, 4.9 * inch],
        )
    )
    story.append(Spacer(1, 0.15 * inch))
    story.append(
        paragraph(
            "The implementation uses an online Q-network for action-value "
            "updates and a separate target Q-network for stable Bellman targets. "
            "The target network is initialized from the online network and "
            "synchronized every 250 optimization steps."
        )
    )

    story.append(heading("4. Replay Buffer and Bellman Update", 2))
    story.append(
        paragraph(
            "The replay buffer stores state, action, reward, next state, and "
            "true termination. Learning starts only after the buffer contains "
            "at least 500 transitions, which prevents very small early samples "
            "from dominating the first gradient steps. Mini-batches of 64 "
            "transitions are sampled randomly."
        )
    )
    story.append(
        paragraph(
            "For each transition, the online network estimates Q(s, a). The "
            "target is r + gamma max_a Q_target(s_next, a) for non-terminal "
            "transitions. For true terminated transitions, the bootstrap term "
            "is masked out. This assignment environment can also truncate at "
            "the time limit; truncation is recorded as episode end but is not "
            "treated as a true success termination inside the replay target."
        )
    )

    story.append(PageBreak())
    story.append(heading("5. Training Methodology", 2))
    story.append(
        paragraph(
            "Training was run headlessly in WSL using CPU execution. Python, "
            "NumPy, PyTorch, and the Gymnasium reset seed were controlled from "
            "the command line. The same seed policy and hyperparameters were "
            "used for both exploration-decay configurations except for the "
            "epsilon decay value."
        )
    )
    story.append(
        styled_table(
            [
                ["Hyperparameter", "Value"],
                ["Episodes per configuration", "500"],
                ["Replay buffer capacity", "50,000 transitions"],
                ["Batch size", "64"],
                ["Warm-up transitions", "500"],
                ["Initial epsilon", "1.00"],
                ["Minimum epsilon", "0.05"],
                ["Config A epsilon decay", "0.995"],
                ["Config B epsilon decay", "0.985"],
                ["Evaluation epsilon", "0.00"],
            ],
            [2.3 * inch, 4.4 * inch],
        )
    )
    story.append(Spacer(1, 0.15 * inch))
    story.append(
        paragraph(
            "Both configurations finished well under the five-hour training "
            "limit. Config A took 82.0 seconds and Config B took 71.0 seconds "
            "on the tested CPU environment."
        )
    )

    story.append(heading("6. Exploration-Decay Results", 2))
    comparison_table = [["Metric", "Config A", "Config B"]]
    summary_by_config = {row["config"]: row for row in config_summary}
    def format_metric(config_name: str, metric: str) -> str:
        value = float(summary_by_config[config_name][metric])
        if metric == "episodes":
            return f"{int(value)}"
        if metric == "wall_clock_seconds":
            return f"{value:.1f}"
        if metric == "final_50_training_success_rate":
            return f"{value * 100:.1f}%"
        if metric == "evaluation_successes":
            episodes = int(
                float(summary_by_config[config_name]["evaluation_episodes"])
            )
            return f"{int(value)}/{episodes}"
        if metric == "mean_final_absolute_error":
            return f"{value:.5f}"
        return f"{value:.3f}"

    for metric, label in [
        ("episodes", "Training episodes"),
        ("wall_clock_seconds", "Wall-clock seconds"),
        ("final_epsilon", "Final epsilon"),
        ("final_20_mean_training_reward", "Mean reward, final 20 train episodes"),
        ("final_50_training_success_rate", "Success rate, final 50 train episodes"),
        ("evaluation_successes", "Evaluation successes"),
        ("mean_evaluation_reward", "Mean evaluation reward"),
        ("mean_final_absolute_error", "Mean final absolute error"),
    ]:
        comparison_table.append(
            [
                label,
                format_metric("config_a", metric),
                format_metric("config_b", metric),
            ]
        )
    story.append(styled_table(comparison_table, [3.0 * inch, 1.85 * inch, 1.85 * inch]))
    story.append(Spacer(1, 0.15 * inch))
    story.append(
        paragraph(
            "Both configurations reached 100 percent greedy evaluation success. "
            "Config B reached minimum epsilon earlier and had a slightly higher "
            "final-20 training reward. Config A produced a slightly higher mean "
            "evaluation reward and lower final absolute error, so Config A was "
            "selected as the final checkpoint."
        )
    )

    story.append(PageBreak())
    story.append(heading("7. Training Curves", 2))
    add_plot(
        story,
        "training_reward.png",
        "Raw training rewards are shown lightly, with a 20-episode moving average.",
    )
    add_plot(
        story,
        "rolling_success_rate.png",
        "Training success rate using a 50-episode rolling window.",
    )

    story.append(PageBreak())
    story.append(heading("8. Exploration and Loss", 2))
    add_plot(
        story,
        "epsilon_over_episodes.png",
        "Config B decays exploration faster and reaches epsilon_min earlier.",
    )
    add_plot(
        story,
        "loss_over_episodes.png",
        "Mean Huber loss over training episodes after replay warm-up.",
    )

    story.append(PageBreak())
    story.append(heading("9. Final Greedy Evaluation", 2))
    story.append(
        paragraph(
            "Final evaluation used epsilon = 0.0 on four required benchmark "
            "goals. Each goal was evaluated in five episodes, for 20 total "
            "episodes per configuration."
        )
    )
    selected_eval_table = [
        ["Goal", "Episodes", "Successes", "Success rate", "Mean reward"]
    ]
    for goal in [-0.8, -0.4, 0.4, 0.8]:
        a_rows = [row for row in config_a_eval if float(row["goal_angle"]) == goal]
        successes = sum(int(row["success"]) for row in a_rows)
        selected_eval_table.append(
            [
                f"{goal:+.1f} rad",
                "5",
                f"{successes}/5",
                f"{successes / 5:.2f}",
                f"{mean(float(row['cumulative_reward']) for row in a_rows):.3f}",
            ]
        )
    selected_eval_table.append(
        [
            "Overall",
            "20",
            f"{int(config_a_summary['successes'])}/20",
            f"{config_a_summary['success_rate']:.2f}",
            f"{config_a_summary['mean_reward']:.3f}",
        ]
    )
    story.append(
        styled_table(
            selected_eval_table,
            [1.3 * inch, 1.2 * inch, 1.4 * inch, 1.4 * inch, 1.4 * inch],
        )
    )
    story.append(Spacer(1, 0.15 * inch))
    story.append(
        paragraph(
            "The selected DQN is Config A. It met the required threshold with "
            "20 successful episodes out of 20."
        )
    )
    story.append(Spacer(1, 0.15 * inch))
    goal_table = [["Goal", "Config A successes", "Config B successes"]]
    for goal in [-0.8, -0.4, 0.4, 0.8]:
        a_rows = [row for row in config_a_eval if float(row["goal_angle"]) == goal]
        b_rows = [row for row in config_b_eval if float(row["goal_angle"]) == goal]
        goal_table.append(
            [
                f"{goal:+.1f} rad",
                f"{sum(int(row['success']) for row in a_rows)}/5",
                f"{sum(int(row['success']) for row in b_rows)}/5",
            ]
        )
    goal_table.append(["Overall", "20/20", "20/20"])
    story.append(styled_table(goal_table, [2.0 * inch, 2.35 * inch, 2.35 * inch]))
    story.append(Spacer(1, 0.15 * inch))
    add_plot(
        story,
        "evaluation_success_by_target.png",
        "Both DQN configurations reached 100 percent success for every benchmark goal.",
    )
    add_plot(
        story,
        "config_comparison.png",
        "Config A had slightly better mean evaluation reward and lower final error.",
    )

    story.append(PageBreak())
    story.append(heading("10. Rule-Based Policy Comparison", 2))
    story.append(
        styled_table(
            [
                ["Metric", "Rule-based", "Selected DQN"],
                [
                    "Successes / 20",
                    f"{int(rule_summary['successes'])}/20",
                    f"{int(config_a_summary['successes'])}/20",
                ],
                [
                    "Success rate",
                    f"{rule_summary['success_rate']:.2f}",
                    f"{config_a_summary['success_rate']:.2f}",
                ],
                [
                    "Mean reward",
                    f"{rule_summary['mean_reward']:.3f}",
                    f"{config_a_summary['mean_reward']:.3f}",
                ],
                [
                    "Mean episode length",
                    f"{rule_summary['mean_length']:.1f}",
                    f"{config_a_summary['mean_length']:.1f}",
                ],
                [
                    "Mean final absolute error",
                    f"{rule_summary['mean_error']:.5f}",
                    f"{config_a_summary['mean_error']:.5f}",
                ],
                [
                    "Qualitative behavior",
                    "Direct target stepping, then HOLD",
                    "Stable greedy DQN policy",
                ],
            ],
            [2.0 * inch, 2.35 * inch, 2.35 * inch],
        )
    )
    story.append(Spacer(1, 0.15 * inch))
    story.append(
        paragraph(
            "The rule-based policy is more sample efficient because it uses the "
            "known structure of the environment and requires no data collection. "
            "The learned DQN needs exploration and replay updates before it "
            "matches that behavior. After training, the selected DQN generalized "
            "across all four target angles and reached the same 20/20 success "
            "rate as the rule-based baseline."
        )
    )
    story.append(
        paragraph(
            "Near the goal, successful behavior requires HOLD rather than "
            "unnecessary target changes. The final DQN behavior was stable "
            "enough to complete every greedy evaluation episode, and the final "
            "absolute error was lower than the rule-based baseline in this run. "
            "A hand-written policy can still outperform a learned agent in this "
            "simple task because it directly encodes the intended target-tracking "
            "logic without exploration noise."
        )
    )

    story.append(heading("11. Limitations and Future Work", 2))
    story.append(
        bullet_list(
            [
                "Only one seed policy was used for the final controlled comparison.",
                "The task controls one joint, not whole-body humanoid behavior.",
                "The reward and success function were kept fixed as required.",
                "More random seeds would better estimate variability.",
                "A future extension could compare different target update intervals or network sizes.",
            ]
        )
    )

    story.append(heading("12. Reproducibility and AI Disclosure", 2))
    story.append(
        paragraph(
            "The submitted README lists exact commands for dependency setup, "
            "training, evaluation, plot generation, checkpoint loading, and "
            "rendering. AI assistance was used to help draft and organize code, "
            "tests, commands, and report text. All DQN components are included "
            "in source form and are intended to be explainable line by line."
        )
    )

    doc.build(story)
    print(f"Saved report to {REPORT_PATH}")


if __name__ == "__main__":
    build_report()
