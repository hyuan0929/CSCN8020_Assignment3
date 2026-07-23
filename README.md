# CSCN8020 Assignment 3: Unitree G1 DQN Left Elbow Control

Student full name: `Haibo Yuan`

Student ID: `9010929`

GitHub repository URL: `https://github.com/hyuan0929/CSCN8020_Assignment3`

Cloneable URL: `https://github.com/hyuan0929/CSCN8020_Assignment3.git`

## Project Summary

This project implements a student-written PyTorch Deep Q-Network for the
Unitree G1 left elbow Gymnasium environment from the primer workshop. The DQN
maps the four-value observation vector to three discrete action values:
decrease, hold, or increase the internal elbow target. Two epsilon-decay
configurations were trained headlessly on CPU, evaluated greedily on four
benchmark goals, compared with the provided rule-based policy, and demonstrated
in the MuJoCo viewer using the saved selected checkpoint.

## Validated Environment

- Windows 11 with WSL 2
- Ubuntu 24.04
- Python 3.14.4
- CPU execution
- WSLg for MuJoCo viewer rendering

The final validated project path was:

```text
/mnt/d/Desktop/8020/MujoCo/unitree/Unitree_MuJoCo_G1_Primer_Workshop
```

## Major Files

| Path | Description |
|---|---|
| `Unitree_MuJoCo_G1_Primer_Workshop.ipynb` | Completed primer workshop notebook. |
| `CSCN8020_Assignment3_DQN.ipynb` | Assignment 3 DQN summary notebook with commands, results, and deliverable map. |
| `src/g1_rl/g1_elbow_env.py` | Approved Gymnasium environment used by the DQN. |
| `src/dqn/q_network.py` | Student-written 4-input, 3-output PyTorch Q-network. |
| `src/dqn/replay_buffer.py` | Student-written bounded random replay buffer. |
| `src/dqn/agent.py` | Student-written DQN agent, epsilon-greedy action selection, Bellman update, target-network sync, checkpointing. |
| `src/train_dqn.py` | Headless training script for epsilon-decay configurations. |
| `src/evaluate_dqn.py` | Greedy DQN and rule-based evaluation script. |
| `src/render_dqn_policy.py` | Saved-checkpoint MuJoCo viewer demonstration script. |
| `src/generate_dqn_plots.py` | Plot and summary-table generation script. |
| `src/generate_report.py` | Technical report generation script. |
| `tests/test_dqn_components.py` | Unit tests for DQN components. |
| `models/selected_dqn.pt` | Selected trained DQN checkpoint. |
| `results/` | Training metrics, evaluation metrics, plots, and comparison summaries. |
| `report/DQN_Assignment_Report.pdf` | Full technical report. |
| `brightspace/CSCN8020_Assignment3_One_Page_Submission.pdf` | One-page Brightspace PDF with repository links. |

## Student-Written DQN Implementation

The DQN implementation uses an online Q-network and a separate target
Q-network. Transitions are stored as state, action, reward, next state, and
true termination flags in replay memory. Learning starts after the warm-up
threshold, samples random mini-batches, computes selected Q-values, builds
Bellman targets using the target network, masks true terminal states, and
optimizes Huber loss with Adam. Epsilon-greedy exploration is used during
training and greedy evaluation uses epsilon equal to 0.0.

## Setup

Run these commands from the repository root in WSL Ubuntu:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PyTorch CPU wheels are not resolved by the default installer, use:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

The project uses the official Unitree MuJoCo repository as an external source
for the original robot assets. If `external/unitree_mujoco` is not already
present, recreate it with:

```bash
git clone https://github.com/unitreerobotics/unitree_mujoco.git external/unitree_mujoco
git -C external/unitree_mujoco checkout ae6a8403e272733e9996ef59990880330496177f
```

## Notebook

Open the primer notebook in VS Code or Jupyter:

```bash
jupyter notebook Unitree_MuJoCo_G1_Primer_Workshop.ipynb
```

Open the Assignment 3 DQN summary notebook:

```bash
jupyter notebook CSCN8020_Assignment3_DQN.ipynb
```

## Environment Validation

Run the supplied environment checker and rule-based baseline before DQN
training:

```bash
source .venv/bin/activate
export PYTHONPATH=src
python -m compileall src tests
python src/test_g1_elbow_env.py
python src/evaluate_dqn.py --policy rule_based --episodes-per-goal 5 --output-csv results/rule_based_evaluation.csv
```

## DQN Tests

```bash
source .venv/bin/activate
export PYTHONPATH=src
python -m unittest tests.test_dqn_components -v
python src/smoke_test_dqn.py
```

## Training

Train Configuration A, the baseline epsilon decay:

```bash
python src/train_dqn.py \
  --config-name config_a \
  --epsilon-decay 0.995 \
  --episodes 500 \
  --output-dir results/config_a \
  --model-path models/config_a_dqn.pt
```

Train Configuration B, the faster epsilon decay:

```bash
python src/train_dqn.py \
  --config-name config_b \
  --epsilon-decay 0.985 \
  --episodes 500 \
  --output-dir results/config_b \
  --model-path models/config_b_dqn.pt
```

## Evaluation

Evaluate both DQN configurations greedily on the required 20 benchmark episodes:

```bash
python src/evaluate_dqn.py \
  --policy dqn \
  --checkpoint models/config_a_dqn.pt \
  --episodes-per-goal 5 \
  --output-csv results/config_a/evaluation_dqn.csv

python src/evaluate_dqn.py \
  --policy dqn \
  --checkpoint models/config_b_dqn.pt \
  --episodes-per-goal 5 \
  --output-csv results/config_b/evaluation_dqn.csv
```

Evaluate the selected checkpoint:

```bash
python src/evaluate_dqn.py \
  --policy dqn \
  --checkpoint models/selected_dqn.pt \
  --episodes-per-goal 5 \
  --output-csv results/selected_dqn_evaluation.csv
```

Evaluate the rule-based baseline:

```bash
python src/evaluate_dqn.py \
  --policy rule_based \
  --episodes-per-goal 5 \
  --output-csv results/rule_based_evaluation.csv
```

## Plots and Report

```bash
python src/generate_dqn_plots.py --output-dir results/plots
python src/generate_report.py
```

## Rendering the Selected Policy

Load the saved DQN checkpoint and render two target angles with epsilon equal
to 0.0:

```bash
python src/render_dqn_policy.py \
  --checkpoint models/selected_dqn.pt \
  --goals -0.8 0.8
```

Single-line version for video recording:

```bash
python src/render_dqn_policy.py --checkpoint models/selected_dqn.pt --goals -0.8 0.8
```

## Final Results Summary

| Metric | Result |
|---|---:|
| Config A final greedy evaluation | 20/20 successes |
| Config B final greedy evaluation | 20/20 successes |
| Rule-based baseline evaluation | 20/20 successes |
| Selected DQN checkpoint | Config A |
| Selected DQN mean evaluation reward | 13.282 |
| Selected DQN mean final absolute error | 0.00491 rad |

## Brightspace Submission Items

Submit the following items to Brightspace:

1. `brightspace/CSCN8020_Assignment3_One_Page_Submission.pdf`
2. GitHub repository URL: `https://github.com/hyuan0929/CSCN8020_Assignment3`
3. Cloneable URL: `https://github.com/hyuan0929/CSCN8020_Assignment3.git`
4. The rendered evaluation video, or an accessible video link.
