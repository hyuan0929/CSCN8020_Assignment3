# CSCN8020 Assignment 3 Submission Checklist

## New Brightspace Requirements

- GitHub repository name: `CSCN8020_Assignment3`
- Repository URL: `https://github.com/hyuan0929/CSCN8020_Assignment3`
- Cloneable URL: `https://github.com/hyuan0929/CSCN8020_Assignment3.git`
- One-page Brightspace PDF: `brightspace/CSCN8020_Assignment3_One_Page_Submission.pdf`
- Demo video: `CSCN8020_Assignment3_DQN_Demo.mp4`
- Student full name: `Haibo Yuan`
- Student ID: `9010929`

## Completed Files

- Completed primer notebook: `Unitree_MuJoCo_G1_Primer_Workshop.ipynb`
- Assignment 3 DQN notebook: `CSCN8020_Assignment3_DQN.ipynb`
- Python DQN source code: `src/dqn/`, `src/train_dqn.py`, `src/evaluate_dqn.py`, `src/render_dqn_policy.py`
- Smoke test: `src/smoke_test_dqn.py`
- Plot generation: `src/generate_dqn_plots.py`
- Report generation: `src/generate_report.py`
- Unit tests: `tests/test_dqn_components.py`
- Selected checkpoint: `models/selected_dqn.pt`
- Config A checkpoint: `models/config_a_dqn.pt`
- Config B checkpoint: `models/config_b_dqn.pt`
- Training metrics: `results/config_a/training_metrics.csv`, `results/config_b/training_metrics.csv`
- Final DQN evaluation: `results/selected_dqn_evaluation.csv`
- Config evaluation files: `results/config_a/evaluation_dqn.csv`, `results/config_b/evaluation_dqn.csv`
- Rule-based baseline: `results/rule_based_evaluation.csv`
- Final evaluation table: `results/config_a/final_evaluation_table.csv`
- Required plots: `results/plots/*.png`
- Technical report: `report/DQN_Assignment_Report.pdf`
- Reproducibility commands: `README.md`
- Dependency file: `requirements.txt`
- Ignore file: `.gitignore`

## Results Summary

- Config A final greedy evaluation: 20/20 successes
- Config B final greedy evaluation: 20/20 successes
- Rule-based baseline evaluation: 20/20 successes
- Selected DQN checkpoint: Config A
- Selected DQN mean evaluation reward: 13.282
- Selected DQN mean final absolute error: 0.00491 rad

## Video Item

The assignment requires a 2-3 minute rendered video. Record it manually from WSLg with:

```bash
cd /mnt/d/Desktop/8020/MujoCo/unitree/Unitree_MuJoCo_G1_Primer_Workshop
source .venv/bin/activate
export PYTHONPATH=src
python src/render_dqn_policy.py --checkpoint models/selected_dqn.pt --goals -0.8 0.8
```

In the video, show:

- The MuJoCo viewer.
- Console output with goal angle, action, elbow angle, and error.
- At least two target angles.
- That the model is loaded from `models/selected_dqn.pt`.
- That the policy uses epsilon = 0.0.
- Whether each episode succeeded.

Do not use the rule-based policy in the video.

## Final Validation Before Submission

- Clone the GitHub repository into a new temporary folder.
- Create and activate a fresh virtual environment.
- Install dependencies with `python -m pip install -r requirements.txt`.
- Run the notebook and the README evaluation commands.
- Confirm that `models/selected_dqn.pt` loads without retraining.
- Confirm that the Brightspace PDF is exactly one page.
- Confirm that no `.venv`, cache folder, duplicate zip, or unnecessary large runtime folder is committed.
