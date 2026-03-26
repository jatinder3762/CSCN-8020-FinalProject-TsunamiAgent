# RL-Based Tsunami Alert Decision System

A complete class-based reinforcement learning project where a Q-learning agent learns when to issue a tsunami warning, request verification, or wait.

## Project Overview

This project simulates an emergency warning controller for tsunami risk management. For each earthquake event episode, the agent observes a discrete disaster state and selects one of four actions:

- `0 = Wait`
- `1 = Verify`
- `2 = Regional Alert`
- `3 = Full Alert`

The objective is to maximize timely correct alerts while minimizing false alarms, missed dangerous events, and delay.

## RL Framing

- **Agent**: Tsunami Alert Decision Agent
- **Environment**: Custom discrete environment (no Gym dependency)
- **Algorithm**: Tabular Q-Learning (`numpy` Q-table)
- **State**: `(Magnitude, Depth, WaveRisk, Confidence, Time)`
- **Episode end**: Alert issued or decision window closes

### State Components

Each dimension has 3 discrete categories:

- Magnitude: Low, Medium, High
- Depth: Deep, Moderate, Shallow
- WaveRisk: Low, Medium, High
- Confidence: Low, Medium, High
- Time: Early, Mid, Late

Total state space size: `3^5 = 243`

### Reward Highlights

- `+100` correct full alert in high-risk case
- `+60` correct regional alert in medium-risk case
- `+30` smart verification under uncertainty
- `-100` missed dangerous alert
- `-60` false full alert
- `-30` false regional alert
- `-10` per-step delay
- `-15` unnecessary verification

Additional penalties/rewards are included for late risky waiting and safe low-risk no-alert outcomes.

## Folder Structure

```text
tsunami_rl_project/
│
├── main.py
├── config.py
├── requirements.txt
├── README.md
├── run_training.py
├── run_evaluation.py
├── notebooks/
│   └── evaluation_demo.ipynb
├── src/
│   ├── __init__.py
│   ├── environment.py
│   ├── agent.py
│   ├── trainer.py
│   ├── evaluator.py
│   ├── plotting.py
│   └── utils.py
├── data/
│   └── .gitkeep
├── outputs/
│   ├── models/
│   ├── plots/
│   └── logs/
└── tests/
    └── test_environment.py
```

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run Training

```bash
python run_training.py --episodes 2500 --seed 42
```

Training outputs:

- `outputs/models/q_table.npy`
- `outputs/models/q_table_metadata.json`
- `outputs/logs/training_history.csv`
- `outputs/logs/training_summary.json`
- `outputs/logs/best_episode_trace.json`
- Training plots in `outputs/plots/`

## Run Evaluation

```bash
python run_evaluation.py --episodes 400 --seed 42
```

Evaluation outputs:

- `outputs/logs/evaluation_history.csv`
- `outputs/logs/evaluation_summary.json`
- Evaluation plots in `outputs/plots/`

## Main Entry Point

Run training, evaluation, or both using one command:

```bash
python main.py --mode both --train-episodes 2500 --eval-episodes 400 --seed 42
```

Modes:

- `--mode train`
- `--mode evaluate`
- `--mode both`

## Notebook Demo

Open the notebook for an interactive demo and evaluation walkthrough:

```bash
jupyter notebook notebooks/evaluation_demo.ipynb
```

The notebook demonstrates:

- training and evaluation execution
- loading CSV logs
- plotting results
- running greedy trajectory samples

## Tests

Run environment sanity tests:

```bash
pytest -q
```

## Sample Outputs Explained

- **training_history.csv**: per-episode reward, steps, epsilon, and error flags.
- **training_summary.json**: aggregate metrics and action distribution.
- **best_episode_trace.json**: step-level trajectory of the best reward episode.
- **evaluation_summary.json**: generalization metrics under greedy policy.
- **plots**: reward trends, moving averages, epsilon decay, step trends, action usage, evaluation summary chart.
