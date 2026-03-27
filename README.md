# RL-Based Tsunami Alert Decision System

A class-based Python reinforcement learning project for tsunami alert decisions, paired with a cleaner Streamlit presentation layer for scenario walkthroughs and live training.

## Primary Demo

The recommended presentation flow is:

```bash
streamlit run app.py
```

This dashboard runs well in VS Code and does not require GIF or video output.

## Professionalized Project Layout

The working RL engine stays intact, while the dashboard layer is now organized under a dedicated presentation package:

```text
FinalProject-TsunamiDSS/
├── app.py
├── config.py
├── main.py
├── run_training.py
├── run_evaluation.py
├── run_simulation.py
├── outputs/
├── src/
│   ├── agent.py
│   ├── environment.py
│   ├── evaluator.py
│   ├── plotting.py
│   ├── simulator.py
│   ├── trainer.py
│   ├── utils.py
│   ├── presentation/
│   │   ├── __init__.py
│   │   ├── dashboard_app.py
│   │   ├── controller.py
│   │   ├── components.py
│   │   ├── charts.py
│   │   ├── models.py
│   │   └── scenarios.py
│   ├── dashboard.py
│   ├── app_controller.py
│   ├── components.py
│   ├── data_models.py
│   └── scenario_generator.py
└── tests/
    ├── test_environment.py
    └── test_app_logic.py
```

The root-level `src/dashboard.py`, `src/app_controller.py`, `src/components.py`, `src/data_models.py`, and `src/scenario_generator.py` remain as light compatibility wrappers so the working app keeps behaving the same way while the internal structure becomes cleaner.

## Dashboard Features

The dashboard provides:

- a sidebar scenario selector
- a scenario summary in the sidebar
- a step slider for instant navigation
- `Previous`, `Next`, `Auto Run`, and `Reset` controls
- current state cards for `Magnitude`, `Depth`, `WaveRisk`, `Confidence`, and `Time`
- selected action display
- reward panel
- per-step explanation text
- decision history table
- progress tracker
- final outcome summary
- reward trajectory chart
- signal profile chart
- action mix chart

## Included Demo Scenarios

- `High-Risk Confirmed Tsunami`
- `Uncertain Moderate-Risk Case`
- `False Alarm Case`
- `Delayed Response Case`

Each scenario is deterministic and presentation-friendly so the dashboard is easy to demonstrate.

## Sidebar Controls

The sidebar includes:

- `View`: switch between `Scenario Demo` and `Live Training`
- `Scenario Selector`: switches between named deterministic scenarios
- scenario summary details: risk level, step count, outcome, and total reward
- short usage guidance for reading the dashboard

## Step Navigation

Use the main navigation controls to inspect the scenario:

- `Previous`: move one step backward
- `Next`: move one step forward
- `Auto Run`: play the currently selected scenario to its final result
- `Reset`: return to the first step
- `Step Slider`: jump directly to any step in the selected scenario

Changing the slider updates the dashboard immediately.

## Dashboard Panels

The main dashboard layout includes:

- `Current State`: metric cards for the active state fields
- `Selected Action`: the action chosen at the current step
- `Reward`: step reward and cumulative reward
- `Explanation`: short reasoning text for the current step
- `Progress Tracker`: current position in the scenario
- `Decision History`: a growing table of decisions through the selected step
- `Final Outcome Summary`: projected or final scenario result
- `Reward Trajectory`: line chart of step and cumulative reward
- `Signal Profile`: bar chart of current warning-signal strength
- `Action Mix`: bar chart showing which decisions have been taken so far

## RL Core

The reinforcement learning project still includes:

- `main.py`
- `run_training.py`
- `run_evaluation.py`
- `config.py`
- `src/environment.py`
- `src/agent.py`
- `src/trainer.py`
- `src/evaluator.py`

The RL setup uses:

- **State**: `(Magnitude, Depth, WaveRisk, Confidence, Time)`
- **Actions**:
  - `Wait`
  - `Verify`
  - `Regional Alert`
  - `Full Alert`

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Training

Train the Q-learning agent with:

```bash
python run_training.py --episodes 2500 --seed 42
```

Training outputs include:

- `outputs/models/q_table.npy`
- `outputs/models/q_table_metadata.json`
- `outputs/logs/training_history.csv`
- `outputs/logs/training_summary.json`
- `outputs/logs/best_episode_trace.json`

## Evaluation

Evaluate the trained model with:

```bash
python run_evaluation.py --episodes 400 --seed 42
```

Evaluation outputs include:

- `outputs/logs/evaluation_history.csv`
- `outputs/logs/evaluation_summary.json`

## Combined CLI Entry Point

Run training, evaluation, or both using:

```bash
python main.py --mode both --train-episodes 2500 --eval-episodes 400 --seed 42
```

Modes:

- `--mode train`
- `--mode evaluate`
- `--mode both`

## Optional Launcher

If you want the project to open the browser automatically on port `8501`, you can still use:

```bash
python start_dashboard.py
```

This launcher starts the primary Streamlit app through `app.py`.

## Notes On Simulation Files

The repository still contains simulation scripts such as `run_simulation.py` and `src/simulator.py`, but GIF or video output is not required for the dashboard presentation flow.

## Tests

Run the test suite with:

```bash
pytest -q
```

The test suite covers:

- environment sanity checks
- deterministic scenario generation
- controller navigation logic
- scenario selection behavior
- presentation package imports
