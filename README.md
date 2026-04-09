# RL-Based Tsunami Alert Decision System

This project demonstrates how reinforcement learning can support tsunami alert decision-making. It combines a Q-learning environment with a Streamlit dashboard so you can both train the agent and present decision scenarios in a clear, step-by-step format.

The project is designed for two main use cases:

- interactive presentation of tsunami decision scenarios through Streamlit
- local experimentation with training and evaluating a Q-learning policy

## What The Project Does

The system models tsunami warning decisions using a compact state made from:

- `Magnitude`
- `Depth`
- `WaveRisk`
- `Confidence`
- `Time`

At each step, the agent chooses one of four actions:

- `Hold / Monitor`
- `Watch / Advisory`
- `Warning`
- `Cancel Alert`

The goal is to balance caution and speed:

- issue strong alerts when risk is high and confidence is strong
- avoid false alarms when the signal is weak or misleading
- reduce delayed warnings and missed alerts

## Main Experience

The easiest way to explore the project is the Streamlit dashboard:

```bash
streamlit run app.py --server.address 127.0.0.1 --server.port 8502
```

The dashboard includes:

- named demo scenarios
- floating step controls with `Previous`, `Next`, `Auto Run`, and `Reset`
- current state cards
- selected action and reward panels
- explanation text with RL math details (collapsed by default)
- decision history
- final outcome summary
- visual charts for reward trend, signal profile, and action mix
- live training monitor with non-technical quality/safety charts

## Demo Scenarios

The dashboard includes deterministic scenarios for presentation and review:

- `High-Risk Confirmed Tsunami`
- `Uncertain Moderate-Risk Case`
- `False Alarm Case`
- `Delayed Response Case`

These scenarios are useful for explaining how the system reacts under different levels of risk and confidence.

## Current Project Structure

The repository now keeps root files minimal and groups runnable scripts under `tools/`:

- `app.py`: main Streamlit entrypoint.
- `src/`: RL environment, agent, trainer/evaluator, and dashboard code.
- `src/core/`: shared runtime factory + protocol interfaces for extensibility.
- `tools/cli/`: training/evaluation/simulation command-line scripts.
- `tools/launch/`: dashboard launch helpers.
- `tools/apps/`: optional standalone Streamlit demos.
- `tests/`: automated tests.
- `docs/ARCHITECTURE.md`: extension and OOP design guide.

## Installation For Other Users

If you are sharing this project with teammates or evaluators, these steps are enough to run it on a fresh machine.

### 1. Prerequisites

Install the following first:

- Python 3.10 or newer
- `pip`
- Git

If you are using VS Code, it is recommended to also install:

- Python extension
- Jupyter extension if you want to open notebooks

### 2. Clone The Repository

```bash
git clone https://github.com/jatinder3762/CSCN-8020-FinalProject-TsunamiAgent.git
cd CSCN-8020-FinalProject-TsunamiAgent
```

If your local folder name is different, just `cd` into that project folder instead.

### 3. Create A Virtual Environment

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

On Windows Command Prompt:

```bat
python -m venv .venv
.venv\Scripts\activate
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Verify The Setup

Run the tests:

```bash
pytest -q
```

If setup is correct, tests should pass.

Recommended quick validation of entrypoints:

```bash
python tools/cli/run_training.py --help
python tools/cli/run_evaluation.py --help
python tools/cli/main.py --help
python tools/launch/start_dashboard.py --help
```

### 6. Quick Start (For Demo/Review)

Run the dashboard:

```bash
streamlit run app.py --server.address 127.0.0.1 --server.port 8502
```

If `8502` is busy, run:

```bash
python tools/launch/start_dashboard.py --port 8503
```

Then open the URL shown in terminal (for example `http://localhost:8503`).

## Running The Project Locally

### Streamlit Dashboard

Start the main dashboard:

```bash
streamlit run app.py --server.address 127.0.0.1 --server.port 8502
```

This is the primary demo command.
The repo includes `.streamlit/config.toml`, so the default project port is `8502`.

If you prefer the helper launcher:

```bash
python tools/launch/start_dashboard.py
```

Run on a different port (for example when `8502` is already in use):

```bash
python tools/launch/start_dashboard.py --port 8503
```

Windows batch launcher:

```bat
tools\launch\start_streamlit.bat
```

### Standalone MDP Visual Simulator

Run the proposal-style interactive MDP visualizer:

```bash
streamlit run tools/apps/mdp_visualizer_app.py
```

### Train The Agent

Run a local training session:

```bash
python tools/cli/run_training.py --episodes 2500 --seed 42
```

This trains the Q-learning agent and writes outputs such as the learned Q-table and training logs.

### Evaluate A Trained Agent

Evaluate the saved model:

```bash
python tools/cli/run_evaluation.py --episodes 400 --seed 42
```

You can also provide a specific model path:

```bash
python tools/cli/run_evaluation.py --episodes 400 --seed 42 --model-path outputs/models/q_table.npy
```

### Run Training And Evaluation Together

```bash
python tools/cli/main.py --mode both --train-episodes 2500 --eval-episodes 400 --seed 42
```

You can also run only one mode:

```bash
python tools/cli/main.py --mode train --train-episodes 2500 --seed 42
python tools/cli/main.py --mode evaluate --eval-episodes 400 --seed 42
```

## Using The Dashboard

When the dashboard opens:

1. Select `Scenario Demo` to walk through predefined tsunami cases.
2. Choose a scenario from the sidebar.
3. Move through the episode using the floating controls:
   - `Previous`
   - `Next`
   - `Auto Run`
   - `Reset`
4. Review the explanation and visual panels as the scenario evolves.

You can also switch to `Live Training` from the sidebar to watch the RL agent train inside Streamlit.

## Output Files

After training or evaluation, the project writes outputs under `outputs/`, including:

- trained Q-table files
- logs and summaries
- live training session CSV output

Typical locations include:

- `outputs/models/q_table.npy`
- `outputs/logs/`

## Notes

- The Streamlit dashboard is the main presentation interface.
- GIF or video output is not required to use or present this project.
- Simulation-related scripts may still exist in the repository, but the dashboard and RL workflow do not depend on video generation.
- For maintainable extension guidance, see `docs/ARCHITECTURE.md`.

## Troubleshooting

### `streamlit` command not found

Activate your virtual environment first, then run:

```bash
pip install -r requirements.txt
```

You can also try:

```bash
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8502
```

### Tests fail because packages are missing

Make sure the virtual environment is active and reinstall dependencies:

```bash
pip install -r requirements.txt
pytest -q
```

### Model file not found during evaluation

Run training first:

```bash
python tools/cli/run_training.py --episodes 2500 --seed 42
```

Then run evaluation again.

### Windows permission error when writing to `outputs/`

If you see a permission error while saving logs/models:

1. Run terminal as Administrator, or
2. Move the project to a user-writable folder (for example under `C:\Users\<you>\Documents\`), then run again.

## Summary

This project is a local, ready-to-run tsunami decision support demo that shows how reinforcement learning can be used to explore alert strategies. The Streamlit dashboard makes the results easier to present, while the CLI scripts make it easy to train and evaluate the model on your own machine.
