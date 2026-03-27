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

- `Wait`
- `Verify`
- `Regional Alert`
- `Full Alert`

The goal is to balance caution and speed:

- issue strong alerts when risk is high and confidence is strong
- avoid false alarms when the signal is weak or misleading
- reduce delayed warnings and missed alerts

## Main Experience

The easiest way to explore the project is the Streamlit dashboard:

```bash
streamlit run app.py
```

The dashboard includes:

- named demo scenarios
- step-by-step navigation with `Previous`, `Next`, `Auto Run`, and a slider
- current state cards
- selected action and reward panels
- explanation text for each decision
- decision history
- progress tracking
- final outcome summary
- visual charts for reward trend, signal profile, and action mix

## Demo Scenarios

The dashboard includes deterministic scenarios for presentation and review:

- `High-Risk Confirmed Tsunami`
- `Uncertain Moderate-Risk Case`
- `False Alarm Case`
- `Delayed Response Case`

These scenarios are useful for explaining how the system reacts under different levels of risk and confidence.

## Local Setup

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

If the environment is set up correctly, the test suite should pass.

## Running The Project Locally

### Streamlit Dashboard

Start the main dashboard:

```bash
streamlit run app.py
```

This is the primary demo command.

If you prefer the helper launcher:

```bash
python start_dashboard.py
```

### Train The Agent

Run a local training session:

```bash
python run_training.py --episodes 2500 --seed 42
```

This trains the Q-learning agent and writes outputs such as the learned Q-table and training logs.

### Evaluate A Trained Agent

Evaluate the saved model:

```bash
python run_evaluation.py --episodes 400 --seed 42
```

You can also provide a specific model path:

```bash
python run_evaluation.py --episodes 400 --seed 42 --model-path outputs/models/q_table.npy
```

### Run Training And Evaluation Together

```bash
python main.py --mode both --train-episodes 2500 --eval-episodes 400 --seed 42
```

You can also run only one mode:

```bash
python main.py --mode train --train-episodes 2500 --seed 42
python main.py --mode evaluate --eval-episodes 400 --seed 42
```

## Using The Dashboard

When the dashboard opens:

1. Select `Scenario Demo` to walk through predefined tsunami cases.
2. Choose a scenario from the sidebar.
3. Move through the episode using:
   - `Previous`
   - `Next`
   - `Auto Run`
   - `Reset`
   - `Step Slider`
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

## Troubleshooting

### `streamlit` command not found

Activate your virtual environment first, then run:

```bash
pip install -r requirements.txt
```

You can also try:

```bash
python -m streamlit run app.py
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
python run_training.py --episodes 2500 --seed 42
```

Then run evaluation again.

## Summary

This project is a local, ready-to-run tsunami decision support demo that shows how reinforcement learning can be used to explore alert strategies. The Streamlit dashboard makes the results easier to present, while the CLI scripts make it easy to train and evaluate the model on your own machine.
