# Architecture Guide

## Goal
This project is organized to keep RL domain logic, orchestration flows, and UI concerns separate so we can extend safely.

## Current Folder Structure

`src/`
- `core/`: extension contracts and runtime assembly.
- `agent.py`: tabular Q-learning implementation.
- `environment.py`: tsunami MDP environment and reward logic.
- `trainer.py`: training workflow and output persistence.
- `evaluator.py`: policy evaluation workflow and metrics.
- `plotting.py`: offline matplotlib output generation.
- `presentation/`: Streamlit UI and deterministic scenario playback.
- compatibility wrapper modules (`dashboard.py`, `components.py`, etc.) for stable imports.

`tests/`
- behavior tests for environment and app wiring.

`tools/cli/`
- `run_training.py`, `run_evaluation.py`, `main.py`, `run_simulation.py`
- CLI entrypoints using shared runtime factory.

`tools/launch/`
- dashboard launch helpers (`start_dashboard.py`, `start_streamlit.bat`).

`tools/apps/`
- optional standalone visualizers (for example `mdp_visualizer_app.py`).

## OOP Extension Points

`src/core/interfaces.py`
- `AgentProtocol`: contract for any trainable policy agent.
- `EnvironmentProtocol`: contract for any environment backend.

`src/core/runtime_factory.py`
- `RuntimeFactory`: centralized object creation for consistent dependency wiring.
- `RuntimeBundle`: immutable container for `(config, environment, agent)`.

This avoids repeated setup logic and makes it easier to:
- swap Q-learning for SARSA or PPO agents.
- replace the simulator environment while keeping trainer/evaluator unchanged.

## How To Extend

### Add a New Agent (example: SARSA)
1. Create new class in `src/` implementing `AgentProtocol`.
2. Add a factory method in `RuntimeFactory` for that agent.
3. Use the new factory method in CLI/app mode selector.

### Add a New Environment Variant
1. Create new environment class implementing `EnvironmentProtocol`.
2. Keep return signature identical to current `step()` contract.
3. Expose selection in `RuntimeFactory`.

### Keep UI Independent
- UI should consume trainer/evaluator outputs and not recreate learning logic.
- Presentation scenarios in `src/presentation/scenarios.py` can evolve without touching training backend.

## Design Principles Followed
- Single Responsibility: environment, agent, trainer, evaluator each own one concern.
- Dependency Inversion (lightweight via Protocols): higher-level flows depend on interfaces, not concrete classes.
- Open/Closed: new agents and environments can be added via factory without rewriting entrypoints.
