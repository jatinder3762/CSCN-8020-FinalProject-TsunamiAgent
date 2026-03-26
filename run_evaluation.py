"""Command-line script for evaluating a trained tsunami Q-learning agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import ProjectConfig
from src.agent import QLearningAgent
from src.environment import TsunamiAlertEnvironment
from src.evaluator import Evaluator
from src.utils import SeedManager


class EvaluationRunner:
    """Builds dependencies and executes policy evaluation."""

    def __init__(self) -> None:
        """Initializes parser and runtime options."""
        self.parser = self._build_parser()

    def _build_parser(self) -> argparse.ArgumentParser:
        """Creates command-line arguments for evaluation."""
        parser = argparse.ArgumentParser(description="Evaluate Q-learning tsunami alert agent.")
        parser.add_argument("--episodes", type=int, default=400, help="Number of evaluation episodes.")
        parser.add_argument("--seed", type=int, default=42, help="Random seed.")
        parser.add_argument(
            "--model-path",
            type=str,
            default="",
            help="Path to saved q_table.npy (defaults to outputs/models/q_table.npy).",
        )
        return parser

    def run(self) -> None:
        """Runs evaluation with a saved Q-table."""
        args = self.parser.parse_args()
        config = ProjectConfig(evaluation_episodes=args.episodes, random_seed=args.seed)
        SeedManager.set_seed(config.random_seed)

        environment = TsunamiAlertEnvironment(config=config, seed=config.random_seed)
        agent = QLearningAgent(
            state_size=config.state_size,
            action_size=config.action_size,
            alpha=config.alpha,
            gamma=config.gamma,
            epsilon=0.0,
            epsilon_decay=1.0,
            min_epsilon=0.0,
            seed=config.random_seed,
        )

        model_path = Path(args.model_path).resolve() if args.model_path else (config.models_dir / "q_table.npy")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        agent.load_q_table(model_path)
        evaluator = Evaluator(config=config, environment=environment, agent=agent)

        summary = evaluator.evaluate(episodes=args.episodes)
        print("Evaluation completed.")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    EvaluationRunner().run()
