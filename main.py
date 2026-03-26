"""Main entry point for training/evaluation workflows."""

from __future__ import annotations

import argparse
import json

from config import ProjectConfig
from src.agent import QLearningAgent
from src.environment import TsunamiAlertEnvironment
from src.evaluator import Evaluator
from src.trainer import Trainer
from src.utils import SeedManager


class Application:
    """Coordinates train/evaluate/both execution modes."""

    def __init__(self) -> None:
        """Creates command-line parser."""
        self.parser = self._build_parser()

    def _build_parser(self) -> argparse.ArgumentParser:
        """Builds argument parser for the main application."""
        parser = argparse.ArgumentParser(description="RL-Based Tsunami Alert Decision System")
        parser.add_argument(
            "--mode",
            choices=["train", "evaluate", "both"],
            default="both",
            help="Select workflow mode.",
        )
        parser.add_argument("--train-episodes", type=int, default=2500, help="Training episodes.")
        parser.add_argument("--eval-episodes", type=int, default=400, help="Evaluation episodes.")
        parser.add_argument("--seed", type=int, default=42, help="Random seed.")
        return parser

    def run(self) -> None:
        """Executes the selected mode and prints summaries."""
        args = self.parser.parse_args()
        SeedManager.set_seed(args.seed)

        if args.mode == "train":
            summary = self._run_training(args.train_episodes, args.seed)
            print("Training completed.")
            print(json.dumps(summary, indent=2))
            return

        if args.mode == "evaluate":
            summary = self._run_evaluation(args.eval_episodes, args.seed)
            print("Evaluation completed.")
            print(json.dumps(summary, indent=2))
            return

        training_summary = self._run_training(args.train_episodes, args.seed)
        evaluation_summary = self._run_evaluation(args.eval_episodes, args.seed)

        print("Training + Evaluation completed.")
        print("Training summary:")
        print(json.dumps(training_summary, indent=2))
        print("Evaluation summary:")
        print(json.dumps(evaluation_summary, indent=2))

    def _run_training(self, train_episodes: int, seed: int) -> dict:
        """Runs a fresh training session."""
        config = ProjectConfig(training_episodes=train_episodes, random_seed=seed)
        environment = TsunamiAlertEnvironment(config=config, seed=seed)
        agent = QLearningAgent(
            state_size=config.state_size,
            action_size=config.action_size,
            alpha=config.alpha,
            gamma=config.gamma,
            epsilon=config.epsilon,
            epsilon_decay=config.epsilon_decay,
            min_epsilon=config.min_epsilon,
            seed=seed,
        )
        trainer = Trainer(config=config, environment=environment, agent=agent)
        return trainer.train()

    def _run_evaluation(self, eval_episodes: int, seed: int) -> dict:
        """Runs evaluation using saved model weights."""
        config = ProjectConfig(evaluation_episodes=eval_episodes, random_seed=seed)
        environment = TsunamiAlertEnvironment(config=config, seed=seed)
        agent = QLearningAgent(
            state_size=config.state_size,
            action_size=config.action_size,
            alpha=config.alpha,
            gamma=config.gamma,
            epsilon=0.0,
            epsilon_decay=1.0,
            min_epsilon=0.0,
            seed=seed,
        )
        agent.load_q_table(config.models_dir / "q_table.npy")
        evaluator = Evaluator(config=config, environment=environment, agent=agent)
        return evaluator.evaluate(episodes=eval_episodes)


if __name__ == "__main__":
    Application().run()
