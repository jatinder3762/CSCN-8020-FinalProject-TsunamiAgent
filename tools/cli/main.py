"""Main entry point for training/evaluation workflows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core import RuntimeFactory
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
        runtime = RuntimeFactory.build_training_bundle(training_episodes=train_episodes, seed=seed)
        trainer = Trainer(config=runtime.config, environment=runtime.environment, agent=runtime.agent)
        return trainer.train()

    def _run_evaluation(self, eval_episodes: int, seed: int) -> dict:
        """Runs evaluation using saved model weights."""
        runtime = RuntimeFactory.build_evaluation_bundle(evaluation_episodes=eval_episodes, seed=seed)
        runtime.agent.load_q_table(runtime.config.models_dir / "q_table.npy")
        evaluator = Evaluator(config=runtime.config, environment=runtime.environment, agent=runtime.agent)
        return evaluator.evaluate(episodes=eval_episodes)


if __name__ == "__main__":
    Application().run()
