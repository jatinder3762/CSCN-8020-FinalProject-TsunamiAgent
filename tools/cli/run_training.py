"""Command-line script for training the tsunami Q-learning agent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core import RuntimeFactory
from src.trainer import Trainer
from src.utils import SeedManager


class TrainingRunner:
    """Builds dependencies and executes full training flow."""

    def __init__(self) -> None:
        """Initializes runner state."""
        self.parser = self._build_parser()

    def _build_parser(self) -> argparse.ArgumentParser:
        """Creates command-line arguments for training."""
        parser = argparse.ArgumentParser(description="Train Q-learning tsunami alert agent.")
        parser.add_argument("--episodes", type=int, default=2500, help="Number of training episodes.")
        parser.add_argument("--seed", type=int, default=42, help="Random seed.")
        return parser

    def run(self) -> None:
        """Runs the training pipeline."""
        args = self.parser.parse_args()
        runtime = RuntimeFactory.build_training_bundle(training_episodes=args.episodes, seed=args.seed)
        SeedManager.set_seed(runtime.config.random_seed)
        trainer = Trainer(config=runtime.config, environment=runtime.environment, agent=runtime.agent)

        summary = trainer.train()
        print("Training completed.")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    TrainingRunner().run()
