"""Command-line script for training the tsunami Q-learning agent."""

from __future__ import annotations

import argparse
import json

from config import ProjectConfig
from src.agent import QLearningAgent
from src.environment import TsunamiAlertEnvironment
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
        config = ProjectConfig(training_episodes=args.episodes, random_seed=args.seed)
        SeedManager.set_seed(config.random_seed)

        environment = TsunamiAlertEnvironment(config=config, seed=config.random_seed)
        agent = QLearningAgent(
            state_size=config.state_size,
            action_size=config.action_size,
            alpha=config.alpha,
            gamma=config.gamma,
            epsilon=config.epsilon,
            epsilon_decay=config.epsilon_decay,
            min_epsilon=config.min_epsilon,
            seed=config.random_seed,
        )
        trainer = Trainer(config=config, environment=environment, agent=agent)

        summary = trainer.train()
        print("Training completed.")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    TrainingRunner().run()
