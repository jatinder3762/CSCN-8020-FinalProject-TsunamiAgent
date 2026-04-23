"""Command-line script for evaluating a trained tsunami Q-learning agent."""

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
            "--disable-safe-override",
            action="store_true",
            help="Disable safe hybrid override and deploy pure greedy RL actions.",
        )
        parser.add_argument(
            "--override-delta",
            type=float,
            default=None,
            help="Override margin threshold delta used by safe hybrid deployment.",
        )
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
        runtime = RuntimeFactory.build_evaluation_bundle(evaluation_episodes=args.episodes, seed=args.seed)
        SeedManager.set_seed(runtime.config.random_seed)
        if bool(args.disable_safe_override):
            runtime.config.use_safe_override = False
        if args.override_delta is not None:
            runtime.config.safe_override_delta = float(args.override_delta)

        model_path = Path(args.model_path).resolve() if args.model_path else (runtime.config.models_dir / "q_table.npy")
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        runtime.agent.load_q_table(model_path)
        evaluator = Evaluator(config=runtime.config, environment=runtime.environment, agent=runtime.agent)

        summary = evaluator.evaluate(episodes=args.episodes)
        print("Evaluation completed.")
        self._print_mdp_details(summary)
        print(json.dumps(summary, indent=2))

    def _print_mdp_details(self, summary: dict[str, object]) -> None:
        """Prints a concise MDP parameter view for easy presentation."""
        mdp = summary.get("mdp_parameters")
        if not isinstance(mdp, dict):
            return

        print("\nMDP Details:")
        print(f"- alpha: {mdp.get('alpha')}")
        print(f"- gamma: {mdp.get('gamma')}")
        print(f"- state_space_size: {mdp.get('state_space_size')}")
        print(f"- action_space_size: {mdp.get('action_space_size')}")

        rewards = mdp.get("reward_inputs")
        if isinstance(rewards, dict):
            print("- reward_inputs:")
            for name, value in rewards.items():
                print(f"  - {name}: {value}")
        deployment = mdp.get("deployment_policy")
        if isinstance(deployment, dict):
            print("- deployment_policy:")
            for name, value in deployment.items():
                print(f"  - {name}: {value}")


if __name__ == "__main__":
    EvaluationRunner().run()
