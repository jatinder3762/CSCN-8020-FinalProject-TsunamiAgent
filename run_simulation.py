"""CLI script for generating movie-style tsunami alert simulations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import ProjectConfig
from src.agent import QLearningAgent
from src.environment import TsunamiAlertEnvironment
from src.simulator import TsunamiCinematicSimulator


class SimulationRunner:
    """Loads trained model and renders single or parallel simulation animations."""

    def __init__(self) -> None:
        """Initializes command-line parser."""
        self.parser = self._build_parser()

    def _build_parser(self) -> argparse.ArgumentParser:
        """Defines command-line options for simulation rendering."""
        parser = argparse.ArgumentParser(description="Render tsunami simulation animations.")
        parser.add_argument("--seed", type=int, default=42, help="Random seed for episode generation.")
        parser.add_argument("--mode", choices=["single", "parallel"], default="single", help="Simulation mode.")
        parser.add_argument("--max-trials", type=int, default=120, help="Search attempts to find a showcase episode.")
        parser.add_argument("--parallel-episodes", type=int, default=9, help="Episodes shown in side-by-side mode.")
        parser.add_argument("--fps", type=int, default=20, help="Animation frames per second.")
        parser.add_argument("--frames-per-step", type=int, default=45, help="Animation frames used per decision step.")
        parser.add_argument(
            "--output",
            type=str,
            default="",
            help="Output animation path. If omitted, mode-specific default is used.",
        )
        parser.add_argument(
            "--model-path",
            type=str,
            default="",
            help="Path to trained q_table.npy (defaults to outputs/models/q_table.npy).",
        )
        parser.add_argument("--show", action="store_true", help="Display interactive preview window.")
        return parser

    def run(self) -> None:
        """Executes simulation pipeline and prints summary."""
        args = self.parser.parse_args()
        config = ProjectConfig(random_seed=args.seed)

        environment = TsunamiAlertEnvironment(config=config, seed=args.seed)
        agent = QLearningAgent(
            state_size=config.state_size,
            action_size=config.action_size,
            alpha=config.alpha,
            gamma=config.gamma,
            epsilon=0.0,
            epsilon_decay=1.0,
            min_epsilon=0.0,
            seed=args.seed,
        )

        model_path = Path(args.model_path).resolve() if args.model_path else (config.models_dir / "q_table.npy")
        if not model_path.exists():
            raise FileNotFoundError(f"Trained model not found: {model_path}. Run training first.")

        agent.load_q_table(model_path)
        simulator = TsunamiCinematicSimulator(
            config=config,
            environment=environment,
            agent=agent,
            fps=args.fps,
            frames_per_step=args.frames_per_step,
        )

        output_path = self._resolve_output_path(config, args.mode, args.output)

        if args.mode == "parallel":
            summary = simulator.run_parallel(
                output_path=output_path,
                episodes=args.parallel_episodes,
                show_preview=args.show,
            )
        else:
            summary = simulator.run(
                output_path=output_path,
                max_trials=args.max_trials,
                show_preview=args.show,
            )

        print("Simulation rendered successfully.")
        print(json.dumps(summary, indent=2))

    @staticmethod
    def _resolve_output_path(config: ProjectConfig, mode: str, output: str) -> Path:
        """Resolves output path using mode-specific defaults when not provided."""
        if output:
            return Path(output)
        if mode == "parallel":
            return config.plots_dir / "tsunami_parallel_episodes.gif"
        return config.plots_dir / "tsunami_cinematic_simulation.gif"


if __name__ == "__main__":
    SimulationRunner().run()
