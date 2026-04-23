"""CLI script for generating movie-style tsunami alert simulations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ProjectConfig
from src.core import RuntimeFactory
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
        runtime = RuntimeFactory.build_evaluation_bundle(evaluation_episodes=1, seed=args.seed)
        config = runtime.config
        if bool(args.disable_safe_override):
            config.use_safe_override = False
        if args.override_delta is not None:
            config.safe_override_delta = float(args.override_delta)

        model_path = Path(args.model_path).resolve() if args.model_path else (config.models_dir / "q_table.npy")
        if not model_path.exists():
            raise FileNotFoundError(f"Trained model not found: {model_path}. Run training first.")

        runtime.agent.load_q_table(model_path)
        simulator = TsunamiCinematicSimulator(
            config=config,
            environment=runtime.environment,
            agent=runtime.agent,
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
