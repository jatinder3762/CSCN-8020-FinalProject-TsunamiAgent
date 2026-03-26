"""Matplotlib plotting utilities for training and evaluation outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from config import ProjectConfig
from src.utils import MathUtils


class PlotManager:
    """Generates and saves all required plots."""

    def __init__(self, config: ProjectConfig) -> None:
        """Initializes plotting manager with output directory config."""
        self.config = config

    def plot_training_metrics(
        self,
        training_records: list[dict[str, float | int | bool | str]],
        action_counts: dict[int, int],
        epsilon_history: list[float],
    ) -> None:
        """Saves training reward, moving average, steps, epsilon, and action plots."""
        if not training_records:
            return

        training_df = pd.DataFrame(training_records)
        episodes = training_df["episode"].to_list()
        rewards = training_df["total_reward"].astype(float).to_list()
        steps = training_df["steps"].astype(float).to_list()
        reward_ma = MathUtils.moving_average(rewards, self.config.moving_average_window)

        self._plot_line(
            x=episodes,
            y=rewards,
            title="Training Reward vs Episode",
            xlabel="Episode",
            ylabel="Total Reward",
            output_path=self.config.plots_dir / "training_reward_vs_episode.png",
            color="#1f77b4",
        )

        self._plot_line(
            x=episodes,
            y=reward_ma,
            title=f"Training Moving Average Reward (window={self.config.moving_average_window})",
            xlabel="Episode",
            ylabel="Moving Avg Reward",
            output_path=self.config.plots_dir / "training_moving_average_reward.png",
            color="#ff7f0e",
        )

        self._plot_line(
            x=episodes,
            y=steps,
            title="Training Steps vs Episode",
            xlabel="Episode",
            ylabel="Steps",
            output_path=self.config.plots_dir / "training_steps_vs_episode.png",
            color="#2ca02c",
        )

        self._plot_line(
            x=episodes,
            y=epsilon_history,
            title="Epsilon Decay",
            xlabel="Episode",
            ylabel="Epsilon",
            output_path=self.config.plots_dir / "training_epsilon_decay.png",
            color="#d62728",
        )

        action_labels = [self.config.action_names[action] for action in sorted(action_counts)]
        action_values = [action_counts[action] for action in sorted(action_counts)]
        self._plot_bar(
            labels=action_labels,
            values=action_values,
            title="Training Action Distribution",
            ylabel="Count",
            output_path=self.config.plots_dir / "training_action_distribution.png",
            color="#9467bd",
        )

    def plot_evaluation_metrics(
        self,
        evaluation_records: list[dict[str, float | int | bool | str]],
        summary_metrics: dict[str, float],
        action_counts: dict[int, int],
    ) -> None:
        """Saves evaluation reward trend and summary/action distribution bar charts."""
        if not evaluation_records:
            return

        evaluation_df = pd.DataFrame(evaluation_records)
        episodes = evaluation_df["episode"].to_list()
        rewards = evaluation_df["total_reward"].astype(float).to_list()

        self._plot_line(
            x=episodes,
            y=rewards,
            title="Evaluation Reward vs Episode",
            xlabel="Episode",
            ylabel="Total Reward",
            output_path=self.config.plots_dir / "evaluation_reward_vs_episode.png",
            color="#17becf",
        )

        metric_labels = [
            "AvgReward",
            "CorrectRate",
            "FalseRate",
            "MissedRate",
        ]
        metric_values = [
            summary_metrics.get("average_reward", 0.0),
            summary_metrics.get("correct_alert_rate", 0.0) * 100.0,
            summary_metrics.get("false_alert_rate", 0.0) * 100.0,
            summary_metrics.get("missed_alert_rate", 0.0) * 100.0,
        ]

        self._plot_bar(
            labels=metric_labels,
            values=metric_values,
            title="Evaluation Summary Metrics",
            ylabel="Value (Rate metrics in %)",
            output_path=self.config.plots_dir / "evaluation_summary_bar.png",
            color="#8c564b",
        )

        action_labels = [self.config.action_names[action] for action in sorted(action_counts)]
        action_values = [action_counts[action] for action in sorted(action_counts)]
        self._plot_bar(
            labels=action_labels,
            values=action_values,
            title="Evaluation Action Distribution",
            ylabel="Count",
            output_path=self.config.plots_dir / "evaluation_action_distribution.png",
            color="#bcbd22",
        )

    def _plot_line(
        self,
        x: list[int],
        y: list[float],
        title: str,
        xlabel: str,
        ylabel: str,
        output_path: Path,
        color: str,
    ) -> None:
        """Draws and saves a single line plot."""
        plt.figure(figsize=(10, 5))
        plt.plot(x, y, color=color, linewidth=1.6)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=140)
        plt.close()

    def _plot_bar(
        self,
        labels: list[str],
        values: list[float],
        title: str,
        ylabel: str,
        output_path: Path,
        color: str,
    ) -> None:
        """Draws and saves a single bar plot."""
        plt.figure(figsize=(10, 5))
        plt.bar(labels, values, color=color)
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(output_path, dpi=140)
        plt.close()
