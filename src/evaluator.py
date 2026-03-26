"""Evaluation pipeline for greedy-policy tsunami alert decisions."""

from __future__ import annotations

from typing import Any

from config import ProjectConfig
from src.agent import QLearningAgent
from src.environment import TsunamiAlertEnvironment
from src.plotting import PlotManager
from src.utils import MathUtils, OutputManager


class Evaluator:
    """Runs policy evaluation and saves metrics/plots."""

    def __init__(self, config: ProjectConfig, environment: TsunamiAlertEnvironment, agent: QLearningAgent) -> None:
        """Initializes evaluator dependencies."""
        self.config = config
        self.environment = environment
        self.agent = agent
        self.plot_manager = PlotManager(config)

        self.evaluation_records: list[dict[str, Any]] = []
        self.action_counts: dict[int, int] = {action: 0 for action in self.config.action_names}

    def evaluate(self, episodes: int | None = None) -> dict[str, Any]:
        """Runs greedy evaluation for the given number of episodes."""
        episode_count = episodes if episodes is not None else self.config.evaluation_episodes
        self.evaluation_records = []
        self.action_counts = {action: 0 for action in self.config.action_names}

        for episode in range(1, episode_count + 1):
            episode_record = self.run_greedy_episode(episode)
            self.evaluation_records.append(episode_record)

        summary = self.summarize_metrics()
        self._save_outputs(summary)
        return summary

    def run_greedy_episode(self, episode: int) -> dict[str, Any]:
        """Runs one evaluation episode with greedy (non-exploratory) action selection."""
        state_idx = self.environment.reset()
        done = False
        steps = 0
        total_reward = 0.0
        final_info: dict[str, Any] = {}

        while not done and steps < self.config.max_steps_per_episode:
            action = self.agent.choose_action(state_idx=state_idx, training=False)
            next_state_idx, reward, done, info = self.environment.step(action)

            self.action_counts[action] += 1
            total_reward += reward
            steps += 1
            state_idx = next_state_idx
            final_info = info

        return {
            "episode": episode,
            "total_reward": total_reward,
            "steps": steps,
            "correct_alert": bool(final_info.get("alert_correct", False)),
            "false_alert": bool(final_info.get("false_alert", False)),
            "missed_alert": bool(final_info.get("missed_alert", False)),
            "actual_risk_level": final_info.get("actual_risk_level", "Unknown"),
            "final_action": final_info.get("action_meaning", "Unknown"),
        }

    def summarize_metrics(self) -> dict[str, Any]:
        """Computes aggregated evaluation metrics."""
        episode_count = len(self.evaluation_records)
        total_reward = sum(float(item["total_reward"]) for item in self.evaluation_records)
        total_steps = sum(int(item["steps"]) for item in self.evaluation_records)

        correct_count = sum(1 for item in self.evaluation_records if bool(item["correct_alert"]))
        false_count = sum(1 for item in self.evaluation_records if bool(item["false_alert"]))
        missed_count = sum(1 for item in self.evaluation_records if bool(item["missed_alert"]))

        return {
            "project_title": self.config.project_title,
            "episodes": episode_count,
            "average_reward": MathUtils.safe_rate(total_reward, episode_count),
            "average_steps": MathUtils.safe_rate(total_steps, episode_count),
            "correct_alert_rate": MathUtils.safe_rate(correct_count, episode_count),
            "false_alert_rate": MathUtils.safe_rate(false_count, episode_count),
            "missed_alert_rate": MathUtils.safe_rate(missed_count, episode_count),
            "action_distribution": {
                self.config.action_names[action]: count for action, count in self.action_counts.items()
            },
        }

    def _save_outputs(self, summary: dict[str, Any]) -> None:
        """Saves evaluation logs and plots to disk."""
        evaluation_log_path = self.config.logs_dir / "evaluation_history.csv"
        evaluation_summary_path = self.config.logs_dir / "evaluation_summary.json"

        OutputManager.save_records_csv(self.evaluation_records, evaluation_log_path)
        OutputManager.save_json(summary, evaluation_summary_path)
        self.plot_manager.plot_evaluation_metrics(self.evaluation_records, summary, self.action_counts)
