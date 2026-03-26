"""Training pipeline for tabular Q-learning tsunami alert decisions."""

from __future__ import annotations

from typing import Any

from config import ProjectConfig
from src.agent import QLearningAgent
from src.environment import TsunamiAlertEnvironment
from src.plotting import PlotManager
from src.utils import LabelFormatter, MathUtils, OutputManager


class Trainer:
    """Runs full training, logging, plotting, and model persistence."""

    def __init__(self, config: ProjectConfig, environment: TsunamiAlertEnvironment, agent: QLearningAgent) -> None:
        """Initializes trainer dependencies and storage buffers."""
        self.config = config
        self.environment = environment
        self.agent = agent
        self.plot_manager = PlotManager(config)

        self.training_records: list[dict[str, Any]] = []
        self.epsilon_history: list[float] = []
        self.action_counts: dict[int, int] = {action: 0 for action in self.config.action_names}
        self.best_episode_trace: list[dict[str, Any]] = []
        self.best_episode_reward: float = float("-inf")

    def train(self) -> dict[str, Any]:
        """Runs training over configured episodes and returns summary metrics."""
        for episode in range(1, self.config.training_episodes + 1):
            episode_record, episode_trace = self.run_episode(episode)
            self.training_records.append(episode_record)

            if float(episode_record["total_reward"]) > self.best_episode_reward:
                self.best_episode_reward = float(episode_record["total_reward"])
                self.best_episode_trace = episode_trace

            self.agent.decay_epsilon()
            self.epsilon_history.append(self.agent.epsilon)
            self.training_records[-1]["epsilon"] = self.agent.epsilon

        summary = self._build_training_summary()
        self.save_outputs(summary)
        return summary

    def run_episode(self, episode: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Executes one training episode and returns episode metrics and trace."""
        state_idx = self.environment.reset()
        done = False
        total_reward = 0.0
        steps = 0

        final_info: dict[str, Any] = {}
        trace: list[dict[str, Any]] = []

        while not done and steps < self.config.max_steps_per_episode:
            state_tuple = self.environment.index_to_state(state_idx)
            action = self.agent.choose_action(state_idx=state_idx, training=True)
            next_state_idx, reward, done, info = self.environment.step(action)

            self.agent.update(state_idx, action, reward, next_state_idx, done)

            total_reward += reward
            steps += 1
            self.action_counts[action] += 1

            trace.append(
                {
                    "step": steps,
                    "state_index": state_idx,
                    "state_text": LabelFormatter.state_name(state_tuple, self.config),
                    "action": action,
                    "action_text": LabelFormatter.action_name(action, self.config),
                    "reward": reward,
                    "next_state_index": next_state_idx,
                    "done": done,
                    "actual_risk_level": info.get("actual_risk_level", "Unknown"),
                }
            )

            state_idx = next_state_idx
            final_info = info

        record = {
            "episode": episode,
            "total_reward": total_reward,
            "steps": steps,
            "epsilon": self.agent.epsilon,
            "correct_alert": bool(final_info.get("alert_correct", False)),
            "false_alert": bool(final_info.get("false_alert", False)),
            "missed_alert": bool(final_info.get("missed_alert", False)),
            "actual_risk_level": final_info.get("actual_risk_level", "Unknown"),
            "final_action": final_info.get("action_meaning", "Unknown"),
        }
        return record, trace

    def save_outputs(self, summary: dict[str, Any]) -> None:
        """Persists logs, summary reports, best trace, model, and plots."""
        OutputManager.ensure_project_directories(self.config)

        training_log_path = self.config.logs_dir / "training_history.csv"
        training_summary_path = self.config.logs_dir / "training_summary.json"
        best_trace_path = self.config.logs_dir / "best_episode_trace.json"
        q_table_path = self.config.models_dir / "q_table.npy"
        model_meta_path = self.config.models_dir / "q_table_metadata.json"

        OutputManager.save_records_csv(self.training_records, training_log_path)
        OutputManager.save_json(summary, training_summary_path)
        OutputManager.save_json({"best_episode_trace": self.best_episode_trace}, best_trace_path)

        self.agent.save_q_table(q_table_path)
        OutputManager.save_json(
            {
                "state_size": self.config.state_size,
                "action_size": self.config.action_size,
                "alpha": self.config.alpha,
                "gamma": self.config.gamma,
                "epsilon_final": self.agent.epsilon,
            },
            model_meta_path,
        )

        self.plot_manager.plot_training_metrics(self.training_records, self.action_counts, self.epsilon_history)

    def _build_training_summary(self) -> dict[str, Any]:
        """Builds aggregate training metrics for reporting."""
        episode_count = len(self.training_records)
        total_reward = sum(float(item["total_reward"]) for item in self.training_records)
        total_steps = sum(int(item["steps"]) for item in self.training_records)

        correct_count = sum(1 for item in self.training_records if bool(item["correct_alert"]))
        false_count = sum(1 for item in self.training_records if bool(item["false_alert"]))
        missed_count = sum(1 for item in self.training_records if bool(item["missed_alert"]))

        average_reward = MathUtils.safe_rate(total_reward, episode_count)
        average_steps = MathUtils.safe_rate(total_steps, episode_count)

        return {
            "project_title": self.config.project_title,
            "episodes": episode_count,
            "average_reward": average_reward,
            "average_steps": average_steps,
            "correct_alert_rate": MathUtils.safe_rate(correct_count, episode_count),
            "false_alert_rate": MathUtils.safe_rate(false_count, episode_count),
            "missed_alert_rate": MathUtils.safe_rate(missed_count, episode_count),
            "final_epsilon": self.agent.epsilon,
            "best_episode_reward": self.best_episode_reward,
            "action_distribution": {
                self.config.action_names[action]: count for action, count in self.action_counts.items()
            },
        }
