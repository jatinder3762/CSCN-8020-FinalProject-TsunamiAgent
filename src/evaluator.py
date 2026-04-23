"""Evaluation pipeline for greedy-policy tsunami alert decisions."""

from __future__ import annotations

from typing import Any

from config import ProjectConfig
from src.agent import QLearningAgent
from src.environment import TsunamiAlertEnvironment
from src.hybrid_policy import SafeHybridOverride
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
        self.override_count: int = 0
        self.decision_count: int = 0
        self.margin_sum: float = 0.0

    def evaluate(self, episodes: int | None = None) -> dict[str, Any]:
        """Runs greedy evaluation for the given number of episodes."""
        episode_count = episodes if episodes is not None else self.config.evaluation_episodes
        self.evaluation_records = []
        self.action_counts = {action: 0 for action in self.config.action_names}
        self.override_count = 0
        self.decision_count = 0
        self.margin_sum = 0.0

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
        override_steps = 0
        final_info: dict[str, Any] = {}

        while not done and steps < self.config.max_steps_per_episode:
            valid_actions = self.environment.get_valid_actions()
            if bool(self.config.use_safe_override):
                state_tuple = self.environment.index_to_state(state_idx)
                decision = SafeHybridOverride.select_action(
                    state_idx=state_idx,
                    state=state_tuple,
                    current_alert_level=int(self.environment.current_alert_level),
                    valid_actions=valid_actions,
                    q_table=self.agent.q_table,
                    delta=float(self.config.safe_override_delta),
                )
                action = decision.deployed_action
                self.override_count += int(decision.used_override)
                self.margin_sum += float(decision.margin)
                override_steps += int(decision.used_override)
            else:
                action = self.agent.choose_action(
                    state_idx=state_idx,
                    training=False,
                    valid_actions=valid_actions,
                )
            next_state_idx, reward, done, info = self.environment.step(action)

            self.action_counts[action] += 1
            total_reward += reward
            steps += 1
            self.decision_count += 1
            state_idx = next_state_idx
            final_info = info

        return {
            "episode": episode,
            "total_reward": total_reward,
            "steps": steps,
            "override_steps": override_steps,
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
            "safe_override_rate": MathUtils.safe_rate(self.override_count, self.decision_count),
            "average_margin": MathUtils.safe_rate(self.margin_sum, self.decision_count),
            "action_distribution": {
                self.config.action_names[action]: count for action, count in self.action_counts.items()
            },
            "mdp_parameters": {
                "mdp_tuple": "M = (S, A, P, R, gamma)",
                "q_learning_update": (
                    "Q(s_t,a_t) <- Q(s_t,a_t) + alpha * [r_t + gamma * max_a' Q(s_(t+1),a') - Q(s_t,a_t)]"
                ),
                "alpha": self.config.alpha,
                "gamma": self.config.gamma,
                "state_space_size": self.config.state_size,
                "action_space_size": self.config.action_size,
                "deployment_policy": {
                    "type": "safe_hybrid_override" if bool(self.config.use_safe_override) else "greedy_rl_only",
                    "use_safe_override": bool(self.config.use_safe_override),
                    "safe_override_delta": float(self.config.safe_override_delta),
                    "override_steps": int(self.override_count),
                    "decision_steps": int(self.decision_count),
                },
                "reward_inputs": {
                    "reward_correct_full_alert": self.config.reward_correct_full_alert,
                    "reward_correct_regional_alert": self.config.reward_correct_regional_alert,
                    "reward_smart_verify": self.config.reward_smart_verify,
                    "penalty_missed_dangerous_alert": self.config.penalty_missed_dangerous_alert,
                    "penalty_false_full_alert": self.config.penalty_false_full_alert,
                    "penalty_false_regional_alert": self.config.penalty_false_regional_alert,
                    "reward_delay_per_step": self.config.reward_delay_per_step,
                    "penalty_unnecessary_verify": self.config.penalty_unnecessary_verify,
                    "penalty_late_wait_in_risk": self.config.penalty_late_wait_in_risk,
                    "reward_partial_regional_on_high": self.config.reward_partial_regional_on_high,
                    "penalty_overreaction_full_on_medium": self.config.penalty_overreaction_full_on_medium,
                    "reward_safe_no_alert_low_risk": self.config.reward_safe_no_alert_low_risk,
                },
            },
        }

    def _save_outputs(self, summary: dict[str, Any]) -> None:
        """Saves evaluation logs and plots to disk."""
        evaluation_log_path = self.config.logs_dir / "evaluation_history.csv"
        evaluation_summary_path = self.config.logs_dir / "evaluation_summary.json"

        OutputManager.save_records_csv(self.evaluation_records, evaluation_log_path)
        OutputManager.save_json(summary, evaluation_summary_path)
        self.plot_manager.plot_evaluation_metrics(self.evaluation_records, summary, self.action_counts)
