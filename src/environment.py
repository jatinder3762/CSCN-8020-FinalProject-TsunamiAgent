"""Custom discrete tsunami alert environment built from scratch."""

from __future__ import annotations

from typing import Any

import numpy as np

from config import ProjectConfig
from src.utils import LabelFormatter, StateCodec


class TsunamiAlertEnvironment:
    """Simulates a tsunami warning decision process with discrete states and actions."""

    def __init__(self, config: ProjectConfig, seed: int | None = None) -> None:
        """Initializes environment state, RNG, and hidden episode variables."""
        self.config = config
        self.rng = np.random.default_rng(config.random_seed if seed is None else seed)

        self.state: tuple[int, int, int, int, int] = (0, 0, 0, 0, 0)
        self.hidden_wave_risk: int = 0
        self.hidden_true_risk: int = 0
        self.current_alert_level: int = 0
        self.warning_issued_step: int | None = None
        self.step_count: int = 0
        self.done: bool = False

    def reset(self) -> int:
        """Starts a new episode and returns encoded initial state index."""
        self.state = self._generate_initial_state()
        self.step_count = 0
        self.current_alert_level = 0
        self.warning_issued_step = None
        self.done = False
        return self.state_to_index(self.state)

    def step(self, action: int) -> tuple[int, float, bool, dict[str, Any]]:
        """Performs one environment step and returns standard RL tuple outputs."""
        if self.done:
            raise RuntimeError("Cannot call step() after episode is done. Call reset() first.")
        if action not in self.config.action_names:
            raise ValueError(f"Invalid action {action}. Valid actions: {list(self.config.action_names)}")

        previous_state = self.state
        previous_alert_level = int(self.current_alert_level)
        valid_actions = self.get_valid_actions()
        action_valid = action in valid_actions
        # Invalid actions are converted to a safe hold action, while a penalty is still applied in reward terms.
        effective_action = action if action_valid else 0

        next_state = self._transition_state(effective_action)
        new_alert_level = self._apply_action_to_alert_level(effective_action, previous_alert_level)
        # Capture when warning is first issued so terminal reward can decay with warning delay.
        if action_valid and effective_action == 2 and self.warning_issued_step is None:
            self.warning_issued_step = self.step_count
        terminal = self._is_terminal(next_state)

        reward, alert_correct, false_alert, missed_alert, reward_terms = self._calculate_reward(
            action=action,
            action_valid=action_valid,
            previous_alert_level=previous_alert_level,
            new_alert_level=new_alert_level,
            previous_state=previous_state,
            next_state=next_state,
            terminal=terminal,
        )

        self.state = next_state
        self.current_alert_level = new_alert_level
        self.done = terminal
        self.step_count += 1

        info: dict[str, Any] = {
            "actual_risk_level": self.config.risk_levels[self.hidden_true_risk],
            "action_meaning": LabelFormatter.action_name(action, self.config),
            "action_valid": bool(action_valid),
            "valid_actions": [LabelFormatter.action_name(action_id, self.config) for action_id in valid_actions],
            "alert_correct": alert_correct,
            "false_alert": false_alert,
            "missed_alert": missed_alert,
            "current_alert_level": self._alert_level_name(self.current_alert_level),
            "step_count": self.step_count,
            "state_tuple": self.state,
            "state_text": LabelFormatter.state_name(self.state, self.config),
            "reward_terms": reward_terms,
        }
        return self.state_to_index(self.state), reward, self.done, info

    def state_to_index(self, state: tuple[int, int, int, int, int]) -> int:
        """Converts a tuple state into an integer index."""
        return StateCodec.encode_state(state, self.config.state_shape)

    def index_to_state(self, index: int) -> tuple[int, int, int, int, int]:
        """Converts an integer index back into tuple state."""
        return StateCodec.decode_state(index, self.config.state_shape)

    def set_state_for_testing(
        self,
        state: tuple[int, int, int, int, int],
        hidden_true_risk: int | None = None,
        hidden_wave_risk: int | None = None,
        current_alert_level: int = 0,
        warning_issued_step: int | None = None,
    ) -> None:
        """Injects deterministic state values for tests."""
        self.state = state
        self.hidden_true_risk = self._clamp(
            hidden_true_risk if hidden_true_risk is not None else self._get_ground_truth_risk(state),
            0,
            2,
        )
        self.hidden_wave_risk = self._clamp(hidden_wave_risk if hidden_wave_risk is not None else state[2], 0, 2)
        self.current_alert_level = self._clamp(current_alert_level, 0, 2)
        self.warning_issued_step = warning_issued_step
        self.step_count = 0
        self.done = False

    def _generate_initial_state(self) -> tuple[int, int, int, int, int]:
        """Builds an initial observable state while creating hidden event truth."""
        magnitude = int(self.rng.choice([0, 1, 2], p=[0.40, 0.38, 0.22]))
        depth = int(self.rng.choice([0, 1, 2], p=[0.36, 0.36, 0.28]))

        self.hidden_wave_risk = self._sample_hidden_wave_risk(magnitude, depth)
        observed_wave_noise = int(self.rng.choice([-1, 0, 1], p=[0.20, 0.60, 0.20]))
        observed_wave_risk = self._clamp(self.hidden_wave_risk + observed_wave_noise, 0, 2)

        confidence = int(self.rng.choice([0, 1, 2], p=[0.42, 0.40, 0.18]))
        time_index = 0

        state = (magnitude, depth, observed_wave_risk, confidence, time_index)
        self.hidden_true_risk = self._get_ground_truth_risk((magnitude, depth, self.hidden_wave_risk, confidence, time_index))
        return state

    def _sample_hidden_wave_risk(self, magnitude: int, depth: int) -> int:
        """Samples hidden wave risk using magnitude and depth as priors."""
        center = (magnitude + depth) / 2.0
        noisy_value = int(round(center + self.rng.normal(0.0, 0.65)))
        return self._clamp(noisy_value, 0, 2)

    def _get_ground_truth_risk(self, state: tuple[int, int, int, int, int]) -> int:
        """Computes hidden risk class from magnitude, depth, and wave risk."""
        magnitude, depth, wave_risk, _, _ = state
        score = (1.4 * magnitude) + (1.2 * depth) + (1.6 * wave_risk)
        if score >= 5.0:
            return 2
        if score >= 2.8:
            return 1
        return 0

    def _transition_state(self, action: int) -> tuple[int, int, int, int, int]:
        """Applies stochastic evidence transitions across a 12-step timeline."""
        magnitude, depth, wave_risk, confidence, time_index = self.state

        time_index = self._clamp(time_index + 1, 0, len(self.config.time_levels) - 1)

        if action in (1, 2):
            if self.rng.random() < 0.82:
                confidence = self._clamp(confidence + 1, 0, 2)

            if self.rng.random() < 0.78:
                if wave_risk < self.hidden_wave_risk:
                    wave_risk += 1
                elif wave_risk > self.hidden_wave_risk:
                    wave_risk -= 1
            elif self.rng.random() < 0.32:
                wave_risk += int(self.rng.choice([-1, 1]))
        elif action == 3:
            if self.rng.random() < 0.45:
                confidence = self._clamp(confidence - 1, 0, 2)
            if self.rng.random() < 0.25:
                wave_risk += int(self.rng.choice([-1, 1]))
        else:
            if self.rng.random() < 0.30:
                confidence = self._clamp(confidence - 1, 0, 2)
            if self.rng.random() < 0.24:
                wave_risk += int(self.rng.choice([-1, 1]))

        if self.hidden_true_risk == 2 and self.rng.random() < 0.38:
            wave_risk += 1
        if self.hidden_true_risk == 0 and self.rng.random() < 0.34:
            wave_risk -= 1

        wave_risk = self._clamp(wave_risk, 0, 2)
        return magnitude, depth, wave_risk, confidence, time_index

    def get_valid_actions(self) -> list[int]:
        """Returns the currently valid action ids based on active alert level."""
        if self.current_alert_level <= 0:
            return [0, 1, 2]
        if self.current_alert_level == 1:
            return [0, 1, 2, 3]
        return [0, 2, 3]

    def _apply_action_to_alert_level(self, action: int, current_alert_level: int) -> int:
        """Updates alert level from the selected action."""
        if action == 1:
            return max(current_alert_level, 1)
        if action == 2:
            return 2
        if action == 3:
            return 0
        return current_alert_level

    @staticmethod
    def _alert_level_name(alert_level: int) -> str:
        """Returns display text for internal alert level id."""
        mapping = {0: "No Alert", 1: "Watch / Advisory", 2: "Warning"}
        return mapping.get(int(alert_level), "Unknown")

    def _calculate_reward(
        self,
        action: int,
        action_valid: bool,
        previous_alert_level: int,
        new_alert_level: int,
        previous_state: tuple[int, int, int, int, int],
        next_state: tuple[int, int, int, int, int],
        terminal: bool,
    ) -> tuple[float, bool, bool, bool, dict[str, float]]:
        """Computes reward and outcome flags for the selected action."""
        reward_terms: dict[str, float] = {
            "r_base": float(self.config.base_step_cost),
            "r_invalid": 0.0,
            "r_churn": 0.0,
            "r_evidence": 0.0,
            "r_overreact": 0.0,
            "r_cancel": 0.0,
            "r_terminal": 0.0,
        }
        alert_correct = False
        false_alert = False
        missed_alert = False

        if not action_valid:
            reward_terms["r_invalid"] += float(self.config.penalty_invalid_action)

        # Penalize alert-level flips to discourage unstable operations.
        if new_alert_level != previous_alert_level:
            reward_terms["r_churn"] += float(self.config.penalty_churn)

        observed_wave = int(next_state[2])
        confidence = int(next_state[3])
        time_index = int(next_state[4])

        danger_signal = observed_wave >= 2 and confidence >= 1
        strong_danger_signal = observed_wave >= 2 and confidence >= 2 and time_index >= 4
        # Evidence penalties encode the cost of keeping alerts too low while ocean evidence strengthens.
        if new_alert_level < 1 and danger_signal:
            reward_terms["r_evidence"] += float(self.config.penalty_ignore_evidence)
        if new_alert_level < 2 and strong_danger_signal:
            reward_terms["r_evidence"] += float(self.config.penalty_ignore_evidence * 0.60)

        # Overreaction penalties protect trust when warning is raised for genuinely low-risk events.
        if new_alert_level >= 2 and self.hidden_true_risk == 0:
            reward_terms["r_overreact"] += float(self.config.penalty_overreact_warning)

        if action_valid and action == 3 and self.hidden_true_risk >= 1:
            reward_terms["r_cancel"] += float(self.config.penalty_risky_cancel)

        # Terminal outcomes enforce asymmetric safety priorities: misses are very costly, timely warnings are rewarded.
        if terminal:
            if self.hidden_true_risk == 2:
                if self.warning_issued_step is not None or new_alert_level >= 2:
                    warning_step = (
                        int(self.warning_issued_step) if self.warning_issued_step is not None else int(self.step_count)
                    )
                    reward_terms["r_terminal"] += float(
                        max(
                            self.config.terminal_warning_floor,
                            self.config.terminal_warning_base
                            - (self.config.terminal_warning_decay_per_step * float(warning_step)),
                        )
                    )
                    alert_correct = True
                else:
                    reward_terms["r_terminal"] += float(self.config.terminal_miss_penalty)
                    missed_alert = True
            elif self.hidden_true_risk == 1:
                if new_alert_level >= 2:
                    reward_terms["r_terminal"] += float(self.config.penalty_false_warning_terminal)
                    false_alert = True
                elif new_alert_level == 1:
                    reward_terms["r_terminal"] += float(self.config.reward_correct_regional_alert)
                    alert_correct = True
                else:
                    reward_terms["r_terminal"] += float(self.config.penalty_missed_dangerous_alert * 0.4)
                    missed_alert = True
            else:
                if new_alert_level >= 2:
                    reward_terms["r_terminal"] += float(self.config.penalty_false_warning_terminal)
                    false_alert = True
                elif new_alert_level == 1:
                    reward_terms["r_terminal"] += float(self.config.penalty_false_watch_terminal)
                    false_alert = True
                else:
                    reward_terms["r_terminal"] += float(self.config.reward_safe_resolution)
                    alert_correct = True

        reward = float(sum(reward_terms.values()))
        return reward, alert_correct, false_alert, missed_alert, reward_terms

    def _is_uncertain(self, state: tuple[int, int, int, int, int]) -> bool:
        """Checks whether the observed state has meaningful uncertainty."""
        observed_wave_risk = state[2]
        confidence = state[3]
        wave_gap = abs(observed_wave_risk - self.hidden_wave_risk)
        return confidence == 0 or (confidence <= 1 and wave_gap >= 1)

    def _is_terminal(self, next_state: tuple[int, int, int, int, int]) -> bool:
        """Determines whether the fixed horizon has been reached."""
        return int(next_state[4]) >= (len(self.config.time_levels) - 1)

    @staticmethod
    def _clamp(value: int, low: int, high: int) -> int:
        """Clamps integer value within bounds."""
        return max(low, min(high, value))
