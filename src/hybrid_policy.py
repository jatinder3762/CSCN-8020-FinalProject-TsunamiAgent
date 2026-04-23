"""Hybrid deployment policy that gates RL actions against a rule baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HybridDecision:
    """Captures the deployed action plus decision diagnostics."""

    deployed_action: int
    rl_action: int
    rule_action: int
    margin: float
    q_rl: float
    q_rule: float
    used_override: bool


class SafeHybridOverride:
    """Implements threshold-based override: deploy RL only when margin >= delta."""

    @staticmethod
    def select_action(
        state_idx: int,
        state: tuple[int, int, int, int, int],
        current_alert_level: int,
        valid_actions: list[int] | tuple[int, ...],
        q_table: np.ndarray,
        delta: float,
    ) -> HybridDecision:
        """Selects action with a rule fallback when RL confidence margin is small."""
        if not valid_actions:
            raise ValueError("valid_actions cannot be empty.")

        action_pool = [int(action) for action in valid_actions]
        rule_action = SafeHybridOverride.rule_baseline_action(state, current_alert_level, action_pool)
        rl_action, q_rl = SafeHybridOverride._best_q_action(state_idx=state_idx, valid_actions=action_pool, q_table=q_table)
        q_rule = float(q_table[state_idx, rule_action])
        margin = float(q_rl - q_rule)

        if margin >= float(delta):
            return HybridDecision(
                deployed_action=rl_action,
                rl_action=rl_action,
                rule_action=rule_action,
                margin=margin,
                q_rl=q_rl,
                q_rule=q_rule,
                used_override=False,
            )

        return HybridDecision(
            deployed_action=rule_action,
            rl_action=rl_action,
            rule_action=rule_action,
            margin=margin,
            q_rl=q_rl,
            q_rule=q_rule,
            used_override=True,
        )

    @staticmethod
    def rule_baseline_action(
        state: tuple[int, int, int, int, int],
        current_alert_level: int,
        valid_actions: list[int] | tuple[int, ...],
    ) -> int:
        """Returns a conservative baseline action from observed risk, confidence, and time."""
        if not valid_actions:
            raise ValueError("valid_actions cannot be empty.")

        magnitude, _, wave_risk, confidence, time_index = [int(value) for value in state]

        # High observed wave risk should escalate quickly, especially when confidence is not low.
        if wave_risk >= 2 and confidence >= 1:
            return SafeHybridOverride._first_valid((2, 1, 0, 3), valid_actions)
        if wave_risk >= 2:
            if time_index >= 4 or magnitude >= 2:
                return SafeHybridOverride._first_valid((1, 2, 0, 3), valid_actions)
            return SafeHybridOverride._first_valid((1, 0, 2, 3), valid_actions)

        # Moderate wave risk prefers advisory unless late and severe.
        if wave_risk == 1:
            if confidence >= 2 and time_index >= 6 and magnitude >= 2:
                return SafeHybridOverride._first_valid((2, 1, 0, 3), valid_actions)
            if confidence >= 1:
                return SafeHybridOverride._first_valid((1, 2, 0, 3), valid_actions)
            if time_index >= 8 and magnitude >= 2:
                return SafeHybridOverride._first_valid((1, 2, 0, 3), valid_actions)
            return SafeHybridOverride._first_valid((0, 1, 2, 3), valid_actions)

        # Low observed wave risk defaults to hold, with cancel allowed when alert has been active.
        if int(current_alert_level) >= 1 and confidence >= 1 and time_index >= 5:
            return SafeHybridOverride._first_valid((3, 0, 1, 2), valid_actions)
        return SafeHybridOverride._first_valid((0, 1, 2, 3), valid_actions)

    @staticmethod
    def _best_q_action(state_idx: int, valid_actions: list[int], q_table: np.ndarray) -> tuple[int, float]:
        """Returns deterministic argmax action among valid actions and its Q value."""
        q_values = [float(q_table[state_idx, action]) for action in valid_actions]
        max_q = max(q_values)
        best_actions = [action for action, value in zip(valid_actions, q_values) if np.isclose(value, max_q)]
        return int(min(best_actions)), float(max_q)

    @staticmethod
    def _first_valid(preferred_actions: tuple[int, ...], valid_actions: list[int] | tuple[int, ...]) -> int:
        """Returns first preferred action that is currently valid."""
        ordered_valid = [int(action) for action in valid_actions]
        valid_set = set(ordered_valid)
        for action in preferred_actions:
            if int(action) in valid_set:
                return int(action)
        return int(ordered_valid[0])
