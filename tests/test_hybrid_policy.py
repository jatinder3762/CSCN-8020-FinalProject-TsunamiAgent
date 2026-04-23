"""Unit tests for safe hybrid override policy behavior."""

from __future__ import annotations

import unittest

import numpy as np

from src.hybrid_policy import SafeHybridOverride


class TestSafeHybridOverride(unittest.TestCase):
    """Validates thresholded RL-vs-rule deployment behavior."""

    def test_rule_baseline_returns_valid_action(self) -> None:
        """Rule baseline should always return one of the currently valid actions."""
        action = SafeHybridOverride.rule_baseline_action(
            state=(2, 1, 2, 1, 5),
            current_alert_level=1,
            valid_actions=[0, 2, 3],
        )
        self.assertIn(action, [0, 2, 3])

    def test_select_action_overrides_to_rule_when_margin_below_delta(self) -> None:
        """If RL advantage over baseline is weak, deployed action should fall back to rule."""
        q_table = np.zeros((1, 4), dtype=np.float64)
        q_table[0, 1] = 10.0
        q_table[0, 2] = 14.0

        decision = SafeHybridOverride.select_action(
            state_idx=0,
            state=(2, 1, 1, 1, 2),
            current_alert_level=0,
            valid_actions=[0, 1, 2],
            q_table=q_table,
            delta=5.0,
        )

        self.assertEqual(decision.rule_action, 1)
        self.assertEqual(decision.rl_action, 2)
        self.assertEqual(decision.deployed_action, 1)
        self.assertTrue(decision.used_override)
        self.assertAlmostEqual(decision.margin, 4.0)

    def test_select_action_keeps_rl_when_margin_meets_delta(self) -> None:
        """If RL margin is at least delta, deployed action should remain RL action."""
        q_table = np.zeros((1, 4), dtype=np.float64)
        q_table[0, 1] = 10.0
        q_table[0, 2] = 15.0

        decision = SafeHybridOverride.select_action(
            state_idx=0,
            state=(2, 1, 1, 1, 2),
            current_alert_level=0,
            valid_actions=[0, 1, 2],
            q_table=q_table,
            delta=5.0,
        )

        self.assertEqual(decision.rule_action, 1)
        self.assertEqual(decision.rl_action, 2)
        self.assertEqual(decision.deployed_action, 2)
        self.assertFalse(decision.used_override)
        self.assertAlmostEqual(decision.margin, 5.0)


if __name__ == "__main__":
    unittest.main()
