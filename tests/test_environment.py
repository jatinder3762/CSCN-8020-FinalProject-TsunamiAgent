"""Unit tests for the custom tsunami alert environment."""

from __future__ import annotations

import unittest

from config import ProjectConfig
from src.environment import TsunamiAlertEnvironment


class TestTsunamiAlertEnvironment(unittest.TestCase):
    """Validates core environment behavior and interface contracts."""

    def setUp(self) -> None:
        """Builds a small deterministic test fixture."""
        self.config = ProjectConfig(training_episodes=10, evaluation_episodes=5, random_seed=123)
        self.environment = TsunamiAlertEnvironment(config=self.config, seed=123)

    def test_reset_returns_valid_state_index(self) -> None:
        """Reset should return an integer state index inside valid range."""
        state_idx = self.environment.reset()
        self.assertIsInstance(state_idx, int)
        self.assertGreaterEqual(state_idx, 0)
        self.assertLess(state_idx, self.config.state_size)

    def test_step_returns_expected_structure(self) -> None:
        """Step output must match standard RL tuple structure."""
        self.environment.reset()
        next_state_idx, reward, done, info = self.environment.step(0)

        self.assertIsInstance(next_state_idx, int)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)

        for key in ("actual_risk_level", "action_meaning", "alert_correct", "false_alert", "missed_alert", "step_count"):
            self.assertIn(key, info)

    def test_terminal_state_occurs_when_time_window_closes(self) -> None:
        """Waiting at mid-time should lead to terminal late-time state."""
        self.environment.set_state_for_testing(
            state=(2, 2, 2, 1, 1),
            hidden_true_risk=2,
            hidden_wave_risk=2,
        )
        _, _, done, _ = self.environment.step(0)
        self.assertTrue(done)

    def test_state_index_roundtrip(self) -> None:
        """Encoding and decoding state indices should be reversible."""
        sample_states = [
            (0, 0, 0, 0, 0),
            (2, 2, 2, 2, 2),
            (1, 0, 2, 1, 2),
            (2, 1, 1, 0, 1),
        ]
        for state in sample_states:
            idx = self.environment.state_to_index(state)
            decoded = self.environment.index_to_state(idx)
            self.assertEqual(state, decoded)


if __name__ == "__main__":
    unittest.main()
