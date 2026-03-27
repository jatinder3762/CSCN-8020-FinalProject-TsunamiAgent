"""Tests for scenario generation and dashboard controller behavior."""

from __future__ import annotations

import unittest

from src.presentation import TsunamiDashboardApp
from src.presentation.controller import DashboardController
from src.presentation.scenarios import ScenarioGenerator


class TestScenarioGenerator(unittest.TestCase):
    """Validates deterministic scenario generation."""

    def test_build_scenarios_returns_expected_named_cases(self) -> None:
        """Scenario generator should build the required named scenarios."""
        generator = ScenarioGenerator()
        scenarios = generator.build_scenarios()

        expected_names = {
            "High-Risk Confirmed Tsunami",
            "Uncertain Moderate-Risk Case",
            "False Alarm Case",
            "Delayed Response Case",
        }

        self.assertEqual(set(scenarios), expected_names)
        for scenario in scenarios.values():
            self.assertGreaterEqual(scenario.step_count, 2)

    def test_scenario_generation_is_deterministic(self) -> None:
        """Repeated generation should produce stable scenario content."""
        generator = ScenarioGenerator()
        first = generator.build_scenarios()
        second = generator.build_scenarios()

        self.assertEqual(
            first["High-Risk Confirmed Tsunami"].steps[0].action,
            second["High-Risk Confirmed Tsunami"].steps[0].action,
        )
        self.assertEqual(
            first["Delayed Response Case"].outcome_label,
            second["Delayed Response Case"].outcome_label,
        )


class TestDashboardController(unittest.TestCase):
    """Validates scenario selection and step navigation behavior."""

    def setUp(self) -> None:
        """Creates a controller backed by a plain dict session state."""
        self.session_state: dict[str, object] = {}
        self.controller = DashboardController(self.session_state, generator=ScenarioGenerator())
        self.controller.initialize()

    def test_controller_initializes_default_scenario(self) -> None:
        """Controller should load the default scenario on init."""
        self.assertEqual(self.controller.selected_scenario_name, "High-Risk Confirmed Tsunami")
        self.assertEqual(self.controller.current_step, 0)

    def test_step_navigation_stays_bounded(self) -> None:
        """Next, previous, and reset should stay within valid bounds."""
        self.controller.previous_step()
        self.assertEqual(self.controller.current_step, 0)

        for _ in range(10):
            self.controller.next_step()
        self.assertEqual(self.controller.current_step, self.controller.current_trace.last_step_index)

        self.controller.reset()
        self.assertEqual(self.controller.current_step, 0)

    def test_set_step_clamps_values(self) -> None:
        """Direct step selection should clamp to safe bounds."""
        self.controller.set_step(99)
        self.assertEqual(self.controller.current_step, self.controller.current_trace.last_step_index)

        self.controller.set_step(-10)
        self.assertEqual(self.controller.current_step, 0)

    def test_scenario_selection_resets_step(self) -> None:
        """Changing scenario should reset the step and update the trace."""
        self.controller.next_step()
        self.assertGreater(self.controller.current_step, 0)

        trace = self.controller.select_scenario("False Alarm Case")

        self.assertEqual(self.controller.selected_scenario_name, "False Alarm Case")
        self.assertEqual(self.controller.current_step, 0)
        self.assertEqual(trace.name, "False Alarm Case")

    def test_initialize_preserves_live_widget_state(self) -> None:
        """Initialization should not overwrite user-selected widget values."""
        self.session_state[self.controller.SCENARIO_WIDGET_KEY] = "Delayed Response Case"
        self.session_state[self.controller.STEP_WIDGET_KEY] = 2
        self.session_state[self.controller.CURRENT_STEP_KEY] = 2
        self.session_state[self.controller.SELECTED_SCENARIO_KEY] = "Delayed Response Case"

        self.controller.initialize()

        self.assertEqual(self.controller.selected_scenario_name, "Delayed Response Case")
        self.assertEqual(self.session_state[self.controller.SCENARIO_WIDGET_KEY], "Delayed Response Case")
        self.assertEqual(self.controller.current_step, 2)
        self.assertEqual(self.session_state[self.controller.STEP_WIDGET_KEY], 2)


class TestPresentationImports(unittest.TestCase):
    """Validates the presentation package export surface."""

    def test_dashboard_app_export_is_available(self) -> None:
        """The package root should expose the main Streamlit dashboard app."""
        app = TsunamiDashboardApp()
        self.assertEqual(app.title, "Tsunami RL Decision Dashboard")


if __name__ == "__main__":
    unittest.main()
