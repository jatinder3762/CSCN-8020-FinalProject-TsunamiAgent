"""Controller logic for the scenario-driven Streamlit dashboard."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

from src.presentation.models import DecisionStep, EpisodeTrace
from src.presentation.scenarios import ScenarioGenerator


class DashboardController:
    """Coordinates scenarios, step navigation, and session state."""

    SCENARIOS_KEY = "dashboard_scenarios"
    SELECTED_SCENARIO_KEY = "dashboard_selected_scenario"
    CURRENT_STEP_KEY = "dashboard_current_step"
    SCENARIO_WIDGET_KEY = "dashboard_scenario_selector"
    STEP_WIDGET_KEY = "dashboard_step_slider"

    def __init__(
        self,
        session_state: MutableMapping[str, Any],
        generator: ScenarioGenerator | None = None,
    ) -> None:
        """Stores session state and scenario generator dependencies."""
        self.session_state = session_state
        self.generator = generator or ScenarioGenerator()

    def initialize(self) -> None:
        """Initializes safe default session state values."""
        if self.SCENARIOS_KEY not in self.session_state:
            self.session_state[self.SCENARIOS_KEY] = self.generator.build_scenarios()

        scenarios = self.scenarios
        default_name = self.generator.default_scenario_name

        selected_name = str(self.session_state.get(self.SELECTED_SCENARIO_KEY, default_name))
        if selected_name not in scenarios:
            selected_name = default_name

        self.session_state[self.SELECTED_SCENARIO_KEY] = selected_name
        if self.SCENARIO_WIDGET_KEY not in self.session_state:
            self.session_state[self.SCENARIO_WIDGET_KEY] = selected_name

        current_step = int(self.session_state.get(self.CURRENT_STEP_KEY, 0))
        self.session_state[self.CURRENT_STEP_KEY] = current_step
        if self.STEP_WIDGET_KEY not in self.session_state:
            self.session_state[self.STEP_WIDGET_KEY] = current_step

        self.set_step(current_step)

    @property
    def scenarios(self) -> dict[str, EpisodeTrace]:
        """Returns the full scenario catalog."""
        return dict(self.session_state[self.SCENARIOS_KEY])

    @property
    def scenario_names(self) -> list[str]:
        """Returns scenario names in display order."""
        return list(self.scenarios.keys())

    @property
    def selected_scenario_name(self) -> str:
        """Returns the currently selected scenario name."""
        return str(self.session_state[self.SELECTED_SCENARIO_KEY])

    @property
    def current_trace(self) -> EpisodeTrace:
        """Returns the current scenario trace."""
        return self.scenarios[self.selected_scenario_name]

    @property
    def current_step(self) -> int:
        """Returns the current visible step index."""
        return int(self.session_state[self.CURRENT_STEP_KEY])

    @property
    def current_decision(self) -> DecisionStep:
        """Returns the current decision step."""
        return self.current_trace.get_step(self.current_step)

    def select_scenario(self, scenario_name: str) -> EpisodeTrace:
        """Selects a scenario and resets the visible step."""
        if scenario_name not in self.scenarios:
            raise KeyError(f"Unknown scenario: {scenario_name}")

        self.session_state[self.SELECTED_SCENARIO_KEY] = scenario_name
        self.reset()
        return self.current_trace

    def set_step(self, step_index: int, sync_widget: bool = True) -> int:
        """Sets the visible step using safe bounds."""
        bounded = max(0, min(int(step_index), self.current_trace.last_step_index))
        self.session_state[self.CURRENT_STEP_KEY] = bounded
        if sync_widget:
            self.session_state[self.STEP_WIDGET_KEY] = bounded
        return bounded

    def next_step(self) -> int:
        """Advances to the next step if possible."""
        return self.set_step(self.current_step + 1)

    def previous_step(self) -> int:
        """Moves back one step if possible."""
        return self.set_step(self.current_step - 1)

    def reset(self) -> int:
        """Resets the current step to the beginning of the scenario."""
        return self.set_step(0)

    def progress_ratio(self) -> float:
        """Returns progress as a value between 0 and 1."""
        return float(self.current_step + 1) / float(self.current_trace.step_count)

    def scenario_summary(self) -> dict[str, str]:
        """Returns summary data for the selected scenario."""
        return self.current_trace.sidebar_summary()

    def visible_history(self) -> list[dict[str, Any]]:
        """Returns history rows through the current visible step."""
        return self.current_trace.history_rows(self.current_step)
