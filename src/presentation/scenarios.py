"""Deterministic named scenarios for the Streamlit dashboard."""

from __future__ import annotations

from src.presentation.models import DecisionStep, EpisodeTrace


class ScenarioGenerator:
    """Builds lightweight deterministic scenarios for dashboard walkthroughs."""

    def __init__(self) -> None:
        """Initializes the supported scenario names."""
        self._default_scenario_name = "High-Risk Confirmed Tsunami"

    @property
    def default_scenario_name(self) -> str:
        """Returns the default scenario used on first load."""
        return self._default_scenario_name

    def build_scenarios(self) -> dict[str, EpisodeTrace]:
        """Builds the full named scenario catalog."""
        return {
            "High-Risk Confirmed Tsunami": self._build_high_risk_confirmed(),
            "Uncertain Moderate-Risk Case": self._build_uncertain_moderate(),
            "False Alarm Case": self._build_false_alarm_case(),
            "Delayed Response Case": self._build_delayed_response_case(),
        }

    def _build_high_risk_confirmed(self) -> EpisodeTrace:
        """Creates a confirmed high-risk case ending in a correct full alert."""
        return EpisodeTrace(
            name="High-Risk Confirmed Tsunami",
            summary=(
                "A shallow, high-magnitude event quickly escalates from verification "
                "to a confirmed full-alert decision."
            ),
            risk_level="High",
            outcome_label="Correct Full Alert",
            steps=(
                DecisionStep(
                    step_index=0,
                    time_label="Early",
                    magnitude="High",
                    depth="Shallow",
                    wave_risk="High",
                    confidence="Medium",
                    action="Verify",
                    reward=30.0,
                    explanation="Agent requested verification due to high wave risk with only medium confidence.",
                    status="Verification Requested",
                ),
                DecisionStep(
                    step_index=1,
                    time_label="Mid",
                    magnitude="High",
                    depth="Shallow",
                    wave_risk="High",
                    confidence="High",
                    action="Full Alert",
                    reward=100.0,
                    explanation="Agent issued full alert because wave risk and confidence were both high.",
                    status="Full Alert Issued",
                ),
            ),
        )

    def _build_uncertain_moderate(self) -> EpisodeTrace:
        """Creates an uncertain case that resolves into a regional alert."""
        return EpisodeTrace(
            name="Uncertain Moderate-Risk Case",
            summary=(
                "A moderate event starts with uncertainty, improves after verification, "
                "and ends with a targeted regional alert."
            ),
            risk_level="Medium",
            outcome_label="Correct Regional Alert",
            steps=(
                DecisionStep(
                    step_index=0,
                    time_label="Early",
                    magnitude="Medium",
                    depth="Moderate",
                    wave_risk="Medium",
                    confidence="Low",
                    action="Wait",
                    reward=-10.0,
                    explanation="Agent waited because confidence was low and the first signal was not yet reliable.",
                    status="Monitoring",
                ),
                DecisionStep(
                    step_index=1,
                    time_label="Mid",
                    magnitude="Medium",
                    depth="Moderate",
                    wave_risk="Medium",
                    confidence="Medium",
                    action="Verify",
                    reward=30.0,
                    explanation="Agent requested verification due to uncertainty in the moderate-risk signal.",
                    status="Verification Requested",
                ),
                DecisionStep(
                    step_index=2,
                    time_label="Late",
                    magnitude="Medium",
                    depth="Moderate",
                    wave_risk="High",
                    confidence="High",
                    action="Regional Alert",
                    reward=60.0,
                    explanation="Agent issued a regional alert after confidence improved and the wave signal strengthened.",
                    status="Regional Alert Issued",
                ),
            ),
        )

    def _build_false_alarm_case(self) -> EpisodeTrace:
        """Creates a low-risk case that results in an unnecessary alert."""
        return EpisodeTrace(
            name="False Alarm Case",
            summary=(
                "A noisy low-risk signal is overinterpreted, leading to an avoidable regional alert."
            ),
            risk_level="Low",
            outcome_label="False Alarm",
            steps=(
                DecisionStep(
                    step_index=0,
                    time_label="Early",
                    magnitude="Low",
                    depth="Deep",
                    wave_risk="Medium",
                    confidence="Low",
                    action="Verify",
                    reward=30.0,
                    explanation="Agent requested verification because the observed wave signal looked inconsistent.",
                    status="Verification Requested",
                ),
                DecisionStep(
                    step_index=1,
                    time_label="Mid",
                    magnitude="Low",
                    depth="Deep",
                    wave_risk="Low",
                    confidence="Medium",
                    action="Regional Alert",
                    reward=-30.0,
                    explanation="Agent issued a regional alert even though the underlying risk had settled to a low level.",
                    status="False Alarm",
                ),
            ),
        )

    def _build_delayed_response_case(self) -> EpisodeTrace:
        """Creates a high-risk case where the warning comes too late."""
        return EpisodeTrace(
            name="Delayed Response Case",
            summary=(
                "A dangerous event is recognized too slowly, producing a warning that arrives later than ideal."
            ),
            risk_level="High",
            outcome_label="Delayed Warning",
            steps=(
                DecisionStep(
                    step_index=0,
                    time_label="Early",
                    magnitude="High",
                    depth="Moderate",
                    wave_risk="Medium",
                    confidence="Low",
                    action="Wait",
                    reward=-10.0,
                    explanation="Agent waited because confidence was still low despite a strong magnitude reading.",
                    status="Monitoring",
                ),
                DecisionStep(
                    step_index=1,
                    time_label="Mid",
                    magnitude="High",
                    depth="Moderate",
                    wave_risk="High",
                    confidence="Medium",
                    action="Wait",
                    reward=-30.0,
                    explanation="Agent delayed again even as wave risk increased, reducing available warning time.",
                    status="Warning Delayed",
                ),
                DecisionStep(
                    step_index=2,
                    time_label="Late",
                    magnitude="High",
                    depth="Moderate",
                    wave_risk="High",
                    confidence="High",
                    action="Full Alert",
                    reward=40.0,
                    explanation="Agent finally issued a full alert, but only after valuable time had already been lost.",
                    status="Delayed Warning",
                ),
            ),
        )
