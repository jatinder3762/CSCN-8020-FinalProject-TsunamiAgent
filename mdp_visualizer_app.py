"""Standalone Streamlit MDP visual simulator mapped to the tsunami RL project.

This app is intentionally deterministic and presentation-friendly:
- It uses explicit scenario trajectories (state observations by timestep).
- It keeps episode state in st.session_state (no manual loops).
- It exposes reward decomposition for each chosen action.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import pandas as pd
import streamlit as st

from config import ProjectConfig


MAX_STEPS = 10


class Observation(TypedDict):
    """One observed state in the simulator at time t."""

    estimated_magnitude: float
    depth_estimate: str
    buoy_confirmed_count: int
    uncertainty: str
    time_elapsed: int


class ScenarioSpec(TypedDict):
    """Scenario definition with deterministic observation path."""

    name: str
    is_true_tsunami: bool
    observations: list[Observation]


@dataclass(frozen=True)
class RewardScale:
    """Reward scale mapped from the project reward configuration."""

    false_warning_penalty: float
    missed_terminal_penalty: float
    delay_penalty: float
    churn_penalty: float
    timely_warning_reward: float


def _build_reward_scale(cfg: ProjectConfig) -> RewardScale:
    """Creates scenario-simulator reward magnitudes mapped to current project config.

    Mapping rationale:
    - Keep proposal magnitudes (approx -500, -5000, -50, +1000) while anchoring to
      existing model semantics in ProjectConfig.
    """
    return RewardScale(
        false_warning_penalty=float(cfg.penalty_false_full_alert * 8.33),     # ~ -500
        missed_terminal_penalty=float(cfg.penalty_missed_dangerous_alert * 50),  # -5000
        delay_penalty=float(cfg.penalty_late_wait_in_risk * 2.5),             # -50
        churn_penalty=-5.0,
        timely_warning_reward=float(cfg.reward_correct_full_alert * 10),       # +1000
    )


def _build_observation_path(
    *,
    magnitude_start: float,
    magnitude_end: float,
    buoy_start: int,
    buoy_end: int,
    depth_rule: str,
    uncertainty_rule: str,
) -> list[Observation]:
    """Builds an 11-point path (t=0..10) for deterministic scenario playback."""
    observations: list[Observation] = []
    for t in range(MAX_STEPS + 1):
        ratio = t / float(MAX_STEPS)
        magnitude = magnitude_start + ((magnitude_end - magnitude_start) * ratio)
        buoy_value = int(round(buoy_start + ((buoy_end - buoy_start) * ratio)))

        if depth_rule == "A":
            depth = "Moderate" if t < 5 else "Shallow"
        else:
            depth = "Deep" if t < 5 else "Moderate"

        if uncertainty_rule == "A":
            uncertainty = "High" if t < 3 else ("Medium" if t < 6 else "Low")
        else:
            uncertainty = "Medium" if t < 4 else "Low"

        observations.append(
            Observation(
                estimated_magnitude=round(float(magnitude), 2),
                depth_estimate=depth,
                buoy_confirmed_count=max(0, min(4, buoy_value)),
                uncertainty=uncertainty,
                time_elapsed=t,
            )
        )
    return observations


def _build_scenarios() -> dict[str, ScenarioSpec]:
    """Builds deterministic Scenario A/B definitions from the proposal."""
    scenario_a = ScenarioSpec(
        name="Scenario A: True Tsunami",
        is_true_tsunami=True,
        observations=_build_observation_path(
            magnitude_start=6.5,
            magnitude_end=8.1,
            buoy_start=0,
            buoy_end=4,
            depth_rule="A",
            uncertainty_rule="A",
        ),
    )
    scenario_b = ScenarioSpec(
        name="Scenario B: False Alarm",
        is_true_tsunami=False,
        observations=_build_observation_path(
            magnitude_start=7.2,
            magnitude_end=6.8,
            buoy_start=1,
            buoy_end=0,
            depth_rule="B",
            uncertainty_rule="B",
        ),
    )
    return {scenario_a["name"]: scenario_a, scenario_b["name"]: scenario_b}


def _initialize_session_state(selected_scenario: str) -> None:
    """Initializes or resets simulator state for the selected scenario."""
    st.session_state["selected_scenario"] = selected_scenario
    st.session_state["current_step"] = 0
    st.session_state["cumulative_return"] = 0.0
    st.session_state["current_alert_level"] = "No Alert"
    st.session_state["warning_ever_issued"] = False
    st.session_state["episode_done"] = False
    st.session_state["last_action"] = "None"
    st.session_state["last_reward"] = 0.0
    st.session_state["last_reward_components"] = {
        "Base Reward": 0.0,
        "False Alarm Penalty": 0.0,
        "Missed Event Penalty": 0.0,
        "Delay Penalty": 0.0,
        "Alert Churn Penalty": 0.0,
        "Timely Warning Reward": 0.0,
    }
    st.session_state["return_history"] = [{"Step": 0, "Cumulative Return": 0.0}]
    st.session_state["transition_log"] = []


def _get_observation(scenarios: dict[str, ScenarioSpec]) -> Observation:
    """Returns current observation for selected scenario and step."""
    scenario_name = str(st.session_state["selected_scenario"])
    step = int(st.session_state["current_step"])
    return scenarios[scenario_name]["observations"][step]


def _apply_action(
    *,
    action_label: str,
    scenarios: dict[str, ScenarioSpec],
    scale: RewardScale,
) -> None:
    """Applies one simulator step using proposal-aligned reward logic."""
    if bool(st.session_state["episode_done"]):
        return

    scenario_name = str(st.session_state["selected_scenario"])
    scenario = scenarios[scenario_name]

    step = int(st.session_state["current_step"])
    obs = scenario["observations"][step]
    previous_alert = str(st.session_state["current_alert_level"])
    next_alert = _action_to_alert_level(action_label)
    next_step = min(MAX_STEPS, step + 1)

    components = {
        "Base Reward": 0.0,
        "False Alarm Penalty": 0.0,
        "Missed Event Penalty": 0.0,
        "Delay Penalty": 0.0,
        "Alert Churn Penalty": 0.0,
        "Timely Warning Reward": 0.0,
    }

    if not scenario["is_true_tsunami"] and next_alert == "Warning":
        components["False Alarm Penalty"] = scale.false_warning_penalty

    if scenario["is_true_tsunami"] and step >= 4 and next_alert not in {"Watch", "Warning"}:
        components["Delay Penalty"] = scale.delay_penalty

    if previous_alert != next_alert:
        components["Alert Churn Penalty"] = scale.churn_penalty

    evidence_is_strong = (
        obs["estimated_magnitude"] >= 7.8
        and obs["buoy_confirmed_count"] >= 3
        and obs["uncertainty"] in {"Low", "Medium"}
    )
    if scenario["is_true_tsunami"] and next_alert == "Warning" and 5 <= step <= 9 and evidence_is_strong:
        components["Timely Warning Reward"] = scale.timely_warning_reward

    warning_after_action = bool(st.session_state["warning_ever_issued"]) or (next_alert == "Warning")
    if scenario["is_true_tsunami"] and next_step >= MAX_STEPS and not warning_after_action:
        components["Missed Event Penalty"] = scale.missed_terminal_penalty

    reward = float(sum(components.values()))
    cumulative_return = float(st.session_state["cumulative_return"]) + reward

    st.session_state["current_step"] = next_step
    st.session_state["cumulative_return"] = cumulative_return
    st.session_state["current_alert_level"] = next_alert
    st.session_state["warning_ever_issued"] = warning_after_action
    st.session_state["episode_done"] = next_step >= MAX_STEPS
    st.session_state["last_action"] = action_label
    st.session_state["last_reward"] = reward
    st.session_state["last_reward_components"] = components

    st.session_state["return_history"].append({"Step": next_step, "Cumulative Return": cumulative_return})
    st.session_state["transition_log"].append(
        {
            "Time": step,
            "Estimated Magnitude": obs["estimated_magnitude"],
            "Depth Estimate": obs["depth_estimate"],
            "Buoy Confirmed Count": f"{obs['buoy_confirmed_count']}/4",
            "Uncertainty": obs["uncertainty"],
            "Previous Alert": previous_alert,
            "Action": action_label,
            "New Alert Level": next_alert,
            "Immediate Reward": round(reward, 2),
            "Cumulative Return": round(cumulative_return, 2),
        }
    )


def _action_to_alert_level(action_label: str) -> str:
    """Maps UI action text to simulator alert levels."""
    mapping = {
        "Monitor (No Alert)": "No Alert",
        "Issue Watch": "Watch",
        "Issue Warning": "Warning",
        "De-escalate/Cancel Alert": "No Alert",
    }
    return mapping[action_label]


def main() -> None:
    """Renders the standalone MDP visual simulator."""
    st.set_page_config(page_title="Tsunami MDP Visual Simulator", layout="wide")
    st.title("Tsunami MDP Visual Simulator")
    st.caption(
        "Interactive proposal-aligned MDP simulator mapped to the current project action/reward semantics."
    )

    cfg = ProjectConfig()
    reward_scale = _build_reward_scale(cfg)
    scenarios = _build_scenarios()

    st.sidebar.header("Scenario Controls")
    selected = st.sidebar.radio("Scenario", options=list(scenarios.keys()))

    if (
        "selected_scenario" not in st.session_state
        or str(st.session_state["selected_scenario"]) != selected
    ):
        _initialize_session_state(selected)

    if st.sidebar.button("Reset Simulation", use_container_width=True):
        _initialize_session_state(selected)

    obs = _get_observation(scenarios)
    current_step = int(st.session_state["current_step"])
    time_ratio = current_step / float(MAX_STEPS)

    col_event, col_actions, col_math = st.columns([1.2, 1.0, 1.2], gap="large")

    with col_event:
        st.subheader("Unfolding Event")
        e1, e2, e3 = st.columns(3)
        e1.metric("Estimated Magnitude", f"{obs['estimated_magnitude']:.2f}")
        e2.metric("Buoy Confirmed", f"{obs['buoy_confirmed_count']}/4")
        e3.metric("Uncertainty", obs["uncertainty"])

        e4, e5, e6 = st.columns(3)
        e4.metric("Depth Estimate", obs["depth_estimate"])
        e5.metric("Current Alert", str(st.session_state["current_alert_level"]))
        e6.metric("Time Elapsed", f"{current_step}/{MAX_STEPS}")

        st.progress(time_ratio)
        if bool(st.session_state["episode_done"]):
            st.warning("Episode completed. Reset simulation or change scenario to continue.")

    with col_actions:
        st.subheader("Alert Controls")
        st.markdown(
            f"**Current Alert Level:** `{st.session_state['current_alert_level']}`"
        )
        st.caption("Action space mapped to project semantics: Wait/Verify/Regional Alert/Full Alert.")

        action_taken: str | None = None
        if st.button("Monitor (No Alert)", use_container_width=True, disabled=bool(st.session_state["episode_done"])):
            action_taken = "Monitor (No Alert)"
        if st.button("Issue Watch", use_container_width=True, disabled=bool(st.session_state["episode_done"])):
            action_taken = "Issue Watch"
        if st.button("Issue Warning", use_container_width=True, disabled=bool(st.session_state["episode_done"])):
            action_taken = "Issue Warning"
        if st.button("De-escalate/Cancel Alert", use_container_width=True, disabled=bool(st.session_state["episode_done"])):
            action_taken = "De-escalate/Cancel Alert"

        if action_taken is not None:
            _apply_action(action_label=action_taken, scenarios=scenarios, scale=reward_scale)
            st.rerun()

    with col_math:
        st.subheader("Mathematical Dashboard")
        st.latex(r"r_{t+1} = r_{base} + P_{false} + P_{miss} + P_{delay} + P_{churn} + R_{timely}")
        st.metric("Immediate Reward r_(t+1)", f"{float(st.session_state['last_reward']):+.2f}")
        st.metric("Cumulative Return G_t", f"{float(st.session_state['cumulative_return']):+.2f}")
        st.caption(f"Last Action: {st.session_state['last_action']}")

        components = dict(st.session_state["last_reward_components"])
        for name, value in components.items():
            st.write(f"- **{name}:** {float(value):+.2f}")

    st.markdown("---")
    st.subheader("Cumulative Return Over Episode")
    return_frame = pd.DataFrame(st.session_state["return_history"])
    if not return_frame.empty:
        st.area_chart(return_frame.set_index("Step")[["Cumulative Return"]], use_container_width=True)

    st.subheader("Transition Log {time, state, action, reward, alert_level}")
    log_frame = pd.DataFrame(st.session_state["transition_log"])
    if log_frame.empty:
        st.info("No actions taken yet. Choose an alert action to start the episode.")
    else:
        st.dataframe(log_frame, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
