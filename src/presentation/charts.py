"""Chart helpers for scenario and training visuals."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.presentation.models import DecisionStep, EpisodeTrace


class DashboardCharts:
    """Renders compact visual insights for the dashboard."""

    _RISK_SCORE = {
        "Low": 1,
        "Deep": 1,
        "Medium": 2,
        "Moderate": 2,
        "High": 3,
        "Shallow": 3,
    }

    @classmethod
    def render_reward_timeline(cls, trace: EpisodeTrace, current_step: int) -> None:
        """Renders step reward and cumulative reward over the visible episode history."""
        st.markdown("### Reward Trajectory")
        history = trace.history_rows(current_step)
        frame = pd.DataFrame(history)
        if frame.empty:
            st.caption("Reward trajectory becomes available as steps are revealed.")
            return

        chart_frame = frame[["Step", "Reward", "Cumulative Reward"]].set_index("Step")
        st.line_chart(chart_frame, use_container_width=True)

    @classmethod
    def render_signal_profile(cls, step: DecisionStep) -> None:
        """Renders the current state risk profile as a compact bar chart."""
        st.markdown("### Signal Profile")
        profile = pd.DataFrame(
            {
                "Signal": ["Magnitude", "Depth", "WaveRisk", "Confidence"],
                "Score": [
                    cls._RISK_SCORE.get(step.magnitude, 0),
                    cls._RISK_SCORE.get(step.depth, 0),
                    cls._RISK_SCORE.get(step.wave_risk, 0),
                    cls._RISK_SCORE.get(step.confidence, 0),
                ],
            }
        ).set_index("Signal")
        st.bar_chart(profile, use_container_width=True)
        st.caption("Higher bars indicate stronger warning signals or higher confidence.")

    @classmethod
    def render_action_mix(cls, trace: EpisodeTrace, current_step: int) -> None:
        """Renders the distribution of actions taken so far in the scenario."""
        visible_steps = trace.steps[: current_step + 1]
        action_counts: dict[str, int] = {}
        for step in visible_steps:
            action_counts[step.action] = action_counts.get(step.action, 0) + 1

        st.markdown("### Action Mix")
        counts = pd.DataFrame(
            {
                "Action": list(action_counts.keys()),
                "Count": list(action_counts.values()),
            }
        ).set_index("Action")
        st.bar_chart(counts, use_container_width=True)
        st.caption("This view helps show how the policy escalates or delays over time.")
