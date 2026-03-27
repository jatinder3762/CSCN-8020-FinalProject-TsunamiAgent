"""Reusable Streamlit UI components for the tsunami dashboard."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from src.presentation.models import DecisionStep, EpisodeTrace


class DashboardComponents:
    """Reusable render helpers for the scenario-driven dashboard."""

    ACTION_COLORS = {
        "Wait": "#6c757d",
        "Verify": "#198754",
        "Regional Alert": "#fd7e14",
        "Full Alert": "#dc3545",
    }

    OUTCOME_COLORS = {
        "Correct Full Alert": "#0f766e",
        "Correct Regional Alert": "#0f766e",
        "False Alarm": "#b45309",
        "Delayed Warning": "#b91c1c",
        "Missed Alert": "#7f1d1d",
        "Scenario In Progress": "#1d4ed8",
    }

    @staticmethod
    def render_state_cards(step: DecisionStep) -> None:
        """Renders the current state as five metric cards."""
        card_data = step.state.to_card_mapping()
        columns = st.columns(len(card_data))
        for column, (label, value) in zip(columns, card_data.items()):
            column.metric(label, value)

    @staticmethod
    def render_action_panel(step: DecisionStep) -> None:
        """Renders the selected action badge and step status."""
        color = DashboardComponents.ACTION_COLORS.get(step.action, "#1f2937")
        st.markdown("### Selected Action")
        st.markdown(
            (
                "<div style='padding:0.9rem 1rem;border-radius:0.85rem;"
                f"background:{color};color:white;font-weight:700;text-align:center;'>"
                f"{step.action}</div>"
            ),
            unsafe_allow_html=True,
        )
        st.caption(f"Status: {step.status}")

    @staticmethod
    def render_reward_panel(step: DecisionStep, cumulative_reward: float) -> None:
        """Renders the step reward and cumulative reward values."""
        st.markdown("### Reward")
        left, right = st.columns(2)
        left.metric("Step Reward", f"{step.reward:+.2f}")
        right.metric("Cumulative Reward", f"{cumulative_reward:+.2f}")

    @staticmethod
    def render_explanation_box(step: DecisionStep) -> None:
        """Renders the explanation text for the current decision."""
        st.markdown("### Explanation")
        st.info(step.explanation)

    @staticmethod
    def render_history_table(trace: EpisodeTrace, current_step: int) -> None:
        """Renders a history table through the currently visible step."""
        st.markdown("### Decision History")
        frame = pd.DataFrame(trace.history_rows(current_step))
        st.dataframe(frame, use_container_width=True, hide_index=True)

    @staticmethod
    def render_progress_indicator(trace: EpisodeTrace, current_step: int) -> None:
        """Renders scenario progress and step count."""
        st.markdown("### Progress Tracker")
        progress = float(current_step + 1) / float(trace.step_count)
        st.progress(progress)
        st.caption(f"Viewing step {current_step + 1} of {trace.step_count}")

    @staticmethod
    def render_outcome_summary(trace: EpisodeTrace, current_step: int) -> None:
        """Renders the current final-outcome summary box."""
        is_complete = current_step >= trace.last_step_index
        title = trace.outcome_label if is_complete else "Scenario In Progress"
        color = DashboardComponents.OUTCOME_COLORS.get(title, "#1d4ed8")
        label = "Final Outcome" if is_complete else "Projected Outcome"

        st.markdown("### Final Outcome Summary")
        st.markdown(
            (
                "<div style='padding:1rem 1.1rem;border-radius:0.9rem;"
                f"background:{color};color:white;'>"
                f"<div style='font-size:0.9rem;opacity:0.9;'>{label}</div>"
                f"<div style='font-size:1.25rem;font-weight:700;'>{title}</div>"
                f"<div style='margin-top:0.35rem;'>Scenario reward: {trace.total_reward:+.2f}</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

        if not is_complete:
            st.caption("Advance to the final step to reveal the confirmed scenario outcome.")

    @staticmethod
    def render_scenario_summary(summary: dict[str, Any], description: str) -> None:
        """Renders the selected scenario summary in the sidebar."""
        st.sidebar.markdown("### Scenario Summary")
        st.sidebar.write(description)
        for label, value in summary.items():
            st.sidebar.write(f"**{label}:** {value}")
