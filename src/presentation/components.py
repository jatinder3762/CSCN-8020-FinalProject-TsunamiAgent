"""Reusable Streamlit UI components for the tsunami dashboard."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from config import ProjectConfig
from src.presentation.models import DecisionStep, EpisodeTrace
from src.utils import StateCodec


class DashboardComponents:
    """Reusable render helpers for the scenario-driven dashboard."""

    ACTION_COLORS = {
        "Wait": "#475569",
        "Verify": "#0f766e",
        "Regional Alert": "#145a86",
        "Full Alert": "#b42318",
        "Hold / Monitor": "#475569",
        "Watch / Advisory": "#145a86",
        "Warning": "#b42318",
        "Cancel Alert": "#7c3e10",
    }

    OUTCOME_COLORS = {
        "Correct Full Alert": "#0f766e",
        "Correct Regional Alert": "#1d8b7f",
        "False Alarm": "#ad6704",
        "Delayed Warning": "#b42318",
        "Missed Alert": "#8a1c14",
        "Scenario In Progress": "#145a86",
    }

    @staticmethod
    def render_state_cards(step: DecisionStep) -> None:
        """Renders the current state as five metric cards."""
        card_data = step.state.to_card_mapping()
        columns = st.columns(len(card_data))
        for column, (label, value) in zip(columns, card_data.items()):
            column.metric(label, value)

    @staticmethod
    def render_state_factor_strip(step: DecisionStep) -> None:
        """Renders input factors as a full-width strip with no text truncation."""
        cards_html = "".join(
            (
                "<div class='factor-card'>"
                f"<div class='factor-label'>{label}</div>"
                f"<div class='factor-value'>{value}</div>"
                "</div>"
            )
            for label, value in step.state.to_card_mapping().items()
        )
        st.markdown(
            f"""
            <style>
            .factor-wrap {{
              border: 2px solid #f0b429;
              border-radius: 12px;
              background: linear-gradient(180deg, #fffef7 0%, #fffcee 100%);
              padding: 0.4rem 0.5rem;
              margin-bottom: 0.4rem;
              box-shadow: 0 2px 6px rgba(124, 82, 0, 0.08);
            }}
            .factor-grid {{
              display: grid;
              grid-template-columns: repeat(5, minmax(135px, 1fr));
              gap: 0.4rem;
            }}
            .factor-card {{
              border: 1px solid #b9cce1;
              border-radius: 9px;
              background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
              padding: 0.3rem 0.44rem;
              min-height: 60px;
              box-shadow: 0 1px 3px rgba(15, 23, 42, 0.07);
              overflow: visible;
            }}
            .factor-label {{
              font-size: 0.72rem;
              font-weight: 700;
              color: #506176;
              line-height: 1.15;
              margin-bottom: 0.05rem;
              white-space: normal;
              overflow: visible;
              text-overflow: clip;
            }}
            .factor-value {{
              font-size: 0.94rem;
              font-weight: 700;
              color: #123b63;
              line-height: 1.2;
              white-space: normal;
              overflow: visible;
              text-overflow: clip;
              overflow-wrap: anywhere;
            }}
            </style>
            <div class="factor-wrap">
              <div class="factor-grid">{cards_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    @staticmethod
    def render_war_room_simulation(
        trace: EpisodeTrace,
        current_step: int,
        autorun_active: bool,
    ) -> None:
        """Renders a war-room style simulation console for Auto Run presentation."""
        step = trace.get_step(current_step)

        score_map = {"Low": 1, "Medium": 2, "High": 3, "Deep": 1, "Moderate": 2, "Shallow": 3}
        magnitude_score = score_map.get(step.magnitude, 1)
        depth_score = score_map.get(step.depth, 1)
        wave_score = score_map.get(step.wave_risk, 1)
        confidence_score = score_map.get(step.confidence, 1)
        threat_index = int(round(((1.7 * magnitude_score) + (1.2 * depth_score) + (1.8 * wave_score)) / 14.1 * 100))
        confidence_pct = int(round((confidence_score / 3.0) * 100))
        buoy_count = {"Low": 1, "Medium": 2, "High": 4}.get(step.wave_risk, 1)
        eta_minutes = max(2, (trace.step_count - (current_step + 1)) * 7)
        status_text = "SIMULATION RUNNING" if autorun_active else "SIMULATION READY"
        status_color = "#22a46a" if autorun_active else "#c87a10"
        waveform_points = "".join(
            (
                f"<span class='wave-dot {'active' if idx <= current_step else ''}'></span>"
                for idx in range(trace.step_count)
            )
        )

        log_rows = []
        for row in trace.history_rows(current_step)[-4:]:
            log_rows.append(
                f"[T{row['Step']}] {row['Action']} | R={float(row['Reward']):+.1f} | CUM={float(row['Cumulative Reward']):+.1f}"
            )
        log_html = "<br/>".join(log_rows) if log_rows else "No actions yet."

        st.markdown(
            f"""
            <style>
            .war-shell {{
              border: 1px solid #2a5a83;
              border-radius: 12px;
              background: radial-gradient(circle at 12% 20%, #102d46 0%, #0c2236 48%, #081827 100%);
              box-shadow: 0 5px 14px rgba(2, 10, 20, 0.3);
              padding: 0.66rem 0.76rem;
              color: #dbe9f8;
              margin-bottom: 0.45rem;
            }}
            .war-head {{
              display:flex;
              justify-content:space-between;
              align-items:center;
              margin-bottom:0.45rem;
            }}
            .war-title {{
              font-size: 0.96rem;
              font-weight: 800;
              color: #b7dbff;
              letter-spacing: 0.02em;
            }}
            .war-status {{
              font-size:0.7rem;
              font-weight: 800;
              border-radius: 999px;
              padding: 0.13rem 0.52rem;
              background: {status_color};
              color: #f8fbff;
              border: 1px solid rgba(255, 255, 255, 0.24);
            }}
            .war-grid {{
              display:grid;
              grid-template-columns: repeat(4, minmax(0, 1fr));
              gap: 0.38rem;
              margin-bottom: 0.44rem;
            }}
            .war-card {{
              border: 1px solid #38658e;
              border-radius: 9px;
              background: linear-gradient(180deg, rgba(21, 51, 81, 0.95), rgba(14, 35, 56, 0.95));
              padding: 0.38rem 0.45rem;
              min-height: 54px;
            }}
            .war-label {{
              font-size: 0.66rem;
              color: #aec9e3;
              font-weight: 700;
              margin-bottom: 0.06rem;
            }}
            .war-value {{
              font-size: 0.95rem;
              color: #f2f8ff;
              font-weight: 800;
            }}
            .wave-track {{
              border: 1px solid #35658d;
              border-radius: 10px;
              background: linear-gradient(90deg, #102b44, #12314d);
              padding: 0.35rem 0.45rem;
              margin-bottom: 0.4rem;
            }}
            .wave-label {{
              font-size: 0.66rem;
              color: #adc8e3;
              margin-bottom: 0.15rem;
              font-weight: 700;
            }}
            .wave-dots {{
              display:flex;
              gap: 0.4rem;
            }}
            .wave-dot {{
              width: 12px;
              height: 12px;
              border-radius: 50%;
              border: 2px solid #5c83a7;
              background: #1b3a58;
            }}
            .wave-dot.active {{
              border-color: #4fc3f7;
              background: #26a0da;
            }}
            .war-log {{
              border: 1px solid #39688f;
              border-radius: 10px;
              background: rgba(9, 28, 45, 0.86);
              padding: 0.38rem 0.45rem;
              font-family: Consolas, Menlo, monospace;
              font-size: 0.67rem;
              color: #d9ecff;
              line-height: 1.45;
            }}
            </style>
            <div class="war-shell">
              <div class="war-head">
                <div class="war-title">Simulation Console</div>
                <div class="war-status">{status_text}</div>
              </div>
              <div class="war-grid">
                <div class="war-card">
                  <div class="war-label">Threat Index</div>
                  <div class="war-value">{threat_index}%</div>
                </div>
                <div class="war-card">
                  <div class="war-label">Confidence</div>
                  <div class="war-value">{confidence_pct}%</div>
                </div>
                <div class="war-card">
                  <div class="war-label">Buoy Confirmation</div>
                  <div class="war-value">{buoy_count}/4</div>
                </div>
                <div class="war-card">
                  <div class="war-label">ETA Window</div>
                  <div class="war-value">{eta_minutes} min</div>
                </div>
              </div>
              <div class="wave-track">
                <div class="wave-label">Scenario Playback Timeline</div>
                <div class="wave-dots">{waveform_points}</div>
              </div>
              <div class="war-log">{log_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    @staticmethod
    def render_action_panel(step: DecisionStep) -> None:
        """Renders the selected action badge and step status."""
        color = DashboardComponents.ACTION_COLORS.get(step.action, "#1f2937")
        st.markdown("### Selected Action")
        st.markdown(
            (
                "<div style='padding:0.55rem 0.75rem;border-radius:0.7rem;font-size:0.93rem;"
                f"background:{color};color:white;font-weight:700;text-align:center;"
                "box-shadow:0 2px 6px rgba(15, 23, 42, 0.2);border:1px solid rgba(255, 255, 255, 0.28);"
                "letter-spacing:0.01em;'>"
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
    def render_mdp_math_panel(
        step: DecisionStep,
        cumulative_reward: float,
        config: ProjectConfig,
        total_steps: int,
        compact: bool = False,
    ) -> None:
        """Renders an MDP visualizer with equations and concrete input values."""
        action_display = DashboardComponents._action_display_label(step.action)
        magnitude_value = DashboardComponents._magnitude_numeric(step.magnitude)
        buoy_count = DashboardComponents._buoy_count(step.wave_risk)
        breakdown = DashboardComponents._reward_breakdown(step, config)
        shell_padding = "0.58rem 0.6rem 0.64rem 0.6rem" if compact else "0.8rem 0.8rem 0.95rem 0.8rem"
        shell_radius = "11px" if compact else "14px"
        head_gap = "0.55rem" if compact else "0.85rem"
        head_font = "0.7rem" if compact else "0.78rem"
        action_pad = "0.45rem 0.66rem" if compact else "0.72rem 0.92rem"
        action_font = "0.84rem" if compact else "0.96rem"
        scene_height = "95px" if compact else "135px"
        city_font = "0.56rem" if compact else "0.64rem"
        buoy_size = "14px" if compact else "19px"
        buoy_bottom = "16px" if compact else "22px"
        tile_gap = "0.35rem" if compact else "0.45rem"
        tile_padding = "0.35rem 0.42rem" if compact else "0.5rem 0.55rem"
        tile_label = "0.65rem" if compact else "0.73rem"
        tile_value = "0.96rem" if compact else "1.18rem"
        mdp_inputs = pd.DataFrame(
            [
                {
                    "Input": "s_t (Magnitude, Depth, WaveRisk, Confidence, Time)",
                    "Value": str(step.state.to_card_mapping()),
                },
                {"Input": "a_t", "Value": step.action},
                {"Input": "r_t = R(s_t, a_t, s_{t+1})", "Value": f"{step.reward:+.2f}"},
                {"Input": "Cumulative Return G_t", "Value": f"{cumulative_reward:+.2f}"},
                {"Input": "Learning Rate alpha", "Value": f"{config.alpha:.3f}"},
                {"Input": "Discount gamma", "Value": f"{config.gamma:.3f}"},
            ]
        )

        st.markdown("### Tsunami MDP Visualizer")
        st.markdown(
            f"""
            <style>
            .mdp-shell {{
              border: 1px solid #c5d6e8;
              border-radius: {shell_radius};
              background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
              padding: {shell_padding};
              margin-bottom: 0.45rem;
              box-shadow: 0 2px 6px rgba(15, 23, 42, 0.08);
            }}
            .mdp-head {{
              display: flex;
              justify-content: space-between;
              align-items: center;
              margin-bottom: 0.4rem;
              color: #1f2937;
            }}
            .mdp-head-title {{
              font-weight: 700;
              font-size: 0.83rem;
              letter-spacing: 0.01em;
              color: #0f335c;
            }}
            .mdp-head-stats {{
              display: flex;
              gap: {head_gap};
              font-family: "Consolas", "Menlo", monospace;
              font-size: {head_font};
              color: #305a84;
            }}
            .mdp-card {{
              border-radius: 10px;
              background: #ffffffdd;
              border: 1px solid #d3deea;
              overflow: hidden;
            }}
            .mdp-action {{
              padding: {action_pad};
              background: linear-gradient(90deg, #0b4f8a, #0f766e);
              color: #ffffff;
              font-weight: 700;
              font-size: {action_font};
            }}
            .mdp-scene {{
              height: {scene_height};
              position: relative;
              background: linear-gradient(180deg, #dce7f2 0 64%, #8fb4dc 64% 100%);
              border-top: 1px solid #bdd0e4;
              border-bottom: 1px solid #bdd0e4;
            }}
            .mdp-city {{
              position: absolute;
              right: 0.9rem;
              top: 0.8rem;
              background: #f8fafc;
              border: 1px solid #64748b;
              border-radius: 14px;
              padding: 0.15rem 0.55rem;
              font-size: {city_font};
              font-weight: 700;
              color: #334155;
            }}
            .mdp-buoy {{
              position: absolute;
              width: {buoy_size};
              height: {buoy_size};
              border-radius: 50%;
              border: 2px solid #dc2626;
              background: #ffffff;
              box-shadow: inset 0 0 0 3px #ef4444;
              bottom: {buoy_bottom};
            }}
            .mdp-line {{
              margin: 0.65rem 0;
            }}
            .mdp-grid {{
              display: grid;
              grid-template-columns: 1fr 1fr;
              gap: {tile_gap};
            }}
            .mdp-tile {{
              border-radius: 8px;
              border-left: 3px solid #94a3b8;
              background: #f8fafc;
              padding: {tile_padding};
            }}
            .mdp-tile-label {{
              font-size: {tile_label};
              color: #475569;
            }}
            .mdp-tile-value {{
              font-size: {tile_value};
              line-height: 1.15;
              font-weight: 700;
              color: #0f335c;
            }}
            </style>
            <div class="mdp-shell">
              <div class="mdp-head">
                <div class="mdp-head-title">Tsunami MDP Visualizer</div>
                <div class="mdp-head-stats">
                  <span>STEP {step.step_index + 1}/{total_steps}</span>
                  <span>MAG {magnitude_value:.1f}</span>
                  <span>BUOYS {buoy_count}</span>
                  <span>RETURN {cumulative_reward:+.0f}</span>
                </div>
              </div>
              <div class="mdp-card">
                <div class="mdp-action">CURRENT ACTION: {action_display}</div>
                <div class="mdp-scene">
                  <div class="mdp-city">COASTAL CITY</div>
                  <div class="mdp-buoy" style="left:22%;"></div>
                  <div class="mdp-buoy" style="left:38%;"></div>
                  <div class="mdp-buoy" style="left:54%;"></div>
                  <div class="mdp-buoy" style="left:70%;"></div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if compact:
            st.caption(
                "R(s,a)=P_safety+P_false+P_delay+B_decision="
                f"{breakdown['safety_penalty']:+.1f}+{breakdown['false_alarm_penalty']:+.1f}"
                f"+{breakdown['churn_penalty']:+.1f}+{breakdown['timeliness_bonus']:+.1f}"
                f"={step.reward:+.1f}"
            )
        else:
            st.latex(
                r"R(s_t,a_t)=P_{safety}+P_{false}+P_{delay}+B_{decision}="
                + f"{breakdown['safety_penalty']:+.1f}"
                + "+"
                + f"{breakdown['false_alarm_penalty']:+.1f}"
                + "+"
                + f"{breakdown['churn_penalty']:+.1f}"
                + "+"
                + f"{breakdown['timeliness_bonus']:+.1f}"
                + "="
                + f"{step.reward:+.1f}"
            )

        st.markdown(
            f"""
            <div class="mdp-grid">
              <div class="mdp-tile" style="border-left-color:#dc2626;background:#fff1f2;">
                <div class="mdp-tile-label">Safety Penalty</div>
                <div class="mdp-tile-value">{breakdown["safety_penalty"]:+.0f}</div>
              </div>
              <div class="mdp-tile" style="border-left-color:#7c3aed;background:#f5f3ff;">
                <div class="mdp-tile-label">False Alarm Penalty</div>
                <div class="mdp-tile-value">{breakdown["false_alarm_penalty"]:+.0f}</div>
              </div>
              <div class="mdp-tile" style="border-left-color:#d97706;background:#fffbeb;">
                <div class="mdp-tile-label">Churn Penalty</div>
                <div class="mdp-tile-value">{breakdown["churn_penalty"]:+.0f}</div>
              </div>
              <div class="mdp-tile" style="border-left-color:#16a34a;background:#f0fdf4;">
                <div class="mdp-tile-label">Timeliness Bonus</div>
                <div class="mdp-tile-value">{breakdown["timeliness_bonus"]:+.0f}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if compact:
            st.caption(f"alpha={config.alpha:.3f}, gamma={config.gamma:.3f}")
        else:
            alpha_col, gamma_col = st.columns(2)
            alpha_col.metric("alpha", f"{config.alpha:.3f}")
            gamma_col.metric("gamma", f"{config.gamma:.3f}")

        reward_inputs = pd.DataFrame(
            [
                {"Reward Parameter": "reward_correct_full_alert", "Value": config.reward_correct_full_alert},
                {"Reward Parameter": "reward_correct_regional_alert", "Value": config.reward_correct_regional_alert},
                {"Reward Parameter": "reward_smart_verify", "Value": config.reward_smart_verify},
                {"Reward Parameter": "penalty_missed_dangerous_alert", "Value": config.penalty_missed_dangerous_alert},
                {"Reward Parameter": "penalty_false_full_alert", "Value": config.penalty_false_full_alert},
                {"Reward Parameter": "penalty_false_regional_alert", "Value": config.penalty_false_regional_alert},
                {"Reward Parameter": "reward_delay_per_step", "Value": config.reward_delay_per_step},
                {"Reward Parameter": "penalty_unnecessary_verify", "Value": config.penalty_unnecessary_verify},
                {"Reward Parameter": "penalty_late_wait_in_risk", "Value": config.penalty_late_wait_in_risk},
                {"Reward Parameter": "reward_partial_regional_on_high", "Value": config.reward_partial_regional_on_high},
                {"Reward Parameter": "penalty_overreaction_full_on_medium", "Value": config.penalty_overreaction_full_on_medium},
                {"Reward Parameter": "reward_safe_no_alert_low_risk", "Value": config.reward_safe_no_alert_low_risk},
            ]
        )
        if compact:
            with st.expander("MDP Equations and Input Values"):
                st.latex(r"\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)")
                st.latex(
                    r"Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha\left[r_t+\gamma\max_{a'}Q(s_{t+1},a')-Q(s_t,a_t)\right]"
                )
                st.dataframe(mdp_inputs, use_container_width=True, hide_index=True)
                st.dataframe(reward_inputs, use_container_width=True, hide_index=True)
        else:
            st.latex(r"\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)")
            st.latex(
                r"Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha\left[r_t+\gamma\max_{a'}Q(s_{t+1},a')-Q(s_t,a_t)\right]"
            )
            st.dataframe(mdp_inputs, use_container_width=True, hide_index=True)
            with st.expander("Reward Inputs Used By R(s, a, s')"):
                st.dataframe(reward_inputs, use_container_width=True, hide_index=True)

    @staticmethod
    def _action_display_label(action: str) -> str:
        """Maps action names to display labels used in the visualizer."""
        mapping = {
            "Wait": "MONITOR",
            "Verify": "VERIFY",
            "Regional Alert": "WATCH",
            "Full Alert": "WARNING",
            "Hold / Monitor": "MONITOR",
            "Watch / Advisory": "WATCH",
            "Warning": "WARNING",
            "Cancel Alert": "CANCEL",
        }
        return mapping.get(action, action.upper())

    @staticmethod
    def _magnitude_numeric(magnitude: str) -> float:
        """Maps magnitude labels to representative Richter-like values."""
        mapping = {"Low": 5.4, "Medium": 7.0, "High": 9.0}
        return float(mapping.get(magnitude, 0.0))

    @staticmethod
    def _buoy_count(wave_risk: str) -> int:
        """Maps wave-risk levels to a stylized buoy count."""
        mapping = {"Low": 1, "Medium": 2, "High": 4}
        return int(mapping.get(wave_risk, 0))

    @staticmethod
    def _reward_breakdown(step: DecisionStep, config: ProjectConfig) -> dict[str, float]:
        """Builds a simple additive reward decomposition for presentation."""
        safety_penalty = 0.0
        false_alarm_penalty = 0.0
        churn_penalty = config.reward_delay_per_step if step.action in ("Wait", "Verify") else 0.0

        status_lower = step.status.lower()
        if "delayed" in status_lower or "missed" in status_lower:
            safety_penalty = config.penalty_late_wait_in_risk

        if "false" in status_lower:
            if step.action == "Full Alert":
                false_alarm_penalty = config.penalty_false_full_alert
            else:
                false_alarm_penalty = config.penalty_false_regional_alert

        timeliness_bonus = float(step.reward) - (safety_penalty + false_alarm_penalty + churn_penalty)
        return {
            "safety_penalty": float(safety_penalty),
            "false_alarm_penalty": float(false_alarm_penalty),
            "churn_penalty": float(churn_penalty),
            "timeliness_bonus": float(timeliness_bonus),
        }

    @staticmethod
    def _safe_level_index(levels: tuple[str, ...], value: str) -> int:
        """Returns a safe level index and falls back to 0 if label is unknown."""
        try:
            return int(levels.index(value))
        except ValueError:
            return 0

    @staticmethod
    def _state_tuple(step: DecisionStep, config: ProjectConfig) -> tuple[int, int, int, int, int]:
        """Builds a discrete encoded state tuple from human-readable labels."""
        time_alias = {"Early": 0, "Mid": max(0, len(config.time_levels) // 2), "Late": len(config.time_levels) - 1}
        time_index = (
            DashboardComponents._safe_level_index(config.time_levels, step.time_label)
            if step.time_label in config.time_levels
            else int(time_alias.get(step.time_label, 0))
        )
        return (
            DashboardComponents._safe_level_index(config.magnitude_levels, step.magnitude),
            DashboardComponents._safe_level_index(config.depth_levels, step.depth),
            DashboardComponents._safe_level_index(config.wave_risk_levels, step.wave_risk),
            DashboardComponents._safe_level_index(config.confidence_levels, step.confidence),
            time_index,
        )

    @staticmethod
    def _action_id(action_name: str, config: ProjectConfig) -> int:
        """Resolves action id from action label."""
        aliases = {
            "Wait": "Hold / Monitor",
            "Verify": "Watch / Advisory",
            "Regional Alert": "Watch / Advisory",
            "Full Alert": "Warning",
            "Cancel": "Cancel Alert",
        }
        canonical_action = aliases.get(action_name, action_name)
        for action_id, label in config.action_names.items():
            if label == canonical_action:
                return int(action_id)
        return -1

    @staticmethod
    def _discounted_return_terms(trace: EpisodeTrace, start_step: int, gamma: float) -> tuple[str, float]:
        """Expands discounted return expression and computes numeric value."""
        rewards = [float(step.reward) for step in trace.steps[start_step:]]
        terms: list[str] = []
        total = 0.0
        for power, reward in enumerate(rewards):
            coeff = float(gamma) ** power
            total += coeff * reward
            terms.append(f"{coeff:.3f}*({reward:+.2f})")
        return " + ".join(terms) if terms else "0.000*(+0.00)", float(total)

    @staticmethod
    def _rl_reward_terms(step: DecisionStep, config: ProjectConfig) -> dict[str, float]:
        """Builds additive reward terms in RL language and reconciles to observed reward."""
        terms: dict[str, float] = {"Base Reward": 0.0}
        status_lower = step.status.lower()

        if step.action in ("Wait", "Verify", "Hold / Monitor"):
            terms["Delay Term"] = float(config.reward_delay_per_step)

        if step.action in ("Verify", "Watch / Advisory"):
            uncertain_signal = step.confidence in ("Low", "Medium")
            terms["Verify Utility"] = float(
                config.reward_smart_verify if uncertain_signal else config.penalty_unnecessary_verify
            )

        if step.action in ("Regional Alert", "Watch / Advisory"):
            if "false alarm" in status_lower or step.wave_risk == "Low":
                terms["Alert Outcome Term"] = float(config.penalty_false_regional_alert)
            else:
                terms["Alert Outcome Term"] = float(config.reward_correct_regional_alert)

        if step.action in ("Full Alert", "Warning"):
            if step.wave_risk == "High":
                terms["Alert Outcome Term"] = float(config.reward_correct_full_alert)
            elif step.wave_risk == "Medium":
                terms["Alert Outcome Term"] = float(config.penalty_overreaction_full_on_medium)
            else:
                terms["Alert Outcome Term"] = float(config.penalty_false_full_alert)

        if "warning delayed" in status_lower and step.action in ("Wait", "Verify", "Hold / Monitor"):
            terms["Late Risk Penalty"] = float(config.penalty_late_wait_in_risk)

        if "missed" in status_lower and step.action in ("Wait", "Verify", "Hold / Monitor"):
            terms["Missed Event Penalty"] = float(config.penalty_missed_dangerous_alert)

        subtotal = sum(terms.values())
        adjustment = float(step.reward) - float(subtotal)
        if abs(adjustment) > 1e-9:
            terms["Observed Transition Adjustment"] = adjustment

        return terms

    @staticmethod
    def render_explanation_box(
        step: DecisionStep,
        trace: EpisodeTrace | None = None,
        current_step: int | None = None,
        config: ProjectConfig | None = None,
    ) -> None:
        """Renders the explanation text and optional detailed RL step calculations."""
        st.markdown("### Explanation")
        st.info(step.explanation)
        if trace is None or current_step is None or config is None:
            return

        state_tuple = DashboardComponents._state_tuple(step, config)
        state_index = StateCodec.encode_state(state_tuple, config.state_shape)
        action_id = DashboardComponents._action_id(step.action, config)

        cumulative_now = trace.cumulative_reward_at(current_step)
        cumulative_prev = trace.cumulative_reward_at(current_step - 1) if current_step > 0 else 0.0
        discount_expr, discounted_return = DashboardComponents._discounted_return_terms(
            trace=trace,
            start_step=current_step,
            gamma=config.gamma,
        )
        reward_terms = DashboardComponents._rl_reward_terms(step, config)
        reward_symbol_map: dict[str, tuple[str, str]] = {
            "Base Reward": ("B_0", "Base reward"),
            "Delay Term": ("P_{delay}", "Delay penalty for waiting/verification"),
            "Verify Utility": ("B_{verify}", "Verification value under uncertainty"),
            "Alert Outcome Term": ("B_{alert}", "Outcome value for alert correctness"),
            "Late Risk Penalty": ("P_{late}", "Penalty for late response under risk"),
            "Missed Event Penalty": ("P_{miss}", "Terminal penalty for missed dangerous event"),
            "Observed Transition Adjustment": ("\\Delta_{obs}", "Alignment adjustment to observed scenario step reward"),
        }

        factor_df = pd.DataFrame(
            [
                {"Factor": "Magnitude", "Label": step.magnitude, "Index": state_tuple[0]},
                {"Factor": "Depth", "Label": step.depth, "Index": state_tuple[1]},
                {"Factor": "WaveRisk", "Label": step.wave_risk, "Index": state_tuple[2]},
                {"Factor": "Confidence", "Label": step.confidence, "Index": state_tuple[3]},
                {"Factor": "Time", "Label": step.time_label, "Index": state_tuple[4]},
            ]
        )

        step_math_df = pd.DataFrame(
            [
                {"Symbol": "s_t", "Computation": "State tuple", "Value": str(state_tuple)},
                {"Symbol": "index(s_t)", "Computation": "StateCodec.encode_state(s_t)", "Value": str(state_index)},
                {"Symbol": "a_t", "Computation": "Action id", "Value": f"{action_id} ({step.action})"},
                {"Symbol": "r_t", "Computation": "Immediate reward R(s_t, a_t)", "Value": f"{step.reward:+.2f}"},
                {"Symbol": "C_t", "Computation": "C_(t-1) + r_t", "Value": f"{cumulative_prev:+.2f} + ({step.reward:+.2f}) = {cumulative_now:+.2f}"},
                {"Symbol": "G_t(gamma)", "Computation": "Sum_k gamma^k * r_(t+k)", "Value": f"{discount_expr} = {discounted_return:+.2f}"},
            ]
        )

        reward_rows = [
            {
                "Symbol": reward_symbol_map.get(name, (name, name))[0],
                "Term (Text)": reward_symbol_map.get(name, (name, name))[1],
                "Value": f"{value:+.2f}",
            }
            for name, value in reward_terms.items()
            if abs(value) > 1e-9
        ]
        if not reward_rows:
            reward_rows = [{"Symbol": "B_0", "Term (Text)": "Base reward", "Value": "+0.00"}]
        reward_df = pd.DataFrame(reward_rows)

        reward_symbol_expr = " + ".join(str(row["Symbol"]) for row in reward_rows)
        reward_numeric_expr = " + ".join(f"({str(row['Value'])})" for row in reward_rows)
        symbol_legend_df = pd.DataFrame(
            [
                {"Symbol": "s_t", "Meaning (Text)": "Current state at step t"},
                {"Symbol": "a_t", "Meaning (Text)": "Action selected at step t"},
                {"Symbol": "r_t", "Meaning (Text)": "Immediate reward after taking a_t"},
                {"Symbol": "C_t", "Meaning (Text)": "Cumulative reward up to step t"},
                {"Symbol": "G_t(\\gamma)", "Meaning (Text)": "Discounted return from step t onward"},
                {"Symbol": "\\alpha", "Meaning (Text)": "Learning rate"},
                {"Symbol": "\\gamma", "Meaning (Text)": "Discount factor"},
                {"Symbol": "P_{delay}", "Meaning (Text)": "Delay penalty component"},
                {"Symbol": "B_{verify}", "Meaning (Text)": "Verification utility component"},
                {"Symbol": "B_{alert}", "Meaning (Text)": "Alert outcome reward/penalty component"},
                {"Symbol": "P_{late}", "Meaning (Text)": "Late-warning penalty component"},
                {"Symbol": "P_{miss}", "Meaning (Text)": "Missed-event terminal penalty component"},
            ]
        )

        with st.expander("RL Step Calculation Details", expanded=False):
            st.latex(
                r"Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha\left[r_t+\gamma\max_{a'}Q(s_{t+1},a')-Q(s_t,a_t)\right]"
            )
            st.latex(r"C_t = \sum_{i=0}^{t} r_i,\quad G_t(\gamma)=\sum_{k=0}^{T-t}\gamma^k r_{t+k}")
            st.latex(rf"r_t = {reward_symbol_expr} = {reward_numeric_expr} = {step.reward:+.2f}")
            st.caption(
                f"alpha = {config.alpha:.3f}, gamma = {config.gamma:.3f}, step = {current_step + 1}/{trace.step_count}"
            )
            st.markdown("#### Symbol Legend (Symbol + Text)")
            st.dataframe(symbol_legend_df, use_container_width=True, hide_index=True)
            st.markdown("#### State-to-Index Mapping")
            st.dataframe(factor_df, use_container_width=True, hide_index=True)
            st.markdown("#### Numeric RL Substitution")
            st.dataframe(step_math_df, use_container_width=True, hide_index=True)
            st.markdown("#### Reward Terms Used at This Step")
            st.dataframe(reward_df, use_container_width=True, hide_index=True)

    @staticmethod
    def render_history_table(
        trace: EpisodeTrace,
        current_step: int,
        max_rows: int | None = None,
        height: int | None = None,
    ) -> None:
        """Renders a history table through the currently visible step."""
        st.markdown("### Decision History")
        rows = trace.history_rows(current_step)
        if max_rows is not None and len(rows) > max_rows:
            rows = rows[-max_rows:]
        frame = pd.DataFrame(rows)
        st.dataframe(frame, use_container_width=True, hide_index=True, height=height)

    @staticmethod
    def render_progress_indicator(trace: EpisodeTrace, current_step: int) -> None:
        """Renders scenario progress and step count."""
        progress = float(current_step + 1) / float(trace.step_count)
        percentage = int(round(progress * 100))
        st.markdown(
            f"""
            <style>
            .progress-shell {{
              border: 1px solid #c5d6e8;
              border-radius: 10px;
              padding: 0.33rem 0.48rem 0.4rem 0.48rem;
              background: linear-gradient(180deg, #ffffff, #f8fbff);
              margin-bottom: 0.3rem;
              box-shadow: 0 2px 6px rgba(15, 23, 42, 0.06);
            }}
            .progress-top {{
              display: flex;
              justify-content: space-between;
              align-items: center;
              margin-bottom: 0.24rem;
            }}
            .progress-title {{
              font-size: 0.74rem;
              font-weight: 700;
              color: #1f2937;
            }}
            .progress-chip {{
              font-size: 0.68rem;
              font-weight: 700;
              color: #ffffff;
              background: linear-gradient(90deg, #0d3b66, #145a86);
              border-radius: 999px;
              padding: 0.1rem 0.42rem;
            }}
            .progress-track {{
              width: 100%;
              height: 9px;
              background: #e5e7eb;
              border-radius: 999px;
              overflow: hidden;
            }}
            .progress-fill {{
              height: 100%;
              border-radius: 999px;
              width: {percentage}%;
              background: linear-gradient(90deg, #0f766e, #1c8aa4, #1d5fa8);
            }}
            .progress-foot {{
              margin-top: 0.2rem;
              font-size: 0.67rem;
              color: #64748b;
            }}
            </style>
            <div class="progress-shell">
              <div class="progress-top">
                <div class="progress-title">Progress</div>
                <div class="progress-chip">{percentage}%</div>
              </div>
              <div class="progress-track"><div class="progress-fill"></div></div>
              <div class="progress-foot">Viewing step {current_step + 1} of {trace.step_count}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

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
                "<div style='padding:0.72rem 0.82rem;border-radius:0.75rem;"
                f"background:{color};color:white;border:1px solid rgba(255,255,255,0.28);"
                "box-shadow:0 3px 8px rgba(15,23,42,0.18);'>"
                f"<div style='font-size:0.78rem;opacity:0.92;'>{label}</div>"
                f"<div style='font-size:1.02rem;font-weight:700;'>{title}</div>"
                f"<div style='margin-top:0.2rem;font-size:0.8rem;'>Scenario reward: {trace.total_reward:+.2f}</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

        if not is_complete:
            st.caption("Advance to the final step to reveal the confirmed scenario outcome.")

    @staticmethod
    def render_scenario_summary(summary: dict[str, Any], description: str) -> None:
        """Renders the selected scenario summary in the sidebar."""
        rows = "".join(
            (
                "<div style='display:flex;justify-content:space-between;gap:0.4rem;"
                "padding:0.3rem 0;border-bottom:1px solid rgba(255,255,255,0.12);'>"
                f"<span style='opacity:0.9;'>{label}</span>"
                f"<span style='font-weight:700;'>{value}</span>"
                "</div>"
            )
            for label, value in summary.items()
        )
        st.sidebar.markdown(
            (
                "<div style='padding:0.65rem 0.7rem;border-radius:10px;"
                "background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.18);'>"
                "<div style='font-weight:700;font-size:0.9rem;margin-bottom:0.28rem;'>Scenario Summary</div>"
                f"<div style='font-size:0.76rem;opacity:0.95;margin-bottom:0.35rem;'>{description}</div>"
                f"{rows}</div>"
            ),
            unsafe_allow_html=True,
        )
