"""Primary Streamlit dashboard for the tsunami RL decision system."""

from __future__ import annotations

from itertools import islice, product
import time
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from config import ProjectConfig
from src.agent import QLearningAgent
from src.core import RuntimeFactory
from src.environment import TsunamiAlertEnvironment
from src.presentation.charts import DashboardCharts
from src.presentation.components import DashboardComponents
from src.presentation.controller import DashboardController


class TsunamiDashboardApp:
    """Renders the scenario-driven and training dashboard experiences."""

    def __init__(self) -> None:
        """Initializes dashboard metadata."""
        self.title = "Tsunami RL Decision Dashboard"
        self.mode_key = "dashboard_mode"
        self.training_summary_key = "dashboard_training_summary"
        self.autorun_state_key = "dashboard_autorun_active"
        self.demo_config = ProjectConfig()

    def run(self) -> None:
        """Builds the Streamlit page and interactive flow."""
        st.set_page_config(page_title=self.title, page_icon=":ocean:", layout="wide")
        controller = DashboardController(st.session_state)
        controller.initialize()
        self._apply_compact_page_style()

        self._render_header()
        sidebar_state = self._render_sidebar(controller)
        if self.autorun_state_key not in st.session_state:
            st.session_state[self.autorun_state_key] = False

        if sidebar_state["mode"] == "Live Training":
            self._render_training_mode(sidebar_state["training_controls"])
            return

        if bool(sidebar_state["auto_run_clicked"]):
            self._autorun_scenario(controller, delay_seconds=0.8)
            return
        self._render_demo_view(controller)

    def _render_header(self) -> None:
        """Renders the page heading and overview copy."""
        st.title(self.title)
        st.caption(
            "Use Scenario Demo for presentation walkthroughs or Live Training to watch the RL agent learn in real time."
        )

    def _apply_compact_page_style(self) -> None:
        """Applies compact page spacing so scenario panels fit in one screen."""
        st.markdown(
            """
            <style>
            :root {
              --gov-bg: #eef3f9;
              --gov-panel: #ffffff;
              --gov-panel-soft: #f7faff;
              --gov-border: #c1cfde;
              --gov-primary: #0d3b66;
              --gov-primary-2: #145a86;
              --gov-success: #0f766e;
              --gov-accent: #f0b429;
              --gov-danger: #b42318;
              --gov-text: #0f172a;
              --gov-muted: #475569;
              --gov-sidebar-bg: #0f2f4d;
              --gov-sidebar-bg-2: #133a5e;
              --gov-sidebar-border: #2a577f;
              --gov-sidebar-text: #e9f2fb;
            }
            [data-testid="stAppViewContainer"] {
              background: radial-gradient(circle at 14% 0%, #f2f7ff 0%, #f6f9fd 56%, #eef3f9 100%) !important;
            }
            [data-testid="stHeader"] {
              background: transparent !important;
            }
            .block-container {
              padding-top: 0.45rem !important;
              padding-bottom: 0.45rem !important;
              color: var(--gov-text) !important;
            }
            h1, h2, h3 {
              margin-top: 0.24rem !important;
              margin-bottom: 0.2rem !important;
              color: var(--gov-primary) !important;
            }
            h1 {
              font-size: 1.66rem !important;
              letter-spacing: 0.01em;
            }
            h2 {
              font-size: 1.22rem !important;
            }
            h3 {
              font-size: 0.95rem !important;
            }
            p, label, li {
              color: var(--gov-text) !important;
            }
            div[data-testid="stMetric"] {
              background: linear-gradient(180deg, #ffffff 0%, var(--gov-panel-soft) 100%) !important;
              border: 1px solid var(--gov-border) !important;
              border-radius: 8px;
              padding: 0.16rem 0.3rem;
              box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08);
            }
            div[data-testid="stMetricLabel"] p {
              font-size: 0.69rem !important;
              white-space: normal !important;
              overflow: visible !important;
              text-overflow: clip !important;
              color: var(--gov-muted) !important;
            }
            div[data-testid="stMetricValue"] {
              font-size: 0.84rem !important;
            }
            div[data-testid="stMetricValue"] > div {
              font-size: 1.62rem !important;
              line-height: 1.0 !important;
              white-space: normal !important;
              overflow: visible !important;
              text-overflow: clip !important;
              word-break: normal !important;
              color: #113b63 !important;
            }
            div[data-testid="stCaptionContainer"] p {
              font-size: 0.68rem !important;
              line-height: 1.2 !important;
              color: var(--gov-muted) !important;
            }
            div[data-testid="stAlert"] {
              padding: 0.5rem 0.62rem !important;
              background: #edf4ff !important;
              border: 1px solid #b8cde6 !important;
              color: var(--gov-text) !important;
              border-radius: 8px !important;
            }
            div[data-baseweb="tab-list"] {
              gap: 0.25rem !important;
            }
            div[data-baseweb="tab-list"] button {
              padding-top: 0.22rem !important;
              padding-bottom: 0.22rem !important;
              font-size: 0.78rem !important;
              background: #f3f7fc !important;
              color: #26496f !important;
              border: 1px solid #bfd2e7 !important;
              border-radius: 999px !important;
            }
            div[data-baseweb="tab-list"] button[aria-selected="true"] {
              background: #dceafb !important;
              border-color: #8cb1d8 !important;
              color: var(--gov-primary) !important;
              font-weight: 700 !important;
            }
            div[data-testid="stButton"] > button {
              min-height: 2rem !important;
              padding: 0.2rem 0.5rem !important;
              border-radius: 999px !important;
              font-size: 0.84rem !important;
              border: 1px solid #9bb3cc !important;
              background: #fdfefe !important;
              color: #113b63 !important;
              font-weight: 700 !important;
              letter-spacing: 0.01em;
            }
            div[data-testid="stButton"] > button * {
              color: inherit !important;
            }
            div[data-testid="stButton"] > button[kind="primary"] {
              background: linear-gradient(90deg, var(--gov-primary), var(--gov-primary-2)) !important;
              color: #ffffff !important;
              border: 1px solid #0f4777 !important;
              font-weight: 700 !important;
              text-shadow: 0 1px 0 rgba(0, 0, 0, 0.18);
            }
            div[data-testid="stButton"] > button:hover {
              border-color: #6f91b6 !important;
              background: #eef4fb !important;
            }
            div[data-testid="stButton"] > button[kind="primary"]:hover {
              background: linear-gradient(90deg, #0a3158, #11496d) !important;
              color: #ffffff !important;
            }
            div[data-testid="stButton"] > button:focus-visible {
              outline: 2px solid var(--gov-accent) !important;
              outline-offset: 1px !important;
            }
            div[data-testid="stButton"] > button:disabled {
              opacity: 0.62 !important;
              color: #4b5b6d !important;
            }
            div[data-testid="stExpander"] details {
              border: 1px solid var(--gov-border) !important;
              border-radius: 10px !important;
              background: #ffffff !important;
              box-shadow: 0 1px 3px rgba(15, 23, 42, 0.07) !important;
            }
            div[data-testid="stSlider"] [role="slider"] {
              background: var(--gov-primary) !important;
            }
            div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div:first-child {
              background: #d8e3ef !important;
            }
            div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div:nth-child(2) {
              background: var(--gov-success) !important;
            }
            [data-testid="stSidebar"] {
              background: linear-gradient(180deg, var(--gov-sidebar-bg) 0%, var(--gov-sidebar-bg-2) 100%) !important;
              border-right: 1px solid var(--gov-sidebar-border) !important;
            }
            [data-testid="stSidebar"] * {
              color: var(--gov-sidebar-text) !important;
            }
            [data-testid="stSidebar"] h1,
            [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] h3 {
              color: #f2f8ff !important;
            }
            [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p,
            [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] li,
            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] span {
              color: var(--gov-sidebar-text) !important;
            }
            [data-testid="stSidebar"] div[data-testid="stSelectbox"] > div,
            [data-testid="stSidebar"] div[data-testid="stNumberInput"] input,
            [data-testid="stSidebar"] div[data-testid="stSlider"] {
              background: rgba(255, 255, 255, 0.09) !important;
              border: 1px solid rgba(197, 220, 244, 0.28) !important;
              border-radius: 8px !important;
            }
            [data-testid="stSidebar"] div[data-baseweb="input"] > div {
              background: rgba(255, 255, 255, 0.1) !important;
              border: 1px solid rgba(198, 219, 241, 0.4) !important;
              border-radius: 10px !important;
            }
            [data-testid="stSidebar"] div[data-baseweb="input"] input {
              color: #f4f9ff !important;
              -webkit-text-fill-color: #f4f9ff !important;
              caret-color: #f4f9ff !important;
            }
            [data-testid="stSidebar"] div[data-baseweb="input"] svg {
              fill: #dcecff !important;
            }
            [data-testid="stSidebar"] div[data-baseweb="select"] > div {
              background: rgba(255, 255, 255, 0.1) !important;
              border: 1px solid rgba(198, 219, 241, 0.4) !important;
              border-radius: 10px !important;
              color: #f4f9ff !important;
            }
            [data-testid="stSidebar"] div[data-baseweb="select"] span,
            [data-testid="stSidebar"] div[data-baseweb="select"] div {
              color: #f4f9ff !important;
            }
            [data-testid="stSidebar"] div[data-baseweb="select"] svg {
              fill: #dcecff !important;
            }
            [data-testid="stSidebar"] div[data-baseweb="select"] input {
              color: #f4f9ff !important;
              -webkit-text-fill-color: #f4f9ff !important;
            }
            [data-testid="stSidebar"] div[data-testid="stButton"] > button {
              border-color: #8cb0d2 !important;
              background: linear-gradient(180deg, rgba(255, 255, 255, 0.17), rgba(255, 255, 255, 0.12)) !important;
              color: #f4f9ff !important;
              font-size: 0.82rem !important;
              font-weight: 700 !important;
              text-shadow: 0 1px 0 rgba(0, 0, 0, 0.22) !important;
            }
            [data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primary"] {
              background: linear-gradient(90deg, #1f72ad, #0f5b95) !important;
              border-color: #2d88c6 !important;
              color: #ffffff !important;
            }
            [data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
              background: linear-gradient(180deg, rgba(255, 255, 255, 0.22), rgba(255, 255, 255, 0.16)) !important;
              border-color: #a6c5e3 !important;
            }
            [data-testid="stSidebar"] div[data-testid="stButton"] > button[kind="primary"]:hover {
              background: linear-gradient(90deg, #2386c8, #1666a2) !important;
              border-color: #56a8dc !important;
            }
            [data-testid="stSidebar"] div[data-testid="stButton"] > button:disabled {
              opacity: 0.9 !important;
              background: linear-gradient(180deg, rgba(255, 255, 255, 0.18), rgba(255, 255, 255, 0.12)) !important;
              border-color: rgba(188, 210, 232, 0.65) !important;
              color: #d9e8f8 !important;
              -webkit-text-fill-color: #d9e8f8 !important;
              text-shadow: none !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

    def _render_sidebar(self, controller: DashboardController) -> dict[str, Any]:
        """Renders sidebar controls for scenario demo and live training modes."""
        st.sidebar.header("Controls")

        if self.mode_key not in st.session_state:
            st.session_state[self.mode_key] = "Scenario Demo"

        mode = st.sidebar.radio(
            "View",
            options=("Scenario Demo", "Live Training"),
            key=self.mode_key,
        )

        training_controls = {
            "seed": 42,
            "train_episodes": 1200,
        }
        auto_run_clicked = False

        if mode == "Scenario Demo":
            selected_name = st.sidebar.selectbox(
                "Scenario Selector",
                options=controller.scenario_names,
                key=controller.SCENARIO_WIDGET_KEY,
            )
            if selected_name != controller.selected_scenario_name:
                controller.select_scenario(selected_name)
            auto_run_clicked = bool(self._render_navigation(controller)["auto_run_clicked"])

            DashboardComponents.render_scenario_summary(
                summary=controller.scenario_summary(),
                description=controller.current_trace.summary,
            )

            st.sidebar.markdown("### How To Read")
            st.sidebar.write("Use Previous and Next to move through the demo one step at a time.")
            st.sidebar.write("Use Auto Run to process the selected scenario to its final result.")
            st.sidebar.write("Use Reset to jump back to the first step.")
            st.sidebar.write("Use Floating Step Controls to keep navigation available while you scroll.")
            st.sidebar.write("Review the charts to see reward momentum, signal strength, and action mix.")
        else:
            seed = st.sidebar.number_input("Training Seed", min_value=1, max_value=99999, value=42, step=1)
            train_episodes = st.sidebar.slider("Training Episodes", 100, 5000, 1200, 100)
            training_controls = {
                "seed": int(seed),
                "train_episodes": int(train_episodes),
            }

            st.sidebar.markdown("### Live Training")
            st.sidebar.write("Start a fresh training session and watch reward and epsilon update as episodes run.")
            st.sidebar.write("The trained Q-table is saved to `outputs/models/q_table.npy` when training finishes.")

        return {
            "mode": mode,
            "training_controls": training_controls,
            "auto_run_clicked": auto_run_clicked,
        }

    def _render_navigation(self, controller: DashboardController) -> dict[str, bool]:
        """Renders fixed step-navigation controls in the sidebar."""
        total = controller.current_trace.step_count
        st.sidebar.markdown("### Floating Step Controls")
        st.sidebar.caption("Fixed navigation menu")
        top_row = st.sidebar.columns(2)
        bottom_row = st.sidebar.columns(2)

        with top_row[0]:
            previous_clicked = st.button("Previous", key="floating_previous", width="stretch")
        with top_row[1]:
            next_clicked = st.button("Next", key="floating_next", width="stretch")
        with bottom_row[0]:
            auto_run_clicked = st.button(
                "Auto Run",
                key="floating_auto_run",
                width="stretch",
                type="primary",
            )
        with bottom_row[1]:
            reset_clicked = st.button("Reset", key="floating_reset", width="stretch")

        if previous_clicked:
            controller.previous_step()
        if next_clicked:
            controller.next_step()
        if reset_clicked:
            controller.reset()

        current = controller.current_step + 1
        st.sidebar.markdown(
            f"""
            <div style="border:1px solid rgba(203,225,247,0.42);background:linear-gradient(180deg,rgba(255,255,255,0.16),rgba(255,255,255,0.09));border-radius:10px;
                        padding:0.3rem 0.45rem;margin-top:0.3rem;">
              <div style="font-size:0.66rem;color:#d5e8fc;font-weight:700;">Current Step</div>
              <div style="font-size:1.12rem;line-height:1.05;color:#ffffff;font-weight:800;">
                {current}/{total}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        return {"auto_run_clicked": auto_run_clicked}

    def _render_demo_view(self, controller: DashboardController) -> None:
        """Renders a compact 2x2 scenario layout to reduce page scrolling."""
        step = controller.current_decision
        trace = controller.current_trace
        cumulative_reward = trace.cumulative_reward_at(controller.current_step)

        st.caption("Section 1/4: Input Factors (Full Width)")
        DashboardComponents.render_state_factor_strip(step)

        st.caption("Section 2/4: Current Decision and Reward")
        section_left, section_right = st.columns(2, gap="small")
        with section_left:
            DashboardComponents.render_action_panel(step)
        with section_right:
            DashboardComponents.render_reward_panel(step, cumulative_reward)
        DashboardComponents.render_outcome_summary(trace, controller.current_step)

        st.caption("Simulation Console (Integrated Auto Run)")
        DashboardComponents.render_war_room_simulation(
            trace=trace,
            current_step=controller.current_step,
            autorun_active=bool(st.session_state.get(self.autorun_state_key, False)),
        )

        st.caption("Section 4/4: Explanation and History")
        DashboardComponents.render_explanation_box(
            step,
            trace=trace,
            current_step=controller.current_step,
            config=self.demo_config,
        )
        DashboardComponents.render_history_table(
            trace,
            controller.current_step,
            max_rows=4,
            height=210,
        )

        st.caption("Charts")
        graph_left, graph_mid, graph_right = st.columns(3, gap="small")
        with graph_left:
            with st.expander("Reward Trend", expanded=True):
                DashboardCharts.render_reward_timeline(trace, controller.current_step)
        with graph_mid:
            with st.expander("Signal Profile", expanded=True):
                DashboardCharts.render_signal_profile(step)
        with graph_right:
            with st.expander("Action Mix", expanded=True):
                DashboardCharts.render_action_mix(trace, controller.current_step)

    def _autorun_scenario(self, controller: DashboardController, delay_seconds: float) -> None:
        """Animates the current scenario to its final result and leaves navigation intact."""
        playback = st.empty()
        st.session_state[self.autorun_state_key] = True

        try:
            while True:
                with playback.container():
                    _ = self._render_demo_view(controller)

                if controller.current_step >= controller.current_trace.last_step_index:
                    break

                time.sleep(delay_seconds)
                controller.set_step(controller.current_step + 1, sync_widget=False)
        finally:
            st.session_state[self.autorun_state_key] = False

        st.rerun()

    def _render_training_mode(self, controls: dict[str, int]) -> None:
        """Renders the live training experience inside Streamlit."""
        st.markdown("## Live Training Session")
        st.write("Run training and monitor policy quality with simple operational charts.")

        cfg = ProjectConfig(training_episodes=controls["train_episodes"], random_seed=controls["seed"])
        st.caption("Section 1/3: Live Training Monitor")
        run_training = st.button("Start Live Training", width="stretch", type="primary")

        progress = st.progress(0)
        status = st.empty()
        metric_columns = st.columns(4, gap="small")
        metric_slots = [column.empty() for column in metric_columns]
        chart_left, chart_mid, chart_right = st.columns(3, gap="small")
        quality_slot = chart_left.empty()
        safety_slot = chart_mid.empty()
        action_slot = chart_right.empty()
        log_caption = st.empty()
        log_slot = st.empty()

        records: list[dict[str, Any]] = []
        step_records: list[dict[str, Any]] = []
        action_counts: dict[str, int] = {name: 0 for name in cfg.action_names.values()}

        if not run_training:
            self._render_training_monitor_snapshot(
                records=records,
                action_counts=action_counts,
                total_episodes=cfg.training_episodes,
                metric_slots=metric_slots,
                quality_slot=quality_slot,
                safety_slot=safety_slot,
                action_slot=action_slot,
                log_caption=log_caption,
                log_slot=log_slot,
            )
            st.caption("Section 2/3: Training Dataset and Model Preview")
            self._render_training_data_preview(cfg)
            if self.training_summary_key in st.session_state:
                st.caption("Section 3/3: Most Recent Session Summary")
                st.json(st.session_state[self.training_summary_key])
            return

        runtime = RuntimeFactory.build_training_bundle(
            training_episodes=cfg.training_episodes,
            seed=cfg.random_seed,
        )
        env = runtime.environment
        agent = runtime.agent

        refresh = max(10, cfg.training_episodes // 60)

        for episode in range(1, cfg.training_episodes + 1):
            result, episode_steps = self._run_training_episode(env, agent, cfg)
            result["episode"] = episode
            records.append(result)
            for step_record in episode_steps:
                step_record["episode"] = episode
            step_records.extend(episode_steps)
            final_action = str(result.get("final_action", "Unknown"))
            action_counts[final_action] = action_counts.get(final_action, 0) + 1
            agent.decay_epsilon()
            records[-1]["epsilon"] = agent.epsilon

            if episode % refresh == 0 or episode == cfg.training_episodes:
                progress.progress(min(100, int((episode / cfg.training_episodes) * 100)))
                status.info(
                    f"Progress {episode}/{cfg.training_episodes} | "
                    f"Current Episode Score {result['total_reward']:+.1f} | Exploration {agent.epsilon:.3f}"
                )
                self._render_training_monitor_snapshot(
                    records=records,
                    action_counts=action_counts,
                    total_episodes=cfg.training_episodes,
                    metric_slots=metric_slots,
                    quality_slot=quality_slot,
                    safety_slot=safety_slot,
                    action_slot=action_slot,
                    log_caption=log_caption,
                    log_slot=log_slot,
                )

        frame = pd.DataFrame(records)
        summary = {
            "episodes": len(frame),
            "average_reward": round(float(frame["total_reward"].mean()), 3),
            "average_steps": round(float(frame["steps"].mean()), 3),
            "correct_alert_rate": round(float(frame["correct"].mean()), 3),
            "false_alert_rate": round(float(frame["false"].mean()), 3),
            "missed_alert_rate": round(float(frame["missed"].mean()), 3),
            "final_epsilon": round(float(agent.epsilon), 4),
            "model_path": str(cfg.models_dir / "q_table.npy"),
            "live_episode_summary_path": str(cfg.logs_dir / "training_live_session.csv"),
            "live_step_trace_path": str(cfg.logs_dir / "training_live_steps.csv"),
        }

        cfg.logs_dir.mkdir(parents=True, exist_ok=True)
        cfg.models_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(cfg.logs_dir / "training_live_session.csv", index=False)
        pd.DataFrame(step_records).to_csv(cfg.logs_dir / "training_live_steps.csv", index=False)
        agent.save_q_table(cfg.models_dir / "q_table.npy")

        st.success("Training completed. Updated model and training log were saved.")
        st.caption("Section 2/3: Training Dataset and Model Preview")
        self._render_training_data_preview(cfg)
        st.caption("Section 3/3: Final Session Summary")
        st.json(summary)
        st.session_state[self.training_summary_key] = summary

    def _render_training_monitor_snapshot(
        self,
        records: list[dict[str, Any]],
        action_counts: dict[str, int],
        total_episodes: int,
        metric_slots: list[Any],
        quality_slot: Any,
        safety_slot: Any,
        action_slot: Any,
        log_caption: Any,
        log_slot: Any,
    ) -> None:
        """Renders non-technical live training charts and latest outcomes."""
        if not records:
            metric_slots[0].metric("Episodes Completed", f"0/{total_episodes}")
            metric_slots[1].metric("Average Decision Score", "+0.0")
            metric_slots[2].metric("Safe Decision Rate", "0.0%")
            metric_slots[3].metric("False + Missed Rate", "0.0%")
            quality_slot.info("Decision Quality trend will appear after training starts.")
            safety_slot.info("Safety trend will appear after training starts.")
            action_slot.info("Action mix will appear after training starts.")
            log_caption.markdown("#### Latest Episode Outcomes")
            log_slot.caption("No episodes yet. Click `Start Live Training` to begin.")
            return

        frame = pd.DataFrame(records)
        completed = len(frame)
        window = min(50, completed)
        # Rolling quality smooths noisy episode rewards so non-technical viewers can read trend direction.
        rolling_quality = frame["total_reward"].rolling(window=window, min_periods=1).mean()
        average_quality = frame["total_reward"].expanding().mean()
        safe_rate = float(frame["correct"].mean()) * 100.0
        false_rate = float(frame["false"].mean()) * 100.0
        missed_rate = float(frame["missed"].mean()) * 100.0

        metric_slots[0].metric("Episodes Completed", f"{completed}/{total_episodes}")
        metric_slots[1].metric("Average Decision Score", f"{float(average_quality.iloc[-1]):+.1f}")
        metric_slots[2].metric("Safe Decision Rate", f"{safe_rate:.1f}%")
        metric_slots[3].metric("False + Missed Rate", f"{(false_rate + missed_rate):.1f}%")

        quality_frame = pd.DataFrame(
            {
                "Episode": frame["episode"],
                "Decision Quality (Rolling)": rolling_quality,
                "Decision Quality (Overall Avg)": average_quality,
            }
        ).set_index("Episode")
        quality_slot.area_chart(quality_frame, width="stretch")

        safety_frame = pd.DataFrame(
            {
                "Episode": frame["episode"],
                # Expanding rates show overall operational behavior as the session evolves.
                "Safe %": frame["correct"].expanding().mean() * 100.0,
                "False Alarm %": frame["false"].expanding().mean() * 100.0,
                "Missed %": frame["missed"].expanding().mean() * 100.0,
            }
        ).set_index("Episode")
        safety_slot.area_chart(safety_frame, width="stretch")

        action_frame = pd.DataFrame(
            {
                "Action": list(action_counts.keys()),
                "Count": list(action_counts.values()),
            }
        ).set_index("Action")
        action_slot.bar_chart(action_frame, width="stretch")

        log_caption.markdown("#### Latest Episode Outcomes")
        recent_frame = frame[["episode", "total_reward", "steps", "final_action", "correct", "false", "missed"]].tail(10)
        recent_frame = recent_frame.rename(
            columns={
                "episode": "Episode",
                "total_reward": "Score",
                "steps": "Steps",
                "final_action": "Final Action",
                "correct": "Safe",
                "false": "False Alarm",
                "missed": "Missed",
            }
        )
        recent_frame["Safe"] = recent_frame["Safe"].map(lambda value: "Yes" if bool(value) else "No")
        recent_frame["False Alarm"] = recent_frame["False Alarm"].map(lambda value: "Yes" if bool(value) else "No")
        recent_frame["Missed"] = recent_frame["Missed"].map(lambda value: "Yes" if bool(value) else "No")
        recent_frame["Score"] = recent_frame["Score"].map(lambda value: f"{float(value):+.1f}")
        log_slot.dataframe(recent_frame, width="stretch", hide_index=True, height=250)

    def _run_training_episode(
        self,
        env: TsunamiAlertEnvironment,
        agent: QLearningAgent,
        cfg: ProjectConfig,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Runs one Q-learning training episode for the live dashboard."""
        state_idx = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        info: dict[str, Any] = {}
        trace: list[dict[str, Any]] = []

        while not done and steps < cfg.max_steps_per_episode:
            state_tuple = env.index_to_state(state_idx)
            action = agent.choose_action(
                state_idx,
                training=True,
                valid_actions=env.get_valid_actions(),
            )
            next_state_idx, reward, done, info = env.step(action)
            next_state_tuple = env.index_to_state(next_state_idx)
            agent.update(state_idx, action, reward, next_state_idx, done)
            steps += 1
            trace.append(
                self._build_live_training_step_record(
                    step=steps,
                    state_idx=state_idx,
                    state_tuple=state_tuple,
                    action=action,
                    reward=reward,
                    next_state_idx=next_state_idx,
                    next_state_tuple=next_state_tuple,
                    info=info,
                    cfg=cfg,
                    done=done,
                )
            )
            state_idx = next_state_idx
            total_reward += reward

        return (
            {
                "total_reward": float(total_reward),
                "steps": int(steps),
                "final_action": str(info.get("action_meaning", "Unknown")),
                "correct": bool(info.get("alert_correct", False)),
                "false": bool(info.get("false_alert", False)),
                "missed": bool(info.get("missed_alert", False)),
            },
            trace,
        )

    @staticmethod
    def _build_live_training_step_record(
        step: int,
        state_idx: int,
        state_tuple: tuple[int, int, int, int, int],
        action: int,
        reward: float,
        next_state_idx: int,
        next_state_tuple: tuple[int, int, int, int, int],
        info: dict[str, Any],
        cfg: ProjectConfig,
        done: bool,
    ) -> dict[str, Any]:
        """Builds one detailed live-training step record for CSV export."""
        reward_terms = info.get("reward_terms", {})
        return {
            "step": int(step),
            "state_index": int(state_idx),
            "magnitude": cfg.magnitude_levels[state_tuple[0]],
            "depth": cfg.depth_levels[state_tuple[1]],
            "wave_risk": cfg.wave_risk_levels[state_tuple[2]],
            "confidence": cfg.confidence_levels[state_tuple[3]],
            "time": cfg.time_levels[state_tuple[4]],
            "action_id": int(action),
            "action_text": cfg.action_names.get(action, f"Unknown({action})"),
            "reward": float(reward),
            "next_state_index": int(next_state_idx),
            "next_magnitude": cfg.magnitude_levels[next_state_tuple[0]],
            "next_depth": cfg.depth_levels[next_state_tuple[1]],
            "next_wave_risk": cfg.wave_risk_levels[next_state_tuple[2]],
            "next_confidence": cfg.confidence_levels[next_state_tuple[3]],
            "next_time": cfg.time_levels[next_state_tuple[4]],
            "actual_risk_level": str(info.get("actual_risk_level", "Unknown")),
            "current_alert_level": str(info.get("current_alert_level", "Unknown")),
            "alert_correct": bool(info.get("alert_correct", False)),
            "false_alert": bool(info.get("false_alert", False)),
            "missed_alert": bool(info.get("missed_alert", False)),
            "done": bool(done),
            "reward_term_base": float(reward_terms.get("r_base", 0.0)),
            "reward_term_invalid": float(reward_terms.get("r_invalid", 0.0)),
            "reward_term_churn": float(reward_terms.get("r_churn", 0.0)),
            "reward_term_evidence": float(reward_terms.get("r_evidence", 0.0)),
            "reward_term_overreact": float(reward_terms.get("r_overreact", 0.0)),
            "reward_term_cancel": float(reward_terms.get("r_cancel", 0.0)),
            "reward_term_terminal": float(reward_terms.get("r_terminal", 0.0)),
        }

    def _render_training_data_preview(self, cfg: ProjectConfig) -> None:
        """Shows a compact preview of the generated training state space and saved model."""
        st.markdown("### Training Data Preview")
        st.caption(
            "Training uses generated discrete state combinations from the environment. "
            "`outputs/models/q_table.npy` is the learned model output, not the raw training dataset."
        )

        metric_columns = st.columns(3)
        metric_columns[0].metric("Total States", str(cfg.state_size))
        metric_columns[1].metric("Actions", str(cfg.action_size))
        metric_columns[2].metric("Max Steps / Episode", str(cfg.max_steps_per_episode))

        preview_left, preview_right = st.columns([1.8, 1.0])
        with preview_left:
            st.markdown("#### State Space Sample")
            st.dataframe(
                self._build_training_state_preview(cfg),
                width="stretch",
                hide_index=True,
            )
        with preview_right:
            st.markdown("#### Action Mapping")
            st.dataframe(
                self._build_action_preview(cfg),
                width="stretch",
                hide_index=True,
            )

        q_table_preview = self._build_q_table_preview(cfg)
        if q_table_preview is not None:
            st.markdown("#### Saved Q-Table Preview")
            st.dataframe(q_table_preview, width="stretch", hide_index=True)

    def _build_training_state_preview(self, cfg: ProjectConfig, limit: int = 12) -> pd.DataFrame:
        """Builds a small preview of the generated state combinations used for training."""
        environment = TsunamiAlertEnvironment(cfg, seed=cfg.random_seed)
        rows: list[dict[str, Any]] = []
        level_ranges = (
            range(len(cfg.magnitude_levels)),
            range(len(cfg.depth_levels)),
            range(len(cfg.wave_risk_levels)),
            range(len(cfg.confidence_levels)),
            range(len(cfg.time_levels)),
        )

        for state_tuple in islice(product(*level_ranges), limit):
            state_index = environment.state_to_index(tuple(int(value) for value in state_tuple))
            rows.append(
                {
                    "State Index": state_index,
                    "Magnitude": cfg.magnitude_levels[state_tuple[0]],
                    "Depth": cfg.depth_levels[state_tuple[1]],
                    "WaveRisk": cfg.wave_risk_levels[state_tuple[2]],
                    "Confidence": cfg.confidence_levels[state_tuple[3]],
                    "Time": cfg.time_levels[state_tuple[4]],
                }
            )

        return pd.DataFrame(rows)

    def _build_action_preview(self, cfg: ProjectConfig) -> pd.DataFrame:
        """Builds a small action reference table for the live training page."""
        return pd.DataFrame(
            [
                {"Action ID": action_id, "Action": action_name}
                for action_id, action_name in cfg.action_names.items()
            ]
        )

    def _build_q_table_preview(self, cfg: ProjectConfig, limit: int = 8) -> pd.DataFrame | None:
        """Builds a compact preview of the saved Q-table when available."""
        model_path = cfg.models_dir / "q_table.npy"
        if not model_path.exists():
            return None

        q_table = np.load(model_path)
        preview = pd.DataFrame(q_table[:limit], columns=list(cfg.action_names.values()))
        preview.insert(0, "State Index", list(range(len(preview))))
        return preview.round(3)
