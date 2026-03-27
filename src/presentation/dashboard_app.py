"""Primary Streamlit dashboard for the tsunami RL decision system."""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
import streamlit as st

from config import ProjectConfig
from src.agent import QLearningAgent
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

    def run(self) -> None:
        """Builds the Streamlit page and interactive flow."""
        st.set_page_config(page_title=self.title, page_icon=":ocean:", layout="wide")
        controller = DashboardController(st.session_state)
        controller.initialize()

        self._render_header()
        sidebar_state = self._render_sidebar(controller)

        if sidebar_state["mode"] == "Live Training":
            self._render_training_mode(sidebar_state["training_controls"])
            return

        navigation_state = self._render_navigation(controller)
        if navigation_state["auto_run_clicked"]:
            self._autorun_scenario(controller, delay_seconds=0.8)
            return

        self._render_demo_view(controller)

    def _render_header(self) -> None:
        """Renders the page heading and overview copy."""
        st.title(self.title)
        st.caption(
            "Use Scenario Demo for presentation walkthroughs or Live Training to watch the RL agent learn in real time."
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

        if mode == "Scenario Demo":
            selected_name = st.sidebar.selectbox(
                "Scenario Selector",
                options=controller.scenario_names,
                key=controller.SCENARIO_WIDGET_KEY,
            )
            if selected_name != controller.selected_scenario_name:
                controller.select_scenario(selected_name)

            DashboardComponents.render_scenario_summary(
                summary=controller.scenario_summary(),
                description=controller.current_trace.summary,
            )

            st.sidebar.markdown("### How To Read")
            st.sidebar.write("Use Previous and Next to move through the demo one step at a time.")
            st.sidebar.write("Use Auto Run to process the selected scenario to its final result.")
            st.sidebar.write("Use Reset to jump back to the first step or the slider to jump directly.")
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
        }

    def _render_navigation(self, controller: DashboardController) -> dict[str, bool]:
        """Renders the step slider and previous/next controls."""
        st.markdown("## Step Navigation")
        top_columns = st.columns([1, 1, 1, 1, 1.1])
        auto_run_clicked = False

        with top_columns[0]:
            if st.button("Previous", use_container_width=True):
                controller.previous_step()
        with top_columns[1]:
            if st.button("Next", use_container_width=True):
                controller.next_step()
        with top_columns[2]:
            if st.button("Auto Run", use_container_width=True):
                auto_run_clicked = True
        with top_columns[3]:
            if st.button("Reset", use_container_width=True):
                controller.reset()
        with top_columns[4]:
            st.metric("Current Step", f"{controller.current_step + 1}/{controller.current_trace.step_count}")

        selected_step = st.slider(
            "Step Slider",
            min_value=0,
            max_value=controller.current_trace.last_step_index,
            key=controller.STEP_WIDGET_KEY,
            format="Step %d",
        )
        if selected_step != controller.current_step:
            controller.set_step(selected_step, sync_widget=False)

        return {"auto_run_clicked": auto_run_clicked}

    def _render_demo_view(self, controller: DashboardController) -> None:
        """Renders the scenario walkthrough panels and charts."""
        self._render_body(controller)

        step = controller.current_decision
        trace = controller.current_trace

        st.markdown("## Decision Insights")
        chart_left, chart_middle, chart_right = st.columns(3)
        with chart_left:
            DashboardCharts.render_reward_timeline(trace, controller.current_step)
        with chart_middle:
            DashboardCharts.render_signal_profile(step)
        with chart_right:
            DashboardCharts.render_action_mix(trace, controller.current_step)

    def _render_body(self, controller: DashboardController) -> None:
        """Renders the main scenario dashboard panels."""
        step = controller.current_decision
        trace = controller.current_trace
        cumulative_reward = trace.cumulative_reward_at(controller.current_step)

        DashboardComponents.render_progress_indicator(trace, controller.current_step)

        st.markdown("## Current State")
        DashboardComponents.render_state_cards(step)

        left, right = st.columns([1.15, 1.0])
        with left:
            DashboardComponents.render_action_panel(step)
            DashboardComponents.render_explanation_box(step)
        with right:
            DashboardComponents.render_reward_panel(step, cumulative_reward)
            DashboardComponents.render_outcome_summary(trace, controller.current_step)

        DashboardComponents.render_history_table(trace, controller.current_step)

    def _autorun_scenario(self, controller: DashboardController, delay_seconds: float) -> None:
        """Animates the current scenario to its final result and leaves navigation intact."""
        playback = st.empty()

        while True:
            with playback.container():
                self._render_demo_view(controller)

            if controller.current_step >= controller.current_trace.last_step_index:
                break

            time.sleep(delay_seconds)
            controller.set_step(controller.current_step + 1, sync_widget=False)

    def _render_training_mode(self, controls: dict[str, int]) -> None:
        """Renders the live training experience inside Streamlit."""
        st.markdown("## Live Training Session")
        st.write("Run live training here to watch episode reward and epsilon evolve over time.")

        if not st.button("Start Live Training", use_container_width=True):
            if self.training_summary_key in st.session_state:
                st.markdown("### Most Recent Training Summary")
                st.json(st.session_state[self.training_summary_key])
            return

        cfg = ProjectConfig(training_episodes=controls["train_episodes"], random_seed=controls["seed"])
        env = TsunamiAlertEnvironment(cfg, seed=cfg.random_seed)
        agent = QLearningAgent(
            cfg.state_size,
            cfg.action_size,
            cfg.alpha,
            cfg.gamma,
            cfg.epsilon,
            cfg.epsilon_decay,
            cfg.min_epsilon,
            cfg.random_seed,
        )

        progress = st.progress(0)
        status = st.empty()
        chart = st.empty()
        records: list[dict[str, Any]] = []
        refresh = max(10, cfg.training_episodes // 60)

        for episode in range(1, cfg.training_episodes + 1):
            result = self._run_training_episode(env, agent, cfg)
            result["episode"] = episode
            records.append(result)
            agent.decay_epsilon()
            records[-1]["epsilon"] = agent.epsilon

            if episode % refresh == 0 or episode == cfg.training_episodes:
                progress.progress(min(100, int((episode / cfg.training_episodes) * 100)))
                status.info(
                    f"Episode {episode}/{cfg.training_episodes} | "
                    f"Reward {result['total_reward']:+.1f} | Epsilon {agent.epsilon:.3f}"
                )
                frame = pd.DataFrame(records)
                chart.line_chart(
                    frame[["episode", "total_reward", "epsilon"]].set_index("episode"),
                    use_container_width=True,
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
        }

        cfg.logs_dir.mkdir(parents=True, exist_ok=True)
        cfg.models_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(cfg.logs_dir / "training_live_session.csv", index=False)
        agent.save_q_table(cfg.models_dir / "q_table.npy")

        st.success("Training completed. Updated model and training log were saved.")
        st.json(summary)
        st.session_state[self.training_summary_key] = summary

    def _run_training_episode(
        self,
        env: TsunamiAlertEnvironment,
        agent: QLearningAgent,
        cfg: ProjectConfig,
    ) -> dict[str, Any]:
        """Runs one Q-learning training episode for the live dashboard."""
        state_idx = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        info: dict[str, Any] = {}

        while not done and steps < cfg.max_steps_per_episode:
            action = agent.choose_action(state_idx, training=True)
            next_state_idx, reward, done, info = env.step(action)
            agent.update(state_idx, action, reward, next_state_idx, done)
            state_idx = next_state_idx
            steps += 1
            total_reward += reward

        return {
            "total_reward": float(total_reward),
            "steps": int(steps),
            "correct": bool(info.get("alert_correct", False)),
            "false": bool(info.get("false_alert", False)),
            "missed": bool(info.get("missed_alert", False)),
        }
