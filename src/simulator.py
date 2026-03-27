"""Cinematic simulation renderer for tsunami decision episodes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from config import ProjectConfig
from src.agent import QLearningAgent
from src.environment import TsunamiAlertEnvironment


class TsunamiCinematicSimulator:
    """Builds movie-style animations from environment-agent episode traces."""

    def __init__(
        self,
        config: ProjectConfig,
        environment: TsunamiAlertEnvironment,
        agent: QLearningAgent,
        fps: int = 20,
        frames_per_step: int = 45,
    ) -> None:
        """Initializes simulator dependencies and animation settings."""
        self.config = config
        self.environment = environment
        self.agent = agent
        self.fps = max(8, int(fps))
        self.frames_per_step = max(20, int(frames_per_step))

    def run(self, output_path: Path, max_trials: int = 120, show_preview: bool = False) -> dict[str, Any]:
        """Selects a showcase episode, renders animation, and returns run summary."""
        episode_data = self._select_showcase_episode(max_trials=max_trials)
        saved_path = self._render_animation(episode_data, output_path, show_preview=show_preview)

        return {
            "saved_animation": str(saved_path),
            "steps": len(episode_data["steps"]),
            "total_reward": episode_data["total_reward"],
            "actual_risk_level": episode_data["actual_risk_level"],
            "final_action": episode_data["steps"][-1]["action_text"],
            "alert_correct": episode_data["steps"][-1]["alert_correct"],
            "false_alert": episode_data["steps"][-1]["false_alert"],
            "missed_alert": episode_data["steps"][-1]["missed_alert"],
        }

    def run_parallel(self, output_path: Path, episodes: int = 9, show_preview: bool = False) -> dict[str, Any]:
        """Renders multiple episodes side-by-side in a monitoring-wall animation."""
        traces = [self._run_single_episode_trace() for _ in range(max(1, episodes))]
        saved_path = self._render_parallel_animation(traces=traces, output_path=output_path, show_preview=show_preview)

        total_reward = sum(float(trace["total_reward"]) for trace in traces)
        correct = sum(1 for trace in traces if bool(trace["steps"][-1]["alert_correct"]))
        false_alert = sum(1 for trace in traces if bool(trace["steps"][-1]["false_alert"]))
        missed = sum(1 for trace in traces if bool(trace["steps"][-1]["missed_alert"]))

        return {
            "saved_animation": str(saved_path),
            "episodes_rendered": len(traces),
            "average_reward": round(total_reward / len(traces), 3),
            "correct_alert_rate": round(correct / len(traces), 3),
            "false_alert_rate": round(false_alert / len(traces), 3),
            "missed_alert_rate": round(missed / len(traces), 3),
        }

    def _select_showcase_episode(self, max_trials: int) -> dict[str, Any]:
        """Searches multiple episodes and picks the most presentation-friendly trace."""
        best_candidate: dict[str, Any] | None = None
        best_score = float("-inf")

        for _ in range(max(1, max_trials)):
            candidate = self._run_single_episode_trace()
            score = self._score_episode_for_demo(candidate)
            if score > best_score:
                best_score = score
                best_candidate = candidate

        if best_candidate is None:
            raise RuntimeError("Unable to generate episode trace for simulation.")
        return best_candidate

    def _run_single_episode_trace(self) -> dict[str, Any]:
        """Runs one greedy episode and stores full transition details."""
        state_idx = self.environment.reset()
        done = False
        total_reward = 0.0
        steps: list[dict[str, Any]] = []

        while not done:
            current_state = self.environment.index_to_state(state_idx)
            action = self.agent.choose_action(state_idx, training=False)
            next_state_idx, reward, done, info = self.environment.step(action)
            next_state = self.environment.index_to_state(next_state_idx)

            total_reward += reward
            steps.append(
                {
                    "state": current_state,
                    "next_state": next_state,
                    "action": action,
                    "action_text": self.config.action_names[action],
                    "reward": reward,
                    "done": done,
                    "actual_risk_level": info["actual_risk_level"],
                    "alert_correct": bool(info["alert_correct"]),
                    "false_alert": bool(info["false_alert"]),
                    "missed_alert": bool(info["missed_alert"]),
                }
            )
            state_idx = next_state_idx

        return {
            "steps": steps,
            "total_reward": round(total_reward, 3),
            "actual_risk_level": steps[-1]["actual_risk_level"],
        }

    def _score_episode_for_demo(self, episode_data: dict[str, Any]) -> float:
        """Ranks episodes by drama and explanatory value for live demos."""
        steps = episode_data["steps"]
        risk = episode_data["actual_risk_level"]
        total_reward = float(episode_data["total_reward"])
        final_action = steps[-1]["action"]

        risk_score = {"Low": 5.0, "Medium": 22.0, "High": 35.0}.get(risk, 0.0)
        step_score = float(len(steps)) * 12.0
        action_score = 18.0 if final_action in (2, 3) else 4.0
        reward_score = min(25.0, max(-10.0, total_reward / 4.0))

        return risk_score + step_score + action_score + reward_score

    def _render_animation(self, episode_data: dict[str, Any], output_path: Path, show_preview: bool) -> Path:
        """Renders wave + alert animation and saves it to GIF or PNG fallback."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig, (wave_ax, hud_ax) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2.1, 1.1]})
        fig.patch.set_facecolor("#081a2b")

        wave_ax.set_xlim(0.0, 20.0)
        wave_ax.set_ylim(-2.2, 2.8)
        wave_ax.axis("off")
        wave_ax.set_facecolor("#0b2945")

        hud_ax.set_xlim(0.0, 1.0)
        hud_ax.set_ylim(0.0, 1.0)
        hud_ax.axis("off")
        hud_ax.set_facecolor("#0e1f30")

        x = np.linspace(0.0, 20.0, 900)
        shoreline_x = 17.1
        wave_ax.axvline(shoreline_x, color="#f4f1de", linewidth=2.0, alpha=0.95)
        wave_ax.text(shoreline_x + 0.1, 2.35, "Coastline", color="#f4f1de", fontsize=10, weight="bold")
        wave_ax.scatter([17.8, 18.5, 19.2], [2.0, 1.85, 1.95], c="#f28482", s=70, marker="^")
        wave_ax.text(17.65, 2.12, "Cities", color="#f28482", fontsize=10, weight="bold")

        # Decision path track to show agent movement across time windows.
        path_y = -1.75
        path_nodes_x = [3.0, 10.0, 17.0]
        wave_ax.plot(path_nodes_x, [path_y, path_y, path_y], color="#f1faee", linewidth=2.2, alpha=0.7)
        for px, label in zip(path_nodes_x, self.config.time_levels):
            wave_ax.scatter([px], [path_y], s=80, color="#f1faee", edgecolors="#1d3557", linewidths=1.5)
            wave_ax.text(px - 0.6, path_y - 0.35, label, fontsize=9, color="#f1faee")
        wave_ax.text(0.6, path_y + 0.28, "Decision Path", fontsize=10, color="#f1faee", weight="bold")

        wave_line, = wave_ax.plot([], [], color="#8ecae6", linewidth=2.2)
        fill_poly = wave_ax.fill_between(x, np.zeros_like(x), np.zeros_like(x), color="#219ebc", alpha=0.65)
        path_marker = wave_ax.scatter([path_nodes_x[0]], [path_y], s=210, color="#ffb703", edgecolors="#023047", linewidths=2.0)

        alert_text = wave_ax.text(10.0, 2.45, "", color="#ffd166", fontsize=16, ha="center", weight="bold")
        title_text = wave_ax.text(0.3, 2.45, "Tsunami Alert Cinematic Simulation", color="#f1faee", fontsize=12, weight="bold")

        hud_text = hud_ax.text(0.03, 0.97, "", va="top", ha="left", fontsize=11, color="#edf6f9", family="monospace")

        frame_plan = self._build_frame_plan(episode_data)
        cumulative_rewards = self._build_cumulative_rewards(episode_data)

        def animate(frame_index: int) -> tuple[Any, ...]:
            nonlocal fill_poly
            frame_info = frame_plan[frame_index]
            step_idx = int(frame_info["step_idx"])
            progress = float(frame_info["progress"])

            if fill_poly is not None:
                fill_poly.remove()

            if step_idx < 0:
                step_data = episode_data["steps"][0]
                state = step_data["state"]
                next_state = step_data["next_state"]
                action_text = "Incoming Seismic Event"
                step_reward = 0.0
                cumulative = 0.0
                alert_correct = False
                false_alert = False
                missed_alert = False
            else:
                step_data = episode_data["steps"][step_idx]
                state = step_data["state"]
                next_state = step_data["next_state"]
                action_text = step_data["action_text"]
                step_reward = float(step_data["reward"])
                cumulative = cumulative_rewards[step_idx]
                alert_correct = bool(step_data["alert_correct"])
                false_alert = bool(step_data["false_alert"])
                missed_alert = bool(step_data["missed_alert"])

            observed_wave = int(state[2])
            risk_level = step_data["actual_risk_level"]
            time_label = self.config.time_levels[int(state[4])]
            amplitude = self._wave_amplitude(observed_wave, risk_level, progress)
            phase = (frame_index / self.frames_per_step) * 0.75
            y = self._wave_profile(x, amplitude, phase, progress)

            wave_line.set_data(x, y)
            fill_poly = wave_ax.fill_between(x, y, -2.2, color="#219ebc", alpha=0.72)

            start_t = int(state[4])
            end_t = int(next_state[4])
            marker_x = path_nodes_x[start_t] + ((path_nodes_x[end_t] - path_nodes_x[start_t]) * progress)
            path_marker.set_offsets(np.array([[marker_x, path_y]]))

            alert_message = self._build_alert_banner(action_text, progress, risk_level)
            alert_text.set_text(alert_message)
            alert_text.set_color(self._alert_banner_color(action_text))
            title_text.set_text(f"Tsunami Alert Cinematic Simulation   |   Risk: {risk_level}")

            hud_text.set_text(
                self._build_hud_text(
                    step_index=step_idx + 1,
                    step_total=len(episode_data["steps"]),
                    state=state,
                    action_text=action_text,
                    risk_level=risk_level,
                    time_label=time_label,
                    step_reward=step_reward,
                    cumulative_reward=cumulative,
                    alert_correct=alert_correct,
                    false_alert=false_alert,
                    missed_alert=missed_alert,
                )
            )
            return wave_line, alert_text, title_text, hud_text, path_marker

        animation = FuncAnimation(fig, animate, frames=len(frame_plan), interval=1000 / self.fps, blit=False)
        saved_path = output_path

        try:
            animation.save(output_path, writer=PillowWriter(fps=self.fps), dpi=120)
        except Exception:
            # Fallback to a still frame if GIF encoder is unavailable.
            saved_path = output_path.with_suffix(".png")
            animate(len(frame_plan) - 1)
            fig.savefig(saved_path, dpi=140, bbox_inches="tight")

        if show_preview:
            plt.show()
        plt.close(fig)
        return saved_path

    def _render_parallel_animation(self, traces: list[dict[str, Any]], output_path: Path, show_preview: bool) -> Path:
        """Renders a side-by-side episode wall animation."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        episode_count = len(traces)
        ncols = int(np.ceil(np.sqrt(episode_count)))
        nrows = int(np.ceil(episode_count / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.4 * nrows), squeeze=False)
        fig.patch.set_facecolor("#061423")
        fig.suptitle("Tsunami Decision Control Room: Parallel Episodes", color="#f1faee", fontsize=16, weight="bold")

        x = np.linspace(0.0, 20.0, 480)
        path_y = -1.75
        path_nodes_x = [3.0, 10.0, 17.0]

        panel_data: list[dict[str, Any]] = []
        for panel_index, axis in enumerate(axes.flatten()):
            if panel_index >= episode_count:
                axis.set_visible(False)
                continue

            axis.set_xlim(0.0, 20.0)
            axis.set_ylim(-2.2, 2.8)
            axis.axis("off")
            axis.set_facecolor("#0c2740")

            axis.axvline(17.1, color="#f4f1de", linewidth=1.4, alpha=0.75)
            axis.plot(path_nodes_x, [path_y, path_y, path_y], color="#f1faee", linewidth=1.8, alpha=0.65)
            axis.scatter(path_nodes_x, [path_y, path_y, path_y], s=35, color="#f1faee", edgecolors="#1d3557", linewidths=1.0)

            title = axis.text(0.6, 2.30, f"Episode {panel_index + 1}", color="#f1faee", fontsize=10, weight="bold")
            action_text = axis.text(10.0, 2.35, "", color="#ffd166", fontsize=10, ha="center", weight="bold")
            reward_text = axis.text(0.6, 1.95, "", color="#caf0f8", fontsize=9)
            line, = axis.plot([], [], color="#8ecae6", linewidth=1.7)
            fill = axis.fill_between(x, np.zeros_like(x), np.zeros_like(x), color="#219ebc", alpha=0.60)
            marker = axis.scatter([path_nodes_x[0]], [path_y], s=90, color="#ffb703", edgecolors="#023047", linewidths=1.2)

            panel_data.append(
                {
                    "axis": axis,
                    "title": title,
                    "action_text": action_text,
                    "reward_text": reward_text,
                    "line": line,
                    "fill": fill,
                    "marker": marker,
                    "trace": traces[panel_index],
                }
            )

        max_steps = max(len(trace["steps"]) for trace in traces)
        frame_plan = self._build_parallel_frame_plan(max_steps=max_steps)

        def animate(frame_index: int) -> tuple[Any, ...]:
            artists: list[Any] = []
            frame_meta = frame_plan[frame_index]
            global_step = int(frame_meta["step_idx"])
            progress = float(frame_meta["progress"])

            for panel_index, panel in enumerate(panel_data):
                trace = panel["trace"]
                steps = trace["steps"]

                if panel["fill"] is not None:
                    panel["fill"].remove()

                if global_step < 0:
                    step = steps[0]
                    action_text_value = "SEISMIC EVENT"
                    cumulative = 0.0
                else:
                    local_step = min(global_step, len(steps) - 1)
                    step = steps[local_step]
                    action_text_value = step["action_text"]
                    cumulative = sum(float(item["reward"]) for item in steps[: local_step + 1])

                state = step["state"]
                next_state = step["next_state"]
                risk = step["actual_risk_level"]
                amp = self._wave_amplitude(int(state[2]), risk, progress)
                phase = ((frame_index / self.frames_per_step) * 0.55) + (panel_index * 0.17)
                y = self._wave_profile(x, amp, phase, progress)

                panel["line"].set_data(x, y)
                panel["fill"] = panel["axis"].fill_between(x, y, -2.2, color="#219ebc", alpha=0.66)

                start_t = int(state[4])
                end_t = int(next_state[4])
                marker_x = path_nodes_x[start_t] + ((path_nodes_x[end_t] - path_nodes_x[start_t]) * progress)
                panel["marker"].set_offsets(np.array([[marker_x, path_y]]))

                panel["title"].set_text(f"Episode {panel_index + 1} | Risk: {risk}")
                panel["action_text"].set_text(action_text_value)
                panel["action_text"].set_color(self._alert_banner_color(action_text_value.title()))
                panel["reward_text"].set_text(f"Reward: {cumulative:+.1f} | Steps: {len(steps)}")

                artists.extend([panel["line"], panel["action_text"], panel["title"], panel["reward_text"], panel["marker"]])

            return tuple(artists)

        animation = FuncAnimation(fig, animate, frames=len(frame_plan), interval=1000 / self.fps, blit=False)
        saved_path = output_path
        try:
            animation.save(output_path, writer=PillowWriter(fps=self.fps), dpi=115)
        except Exception:
            saved_path = output_path.with_suffix(".png")
            animate(len(frame_plan) - 1)
            fig.savefig(saved_path, dpi=140, bbox_inches="tight")

        if show_preview:
            plt.show()
        plt.close(fig)
        return saved_path

    def _build_frame_plan(self, episode_data: dict[str, Any]) -> list[dict[str, float | int]]:
        """Creates intro/step/outro frame schedule for smooth animation."""
        plan: list[dict[str, float | int]] = []
        intro_frames = int(self.frames_per_step * 0.7)
        outro_frames = int(self.frames_per_step * 0.9)

        for k in range(intro_frames):
            plan.append({"step_idx": -1, "progress": k / max(1, intro_frames - 1)})

        for step_idx in range(len(episode_data["steps"])):
            for k in range(self.frames_per_step):
                plan.append({"step_idx": step_idx, "progress": k / max(1, self.frames_per_step - 1)})

        for k in range(outro_frames):
            plan.append({"step_idx": len(episode_data["steps"]) - 1, "progress": k / max(1, outro_frames - 1)})
        return plan

    def _build_parallel_frame_plan(self, max_steps: int) -> list[dict[str, float | int]]:
        """Creates frame schedule for the multi-episode wall animation."""
        plan: list[dict[str, float | int]] = []
        intro_frames = int(self.frames_per_step * 0.6)
        outro_frames = int(self.frames_per_step * 0.7)

        for k in range(intro_frames):
            plan.append({"step_idx": -1, "progress": k / max(1, intro_frames - 1)})

        for step_idx in range(max_steps):
            for k in range(self.frames_per_step):
                plan.append({"step_idx": step_idx, "progress": k / max(1, self.frames_per_step - 1)})

        for k in range(outro_frames):
            plan.append({"step_idx": max_steps - 1, "progress": k / max(1, outro_frames - 1)})
        return plan

    @staticmethod
    def _build_cumulative_rewards(episode_data: dict[str, Any]) -> list[float]:
        """Computes cumulative reward trace for HUD visualization."""
        cumulative = 0.0
        values: list[float] = []
        for step in episode_data["steps"]:
            cumulative += float(step["reward"])
            values.append(round(cumulative, 3))
        return values

    @staticmethod
    def _wave_profile(x: np.ndarray, amplitude: float, phase: float, progress: float) -> np.ndarray:
        """Builds layered wave shape with an advancing surge near shore."""
        harmonic = 0.33 * amplitude * np.sin(2.0 * np.pi * ((x / 2.9) + 0.9 * phase))
        carrier = amplitude * np.sin(2.0 * np.pi * ((x / 5.6) + phase))
        surge_center = 10.8 + (7.0 * progress)
        surge = (amplitude * 0.75) * np.exp(-((x - surge_center) ** 2) / 14.0)
        return carrier + harmonic + surge - 0.2

    @staticmethod
    def _wave_amplitude(observed_wave: int, risk_level: str, progress: float) -> float:
        """Maps state/risk to visual wave amplitude for dramatic clarity."""
        base = {0: 0.25, 1: 0.6, 2: 1.05}.get(observed_wave, 0.3)
        risk_boost = {"Low": 0.08, "Medium": 0.25, "High": 0.52}.get(risk_level, 0.0)
        pulse = 0.18 * np.sin(np.pi * progress)
        return base + risk_boost + pulse

    @staticmethod
    def _build_alert_banner(action_text: str, progress: float, risk_level: str) -> str:
        """Generates animated alert banner text."""
        if action_text in ("Regional Alert", "Full Alert"):
            flash = " !!!" if progress > 0.45 else ""
            return f"{action_text.upper()}{flash}"
        if action_text == "Verify":
            return f"VERIFYING SIGNALS ({risk_level.upper()} RISK)"
        if action_text == "Wait":
            return "MONITORING WAVE DATA"
        return "SEISMIC EVENT DETECTED"

    @staticmethod
    def _alert_banner_color(action_text: str) -> str:
        """Returns color code by action severity."""
        normalized = action_text.strip().lower()
        if normalized == "full alert":
            return "#ff595e"
        if normalized == "regional alert":
            return "#ffca3a"
        if normalized == "verify":
            return "#8ac926"
        if normalized == "wait":
            return "#90e0ef"
        return "#ffd166"

    def _build_hud_text(
        self,
        step_index: int,
        step_total: int,
        state: tuple[int, int, int, int, int],
        action_text: str,
        risk_level: str,
        time_label: str,
        step_reward: float,
        cumulative_reward: float,
        alert_correct: bool,
        false_alert: bool,
        missed_alert: bool,
    ) -> str:
        """Composes right-side dashboard text per frame."""
        magnitude = self.config.magnitude_levels[state[0]]
        depth = self.config.depth_levels[state[1]]
        wave = self.config.wave_risk_levels[state[2]]
        confidence = self.config.confidence_levels[state[3]]

        return (
            "DEMO HUD\n"
            "--------\n"
            f"Step: {step_index}/{step_total}\n"
            f"Time Window: {time_label}\n"
            f"Actual Risk: {risk_level}\n\n"
            "Observed State\n"
            f"- Magnitude : {magnitude}\n"
            f"- Depth     : {depth}\n"
            f"- WaveRisk  : {wave}\n"
            f"- Confidence: {confidence}\n\n"
            f"Agent Action: {action_text}\n"
            f"Step Reward : {step_reward:+.1f}\n"
            f"Cumulative  : {cumulative_reward:+.1f}\n\n"
            f"Correct Alert: {alert_correct}\n"
            f"False Alert  : {false_alert}\n"
            f"Missed Alert : {missed_alert}\n"
        )
