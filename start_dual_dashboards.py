"""Launches both Streamlit dashboards on different ports for side-by-side use."""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_HOST = "127.0.0.1"
DEFAULT_MAIN_PORT = 8502
DEFAULT_VISUALIZER_PORT = 8503


def _build_parser() -> argparse.ArgumentParser:
    """Builds CLI options for launching both Streamlit apps."""
    parser = argparse.ArgumentParser(
        description="Run app.py and mdp_visualizer_app.py together on different ports."
    )
    parser.add_argument("--host", type=str, default=DEFAULT_HOST, help="Server host (default: 127.0.0.1).")
    parser.add_argument(
        "--main-port",
        type=int,
        default=DEFAULT_MAIN_PORT,
        help="Port for app.py (default: 8502).",
    )
    parser.add_argument(
        "--visualizer-port",
        type=int,
        default=DEFAULT_VISUALIZER_PORT,
        help="Port for mdp_visualizer_app.py (default: 8503).",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open browser tabs automatically.",
    )
    return parser


def _is_port_open(host: str, port: int) -> bool:
    """Checks whether host:port accepts TCP connections."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def _wait_for_port(host: str, port: int, timeout_s: float = 20.0) -> bool:
    """Waits until the requested host:port is reachable."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _is_port_open(host, port):
            return True
        time.sleep(0.25)
    return False


def _assert_port_available(host: str, port: int, label: str) -> None:
    """Raises if a requested port is already occupied."""
    if _is_port_open(host, port):
        raise RuntimeError(
            f"{label} port {port} is already in use on {host}. "
            "Pick another port and retry."
        )


def _start_streamlit(app_file: str, host: str, port: int) -> subprocess.Popen[bytes]:
    """Starts one Streamlit process."""
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        app_file,
        "--server.address",
        host,
        "--server.port",
        str(port),
    ]
    return subprocess.Popen(cmd, cwd=PROJECT_ROOT)


def main() -> int:
    """Launches both apps and keeps the parent process attached."""
    args = _build_parser().parse_args()
    host = str(args.host)
    main_port = int(args.main_port)
    visualizer_port = int(args.visualizer_port)
    auto_open_browser = not bool(args.no_browser)

    if main_port == visualizer_port:
        print("main-port and visualizer-port must be different.")
        return 1

    try:
        _assert_port_available(host, main_port, "Main dashboard")
        _assert_port_available(host, visualizer_port, "MDP visualizer")
    except RuntimeError as exc:
        print(str(exc))
        return 1

    main_process = _start_streamlit("app.py", host, main_port)
    visualizer_process = _start_streamlit("mdp_visualizer_app.py", host, visualizer_port)

    main_url = f"http://{host}:{main_port}"
    visualizer_url = f"http://{host}:{visualizer_port}"

    main_ready = _wait_for_port(host, main_port)
    visualizer_ready = _wait_for_port(host, visualizer_port)

    if main_ready:
        print(f"Main dashboard: {main_url}")
        if auto_open_browser:
            webbrowser.open(main_url)
    else:
        print("Main dashboard did not become reachable in time.")

    if visualizer_ready:
        print(f"MDP visualizer: {visualizer_url}")
        if auto_open_browser:
            webbrowser.open(visualizer_url)
    else:
        print("MDP visualizer did not become reachable in time.")

    try:
        # Keep parent process alive while children run.
        while True:
            main_exit = main_process.poll()
            viz_exit = visualizer_process.poll()
            if main_exit is not None or viz_exit is not None:
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        for proc in (main_process, visualizer_process):
            if proc.poll() is None:
                proc.terminate()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
