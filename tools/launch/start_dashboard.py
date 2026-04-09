"""Single-command launcher for Streamlit dashboard with configurable port.

Behavior:
- Default to port 8502 unless overridden by CLI args.
- If the target port is occupied, terminate occupying process(es).
- Start Streamlit app.
- Open browser automatically.
"""

from __future__ import annotations

import argparse
import os
import re
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path


DEFAULT_PORT = 8502
DEFAULT_HOST = "127.0.0.1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _build_parser() -> argparse.ArgumentParser:
    """Creates CLI arguments for custom host and port."""
    parser = argparse.ArgumentParser(
        description="Launch Streamlit dashboard with optional custom host/port."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Dashboard port (default: 8502).",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_HOST,
        help="Dashboard host/address (default: 127.0.0.1).",
    )
    return parser


def _find_pids_on_port(port: int) -> set[int]:
    """Returns process IDs using the given TCP port (Windows netstat parsing)."""
    result = subprocess.run(
        ["netstat", "-ano", "-p", "tcp"],
        capture_output=True,
        text=True,
        check=False,
    )
    pids: set[int] = set()
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line or f":{port}" not in line:
            continue
        parts = re.split(r"\s+", line)
        if len(parts) < 5:
            continue
        local_addr = parts[1]
        if not local_addr.endswith(f":{port}"):
            continue
        pid_text = parts[-1]
        if pid_text.isdigit():
            pid = int(pid_text)
            if pid > 0 and pid != os.getpid():
                pids.add(pid)
    return pids


def _kill_pid(pid: int) -> None:
    """Force-kills a PID on Windows."""
    subprocess.run(
        ["taskkill", "/PID", str(pid), "/T", "/F"],
        capture_output=True,
        text=True,
        check=False,
    )


def _is_port_open(host: str, port: int) -> bool:
    """Checks whether a TCP host:port is accepting connections."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def _wait_for_port(host: str, port: int, timeout_s: float = 25.0) -> bool:
    """Waits until the TCP port is open or timeout occurs."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _is_port_open(host, port):
            return True
        time.sleep(0.25)
    return False


def _clear_port(port: int, retries: int = 6) -> bool:
    """Attempts to clear a port by terminating occupying PIDs."""
    for _ in range(max(1, retries)):
        pids = _find_pids_on_port(port)
        if not pids:
            return True
        print(f"Port {port} is in use by PID(s): {sorted(pids)}. Clearing port...")
        for pid in pids:
            _kill_pid(pid)
        time.sleep(1.0)
    return len(_find_pids_on_port(port)) == 0


def main() -> int:
    """Launches Streamlit dashboard with configurable host/port behavior."""
    args = _build_parser().parse_args()
    host = str(args.host)
    port = int(args.port)
    url = f"http://{host}:{port}"

    print(f"Preparing dashboard on {url}")

    if not _clear_port(port):
        print(
            f"Unable to free port {port}. Close the process using this port and rerun "
            "or run terminal as Administrator."
        )
        return 1

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app.py",
        "--server.address",
        host,
        "--server.port",
        str(port),
    ]
    process = subprocess.Popen(cmd, cwd=PROJECT_ROOT)

    if _wait_for_port(host, port):
        print(f"Dashboard is running at {url}")
        webbrowser.open(url)
    else:
        print("Dashboard did not open in time. Check terminal logs.")

    try:
        return process.wait()
    except KeyboardInterrupt:
        process.terminate()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
