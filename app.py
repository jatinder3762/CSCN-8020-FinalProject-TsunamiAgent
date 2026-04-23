"""Primary Streamlit entrypoint for the tsunami decision dashboard."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.presentation import TsunamiDashboardApp


def main() -> None:
    """Launches the Streamlit dashboard application."""
    TsunamiDashboardApp().run()


def _is_running_via_streamlit() -> bool:
    """Detects whether the script is executing inside Streamlit's runtime."""
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    return get_script_run_ctx(suppress_warning=True) is not None


def _launch_streamlit() -> int:
    """Re-invokes this script through Streamlit when launched with plain Python."""
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(Path(__file__).resolve()),
        *sys.argv[1:],
    ]
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    if _is_running_via_streamlit():
        main()
    else:
        raise SystemExit(_launch_streamlit())
