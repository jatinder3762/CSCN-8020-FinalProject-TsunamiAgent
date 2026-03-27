"""Primary Streamlit entrypoint for the tsunami decision dashboard."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.presentation import TsunamiDashboardApp


def main() -> None:
    """Launches the Streamlit dashboard application."""
    TsunamiDashboardApp().run()


if __name__ == "__main__":
    main()
