"""Utility classes for IO, reproducibility, math helpers, and readable labels."""

from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from config import ProjectConfig


class SeedManager:
    """Controls deterministic behavior across Python and NumPy."""

    @staticmethod
    def set_seed(seed: int) -> np.random.Generator:
        """Sets process-level seeds and returns a seeded NumPy generator."""
        random.seed(seed)
        np.random.seed(seed)
        return np.random.default_rng(seed)


class StateCodec:
    """Encodes and decodes discrete tuple states to integer indices."""

    @staticmethod
    def encode_state(state: tuple[int, int, int, int, int], shape: tuple[int, int, int, int, int]) -> int:
        """Converts a state tuple into a unique integer index."""
        index = 0
        for value, base in zip(state, shape):
            index = index * base + value
        return index

    @staticmethod
    def decode_state(index: int, shape: tuple[int, int, int, int, int]) -> tuple[int, int, int, int, int]:
        """Converts a state index back to a state tuple."""
        values: list[int] = []
        for base in reversed(shape):
            values.append(index % base)
            index //= base
        values.reverse()
        return tuple(values)  # type: ignore[return-value]


class MathUtils:
    """Mathematical helper methods for metrics and smoothing."""

    @staticmethod
    def moving_average(values: list[float], window: int) -> list[float]:
        """Computes a trailing moving average list."""
        if not values:
            return []
        if window <= 1:
            return values[:]

        series = pd.Series(values, dtype="float64")
        return series.rolling(window=window, min_periods=1).mean().to_list()

    @staticmethod
    def safe_rate(numerator: float, denominator: float) -> float:
        """Returns numerator/denominator, handling division by zero safely."""
        if denominator == 0:
            return 0.0
        return float(numerator) / float(denominator)


class OutputManager:
    """Handles path creation and serialization of structured outputs."""

    @staticmethod
    def ensure_directory(path: Path) -> None:
        """Creates a directory if needed."""
        path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def ensure_project_directories(config: "ProjectConfig") -> None:
        """Ensures all configured project directories exist."""
        for directory in (config.data_dir, config.outputs_dir, config.models_dir, config.plots_dir, config.logs_dir):
            OutputManager.ensure_directory(directory)

    @staticmethod
    def save_records_csv(records: list[dict[str, Any]], path: Path) -> None:
        """Saves a list of dictionaries to CSV."""
        OutputManager.ensure_directory(path.parent)
        frame = pd.DataFrame(records)
        frame.to_csv(path, index=False)

    @staticmethod
    def save_json(data: dict[str, Any], path: Path) -> None:
        """Saves a dictionary to JSON file."""
        OutputManager.ensure_directory(path.parent)
        with path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)

    @staticmethod
    def load_json(path: Path) -> dict[str, Any]:
        """Loads JSON content from file."""
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def timestamp() -> str:
        """Returns a compact UTC timestamp string for output files."""
        return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


class LabelFormatter:
    """Builds readable state/action labels from index-based representations."""

    @staticmethod
    def action_name(action: int, config: "ProjectConfig") -> str:
        """Maps action index to action text."""
        return config.action_names.get(action, f"Unknown({action})")

    @staticmethod
    def state_name(state: tuple[int, int, int, int, int], config: "ProjectConfig") -> str:
        """Formats a state tuple into a human-readable description."""
        return (
            f"Magnitude={config.magnitude_levels[state[0]]}, "
            f"Depth={config.depth_levels[state[1]]}, "
            f"WaveRisk={config.wave_risk_levels[state[2]]}, "
            f"Confidence={config.confidence_levels[state[3]]}, "
            f"Time={config.time_levels[state[4]]}"
        )
