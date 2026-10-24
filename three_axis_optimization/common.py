"""Shared utilities for the three-axis VLA optimization experiments."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GENZ_ROOT = ROOT / "genz"
if str(GENZ_ROOT) not in sys.path:
    sys.path.insert(0, str(GENZ_ROOT))

from GenZ.LLM_inference.utils import get_inference_system  # noqa: E402
from GenZ.unit import Unit  # noqa: E402

TIME = "Latency (msec)"
COMPUTE = "Compute time (msec)"
MEMORY = "Memory time (msec)"
COMM = "Communication time (msec)"
INPUT_A = "Input_a (MB)"
INPUT_W = "Input_w (MB)"
OUTPUT = "Output (MB)"
TOTAL_DATA = "Total Data (MB)"
FLOPS = "Num ops (MFLOP)"
GROUP_KEYS = ["stage", "stage_invocation", "repeat_instance"]


def load_profile(profile_dir: Path) -> pd.DataFrame:
    path = profile_dir / "operator_breakdown.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run pipeline_prototype/profile_and_pipeline.py first."
        )
    frame = pd.read_csv(path).reset_index(drop=True)
    required = {
        TIME, COMPUTE, MEMORY, COMM, INPUT_A, INPUT_W, OUTPUT, TOTAL_DATA,
        FLOPS, "stage", "stage_invocation", "repeat_instance", "Layer Name",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Profile is missing columns: {sorted(missing)}")
    frame["source_indices"] = [str(index) for index in frame.index]
    return frame


def system_for(gpu: str, bits: str):
    return get_inference_system(system_name=gpu, bits=bits)


def transfer_ms(size_mb: float, raw_bandwidth: float) -> float:
    """Convert a GenZ decimal-MB transfer to milliseconds."""
    return Unit().unit_to_raw(float(size_mb), type="M") / raw_bandwidth * 1e3


def stage_latency(operators: pd.DataFrame) -> pd.Series:
    return operators.groupby("stage", sort=False)[TIME].sum()


def effective_roofline(compute: float, memory: float, communication: float) -> float:
    return max(float(compute), float(memory), float(communication), 0.0)


def assert_valid_operators(operators: pd.DataFrame) -> None:
    numeric = [TIME, COMPUTE, MEMORY, COMM, INPUT_A, INPUT_W, OUTPUT, TOTAL_DATA, FLOPS]
    if operators[numeric].isna().any().any():
        raise AssertionError("NaN found in modeled operator counters")
    if (operators[numeric] < -1e-12).any().any():
        raise AssertionError("Negative modeled operator counter")
    lower = operators[[COMPUTE, MEMORY, COMM]].max(axis=1)
    if (operators[TIME] + 1e-10 < lower).any():
        raise AssertionError("Latency is below its resource lower bound")
