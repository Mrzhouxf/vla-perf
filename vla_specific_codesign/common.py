"""Shared profile and hardware helpers."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from three_axis_optimization.common import (  # re-use validated GenZ units
    COMM, COMPUTE, FLOPS, INPUT_A, INPUT_W, MEMORY, OUTPUT, TIME, TOTAL_DATA,
    assert_valid_operators, load_profile, system_for, transfer_ms,
)

STAGES = ["vision_encoder", "vlm_prefill", "action_expert"]


def load_validated_profile(path: Path) -> pd.DataFrame:
    operators = load_profile(path)
    assert_valid_operators(operators)
    return operators


def stage_totals(operators: pd.DataFrame) -> pd.Series:
    return operators.groupby("stage", sort=False)[TIME].sum().reindex(STAGES, fill_value=0.0)


def repeat_number(value: str) -> int:
    text = str(value)
    return int(text.split("repeat", 1)[1].split("/", 1)[0]) if "repeat" in text else -1
