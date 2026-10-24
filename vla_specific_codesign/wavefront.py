"""Exact layer-level Prefix-to-Action-Expert wavefront scheduler."""
from __future__ import annotations

import pandas as pd

from .common import COMM, COMPUTE, MEMORY, TIME, repeat_number


RESOURCE_COLUMNS = [COMPUTE, MEMORY, COMM]


def layer_table(operators: pd.DataFrame, stage: str, invocation: int | None = None) -> pd.DataFrame:
    selected = operators[operators.stage == stage].copy()
    if invocation is not None:
        selected = selected[selected.stage_invocation == invocation]
    selected["layer"] = selected.repeat_instance.map(repeat_number)
    selected = selected[selected.layer >= 0]
    columns = [TIME, *RESOURCE_COLUMNS]
    return selected.groupby("layer", sort=True)[columns].sum().reset_index()


def pair_latency(prefix: pd.Series, action: pd.Series, overlap_efficiency: float) -> dict:
    if not 0 <= overlap_efficiency <= 1:
        raise ValueError("overlap_efficiency must be in [0, 1]")
    serial = float(prefix[TIME] + action[TIME])
    ideal = max(
        float(prefix[COMPUTE] + action[COMPUTE]),
        float(prefix[MEMORY] + action[MEMORY]),
        float(prefix[COMM] + action[COMM]),
    )
    ideal = min(serial, max(float(prefix[TIME]), float(action[TIME]), ideal))
    conservative = serial - overlap_efficiency * (serial - ideal)
    return {
        "serial_pair_ms": serial,
        "ideal_resource_ms": ideal,
        "scheduled_pair_ms": conservative,
        "overlap_saved_ms": serial - conservative,
    }


def schedule_wavefront(prefix_layers: pd.DataFrame, action_layers: pd.DataFrame,
                       overlap_efficiency: float = 0.8) -> tuple[float, pd.DataFrame]:
    """Schedule P0, then (P_l || A_l-1), then A_last.

    This is dependency-exact for a prefix mask that prevents prefix tokens from
    attending to the suffix.  Action layer l becomes ready only after both
    prefix KV layer l and action hidden layer l-1 are complete.
    """
    if len(prefix_layers) != len(action_layers) or len(prefix_layers) == 0:
        raise ValueError("Prefix and Action Expert must expose the same non-zero layer count")
    records = []
    current = 0.0

    first = prefix_layers.iloc[0]
    duration = float(first[TIME])
    records.append({
        "diagonal": 0, "prefix_layer": int(first.layer), "action_layer": -1,
        "start_ms": current, "end_ms": current + duration,
        "serial_pair_ms": duration, "ideal_resource_ms": duration,
        "scheduled_pair_ms": duration, "overlap_saved_ms": 0.0,
        "dependency": "pipeline fill: prefix layer 0",
    })
    current += duration

    for layer in range(1, len(prefix_layers)):
        prefix, action = prefix_layers.iloc[layer], action_layers.iloc[layer - 1]
        timing = pair_latency(prefix, action, overlap_efficiency)
        records.append({
            "diagonal": layer, "prefix_layer": int(prefix.layer),
            "action_layer": int(action.layer), "start_ms": current,
            "end_ms": current + timing["scheduled_pair_ms"], **timing,
            "dependency": "P_l depends P_l-1; A_l-1 depends P_l-1 and A_l-2",
        })
        current += timing["scheduled_pair_ms"]

    last = action_layers.iloc[-1]
    duration = float(last[TIME])
    records.append({
        "diagonal": len(prefix_layers), "prefix_layer": -1,
        "action_layer": int(last.layer), "start_ms": current,
        "end_ms": current + duration, "serial_pair_ms": duration,
        "ideal_resource_ms": duration, "scheduled_pair_ms": duration,
        "overlap_saved_ms": 0.0, "dependency": "pipeline drain: final action layer",
    })
    current += duration
    return current, pd.DataFrame(records)


def wavefront_bounds(prefix_layers: pd.DataFrame, action_layers: pd.DataFrame) -> dict:
    serial = float(prefix_layers[TIME].sum() + action_layers[TIME].sum())
    ideal, _ = schedule_wavefront(prefix_layers, action_layers, overlap_efficiency=1.0)
    return {
        "serial_region_ms": serial,
        "ideal_wavefront_ms": ideal,
        "maximum_overlap_saved_ms": serial - ideal,
    }


def prefix_kv_ready_table(operators: pd.DataFrame) -> pd.DataFrame:
    prefix = operators[operators.stage == "vlm_prefill"].copy()
    prefix["layer"] = prefix.repeat_instance.map(repeat_number)
    prefix = prefix[prefix.layer >= 0]
    rows = []
    for layer, group in prefix.groupby("layer", sort=True):
        qkv = group[group["Layer Name"] == "QKV"]
        if len(qkv) != 1:
            raise ValueError(f"Expected one Prefix QKV producer at layer {layer}")
        rows.append({
            "layer": int(layer),
            "prefix_layer_ms": float(group[TIME].sum()),
            "kv_ready_offset_ms": float(qkv[TIME].iloc[0]),
        })
    return pd.DataFrame(rows)


def schedule_streaming_wavefront(operators: pd.DataFrame, action_layers: pd.DataFrame,
                                 overlap_efficiency: float = 0.8):
    """Release A_l when Prefix QKV_l is ready, before Prefix layer l finishes.

    The no-interference dependency schedule is bounded by conserved aggregate
    compute/memory/communication work.  `overlap_efficiency` interpolates from
    serial execution to that joint dependency/resource lower bound.
    """
    if not 0 <= overlap_efficiency <= 1:
        raise ValueError("overlap_efficiency must be in [0, 1]")
    prefix_layers = layer_table(operators, "vlm_prefill")
    ready = prefix_kv_ready_table(operators)
    if len(prefix_layers) != len(action_layers) or len(ready) != len(action_layers):
        raise ValueError("Prefix KV producers and Action layers must align")

    prefix_start, prefix_end = [], []
    current = 0.0
    for _, layer in prefix_layers.iterrows():
        prefix_start.append(current)
        current += float(layer[TIME])
        prefix_end.append(current)

    action_end = 0.0
    records = []
    for layer in range(len(action_layers)):
        kv_ready = prefix_start[layer] + float(ready.iloc[layer].kv_ready_offset_ms)
        start = max(kv_ready, action_end)
        end = start + float(action_layers.iloc[layer][TIME])
        records.append({
            "layer": layer,
            "prefix_start_ms": prefix_start[layer],
            "prefix_kv_ready_ms": kv_ready,
            "prefix_end_ms": prefix_end[layer],
            "action_dependency_start_ms": start,
            "action_dependency_end_ms": end,
            "kv_released_before_prefix_end_ms": max(0.0, prefix_end[layer] - kv_ready),
        })
        action_end = end

    dependency_makespan = max(prefix_end[-1], action_end)
    serial = float(prefix_layers[TIME].sum() + action_layers[TIME].sum())
    resource_lower = max(
        float(prefix_layers[COMPUTE].sum() + action_layers[COMPUTE].sum()),
        float(prefix_layers[MEMORY].sum() + action_layers[MEMORY].sum()),
        float(prefix_layers[COMM].sum() + action_layers[COMM].sum()),
    )
    ideal = min(serial, max(dependency_makespan, resource_lower))
    scheduled = serial - overlap_efficiency * (serial - ideal)
    summary = {
        "serial_region_ms": serial,
        "dependency_only_makespan_ms": dependency_makespan,
        "aggregate_resource_lower_bound_ms": resource_lower,
        "ideal_streaming_wavefront_ms": ideal,
        "scheduled_streaming_wavefront_ms": scheduled,
        "latency_saved_ms": serial - scheduled,
    }
    return scheduled, pd.DataFrame(records), summary
