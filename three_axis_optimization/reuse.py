"""Similarity-gated cross-frame activation/KV reuse what-if model."""
from __future__ import annotations

import re

import pandas as pd

from .common import (
    COMM, COMPUTE, FLOPS, INPUT_A, INPUT_W, MEMORY, OUTPUT, TIME, TOTAL_DATA,
    effective_roofline, transfer_ms,
)


def _layer_number(value: str) -> int:
    match = re.search(r"repeat(\d+)", str(value))
    return int(match.group(1)) if match else 0


def base_reuse_ratio(similarity: float, threshold: float, maximum: float) -> float:
    if not 0 <= similarity <= 1 or not 0 <= threshold <= 1 or not 0 <= maximum <= 1:
        raise ValueError("similarity, threshold and maximum reuse must be in [0, 1]")
    # `maximum` is the implementation's upper coverage, while similarity is
    # the measured fraction likely to be unchanged.  Multiplication preserves
    # sensitivity above the gate (unlike min(similarity, maximum), which would
    # flatten every value once similarity exceeds a smaller maximum).
    return maximum * similarity if similarity >= threshold else 0.0


def reuse_ratio_for_row(row: pd.Series, similarity: float, threshold: float,
                        maximum: float, layer_decay: float,
                        vlm_propagation: float, visual_prefix_fraction: float) -> float:
    base = base_reuse_ratio(similarity, threshold, maximum)
    layer = _layer_number(row["repeat_instance"])
    stage = str(row["stage"])
    if stage == "vision_encoder":
        return base * layer_decay ** layer
    if stage == "vlm_prefill":
        # VLM hidden states are globally coupled by attention.  Only a guarded
        # fraction of the visual prefix is allowed to propagate as a cache hit.
        return base * visual_prefix_fraction * vlm_propagation * layer_decay ** layer
    # Robot state, diffusion timestep and noisy actions change every frame/step.
    return 0.0


def _residency(row: pd.Series) -> tuple[str, str, str]:
    """Return GenZ input-A/input-W/output residency for the current π0 graph."""
    name, stage = str(row["Layer Name"]), str(row["stage"])
    action_attention = (
        stage == "action_expert"
        and (name in {"QKV", "QKV Self", "Out Proj"}
             or name.startswith("Logit") or name.startswith("Attend"))
    )
    if action_attention:  # ResidencyInfo.AC_onchip
        return "on", "off", "on"
    if name.startswith("Logit"):  # ResidencyInfo.C_onchip
        return "off", "off", "on"
    if name.startswith("Attend"):  # ResidencyInfo.A_onchip
        return "on", "off", "off"
    return "off", "off", "off"


def _memory_time_ms(row: pd.Series, system, input_a: float, produced_output: float,
                    cache_read: float) -> float:
    if float(row[COMPUTE]) == 0:
        return 0.0  # Matches GenZ's special zero-compute behavior.
    locations = _residency(row)
    traffic = (input_a, float(row[INPUT_W]), produced_output)
    total = 0.0
    for size_mb, location in zip(traffic, locations):
        bandwidth = system.onchip_mem_bw if location == "on" else system.offchip_mem_bw
        total += transfer_ms(size_mb, bandwidth)
    # A cross-frame cache is much larger than SRAM/register capacity and is
    # therefore always charged to off-chip memory, regardless of output pinning.
    total += transfer_ms(cache_read, system.offchip_mem_bw)
    return total / system.memory_efficiency


def apply_reuse(operators: pd.DataFrame, system, similarity: float = 0.92,
                threshold: float = 0.85, maximum: float = 0.80,
                layer_decay: float = 0.97, vlm_propagation: float = 0.25,
                visual_prefix_fraction: float = 768 / 800) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = operators.copy()
    result["Reuse cache read (MB)"] = 0.0
    records = []
    for index, row in result.iterrows():
        ratio = reuse_ratio_for_row(row, similarity, threshold, maximum, layer_decay,
                                    vlm_propagation, visual_prefix_fraction)
        old_compute, old_memory, old_latency = float(row[COMPUTE]), float(row[MEMORY]), float(row[TIME])
        old_flops, old_output = float(row[FLOPS]), float(row[OUTPUT])
        if old_compute == 0 or old_latency == 0:
            ratio = 0.0
        requested_ratio = ratio
        new_compute = old_compute * (1 - ratio)
        # Weights are still fetched once for sparse recomputation.  Reused
        # outputs must be read from the cache, so output traffic is retained.
        new_input_a = float(row[INPUT_A]) * (1 - ratio)
        produced_output = old_output * (1 - ratio)
        cache_read = old_output * ratio
        new_total = float(row[INPUT_W]) + new_input_a + produced_output + cache_read
        new_memory = _memory_time_ms(row, system, new_input_a, produced_output, cache_read)
        new_latency = effective_roofline(new_compute, new_memory, float(row[COMM]))
        cache_profitable = new_latency <= old_latency + 1e-15
        if not cache_profitable:
            ratio = 0.0
            new_compute, new_memory, new_latency = old_compute, old_memory, old_latency
            new_input_a, produced_output = float(row[INPUT_A]), old_output
            cache_read, new_total = 0.0, float(row[TOTAL_DATA])
        result.at[index, COMPUTE] = new_compute
        result.at[index, MEMORY] = new_memory
        result.at[index, TIME] = new_latency
        result.at[index, INPUT_A] = new_input_a
        # OUTPUT is the producer write that a following fusion may eliminate.
        # Cache reads are retained separately and are never counted as a free
        # producer write in the composition model.
        result.at[index, OUTPUT] = produced_output
        result.at[index, "Reuse cache read (MB)"] = cache_read
        result.at[index, TOTAL_DATA] = new_total
        result.at[index, FLOPS] = old_flops * (1 - ratio)
        records.append({
            "operator_index": index,
            "stage": row["stage"],
            "stage_invocation": row["stage_invocation"],
            "repeat_instance": row["repeat_instance"],
            "operator": row["Layer Name"],
            "requested_reuse_ratio": requested_ratio,
            "reuse_ratio": ratio,
            "cache_profitable": cache_profitable,
            "baseline_latency_ms": old_latency,
            "reused_latency_ms": new_latency,
            "latency_saved_ms": max(0.0, old_latency - new_latency),
            "baseline_compute_ms": old_compute,
            "reused_compute_ms": new_compute,
            "baseline_memory_ms": old_memory,
            "reused_memory_ms": new_memory,
            "baseline_flops_mflop": old_flops,
            "executed_flops_mflop": old_flops * (1 - ratio),
            "cache_read_mb": cache_read,
            "cache_output_mb": old_output if ratio > 0 else 0.0,
            "input_a_residency": _residency(row)[0],
            "output_residency": _residency(row)[2],
        })
    return result, pd.DataFrame(records)
