#!/usr/bin/env python3
"""Fast-OverlaPIM-style ready-time mapping experiment for the π0 VLM prefix.

Fast-OverlaPIM is PIM-specific: its resources are banks and its tasks are data
spaces.  This GPU adaptation uses *logical execution partitions* as resources
and VLM prefix chunks as data spaces.  This mapping is disabled by default:
stock SigLIP does not expose incrementally-ready output tokens and the baseline
does not specify a partitioned GPU execution model.  It compares, only under
explicit streaming assumptions:

1. raw mapping: each Transformer layer waits for a group containing chunks
   with different input ready times;
2. transformed mapping: tasks are globally ordered by ready time and placed on
   the earliest free logical resource, allowing an early VLM chunk to progress
   to its next layer without waiting for late vision chunks.

This is a reproducible analytical mapping study; it is not a CUDA scheduler.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_tasks(operators: pd.DataFrame, vision_latency_ms: float, chunks: int) -> pd.DataFrame:
    vlm = operators[operators["stage"] == "vlm_prefill"].copy()
    # `repeatN` is the materialized Transformer block index emitted by the profiler.
    vlm["layer"] = pd.to_numeric(vlm["repeat_instance"].str.extract(r"repeat(\d+)")[0], errors="coerce")
    # The root embedding/classifier rows carry zero latency and are outside the
    # repeated Transformer blocks being mapped here.
    vlm = vlm.dropna(subset=["layer"])
    vlm["layer"] = vlm["layer"].astype(int)
    per_layer = vlm.groupby("layer")["Latency (msec)"].sum().sort_index()
    attention = vlm[vlm["Op Type"].isin(["Logit", "Attend"])]["Latency (msec)"].sum()
    attention_fraction = attention / vlm["Latency (msec)"].sum()
    # Linear non-attention work + quadratic causal-attention work, normalized so
    # the chunk durations sum exactly to the GenZ VLM latency.
    boundaries = [index / chunks for index in range(chunks + 1)]
    shares = [
        (1 - attention_fraction) * (boundaries[i + 1] - boundaries[i])
        + attention_fraction * (boundaries[i + 1] ** 2 - boundaries[i] ** 2)
        for i in range(chunks)
    ]
    rows = []
    for layer, layer_latency in per_layer.items():
        for chunk, share in enumerate(shares):
            rows.append({
                "layer": layer,
                "chunk": chunk,
                "duration_ms": layer_latency * share,
                "vision_ready_ms": vision_latency_ms * (chunk + 1) / chunks,
            })
    return pd.DataFrame(rows)


def raw_mapping(tasks: pd.DataFrame, resources: int) -> pd.DataFrame:
    """Original layer-major grouping: the group waits for its latest input."""
    records, group_end = [], 0.0
    for layer, group in tasks.groupby("layer", sort=True):
        group_ready = max(group_end, group["vision_ready_ms"].max())
        resource_end = [group_ready] * resources
        for _, task in group.sort_values("chunk").iterrows():
            resource = min(range(resources), key=lambda index: resource_end[index])
            start = max(resource_end[resource], group_ready)
            end = start + task.duration_ms
            resource_end[resource] = end
            records.append({**task.to_dict(), "mapping": "raw_layer_major", "resource": resource,
                            "start_ms": start, "end_ms": end,
                            "wait_for_ready_ms": start - task.vision_ready_ms})
        group_end = max(resource_end)
    return pd.DataFrame(records)


def transformed_mapping(tasks: pd.DataFrame, resources: int) -> pd.DataFrame:
    """Ready-time transformation with legal causal chunk/layer dependencies."""
    pending = {(int(row.layer), int(row.chunk)): row for row in tasks.itertuples(index=False)}
    finish, resource_end, records = {}, [0.0] * resources, []
    while pending:
        candidates = []
        for (layer, chunk), task in pending.items():
            has_previous_layer = layer == 0 or (layer - 1, chunk) in finish
            has_previous_chunk = chunk == 0 or (layer, chunk - 1) in finish
            if has_previous_layer and has_previous_chunk:
                parent_layer_end = 0.0 if layer == 0 else finish[(layer - 1, chunk)]
                parent_chunk_end = 0.0 if chunk == 0 else finish[(layer, chunk - 1)]
                ready = max(task.vision_ready_ms, parent_layer_end, parent_chunk_end)
                candidates.append((ready, layer, chunk, task))
        resource = min(range(resources), key=lambda index: resource_end[index])
        ready, layer, chunk, task = min(
            candidates, key=lambda item: (max(resource_end[resource], item[0]), item[0], item[1], item[2]))
        start = max(resource_end[resource], ready)
        end = start + task.duration_ms
        resource_end[resource], finish[(layer, chunk)] = end, end
        records.append({"layer": layer, "chunk": chunk, "duration_ms": task.duration_ms,
                        "vision_ready_ms": task.vision_ready_ms, "mapping": "ready_time_transformed",
                        "resource": resource, "start_ms": start, "end_ms": end,
                        "wait_for_ready_ms": start - task.vision_ready_ms})
        del pending[(layer, chunk)]
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--resources", type=int, default=2, help="fixed logical GPU execution partitions")
    parser.add_argument("--chunks", type=int, default=3)
    parser.add_argument("--assume-streaming-vision", action="store_true",
                        help="enable only for an encoder proven to emit usable token chunks")
    parser.add_argument("--assume-partitioned-gpu", action="store_true",
                        help="enable only with a validated fixed-SM/HBM partition model")
    args = parser.parse_args()
    if args.resources < 1 or args.chunks < 1:
        raise ValueError("resources and chunks must be positive")

    stages = pd.read_csv(args.profile_dir / "stage_breakdown.csv")
    operators = pd.read_csv(args.profile_dir / "operator_breakdown.csv")
    vision_ms = float(stages.loc[stages.stage == "vision_encoder", "baseline_latency_ms"].iloc[0])
    vlm_ms = float(stages.loc[stages.stage == "vlm_prefill", "baseline_latency_ms"].iloc[0])
    enabled = args.assume_streaming_vision and args.assume_partitioned_gpu
    if enabled:
        tasks = build_tasks(operators, vision_ms, args.chunks)
        # Each partition receives 1/resources of the original fixed GPU
        # capacity, so task latency scales with the partition count.
        tasks["duration_ms"] *= args.resources
        raw, transformed = raw_mapping(tasks, args.resources), transformed_mapping(tasks, args.resources)
        raw_end, transformed_end = raw.end_ms.max(), transformed.end_ms.max()
    else:
        tasks = pd.DataFrame([{"layer": 0, "chunk": 0, "duration_ms": vlm_ms,
                               "vision_ready_ms": vision_ms}])
        raw = tasks.assign(mapping="dependency_safe_no_streaming", resource=0,
                           start_ms=vision_ms, end_ms=vision_ms + vlm_ms, wait_for_ready_ms=0.0)
        transformed = raw.copy()
        raw_end = transformed_end = float(raw.end_ms.iloc[0])
    summary = pd.DataFrame([{
        "resources": args.resources, "chunks": args.chunks,
        "mapping_enabled": enabled,
        "raw_mapping_makespan_ms": raw_end, "ready_time_transformed_makespan_ms": transformed_end,
        "mapping_speedup": raw_end / transformed_end,
        "raw_total_wait_for_input_ms": raw.wait_for_ready_ms.sum(),
        "transformed_total_wait_for_input_ms": transformed.wait_for_ready_ms.sum(),
    }])
    tasks.to_csv(args.profile_dir / "fast_overlapim_task_spaces.csv", index=False)
    raw.to_csv(args.profile_dir / "fast_overlapim_raw_mapping_schedule.csv", index=False)
    transformed.to_csv(args.profile_dir / "fast_overlapim_transformed_mapping_schedule.csv", index=False)
    summary.to_csv(args.profile_dir / "fast_overlapim_summary.csv", index=False)
    report = f"""# Fast-OverlaPIM-style VLM mapping experiment

固定 {args.resources} 个逻辑执行分区、{args.chunks} 个视觉/VLM prefix data spaces。

| Mapping | Vision→VLM makespan from image input (ms) | Accumulated task wait (task·ms) |
|---|---:|---:|
| Raw layer-major mapping | {raw_end:.4f} | {raw.wait_for_ready_ms.sum():.4f} |
| Ready-time transformed mapping | {transformed_end:.4f} | {transformed.wait_for_ready_ms.sum():.4f} |

Transformation speedup: **{raw_end / transformed_end:.4f}×**.

`mapping_enabled={enabled}`. With the default `False`, this is the dependency-safe result: SigLIP completes before the VLM starts, so Fast-OverlaPIM has no legal cross-stage data spaces to transform. To enable the analytical mapping experiment, both `--assume-streaming-vision` and `--assume-partitioned-gpu` must be supplied. Then `fast_overlapim_raw_mapping_schedule.csv` preserves a group containing chunks with different ready times, while `fast_overlapim_transformed_mapping_schedule.csv` sorts ready tasks and uses the same fixed resource budget.

This is a GPU-adapted analytical experiment, not a claim that CUDA streams equal independent PIM banks. When enabled, causal dependencies include both the prior layer of a chunk and the prior chunk of a layer; cross-chunk attention work remains a quadratic approximation in `fast_overlapim_task_spaces.csv`.
"""
    (args.profile_dir / "FAST_OVERLAPIM_REPORT.md").write_text(report, encoding="utf-8")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
