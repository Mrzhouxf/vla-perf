"""Fine-grained ready-time pipeline model.

The model splits a legal producer-consumer segment into tiles.  The first tile
pays the serial fill/drain bubble; subsequent tiles approach the segment's
aggregate compute/memory/communication lower bound.  This is an analytical
double-buffered wavefront, not a claim about CUDA kernel scheduling.
"""
from __future__ import annotations

import pandas as pd

from .common import COMM, COMPUTE, GROUP_KEYS, MEMORY, TIME


def _is_attention_barrier(producer: str, consumer: str) -> bool:
    # Self-attention Q/K/V must be complete before a full-key query row can be
    # evaluated in the unfused baseline.  Online-attention fusion removes this
    # explicit edge by collapsing the chain before pipeline simulation.
    return "QKV" in producer and "Logit" in consumer


def can_stream(producer: str, consumer: str) -> bool:
    if _is_attention_barrier(str(producer), str(consumer)):
        return False
    if producer in {"embeddings", "classifier"} or consumer in {"embeddings", "classifier"}:
        return False
    return True


def segment_latency(segment: pd.DataFrame, tiles: int) -> dict:
    if tiles < 1:
        raise ValueError("tiles must be positive")
    serial = float(segment[TIME].sum())
    compute = float(segment[COMPUTE].sum())
    memory = float(segment[MEMORY].sum())
    communication = float(segment[COMM].sum())
    lower = max(compute, memory, communication, 0.0)
    if len(segment) < 2 or tiles == 1 or serial == 0:
        pipelined = serial
    else:
        # At one tile this equals serial; at infinite tiles it approaches the
        # conserved aggregate resource lower bound.
        pipelined = lower + (serial - lower) / tiles
    pipelined = min(serial, max(lower, pipelined))
    return {
        "operators": len(segment),
        "operator_chain": " -> ".join(segment["Layer Name"].astype(str)),
        "serial_latency_ms": serial,
        "compute_work_ms": compute,
        "memory_work_ms": memory,
        "communication_work_ms": communication,
        "resource_lower_bound_ms": lower,
        "pipeline_latency_ms": pipelined,
        "latency_saved_ms": serial - pipelined,
    }


def simulate_pipeline(operators: pd.DataFrame, tiles: int = 8) -> tuple[pd.Series, pd.DataFrame]:
    records = []
    for keys, group in operators.groupby(GROUP_KEYS, sort=False, dropna=False):
        group = group.sort_index()
        start = 0
        rows = list(group.iterrows())
        for position in range(1, len(rows) + 1):
            boundary = position == len(rows)
            if not boundary:
                boundary = not can_stream(
                    str(rows[position - 1][1]["Layer Name"]),
                    str(rows[position][1]["Layer Name"]),
                )
            if boundary:
                segment = group.iloc[start:position]
                record = dict(zip(GROUP_KEYS, keys))
                record.update(segment_latency(segment, tiles))
                record["tiles"] = tiles
                records.append(record)
                start = position
    details = pd.DataFrame(records)
    latency = details.groupby("stage", sort=False)["pipeline_latency_ms"].sum()
    return latency, details
