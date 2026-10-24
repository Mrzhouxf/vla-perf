"""Prefix-KV producer to first Action Attention direct-stream transform."""
from __future__ import annotations

import pandas as pd

from .common import COMM, COMPUTE, INPUT_W, MEMORY, TIME, transfer_ms


def apply_first_pass_kv_streaming(operators: pd.DataFrame, system) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = operators.copy()
    records = []
    mask = (
        (result.stage == "action_expert")
        & (result.stage_invocation == 1)
        & result["Layer Name"].isin(["Logit Pre", "Attend Pre"])
    )
    for index, row in result[mask].iterrows():
        kv_mb = float(row[INPUT_W])
        offchip = transfer_ms(kv_mb, system.offchip_mem_bw) / system.memory_efficiency
        onchip = transfer_ms(kv_mb, system.onchip_mem_bw) / system.memory_efficiency
        saved = max(0.0, offchip - onchip)
        new_memory = max(0.0, float(row[MEMORY]) - saved)
        new_latency = max(float(row[COMPUTE]), new_memory, float(row[COMM]))
        result.at[index, MEMORY] = new_memory
        result.at[index, TIME] = new_latency
        records.append({
            "operator_index": index,
            "repeat_instance": row.repeat_instance,
            "operator": row["Layer Name"],
            "prefix_kv_mb": kv_mb,
            "baseline_memory_ms": float(row[MEMORY]),
            "streamed_memory_ms": new_memory,
            "hbm_to_onchip_time_saved_ms": saved,
            "baseline_latency_ms": float(row[TIME]),
            "streamed_latency_ms": new_latency,
            "standalone_latency_saved_ms": float(row[TIME]) - new_latency,
            "cache_write_retained": True,
            "later_denoise_hbm_reads_retained": True,
        })
    return result, pd.DataFrame(records)
