"""Traffic-conserving QKV/online-attention and gated-MLP fusion model."""
from __future__ import annotations

import pandas as pd

from .common import (
    COMM, COMPUTE, FLOPS, GROUP_KEYS, INPUT_A, INPUT_W, MEMORY, OUTPUT,
    TIME, TOTAL_DATA, effective_roofline, transfer_ms,
)


def _candidate(group: pd.DataFrame, position: int, system):
    rows = list(group.iterrows())
    if position + 2 < len(rows):
        indices = [rows[position + offset][0] for offset in range(3)]
        qkv, logit, attend = [rows[position + offset][1] for offset in range(3)]
        valid = (
            str(qkv["Layer Name"]) in {"QKV", "QKV Self"}
            and str(logit["Layer Name"]).startswith("Logit")
            and str(attend["Layer Name"]).startswith("Attend")
        )
        if valid:
            stage = str(qkv["stage"])
            if stage in {"vision_encoder", "vlm_prefill"} or qkv["Layer Name"] == "QKV Self":
                reads = float(logit[INPUT_A]) + float(logit[INPUT_W]) + float(attend[INPUT_W])
                saved_hbm = float(qkv[OUTPUT]) + min(float(qkv[OUTPUT]), reads)
                coverage = "full direct Q/K/V edge"
            else:
                query = min(float(qkv[OUTPUT]), float(logit[INPUT_A]))
                saved_hbm = 2 * query
                coverage = "action-query edge only; prefix K/V retained"
            saved_onchip = float(logit[OUTPUT]) + float(attend[INPUT_A])
            return _record("qkv_online_attention", indices, [qkv, logit, attend],
                           saved_hbm, saved_onchip, coverage, system)
    if position + 1 < len(rows):
        indices = [rows[position][0], rows[position + 1][0]]
        up, down = rows[position][1], rows[position + 1][1]
        if up["Layer Name"] == "up+gate" and down["Layer Name"] == "down":
            saved_hbm = float(up[OUTPUT]) + float(down[INPUT_A])
            return _record("mlp_activation_down", indices, [up, down], saved_hbm, 0.0,
                           "tile-local gate/up activation", system)
    return None


def _record(kind, indices, rows, saved_hbm, saved_onchip, coverage, system):
    compute = sum(float(row[COMPUTE]) for row in rows)
    memory = sum(float(row[MEMORY]) for row in rows)
    communication = sum(float(row[COMM]) for row in rows)
    saved_hbm_ms = transfer_ms(saved_hbm, system.offchip_mem_bw)
    saved_onchip_ms = transfer_ms(saved_onchip, system.onchip_mem_bw)
    adjusted_memory = max(0.0, memory - saved_hbm_ms - saved_onchip_ms)
    fused = effective_roofline(compute, adjusted_memory, communication)
    baseline = sum(float(row[TIME]) for row in rows)
    return {
        "fusion_type": kind,
        "indices": indices,
        "operator_chain": " -> ".join(str(row["Layer Name"]) for row in rows),
        "baseline_latency_ms": baseline,
        "fused_latency_ms": min(baseline, fused),
        "latency_saved_ms": max(0.0, baseline - fused),
        "saved_hbm_mb": saved_hbm,
        "saved_onchip_mb": saved_onchip,
        "saved_hbm_time_ms": saved_hbm_ms,
        "saved_onchip_time_ms": saved_onchip_ms,
        "compute_work_ms": compute,
        "adjusted_memory_work_ms": adjusted_memory,
        "communication_work_ms": communication,
        "coverage": coverage,
    }


def apply_fusion(operators: pd.DataFrame, system) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse non-overlapping fusion chains and return their audit records."""
    fused_rows, records = [], []
    for keys, group in operators.groupby(GROUP_KEYS, sort=False, dropna=False):
        group = group.sort_index()
        rows = list(group.iterrows())
        position = 0
        while position < len(rows):
            candidate = _candidate(group, position, system)
            if candidate is None:
                fused_rows.append(rows[position][1].copy())
                position += 1
                continue
            length = len(candidate["indices"])
            chain = [rows[position + offset][1] for offset in range(length)]
            row = chain[0].copy()
            row["Layer Name"] = candidate["operator_chain"]
            row["Op Type"] = "Fused"
            row[COMPUTE] = candidate["compute_work_ms"]
            row[MEMORY] = candidate["adjusted_memory_work_ms"]
            row[COMM] = candidate["communication_work_ms"]
            row[TIME] = candidate["fused_latency_ms"]
            row[FLOPS] = sum(float(item[FLOPS]) for item in chain)
            row[INPUT_A] = float(chain[0][INPUT_A])
            row[INPUT_W] = sum(float(item[INPUT_W]) for item in chain)
            row[OUTPUT] = float(chain[-1][OUTPUT])
            row[TOTAL_DATA] = max(0.0, sum(float(item[TOTAL_DATA]) for item in chain) - candidate["saved_hbm_mb"])
            row["source_indices"] = ";".join(str(item["source_indices"]) for item in chain)
            fused_rows.append(row)
            record = dict(zip(GROUP_KEYS, keys))
            record.update({key: value for key, value in candidate.items() if key != "indices"})
            record["source_indices"] = ";".join(map(str, candidate["indices"]))
            records.append(record)
            position += length
    result = pd.DataFrame(fused_rows).reset_index(drop=True)
    return result, pd.DataFrame(records)
