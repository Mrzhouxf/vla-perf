#!/usr/bin/env python3
"""Operator-level VLA profiling and dependency-safe fusion what-if model.

The input is the validated GenZ operator breakdown produced by
``pipeline_prototype/profile_and_pipeline.py``.  This file never modifies GenZ.

Fusion changes only intermediate tensor residency.  FLOPs are conserved, and
the fused latency of a chain is the roofline lower bound
``max(sum(compute), adjusted_memory, sum(communication))``.  HBM/on-chip bytes
are subtracted only for explicitly matched producer-consumer edges.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
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


def transfer_ms(size_mb: float, raw_bandwidth: float) -> float:
    return Unit().unit_to_raw(float(size_mb), type="M") / raw_bandwidth * 1e3


def stage_operator_breakdown(ops: pd.DataFrame) -> pd.DataFrame:
    columns = [TIME, COMPUTE, MEMORY, COMM, INPUT_A, INPUT_W, OUTPUT, "Total Data (MB)", "Num ops (MFLOP)"]
    result = ops.groupby(["stage", "Layer Name", "Op Type", "Bound"], dropna=False)[columns].sum().reset_index()
    stage_latency = result.groupby("stage")[TIME].transform("sum")
    result["stage_latency_share_pct"] = 100 * result[TIME] / stage_latency
    result["compute_utilization_pct"] = result.apply(lambda row: 100 * row[COMPUTE] / row[TIME] if row[TIME] else 0, axis=1)
    result["memory_utilization_pct"] = result.apply(lambda row: 100 * row[MEMORY] / row[TIME] if row[TIME] else 0, axis=1)
    return result.sort_values(["stage", TIME], ascending=[True, False])


def execution_groups(ops: pd.DataFrame):
    keys = ["stage", "stage_invocation", "repeat_instance"]
    for key, group in ops.groupby(keys, sort=False, dropna=False):
        yield key, group.sort_index()


def find_candidates(ops: pd.DataFrame, system) -> pd.DataFrame:
    records = []
    candidate_id = 0
    for (stage, invocation, repeat_instance), group in execution_groups(ops):
        rows = list(group.iterrows())
        position = 0
        while position < len(rows):
            # QKV + online attention. This subsumes the Logit+Attend pair.
            if position + 2 < len(rows):
                index0, qkv = rows[position]
                index1, logit = rows[position + 1]
                index2, attend = rows[position + 2]
                qkv_names = {"QKV", "QKV Self"}
                valid_pair = str(logit["Layer Name"]).startswith("Logit") and str(attend["Layer Name"]).startswith("Attend")
                if qkv["Layer Name"] in qkv_names and valid_pair:
                    if stage in {"vision_encoder", "vlm_prefill"} or qkv["Layer Name"] == "QKV Self":
                        consumer_reads = float(logit[INPUT_A]) + float(logit[INPUT_W]) + float(attend[INPUT_W])
                        saved_hbm_mb = float(qkv[OUTPUT]) + min(float(qkv[OUTPUT]), consumer_reads)
                        coverage = "full Q/K/V producer-consumer edge"
                    else:
                        # Prefix attention consumes prefix K/V; only the action Q
                        # is a direct edge from this QKV row.
                        q_bytes = min(float(qkv[OUTPUT]), float(logit[INPUT_A]))
                        saved_hbm_mb = 2 * q_bytes
                        coverage = "query edge only; prefix K/V remains in HBM"
                    saved_onchip_mb = float(logit[OUTPUT]) + float(attend[INPUT_A])
                    records.append(candidate_record(
                        candidate_id, stage, invocation, repeat_instance, "qkv_online_attention",
                        [index0, index1, index2], [qkv, logit, attend], saved_hbm_mb,
                        saved_onchip_mb, coverage, system))
                    candidate_id += 1

            # Online softmax/attention alone. GenZ already keeps score tiles on-chip.
            if position + 1 < len(rows):
                index0, logit = rows[position]
                index1, attend = rows[position + 1]
                if str(logit["Layer Name"]).startswith("Logit") and str(attend["Layer Name"]).startswith("Attend"):
                    saved_onchip_mb = float(logit[OUTPUT]) + float(attend[INPUT_A])
                    records.append(candidate_record(
                        candidate_id, stage, invocation, repeat_instance, "online_attention_only",
                        [index0, index1], [logit, attend], 0.0, saved_onchip_mb,
                        "score matrix is already on-chip in GenZ; no HBM saving", system))
                    candidate_id += 1

            # Gate/up projection + activation + down projection.
            if position + 1 < len(rows):
                index0, up = rows[position]
                index1, down = rows[position + 1]
                if up["Layer Name"] == "up+gate" and down["Layer Name"] == "down":
                    saved_hbm_mb = float(up[OUTPUT]) + float(down[INPUT_A])
                    records.append(candidate_record(
                        candidate_id, stage, invocation, repeat_instance, "mlp_activation_down",
                        [index0, index1], [up, down], saved_hbm_mb, 0.0,
                        "tile-local gate/up activation consumed by down projection", system))
                    candidate_id += 1
            position += 1
    return pd.DataFrame(records)


def candidate_record(candidate_id, stage, invocation, repeat_instance, fusion_type,
                     indices, rows, saved_hbm_mb, saved_onchip_mb, coverage, system):
    baseline = sum(float(row[TIME]) for row in rows)
    compute = sum(float(row[COMPUTE]) for row in rows)
    memory = sum(float(row[MEMORY]) for row in rows)
    communication = sum(float(row[COMM]) for row in rows)
    saved_hbm_ms = transfer_ms(saved_hbm_mb, system.offchip_mem_bw)
    saved_onchip_ms = transfer_ms(saved_onchip_mb, system.onchip_mem_bw)
    fused_memory = max(0.0, memory - saved_hbm_ms - saved_onchip_ms)
    fused = max(compute, fused_memory, communication)
    return {
        "candidate_id": candidate_id, "stage": stage, "stage_invocation": invocation,
        "repeat_instance": repeat_instance, "fusion_type": fusion_type,
        "operator_indices": ";".join(map(str, indices)),
        "operator_chain": " -> ".join(str(row["Layer Name"]) for row in rows),
        "baseline_latency_ms": baseline, "fused_latency_ms": fused,
        "latency_saved_ms": max(0.0, baseline - fused),
        "saved_hbm_mb": saved_hbm_mb, "saved_onchip_mb": saved_onchip_mb,
        "saved_hbm_time_ms": saved_hbm_ms, "saved_onchip_time_ms": saved_onchip_ms,
        "compute_work_ms": compute, "adjusted_memory_work_ms": fused_memory,
        "flops_conserved": True, "coverage": coverage,
    }


def summarize_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    numeric = ["baseline_latency_ms", "fused_latency_ms", "latency_saved_ms", "saved_hbm_mb",
               "saved_onchip_mb", "saved_hbm_time_ms", "saved_onchip_time_ms"]
    summary = candidates.groupby(["stage", "fusion_type"])[numeric].sum().reset_index()
    counts = candidates.groupby(["stage", "fusion_type"]).size().rename("occurrences").reset_index()
    summary = summary.merge(counts, on=["stage", "fusion_type"])
    summary["chain_speedup"] = summary["baseline_latency_ms"] / summary["fused_latency_ms"]
    return summary.sort_values("latency_saved_ms", ascending=False)


def scenario_summary(ops: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    baseline_by_stage = ops.groupby("stage")[TIME].sum()
    selections = {
        "baseline": set(),
        "online_attention_only": {"online_attention_only"},
        "qkv_online_attention": {"qkv_online_attention"},
        "mlp_activation_down": {"mlp_activation_down"},
        "combined_qkv_attention_mlp": {"qkv_online_attention", "mlp_activation_down"},
    }
    records = []
    for scenario, fusion_types in selections.items():
        selected = candidates[candidates.fusion_type.isin(fusion_types)]
        saved_by_stage = selected.groupby("stage")["latency_saved_ms"].sum()
        for stage, baseline in baseline_by_stage.items():
            saved = float(saved_by_stage.get(stage, 0.0))
            records.append({"scope": stage, "scenario": scenario, "baseline_latency_ms": baseline,
                            "fused_latency_ms": baseline - saved, "latency_saved_ms": saved,
                            "speedup": baseline / (baseline - saved)})
        baseline_total = float(baseline_by_stage.sum())
        saved_total = float(selected.latency_saved_ms.sum())
        records.append({"scope": "end_to_end", "scenario": scenario, "baseline_latency_ms": baseline_total,
                        "fused_latency_ms": baseline_total - saved_total, "latency_saved_ms": saved_total,
                        "speedup": baseline_total / (baseline_total - saved_total)})
    return pd.DataFrame(records)


def adjacent_pairs(ops: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    eligible = {}
    for row in candidates.itertuples(index=False):
        indices = tuple(map(int, row.operator_indices.split(";")))
        for left, right in zip(indices, indices[1:]):
            eligible.setdefault((left, right), set()).add(row.fusion_type)
    records = []
    for (stage, invocation, repeat_instance), group in execution_groups(ops):
        rows = list(group.iterrows())
        for (left_index, left), (right_index, right) in zip(rows, rows[1:]):
            records.append({"stage": stage, "stage_invocation": invocation,
                            "repeat_instance": repeat_instance,
                            "producer": left["Layer Name"], "consumer": right["Layer Name"],
                            "pair_latency_ms": float(left[TIME]) + float(right[TIME]),
                            "eligible_fusions": ";".join(sorted(eligible.get((left_index, right_index), set()))) or "none"})
    return pd.DataFrame(records)


def operator_instances(ops: pd.DataFrame) -> pd.DataFrame:
    result = ops.copy()
    result.insert(0, "operator_instance_id", range(len(result)))
    group_keys = ["stage", "stage_invocation", "repeat_instance"]
    result["previous_operator"] = result.groupby(group_keys, sort=False)["Layer Name"].shift(1).fillna("stage/layer start")
    result["next_operator"] = result.groupby(group_keys, sort=False)["Layer Name"].shift(-1).fillna("stage/layer end")
    result["compute_utilization_pct"] = result.apply(lambda row: 100 * row[COMPUTE] / row[TIME] if row[TIME] else 0, axis=1)
    result["memory_utilization_pct"] = result.apply(lambda row: 100 * row[MEMORY] / row[TIME] if row[TIME] else 0, axis=1)
    return result


def write_report(output, args, stage_ops, candidate_summary, scenarios):
    combined = scenarios[(scenarios.scope == "end_to_end") & (scenarios.scenario == "combined_qkv_attention_mlp")].iloc[0]
    stage_table = scenarios[(scenarios.scope != "end_to_end") & (scenarios.scenario == "combined_qkv_attention_mlp")]
    unsupported = pd.DataFrame([
        {"candidate": "AdaRMSNorm + QKV + RoPE", "status": "not represented in current GenZ π0 graph"},
        {"candidate": "Output projection + Euler + next input projection", "status": "Euler/action boundary absent from current GenZ graph"},
        {"candidate": "VLM layer KV -> first AE layer attention", "status": "needs a joint layer-level π0 graph and resource-sharing model"},
    ])
    unsupported.to_csv(output / "unsupported_vla_specific_candidates.csv", index=False)
    lines = [
        "# VLA operator fusion experiment",
        "",
        f"Configuration: {args.gpu} / {args.bits}. FLOPs are unchanged; only explicitly matched intermediate traffic is removed.",
        f"Combined QKV-attention + MLP fusion changes the modeled end-to-end latency from **{combined.baseline_latency_ms:.4f} ms** to **{combined.fused_latency_ms:.4f} ms** ({combined.speedup:.4f}×).",
        "",
        "## Per-stage result",
        "",
        stage_table.to_markdown(index=False, floatfmt=".5f"),
        "",
        "## Fusion candidates",
        "",
        candidate_summary.to_markdown(index=False, floatfmt=".5f"),
        "",
        "## Operator latency breakdown",
        "",
        stage_ops.to_markdown(index=False, floatfmt=".5f"),
        "",
        "## Important boundaries",
        "",
        "`Logit -> Attend` is already modeled with its score tensor on-chip by GenZ (`C_onchip/A_onchip`). The online-attention result therefore removes only on-chip score materialization, not fictitious HBM traffic. Softmax exp/max/sum arithmetic is not a separate GenZ row, so its arithmetic latency is absent from both baseline and fused results.",
        "",
        "AdaRMSNorm/RoPE/Euler fusion is not assigned a numerical speedup because those operators are absent from the current model. See `unsupported_vla_specific_candidates.csv`.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def plot_results(scenarios: pd.DataFrame, output: Path):
    view = scenarios[(scenarios.scope != "end_to_end") & scenarios.scenario.isin(["baseline", "combined_qkv_attention_mlp"])]
    pivot = view.pivot(index="scope", columns="scenario", values="fused_latency_ms")
    pivot.plot(kind="bar", figsize=(8, 4.5), rot=15)
    plt.ylabel("Latency (ms)")
    plt.title("Per-stage operator-fusion latency")
    plt.tight_layout()
    plt.savefig(output / "operator_fusion_stage_latency.png", dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--bits", default="bf16")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    ops = pd.read_csv(args.profile_dir / "operator_breakdown.csv")
    system = get_inference_system(system_name=args.gpu, bits=args.bits)
    stage_ops = stage_operator_breakdown(ops)
    candidates = find_candidates(ops, system)
    candidate_summary = summarize_candidates(candidates)
    scenarios = scenario_summary(ops, candidates)
    pairs = adjacent_pairs(ops, candidates)
    instances = operator_instances(ops)

    instances.to_csv(args.output / "operator_instance_breakdown.csv", index=False)
    stage_ops.to_csv(args.output / "operator_stage_breakdown.csv", index=False)
    candidates.to_csv(args.output / "fusion_candidates_detailed.csv", index=False)
    candidate_summary.to_csv(args.output / "fusion_candidates_summary.csv", index=False)
    scenarios.to_csv(args.output / "fusion_scenarios.csv", index=False)
    pairs.to_csv(args.output / "adjacent_operator_pairs.csv", index=False)
    plot_results(scenarios, args.output)
    write_report(args.output, args, stage_ops, candidate_summary, scenarios)
    print(scenarios[scenarios.scope == "end_to_end"].to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
