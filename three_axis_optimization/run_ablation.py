#!/usr/bin/env python3
"""Run pipeline, operator-fusion and cross-frame-reuse ablations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .common import TIME, assert_valid_operators, load_profile, stage_latency, system_for
from .fusion import apply_fusion
from .pipeline import simulate_pipeline
from .reuse import apply_reuse

STAGE_ORDER = ["vision_encoder", "vlm_prefill", "action_expert"]
SCENARIOS = [
    ("baseline", False, False, False),
    ("pipeline_only", True, False, False),
    ("fusion_only", False, True, False),
    ("reuse_only", False, False, True),
    ("pipeline_fusion", True, True, False),
    ("pipeline_reuse", True, False, True),
    ("fusion_reuse", False, True, True),
    ("all_three", True, True, True),
]


def evaluate_frame(operators: pd.DataFrame, system, use_pipeline: bool,
                   use_fusion: bool, tiles: int) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    working = operators.copy()
    fusion_details = pd.DataFrame()
    pipeline_details = pd.DataFrame()
    if use_fusion:
        working, fusion_details = apply_fusion(working, system)
    assert_valid_operators(working)
    if use_pipeline:
        latency, pipeline_details = simulate_pipeline(working, tiles)
    else:
        latency = stage_latency(working)
    return latency.reindex(STAGE_ORDER, fill_value=0.0), fusion_details, pipeline_details


def run_scenarios(operators: pd.DataFrame, system, args) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_stage = stage_latency(operators).reindex(STAGE_ORDER, fill_value=0.0)
    baseline_total = float(baseline_stage.sum())
    reuse_ops, _ = apply_reuse(
        operators, system, similarity=args.similarity, threshold=args.reuse_threshold,
        maximum=args.max_reuse, layer_decay=args.layer_decay,
        vlm_propagation=args.vlm_propagation,
        visual_prefix_fraction=args.visual_prefix_fraction,
    )
    scenario_rows, stage_rows = [], []
    for name, pipeline_enabled, fusion_enabled, reuse_enabled in SCENARIOS:
        first, _, _ = evaluate_frame(
            operators, system, pipeline_enabled, fusion_enabled, args.tiles
        )
        if reuse_enabled:
            later, _, _ = evaluate_frame(
                reuse_ops, system, pipeline_enabled, fusion_enabled, args.tiles
            )
            later = later.copy()
            later.loc["vision_encoder"] += args.validation_overhead_ms
        else:
            later = first
        average = (first + (args.frames - 1) * later) / args.frames
        total = float(average.sum())
        for stage in STAGE_ORDER:
            stage_rows.append({
                "scenario": name,
                "stage": stage,
                "baseline_latency_ms": float(baseline_stage[stage]),
                "first_frame_latency_ms": float(first[stage]),
                "later_frame_latency_ms": float(later[stage]),
                "average_latency_ms": float(average[stage]),
                "stage_speedup": float(baseline_stage[stage] / average[stage]) if average[stage] else 1.0,
            })
        scenario_rows.append({
            "scenario": name,
            "pipeline": pipeline_enabled,
            "fusion": fusion_enabled,
            "reuse": reuse_enabled,
            "frames": args.frames,
            "baseline_average_latency_ms": baseline_total,
            "first_frame_latency_ms": float(first.sum()),
            "later_frame_latency_ms": float(later.sum()),
            "average_latency_ms": total,
            "latency_saved_ms": baseline_total - total,
            "speedup": baseline_total / total,
            "latency_reduction_pct": 100 * (baseline_total - total) / baseline_total,
        })
    return pd.DataFrame(scenario_rows), pd.DataFrame(stage_rows)


def reuse_sweep(operators: pd.DataFrame, system, args) -> pd.DataFrame:
    baseline = float(operators[TIME].sum())
    rows = []
    for similarity in [0.70, 0.80, 0.85, 0.90, 0.92, 0.95, 0.98]:
        reused, details = apply_reuse(
            operators, system, similarity=similarity, threshold=args.reuse_threshold,
            maximum=args.max_reuse, layer_decay=args.layer_decay,
            vlm_propagation=args.vlm_propagation,
            visual_prefix_fraction=args.visual_prefix_fraction,
        )
        later = float(reused[TIME].sum()) + args.validation_overhead_ms
        average = (baseline + (args.frames - 1) * later) / args.frames
        rows.append({
            "similarity": similarity,
            "cache_gate_passed": similarity >= args.reuse_threshold,
            "later_frame_latency_ms": later,
            "average_latency_ms": average,
            "speedup": baseline / average,
            "latency_reduction_pct": 100 * (baseline - average) / baseline,
            "modeled_cache_footprint_mb": float(details["cache_output_mb"].sum()),
        })
    return pd.DataFrame(rows)


def plot_results(ablation: pd.DataFrame, stages: pd.DataFrame,
                 sweep: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    view = stages.pivot(index="scenario", columns="stage", values="average_latency_ms")
    view = view.reindex(ablation.scenario)
    view[STAGE_ORDER].plot(kind="bar", stacked=True, ax=axes[0], rot=35)
    axes[0].set_ylabel("average latency (ms)")
    axes[0].set_title("Three-axis ablation")
    axes[0].legend(fontsize=8)
    axes[1].plot(sweep.similarity, sweep.speedup, marker="o")
    axes[1].axvline(float(sweep.loc[sweep.cache_gate_passed, "similarity"].min()),
                    color="gray", linestyle="--", label="reuse threshold")
    axes[1].set_xlabel("inter-frame similarity")
    axes[1].set_ylabel("average speedup")
    axes[1].set_title("Reuse sensitivity")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output / "three_axis_ablation.png", dpi=180)
    plt.close(fig)


def write_report(args, ablation, stages, pipeline_details, fusion_details,
                 reuse_details, sweep, output: Path) -> None:
    singles = ablation[ablation.scenario.isin(["baseline", "pipeline_only", "fusion_only", "reuse_only", "all_three"])]
    all_three = ablation[ablation.scenario == "all_three"].iloc[0]
    cache_mb = float(reuse_details.cache_output_mb.sum())
    requested = int((reuse_details.requested_reuse_ratio > 0).sum())
    applied = int((reuse_details.reuse_ratio > 0).sum())
    fallbacks = requested - applied
    lines = [
        "# VLA-Perf 三类优化统一消融实验",
        "",
        f"配置：**{args.gpu}/{args.bits}**，{args.frames} 帧；流水线 tile={args.tiles}；跨帧相似度={args.similarity:.2f}，阈值={args.reuse_threshold:.2f}。",
        f"三项同时启用时，平均端到端延迟从 **{all_three.baseline_average_latency_ms:.4f} ms** 降至 **{all_three.average_latency_ms:.4f} ms**，模拟加速 **{all_three.speedup:.4f}×**。",
        "",
        "## 单项与组合消融",
        "",
        ablation.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## 三个阶段的延迟",
        "",
        stages.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## 1. Ready-time 流水线",
        "",
        "算子链按 token/query tile 切分。producer 的一个 tile ready 后，consumer 即可启动；QKV→Logit 在未融合时保留全 K/V 屏障。每个 pipeline segment 保持 compute、memory、communication 工作量不变，其延迟在串行值和聚合资源下界之间插值。该结果代表支持双缓冲/持久化调度时的 analytical estimate，不是现有 CUDA runtime 的直接测量。",
        "",
        pipeline_details.groupby("stage")[["serial_latency_ms", "resource_lower_bound_ms", "pipeline_latency_ms", "latency_saved_ms"]].sum().reset_index().to_markdown(index=False, floatfmt=".6f"),
        "",
        "## 2. 算子融合",
        "",
        "实现 QKV→online attention 和 gate/up→activation→down 两类融合。FLOPs 不减少，仅扣除直接 producer-consumer 中间张量的 HBM/片上物化；Action prefix attention 只扣 query 边，prefix K/V 仍保留。",
        "",
        fusion_details.groupby(["stage", "fusion_type"])[["baseline_latency_ms", "fused_latency_ms", "latency_saved_ms", "saved_hbm_mb", "saved_onchip_mb"]].sum().reset_index().to_markdown(index=False, floatfmt=".6f"),
        "",
        "## 3. 跨帧数据重用",
        "",
        f"相似度低于阈值时完全回退；通过阈值后，首层视觉 token 复用率为 similarity×{args.max_reuse:.2f}，再逐层乘以 {args.layer_decay:.2f} 的可信度衰减。VLM 只允许 visual-prefix_fraction×{args.vlm_propagation:.2f} 的保守传播，Action Expert 因状态、噪声和 timestep 改变而默认不复用。命中部分按 HBM cache read 计费，只消除对应计算、输入读取和 producer write；若 cache 路径慢于重算则逐算子回退。本次 {requested} 个候选中应用 {applied} 个、回退 {fallbacks} 个；每个后续帧加入 {args.validation_overhead_ms:.4f} ms 验证开销；模拟缓存足迹为 {cache_mb:.2f} MB。",
        "",
        sweep.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## 解释边界",
        "",
        "- 所有结果均为基于 GenZ counters 的 Roofline/traffic what-if 模拟，不是实际 CUDA/Triton kernel 实测。",
        "- 当前 π0 GenZ 图缺少 AdaRMSNorm、RoPE、激活函数、Residual 和 Euler Update，三类优化都不会为这些缺失算子虚构收益。",
        "- Online attention 的 score 已由 GenZ 建模为片上数据，因此没有重复扣除完整 score matrix 的 HBM 流量。",
        "- 跨帧复用是近似计算，需要真实特征误差、动作成功率和回退率实验才能证明精度安全。",
    ]
    (output / "REPORT_CN.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--bits", default="bf16")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tiles", type=int, default=8)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--similarity", type=float, default=0.92)
    parser.add_argument("--reuse-threshold", type=float, default=0.85)
    parser.add_argument("--max-reuse", type=float, default=0.80)
    parser.add_argument("--layer-decay", type=float, default=0.97)
    parser.add_argument("--vlm-propagation", type=float, default=0.25)
    parser.add_argument("--visual-prefix-fraction", type=float, default=768 / 800)
    parser.add_argument("--validation-overhead-ms", type=float, default=0.02)
    args = parser.parse_args()
    if args.tiles < 1 or args.frames < 1:
        raise ValueError("tiles and frames must be positive")
    args.output.mkdir(parents=True, exist_ok=True)

    operators = load_profile(args.profile_dir)
    assert_valid_operators(operators)
    system = system_for(args.gpu, args.bits)
    ablation, stages = run_scenarios(operators, system, args)
    _, fusion_details = apply_fusion(operators, system)
    _, pipeline_details = simulate_pipeline(operators, args.tiles)
    _, reuse_details = apply_reuse(
        operators, system, similarity=args.similarity, threshold=args.reuse_threshold,
        maximum=args.max_reuse, layer_decay=args.layer_decay,
        vlm_propagation=args.vlm_propagation,
        visual_prefix_fraction=args.visual_prefix_fraction,
    )
    sweep = reuse_sweep(operators, system, args)

    ablation.to_csv(args.output / "ablation_summary.csv", index=False)
    stages.to_csv(args.output / "stage_ablation.csv", index=False)
    pipeline_details.to_csv(args.output / "pipeline_segments.csv", index=False)
    fusion_details.to_csv(args.output / "fusion_candidates.csv", index=False)
    reuse_details.to_csv(args.output / "reuse_operator_details.csv", index=False)
    sweep.to_csv(args.output / "reuse_similarity_sweep.csv", index=False)
    (args.output / "settings.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    plot_results(ablation, stages, sweep, args.output)
    write_report(args, ablation, stages, pipeline_details, fusion_details,
                 reuse_details, sweep, args.output)
    print(ablation.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
