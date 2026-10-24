#!/usr/bin/env python3
"""Run the exact VLA-specific operator/pipeline co-design ablation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from three_axis_optimization.fusion import apply_fusion

from .common import TIME, load_validated_profile, stage_totals, system_for
from .denoise_boundary import BoundarySpec, model_boundary
from .kv_streaming import apply_first_pass_kv_streaming
from .wavefront import (
    layer_table, schedule_streaming_wavefront, schedule_wavefront, wavefront_bounds,
)


def apply_action_core_fusion(operators: pd.DataFrame, system) -> tuple[pd.DataFrame, pd.DataFrame]:
    action = operators[operators.stage == "action_expert"].copy()
    non_action = operators[operators.stage != "action_expert"].copy()
    fused_action, details = apply_fusion(action, system)
    return pd.concat([non_action, fused_action], ignore_index=True), details


def wavefront_region(operators: pd.DataFrame, overlap_efficiency: float):
    prefix = layer_table(operators, "vlm_prefill")
    action = layer_table(operators, "action_expert", invocation=1)
    latency, schedule = schedule_wavefront(prefix, action, overlap_efficiency)
    return latency, schedule, wavefront_bounds(prefix, action)


def core_parts(operators: pd.DataFrame) -> dict:
    stages = stage_totals(operators)
    first_action = float(operators[(operators.stage == "action_expert") &
                                   (operators.stage_invocation == 1)][TIME].sum())
    return {
        "vision": float(stages.vision_encoder),
        "vlm": float(stages.vlm_prefill),
        "action_all": float(stages.action_expert),
        "action_first": first_action,
        "action_later": float(stages.action_expert - first_action),
    }


def run_ablation(operators: pd.DataFrame, system, launch_us: float,
                 overlap_efficiency: float, boundary_spec: BoundarySpec):
    original = core_parts(operators)
    kv_ops, kv_details = apply_first_pass_kv_streaming(operators, system)
    kv = core_parts(kv_ops)
    fused_ops, fusion_details = apply_action_core_fusion(operators, system)
    fused = core_parts(fused_ops)
    kv_fused_ops, _ = apply_action_core_fusion(kv_ops, system)
    kv_fused = core_parts(kv_fused_ops)

    wave, wave_schedule, bounds = wavefront_region(operators, overlap_efficiency)
    kv_action_layers = layer_table(kv_ops, "action_expert", invocation=1)
    wave_kv, wave_kv_schedule, streaming_bounds = schedule_streaming_wavefront(
        kv_ops, kv_action_layers, overlap_efficiency
    )
    fused_action_layers = layer_table(kv_fused_ops, "action_expert", invocation=1)
    wave_all, wave_all_schedule, all_bounds = schedule_streaming_wavefront(
        kv_fused_ops, fused_action_layers, overlap_efficiency
    )
    boundary, boundary_details = model_boundary(system, launch_us, boundary_spec)

    baseline_core = original["vision"] + original["vlm"] + original["action_all"]
    baseline = baseline_core + boundary["baseline_boundary_ms"]
    rows = []

    def add(name, latency, exact, description):
        rows.append({
            "scenario": name,
            "end_to_end_latency_ms": latency,
            "latency_saved_ms": baseline - latency,
            "speedup": baseline / latency,
            "latency_reduction_pct": 100 * (baseline - latency) / baseline,
            "model_equivalent": exact,
            "description": description,
        })

    add("extended_baseline", baseline, True,
        "GenZ core plus explicit action input/output projections and Euler updates")
    add("prefix_action_wavefront", original["vision"] + wave + original["action_later"]
        + boundary["baseline_boundary_ms"], True,
        "dependency-exact P_l || A_l-1 scheduling; no stale KV")
    add("streaming_kv_only", kv["vision"] + kv["vlm"] + kv["action_all"]
        + boundary["baseline_boundary_ms"], True,
        "first-pass prefix K/V consumed on-chip while cache write is retained")
    add("wavefront_streaming_kv", kv["vision"] + wave_kv + kv["action_later"]
        + boundary["baseline_boundary_ms"], True,
        "exact wavefront with direct prefix-KV producer-consumer edge")
    add("action_core_fusion", fused["vision"] + fused["vlm"] + fused["action_all"]
        + boundary["baseline_boundary_ms"], True,
        "supporting QKV-online-attention and gated-MLP traffic fusion in AE")
    add("denoise_boundary_fusion", baseline_core + boundary["fused_boundary_ms"], True,
        "action OutProj + Euler + next-step Action InProj")
    add("vla_action_kernel_suite", fused["vision"] + fused["vlm"] + fused["action_all"]
        + boundary["fused_boundary_ms"], True,
        "AE core traffic fusion plus flow-matching denoise-boundary fusion")
    all_latency = kv_fused["vision"] + wave_all + kv_fused["action_later"] + boundary["fused_boundary_ms"]
    add("full_operator_pipeline_codesign", all_latency, True,
        "streaming KV producer-consumer + exact wavefront + repeated AE fusion")

    artifacts = {
        "kv_details": kv_details,
        "fusion_details": fusion_details,
        "wave_schedule": wave_schedule,
        "wave_kv_schedule": wave_kv_schedule,
        "wave_all_schedule": wave_all_schedule,
        "boundary_details": boundary_details,
        "boundary_summary": boundary,
        "wavefront_bounds": bounds,
        "all_wavefront_bounds": all_bounds,
        "streaming_wavefront_bounds": streaming_bounds,
        "baseline_core_ms": baseline_core,
        "baseline_extended_ms": baseline,
    }
    return pd.DataFrame(rows), artifacts


def launch_sensitivity(operators, system, overlap_efficiency, spec):
    rows = []
    for launch_us in [0, 1, 2, 3, 5, 8, 10]:
        table, artifacts = run_ablation(operators, system, launch_us, overlap_efficiency, spec)
        baseline = table[table.scenario == "extended_baseline"].iloc[0]
        full = table[table.scenario == "full_operator_pipeline_codesign"].iloc[0]
        boundary = artifacts["boundary_summary"]
        rows.append({
            "kernel_launch_us": launch_us,
            "baseline_latency_ms": baseline.end_to_end_latency_ms,
            "codesign_latency_ms": full.end_to_end_latency_ms,
            "codesign_speedup": full.speedup,
            "codesign_reduction_pct": full.latency_reduction_pct,
            "boundary_fusion_saved_ms": boundary["boundary_saved_ms"],
        })
    return pd.DataFrame(rows)


def with_denoise_steps(operators: pd.DataFrame, steps: int) -> pd.DataFrame:
    if steps < 1:
        raise ValueError("steps must be positive")
    non_action = operators[operators.stage != "action_expert"].copy()
    template = operators[(operators.stage == "action_expert") &
                         (operators.stage_invocation == 1)].copy()
    copies = []
    for invocation in range(1, steps + 1):
        copy = template.copy()
        copy["stage_invocation"] = invocation
        copy["source_indices"] = copy.source_indices.astype(str) + f":denoise{invocation}"
        copies.append(copy)
    return pd.concat([non_action, *copies], ignore_index=True)


def denoise_step_sensitivity(operators, system, launch_us, overlap_efficiency, args):
    rows = []
    for steps in [4, 6, 8, 10, 16, 20]:
        scaled = with_denoise_steps(operators, steps)
        spec = BoundarySpec(args.action_tokens, args.action_dof, args.expert_width, steps)
        table, _ = run_ablation(scaled, system, launch_us, overlap_efficiency, spec)
        baseline = table[table.scenario == "extended_baseline"].iloc[0]
        full = table[table.scenario == "full_operator_pipeline_codesign"].iloc[0]
        rows.append({
            "denoise_steps": steps,
            "baseline_latency_ms": baseline.end_to_end_latency_ms,
            "codesign_latency_ms": full.end_to_end_latency_ms,
            "codesign_speedup": full.speedup,
            "codesign_reduction_pct": full.latency_reduction_pct,
        })
    return pd.DataFrame(rows)


def overlap_sensitivity(operators, system, launch_us, spec):
    rows = []
    for efficiency in [0.0, 0.25, 0.5, 0.75, 0.8, 1.0]:
        table, _ = run_ablation(operators, system, launch_us, efficiency, spec)
        wave = table[table.scenario == "wavefront_streaming_kv"].iloc[0]
        full = table[table.scenario == "full_operator_pipeline_codesign"].iloc[0]
        rows.append({
            "overlap_efficiency": efficiency,
            "wavefront_kv_saved_ms": wave.latency_saved_ms,
            "full_codesign_latency_ms": full.end_to_end_latency_ms,
            "full_codesign_speedup": full.speedup,
            "full_codesign_reduction_pct": full.latency_reduction_pct,
        })
    return pd.DataFrame(rows)


def plot_results(ablation, sensitivity, output):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(ablation.scenario, ablation.end_to_end_latency_ms)
    axes[0].tick_params(axis="x", rotation=40)
    axes[0].set_ylabel("latency (ms)")
    axes[0].set_title("Exact VLA-specific co-design ablation")
    axes[1].plot(sensitivity.kernel_launch_us, sensitivity.codesign_speedup, marker="o")
    axes[1].set_xlabel("assumed kernel launch overhead (us)")
    axes[1].set_ylabel("full co-design speedup")
    axes[1].set_title("Launch-overhead sensitivity")
    fig.tight_layout()
    fig.savefig(output / "codesign_ablation.png", dpi=180)
    plt.close(fig)


def write_report(args, ablation, artifacts, sensitivity, denoise_sensitivity,
                 overlap_sensitivity_table, output):
    full = ablation[ablation.scenario == "full_operator_pipeline_codesign"].iloc[0]
    wave = ablation[ablation.scenario == "prefix_action_wavefront"].iloc[0]
    kv = artifacts["kv_details"]
    boundary = artifacts["boundary_summary"]
    fusion = artifacts["fusion_details"]
    lines = [
        "# Exact VLA Operator–Pipeline Co-Design 实验报告",
        "",
        f"配置：{args.gpu}/{args.bits}，Prefix/Action 18层，{args.denoise_steps}次去噪；共运行假设 overlap efficiency={args.overlap_efficiency:.2f}，kernel launch={args.kernel_launch_us:.1f} us。",
        f"扩展基线为 **{full.end_to_end_latency_ms + full.latency_saved_ms:.4f} ms**；完整共设计为 **{full.end_to_end_latency_ms:.4f} ms**，模拟加速 **{full.speedup:.4f}×**。所有主方案保持当前帧模型计算等价，不使用stale KV，也不减少去噪步数。",
        "",
        "## 消融",
        "",
        ablation.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## 创新1：精确Prefix→Action层级Wavefront",
        "",
        "依赖为 `P_l <- P_l-1`、`A_l <- A_l-1` 和 `A_l <- KV(P_l)`，因此合法对角调度是 `P0; (P1||A0); ...; (P17||A16); A17`。该方案只重排第一次Action Expert，后续去噪保持串行。",
        f"默认资源感知调度单独节省 **{wave.latency_saved_ms:.6f} ms**。其理想资源上界最多隐藏 {artifacts['wavefront_bounds']['maximum_overlap_saved_ms']:.6f} ms。",
        "",
        "## 创新2：Prefix-KV Producer–Consumer Streaming",
        "",
        f"第一次Action Attention共直接消费 **{kv.prefix_kv_mb.sum():.4f} MB** Prefix K/V；KV cache写回仍保留，后续{args.denoise_steps-1}次读取也保留，因此没有把缓存访问虚构为全部消失。HBM→片上路径减少的资源时间为 **{kv.hbm_to_onchip_time_saved_ms.sum():.6f} ms**。更关键的是，A_l在P_l的QKV完成后即可释放，不再等待整个P_l结束；流式wavefront的资源下界为 {artifacts['streaming_wavefront_bounds']['ideal_streaming_wavefront_ms']:.6f} ms。",
        "",
        "## 创新3：Flow-matching Denoise Boundary Fusion",
        "",
        f"将 `Action OutProj + Euler + Next Action InProj` 合并。边界kernel由 {boundary['baseline_kernels']} 个降至 {boundary['fused_kernels']} 个，删除 {boundary['launches_removed']} 次启动；FLOPs保持不变，流量减少 {boundary['traffic_saved_mb']:.6f} MB，边界延迟节省 {boundary['boundary_saved_ms']:.6f} ms。",
        "",
        "## Supporting Action Expert Fusion",
        "",
        fusion.groupby("fusion_type")[["baseline_latency_ms", "fused_latency_ms", "latency_saved_ms", "saved_hbm_mb", "saved_onchip_mb"]].sum().reset_index().to_markdown(index=False, floatfmt=".6f"),
        "",
        "普通QKV/MLP融合仅作为支撑技术，不单独声明论文核心创新。",
        "",
        "## Kernel-launch敏感性",
        "",
        sensitivity.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## 去噪步数敏感性",
        "",
        denoise_sensitivity.to_markdown(index=False, floatfmt=".6f"),
        "",
        "Action fusion收益随重复次数增长；Wavefront只覆盖第一次AE，因此其绝对收益不会随去噪次数线性增长。",
        "",
        "## 共运行效率敏感性",
        "",
        overlap_sensitivity_table.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## 解释边界",
        "",
        "- 这是基于GenZ逐算子counter的扩展Roofline实验，不是CUDA/Triton实测。",
        "- overlap efficiency和kernel launch均显式做成参数；论文最终数字必须由Nsight实测校准。",
        "- 当前未为AdaRMSNorm/RoPE虚构额外收益；它们需要加入完整DAG并实现真实kernel后再报告。",
        "- Wavefront精确性依赖Prefix attention mask阻止Prefix读取Action suffix，实际runtime仍需数值等价测试。",
    ]
    (output / "REPORT_CN.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--bits", default="bf16")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overlap-efficiency", type=float, default=0.8)
    parser.add_argument("--kernel-launch-us", type=float, default=3.0)
    parser.add_argument("--denoise-steps", type=int, default=10)
    parser.add_argument("--action-tokens", type=int, default=50)
    parser.add_argument("--action-dof", type=int, default=14)
    parser.add_argument("--expert-width", type=int, default=1024)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    operators = load_validated_profile(args.profile_dir)
    system = system_for(args.gpu, args.bits)
    spec = BoundarySpec(args.action_tokens, args.action_dof, args.expert_width, args.denoise_steps)
    ablation, artifacts = run_ablation(
        operators, system, args.kernel_launch_us, args.overlap_efficiency, spec
    )
    sensitivity = launch_sensitivity(operators, system, args.overlap_efficiency, spec)
    denoise_sensitivity = denoise_step_sensitivity(
        operators, system, args.kernel_launch_us, args.overlap_efficiency, args
    )
    overlap_sensitivity_table = overlap_sensitivity(
        operators, system, args.kernel_launch_us, spec
    )

    ablation.to_csv(args.output / "codesign_ablation.csv", index=False)
    artifacts["wave_schedule"].to_csv(args.output / "wavefront_schedule.csv", index=False)
    artifacts["wave_kv_schedule"].to_csv(args.output / "wavefront_kv_schedule.csv", index=False)
    artifacts["wave_all_schedule"].to_csv(args.output / "full_codesign_schedule.csv", index=False)
    artifacts["kv_details"].to_csv(args.output / "streaming_kv_edges.csv", index=False)
    artifacts["fusion_details"].to_csv(args.output / "supporting_ae_fusions.csv", index=False)
    artifacts["boundary_details"].to_csv(args.output / "denoise_boundary_details.csv", index=False)
    sensitivity.to_csv(args.output / "kernel_launch_sensitivity.csv", index=False)
    denoise_sensitivity.to_csv(args.output / "denoise_step_sensitivity.csv", index=False)
    overlap_sensitivity_table.to_csv(args.output / "overlap_efficiency_sensitivity.csv", index=False)
    (args.output / "settings.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    plot_results(ablation, sensitivity, args.output)
    write_report(args, ablation, artifacts, sensitivity, denoise_sensitivity,
                 overlap_sensitivity_table, args.output)
    print(ablation.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
