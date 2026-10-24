#!/usr/bin/env python3
"""Fine-grained π0 latency profiler and conservative overlap-pipeline prototype.

This program deliberately lives outside the baseline implementation.  It calls
GenZ exactly as the existing π0 scripts do and exports every GenZ operator-level
counter.  It deliberately does not invent an additional DMA/compute-overlap
schedule: GenZ's memory counter combines reads and output writes, while GenZ
already uses roofline latency (the maximum of compute, memory and communication).
A further overlap estimate needs a concrete runtime, read/write split, buffer
capacity and shared-HBM arbitration model, none of which is in this repository.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GENZ_ROOT = ROOT / "genz"
if str(GENZ_ROOT) not in sys.path:
    sys.path.insert(0, str(GENZ_ROOT))

from GenZ import parallel_decode_modeling, prefill_moddeling  # noqa: E402

TIME = "Latency (msec)"
COMPUTE = "Compute time (msec)"
MEMORY = "Memory time (msec)"
COMM = "Communication time (msec)"


@dataclass(frozen=True)
class StageSpec:
    name: str
    model: str
    kind: str
    multiplier: int
    run: Callable[[str, str], dict]


def stage_specs(vision_tokens: int, vlm_tokens: int, action_tokens: int, denoise_steps: int) -> list[StageSpec]:
    return [
        StageSpec(
            "vision_encoder", "vla/pi0-vision", "prefill", 1,
            lambda gpu, bits: prefill_moddeling(
                model="vla/pi0-vision", batch_size=3, input_tokens=vision_tokens,
                system_name=gpu, bits=bits, tensor_parallel=1, pipeline_parallel=1, debug=False),
        ),
        StageSpec(
            "vlm_prefill", "vla/pi0-vlm", "prefill", 1,
            lambda gpu, bits: prefill_moddeling(
                model="vla/pi0-vlm", batch_size=1, input_tokens=vlm_tokens,
                system_name=gpu, bits=bits, tensor_parallel=1, pipeline_parallel=1, debug=False),
        ),
        StageSpec(
            "action_expert", "vla/pi0-action-expert", "parallel_decode", denoise_steps,
            lambda gpu, bits: parallel_decode_modeling(
                model="vla/pi0-action-expert", batch_size=1, input_tokens=vlm_tokens,
                output_tokens_parallel=action_tokens, self_attention=True,
                system_name=gpu, bits=bits, tensor_parallel=1, pipeline_parallel=1, debug=False),
        ),
    ]


def expand_repeats(df: pd.DataFrame) -> pd.DataFrame:
    """Materialize GenZ Repeat blocks so execution order is retained."""
    rows = []

    def visit(start: int, stop: int, provenance: str) -> None:
        index = start
        while index < stop:
            source = df.iloc[index]
            if str(source["Op Type"]) == "Repeat":
                depth, end = 1, index + 1
                while depth:
                    nested_type = str(df.iloc[end]["Op Type"])
                    depth += nested_type == "Repeat"
                    depth -= nested_type == "EndRepeat"
                    end += 1
                for repeat_index in range(int(source["Dimension"])):
                    visit(index + 1, end - 1, f"{provenance}repeat{repeat_index}/")
                index = end
                continue
            if str(source["Op Type"]) != "EndRepeat":
                row = source.copy()
                row["repeat_instance"] = provenance.rstrip("/") or "root"
                rows.append(row)
            index += 1

    visit(0, len(df), "")
    return pd.DataFrame(rows)


def dependency_safe_schedule(tasks: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """Reference schedule that preserves GenZ critical paths and operator order."""
    current = 0.0
    rows = []
    for _, task in tasks.iterrows():
        start, end = current, current + float(task[TIME])
        current = end
        rows.append({
            "stage": task["stage"], "operator": task["Layer Name"], "op_type": task["Op Type"],
            "start_ms": start, "end_ms": end,
            "compute_work_ms": float(task[COMPUTE]), "memory_work_ms": float(task[MEMORY]),
            "communication_work_ms": float(task[COMM]),
        })
    return current, pd.DataFrame(rows)


def write_report(output: Path, stages: pd.DataFrame, latency_types: pd.DataFrame,
                 pipeline: pd.DataFrame, top_ops: pd.DataFrame, args: argparse.Namespace) -> None:
    total = float(stages["baseline_latency_ms"].sum())
    bottleneck = stages.loc[stages["baseline_latency_ms"].idxmax()]
    lines = [
        "# π0 GPU 延迟预实验报告",
        "",
        f"- 配置：{args.gpu} / {args.bits} / 3 cameras / vision={args.vision_tokens} tokens / VLM={args.vlm_tokens} tokens / action={args.action_tokens} tokens / {args.denoise_steps} denoising steps。",
        f"- 基线端到端延迟：**{total:.3f} ms**；最大阶段：**{bottleneck['stage']} ({bottleneck['baseline_latency_ms']:.3f} ms, {bottleneck['baseline_share_pct']:.1f}%)**。",
        "- `operator_breakdown.csv` 是逐算子、已展开 Transformer repeat 次数的原始细分；`category_breakdown.csv` 给出计算、访存、通信及有效关键路径占比。",
        "",
        "## 阶段延迟",
        "",
        stages.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 资源类型工作量占比",
        "",
        latency_types.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 流水线实验",
        "",
        pipeline.to_markdown(index=False, floatfmt=".4f"),
        "",
        "当前不报告额外的访存/计算重叠收益：GenZ 的 Memory time 同时包含 activation、weight 和 output 写回，而单算子 Latency 已是 Roofline 临界路径。SigLIP 使用全 token self-attention，默认不允许视觉 token 在 encoder 完成前解锁 VLM。",
        "",
        "## 最慢算子（按有效关键路径延迟）",
        "",
        top_ops.to_markdown(index=False, floatfmt=".5f"),
        "",
        "## 解释口径",
        "",
        "GenZ 的 `Latency` 是单算子 Roofline 临界路径（不是 Compute + Memory 的简单相加）；因此计算/访存/通信列是资源工作量和瓶颈诊断，三者百分比不会强制相加为 100%。Action Expert 的单次结果已乘以去噪步数。",
    ]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def plot_stage_breakdown(stages: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = range(len(stages))
    ax.bar(x, stages["compute_work_ms"], label="compute work")
    ax.bar(x, stages["memory_work_ms"], bottom=stages["compute_work_ms"], label="memory work")
    ax.bar(x, stages["communication_work_ms"], bottom=stages["compute_work_ms"] + stages["memory_work_ms"], label="communication work")
    ax.plot(x, stages["baseline_latency_ms"], color="black", marker="o", label="effective critical-path latency")
    ax.set_xticks(list(x), stages["stage"], rotation=15, ha="right")
    ax.set_ylabel("ms")
    ax.set_title("π0 stage resource work and effective latency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "stage_latency_breakdown.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="H100")
    parser.add_argument("--bits", default="bf16")
    parser.add_argument("--vision-tokens", type=int, default=256)
    parser.add_argument("--vlm-tokens", type=int, default=800)
    parser.add_argument("--action-tokens", type=int, default=50)
    parser.add_argument("--denoise-steps", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    operator_frames, stage_rows, schedule_frames = [], [], []
    for spec in stage_specs(args.vision_tokens, args.vlm_tokens, args.action_tokens, args.denoise_steps):
        result = spec.run(args.gpu, args.bits)
        ops = expand_repeats(result["model_df"])
        # Denoising is a true sequential repetition of the whole DiT pass.
        ops = pd.concat([ops.assign(stage_invocation=index + 1) for index in range(spec.multiplier)], ignore_index=True)
        ops["stage"] = spec.name
        effective = float(ops[TIME].sum())
        dependency_safe, schedule = dependency_safe_schedule(ops)
        schedule_frames.append(schedule)
        stage_rows.append({
            "stage": spec.name, "model": spec.model, "invocations": spec.multiplier,
            "baseline_latency_ms": effective, "compute_work_ms": float(ops[COMPUTE].sum()),
            "memory_work_ms": float(ops[MEMORY].sum()), "communication_work_ms": float(ops[COMM].sum()),
            "dependency_safe_latency_ms": dependency_safe, "total_ops_mflop": float(ops["Num ops (MFLOP)"].sum()),
            "total_data_mb": float(ops["Total Data (MB)"].sum()),
        })
        operator_frames.append(ops)

    operators = pd.concat(operator_frames, ignore_index=True)
    stages = pd.DataFrame(stage_rows)
    stages["baseline_share_pct"] = 100 * stages["baseline_latency_ms"] / stages["baseline_latency_ms"].sum()
    stages["schedule_consistency_error_ms"] = stages["dependency_safe_latency_ms"] - stages["baseline_latency_ms"]
    for column, share_column in (("compute_work_ms", "compute_work_share_pct"),
                                 ("memory_work_ms", "memory_work_share_pct"),
                                 ("communication_work_ms", "communication_work_share_pct")):
        total_work = float(stages[column].sum())
        stages[share_column] = 100 * stages[column] / total_work if total_work else 0.0
    operators["stage_effective_share_pct"] = 100 * operators[TIME] / operators.groupby("stage")[TIME].transform("sum")
    categories = (operators.groupby(["stage", "Bound"], dropna=False)[[TIME, COMPUTE, MEMORY, COMM, "Num ops (MFLOP)", "Total Data (MB)"]]
                  .sum().reset_index())
    categories["stage_effective_share_pct"] = 100 * categories[TIME] / categories.groupby("stage")[TIME].transform("sum")
    latency_types = stages.melt(
        id_vars="stage", value_vars=["baseline_latency_ms", "compute_work_ms", "memory_work_ms", "communication_work_ms"],
        var_name="latency_type", value_name="work_ms")
    type_totals = latency_types.groupby("latency_type")["work_ms"].transform("sum")
    latency_types["share_within_latency_type_pct"] = (100 * latency_types["work_ms"] / type_totals).fillna(0.0)

    baseline_total = float(stages["baseline_latency_ms"].sum())
    pipeline = pd.DataFrame([
        {"scenario": "baseline_serial", "end_to_end_latency_ms": baseline_total,
         "speedup_vs_baseline": 1.0, "assumption": "GenZ per-operator critical paths; stages serial"},
        {"scenario": "dependency_safe_current_model", "end_to_end_latency_ms": baseline_total,
         "speedup_vs_baseline": 1.0, "assumption": "no legal cross-stage overlap in stock SigLIP→VLM graph"},
    ])
    top_ops = (operators.groupby(["stage", "Layer Name", "Op Type", "Bound"], dropna=False)[[
        TIME, COMPUTE, MEMORY, COMM, "Total Data (MB)", "Num ops (MFLOP)"]].sum().reset_index())
    top_ops["stage_effective_share_pct"] = 100 * top_ops[TIME] / top_ops.groupby("stage")[TIME].transform("sum")
    top_ops = top_ops.sort_values(TIME, ascending=False).head(30)

    operators.to_csv(args.output / "operator_breakdown.csv", index=False)
    categories.to_csv(args.output / "category_breakdown.csv", index=False)
    latency_types.to_csv(args.output / "latency_type_summary.csv", index=False)
    stages.to_csv(args.output / "stage_breakdown.csv", index=False)
    pipeline.to_csv(args.output / "pipeline_summary.csv", index=False)
    pd.concat(schedule_frames, ignore_index=True).to_csv(args.output / "dependency_safe_operator_schedule.csv", index=False)
    top_ops.to_csv(args.output / "top_operators.csv", index=False)
    plot_stage_breakdown(stages, args.output)
    write_report(args.output, stages, latency_types, pipeline, top_ops, args)
    print(f"Wrote detailed profile and pipeline experiment to {args.output}")


if __name__ == "__main__":
    main()
