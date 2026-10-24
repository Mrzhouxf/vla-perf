"""Launch- and traffic-aware flow-matching denoise-boundary fusion model."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .common import transfer_ms


@dataclass(frozen=True)
class BoundarySpec:
    action_tokens: int = 50
    action_dof: int = 14
    expert_width: int = 1024
    denoise_steps: int = 10
    action_bytes: int = 4
    weight_bytes: int = 2
    embedding_bytes: int = 2


def _projection(direction: str, spec: BoundarySpec) -> tuple[float, float]:
    tokens, dof, width = spec.action_tokens, spec.action_dof, spec.expert_width
    flops = 2.0 * tokens * dof * width
    weight_mb = dof * width * spec.weight_bytes / 1e6
    action_mb = tokens * dof * spec.action_bytes / 1e6
    embedding_mb = tokens * width * spec.embedding_bytes / 1e6
    traffic_mb = weight_mb + action_mb + embedding_mb
    if direction not in {"input", "output"}:
        raise ValueError(direction)
    return flops, traffic_mb


def _euler(spec: BoundarySpec) -> tuple[float, float]:
    elements = spec.action_tokens * spec.action_dof
    return 3.0 * elements, 3.0 * elements * spec.action_bytes / 1e6


def roofline_ms(flops: float, traffic_mb: float, system) -> tuple[float, float, float]:
    compute = flops / system.op_per_sec * 1e3 / system.compute_efficiency
    memory = transfer_ms(traffic_mb, system.offchip_mem_bw) / system.memory_efficiency
    return max(compute, memory), compute, memory


def model_boundary(system, launch_us: float, spec: BoundarySpec = BoundarySpec()) -> tuple[dict, pd.DataFrame]:
    if launch_us < 0:
        raise ValueError("launch_us must be non-negative")
    launch_ms = launch_us / 1000.0
    in_flops, in_mb = _projection("input", spec)
    out_flops, out_mb = _projection("output", spec)
    euler_flops, euler_mb = _euler(spec)
    in_ms, _, _ = roofline_ms(in_flops, in_mb, system)
    out_ms, _, _ = roofline_ms(out_flops, out_mb, system)
    euler_ms, _, _ = roofline_ms(euler_flops, euler_mb, system)

    baseline_kernel = in_ms + out_ms + euler_ms + 3 * launch_ms
    baseline = spec.denoise_steps * baseline_kernel

    # Transition: output projection + Euler + next input projection.  Velocity
    # and updated action remain internal; the final next-step embedding is kept.
    action_mb = spec.action_tokens * spec.action_dof * spec.action_bytes / 1e6
    embed_mb = spec.action_tokens * spec.expert_width * spec.embedding_bytes / 1e6
    weight_mb = 2 * spec.action_dof * spec.expert_width * spec.weight_bytes / 1e6
    transition_mb = embed_mb + action_mb + weight_mb + embed_mb
    transition_flops = out_flops + euler_flops + in_flops
    transition_ms, _, _ = roofline_ms(transition_flops, transition_mb, system)

    # Final step has no next input projection.
    final_mb = embed_mb + action_mb + spec.action_dof * spec.expert_width * spec.weight_bytes / 1e6
    final_flops = out_flops + euler_flops
    final_ms, _, _ = roofline_ms(final_flops, final_mb, system)
    fused = (
        in_ms + launch_ms
        + (spec.denoise_steps - 1) * (transition_ms + launch_ms)
        + final_ms + launch_ms
    )
    baseline_kernels = 3 * spec.denoise_steps
    fused_kernels = spec.denoise_steps + 1
    baseline_traffic = spec.denoise_steps * (in_mb + out_mb + euler_mb)
    fused_traffic = in_mb + (spec.denoise_steps - 1) * transition_mb + final_mb
    summary = {
        "baseline_boundary_ms": baseline,
        "fused_boundary_ms": fused,
        "boundary_saved_ms": baseline - fused,
        "boundary_speedup": baseline / fused,
        "baseline_kernels": baseline_kernels,
        "fused_kernels": fused_kernels,
        "launches_removed": baseline_kernels - fused_kernels,
        "baseline_traffic_mb": baseline_traffic,
        "fused_traffic_mb": fused_traffic,
        "traffic_saved_mb": baseline_traffic - fused_traffic,
        "flops_conserved": True,
    }
    details = pd.DataFrame([
        {"chain": "standalone input projection", "occurrences": 1,
         "roofline_ms_each": in_ms, "launch_ms_each": launch_ms},
        {"chain": "fused output + Euler + next input", "occurrences": spec.denoise_steps - 1,
         "roofline_ms_each": transition_ms, "launch_ms_each": launch_ms},
        {"chain": "fused final output + Euler", "occurrences": 1,
         "roofline_ms_each": final_ms, "launch_ms_each": launch_ms},
    ])
    return summary, details
