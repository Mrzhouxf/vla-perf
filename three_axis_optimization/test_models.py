from __future__ import annotations

import unittest
from argparse import Namespace
from pathlib import Path

from .common import COMPUTE, FLOPS, MEMORY, TIME, load_profile, system_for
from .fusion import apply_fusion
from .pipeline import simulate_pipeline
from .reuse import apply_reuse
from .run_ablation import run_scenarios

ROOT = Path(__file__).resolve().parents[1]


class OptimizationModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ops = load_profile(ROOT / "pipeline_prototype" / "results")
        cls.system = system_for("H100", "bf16")

    def test_pipeline_one_tile_is_baseline_and_respects_lower_bound(self):
        baseline = self.ops.groupby("stage")[TIME].sum().sort_index()
        one_tile, _ = simulate_pipeline(self.ops, tiles=1)
        self.assertTrue((abs(one_tile.sort_index() - baseline) < 1e-10).all())
        _, details = simulate_pipeline(self.ops, tiles=8)
        self.assertTrue((details.pipeline_latency_ms + 1e-12 >= details.resource_lower_bound_ms).all())
        self.assertTrue((details.pipeline_latency_ms <= details.serial_latency_ms + 1e-12).all())

    def test_fusion_conserves_flops_and_is_non_overlapping(self):
        fused, details = apply_fusion(self.ops, self.system)
        self.assertAlmostEqual(float(fused[FLOPS].sum()), float(self.ops[FLOPS].sum()), places=5)
        source = [token for value in fused.source_indices for token in str(value).split(";")]
        self.assertEqual(len(source), len(set(source)))
        self.assertTrue((details.fused_latency_ms <= details.baseline_latency_ms + 1e-12).all())

    def test_reuse_gate_and_action_safety(self):
        below, below_details = apply_reuse(self.ops, self.system, similarity=0.80, threshold=0.85)
        self.assertTrue((abs(below[TIME] - self.ops[TIME]) < 1e-12).all())
        self.assertTrue((abs(below[MEMORY] - self.ops[MEMORY]) < 1e-12).all())
        self.assertTrue((below_details.reuse_ratio == 0).all())
        reused, active = apply_reuse(self.ops, self.system, similarity=0.92, threshold=0.85)
        self.assertTrue((active.loc[active.stage == "action_expert", "reuse_ratio"] == 0).all())
        self.assertGreater(active.loc[active.stage == "vision_encoder", "reuse_ratio"].max(), 0)
        self.assertLess(float(reused[FLOPS].sum()), float(self.ops[FLOPS].sum()))
        action = self.ops.stage == "action_expert"
        self.assertAlmostEqual(float(reused.loc[action, FLOPS].sum()), float(self.ops.loc[action, FLOPS].sum()))
        fallbacks = active[(active.requested_reuse_ratio > 0) & (~active.cache_profitable)]
        self.assertGreater(len(fallbacks), 0)
        self.assertTrue((fallbacks.reuse_ratio == 0).all())

    def test_all_three_is_composed_on_transformed_counters(self):
        args = Namespace(
            similarity=0.92, reuse_threshold=0.85, max_reuse=0.80,
            layer_decay=0.97, vlm_propagation=0.25,
            visual_prefix_fraction=768 / 800, validation_overhead_ms=0.02,
            tiles=8, frames=8,
        )
        ablation, _ = run_scenarios(self.ops, self.system, args)
        values = ablation.set_index("scenario").average_latency_ms
        self.assertLessEqual(values["all_three"], values["pipeline_fusion"] + 1e-12)
        self.assertLessEqual(values["all_three"], values["fusion_reuse"] + 1e-12)
        self.assertGreater(values["all_three"], 0)


if __name__ == "__main__":
    unittest.main()
