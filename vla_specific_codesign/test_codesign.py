from __future__ import annotations

import unittest
from pathlib import Path

from .common import FLOPS, load_validated_profile, system_for
from .denoise_boundary import BoundarySpec, model_boundary
from .kv_streaming import apply_first_pass_kv_streaming
from .run_experiment import apply_action_core_fusion, run_ablation
from .wavefront import layer_table, schedule_streaming_wavefront, schedule_wavefront

ROOT = Path(__file__).resolve().parents[1]


class CodesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ops = load_validated_profile(ROOT / "pipeline_prototype" / "results")
        cls.system = system_for("H100", "bf16")

    def test_wavefront_layer_count_dependencies_and_bounds(self):
        prefix = layer_table(self.ops, "vlm_prefill")
        action = layer_table(self.ops, "action_expert", invocation=1)
        self.assertEqual(len(prefix), 18)
        self.assertEqual(len(action), 18)
        serial = float(prefix["Latency (msec)"].sum() + action["Latency (msec)"].sum())
        no_overlap, schedule = schedule_wavefront(prefix, action, 0.0)
        ideal, _ = schedule_wavefront(prefix, action, 1.0)
        self.assertAlmostEqual(no_overlap, serial)
        self.assertLessEqual(ideal, no_overlap)
        self.assertEqual(len(schedule), 19)

    def test_streaming_keeps_later_iterations_unchanged(self):
        streamed, details = apply_first_pass_kv_streaming(self.ops, self.system)
        self.assertAlmostEqual(float(details.prefix_kv_mb.sum()), 14.0625)
        later = self.ops.stage_invocation > 1
        self.assertTrue((streamed.loc[later, "Latency (msec)"] == self.ops.loc[later, "Latency (msec)"]).all())
        self.assertTrue(details.cache_write_retained.all())
        self.assertTrue(details.later_denoise_hbm_reads_retained.all())
        action = layer_table(streamed, "action_expert", invocation=1)
        latency, schedule, summary = schedule_streaming_wavefront(streamed, action, 0.8)
        self.assertGreater((schedule.prefix_end_ms - schedule.prefix_kv_ready_ms).min(), 0)
        self.assertGreater(summary["latency_saved_ms"], 0)
        self.assertEqual(latency, summary["scheduled_streaming_wavefront_ms"])

    def test_boundary_conserves_flops_and_removes_expected_launches(self):
        summary, _ = model_boundary(self.system, 3.0, BoundarySpec())
        self.assertTrue(summary["flops_conserved"])
        self.assertEqual(summary["baseline_kernels"], 30)
        self.assertEqual(summary["fused_kernels"], 11)
        self.assertGreater(summary["boundary_saved_ms"], 0)

    def test_action_fusion_conserves_nominal_flops(self):
        fused, _ = apply_action_core_fusion(self.ops, self.system)
        self.assertAlmostEqual(float(fused[FLOPS].sum()), float(self.ops[FLOPS].sum()), places=5)

    def test_full_codesign_is_not_factorized_speedup(self):
        table, _ = run_ablation(self.ops, self.system, 3.0, 0.8, BoundarySpec())
        values = table.set_index("scenario").end_to_end_latency_ms
        self.assertLess(values["full_operator_pipeline_codesign"], values["extended_baseline"])
        self.assertLessEqual(values["full_operator_pipeline_codesign"], values["vla_action_kernel_suite"])
        self.assertTrue(table.model_equivalent.all())


if __name__ == "__main__":
    unittest.main()
