import sys
import unittest
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "genz"))

from analyze_operator_fusion import (  # noqa: E402
    COMPUTE, MEMORY, COMM, TIME, INPUT_A, INPUT_W, OUTPUT,
    candidate_record, find_candidates, scenario_summary,
)
from GenZ.LLM_inference.utils import get_inference_system  # noqa: E402


class FusionModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.system = get_inference_system(system_name="H100", bits="bf16")

    def test_fused_chain_never_drops_compute(self):
        rows = [
            pd.Series({"Layer Name": "up+gate", TIME: 2.0, COMPUTE: 1.5, MEMORY: 2.0,
                       COMM: 0.0, INPUT_A: 1.0, INPUT_W: 8.0, OUTPUT: 4.0}),
            pd.Series({"Layer Name": "down", TIME: 1.0, COMPUTE: 0.8, MEMORY: 1.0,
                       COMM: 0.0, INPUT_A: 2.0, INPUT_W: 4.0, OUTPUT: 1.0}),
        ]
        record = candidate_record(0, "action_expert", 1, "repeat0", "mlp_activation_down",
                                  [0, 1], rows, 6.0, 0.0, "test", self.system)
        self.assertGreaterEqual(record["fused_latency_ms"], 2.3)
        self.assertLessEqual(record["fused_latency_ms"], record["baseline_latency_ms"])

    def test_combined_scenario_does_not_double_count_online_attention(self):
        profile = ROOT / "pipeline_prototype" / "results" / "operator_breakdown.csv"
        ops = pd.read_csv(profile)
        candidates = find_candidates(ops, self.system)
        scenarios = scenario_summary(ops, candidates)
        combined = scenarios[(scenarios.scope == "end_to_end") &
                             (scenarios.scenario == "combined_qkv_attention_mlp")].iloc[0]
        selected = candidates[candidates.fusion_type.isin({"qkv_online_attention", "mlp_activation_down"})]
        self.assertAlmostEqual(combined.latency_saved_ms, selected.latency_saved_ms.sum(), places=12)

    def test_expected_action_candidate_counts(self):
        profile = ROOT / "pipeline_prototype" / "results" / "operator_breakdown.csv"
        candidates = find_candidates(pd.read_csv(profile), self.system)
        action = candidates[candidates.stage == "action_expert"]
        self.assertEqual((action.fusion_type == "qkv_online_attention").sum(), 360)
        self.assertEqual((action.fusion_type == "mlp_activation_down").sum(), 180)


if __name__ == "__main__":
    unittest.main()
