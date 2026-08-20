import json
import unittest
from pathlib import Path

from src.benchmark_contract import validate_dataset
from src.benchmark_repository import load_benchmark_repository
from src.benchmark_scenario import build_scenario


class BenchmarkScenarioTests(unittest.TestCase):
    def test_repository_loads_first_reproducible_scenario(self):
        root = Path(__file__).parents[1]
        dataset = load_benchmark_repository(root / "data" / "convertible_benchmark.json")
        report = validate_dataset(dataset)
        self.assertEqual(report["issue_count"], 10)
        self.assertEqual(report["market_snapshot_count"], 1)
        self.assertEqual(report["scenario_count"], 1)
        self.assertEqual(report["scenario_issue_count"], 1)
        self.assertFalse(report["commercial_demo_ready"])

    def test_committed_scenario_byte_values_match_engine_regeneration(self):
        root = Path(__file__).parents[1]
        dataset = load_benchmark_repository(root / "data" / "convertible_benchmark.json")
        issues = {item["id"]: item for item in dataset["issues"]}
        snapshots = {item["id"]: item for item in dataset["market_snapshots"]}
        scenario_path = root / "data" / "scenarios" / "mstr-2028-0625__2024-09-17__base-assumption.json"
        committed = json.loads(scenario_path.read_text(encoding="utf-8"))
        regenerated = build_scenario(
            issues[committed["issue_id"]],
            snapshots[committed["snapshot_id"]],
            committed["assumptions"],
            scenario_id=committed["id"],
        )
        self.assertEqual(regenerated, committed)

    def test_unobserved_credit_spread_and_volatility_remain_assumptions(self):
        root = Path(__file__).parents[1]
        scenario = json.loads(
            (root / "data" / "scenarios" / "mstr-2028-0625__2024-09-17__base-assumption.json").read_text(encoding="utf-8")
        )
        snapshot = json.loads(
            (root / "data" / "market_snapshots" / "mstr-2028-0625__2024-09-17.json").read_text(encoding="utf-8")
        )
        self.assertIsNone(snapshot["credit_spread_if_observed"])
        self.assertIsNone(snapshot["model_inputs"])
        self.assertEqual(scenario["assumptions"]["model"]["credit_spread"], 0.02)
        self.assertEqual(scenario["assumptions"]["model"]["equity_volatility"], 0.6)
        self.assertTrue(scenario["claim_boundary"]["scenario_not_observed_market_value"])
        self.assertFalse(scenario["claim_boundary"]["calibrated_to_market"])
        self.assertFalse(scenario["claim_boundary"]["fair_value_claim"])
        self.assertFalse(scenario["claim_boundary"]["recommendation"])


if __name__ == "__main__":
    unittest.main()
