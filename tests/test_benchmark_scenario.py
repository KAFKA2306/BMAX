import json
import unittest
from pathlib import Path

from src.benchmark_contract import validate_dataset
from src.benchmark_repository import load_benchmark_repository
from src.benchmark_scenario import build_scenario


class BenchmarkScenarioTests(unittest.TestCase):
    def test_repository_loads_three_reproducible_scenarios(self):
        root = Path(__file__).parents[1]
        dataset = load_benchmark_repository(root / "data" / "convertible_benchmark.json")
        report = validate_dataset(dataset)
        self.assertEqual(report["issue_count"], 10)
        self.assertEqual(report["market_snapshot_count"], 3)
        self.assertEqual(report["scenario_count"], 3)
        self.assertEqual(report["scenario_issue_count"], 3)
        self.assertTrue(report["commercial_demo_ready"])

    def test_all_committed_scenario_byte_values_match_engine_regeneration(self):
        root = Path(__file__).parents[1]
        dataset = load_benchmark_repository(root / "data" / "convertible_benchmark.json")
        issues = {item["id"]: item for item in dataset["issues"]}
        snapshots = {item["id"]: item for item in dataset["market_snapshots"]}
        scenario_paths = sorted((root / "data" / "scenarios").glob("*.json"))
        self.assertEqual(len(scenario_paths), 3)
        for scenario_path in scenario_paths:
            with self.subTest(path=scenario_path.name):
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
        snapshots = {
            path.stem: json.loads(path.read_text(encoding="utf-8"))
            for path in (root / "data" / "market_snapshots").glob("*.json")
        }
        for scenario_path in sorted((root / "data" / "scenarios").glob("*.json")):
            with self.subTest(path=scenario_path.name):
                scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
                snapshot = snapshots[scenario["snapshot_id"]]
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
