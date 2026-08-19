import json
import unittest
from pathlib import Path

from src.benchmark_contract import BenchmarkContractError, validate_dataset


def valid_dataset():
    return {
        "schema_version": "bmax.convertible-benchmark.v1",
        "issuers": [{"id":"issuer-1","legal_name":"Example Corp","ticker":"EXM","cik":"0000000001","jurisdiction":"US-DE"}],
        "evidence": [{"id":"ev-1","accession_no":"0000000000-26-000001","filing_type":"8-K","filed_at":"2026-01-02","source_url":"https://www.sec.gov/Archives/example","section_ref":"Exhibit 4.1","observed_at":"2026-08-11"}],
        "issues": [{
            "id":"issue-1","issuer_id":"issuer-1","principal":100000000,"issue_date":"2026-01-02","maturity_date":"2031-01-02","coupon":0.01,"currency":"USD",
            "conversion_terms":{"conversion_price":125.0,"conversion_ratio":8.0,"reference_share":"EXM"},
            "protection_terms": {name:{"status":"ABSENT","evidence_id":"ev-1"} for name in ("call_terms","put_terms","redemption_terms","fundamental_change_terms")},
            "evidence_map": {field:"ev-1" for field in ("principal","issue_date","maturity_date","coupon","currency","conversion_terms.conversion_price","conversion_terms.conversion_ratio","conversion_terms.reference_share")},
        }],
        "market_snapshots": [], "scenarios": [],
    }


class BenchmarkContractTests(unittest.TestCase):
    def test_valid_contract_reports_coverage_without_claiming_demo_readiness(self):
        report = validate_dataset(valid_dataset())
        self.assertEqual(report["field_evidence_coverage"], 1.0)
        self.assertEqual(report["scenario_ready_count"], 1)
        self.assertFalse(report["commercial_demo_ready"])

    def test_repository_fixture_is_valid_but_not_commercial_demo_ready(self):
        dataset_path = Path(__file__).parents[1] / "data" / "convertible_benchmark.json"
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
        report = validate_dataset(dataset)
        self.assertEqual(report["issuer_count"], 4)
        self.assertEqual(report["issue_count"], 4)
        self.assertEqual(report["field_evidence_coverage"], 1.0)
        self.assertEqual(report["scenario_ready_count"], 4)
        self.assertFalse(report["commercial_demo_ready"])

    def test_unverified_is_distinct_from_verified_absence(self):
        data = valid_dataset()
        data["issues"][0]["protection_terms"]["put_terms"] = {"status":"UNVERIFIED","evidence_id":None}
        self.assertEqual(validate_dataset(data)["scenario_ready_count"], 0)

    def test_unverified_cannot_carry_confirming_evidence(self):
        data = valid_dataset()
        data["issues"][0]["protection_terms"]["put_terms"] = {"status":"UNVERIFIED","evidence_id":"ev-1"}
        with self.assertRaises(BenchmarkContractError):
            validate_dataset(data)

    def test_missing_field_evidence_fails_coverage_closed(self):
        data = valid_dataset()
        del data["issues"][0]["evidence_map"]["coupon"]
        report = validate_dataset(data)
        self.assertEqual(report["field_evidence_coverage"], 0.0)
        self.assertFalse(report["commercial_demo_ready"])

    def test_contract_layer_rejects_model_outputs(self):
        data = valid_dataset()
        data["issues"][0]["fair_value"] = 123.45
        with self.assertRaises(BenchmarkContractError):
            validate_dataset(data)

    def test_scenario_rejects_fair_value_assertion(self):
        data = valid_dataset()
        data["market_snapshots"] = [{"id":"snap-1","issue_id":"issue-1","as_of":"2026-08-11","equity_price":100.0,"risk_free_curve_ref":"ust-2026-08-11","credit_spread_if_observed":None}]
        data["scenarios"] = [{"id":"sc-1","issue_id":"issue-1","snapshot_id":"snap-1","model_version":"research-v1","assumptions":{},"model_output":{"bond_floor":90.0,"conversion_value":80.0,"option_component":15.0,"var":20.0,"expected_shortfall":25.0,"fair_value":105.0}}]
        with self.assertRaises(BenchmarkContractError):
            validate_dataset(data)


if __name__ == "__main__":
    unittest.main()
