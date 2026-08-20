from pathlib import Path
import unittest


class BenchmarkPageContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.html = Path("benchmark/index.html").read_text(encoding="utf-8")

    def test_three_step_comparison_and_canonical_data_source(self) -> None:
        self.assertIn('id="issuer"', self.html)
        self.assertIn('id="issues"', self.html)
        self.assertIn('id="compare"', self.html)
        self.assertIn("../data/convertible_benchmark.json", self.html)
        self.assertIn("selected.size>=3", self.html)

    def test_evidence_and_claim_boundaries_are_visible(self) -> None:
        self.assertIn("SEC提出資料", self.html)
        self.assertIn("公正価値や推奨条件として扱いません", self.html)
        self.assertIn("evidence(issue,field)", self.html)
        self.assertIn("filing_opened", self.html)
        self.assertIn("poc_cta_clicked", self.html)

    def test_contract_market_model_layers_are_not_relabelled(self) -> None:
        self.assertIn('class="tag contract"', self.html)
        self.assertIn("../data/market_snapshots/", self.html)
        self.assertIn("Equity ${m.equity.observation}", self.html)
        self.assertIn("U.S. Treasury par yield", self.html)
        self.assertIn("m.equity.source_url", self.html)
        self.assertIn("m.risk_free_curve.source_url", self.html)
        self.assertIn("'observed'", self.html)
        self.assertIn("Scenario theoretical price", self.html)
        self.assertIn("model_output.theoretical_price", self.html)
        self.assertNotIn("fair value", self.html.lower())
        self.assertNotIn("recommendation", self.html.lower())


if __name__ == "__main__":
    unittest.main()
