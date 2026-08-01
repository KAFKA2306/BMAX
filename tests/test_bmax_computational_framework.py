import unittest

import numpy as np

from src.bmax_computational_framework import (
    BMAXIntegratedEngine,
    BMXRiskEngine,
    BitcoinBlackScholesEngine,
    BitcoinMarketParams,
    ConvertibleBondEngine,
    ConvertibleBondParams,
    ETFLiquidityEngine,
    ThreeLayerAssetModel,
)


class BlackScholesTests(unittest.TestCase):
    def test_zero_volatility_is_finite(self) -> None:
        engine = BitcoinBlackScholesEngine(risk_free_rate=0.01)
        price = engine.call_price(100, 90, 1, 0)
        self.assertTrue(np.isfinite(price))
        self.assertGreaterEqual(price, 0)


class CorrelationAndSimulationTests(unittest.TestCase):
    def test_correlation_matrix_remains_a_correlation_matrix(self) -> None:
        model = ThreeLayerAssetModel(BitcoinMarketParams(), 0.7, 0.5)
        self.assertTrue(np.allclose(np.diag(model.correlation_matrix), 1.0))
        self.assertGreaterEqual(
            float(np.linalg.eigvalsh(model.correlation_matrix).min()),
            -1e-10,
        )

    def test_seed_makes_simulation_reproducible(self) -> None:
        model = ThreeLayerAssetModel(BitcoinMarketParams())
        first = model.simulate_price_paths(
            (45_000, 150, 1_000), n_steps=10, n_simulations=20, seed=42
        )
        second = model.simulate_price_paths(
            (45_000, 150, 1_000), n_steps=10, n_simulations=20, seed=42
        )
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(first.shape, (20, 3, 11))


class ConvertibleBondTests(unittest.TestCase):
    def test_price_never_falls_below_floor_or_conversion_value(self) -> None:
        bs = BitcoinBlackScholesEngine()
        params = ConvertibleBondParams()
        engine = ConvertibleBondEngine(params, bs)
        price = engine.price(150, 0.6)
        self.assertGreaterEqual(price, engine.bond_floor())
        self.assertGreaterEqual(price, engine.conversion_value(150))

    def test_coupon_increases_bond_floor(self) -> None:
        bs = BitcoinBlackScholesEngine()
        zero_coupon = ConvertibleBondEngine(
            ConvertibleBondParams(coupon_rate=0), bs
        ).bond_floor()
        coupon = ConvertibleBondEngine(
            ConvertibleBondParams(coupon_rate=0.03), bs
        ).bond_floor()
        self.assertGreater(coupon, zero_coupon)


class LiquidityTests(unittest.TestCase):
    def test_ratio_requires_observed_etf_liquidity(self) -> None:
        engine = ETFLiquidityEngine()
        with self.assertRaises(ValueError):
            engine.liquidity_transformation_ratio(
                np.array([0.2, 0.4]), np.array([0.5, 0.5])
            )

    def test_measured_ratio(self) -> None:
        engine = ETFLiquidityEngine()
        ratio = engine.liquidity_transformation_ratio(
            np.array([0.2, 0.4]),
            np.array([0.5, 0.5]),
            observed_etf_liquidity=0.6,
        )
        self.assertAlmostEqual(ratio, 2.0)


class RiskTests(unittest.TestCase):
    def test_var_and_expected_shortfall_are_positive_losses(self) -> None:
        returns = np.array([-0.10, -0.05, -0.02, 0.01, 0.02, 0.03])
        engine = BMXRiskEngine(0.05)
        var = engine.bitcoin_aware_var(returns)
        expected_shortfall = engine.expected_shortfall(returns, var)
        self.assertGreaterEqual(var, 0)
        self.assertGreaterEqual(expected_shortfall, var)


class IntegratedEngineTests(unittest.TestCase):
    def test_unobserved_liquidity_is_not_invented(self) -> None:
        result = BMAXIntegratedEngine().comprehensive_analysis(
            (45_000, 150, 1_050),
            {
                "volatility": 0.6,
                "regime": "normal",
                "seed": 1,
                "n_steps": 10,
                "n_simulations": 20,
            },
        )
        self.assertIsNone(result["liquidity_transformation_ratio"])
        self.assertFalse(
            result["convertible_bond_analysis"]["calibrated_to_market"]
        )


if __name__ == "__main__":
    unittest.main()
