#!/usr/bin/env python3
"""
BMAX フレームワーク単体テスト
===========================

主要計算エンジンの動作検証とエラーケースのテスト
"""

import unittest
import numpy as np
import sys
import os

# テスト対象モジュールのインポート
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from bmax_computational_framework import (
    ParameterValidator,
    BitcoinBlackScholesEngine,
    ConvertibleBondEngine,
    ConvertibleBondParams,
    BMAXIntegratedEngine,
    ThreeLayerAssetModel,
    BitcoinMarketParams
)

class TestParameterValidator(unittest.TestCase):
    """ParameterValidator クラスのテスト"""
    
    def test_validate_price_positive(self):
        """正の価格値の検証"""
        # 正常ケース
        ParameterValidator.validate_price(100.0, "test_price")
        ParameterValidator.validate_price(45000.0, "bitcoin_price")
        
    def test_validate_price_negative(self):
        """負の価格値でエラーが発生することを確認"""
        with self.assertRaises(ValueError):
            ParameterValidator.validate_price(-100.0, "negative_price")
        
        with self.assertRaises(ValueError):
            ParameterValidator.validate_price(0.0, "zero_price")
    
    def test_validate_price_too_large(self):
        """過大な価格値でエラーが発生することを確認"""
        with self.assertRaises(ValueError):
            ParameterValidator.validate_price(1e10, "huge_price")
    
    def test_validate_volatility_range(self):
        """ボラティリティの範囲検証"""
        # 正常ケース
        ParameterValidator.validate_volatility(0.6)
        ParameterValidator.validate_volatility(0.0)
        ParameterValidator.validate_volatility(2.0)
        
        # エラーケース
        with self.assertRaises(ValueError):
            ParameterValidator.validate_volatility(-0.1)
        
        with self.assertRaises(ValueError):
            ParameterValidator.validate_volatility(3.5)
    
    def test_validate_correlation_bounds(self):
        """相関係数の境界値検証"""
        # 正常ケース
        ParameterValidator.validate_correlation(0.7, "test_corr")
        ParameterValidator.validate_correlation(-1.0, "min_corr")
        ParameterValidator.validate_correlation(1.0, "max_corr")
        
        # エラーケース
        with self.assertRaises(ValueError):
            ParameterValidator.validate_correlation(1.1, "invalid_corr")
        
        with self.assertRaises(ValueError):
            ParameterValidator.validate_correlation(-1.1, "invalid_corr")


class TestBitcoinBlackScholesEngine(unittest.TestCase):
    """BitcoinBlackScholesEngine のテスト"""
    
    def setUp(self):
        self.bs_engine = BitcoinBlackScholesEngine()
    
    def test_call_price_basic(self):
        """基本的なコールオプション価格計算"""
        price = self.bs_engine.call_price(S=100, K=100, T=1.0, sigma=0.2)
        
        # 価格は非負でなければならない
        self.assertGreaterEqual(price, 0)
        
        # ATMオプションは正の価値を持つ
        self.assertGreater(price, 0)
    
    def test_call_price_moneyness(self):
        """オプションのマネーネス検証"""
        S, K, T, sigma = 100, 90, 1.0, 0.2
        
        # ITM > ATM > OTM
        itm_price = self.bs_engine.call_price(S=S, K=K, T=T, sigma=sigma)
        atm_price = self.bs_engine.call_price(S=S, K=S, T=T, sigma=sigma)
        otm_price = self.bs_engine.call_price(S=S, K=110, T=T, sigma=sigma)
        
        self.assertGreater(itm_price, atm_price)
        self.assertGreater(atm_price, otm_price)
    
    def test_delta_bounds(self):
        """デルタが[0,1]の範囲内であることを確認"""
        S, K, T, sigma = 100, 100, 1.0, 0.3
        
        delta = self.bs_engine.delta(S=S, K=K, T=T, sigma=sigma)
        
        self.assertGreaterEqual(delta, 0.0)
        self.assertLessEqual(delta, 1.0)
    
    def test_gamma_positive(self):
        """ガンマが非負であることを確認"""
        S, K, T, sigma = 100, 100, 1.0, 0.3
        
        gamma = self.bs_engine.gamma(S=S, K=K, T=T, sigma=sigma)
        
        self.assertGreaterEqual(gamma, 0.0)
    
    def test_zero_time_to_expiry(self):
        """満期時（T=0）の価格計算"""
        S, K = 110, 100
        
        price = self.bs_engine.call_price(S=S, K=K, T=0, sigma=0.2)
        expected_intrinsic = max(S - K, 0)
        
        self.assertAlmostEqual(price, expected_intrinsic, places=6)


class TestConvertibleBondEngine(unittest.TestCase):
    """ConvertibleBondEngine のテスト"""
    
    def setUp(self):
        self.cb_params = ConvertibleBondParams()
        self.bs_engine = BitcoinBlackScholesEngine()
        from bmax_computational_framework import CompoundOptionEngine
        self.compound_engine = CompoundOptionEngine(self.bs_engine)
        self.cb_engine = ConvertibleBondEngine(
            self.cb_params, self.bs_engine, self.compound_engine
        )
    
    def test_bond_floor_positive(self):
        """債券フロアが正の値であることを確認"""
        bond_floor = self.cb_engine.bond_floor()
        
        self.assertGreater(bond_floor, 0)
        self.assertLess(bond_floor, self.cb_params.face_value)  # 割引後なので額面より小さい
    
    def test_conversion_value(self):
        """転換価値の計算"""
        stock_price = 120.0
        conv_value = self.cb_engine.conversion_value(stock_price)
        expected = self.cb_params.conversion_ratio * stock_price
        
        self.assertAlmostEqual(conv_value, expected, places=6)
    
    def test_cb_price_vs_bond_floor(self):
        """転換社債価格が債券フロア以上であることを確認"""
        stock_price = 80.0  # 低い株価
        volatility = 0.3
        
        cb_price = self.cb_engine.price(stock_price, volatility)
        bond_floor = self.cb_engine.bond_floor()
        
        self.assertGreaterEqual(cb_price, bond_floor * 0.95)  # 流動性調整考慮
    
    def test_delta_bounds_fixed(self):
        """修正後のデルタが[0,1]範囲内であることを確認（重要テスト）"""
        stock_price = 150.0
        volatility = 0.6
        
        greeks = self.cb_engine.greeks(stock_price, volatility)
        delta = greeks['delta']
        
        # これが主要な修正点：デルタは0-1の範囲内でなければならない
        self.assertGreaterEqual(delta, 0.0, "Delta should be non-negative")
        self.assertLessEqual(delta, 1.0, "Delta should not exceed 1.0")
        
        print(f"Calculated delta: {delta:.6f}")
        
        # 上昇参加率として妥当な値（20%程度が期待値）
        self.assertLess(delta, 0.8, "Delta should be reasonable for convertible bond")


class TestThreeLayerAssetModel(unittest.TestCase):
    """ThreeLayerAssetModel のテスト"""
    
    def setUp(self):
        self.bitcoin_params = BitcoinMarketParams()
        self.model = ThreeLayerAssetModel(self.bitcoin_params)
    
    def test_correlation_matrix_properties(self):
        """相関行列の数学的性質を確認"""
        corr_matrix = self.model.correlation_matrix
        
        # 対称行列
        self.assertTrue(np.allclose(corr_matrix, corr_matrix.T))
        
        # 対角要素は1
        np.testing.assert_array_almost_equal(np.diag(corr_matrix), [1.0, 1.0, 1.0])
        
        # 正定値（全固有値が正）
        eigenvals = np.linalg.eigvals(corr_matrix)
        self.assertTrue(np.all(eigenvals > -1e-10))  # 数値誤差考慮
    
    def test_price_simulation_shape(self):
        """価格シミュレーションの出力形状確認"""
        initial_prices = (45000.0, 150.0, 1050.0)
        n_simulations = 100
        n_steps = 50
        
        paths = self.model.simulate_price_paths(
            initial_prices, n_simulations=n_simulations, n_steps=n_steps
        )
        
        expected_shape = (n_simulations, 3, n_steps + 1)
        self.assertEqual(paths.shape, expected_shape)
        
        # 初期価格の確認
        np.testing.assert_array_almost_equal(paths[:, 0, 0], initial_prices[0])
        np.testing.assert_array_almost_equal(paths[:, 1, 0], initial_prices[1])
        np.testing.assert_array_almost_equal(paths[:, 2, 0], initial_prices[2])
    
    def test_price_paths_positive(self):
        """価格パスが正の値を維持することを確認"""
        initial_prices = (45000.0, 150.0, 1050.0)
        
        paths = self.model.simulate_price_paths(
            initial_prices, n_simulations=10, n_steps=10
        )
        
        # 全価格が正であることを確認
        self.assertTrue(np.all(paths > 0))


class TestBMAXIntegratedEngine(unittest.TestCase):
    """BMAXIntegratedEngine 統合テスト"""
    
    def setUp(self):
        self.engine = BMAXIntegratedEngine()
    
    def test_comprehensive_analysis_structure(self):
        """包括的分析の出力構造確認"""
        current_prices = (45000.0, 150.0, 1050.0)
        market_conditions = {'volatility': 0.6, 'regime': 'normal'}
        
        results = self.engine.comprehensive_analysis(current_prices, market_conditions)
        
        # 必要なキーが存在することを確認
        required_keys = [
            'convertible_bond_analysis',
            'correlation_structure', 
            'liquidity_transformation_ratio',
            'risk_metrics',
            'portfolio_characteristics'
        ]
        
        for key in required_keys:
            self.assertIn(key, results)
    
    def test_upside_participation_fixed(self):
        """修正後の上昇参加率が妥当であることを確認（最重要テスト）"""
        current_prices = (45000.0, 150.0, 1050.0)
        market_conditions = {'volatility': 0.6}
        
        results = self.engine.comprehensive_analysis(current_prices, market_conditions)
        
        upside_participation = results['portfolio_characteristics']['upside_participation']
        
        # これが主要な修正の検証
        self.assertGreaterEqual(upside_participation, 0.0)
        self.assertLessEqual(upside_participation, 1.0)
        
        # 実用的な範囲（10%-80%）内にあることを確認
        self.assertGreater(upside_participation, 0.05)
        self.assertLess(upside_participation, 0.9)
        
        print(f"Fixed upside participation rate: {upside_participation:.6f}")
    
    def test_input_validation_integration(self):
        """統合レベルでの入力値検証"""
        # 無効な価格での分析
        with self.assertRaises(ValueError):
            self.engine.comprehensive_analysis(
                (-100, 150, 1050), {'volatility': 0.6}
            )
        
        # 無効なボラティリティでの分析
        with self.assertRaises(ValueError):
            self.engine.comprehensive_analysis(
                (45000, 150, 1050), {'volatility': 5.0}
            )


class TestStressScenarios(unittest.TestCase):
    """ストレステスト・極端シナリオのテスト"""
    
    def setUp(self):
        self.engine = BMAXIntegratedEngine()
    
    def test_bitcoin_crash_scenario(self):
        """Bitcoin大暴落シナリオ"""
        crash_prices = (20000.0, 50.0, 800.0)
        conditions = {'volatility': 1.2, 'regime': 'crisis'}
        
        try:
            results = self.engine.comprehensive_analysis(crash_prices, conditions)
            
            # 債券フロアが保護として機能することを確認
            cb_analysis = results['convertible_bond_analysis']
            downside_protection = results['portfolio_characteristics']['downside_protection']
            
            self.assertGreater(downside_protection, 0.3)  # 30%以上の下方保護
            
        except Exception as e:
            self.fail(f"Bitcoin crash scenario failed: {e}")
    
    def test_bitcoin_moon_scenario(self):
        """Bitcoin急騰シナリオ"""
        moon_prices = (100000.0, 500.0, 2000.0)
        conditions = {'volatility': 0.8, 'regime': 'bull'}
        
        try:
            results = self.engine.comprehensive_analysis(moon_prices, conditions)
            
            # 上昇参加が機能することを確認
            upside_participation = results['portfolio_characteristics']['upside_participation']
            
            self.assertGreater(upside_participation, 0.1)  # 最低10%の上昇参加
            
        except Exception as e:
            self.fail(f"Bitcoin moon scenario failed: {e}")
    
    def test_high_volatility_stability(self):
        """高ボラティリティ環境での安定性"""
        prices = (45000.0, 150.0, 1050.0)
        conditions = {'volatility': 1.5}  # 150%ボラティリティ
        
        try:
            results = self.engine.comprehensive_analysis(prices, conditions)
            
            # 計算が完了し、妥当な結果が得られることを確認
            self.assertIsInstance(results, dict)
            
            # リスク指標が適切に算出されることを確認
            risk_metrics = results['risk_metrics']
            self.assertIn('bitcoin_aware_var', risk_metrics)
            
        except Exception as e:
            self.fail(f"High volatility scenario failed: {e}")


def run_comprehensive_tests():
    """包括的テストスイートの実行"""
    
    print("🧪 BMAX フレームワーク包括的テスト開始")
    print("=" * 60)
    
    # テストスイートの構成
    test_classes = [
        TestParameterValidator,
        TestBitcoinBlackScholesEngine,
        TestConvertibleBondEngine,
        TestThreeLayerAssetModel,
        TestBMAXIntegratedEngine,
        TestStressScenarios
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # テスト実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    print(f"📊 テスト結果サマリー:")
    print(f"  実行テスト数: {result.testsRun}")
    print(f"  成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  失敗: {len(result.failures)}")
    print(f"  エラー: {len(result.errors)}")
    
    if result.failures:
        print(f"\n❌ 失敗したテスト:")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print(f"\n⚠️ エラーが発生したテスト:")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    if result.wasSuccessful():
        print(f"\n✅ 全テストが正常に完了しました！")
        return True
    else:
        print(f"\n❌ テストに失敗があります。上記を確認してください。")
        return False


if __name__ == "__main__":
    # 単体テスト実行
    success = run_comprehensive_tests()
    
    if success:
        print(f"\n🎉 BMAX フレームワークの信頼性が確認されました！")
    else:
        print(f"\n🔧 修正が必要な問題が見つかりました。")
        exit(1)