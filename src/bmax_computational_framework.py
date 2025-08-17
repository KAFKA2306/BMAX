#!/usr/bin/env python3
"""
BMAX REX 転換社債ETF 計算フレームワーク
====================================

ビットコイン関連企業転換社債の数学的価格決定理論を実装する
包括的計算フレームワーク

主要コンポーネント:
1. ハイブリッド証券価格決定エンジン
2. 三層資産モデル実装
3. 複合オプション価格計算
4. ETF流動性変換メカニズム
5. リスク測定・ヘッジシステム

作成者: 金融工学研究チーム
バージョン: 1.0 (革新的実装)
"""

import numpy as np
import pandas as pd
from scipy import stats, optimize
from scipy.special import ndtr
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable
from abc import ABC, abstractmethod
from functools import lru_cache
import multiprocessing as mp
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 入力値検証システム
# =============================================================================

class ParameterValidator:
    """パラメータ検証クラス"""
    
    @staticmethod
    def validate_price(price: float, name: str = "price") -> None:
        """価格パラメータの検証"""
        if not isinstance(price, (int, float)):
            raise TypeError(f"{name} must be a number")
        if price <= 0:
            raise ValueError(f"{name} must be positive, got {price}")
        if price > 1e9:  # 10億ドル制限
            raise ValueError(f"{name} too large, got {price}")
    
    @staticmethod
    def validate_volatility(volatility: float) -> None:
        """ボラティリティの検証"""
        if not isinstance(volatility, (int, float)):
            raise TypeError("Volatility must be a number")
        if volatility < 0:
            raise ValueError(f"Volatility must be non-negative, got {volatility}")
        if volatility > 3.0:  # 300%制限
            raise ValueError(f"Volatility too high, got {volatility} (max 3.0)")
    
    @staticmethod
    def validate_correlation(correlation: float, name: str = "correlation") -> None:
        """相関係数の検証"""
        if not isinstance(correlation, (int, float)):
            raise TypeError(f"{name} must be a number")
        if not (-1.0 <= correlation <= 1.0):
            raise ValueError(f"{name} must be between -1 and 1, got {correlation}")
    
    @staticmethod
    def validate_time_to_maturity(time: float) -> None:
        """満期までの時間の検証"""
        if not isinstance(time, (int, float)):
            raise TypeError("Time to maturity must be a number")
        if time < 0:
            raise ValueError(f"Time to maturity must be non-negative, got {time}")
        if time > 50:  # 50年制限
            raise ValueError(f"Time to maturity too long, got {time} years")
    
    @staticmethod
    def validate_bitcoin_stock_consistency(bitcoin_price: float, stock_price: float) -> None:
        """Bitcoin価格と株価の整合性チェック"""
        if stock_price > bitcoin_price * 0.1:
            import warnings
            warnings.warn(f"Warning: 株価({stock_price})がビットコイン価格({bitcoin_price})の10%を超えています")

# =============================================================================
# 基礎データ構造
# =============================================================================

@dataclass
class ConvertibleBondParams:
    """転換社債パラメータ"""
    face_value: float = 1000.0          # 額面価格
    conversion_ratio: float = 10.0      # 転換比率
    conversion_price: float = 100.0     # 転換価格
    coupon_rate: float = 0.0            # クーポン率（多くは0%）
    maturity: float = 5.0               # 満期（年）
    credit_spread: float = 0.02         # 信用スプレッド
    call_protection: float = 2.0        # コール保護期間

@dataclass
class BitcoinMarketParams:
    """ビットコイン市場パラメータ"""
    current_price: float = 45000.0      # 現在価格（USD）
    drift: float = 0.15                 # ドリフト率（年率）
    volatility: float = 0.80            # ボラティリティ（年率）
    jump_intensity: float = 0.1         # ジャンプ頻度
    jump_mean: float = 0.0              # ジャンプ平均
    jump_std: float = 0.15              # ジャンプ標準偏差

@dataclass
class MarketRegime:
    """市場レジーム"""
    regime_name: str                    # レジーム名
    transition_matrix: np.ndarray       # 遷移確率行列
    volatility_multiplier: float       # ボラティリティ倍率
    correlation_modifier: float        # 相関修正項

# =============================================================================
# 抽象基底クラス
# =============================================================================

class PricingEngine(ABC):
    """価格決定エンジンの抽象基底クラス"""
    
    @abstractmethod
    def price(self, *args, **kwargs) -> float:
        """価格計算の抽象メソッド"""
        pass
    
    @abstractmethod
    def greeks(self, *args, **kwargs) -> Dict[str, float]:
        """Greeks計算の抽象メソッド"""
        pass

# =============================================================================
# Black-Scholes基底クラス（Bitcoinアプリケーション用拡張）
# =============================================================================

class BitcoinBlackScholesEngine:
    """Bitcoin特化型Black-Scholesエンジン（パフォーマンス最適化済み）"""
    
    def __init__(self, risk_free_rate: float = 0.045):
        self.risk_free_rate = risk_free_rate
        self._cache_size = 1000  # キャッシュサイズ
    
    def d1(self, S: float, K: float, T: float, sigma: float, q: float = 0.0) -> float:
        """d1パラメータ計算（配当調整済み）"""
        return (np.log(S/K) + (self.risk_free_rate - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    def d2(self, S: float, K: float, T: float, sigma: float, q: float = 0.0) -> float:
        """d2パラメータ計算"""
        return self.d1(S, K, T, sigma, q) - sigma * np.sqrt(T)
    
    @lru_cache(maxsize=1000)
    def _cached_call_price_core(self, S: float, K: float, T: float, sigma: float, q: float = 0.0) -> float:
        """キャッシュされたコールオプション価格計算（コア部分）"""
        if T <= 0:
            return max(S - K, 0)
        
        d1_val = self.d1(S, K, T, sigma, q)
        d2_val = self.d2(S, K, T, sigma, q)
        
        call = (S * np.exp(-q * T) * ndtr(d1_val) - 
                K * np.exp(-self.risk_free_rate * T) * ndtr(d2_val))
        
        return max(call, 0)
    
    def call_price(self, S: float, K: float, T: float, sigma: float, q: float = 0.0) -> float:
        """コールオプション価格（Bitcoin企業株式用）"""
        # 入力値検証
        ParameterValidator.validate_price(S, "Stock price")
        ParameterValidator.validate_price(K, "Strike price")
        ParameterValidator.validate_time_to_maturity(T)
        ParameterValidator.validate_volatility(sigma)
        
        # キャッシュされた計算を使用
        return self._cached_call_price_core(S, K, T, sigma, q)
    
    def delta(self, S: float, K: float, T: float, sigma: float, q: float = 0.0) -> float:
        """デルタ計算"""
        if T <= 0:
            return 1.0 if S > K else 0.0
        
        d1_val = self.d1(S, K, T, sigma, q)
        return np.exp(-q * T) * ndtr(d1_val)
    
    def gamma(self, S: float, K: float, T: float, sigma: float, q: float = 0.0) -> float:
        """ガンマ計算"""
        if T <= 0:
            return 0.0
        
        d1_val = self.d1(S, K, T, sigma, q)
        return (np.exp(-q * T) * stats.norm.pdf(d1_val)) / (S * sigma * np.sqrt(T))
    
    def vega(self, S: float, K: float, T: float, sigma: float, q: float = 0.0) -> float:
        """ベガ計算"""
        if T <= 0:
            return 0.0
        
        d1_val = self.d1(S, K, T, sigma, q)
        return S * np.exp(-q * T) * stats.norm.pdf(d1_val) * np.sqrt(T) / 100

# =============================================================================
# 三層資産モデル実装
# =============================================================================

class ThreeLayerAssetModel:
    """Bitcoin → 企業株式 → 転換社債の三層資産モデル"""
    
    def __init__(self, 
                 bitcoin_params: BitcoinMarketParams,
                 stock_correlation: float = 0.7,
                 cb_correlation: float = 0.5):
        
        self.bitcoin_params = bitcoin_params
        self.stock_correlation = stock_correlation      # Bitcoin-Stock相関
        self.cb_correlation = cb_correlation            # Stock-CB相関
        
        # 相関行列構築
        self.correlation_matrix = self._build_correlation_matrix()
    
    def _build_correlation_matrix(self) -> np.ndarray:
        """三層間相関行列の構築（数値安定化対応）"""
        # 入力値検証
        ParameterValidator.validate_correlation(self.stock_correlation, "stock_correlation")
        ParameterValidator.validate_correlation(self.cb_correlation, "cb_correlation")
        
        # Bitcoin-Stock-CB の三次元相関行列
        rho_bs = self.stock_correlation
        rho_sc = self.cb_correlation
        rho_bc = rho_bs * rho_sc  # 間接相関（合成）
        
        correlation_matrix = np.array([
            [1.0,    rho_bs, rho_bc],  # Bitcoin
            [rho_bs, 1.0,    rho_sc],  # Stock
            [rho_bc, rho_sc, 1.0]      # Convertible Bond
        ])
        
        # 正定値性の確保
        eigenvals = np.linalg.eigvals(correlation_matrix)
        min_eigenval = np.min(eigenvals)
        
        if min_eigenval < 1e-8:
            # 微小な正の値を対角要素に追加
            regularization = 1e-6
            correlation_matrix += np.eye(3) * regularization
            print(f"Warning: 相関行列が非正定値でした。正則化パラメータ {regularization} を追加しました。")
        
        return correlation_matrix
    
    def _simulate_chunk(self, args: tuple) -> np.ndarray:
        """並列処理用のシミュレーションチャンク"""
        chunk_size, initial_prices, time_horizon, n_steps, L = args
        
        dt = time_horizon / n_steps
        bitcoin_init, stock_init, cb_init = initial_prices
        
        # 価格パス初期化
        paths = np.zeros((chunk_size, 3, n_steps + 1))
        paths[:, 0, 0] = bitcoin_init   # Bitcoin
        paths[:, 1, 0] = stock_init     # Stock
        paths[:, 2, 0] = cb_init        # Convertible Bond
        
        for sim in range(chunk_size):
            for t in range(1, n_steps + 1):
                # 独立な正規乱数
                z_independent = np.random.normal(0, 1, 3)
                
                # 相関のある乱数
                z_correlated = L @ z_independent
                
                # Bitcoin価格更新（ジャンプ拡散モデル）
                bitcoin_drift = self.bitcoin_params.drift * dt
                bitcoin_diffusion = self.bitcoin_params.volatility * np.sqrt(dt) * z_correlated[0]
                
                # ジャンプ項
                jump_occurs = np.random.poisson(self.bitcoin_params.jump_intensity * dt)
                jump_size = 0
                if jump_occurs > 0:
                    jump_size = np.random.normal(
                        self.bitcoin_params.jump_mean, 
                        self.bitcoin_params.jump_std
                    )
                
                bitcoin_return = bitcoin_drift + bitcoin_diffusion + jump_size
                paths[sim, 0, t] = paths[sim, 0, t-1] * np.exp(bitcoin_return)
                
                # 株価更新（Bitcoin連動）
                stock_vol = 0.6  # 株式ボラティリティ（Bitcoin より低い）
                stock_drift = 0.12 * dt  # 株式期待リターン
                stock_diffusion = stock_vol * np.sqrt(dt) * z_correlated[1]
                
                stock_return = stock_drift + stock_diffusion
                paths[sim, 1, t] = paths[sim, 1, t-1] * np.exp(stock_return)
                
                # 転換社債価格更新（株価連動＋債券フロア効果）
                cb_vol = 0.4  # CB ボラティリティ（株式より低い）
                cb_drift = 0.08 * dt  # CB期待リターン
                cb_diffusion = cb_vol * np.sqrt(dt) * z_correlated[2]
                
                cb_return = cb_drift + cb_diffusion
                paths[sim, 2, t] = paths[sim, 2, t-1] * np.exp(cb_return)
        
        return paths
    
    def simulate_price_paths(self, 
                           initial_prices: Tuple[float, float, float],
                           time_horizon: float = 1.0,
                           n_steps: int = 252,
                           n_simulations: int = 10000,
                           use_parallel: bool = True) -> np.ndarray:
        """
        三層資産価格パスの同時シミュレーション（並列処理対応）
        
        Args:
            use_parallel: 並列処理を使用するかどうか
        
        Returns:
            価格パス配列 [simulations, assets, time_steps]
        """
        
        # Choleskyフレームワークによる相関のある乱数生成（エラーハンドリング付き）
        try:
            L = np.linalg.cholesky(self.correlation_matrix)
        except np.linalg.LinAlgError:
            # Choleskyが失敗した場合はSVD分解を使用
            print("Warning: Cholesky分解が失敗しました。SVD分解を使用します。")
            U, s, Vt = np.linalg.svd(self.correlation_matrix)
            # 負の固有値を0にクリップ
            s = np.maximum(s, 1e-10)
            L = U @ np.diag(np.sqrt(s))
        
        if use_parallel and n_simulations >= 1000:
            # 並列処理を使用
            n_cores = min(mp.cpu_count(), 4)  # 最大4コア使用
            chunk_size = n_simulations // n_cores
            
            # 残りのシミュレーション数を調整
            remainder = n_simulations % n_cores
            chunk_sizes = [chunk_size] * n_cores
            if remainder > 0:
                chunk_sizes[-1] += remainder
            
            # 並列処理用の引数
            args_list = [(size, initial_prices, time_horizon, n_steps, L) 
                        for size in chunk_sizes if size > 0]
            
            try:
                with mp.Pool(len(args_list)) as pool:
                    results = pool.map(self._simulate_chunk, args_list)
                
                # 結果を結合
                paths = np.concatenate(results, axis=0)
                
            except Exception as e:
                print(f"Warning: 並列処理が失敗しました。シーケンシャル処理にフォールバック: {e}")
                use_parallel = False
        
        if not use_parallel or n_simulations < 1000:
            # シーケンシャル処理
            paths = self._simulate_chunk((n_simulations, initial_prices, time_horizon, n_steps, L))
        
        return paths
    
    def correlation_analysis(self, price_paths: np.ndarray) -> Dict[str, float]:
        """価格パスから実現相関の分析"""
        
        # リターン計算
        returns = np.diff(np.log(price_paths), axis=2)
        
        # 各シミュレーションでの相関計算
        correlations = []
        
        for sim in range(returns.shape[0]):
            sim_returns = returns[sim, :, :].T  # [time, assets]
            corr_matrix = np.corrcoef(sim_returns.T)
            correlations.append(corr_matrix)
        
        avg_correlation = np.mean(correlations, axis=0)
        
        return {
            'bitcoin_stock': avg_correlation[0, 1],
            'bitcoin_cb': avg_correlation[0, 2],
            'stock_cb': avg_correlation[1, 2],
            'correlation_matrix': avg_correlation
        }

# =============================================================================
# 複合オプション価格決定エンジン
# =============================================================================

class CompoundOptionEngine(PricingEngine):
    """複合オプション（Bitcoin → Stock → CB）価格決定エンジン"""
    
    def __init__(self, bs_engine: BitcoinBlackScholesEngine):
        self.bs_engine = bs_engine
    
    def compound_call_price(self, 
                          S: float,              # 現在の株価
                          K1: float,             # 第1段階行使価格
                          K2: float,             # 第2段階行使価格（転換価格）
                          T1: float,             # 第1段階満期
                          T2: float,             # 第2段階満期
                          sigma: float,          # ボラティリティ
                          correlation: float = 0.7) -> float:
        """
        複合コールオプション価格計算
        
        Bitcoin価格変動 → 株価変動 → 転換オプション価値
        の二段階価格形成を反映
        """
        
        if T1 <= 0 or T2 <= 0:
            return max(self.bs_engine.call_price(S, K2, T2, sigma) - K1, 0)
        
        # 臨界株価の計算
        def critical_stock_price_equation(S_star):
            option_value = self.bs_engine.call_price(S_star, K2, T2 - T1, sigma)
            return option_value - K1
        
        try:
            # Newton法で臨界価格を求解
            S_star = optimize.newton(critical_stock_price_equation, K1 + K2)
        except:
            S_star = K1 + K2  # 収束しない場合のフォールバック
        
        # 複合オプション価格の計算
        # Geske (1979) モデルの実装
        
        sigma1 = sigma * np.sqrt(T1)
        sigma2 = sigma * np.sqrt(T2)
        rho = np.sqrt(T1 / T2)  # 時間相関
        
        d1 = (np.log(S / S_star) + (self.bs_engine.risk_free_rate + 0.5 * sigma**2) * T1) / sigma1
        d2 = d1 - sigma1
        d3 = (np.log(S / K2) + (self.bs_engine.risk_free_rate + 0.5 * sigma**2) * T2) / sigma2
        d4 = d3 - sigma2
        
        # 二変量正規分布の累積分布関数
        from scipy.stats import multivariate_normal
        
        # M(d1, d3; rho) の計算
        M_d1_d3 = multivariate_normal.cdf([d1, d3], cov=[[1, rho], [rho, 1]])
        M_d2_d4 = multivariate_normal.cdf([d2, d4], cov=[[1, rho], [rho, 1]])
        
        compound_price = (S * M_d1_d3 - 
                         K2 * np.exp(-self.bs_engine.risk_free_rate * T2) * M_d2_d4 -
                         K1 * np.exp(-self.bs_engine.risk_free_rate * T1) * ndtr(d2))
        
        return max(compound_price, 0)
    
    def price(self, *args, **kwargs) -> float:
        """価格計算（基底クラス実装）"""
        return self.compound_call_price(*args, **kwargs)
    
    def greeks(self, S: float, K1: float, K2: float, T1: float, T2: float, 
               sigma: float, **kwargs) -> Dict[str, float]:
        """複合オプションのGreeks計算"""
        
        # 数値微分によるGreeks計算
        h = 0.01
        base_price = self.compound_call_price(S, K1, K2, T1, T2, sigma)
        
        # Delta
        delta = (self.compound_call_price(S + h, K1, K2, T1, T2, sigma) - base_price) / h
        
        # Gamma
        gamma = (self.compound_call_price(S + h, K1, K2, T1, T2, sigma) - 
                2 * base_price + 
                self.compound_call_price(S - h, K1, K2, T1, T2, sigma)) / (h**2)
        
        # Vega
        vega = (self.compound_call_price(S, K1, K2, T1, T2, sigma + 0.01) - base_price) / 0.01
        
        # Theta1 (T1に対する)
        theta1 = -(self.compound_call_price(S, K1, K2, T1 - 1/365, T2, sigma) - base_price) * 365
        
        # Theta2 (T2に対する)
        theta2 = -(self.compound_call_price(S, K1, K2, T1, T2 - 1/365, sigma) - base_price) * 365
        
        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta1': theta1,
            'theta2': theta2
        }

# =============================================================================
# 転換社債価格決定エンジン
# =============================================================================

class ConvertibleBondEngine(PricingEngine):
    """転換社債総合価格決定エンジン"""
    
    def __init__(self, 
                 cb_params: ConvertibleBondParams,
                 bs_engine: BitcoinBlackScholesEngine,
                 compound_engine: CompoundOptionEngine):
        
        self.cb_params = cb_params
        self.bs_engine = bs_engine
        self.compound_engine = compound_engine
    
    def bond_floor(self, credit_spread: float = None) -> float:
        """債券フロア価値計算"""
        
        if credit_spread is None:
            credit_spread = self.cb_params.credit_spread
        
        # ゼロクーポン債の現在価値
        discount_rate = self.bs_engine.risk_free_rate + credit_spread
        bond_floor = self.cb_params.face_value * np.exp(-discount_rate * self.cb_params.maturity)
        
        return bond_floor
    
    def conversion_value(self, stock_price: float) -> float:
        """転換価値計算"""
        return self.cb_params.conversion_ratio * stock_price
    
    def conversion_premium(self, cb_price: float, stock_price: float) -> float:
        """転換プレミアム計算"""
        conv_value = self.conversion_value(stock_price)
        if conv_value > 0:
            return (cb_price - conv_value) / conv_value
        return 0.0
    
    def option_value(self, stock_price: float, volatility: float) -> float:
        """転換オプション価値（アメリカンスタイル近似）"""
        
        # Black-Scholes近似（ヨーロピアンスタイル）
        european_value = self.bs_engine.call_price(
            S=stock_price,
            K=self.cb_params.conversion_price,
            T=self.cb_params.maturity,
            sigma=volatility
        ) * self.cb_params.conversion_ratio
        
        # アメリカンスタイル調整（簡易近似）
        # より精密にはBinomial TreeやMonte Carloが必要
        american_premium = 0.05 * european_value  # 経験的調整
        
        return european_value + american_premium
    
    def price(self, stock_price: float, volatility: float, 
              credit_spread: float = None) -> float:
        """転換社債価格総合計算"""
        
        # 入力値検証
        ParameterValidator.validate_price(stock_price, "Stock price")
        ParameterValidator.validate_volatility(volatility)
        
        # 構成要素の計算
        bond_floor_val = self.bond_floor(credit_spread)
        option_val = self.option_value(stock_price, volatility)
        
        # 流動性プレミアム/ディスカウント
        liquidity_adjustment = -0.02 * bond_floor_val  # 2% 流動性ディスカウント
        
        # 総合価値
        total_value = max(bond_floor_val, bond_floor_val + option_val) + liquidity_adjustment
        
        return total_value
    
    def greeks(self, stock_price: float, volatility: float, **kwargs) -> Dict[str, float]:
        """転換社債のGreeks計算"""
        
        h = 0.01
        base_price = self.price(stock_price, volatility)
        
        # Delta (株価感応度) - Calculate as percentage change ratio
        delta_absolute = (self.price(stock_price + h, volatility) - base_price) / h
        
        # Normalize delta by current CB price to get percentage sensitivity
        # This gives us the percentage change in CB price per 1% change in stock price
        delta = (delta_absolute * stock_price) / base_price if base_price > 0 else 0.0
        
        # Ensure delta is bounded [0, 1] for convertible bonds
        delta = max(0.0, min(1.0, delta))
        
        # Gamma
        gamma = (self.price(stock_price + h, volatility) - 
                2 * base_price + 
                self.price(stock_price - h, volatility)) / (h**2)
        
        # Vega
        vega = (self.price(stock_price, volatility + 0.01) - base_price) / 0.01
        
        # Credit Spread Sensitivity
        credit_sens = (self.price(stock_price, volatility, 
                                 self.cb_params.credit_spread + 0.01) - base_price) / 0.01
        
        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'credit_sensitivity': credit_sens,
            'conversion_ratio': self.cb_params.conversion_ratio
        }

# =============================================================================
# ETF流動性変換エンジン
# =============================================================================

class ETFLiquidityEngine:
    """ETF流動性変換メカニズム"""
    
    def __init__(self, 
                 creation_unit_size: int = 50000,
                 transaction_costs: float = 0.001):
        
        self.creation_unit_size = creation_unit_size
        self.transaction_costs = transaction_costs
    
    def liquidity_transformation_ratio(self, 
                                     individual_liquidity: np.ndarray,
                                     weights: np.ndarray) -> float:
        """
        個別転換社債流動性のETF流動性への変換比率
        
        Args:
            individual_liquidity: 個別転換社債の流動性指標配列
            weights: ポートフォリオウェイト
            
        Returns:
            流動性変換比率
        """
        
        # 加重平均個別流動性
        weighted_individual = np.sum(weights * individual_liquidity)
        
        # ETF構造による流動性増強効果
        # Kyle (1985) モデルの拡張
        pooling_benefit = 1.2  # プーリング効果
        creation_redemption_benefit = 1.15  # 作成・償還効果
        market_making_benefit = 1.1  # マーケットメイキング効果
        
        # 総合流動性変換比率
        transformation_ratio = (pooling_benefit * 
                               creation_redemption_benefit * 
                               market_making_benefit)
        
        etf_liquidity = weighted_individual * transformation_ratio
        
        return etf_liquidity / weighted_individual
    
    def premium_discount_dynamics(self, 
                                nav: float, 
                                market_price: float,
                                trading_volume: float) -> Dict[str, float]:
        """ETF プレミアム・ディスカウント動態分析"""
        
        premium_discount = (market_price - nav) / nav
        
        # 取引量と乖離の関係（負の相関）
        volume_effect = -0.001 * np.log(1 + trading_volume / 1000000)
        
        # 裁定限界
        arbitrage_threshold = self.transaction_costs * 2
        
        # 収束速度（平均回帰）
        reversion_speed = 0.1 if abs(premium_discount) > arbitrage_threshold else 0.3
        
        return {
            'premium_discount': premium_discount,
            'arbitrage_threshold': arbitrage_threshold,
            'reversion_speed': reversion_speed,
            'volume_effect': volume_effect
        }

# =============================================================================
# リスク測定・管理エンジン
# =============================================================================

class BMXRiskEngine:
    """BMAX特化型リスク測定・管理エンジン"""
    
    def __init__(self, confidence_level: float = 0.05):
        self.confidence_level = confidence_level
    
    def bitcoin_aware_var(self, 
                         returns: np.ndarray,
                         bitcoin_regime: str = 'normal') -> float:
        """Bitcoin状況を考慮したVaR計算"""
        
        # レジーム別調整
        regime_adjustments = {
            'bull': 0.8,      # 強気相場：リスク低下
            'normal': 1.0,    # 通常市場
            'bear': 1.3,      # 弱気相場：リスク増加
            'crisis': 1.8     # 危機：リスク大幅増加
        }
        
        adjustment = regime_adjustments.get(bitcoin_regime, 1.0)
        
        # Historical VaR
        var_historical = np.percentile(returns, self.confidence_level * 100)
        
        # Bitcoin adjusted VaR
        bitcoin_aware_var = var_historical * adjustment
        
        return bitcoin_aware_var
    
    def expected_shortfall(self, returns: np.ndarray, var: float) -> float:
        """期待ショートフォール（条件付きVaR）"""
        tail_returns = returns[returns <= var]
        return np.mean(tail_returns) if len(tail_returns) > 0 else var
    
    def tail_risk_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """テールリスク指標群"""
        
        # 基本統計量
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        # 極値統計（GPD）パラメータ推定
        threshold = np.percentile(returns, 5)  # 5%閾値
        exceedances = returns[returns < threshold] - threshold
        
        if len(exceedances) > 10:  # 十分なデータがある場合
            # 形状パラメータ（Hill推定量）
            sorted_exc = np.sort(-exceedances)
            k = len(sorted_exc) // 4  # サンプルサイズの1/4
            hill_estimator = np.mean(np.log(sorted_exc[:k])) - np.log(sorted_exc[k-1])
            
            # スケールパラメータ
            scale_param = sorted_exc[k-1] * hill_estimator
        else:
            hill_estimator = 0.1  # デフォルト値
            scale_param = abs(threshold) * 0.1
        
        return {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'hill_estimator': hill_estimator,
            'scale_parameter': scale_param,
            'tail_index': hill_estimator
        }
    
    def optimal_hedge_ratio(self, 
                          asset_returns: np.ndarray,
                          hedge_returns: np.ndarray) -> Dict[str, float]:
        """最適ヘッジ比率計算"""
        
        # 最小分散ヘッジ比率
        covariance = np.cov(asset_returns, hedge_returns)[0, 1]
        hedge_variance = np.var(hedge_returns)
        
        min_var_ratio = covariance / hedge_variance if hedge_variance > 0 else 0
        
        # 時変ヘッジ比率（GARCH based）
        # 簡易実装（実際にはGARCHモデルが必要）
        rolling_window = 60  # 60日ローリング
        
        if len(asset_returns) >= rolling_window:
            rolling_cov = np.array([
                np.cov(asset_returns[i:i+rolling_window], 
                      hedge_returns[i:i+rolling_window])[0, 1]
                for i in range(len(asset_returns) - rolling_window + 1)
            ])
            
            rolling_var = np.array([
                np.var(hedge_returns[i:i+rolling_window])
                for i in range(len(hedge_returns) - rolling_window + 1)
            ])
            
            time_varying_ratios = rolling_cov / rolling_var
            avg_time_varying = np.mean(time_varying_ratios)
        else:
            avg_time_varying = min_var_ratio
        
        return {
            'minimum_variance_ratio': min_var_ratio,
            'time_varying_average': avg_time_varying,
            'hedge_effectiveness': 1 - (np.var(asset_returns - min_var_ratio * hedge_returns) / np.var(asset_returns))
        }
    
    def enhanced_risk_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        """追加リスク指標の計算"""
        if len(returns) == 0:
            return {}
        
        # 基本統計量
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # VaR指標
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # Expected Shortfall (Conditional VaR)
        tail_returns_95 = returns[returns <= var_95]
        tail_returns_99 = returns[returns <= var_99]
        
        es_95 = np.mean(tail_returns_95) if len(tail_returns_95) > 0 else var_95
        es_99 = np.mean(tail_returns_99) if len(tail_returns_99) > 0 else var_99
        
        # 最大ドローダウン
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        # Sharpe比とSortino比（年率換算）
        sharpe_ratio = (mean_return / std_return * np.sqrt(252)) if std_return > 0 else 0
        
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else std_return
        sortino_ratio = (mean_return / downside_std * np.sqrt(252)) if downside_std > 0 else 0
        
        # Calmar比（年率リターン/最大ドローダウン）
        annual_return = mean_return * 252
        calmar_ratio = abs(annual_return / max_drawdown) if max_drawdown < 0 else 0
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'expected_shortfall_95': es_95,
            'expected_shortfall_99': es_99,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'mean_return': mean_return,
            'volatility': std_return
        }
    
    def sensitivity_analysis(self, base_prices: tuple, base_conditions: dict, engine=None) -> Dict[str, Dict]:
        """主要パラメータの感度分析"""
        if engine is None:
            # 外部からエンジンを渡されない場合のフォールバック
            print("Warning: No engine provided for sensitivity analysis")
            return {}
        
        # engine = BMAXIntegratedEngine()  # 外部から渡されたエンジンを使用
        base_result = engine.comprehensive_analysis(base_prices, base_conditions)
        base_cb_price = base_result['convertible_bond_analysis']['theoretical_price']
        
        sensitivities = {}
        
        # Bitcoin価格感度
        btc_shifts = [-0.3, -0.2, -0.1, 0.1, 0.2, 0.3]
        btc_sensitivity = {}
        
        for shift in btc_shifts:
            shifted_btc = base_prices[0] * (1 + shift)
            shifted_prices = (shifted_btc, base_prices[1], base_prices[2])
            
            try:
                result = engine.comprehensive_analysis(shifted_prices, base_conditions)
                new_price = result['convertible_bond_analysis']['theoretical_price']
                price_change = (new_price - base_cb_price) / base_cb_price
                btc_sensitivity[f'btc_{shift:+.0%}'] = price_change
            except Exception as e:
                btc_sensitivity[f'btc_{shift:+.0%}'] = np.nan
        
        sensitivities['bitcoin_price'] = btc_sensitivity
        
        # ボラティリティ感度
        vol_shifts = [-0.3, -0.2, -0.1, 0.1, 0.2, 0.3]
        vol_sensitivity = {}
        base_vol = base_conditions.get('volatility', 0.6)
        
        for shift in vol_shifts:
            shifted_vol = base_vol * (1 + shift)
            shifted_conditions = base_conditions.copy()
            shifted_conditions['volatility'] = shifted_vol
            
            try:
                result = engine.comprehensive_analysis(base_prices, shifted_conditions)
                new_price = result['convertible_bond_analysis']['theoretical_price']
                price_change = (new_price - base_cb_price) / base_cb_price
                vol_sensitivity[f'vol_{shift:+.0%}'] = price_change
            except Exception as e:
                vol_sensitivity[f'vol_{shift:+.0%}'] = np.nan
        
        sensitivities['volatility'] = vol_sensitivity
        
        # 株価感度
        stock_shifts = [-0.3, -0.2, -0.1, 0.1, 0.2, 0.3]
        stock_sensitivity = {}
        
        for shift in stock_shifts:
            shifted_stock = base_prices[1] * (1 + shift)
            shifted_prices = (base_prices[0], shifted_stock, base_prices[2])
            
            try:
                result = engine.comprehensive_analysis(shifted_prices, base_conditions)
                new_price = result['convertible_bond_analysis']['theoretical_price']
                price_change = (new_price - base_cb_price) / base_cb_price
                stock_sensitivity[f'stock_{shift:+.0%}'] = price_change
            except Exception as e:
                stock_sensitivity[f'stock_{shift:+.0%}'] = np.nan
        
        sensitivities['stock_price'] = stock_sensitivity
        
        return sensitivities

# =============================================================================
# 統合BMAX分析エンジン
# =============================================================================

class BMAXIntegratedEngine:
    """BMAX ETF統合分析エンジン"""
    
    def __init__(self):
        # 各種エンジンの初期化
        self.bs_engine = BitcoinBlackScholesEngine()
        self.compound_engine = CompoundOptionEngine(self.bs_engine)
        self.three_layer_model = ThreeLayerAssetModel(BitcoinMarketParams())
        self.liquidity_engine = ETFLiquidityEngine()
        self.risk_engine = BMXRiskEngine()
        
        # デフォルトパラメータ
        self.cb_params = ConvertibleBondParams()
        self.cb_engine = ConvertibleBondEngine(
            self.cb_params, self.bs_engine, self.compound_engine
        )
    
    def comprehensive_analysis(self, 
                             current_prices: Tuple[float, float, float],
                             market_conditions: Dict) -> Dict:
        """BMAX の包括的分析"""
        
        bitcoin_price, stock_price, cb_price = current_prices
        
        # 入力値検証
        ParameterValidator.validate_price(bitcoin_price, "Bitcoin price")
        ParameterValidator.validate_price(stock_price, "Stock price")
        ParameterValidator.validate_price(cb_price, "Convertible bond price")
        ParameterValidator.validate_bitcoin_stock_consistency(bitcoin_price, stock_price)
        
        volatility = market_conditions.get('volatility', 0.6)
        ParameterValidator.validate_volatility(volatility)
        volatility = market_conditions.get('volatility', 0.6)
        bitcoin_regime = market_conditions.get('regime', 'normal')
        
        # 1. 転換社債価格分析
        cb_analysis = {
            'theoretical_price': self.cb_engine.price(stock_price, volatility),
            'bond_floor': self.cb_engine.bond_floor(),
            'conversion_value': self.cb_engine.conversion_value(stock_price),
            'option_value': self.cb_engine.option_value(stock_price, volatility),
            'greeks': self.cb_engine.greeks(stock_price, volatility)
        }
        
        # 2. 三層相関分析
        price_paths = self.three_layer_model.simulate_price_paths(
            current_prices, time_horizon=1.0, n_simulations=1000
        )
        correlation_analysis = self.three_layer_model.correlation_analysis(price_paths)
        
        # 3. 流動性変換効果
        individual_liquidity = np.array([0.3, 0.4, 0.2, 0.5, 0.3])  # サンプル
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        
        liquidity_ratio = self.liquidity_engine.liquidity_transformation_ratio(
            individual_liquidity, weights
        )
        
        # 4. リスク分析
        returns = np.diff(np.log(price_paths[:, 2, :]), axis=1).flatten()  # CB returns
        var = self.risk_engine.bitcoin_aware_var(returns, bitcoin_regime)
        es = self.risk_engine.expected_shortfall(returns, var)
        tail_metrics = self.risk_engine.tail_risk_metrics(returns)
        enhanced_metrics = self.risk_engine.enhanced_risk_metrics(returns)
        
        # 5. 統合結果
        integrated_results = {
            'convertible_bond_analysis': cb_analysis,
            'correlation_structure': correlation_analysis,
            'liquidity_transformation_ratio': liquidity_ratio,
            'risk_metrics': {
                'bitcoin_aware_var': var,
                'expected_shortfall': es,
                'tail_risk': tail_metrics,
                'enhanced_metrics': enhanced_metrics
            },
            'portfolio_characteristics': {
                'diversification_benefit': correlation_analysis['bitcoin_cb'],
                'downside_protection': cb_analysis['bond_floor'] / cb_analysis['theoretical_price'],
                'upside_participation': cb_analysis['greeks']['delta']
            }
        }
        
        return integrated_results
    
    def scenario_analysis(self, scenarios: List[Dict]) -> pd.DataFrame:
        """シナリオ分析の実行"""
        
        results = []
        
        for scenario in scenarios:
            prices = scenario['prices']
            conditions = scenario['conditions']
            
            analysis = self.comprehensive_analysis(prices, conditions)
            
            scenario_result = {
                'scenario_name': scenario.get('name', 'Unnamed'),
                'cb_price': analysis['convertible_bond_analysis']['theoretical_price'],
                'var': analysis['risk_metrics']['bitcoin_aware_var'],
                'expected_shortfall': analysis['risk_metrics']['expected_shortfall'],
                'liquidity_ratio': analysis['liquidity_transformation_ratio'],
                'correlation_bitcoin_cb': analysis['correlation_structure']['bitcoin_cb']
            }
            
            results.append(scenario_result)
        
        return pd.DataFrame(results)
    
    def perform_sensitivity_analysis(self, base_prices: tuple, base_conditions: dict) -> Dict[str, Dict]:
        """感度分析の実行"""
        return self.risk_engine.sensitivity_analysis(base_prices, base_conditions, self)

# =============================================================================
# 使用例とテスト
# =============================================================================

def run_bmax_analysis():
    """BMAX分析の実行例"""
    
    print("🚀 BMAX REX 転換社債ETF 包括的分析システム")
    print("=" * 60)
    
    # エンジン初期化
    bmax_engine = BMAXIntegratedEngine()
    
    # 現在価格設定（サンプル）
    current_prices = (45000.0, 150.0, 1050.0)  # Bitcoin, Stock, CB
    
    # 市場条件
    market_conditions = {
        'volatility': 0.65,
        'regime': 'normal'
    }
    
    # 包括的分析実行
    results = bmax_engine.comprehensive_analysis(current_prices, market_conditions)
    
    # 結果表示
    print(f"\n📊 転換社債分析:")
    cb_analysis = results['convertible_bond_analysis']
    print(f"  理論価格: ${cb_analysis['theoretical_price']:.2f}")
    print(f"  債券フロア: ${cb_analysis['bond_floor']:.2f}")
    print(f"  転換価値: ${cb_analysis['conversion_value']:.2f}")
    print(f"  オプション価値: ${cb_analysis['option_value']:.2f}")
    
    print(f"\n🔗 相関構造分析:")
    corr = results['correlation_structure']
    print(f"  Bitcoin-CB相関: {corr['bitcoin_cb']:.3f}")
    print(f"  Stock-CB相関: {corr['stock_cb']:.3f}")
    
    print(f"\n⚡ 流動性変換効果:")
    print(f"  変換比率: {results['liquidity_transformation_ratio']:.3f}")
    
    print(f"\n⚠️ リスク指標:")
    risk = results['risk_metrics']
    print(f"  Bitcoin-Aware VaR: {risk['bitcoin_aware_var']:.4f}")
    print(f"  Expected Shortfall: {risk['expected_shortfall']:.4f}")
    
    print(f"\n🎯 ポートフォリオ特性:")
    portfolio = results['portfolio_characteristics']
    print(f"  多様化便益: {portfolio['diversification_benefit']:.3f}")
    print(f"  下方保護: {portfolio['downside_protection']:.3f}")
    print(f"  上昇参加: {portfolio['upside_participation']:.3f}")
    
    print(f"\n✅ BMAX統合分析完了")

if __name__ == "__main__":
    run_bmax_analysis()