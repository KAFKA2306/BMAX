"""BMAX転換社債ETFの研究用計算コア。

このモジュールは実市場価格を再現する校正済みモデルではない。入力された仮定を
一貫した数式で計算し、非有限値、再現不能な乱数、債券フロア割れ、未観測の流動性
倍率などを黙って受け入れないことを目的とする。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import ndtr
from scipy.stats import multivariate_normal


class ParameterValidator:
    @staticmethod
    def _finite_number(value: float, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise TypeError(f"{name} must be a number")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{name} must be finite")
        return result

    @classmethod
    def validate_price(cls, price: float, name: str = "price") -> float:
        result = cls._finite_number(price, name)
        if result <= 0:
            raise ValueError(f"{name} must be positive, got {result}")
        return result

    @classmethod
    def validate_volatility(cls, volatility: float) -> float:
        result = cls._finite_number(volatility, "volatility")
        if result < 0:
            raise ValueError("volatility must be non-negative")
        if result > 3:
            raise ValueError("volatility above 300% requires an explicit model extension")
        return result

    @classmethod
    def validate_correlation(cls, correlation: float, name: str = "correlation") -> float:
        result = cls._finite_number(correlation, name)
        if not -1 <= result <= 1:
            raise ValueError(f"{name} must be between -1 and 1")
        return result

    @classmethod
    def validate_time_to_maturity(cls, time: float) -> float:
        result = cls._finite_number(time, "time_to_maturity")
        if result < 0 or result > 50:
            raise ValueError("time_to_maturity must be between 0 and 50 years")
        return result

    @classmethod
    def validate_bitcoin_stock_consistency(
        cls, bitcoin_price: float, stock_price: float
    ) -> None:
        """価格単位が異なるため、価格水準同士の比率判定は行わない。"""
        cls.validate_price(bitcoin_price, "Bitcoin price")
        cls.validate_price(stock_price, "Stock price")


@dataclass(frozen=True)
class ConvertibleBondParams:
    face_value: float = 1_000.0
    conversion_ratio: float = 10.0
    conversion_price: float = 100.0
    coupon_rate: float = 0.0
    maturity: float = 5.0
    credit_spread: float = 0.02
    call_protection: float = 2.0

    def __post_init__(self) -> None:
        ParameterValidator.validate_price(self.face_value, "face_value")
        ParameterValidator.validate_price(self.conversion_ratio, "conversion_ratio")
        ParameterValidator.validate_price(self.conversion_price, "conversion_price")
        ParameterValidator.validate_time_to_maturity(self.maturity)
        if self.maturity == 0:
            raise ValueError("convertible-bond maturity must be positive")
        if not 0 <= self.coupon_rate <= 1:
            raise ValueError("coupon_rate must be in [0, 1]")
        if self.credit_spread < 0 or not math.isfinite(self.credit_spread):
            raise ValueError("credit_spread must be finite and non-negative")
        if not 0 <= self.call_protection <= self.maturity:
            raise ValueError("call_protection must be between 0 and maturity")


@dataclass(frozen=True)
class BitcoinMarketParams:
    current_price: float = 45_000.0
    drift: float = 0.15
    volatility: float = 0.80
    jump_intensity: float = 0.10
    jump_mean: float = 0.0
    jump_std: float = 0.15

    def __post_init__(self) -> None:
        ParameterValidator.validate_price(self.current_price, "bitcoin_current_price")
        ParameterValidator.validate_volatility(self.volatility)
        if self.jump_intensity < 0 or self.jump_std < 0:
            raise ValueError("jump_intensity and jump_std must be non-negative")
        for value, name in ((self.drift, "drift"), (self.jump_mean, "jump_mean")):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class MarketRegime:
    regime_name: str
    transition_matrix: np.ndarray
    volatility_multiplier: float
    correlation_modifier: float


class BitcoinBlackScholesEngine:
    """配当利回りを受け取る欧州コール価格エンジン。"""

    def __init__(self, risk_free_rate: float = 0.045):
        if not math.isfinite(risk_free_rate):
            raise ValueError("risk_free_rate must be finite")
        self.risk_free_rate = float(risk_free_rate)

    def d1(self, S: float, K: float, T: float, sigma: float, q: float = 0.0) -> float:
        if T <= 0 or sigma <= 0:
            raise ValueError("d1 is undefined when T or sigma is zero")
        return (
            math.log(S / K)
            + (self.risk_free_rate - q + 0.5 * sigma**2) * T
        ) / (sigma * math.sqrt(T))

    def d2(self, S: float, K: float, T: float, sigma: float, q: float = 0.0) -> float:
        return self.d1(S, K, T, sigma, q) - sigma * math.sqrt(T)

    @lru_cache(maxsize=1_000)
    def _cached_call_price_core(
        self, S: float, K: float, T: float, sigma: float, q: float = 0.0
    ) -> float:
        if T == 0:
            return max(S - K, 0.0)
        if sigma == 0:
            return max(
                S * math.exp(-q * T)
                - K * math.exp(-self.risk_free_rate * T),
                0.0,
            )
        d1_value = self.d1(S, K, T, sigma, q)
        d2_value = d1_value - sigma * math.sqrt(T)
        return max(
            S * math.exp(-q * T) * ndtr(d1_value)
            - K * math.exp(-self.risk_free_rate * T) * ndtr(d2_value),
            0.0,
        )

    def call_price(
        self, S: float, K: float, T: float, sigma: float, q: float = 0.0
    ) -> float:
        S = ParameterValidator.validate_price(S, "stock_price")
        K = ParameterValidator.validate_price(K, "strike_price")
        T = ParameterValidator.validate_time_to_maturity(T)
        sigma = ParameterValidator.validate_volatility(sigma)
        if not math.isfinite(q):
            raise ValueError("dividend_yield must be finite")
        return self._cached_call_price_core(S, K, T, sigma, float(q))

    def delta(
        self, S: float, K: float, T: float, sigma: float, q: float = 0.0
    ) -> float:
        self.call_price(S, K, T, sigma, q)
        if T == 0:
            return 1.0 if S > K else 0.0
        if sigma == 0:
            forward_intrinsic = S * math.exp(-q * T) > K * math.exp(
                -self.risk_free_rate * T
            )
            return math.exp(-q * T) if forward_intrinsic else 0.0
        return math.exp(-q * T) * ndtr(self.d1(S, K, T, sigma, q))

    def gamma(
        self, S: float, K: float, T: float, sigma: float, q: float = 0.0
    ) -> float:
        self.call_price(S, K, T, sigma, q)
        if T == 0 or sigma == 0:
            return 0.0
        return (
            math.exp(-q * T)
            * stats.norm.pdf(self.d1(S, K, T, sigma, q))
            / (S * sigma * math.sqrt(T))
        )

    def vega(
        self, S: float, K: float, T: float, sigma: float, q: float = 0.0
    ) -> float:
        self.call_price(S, K, T, sigma, q)
        if T == 0 or sigma == 0:
            return 0.0
        return (
            S
            * math.exp(-q * T)
            * stats.norm.pdf(self.d1(S, K, T, sigma, q))
            * math.sqrt(T)
            / 100
        )


class ThreeLayerAssetModel:
    """三資産の相関付きシナリオ生成器。価格決定モデルではない。"""

    def __init__(
        self,
        bitcoin_params: BitcoinMarketParams,
        stock_correlation: float = 0.7,
        cb_correlation: float = 0.5,
    ):
        self.bitcoin_params = bitcoin_params
        self.stock_correlation = ParameterValidator.validate_correlation(
            stock_correlation, "bitcoin_stock_correlation"
        )
        self.cb_correlation = ParameterValidator.validate_correlation(
            cb_correlation, "stock_cb_correlation"
        )
        self.correlation_matrix = self._build_correlation_matrix()

    def _build_correlation_matrix(self) -> np.ndarray:
        rho_bs = self.stock_correlation
        rho_sc = self.cb_correlation
        rho_bc = rho_bs * rho_sc
        matrix = np.array(
            [
                [1.0, rho_bs, rho_bc],
                [rho_bs, 1.0, rho_sc],
                [rho_bc, rho_sc, 1.0],
            ],
            dtype=float,
        )
        eigenvalues = np.linalg.eigvalsh(matrix)
        if float(eigenvalues.min()) < -1e-10:
            raise ValueError("correlation matrix is not positive semidefinite")
        if not np.allclose(np.diag(matrix), 1.0):
            raise ValueError("correlation matrix diagonal must remain one")
        return matrix

    def simulate_price_paths(
        self,
        initial_prices: Tuple[float, float, float],
        time_horizon: float = 1.0,
        n_steps: int = 252,
        n_simulations: int = 10_000,
        use_parallel: bool = False,
        *,
        seed: int | None = 0,
        stock_drift: float = 0.12,
        stock_volatility: float = 0.60,
        cb_drift: float = 0.08,
        cb_volatility: float = 0.40,
    ) -> np.ndarray:
        del use_parallel  # 再現性を壊す暗黙fork乱数を使用しない。
        prices = np.asarray(initial_prices, dtype=float)
        if prices.shape != (3,) or not np.isfinite(prices).all() or np.any(prices <= 0):
            raise ValueError("initial_prices must contain three finite positive prices")
        if time_horizon <= 0 or not math.isfinite(time_horizon):
            raise ValueError("time_horizon must be finite and positive")
        if not isinstance(n_steps, int) or n_steps < 1:
            raise ValueError("n_steps must be a positive integer")
        if not isinstance(n_simulations, int) or n_simulations < 1:
            raise ValueError("n_simulations must be a positive integer")
        stock_volatility = ParameterValidator.validate_volatility(stock_volatility)
        cb_volatility = ParameterValidator.validate_volatility(cb_volatility)

        rng = np.random.default_rng(seed)
        dt = time_horizon / n_steps
        eigenvalues, eigenvectors = np.linalg.eigh(self.correlation_matrix)
        transform = eigenvectors @ np.diag(np.sqrt(np.clip(eigenvalues, 0, None)))
        independent = rng.standard_normal((n_simulations, n_steps, 3))
        shocks = independent @ transform.T

        volatilities = np.array(
            [self.bitcoin_params.volatility, stock_volatility, cb_volatility]
        )
        drifts = np.array([self.bitcoin_params.drift, stock_drift, cb_drift])
        log_returns = (
            (drifts - 0.5 * volatilities**2) * dt
            + shocks * volatilities * math.sqrt(dt)
        )

        if self.bitcoin_params.jump_intensity > 0:
            counts = rng.poisson(
                self.bitcoin_params.jump_intensity * dt,
                size=(n_simulations, n_steps),
            )
            jump_noise = rng.standard_normal((n_simulations, n_steps))
            jump_sum = (
                counts * self.bitcoin_params.jump_mean
                + np.sqrt(counts) * self.bitcoin_params.jump_std * jump_noise
            )
            log_returns[:, :, 0] += jump_sum

        cumulative = np.cumsum(log_returns, axis=1)
        paths = np.empty((n_simulations, 3, n_steps + 1), dtype=float)
        paths[:, :, 0] = prices
        paths[:, :, 1:] = prices[None, :, None] * np.exp(
            cumulative.transpose(0, 2, 1)
        )
        if not np.isfinite(paths).all() or np.any(paths <= 0):
            raise FloatingPointError("simulation produced invalid prices")
        return paths

    def correlation_analysis(self, price_paths: np.ndarray) -> Dict[str, float | np.ndarray]:
        paths = np.asarray(price_paths, dtype=float)
        if paths.ndim != 3 or paths.shape[1] != 3 or paths.shape[2] < 2:
            raise ValueError("price_paths must have shape [simulations, 3, steps+1]")
        if not np.isfinite(paths).all() or np.any(paths <= 0):
            raise ValueError("price_paths must be finite and positive")
        returns = np.diff(np.log(paths), axis=2)
        samples = returns.transpose(0, 2, 1).reshape(-1, 3)
        if np.any(np.std(samples, axis=0) == 0):
            raise ValueError("realized correlation is undefined for zero-variance assets")
        matrix = np.corrcoef(samples, rowvar=False)
        return {
            "bitcoin_stock": float(matrix[0, 1]),
            "bitcoin_cb": float(matrix[0, 2]),
            "stock_cb": float(matrix[1, 2]),
            "correlation_matrix": matrix,
        }


class CompoundOptionEngine:
    """標準的な同一原資産上の欧州call-on-call（Geske型）近似。"""

    def __init__(self, bs_engine: BitcoinBlackScholesEngine):
        self.bs_engine = bs_engine

    def compound_call_price(
        self,
        S: float,
        K1: float,
        K2: float,
        T1: float,
        T2: float,
        sigma: float,
        correlation: float | None = None,
    ) -> float:
        S = ParameterValidator.validate_price(S, "underlying_price")
        K1 = ParameterValidator.validate_price(K1, "compound_strike")
        K2 = ParameterValidator.validate_price(K2, "underlying_strike")
        T1 = ParameterValidator.validate_time_to_maturity(T1)
        T2 = ParameterValidator.validate_time_to_maturity(T2)
        sigma = ParameterValidator.validate_volatility(sigma)
        if T2 <= T1:
            raise ValueError("T2 must be greater than T1 for a call-on-call")
        if T1 == 0:
            return max(self.bs_engine.call_price(S, K2, T2, sigma) - K1, 0.0)
        if sigma == 0:
            return max(self.bs_engine.call_price(S, K2, T2, sigma) - K1, 0.0)

        rho = math.sqrt(T1 / T2)
        if correlation is not None and not math.isclose(correlation, rho, abs_tol=1e-8):
            raise ValueError(
                "For the standard same-underlying compound option, correlation is "
                "sqrt(T1/T2) and cannot be supplied independently."
            )

        def equation(critical_price: float) -> float:
            return (
                self.bs_engine.call_price(
                    critical_price, K2, T2 - T1, sigma
                )
                - K1
            )

        lower = 1e-12
        upper = max(S, K1 + K2, 1.0)
        for _ in range(100):
            if equation(upper) > 0:
                break
            upper *= 2
        else:
            raise RuntimeError("failed to bracket the compound-option critical price")
        critical = optimize.brentq(equation, lower, upper, maxiter=500)

        sigma_t1 = sigma * math.sqrt(T1)
        sigma_t2 = sigma * math.sqrt(T2)
        a1 = (
            math.log(S / critical)
            + (self.bs_engine.risk_free_rate + 0.5 * sigma**2) * T1
        ) / sigma_t1
        a2 = a1 - sigma_t1
        b1 = (
            math.log(S / K2)
            + (self.bs_engine.risk_free_rate + 0.5 * sigma**2) * T2
        ) / sigma_t2
        b2 = b1 - sigma_t2
        covariance = [[1.0, rho], [rho, 1.0]]
        price = (
            S * multivariate_normal.cdf([a1, b1], cov=covariance)
            - K2
            * math.exp(-self.bs_engine.risk_free_rate * T2)
            * multivariate_normal.cdf([a2, b2], cov=covariance)
            - K1
            * math.exp(-self.bs_engine.risk_free_rate * T1)
            * ndtr(a2)
        )
        return max(float(price), 0.0)

    def price(self, *args, **kwargs) -> float:
        return self.compound_call_price(*args, **kwargs)

    def greeks(
        self,
        S: float,
        K1: float,
        K2: float,
        T1: float,
        T2: float,
        sigma: float,
        **kwargs,
    ) -> Dict[str, float]:
        step = max(1e-4, S * 1e-4)
        base = self.price(S, K1, K2, T1, T2, sigma, **kwargs)
        up = self.price(S + step, K1, K2, T1, T2, sigma, **kwargs)
        down = self.price(S - step, K1, K2, T1, T2, sigma, **kwargs)
        volatility_step = 1e-4
        return {
            "delta": (up - down) / (2 * step),
            "gamma": (up - 2 * base + down) / step**2,
            "vega": (
                self.price(
                    S, K1, K2, T1, T2, sigma + volatility_step, **kwargs
                )
                - base
            )
            / volatility_step,
        }


class ConvertibleBondEngine:
    """債券フロアと欧州転換オプションを分離した研究用近似。"""

    def __init__(
        self,
        cb_params: ConvertibleBondParams,
        bs_engine: BitcoinBlackScholesEngine,
        compound_engine: CompoundOptionEngine | None = None,
    ):
        self.cb_params = cb_params
        self.bs_engine = bs_engine
        self.compound_engine = compound_engine

    def bond_floor(self, credit_spread: float | None = None) -> float:
        spread = self.cb_params.credit_spread if credit_spread is None else float(credit_spread)
        if not math.isfinite(spread) or spread < 0:
            raise ValueError("credit_spread must be finite and non-negative")
        discount_rate = self.bs_engine.risk_free_rate + spread
        maturity = self.cb_params.maturity
        principal = self.cb_params.face_value * math.exp(-discount_rate * maturity)
        annual_coupon = self.cb_params.face_value * self.cb_params.coupon_rate
        if annual_coupon == 0:
            return principal
        if abs(discount_rate) < 1e-12:
            coupon_value = annual_coupon * maturity
        else:
            coupon_value = annual_coupon * (
                1 - math.exp(-discount_rate * maturity)
            ) / discount_rate
        return principal + coupon_value

    def conversion_value(self, stock_price: float) -> float:
        stock_price = ParameterValidator.validate_price(stock_price, "stock_price")
        return self.cb_params.conversion_ratio * stock_price

    def conversion_premium(self, cb_price: float, stock_price: float) -> float:
        cb_price = ParameterValidator.validate_price(cb_price, "cb_price")
        conversion_value = self.conversion_value(stock_price)
        return (cb_price - conversion_value) / conversion_value

    def option_value(self, stock_price: float, volatility: float) -> float:
        return self.bs_engine.call_price(
            S=stock_price,
            K=self.cb_params.conversion_price,
            T=self.cb_params.maturity,
            sigma=volatility,
        ) * self.cb_params.conversion_ratio

    def price(
        self,
        stock_price: float,
        volatility: float,
        credit_spread: float | None = None,
    ) -> float:
        stock_price = ParameterValidator.validate_price(stock_price, "stock_price")
        volatility = ParameterValidator.validate_volatility(volatility)
        floor = self.bond_floor(credit_spread)
        conversion = self.conversion_value(stock_price)
        theoretical = floor + self.option_value(stock_price, volatility)
        value = max(floor, conversion, theoretical)
        if not math.isfinite(value):
            raise FloatingPointError("convertible-bond price is non-finite")
        return value

    def greeks(self, stock_price: float, volatility: float, **kwargs) -> Dict[str, float]:
        step = max(1e-4, stock_price * 1e-4)
        base = self.price(stock_price, volatility, **kwargs)
        up = self.price(stock_price + step, volatility, **kwargs)
        down = self.price(stock_price - step, volatility, **kwargs)
        volatility_step = 1e-4
        delta = (up - down) / (2 * step)
        gamma = (up - 2 * base + down) / step**2
        vega = (
            self.price(stock_price, volatility + volatility_step, **kwargs) - base
        ) / volatility_step
        spread_step = 1e-4
        base_spread = self.cb_params.credit_spread
        credit_sensitivity = (
            self.price(
                stock_price,
                volatility,
                credit_spread=base_spread + spread_step,
            )
            - base
        ) / spread_step
        return {
            "delta": float(delta),
            "gamma": float(gamma),
            "vega": float(vega),
            "credit_sensitivity": float(credit_sensitivity),
            "conversion_ratio": self.cb_params.conversion_ratio,
        }


class ETFLiquidityEngine:
    """観測済みETF流動性と構成銘柄流動性を比較する計測器。"""

    def __init__(self, creation_unit_size: int = 50_000, transaction_costs: float = 0.001):
        if creation_unit_size <= 0:
            raise ValueError("creation_unit_size must be positive")
        if transaction_costs < 0 or not math.isfinite(transaction_costs):
            raise ValueError("transaction_costs must be finite and non-negative")
        self.creation_unit_size = int(creation_unit_size)
        self.transaction_costs = float(transaction_costs)

    def liquidity_transformation_ratio(
        self,
        individual_liquidity: np.ndarray,
        weights: np.ndarray,
        observed_etf_liquidity: float | None = None,
    ) -> float:
        liquidity = np.asarray(individual_liquidity, dtype=float)
        portfolio_weights = np.asarray(weights, dtype=float)
        if liquidity.ndim != 1 or portfolio_weights.shape != liquidity.shape:
            raise ValueError("liquidity and weights must be one-dimensional and aligned")
        if not np.isfinite(liquidity).all() or np.any(liquidity < 0):
            raise ValueError("individual liquidity must be finite and non-negative")
        if not np.isfinite(portfolio_weights).all() or np.any(portfolio_weights < 0):
            raise ValueError("weights must be finite and non-negative")
        if not math.isclose(float(portfolio_weights.sum()), 1.0, abs_tol=1e-8):
            raise ValueError("weights must sum to one")
        weighted = float(np.dot(portfolio_weights, liquidity))
        if weighted <= 0:
            raise ValueError("weighted constituent liquidity must be positive")
        if observed_etf_liquidity is None:
            raise ValueError(
                "observed_etf_liquidity is required; a transformation multiplier "
                "cannot be inferred from constituent liquidity alone"
            )
        observed = ParameterValidator.validate_price(
            observed_etf_liquidity, "observed_etf_liquidity"
        )
        return observed / weighted

    def premium_discount_dynamics(
        self, nav: float, market_price: float, trading_volume: float
    ) -> Dict[str, float]:
        nav = ParameterValidator.validate_price(nav, "nav")
        market_price = ParameterValidator.validate_price(market_price, "market_price")
        if trading_volume < 0 or not math.isfinite(trading_volume):
            raise ValueError("trading_volume must be finite and non-negative")
        premium_discount = (market_price - nav) / nav
        return {
            "premium_discount": premium_discount,
            "arbitrage_cost_band": self.transaction_costs * 2,
            "trading_volume": float(trading_volume),
            "model_status": "descriptive_only",
        }


class BMXRiskEngine:
    """損失を正値で返す履歴分位ベースのリスク計測器。"""

    def __init__(self, confidence_level: float = 0.05):
        if not 0 < confidence_level < 0.5:
            raise ValueError("confidence_level must be in (0, 0.5)")
        self.confidence_level = float(confidence_level)

    @staticmethod
    def _returns(values: np.ndarray) -> np.ndarray:
        returns = np.asarray(values, dtype=float).reshape(-1)
        if returns.size < 2 or not np.isfinite(returns).all():
            raise ValueError("returns must contain at least two finite observations")
        if np.any(returns <= -1):
            raise ValueError("simple returns cannot be less than or equal to -100%")
        return returns

    def bitcoin_aware_var(self, returns: np.ndarray, bitcoin_regime: str = "normal") -> float:
        values = self._returns(returns)
        adjustments = {"bull": 0.8, "normal": 1.0, "bear": 1.3, "crisis": 1.8}
        if bitcoin_regime not in adjustments:
            raise ValueError(f"unknown bitcoin_regime: {bitcoin_regime}")
        historical_loss = max(0.0, -float(np.quantile(values, self.confidence_level)))
        return historical_loss * adjustments[bitcoin_regime]

    def expected_shortfall(self, returns: np.ndarray, var: float) -> float:
        values = self._returns(returns)
        if var < 0 or not math.isfinite(var):
            raise ValueError("var must be a finite non-negative loss")
        tail = values[values <= -var]
        return var if tail.size == 0 else max(0.0, -float(tail.mean()))

    def tail_risk_metrics(self, returns: np.ndarray) -> Dict[str, float | int | None]:
        values = self._returns(returns)
        threshold = float(np.quantile(values, self.confidence_level))
        losses = -values[values <= threshold]
        positive_losses = np.sort(losses[losses > 0])[::-1]
        hill: float | None = None
        if positive_losses.size >= 8:
            k = max(4, positive_losses.size // 4)
            reference = positive_losses[k - 1]
            if reference > 0:
                hill = float(np.mean(np.log(positive_losses[:k] / reference)))
        return {
            "skewness": float(stats.skew(values)),
            "excess_kurtosis": float(stats.kurtosis(values)),
            "tail_threshold_return": threshold,
            "tail_observations": int(losses.size),
            "hill_tail_index": hill,
        }

    def optimal_hedge_ratio(
        self, asset_returns: np.ndarray, hedge_returns: np.ndarray
    ) -> Dict[str, float]:
        asset = self._returns(asset_returns)
        hedge = self._returns(hedge_returns)
        if asset.shape != hedge.shape:
            raise ValueError("asset_returns and hedge_returns must be aligned")
        asset_variance = float(np.var(asset))
        hedge_variance = float(np.var(hedge))
        if asset_variance <= 0 or hedge_variance <= 0:
            raise ValueError("hedge ratio is undefined for zero-variance series")
        covariance = float(np.cov(asset, hedge, ddof=0)[0, 1])
        ratio = covariance / hedge_variance
        hedged_variance = float(np.var(asset - ratio * hedge))
        return {
            "minimum_variance_ratio": ratio,
            "hedge_effectiveness": 1 - hedged_variance / asset_variance,
        }

    def enhanced_risk_metrics(self, returns: np.ndarray) -> Dict[str, float]:
        values = self._returns(returns)
        mean_return = float(np.mean(values))
        volatility = float(np.std(values))
        var_95 = max(0.0, -float(np.quantile(values, 0.05)))
        var_99 = max(0.0, -float(np.quantile(values, 0.01)))
        es_95 = self.expected_shortfall(values, var_95)
        es_99 = self.expected_shortfall(values, var_99)
        cumulative = np.cumprod(1 + values)
        running_max = np.maximum.accumulate(cumulative)
        max_drawdown = float(np.min(cumulative / running_max - 1))
        sharpe = mean_return / volatility * math.sqrt(252) if volatility > 0 else 0.0
        downside = values[values < 0]
        downside_std = float(np.std(downside)) if downside.size else 0.0
        sortino = mean_return / downside_std * math.sqrt(252) if downside_std > 0 else 0.0
        annual_return = mean_return * 252
        calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0.0
        return {
            "var_95_loss": var_95,
            "var_99_loss": var_99,
            "expected_shortfall_95_loss": es_95,
            "expected_shortfall_99_loss": es_99,
            "max_drawdown": max_drawdown,
            "sharpe_ratio_zero_rate": sharpe,
            "sortino_ratio_zero_target": sortino,
            "calmar_ratio_arithmetic_return": calmar,
            "mean_daily_return": mean_return,
            "daily_volatility": volatility,
        }

    def sensitivity_analysis(
        self, base_prices: tuple, base_conditions: dict, engine=None
    ) -> Dict[str, Dict[str, float]]:
        if engine is None:
            raise ValueError("engine is required for sensitivity analysis")
        base_result = engine.comprehensive_analysis(base_prices, base_conditions)
        base_price = base_result["convertible_bond_analysis"]["theoretical_price"]
        if base_price <= 0:
            raise ValueError("base theoretical price must be positive")

        results: Dict[str, Dict[str, float]] = {}
        for label, index in (("bitcoin_price", 0), ("stock_price", 1)):
            series: Dict[str, float] = {}
            for shift in (-0.3, -0.2, -0.1, 0.1, 0.2, 0.3):
                shifted = list(base_prices)
                shifted[index] *= 1 + shift
                value = engine.comprehensive_analysis(tuple(shifted), base_conditions)[
                    "convertible_bond_analysis"
                ]["theoretical_price"]
                series[f"{shift:+.0%}"] = (value - base_price) / base_price
            results[label] = series

        volatility_series: Dict[str, float] = {}
        base_volatility = float(base_conditions.get("volatility", 0.6))
        for shift in (-0.3, -0.2, -0.1, 0.1, 0.2, 0.3):
            conditions = dict(base_conditions)
            conditions["volatility"] = base_volatility * (1 + shift)
            value = engine.comprehensive_analysis(base_prices, conditions)[
                "convertible_bond_analysis"
            ]["theoretical_price"]
            volatility_series[f"{shift:+.0%}"] = (value - base_price) / base_price
        results["volatility"] = volatility_series
        return results


class BMAXIntegratedEngine:
    """仮定を明示して価格近似とシナリオ統計をまとめる。"""

    def __init__(self):
        self.bs_engine = BitcoinBlackScholesEngine()
        self.compound_engine = CompoundOptionEngine(self.bs_engine)
        self.three_layer_model = ThreeLayerAssetModel(BitcoinMarketParams())
        self.liquidity_engine = ETFLiquidityEngine()
        self.risk_engine = BMXRiskEngine()
        self.cb_params = ConvertibleBondParams()
        self.cb_engine = ConvertibleBondEngine(
            self.cb_params, self.bs_engine, self.compound_engine
        )

    def comprehensive_analysis(
        self,
        current_prices: Tuple[float, float, float],
        market_conditions: Dict,
    ) -> Dict:
        if len(current_prices) != 3:
            raise ValueError("current_prices must be (bitcoin, stock, convertible_bond)")
        bitcoin_price, stock_price, cb_market_price = current_prices
        ParameterValidator.validate_bitcoin_stock_consistency(
            bitcoin_price, stock_price
        )
        ParameterValidator.validate_price(cb_market_price, "convertible_bond_market_price")
        volatility = ParameterValidator.validate_volatility(
            market_conditions.get("volatility", 0.6)
        )
        regime = market_conditions.get("regime", "normal")
        simulations = int(market_conditions.get("n_simulations", 1_000))
        steps = int(market_conditions.get("n_steps", 252))
        seed = market_conditions.get("seed", 0)

        theoretical_price = self.cb_engine.price(stock_price, volatility)
        bond_floor = self.cb_engine.bond_floor()
        conversion_value = self.cb_engine.conversion_value(stock_price)
        option_value = self.cb_engine.option_value(stock_price, volatility)
        cb_analysis = {
            "market_price_input": float(cb_market_price),
            "theoretical_price": theoretical_price,
            "model_price_gap": float(cb_market_price - theoretical_price),
            "bond_floor": bond_floor,
            "conversion_value": conversion_value,
            "option_value": option_value,
            "greeks": self.cb_engine.greeks(stock_price, volatility),
            "pricing_model": "bond_floor_plus_european_conversion_option_approximation",
            "calibrated_to_market": False,
        }

        paths = self.three_layer_model.simulate_price_paths(
            current_prices,
            n_steps=steps,
            n_simulations=simulations,
            seed=seed,
            stock_volatility=float(
                market_conditions.get("stock_path_volatility", 0.60)
            ),
            cb_volatility=float(market_conditions.get("cb_path_volatility", 0.40)),
        )
        correlations = self.three_layer_model.correlation_analysis(paths)
        cb_log_returns = np.diff(np.log(paths[:, 2, :]), axis=1).reshape(-1)
        var_loss = self.risk_engine.bitcoin_aware_var(cb_log_returns, regime)
        expected_shortfall_loss = self.risk_engine.expected_shortfall(
            cb_log_returns, var_loss
        )

        liquidity_ratio = None
        liquidity_status = "not_estimated_without_observed_etf_liquidity"
        if all(
            key in market_conditions
            for key in (
                "individual_liquidity",
                "liquidity_weights",
                "observed_etf_liquidity",
            )
        ):
            liquidity_ratio = self.liquidity_engine.liquidity_transformation_ratio(
                np.asarray(market_conditions["individual_liquidity"], dtype=float),
                np.asarray(market_conditions["liquidity_weights"], dtype=float),
                float(market_conditions["observed_etf_liquidity"]),
            )
            liquidity_status = "measured_ratio"

        bitcoin_cb_correlation = float(correlations["bitcoin_cb"])
        return {
            "convertible_bond_analysis": cb_analysis,
            "correlation_structure": correlations,
            "liquidity_transformation_ratio": liquidity_ratio,
            "liquidity_model_status": liquidity_status,
            "risk_metrics": {
                "bitcoin_aware_var_loss": var_loss,
                "expected_shortfall_loss": expected_shortfall_loss,
                "tail_risk": self.risk_engine.tail_risk_metrics(cb_log_returns),
                "enhanced_metrics": self.risk_engine.enhanced_risk_metrics(
                    cb_log_returns
                ),
            },
            "portfolio_characteristics": {
                "bitcoin_cb_correlation": bitcoin_cb_correlation,
                "diversification_proxy_one_minus_absolute_correlation": (
                    1 - abs(bitcoin_cb_correlation)
                ),
                "bond_floor_share": min(1.0, bond_floor / theoretical_price),
                "conversion_delta": cb_analysis["greeks"]["delta"],
            },
            "simulation_assumptions": {
                "seed": seed,
                "n_simulations": simulations,
                "n_steps": steps,
                "scenario_not_forecast": True,
            },
        }

    def scenario_analysis(self, scenarios: List[Dict]) -> pd.DataFrame:
        rows = []
        for scenario in scenarios:
            analysis = self.comprehensive_analysis(
                scenario["prices"], scenario["conditions"]
            )
            rows.append(
                {
                    "scenario_name": scenario.get("name", "Unnamed"),
                    "cb_theoretical_price": analysis["convertible_bond_analysis"][
                        "theoretical_price"
                    ],
                    "var_loss": analysis["risk_metrics"][
                        "bitcoin_aware_var_loss"
                    ],
                    "expected_shortfall_loss": analysis["risk_metrics"][
                        "expected_shortfall_loss"
                    ],
                    "liquidity_ratio": analysis[
                        "liquidity_transformation_ratio"
                    ],
                    "bitcoin_cb_correlation": analysis["correlation_structure"][
                        "bitcoin_cb"
                    ],
                }
            )
        return pd.DataFrame(rows)

    def perform_sensitivity_analysis(
        self, base_prices: tuple, base_conditions: dict
    ) -> Dict[str, Dict[str, float]]:
        return self.risk_engine.sensitivity_analysis(
            base_prices, base_conditions, self
        )


def run_bmax_analysis() -> None:
    engine = BMAXIntegratedEngine()
    results = engine.comprehensive_analysis(
        (45_000.0, 150.0, 1_050.0),
        {"volatility": 0.65, "regime": "normal", "seed": 0},
    )
    cb = results["convertible_bond_analysis"]
    print("BMAX research scenario; not a market forecast")
    print(f"theoretical convertible-bond value: {cb['theoretical_price']:.2f}")
    print(f"bond floor: {cb['bond_floor']:.2f}")
    print(f"input market price: {cb['market_price_input']:.2f}")
    print(
        "5% scenario VaR loss: "
        f"{results['risk_metrics']['bitcoin_aware_var_loss']:.6f}"
    )


if __name__ == "__main__":
    run_bmax_analysis()
