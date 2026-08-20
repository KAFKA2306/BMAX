"""Build reproducible convertible-bond scenarios from contract + market evidence.

Observed values come only from a validated market snapshot. Missing market inputs are
never backfilled. Model assumptions are explicit and remain separate from observations.
"""
from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np

from .bmax_computational_framework import (
    BMXRiskEngine,
    BitcoinBlackScholesEngine,
    BitcoinMarketParams,
    ConvertibleBondEngine,
    ConvertibleBondParams,
    ThreeLayerAssetModel,
)

MODEL_VERSION = "bmax.convertible-scenario.v1"


def _years(start: str, end: str) -> float:
    return (date.fromisoformat(end) - date.fromisoformat(start)).days / 365.25


def build_scenario(
    issue: dict[str, Any],
    snapshot: dict[str, Any],
    assumptions: dict[str, Any],
    *,
    scenario_id: str,
) -> dict[str, Any]:
    if snapshot["issue_id"] != issue["id"]:
        raise ValueError("snapshot belongs to another issue")

    tenor = str(assumptions["risk_free_tenor"])
    curve = snapshot["risk_free_curve"]
    if tenor not in curve["points"]:
        raise ValueError(f"risk-free tenor {tenor!r} is not observed in snapshot")
    risk_free_rate = float(curve["points"][tenor])

    maturity = _years(snapshot["as_of"], issue["maturity_date"])
    if maturity <= 0:
        raise ValueError("scenario snapshot must precede maturity")

    call_terms = issue["protection_terms"]["call_terms"]
    call_protection = 0.0
    if call_terms["status"] == "PRESENT":
        earliest = call_terms.get("earliest_date")
        if earliest:
            call_protection = max(0.0, min(maturity, _years(snapshot["as_of"], earliest)))

    model = assumptions["model"]
    equity_volatility = float(model["equity_volatility"])
    credit_spread = float(model["credit_spread"])

    cb_params = ConvertibleBondParams(
        face_value=1_000.0,
        conversion_ratio=float(issue["conversion_terms"]["conversion_ratio"]),
        conversion_price=float(issue["conversion_terms"]["conversion_price"]),
        coupon_rate=float(issue["coupon"]),
        maturity=maturity,
        credit_spread=credit_spread,
        call_protection=call_protection,
    )
    bs_engine = BitcoinBlackScholesEngine(risk_free_rate=risk_free_rate)
    cb_engine = ConvertibleBondEngine(cb_params, bs_engine)
    stock_price = float(snapshot["equity"]["price"])

    bond_floor = cb_engine.bond_floor()
    conversion_value = cb_engine.conversion_value(stock_price)
    option_component = cb_engine.option_value(stock_price, equity_volatility)
    theoretical_price = cb_engine.price(stock_price, equity_volatility)

    risk = assumptions["risk_simulation"]
    normalized_level = float(risk["normalized_initial_level"])
    bitcoin_params = BitcoinMarketParams(
        current_price=normalized_level,
        drift=float(risk["bitcoin_drift"]),
        volatility=float(risk["bitcoin_volatility"]),
        jump_intensity=float(risk["bitcoin_jump_intensity"]),
        jump_mean=float(risk["bitcoin_jump_mean"]),
        jump_std=float(risk["bitcoin_jump_std"]),
    )
    path_model = ThreeLayerAssetModel(
        bitcoin_params,
        stock_correlation=float(risk["bitcoin_stock_correlation"]),
        cb_correlation=float(risk["stock_cb_correlation"]),
    )
    paths = path_model.simulate_price_paths(
        (normalized_level, normalized_level, normalized_level),
        time_horizon=float(risk["time_horizon_years"]),
        n_steps=int(risk["n_steps"]),
        n_simulations=int(risk["n_simulations"]),
        seed=int(risk["seed"]),
        stock_drift=float(risk["stock_drift"]),
        stock_volatility=float(risk["stock_volatility"]),
        cb_drift=float(risk["cb_drift"]),
        cb_volatility=float(risk["cb_volatility"]),
    )
    cb_log_returns = np.diff(np.log(paths[:, 2, :]), axis=1).reshape(-1)
    risk_engine = BMXRiskEngine(confidence_level=float(risk["tail_probability"]))
    var_loss = risk_engine.bitcoin_aware_var(cb_log_returns, str(risk["bitcoin_regime"]))
    expected_shortfall_loss = risk_engine.expected_shortfall(cb_log_returns, var_loss)

    def rounded(value: float) -> float:
        return round(float(value), 12)

    return {
        "id": scenario_id,
        "issue_id": issue["id"],
        "snapshot_id": snapshot["id"],
        "model_version": MODEL_VERSION,
        "assumptions": assumptions,
        "model_output": {
            "bond_floor": rounded(bond_floor),
            "conversion_value": rounded(conversion_value),
            "option_component": rounded(option_component),
            "theoretical_price": rounded(theoretical_price),
            "var": rounded(var_loss),
            "expected_shortfall": rounded(expected_shortfall_loss),
            "units": {
                "bond_floor": "USD per 1000 USD face value",
                "conversion_value": "USD per 1000 USD face value",
                "option_component": "USD per 1000 USD face value",
                "theoretical_price": "USD per 1000 USD face value",
                "var": "one-step log-return loss fraction",
                "expected_shortfall": "one-step log-return loss fraction",
            },
        },
        "claim_boundary": {
            "scenario_not_observed_market_value": True,
            "calibrated_to_market": False,
            "fair_value_claim": False,
            "recommendation": False,
        },
    }
