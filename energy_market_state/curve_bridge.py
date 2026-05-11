from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.interpolate import PchipInterpolator
except Exception:  # pragma: no cover - optional runtime fallback
    PchipInterpolator = None


FORWARD_HORIZON_MONTHS = 24

LINEAR_LABEL = "Linear (valuation default)"
MONOTONE_LABEL = "Monotone cubic (visual smoothing only)"

POWER_BUCKETS = ["Baseload", "Peak", "Offpeak", "Weekday", "Weekend"]

EXPECTED_FORWARD_NODES = {
    "BRENT": 8,
    "WTI": 8,
    "TTF": 8,
    "EUA": 3,
    "DE_POWER": 6,
}


@dataclass(frozen=True)
class CommodityCurveConfig:
    commodity: str
    display_name: str
    prompt_column: str
    unit: str
    methodology_family: str
    color: str
    seasonality_strength: float
    slope_scale: float
    front_scale: float
    driver_name: str
    stale_warning_days: int
    max_slope_warning_pct: float


COMMODITY_CURVE_CONFIG: dict[str, CommodityCurveConfig] = {
    "BRENT": CommodityCurveConfig(
        commodity="BRENT",
        display_name="Brent",
        prompt_column="oil_brent_usd_bbl",
        unit="USD/bbl",
        methodology_family="tradable_futures_interpolation",
        color="#0f4c81",
        seasonality_strength=0.08,
        slope_scale=0.12,
        front_scale=0.04,
        driver_name="Local calendar spread",
        stale_warning_days=3,
        max_slope_warning_pct=12.0,
    ),
    "WTI": CommodityCurveConfig(
        commodity="WTI",
        display_name="WTI",
        prompt_column="oil_wti_usd_bbl",
        unit="USD/bbl",
        methodology_family="tradable_futures_interpolation",
        color="#1f6fb2",
        seasonality_strength=0.07,
        slope_scale=0.13,
        front_scale=0.04,
        driver_name="Local calendar spread",
        stale_warning_days=3,
        max_slope_warning_pct=12.0,
    ),
    "TTF": CommodityCurveConfig(
        commodity="TTF",
        display_name="TTF Gas",
        prompt_column="gas_ttf_eur_mwh",
        unit="EUR/MWh",
        methodology_family="seasonal_hybrid_interpolation",
        color="#4b8fd8",
        seasonality_strength=0.48,
        slope_scale=0.15,
        front_scale=0.08,
        driver_name="Seasonal factor",
        stale_warning_days=3,
        max_slope_warning_pct=20.0,
    ),
    "EUA": CommodityCurveConfig(
        commodity="EUA",
        display_name="EUA",
        prompt_column="carbon_eua_etc_proxy_price",
        unit="EUR/tCO2e",
        methodology_family="carry_interpolation",
        color="#8ab4f8",
        seasonality_strength=0.06,
        slope_scale=0.10,
        front_scale=0.02,
        driver_name="Carry slope",
        stale_warning_days=3,
        max_slope_warning_pct=12.0,
    ),
    "DE_POWER": CommodityCurveConfig(
        commodity="DE_POWER",
        display_name="German Power",
        prompt_column="power_de_day_ahead_eur_mwh",
        unit="EUR/MWh",
        methodology_family="baseload_bucketed_shaping",
        color="#123a72",
        seasonality_strength=0.60,
        slope_scale=0.18,
        front_scale=0.12,
        driver_name="Profile factor",
        stale_warning_days=1,
        max_slope_warning_pct=35.0,
    ),
}


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def to_month_period(timestamp: pd.Timestamp) -> pd.Period:
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts.to_period("M")


def delivery_month_from(valuation_date: pd.Timestamp, tenor_month: int) -> pd.Timestamp:
    return (to_month_period(valuation_date) + tenor_month).to_timestamp()


def delivery_month_end(delivery_month_start: pd.Timestamp) -> pd.Timestamp:
    return (pd.Timestamp(delivery_month_start) + pd.offsets.MonthEnd(0)).normalize()


def contract_label(delivery_month_start: pd.Timestamp) -> str:
    return pd.Timestamp(delivery_month_start).strftime("%b-%y")


def commodity_select_order() -> list[str]:
    return ["BRENT", "WTI", "TTF", "DE_POWER", "EUA"]


def interpolation_options() -> list[str]:
    return [LINEAR_LABEL, MONOTONE_LABEL]


def scenario_options() -> list[str]:
    return ["Base", "Parallel shift", "Front-end shock", "Steepener", "Flattener"]


def node_tenors_for_commodity(
    commodity: str,
    valuation_date: pd.Timestamp,
    horizon_months: int = FORWARD_HORIZON_MONTHS,
) -> np.ndarray:
    if commodity in {"BRENT", "WTI"}:
        tenors = np.array([1, 2, 3, 4, 5, 6, 9, 12, 18, 24], dtype=int)
    elif commodity == "TTF":
        tenors = np.array([1, 2, 3, 4, 5, 6, 9, 12, 18, 24], dtype=int)
    elif commodity == "DE_POWER":
        tenors = np.array([1, 2, 3, 6, 9, 12, 18, 24], dtype=int)
    elif commodity == "EUA":
        horizon_tenors = {1, horizon_months}
        base_period = to_month_period(valuation_date)
        for tenor in range(1, horizon_months + 1):
            delivery_start = delivery_month_from(valuation_date, tenor)
            if delivery_start.month == 12:
                horizon_tenors.add(tenor)
        tenors = np.array(sorted(horizon_tenors), dtype=int)
    else:  # pragma: no cover - protected by commodity config
        raise ValueError(f"Unsupported commodity: {commodity}")

    return tenors[tenors <= horizon_months]


def build_monthly_linear_curve(
    node_tenors: np.ndarray,
    node_prices: np.ndarray,
    horizon_months: int = FORWARD_HORIZON_MONTHS,
) -> tuple[np.ndarray, np.ndarray]:
    monthly_tenors = np.arange(1, horizon_months + 1, dtype=float)
    monthly_prices = np.interp(monthly_tenors, node_tenors.astype(float), node_prices.astype(float))
    return monthly_tenors, monthly_prices


def build_visual_curve(
    node_tenors: np.ndarray,
    node_prices: np.ndarray,
    horizon_months: int,
    interpolation_method: str,
) -> tuple[np.ndarray, np.ndarray]:
    if interpolation_method == LINEAR_LABEL or PchipInterpolator is None:
        return build_monthly_linear_curve(node_tenors, node_prices, horizon_months)

    tenor_dense = np.linspace(1.0, float(horizon_months), max(horizon_months * 8, horizon_months))
    interpolator = PchipInterpolator(node_tenors.astype(float), node_prices.astype(float))
    dense_prices = interpolator(tenor_dense)
    return tenor_dense, dense_prices


def tenor_array_to_dates(valuation_date: pd.Timestamp, tenor_array: np.ndarray) -> list[pd.Timestamp]:
    valuation_ts = pd.Timestamp(valuation_date)
    if valuation_ts.tzinfo is not None:
        valuation_ts = valuation_ts.tz_convert(None)
    return [
        valuation_ts + pd.to_timedelta(float(tenor) * 30.4375, unit="D")
        for tenor in tenor_array
    ]


def scenario_shock_profile(
    tenor_months: np.ndarray,
    scenario_name: str,
    shock_size_decimal: float,
) -> np.ndarray:
    if scenario_name == "Base":
        return np.zeros_like(tenor_months, dtype=float)

    min_tenor = float(np.min(tenor_months))
    max_tenor = float(np.max(tenor_months))
    denom = max(max_tenor - min_tenor, 1.0)
    weights = (tenor_months.astype(float) - min_tenor) / denom

    if scenario_name == "Parallel shift":
        return np.full_like(tenor_months, shock_size_decimal, dtype=float)
    if scenario_name == "Front-end shock":
        return shock_size_decimal * (1.0 - weights)
    if scenario_name == "Steepener":
        return shock_size_decimal * (1.0 - 2.0 * weights)
    if scenario_name == "Flattener":
        return -shock_size_decimal * (1.0 - 2.0 * weights)

    raise ValueError(f"Unsupported scenario: {scenario_name}")


def apply_scenario_to_nodes(
    node_tenors: np.ndarray,
    node_prices: np.ndarray,
    scenario_name: str,
    shock_size_decimal: float,
) -> np.ndarray:
    profile = scenario_shock_profile(node_tenors, scenario_name, shock_size_decimal)
    return node_prices * (1.0 + profile)


def compute_term_structure(front_price: float, back_price: float) -> str:
    if front_price == 0.0:
        return "Flat"
    relative_gap = (back_price - front_price) / abs(front_price)
    if relative_gap > 0.005:
        return "Contango"
    if relative_gap < -0.005:
        return "Backwardation"
    return "Flat"


def compute_max_local_slope_pct_per_month(curve_prices: np.ndarray) -> float:
    if len(curve_prices) < 2:
        return 0.0

    prior = curve_prices[:-1]
    nxt = curve_prices[1:]
    denom = np.where(np.abs(prior) < 1e-9, np.nan, np.abs(prior))
    local = np.abs((nxt - prior) / denom)
    return float(np.nanmax(local) * 100.0)
