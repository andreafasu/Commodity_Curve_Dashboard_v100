from __future__ import annotations

from math import erf, exp, log, pi, sqrt

import numpy as np

try:
    from scipy.interpolate import PchipInterpolator
except Exception:  # pragma: no cover - optional runtime fallback
    PchipInterpolator = None


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def normal_pdf(x: float) -> float:
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


def black76_price_greeks(
    forward: float,
    strike: float,
    volatility: float,
    time_to_expiry: float,
    option_type: str,
    discount_factor: float = 1.0,
) -> dict[str, float]:
    option_sign = 1.0 if option_type.lower() == "call" else -1.0

    if (
        forward <= 0.0
        or strike <= 0.0
        or time_to_expiry <= 0.0
        or volatility <= 0.0
    ):
        intrinsic = max(option_sign * (forward - strike), 0.0)
        delta = 1.0 if option_sign * (forward - strike) > 0.0 else 0.0
        if option_sign < 0.0:
            delta -= 1.0
        return {
            "price": discount_factor * intrinsic,
            "delta": discount_factor * delta,
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
            "implied_vol": max(volatility, 0.0),
        }

    sigma_root_t = volatility * sqrt(time_to_expiry)
    d1 = (log(forward / strike) + 0.5 * volatility * volatility * time_to_expiry) / sigma_root_t
    d2 = d1 - sigma_root_t

    if option_sign > 0.0:
        price = discount_factor * (forward * normal_cdf(d1) - strike * normal_cdf(d2))
        delta = discount_factor * normal_cdf(d1)
    else:
        price = discount_factor * (strike * normal_cdf(-d2) - forward * normal_cdf(-d1))
        delta = -discount_factor * normal_cdf(-d1)

    gamma = discount_factor * normal_pdf(d1) / (forward * sigma_root_t)
    vega = discount_factor * forward * normal_pdf(d1) * sqrt(time_to_expiry)
    theta = -discount_factor * forward * normal_pdf(d1) * volatility / (2.0 * sqrt(time_to_expiry))

    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "implied_vol": volatility,
    }


def build_prompt_anchored_curve(
    prompt_price: float,
    delivery_month_numbers: np.ndarray,
    seasonal_multipliers: np.ndarray,
    carry_signal: float,
    front_signal: float,
    seasonality_strength: float,
    slope_scale: float,
    front_scale: float,
    front_decay_years: float = 0.45,
) -> np.ndarray:
    tenors_years = delivery_month_numbers / 12.0
    seasonality = 1.0 + seasonality_strength * (seasonal_multipliers - 1.0)
    slope_component = np.exp(slope_scale * carry_signal * tenors_years)
    front_component = 1.0 + front_scale * front_signal * np.exp(-tenors_years / front_decay_years)

    raw_curve = seasonality * slope_component * front_component
    normalized_curve = raw_curve / raw_curve[0]
    return prompt_price * normalized_curve


def smile_volatility(
    atm_volatility: float,
    forward: float,
    strike: float,
    time_to_expiry: float,
    skew: float,
    smile: float,
    term_slope: float,
    min_volatility: float = 0.06,
    max_volatility: float = 2.50,
) -> float:
    if forward <= 0.0 or strike <= 0.0:
        return atm_volatility

    log_moneyness = log(strike / forward)
    term_adjustment = 1.0 + term_slope * (1.0 / sqrt(max(time_to_expiry, 1.0 / 12.0)) - 1.0)
    volatility = atm_volatility * term_adjustment * (1.0 + skew * log_moneyness + smile * log_moneyness * log_moneyness)
    return float(np.clip(volatility, min_volatility, max_volatility))


def smooth_monthly_curve(
    tenor_months: np.ndarray,
    forward_prices: np.ndarray,
    num_points: int = 120,
) -> tuple[np.ndarray, np.ndarray]:
    x_new = np.linspace(float(tenor_months.min()), float(tenor_months.max()), num_points)
    if len(tenor_months) < 3 or PchipInterpolator is None:
        y_new = np.interp(x_new, tenor_months, forward_prices)
        return x_new, y_new

    interpolator = PchipInterpolator(tenor_months, forward_prices)
    y_new = interpolator(x_new)
    return x_new, y_new


def apply_curve_shock(
    forward_prices: np.ndarray,
    tenor_months: np.ndarray,
    parallel_shift_pct: float = 0.0,
    front_shift_pct: float = 0.0,
    back_shift_pct: float = 0.0,
) -> np.ndarray:
    weights = (tenor_months - tenor_months.min()) / max(float(tenor_months.max() - tenor_months.min()), 1.0)
    shock_profile = parallel_shift_pct + front_shift_pct * (1.0 - weights) + back_shift_pct * weights
    return forward_prices * (1.0 + shock_profile)

