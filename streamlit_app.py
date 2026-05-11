from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from energy_market_state.curve_bridge import (  # noqa: E402
    COMMODITY_CURVE_CONFIG,
    FORWARD_HORIZON_MONTHS,
    LINEAR_LABEL,
    MONOTONE_LABEL,
    POWER_BUCKETS,
    apply_scenario_to_nodes,
    build_monthly_linear_curve,
    build_visual_curve,
    commodity_select_order,
    compute_term_structure,
    scenario_options,
    tenor_array_to_dates,
)


DATA_PATH = PROJECT_ROOT / "runtime" / "curve_dashboard" / "curve_dashboard_data.csv"
METRICS_PATH = PROJECT_ROOT / "runtime" / "curve_dashboard" / "curve_dashboard_metrics.csv"
POWER_PROFILE_PATH = PROJECT_ROOT / "runtime" / "curve_dashboard" / "power_hourly_profile_2025.csv"

METHODLOGY_TEXT = {
    "BRENT": (
        "Oil curves are modeled using tradable futures interpolation because the market already "
        "provides relatively granular liquid forward points. The valuation curve is linear by default, "
        "with monotone cubic available only as visual smoothing."
    ),
    "WTI": (
        "WTI is treated like a tradable futures strip problem: observed forward nodes anchor the curve, "
        "while interpolation fills missing tenors and spread analytics explain the shape."
    ),
    "TTF": (
        "TTF is a hybrid case: it behaves like a tradable futures curve, but strong seasonality means "
        "winter-summer structure matters. The modeled curve combines node interpolation with seasonal context."
    ),
    "EUA": (
        "EUA is handled as a carry and term-structure problem. The curve is anchored by smooth financial-like "
        "forward points, with focus on Dec-Dec carry and slope rather than shaping."
    ),
    "DE_POWER": (
        "German Power uses baseload forward anchors combined with normalized previous-year hourly profile factors. "
        "For each delivery period, the hourly-shaped forward curve is rescaled so that its average equals the "
        "observable baseload forward price. This illustrates how a baseload forward can be distributed into a "
        "granular intraday shape while preserving the market anchor."
    ),
}

SCENARIO_NOTE = (
    "`Steepener` and `Flattener` are simplified generic curve-shape stresses for v1. "
    "They are intended as intuitive front-vs-back overlays, not full commodity-specific market-convention models."
)

THEME_CSS = """
<style>
    .stApp {
        background: linear-gradient(180deg, #ffffff 0%, #f6faff 100%);
        color: #0f172a;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fbff 0%, #edf5ff 100%);
    }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #d8e7fb;
        border-radius: 14px;
        padding: 0.8rem 1rem;
        box-shadow: 0 8px 24px rgba(15, 76, 129, 0.06);
    }
    h1, h2, h3 {
        color: #123a72;
    }
</style>
"""


def format_value(value: float | str | None, unit: str = "") -> str:
    if value is None or pd.isna(value):
        return "n/a"
    if unit is None or pd.isna(unit):
        unit = ""
    if isinstance(value, str):
        return value
    if unit == "%":
        return f"{value:.1f}%"
    if unit == "%/month":
        return f"{value:.1f}%/month"
    if unit == "days":
        return f"{int(value)} days"
    if unit == "count":
        return f"{int(value)}"
    if unit:
        return f"{value:,.2f} {unit}"
    return f"{value:,.2f}"


def parse_metric_value(value: object) -> object:
    if pd.isna(value):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return value
        try:
            return float(stripped)
        except ValueError:
            return value
    return value


def inject_theme() -> None:
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def metric_pair(metric_map: dict[str, tuple[object, str]], metric_name: str, default_unit: str = "") -> tuple[object, str]:
    return metric_map.get(metric_name, (None, default_unit))


def normalize_factor_array(values: np.ndarray) -> np.ndarray:
    cleaned = np.asarray(values, dtype=float)
    cleaned = np.where(np.isfinite(cleaned), cleaned, np.nan)
    mean_value = np.nanmean(cleaned)
    if not np.isfinite(mean_value) or abs(mean_value) < 1e-9:
        return np.ones_like(cleaned, dtype=float)
    normalized = cleaned / mean_value
    return np.where(np.isfinite(normalized), normalized, 1.0)


@st.cache_data(show_spinner=False)
def load_dashboard_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not DATA_PATH.exists() or not METRICS_PATH.exists():
        raise FileNotFoundError(
            "Dashboard CSVs are missing. Run `python scripts/build_forward_curve_dashboard_dataset.py` first."
        )

    data_df = pd.read_csv(DATA_PATH, low_memory=False)
    metrics_df = pd.read_csv(METRICS_PATH, low_memory=False)

    for column in ["valuation_date", "delivery_start", "delivery_end", "x_timestamp"]:
        if column in data_df.columns:
            data_df[column] = pd.to_datetime(data_df[column], errors="coerce")
    if "valuation_date" in metrics_df.columns:
        metrics_df["valuation_date"] = pd.to_datetime(metrics_df["valuation_date"], errors="coerce")

    metrics_df["metric_value"] = metrics_df["metric_value"].apply(parse_metric_value)
    return data_df, metrics_df


@st.cache_data(show_spinner=False)
def load_power_hourly_profile_template() -> pd.DataFrame:
    if not POWER_PROFILE_PATH.exists():
        return pd.DataFrame(
            columns=["profile_year", "month_of_year", "is_weekend", "hour_local", "raw_price", "raw_factor"]
        )

    template = pd.read_csv(POWER_PROFILE_PATH, low_memory=False)
    if "is_weekend" in template.columns:
        template["is_weekend"] = template["is_weekend"].astype(str).str.lower().map({"true": True, "false": False}).fillna(False)
    return template


def curve_subset_for_selection(
    data_df: pd.DataFrame,
    commodity: str,
    valuation_date: pd.Timestamp,
    power_bucket: str,
    horizon_months: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    commodity_rows = data_df[
        (data_df["commodity"] == commodity)
        & (data_df["valuation_date"] == valuation_date)
        & (data_df["tenor_month"].fillna(0).astype(float) <= horizon_months)
    ].copy()

    observed = commodity_rows[commodity_rows["series_type"] == "observed_forward_node"].sort_values("tenor_month")
    drivers = commodity_rows[commodity_rows["series_type"] == "method_driver"].copy()

    if commodity == "DE_POWER" and power_bucket != "Baseload":
        modeled = commodity_rows[
            (commodity_rows["series_type"] == "power_bucket_price") & (commodity_rows["bucket_name"] == power_bucket)
        ].sort_values("tenor_month")
        drivers = drivers[drivers["bucket_name"] == power_bucket].sort_values("tenor_month")
    else:
        modeled = commodity_rows[commodity_rows["series_type"] == "modeled_forward_curve"].sort_values("tenor_month")
        bucket_match = "Baseload" if commodity == "DE_POWER" else ""
        drivers = drivers[drivers["bucket_name"] == bucket_match].sort_values("tenor_month")

    return observed, modeled, drivers


def historical_subset_for_selection(
    data_df: pd.DataFrame,
    commodity: str,
    historical_start: pd.Timestamp,
    valuation_date: pd.Timestamp,
) -> pd.DataFrame:
    historical = data_df[
        (data_df["commodity"] == commodity) & (data_df["series_type"] == "historical_realized")
    ].copy()
    historical = historical[(historical["x_timestamp"] >= historical_start) & (historical["x_timestamp"] <= valuation_date)]
    return historical.sort_values("x_timestamp")


def smooth_historical_display(historical: pd.DataFrame, commodity: str, display_mode: str) -> pd.DataFrame:
    if historical.empty or commodity != "DE_POWER" or display_mode == "Raw daily":
        return historical

    window = 7 if display_mode == "7d rolling average" else 30
    smoothed = historical.sort_values("x_timestamp").copy()
    smoothed["price_value"] = smoothed["price_value"].astype(float).rolling(window, min_periods=1).mean()
    return smoothed


def power_baseload_subsets(
    data_df: pd.DataFrame,
    valuation_date: pd.Timestamp,
    horizon_months: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    commodity_rows = data_df[
        (data_df["commodity"] == "DE_POWER")
        & (data_df["valuation_date"] == valuation_date)
        & (data_df["tenor_month"].fillna(0).astype(float) <= horizon_months)
    ].copy()
    baseload_curve = commodity_rows[commodity_rows["series_type"] == "modeled_forward_curve"].sort_values("tenor_month")
    baseload_driver = commodity_rows[
        (commodity_rows["series_type"] == "method_driver") & (commodity_rows["bucket_name"] == "Baseload")
    ].sort_values("tenor_month")
    return baseload_curve, baseload_driver


def build_hourly_shaped_forward_curve(
    baseload_curve: pd.DataFrame,
    profile_template: pd.DataFrame,
) -> tuple[list[pd.Timestamp], list[float]]:
    if baseload_curve.empty or profile_template.empty:
        return [], []

    template = profile_template.copy()
    template["month_of_year"] = template["month_of_year"].astype(int)
    template["hour_local"] = template["hour_local"].astype(int)
    template["raw_factor"] = pd.to_numeric(template["raw_factor"], errors="coerce").fillna(1.0)
    template = template[["month_of_year", "is_weekend", "hour_local", "raw_factor"]]

    overlay_x: list[pd.Timestamp] = []
    overlay_y: list[float] = []

    for row in baseload_curve.itertuples(index=False):
        anchor_start = row.delivery_start if pd.notna(row.delivery_start) else row.x_timestamp
        if pd.isna(anchor_start) or pd.isna(row.price_value):
            continue

        delivery_start = pd.Timestamp(anchor_start).normalize()
        if pd.notna(row.delivery_end):
            delivery_end = pd.Timestamp(row.delivery_end).normalize() + pd.Timedelta(days=1)
        else:
            delivery_end = (delivery_start + pd.offsets.MonthEnd(0)) + pd.Timedelta(days=1)

        local_hours = pd.date_range(
            delivery_start.tz_localize("Europe/Berlin"),
            delivery_end.tz_localize("Europe/Berlin"),
            freq="h",
            inclusive="left",
        )
        if len(local_hours) == 0:
            continue

        hourly_frame = pd.DataFrame({"timestamp_local": local_hours})
        hourly_frame["month_of_year"] = hourly_frame["timestamp_local"].dt.month.astype(int)
        hourly_frame["hour_local"] = hourly_frame["timestamp_local"].dt.hour.astype(int)
        hourly_frame["is_weekend"] = (hourly_frame["timestamp_local"].dt.dayofweek >= 5).astype(bool)
        hourly_frame = hourly_frame.merge(
            template,
            on=["month_of_year", "is_weekend", "hour_local"],
            how="left",
        )
        raw_factors = hourly_frame["raw_factor"].fillna(1.0).astype(float).to_numpy()
        normalized_factors = normalize_factor_array(raw_factors)
        hourly_prices = float(row.price_value) * normalized_factors

        overlay_x.extend(hourly_frame["timestamp_local"].dt.tz_localize(None).tolist())
        overlay_y.extend(hourly_prices.tolist())

    return overlay_x, overlay_y


def metric_lookup(
    metrics_df: pd.DataFrame,
    commodity: str,
    valuation_date: pd.Timestamp,
) -> dict[str, tuple[object, str]]:
    current = metrics_df[
        (metrics_df["commodity"] == commodity) & (metrics_df["valuation_date"] == valuation_date)
    ]
    return {
        row.metric_name: (row.metric_value, row.metric_unit)
        for row in current.itertuples(index=False)
    }


def build_spread_table(
    metric_map: dict[str, tuple[object, str]],
    commodity: str,
    power_bucket: str,
) -> pd.DataFrame:
    if commodity in {"BRENT", "WTI"}:
        metric_names = ["M1-M2", "M2-M3", "M1-M6"]
    elif commodity == "TTF":
        metric_names = ["M1-M2", "M1-M6", "Winter-Summer"]
    elif commodity == "EUA":
        metric_names = ["Dec1-Dec2", "Dec1-Dec3", "Carry slope"]
    else:
        metric_names = ["M1-M2 Baseload", "M1-M6 Baseload"]
        premium_value = 0.0 if power_bucket == "Baseload" else metric_map.get(
            f"{power_bucket} premium vs Baseload", (float("nan"), COMMODITY_CURVE_CONFIG[commodity].unit)
        )[0]
        premium_unit = COMMODITY_CURVE_CONFIG[commodity].unit
        table = pd.DataFrame(
            [
                {"Spread": "M1-M2 Baseload", "Value": format_value(*metric_map.get("M1-M2 Baseload", (None, "")))},
                {"Spread": "M1-M6 Baseload", "Value": format_value(*metric_map.get("M1-M6 Baseload", (None, "")))},
                {
                    "Spread": "Selected bucket premium vs Baseload",
                    "Value": format_value(premium_value, premium_unit),
                },
            ]
        )
        return table

    rows = []
    for metric_name in metric_names:
        metric_value, metric_unit = metric_map.get(metric_name, (None, ""))
        rows.append({"Spread": metric_name, "Value": format_value(metric_value, metric_unit)})
    return pd.DataFrame(rows)


def build_display_curve(
    commodity: str,
    valuation_date: pd.Timestamp,
    observed_nodes: pd.DataFrame,
    modeled_curve: pd.DataFrame,
    interpolation_method: str,
    horizon_months: int,
) -> tuple[list[pd.Timestamp], list[float]]:
    if commodity == "DE_POWER" or interpolation_method == LINEAR_LABEL:
        return modeled_curve["x_timestamp"].tolist(), modeled_curve["price_value"].astype(float).tolist()

    tenor_dense, price_dense = build_visual_curve(
        observed_nodes["tenor_month"].astype(float).to_numpy(),
        observed_nodes["price_value"].astype(float).to_numpy(),
        horizon_months,
        interpolation_method,
    )
    x_dates = tenor_array_to_dates(valuation_date, tenor_dense)
    return x_dates, price_dense.tolist()


def build_shocked_curve(
    commodity: str,
    valuation_date: pd.Timestamp,
    observed_nodes: pd.DataFrame,
    power_bucket_factors: pd.DataFrame,
    scenario_name: str,
    shock_size_decimal: float,
    interpolation_method: str,
    horizon_months: int,
    power_bucket: str,
) -> tuple[list[pd.Timestamp], list[float]]:
    shocked_nodes = apply_scenario_to_nodes(
        observed_nodes["tenor_month"].astype(float).to_numpy(),
        observed_nodes["price_value"].astype(float).to_numpy(),
        scenario_name,
        shock_size_decimal,
    )
    monthly_tenors, shocked_linear = build_monthly_linear_curve(
        observed_nodes["tenor_month"].astype(float).to_numpy(),
        shocked_nodes,
        horizon_months,
    )

    if commodity == "DE_POWER" and power_bucket != "Baseload":
        factor_curve = power_bucket_factors.sort_values("tenor_month")["driver_value"].astype(float).to_numpy()
        shocked_linear = shocked_linear * factor_curve

    if commodity == "DE_POWER" or interpolation_method == LINEAR_LABEL:
        return tenor_array_to_dates(valuation_date, monthly_tenors), shocked_linear.tolist()

    tenor_dense, price_dense = build_visual_curve(
        observed_nodes["tenor_month"].astype(float).to_numpy(),
        shocked_nodes,
        horizon_months,
        interpolation_method,
    )
    return tenor_array_to_dates(valuation_date, tenor_dense), price_dense.tolist()


def build_key_spread_card(
    commodity: str,
    metric_map: dict[str, tuple[object, str]],
) -> tuple[str, str]:
    fallback_value, fallback_unit = metric_pair(metric_map, "key_spread", COMMODITY_CURVE_CONFIG[commodity].unit)

    if commodity in {"BRENT", "WTI"}:
        value, unit = metric_pair(metric_map, "M1-M2", COMMODITY_CURVE_CONFIG[commodity].unit)
        return "M1-M2 spread", format_value(value if not pd.isna(value) else fallback_value, unit or fallback_unit)

    if commodity == "TTF":
        winter_value, winter_unit = metric_pair(metric_map, "Winter-Summer", COMMODITY_CURVE_CONFIG[commodity].unit)
        if winter_value is not None and not pd.isna(winter_value):
            return "Winter-Summer spread", format_value(winter_value, winter_unit)
        m12_value, m12_unit = metric_pair(metric_map, "M1-M2", COMMODITY_CURVE_CONFIG[commodity].unit)
        return "M1-M2 spread", format_value(m12_value if not pd.isna(m12_value) else fallback_value, m12_unit or fallback_unit)

    if commodity == "EUA":
        value, unit = metric_pair(metric_map, "Dec1-Dec2", COMMODITY_CURVE_CONFIG[commodity].unit)
        return "Dec1-Dec2 spread", format_value(value if not pd.isna(value) else fallback_value, unit or fallback_unit)

    value, unit = metric_pair(metric_map, "M1-M2 Baseload", COMMODITY_CURVE_CONFIG[commodity].unit)
    return "M1-M2 Baseload", format_value(value if not pd.isna(value) else fallback_value, unit or fallback_unit)


def build_realized_vol_display(metric_map: dict[str, tuple[object, str]]) -> tuple[str, str | None]:
    raw_value, unit = metric_pair(metric_map, "realized_vol", "%")
    if raw_value is None or pd.isna(raw_value):
        return "n/a", None

    display_value = float(raw_value)
    if display_value > 300.0:
        return format_value(300.0, unit), "Capped at 300% for display robustness after robust log-return estimation."
    return format_value(display_value, unit), None


def midpoint_timestamp(start: pd.Timestamp, end: pd.Timestamp) -> pd.Timestamp:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    return start_ts + (end_ts - start_ts) / 2


def methodology_box_text(commodity: str, is_power: bool) -> str:
    note = METHODLOGY_TEXT[commodity]
    if is_power:
        note += "\n\nObserved markers show baseload anchors; the selected bucket line is the shaped curve built off those anchors."
        note += "\n\nBucketed shaping remains the v1 valuation methodology. The hourly-shaped forward curve is shown for methodology illustration only."
    note += f"\n\n{SCENARIO_NOTE}"
    note += "\n\nSynthetic forward nodes shown as observed anchors are demo-grade market nodes and are flagged in the source CSV with `is_synthetic = True`."
    return note


def quality_panel(metric_map: dict[str, tuple[object, str]], synthetic_nodes_present: bool | None) -> None:
    status_value = metric_pair(metric_map, "curve_construction_status")[0]
    if status_value is None or pd.isna(status_value):
        status_value = "n/a"
    missing_nodes_value = metric_pair(metric_map, "missing_forward_nodes", "count")[0]
    expected_nodes_value = metric_pair(metric_map, "expected_forward_nodes", "count")[0]
    source_age_value = metric_pair(metric_map, "source_age_days", "days")[0]
    max_local_slope = metric_pair(metric_map, "max_local_slope_pct_per_month", "%/month")

    if missing_nodes_value is None or pd.isna(missing_nodes_value) or expected_nodes_value is None or pd.isna(expected_nodes_value):
        missing_nodes_text = "n/a"
    else:
        missing_nodes_text = f"{int(missing_nodes_value)} / {int(expected_nodes_value)}"

    if source_age_value is None or pd.isna(source_age_value):
        source_age_text = "n/a"
    else:
        source_age_text = f"{int(source_age_value)} days"

    if synthetic_nodes_present is None:
        synthetic_nodes_text = "n/a"
    else:
        synthetic_nodes_text = "Yes" if synthetic_nodes_present else "No"

    status_color = "#157347" if status_value == "PASS" else "#b86e00"
    status_bg = "#eaf7ef" if status_value == "PASS" else "#fff5e6"
    box = f"""
    <div style="border:1px solid #d8e7fb;border-radius:14px;padding:0.95rem 1rem;background:#f8fbff;box-shadow:0 8px 24px rgba(15, 76, 129, 0.06);">
      <div style="font-size:0.82rem;color:#45627f;font-weight:700;letter-spacing:0.02em;text-transform:uppercase;">Curve controls</div>
      <div style="margin-top:0.55rem;display:inline-block;padding:0.18rem 0.55rem;border-radius:999px;background:{status_bg};color:{status_color};font-size:0.78rem;font-weight:700;">Status: {status_value}</div>
      <div style="font-size:0.87rem;margin-top:0.65rem;">Missing nodes: <b>{missing_nodes_text}</b></div>
      <div style="font-size:0.87rem;">Source age: <b>{source_age_text}</b></div>
      <div style="font-size:0.87rem;">Max local slope: <b>{format_value(*max_local_slope)}</b></div>
      <div style="font-size:0.87rem;">Synthetic nodes: <b>{synthetic_nodes_text}</b></div>
    </div>
    """
    st.markdown(box, unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="Historical-to-Forward Curve Bridge", layout="wide")
    inject_theme()
    st.title("Historical-to-Forward Curve Bridge")
    st.caption("Observed market anchors vs modeled valuation curve, with commodity-specific construction methodology.")

    data_df, metrics_df = load_dashboard_data()
    power_hourly_profile_template = load_power_hourly_profile_template()

    available_valuation_dates = sorted(
        data_df.loc[data_df["series_type"] == "observed_forward_node", "valuation_date"].dropna().unique().tolist()
    )
    historical_min_date = data_df.loc[data_df["series_type"] == "historical_realized", "x_timestamp"].min()

    with st.sidebar:
        st.header("Controls")
        commodity = st.selectbox(
            "Commodity",
            options=commodity_select_order(),
            index=0,
            format_func=lambda key: COMMODITY_CURVE_CONFIG[key].display_name,
        )
        valuation_date = st.selectbox(
            "Valuation date",
            options=available_valuation_dates,
            index=len(available_valuation_dates) - 1,
            format_func=lambda ts: pd.Timestamp(ts).strftime("%Y-%m-%d"),
        )
        default_hist_start = max(pd.Timestamp(historical_min_date), pd.Timestamp(valuation_date) - pd.Timedelta(days=365))
        historical_start = st.date_input(
            "Historical start date",
            value=default_hist_start.date(),
            min_value=pd.Timestamp(historical_min_date).date(),
            max_value=pd.Timestamp(valuation_date).date(),
        )
        horizon_months = st.slider("Forward horizon (months)", min_value=6, max_value=FORWARD_HORIZON_MONTHS, value=24, step=1)
        interpolation_method = st.selectbox(
            "Interpolation method",
            options=[LINEAR_LABEL, MONOTONE_LABEL],
            index=0,
            disabled=(commodity == "DE_POWER"),
            help="Power uses baseload shaping; interpolation display controls apply to tradable-node curves.",
        )
        scenario_name = st.selectbox("Scenario", options=scenario_options(), index=0)
        shock_size_pct = st.slider("Shock size %", min_value=0.0, max_value=25.0, value=5.0, step=0.5)
        show_historical = st.checkbox("Show historical data", value=True)
        show_observed = st.checkbox("Show observed forwards", value=True)
        show_modeled = st.checkbox("Show modeled curve", value=True)
        show_shocked = st.checkbox("Show shocked curve", value=(scenario_name != "Base"))
        show_method_driver = st.checkbox("Show method driver", value=True)
        power_bucket = "Baseload"
        historical_display_mode = "Raw daily"
        show_hourly_shaped_forward_curve = False
        if commodity == "DE_POWER":
            power_bucket = st.selectbox("Power bucket", options=POWER_BUCKETS, index=0)
            historical_display_mode = st.selectbox(
                "Historical display",
                options=["Raw daily", "7d rolling average", "30d rolling average"],
                index=1,
            )
            show_hourly_shaped_forward_curve = st.checkbox("Show hourly-shaped forward curve", value=False)

    valuation_date = pd.Timestamp(valuation_date).normalize()
    historical_start = pd.Timestamp(historical_start).normalize()
    config = COMMODITY_CURVE_CONFIG[commodity]

    historical = historical_subset_for_selection(data_df, commodity, historical_start, valuation_date)
    historical_display = smooth_historical_display(historical, commodity, historical_display_mode)
    observed_nodes, modeled_curve, method_driver = curve_subset_for_selection(
        data_df=data_df,
        commodity=commodity,
        valuation_date=valuation_date,
        power_bucket=power_bucket,
        horizon_months=horizon_months,
    )
    metric_map = metric_lookup(metrics_df, commodity, valuation_date)

    if observed_nodes.empty or modeled_curve.empty:
        st.error("No forward curve data available for the selected filters.")
        return

    baseload_curve = pd.DataFrame()
    baseload_driver = pd.DataFrame()
    hourly_shaped_x: list[pd.Timestamp] = []
    hourly_shaped_y: list[float] = []
    if commodity == "DE_POWER":
        baseload_curve, baseload_driver = power_baseload_subsets(data_df, valuation_date, horizon_months)
        if show_hourly_shaped_forward_curve:
            hourly_shaped_x, hourly_shaped_y = build_hourly_shaped_forward_curve(
                baseload_curve=baseload_curve,
                profile_template=power_hourly_profile_template,
            )

    base_x, base_y = build_display_curve(
        commodity=commodity,
        valuation_date=valuation_date,
        observed_nodes=observed_nodes,
        modeled_curve=modeled_curve,
        interpolation_method=interpolation_method,
        horizon_months=horizon_months,
    )

    displayed_front = float(modeled_curve["price_value"].iloc[0])
    displayed_back = float(modeled_curve["price_value"].iloc[-1])
    key_spread_label, key_spread_value = build_key_spread_card(commodity, metric_map)
    realized_vol_display, realized_vol_note = build_realized_vol_display(metric_map)
    synthetic_nodes_present = None if observed_nodes.empty else bool(observed_nodes["is_synthetic"].fillna(False).astype(bool).any())

    top_cols = st.columns([1, 1, 1, 1, 1, 1.35])
    top_cols[0].metric("Front price", format_value(displayed_front, config.unit))
    top_cols[1].metric("Back price", format_value(displayed_back, config.unit))
    top_cols[2].metric("Term structure", compute_term_structure(displayed_front, displayed_back))
    top_cols[3].metric(key_spread_label, key_spread_value)
    with top_cols[4]:
        st.metric("Realized vol", realized_vol_display)
        if realized_vol_note:
            st.caption(realized_vol_note)
    with top_cols[5]:
        quality_panel(metric_map, synthetic_nodes_present)

    driver_name = method_driver["driver_name"].iloc[0] if not method_driver.empty else ""
    driver_unit = method_driver["unit"].iloc[0] if not method_driver.empty else ""
    observed_label = "Observed baseload anchors" if commodity == "DE_POWER" else "Observed market anchors"
    modeled_label = "Modeled forward curve"
    if commodity == "DE_POWER":
        modeled_label = "Baseload forward curve" if power_bucket == "Baseload" else f"{power_bucket} shaped curve"
    historical_label = "Historical realized"
    if commodity == "DE_POWER" and historical_display_mode != "Raw daily":
        historical_label = f"Historical realized ({historical_display_mode})"

    figure = go.Figure()
    if show_historical and not historical_display.empty:
        figure.add_trace(
            go.Scatter(
                x=historical_display["x_timestamp"],
                y=historical_display["price_value"],
                mode="lines",
                name=historical_label,
                line=dict(color="#7a92b2", width=1.8),
                yaxis="y",
            )
        )

    if show_observed:
        figure.add_trace(
            go.Scatter(
                x=observed_nodes["x_timestamp"],
                y=observed_nodes["price_value"],
                mode="markers",
                name=observed_label,
                marker=dict(color=config.color, size=8, symbol="circle"),
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}<extra></extra>",
                yaxis="y",
            )
        )

    if show_modeled:
        figure.add_trace(
            go.Scatter(
                x=base_x,
                y=base_y,
                mode="lines",
                name=modeled_label,
                line=dict(color=config.color, width=2.5),
                yaxis="y",
            )
        )

    if commodity == "DE_POWER" and show_hourly_shaped_forward_curve and hourly_shaped_x and hourly_shaped_y:
        figure.add_trace(
            go.Scatter(
                x=hourly_shaped_x,
                y=hourly_shaped_y,
                mode="lines",
                name="Hourly-shaped forward curve",
                line=dict(color="rgba(31, 111, 178, 0.45)", width=1.2, dash="dash"),
                opacity=0.45,
                yaxis="y",
            )
        )

    if show_shocked and scenario_name != "Base":
        shocked_x, shocked_y = build_shocked_curve(
            commodity=commodity,
            valuation_date=valuation_date,
            observed_nodes=observed_nodes,
            power_bucket_factors=method_driver,
            scenario_name=scenario_name,
            shock_size_decimal=shock_size_pct / 100.0,
            interpolation_method=interpolation_method,
            horizon_months=horizon_months,
            power_bucket=power_bucket,
        )
        figure.add_trace(
            go.Scatter(
                x=shocked_x,
                y=shocked_y,
                mode="lines",
                name=f"{scenario_name} ({shock_size_pct:.1f}%)",
                line=dict(color="#5a9be6", width=2, dash="dash"),
                yaxis="y",
            )
        )

    if show_method_driver and not method_driver.empty:
        figure.add_trace(
            go.Scatter(
                x=method_driver["x_timestamp"],
                y=method_driver["driver_value"].astype(float),
                mode="lines",
                name=driver_name,
                line=dict(color="#325f94", width=1.5, dash="dot"),
                yaxis="y2",
            )
        )

    if not historical_display.empty:
        figure.add_vrect(
            x0=historical_display["x_timestamp"].min(),
            x1=valuation_date,
            fillcolor="rgba(18, 58, 114, 0.05)",
            line_width=0,
            layer="below",
        )
        figure.add_annotation(
            x=midpoint_timestamp(historical_display["x_timestamp"].min(), valuation_date),
            y=1.05,
            xref="x",
            yref="paper",
            text="Historical realized",
            showarrow=False,
            font=dict(color="#45627f"),
            bgcolor="rgba(255,255,255,0.85)",
        )

    forward_end_candidates = [pd.Timestamp(x) for x in base_x]
    if commodity == "DE_POWER" and show_hourly_shaped_forward_curve and hourly_shaped_x:
        forward_end_candidates.extend(pd.Timestamp(x) for x in hourly_shaped_x)
    if show_shocked and scenario_name != "Base":
        forward_end_candidates.extend(pd.Timestamp(x) for x in shocked_x)
    if forward_end_candidates:
        forward_end = max(forward_end_candidates)
        if forward_end > valuation_date:
            figure.add_annotation(
                x=midpoint_timestamp(valuation_date, forward_end),
                y=1.05,
                xref="x",
                yref="paper",
                text="Forward curve",
                showarrow=False,
                font=dict(color="#45627f"),
                bgcolor="rgba(255,255,255,0.85)",
            )

    figure.add_shape(
        type="line",
        x0=valuation_date,
        x1=valuation_date,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(color="#334155", width=1.5, dash="dash"),
    )
    figure.add_annotation(
        x=valuation_date,
        y=1,
        xref="x",
        yref="paper",
        text="Valuation date",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        yshift=6,
        font=dict(color="#334155"),
        bgcolor="rgba(255,255,255,0.85)",
    )
    figure.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        height=650,
        margin=dict(l=20, r=20, t=20, b=20),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#d8e7fb",
            borderwidth=1,
        ),
        xaxis=dict(title="Historical data  |  Valuation date  |  Forward curve"),
        yaxis=dict(title=config.unit, gridcolor="#dce8f7", zeroline=False),
        yaxis2=dict(
            title=f"{driver_name} ({driver_unit})" if driver_name else "",
            overlaying="y",
            side="right",
            showgrid=False,
            zeroline=False,
        ),
    )
    st.plotly_chart(figure, width="stretch")

    st.caption("Observed forward nodes are synthetic demo-grade market anchors and are flagged in the source CSV with `is_synthetic = True`.")
    if show_method_driver and method_driver.empty:
        st.caption("Method driver is not available for the selected filters.")

    bottom_left, bottom_right = st.columns([1.0, 1.2])
    with bottom_left:
        st.subheader("Spread analysis")
        spread_table = build_spread_table(metric_map, commodity, power_bucket)
        st.dataframe(spread_table, width="stretch", hide_index=True)

    with bottom_right:
        st.subheader("Methodology explanation")
        st.markdown(methodology_box_text(commodity, commodity == "DE_POWER"))


if __name__ == "__main__":
    main()
