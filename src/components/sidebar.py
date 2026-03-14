from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import streamlit as st

from src.services.market_data import calculate_time_to_expiry, get_yahoo_risk_free_rate, get_yahoo_spot_price


@dataclass(frozen=True)
class SidebarConfig:
    setup_mode: str
    ticker: str | None
    model_type: str
    current_price: float
    strike_price: float
    time_to_maturity: float
    volatility: float
    risk_free_rate: float
    model_params: dict


def _inject_sidebar_styles() -> None:
    st.sidebar.markdown("""
        <style>
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] input,
        section[data-testid="stSidebar"] select,
        section[data-testid="stSidebar"] div[data-baseweb="select"],
        section[data-testid="stSidebar"] .stNumberInput,
        section[data-testid="stSidebar"] .stCheckbox,
        section[data-testid="stSidebar"] span {
            font-size: 1.2rem !important;
        }

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            font-size: 2rem !important;
        }

        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] label {
            font-size: 1.15rem !important;
        }

        section[data-testid="stSidebar"] .stElementContainer {
            margin-bottom: 0.15rem !important;
        }

        section[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
            font-size: 1.0rem !important;
        }

        section[data-testid="stSidebar"] [data-testid="stMetricValue"] {
            font-size: 1.5rem !important;
            line-height: 1.05 !important;
        }
        </style>
    """, unsafe_allow_html=True)


def render_sidebar() -> SidebarConfig:
    _inject_sidebar_styles()

    st.sidebar.markdown("# Trade Setup")
    setup_mode = st.sidebar.radio(
        "Setup Mode",
        ["Manual Entry", "Ticker Analysis"],
        horizontal=True,
    )
    model_type = st.sidebar.selectbox(
        "Model Type",
        ["Black-Scholes", "Binomial", "Monte Carlo"],
        index=0,
    )

    st.sidebar.markdown("## Contract Details")
    strike_price = st.sidebar.number_input("Strike Price", value=100.00, step=0.01, format="%.2f")

    default_trade_date = date.today()
    default_expiry_date = default_trade_date + timedelta(days=365)
    trade_date = st.sidebar.date_input("Trade Date", value=default_trade_date)
    expiry_date = st.sidebar.date_input("Expiry Date", value=default_expiry_date, min_value=trade_date)

    days_to_expiry, time_to_maturity = calculate_time_to_expiry(trade_date, expiry_date)
    days_col, years_col = st.sidebar.columns(2)
    days_col.metric("Days Remaining", f"{days_to_expiry}")
    years_col.metric("Year Fraction", f"{time_to_maturity:.4f}")

    st.sidebar.markdown("## Market Inputs")
    ticker = None
    if setup_mode == "Ticker Analysis":
        ticker = st.sidebar.text_input("Ticker", value="AAPL").strip().upper()
        try:
            spot_quote = get_yahoo_spot_price(ticker)
            current_price = spot_quote.price
            st.sidebar.caption(f"Spot auto-filled from {spot_quote.source}.")
            st.sidebar.metric("Spot Price", f"{current_price:.2f}")
        except Exception as exc:
            st.sidebar.warning(f"Yahoo Finance fetch failed: {exc}. Falling back to manual spot input.")
            current_price = st.sidebar.number_input("Spot Price", value=100.00, step=0.01, format="%.2f")
    else:
        current_price = st.sidebar.number_input("Spot Price", value=100.00, step=0.01, format="%.2f")

    volatility = st.sidebar.number_input("Volatility (σ)", value=0.20, step=0.01, format="%.2f")

    rate_source = st.sidebar.radio(
        "Risk-Free Rate Source",
        ["Yahoo Finance (Auto)", "Manual"] if setup_mode == "Ticker Analysis" else ["Manual", "Yahoo Finance (Auto)"],
        horizontal=True,
    )
    if rate_source == "Yahoo Finance (Auto)":
        try:
            rate_quote = get_yahoo_risk_free_rate(time_to_maturity)
            risk_free_rate = rate_quote.rate
            st.sidebar.caption(f"Auto-filled from {rate_quote.label}.")
            st.sidebar.metric("Risk-Free Rate", f"{risk_free_rate:.2%}")
        except Exception as exc:
            st.sidebar.warning(f"Yahoo Finance fetch failed: {exc}. Falling back to manual input.")
            risk_free_rate = st.sidebar.number_input("Risk-Free Rate", value=0.05, step=0.01, format="%.2f")
    else:
        risk_free_rate = st.sidebar.number_input("Risk-Free Rate", value=0.05, step=0.01, format="%.2f")

    model_params = {}
    if model_type == "Binomial":
        model_params["steps"] = st.sidebar.slider("Number of Steps", 10, 1000, 100)
        model_params["option_style"] = st.sidebar.selectbox("Option Style", ["European", "American"])
    elif model_type == "Monte Carlo":
        model_params["n_simulations"] = st.sidebar.slider("Number of Simulations", 1000, 50000, 10000)
        model_params["n_steps"] = st.sidebar.slider("Time Steps", 50, 500, 100)

    st.sidebar.markdown("## Advanced Settings")
    with st.sidebar.expander("Advanced Market Parameters", expanded=False):
        vol_term_structure = st.checkbox("Use Volatility Term Structure", False)
        if vol_term_structure:
            vol_3m = st.number_input("3-Month Volatility", value=volatility * 0.9, step=0.01, format="%.2f")
            vol_6m = st.number_input("6-Month Volatility", value=volatility, step=0.01, format="%.2f")
            vol_12m = st.number_input("12-Month Volatility", value=volatility * 1.1, step=0.01, format="%.2f")
            model_params["vol_term_structure"] = {
                0.25: vol_3m,
                0.5: vol_6m,
                1.0: vol_12m,
            }

        rate_term_structure = st.checkbox("Use Rate Term Structure", False)
        if rate_term_structure:
            rate_3m = st.number_input("3-Month Rate", value=risk_free_rate * 0.8, step=0.001, format="%.3f")
            rate_6m = st.number_input("6-Month Rate", value=risk_free_rate, step=0.001, format="%.3f")
            rate_12m = st.number_input("12-Month Rate", value=risk_free_rate * 1.2, step=0.001, format="%.3f")
            model_params["rate_term_structure"] = {
                0.25: rate_3m,
                0.5: rate_6m,
                1.0: rate_12m,
            }

        div_yield = st.number_input("Dividend Yield", value=0.0, step=0.001, format="%.3f", key="div_yield_slider")
        if div_yield > 0:
            model_params["dividend_yield"] = div_yield

        skew = st.slider("Volatility Skew", -0.2, 0.2, 0.0, 0.01, key="vol_skew_slider")
        if skew != 0:
            model_params["skew"] = skew

    return SidebarConfig(
        setup_mode=setup_mode,
        ticker=ticker,
        model_type=model_type,
        current_price=current_price,
        strike_price=strike_price,
        time_to_maturity=time_to_maturity,
        volatility=volatility,
        risk_free_rate=risk_free_rate,
        model_params=model_params,
    )
