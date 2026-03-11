from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math

import streamlit as st
import yfinance as yf


@dataclass(frozen=True)
class MarketRateQuote:
    rate: float
    label: str
    source: str


def calculate_time_to_expiry(trade_date: date, expiry_date: date) -> tuple[int, float]:
    days_to_expiry = max((expiry_date - trade_date).days, 0)
    return days_to_expiry, max(days_to_expiry / 365.0, 0.0)


YAHOO_TREASURY_TICKERS = {
    0.25: ("^IRX", "13W Treasury Bill"),
    5.00: ("^FVX", "5Y Treasury Yield"),
    10.00: ("^TNX", "10Y Treasury Yield"),
    30.00: ("^TYX", "30Y Treasury Yield"),
}


def _nearest_supported_tenor(time_to_maturity: float) -> float:
    maturities = sorted(YAHOO_TREASURY_TICKERS.keys())
    target = max(time_to_maturity, 0.0)
    return min(maturities, key=lambda maturity: abs(maturity - target))


@st.cache_data(ttl=3600, show_spinner=False)
def get_yahoo_risk_free_rate(time_to_maturity: float) -> MarketRateQuote:
    """Fetch a Treasury proxy from Yahoo Finance and normalize it to decimal form.

    Yahoo yield tickers generally return values like 4.25 for 4.25%, so the pricing
    stack needs them divided by 100.
    """
    nearest = _nearest_supported_tenor(time_to_maturity)
    symbol, label = YAHOO_TREASURY_TICKERS[nearest]
    history = yf.Ticker(symbol).history(period="5d", interval="1d")

    if history.empty:
        raise ValueError(f"No Yahoo Finance data returned for {symbol}")

    latest_close = float(history["Close"].dropna().iloc[-1])
    if math.isnan(latest_close):
        raise ValueError(f"Latest Yahoo Finance close is NaN for {symbol}")

    return MarketRateQuote(
        rate=latest_close / 100.0,
        label=label,
        source=f"Yahoo Finance {symbol}",
    )
