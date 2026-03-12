"""
Enrich a saved trade with market data, IV, Greeks, and trade quality scores.
Call after saving a trade so that trade_quality_score, implied_vol_*, delta_*, etc. are populated.
"""
from datetime import datetime
import pandas as pd

from db import get_session, Trade
from ingest import get_market_data, get_vix_for_day, compute_realized_vol
from quant import (
    implied_volatility,
    bs_greeks,
    compute_trade_scores,
    MIN_T,
)

# Risk-free rate proxy (could be fetched from FRED in the future)
DEFAULT_R = 0.05


def _years_from_minutes(minutes: float) -> float:
    # 252 trading days, 390 minutes per day
    return max(minutes / (252 * 390), MIN_T)


def _get_underlying_at_time(md: pd.DataFrame, dt: datetime):
    """Return (S, timestamp) for the bar closest to dt. dt and md.index should be timezone-aware."""
    if md.empty:
        return None, None
    target = pd.to_datetime(dt, utc=True)
    if md.index.tz is None:
        md = md.copy()
        md.index = md.index.tz_localize("UTC")
    idx = md.index.get_indexer([target], method="nearest")[0]
    row = md.iloc[idx]
    return float(row["Close"]), md.index[idx]


def enrich_trade(trade_id: int) -> str | None:
    """
    Load trade by id, fetch market data, compute IV/Greeks/scores, update and commit.
    Returns None on success, or an error message string.
    """
    session = get_session()
    try:
        trade = session.query(Trade).filter_by(id=trade_id).first()
        if not trade:
            return "Trade not found."

        ticker = trade.ticker or "^SPX"
        md = get_market_data(ticker, "1m")
        if md.empty:
            return "No market data for ticker."

        entry_dt = pd.to_datetime(trade.entry_time)
        exit_dt = pd.to_datetime(trade.exit_time)
        if entry_dt.tzinfo is None:
            entry_dt = entry_dt.tz_localize("UTC")
        if exit_dt.tzinfo is None:
            exit_dt = exit_dt.tz_localize("UTC")

        S_entry, _ = _get_underlying_at_time(md, entry_dt)
        S_exit, _ = _get_underlying_at_time(md, exit_dt)
        if S_entry is None:
            return "Could not resolve underlying price at entry."
        if S_exit is None:
            S_exit = S_entry

        trade.underlying_entry_price = S_entry
        trade.underlying_exit_price = S_exit

        # Time to expiry: use hold duration as proxy for 0DTE (remaining life)
        hold_minutes = (exit_dt - entry_dt).total_seconds() / 60.0
        trade.hold_time_minutes = hold_minutes
        T = _years_from_minutes(hold_minutes)

        r = DEFAULT_R
        K = float(trade.strike or 0)
        option_type = (trade.option_type or "call").lower()
        entry_price = float(trade.entry_price or 0)
        exit_price = float(trade.exit_price or 0)

        # VIX and realized vol for vol_ratio
        vix = get_vix_for_day(entry_dt)
        trade.vix_at_entry = vix
        vix_annual = (vix / 100.0) if vix else 0.20
        real_vol_5m = compute_realized_vol(md, entry_dt, 5, "1m")
        real_vol_15m = compute_realized_vol(md, entry_dt, 15, "1m")
        trade.realized_vol_5m = real_vol_5m
        trade.realized_vol_15m = real_vol_15m

        # IV at entry and exit
        iv_entry = implied_volatility(entry_price, S_entry, K, T, r, option_type)
        # At exit, remaining time is negligible for 0DTE; use 1-min floor for numerical stability
        T_exit = _years_from_minutes(1.0)
        iv_exit = implied_volatility(exit_price, S_exit, K, T_exit, r, option_type) if exit_price else None
        trade.implied_vol_entry = iv_entry
        trade.implied_vol_exit = iv_exit

        # Greeks at entry
        if iv_entry is not None:
            greeks_e = bs_greeks(S_entry, K, T, r, iv_entry, option_type)
            trade.delta_entry = greeks_e["delta"]
            trade.gamma_entry = greeks_e["gamma"]
            trade.theta_entry = greeks_e["theta"]
            trade.vega_entry = greeks_e["vega"]
        # Greeks at exit (use same T for simplicity)
        if iv_exit is not None:
            greeks_x = bs_greeks(S_exit, K, T, r, iv_exit, option_type)
            trade.delta_exit = greeks_x["delta"]
            trade.gamma_exit = greeks_x["gamma"]
            trade.theta_exit = greeks_x["theta"]
            trade.vega_exit = greeks_x["vega"]

        # Vol ratio: IV / realized or IV / VIX
        ref_vol = vix_annual if vix_annual > 0 else 0.20
        vol_ratio = (iv_entry / ref_vol) if iv_entry and ref_vol else 1.0
        trade.vol_ratio = vol_ratio

        # Scores calculation proxy variables
        import numpy as np
        execution_slippage = abs((trade.delta_entry or 0.5) * 0.05)
        pst_time = entry_dt.tz_convert('America/Los_Angeles')
        time_val = pst_time.hour + pst_time.minute / 60.0
        # Better expectancy early in the session
        if 6.5 <= time_val <= 8.5:
            entry_time_expectancy = 0.9
        elif 8.5 < time_val <= 11.5:
            entry_time_expectancy = 0.6
        else:
            entry_time_expectancy = 0.3
            
        gamma_exposure = abs((trade.gamma_entry or 0) * (trade.contracts or 1) * 100)
        scores = compute_trade_scores(
            theoretical_edge=0.05,
            vol_ratio=vol_ratio,
            delta=trade.delta_entry or 0.5,
            gamma_exposure=gamma_exposure,
            execution_slippage=execution_slippage,
            entry_time_expectancy=entry_time_expectancy,
            hold_time_minutes=hold_minutes,
        )
        trade.entry_execution_score = scores["execution_score"]
        trade.volatility_edge_score = scores["volatility_edge_score"]
        trade.timing_score = scores["timing_score"]
        trade.risk_reward_score = scores["risk_reward_score"]
        trade.trade_quality_score = scores["total_score"]

        session.commit()
        return None
    except Exception as e:
        session.rollback()
        return str(e)
    finally:
        session.close()
