import pytest
import numpy as np
from spx_journal.quant import (
    bs_price, 
    bs_greeks, 
    implied_volatility, 
    kelly_fraction,
    compute_trade_scores
)

def test_bs_pricing_time_floor():
    # If T is 0, it should use the 1 minute floor (1/525600) and not raise exception
    # Intrinsic value of a 5000 call with spot 5050 is 50. Output should be very close to 50.
    price = bs_price(S=5050, K=5000, T=0, r=0.05, sigma=0.2, option_type='call')
    assert not np.isnan(price)
    assert price >= 50.0  # Should be at least intrinsic value
    assert price < 50.1   # Time premium should be negligible

def test_iv_inversion():
    # Normal case: Brent should work perfectly
    S, K, T, r = 5000, 5050, 7/365, 0.05
    target_vol = 0.15
    price = bs_price(S, K, T, r, target_vol, 'call')
    
    iv = implied_volatility(price, S, K, T, r, 'call')
    assert iv is not None
    assert pytest.approx(iv, abs=1e-4) == target_vol

def test_iv_climate_bounds():
    # If price implies vol > 6.0, it should clamp to 6.0
    S, K, T, r = 5000, 5000, 1/365, 0.05
    # A 1-day ATM call worth 500 implies massive vol
    iv_high = implied_volatility(500, S, K, T, r, 'call')
    assert iv_high == 6.0

    # If price implies vol < 0.01 (e.g., exactly intrinsic), it should clamp to 0.01
    iv_low = implied_volatility(0.001, S, K, T, r, 'call')
    assert iv_low == 0.01

def test_kelly_fraction():
    # f = (bp - q) / b
    # Let avg win = 200, avg loss = 100 => b = 2
    # win rate p = 0.5, q = 0.5
    # f = (2*0.5 - 0.5) / 2 = 0.5 / 2 = 0.25 (25%)
    f = kelly_fraction(avg_win=200, avg_loss=100, win_rate=0.5)
    assert pytest.approx(f, abs=1e-4) == 0.25

    # Negative edge -> f = 0
    f_neg = kelly_fraction(avg_win=100, avg_loss=100, win_rate=0.4)
    assert f_neg == 0.0

def test_trade_quality_components():
    # Mock some data
    result = compute_trade_scores(
        theoretical_edge=0.05,
        vol_ratio=1.0,  # perfect
        delta=0.3,
        gamma_exposure=500,
        execution_slippage=0.01,
        entry_time_expectancy=1.5,
        hold_time_minutes=15
    )
    
    assert "volatility_edge_score" in result
    assert "execution_score" in result
    assert "timing_score" in result
    assert "risk_reward_score" in result
    assert "total_score" in result
    assert 0 <= result["total_score"] <= 100
