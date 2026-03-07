import pytest
import numpy as np
import pandas as pd
from montecarlo import simulate_equity_paths, block_bootstrap, calculate_risk_metrics

def test_block_bootstrap():
    returns = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    block_size = 3
    num_samples = 15
    
    samples = block_bootstrap(returns, block_size, num_samples, seed=42)
    assert len(samples) >= num_samples
    
    # Verify blocks are contiguous (e.g., if we see a 3, the next should be 4 unless it was the end of a block)
    # Just checking the function runs and produces expected length
    assert isinstance(samples, np.ndarray)

def test_simulate_equity_paths():
    trades = pd.DataFrame({
        'pnl': [50, -20, 10, -5, 30, -100, 40, 20, -10, 50]
    })
    
    paths = simulate_equity_paths(trades, num_simulations=100, initial_capital=200, block_size=3, seed=42)
    
    assert paths.shape[1] == 100
    # Expected number of steps per path is len(trades) by default or can be scaled, 
    # but the implementation should produce at least a valid array of shapes
    assert paths.shape[0] == len(trades) + 1  # includes initial capital

def test_calculate_risk_metrics():
    paths = np.array([
        [200, 250, 300, 350],
        [200, 150, 80, 0],    # Ruin
        [200, 180, 220, 250]
    ]).T
    
    metrics = calculate_risk_metrics(paths, ruin_level=0)
    assert "probability_of_ruin" in metrics
    assert "expected_worst_drawdown" in metrics
    
    assert pytest.approx(metrics["probability_of_ruin"], abs=1e-2) == 1/3
    assert metrics["expected_worst_drawdown"] < 0
