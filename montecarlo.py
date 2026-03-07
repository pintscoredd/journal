import numpy as np
import pandas as pd

def block_bootstrap(data_array, block_size, num_samples, seed=None):
    if seed is not None:
        np.random.seed(seed)
        
    n = len(data_array)
    num_blocks = int(np.ceil(num_samples / block_size))
    
    # In block bootstrap, we pick random starting indices
    # up to n - block_size to ensure we have a full block
    max_start_idx = n - block_size
    if max_start_idx < 0:
        # If block_size > data length, just do normal replacement or truncate block
        max_start_idx = 0
        block_size = min(block_size, n)
        
    start_indices = np.random.randint(0, max_start_idx + 1, size=num_blocks)
    
    sampled = []
    for start_idx in start_indices:
        block = data_array[start_idx : start_idx + block_size]
        sampled.extend(block)
        
    return np.array(sampled[:num_samples])

def simulate_equity_paths(trades_df, num_simulations=10000, initial_capital=200, block_size=3, seed=None):
    if trades_df.empty:
        return np.array([[initial_capital] * num_simulations])
    
    pnls = trades_df['pnl'].values
    n_trades = len(pnls)
    
    if seed is not None:
        np.random.seed(seed)
        
    paths = np.zeros((n_trades + 1, num_simulations))
    paths[0, :] = initial_capital
    
    for i in range(num_simulations):
        sampled_pnls = block_bootstrap(pnls, block_size, n_trades)
        paths[1:, i] = initial_capital + np.cumsum(sampled_pnls)
        
    return paths

def calculate_risk_metrics(paths_array, ruin_level=0):
    num_sims = paths_array.shape[1]
    
    # Ruin probability
    min_capitals = np.min(paths_array, axis=0)
    ruined = np.sum(min_capitals <= ruin_level)
    prob_ruin = ruined / num_sims if num_sims > 0 else 0
    
    # Drawdowns
    # peak to trough
    running_max = np.maximum.accumulate(paths_array, axis=0)
    drawdowns = paths_array - running_max
    
    worst_drawdowns_per_sim = np.min(drawdowns, axis=0)
    expected_worst_dd = np.mean(worst_drawdowns_per_sim)
    
    return {
        "probability_of_ruin": prob_ruin,
        "expected_worst_drawdown": expected_worst_dd,
        "median_final_capital": np.median(paths_array[-1, :]),
        "mean_final_capital": np.mean(paths_array[-1, :]),
    }
