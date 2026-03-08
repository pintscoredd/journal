import numpy as np
import scipy.stats as stats
from scipy.optimize import brentq, newton, bisect

MIN_T = 1 / 525600  # 1 minute minimum time floor

def _d1_d2(S, K, T, r, sigma):
    T = max(T, MIN_T)
    sigma = max(sigma, 1e-8)  # Prevent divide by zero
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2

def bs_price(S, K, T, r, sigma, option_type='call'):
    T = max(T, MIN_T)
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    
    if option_type.lower() == 'call':
        price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
        
    return price

def bs_greeks(S, K, T, r, sigma, option_type='call'):
    T = max(T, MIN_T)
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    
    # PDF of d1
    nd1 = stats.norm.pdf(d1)
    
    # Delta
    if option_type.lower() == 'call':
        delta = stats.norm.cdf(d1)
    else:
        delta = stats.norm.cdf(d1) - 1.0
        
    # Gamma (same for both)
    gamma = nd1 / (S * sigma * np.sqrt(T))
    
    # Theta
    # Formula uses T in years, return value is per year, common to divide by 365 for daily
    if option_type.lower() == 'call':
        theta = (- (S * sigma * nd1) / (2 * np.sqrt(T)) 
                 - r * K * np.exp(-r * T) * stats.norm.cdf(d2))
    else:
        theta = (- (S * sigma * nd1) / (2 * np.sqrt(T)) 
                 + r * K * np.exp(-r * T) * stats.norm.cdf(-d2))
                 
    # Vega
    vega = S * np.sqrt(T) * nd1
    
    return {
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega
    }

def implied_volatility(price, S, K, T, r, option_type='call'):
    T = max(T, MIN_T)
    
    # Avoid intrinsic issues and deep OTM issues.
    if option_type.lower() == 'call':
        intrinsic = max(0.0, S - K * np.exp(-r * T))
    else:
        intrinsic = max(0.0, K * np.exp(-r * T) - S)
        
    if price < intrinsic:
        return 0.01  # Price is below intrinsic, vol is effectively ~0, clamp to floor
        
    def objective(sigma):
        return bs_price(S, K, T, r, sigma, option_type) - price

    iv = None
    
    # IV solver fallback chain
    # 1. Try Brent
    try:
        iv = brentq(objective, 1e-4, 10.0, maxiter=500)
    except Exception:
        # 2. Try Newton
        try:
            # Need fprime (vega)
            def fprime(sigma):
                return bs_greeks(S, K, T, r, sigma, option_type)["vega"]
            iv = newton(objective, x0=0.2, fprime=fprime, maxiter=200)
        except Exception:
            # 3. Try Bisection Fallback
            try:
                iv = bisect(objective, 1e-4, 10.0, maxiter=200)
            except Exception:
                pass
                
    if iv is None or np.isnan(iv):
        return None
        
    return min(max(iv, 0.01), 6.0)

def kelly_fraction(avg_win, avg_loss, win_rate):
    """
    f = (bp - q) / b
    where b = avg_win / avg_loss
    p = win_rate
    q = 1 - p
    """
    if avg_loss == 0 or np.isnan(avg_loss):
        return 0.0  # safe guard
    b = abs(avg_win / avg_loss)
    p = win_rate
    q = 1 - p
    
    if b == 0:
        return 0.0
        
    f = (b * p - q) / b
    return max(0.0, min(1.0, f))

def compute_trade_scores(
    theoretical_edge, 
    vol_ratio, 
    delta, 
    gamma_exposure, 
    execution_slippage,
    entry_time_expectancy,
    hold_time_minutes
):
    """
    Computes component scores 0-100 and a weighted final score.
    Weights: 
    volatility_edge_score: 35%
    execution_score: 25%
    timing_score: 25%
    risk_reward_score: 15%
    """
    # 1. Volatility Edge Score (ideal vol_ratio is 0.9..1.1)
    if vol_ratio < 0.7 or vol_ratio > 1.3:
        vol_score = 30.0
    elif 0.9 <= vol_ratio <= 1.1:
        vol_score = 100.0
    else:
        vol_score = 70.0  # in between

    # 2. Execution Score (scale based on relative slippage to theoretical edge)
    # Penalize negative edge (Slippage relative to 1m best bars)
    # Simple proxy:
    exec_score = 100 * np.exp(-50 * abs(execution_slippage))
    exec_score = min(max(exec_score, 0), 100)
    
    # 3. Timing Score based on expectancies
    time_score = min(max(50 + 50 * entry_time_expectancy, 0), 100)
    
    # 4. Risk / Reward Profile Score (Greeks Profile)
    # E.g. gamma_exposure exploding penalizes the score
    risk_score = 100 * np.exp(-0.001 * abs(gamma_exposure))
    risk_score = min(max(risk_score, 0), 100)
    
    total = (0.35 * vol_score) + (0.25 * exec_score) + (0.25 * time_score) + (0.15 * risk_score)
    
    return {
        "volatility_edge_score": vol_score,
        "execution_score": exec_score,
        "timing_score": time_score,
        "risk_reward_score": risk_score,
        "total_score": round(total, 1)
    }

