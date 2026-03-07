import os
import time
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pytz

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data_cache')
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_TTL_SECONDS = 3600  # 1 hour

def fetch_yfinance_with_retry(ticker: str, interval: str, period: str = "7d", retries: int = 3) -> pd.DataFrame:
    for attempt in range(retries):
        try:
            t = yf.Ticker(ticker)
            df = t.history(interval=interval, period=period)
            if not df.empty:
                # Ensure timezone aware index (UTC)
                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC')
                else:
                    df.index = df.index.tz_convert('UTC')
                return df
        except Exception as e:
            time.sleep(1 + attempt * 2)
    return pd.DataFrame()

def get_market_data(ticker: str, interval: str = "1m") -> pd.DataFrame:
    # Use SPX index standard representation for yfinance if ^SPX is passed
    yf_ticker = ticker
    
    safe_ticker = yf_ticker.replace("^", "")
    cache_file = os.path.join(CACHE_DIR, f"{safe_ticker}_{interval}.parquet")
    
    # Check cache TTL
    if os.path.exists(cache_file):
        mtime = os.path.getmtime(cache_file)
        if (time.time() - mtime) < CACHE_TTL_SECONDS:
            try:
                df = pd.read_parquet(cache_file)
                if not df.empty:
                    # In pyarrow/fastparquet sometimes index loses tz when written, ensure UTC
                    if df.index.tz is None:
                        df.index = df.index.tz_localize('UTC')
                    return df
            except Exception:
                pass # Fallback to fetch
                
    # Fetch fresh
    # Period for 1m bars in yfinance is max 7d
    period = "7d" if interval == "1m" else "60d"
    df = fetch_yfinance_with_retry(yf_ticker, interval, period)
    
    if not df.empty:
        try:
            df.to_parquet(cache_file)
        except Exception:
            pass # Cache write fail shouldn't break app
            
    return df

def get_vix_for_day(date_obj: datetime) -> float:
    # 1 day cache effectively
    cache_file = os.path.join(CACHE_DIR, "VIX_daily.parquet")
    df = pd.DataFrame()
    
    if os.path.exists(cache_file):
        mtime = os.path.getmtime(cache_file)
        # VIX daily data doesn't change much historically, but for current day we cache 1 hr
        if (time.time() - mtime) < CACHE_TTL_SECONDS:
            try:
                df = pd.read_parquet(cache_file)
            except (OSError, ValueError):
                pass
                
    if df.empty:
        df = fetch_yfinance_with_retry("^VIX", "1d", "60d")
        if not df.empty:
            df.to_parquet(cache_file)
            
    if df.empty:
        return 0.0
        
    # Find closest previous close
    target_dt = pd.to_datetime(date_obj, utc=True)
    valid_dates = df[df.index <= target_dt]
    if not valid_dates.empty:
        return valid_dates.iloc[-1]['Close']
    return 0.0

def _get_annualization_factor(interval: str) -> float:
    # Approx 252 trading days, 390 mins proper day
    if interval == "1m":
        return 252 * 390
    elif interval == "5m":
        return 252 * (390 / 5)
    elif interval == "15m":
        return 252 * (390 / 15)
    elif interval == "60m" or interval == "1h":
        return 252 * (390 / 60)
    return 252

def compute_realized_vol(df: pd.DataFrame, end_time: datetime, window_mins: int, interval: str="1m") -> float:
    if df.empty:
        return 0.0
        
    start_time = end_time - timedelta(minutes=window_mins)
    
    # Filter
    mask = (df.index >= pd.to_datetime(start_time, utc=True)) & (df.index <= pd.to_datetime(end_time, utc=True))
    window_df = df.loc[mask].copy()
    
    if len(window_df) < 2:
        return 0.0
        
    window_df['log_ret'] = np.log(window_df['Close'] / window_df['Close'].shift(1))
    std_dev = window_df['log_ret'].std()
    
    if pd.isna(std_dev):
        return 0.0
        
    ann_factor = np.sqrt(_get_annualization_factor(interval))
    return float(std_dev * ann_factor)

def import_trades_csv(file_path_or_buffer) -> pd.DataFrame:
    # A generic parser that takes Robinhood-like CSV and maps to our model
    df = pd.read_csv(file_path_or_buffer)
    # Expected user mapping done in UI. Here we return raw for UI to present.
    return df
