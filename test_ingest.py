import pytest
import pandas as pd
import os
from io import StringIO
from ingest import import_trades_csv, get_market_data

def test_import_trades_csv():
    csv_data = """Activity Date,Process Date,Settle Date,Instrument,Description,Trans Code,Quantity,Price,Amount
2024-05-15,2024-05-15,2024-05-16,SPY,SPY 05/15/2024 515.00 Call,Buy,1,1.50,-150.00
2024-05-15,2024-05-15,2024-05-16,SPY,SPY 05/15/2024 515.00 Call,Sell,1,2.50,250.00
"""
    stream = StringIO(csv_data)
    df = import_trades_csv(stream)
    assert not df.empty
    assert len(df) == 2
    assert "Activity Date" in df.columns
    assert "SPY" in df["Instrument"].values

def test_get_market_data_empty_fallback():
    # If network fails or ticker is garbage, should return empty dataframe gracefully
    df = get_market_data("INVALID_TICKER_XXX", interval="1m")
    assert isinstance(df, pd.DataFrame)
