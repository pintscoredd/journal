# SPX 0DTE Journal — Project Analysis

Senior software engineer & quantitative developer review. Focus: architecture, performance, math/statistics, signal quality, bugs, free-API features, refactoring.

---

## 1. Architecture and Modularity

### Strengths
- Clear separation: `db.py` (persistence), `quant.py` (pricing/scoring), `ingest.py` (market data), `montecarlo.py` (risk), `ai_adapter.py` (LLM), `ui.py` (Streamlit).
- Single entrypoint `app.py` with page routing; `ui` holds all page renderers.

### Issues
- **UI owns too much logic**: Trade replay (VWAP/EMA, chart HTML), data loading, and session handling live in `ui.py`. Replay and “enrichment” should live in a service layer (e.g. `services/replay.py`, `services/enrichment.py`).
- **No service layer**: Business flow (save trade → enrich with market data → compute IV/Greeks → compute scores → save) is not centralized. Manual entry and CSV ingest don’t share one pipeline.
- **DB session lifecycle**: `get_session()` creates a new engine every time (`get_engine()` is called in each `get_session()`). For SQLite this is usually fine; for Supabase/Postgres you should reuse one engine (and optionally connection pool) per process.
- **Config split**: DB choice comes from `os.environ` in `db.py`, while the UI has a “Use Supabase” checkbox in `st.session_state` that never affects `db.get_engine()`. So the checkbox is misleading.

**Recommendation**: Introduce a small service layer (e.g. `trade_service.py`) that: (1) takes raw trade input, (2) fetches market data and VIX, (3) computes T, IV, Greeks, scores, (4) persists the trade. Both “Manual Quick Entry” and CSV ingest should call this.

---

## 2. Performance Bottlenecks

- **Repeated engine creation**: `db.get_session()` → `get_engine()` each time. Cache the engine (e.g. with `@st.cache_resource` or a module-level engine) so only one engine exists per process.
- **Monte Carlo**: 10k paths × N trades with Python loop (`for i in range(num_simulations)`). Can be vectorized: draw all block starts in one go, build a 2D array of PnL paths, then `np.cumsum(..., axis=0)`.
- **Trade viewer**: `get_all_trades_df()` runs on every interaction and builds a full DataFrame from ORM. For large histories, consider pagination or loading only the selected trade.
- **Replay chart**: Building `chart_data` with a Python loop over `plot_md.iterrows()` is slow. Prefer vectorized construction, e.g. `plot_md.assign(time=plot_md.index.astype(np.int64)//10**9)` and then `.to_dict('records')`.
- **AI cache**: Cache key includes full prompt + quant JSON. Good. No TTL on `AICache`; old entries never expire (storage grows; consider periodic cleanup or TTL).
- **Market data**: Parquet cache with 1h TTL is good. `fetch_cached_market_data` uses `@st.cache_data(ttl=3600)` on top, so Streamlit and file cache can both be used; ensure cache keys match (ticker + interval).

---

## 3. Mathematical / Statistical Improvements

### Time-to-expiry (T) for 0DTE
- `quant.py` uses `MIN_T = 1/525600` (1 minute in years). For 0DTE, T should be computed from entry/exit time to market close (e.g. 16:00 ET), not a fixed floor, so that IV and Greeks reflect same-day expiry. Recommendation: add a helper `years_to_expiry(entry_dt, exit_dt, expiry_date, market_close_et="16:00")` and use it in IV/Greeks and in scoring.

### Realized volatility
- `ingest.compute_realized_vol` uses log returns and annualization. For 0DTE, intraday (e.g. 5m/15m) realized vol is appropriate. Consider also reporting a “vol of vol” or using a 5m/15m ratio as a regime proxy (e.g. mean-reversion vs trend).

### Vol ratio and scoring
- `compute_trade_scores` uses a step function for `vol_ratio` (0.9–1.1 → 100, 0.7–1.3 → 70, else 30). A smooth curve (e.g. Gaussian-style decay from 1.0) would avoid discontinuities and better reflect “slightly rich/cheap” vs “way off”.
- **Unused inputs**: `theoretical_edge`, `delta`, and `hold_time_minutes` are accepted but not used. Either remove them or incorporate (e.g. hold_time for time-decay penalty, delta for directional bias check).

### Kelly and sizing
- `kelly_fraction` is correct. `recommended_contracts` is on the model but never computed or displayed. You could derive it from Kelly and current capital/risk (e.g. `kelly_fraction * capital / (avg_loss per contract)` then round).

### Monte Carlo
- Block bootstrap preserves order and autocorrelation, which is good for PnL series. For very small samples (e.g. &lt; 20 trades), block_size=3 may be large relative to N; consider `block_size = max(1, min(3, n//5))` or similar.
- `calculate_risk_metrics`: “Expected worst drawdown” is reported as a single number; consider also percentiles (e.g. 5th/95th) of final capital and of max drawdown for better risk communication.

---

## 4. Signal Quality Improvements

- **VIX at entry**: `get_vix_for_day` returns the latest close ≤ trade date. For same-day entries, VIX at exact entry time would require intraday VIX (not free from Yahoo). As a free improvement, you could add VIX at previous day close and optionally a “VIX term” (e.g. VIX9d if available) for 0DTE context.
- **VWAP**: Replay uses full-window VWAP. For 0DTE, “VWAP from open” (or from 9:30) is more standard. Ensure the window starts at market open when data exists.
- **EMA rejection logic**: The AI prompt asks for “rejection from EMAs” and “prior 3 candles.” The logic is in the LLM, not in code. You could add a small signal module that returns a boolean or score: “price within X% of EMA5/14/25 at entry” and “prior N candles show wick away from EMA,” and pass that into the prompt or into scoring.
- **Execution score**: Currently `100 * exp(-50 * |slippage|)`. You don’t have a consistent “theoretical best” price (e.g. from 1m bar). Enrichment could compute “best bid/ask or mid in the 1m bar of entry” and store a slippage field, then use it in scoring.

---

## 5. Bugs and Logical Flaws

### Critical
1. **Missing import in `ui.py`**: `Secret` is used in `render_settings()` (Export Encrypted Snippet) but not imported from `db`. This raises `NameError` when the button is clicked. **Fix**: `from db import get_session, Trade, Secret`.
2. **`expiry` never set**: Manual trade entry creates `Trade(...)` without `expiry`. IV and any expiry-based logic will fail or use None. **Fix**: Add expiry to the form (default to trade date for 0DTE) and set `expiry=trade_date` (or parsed expiry).
3. **Supabase checkbox ineffective**: `db.get_engine()` reads only `os.environ["USE_SUPABASE"]`. The Settings checkbox sets `st.session_state["use_supabase"]`. So toggling the checkbox doesn’t switch DB. **Fix**: Either (a) set `os.environ["USE_SUPABASE"] = "true"/"false"` when the user saves settings (with a warning that app restart may be needed), or (b) pass a config object into `get_engine(config)` and have the UI pass session_state into that config (requires refactoring how/where the engine is created).

### Moderate
4. **Quant pipeline never run**: New trades (manual or CSV) are saved without calling `implied_volatility`, `bs_greeks`, or `compute_trade_scores`. So `implied_vol_entry`, `delta_entry`, `trade_quality_score`, etc. stay NULL. **Fix**: After saving a trade, call an enrichment function that fetches underlying price (and optionally VIX), computes T, IV, Greeks, scores, then updates the trade row.
5. **Bare `except` in `utils.safely_divide`**: Catches all exceptions. Prefer `except (ZeroDivisionError, TypeError):` (or similar) so programming errors aren’t swallowed.
6. **Bare `except` in `ingest.get_vix_for_day`**: Same; narrow to specific exceptions (e.g. `OSError`, `ValueError`).
7. **Monte Carlo with 1 trade**: `block_bootstrap` with `n=1` gives `max_start_idx=0`, `block_size=1`, one block repeated; paths are identical. Acceptable but could document or short-circuit “N&lt;2 → no MC” in the UI.

### Minor
8. **`parse_time`**: Fallback returns 07:00:00; invalid input is silently normalized. Consider logging or returning an error so the user knows the time was interpreted.
9. **CSV “Save to DB (Dummy Implementation)”**: Button says ingestion “would” process; it doesn’t. Either implement full CSV→enrich→save or rename to “Preview only (not saved).”
10. **Weekly report**: Uses `mock_agg_data` instead of real aggregates from `get_all_trades_df()`. Replace with real expectancy, win_rate, total_pnl from the last 7 days (or configurable window).

---

## 6. Feature Improvements Using Free APIs

- **Yahoo Finance (already used)**: You already use 1m/5m and VIX. Could add: VIX open/close for the day, or another index (e.g. SPX vs ETF) for basis.
- **FRED (free)**: Risk-free rate: e.g. DFF or DTB3 for `r` in Black–Scholes instead of hardcoding. Could cache daily.
- **Alternative free data**: Alpha Vantage (free tier), Polygon (free tier) for redundancy or backup when Yahoo fails. Abstract “market data provider” behind a small interface so you can swap.
- **Earnings / calendar**: No earnings calendar. For 0DTE, avoiding earnings days improves signal. Consider a free earnings API or scraping (with care) to tag “earnings day” on the trade date.
- **IV surface proxy**: You store `vol_skew` and `vol_term_slope` but don’t populate them. With only one option per trade you can’t build a full surface; you could approximate “skew” by comparing IV of the traded strike to ATM (e.g. from another strike’s IV if you had it, or leave as NULL until you have multi-strike data).

---

## 7. Refactoring Opportunities

- **Extract replay into a module**: Move “fetch market data → slice window → compute VWAP/EMAs → build chart_data + entry_context” into e.g. `services/replay.py`. UI only calls `get_replay_data(trade_id)` and renders the chart.
- **Unify trade creation**: One function `create_trade_from_manual(...)` and `create_trades_from_csv(df)` that both call a shared `enrich_and_save_trade(trade_dict)` (or per-row for CSV).
- **Config object**: Replace scattered `os.environ` and `st.session_state` with a single `AppConfig` (from env + secrets + session_state) passed into DB and services. Easier to test and to support “Use Supabase” from the UI.
- **Type hints**: Add type hints across `quant.py`, `ingest.py`, `montecarlo.py`, and service layer for better IDE support and clarity.
- **Constants**: Move `MIN_T`, `CACHE_TTL_SECONDS`, default block_size, score weights, etc. to a `config.py` or `constants.py` so they’re easy to tune and document.
- **Tests**: Add integration test that: creates a trade, runs enrichment (with mocked yfinance), and asserts that `implied_vol_entry` and `trade_quality_score` are set. Add test for “Export Encrypted Snippet” (with mocked Secret table) to prevent regression on the `Secret` import.

---

## 8. Suggested Code Changes (Summary)

| Priority | Change |
|----------|--------|
| Critical | Add `Secret` to `from db import ...` in `ui.py`. |
| Critical | Set `expiry` on manual trade (e.g. trade date); add expiry to form if you want user override. |
| Critical | Wire “Use Supabase” to DB: set env on save or refactor engine to accept config. |
| High | Implement “enrich after save”: fetch market data, compute T, IV, Greeks, scores, update trade. |
| High | Use or remove `theoretical_edge`, `delta`, `hold_time_minutes` in `compute_trade_scores`. |
| Medium | Cache `get_engine()` (e.g. `@st.cache_resource`) and reuse in `get_session()`. |
| Medium | Vectorize Monte Carlo inner loop (all paths in one go). |
| Medium | Replace mock_agg_data in weekly report with real aggregates from DB. |
| Low | Smooth vol_ratio scoring; add T from market close for 0DTE; narrow `except` clauses. |

---

## 9. Example Code Snippets

### 9.1 Fix: Import `Secret` in `ui.py`
```python
from db import get_session, Trade, Secret
```

### 9.2 Fix: Set `expiry` and optional enrichment in manual entry
```python
# In render_new_trade(), when creating new_trade:
new_trade = Trade(
    ...
    expiry=trade_date,  # 0DTE: expiry is trade date
    ...
)
session.add(new_trade)
session.commit()
# Optional: trigger enrichment job or sync call to compute IV/Greeks/scores
```

### 9.3 Improved vol_ratio scoring (smooth)
```python
# In quant.compute_trade_scores, replace step function with:
deviation = abs(vol_ratio - 1.0)
vol_score = 100.0 * np.exp(-3.0 * deviation)  # smooth decay from 1.0
vol_score = min(max(vol_score, 0), 100)
```

### 9.4 Vectorized Monte Carlo (sketch)
```python
def simulate_equity_paths_vec(trades_df, num_simulations=10000, initial_capital=200, block_size=3, seed=None):
    if trades_df.empty:
        return np.array([[initial_capital] * num_simulations])
    pnls = trades_df['pnl'].values
    n_trades = len(pnls)
    if seed is not None:
        np.random.seed(seed)
    max_start = max(0, n_trades - block_size)
    num_blocks = int(np.ceil(n_trades / block_size))
    # Shape (num_simulations, num_blocks); each row is start indices for one path
    starts = np.random.randint(0, max_start + 1, size=(num_simulations, num_blocks))
    # Build PnL matrix: (num_simulations, n_trades) by taking blocks
    # (e.g. with advanced indexing or a loop over block index)
    sampled = np.zeros((num_simulations, n_trades))
    for b in range(num_blocks):
        start = starts[:, b]
        for j in range(block_size):
            col = b * block_size + j
            if col < n_trades:
                sampled[:, col] = pnls[start + j]
    paths = np.column_stack([np.full(num_simulations, initial_capital), initial_capital + np.cumsum(sampled, axis=1)])
    return paths.T  # (n_trades+1, num_simulations)
```

### 9.5 Cache DB engine
```python
# db.py
@functools.lru_cache(maxsize=1)
def get_engine():
    ...
    return create_engine(database_url)
# Then get_session() uses get_engine() as now; only one engine per process.
```
(Note: with `lru_cache`, `get_engine()` arguments must be hashable; current implementation has no args, so it’s fine. If you later add config, use a hashable config or a cache key.)

### 9.6 Real weekly aggregates in reports
```python
# In render_reports(), replace mock_agg_data with:
df = get_all_trades_df()
if not df.empty:
    df = df.sort_values('entry_time')
    week_start = df['entry_time'].max() - pd.Timedelta(days=7)
    week_df = df[df['entry_time'] >= week_start]
    if not week_df.empty:
        mock_agg_data = {
            "expectancy": week_df['pnl'].mean(),
            "win_rate": (week_df['pnl'] > 0).mean(),
            "total_pnl": week_df['pnl'].sum()
        }
    else:
        mock_agg_data = {"expectancy": 0, "win_rate": 0, "total_pnl": 0}
else:
    mock_agg_data = {"expectancy": 0, "win_rate": 0, "total_pnl": 0}
```

---

End of analysis. Implement critical and high-priority items first, then performance and refactors.
