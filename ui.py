import json
import base64
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

from db import get_session, Trade
from secrets_store import store_api_key, get_master_key, encrypt_key
from ingest import get_market_data, import_trades_csv
from ai_adapter import AIAdapter
from montecarlo import simulate_equity_paths, calculate_risk_metrics

# --- CACHING ---
@st.cache_data(ttl=3600)
def fetch_cached_market_data(ticker, interval):
    return get_market_data(ticker, interval)

@st.cache_data
def run_cached_monte_carlo(trades_df, num_sims, initial, block_size):
    if trades_df.empty:
        return None, None
    paths = simulate_equity_paths(trades_df, num_sims, initial, block_size)
    metrics = calculate_risk_metrics(paths)
    return paths, metrics

# --- DATA FETCH ---
def get_all_trades_df():
    session = get_session()
    try:
        trades = session.query(Trade).order_by(Trade.entry_time.asc()).all()
        if not trades:
            return pd.DataFrame()
        # Convert to records
        data = []
        for t in trades:
            d = t.__dict__.copy()
            d.pop('_sa_instance_state', None)
            data.append(d)
        return pd.DataFrame(data)
    finally:
        session.close()

# --- PAGES ---
def render_dashboard():
    st.header("Dashboard")
    df = get_all_trades_df()
    if df.empty:
        st.info("No trades found. Please import some trades.")
        return

    # High level KPIs
    df['cum_pnl'] = df['pnl'].cumsum()
    total_pnl = df['pnl'].sum()
    win_rate = len(df[df['pnl'] > 0]) / len(df) if len(df) > 0 else 0
    avg_win = df[df['pnl'] > 0]['pnl'].mean() if len(df[df['pnl'] > 0]) > 0 else 0
    avg_loss = df[df['pnl'] <= 0]['pnl'].mean() if len(df[df['pnl'] <= 0]) > 0 else 0
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total PnL", f"${total_pnl:.2f}")
    c2.metric("Win Rate", f"{win_rate*100:.1f}%")
    c3.metric("Avg Win", f"${avg_win:.2f}")
    c4.metric("Avg Loss", f"${avg_loss:.2f}")

    # Equity Curve
    fig = px.line(df, x='entry_time', y='cum_pnl', title='Equity Curve')
    st.plotly_chart(fig, use_container_width=True)
    
    # Expectancy by 15m bucket
    st.subheader("Rolling Expectancy by Time Of Day (15m Bins)")
    df['time_bucket'] = pd.to_datetime(df['entry_time']).dt.floor('15T').dt.time
    expectancy_df = df.groupby('time_bucket')['pnl'].mean().reset_index()
    # convert time_bucket to string for plotting
    expectancy_df['time_bucket_str'] = expectancy_df['time_bucket'].apply(lambda x: x.strftime("%H:%M"))
    fig2 = px.bar(expectancy_df, x='time_bucket_str', y='pnl', title='Average PnL by 15m Entry Window')
    st.plotly_chart(fig2, use_container_width=True)
    
    # Monte Carlo Sample
    st.subheader("Monte Carlo Simulation (Block Bootstrap)")
    num_sims = st.session_state.get('mc_sims', 1000)
    paths, metrics = run_cached_monte_carlo(df, num_sims, initial=st.session_state.get('capital', 200), block_size=3)
    if paths is not None:
        cr1, cr2 = st.columns(2)
        cr1.metric("Probability of Ruin", f"{metrics['probability_of_ruin']*100:.1f}%")
        cr2.metric("Expected Worst Drawdown", f"${metrics['expected_worst_drawdown']:.2f}")
        
        # Plot a subset of paths
        plot_paths = paths[:, :min(50, num_sims)]
        fig_mc = go.Figure()
        for i in range(plot_paths.shape[1]):
            fig_mc.add_trace(go.Scatter(y=plot_paths[:, i], mode='lines', line=dict(width=1, color='rgba(0,0,255,0.1)'), showlegend=False))
        fig_mc.update_layout(title=f"Sample 50 Equity Paths (from {num_sims})")
        st.plotly_chart(fig_mc, use_container_width=True)

def render_new_trade():
    st.header("Ingest Trades")
    
    file = st.file_uploader("Upload CSV (Robinhood format example)", type=["csv"])
    if file:
        df = import_trades_csv(file)
        st.dataframe(df)
        if st.button("Save to DB (Dummy Implementation)"):
            st.success("Ingestion logic would process this and enrich quant DB elements here.")
    
    st.subheader("Manual Quick Entry")
    with st.form("manual_entry"):
        col1, col2 = st.columns(2)
        ticker = col1.text_input("Ticker", "^SPX")
        option_type = col2.selectbox("Option Type", ["call", "put"])
        
        col3, col4 = st.columns(2)
        strike = col3.number_input("Strike", value=5000.0, step=5.0)
        contracts = col4.number_input("Contracts", min_value=1, value=1, step=1)
        
        c1, c2 = st.columns(2)
        entry_price = c1.number_input("Entry Price", value=5.0, step=0.1)
        exit_price = c2.number_input("Exit Price", value=6.0, step=0.1)
        
        t1, t2 = st.columns(2)
        trade_date = t1.date_input("Trade Date", value=datetime.today())
        # Default to regular market hours to ensure we get data
        entry_time_input = t2.time_input("Entry Time (PST)", value=datetime.strptime("07:00", "%H:%M").time())
        
        submitted = st.form_submit_button("Save Trade")
        if submitted:
            import uuid
            session = get_session()
            try:
                # Combine date and time
                entry_dt = datetime.combine(trade_date, entry_time_input)
                # Ensure UTC awareness for yfinance lookup
                entry_dt_utc = pd.to_datetime(entry_dt).tz_localize('America/Los_Angeles').tz_convert('UTC')
                exit_dt_utc = entry_dt_utc + timedelta(minutes=15) # default pseudo hold time
                
                new_trade = Trade(
                    trade_uuid=str(uuid.uuid4()),
                    ticker=ticker,
                    option_type=option_type,
                    strike=strike,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    entry_time=entry_dt_utc,
                    exit_time=exit_dt_utc,
                    contracts=contracts,
                    pnl=(exit_price - entry_price) * 100 * contracts
                )
                session.add(new_trade)
                session.commit()
                st.success(f"Trade for {ticker} successfully saved: ${new_trade.pnl:.2f} PnL!")
            except Exception as e:
                st.error(f"Error saving trade: {e}")
            finally:
                session.close()

def render_trade_viewer():
    st.header("Trade Viewer & Replay")
    df = get_all_trades_df()
    if df.empty:
        st.info("No trades to display.")
        return
        
    trade_id = st.selectbox("Select Trade", df['id'].values, format_func=lambda x: f"Trade {x} | {df[df['id']==x]['entry_time'].values[0]}")
    selected = df[df['id'] == trade_id].iloc[0]
    
    st.write(f"**Score**: {selected.get('trade_quality_score', 'N/A')}/100")
    st.write(f"PnL: ${selected['pnl']}")
    
    # Replay Chart
    st.subheader("Replay Chart")
    try:
        md = fetch_cached_market_data(selected['ticker'], "1m")
        if not md.empty:
            # Parse db time safely
            db_entry = pd.to_datetime(selected['entry_time'])
            db_exit = pd.to_datetime(selected['exit_time'])
            if db_entry.tzinfo is None:
                db_entry = db_entry.tz_localize('UTC')
            if db_exit.tzinfo is None:
                db_exit = db_exit.tz_localize('UTC')
                
            start_date = db_entry - timedelta(hours=1)
            end_date = db_exit + timedelta(hours=1)
            mask = (md.index >= start_date) & (md.index <= end_date)
            plot_md = md.loc[mask].copy()
            
            if not plot_md.empty:
                # Calculate VWAP
                plot_md['typ'] = (plot_md['High'] + plot_md['Low'] + plot_md['Close']) / 3
                plot_md['vwap'] = (plot_md['typ'] * plot_md['Volume']).cumsum() / plot_md['Volume'].cumsum()
                plot_md['ma5'] = plot_md['Close'].rolling(5).mean()
                
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=plot_md.index, open=plot_md['Open'], high=plot_md['High'], low=plot_md['Low'], close=plot_md['Close'], name="1m Price"))
                fig.add_trace(go.Scatter(x=plot_md.index, y=plot_md['vwap'], mode='lines', name="VWAP", line=dict(color='yellow')))
                fig.add_trace(go.Scatter(x=plot_md.index, y=plot_md['ma5'], mode='lines', name="5m MA", line=dict(color='orange')))
                
                # Markers
                fig.add_vline(x=db_entry.timestamp() * 1000, line_color="green", annotation_text="Entry")
                fig.add_vline(x=db_exit.timestamp() * 1000, line_color="red", annotation_text="Exit")
                
                # Make chart background fit the dark theme beautifully
                fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"No market data bounded between {start_date.strftime('%Y-%m-%d %H:%M')} and {end_date.strftime('%Y-%m-%d %H:%M')}.")
                st.info("Tip: Double check if your entry time was during regular market hours!")
    except Exception as e:
        st.warning(f"Failed to load replay: {e}")

    # AI Critique
    if st.button("Generate AI Critique"):
        provider = st.session_state.get('ai_provider', 'noop')
        adapter = AIAdapter(provider=provider)
        with open("single_trade_critique.txt", "r") as f:
            template = f.read()
        
        # Serialize only numeric quant aspects to pass
        quant_dict = {
            "delta": selected.get("delta_entry", 0),
            "gamma_exposure": selected.get("gamma_entry", 0),
            "vol_ratio": selected.get("vol_ratio", 1.0),
            "execution_score": selected.get("entry_execution_score", 0),
            "trade_quality_score": selected.get("trade_quality_score", 0),
            "hold_minutes": selected.get("hold_time_minutes", 0)
        }
        res = adapter.get_critique(template, quant_dict, model="")
        st.markdown(res)

def render_reports():
    st.header("Weekly Report Generation")
    
    if st.button("Generate Weekly Auto Report"):
        provider = st.session_state.get('ai_provider', 'noop')
        adapter = AIAdapter(provider=provider)
        with open("weekly_report.txt", "r") as f:
            template = f.read()
            
        mock_agg_data = {"expectancy": 15.50, "win_rate": 0.65, "total_pnl": 500}
        with st.spinner("Analyzing weekly data..."):
            res = adapter.get_critique(template, mock_agg_data)
        st.success("Report Generated:")
        st.markdown(res)

def render_settings():
    st.header("Settings & Secure Keys")
    
    st.subheader("Preferences")
    st.selectbox("Default Market Data Provider", ["yfinance", "polygon (optional)"], key="md_provider")
    st.selectbox("AI Critique Provider", ["noop", "gemini", "groq"], key="ai_provider")
    st.number_input("Monte Carlo Sample Size", 100, 50000, 10000, key="mc_sims")
    st.number_input("Base Capital for sizing", 50, 100000, 200, key="capital")
    st.checkbox("Use Supabase Hosted DB", value=False, key="use_supabase")
    
    st.subheader("AI API Keys Secure Store")
    st.info("Keys are AES encryptly stored into the SQLite 'secrets' table using PBKDF2 Master Password or keys from secrets.toml")
    
    master = get_master_key()
    if master:
        st.success("Master Key is Active.")
    else:
        st.error("No Master Key generated! Add MASTER_PASSWORD to secrets.toml or environment variables.")
        st.code("MASTER_PASSWORD = \"your-secure-password\"")
        
    with st.form("api_key_form"):
        p = st.selectbox("Provider", ["gemini_api_key", "groq_api_key", "polygon_api_key"])
        val = st.text_input("Plain Text Key", type="password")
        sub = st.form_submit_button("Encrypt & Store")
        if sub and val:
            if not master:
                st.error("Enable master key first to encrypt.")
            else:
                store_api_key(p, val)
                st.success(f"Stored encrypted key for {p}")
                
    if st.button("Export Encrypted Snippet to GUI"):
        # For user to copy to secrets.toml in cloud if they want
        if not master:
            st.error("No master key")
        else:
            try:
                session = get_session()
                records = session.query(Secret).all()
                snippet = "[ai]\n"
                for r in records:
                    b64_enc = base64.b64encode(r.encrypted_key).decode()
                    snippet += f"{r.provider} = \"{b64_enc}\"\n"
                st.code(snippet, language="toml")
            except Exception as e:
                st.error(f"Error {e}")
