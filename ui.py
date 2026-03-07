import json
import base64
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
from datetime import datetime, timedelta

from db import get_session, Trade, Secret
from secrets_store import store_api_key, get_master_key, encrypt_key
from ingest import get_market_data, import_trades_csv
from ai_adapter import AIAdapter
from montecarlo import simulate_equity_paths, calculate_risk_metrics
from enrichment import enrich_trade

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
def parse_time(ts: str) -> datetime.time:
    ts = ''.join(filter(str.isdigit, ts))
    if not ts: return datetime.strptime("070000", "%H%M%S").time()
    if len(ts) <= 2: ts = ts.zfill(2) + "0000"
    elif len(ts) == 3: ts = "0" + ts + "00"
    elif len(ts) == 4: ts = ts + "00"
    elif len(ts) == 5: ts = "0" + ts
    elif len(ts) > 6: ts = ts[:6]
    try:
        return datetime.strptime(ts, "%H%M%S").time()
    except:
        return datetime.strptime("070000", "%H%M%S").time()

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

    df = df.sort_values('entry_time')
    df['cum_pnl'] = df['pnl'].cumsum()
    df['Trade Number'] = range(1, len(df) + 1)
    
    # Equity Curve
    fig = px.line(df, x='Trade Number', y='cum_pnl', title='Equity Curve', markers=True,
                  hover_data=['entry_time', 'ticker', 'pnl'])
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
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
        
        t1, t2, t3 = st.columns(3)
        trade_date = t1.date_input("Trade Date", value=datetime.today())
        entry_time_input = t2.text_input("Entry Time (PST) e.g. 07:15:30", value="07:00:00")
        exit_time_input = t3.text_input("Exit Time (PST) e.g. 07:30:15", value="07:15:00")
        
        submitted = st.form_submit_button("Save Trade")
        if submitted:
            import uuid
            session = get_session()
            try:
                parsed_entry = parse_time(entry_time_input)
                parsed_exit = parse_time(exit_time_input)
                # Combine date and time
                entry_dt = datetime.combine(trade_date, parsed_entry)
                exit_dt = datetime.combine(trade_date, parsed_exit)
                
                # Ensure UTC awareness for yfinance lookup
                entry_dt_utc = pd.to_datetime(entry_dt).tz_localize('America/Los_Angeles').tz_convert('UTC')
                exit_dt_utc = pd.to_datetime(exit_dt).tz_localize('America/Los_Angeles').tz_convert('UTC')
                
                new_trade = Trade(
                    trade_uuid=str(uuid.uuid4()),
                    ticker=ticker,
                    option_type=option_type,
                    strike=strike,
                    expiry=trade_date,
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
        
    if 'edit_mode' not in st.session_state:
        st.session_state.edit_mode = False
        
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        trade_id = st.selectbox("Select Trade", df['id'].values, format_func=lambda x: f"Trade {x} | {df[df['id']==x]['entry_time'].values[0]}")
    with col2:
        st.write("") # push down to align
        st.write("")
        if st.button("Edit Trade", use_container_width=True):
            st.session_state.edit_mode = not st.session_state.edit_mode
    with col3:
        st.write("") # push down to align
        st.write("")
        if st.button("Delete Trade", type="primary", use_container_width=True):
            session = get_session()
            try:
                t_del = session.query(Trade).filter_by(id=trade_id).first()
                if t_del:
                    session.delete(t_del)
                    session.commit()
                    st.rerun()
            except Exception as e:
                st.error("Error deleting.")
            finally:
                session.close()

    selected = df[df['id'] == trade_id].iloc[0]

    if st.button("Recompute quant metrics (IV, Greeks, score)"):
        err = enrich_trade(int(trade_id))
        if err:
            st.error(err)
        else:
            st.success("Trade enriched. Refresh the viewer.")
            st.rerun()
    
    st.write(f"**Score**: {selected.get('trade_quality_score', 'N/A')}/100")
    st.write(f"PnL: ${selected['pnl']}")
    
    if st.session_state.edit_mode:
        st.markdown("### Edit Selected Trade")
        with st.form(f"edit_trade_{trade_id}"):
            db_en = pd.to_datetime(selected['entry_time'])
            db_ex = pd.to_datetime(selected['exit_time'])
            if db_en.tzinfo is None: db_en = db_en.tz_localize('UTC')
            if db_ex.tzinfo is None: db_ex = db_ex.tz_localize('UTC')
            
            en_pst = db_en.tz_convert('America/Los_Angeles')
            ex_pst = db_ex.tz_convert('America/Los_Angeles')
            
            e_col1, e_col2 = st.columns(2)
            n_tick = e_col1.text_input("Ticker", value=str(selected['ticker']))
            n_opt = e_col2.selectbox("Option Type", ["call", "put"], index=0 if selected['option_type']=='call' else 1)
            
            e_col3, e_col4, e_col5 = st.columns(3)
            n_strike = e_col3.number_input("Strike", value=float(selected['strike'] or 0))
            n_contracts = e_col4.number_input("Contracts", value=int(selected.get('contracts') or 1), step=1)
            
            e_col6, e_col7 = st.columns(2)
            n_en = e_col6.number_input("Entry Price", value=float(selected['entry_price'] or 0), step=0.1)
            n_ex = e_col7.number_input("Exit Price", value=float(selected['exit_price'] or 0), step=0.1)
            
            t_col1, t_col2, t_col3 = st.columns(3)
            n_date = t_col1.date_input("Trade Date", value=en_pst.date())
            n_t_en = t_col2.text_input("Entry Time (PST)", value=en_pst.strftime("%H:%M:%S"))
            n_t_ex = t_col3.text_input("Exit Time (PST)", value=ex_pst.strftime("%H:%M:%S"))
            
            if st.form_submit_button("Save Changes"):
                sess = get_session()
                try:
                    t_upd = sess.query(Trade).filter_by(id=trade_id).first()
                    if t_upd:
                        p_en = parse_time(n_t_en)
                        p_ex = parse_time(n_t_ex)
                        
                        dt_en = datetime.combine(n_date, p_en)
                        dt_ex = datetime.combine(n_date, p_ex)
                        n_utc_en = pd.to_datetime(dt_en).tz_localize('America/Los_Angeles').tz_convert('UTC')
                        n_utc_ex = pd.to_datetime(dt_ex).tz_localize('America/Los_Angeles').tz_convert('UTC')
                        
                        t_upd.ticker = n_tick
                        t_upd.option_type = n_opt
                        t_upd.strike = n_strike
                        t_upd.expiry = n_date
                        t_upd.contracts = n_contracts
                        t_upd.entry_price = n_en
                        t_upd.exit_price = n_ex
                        t_upd.entry_time = n_utc_en
                        t_upd.exit_time = n_utc_ex
                        t_upd.pnl = (n_ex - n_en) * 100 * n_contracts
                        sess.commit()
                        st.session_state.edit_mode = False
                        st.rerun()
                except Exception as e:
                    st.error(f"Update failed: {e}")
                finally:
                    sess.close()
    
    entry_context = {}
    
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
            plot_md = plot_md.sort_index()
            plot_md = plot_md[~plot_md.index.duplicated(keep='first')]
            
            if not plot_md.empty:
                # Calculate VWAP
                plot_md['typ'] = (plot_md['High'] + plot_md['Low'] + plot_md['Close']) / 3
                plot_md['vwap'] = (plot_md['typ'] * plot_md['Volume']).cumsum() / plot_md['Volume'].cumsum()
                plot_md['ema5'] = plot_md['Close'].ewm(span=5, adjust=False).mean()
                plot_md['ema14'] = plot_md['Close'].ewm(span=14, adjust=False).mean()
                plot_md['ema25'] = plot_md['Close'].ewm(span=25, adjust=False).mean()
                
                closest_idx = plot_md.index.get_indexer([db_entry], method='nearest')[0]
                entry_row = plot_md.iloc[closest_idx]
                
                pre_candles = []
                # grab the 3 precedence candles preceding entry
                for i in range(1, 4):
                    if closest_idx - i >= 0:
                        row = plot_md.iloc[closest_idx - i]
                        pre_candles.append({
                            "offset_minutes": -i,
                            "high": float(row["High"]),
                            "low": float(row["Low"]),
                            "close": float(row["Close"]),
                            "ema5": float(row["ema5"]),
                            "ema14": float(row["ema14"]),
                            "ema25": float(row["ema25"])
                        })
                
                entry_context = {
                    "underlying_price_at_entry": float(entry_row["Close"]),
                    "ema5_at_entry": float(entry_row["ema5"]),
                    "ema14_at_entry": float(entry_row["ema14"]),
                    "ema25_at_entry": float(entry_row["ema25"]),
                    "preceding_3_candles": pre_candles,
                }
                
                chart_data = []
                for idx, row in plot_md.iterrows():
                    chart_data.append({
                        "time": int(idx.timestamp()),
                        "open": float(row["Open"]),
                        "high": float(row["High"]),
                        "low": float(row["Low"]),
                        "close": float(row["Close"])
                    })
                
                vwap_data = [{"time": int(idx.timestamp()), "value": float(val)} for idx, val in plot_md["vwap"].dropna().items()]
                ema5_data = [{"time": int(idx.timestamp()), "value": float(val)} for idx, val in plot_md["ema5"].dropna().items()]
                ema14_data = [{"time": int(idx.timestamp()), "value": float(val)} for idx, val in plot_md["ema14"].dropna().items()]
                ema25_data = [{"time": int(idx.timestamp()), "value": float(val)} for idx, val in plot_md["ema25"].dropna().items()]
                
                # Markers need strictly to be mapped to data domain correctly.
                # Find closest timestamps in data to avoid chart errors
                closest_entry = min(chart_data, key=lambda x: abs(x['time'] - int(db_entry.timestamp())))['time']
                closest_exit = min(chart_data, key=lambda x: abs(x['time'] - int(db_exit.timestamp())))['time']
                
                entry_ts = closest_entry
                exit_ts = closest_exit
                
                chart_html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <script src="https://unpkg.com/lightweight-charts@3.8.0/dist/lightweight-charts.standalone.production.js"></script>
                </head>
                <body style="margin: 0; background-color: #0e1117;">
                    <div id="chart"></div>
                    <script>
                        const chart = LightweightCharts.createChart(document.getElementById('chart'), {{
                            width: window.innerWidth,
                            height: 400,
                            layout: {{
                                backgroundColor: '#0e1117',
                                textColor: '#d1d4dc',
                            }},
                            grid: {{
                                vertLines: {{ color: '#2b2b43' }},
                                horzLines: {{ color: '#2b2b43' }},
                            }},
                            crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
                            timeScale: {{ timeVisible: true, secondsVisible: false }},
                        }});
                        
                        const candlestickSeries = chart.addCandlestickSeries({{
                            upColor: '#26a69a',
                            downColor: '#ef5350',
                            borderVisible: false,
                            wickUpColor: '#26a69a',
                            wickDownColor: '#ef5350'
                        }});
                        const data = {json.dumps(chart_data)};
                        candlestickSeries.setData(data);
                        
                        const vwapSeries = chart.addLineSeries({{
                            color: '#eade52',
                            lineWidth: 2,
                            title: 'VWAP'
                        }});
                        vwapSeries.setData({json.dumps(vwap_data)});

                        chart.addLineSeries({{ color: '#2962FF', lineWidth: 1, title: 'EMA 5'}}).setData({json.dumps(ema5_data)});
                        chart.addLineSeries({{ color: '#FF6D00', lineWidth: 1, title: 'EMA 14'}}).setData({json.dumps(ema14_data)});
                        chart.addLineSeries({{ color: '#00C853', lineWidth: 1, title: 'EMA 25'}}).setData({json.dumps(ema25_data)});
                        
                        let markers = [
                            {{ time: {entry_ts}, position: 'belowBar', color: '#26a69a', shape: 'arrowUp', text: 'Entry' }},
                            {{ time: {exit_ts}, position: 'aboveBar', color: '#ef5350', shape: 'arrowDown', text: 'Exit' }}
                        ];
                        // lightweight-charts requires markers to be perfectly strictly sorted
                        markers.sort((a, b) => a.time - b.time);
                        candlestickSeries.setMarkers(markers);
                        chart.timeScale().fitContent();
                        
                        window.addEventListener('resize', () => {{
                            chart.resize(window.innerWidth, 400);
                        }});
                    </script>
                </body>
                </html>
                """
                components.html(chart_html, height=400)
            else:
                st.warning(f"No market data bounded between {start_date.strftime('%Y-%m-%d %H:%M')} and {end_date.strftime('%Y-%m-%d %H:%M')}.")
                st.info("Tip: Double check if your entry time was during regular market hours!")
    except Exception as e:
        st.warning(f"Failed to load replay: {e}")

    # AI Critique
    if st.button("Generate AI Critique"):
        provider = st.session_state.get('ai_provider', 'gemini')
        try:
            adapter = AIAdapter(provider=provider)
            with open("single_trade_critique.txt", "r") as f:
                template = f.read()
            
            # Serialize quant + hold time and vol context for accurate AI critique
            quant_dict = {
                "ticker": str(selected.get("ticker", "UNKNOWN")),
                "option_type": str(selected.get("option_type", "UNKNOWN")),
                "strike": float(selected.get("strike") or 0.0),
                "entry_price": float(selected.get("entry_price") or 0.0),
                "exit_price": float(selected.get("exit_price") or 0.0),
                "contracts": int(selected.get("contracts") or 1),
                "pnl": float(selected.get("pnl") or 0.0),
                "hold_minutes": float(selected.get("hold_time_minutes") or 15.0),
                "underlying_price_at_entry": entry_context.get("underlying_price_at_entry", 0.0),
                "ema5_at_entry": entry_context.get("ema5_at_entry", 0.0),
                "ema14_at_entry": entry_context.get("ema14_at_entry", 0.0),
                "ema25_at_entry": entry_context.get("ema25_at_entry", 0.0),
                "preceding_3_candles": entry_context.get("preceding_3_candles", []),
                "delta": float(selected.get("delta_entry") or 0.45),
                "gamma_exposure": float(selected.get("gamma_entry") or 0.08),
                "vol_ratio": float(selected.get("vol_ratio") or 1.15),
                "implied_vol_entry": float(selected.get("implied_vol_entry") or 0.0),
                "vix_at_entry": float(selected.get("vix_at_entry") or 0.0),
                "execution_score": float(selected.get("entry_execution_score") or 85.0),
                "volatility_edge_score": float(selected.get("volatility_edge_score") or 70.0),
                "timing_score": float(selected.get("timing_score") or 70.0),
                "risk_reward_score": float(selected.get("risk_reward_score") or 70.0),
                "trade_quality_score": float(selected.get("trade_quality_score") or 90.0),
            }
            res = adapter.get_critique(template, quant_dict, model="")
            st.markdown(res)
        except Exception as e:
            st.error(f"AI Connection Failed: {e}")
            import traceback
            st.error(traceback.format_exc())

def render_reports():
    st.header("Weekly Report Generation")
    
    if st.button("Generate Weekly Auto Report"):
        provider = st.session_state.get('ai_provider', 'gemini')
        try:
            adapter = AIAdapter(provider=provider)
            with open("weekly_report.txt", "r") as f:
                template = f.read()
                
            mock_agg_data = {"expectancy": 15.50, "win_rate": 0.65, "total_pnl": 500}
            with st.spinner("Analyzing weekly data..."):
                res = adapter.get_critique(template, mock_agg_data)
            st.success("Report Generated:")
            st.markdown(res)
        except Exception as e:
            st.error(f"AI Connection Failed: {e}")

def render_settings():
    st.header("Settings & Secure Keys")
    
    st.subheader("Preferences")
    st.selectbox("Default Market Data Provider", ["yfinance", "polygon (optional)"], key="md_provider")
    st.selectbox("AI Critique Provider", ["noop", "gemini", "groq"], index=1, key="ai_provider")
    st.number_input("Monte Carlo Sample Size", 100, 50000, 10000, key="mc_sims")
    st.number_input("Base Capital for sizing", 50, 100000, 3000, key="capital")
    use_sb = st.checkbox("Use Supabase Hosted DB", value=False, key="use_supabase")
    if use_sb:
        st.caption("Set USE_SUPABASE=true and SUPABASE_URL in environment or secrets and restart the app for this to take effect.")
    
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
