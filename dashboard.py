"""
Vietnam Stock Dashboard — Dash Web Application
================================================
Interactive market visualization for CafeF OHLCV data.

Usage:
    python dashboard.py
    Then open http://localhost:8050 in your browser.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, html, dcc, callback, Input, Output, dash_table
from signal_rsi import get_latest_signals, compute_rsi, RSI_OVERSOLD, RSI_OVERBOUGHT

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
DATA_PATH = Path(__file__).parent / "outputs" / "pipeline_data" / "processed" / "cafef_ohlcv.csv"

print("Loading data...")
market_data = pd.read_csv(DATA_PATH, encoding="utf-8-sig", parse_dates=["trading_date"])
market_data["ticker"] = market_data["ticker"].astype(str).str.upper().str.strip()
market_data["exchange"] = market_data["exchange"].astype(str).str.upper().str.strip()
market_data = market_data.sort_values(["exchange", "ticker", "trading_date"])

exchanges = sorted(market_data["exchange"].unique())
print(f"Loaded {len(market_data):,} rows | Exchanges: {exchanges}")

# Load pre-computed RSI signals
SIGNALS_PATH = Path(__file__).parent / "outputs" / "rsi_signals.csv"
print("Loading pre-computed RSI signals...")
if SIGNALS_PATH.exists():
    all_signals = pd.read_csv(SIGNALS_PATH, parse_dates=["Signal_Date"])
    latest_signals = get_latest_signals(all_signals)
    print(f"Loaded {len(all_signals):,} signals ({len(latest_signals):,} latest)")
else:
    print("Warning: No pre-computed signals found. Run the pipeline first.")
    all_signals = pd.DataFrame(columns=["Ticker", "Exchange", "Close", "Signal", "Indicator", "Signal_Date"])


# ---------------------------------------------------------------------------
# Technical Indicators
# ---------------------------------------------------------------------------
def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    close = frame["close"]
    frame["sma_20"] = close.rolling(20, min_periods=1).mean()
    frame["ema_20"] = close.ewm(span=20, adjust=False).mean()
    rolling_mean = close.rolling(20, min_periods=1).mean()
    rolling_std = close.rolling(20, min_periods=1).std().fillna(0)
    frame["bb_upper"] = rolling_mean + 2 * rolling_std
    frame["bb_lower"] = rolling_mean - 2 * rolling_std
    change = close.diff()
    gain = change.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-change.clip(upper=0)).rolling(14, min_periods=14).mean()
    rs = gain / loss.replace(0, pd.NA)
    frame["rsi_14"] = (100 - (100 / (1 + rs))).fillna(50)
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    frame["macd"] = ema_12 - ema_26
    frame["macd_signal"] = frame["macd"].ewm(span=9, adjust=False).mean()
    return frame


# ---------------------------------------------------------------------------
# Chart Builders
# ---------------------------------------------------------------------------
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#c8d6e5"),
    hovermode="x unified",
    margin=dict(l=50, r=16, t=40, b=16),
    legend=dict(
        orientation="h", y=1.06, x=0,
        bgcolor="rgba(0,0,0,0)", font=dict(size=11)
    ),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zeroline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zeroline=False),
)


def build_price_chart(data: pd.DataFrame, overlays: list, ticker: str, exchange: str) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.04, row_heights=[0.72, 0.28],
    )
    fig.add_trace(
        go.Candlestick(
            x=data["trading_date"],
            open=data["open"], high=data["high"],
            low=data["low"], close=data["close"],
            name="OHLC",
            increasing_line_color="#00d2a0", increasing_fillcolor="#00d2a0",
            decreasing_line_color="#ff6b6b", decreasing_fillcolor="#ff6b6b",
        ),
        row=1, col=1,
    )

    overlay_specs = {
        "SMA 20": ("sma_20", "#3b82f6"),
        "EMA 20": ("ema_20", "#a78bfa"),
        "Bollinger Bands": ("bb_upper", "#64748b"),
    }
    for label in overlays:
        col_name, color = overlay_specs[label]
        fig.add_trace(
            go.Scatter(
                x=data["trading_date"], y=data[col_name],
                name=label, line=dict(color=color, width=1.5),
            ),
            row=1, col=1,
        )
        if label == "Bollinger Bands":
            fig.add_trace(
                go.Scatter(
                    x=data["trading_date"], y=data["bb_lower"],
                    name="BB Lower",
                    line=dict(color=color, width=1.2, dash="dot"),
                ),
                row=1, col=1,
            )

    colors = [
        "#00d2a0" if c >= o else "#ff6b6b"
        for o, c in zip(data["open"], data["close"])
    ]
    fig.add_trace(
        go.Bar(
            x=data["trading_date"], y=data["volume"],
            name="Volume", marker_color=colors, opacity=0.6,
        ),
        row=2, col=1,
    )

    fig.update_layout(
        **CHART_LAYOUT,
        height=520,
        title=dict(text=f"{ticker} — {exchange}", font=dict(size=16, color="#f1f5f9")),
        xaxis_rangeslider_visible=False,
    )
    fig.update_yaxes(title_text="Price", row=1, col=1, gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(title_text="Volume", row=2, col=1, gridcolor="rgba(255,255,255,0.06)")
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", row=1, col=1)
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", row=2, col=1)
    return fig


def build_oscillator_chart(data: pd.DataFrame, indicator: str) -> go.Figure:
    fig = go.Figure()
    if indicator == "RSI 14":
        fig.add_trace(
            go.Scatter(
                x=data["trading_date"], y=data["rsi_14"],
                name="RSI 14", line=dict(color="#a78bfa", width=2),
                fill="tozeroy", fillcolor="rgba(167,139,250,0.08)",
            )
        )
        fig.add_hline(y=70, line_dash="dot", line_color="rgba(255,107,107,0.5)")
        fig.add_hline(y=30, line_dash="dot", line_color="rgba(0,210,160,0.5)")
        fig.update_yaxes(range=[0, 100], title_text="RSI")
    else:
        histogram = data["macd"] - data["macd_signal"]
        colors = ["#00d2a0" if v >= 0 else "#ff6b6b" for v in histogram]
        fig.add_trace(
            go.Bar(
                x=data["trading_date"], y=histogram,
                name="Histogram", marker_color=colors, opacity=0.5,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=data["trading_date"], y=data["macd"],
                name="MACD", line=dict(color="#3b82f6", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=data["trading_date"], y=data["macd_signal"],
                name="Signal", line=dict(color="#ff6b6b", width=1.5),
            )
        )
        fig.update_yaxes(title_text="MACD")

    fig.update_layout(
        **CHART_LAYOUT,
        height=220,
        title=dict(text=indicator, font=dict(size=13, color="#94a3b8")),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)")
    return fig


# ---------------------------------------------------------------------------
# Dash Application
# ---------------------------------------------------------------------------
app = Dash(
    __name__,
    title="Vietnam Stock Dashboard",
    update_title=None,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

default_exchange = "HOSE" if "HOSE" in exchanges else exchanges[0]
default_tickers = sorted(
    market_data.loc[market_data["exchange"] == default_exchange, "ticker"].unique()
)
default_ticker = "FPT" if "FPT" in default_tickers else default_tickers[0]

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
app.layout = html.Div(
    id="app-container",
    children=[
        # ---- Sidebar ----
        html.Aside(
            id="sidebar",
            children=[
                html.Div(
                    className="logo-section",
                    children=[
                        html.Div("📈", className="logo-icon"),
                        html.H1("VN Stocks", className="logo-title"),
                        html.P("CafeF Market Data", className="logo-subtitle"),
                    ],
                ),
                html.Hr(className="divider"),
                html.Label("Exchange", className="control-label"),
                dcc.Dropdown(
                    id="exchange-dropdown",
                    options=[{"label": e, "value": e} for e in exchanges],
                    value=default_exchange,
                    clearable=False,
                    className="dark-dropdown",
                ),
                html.Label("Ticker", className="control-label"),
                dcc.Dropdown(
                    id="ticker-dropdown",
                    options=[{"label": t, "value": t} for t in default_tickers],
                    value=default_ticker,
                    clearable=False,
                    searchable=True,
                    className="dark-dropdown",
                ),
                html.Label("Overlays", className="control-label"),
                dcc.Checklist(
                    id="overlay-checklist",
                    options=[
                        {"label": " SMA 20", "value": "SMA 20"},
                        {"label": " EMA 20", "value": "EMA 20"},
                        {"label": " Bollinger Bands", "value": "Bollinger Bands"},
                    ],
                    value=["SMA 20", "EMA 20"],
                    className="dark-checklist",
                    inputClassName="checklist-input",
                    labelClassName="checklist-label",
                ),
                html.Label("Momentum", className="control-label"),
                dcc.RadioItems(
                    id="oscillator-radio",
                    options=[
                        {"label": " RSI 14", "value": "RSI 14"},
                        {"label": " MACD", "value": "MACD"},
                    ],
                    value="RSI 14",
                    className="dark-radio",
                    inputClassName="radio-input",
                    labelClassName="radio-label",
                ),
                html.Div(className="spacer"),
                html.Div(
                    className="sidebar-footer",
                    children=[
                        html.P(
                            f"{len(market_data):,} data points",
                            className="footer-stat",
                        ),
                        html.P(
                            f"{len(exchanges)} exchanges",
                            className="footer-stat",
                        ),
                    ],
                ),
            ],
        ),
        # ---- Main Content ----
        html.Main(
            id="main-content",
            children=[
                # Tab Navigation
                dcc.Tabs(
                    id="main-tabs",
                    value="charts",
                    className="dark-tabs",
                    children=[
                        dcc.Tab(label="📊 Charts", value="charts", className="dark-tab", selected_className="dark-tab--selected"),
                        dcc.Tab(label="⚡ RSI Signals", value="signals", className="dark-tab", selected_className="dark-tab--selected"),
                    ],
                ),
                # Tab Content
                html.Div(id="tab-content"),
            ],
        ),
    ],
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
@callback(
    Output("ticker-dropdown", "options"),
    Output("ticker-dropdown", "value"),
    Input("exchange-dropdown", "value"),
)
def update_tickers(exchange: str):
    tickers = sorted(
        market_data.loc[market_data["exchange"] == exchange, "ticker"].unique()
    )
    options = [{"label": t, "value": t} for t in tickers]
    value = "FPT" if "FPT" in tickers else (tickers[0] if tickers else None)
    return options, value


@callback(
    Output("tab-content", "children"),
    Input("main-tabs", "value"),
    Input("exchange-dropdown", "value"),
    Input("ticker-dropdown", "value"),
    Input("overlay-checklist", "value"),
    Input("oscillator-radio", "value"),
)
def render_tab(tab: str, exchange: str, ticker: str, overlays: list, oscillator: str):
    if tab == "signals":
        return _build_signals_tab(exchange)
    return _build_charts_tab(exchange, ticker, overlays, oscillator)


def _build_signals_tab(exchange: str) -> html.Div:
    """Build the RSI Signals table view."""
    # Filter signals for the selected exchange
    ex_signals = all_signals[all_signals["Exchange"] == exchange].copy()
    ex_latest = get_latest_signals(ex_signals)

    # Recent signals (last 30 days)
    if not ex_signals.empty:
        cutoff = ex_signals["Signal_Date"].max() - pd.Timedelta(days=30)
        recent = ex_signals[ex_signals["Signal_Date"] >= cutoff].copy()
    else:
        recent = ex_signals

    buy_count = (ex_latest["Signal"] == "BUY").sum() if not ex_latest.empty else 0
    sell_count = (ex_latest["Signal"] == "SELL").sum() if not ex_latest.empty else 0
    hold_count = (ex_latest["Signal"] == "HOLD").sum() if not ex_latest.empty else 0

    # Format for display
    display_df = recent.head(500).copy()
    if not display_df.empty:
        display_df["Signal_Date"] = display_df["Signal_Date"].dt.strftime("%Y-%m-%d")
        display_df["Close"] = display_df["Close"].apply(lambda x: f"{x:,.2f}")
        display_df["Indicator"] = display_df["Indicator"].apply(lambda x: f"{x:.2f}")

    return html.Div([
        # Signal summary cards
        html.Div(
            className="metrics-row",
            children=[
                _metric_card("Exchange", exchange, None, None),
                _metric_card("Total Signals (30d)", f"{len(recent):,}", None, None),
                _metric_card("BUY", str(buy_count), f"RSI < {RSI_OVERSOLD}", True),
                _metric_card("SELL", str(sell_count), f"RSI > {RSI_OVERBOUGHT}", False),
                _metric_card("HOLD", str(hold_count), f"{RSI_OVERSOLD} ≤ RSI ≤ {RSI_OVERBOUGHT}", None),
                _metric_card("Tickers Scanned", f"{recent['Ticker'].nunique():,}" if not recent.empty else "0", None, None, accent=True),
            ],
        ),
        # Signals table
        html.Div(
            className="chart-card signals-table-card",
            children=[
                html.H3(
                    f"⚡ RSI Trading Signals — {exchange} (Last 30 Days)",
                    className="signals-title",
                ),
                html.P(
                    f"BUY when RSI(14) < {RSI_OVERSOLD} (oversold)  ·  SELL when RSI(14) > {RSI_OVERBOUGHT} (overbought)",
                    className="signals-subtitle",
                ),
                dash_table.DataTable(
                    id="signals-table",
                    columns=[
                        {"name": "Ticker", "id": "Ticker"},
                        {"name": "Exchange", "id": "Exchange"},
                        {"name": "Close", "id": "Close"},
                        {"name": "Signal", "id": "Signal"},
                        {"name": "Indicator (RSI)", "id": "Indicator"},
                        {"name": "Signal Date", "id": "Signal_Date"},
                    ],
                    data=display_df.to_dict("records"),
                    page_size=20,
                    sort_action="native",
                    filter_action="native",
                    style_table={"overflowX": "auto"},
                    style_header={
                        "backgroundColor": "#1e293b",
                        "color": "#94a3b8",
                        "fontWeight": "600",
                        "fontSize": "0.75rem",
                        "textTransform": "uppercase",
                        "letterSpacing": "0.05em",
                        "border": "1px solid rgba(255,255,255,0.06)",
                        "padding": "10px 14px",
                    },
                    style_cell={
                        "backgroundColor": "#111827",
                        "color": "#e2e8f0",
                        "border": "1px solid rgba(255,255,255,0.04)",
                        "padding": "10px 14px",
                        "fontSize": "0.9rem",
                        "fontFamily": "Inter, sans-serif",
                    },
                    style_data_conditional=[
                        {
                            "if": {"filter_query": '{Signal} = "BUY"', "column_id": "Signal"},
                            "color": "#00d2a0",
                            "fontWeight": "700",
                        },
                        {
                            "if": {"filter_query": '{Signal} = "SELL"', "column_id": "Signal"},
                            "color": "#ff6b6b",
                            "fontWeight": "700",
                        },
                        {
                            "if": {"filter_query": '{Signal} = "HOLD"', "column_id": "Signal"},
                            "color": "#94a3b8",
                            "fontWeight": "500",
                        },
                        {
                            "if": {"state": "active"},
                            "backgroundColor": "#1e293b",
                            "border": "1px solid rgba(59,130,246,0.3)",
                        },
                    ],
                    style_filter={
                        "backgroundColor": "#1e293b",
                        "color": "#e2e8f0",
                        "border": "1px solid rgba(255,255,255,0.08)",
                    },
                    style_as_list_view=True,
                ),
            ],
        ),
    ])


def _build_charts_tab(exchange: str, ticker: str, overlays: list, oscillator: str) -> html.Div:
    """Build the Charts tab content."""
    empty_fig = go.Figure()
    empty_fig.update_layout(**CHART_LAYOUT, height=400)

    if not ticker:
        return html.Div([
            html.Div(className="metrics-row"),
            html.Div(className="chart-card", children=[dcc.Graph(figure=empty_fig)]),
        ])

    ticker_data = market_data.loc[
        (market_data["exchange"] == exchange) & (market_data["ticker"] == ticker)
    ].copy()

    if ticker_data.empty:
        return html.Div([html.P("No data available.", className="no-data")])

    # Last 1 year
    max_date = ticker_data["trading_date"].max()
    start_date = max_date - pd.DateOffset(years=1)
    selected = ticker_data.loc[ticker_data["trading_date"] >= start_date]
    if selected.empty:
        selected = ticker_data
    selected = add_indicators(selected)

    # Metrics
    latest = selected.iloc[-1]
    prev_close = selected.iloc[-2]["close"] if len(selected) > 1 else latest["close"]
    pct = ((latest["close"] / prev_close) - 1) * 100 if prev_close else 0
    is_up = pct >= 0

    # Charts
    price_fig = build_price_chart(selected, overlays or [], ticker, exchange)
    osc_fig = build_oscillator_chart(selected, oscillator)

    info = (
        f"{len(selected):,} trading days  ·  "
        f"{selected['trading_date'].min():%d/%m/%Y} → "
        f"{selected['trading_date'].max():%d/%m/%Y}"
    )

    return html.Div([
        html.Div(
            className="metrics-row",
            children=[
                _metric_card("Latest Close", f"{latest['close']:,.2f}", f"{pct:+.2f}%", is_up),
                _metric_card("Volume", f"{latest['volume']:,.0f}", None, None),
                _metric_card("Period High", f"{selected['high'].max():,.2f}", None, True),
                _metric_card("Period Low", f"{selected['low'].min():,.2f}", None, False),
                _metric_card("RSI 14", f"{latest['rsi_14']:.1f}", None, None, accent=True),
            ],
        ),
        html.Div(
            className="chart-card",
            children=[dcc.Graph(figure=price_fig, config={"displayModeBar": False})],
        ),
        html.Div(
            className="chart-card",
            children=[dcc.Graph(figure=osc_fig, config={"displayModeBar": False})],
        ),
        html.Div(info, className="info-bar"),
    ])


def _metric_card(
    label: str, value: str, sub: str | None, is_up: bool | None, accent: bool = False
) -> html.Div:
    children = [
        html.Span(label, className="metric-label"),
        html.Span(value, className="metric-value"),
    ]
    if sub is not None:
        cls = "metric-sub up" if is_up else "metric-sub down"
        children.append(html.Span(sub, className=cls))

    card_cls = "metric-card"
    if accent:
        card_cls += " accent"
    elif is_up is True:
        card_cls += " up"
    elif is_up is False:
        card_cls += " down"
    return html.Div(children, className=card_cls)


# ---------------------------------------------------------------------------
# Inline CSS (premium dark theme)
# ---------------------------------------------------------------------------
app.index_string = '''<!DOCTYPE html>
<html lang="vi">
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        /* ===== RESET ===== */
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        html { font-size: 14px; }
        body {
            font-family: "Inter", -apple-system, BlinkMacSystemFont, sans-serif;
            background: #0b0f19;
            color: #e2e8f0;
            overflow-x: hidden;
        }

        /* ===== APP GRID ===== */
        #app-container {
            display: flex;
            min-height: 100vh;
        }

        /* ===== SIDEBAR ===== */
        #sidebar {
            width: 260px;
            min-width: 260px;
            background: linear-gradient(180deg, #111827 0%, #0d1117 100%);
            border-right: 1px solid rgba(255,255,255,0.06);
            padding: 24px 20px;
            display: flex;
            flex-direction: column;
            position: sticky;
            top: 0;
            height: 100vh;
            overflow-y: auto;
        }
        .logo-section { text-align: center; padding: 8px 0 4px; }
        .logo-icon { font-size: 32px; margin-bottom: 4px; }
        .logo-title {
            font-size: 1.35rem; font-weight: 800;
            background: linear-gradient(135deg, #3b82f6, #a78bfa);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .logo-subtitle { font-size: 0.75rem; color: #64748b; margin-top: 2px; }
        .divider {
            border: none;
            border-top: 1px solid rgba(255,255,255,0.06);
            margin: 16px 0;
        }
        .control-label {
            display: block;
            font-size: 0.7rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #64748b;
            margin: 16px 0 6px;
        }
        .spacer { flex: 1; }
        .sidebar-footer {
            border-top: 1px solid rgba(255,255,255,0.06);
            padding-top: 12px;
            margin-top: 12px;
        }
        .footer-stat { font-size: 0.75rem; color: #475569; margin: 2px 0; }

        /* ===== DROPDOWN DARK ===== */
        .dark-dropdown .Select-control,
        .dark-dropdown .Select-menu-outer {
            background: #1e293b !important;
            border-color: rgba(255,255,255,0.08) !important;
            color: #e2e8f0 !important;
        }
        .dark-dropdown .Select-value-label,
        .dark-dropdown .Select-placeholder,
        .dark-dropdown input { color: #e2e8f0 !important; }
        .dark-dropdown .Select-option {
            background: #1e293b !important;
            color: #e2e8f0 !important;
        }
        .dark-dropdown .Select-option.is-focused {
            background: #334155 !important;
        }

        /* ===== CHECKLIST & RADIO ===== */
        .dark-checklist, .dark-radio {
            display: flex; flex-direction: column; gap: 6px;
        }
        .checklist-label, .radio-label {
            display: flex; align-items: center; gap: 6px;
            font-size: 0.85rem; color: #cbd5e1; cursor: pointer;
            padding: 4px 0;
        }
        .checklist-label:hover, .radio-label:hover { color: #f1f5f9; }
        .checklist-input, .radio-input {
            accent-color: #3b82f6;
            width: 14px; height: 14px;
        }

        /* ===== MAIN CONTENT ===== */
        #main-content {
            flex: 1;
            padding: 24px 28px;
            min-width: 0;
            background: #0b0f19;
        }

        /* ===== METRICS ===== */
        .metrics-row {
            display: flex;
            gap: 12px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .metric-card {
            flex: 1;
            min-width: 140px;
            background: linear-gradient(145deg, #151c2c, #111827);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 12px;
            padding: 16px 18px;
            display: flex;
            flex-direction: column;
            gap: 4px;
            transition: all 0.25s ease;
        }
        .metric-card:hover {
            border-color: rgba(59,130,246,0.3);
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(0,0,0,0.3);
        }
        .metric-card.up { border-left: 3px solid #00d2a0; }
        .metric-card.down { border-left: 3px solid #ff6b6b; }
        .metric-card.accent { border-left: 3px solid #a78bfa; }
        .metric-label {
            font-size: 0.7rem; font-weight: 600;
            text-transform: uppercase; letter-spacing: 0.06em;
            color: #64748b;
        }
        .metric-value {
            font-size: 1.4rem; font-weight: 700;
            color: #f1f5f9;
        }
        .metric-sub {
            font-size: 0.85rem; font-weight: 600;
        }
        .metric-sub.up { color: #00d2a0; }
        .metric-sub.down { color: #ff6b6b; }

        /* ===== CHART CARDS ===== */
        .chart-card {
            background: linear-gradient(145deg, #151c2c, #111827);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 14px;
            padding: 12px 8px 4px;
            margin-bottom: 16px;
            transition: border-color 0.3s ease;
        }
        .chart-card:hover {
            border-color: rgba(59,130,246,0.2);
        }

        /* ===== INFO BAR ===== */
        .info-bar {
            font-size: 0.8rem;
            color: #475569;
            padding: 4px 8px;
            text-align: right;
        }

        .no-data {
            text-align: center;
            color: #64748b;
            padding: 40px;
            font-size: 1rem;
        }

        /* ===== TABS ===== */
        .dark-tabs {
            margin-bottom: 20px;
            border-bottom: 1px solid rgba(255,255,255,0.06);
        }
        .dark-tab {
            background: transparent !important;
            color: #64748b !important;
            border: none !important;
            padding: 10px 20px !important;
            font-family: "Inter", sans-serif !important;
            font-size: 0.9rem !important;
            font-weight: 600 !important;
            cursor: pointer;
            transition: color 0.2s ease;
        }
        .dark-tab:hover {
            color: #cbd5e1 !important;
        }
        .dark-tab--selected {
            color: #f1f5f9 !important;
            border-bottom: 2px solid #3b82f6 !important;
            background: transparent !important;
        }

        /* ===== SIGNALS TABLE ===== */
        .signals-table-card {
            padding: 20px 16px;
        }
        .signals-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: #f1f5f9;
            margin-bottom: 4px;
        }
        .signals-subtitle {
            font-size: 0.8rem;
            color: #64748b;
            margin-bottom: 16px;
        }
        /* DataTable pagination & filter styling */
        .dash-table-container .previous-next-container {
            background: #111827 !important;
            color: #e2e8f0 !important;
        }
        .dash-table-container .page-number {
            color: #e2e8f0 !important;
        }
        .dash-table-container button.previous-page,
        .dash-table-container button.next-page,
        .dash-table-container button.first-page,
        .dash-table-container button.last-page {
            color: #e2e8f0 !important;
            fill: #e2e8f0 !important;
        }

        /* ===== PLOTLY OVERRIDES ===== */
        .js-plotly-plot .plotly .modebar { display: none !important; }

        /* ===== RESPONSIVE ===== */
        @media (max-width: 900px) {
            #app-container { flex-direction: column; }
            #sidebar {
                width: 100%; min-width: 100%;
                height: auto; position: relative;
                flex-direction: row; flex-wrap: wrap;
                padding: 12px;
                gap: 8px;
            }
            .logo-section { width: 100%; }
            .divider { display: none; }
            .spacer { display: none; }
            .sidebar-footer { display: none; }
            .control-label { margin: 8px 0 4px; }
            #main-content { padding: 16px; }
            .metrics-row { gap: 8px; }
            .metric-card { min-width: 120px; padding: 12px; }
        }

        /* Scrollbar styling */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb {
            background: #1e293b;
            border-radius: 3px;
        }
        ::-webkit-scrollbar-thumb:hover { background: #334155; }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>'''


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"\n{'='*50}")
    print(f"  Vietnam Stock Dashboard")
    print(f"  Open http://localhost:8050 in your browser")
    print(f"{'='*50}\n")
    app.run(debug=False, host="0.0.0.0", port=8050)
