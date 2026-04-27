"""
S&P 500 Stock Explorer
- Tab 1: pick any S&P 500 stock, view price chart and performance metrics
- Tab 2: explore performers across the S&P 500 with sector/industry filters,
         interactive sortable table, and benchmark comparisons
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import requests
from io import StringIO
from datetime import date, timedelta


# ---------- Data loading ----------

@st.cache_data
def load_tickers() -> list[str]:
    """Load the S&P 500 ticker list from tickers.csv (one comma-separated row)."""
    with open("tickers.csv", "r", encoding="utf-8") as f:
        text = f.read().strip()
    return sorted({t.strip() for t in text.split(",") if t.strip()})


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_history(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Fetch adjusted daily price history for a single ticker. Cached for 1 hour."""
    yf_ticker = ticker.replace(".", "-")  # yfinance prefers dashes for class shares
    df = yf.Ticker(yf_ticker).history(
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        auto_adjust=True,  # true total-return adjusted close
    )
    return df


@st.cache_data(ttl=86400, show_spinner=False)
def get_company_info(ticker: str) -> dict:
    """Fetch company name + sector via yfinance.info. Cached 24 hours."""
    yf_ticker = ticker.replace(".", "-")
    try:
        info = yf.Ticker(yf_ticker).info
        return {
            "name": info.get("longName") or info.get("shortName") or ticker,
            "sector": info.get("sector") or "—",
            "industry": info.get("industry") or "—",
        }
    except Exception:
        return {"name": ticker, "sector": "—", "industry": "—"}


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_sp500_metadata() -> pd.DataFrame:
    """
    Pull ticker → name / GICS sector / GICS sub-industry mapping from Wikipedia's
    S&P 500 list. One HTTP call vs 503 yfinance calls. Cached for 24 hours.

    Wikipedia rejects requests without a User-Agent header (403), so we fetch
    via `requests` with an identifying UA, then hand the HTML to pandas.

    Returns columns: ticker, name, sector, industry. Empty df on failure.
    """
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; SP500-Streamlit-App/1.0)"
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text))
        df = tables[0].copy()  # first table is the constituents list
        df = df.rename(columns={
            "Symbol": "ticker",
            "Security": "name",
            "GICS Sector": "sector",
            "GICS Sub-Industry": "industry",
        })
        df["ticker"] = df["ticker"].astype(str).str.strip()
        # Wikipedia uses the same dot format as our tickers.csv (BRK.B etc.)
        return df[["ticker", "name", "sector", "industry"]]
    except Exception as e:
        # Surface the failure reason via Streamlit so it's debuggable
        st.session_state["_metadata_error"] = f"{type(e).__name__}: {e}"
        return pd.DataFrame(columns=["ticker", "name", "sector", "industry"])


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_all_returns(tickers: tuple, start: date, end: date) -> pd.DataFrame:
    """
    Batch download every ticker's history in one call and compute period return.
    Returns DataFrame with columns: ticker, start_price, end_price, return.
    Cached for 24 hours.
    """
    yf_tickers = [t.replace(".", "-") for t in tickers]
    data = yf.download(
        yf_tickers,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="ticker",
    )

    # Require ~80% coverage of the period — keeps recent IPOs from showing up
    # with deceptive "returns" computed off partial windows.
    expected_trading_days = max(1, (end - start).days * 252 // 365)
    min_required = int(0.8 * expected_trading_days)

    rows = []
    for orig, yf_t in zip(tickers, yf_tickers):
        try:
            if isinstance(data.columns, pd.MultiIndex):
                close = data[yf_t]["Close"].dropna()
            else:
                close = data["Close"].dropna()
            if len(close) < 2 or len(close) < min_required:
                continue
            start_price = float(close.iloc[0])
            end_price = float(close.iloc[-1])
            rows.append({
                "ticker": orig,
                "start_price": start_price,
                "end_price": end_price,
                "return": (end_price / start_price) - 1,
            })
        except (KeyError, IndexError, AttributeError):
            continue

    return pd.DataFrame(rows)


# ---------- Metrics & helpers ----------

def compute_metrics(df: pd.DataFrame) -> dict:
    if df.empty or len(df) < 2:
        return {}
    start_price = df["Close"].iloc[0]
    end_price = df["Close"].iloc[-1]
    total_return = (end_price / start_price) - 1
    days = (df.index[-1] - df.index[0]).days
    annualized = ((end_price / start_price) ** (365.25 / days)) - 1 if days > 0 else None
    cummax = df["Close"].cummax()
    drawdown = (df["Close"] - cummax) / cummax
    max_dd = drawdown.min()
    daily_returns = df["Close"].pct_change().dropna()
    vol_annualized = daily_returns.std() * (252 ** 0.5) if len(daily_returns) > 1 else None
    return {
        "latest_price": float(end_price),
        "total_return": float(total_return),
        "annualized": float(annualized) if annualized is not None else None,
        "max_drawdown": float(max_dd),
        "volatility": float(vol_annualized) if vol_annualized is not None else None,
        "trading_days": len(df),
    }


def date_from_preset(preset: str, today: date) -> tuple[date, date]:
    if preset == "YTD":
        return date(today.year, 1, 1), today
    if preset == "1Y":
        return today - timedelta(days=365), today
    if preset == "3Y":
        return today - timedelta(days=365 * 3), today
    if preset == "5Y":
        return today - timedelta(days=365 * 5), today
    if preset == "10Y":
        return today - timedelta(days=365 * 10), today
    if preset == "Max":
        return date(1990, 1, 1), today
    return today - timedelta(days=365), today


def format_pct(x) -> str:
    return f"{x:.1%}" if x is not None else "—"


def format_money(x) -> str:
    return f"${x:,.2f}" if x is not None else "—"


# ---------- UI ----------

st.set_page_config(page_title="S&P 500 Explorer", layout="wide", page_icon="📈")

st.title("S&P 500 Stock Explorer")
st.caption(
    "Adjusted close prices via Yahoo Finance. Returns are total-return (split + dividend adjusted). "
    "Sector classifications from Wikipedia (GICS)."
)

tickers = load_tickers()
today = date.today()

tab_explorer, tab_analysis = st.tabs(["📈 Stock Explorer", "📊 Performers Explorer"])


# =========================================
# TAB 1 — Single stock explorer
# =========================================
with tab_explorer:
    c1, c2 = st.columns([1, 2])
    with c1:
        default_idx = tickers.index("AAPL") if "AAPL" in tickers else 0
        ticker = st.selectbox("Stock", tickers, index=default_idx, key="explorer_ticker")
    with c2:
        preset = st.radio(
            "Time range",
            ["YTD", "1Y", "3Y", "5Y", "10Y", "Max", "Custom"],
            index=1,
            horizontal=True,
            key="explorer_preset",
        )

    if preset == "Custom":
        cc1, cc2 = st.columns(2)
        with cc1:
            start_date = st.date_input("Start", today - timedelta(days=365), key="explorer_start")
        with cc2:
            end_date = st.date_input("End", today, key="explorer_end")
    else:
        start_date, end_date = date_from_preset(preset, today)

    if start_date >= end_date:
        st.error("Start date must be before end date.")
        st.stop()

    info = get_company_info(ticker)
    st.subheader(f"{info['name']} ({ticker})")
    st.caption(f"{info['sector']} • {info['industry']}")

    with st.spinner(f"Loading {ticker} price history..."):
        df = fetch_history(ticker, start_date, end_date)

    if df.empty:
        st.warning(
            f"No price data for **{ticker}** between {start_date} and {end_date}. "
            f"This usually means the company wasn't public over that range — try a shorter window."
        )
    else:
        metrics = compute_metrics(df)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Latest close", format_money(metrics["latest_price"]))
        m2.metric("Period return", format_pct(metrics["total_return"]))
        m3.metric("Annualized", format_pct(metrics["annualized"]))
        m4.metric("Max drawdown", format_pct(metrics["max_drawdown"]))
        m5.metric("Volatility (ann.)", format_pct(metrics["volatility"]))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Close"], mode="lines", name=ticker,
            line=dict(width=2),
            hovertemplate="%{x|%b %d, %Y}<br>$%{y:.2f}<extra></extra>",
        ))
        fig.update_layout(
            title=f"{ticker} adjusted close — {start_date} to {end_date}",
            xaxis_title=None, yaxis_title="Price ($)", height=500,
            hovermode="x unified",
            margin=dict(l=10, r=10, t=50, b=10), showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Show raw price data"):
            table = df[["Open", "High", "Low", "Close", "Volume"]].sort_index(ascending=False)
            table.index = table.index.date
            st.dataframe(table, use_container_width=True)


# =========================================
# TAB 2 — Performers explorer
# =========================================
with tab_analysis:
    st.markdown(
        "Explore performers across the S&P 500. Sort by any column, filter by sector or "
        "industry, and compare to your equal-weight benchmark and the SPY index."
    )

    # ----- Period selector -----
    fc1, _ = st.columns([1, 4])
    with fc1:
        analysis_period = st.selectbox(
            "Period",
            ["YTD", "1Y", "3Y", "5Y"],
            index=1,
            key="analysis_period",
        )

    a_start, a_end = date_from_preset(analysis_period, today)

    # ----- Fetch data -----
    with st.spinner(
        f"Fetching {analysis_period} returns for all {len(tickers)} S&P 500 tickers — "
        f"~30-60 seconds the first time, then cached for 24 hours..."
    ):
        returns_df = fetch_all_returns(tuple(tickers), a_start, a_end)
        spy_df = fetch_history("SPY", a_start, a_end)
        metadata = fetch_sp500_metadata()

    if returns_df.empty:
        st.error("Failed to fetch return data. yfinance may be rate-limiting; try again in a moment.")
    else:
        # ----- Merge sector/industry metadata -----
        if not metadata.empty:
            merged = returns_df.merge(metadata, on="ticker", how="left")
            merged["name"] = merged["name"].fillna(merged["ticker"])
            merged["sector"] = merged["sector"].fillna("Unknown")
            merged["industry"] = merged["industry"].fillna("Unknown")
        else:
            merged = returns_df.copy()
            merged["name"] = merged["ticker"]
            merged["sector"] = "Unknown"
            merged["industry"] = "Unknown"
            err = st.session_state.get("_metadata_error", "unknown error")
            st.warning(
                f"Couldn't fetch sector data from Wikipedia ({err}) — sectors will show as Unknown. "
                f"Filters will still work but be uninformative."
            )

        # ----- Sector & industry filters -----
        all_sectors = sorted(merged["sector"].unique())
        sf1, sf2 = st.columns(2)
        with sf1:
            selected_sectors = st.multiselect(
                "Filter by sector",
                all_sectors,
                default=all_sectors,
                key="analysis_sectors",
                help="Select one or more GICS sectors. Empty = no results.",
            )

        sector_filtered = (
            merged[merged["sector"].isin(selected_sectors)]
            if selected_sectors else merged.iloc[0:0]
        )
        all_industries = sorted(sector_filtered["industry"].unique())
        with sf2:
            selected_industries = st.multiselect(
                "Filter by industry (sub-sector)",
                all_industries,
                default=all_industries,
                key="analysis_industries",
                help="Industries are limited to those within your selected sectors.",
            )

        filtered = (
            sector_filtered[sector_filtered["industry"].isin(selected_industries)]
            if selected_industries else sector_filtered.iloc[0:0]
        )

        if filtered.empty:
            st.warning("No tickers match the current filters. Adjust sector / industry selections.")
        else:
            # ----- Summary metrics -----
            full_avg = float(merged["return"].mean())
            filtered_avg = float(filtered["return"].mean())
            spy_return = (
                float((spy_df["Close"].iloc[-1] / spy_df["Close"].iloc[0]) - 1)
                if not spy_df.empty
                else None
            )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Tickers shown", f"{len(filtered)} / {len(merged)}")
            m2.metric(
                "Filtered avg",
                format_pct(filtered_avg),
                delta=f"{(filtered_avg - full_avg):.1%} vs full S&P",
            )
            m3.metric("Full S&P equal-weight (benchmark)", format_pct(full_avg))
            m4.metric("SPY (cap-weighted)", format_pct(spy_return))

            # ----- Default sort order -----
            sc1, _ = st.columns([1, 4])
            with sc1:
                sort_order = st.radio(
                    "Default sort",
                    ["Worst → Best", "Best → Worst"],
                    horizontal=True,
                    key="analysis_sort",
                )

            ascending = (sort_order == "Worst → Best")
            display = filtered.sort_values("return", ascending=ascending).reset_index(drop=True)

            # ----- Interactive table -----
            st.subheader(
                f"All {len(display)} matching tickers — click any column header to re-sort"
            )
            display_table = display[
                ["ticker", "name", "sector", "industry", "start_price", "end_price", "return"]
            ].copy()
            # Convert decimal return to display percent (12.3 not 0.123) for column formatter
            display_table["return"] = display_table["return"] * 100
            display_table.columns = [
                "Ticker", "Name", "Sector", "Industry",
                "Start price", "End price", f"{analysis_period} return",
            ]

            st.dataframe(
                display_table,
                use_container_width=True,
                hide_index=True,
                height=600,
                column_config={
                    "Start price": st.column_config.NumberColumn(format="$%.2f"),
                    "End price": st.column_config.NumberColumn(format="$%.2f"),
                    f"{analysis_period} return": st.column_config.NumberColumn(format="%.1f%%"),
                },
            )

            # ----- By-sector bar chart (always full S&P, not filtered) -----
            st.subheader(f"Average {analysis_period} return by sector")
            st.caption("Across the full S&P 500 (not affected by your filters above)")
            sector_summary = (
                merged.groupby("sector")["return"]
                .agg(["mean", "count"])
                .reset_index()
                .sort_values("mean")
            )
            fig_sector = go.Figure()
            fig_sector.add_trace(go.Bar(
                x=sector_summary["mean"] * 100,
                y=sector_summary["sector"],
                orientation="h",
                marker_color=["#e74c3c" if x < 0 else "#27ae60" for x in sector_summary["mean"]],
                hovertemplate="<b>%{y}</b><br>Avg return: %{x:.1f}%%<br>%{customdata} tickers<extra></extra>",
                customdata=sector_summary["count"],
            ))
            fig_sector.add_vline(
                x=full_avg * 100, line_dash="dash", line_color="black",
                annotation_text=f"S&P avg: {full_avg:.1%}",
                annotation_position="top",
            )
            fig_sector.update_layout(
                xaxis_title="Avg return (%)",
                yaxis_title=None,
                height=420,
                margin=dict(l=10, r=10, t=10, b=10),
                showlegend=False,
            )
            st.plotly_chart(fig_sector, use_container_width=True)

            # ----- Distribution (filtered) -----
            st.subheader(f"Return distribution — your filtered set ({len(filtered)} tickers)")
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=filtered["return"],
                nbinsx=30,
                marker_color="#4a90e2",
                name="Filtered",
            ))
            fig_dist.add_vline(
                x=full_avg, line_dash="dash", line_color="green",
                annotation_text=f"S&P avg: {full_avg:.1%}",
                annotation_position="top",
            )
            fig_dist.add_vline(
                x=filtered_avg, line_dash="dash", line_color="purple",
                annotation_text=f"Filtered avg: {filtered_avg:.1%}",
                annotation_position="bottom",
            )
            if spy_return is not None:
                fig_dist.add_vline(
                    x=spy_return, line_dash="dash", line_color="orange",
                    annotation_text=f"SPY: {spy_return:.1%}",
                    annotation_position="top",
                )
            fig_dist.update_layout(
                xaxis_title=f"{analysis_period} return",
                yaxis_title="Number of stocks",
                height=400,
                bargap=0.05,
                xaxis_tickformat=".0%",
            )
            st.plotly_chart(fig_dist, use_container_width=True)
