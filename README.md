# S&P 500 Stock Explorer

A Streamlit app for exploring S&P 500 constituents — adjusted-close price charts, performance metrics, and a sector-aware performers tab with sortable, filterable tables.

## Features

- **Stock Explorer** — pick any S&P 500 ticker, view adjusted-close price chart, and see latest close, period return, annualized return, max drawdown, and annualized volatility for any time range (YTD, 1Y, 3Y, 5Y, 10Y, Max, or custom dates).
- **Performers Explorer** — see how every S&P 500 stock has performed over a chosen period, with GICS sector and industry filters, an interactive sortable table, by-sector bar chart, and return-distribution histogram. Equal-weight and SPY (cap-weighted) benchmarks for comparison.

## Data sources

- Prices: [yfinance](https://github.com/ranaroussi/yfinance) (Yahoo Finance, total-return adjusted).
- Sector / industry: Wikipedia's S&P 500 list (GICS classifications).

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Stack

Streamlit · yfinance · plotly · pandas · lxml
