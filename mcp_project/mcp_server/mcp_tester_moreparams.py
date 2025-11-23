import mcp
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
#from fastmcp import tool  # assuming fastmcp is installed and imported like this
from fastmcp import FastMCP
logging.basicConfig(level=logging.INFO)


#@mcp.tool()
def portfolio_performance(holdings: dict) -> dict:
    """
    Track portfolio performance with advanced risk metrics and insights.
    Args:
        holdings: {"AAPL": 10, "MSFT": 5, "CASH": 5000}
    Returns:
        Detailed portfolio analytics dictionary.
    """

    tickers = [t for t in holdings.keys() if t != "CASH"]
    data = yf.download(tickers, period="6mo", interval="1d", group_by="ticker", auto_adjust=True, progress=False)

    total_value = holdings.get("CASH", 0.0)
    daily_change_value = 0.0
    portfolio_history = pd.DataFrame()

    sector_exposure = {}
    industry_exposure = {}
    dividends_total = 0.0
    projected_income = 0.0
    top_gainers = []
    top_losers = []

    # Risk calculation prep
    returns_list = []
    weights = []
    betas = []

    for ticker in tickers:
        stock = yf.Ticker(ticker)
        hist = data[ticker]["Close"] if len(tickers) > 1 else data["Close"]

        qty = holdings[ticker]
        latest_price = hist.iloc[-1]
        prev_price = hist.iloc[-2]

        total_value += latest_price * qty
        daily_change_value += (latest_price - prev_price) * qty

        # Portfolio history
        portfolio_history[ticker] = hist.pct_change().dropna()
        returns_list.append(hist.pct_change().dropna())
        weights.append((latest_price * qty) / total_value)

        # Sector & industry mapping
        info = stock.info
        sector_exposure[info.get("sector", "Unknown")] = sector_exposure.get(info.get("sector", "Unknown"), 0) + latest_price * qty
        industry_exposure[info.get("industry", "Unknown")] = industry_exposure.get(info.get("industry", "Unknown"), 0) + latest_price * qty

        # Dividend & income
        divs = stock.dividends
        if not divs.empty:
            annual_div = divs[-4:].sum() if len(divs) >= 4 else divs.sum()
            projected_income += annual_div * qty
            dividends_total += (annual_div / latest_price) * 100  # yield %

    # Normalize exposures
    for sector in sector_exposure:
        sector_exposure[sector] = round((sector_exposure[sector] / total_value) * 100, 2)
    for industry in industry_exposure:
        industry_exposure[industry] = round((industry_exposure[industry] / total_value) * 100, 2)

    # Risk metrics
    portfolio_returns = portfolio_history.mean(axis=1)
    volatility = np.std(portfolio_returns) * np.sqrt(252)
    sharpe = np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252) if np.std(portfolio_returns) != 0 else 0
    sortino = np.mean(portfolio_returns) / np.std(portfolio_returns[portfolio_returns < 0]) * np.sqrt(252) if np.std(portfolio_returns[portfolio_returns < 0]) != 0 else 0
    max_drawdown = (portfolio_history.sum(axis=1).cumsum().cummax() - portfolio_history.sum(axis=1).cumsum()).max()

    # Value at Risk (95%)
    var_95 = np.percentile(portfolio_returns, 5) * total_value

    # Narrative
    narrative = f"Your portfolio is worth ${total_value:,.2f}, with a daily change of ${daily_change_value:,.2f}. "
    narrative += f"Top sector exposure: {max(sector_exposure, key=sector_exposure.get)} ({max(sector_exposure.values()):.2f}%). "
    narrative += f"Annualized volatility is {volatility:.2%}, Sharpe ratio {sharpe:.2f}."

    return {
        "total_value": round(total_value, 2),
        "daily_change": round(daily_change_value, 2),
        "sector_exposure": sector_exposure,
        "industry_exposure": industry_exposure,
        "dividend_yield_avg": round(dividends_total / len(tickers), 2) if tickers else 0,
        "projected_annual_income": round(projected_income, 2),
        "risk_metrics": {
            "volatility": round(volatility, 4),
            "sharpe": round(sharpe, 4),
            "sortino": round(sortino, 4),
            "max_drawdown": round(max_drawdown, 4),
            "value_at_risk_95": round(var_95, 2)
        },
        "narrative": narrative
    }

# Testing
if __name__ == "__main__":
    holdings = {"AAPL": 10, "MSFT": 5, "CASH": 5000}
    print(portfolio_performance(holdings))
