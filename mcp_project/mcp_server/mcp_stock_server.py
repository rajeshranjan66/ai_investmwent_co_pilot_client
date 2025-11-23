import asyncio
import math
from datetime import datetime
from typing import Optional, Dict, Any
from cachetools import cached, TTLCache
import pandas as pd
import time

import numpy as np
import yfinance as yf
from dotenv import load_dotenv
from fastmcp import FastMCP
from pandas import DataFrame
import logging
import functools
from cachetools.keys import hashkey
from statsmodels.tsa.arima.model import ARIMA
#from .cache_utils import CacheWithStats

# cache 500 records for 30 sec,
#financialdata_cache = TTLCache(maxsize=100, ttl=300)
#pricedata_cache = TTLCache(maxsize=100, ttl=30)
#Stores up to 100 different stock symbols
#Each entry expires after 30 seconds (TTL = Time To Live)
#Automatically removes old entries when full

price_cache = TTLCache(maxsize=100, ttl=300)


from cachetools.keys import hashkey
import functools

from cachetools.keys import hashkey
import functools

from cachetools.keys import hashkey
import functools
import time


# Update your CacheWithStats class
class CacheWithStats:
    def __init__(self, cache):
        self._cache = cache
        self.hits = 0
        self.misses = 0
        self._last_reset = time.time()

    def __call__(self, func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Create a proper cache key
            key = self._generate_key(func.__name__, args, kwargs)

            if key in self._cache:
                self.hits += 1
                logging.debug(f"Cache HIT for {func.__name__} with key {key}")
                return self._cache[key]
            else:
                self.misses += 1
                logging.debug(f"Cache MISS for {func.__name__} with key {key}")
                result = await func(*args, **kwargs)
                self._cache[key] = result
                return result

        return async_wrapper

    def _generate_key(self, func_name, args, kwargs):
        """Generate a consistent cache key"""
        # Convert args to a tuple representation
        args_repr = tuple(repr(arg) for arg in args)

        # Convert kwargs to a sorted tuple of (key, value) pairs
        kwargs_repr = tuple(sorted((k, repr(v)) for k, v in kwargs.items()))

        # Combine everything into a hashable key
        return hashkey(func_name, args_repr, kwargs_repr)

    def __getattr__(self, name):
        return getattr(self._cache, name)

# Wrap the price_cache for statistics tracking
tracked_cache = CacheWithStats(price_cache)




load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
mcp = FastMCP("stocksAnalysisMCPServer", "1.0.0", "A server to analyze stock data using yfinance")



#tested

@mcp.tool()
@tracked_cache
async def fetch_stock_info(symbol: str) -> dict:
    """
    Get general company info for a stock symbol.
    Args: symbol (str)
    Returns: dict with company profile and key stats.
    """
    logging.debug(f"**************fetch_stock_info called with: {symbol}")
    logging.debug(f"***************Cache stats - Hits: {tracked_cache.hits}, Misses: {tracked_cache.misses}")
    logging.debug(f"═══════════════════════════════════════════════════")
    logging.debug(f"fetch_stock_info called with: '{symbol}'")
    logging.debug(f"Type of symbol: {type(symbol)}, Value: '{symbol}'")
    logging.debug(f"Cache stats - Hits: {tracked_cache.hits}, Misses: {tracked_cache.misses}")
    logging.debug(f"Current cache size: {len(price_cache)}")
    logging.debug(f"Cache keys: {list(price_cache.keys())}")
    logging.debug(f"═══════════════════════════════════════════════════")
    try:
        ticker = yf.Ticker(symbol)

        # fast_info: quick access to common fields
        fast_info = ticker.fast_info if hasattr(ticker, "fast_info") else {}

        # get_info(): more detailed profile (non-deprecated version of .info)
        try:
            full_info = ticker.get_info()

            #logging.info(f"full_info keys for {symbol}: {list(full_info.keys())}")
        except Exception as e:
            #logging.warning(f"Could not fetch detailed info for {symbol}: {e}")
            full_info = {}

        result = {
            "symbol": symbol,
            "company_name": full_info.get("shortName"),
            "market_cap": full_info.get("marketCap"),
            "pe_ratio": full_info.get("trailingPE"),
            "52_week_high": full_info.get("fiftyTwoWeekHigh"),
            "52_week_low": full_info.get("fiftyTwoWeekLow"),
            "company_name": full_info.get("shortName"),
            "long_name": full_info.get("longName"),
            "symbol": full_info.get("symbol"),
            "exchange": full_info.get("exchange"),
            "sector": full_info.get("sector"),
            "industry": full_info.get("industry"),
            "market_cap": full_info.get("marketCap"),
            "enterprise_value": full_info.get("enterpriseValue"),
            "shares_outstanding": full_info.get("sharesOutstanding"),
            "pe_ratio_trailing": full_info.get("trailingPE"),
            "pe_ratio_forward": full_info.get("forwardPE"),
            "peg_ratio": full_info.get("pegRatio"),
            "dividend_rate": full_info.get("dividendRate"),
            "dividend_yield": full_info.get("dividendYield"),
            "ex_dividend_date": full_info.get("exDividendDate"),
            "payout_ratio": full_info.get("payoutRatio"),
            "five_year_avg_dividend_yield": full_info.get("fiveYearAvgDividendYield"),
            "fifty_two_week_high": full_info.get("fiftyTwoWeekHigh"),
            "fifty_two_week_low": full_info.get("fiftyTwoWeekLow"),
            "regular_market_price": full_info.get("regularMarketPrice"),
            "beta": full_info.get("beta"),
            "profit_margins": full_info.get("profitMargins"),
            "gross_margins": full_info.get("grossMargins"),
            "ebitda_margins": full_info.get("ebitdaMargins"),
            "revenue": full_info.get("revenue"),
            "gross_profits": full_info.get("grossProfits"),
            "total_cash": full_info.get("totalCash"),
            "total_debt": full_info.get("totalDebt"),
            "debt_to_equity": full_info.get("debtToEquity"),
            "website": full_info.get("website"),
            "address": full_info.get("address1"),
            "city": full_info.get("city"),
            "state": full_info.get("state"),
            "country": full_info.get("country"),
            "full_time_employees": full_info.get("fullTimeEmployees"),
            "earnings_growth": full_info.get("earningsGrowth"),
            "revenue_growth": full_info.get("revenueGrowth"),
            "operating_margins": full_info.get("operatingMargins"),
            "recommendation_mean": full_info.get("recommendationMean"),
            "recommendation_key": full_info.get("recommendationKey"),
            "logo_url": full_info.get("logo_url"),
            "phone": full_info.get("phone"),
            "summary": full_info.get("summary"),
            "general_info": {},


            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Merge fast_info + full_info
        result["general_info"].update(fast_info if fast_info else {})
        result["general_info"].update(full_info if full_info else {})

        if not result["general_info"]:
            return {
                "error": f"No stock info available for {symbol}",
                "symbol": symbol,
                "last_updated": result["last_updated"]
            }

        return result

    except Exception as e:
        logging.error(f"Error fetching stock info for {symbol}: {str(e)}")
        return {
            "error": f"Failed to fetch stock info: {str(e)}",
            "symbol": symbol,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
# def fetch_stock_info(symbol: str) -> dict:
#     """
#     Retrieve general information about a company based on its stock symbol.
#
#     Args:
#         symbol (str): The stock symbol of the company (e.g., "AAPL", "MSFT").
#
#     Returns:
#         dict: A dictionary containing the company's general information, including:
#               - `symbol` (str): The stock symbol.
#               - `general_info` (dict): Merged data from `fast_info` and `get_info`.
#               - `last_updated` (str): Timestamp of the data retrieval.
#               - `error` (str, optional): Error message if the data retrieval fails.
#     """
#     try:
#         ticker = yf.Ticker(symbol)
#
#         # fast_info: quick access to common fields
#         fast_info = ticker.fast_info if hasattr(ticker, "fast_info") else {}
#
#         # get_info(): more detailed profile (non-deprecated version of .info)
#         try:
#             full_info = ticker.get_info()
#         except Exception as e:
#             logging.warning(f"Could not fetch detailed info for {symbol}: {e}")
#             full_info = {}
#
#         result = {
#             "symbol": symbol,
#             "general_info": {},
#             "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         }
#
#         # Merge fast_info + full_info
#         result["general_info"].update(fast_info if fast_info else {})
#         result["general_info"].update(full_info if full_info else {})
#
#         if not result["general_info"]:
#             return {
#                 "error": f"No stock info available for {symbol}",
#                 "symbol": symbol,
#                 "last_updated": result["last_updated"]
#             }
#
#         return result
#
#     except Exception as e:
#         logging.error(f"Error fetching stock info for {symbol}: {str(e)}")
#         return {
#             "error": f"Failed to fetch stock info: {str(e)}",
#             "symbol": symbol,
#             "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         }


#Tested

@tracked_cache
@mcp.tool()
def fetch_quarterly_financials(symbol: str) -> dict:
    """
        Get quarterly financials and ratios for a stock.
        Args: symbol (str)
        Returns: list of dicts per quarter with metrics.
        """
    ticker = yf.Ticker(symbol)
    income = ticker.quarterly_financials
    balance = ticker.quarterly_balance_sheet
    cashflow = ticker.quarterly_cashflow

    if (income is None or income.empty) and (balance is None or balance.empty) and (cashflow is None or cashflow.empty):
        return {"error": f"No quarterly financial statements available for {symbol}"}

    df_income = income.T if income is not None else pd.DataFrame()
    df_balance = balance.T if balance is not None else pd.DataFrame()
    df_cashflow = cashflow.T if cashflow is not None else pd.DataFrame()

    combined = df_income.join(df_balance, how="outer", lsuffix="_inc", rsuffix="_bal")
    combined = combined.join(df_cashflow, how="outer", rsuffix="_cf")
    combined.sort_index(ascending=False, inplace=True)
    combined = combined.reset_index()

    records = []
    for i, row in combined.iterrows():
        quarter_data = row.to_dict()
        try:
            revenue = row.get("Total Revenue", None)
            net_income = row.get("Net Income", None)
            ebitda = row.get("EBITDA", None)
            operating_income = row.get("Operating Income", None)
            gross_profit = row.get("Gross Profit", None)
            total_assets = row.get("Total Assets", None)
            total_liabilities = row.get("Total Liab", None)
            equity = row.get("Total Stockholder Equity", None)
            free_cash_flow = row.get("Free Cash Flow", None)
            capex = row.get("Capital Expenditures", None)
            current_assets = row.get("Total Current Assets", None)
            current_liabilities = row.get("Total Current Liabilities", None)
            inventory = row.get("Inventory", None)
            receivables = row.get("Net Receivables", None)
            shares_outstanding = row.get("Common Stock", None)
            dividends = row.get("Dividends Paid", None)
            tax_expense = row.get("Income Tax Expense", None)
            interest_expense = row.get("Interest Expense", None)

            eps = net_income / shares_outstanding if net_income and shares_outstanding else None
            roe = net_income / equity if net_income and equity else None
            roa = net_income / total_assets if net_income and total_assets else None
            current_ratio = current_assets / current_liabilities if current_assets and current_liabilities else None
            quick_ratio = (current_assets - inventory) / current_liabilities if current_assets and inventory and current_liabilities else None
            debt_to_equity = total_liabilities / equity if total_liabilities and equity else None
            interest_coverage = operating_income / interest_expense if operating_income and interest_expense else None
            book_value_per_share = equity / shares_outstanding if equity and shares_outstanding else None
            gross_margin = gross_profit / revenue if gross_profit and revenue else None
            operating_margin = operating_income / revenue if operating_income and revenue else None
            net_margin = net_income / revenue if net_income and revenue else None
            dividend_payout_ratio = dividends / net_income if dividends and net_income else None
            effective_tax_rate = tax_expense / pre_tax_income if tax_expense and (pre_tax_income := row.get("Pretax Income", None)) else None
            inventory_turnover = revenue / inventory if revenue and inventory else None
            receivables_turnover = revenue / receivables if revenue and receivables else None

            quarter_data.update({
                "EPS": eps,
                "EBITDA": ebitda,
                "Operating_Income": operating_income,
                "Gross_Profit": gross_profit,
                "Total_Assets": total_assets,
                "Total_Liabilities": total_liabilities,
                "Shareholder_Equity": equity,
                "Free_Cash_Flow": free_cash_flow,
                "Return_on_Equity": roe,
                "Return_on_Assets": roa,
                "Current_Ratio": current_ratio,
                "Quick_Ratio": quick_ratio,
                "Debt_to_Equity": debt_to_equity,
                "Interest_Coverage": interest_coverage,
                "Book_Value_Per_Share": book_value_per_share,
                "Gross_Margin": gross_margin,
                "Operating_Margin": operating_margin,
                "Net_Margin": net_margin,
                "Dividend_Payout_Ratio": dividend_payout_ratio,
                "Effective_Tax_Rate": effective_tax_rate,
                "Capital_Expenditures": capex,
                "Inventory_Turnover": inventory_turnover,
                "Receivables_Turnover": receivables_turnover,
            })

            # Growth rates (QoQ)
            if i < len(combined) - 1:
                prev_row = combined.iloc[i + 1]
                prev_revenue = prev_row.get("Total Revenue", None)
                prev_net_income = prev_row.get("Net Income", None)
                quarter_data["Revenue_Growth_QoQ"] = ((revenue - prev_revenue) / prev_revenue) if revenue and prev_revenue else None
                quarter_data["Net_Income_Growth_QoQ"] = ((net_income - prev_net_income) / prev_net_income) if net_income and prev_net_income else None
        except Exception:
            pass
        records.append(quarter_data)

    return records
# def fetch_quarterly_financials(symbol: str) -> pd.DataFrame:
#     """
#     Retrieve quarterly financial statements for a given stock symbol.
#
#     Args:
#         symbol (str): The stock symbol of the company (e.g., "AAPL", "TSLA").
#
#     Returns:
#         pd.DataFrame: A DataFrame containing quarterly financial data, including:
#                       - Income Statement, Balance Sheet, and Cash Flow metrics.
#                       - Rows represent fiscal quarters, and columns represent financial metrics.
#     """
#     ticker = yf.Ticker(symbol)
#
#     # Pull the main quarterly statements
#     income = ticker.quarterly_financials
#     balance = ticker.quarterly_balance_sheet
#     cashflow = ticker.quarterly_cashflow
#
#     # If all are empty or None, consider it an error
#     if (income is None or income.empty) and (balance is None or balance.empty) and (cashflow is None or cashflow.empty):
#         raise ValueError(f"No quarterly financial statements available for {symbol}")
#
#     # Transpose so quarters are rows
#     df_income = income.T if income is not None else pd.DataFrame()
#     df_balance = balance.T if balance is not None else pd.DataFrame()
#     df_cashflow = cashflow.T if cashflow is not None else pd.DataFrame()
#
#     # Merge into one DataFrame on index = period (quarter)
#     combined = df_income.join(df_balance, how="outer", lsuffix="_inc", rsuffix="_bal")
#     combined = combined.join(df_cashflow, how="outer", rsuffix="_cf")
#
#     # Sort by index descending (most recent first)
#     combined.sort_index(ascending=False, inplace=True)
#
#     return combined

#tested
@tracked_cache
@mcp.tool()
def fetch_annual_financials(symbol: str) -> dict:

    """
    Get annual financials and ratios for a stock.
    Args: symbol (str)
    Returns: list of dicts per year with metrics.
    """
    ticker = yf.Ticker(symbol)
    income = ticker.financials
    balance = ticker.balance_sheet
    cashflow = ticker.cashflow

    if (income is None or income.empty) and (balance is None or balance.empty) and (cashflow is None or cashflow.empty):
        return {"error": f"No financial statements available for {symbol}"}

    df_income = income.T if income is not None else pd.DataFrame()
    df_balance = balance.T if balance is not None else pd.DataFrame()
    df_cashflow = cashflow.T if cashflow is not None else pd.DataFrame()

    combined = df_income.join(df_balance, how="outer", lsuffix="_inc", rsuffix="_bal")
    combined = combined.join(df_cashflow, how="outer", rsuffix="_cf")
    combined.sort_index(ascending=False, inplace=True)
    combined = combined.reset_index()

    # Calculate additional metrics
    records = []
    for i, row in combined.iterrows():
        year_data = row.to_dict()
        try:
            # Extract values safely
            revenue = row.get("Total Revenue", None)
            net_income = row.get("Net Income", None)
            ebitda = row.get("EBITDA", None)
            operating_income = row.get("Operating Income", None)
            gross_profit = row.get("Gross Profit", None)
            total_assets = row.get("Total Assets", None)
            total_liabilities = row.get("Total Liab", None)
            equity = row.get("Total Stockholder Equity", None)
            free_cash_flow = row.get("Free Cash Flow", None)
            capex = row.get("Capital Expenditures", None)
            current_assets = row.get("Total Current Assets", None)
            current_liabilities = row.get("Total Current Liabilities", None)
            inventory = row.get("Inventory", None)
            receivables = row.get("Net Receivables", None)
            shares_outstanding = row.get("Common Stock", None)
            dividends = row.get("Dividends Paid", None)
            tax_expense = row.get("Income Tax Expense", None)
            interest_expense = row.get("Interest Expense", None)

            # Ratios and growth
            eps = net_income / shares_outstanding if net_income and shares_outstanding else None
            roe = net_income / equity if net_income and equity else None
            roa = net_income / total_assets if net_income and total_assets else None
            current_ratio = current_assets / current_liabilities if current_assets and current_liabilities else None
            quick_ratio = (current_assets - inventory) / current_liabilities if current_assets and inventory and current_liabilities else None
            debt_to_equity = total_liabilities / equity if total_liabilities and equity else None
            interest_coverage = operating_income / interest_expense if operating_income and interest_expense else None
            book_value_per_share = equity / shares_outstanding if equity and shares_outstanding else None
            gross_margin = gross_profit / revenue if gross_profit and revenue else None
            operating_margin = operating_income / revenue if operating_income and revenue else None
            net_margin = net_income / revenue if net_income and revenue else None
            dividend_payout_ratio = dividends / net_income if dividends and net_income else None
            effective_tax_rate = tax_expense / pre_tax_income if tax_expense and (pre_tax_income := row.get("Pretax Income", None)) else None
            inventory_turnover = revenue / inventory if revenue and inventory else None
            receivables_turnover = revenue / receivables if revenue and receivables else None

            # Add to year_data
            year_data.update({
                "EPS": eps,
                "EBITDA": ebitda,
                "Operating_Income": operating_income,
                "Gross_Profit": gross_profit,
                "Total_Assets": total_assets,
                "Total_Liabilities": total_liabilities,
                "Shareholder_Equity": equity,
                "Free_Cash_Flow": free_cash_flow,
                "Return_on_Equity": roe,
                "Return_on_Assets": roa,
                "Current_Ratio": current_ratio,
                "Quick_Ratio": quick_ratio,
                "Debt_to_Equity": debt_to_equity,
                "Interest_Coverage": interest_coverage,
                "Book_Value_Per_Share": book_value_per_share,
                "Gross_Margin": gross_margin,
                "Operating_Margin": operating_margin,
                "Net_Margin": net_margin,
                "Dividend_Payout_Ratio": dividend_payout_ratio,
                "Effective_Tax_Rate": effective_tax_rate,
                "Capital_Expenditures": capex,
                "Inventory_Turnover": inventory_turnover,
                "Receivables_Turnover": receivables_turnover,
            })

            # Growth rates (YoY)
            if i < len(combined) - 1:
                prev_row = combined.iloc[i + 1]
                prev_revenue = prev_row.get("Total Revenue", None)
                prev_net_income = prev_row.get("Net Income", None)
                year_data["Revenue_Growth_YoY"] = ((revenue - prev_revenue) / prev_revenue) if revenue and prev_revenue else None
                year_data["Net_Income_Growth_YoY"] = ((net_income - prev_net_income) / prev_net_income) if net_income and prev_net_income else None
        except Exception:
            pass
        records.append(year_data)

    return records[:2]
# def fetch_annual_financials(symbol: str) -> pd.DataFrame:
#     """
#     Retrieve annual financial statements for a given stock symbol.
#
#     Args:
#         symbol (str): The stock symbol of the company (e.g., "AAPL", "TSLA").
#
#     Returns:
#         pd.DataFrame: A DataFrame containing annual financial data, including:
#                       - Income Statement, Balance Sheet, and Cash Flow metrics.
#                       - Rows represent fiscal years, and columns represent financial metrics.
#     """
#     ticker = yf.Ticker(symbol)
#
#     # Pull the main statements: Income, Balance Sheet, Cash Flow
#     income = ticker.financials
#     balance = ticker.balance_sheet
#     cashflow = ticker.cashflow
#
#     # If all are empty or None, consider it an error
#     if (income is None or income.empty) and (balance is None or balance.empty) and (cashflow is None or cashflow.empty):
#         raise ValueError(f"No financial statements available for {symbol}")
#
#     # Transpose for annual periods as rows
#     df_income = income.T if income is not None else pd.DataFrame()
#     df_balance = balance.T if balance is not None else pd.DataFrame()
#     df_cashflow = cashflow.T if cashflow is not None else pd.DataFrame()
#
#     # Merge into one DataFrame on index = period (Date)
#     combined = df_income.join(df_balance, how="outer", lsuffix="_inc", rsuffix="_bal")
#     combined = combined.join(df_cashflow, how="outer", rsuffix="_cf")
#
#     # Sort by index descending (most recent first)
#     combined.sort_index(ascending=False, inplace=True)
#
#     return combined

#tested


@mcp.tool()
@tracked_cache
async def get_stock_price(symbol: str) -> Optional[float]:
    """
    Retrieve the latest stock price for a given symbol.

    Args:
        symbol (str): The stock symbol (e.g., "AAPL", "MSFT").

    Returns:
        float: The latest stock price if successful.
        None: If the price data is unavailable or invalid.
    """
    logging.debug(f"**************fetch_stock_info called with: {symbol}")
    logging.debug(f"***************Cache stats - Hits: {tracked_cache.hits}, Misses: {tracked_cache.misses}")
    logging.debug(f"═══════════════════════════════════════════════════")
    logging.debug(f"fetch_stock_info called with: '{symbol}'")
    logging.debug(f"Type of symbol: {type(symbol)}, Value: '{symbol}'")
    logging.debug(f"Cache stats - Hits: {tracked_cache.hits}, Misses: {tracked_cache.misses}")
    logging.debug(f"Current cache size: {len(price_cache)}")
    logging.debug(f"Cache keys: {list(price_cache.keys())}")
    logging.debug(f"═══════════════════════════════════════════════════")
    try:
        ticker = yf.Ticker(symbol)

        # Get data with timeout and validation
        data = ticker.history(
            period="1d",
            interval="1m",  # More frequent data for accuracy
            prepost=False,  # Only regular market hours
            timeout=10  # Fail fast if no response
        )

        if data.empty:
            logging.error(f"No price data for {symbol}")
            return None

        # Validate we have closing price
        if "Close" not in data.columns:
            logging.error(f"Missing Close price for {symbol}")
            return None

        latest_price = data["Close"].iloc[-1]

        # Price sanity check
        if not isinstance(latest_price, (float, int)) or latest_price <= 0:
            logging.error(f"Invalid price {latest_price} for {symbol}")
            return None

       # logging.info(f"Price retrieved for {symbol}: ${latest_price:.2f}")
        return float(latest_price)

    except Exception as e:
        logging.error(f"Failed to get price for {symbol}: {str(e)}")
        return None


#tested


#@cached(price_cache)
@tracked_cache
@mcp.tool()
def get_stock_history(symbol: str, period: str = "1mo") -> DataFrame:
    """
    Retrieve historical stock price data with technical indicators.

    Args:
        symbol (str): The stock symbol (e.g., "AAPL", "MSFT").
        period (str): The time period for the data (default: "1mo").
                      Valid values: "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max".

    Returns:
        DataFrame: A DataFrame containing OHLCV data, technical indicators, and metadata.
                   Metadata is stored in the `attrs` attribute of the DataFrame.
    """


    # Input validation
    valid_periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
    if period not in valid_periods:
        logging.error(f"Invalid period: {period}. Valid periods: {valid_periods}")
        return DataFrame()

    try:
        ticker = yf.Ticker(symbol)

        # Get historical data with modern API
        history = ticker.history(
            period=period,
            interval="1d",
            actions=True,
            auto_adjust=True,
            prepost=False  # Disable pre/post market data for consistency
        )

        if history.empty:
            logging.warning(f"No data for {symbol} (period: {period})")
            return DataFrame()

        # Validate required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in history.columns]
        if missing_cols:
            logging.error(f"Missing columns: {missing_cols}")
            return DataFrame()

        # Calculate technical indicators
        def safe_calc(series, window):
            try:
                return series.rolling(window=window).mean()
            except:
                return None

        indicators = {
            'Daily_Return': history['Close'].pct_change(),
            'MA5': safe_calc(history['Close'], 5),
            'MA20': safe_calc(history['Close'], 20),
            'Daily_Range': history['High'] - history['Low'],
            'Volume_MA5': safe_calc(history['Volume'], 5)
        }

        # Add indicators only if calculation succeeded
        for name, values in indicators.items():
            if values is not None:
                history[name] = values.round(4)

        # Add metadata
        metadata = {
            'symbol': symbol,
            'period': period,
            'start_date': history.index[0].strftime('%Y-%m-%d'),
            'end_date': history.index[-1].strftime('%Y-%m-%d'),
            'trading_days': len(history),
            'summary_stats': {
                'start_price': float(history['Open'].iloc[0]),
                'end_price': float(history['Close'].iloc[-1]),
                'period_high': float(history['High'].max()),
                'period_low': float(history['Low'].min()),
                'avg_volume': int(history['Volume'].mean()),
                'total_volume': int(history['Volume'].sum()),
                'price_change': float(history['Close'].iloc[-1] - history['Open'].iloc[0]),
                'price_change_pct': round(float(
                    (history['Close'].iloc[-1] - history['Open'].iloc[0]) /
                    history['Open'].iloc[0] * 100), 2)
            }
        }

        history.attrs.update(metadata)
        return history

    except Exception as e:
        logging.error(f"Error processing {symbol}: {str(e)}", exc_info=True)
        return DataFrame()

@tracked_cache
@mcp.tool()
def fetch_technical_indicators(symbol: str, period: str = "1mo") -> dict:
    """
    Get daily technical indicators for a stock.
    Args: symbol (str), period (str)
    Returns: dict with indicators per day.
    """
    import pandas as pd
    import yfinance as yf

    def safe_calc(series, window):
        try:
            return series.rolling(window=window).mean()
        except Exception:
            return None

    def calc_atr(df, window=14):
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=window).mean()

    def calc_obv(df):
        obv = [0]
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i - 1]:
                obv.append(obv[-1] + df['Volume'].iloc[i])
            elif df['Close'].iloc[i] < df['Close'].iloc[i - 1]:
                obv.append(obv[-1] - df['Volume'].iloc[i])
            else:
                obv.append(obv[-1])
        return pd.Series(obv, index=df.index)

    def calc_adx(df, window=14):
        up = df['High'].diff()
        down = -df['Low'].diff()
        plus_dm = np.where((up > down) & (up > 0), up, 0)
        minus_dm = np.where((down > up) & (down > 0), down, 0)
        tr = calc_atr(df, window)
        plus_di = 100 * pd.Series(plus_dm).rolling(window).sum() / tr
        minus_di = 100 * pd.Series(minus_dm).rolling(window).sum() / tr
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        adx = dx.rolling(window).mean()
        return adx

    try:
        ticker = yf.Ticker(symbol)
        history = ticker.history(period=period, interval="1d", auto_adjust=True, prepost=False)
        if history.empty:
            return {"error": f"No data for {symbol} ({period})"}

        # Indicators
        indicators = {
            "SMA5": safe_calc(history["Close"], 5),
            "SMA10": safe_calc(history["Close"], 10),
            "SMA20": safe_calc(history["Close"], 20),

            "SMA50": safe_calc(history["Close"], 50),
            "EMA12": history["Close"].ewm(span=12, adjust=False).mean(),
            "EMA26": history["Close"].ewm(span=26, adjust=False).mean(),
            "RSI14": 100 - (100 / (
                        1 + history["Close"].pct_change().rolling(14).mean() / history["Close"].pct_change().rolling(
                    14).std())),
            "MACD": history["Close"].ewm(span=12, adjust=False).mean() - history["Close"].ewm(span=26,
                                                                                              adjust=False).mean(),
            "Bollinger_Upper": safe_calc(history["Close"], 20) + 2 * history["Close"].rolling(20).std(),
            "Bollinger_Lower": safe_calc(history["Close"], 20) - 2 * history["Close"].rolling(20).std(),
            "Daily_Range": history["High"] - history["Low"],
            "Volume_MA5": safe_calc(history["Volume"], 5),
            "ATR14": calc_atr(history, 14),
            "ADX14": calc_adx(history, 14),
            "OBV": calc_obv(history)
        }

        for name, values in indicators.items():
            if values is not None:
                history[name] = values.round(4)

        # Prepare output
        result = history.reset_index().to_dict(orient="records")
        return {"symbol": symbol, "period": period, "indicators": result}

    except Exception as e:
        return {"error": str(e)}

@tracked_cache
@mcp.tool()
def fetch_dividends(symbol: str) -> dict:
    """
    Retrieve historical dividend data for a given stock symbol.

    Args:
        symbol (str): The stock symbol of the company (e.g., "AAPL", "TSLA").

    Returns:
        dict: A dictionary containing:
              - `dividend_data` (dict): Historical dividend data and statistics.
              - `dividend_summary` (dict): Summary of dividend-related metrics.
              - `last_updated` (str): Timestamp of the data retrieval.
              - `error` (str, optional): Error message if the data retrieval fails.
    """
    try:
        ticker = yf.Ticker(symbol)

        # Get dividend data
        dividends = ticker.history(period="max")['Dividends']

        # Get additional dividend info from ticker.info
        info = ticker.info

        result = {
            "symbol": symbol,
            "dividend_data": {},
            "dividend_summary": {},
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Process historical dividends
        if not dividends.empty:
            # Filter out zero dividends and create historical data
            div_history = {}
            for date, amount in dividends[dividends > 0].items():
                div_history[date.strftime("%Y-%m-%d")] = float(amount)

            result["dividend_data"] = {
                "history": div_history,
                "total_dividends_paid": len(div_history),
                "latest_dividend": float(dividends[dividends > 0].iloc[-1]) if len(
                    dividends[dividends > 0]) > 0 else None,
                "latest_dividend_date": dividends[dividends > 0].index[-1].strftime("%Y-%m-%d") if len(
                    dividends[dividends > 0]) > 0 else None
            }

            # Calculate some basic statistics
            if div_history:
                dividend_values = list(div_history.values())
                result["dividend_data"]["statistics"] = {
                    "average_dividend": float(sum(dividend_values) / len(dividend_values)),
                    "minimum_dividend": float(min(dividend_values)),
                    "maximum_dividend": float(max(dividend_values)),
                    "total_dividend_amount": float(sum(dividend_values))
                }

        # Add dividend summary from info
        dividend_summary = {}
        dividend_fields = {
            "dividendRate": "annual_dividend_rate",
            "dividendYield": "dividend_yield",
            "payoutRatio": "payout_ratio",
            "fiveYearAvgDividendYield": "five_year_avg_dividend_yield",
            "trailingAnnualDividendRate": "trailing_annual_dividend_rate",
            "trailingAnnualDividendYield": "trailing_annual_dividend_yield"
        }

        for api_field, result_field in dividend_fields.items():
            if api_field in info and info[api_field] is not None:
                dividend_summary[result_field] = float(info[api_field])

        # Add dividend dates
        date_fields = {
            "dividendDate": "next_dividend_date",
            "exDividendDate": "ex_dividend_date",
            "lastDividendDate": "last_dividend_date"
        }

        for api_field, result_field in date_fields.items():
            if api_field in info and info[api_field] is not None:
                try:
                    date_value = datetime.fromtimestamp(info[api_field])
                    dividend_summary[result_field] = date_value.strftime("%Y-%m-%d")
                except (TypeError, ValueError) as e:
                    logging.warning(f"Could not process {api_field}: {e}")

        if dividend_summary:
            result["dividend_summary"] = dividend_summary

        # Check if we have any dividend data
        if not result["dividend_data"] and not result["dividend_summary"]:
            logging.warning(f"No dividend data found for {symbol}")
            return {
                "error": f"No dividend data available for {symbol}",
                "symbol": symbol,
                "last_updated": result["last_updated"]
            }

        return result

    except Exception as e:
        logging.error(f"Error fetching dividends for {symbol}: {str(e)}")
        return {
            "error": f"Failed to fetch dividends: {str(e)}",
            "symbol": symbol,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }


@tracked_cache
@mcp.tool()
def fetch_actions(symbol: str) -> dict:
    """
    Retrieve corporate actions (dividends and stock splits) for a given stock symbol.

    Args:
        symbol (str): The stock symbol of the company (e.g., "AAPL", "TSLA").

    Returns:
        dict: A dictionary containing:
              - `actions` (dict): Historical dividend and stock split data.
              - `summary` (dict): Summary statistics for dividends and splits.
              - `last_updated` (str): Timestamp of the data retrieval.
              - `error` (str, optional): Error message if the data retrieval fails.
    """
    try:
        ticker = yf.Ticker(symbol)

        # Get historical data with actions
        history = ticker.history(period="max")

        result = {
            "symbol": symbol,
            "actions": {
                "dividends": {},
                "splits": {}
            },
            "summary": {
                "total_dividends": 0,
                "total_splits": 0,
                "dividend_stats": {},
                "split_stats": {}
            },
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Process dividends
        if 'Dividends' in history.columns:
            dividends = history['Dividends']
            div_data = {}

            # Filter out zero dividends
            non_zero_dividends = dividends[dividends > 0]

            if not non_zero_dividends.empty:
                for date, amount in non_zero_dividends.items():
                    div_data[date.strftime("%Y-%m-%d")] = float(amount)

                dividend_values = list(div_data.values())
                result["actions"]["dividends"] = div_data
                result["summary"]["dividend_stats"] = {
                    "total_dividends_paid": len(div_data),
                    "first_dividend_date": min(div_data.keys()),
                    "last_dividend_date": max(div_data.keys()),
                    "average_dividend": float(sum(dividend_values) / len(dividend_values)),
                    "minimum_dividend": float(min(dividend_values)),
                    "maximum_dividend": float(max(dividend_values)),
                    "total_dividend_amount": float(sum(dividend_values))
                }
                result["summary"]["total_dividends"] = len(div_data)

        # Process stock splits
        if 'Stock Splits' in history.columns:
            splits = history['Stock Splits']
            split_data = {}

            # Filter out non-split events
            non_zero_splits = splits[splits != 0]

            if not non_zero_splits.empty:
                for date, ratio in non_zero_splits.items():
                    split_data[date.strftime("%Y-%m-%d")] = float(ratio)

                split_values = list(split_data.values())
                result["actions"]["splits"] = split_data
                result["summary"]["split_stats"] = {
                    "total_splits": len(split_data),
                    "first_split_date": min(split_data.keys()),
                    "last_split_date": max(split_data.keys()),
                    "split_ratios": split_data
                }
                result["summary"]["total_splits"] = len(split_data)

        # Add additional info from ticker.info
        try:
            info = ticker.info

            # Add upcoming dividend information if available
            if 'dividendDate' in info and info['dividendDate']:
                next_div_date = datetime.fromtimestamp(info['dividendDate'])
                result["upcoming_dividend"] = {
                    "date": next_div_date.strftime("%Y-%m-%d"),
                    "amount": info.get('dividendRate', None)
                }

            # Add ex-dividend date if available
            if 'exDividendDate' in info and info['exDividendDate']:
                ex_div_date = datetime.fromtimestamp(info['exDividendDate'])
                result["ex_dividend_date"] = ex_div_date.strftime("%Y-%m-%d")

        except Exception as e:
            logging.warning(f"Error processing additional info: {str(e)}")

        # Check if we have any data
        if not result["actions"]["dividends"] and not result["actions"]["splits"]:
            logging.warning(f"No corporate actions found for {symbol}")
            return {
                "error": f"No corporate actions available for {symbol}",
                "symbol": symbol,
                "last_updated": result["last_updated"]
            }

        return result

    except Exception as e:
        logging.error(f"Error fetching actions for {symbol}: {str(e)}")
        return {
            "error": f"Failed to fetch actions: {str(e)}",
            "symbol": symbol,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }


@tracked_cache
@mcp.tool()
def fetch_calendar(symbol: str) -> dict:
    """
    Retrieve upcoming company events, such as earnings dates, for a given stock symbol.

    Args:
        symbol (str): The stock symbol of the company (e.g., "AAPL", "TSLA").

    Returns:
        dict: A dictionary containing:
              - `events` (dict): Upcoming earnings and other event details.
              - `last_updated` (str): Timestamp of the data retrieval.
              - `error` (str, optional): Error message if the data retrieval fails.
    """
    try:
        ticker = yf.Ticker(symbol)
        calendar = ticker.calendar
        earnings_dates = ticker.earnings_dates

        result = {
            "symbol": symbol,
            "events": {},
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Process calendar events
        if calendar is not None and not calendar.empty:
            try:
                calendar_dict = {}

                # Process Earnings Date
                if 'Earnings Date' in calendar.index:
                    earnings_date = calendar.loc['Earnings Date'].iloc[0]
                    calendar_dict['earnings_date'] = earnings_date.strftime("%Y-%m-%d") if pd.notnull(
                        earnings_date) else None

                # Process Earnings High/Low
                if 'Earnings High' in calendar.index:
                    calendar_dict['earnings_estimate_high'] = float(
                        calendar.loc['Earnings High'].iloc[0]) if pd.notnull(
                        calendar.loc['Earnings High'].iloc[0]) else None

                if 'Earnings Low' in calendar.index:
                    calendar_dict['earnings_estimate_low'] = float(calendar.loc['Earnings Low'].iloc[0]) if pd.notnull(
                        calendar.loc['Earnings Low'].iloc[0]) else None

                # Process Revenue Forecast
                if 'Revenue Forecast' in calendar.index:
                    calendar_dict['revenue_forecast'] = float(calendar.loc['Revenue Forecast'].iloc[0]) if pd.notnull(
                        calendar.loc['Revenue Forecast'].iloc[0]) else None

                result["events"]["upcoming_earnings"] = calendar_dict

            except Exception as e:
                logging.warning(f"Error processing calendar data: {str(e)}")

        # Process detailed earnings dates
        if earnings_dates is not None and not earnings_dates.empty:
            try:
                earnings_list = []
                for date, row in earnings_dates.iterrows():
                    earnings_event = {
                        "date": date.strftime("%Y-%m-%d"),
                        "estimate": float(row.get('EPS Estimate', None)) if pd.notnull(
                            row.get('EPS Estimate', None)) else None,
                        "actual": float(row.get('Reported EPS', None)) if pd.notnull(
                            row.get('Reported EPS', None)) else None,
                        "surprise": float(row.get('Surprise(%)', None)) if pd.notnull(
                            row.get('Surprise(%)', None)) else None
                    }
                    earnings_list.append(earnings_event)

                result["events"]["earnings_history"] = earnings_list[:4]  # Last 4 earnings dates

            except Exception as e:
                logging.warning(f"Error processing earnings dates: {str(e)}")

        # Add additional company events if available
        try:
            # Get company information
            info = ticker.info

            # Add next dividend date if available
            if 'dividendDate' in info and info['dividendDate'] is not None:
                div_date = datetime.fromtimestamp(info['dividendDate'])
                result["events"]["next_dividend"] = {
                    "date": div_date.strftime("%Y-%m-%d"),
                    "amount": info.get('dividendRate', None)
                }

            # Add ex-dividend date if available
            if 'exDividendDate' in info and info['exDividendDate'] is not None:
                ex_div_date = datetime.fromtimestamp(info['exDividendDate'])
                result["events"]["ex_dividend_date"] = ex_div_date.strftime("%Y-%m-%d")

        except Exception as e:
            logging.warning(f"Error processing additional events: {str(e)}")

        # Check if we have any events
        if not any(result["events"].values()):
            logging.warning(f"No calendar events found for {symbol}")
            return {
                "error": f"No calendar events available for {symbol}",
                "symbol": symbol,
                "last_updated": result["last_updated"]
            }

        return result

    except Exception as e:
        logging.error(f"Error fetching calendar for {symbol}: {str(e)}")
        return {
            "error": f"Failed to fetch calendar: {str(e)}",
            "symbol": symbol,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

@tracked_cache
@mcp.tool()
def fetch_earnings(symbol: str) -> dict:
    """
    Retrieve annual earnings history for a given stock symbol.

    Args:
        symbol (str): The stock symbol of the company (e.g., "AAPL", "TSLA").

    Returns:
        dict: A dictionary containing:
              - Years as keys and earnings data (revenue and net income) as values.
              - `error` (str, optional): Error message if the data retrieval fails.
    """
    try:
        ticker = yf.Ticker(symbol)

        # Income statement -> Net Income (earnings)
        income_stmt = ticker.income_stmt
        # Financials -> Total Revenue
        financials = ticker.financials

        if income_stmt is None or financials is None:
            return {"message": f"No earnings history found for {symbol}"}

        earnings_dict = {}

        for year in income_stmt.columns:
            revenue_val = financials.loc["Total Revenue"].get(year, float("nan"))
            earnings_val = income_stmt.loc["Net Income"].get(year, float("nan"))

            # Replace NaN with None
            revenue = None if (revenue_val is None or math.isnan(revenue_val)) else int(revenue_val)
            earnings = None if (earnings_val is None or math.isnan(earnings_val)) else int(earnings_val)

            earnings_dict[str(year.year)] = {
                "Revenue": revenue,
                "Earnings": earnings
            }

        return earnings_dict

    except Exception as e:
        return {"error": str(e)}

@tracked_cache
@mcp.tool()
def fetch_institutional_holders(symbol: str) -> dict:
    """
    Retrieve institutional holders and ownership summary for a given stock symbol.

    Args:
        symbol (str): The stock symbol of the company (e.g., "AAPL", "MSFT").

    Returns:
        dict: A dictionary containing:
              - `institutional_holders` (list): List of institutional holders and their details.
              - `ownership_summary` (dict): Summary of major holders and ownership percentages.
              - `metadata` (dict): Metadata about the data retrieval.
              - `error` (str, optional): Error message if the data retrieval fails.
    """
    result = {
        "institutional_holders": [],
        "ownership_summary": {},
        "metadata": {
            "symbol": symbol,
            "last_updated": datetime.now().isoformat(),
            "data_source": "yfinance"
        }
    }

    try:
        ticker = yf.Ticker(symbol)

        # Get institutional holders (new API method)
        inst_holders = ticker.get_institutional_holders()

        # Process institutional holders
        if inst_holders is not None and not inst_holders.empty:
            holders_list = []
            for _, row in inst_holders.iterrows():
                pct_held = row.get('% Out')
                holder_data = {
                    "holder": str(row.get('Holder', 'N/A')),
                    "shares": int(row.get('Shares', 0)),
                    "date_reported": str(row.get('Date Reported', 'N/A')),
                    "value": float(row.get('Value', 0)),
                    "pct_held": float(pct_held) if pct_held is not None else None
                }
                holders_list.append(holder_data)

            result["institutional_holders"] = holders_list
            result["metadata"]["total_institutions"] = len(holders_list)
            result["metadata"]["total_shares"] = int(inst_holders['Shares'].sum())

        # Get major holders (ownership breakdown)
        major_holders = ticker.get_major_holders()

        # Process ownership summary
        if major_holders is not None and not major_holders.empty:
            ownership = {}
            for _, row in major_holders.iterrows():
                if len(row) >= 2:
                    key = str(row.iloc[0]).strip()
                    val = str(row.iloc[1]).strip()
                    if '%' in val:
                        val = float(val.replace('%', '')) / 100
                    ownership[key] = val
            result["ownership_summary"] = ownership

    except Exception as e:
        result["error"] = f"Failed to fetch data for {symbol}"
        result["details"] = str(e)

    return result


#new addition of mcp tools for quant-grade analysis

# def calculate_sharpe_ratio(returns, risk_free_rate=0.05):
#     """
#     Calculate the annualized Sharpe Ratio for a given series of returns.
#
#     Args:
#         returns (pd.Series or list): Daily returns of the asset.
#         risk_free_rate (float, optional): Annual risk-free rate. Default is 0.05 (5%).
#
#     Returns:
#         float: Annualized Sharpe Ratio. Returns NaN if the standard deviation of returns is zero.
#     """
#     # Convert to pandas.Series if input is a list
#     if isinstance(returns, list):
#         returns = pd.Series(returns)
#
#     excess_returns = returns - (risk_free_rate / 252)
#     return np.sqrt(252) * excess_returns.mean() / returns.std() if returns.std() != 0 else np.nan


# def calculate_max_drawdown(price_series):
#     """
#     Calculate the maximum drawdown percentage for a given price series.
#
#     Args:
#         price_series (pd.Series or list): Series or list of asset prices.
#
#     Returns:
#         float: Maximum drawdown as a percentage (negative value).
#     """
#     # Convert to pandas.Series if input is a list
#     if isinstance(price_series, list):
#         price_series = pd.Series(price_series)
#
#     rolling_max = price_series.cummax()
#     drawdown = (price_series / rolling_max) - 1
#     return drawdown.min() * 100  # in %

# def calculate_sortino_ratio(returns: pd.Series, risk_free_rate=0.05):
#     """Annualized Sortino ratio."""
#     downside_returns = returns[returns < 0]
#     if downside_returns.std() == 0:
#         return np.nan
#     excess_returns = returns - (risk_free_rate / 252)
#     return np.sqrt(252) * excess_returns.mean() / downside_returns.std()

# def calculate_beta(stock_returns, benchmark_returns):
#     """
#     Calculate the beta of a stock relative to a benchmark.
#
#     Args:
#         stock_returns (pd.Series): Daily returns of the stock.
#         benchmark_returns (pd.Series): Daily returns of the benchmark (e.g., S&P 500).
#
#     Returns:
#         float: Beta value. Returns NaN if the benchmark variance is zero.
#     """
#     # Align the stock and benchmark returns by their indices
#     aligned = pd.concat([stock_returns, benchmark_returns], axis=1).dropna()
#     aligned.columns = ["stock", "benchmark"]
#
#     # Calculate covariance and variance
#     covariance = np.cov(aligned["stock"], aligned["benchmark"])[0][1]
#     benchmark_variance = np.var(aligned["benchmark"])
#
#     return covariance / benchmark_variance if benchmark_variance != 0 else np.nan

# def annualized_volatility(returns):
#     """
#     Calculate the annualized volatility of daily returns.
#
#     Args:
#         returns (pd.Series): A pandas Series of daily returns.
#
#     Returns:
#         float: The annualized volatility, expressed as a percentage.
#     """
#     return returns.std() * np.sqrt(252)

# @mcp.tool()
# def quant_analysis(holdings: dict) -> dict:
#     """
#     Quant-level portfolio tracker with advanced metrics and equity curve.
#     Args:
#         holdings: {"AAPL": 10, "MSFT": 5, "CASH": 5000}
#     Returns:
#         dict: comprehensive portfolio analytics and equity curve data.
#     """
#     try:
#         tickers = [t for t in holdings.keys() if t.upper() != "CASH"]
#         cash_value = holdings.get("CASH", 0)
#         if not tickers and cash_value == 0:
#             return {"error": "No holdings provided."}
#
#         end_date = datetime.now()
#         start_date = end_date - timedelta(days=365)
#
#         # Download price history for all tickers + benchmark
#         all_tickers = tickers + ["SPY"]
#         hist = yf.download(all_tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)
#
#         # Use 'Close' if 'Adj Close' not present
#         if isinstance(hist, pd.DataFrame):
#             if "Adj Close" in hist.columns:
#                 hist = hist["Adj Close"]
#             elif "Close" in hist.columns:
#                 hist = hist["Close"]
#             else:
#                 raise KeyError("Neither 'Adj Close' nor 'Close' column found in the data.")
#
#         # Separate benchmark data
#         spy_prices = hist["SPY"]
#         hist = hist.drop(columns="SPY")
#         hist = hist.dropna(how='all')  # drop dates with no data at all
#
#         # Calculate latest prices & values
#         latest_prices = hist.iloc[-1]
#         prev_prices = hist.iloc[-2]
#         stock_values = {}
#         for t in tickers:
#             qty = holdings[t]
#             price = latest_prices[t]
#             stock_values[t] = price * qty
#         total_stock_value = sum(stock_values.values())
#         total_value = total_stock_value + cash_value
#
#         # Daily portfolio value & change
#         prev_value = sum(prev_prices[t] * holdings[t] for t in tickers) + cash_value
#         daily_change = total_value - prev_value
#         daily_change_pct = (daily_change / prev_value) * 100 if prev_value != 0 else 0
#
#         # Sector exposure
#         sector_exposure = {}
#         for t in tickers:
#             try:
#                 info = yf.Ticker(t).info
#                 sector = info.get("sector", "Unknown")
#             except Exception:
#                 sector = "Unknown"
#             sector_exposure[sector] = sector_exposure.get(sector, 0) + stock_values[t]
#         for s in sector_exposure:
#             sector_exposure[s] = round((sector_exposure[s] / total_value) * 100, 2)
#
#         # Portfolio weights for returns calculation
#         weights = np.array([stock_values[t] / total_stock_value for t in tickers])
#
#         # Calculate daily returns for each stock & portfolio
#         daily_returns = hist.pct_change().dropna()
#         portfolio_returns = daily_returns.dot(weights)
#
#         # Benchmark returns aligned
#         spy_returns = spy_prices.pct_change().dropna()
#         aligned = pd.concat([portfolio_returns, spy_returns], axis=1).dropna()
#         aligned.columns = ["portfolio", "spy"]
#
#         # Performance Metrics
#         cagr = (portfolio_returns.add(1).prod() ** (252 / len(portfolio_returns))) - 1
#         ann_vol = annualized_volatility(portfolio_returns)
#         sharpe = calculate_sharpe_ratio(portfolio_returns)
#         max_dd = calculate_max_drawdown((1 + portfolio_returns).cumprod())
#         beta = calculate_beta(aligned["portfolio"], aligned["spy"])
#
#         # Diversification Index (Herfindahl Index)
#         diversification_index = 1 / np.sum(weights ** 2) if len(weights) > 0 else None
#
#         # Contribution of each holding to total portfolio value (%)
#         contributions = {t: round((stock_values[t] / total_value) * 100, 2) for t in tickers}
#
#         # Equity Curve for portfolio value over time
#         portfolio_value_curve = (1 + portfolio_returns).cumprod() * total_stock_value + cash_value
#         equity_curve = portfolio_value_curve.round(2).to_dict()
#
#         return {
#             "total_value": round(total_value, 2),
#             "cash_value": round(cash_value, 2),
#             "daily_change": round(daily_change, 2),
#             "daily_change_pct": round(daily_change_pct, 2),
#             "sector_exposure": sector_exposure,
#             "holdings_value": stock_values,
#             "contribution_pct": contributions,
#             "performance_metrics": {
#                 "CAGR_pct": round(cagr * 100, 2),
#                 "Annual_Volatility_pct": round(ann_vol * 100, 2),
#                 "Sharpe_Ratio": round(sharpe, 2),
#                 "Max_Drawdown_pct": round(max_dd, 2),
#                 "Beta_vs_SPY": round(beta, 2),
#                 "Diversification_Index": round(diversification_index, 2)
#             },
#             "equity_curve": equity_curve,
#             "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         }
#
#     except Exception as e:
#         logging.exception("Error in portfolio_performance")
#         return {"error": str(e)}

@tracked_cache
@mcp.tool()
def portfolio_quant_analysis(holdings: dict) -> dict:
    """
    Detailed quantitative research analysis with statistical metrics , risk metrics, narrative about portfolio,Top sector exposure,Annualized volatility
    and optimization suggestions. Best for in-depth quant review.
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

@tracked_cache
@mcp.tool()
def forecast_stock(symbol: str, forecast_days: int, p: int, d: int, q: int) -> dict:
    """
    Forecast stock prices using ARIMA.

    Args:
        symbol (str): Stock symbol (e.g., "AAPL").
        forecast_days (int): Number of days to forecast.
        p (int): ARIMA parameter for autoregression.
        d (int): ARIMA parameter for differencing.
        q (int): ARIMA parameter for moving average.
        p (AutoRegressive order): The number of lag observations included in the model. It determines how many past values are used to predict the current value.
        d (Differencing order): The number of times the data needs to be differenced to make it stationary. It accounts for trends in the data.
        q (Moving Average order): The size of the moving average window, which determines how many past forecast errors are used to predict the current value.

    Returns:
        dict: Forecasted prices and metadata.
    """
    try:
        # Fetch historical data
        ticker = yf.Ticker(symbol)
        history = ticker.history(period="1y", interval="1d")
        if history.empty:
            return {"error": f"No historical data available for {symbol}"}

        # Use the 'Close' price for forecasting
        data = history['Close'].dropna()

        # Fit ARIMA model
        model = ARIMA(data, order=(p, d, q))
        fitted_model = model.fit()

        # Generate forecasts
        forecast = fitted_model.get_forecast(steps=forecast_days)
        forecast_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
        forecast_values = forecast.predicted_mean
        conf_int = forecast.conf_int()

        # Prepare results
        result = {
            "symbol": symbol,
            "forecast": {
                "dates": forecast_index.strftime("%Y-%m-%d").tolist(),
                "prices": forecast_values.tolist(),
                "confidence_intervals": conf_int.values.tolist()
            },
            "model_summary": str(fitted_model.summary()),
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        return result

    except Exception as e:
        return {"error": str(e)}





# Internal, non-tool function for statistics logic
def _get_cache_statistics_logic() -> Dict[str, Any]:
    """Shared logic for getting cache statistics."""
    total_requests = tracked_cache.hits + tracked_cache.misses
    hit_ratio = tracked_cache.hits / total_requests if total_requests > 0 else 0

    return {
        "hits": tracked_cache.hits,
        "misses": tracked_cache.misses,
        "total_requests": total_requests,
        "hit_ratio": f"{hit_ratio:.2%}",
        "max_size": price_cache.maxsize,
        "current_size": len(price_cache),
        "ttl_seconds": price_cache.ttl,
        "last_reset_seconds_ago": time.time() - tracked_cache._last_reset,
        "performance": "Excellent" if hit_ratio > 0.7 else "Good" if hit_ratio > 0.4 else "Needs improvement"
    }

@mcp.tool()
def get_cache_stats() -> Dict[str, Any]:
    """Retrieve statistics about the price cache."""
    return _get_cache_statistics_logic()

@mcp.tool()
def reset_cache() -> Dict[str, Any]:
    """Clear the price cache and reset statistics"""
    price_cache.clear()
    tracked_cache.hits = 0
    tracked_cache.misses = 0
    tracked_cache._last_reset = time.time()

    return {
        "message": "Cache cleared successfully",
        "new_stats": _get_cache_statistics_logic()
    }





if __name__ == "__main__":
    mcp.run(transport="stdio")