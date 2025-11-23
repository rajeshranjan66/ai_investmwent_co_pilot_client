# from statsmodels.tsa.arima.model import ARIMA
# import yfinance as yf
# import pandas as pd
# from fastmcp import FastMCP
# from datetime import datetime
#
# mcp = FastMCP("stockForecastMCPServer", "1.0.0", "A server to forecast stock prices using ARIMA")
#
# @mcp.tool()
# def forecast_stock(symbol: str, forecast_days: int, p: int, d: int, q: int) -> dict:
#     """
#     Forecast stock prices using ARIMA.
#
#     Args:
#         symbol (str): Stock symbol (e.g., "AAPL").
#         forecast_days (int): Number of days to forecast.
#         p (int): ARIMA parameter for autoregression.
#         d (int): ARIMA parameter for differencing.
#         q (int): ARIMA parameter for moving average.
#         p (AutoRegressive order): The number of lag observations included in the model. It determines how many past values are used to predict the current value.
#         d (Differencing order): The number of times the data needs to be differenced to make it stationary. It accounts for trends in the data.
#         q (Moving Average order): The size of the moving average window, which determines how many past forecast errors are used to predict the current value.
#
#     Returns:
#         dict: Forecasted prices and metadata.
#     """
#     try:
#         # Fetch historical data
#         ticker = yf.Ticker(symbol)
#         history = ticker.history(period="1y", interval="1d")
#         if history.empty:
#             return {"error": f"No historical data available for {symbol}"}
#
#         # Use the 'Close' price for forecasting
#         data = history['Close'].dropna()
#
#         # Fit ARIMA model
#         model = ARIMA(data, order=(p, d, q))
#         fitted_model = model.fit()
#
#         # Generate forecasts
#         forecast = fitted_model.get_forecast(steps=forecast_days)
#         forecast_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
#         forecast_values = forecast.predicted_mean
#         conf_int = forecast.conf_int()
#
#         # Prepare results
#         result = {
#             "symbol": symbol,
#             "forecast": {
#                 "dates": forecast_index.strftime("%Y-%m-%d").tolist(),
#                 "prices": forecast_values.tolist(),
#                 "confidence_intervals": conf_int.values.tolist()
#             },
#             "model_summary": str(fitted_model.summary()),
#             "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         }
#         return result
#
#     except Exception as e:
#         return {"error": str(e)}
#
# if __name__ == "__main__":
#     mcp.run(transport="stdio")