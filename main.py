from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import yfinance as yf
import pandas as pd
import math
import urllib.parse
import json
import numpy as np # Import numpy for NaN handling
import traceback # Import traceback for detailed error logging
import ta.trend # Import ta.trend for trend indicators (e.g., EMA, MACD, ADX)
import ta.momentum # Import ta.momentum for momentum indicators (e.g., RSI)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or your Thunkable domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    """
    Root endpoint for the API. Returns a simple message to confirm the API is running.
    """
    return {"message": "API is running!"}

# Helper function to safely convert a pandas Series to a list,
# replacing NaN values with None and ensuring standard Python floats.
def _safe_list(series: pd.Series):
    """
    Converts a pandas Series to a list, replacing any NaN values with None.
    Also ensures numeric values are standard Python floats for JSON compatibility.
    Includes robust error handling for non-float convertible values.
    """
    cleaned_list = []
    for x in series:
        try:
            # Attempt to convert to standard Python float.
            # This will raise ValueError or TypeError if x is not convertible (e.g., a string).
            float_x = float(x)
            if pd.isna(float_x): # Check if the converted float is NaN
                cleaned_list.append(None)
            else:
                cleaned_list.append(round(float_x, 2))
        except (ValueError, TypeError): # Catch errors if conversion to float fails (e.g., x is a string)
            cleaned_list.append(None) # Treat non-convertible values as missing data
    return cleaned_list

@app.get("/indicators")
def get_indicators(ticker: str = Query(...)):
    """
    Fetches the latest values of various technical indicators for a given stock ticker.

    Args:
        ticker (str): The stock ticker symbol (e.g., "AAPL", "MSFT").

    Returns:
        JSONResponse: A dictionary containing the latest indicator values or an error message.
    """
    try:
        # Download 6 months of daily data for the given ticker to ensure enough data for all indicators
        # Increased period from 3mo to 6mo to provide even more data for indicator calculations.
        data = yf.download(ticker, period="6mo", interval="1d")

        # Check if no data was returned for the ticker
        if data.empty:
            return JSONResponse(status_code=404, content={"error": "Invalid ticker or no data found. Please check the ticker symbol."})

        # Ensure 'Close', 'High', 'Low' columns are 1-dimensional Series using .squeeze()
        close = data["Close"].squeeze()
        high = data["High"].squeeze()
        low = data["Low"].squeeze()

        # Calculate Exponential Moving Average (EMA) 20 using ta.trend
        # fillna=False ensures that leading NaNs (due to insufficient data) are preserved
        ema_20 = ta.trend.ema_indicator(close, window=20, fillna=False).iloc[-1]

        # Calculate MACD and MACD Signal using ta.trend
        macd_line = ta.trend.macd(close, window_fast=12, window_slow=26, fillna=False).iloc[-1]
        macd_signal = ta.trend.macd_signal(close, window_fast=12, window_slow=26, window_sign=9, fillna=False).iloc[-1]

        # Calculate Relative Strength Index (RSI) using ta.momentum
        rsi_series = ta.momentum.rsi(close, window=14, fillna=False)
        rsi_val = rsi_series.iloc[-1]

        # Calculate Stochastic RSI (maintaining original calculation logic: Stochastic of RSI)
        min_rsi = rsi_series.rolling(14).min()
        max_rsi = rsi_series.rolling(14).max()

        stoch_rsi_denominator = max_rsi - min_rsi
        stoch_rsi_numerator = rsi_series - min_rsi # Use rsi_series directly here

        # Conditional calculation: if denominator is zero, set stoch_rsi to 0.0, else calculate
        stoch_rsi_series = pd.Series(np.where(
            stoch_rsi_denominator == 0,
            0.0,  # If denominator is zero, set to 0 (RSI is flat at min_rsi)
            (stoch_rsi_numerator / stoch_rsi_denominator) * 100
        ), index=rsi_series.index)
        stoch_rsi_series = stoch_rsi_series.replace([np.inf, -np.inf], np.nan) # Replace inf with nan if any

        stoch_val = stoch_rsi_series.iloc[-1] # Get the latest Stochastic RSI value

        # Calculate Average Directional Index (ADX) using ta.trend (new indicator)
        adx_val = ta.trend.adx(high, low, close, window=14, fillna=False).iloc[-1]


        # Helper to safely round a float or return None if it's NaN/None
        def safe_round(value):
            # If the value is a Pandas Series, extract its scalar item.
            if isinstance(value, pd.Series):
                if not value.empty: # Ensure the series is not empty
                    value = value.iloc[0] # Get the scalar item
                else:
                    return None # Handle empty series as None

            try:
                # If value is NaN from pandas/numpy (or now a standard float NaN), return None.
                # pd.isna() reliably checks for various forms of NaN.
                if pd.isna(value):
                    return None
                # Otherwise, convert to standard Python float and then round.
                return round(float(value), 2)
            except (ValueError, TypeError): # Catch errors if conversion to float fails (e.g., value is a string)
                return None # Treat non-convertible values as missing data

        result = {
            "ticker": ticker.upper(),
            "ema_20": safe_round(ema_20),
            "macd": safe_round(macd_line), # Changed from 'macd' to 'macd_line' for clarity
            "macd_signal": safe_round(macd_signal),
            "rsi": safe_round(rsi_val),
            "stoch_rsi": safe_round(stoch_val),
            "adx": safe_round(adx_val) # Add ADX to the result
        }

        return result

    except Exception as e:
        # Log the full traceback for debugging in the server logs
        traceback.print_exc()
        # Return a more specific error message to the client
        return JSONResponse(status_code=500, content={"error": f"An unexpected error occurred: {type(e).__name__} - {str(e)}. Please check the server logs for more details."})

@app.get("/timeseries")
def get_timeseries(ticker: str = Query(...)):
    """
    Fetches time series data for various technical indicators for a given stock ticker.

    Args:
        ticker (str): The stock ticker symbol (e.g., "AAPL", "MSFT").

    Returns:
        JSONResponse: A dictionary containing lists of historical indicator values or an error message.
    """
    try:
        # Download 6 months of daily data
        data = yf.download(ticker, period="6mo", interval="1d")
        if data.empty:
            return JSONResponse(status_code=404, content={"error": "Invalid ticker or no data found."})

        # Ensure 'Close', 'High', 'Low' columns are 1-dimensional Series using .squeeze()
        close = data["Close"].squeeze()
        high = data["High"].squeeze()
        low = data["Low"].squeeze()

        # Calculate EMA 20 using ta.trend
        ema_20 = ta.trend.ema_indicator(close, window=20, fillna=False)

        # Calculate MACD and MACD Signal using ta.trend
        macd_line = ta.trend.macd(close, window_fast=12, window_slow=26, fillna=False)
        macd_signal = ta.trend.macd_signal(close, window_fast=12, window_slow=26, window_sign=9, fillna=False)

        # Calculate Relative Strength Index (RSI) using ta.momentum
        rsi = ta.momentum.rsi(close, window=14, fillna=False)

        # Calculate Stochastic RSI (maintaining original calculation logic: Stochastic of RSI)
        min_rsi = rsi.rolling(14).min()
        max_rsi = rsi.rolling(14).max()
        
        stoch_rsi_denominator = max_rsi - min_rsi
        stoch_rsi_numerator = rsi - min_rsi

        stoch_rsi = pd.Series(np.where(
            stoch_rsi_denominator == 0,
            0.0, # If denominator is zero, set to 0
            (stoch_rsi_numerator / stoch_rsi_denominator) * 100
        ), index=rsi.index)
        stoch_rsi = stoch_rsi.replace([np.inf, -np.inf], np.nan)

        # Calculate Average Directional Index (ADX) using ta.trend
        adx = ta.trend.adx(high, low, close, window=14, fillna=False)


        return {
            "labels": list(data.index.strftime("%Y-%m-%d")),
            "close": _safe_list(close), # _safe_list handles rounding and NaNs internally
            "ema_20": _safe_list(ema_20), # _safe_list handles rounding and NaNs internally
            "macd": _safe_list(macd_line), # _safe_list handles rounding and NaNs internally
            "macd_signal": _safe_list(macd_signal), # _safe_list handles rounding and NaNs internally
            "rsi": _safe_list(rsi), # _safe_list handles rounding and NaNs internally
            "stoch_rsi": _safe_list(stoch_rsi), # _safe_list handles rounding and NaNs internally
            "adx": _safe_list(adx) # Add ADX to the timeseries result
        }

    except Exception as e:
        # Log the full traceback for debugging in the server logs
        traceback.print_exc()
        # Return a more specific error message to the client
        return JSONResponse(status_code=500, content={"error": f"An unexpected error occurred: {type(e).__name__} - {str(e)}. Please check the server logs for more details."})

@app.get("/chart")
def get_combined_chart_url(ticker: str = Query(...)):
    """
    Generates a QuickChart.io URL for a combined chart of various technical indicators
    for a given stock ticker.

    Args:
        ticker (str): The stock ticker symbol (e.g., "AAPL", "MSFT").

    Returns:
        JSONResponse: A dictionary containing the chart URL or an error message.
    """
    try:
        # Download 6 months of daily data
        data = yf.download(ticker, period="6mo", interval="1d")
        if data.empty:
            return JSONResponse(status_code=404, content={"error": "Invalid ticker or no data found."})

        # Ensure 'Close', 'High', 'Low' columns are 1-dimensional Series using .squeeze()
        close = data["Close"].squeeze()
        high = data["High"].squeeze()
        low = data["Low"].squeeze()

        # Calculate EMA 20 using ta.trend
        ema_20 = ta.trend.ema_indicator(close, window=20, fillna=False)

        # Calculate MACD and MACD Signal using ta.trend
        macd_line = ta.trend.macd(close, window_fast=12, window_slow=26, fillna=False)
        macd_signal = ta.trend.macd_signal(close, window_fast=12, window_slow=26, window_sign=9, fillna=False)

        # Calculate RSI using ta.momentum
        rsi = ta.momentum.rsi(close, window=14, fillna=False)

        # Calculate Stochastic RSI (maintaining original calculation logic: Stochastic of RSI)
        min_rsi = rsi.rolling(14).min()
        max_rsi = rsi.rolling(14).max()
        
        stoch_rsi_denominator = max_rsi - min_rsi
        stoch_rsi_numerator = rsi - min_rsi

        stoch_rsi = pd.Series(np.where(
            stoch_rsi_denominator == 0,
            0.0, # If denominator is zero, set to 0
            (stoch_rsi_numerator / stoch_rsi_denominator) * 100
        ), index=rsi.index)
        stoch_rsi = stoch_rsi.replace([np.inf, -np.inf], np.nan)

        # Calculate Average Directional Index (ADX) using ta.trend
        adx = ta.trend.adx(high, low, close, window=14, fillna=False)


        labels = list(data.index.strftime("%Y-%m-%d"))

        chart_data = {
            "type": "line",
            "data": {
                "labels": labels,
                "datasets": [
                    {"label": "Close", "data": _safe_list(close), "borderColor": "blue", "fill": False},
                    {"label": "EMA 20", "data": _safe_list(ema_20), "borderColor": "orange", "fill": False},
                    {"label": "MACD", "data": _safe_list(macd_line), "borderColor": "purple", "fill": False},
                    {"label": "MACD Signal", "data": _safe_list(macd_signal), "borderColor": "pink", "fill": False},
                    {"label": "RSI", "data": _safe_list(rsi), "borderColor": "green", "fill": False},
                    {"label": "Stoch RSI", "data": _safe_list(stoch_rsi), "borderColor": "red", "fill": False},
                    {"label": "ADX", "data": _safe_list(adx), "borderColor": "brown", "fill": False}, # Add ADX to the chart
                ]
            },
            "options": {
                "title": {"display": True, "text": f"{ticker.upper()} Technical Indicators"},
                "scales": {"y": {"beginAtZero": False}}
            }
        }

        # Encode the chart data to be used in the QuickChart.io URL
        encoded = urllib.parse.quote(json.dumps(chart_data))
        url = f"https://quickchart.io/chart?c={encoded}"
        return {"chart_url": url}

    except Exception as e:
        # Log the full traceback for debugging in the server logs
        traceback.print_exc()
        # Return a more specific error message to the client
        return JSONResponse(status_code=500, content={"error": f"An unexpected error occurred: {type(e).__name__} - {str(e)}. Please check the server logs for more details."})
