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
    """
    cleaned_list = []
    for x in series:
        if pd.isna(x): # Check for NaN or None (including numpy NaNs)
            cleaned_list.append(None)
        else:
            # Explicitly convert to standard Python float before rounding
            cleaned_list.append(round(float(x), 2))
    return cleaned_list

@app.get("/indicators")
def get_indicators(ticker: str = Query(...)):
    """
    Fetches the latest values of various technical indicators for a given stock ticker.

    Args:
        ticker (str): The stock ticker symbol (e.g., "AAPL", "MSFT").

    Returns:
        JSONResponse: A dictionary containing the latest indicator values or an message.
    """
    try:
        # Download 1 month of daily data for the given ticker
        data = yf.download(ticker, period="1mo", interval="1d")

        # Check if no data was returned for the ticker
        if data.empty:
            return JSONResponse(status_code=404, content={"error": "Invalid ticker or no data found. Please check the ticker symbol."})

        close = data["Close"]

        # Calculate Exponential Moving Average (EMA) 20
        # adjust=False matches calculation methods found in many charting platforms.
        ema_20 = close.ewm(span=20, adjust=False).mean().iloc[-1]

        # Calculate MACD (Moving Average Convergence Divergence)
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        macd = macd_line.iloc[-1]
        signal_line = macd_line.ewm(span=9, adjust=False).mean().iloc[-1]

        # Calculate Relative Strength Index (RSI)
        delta = close.diff()
        gain = delta.clip(lower=0) # Positive changes
        loss = -1 * delta.clip(upper=0) # Negative changes (as positive values)

        # Average gains and losses over 14 periods
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()

        # Handle division by zero for rs (when avg_loss is 0)
        # Replace NaN in rs where avg_loss is 0 with infinity, then handle in RSI formula
        rs = avg_gain / avg_loss
        rs = rs.replace([np.inf, -np.inf], np.nan) # Replace inf with nan if any

        # Calculate RSI: 100 - (100 / (1 + RS))
        rsi = 100 - (100 / (1 + rs))
        rsi_val = rsi.iloc[-1] # Get the latest RSI value

        # Calculate Stochastic RSI
        # Min and Max RSI over 14 periods
        min_rsi = rsi.rolling(14).min()
        max_rsi = rsi.rolling(14).max()

        # Handle division by zero for stoch_rsi (when max_rsi - min_rsi is 0)
        # Add a small epsilon to denominator or check for zero difference
        stoch_rsi = ((rsi - min_rsi) / (max_rsi - min_rsi)) * 100
        # If max_rsi == min_rsi, stoch_rsi would be NaN, so we handle it gracefully.
        stoch_rsi = stoch_rsi.replace([np.inf, -np.inf], np.nan) # Replace inf with nan if any

        stoch_val = stoch_rsi.iloc[-1] # Get the latest Stochastic RSI value

        # Helper to safely round a float or return None if it's NaN/None
        def safe_round(value):
            # If the value is a Pandas Series with a single item, extract it.
            if isinstance(value, pd.Series) and len(value) == 1:
                value = value.iloc[0]

            # If it's NaN from pandas/numpy (or now a standard float NaN), return None.
            if pd.isna(value):
                return None
            # Otherwise, convert to standard float and then round.
            return round(float(value), 2)

        result = {
            "ticker": ticker.upper(),
            "ema_20": safe_round(ema_20),
            "macd": safe_round(macd),
            "macd_signal": safe_round(signal_line),
            "rsi": safe_round(rsi_val),
            "stoch_rsi": safe_round(stoch_val)
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
        # Download 1 month of daily data
        data = yf.download(ticker, period="1mo", interval="1d")
        if data.empty:
            return JSONResponse(status_code=404, content={"error": "Invalid ticker or no data found."})

        close = data["Close"]

        # Calculate EMA 20
        ema_20 = close.ewm(span=20, adjust=False).mean()

        # Calculate MACD
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()

        # Calculate RSI
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -1 * delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        rs = rs.replace([np.inf, -np.inf], np.nan)
        rsi = 100 - (100 / (1 + rs))

        # Calculate Stoch RSI
        min_rsi = rsi.rolling(14).min()
        max_rsi = rsi.rolling(14).max()
        stoch_rsi = ((rsi - min_rsi) / (max_rsi - min_rsi)) * 100
        stoch_rsi = stoch_rsi.replace([np.inf, -np.inf], np.nan)


        return {
            "labels": list(data.index.strftime("%Y-%m-%d")),
            "close": _safe_list(close.round(2)), # Use helper for clean list
            "ema_20": _safe_list(ema_20.round(2)), # Use helper for clean list
            "macd": _safe_list(macd_line.round(2)), # Use helper for clean list
            "macd_signal": _safe_list(macd_signal.round(2)), # Use helper for clean list
            "rsi": _safe_list(rsi.round(2)), # Use helper for clean list
            "stoch_rsi": _safe_list(stoch_rsi.round(2)), # Use helper for clean list
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
        # Download 1 month of daily data
        data = yf.download(ticker, period="1mo", interval="1d")
        if data.empty:
            return JSONResponse(status_code=404, content={"error": "Invalid ticker or no data found."})

        close = data["Close"]
        ema_20 = close.ewm(span=20, adjust=False).mean()
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()

        # RSI
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -1 * delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        rs = rs.replace([np.inf, -np.inf], np.nan)
        rsi = 100 - (100 / (1 + rs))

        # Stoch RSI
        min_rsi = rsi.rolling(14).min()
        max_rsi = rsi.rolling(14).max()
        stoch_rsi = ((rsi - min_rsi) / (max_rsi - min_rsi)) * 100
        stoch_rsi = stoch_rsi.replace([np.inf, -np.inf], np.nan)


        labels = list(data.index.strftime("%Y-%m-%d"))

        chart_data = {
            "type": "line",
            "data": {
                "labels": labels,
                "datasets": [
                    {"label": "Close", "data": _safe_list(close.round(2)), "borderColor": "blue", "fill": False},
                    {"label": "EMA 20", "data": _safe_list(ema_20.round(2)), "borderColor": "orange", "fill": False},
                    {"label": "MACD", "data": _safe_list(macd_line.round(2)), "borderColor": "purple", "fill": False},
                    {"label": "MACD Signal", "data": _safe_list(macd_signal.round(2)), "borderColor": "pink", "fill": False},
                    {"label": "RSI", "data": _safe_list(rsi.round(2)), "borderColor": "green", "fill": False},
                    {"label": "Stoch RSI", "data": _safe_list(stoch_rsi.round(2)), "borderColor": "red", "fill": False},
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

