from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np

app = FastAPI()

# CORS setup for Thunkable
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Utility functions
def calculate_ema(close_prices, period=20):
    return close_prices.ewm(span=period, adjust=False).mean()

def calculate_macd(close_prices, fast=12, slow=26, signal=9):
    ema_fast = close_prices.ewm(span=fast, adjust=False).mean()
    ema_slow = close_prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_rsi(close_prices, period=14):
    delta = close_prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_stochrsi(rsi_series, period=14):
    min_rsi = rsi_series.rolling(window=period).min()
    max_rsi = rsi_series.rolling(window=period).max()
    stoch_rsi = (rsi_series - min_rsi) / (max_rsi - min_rsi)
    return stoch_rsi * 100  # Convert to %


@app.get("/indicators")
def get_indicators(ticker: str = Query(...)):
    try:
        df = yf.download(ticker, period="6mo", interval="1d")
        if df.empty:
            return {"error": "No data found for this ticker."}

        close = df['Close'].dropna()

        ema_20 = calculate_ema(close, period=20).iloc[-1]
        macd_line, signal_line, hist = calculate_macd(close)
        macd_val = macd_line.iloc[-1]
        signal_val = signal_line.iloc[-1]
        rsi_series = calculate_rsi(close)
        rsi_val = rsi_series.iloc[-1]
        stoch_rsi_series = calculate_stochrsi(rsi_series)
        stoch_rsi_val = stoch_rsi_series.iloc[-1]

            return {
    "ticker": ticker,
    "ema_20": {ticker: round(ema_20, 2)},
    "macd": {ticker: round(macd_val, 2)},
    "macd_signal": {ticker: round(signal_val, 2)},
    "rsi": {ticker: round(rsi_val, 2)},
    "stoch_rsi": {ticker: round(stoch_rsi_val, 2)}
}


    except Exception as e:
        return {"error": str(e)}
