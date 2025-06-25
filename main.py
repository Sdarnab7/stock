from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import yfinance as yf
import pandas as pd
import math

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
    return {"message": "API is running!"}

@app.get("/indicators")
def get_indicators(ticker: str = Query(...)):
    try:
        data = yf.download(ticker, period="1mo", interval="1d")
        if data.empty:
            return JSONResponse(status_code=404, content={"error": "Invalid ticker or no data"})

        close = data["Close"]

        ema_20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        macd = macd_line.iloc[-1]
        signal_line = macd_line.ewm(span=9, adjust=False).mean().iloc[-1]

        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -1 * delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi_val = rsi.iloc[-1]

        min_rsi = rsi.rolling(14).min()
        max_rsi = rsi.rolling(14).max()
        stoch_rsi = ((rsi - min_rsi) / (max_rsi - min_rsi)) * 100
        stoch_val = stoch_rsi.iloc[-1]

        def safe(value):
            return None if (value is None or isinstance(value, float) and math.isnan(value)) else round(value, 2)

        result = {
            "ticker": ticker.upper(),
            "ema_20": safe(ema_20),
            "macd": safe(macd),
            "macd_signal": safe(signal_line),
            "rsi": safe(rsi_val),
            "stoch_rsi": safe(stoch_val)
        }

        return result

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

