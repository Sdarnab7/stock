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
@app.get("/timeseries")
def get_timeseries(ticker: str = Query(...)):
    try:
        data = yf.download(ticker, period="1mo", interval="1d")
        if data.empty:
            return JSONResponse(status_code=404, content={"error": "Invalid ticker or no data"})

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
        rsi = 100 - (100 / (1 + rs))

        # Stoch RSI
        min_rsi = rsi.rolling(14).min()
        max_rsi = rsi.rolling(14).max()
        stoch_rsi = ((rsi - min_rsi) / (max_rsi - min_rsi)) * 100

        return {
            "labels": list(data.index.strftime("%Y-%m-%d")),
            "close": list(close.round(2)),
            "ema_20": list(ema_20.round(2)),
            "macd": list(macd_line.round(2)),
            "macd_signal": list(macd_signal.round(2)),
            "rsi": list(rsi.round(2)),
            "stoch_rsi": list(stoch_rsi.round(2)),
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
@app.get("/chart")
def get_combined_chart_url(ticker: str = Query(...)):
    try:
        data = yf.download(ticker, period="1mo", interval="1d")
        close = data["Close"]
        ema_20 = close.ewm(span=20, adjust=False).mean()
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        macd_signal = macd_line.ewm(span=9, adjust=False).mean()

        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -1 * delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        min_rsi = rsi.rolling(14).min()
        max_rsi = rsi.rolling(14).max()
        stoch_rsi = ((rsi - min_rsi) / (max_rsi - min_rsi)) * 100

        labels = list(data.index.strftime("%Y-%m-%d"))

        chart_data = {
            "type": "line",
            "data": {
                "labels": labels,
                "datasets": [
                    {"label": "Close", "data": list(close.round(2)), "borderColor": "blue", "fill": False},
                    {"label": "EMA 20", "data": list(ema_20.round(2)), "borderColor": "orange", "fill": False},
                    {"label": "MACD", "data": list(macd_line.round(2)), "borderColor": "purple", "fill": False},
                    {"label": "MACD Signal", "data": list(macd_signal.round(2)), "borderColor": "pink", "fill": False},
                    {"label": "RSI", "data": list(rsi.round(2)), "borderColor": "green", "fill": False},
                    {"label": "Stoch RSI", "data": list(stoch_rsi.round(2)), "borderColor": "red", "fill": False},
                ]
            },
            "options": {
                "title": {"display": True, "text": f"{ticker.upper()} Technical Indicators"},
                "scales": {"y": {"beginAtZero": False}}
            }
        }

        encoded = urllib.parse.quote(json.dumps(chart_data))
        url = f"https://quickchart.io/chart?c={encoded}"
        return {"chart_url": url}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


