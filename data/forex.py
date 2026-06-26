import requests
import yfinance as yf
from config import ALPHA_VANTAGE_BASE, ALPHA_VANTAGE_KEY


def get_forex_price(pair: str) -> dict | None:
    """pair like 'EURUSD' or 'EUR/USD'"""
    pair = pair.upper().replace("/", "").replace("-", "")
    from_cur = pair[:3]
    to_cur = pair[3:]

    # Try Alpha Vantage first
    if ALPHA_VANTAGE_KEY and ALPHA_VANTAGE_KEY != "demo":
        data = _alpha_vantage_forex(from_cur, to_cur)
        if data:
            return data

    # Fallback: yfinance
    return _yfinance_forex(pair, from_cur, to_cur)


def _alpha_vantage_forex(from_cur: str, to_cur: str) -> dict | None:
    try:
        params = {
            "function": "CURRENCY_EXCHANGE_RATE",
            "from_currency": from_cur,
            "to_currency": to_cur,
            "apikey": ALPHA_VANTAGE_KEY,
        }
        r = requests.get(ALPHA_VANTAGE_BASE, params=params, timeout=10)
        r.raise_for_status()
        d = r.json().get("Realtime Currency Exchange Rate", {})
        if not d:
            return None
        price = float(d["5. Exchange Rate"])
        bid = float(d.get("8. Bid Price", price))
        ask = float(d.get("9. Ask Price", price))
        return {
            "pair": f"{from_cur}/{to_cur}",
            "price": price,
            "bid": bid,
            "ask": ask,
            "change_24h": None,
        }
    except Exception:
        return None


def _yfinance_forex(pair: str, from_cur: str, to_cur: str) -> dict | None:
    try:
        ticker = yf.Ticker(f"{pair}=X")
        hist = ticker.history(period="2d")
        if hist.empty:
            return None
        price = float(hist["Close"].iloc[-1])
        prev = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else price
        change = ((price - prev) / prev) * 100
        return {
            "pair": f"{from_cur}/{to_cur}",
            "price": price,
            "bid": None,
            "ask": None,
            "change_24h": change,
        }
    except Exception:
        return None


def get_multiple_forex(pairs: list[str]) -> list[dict]:
    results = []
    for p in pairs:
        data = get_forex_price(p)
        if data:
            results.append(data)
    return results
