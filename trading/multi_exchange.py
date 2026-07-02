"""
Multi-exchange price aggregation.
Fetches prices from Binance, Coinbase, and Kraken (all free, no auth).
Benefits:
  1. Best-price execution — use cheapest ask for buys
  2. Divergence signal — large spread between exchanges = momentum signal
  3. More reliable pricing — falls back if one exchange is down
"""
import time
import logging
import requests

logger = logging.getLogger(__name__)

_cache: dict[str, dict] = {}
_CACHE_TTL = 15      # 15 seconds — prices move fast
_TIMEOUT   = 4       # seconds per request

# Kraken uses different ticker symbols
_KRAKEN_MAP = {
    "BTC": "XXBTZUSD", "ETH": "XETHZUSD", "LTC": "XLTCZUSD",
    "XRP": "XXRPZUSD", "ADA": "ADAUSD",   "SOL": "SOLUSD",
    "DOGE": "XDGUSD",  "DOT": "DOTUSD",   "LINK": "LINKUSD",
    "AVAX": "AVAXUSD", "BNB": "BNBUSD",   "TRX": "TRXUSD",
    "MATIC": "MATICUSD",
}

_COINBASE_SUPPORTED = {
    "BTC", "ETH", "SOL", "AVAX", "DOGE", "ADA", "XRP",
    "DOT", "LINK", "LTC", "BNB", "MATIC", "TRX",
}


def _binance_price(symbol: str) -> float | None:
    try:
        r = requests.get(
            "https://api.binance.com/api/v3/ticker/price",
            params={"symbol": f"{symbol}USDT"},
            timeout=_TIMEOUT,
        )
        if r.status_code == 200:
            return float(r.json()["price"])
    except Exception:
        pass
    return None


def _coinbase_price(symbol: str) -> float | None:
    if symbol not in _COINBASE_SUPPORTED:
        return None
    try:
        r = requests.get(
            f"https://api.coinbase.com/v2/prices/{symbol}-USD/spot",
            timeout=_TIMEOUT,
        )
        if r.status_code == 200:
            return float(r.json()["data"]["amount"])
    except Exception:
        pass
    return None


def _kraken_price(symbol: str) -> float | None:
    pair = _KRAKEN_MAP.get(symbol)
    if not pair:
        return None
    try:
        r = requests.get(
            "https://api.kraken.com/0/public/Ticker",
            params={"pair": pair},
            timeout=_TIMEOUT,
        )
        if r.status_code == 200:
            data = r.json()
            if not data.get("error"):
                result = list(data["result"].values())[0]
                return float(result["c"][0])   # last trade price
    except Exception:
        pass
    return None


def get_best_price(symbol: str) -> dict:
    """
    Fetch price from all three exchanges and return aggregated data.

    Returns:
      best_price     : float  — lowest ask price found (best for buying)
      avg_price      : float  — average across exchanges
      divergence     : float  — (max - min) / avg as fraction; >0.003 = meaningful spread
      prices         : dict   — {exchange: price}
      divergence_signal : float [-1,+1] — positive if Coinbase/Kraken leading Binance (buy momentum)
    """
    sym = symbol.upper()
    cached = _cache.get(sym, {})
    if cached and time.time() - cached.get("_ts", 0) < _CACHE_TTL:
        return {k: v for k, v in cached.items() if k != "_ts"}

    prices: dict[str, float] = {}
    binance = _binance_price(sym)
    if binance:
        prices["binance"] = binance
    coinbase = _coinbase_price(sym)
    if coinbase:
        prices["coinbase"] = coinbase
    kraken = _kraken_price(sym)
    if kraken:
        prices["kraken"] = kraken

    if not prices:
        return {"best_price": None, "avg_price": None,
                "divergence": 0.0, "prices": {}, "divergence_signal": 0.0}

    vals       = list(prices.values())
    avg_price  = sum(vals) / len(vals)
    best_price = min(vals)    # lowest = best ask for buying
    divergence = (max(vals) - min(vals)) / avg_price if avg_price > 0 else 0.0

    # Divergence signal: if Coinbase/Kraken are higher than Binance, retail is buying
    # there first → often leads Binance by a few minutes (bullish for Binance)
    divergence_signal = 0.0
    if binance and len(prices) > 1:
        others_avg = sum(v for k, v in prices.items() if k != "binance") / (len(prices) - 1)
        lead = (others_avg - binance) / binance
        divergence_signal = round(max(-1.0, min(1.0, lead / 0.005)), 4)  # 0.5% = ±1.0

    result = {
        "best_price":        round(best_price, 8),
        "avg_price":         round(avg_price, 8),
        "divergence":        round(divergence, 6),
        "prices":            prices,
        "divergence_signal": divergence_signal,
    }
    _cache[sym] = {**result, "_ts": time.time()}
    return result
