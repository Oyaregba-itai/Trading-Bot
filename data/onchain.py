"""
On-chain crypto signals + Binance order book proxy.
All sources are free, no API key required.

Signals returned (all normalised -1..+1 or 0..1):
  - exchange_netflow:   negative = coins leaving exchanges (bullish), positive = entering (bearish)
  - order_book_imbal:   bid-heavy = positive (buy pressure), ask-heavy = negative (sell pressure)
  - funding_rate_norm:  positive = longs paying shorts (over-leveraged longs, slight bearish)
  - open_interest_chg:  rising OI + rising price = trend continuation; falling OI = reversal risk
"""
import time
import logging
import requests

logger = logging.getLogger(__name__)

# In-memory cache: symbol → {signal_name: value, "_ts": timestamp}
_cache: dict[str, dict] = {}
_CACHE_TTL = 300    # 5 minutes
_REQ_TIMEOUT = 8    # seconds

# Binance ticker map for our symbols
_BINANCE_TICKER = {
    "BTC": "BTCUSDT", "ETH": "ETHUSDT", "BNB": "BNBUSDT",
    "SOL": "SOLUSDT", "AVAX": "AVAXUSDT", "DOGE": "DOGEUSDT",
    "ADA": "ADAUSDT", "XRP": "XRPUSDT", "DOT": "DOTUSDT",
    "LINK": "LINKUSDT", "LTC": "LTCUSDT", "PEPE": "PEPEUSDT",
    "SHIB": "SHIBUSDT", "WIF": "WIFUSDT", "BONK": "BONKUSDT",
    "FLOKI": "FLOKIUSDT", "SUI": "SUIUSDT", "TRX": "TRXUSDT",
}


def _binance_get(path: str, params: dict = None) -> dict | list | None:
    try:
        r = requests.get(
            f"https://api.binance.com{path}",
            params=params, timeout=_REQ_TIMEOUT,
            headers={"User-Agent": "TradingBot/1.0"},
        )
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        logger.debug("Binance REST error %s: %s", path, e)
    return None


def _binance_futures_get(path: str, params: dict = None) -> dict | list | None:
    try:
        r = requests.get(
            f"https://fapi.binance.com{path}",
            params=params, timeout=_REQ_TIMEOUT,
            headers={"User-Agent": "TradingBot/1.0"},
        )
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        logger.debug("Binance Futures REST error %s: %s", path, e)
    return None


# ── Order book imbalance ──────────────────────────────────────────────────────

def _get_order_book_imbalance(ticker: str, depth: int = 20) -> float:
    """
    Fetch top-N bid/ask levels and return imbalance in [-1, +1].
    > 0 = bid-heavy (buy pressure), < 0 = ask-heavy (sell pressure).
    """
    data = _binance_get("/api/v3/depth", {"symbol": ticker, "limit": depth})
    if not data:
        return 0.0
    try:
        bid_vol = sum(float(row[1]) for row in data["bids"])
        ask_vol = sum(float(row[1]) for row in data["asks"])
        total = bid_vol + ask_vol
        if total == 0:
            return 0.0
        return round((bid_vol - ask_vol) / total, 4)
    except Exception:
        return 0.0


# ── Funding rate (perpetual futures) ─────────────────────────────────────────

def _get_funding_rate(ticker: str) -> float:
    """
    Current funding rate for perpetual contract.
    Positive = longs paying shorts (market over-bought).
    Normalised to [-1, +1] assuming ±0.1% is extreme.
    """
    data = _binance_futures_get("/fapi/v1/premiumIndex", {"symbol": ticker})
    if not data:
        return 0.0
    try:
        rate = float(data.get("lastFundingRate", 0))
        return round(max(-1.0, min(1.0, rate / 0.001)), 4)   # 0.1% = extreme
    except Exception:
        return 0.0


# ── Open interest change ──────────────────────────────────────────────────────

def _get_oi_change(ticker: str) -> float:
    """
    Compare current OI vs 8h ago. Positive = rising OI, negative = falling.
    Rising OI + rising price → trend continuation.
    Normalised to [-1, +1].
    """
    data = _binance_futures_get(
        "/futures/data/openInterestHist",
        {"symbol": ticker, "period": "1h", "limit": 9},
    )
    if not data or len(data) < 2:
        return 0.0
    try:
        oldest = float(data[0]["sumOpenInterestValue"])
        newest = float(data[-1]["sumOpenInterestValue"])
        if oldest == 0:
            return 0.0
        chg = (newest - oldest) / oldest
        return round(max(-1.0, min(1.0, chg * 10)), 4)   # 10% change = 1.0
    except Exception:
        return 0.0


# ── Exchange net flow proxy via large trade imbalance ────────────────────────

def _get_taker_buy_ratio(ticker: str) -> float:
    """
    Taker buy volume ratio over last 4h (15m candles).
    > 0.5 = buyers more aggressive (bullish), < 0.5 = sellers.
    Returns normalised to [-1, +1].
    """
    data = _binance_futures_get(
        "/futures/data/takerlongshortRatio",
        {"symbol": ticker, "period": "15m", "limit": 16},
    )
    if not data:
        # Fall back to spot aggregate trades sentiment
        agg = _binance_get("/api/v3/aggTrades", {"symbol": ticker, "limit": 500})
        if not agg:
            return 0.0
        try:
            buy_vol = sum(float(t["q"]) for t in agg if not t["m"])   # m=False → taker is buyer
            sell_vol = sum(float(t["q"]) for t in agg if t["m"])
            total = buy_vol + sell_vol
            if total == 0:
                return 0.0
            ratio = buy_vol / total
            return round((ratio - 0.5) * 2, 4)
        except Exception:
            return 0.0
    try:
        ratios = [float(r["buySellRatio"]) for r in data]
        avg = sum(ratios) / len(ratios)
        return round((avg - 1.0) / 1.0, 4)   # 1.0 = equal, >1 = buyers dominant
    except Exception:
        return 0.0


# ── Main public function ──────────────────────────────────────────────────────

def get_onchain_signals(symbol: str) -> dict:
    """
    Returns dict of on-chain/microstructure signals for a crypto symbol.
    Non-crypto symbols return neutral values (0.0) without API calls.

    Keys:
      order_book_imbal   : [-1, +1]  bid vs ask pressure
      funding_rate_norm  : [-1, +1]  futures funding rate
      oi_change          : [-1, +1]  open interest change (8h)
      taker_buy_ratio    : [-1, +1]  aggressive buyer vs seller dominance
      composite_signal   : [-1, +1]  weighted average of above
    """
    neutral = {
        "order_book_imbal":  0.0,
        "funding_rate_norm": 0.0,
        "oi_change":         0.0,
        "taker_buy_ratio":   0.0,
        "composite_signal":  0.0,
    }

    ticker = _BINANCE_TICKER.get(symbol.upper())
    if not ticker:
        return neutral

    # Serve from cache if fresh
    cached = _cache.get(symbol.upper(), {})
    if cached and (time.time() - cached.get("_ts", 0)) < _CACHE_TTL:
        return {k: v for k, v in cached.items() if k != "_ts"}

    try:
        ob  = _get_order_book_imbalance(ticker)
        fr  = _get_funding_rate(ticker)
        oi  = _get_oi_change(ticker)
        tbr = _get_taker_buy_ratio(ticker)

        # Composite: ob and tbr are most reliable; funding rate is contrarian at extremes
        composite = round(
            ob  * 0.35
            + tbr * 0.35
            + oi  * 0.20
            - fr  * 0.10,   # negative: high funding = overcrowded longs (slight bearish signal)
            4,
        )
        composite = max(-1.0, min(1.0, composite))

        result = {
            "order_book_imbal":  ob,
            "funding_rate_norm": fr,
            "oi_change":         oi,
            "taker_buy_ratio":   tbr,
            "composite_signal":  composite,
        }
        _cache[symbol.upper()] = {**result, "_ts": time.time()}
        return result

    except Exception as e:
        logger.debug("on-chain signals failed for %s: %s", symbol, e)
        return neutral
