"""
Real-time crypto price feed via Binance public WebSocket.
Maintains an in-memory price cache updated live.
Falls back to REST polling for symbols not on Binance.
"""
import asyncio
import json
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Price cache: symbol → {"price": float, "ts": float, "bid": float, "ask": float}
_price_cache: dict[str, dict] = {}

# Binance symbol map: our symbol → Binance stream symbol (lowercase)
_BINANCE_SYMBOLS: dict[str, str] = {
    "BTC":  "btcusdt",
    "ETH":  "ethusdt",
    "BNB":  "bnbusdt",
    "SOL":  "solusdt",
    "AVAX": "avaxusdt",
    "DOGE": "dogeusdt",
    "ADA":  "adausdt",
    "XRP":  "xrpusdt",
    "DOT":  "dotusdt",
    "LINK": "linkusdt",
    "LTC":  "ltcusdt",
    "PEPE": "pepeusdt",
    "SHIB": "shibusdt",
    "WIF":  "wifusdt",
    "BONK": "bonkusdt",
    "FLOKI":"flokiusdt",
    "SUI":  "suiusdt",
    "TRX":  "trxusdt",
}

_ws_task: Optional[asyncio.Task] = None
_running = False
_MAX_CACHE_AGE = 60   # seconds before cache entry considered stale


def get_cached_price(symbol: str) -> Optional[float]:
    """
    Return live WebSocket price if fresh (<60s), else None.
    Caller falls back to REST polling when None is returned.
    """
    entry = _price_cache.get(symbol.upper())
    if entry and (time.time() - entry["ts"]) < _MAX_CACHE_AGE:
        return entry["price"]
    return None


def get_cached_spread(symbol: str) -> Optional[tuple[float, float]]:
    """Return (bid, ask) if available."""
    entry = _price_cache.get(symbol.upper())
    if entry and (time.time() - entry["ts"]) < _MAX_CACHE_AGE:
        bid = entry.get("bid")
        ask = entry.get("ask")
        if bid and ask:
            return bid, ask
    return None


def get_order_book_imbalance(symbol: str) -> float:
    """
    Returns order book imbalance in [-1, +1].
    +1 = all bids (strong buy pressure)
    -1 = all asks (strong sell pressure)
    0  = balanced / unknown
    """
    entry = _price_cache.get(symbol.upper())
    if not entry:
        return 0.0
    bid_vol = entry.get("bid_qty", 0.0)
    ask_vol = entry.get("ask_qty", 0.0)
    total = bid_vol + ask_vol
    if total == 0:
        return 0.0
    return round((bid_vol - ask_vol) / total, 4)


async def _connect_binance_ws():
    """
    Connect to Binance combined stream for all tracked symbols.
    Processes ticker updates (price, bid, ask) and writes to _price_cache.
    Auto-reconnects on disconnect.
    """
    try:
        import websockets
    except ImportError:
        logger.warning("websockets not installed — real-time price feed disabled")
        return

    streams = "/".join(f"{sym}@bookTicker" for sym in _BINANCE_SYMBOLS.values())
    url = f"wss://stream.binance.com:9443/stream?streams={streams}"
    reverse_map = {v: k for k, v in _BINANCE_SYMBOLS.items()}

    global _running
    while _running:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                logger.info("Binance WebSocket connected — streaming %d symbols", len(_BINANCE_SYMBOLS))
                async for raw in ws:
                    if not _running:
                        break
                    try:
                        msg = json.loads(raw)
                        data = msg.get("data", msg)
                        stream = msg.get("stream", "")
                        # bookTicker payload: s=symbol, b=bestBid, B=bidQty, a=bestAsk, A=askQty
                        sym_lower = data.get("s", "").lower()
                        our_sym   = reverse_map.get(sym_lower)
                        if not our_sym:
                            continue
                        bid   = float(data.get("b", 0))
                        ask   = float(data.get("a", 0))
                        price = (bid + ask) / 2 if bid and ask else bid or ask
                        if price > 0:
                            _price_cache[our_sym] = {
                                "price":   price,
                                "bid":     bid,
                                "ask":     ask,
                                "bid_qty": float(data.get("B", 0)),
                                "ask_qty": float(data.get("A", 0)),
                                "ts":      time.time(),
                            }
                    except Exception:
                        pass
        except Exception as e:
            if _running:
                logger.warning("Binance WS disconnected (%s) — reconnecting in 5s", e)
                await asyncio.sleep(5)


async def start_price_feed():
    """Start the WebSocket price feed in the background. Call once on bot startup."""
    global _ws_task, _running
    if _ws_task and not _ws_task.done():
        return
    _running = True
    _ws_task = asyncio.create_task(_connect_binance_ws())
    logger.info("WebSocket price feed started")


def stop_price_feed():
    """Stop the WebSocket feed gracefully."""
    global _running
    _running = False
    if _ws_task and not _ws_task.done():
        _ws_task.cancel()
