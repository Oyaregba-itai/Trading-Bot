"""
Congressional trade signal.
US politicians must disclose stock trades within 45 days (STOCK Act).
When multiple politicians buy a stock, it's a bullish signal.
When they sell, it's a bearish signal.

Data source: Capitol Trades public API (free, no auth required).
Only applies to US stock symbols (AAPL, NVDA, TSLA, etc.).
"""
import logging
import requests
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

_BASE = "https://api.capitoltrades.com/trades"
_CACHE: dict[str, tuple[float, float, str]] = {}  # symbol → (score, timestamp, reason)
_CACHE_TTL = 3600 * 6  # 6 hours


def get_congressional_signal(symbol: str) -> tuple[float, str]:
    """
    Returns (sentiment_score, reason).
    score > 0  = net buying by politicians (bullish)
    score < 0  = net selling (bearish)
    score = 0  = no recent activity or not a stock

    Only meaningful for US stocks tracked by Capitol Trades.
    """
    import time
    from config import STOCK_SYMBOLS

    if symbol not in STOCK_SYMBOLS:
        return 0.0, "not a US stock"

    # Check cache
    if symbol in _CACHE:
        score, ts, reason = _CACHE[symbol]
        if time.time() - ts < _CACHE_TTL:
            return score, reason

    try:
        # Capitol Trades API — last 60 days of trades for this ticker
        params = {
            "ticker": symbol,
            "pageSize": 50,
            "page": 1,
        }
        r = requests.get(_BASE, params=params, timeout=10,
                         headers={"User-Agent": "TradingBot/1.0"})

        if r.status_code != 200:
            return 0.0, f"Capitol Trades API error {r.status_code}"

        data = r.json()
        trades_raw = data.get("data", [])
        if not trades_raw:
            result = (0.0, "no congressional trades found")
            _CACHE[symbol] = (*result, time.time())
            return result

        # Count buys vs sells in last 60 days
        cutoff = datetime.now(timezone.utc) - timedelta(days=60)
        buys  = 0
        sells = 0
        buy_volume  = 0.0
        sell_volume = 0.0

        for t in trades_raw:
            try:
                tx_date_str = t.get("txDate") or t.get("transactionDate", "")
                if not tx_date_str:
                    continue
                tx_date = datetime.fromisoformat(tx_date_str.replace("Z", "+00:00"))
                if tx_date.tzinfo is None:
                    tx_date = tx_date.replace(tzinfo=timezone.utc)
                if tx_date < cutoff:
                    continue

                tx_type = (t.get("type") or t.get("txType") or "").lower()
                # Estimate size — Capitol Trades uses ranges like "$1,001-$15,000"
                size_str = str(t.get("size") or t.get("amount") or "0")
                size_val = _parse_size(size_str)

                if "purchase" in tx_type or "buy" in tx_type:
                    buys += 1
                    buy_volume += size_val
                elif "sale" in tx_type or "sell" in tx_type:
                    sells += 1
                    sell_volume += size_val
            except Exception:
                continue

        total = buys + sells
        if total == 0:
            result = (0.0, "no recent congressional trades (60d)")
            _CACHE[symbol] = (*result, time.time())
            return result

        net_score = (buys - sells) / total   # -1 to +1
        # Weight by volume if available
        if buy_volume + sell_volume > 0:
            net_score = (buy_volume - sell_volume) / (buy_volume + sell_volume)

        reason = (f"Congress: {buys} buy(s), {sells} sell(s) in 60d "
                  f"(net {'bullish' if net_score > 0 else 'bearish'} {abs(net_score):.0%})")

        result = (round(net_score * 0.3, 3), reason)   # scale to ±0.3 max impact
        _CACHE[symbol] = (*result, time.time())
        return result

    except Exception as e:
        logger.debug("Congressional signal error %s: %s", symbol, e)
        return 0.0, "congressional data unavailable"


def _parse_size(size_str: str) -> float:
    """Parse Capitol Trades size range strings like '$1,001-$15,000' → midpoint."""
    import re
    nums = re.findall(r"[\d,]+", str(size_str).replace("$", ""))
    vals = []
    for n in nums:
        try:
            vals.append(float(n.replace(",", "")))
        except ValueError:
            pass
    if not vals:
        return 0.0
    return sum(vals) / len(vals)
