"""
Advanced trading intelligence features:
  1. Multi-timeframe confirmation (4H + 1D must agree)
  2. Stale position cleanup (flat for 5+ days = free the cash)
  3. Daily loss limit (pause trading if down $300 today)
  4. Earnings blackout (avoid stocks 2 days before earnings)
  5. Crypto funding rates (negative funding = bearish futures sentiment)
  6. Social volume spike (Reddit mention surge = something happening)
"""
import logging
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)


# ── 1. Multi-Timeframe Confirmation ──────────────────────────────────────────

def multi_timeframe_signal(symbol: str) -> tuple[str, float, str]:
    """
    Fetch 4H candles and generate a short-term trend signal.
    Returns (signal, confidence, reason)
    Compares with daily to check alignment.
    """
    try:
        import yfinance as yf
        import numpy as np
        from config import CRYPTO_IDS, COMMODITY_SYMBOLS

        # Map symbol to yfinance ticker
        if symbol in CRYPTO_IDS:
            ticker = f"{symbol}-USD"
        elif symbol in COMMODITY_SYMBOLS:
            from config import COMMODITY_SYMBOLS as CS
            ticker = CS[symbol]
        elif len(symbol) == 6 and symbol.isalpha():
            ticker = f"{symbol}=X"
        else:
            ticker = symbol

        df4h = yf.download(ticker, period="10d", interval="4h",
                           auto_adjust=True, progress=False)
        if df4h is None or len(df4h) < 10:
            return "NEUTRAL", 0.5, "insufficient 4H data"

        close = df4h["Close"].squeeze()
        # Simple 4H trend: EMA9 vs EMA21
        ema9  = float(close.ewm(span=9).mean().iloc[-1])
        ema21 = float(close.ewm(span=21).mean().iloc[-1])
        price = float(close.iloc[-1])

        # 4H RSI
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rsi4h = float((100 - 100 / (1 + gain / loss.replace(0, 1e-9))).iloc[-1])

        # Momentum: last 3 candles direction
        last3 = close.tail(3).values
        rising = sum(1 for i in range(1, len(last3)) if last3[i] > last3[i-1])

        if ema9 > ema21 and rsi4h < 70 and rising >= 1:
            conf = 0.60 + min((ema9 - ema21) / ema21 * 10, 0.15)
            return "BUY", round(conf, 3), f"4H EMA9>{ema21:.4f}, RSI4H={rsi4h:.0f}"
        elif ema9 < ema21 and rsi4h > 30 and rising <= 1:
            conf = 0.60 + min((ema21 - ema9) / ema21 * 10, 0.15)
            return "SELL", round(conf, 3), f"4H EMA9<{ema21:.4f}, RSI4H={rsi4h:.0f}"
        else:
            return "NEUTRAL", 0.5, f"4H mixed: EMA9={ema9:.4f}, RSI={rsi4h:.0f}"

    except Exception as e:
        logger.debug("4H signal error %s: %s", symbol, e)
        return "NEUTRAL", 0.5, "4H error"


def timeframe_aligned(symbol: str, daily_signal: str) -> tuple[bool, str]:
    """
    Returns (aligned, reason).
    Trade only if 4H signal agrees with daily ML signal.
    Neutral 4H = allow (don't block, just don't boost).
    """
    signal4h, conf4h, reason = multi_timeframe_signal(symbol)
    if signal4h == "NEUTRAL":
        return True, f"4H neutral — proceeding on daily signal"
    if signal4h == daily_signal:
        return True, f"4H confirms: {signal4h} ({conf4h:.0%}) — {reason}"
    # Disagreement: block the trade
    return False, f"4H disagrees: {signal4h} vs daily {daily_signal} — skipping"


# ── 2. Stale Position Cleanup ─────────────────────────────────────────────────

def check_stale_positions() -> list[dict]:
    """
    Find positions open > 24 hours with no meaningful movement (between -0.3% and +0.5%).
    These are dead money — free the cash for better opportunities.
    Returns list of {symbol, profit_pct, days_open, reason}
    """
    stale = []
    try:
        import database as db
        from trading.demo_wallet import get_portfolio_value
        from trading.auto_trader import _get_price

        positions = db.get_all_positions()
        now = datetime.now(timezone.utc)

        for pos in positions:
            pos = dict(pos)
            opened_at = pos.get("opened_at", "")
            try:
                opened = datetime.fromisoformat(opened_at.replace("Z", "+00:00"))
                if opened.tzinfo is None:
                    opened = opened.replace(tzinfo=timezone.utc)
                days_open = (now - opened).total_seconds() / 86400
            except Exception:
                continue

            if days_open < 1:
                continue

            price = _get_price(pos["symbol"])
            if price is None:
                continue

            entry = pos["entry_price"]
            profit_pct = (price / entry - 1) * 100

            if -0.3 <= profit_pct <= 0.5:
                stale.append({
                    "symbol":     pos["symbol"],
                    "profit_pct": profit_pct,
                    "days_open":  days_open,
                    "price":      price,
                    "reason":     f"Flat for {days_open:.1f} days ({profit_pct:+.2f}%)",
                })
    except Exception as e:
        logger.error("Stale position check error: %s", e)
    return stale


# ── 3. Daily Loss Limit ───────────────────────────────────────────────────────

DAILY_LOSS_LIMIT = -300.0   # Pause all new trades if down $300 today

def daily_pnl_today() -> float:
    """Sum up P&L from all trades closed today (UTC)."""
    try:
        import database as db
        trades = db.get_all_closed_trades()
        today = datetime.now(timezone.utc).date().isoformat()
        return sum(
            (t["pnl"] or 0) for t in trades
            if t["closed_at"] and t["closed_at"][:10] == today
        )
    except Exception:
        return 0.0


def daily_loss_limit_hit() -> tuple[bool, float]:
    """Returns (limit_hit, today_pnl)."""
    pnl = daily_pnl_today()
    return pnl <= DAILY_LOSS_LIMIT, pnl


# ── 4. Earnings Blackout ──────────────────────────────────────────────────────

def earnings_blackout(symbol: str) -> tuple[bool, str]:
    """
    Returns (in_blackout, reason).
    Block stock trades within 2 days before or after earnings.
    """
    from config import STOCK_SYMBOLS
    if symbol not in STOCK_SYMBOLS:
        return False, "not a stock"
    try:
        import yfinance as yf
        cal = yf.Ticker(symbol).calendar
        if cal is None or cal.empty:
            return False, "no earnings data"

        now = datetime.now(timezone.utc).date()
        # calendar index contains 'Earnings Date'
        if "Earnings Date" in cal.index:
            earn_val = cal.loc["Earnings Date"]
            # Could be a single value or a list
            dates = earn_val if hasattr(earn_val, '__iter__') else [earn_val]
            for d in dates:
                try:
                    earn_date = d.date() if hasattr(d, 'date') else d
                    diff = abs((earn_date - now).days)
                    if diff <= 2:
                        return True, f"earnings on {earn_date} ({diff}d away) — blackout active"
                except Exception:
                    continue
        return False, "no upcoming earnings"
    except Exception:
        return False, "earnings check failed"


# ── 5. Crypto Funding Rates ───────────────────────────────────────────────────

def get_funding_rate(symbol: str) -> tuple[float, str]:
    """
    Fetch latest perpetual futures funding rate from Binance (free, no auth).
    Positive funding = longs paying shorts (market bullish/overcrowded)
    Negative funding = shorts paying longs (market bearish/oversold = contrarian buy)
    Returns (rate, interpretation)
    """
    from config import CRYPTO_IDS
    if symbol not in CRYPTO_IDS:
        return 0.0, "not crypto"
    try:
        import requests
        r = requests.get(
            f"https://fapi.binance.com/fapi/v1/fundingRate",
            params={"symbol": f"{symbol}USDT", "limit": 1},
            timeout=5
        )
        if r.status_code != 200:
            return 0.0, "unavailable"
        data = r.json()
        if not data:
            return 0.0, "no data"
        rate = float(data[0]["fundingRate"]) * 100  # convert to %

        if rate > 0.05:
            interp = f"HIGH positive ({rate:+.3f}%) — market very bullish/overcrowded, risky"
        elif rate > 0.01:
            interp = f"positive ({rate:+.3f}%) — mild bullish bias"
        elif rate < -0.05:
            interp = f"NEGATIVE ({rate:+.3f}%) — market oversold, contrarian BUY signal"
        elif rate < -0.01:
            interp = f"slightly negative ({rate:+.3f}%) — mild bearish futures"
        else:
            interp = f"neutral ({rate:+.3f}%)"
        return rate, interp
    except Exception:
        return 0.0, "unavailable"


def funding_rate_size_mult(symbol: str, signal: str) -> float:
    """
    Adjust position size based on funding rate.
    Negative funding + BUY signal = strong contrarian setup, boost size
    Very positive funding + BUY signal = overcrowded trade, reduce size
    """
    rate, _ = get_funding_rate(symbol)
    if rate == 0.0:
        return 1.0
    if signal == "BUY":
        if rate < -0.05:   return 1.20   # contrarian buy — good setup
        if rate < -0.01:   return 1.10
        if rate > 0.08:    return 0.70   # too crowded — risky
        if rate > 0.05:    return 0.85
    return 1.0


# ── 6. Social Volume Spike ────────────────────────────────────────────────────

def social_volume_spike(symbol: str) -> tuple[bool, str]:
    """
    Check Reddit post count for the symbol in last hour vs normal.
    A spike (3x+ normal) means something is happening — news, pump, etc.
    Returns (spike_detected, message)
    """
    try:
        import feedparser, time
        sub_map = {
            "BTC": "Bitcoin", "ETH": "ethereum", "SOL": "solana",
            "DOGE": "dogecoin", "SHIB": "SHIBArmy", "PEPE": "pepecoin",
            "BNB": "binance", "XRP": "Ripple", "ADA": "cardano",
        }
        sub = sub_map.get(symbol, "CryptoCurrency")
        feed = feedparser.parse(
            f"https://www.reddit.com/r/{sub}/new.rss?limit=25",
            request_headers={"User-Agent": "TradingBot/1.0"}
        )
        if not feed.entries:
            return False, "no data"

        now = time.time()
        one_hour_ago = now - 3600
        recent = sum(
            1 for e in feed.entries
            if hasattr(e, "published_parsed") and
            time.mktime(e.published_parsed) > one_hour_ago
        )
        # Normal baseline: ~2-3 posts/hour for mid-size subs
        if recent >= 8:
            return True, f"{recent} Reddit posts in last hour — unusually high activity on {symbol}"
        return False, f"{recent} posts/hour — normal"
    except Exception:
        return False, "unavailable"
