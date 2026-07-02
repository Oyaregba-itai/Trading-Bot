"""
Advanced trading intelligence features:
  1. Multi-timeframe confirmation (4H + 1D must agree)
  2. Stale position cleanup (flat for 5+ days = free the cash)
  3. Daily loss limit (pause trading if down $300 today)
  4. Earnings blackout (avoid stocks 2 days before earnings)
  5. Crypto funding rates (negative funding = bearish futures sentiment)
  6. Social volume spike (Reddit mention surge = something happening)
  7. Re-entry cooldown (don't immediately re-buy after a loss)
  8. Per-symbol circuit breaker (pause symbol after 3 consecutive losses)
  9. Correlation filter (don't double-up on highly correlated symbols)
  10. Macro event blackout (avoid trading around Fed/NFP/CPI events)
  11. Portfolio heat limit (cap total at-risk across all open positions)
  12. Engulfing candle + volume confirmation entry filter
  13. Trading session quality (scale size by liquidity of current session)
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


# ── 7. Re-entry Cooldown ──────────────────────────────────────────────────────

# How long to wait before re-buying a symbol that just closed as a LOSS,
# scaled to how fast that timeframe moves. Without this, a ranging market
# near a support/resistance level causes immediate re-entry → re-stop-out
# loops (e.g. USDCAD or AVAX getting stopped out 3x in a row at the same price).
_COOLDOWN_HOURS = {"5m": 1, "1h": 4, "1d": 24}


def reentry_cooldown_active(symbol: str, timeframe: str = "1h") -> tuple[bool, str]:
    """
    Returns (cooldown_active, reason).
    Blocks re-entry only after a LOSS/STOP_LOSS/TRAILING_STOP close, not after wins.
    """
    try:
        import database as db
        last = db.get_last_closed_trade(symbol)
        if not last:
            return False, "no prior trade"
        last = dict(last)
        if last.get("result") != "LOSS":
            return False, "last trade was a win"

        closed_at = last.get("closed_at")
        if not closed_at:
            return False, "no close time"

        closed = datetime.fromisoformat(closed_at.replace("Z", "+00:00"))
        if closed.tzinfo is None:
            closed = closed.replace(tzinfo=timezone.utc)

        hours_since = (datetime.now(timezone.utc) - closed).total_seconds() / 3600
        cooldown = _COOLDOWN_HOURS.get(timeframe, 4)

        if hours_since < cooldown:
            remaining = cooldown - hours_since
            return True, f"lost {hours_since:.1f}h ago — cooling down {remaining:.1f}h more ({timeframe})"
        return False, f"cooldown cleared ({hours_since:.1f}h since last loss)"
    except Exception as e:
        logger.debug("Cooldown check error %s: %s", symbol, e)
        return False, "cooldown check failed"


# ── 8. Per-symbol Circuit Breaker ─────────────────────────────────────────────

def symbol_circuit_breaker(symbol: str) -> tuple[bool, str]:
    """
    Block a symbol that has 3+ consecutive losses until 24h after the last loss.
    Prevents the bot from repeatedly re-entering a clearly broken setup.
    """
    try:
        import database as db
        consecutive = db.get_consecutive_losses(symbol)
        if consecutive < 3:
            return False, f"{consecutive} consecutive losses — ok"

        last = db.get_last_closed_trade(symbol)
        if not last:
            return False, "no trade history"
        last = dict(last)
        closed_at = last.get("closed_at", "")
        if not closed_at:
            return False, "no close time"

        closed = datetime.fromisoformat(closed_at.replace("Z", "+00:00"))
        if closed.tzinfo is None:
            closed = closed.replace(tzinfo=timezone.utc)
        hours_since = (datetime.now(timezone.utc) - closed).total_seconds() / 3600

        if hours_since < 24:
            return True, f"{consecutive} consecutive losses — circuit breaker active ({24-hours_since:.0f}h remaining)"
        return False, f"{consecutive} consecutive losses but 24h passed — circuit reset"
    except Exception as e:
        logger.debug("Circuit breaker error %s: %s", symbol, e)
        return False, "circuit breaker check failed"


# ── 9. Correlation Filter ─────────────────────────────────────────────────────

# Predefined correlation groups — symbols within a group move together
_CORR_GROUPS: list[set] = [
    {"BTC", "ETH", "BNB", "SOL", "AVAX", "LTC", "XRP", "ADA", "DOT", "LINK"},
    {"DOGE", "SHIB", "PEPE", "WIF", "BONK", "FLOKI", "SUI", "TRX"},
    {"EURUSD", "GBPUSD", "EURGBP", "AUDUSD"},
    {"USDJPY", "USDCAD", "USDCHF"},
    {"GOLD", "SILVER"},
    {"OIL", "NATGAS"},
    {"AAPL", "MSFT", "NVDA", "AMD", "TSLA", "GOOGL", "META", "AMZN"},
]


def correlation_filter(symbol: str) -> tuple[float, str]:
    """
    Returns (size_multiplier, reason).
    If we already have open positions in the same correlation group,
    reduce new position size to avoid doubling exposure.
    1 existing correlated position → 0.7x size
    2+ existing correlated positions → 0.5x size
    """
    try:
        import database as db
        open_syms = {p["symbol"] for p in (db.get_all_positions() or [])}
        if not open_syms:
            return 1.0, "no open positions"

        sym_group = next((g for g in _CORR_GROUPS if symbol in g), None)
        if sym_group is None:
            return 1.0, "no correlation group"

        correlated_open = open_syms & sym_group
        if not correlated_open:
            return 1.0, "no correlated positions open"

        n = len(correlated_open)
        mult = 0.7 if n == 1 else 0.5
        return mult, f"{n} correlated position(s) open ({', '.join(correlated_open)}) — size {mult:.0%}"
    except Exception as e:
        logger.debug("Correlation filter error %s: %s", symbol, e)
        return 1.0, "correlation check failed"


# ── 10. Macro Event Blackout ──────────────────────────────────────────────────

def _first_friday(year: int, month: int) -> int:
    """Return the day-of-month of the first Friday in a given year/month."""
    import calendar
    cal = calendar.monthcalendar(year, month)
    for week in cal:
        if week[calendar.FRIDAY] != 0:
            return week[calendar.FRIDAY]
    return 7


# FOMC meeting dates 2026 (approximate — decisions announced ~14:00 ET = 19:00 UTC)
_FOMC_2026 = [
    (1, 28), (3, 18), (5, 6), (6, 17),
    (7, 29), (9, 16), (10, 28), (12, 9),
]


def macro_event_blackout(symbol: str) -> tuple[bool, str]:
    """
    Returns (in_blackout, reason).
    Blocks forex, crypto, and stock trades 30 min before and 90 min after:
      - NFP (Non-Farm Payrolls): first Friday of month, 08:30 ET (13:30 UTC)
      - FOMC rate decisions: hardcoded 2026 dates, ~19:00 UTC
      - CPI: second Wednesday of month, 08:30 ET (13:30 UTC)

    Crypto is included because macro news causes large moves on BTC/ETH too.
    """
    from config import STOCK_SYMBOLS
    # Only apply to forex and stocks (and crypto for NFP/FOMC)
    from config import CRYPTO_IDS, COMMODITY_SYMBOLS
    is_forex = len(symbol) == 6 and symbol.isalpha()
    is_stock = symbol in STOCK_SYMBOLS
    is_crypto = symbol in CRYPTO_IDS
    if not (is_forex or is_stock or is_crypto):
        return False, "not affected by macro events"

    now = datetime.now(timezone.utc)
    y, m, d, h, mi = now.year, now.month, now.day, now.hour, now.minute
    now_mins = h * 60 + mi  # minutes since midnight UTC

    def in_window(event_utc_h: int, event_utc_m: int, before_mins=30, after_mins=90) -> bool:
        event_mins = event_utc_h * 60 + event_utc_m
        return (event_mins - before_mins) <= now_mins <= (event_mins + after_mins)

    # NFP — first Friday of each month at 13:30 UTC
    if now.weekday() == 4:  # Friday
        nfp_day = _first_friday(y, m)
        if d == nfp_day and in_window(13, 30):
            return True, f"NFP blackout — Non-Farm Payrolls at 13:30 UTC"

    # FOMC — hardcoded 2026 dates at 19:00 UTC
    if y == 2026:
        for fomc_m, fomc_d in _FOMC_2026:
            if m == fomc_m and d == fomc_d and in_window(19, 0, before_mins=30, after_mins=120):
                return True, f"FOMC blackout — Fed rate decision"

    # CPI — second Wednesday of month at 13:30 UTC
    if now.weekday() == 2:  # Wednesday
        import calendar as cal_mod
        weeks = cal_mod.monthcalendar(y, m)
        wednesdays = [w[cal_mod.WEDNESDAY] for w in weeks if w[cal_mod.WEDNESDAY] != 0]
        if len(wednesdays) >= 2 and d == wednesdays[1] and in_window(13, 30):
            return True, f"CPI blackout — Consumer Price Index release at 13:30 UTC"

    return False, "no macro events"


# ── 11. Portfolio Heat Limit ──────────────────────────────────────────────────

PORTFOLIO_HEAT_LIMIT = 0.25   # Max 25% of wallet at risk simultaneously

def portfolio_heat_ok(new_trade_cash: float, equity: float) -> tuple[bool, str]:
    """
    Returns (ok, reason).
    Calculates total capital currently at risk (cost of all open positions)
    and blocks a new trade if it would push total exposure above 25%.
    """
    try:
        import database as db
        positions = db.get_all_positions() or []
        current_at_risk = sum(p["cost"] for p in positions)
        total_after = current_at_risk + new_trade_cash
        heat = total_after / max(equity, 1)
        if heat > PORTFOLIO_HEAT_LIMIT:
            return False, f"portfolio heat {heat:.0%} would exceed {PORTFOLIO_HEAT_LIMIT:.0%} limit"
        return True, f"portfolio heat {heat:.0%} — ok"
    except Exception as e:
        logger.debug("Portfolio heat error: %s", e)
        return True, "heat check failed — allowing"


# ── 12. Engulfing Candle + Volume Confirmation ────────────────────────────────

def engulfing_confirmed(symbol: str, timeframe: str = "1h") -> tuple[bool, str]:
    """
    Returns (confirmed, reason).
    A bullish engulfing candle with above-average volume is a high-quality entry signal.
    Not blocking — returns False if pattern absent, but bot can still trade (just no boost).
    Used as a quality score bonus, not a hard gate.
    """
    try:
        from ml.trainer import fetch_training_data
        df = fetch_training_data(symbol, timeframe)
        if df is None or len(df) < 22:
            return False, "insufficient data"

        prev  = df.iloc[-2]
        curr  = df.iloc[-1]

        prev_body = abs(prev["Close"] - prev["Open"])
        curr_body = abs(curr["Close"] - curr["Open"])

        # Bullish engulfing: current green candle body engulfs previous candle body
        bullish_engulf = (
            curr["Close"] > curr["Open"] and       # current is green
            prev["Close"] < prev["Open"] and       # previous is red
            curr["Open"] <= prev["Close"] and      # opens below prev close
            curr["Close"] >= prev["Open"] and      # closes above prev open
            curr_body > prev_body * 0.8            # body is substantial
        )

        if not bullish_engulf:
            return False, "no engulfing pattern"

        # Volume confirmation: current volume > 1.2x 20-period average
        avg_vol = float(df["Volume"].tail(20).mean())
        curr_vol = float(curr["Volume"])
        if avg_vol > 0 and curr_vol > avg_vol * 1.2:
            return True, f"bullish engulfing + volume {curr_vol/avg_vol:.1f}x avg"

        return False, f"engulfing pattern but volume weak ({curr_vol/avg_vol:.1f}x avg)"
    except Exception as e:
        logger.debug("Engulfing check error %s: %s", symbol, e)
        return False, "engulfing check failed"


# ── 13. Trading Session Quality ───────────────────────────────────────────────

def session_size_mult(symbol: str) -> tuple[float, str]:
    """
    Returns (size_multiplier, session_name).
    Scale position size by current session liquidity.
    London-NY overlap is the highest quality session for forex.
    Asian session has lower liquidity → smaller forex positions.
    Crypto is unaffected (24/7 equal liquidity).
    """
    from config import CRYPTO_IDS, COMMODITY_SYMBOLS
    if symbol in CRYPTO_IDS:
        return 1.0, "crypto 24/7"

    now  = datetime.now(timezone.utc)
    hour = now.hour

    is_forex = len(symbol) == 6 and symbol.isalpha()

    if is_forex:
        if 13 <= hour < 16:
            return 1.2, "London-NY overlap (peak liquidity)"
        elif 7 <= hour < 13 or 16 <= hour < 21:
            return 1.0, "London or NY session"
        else:
            return 0.7, "Asian/off-hours session (low forex liquidity)"

    if symbol in COMMODITY_SYMBOLS:
        if 13 <= hour < 21:
            return 1.0, "commodity NY session"
        return 0.85, "commodity off-hours"

    # Stocks: only trade during market hours (already gated elsewhere)
    return 1.0, "stock market hours"
