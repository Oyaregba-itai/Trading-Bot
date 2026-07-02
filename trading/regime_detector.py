"""
Market regime classifier.
Detects the current market environment for a symbol to guide strategy.

Regimes:
  BULL_TREND     — Strong uptrend, EMAs aligned up, good for BUY entries
  BEAR_TREND     — Strong downtrend, EMAs aligned down, skip BUY entries
  RANGING        — Low trend strength, price oscillating — skip new entries
  HIGH_VOLATILITY — Abnormal volatility spike — reduce position size

Uses price data only (no external APIs), so it works for all asset classes.
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)

REGIME_BULL      = "BULL_TREND"
REGIME_BEAR      = "BEAR_TREND"
REGIME_RANGING   = "RANGING"
REGIME_HIGH_VOL  = "HIGH_VOLATILITY"
REGIME_UNKNOWN   = "UNKNOWN"

# Cache regime per symbol to avoid repeated fetches within same cycle
_cache: dict[str, tuple[str, float, str]] = {}
_cache_ts: dict[str, float] = {}
_CACHE_TTL = 900  # 15 minutes


def _adx_like(high, low, close, period=14) -> float:
    """
    Simplified trend strength score (0-100, similar to ADX).
    High value = strong trend (either direction).
    Low value = ranging/choppy.
    """
    import pandas as pd
    h = pd.Series(high)
    l = pd.Series(low)
    c = pd.Series(close)

    up   = h.diff()
    down = -l.diff()
    plus_dm  = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    pdi = 100 * pd.Series(plus_dm).rolling(period).mean() / atr.replace(0, 1e-9)
    mdi = 100 * pd.Series(minus_dm).rolling(period).mean() / atr.replace(0, 1e-9)

    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, 1e-9)
    adx = dx.rolling(period).mean()
    return float(adx.iloc[-1]) if not adx.empty else 20.0


def detect_regime(symbol: str, timeframe: str = "1h") -> tuple[str, float, str]:
    """
    Returns (regime, confidence, reason).

    Trading rules applied by auto_trader:
      RANGING      → skip BUY entries entirely
      BEAR_TREND   → skip BUY entries
      HIGH_VOL     → allow but halve position size
      BULL_TREND   → trade normally
      UNKNOWN      → trade normally (no data = don't block)
    """
    import time
    cache_key = f"{symbol}_{timeframe}"
    now = time.time()
    if cache_key in _cache and now - _cache_ts.get(cache_key, 0) < _CACHE_TTL:
        return _cache[cache_key]

    try:
        from ml.trainer import fetch_training_data
        df = fetch_training_data(symbol, timeframe)
        if df is None or len(df) < 50:
            return REGIME_UNKNOWN, 0.5, "insufficient data"

        close  = df["Close"].values
        high   = df["High"].values
        low    = df["Low"].values

        # ── EMAs ─────────────────────────────────────────────────────────────
        import pandas as pd
        s = pd.Series(close)
        ema20  = float(s.ewm(span=20).mean().iloc[-1])
        ema50  = float(s.ewm(span=50).mean().iloc[-1])
        ema200 = float(s.ewm(span=min(200, len(s))).mean().iloc[-1])
        price  = float(close[-1])

        ema20_slope  = float(s.ewm(span=20).mean().diff(3).iloc[-1])  # rising/falling

        # ── Volatility rank ───────────────────────────────────────────────────
        returns = pd.Series(close).pct_change().dropna()
        vol_current = float(returns.tail(14).std())
        vol_90      = float(returns.tail(90).std()) if len(returns) >= 90 else vol_current
        vol_rank    = vol_current / (vol_90 + 1e-9)

        # ── Trend strength (ADX-like) ─────────────────────────────────────────
        adx = _adx_like(high, low, close)

        # ── Ranging detection: price oscillating around EMA ───────────────────
        # Count how many of last 20 closes crossed EMA20
        tail20 = close[-20:]
        ema20_series = s.ewm(span=20).mean().values[-20:]
        crossings = sum(
            1 for i in range(1, len(tail20))
            if (tail20[i-1] - ema20_series[i-1]) * (tail20[i] - ema20_series[i]) < 0
        )

        # ── Classification ────────────────────────────────────────────────────

        # HIGH_VOLATILITY: current vol > 2.5x 90-day average
        if vol_rank > 2.5:
            reason = f"vol {vol_rank:.1f}x above normal (ADX={adx:.0f})"
            result = (REGIME_HIGH_VOL, min(0.5 + vol_rank * 0.1, 0.95), reason)
            _cache[cache_key] = result
            _cache_ts[cache_key] = now
            return result

        # RANGING: low ADX and price crossing EMA frequently
        if adx < 20 and crossings >= 4:
            reason = f"ranging — ADX={adx:.0f}, {crossings} EMA crossings in 20 periods"
            result = (REGIME_RANGING, 0.65 + crossings * 0.02, reason)
            _cache[cache_key] = result
            _cache_ts[cache_key] = now
            return result

        # BULL_TREND: EMAs aligned up, positive slope, price above EMAs
        if ema20 > ema50 and price > ema20 and ema20_slope > 0 and adx > 20:
            gap    = (ema20 - ema50) / ema50
            conf   = min(0.60 + gap * 100 + adx / 200, 0.92)
            reason = f"bull — EMA20>{ema50:.2f}, ADX={adx:.0f}, slope+{ema20_slope:.4f}"
            result = (REGIME_BULL, conf, reason)
            _cache[cache_key] = result
            _cache_ts[cache_key] = now
            return result

        # BEAR_TREND: EMAs aligned down, negative slope
        if ema20 < ema50 and price < ema20 and ema20_slope < 0 and adx > 20:
            gap    = (ema50 - ema20) / ema50
            conf   = min(0.60 + gap * 100 + adx / 200, 0.92)
            reason = f"bear — EMA20<{ema50:.2f}, ADX={adx:.0f}, slope{ema20_slope:.4f}"
            result = (REGIME_BEAR, conf, reason)
            _cache[cache_key] = result
            _cache_ts[cache_key] = now
            return result

        # Default: weak trend but not clearly ranging
        reason = f"weak trend — ADX={adx:.0f}, crossings={crossings}"
        result = (REGIME_BULL if ema20 > ema50 else REGIME_BEAR, 0.50, reason)
        _cache[cache_key] = result
        _cache_ts[cache_key] = now
        return result

    except Exception as e:
        logger.debug("Regime detection error %s: %s", symbol, e)
        return REGIME_UNKNOWN, 0.5, f"error: {e}"


def regime_allows_buy(symbol: str, timeframe: str = "1h") -> tuple[bool, float, str]:
    """
    Convenience wrapper used by auto_trader.
    Returns (allow_buy, size_multiplier, reason).
      RANGING / BEAR  → block, size_mult irrelevant
      HIGH_VOL        → allow but size_mult = 0.5
      BULL / UNKNOWN  → allow, size_mult = 1.0
    """
    regime, conf, reason = detect_regime(symbol, timeframe)

    if regime == REGIME_RANGING:
        return False, 0.0, f"Ranging market — {reason}"
    if regime == REGIME_BEAR:
        return False, 0.0, f"Bear trend — {reason}"
    if regime == REGIME_HIGH_VOL:
        return True, 0.5, f"High volatility — half size — {reason}"
    return True, 1.0, f"{regime} — {reason}"
