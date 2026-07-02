"""
Cross-asset confirmation signals.
Uses free yfinance data — no API key needed.

Signals:
  DXY  (Dollar Index)  : rising DXY = bearish for risk assets / crypto / commodities
  VIX  (Fear gauge)    : rising VIX = risk-off, avoid new longs
  TLT  (Bond prices)   : falling TLT (rising yields) = risk-off for growth stocks
  SPY  (S&P 500 trend) : overall risk sentiment

All signals normalised to [-1, +1]. Cached 30 minutes.
"""
import time
import logging
import pandas as pd

logger = logging.getLogger(__name__)

_cache: dict[str, dict] = {}
_CACHE_TTL = 1800   # 30 minutes


def _fetch(ticker: str, period: str = "30d", interval: str = "1d") -> pd.DataFrame | None:
    try:
        import yfinance as yf
        df = yf.Ticker(ticker).history(period=period, interval=interval)
        if df is not None and len(df) >= 5:
            return df
    except Exception as e:
        logger.debug("cross_asset fetch %s: %s", ticker, e)
    return None


def _pct_change_norm(series: pd.Series, periods: int = 5, scale: float = 0.05) -> float:
    """Normalise recent % change to [-1, +1]. scale = 5% move → ±1.0"""
    try:
        chg = float(series.pct_change(periods).iloc[-1])
        return round(max(-1.0, min(1.0, chg / scale)), 4)
    except Exception:
        return 0.0


def _get_dxy_signal() -> float:
    """
    DXY rising = USD strengthening.
    Bearish for: BTC, ETH, most crypto, EURUSD, GBPUSD, GOLD, OIL
    Bullish for: USDJPY, USDCAD (USD pairs where USD is base)
    Returns raw DXY momentum in [-1, +1]. Caller inverts for USD pairs.
    """
    cached = _cache.get("DXY", {})
    if cached and time.time() - cached.get("_ts", 0) < _CACHE_TTL:
        return cached["value"]
    df = _fetch("DX-Y.NYB", period="30d", interval="1d")
    val = _pct_change_norm(df["Close"], periods=5, scale=0.03) if df is not None else 0.0
    _cache["DXY"] = {"value": val, "_ts": time.time()}
    return val


def _get_vix_signal() -> float:
    """
    VIX > 25 = elevated fear → reduce longs.
    VIX < 15 = complacency → ok to trade.
    Returns risk-off score in [0, 1]. Higher = more fearful.
    """
    cached = _cache.get("VIX", {})
    if cached and time.time() - cached.get("_ts", 0) < _CACHE_TTL:
        return cached["value"]
    df = _fetch("^VIX", period="30d", interval="1d")
    if df is None:
        _cache["VIX"] = {"value": 0.0, "_ts": time.time()}
        return 0.0
    try:
        vix = float(df["Close"].iloc[-1])
        # <15 = calm (0), 15-25 = normal (0.3), 25-35 = fearful (0.7), >35 = panic (1.0)
        if vix < 15:
            val = 0.0
        elif vix < 25:
            val = round((vix - 15) / 10 * 0.4, 4)
        elif vix < 35:
            val = round(0.4 + (vix - 25) / 10 * 0.4, 4)
        else:
            val = min(1.0, round(0.8 + (vix - 35) / 20 * 0.2, 4))
    except Exception:
        val = 0.0
    _cache["VIX"] = {"value": val, "_ts": time.time()}
    return val


def _get_spy_trend() -> float:
    """S&P 500 5-day trend. Positive = risk-on, negative = risk-off."""
    cached = _cache.get("SPY", {})
    if cached and time.time() - cached.get("_ts", 0) < _CACHE_TTL:
        return cached["value"]
    df = _fetch("SPY", period="30d", interval="1d")
    val = _pct_change_norm(df["Close"], periods=5, scale=0.05) if df is not None else 0.0
    _cache["SPY"] = {"value": val, "_ts": time.time()}
    return val


def _get_bond_yield_signal() -> float:
    """
    TLT (20yr treasury ETF) trend. Falling TLT = rising yields = risk-off for growth.
    Returns [-1, +1]. Negative = yields rising (risk-off).
    """
    cached = _cache.get("TLT", {})
    if cached and time.time() - cached.get("_ts", 0) < _CACHE_TTL:
        return cached["value"]
    df = _fetch("TLT", period="30d", interval="1d")
    val = _pct_change_norm(df["Close"], periods=5, scale=0.03) if df is not None else 0.0
    _cache["TLT"] = {"value": val, "_ts": time.time()}
    return val


# ── Asset-class routing ───────────────────────────────────────────────────────

def get_cross_asset_signal(symbol: str) -> dict:
    """
    Returns a confirmation score for the symbol based on macro cross-asset signals.

    Keys:
      confirmation : float [-1, +1]  — positive = macro tailwind, negative = headwind
      vix_risk_off : float [0, 1]    — how fearful the market is (0 = calm)
      reason       : str             — human-readable explanation
    """
    from config import CRYPTO_IDS, COMMODITY_SYMBOLS

    try:
        dxy   = _get_dxy_signal()
        vix   = _get_vix_signal()
        spy   = _get_spy_trend()
        bonds = _get_bond_yield_signal()

        sym = symbol.upper()
        is_crypto    = sym in CRYPTO_IDS
        is_commodity = sym in COMMODITY_SYMBOLS
        is_forex     = len(sym) == 6 and sym.isalpha()
        is_usd_base  = sym.startswith("USD")  # USDJPY, USDCAD, USDCHF

        if is_crypto:
            # Crypto inversely correlated with DXY; positively with SPY and bonds
            conf = -dxy * 0.40 + spy * 0.35 + bonds * 0.15 - vix * 0.10
            reason = f"DXY={dxy:+.2f} SPY={spy:+.2f} VIX_risk={vix:.2f}"

        elif is_commodity and sym in ("GOLD", "SILVER"):
            # Gold rises on DXY weakness and fear
            conf = -dxy * 0.50 + vix * 0.30 - spy * 0.20
            reason = f"DXY={dxy:+.2f} VIX={vix:.2f} SPY={spy:+.2f}"

        elif is_commodity:
            # Oil correlates with risk-on
            conf = spy * 0.50 - dxy * 0.30 - vix * 0.20
            reason = f"SPY={spy:+.2f} DXY={dxy:+.2f} VIX={vix:.2f}"

        elif is_forex and is_usd_base:
            # USD/xxx pairs: DXY bullish = pair rises
            conf = dxy * 0.60 + spy * 0.20 - vix * 0.20
            reason = f"DXY={dxy:+.2f} (USD base pair)"

        elif is_forex:
            # xxx/USD pairs: DXY bullish = pair falls
            conf = -dxy * 0.60 + bonds * 0.20 - vix * 0.20
            reason = f"DXY={dxy:+.2f} (non-USD base)"

        else:
            # Stocks
            conf = spy * 0.50 + bonds * 0.25 - vix * 0.25
            reason = f"SPY={spy:+.2f} TLT={bonds:+.2f} VIX={vix:.2f}"

        conf = round(max(-1.0, min(1.0, conf)), 4)
        return {"confirmation": conf, "vix_risk_off": vix, "reason": reason}

    except Exception as e:
        logger.debug("cross_asset signal error %s: %s", symbol, e)
        return {"confirmation": 0.0, "vix_risk_off": 0.0, "reason": "unavailable"}
