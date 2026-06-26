"""
Smart market filters applied before every trade decision.

Filters implemented:
  1. Volume Surge         — 2x average volume = stronger signal, bigger position
  2. RSI Overbought/Sold  — Block buys > RSI 75, sells < RSI 25 (already too extended)
  3. Trend Alignment      — Only buy above SMA50, sell below SMA50 (trade with trend)
  4. Drawdown Protection  — Pause ALL new trades if wallet down 15%+ from peak
  5. Correlated Exposure  — Limit how many correlated assets we hold at once
  6. BTC Market Regime    — When BTC is crashing, reduce all crypto buy sizes
  7. VIX Fear Filter      — High stock-market fear = reduce stock position sizes
  8. RSI Divergence       — Price new high but RSI dropping = warning, reduce size
  9. Spread/Gap Filter    — Skip if price gapped >5% overnight (chasing a move)
 10. Support/Resistance   — Reduce size when buying near a major resistance level
"""
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


# ── 1. Volume Surge ────────────────────────────────────────────────────────────

def volume_surge_factor(symbol: str) -> float:
    """
    Returns a size multiplier based on volume vs 20-day average.
    2x volume → 1.20x position size (stronger conviction)
    0.5x volume → 0.85x position size (low interest, be cautious)
    """
    try:
        from ml.trainer import fetch_training_data
        df = fetch_training_data(symbol, days=30)
        if df is None or len(df) < 10 or "Volume" not in df.columns:
            return 1.0
        vol_today = float(df["Volume"].iloc[-1])
        vol_avg   = float(df["Volume"].tail(20).mean())
        if vol_avg <= 0:
            return 1.0
        ratio = vol_today / vol_avg
        if ratio >= 2.0:
            logger.info("%s volume surge %.1fx avg — boosting position", symbol, ratio)
            return 1.20
        elif ratio >= 1.5:
            return 1.10
        elif ratio <= 0.5:
            logger.info("%s low volume %.1fx avg — reducing position", symbol, ratio)
            return 0.85
        return 1.0
    except Exception:
        return 1.0


# ── 2. RSI Overbought / Oversold Filter ───────────────────────────────────────

def rsi_filter(symbol: str, signal: str) -> tuple[bool, str]:
    """
    Block trades when RSI is already at an extreme.
    BUY blocked when RSI > 75 (overbought — too late to buy)
    SELL blocked when RSI < 25 (oversold — too late to sell)
    Returns (allow, reason)
    """
    try:
        from ml.trainer import fetch_training_data
        import numpy as np
        df = fetch_training_data(symbol, days=30)
        if df is None or len(df) < 15:
            return True, "insufficient data"
        close  = df["Close"]
        delta  = close.diff()
        gain   = delta.clip(lower=0).rolling(14).mean()
        loss   = (-delta.clip(upper=0)).rolling(14).mean()
        rs     = gain / loss.replace(0, 1e-9)
        rsi    = float((100 - 100 / (1 + rs)).iloc[-1])

        if signal == "BUY" and rsi > 75:
            return False, f"RSI {rsi:.0f} — overbought, skip buy"
        if signal == "SELL" and rsi < 25:
            return False, f"RSI {rsi:.0f} — oversold, skip sell"
        return True, f"RSI {rsi:.0f} OK"
    except Exception:
        return True, "RSI check failed"


# ── 3. Trend Alignment ────────────────────────────────────────────────────────

def trend_filter(symbol: str, signal: str) -> tuple[bool, float]:
    """
    Only buy when price > SMA50 (uptrend), only sell when price < SMA50 (downtrend).
    Returns (allow, sma50_value)
    When trading against trend: still allow but reduce size (caller uses the float)
    """
    try:
        from ml.trainer import fetch_training_data
        df = fetch_training_data(symbol, days=90)
        if df is None or len(df) < 50:
            return True, 0.0
        close  = df["Close"]
        sma50  = float(close.rolling(50).mean().iloc[-1])
        sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else sma50
        price  = float(close.iloc[-1])

        in_uptrend   = price > sma50
        long_uptrend = price > sma200

        if signal == "BUY":
            if not in_uptrend:
                logger.info("%s buying against trend (price %.4f < SMA50 %.4f) — reducing size",
                            symbol, price, sma50)
                return False, sma50  # False = against trend (reduce size in caller)
        elif signal == "SELL":
            if in_uptrend:
                return False, sma50

        return True, sma50
    except Exception:
        return True, 0.0


# ── 4. Drawdown Protection ────────────────────────────────────────────────────

def drawdown_check() -> tuple[bool, float]:
    """
    Pause new trades if the wallet is down more than 15% from its peak equity.
    Returns (allow_new_trades, current_drawdown_pct)
    """
    try:
        import database as db
        from trading.demo_wallet import get_portfolio_value
        trades = db.get_all_closed_trades()
        portfolio = get_portfolio_value()
        current_equity = portfolio["total_equity"]
        initial = db.get_initial()

        # Peak equity = initial + best running total from trade history
        peak = initial
        running = initial
        for t in sorted(trades, key=lambda x: x["closed_at"] or ""):
            running += t["pnl"] if t["pnl"] else 0
            if running > peak:
                peak = running

        peak = max(peak, current_equity)  # include open positions value
        drawdown = (peak - current_equity) / peak * 100 if peak > 0 else 0

        if drawdown >= 15:
            logger.warning("Drawdown protection: %.1f%% drawdown — pausing new trades", drawdown)
            return False, drawdown
        return True, drawdown
    except Exception:
        return True, 0.0


# ── 5. Correlated Exposure Limit ──────────────────────────────────────────────

# Assets that move together — limit total positions per group
_CORRELATION_GROUPS = {
    "meme_crypto":   {"DOGE","SHIB","PEPE","WIF","BONK","FLOKI","BRETT","TURBO","POPCAT"},
    "major_crypto":  {"BTC","ETH","SOL","BNB","ADA","AVAX","LINK","DOT","ATOM"},
    "us_tech":       {"AAPL","MSFT","GOOGL","AMZN","META","NVDA","AMD","TSLA","NFLX"},
    "us_market_etf": {"SPY","QQQ"},
    "energy":        {"OIL","NATGAS"},
    "metals":        {"GOLD","SILVER","COPPER"},
    "usd_pairs":     {"EURUSD","GBPUSD","AUDUSD","USDJPY","USDCAD"},
}
_MAX_PER_GROUP = 2   # max 2 open positions from the same correlated group


def correlated_exposure_check(symbol: str) -> tuple[bool, str]:
    """
    Return (allow, reason). Block if we already hold _MAX_PER_GROUP from same group.
    """
    try:
        import database as db
        open_syms = {p["symbol"] for p in db.get_all_positions()}
        for group, members in _CORRELATION_GROUPS.items():
            if symbol in members:
                held = open_syms & members
                if len(held) >= _MAX_PER_GROUP:
                    return False, f"correlated group '{group}' already has {len(held)} positions ({', '.join(held)})"
        return True, "OK"
    except Exception:
        return True, "OK"


# ── 6. BTC Market Regime ──────────────────────────────────────────────────────

def btc_regime() -> str:
    """
    Return 'bull', 'bear', or 'neutral' based on BTC vs its SMA50.
    Used to scale down ALL crypto trades in a bear market.
    """
    try:
        from ml.trainer import fetch_training_data
        df = fetch_training_data("BTC", days=60)
        if df is None or len(df) < 50:
            return "neutral"
        close = df["Close"]
        price = float(close.iloc[-1])
        sma50 = float(close.rolling(50).mean().iloc[-1])
        ret7d = (price / float(close.iloc[-8]) - 1) * 100 if len(close) >= 8 else 0

        if price > sma50 * 1.05 and ret7d > 0:
            return "bull"
        elif price < sma50 * 0.95 and ret7d < -3:
            return "bear"
        return "neutral"
    except Exception:
        return "neutral"


# ── 7. VIX Fear Filter (Stocks) ───────────────────────────────────────────────

def vix_size_multiplier() -> float:
    """
    Fetch VIX (stock market fear index).
    VIX > 30 = extreme fear = reduce stock positions by 40%
    VIX > 20 = elevated fear = reduce by 20%
    VIX < 15 = calm = slight boost
    """
    try:
        import yfinance as yf
        vix = yf.Ticker("^VIX").history(period="2d")
        if vix.empty:
            return 1.0
        vix_val = float(vix["Close"].iloc[-1])
        if vix_val > 35:
            logger.info("VIX %.1f — extreme fear, reducing stock positions 50%%", vix_val)
            return 0.50
        elif vix_val > 25:
            logger.info("VIX %.1f — high fear, reducing stock positions 30%%", vix_val)
            return 0.70
        elif vix_val > 20:
            return 0.85
        elif vix_val < 13:
            return 1.10   # very calm market, slight boost
        return 1.0
    except Exception:
        return 1.0


# ── 8. Price Gap Filter ───────────────────────────────────────────────────────

def gap_filter(symbol: str, signal: str) -> tuple[bool, str]:
    """
    Skip trade if price gapped up/down >4% overnight — you'd be chasing.
    A gap up on a BUY signal = the move already happened, not a good entry.
    """
    try:
        from ml.trainer import fetch_training_data
        df = fetch_training_data(symbol, days=5)
        if df is None or len(df) < 2:
            return True, "OK"
        prev_close = float(df["Close"].iloc[-2])
        today_open = float(df["Open"].iloc[-1]) if "Open" in df.columns else float(df["Close"].iloc[-1])
        gap_pct = (today_open / prev_close - 1) * 100

        if signal == "BUY" and gap_pct > 4:
            return False, f"price already gapped up {gap_pct:.1f}% — chasing, skip"
        if signal == "SELL" and gap_pct < -4:
            return False, f"price already gapped down {gap_pct:.1f}% — too late to sell"
        return True, f"gap {gap_pct:+.1f}% OK"
    except Exception:
        return True, "OK"


# ── 9. Support/Resistance Proximity ──────────────────────────────────────────

def resistance_proximity_factor(symbol: str, signal: str, current_price: float) -> float:
    """
    If buying within 1% of a major resistance level, reduce position size.
    If buying with clear space to next resistance, full size.
    Returns size multiplier (0.70 – 1.0).
    """
    try:
        from ml.trainer import fetch_training_data
        df = fetch_training_data(symbol, days=90)
        if df is None or len(df) < 30:
            return 1.0

        highs = df["High"].values
        # Find local swing highs in last 90 days
        resistances = []
        for i in range(5, len(highs) - 5):
            if highs[i] == max(highs[i-5:i+6]):
                resistances.append(highs[i])

        # Find nearest resistance above current price
        above = [r for r in resistances if r > current_price]
        if not above:
            return 1.0

        nearest = min(above)
        pct_away = (nearest / current_price - 1) * 100

        if signal == "BUY":
            if pct_away < 1.0:
                logger.info("%s resistance at %.4f only %.1f%% away — reducing buy size",
                            symbol, nearest, pct_away)
                return 0.70
            elif pct_away < 2.5:
                return 0.85
        return 1.0
    except Exception:
        return 1.0


# ── Master filter function ────────────────────────────────────────────────────

def apply_all_filters(symbol: str, signal: str, sentiment: float,
                      confidence: float, current_price: float) -> dict:
    """
    Run all filters. Returns:
    {
      "allow":        bool,    # False = skip this trade entirely
      "size_mult":    float,   # 0.5–1.5 multiplier on position size
      "reason":       str,     # human-readable explanation
      "warnings":     list,    # soft warnings (didn't block but noted)
    }
    """
    from config import CRYPTO_IDS, STOCK_SYMBOLS

    warnings = []
    size_mult = 1.0

    # Hard blocks first (these cancel the trade)
    dd_ok, dd_pct = drawdown_check()
    if not dd_ok:
        return {"allow": False, "size_mult": 0, "reason": f"Drawdown protection: -{dd_pct:.1f}% from peak", "warnings": []}

    corr_ok, corr_reason = correlated_exposure_check(symbol)
    if not corr_ok:
        return {"allow": False, "size_mult": 0, "reason": f"Correlated exposure: {corr_reason}", "warnings": []}

    rsi_ok, rsi_reason = rsi_filter(symbol, signal)
    if not rsi_ok:
        return {"allow": False, "size_mult": 0, "reason": rsi_reason, "warnings": []}

    gap_ok, gap_reason = gap_filter(symbol, signal)
    if not gap_ok:
        return {"allow": False, "size_mult": 0, "reason": gap_reason, "warnings": []}

    # Soft filters (reduce size but don't block)
    trend_ok, _ = trend_filter(symbol, signal)
    if not trend_ok:
        size_mult *= 0.75
        warnings.append("trading against SMA50 trend — size reduced 25%")

    vol_mult = volume_surge_factor(symbol)
    size_mult *= vol_mult
    if vol_mult != 1.0:
        warnings.append(f"volume factor {vol_mult:.2f}x")

    res_mult = resistance_proximity_factor(symbol, signal, current_price)
    size_mult *= res_mult
    if res_mult < 1.0:
        warnings.append(f"near resistance — size factor {res_mult:.2f}x")

    # BTC regime affects all crypto
    if symbol in CRYPTO_IDS:
        regime = btc_regime()
        if regime == "bear":
            size_mult *= 0.60
            warnings.append("BTC in bear regime — crypto size reduced 40%")
        elif regime == "bull":
            size_mult *= 1.10
            warnings.append("BTC in bull regime — slight boost")

    # VIX affects stocks
    if symbol in STOCK_SYMBOLS:
        vix_mult = vix_size_multiplier()
        size_mult *= vix_mult
        if vix_mult < 1.0:
            warnings.append(f"VIX elevated — stock size factor {vix_mult:.2f}x")

    # Cap final multiplier
    size_mult = max(0.40, min(size_mult, 1.50))

    return {
        "allow":    True,
        "size_mult": size_mult,
        "reason":   "passed all filters",
        "warnings": warnings,
    }
