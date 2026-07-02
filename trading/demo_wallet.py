"""
Paper-trading wallet backed by SQLite.
Executes virtual buys and sells, tracks positions, cash, and trade history.

Smart features:
  - Volatility-based stop loss / take profit per asset class
  - Sentiment + confidence-based position sizing
  - Trailing stop loss (locks in profits as price rises)
  - Market hours gating (no forex/stock trades at 1 AM)
"""
from datetime import datetime, timezone
import logging
import database as db

logger = logging.getLogger(__name__)

MIN_CONFIDENCE  = 0.65    # Only trade if model confidence ≥ 65%
MIN_RR_RATIO    = 2.0     # Minimum take-profit / stop-loss ratio (2:1)
MIN_CASH_RESERVE = 500.0  # Always keep $500 in cash


# ── Volatility presets per asset class ────────────────────────────────────────
# Meme coins are wild — give them room to breathe, bigger reward
# Forex moves slowly — tight stop, modest target
# Stocks & commodities in between

_VOLATILITY = {
    "5m": {
        "meme":      {"sl": 0.005, "tp": 0.010, "size": 0.10},
        "crypto":    {"sl": 0.003, "tp": 0.006, "size": 0.15},
        "forex":     {"sl": 0.001, "tp": 0.002, "size": 0.20},
        "stock":     {"sl": 0.002, "tp": 0.004, "size": 0.15},
        "commodity": {"sl": 0.002, "tp": 0.004, "size": 0.15},
    },
    "1h": {
        "meme":      {"sl": 0.04,  "tp": 0.08,  "size": 0.15},
        "crypto":    {"sl": 0.015, "tp": 0.03,  "size": 0.20},
        "forex":     {"sl": 0.003, "tp": 0.006, "size": 0.20},
        "stock":     {"sl": 0.010, "tp": 0.020, "size": 0.20},
        "commodity": {"sl": 0.010, "tp": 0.020, "size": 0.20},
    },
    "1d": {
        "meme":      {"sl": 0.08,  "tp": 0.16,  "size": 0.15},
        "crypto":    {"sl": 0.04,  "tp": 0.08,  "size": 0.20},
        "forex":     {"sl": 0.010, "tp": 0.020, "size": 0.20},
        "stock":     {"sl": 0.030, "tp": 0.060, "size": 0.20},
        "commodity": {"sl": 0.025, "tp": 0.050, "size": 0.20},
    },
}

# Trailing stop tiers: when price rises X%, move stop to Y% above entry
# e.g. up 4% → stop moves to breakeven (0%), up 8% → stop at +3%
_TRAIL_TIERS = [
    (0.12, 0.08),   # up 12%+ → stop at +8% (near TP, lock in most gain)
    (0.08, 0.04),   # up 8%+  → stop at +4%
    (0.04, 0.00),   # up 4%+  → stop at breakeven (entry price)
]


def _asset_class(symbol: str) -> str:
    from config import MEME_COIN_SYMBOLS, CRYPTO_IDS, COMMODITY_SYMBOLS, STOCK_SYMBOLS
    if symbol in MEME_COIN_SYMBOLS:      return "meme"
    if symbol in COMMODITY_SYMBOLS:      return "commodity"
    if symbol in STOCK_SYMBOLS:          return "stock"
    if len(symbol) == 6 and symbol.isalpha(): return "forex"
    if symbol in CRYPTO_IDS:             return "crypto"
    return "stock"


def _get_adr_pct(symbol: str, timeframe: str = "1h", periods: int = 14) -> float | None:
    """
    Average Daily Range as a % of price over last N periods.
    Returns None if data unavailable (falls back to fixed presets).
    """
    try:
        from ml.trainer import fetch_training_data
        df = fetch_training_data(symbol, timeframe)
        if df is None or len(df) < periods + 1:
            return None
        adr      = (df["High"] - df["Low"]).tail(periods).mean()
        price    = float(df["Close"].iloc[-1])
        if price <= 0:
            return None
        return float(adr / price)
    except Exception:
        return None


def _trade_params(symbol: str, sentiment: float, confidence: float, equity: float,
                  timeframe: str = "1h") -> dict:
    """Return position size (cash), stop loss price multiplier, take profit multiplier."""
    cls    = _asset_class(symbol)
    tf_map = _VOLATILITY.get(timeframe, _VOLATILITY["1h"])
    preset = tf_map.get(cls, tf_map["stock"])

    # Try ADR-based dynamic SL/TP — more adaptive than fixed presets
    adr_pct = _get_adr_pct(symbol, timeframe)
    if adr_pct and adr_pct > 0:
        # SL = 0.5x ADR, TP = 1.5x ADR (scales with actual symbol volatility)
        tf_tp_mult = {"5m": 0.8, "1h": 1.5, "1d": 2.0}.get(timeframe, 1.5)
        dyn_sl = max(adr_pct * 0.5, preset["sl"] * 0.5)   # floor at half preset
        dyn_tp = max(adr_pct * tf_tp_mult, preset["tp"] * 0.5)
        # Cap to 2x preset to avoid huge stops on volatile days
        sl_pct = min(dyn_sl, preset["sl"] * 2.0)
        tp_pct = min(dyn_tp, preset["tp"] * 2.0)
    else:
        sl_pct = preset["sl"]
        tp_pct = preset["tp"]

    base_size = preset["size"]

    # Sentiment-based sizing: positive news = bigger bet, negative = smaller
    if sentiment > 0.2:
        sent_mult = 1.20
    elif sentiment > 0.1:
        sent_mult = 1.10
    elif sentiment < -0.2:
        sent_mult = 0.60   # strong negative news — trade cautiously
    elif sentiment < -0.1:
        sent_mult = 0.80
    else:
        sent_mult = 1.00

    # Confidence-based sizing: more confident = larger position
    if confidence >= 0.75:
        conf_mult = 1.20
    elif confidence >= 0.65:
        conf_mult = 1.10
    else:
        conf_mult = 1.00

    final_size = min(base_size * sent_mult * conf_mult, 0.30)  # cap at 30%

    return {
        "position_cash": equity * final_size,
        "sl_pct":        sl_pct,
        "tp_pct":        tp_pct,
        "asset_class":   cls,
    }


def _is_market_open(symbol: str) -> tuple[bool, str]:
    """Return (is_open, reason). Crypto = always open. Forex/stocks = business hours only."""
    from config import CRYPTO_IDS
    if symbol in CRYPTO_IDS:
        return True, "24/7"

    now   = datetime.now(timezone.utc)
    hour  = now.hour
    wday  = now.weekday()   # 0=Mon … 6=Sun

    # Weekends — forex & stocks closed
    if wday >= 5:
        return False, f"weekend (markets reopen Monday)"

    cls = _asset_class(symbol)

    if cls == "forex":
        # Sydney opens 21:00 UTC Sun, New York closes 21:00 UTC Fri
        # Best liquidity: London 07-16 UTC, NY 13-21 UTC
        if 7 <= hour <= 21:
            return True, "forex session open"
        return False, f"forex illiquid ({hour}:00 UTC — best hours 07-21 UTC)"

    if cls == "stock":
        # NYSE/NASDAQ: 13:30–20:00 UTC
        if 13 <= hour <= 20:
            return True, "stock market open"
        return False, f"stock market closed ({hour}:00 UTC — opens 13:30 UTC)"

    # Commodities trade nearly 24h on futures
    return True, "commodity futures open"


# ── Core wallet functions ──────────────────────────────────────────────────────

def get_portfolio_value(prices: dict | None = None) -> dict:
    cash      = db.get_cash()
    positions = db.get_all_positions()
    pos_value = 0.0
    pos_details = []

    for pos in positions:
        import math
        sym     = pos["symbol"]
        current = (prices or {}).get(sym, pos["entry_price"])
        if current is None or (isinstance(current, float) and math.isnan(current)):
            current = pos["entry_price"]
        value   = current * pos["quantity"]
        cost    = pos["cost"]
        pnl     = value - cost
        pnl_pct = (pnl / cost) * 100 if cost > 0 else 0
        pos_value += value
        pos_details.append({
            "symbol":        sym,
            "quantity":      pos["quantity"],
            "entry_price":   pos["entry_price"],
            "current_price": current,
            "cost":          cost,
            "value":         value,
            "pnl":           pnl,
            "pnl_pct":       pnl_pct,
            "stop_loss":     pos["stop_loss"],
            "take_profit":   pos["take_profit"],
            "confidence":    pos["confidence"],
            "opened_at":     pos["opened_at"],
        })

    return {
        "cash":            cash,
        "positions_value": pos_value,
        "total_equity":    cash + pos_value,
        "positions":       pos_details,
    }


def execute_buy(symbol: str, asset_type: str, current_price: float,
                confidence: float, signal: str = "BUY",
                sentiment: float = 0.0, timeframe: str = "1h") -> dict | None:
    """
    Execute a smart paper buy with all market filters applied.
    """
    if confidence < MIN_CONFIDENCE:
        return None

    # Market hours gate
    is_open, reason = _is_market_open(symbol)
    if not is_open:
        logger.info("Skipping %s — %s", symbol, reason)
        return None

    # No duplicate positions
    if db.get_position(symbol):
        return None

    # Run all market intelligence filters
    from trading.market_filters import apply_all_filters
    filters = apply_all_filters(symbol, signal, sentiment, confidence, current_price)
    if not filters["allow"]:
        logger.info("Skipping %s — %s", symbol, filters["reason"])
        return None
    if filters["warnings"]:
        logger.info("%s filter notes: %s", symbol, "; ".join(filters["warnings"]))

    cash      = db.get_cash()
    portfolio = get_portfolio_value()
    equity    = portfolio["total_equity"]

    params    = _trade_params(symbol, sentiment, confidence, equity, timeframe)
    available = cash - MIN_CASH_RESERVE
    if available < 10:
        return None

    # Enforce minimum 2:1 risk/reward — skip trades where SL is too close to TP
    rr_ratio = params["tp_pct"] / params["sl_pct"] if params["sl_pct"] > 0 else 0
    if rr_ratio < MIN_RR_RATIO:
        logger.info("Skipping %s — R/R %.1f:1 below minimum %.1f:1", symbol, rr_ratio, MIN_RR_RATIO)
        return None

    # Apply filter size multiplier on top of sentiment/confidence sizing
    adjusted_cash = params["position_cash"] * filters["size_mult"]
    trade_cash    = min(adjusted_cash, available)
    quantity      = trade_cash / current_price
    cost          = trade_cash
    stop_loss     = current_price * (1 - params["sl_pct"])
    take_profit   = current_price * (1 + params["tp_pct"])

    db.set_cash(cash - cost)
    db.open_position(symbol, asset_type, current_price, quantity, cost,
                     stop_loss, take_profit, confidence)

    logger.info(
        "BUY %s @ %.4f | size=%.0f (mult=%.2f) SL=%.4f TP=%.4f conf=%.0f%% sent=%+.2f",
        symbol, current_price, trade_cash, filters["size_mult"],
        stop_loss, take_profit, confidence * 100, sentiment
    )

    return {
        "action":      "BUY",
        "symbol":      symbol,
        "price":       current_price,
        "quantity":    quantity,
        "cost":        cost,
        "stop_loss":   stop_loss,
        "take_profit": take_profit,
        "confidence":  confidence,
        "sl_pct":      params["sl_pct"],
        "tp_pct":      params["tp_pct"],
        "asset_class": params["asset_class"],
        "size_mult":   filters["size_mult"],
        "warnings":    filters["warnings"],
        "timeframe":   timeframe,
    }


def execute_sell(symbol: str, current_price: float, reason: str = "SIGNAL",
                 timeframe: str = "1h") -> dict | None:
    """Close a position. reason: SIGNAL | STOP_LOSS | TAKE_PROFIT | MANUAL | TRAILING_STOP"""
    pos = db.get_position(symbol)
    if not pos:
        return None

    quantity    = pos["quantity"]
    entry_price = pos["entry_price"]
    cost        = pos["cost"]
    revenue     = current_price * quantity
    pnl         = revenue - cost
    pnl_pct     = (pnl / cost) * 100 if cost > 0 else 0
    result      = "WIN" if pnl > 0 else "LOSS"

    db.set_cash(db.get_cash() + revenue)
    db.record_trade(
        symbol, pos["asset_type"],
        entry_price, current_price, quantity, cost, revenue,
        pnl, pnl_pct, result, pos["confidence"], reason,
        pos["opened_at"], timeframe=timeframe,
    )
    db.close_position(symbol)

    return {
        "action":      "SELL",
        "symbol":      symbol,
        "entry_price": entry_price,
        "exit_price":  current_price,
        "quantity":    quantity,
        "pnl":         pnl,
        "pnl_pct":     pnl_pct,
        "result":      result,
        "reason":      reason,
    }


def check_stop_take(symbol: str, current_price: float) -> dict | None:
    """
    Check stop-loss / take-profit / trailing stop.
    Also updates the trailing stop when price rises past thresholds.
    """
    pos = db.get_position(symbol)
    if not pos:
        return None

    entry       = pos["entry_price"]
    stop_loss   = pos["stop_loss"]
    take_profit = pos["take_profit"]
    gain_pct    = (current_price / entry - 1)

    # ── Trailing Stop: move stop up as price rises ────────────────────────────
    new_stop = stop_loss
    for gain_threshold, lock_pct in _TRAIL_TIERS:
        if gain_pct >= gain_threshold:
            candidate = entry * (1 + lock_pct)
            if candidate > new_stop:
                new_stop = candidate
            break

    if new_stop > stop_loss:
        db.update_stop_loss(symbol, new_stop)
        logger.info(
            "Trailing stop updated %s: %.4f → %.4f (price %.4f, gain +%.1f%%)",
            symbol, stop_loss, new_stop, current_price, gain_pct * 100
        )
        stop_loss = new_stop

    # ── Check triggers ────────────────────────────────────────────────────────
    if current_price <= stop_loss:
        reason = "TRAILING_STOP" if new_stop > pos["stop_loss"] else "STOP_LOSS"
        return execute_sell(symbol, current_price, reason=reason)

    if take_profit and current_price >= take_profit:
        return execute_sell(symbol, current_price, reason="TAKE_PROFIT")

    return None


def smart_position_review(symbol: str, current_price: float,
                           sentiment_score: float, fear_greed: int) -> dict | None:
    """
    Re-evaluate an open position every 15 minutes.
    Exits early to lock in profit or cut losses before SL/TP triggers.
    Returns sell result dict if exited, None if holding.

    Exit early if:
      - In profit AND model flipped to SELL with high confidence
      - In profit AND price has been falling for 3+ candles (momentum reversal)
      - In profit AND sentiment turned strongly negative
      - In ANY loss AND model now says SELL with high confidence (cut fast)

    Hold / tighten if:
      - Model still says BUY with high confidence → raise trailing stop
      - In profit, no reversal signals → hold for full take profit
    """
    pos = db.get_position(symbol)
    if not pos:
        return None

    entry      = pos["entry_price"]
    profit_pct = (current_price / entry - 1) * 100
    pos_dict   = dict(pos)

    # ── Re-run ML prediction with current market data ─────────────────────────
    try:
        from ml.trainer import predict_symbol
        pred = predict_symbol(symbol, sentiment_score, fear_greed)
        if "error" in pred:
            return None
        current_signal = pred["signal"]
        current_conf   = pred["confidence"]
    except Exception:
        return None

    # ── Check price momentum: last 4 candles direction ────────────────────────
    momentum_falling = False
    try:
        from ml.trainer import fetch_training_data
        df = fetch_training_data(symbol, days=10)
        if df is not None and len(df) >= 4:
            recent = df["Close"].tail(4).values
            # Falling = at least 3 of last 4 closes are declining
            declines = sum(1 for i in range(1, len(recent)) if recent[i] < recent[i-1])
            momentum_falling = declines >= 3
    except Exception:
        pass

    # ── Decision logic ────────────────────────────────────────────────────────

    # SCENARIO 1: In profit + model flipped to SELL → exit now, keep the gain
    if profit_pct >= 0.3 and current_signal == "SELL" and current_conf >= 0.62:
        logger.info("%s early exit: in profit +%.2f%%, model flipped SELL %.0f%%",
                    symbol, profit_pct, current_conf * 100)
        return execute_sell(symbol, current_price, reason="EARLY_EXIT_REVERSAL")

    # SCENARIO 2: In profit + price falling 3 candles in a row + sentiment turned negative
    if profit_pct >= 0.5 and momentum_falling and sentiment_score < -0.05:
        logger.info("%s early exit: momentum reversing, sentiment %.3f, profit +%.2f%%",
                    symbol, sentiment_score, profit_pct)
        return execute_sell(symbol, current_price, reason="EARLY_EXIT_MOMENTUM")

    # SCENARIO 3: In ANY loss + model now says SELL with conviction → cut fast
    if profit_pct < -0.5 and current_signal == "SELL" and current_conf >= 0.65:
        logger.info("%s cut loss: model SELL %.0f%% at loss %.2f%%",
                    symbol, current_conf * 100, profit_pct)
        return execute_sell(symbol, current_price, reason="CUT_LOSS")

    # SCENARIO 4: In profit + model still strongly BUY → tighten trailing stop
    if profit_pct >= 1.0 and current_signal == "BUY" and current_conf >= 0.65:
        # Move stop to 50% of current gain (lock in half)
        locked_stop = entry * (1 + (profit_pct / 200))  # half the profit
        current_sl  = pos_dict["stop_loss"] or 0
        if locked_stop > current_sl:
            db.update_stop_loss(symbol, locked_stop)
            logger.info("%s tightened stop to %.4f (locking in %.1f%% of +%.1f%% gain)",
                        symbol, locked_stop, profit_pct / 2, profit_pct)

    return None  # Hold the position
