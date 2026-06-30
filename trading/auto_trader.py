"""
Auto-trading engine. Runs as an APScheduler job.
Each cycle:
  1. Check all open positions for stop-loss / take-profit
  2. For each watched symbol, get ML prediction + live sentiment
  3. Execute buy/sell if confidence threshold is met
  4. Log every action
"""
import asyncio
import logging
from datetime import datetime, timezone
from typing import Callable

logger = logging.getLogger(__name__)

# Symbols actively being auto-traded {chat_id: set of symbols}
_watched: dict[int, set] = {}
# Notification callback: (chat_id, message) -> None
_notify_cb: Callable | None = None

# Activity tracking
_last_cycle_time: datetime | None = None
_last_cycle_scanned: int = 0
_last_cycle_trades: int = 0
_last_cycle_skipped: int = 0
_total_cycles: int = 0

DEFAULT_SYMBOLS = [
    # Crypto
    "BTC", "ETH", "SOL", "BNB", "DOGE",
    # Forex
    "EURUSD", "GBPUSD",
    # Commodities
    "GOLD", "OIL",
    # Stocks
    "AAPL", "TSLA", "NVDA",
]


def start_watching(chat_id: int, symbols: list[str]):
    _watched[chat_id] = set(s.upper() for s in symbols)
    from database import save_autotrade_session
    save_autotrade_session(chat_id, _watched[chat_id])


def stop_watching(chat_id: int):
    _watched.pop(chat_id, None)
    from database import delete_autotrade_session
    delete_autotrade_session(chat_id)


def restore_sessions():
    """Load saved autotrade sessions from DB on bot startup.
    Falls back to DEFAULT_CHAT_ID env var so Railway restarts auto-resume trading."""
    import os
    from database import load_all_autotrade_sessions, get_last_chat_id, save_autotrade_session

    sessions = load_all_autotrade_sessions()

    if sessions:
        _watched.update(sessions)
        total = sum(len(v) for v in sessions.values())
        logger.info("Restored %d autotrade session(s) with %d symbols", len(sessions), total)
    else:
        # Try env var first (set on Railway so deploys always auto-resume)
        env_chat_id = os.environ.get("DEFAULT_CHAT_ID")
        chat_id = int(env_chat_id) if env_chat_id else get_last_chat_id()
        if chat_id:
            all_syms = set(DEFAULT_SYMBOLS)
            try:
                from handlers.ml_handlers import ALL_SYMBOLS
                all_syms = set(ALL_SYMBOLS)
            except Exception:
                pass
            _watched[chat_id] = all_syms
            save_autotrade_session(chat_id, all_syms)
            logger.info("Auto-started trading for chat %d with %d symbols", chat_id, len(all_syms))


def _ensure_default_session():
    """If _watched is empty, auto-start for the owner's chat ID."""
    import os
    if _watched:
        return
    # Try env var first, then fall back to hardcoded owner ID
    env_id = os.environ.get("DEFAULT_CHAT_ID", "7819653477")
    try:
        chat_id = int(env_id)
    except (ValueError, TypeError):
        chat_id = 7819653477
    all_syms = set(DEFAULT_SYMBOLS)
    try:
        from handlers.ml_handlers import ALL_SYMBOLS
        all_syms = set(ALL_SYMBOLS)
    except Exception:
        pass
    _watched[chat_id] = all_syms
    logger.info("Auto-started session for chat %d (%d symbols)", chat_id, len(all_syms))


def get_watched(chat_id: int) -> set:
    _ensure_default_session()
    return _watched.get(chat_id, set())


def is_watching(chat_id: int) -> bool:
    _ensure_default_session()
    return bool(_watched.get(chat_id))


def set_notify_callback(cb: Callable):
    global _notify_cb
    _notify_cb = cb


async def _notify(chat_id: int, message: str):
    if _notify_cb:
        try:
            await _notify_cb(chat_id, message)
        except Exception as e:
            logger.error("Notify error: %s", e)


def _get_price(symbol: str) -> float | None:
    try:
        from config import CRYPTO_IDS, COMMODITY_SYMBOLS
        if symbol in CRYPTO_IDS:
            from data.crypto import get_crypto_price
            d = get_crypto_price(symbol)
            return d["price"] if d else None
        elif symbol in COMMODITY_SYMBOLS:
            from data.stocks import get_commodity_price
            d = get_commodity_price(symbol)
            return d["price"] if d else None
        elif len(symbol) == 6:
            from data.forex import get_forex_price
            d = get_forex_price(symbol)
            return d["price"] if d else None
        else:
            from data.stocks import get_stock_price
            d = get_stock_price(symbol)
            return d["price"] if d else None
    except Exception:
        return None


def _asset_type(symbol: str) -> str:
    from config import CRYPTO_IDS, COMMODITY_SYMBOLS
    if symbol in CRYPTO_IDS:
        return "crypto"
    if symbol in COMMODITY_SYMBOLS:
        return "commodity"
    if len(symbol) == 6:
        return "forex"
    return "stock"


def get_activity_status() -> dict:
    """Returns current bot activity stats for /status command."""
    import database as db
    positions = db.get_all_positions() or []
    return {
        "last_cycle":    _last_cycle_time,
        "scanned":       _last_cycle_scanned,
        "trades":        _last_cycle_trades,
        "skipped":       _last_cycle_skipped,
        "total_cycles":  _total_cycles,
        "open_positions": len(positions),
        "watching":      sum(len(v) for v in _watched.values()),
    }


async def run_trading_cycle(app=None):
    """Called by APScheduler every N minutes."""
    global _last_cycle_time, _last_cycle_scanned, _last_cycle_trades, _last_cycle_skipped, _total_cycles
    from trading.demo_wallet import execute_buy, execute_sell, check_stop_take
    from ml.trainer import predict_symbol
    from data.sentiment import aggregate_sentiment
    from data.crypto import get_fear_greed
    from trading.smart_features import (daily_loss_limit_hit, check_stale_positions,
                                        timeframe_aligned, earnings_blackout,
                                        funding_rate_size_mult, social_volume_spike,
                                        reentry_cooldown_active)
    from utils.formatters import fmt_price

    _ensure_default_session()
    if not _watched:
        return

    _last_cycle_time = datetime.now(timezone.utc)
    _total_cycles += 1
    cycle_scanned = 0
    cycle_trades = 0
    cycle_skipped = 0

    fg_data = get_fear_greed()
    fear_greed = fg_data["value"] if fg_data else 50

    # ── Daily loss limit check ────────────────────────────────────────────────
    loss_hit, today_pnl = daily_loss_limit_hit()
    if loss_hit:
        logger.warning("Daily loss limit hit (%.2f) — pausing new trades until midnight", today_pnl)
        # Only check SL/TP on existing positions, no new entries
        all_syms: set[str] = set()
        for syms in _watched.values():
            all_syms.update(syms)
        for symbol in all_syms:
            price = _get_price(symbol)
            if price:
                result = check_stop_take(symbol, price)
                if result:
                    msg = _format_exit_msg(result, price)
                    for chat_id, syms in _watched.items():
                        if symbol in syms:
                            await _notify(chat_id, msg)
        return

    # ── Stale position cleanup ────────────────────────────────────────────────
    stale = check_stale_positions()
    for s in stale:
        result = execute_sell(s["symbol"], s["price"], reason="STALE_POSITION")
        if result:
            msg = (f"⏰ *Stale Position Closed: {s['symbol']}*\n\n"
                   f"{s['reason']}\n"
                   f"P&L: {result['pnl']:+.2f} ({result['pnl_pct']:+.2f}%)\n"
                   f"Cash freed up for better opportunities.")
            for chat_id, syms in _watched.items():
                if s["symbol"] in syms:
                    await _notify(chat_id, msg)

    # Collect all symbols being watched across all users
    all_symbols: set[str] = set()
    for syms in _watched.values():
        all_symbols.update(syms)

    for symbol in all_symbols:
        price = _get_price(symbol)
        if price is None:
            continue

        cycle_scanned += 1

        # 1. Check stop-loss / take-profit / trailing stop
        result = check_stop_take(symbol, price)
        if result:
            msg = _format_exit_msg(result, price)
            for chat_id, syms in _watched.items():
                if symbol in syms:
                    await _notify(chat_id, msg)
            cycle_trades += 1
            continue

        # 2. Get sentiment
        try:
            sentiment_result = aggregate_sentiment(symbol, log_to_db=True)
            sentiment_score = sentiment_result.composite
        except Exception:
            sentiment_score = 0.0

        # Block stablecoins
        from config import STABLECOIN_SYMBOLS
        if symbol in STABLECOIN_SYMBOLS:
            continue

        # 3. Collect signals from all available timeframes
        from database import get_model_meta, get_position
        TIMEFRAMES = ["5m", "1h", "1d"]
        tf_signals: dict[str, dict] = {}
        for tf in TIMEFRAMES:
            meta = get_model_meta(symbol, tf)
            if not meta:
                continue
            meta = dict(meta)
            n   = meta.get("n_samples", 0) or 0
            acc = meta.get("accuracy", 0) or 0
            rec = meta.get("recall_s", 0) or 0
            if acc > 0.82 or (acc > 0.72 and n < 300) or (rec < 0.15 and acc > 0.68):
                logger.info("Skipping %s %s — overfit (acc=%.1f%%, n=%d)", symbol, tf, acc*100, n)
                continue
            try:
                p = predict_symbol(symbol, sentiment_score, fear_greed, timeframe=tf)
                if "error" not in p:
                    tf_signals[tf] = p
            except Exception as e:
                logger.error("Prediction error %s %s: %s", symbol, tf, e)

        if not tf_signals:
            cycle_skipped += 1
            continue

        # Confluence: count agreements
        buy_tfs  = [tf for tf, p in tf_signals.items() if p["signal"] == "BUY"]
        sell_tfs = [tf for tf, p in tf_signals.items() if p["signal"] == "SELL"]

        n_avail = len(tf_signals)
        if n_avail >= 2:
            if len(buy_tfs) >= 2:
                signal = "BUY"
            elif len(sell_tfs) >= 2:
                signal = "SELL"
            else:
                cycle_skipped += 1
                continue  # Timeframes disagree — skip
        else:
            # Only 1 timeframe trained — use it directly
            signal = next(iter(tf_signals.values()))["signal"]

        # Pick best timeframe for this direction (highest confidence)
        agree_tfs = buy_tfs if signal == "BUY" else sell_tfs
        if not agree_tfs:
            agree_tfs = list(tf_signals.keys())
        best_tf   = max(agree_tfs, key=lambda tf: tf_signals[tf]["confidence"])
        pred       = tf_signals[best_tf]
        confidence = pred["confidence"]
        asset_type = _asset_type(symbol)

        # 3. Smart position review for EXISTING positions
        if get_position(symbol):
            from trading.demo_wallet import smart_position_review
            smart_result = smart_position_review(symbol, price, sentiment_score, fear_greed)
            if smart_result:
                msg = _format_exit_msg(smart_result, price)
                for chat_id, syms in _watched.items():
                    if symbol in syms:
                        await _notify(chat_id, msg)
                cycle_trades += 1
            continue

        # 4. Open new position — run extra smart checks first
        if signal == "BUY" and confidence >= 0.60:

            cooling, cool_reason = reentry_cooldown_active(symbol, best_tf)
            if cooling:
                logger.info("Skipping %s — %s", symbol, cool_reason)
                cycle_skipped += 1
                continue

            blacked_out, earn_reason = earnings_blackout(symbol)
            if blacked_out:
                logger.info("Skipping %s — %s", symbol, earn_reason)
                cycle_skipped += 1
                continue

            tf_ok, tf_reason = timeframe_aligned(symbol, signal)
            if not tf_ok:
                logger.info("Skipping %s — %s", symbol, tf_reason)
                cycle_skipped += 1
                continue

            spike, spike_msg = social_volume_spike(symbol)
            if spike:
                logger.info("%s social spike: %s", symbol, spike_msg)

            fund_mult = funding_rate_size_mult(symbol, signal)

            trade = execute_buy(symbol, asset_type, price, confidence, signal,
                                sentiment=sentiment_score * fund_mult, timeframe=best_tf)
            if not trade:
                cycle_skipped += 1
            if trade:
                # For crypto symbols, also place real order on Binance Testnet
                binance_note = ""
                if asset_type == "crypto":
                    try:
                        from trading.binance_broker import is_crypto_symbol, place_buy_order, is_available
                        if is_available() and is_crypto_symbol(symbol):
                            b_order = place_buy_order(
                                symbol,
                                trade["cost"],
                                trade["stop_loss"],
                                trade["take_profit"],
                            )
                            if b_order:
                                binance_note = "\n🔗 Order placed on Binance Testnet"
                    except Exception as _be:
                        logger.warning("Binance order skipped for %s: %s", symbol, _be)

                msg = _format_buy_msg(trade, pred)
                if spike:
                    msg += f"\n⚡ Social spike: {spike_msg}"
                if binance_note:
                    msg += binance_note
                for chat_id, syms in _watched.items():
                    if symbol in syms:
                        await _notify(chat_id, msg)
                cycle_trades += 1

        elif signal == "SELL":
            pos = get_position(symbol)
            if pos and confidence >= 0.60:
                result = execute_sell(symbol, price, reason="SIGNAL")
                if result:
                    msg = _format_exit_msg(result, price)
                    for chat_id, syms in _watched.items():
                        if symbol in syms:
                            await _notify(chat_id, msg)
                    cycle_trades += 1

    # ── Cycle summary ─────────────────────────────────────────────────────────
    _last_cycle_scanned = cycle_scanned
    _last_cycle_trades  = cycle_trades
    _last_cycle_skipped = cycle_skipped

    import database as db
    from utils.formatters import fmt_price
    open_positions = db.get_all_positions() or []
    open_count = len(open_positions)
    now_str = datetime.now(timezone.utc).strftime("%H:%M UTC")

    lines = [f"🤖 *Cycle #{_total_cycles} complete* — {now_str}"]
    lines.append(f"Scanned {cycle_scanned} symbols | Skipped: {cycle_skipped}")

    if cycle_trades > 0:
        lines.append(f"*{cycle_trades} new trade(s) this cycle*")
    else:
        lines.append("No new trades this cycle")

    # Show open positions with live P&L
    if open_positions:
        lines.append(f"\n*Open Positions ({open_count}):*")
        total_target = 0.0
        for pos in open_positions:
            sym    = pos["symbol"]
            entry  = pos["entry_price"]
            sl     = pos["stop_loss"] or 0
            tp     = pos["take_profit"] or 0
            cost   = pos["cost"] or 0
            price  = _get_price(sym) or entry
            pnl    = (price - entry) * pos["quantity"]
            pnl_pct = (price / entry - 1) * 100 if entry > 0 else 0
            tp_gain = (tp / entry - 1) * 100 if entry > 0 else 0
            tp_profit = (tp - entry) * pos["quantity"]
            total_target += tp_profit
            arrow = "▲" if pnl >= 0 else "▼"
            lines.append(
                f"  • *{sym}*: {arrow} {pnl:+.2f} ({pnl_pct:+.1f}%)  "
                f"| TP: {fmt_price(tp)} (+{tp_gain:.0f}%)"
            )
        lines.append(f"\n💰 *Total profit if all TPs hit: +${total_target:.2f}*")
    else:
        lines.append("No open positions.")

    summary = "\n".join(lines)
    for chat_id in _watched:
        await _notify(chat_id, summary)


def _est_days_to_tp(asset_class: str, tp_pct: float, timeframe: str = "1h") -> str:
    """Estimate time to reach take profit based on timeframe volatility."""
    moves = {
        "5m": {"meme": 0.05, "crypto": 0.02, "stock": 0.01, "commodity": 0.008, "forex": 0.003},
        "1h": {"meme": 0.35, "crypto": 0.12, "stock": 0.06, "commodity": 0.05,  "forex": 0.015},
        "1d": {"meme": 8.0,  "crypto": 3.0,  "stock": 1.5,  "commodity": 1.2,   "forex": 0.5},
    }
    per_period = moves.get(timeframe, moves["1h"]).get(asset_class, 0.06)
    periods = tp_pct / per_period
    if timeframe == "5m":
        mins = periods * 5
        return f"~{int(mins)} min" if mins < 60 else f"~{mins/60:.0f} hrs"
    elif timeframe == "1h":
        if periods < 4:   return f"~{max(1,int(periods))}-{int(periods)+2} hours"
        elif periods < 24: return f"~{int(periods)} hours"
        elif periods < 48: return "~1-2 days"
        else:              return f"~{periods/24:.0f} days"
    else:  # 1d
        if periods < 1:  return "hours to ~1 day"
        elif periods < 3: return f"~{periods:.0f}-{periods+1:.0f} days"
        else:             return f"~{periods:.0f} days"


def _format_buy_msg(trade: dict, pred: dict) -> str:
    from utils.formatters import fmt_price
    sl_pct   = trade.get("sl_pct", 0.05) * 100
    tp_pct   = trade.get("tp_pct", 0.12) * 100
    cls      = trade.get("asset_class", "").title()
    tf       = trade.get("timeframe", "1h")
    sent     = pred.get("sentiment", 0)
    mult     = trade.get("size_mult", 1.0)
    warnings = trade.get("warnings", [])
    cost     = trade.get("cost", 0)
    qty      = trade.get("quantity", 0)
    entry    = trade.get("price", 0)
    sl       = trade.get("stop_loss", 0)
    tp       = trade.get("take_profit", 0)

    # $ amounts
    tp_profit  = (tp - entry) * qty if tp and entry else cost * (tp_pct / 100)
    sl_risk    = (entry - sl) * qty if sl and entry else cost * (sl_pct / 100)
    rr_ratio   = tp_profit / sl_risk if sl_risk > 0 else 0
    est_time   = _est_days_to_tp(cls.lower(), tp_pct, tf)
    tf_label   = {"5m": "Scalp (5-min)", "1h": "Intraday (1-hr)", "1d": "Swing (Daily)"}.get(tf, tf)

    sent_note = ("Strong positive news" if sent > 0.2 else
                 "Mild positive news"   if sent > 0.1 else
                 "Negative news — reduced size" if sent < -0.1 else
                 "Neutral")

    lines = [
        f"📈 *New Trade Opened*",
        f"",
        f"*{trade['symbol']}* ({cls}) — _{tf_label}_",
        f"Entry Price:  {fmt_price(entry)}",
        f"Position Size: {fmt_price(cost)} ({qty:.6f} units)",
        f"",
        f"🎯 *Take Profit:* {fmt_price(tp)} (+{tp_pct:.1f}%) → *+${tp_profit:.2f} profit*",
        f"🛑 *Stop Loss:*   {fmt_price(sl)} (-{sl_pct:.1f}%) → *-${sl_risk:.2f} risk*",
        f"⚖️ Risk/Reward:  1 : {rr_ratio:.1f}",
        f"⏱ Est. close:   {est_time}",
        f"",
        f"🤖 ML Confidence: *{trade['confidence']*100:.1f}%*",
        f"📰 Sentiment: {sent:+.3f} — {sent_note}",
        f"🔒 Trailing stop: active (locks profit as price rises)",
    ]
    if warnings:
        lines.append(f"⚠️ Notes: {'; '.join(warnings)}")
    return "\n".join(lines)


def _format_exit_msg(result: dict, price: float) -> str:
    from utils.formatters import fmt_price
    emoji = "✅" if result["result"] == "WIN" else "❌"
    reason_map = {
        "STOP_LOSS":             "Stop Loss Hit",
        "TAKE_PROFIT":           "Take Profit Hit",
        "SIGNAL":                "ML Sell Signal",
        "MANUAL":                "Manual Close",
        "TRAILING_STOP":         "Trailing Stop Hit — profit was locked in",
        "EARLY_EXIT_REVERSAL":   "Smart Exit — model flipped SELL while in profit",
        "EARLY_EXIT_MOMENTUM":   "Smart Exit — price reversing, locked in gains early",
        "CUT_LOSS":              "Smart Exit — model says SELL, cutting loss before it grows",
    }
    reason = reason_map.get(result["reason"], result["reason"])
    return (
        f"{emoji} *Auto-Trade: Position Closed*\n\n"
        f"Symbol: *{result['symbol']}*\n"
        f"Reason: {reason}\n"
        f"Entry: {fmt_price(result['entry_price'])}  →  Exit: {fmt_price(result['exit_price'])}\n"
        f"P&L: {result['pnl']:+.2f} ({result['pnl_pct']:+.2f}%)\n"
        f"Result: *{result['result']}*"
    )
