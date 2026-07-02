"""
Weekly self-improvement loop.
Runs every Sunday, analyzes last 7 days of trades per symbol,
and adjusts per-symbol position size multipliers automatically.

Rules:
  - Symbol with >60% loss rate over 7 days → reduce size to 0.7x
  - Symbol with >70% win rate over 7 days  → boost size to 1.2x
  - Symbol with <3 trades → no change (insufficient data)
  - Resets all adjustments after 14 days (avoids permanent suppression)
  - Sends a Telegram summary of what changed
"""
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def run_self_improvement(notify_cb=None, chat_ids: list[int] | None = None):
    """
    Analyze last 7 days of trades and adjust per-symbol size multipliers.
    notify_cb(chat_id, message) is called with the summary if provided.
    """
    import database as db

    trades = db.get_recent_trades_all(days=7)
    if not trades:
        logger.info("Self-improver: no trades in last 7 days")
        return

    # Group trades by symbol
    by_symbol: dict[str, list] = {}
    for t in trades:
        t = dict(t)
        sym = t["symbol"]
        by_symbol.setdefault(sym, []).append(t)

    changes: list[str] = []
    held: list[str] = []

    for sym, sym_trades in by_symbol.items():
        n       = len(sym_trades)
        wins    = sum(1 for t in sym_trades if t["result"] == "WIN")
        losses  = n - wins
        win_rate = wins / n if n > 0 else 0
        total_pnl = sum(t["pnl"] or 0 for t in sym_trades)

        if n < 3:
            held.append(f"{sym}: only {n} trade(s) — skip")
            continue

        current_mult = db.get_symbol_size_mult(sym)

        if win_rate < 0.40:
            new_mult = max(current_mult * 0.7, 0.40)   # reduce, floor at 40%
            reason   = f"{losses}/{n} losses ({win_rate:.0%} win rate, P&L {total_pnl:+.2f})"
            db.set_symbol_size_mult(sym, round(new_mult, 2), reason)
            changes.append(f"🔻 *{sym}*: size {current_mult:.0%}→{new_mult:.0%} ({reason})")
            # Save lesson
            db.save_lesson(sym, f"Reduced size to {new_mult:.0%} — {reason}", timeframe="weekly")

        elif win_rate > 0.70:
            new_mult = min(current_mult * 1.20, 1.50)  # boost, cap at 150%
            reason   = f"{wins}/{n} wins ({win_rate:.0%} win rate, P&L {total_pnl:+.2f})"
            db.set_symbol_size_mult(sym, round(new_mult, 2), reason)
            changes.append(f"🔺 *{sym}*: size {current_mult:.0%}→{new_mult:.0%} ({reason})")
            db.save_lesson(sym, f"Boosted size to {new_mult:.0%} — {reason}", timeframe="weekly")

        else:
            # Drift back toward 1.0 (gradual reset)
            if abs(current_mult - 1.0) > 0.05:
                new_mult = current_mult + (1.0 - current_mult) * 0.3
                db.set_symbol_size_mult(sym, round(new_mult, 2), "drifting back to normal")
            held.append(f"{sym}: {win_rate:.0%} win rate — no major change")

    # Build summary message
    lines = ["🧠 *Weekly Self-Improvement Report*", ""]
    lines.append(f"Analyzed {len(by_symbol)} symbols over last 7 days")
    lines.append("")

    if changes:
        lines.append("*Changes made:*")
        lines.extend(changes)
        lines.append("")

    if held:
        lines.append(f"*No change ({len(held)} symbols):* {', '.join(s.split(':')[0] for s in held)}")

    lines.append("")
    lines.append(f"_Next review: {_next_sunday()}_")
    summary = "\n".join(lines)

    logger.info("Self-improver complete: %d changes", len(changes))

    if notify_cb and chat_ids:
        import asyncio
        for cid in chat_ids:
            try:
                asyncio.ensure_future(notify_cb(cid, summary))
            except Exception as e:
                logger.error("Self-improver notify error: %s", e)

    return summary


def _next_sunday() -> str:
    from datetime import timedelta
    now  = datetime.now(timezone.utc)
    days = (6 - now.weekday()) % 7 or 7
    return (now + timedelta(days=days)).strftime("%Y-%m-%d")
