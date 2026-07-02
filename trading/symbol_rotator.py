"""
Automatic symbol rotation.
Weekly job that evaluates per-symbol performance over the last 30 days,
drops consistent losers, and adds high-volume alternatives from a pool.
"""
import logging
from database import get_strategy_performance

logger = logging.getLogger(__name__)

# Candidate pool to pull replacements from
_CANDIDATE_POOL = [
    # Crypto
    "BTC", "ETH", "SOL", "BNB", "AVAX", "DOGE", "ADA", "XRP", "LINK", "DOT",
    "LTC", "MATIC", "ATOM", "NEAR", "OP", "ARB",
    # Stocks
    "AAPL", "NVDA", "TSLA", "MSFT", "GOOGL", "META", "AMZN", "AMD",
    # Forex
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
    # Commodities
    "GOLD", "OIL",
]

_MIN_TRADES     = 3     # ignore symbols with too few trades
_DROP_WIN_RATE  = 35.0  # drop if win rate below this %
_DROP_MIN_PNL   = -50.0 # drop if total P&L below this $


def run_symbol_rotation(current_symbols: set, notify_cb=None, chat_ids=None) -> tuple[set, str]:
    """
    Evaluate current_symbols and return (new_symbol_set, summary_message).
    Drops underperformers and adds replacements from the pool.
    """
    stats = get_strategy_performance(days=30)
    perf = {}
    for row in stats:
        sym = row["symbol"]
        if sym not in perf:
            perf[sym] = {"trades": 0, "total_pnl": 0.0, "win_rate": 0.0}
        perf[sym]["trades"]    += row["trades"]
        perf[sym]["total_pnl"] += row["total_pnl"]
        # Weight win rate by trade count
        perf[sym]["win_rate"] = (
            perf[sym]["win_rate"] * (perf[sym]["trades"] - row["trades"])
            + row["win_rate"] * row["trades"]
        ) / perf[sym]["trades"]

    to_drop = []
    for sym in list(current_symbols):
        p = perf.get(sym)
        if not p or p["trades"] < _MIN_TRADES:
            continue  # not enough data to judge
        if p["win_rate"] < _DROP_WIN_RATE and p["total_pnl"] < _DROP_MIN_PNL:
            to_drop.append((sym, p["win_rate"], p["total_pnl"]))

    # Add replacements — prefer symbols not already watched and not recently dropped
    already_in = current_symbols | {s for s, _, _ in to_drop}
    candidates = [s for s in _CANDIDATE_POOL if s not in already_in]
    to_add = candidates[: len(to_drop)]

    new_symbols = (current_symbols - {s for s, _, _ in to_drop}) | set(to_add)

    lines = ["Symbol Rotation Report (30-day)\n"]
    if to_drop:
        lines.append("Dropped (underperformers):")
        for sym, wr, pnl in to_drop:
            lines.append(f"  - {sym}: {wr:.0f}% win, ${pnl:+.0f} P&L")
    else:
        lines.append("No symbols dropped — all performing adequately.")

    if to_add:
        lines.append("\nAdded:")
        for sym in to_add:
            lines.append(f"  + {sym}")

    lines.append(f"\nActive symbols: {len(new_symbols)}")

    summary = "\n".join(lines)

    if notify_cb and chat_ids:
        for cid in chat_ids:
            try:
                notify_cb(cid, summary)
            except Exception:
                pass

    logger.info("Symbol rotation: dropped %d, added %d", len(to_drop), len(to_add))
    return new_symbols, summary
