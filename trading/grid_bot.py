"""
Grid Trading Bot
Places virtual buy/sell orders at evenly spaced price levels.
Profits when price oscillates up and down within the range.

Usage: /grid BTC 60000 70000 10
  → Creates 10 grid levels between $60k-$70k
  → Buys when price drops to a level, sells at the next level up
"""
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def create_grid_bot(chat_id: int, symbol: str, low: float, high: float, levels: int):
    """Create a new grid bot. Returns (grid_id, summary_text)."""
    import database as db

    if low >= high:
        return None, "Low price must be less than high price."
    if levels < 2 or levels > 50:
        return None, "Levels must be between 2 and 50."

    grid_size = (high - low) / levels
    step = grid_size

    # Build price levels
    price_levels = [round(low + i * step, 8) for i in range(levels + 1)]

    # Initial state: nothing bought yet, track which levels have open buys
    state = {
        "levels": price_levels,
        "open_buys": {},   # level_index -> {"price": x, "qty": y}
        "realized_profit": 0.0,
    }

    grid_id = db.create_grid(
        chat_id, symbol, low, high, levels, grid_size,
        json.dumps(state)
    )

    lines = [
        f"Grid Bot Created — {symbol}",
        "",
        f"Range: ${low:,.4f} — ${high:,.4f}",
        f"Levels: {levels} ({levels+1} price points)",
        f"Grid size: ${grid_size:,.4f} per level",
        f"",
        f"How it works:",
        f"• Bot buys when price drops to each level",
        f"• Sells at the next level up (+${grid_size:,.4f})",
        f"• Profits from every up/down oscillation",
        f"",
        f"Use /gridstop {symbol} to stop.",
    ]
    return grid_id, "\n".join(lines)


def run_grid_cycle(app=None):
    """
    Called every 2 minutes by the position monitor.
    Checks all active grids and executes virtual trades.
    Returns list of (chat_id, message) notifications.
    """
    import database as db
    from trading.auto_trader import _get_price

    notifications = []
    grids = db.get_active_grids()

    for grid in grids:
        grid = dict(grid)
        symbol   = grid["symbol"]
        chat_id  = grid["chat_id"]
        grid_id  = grid["id"]
        cash     = db.get_cash()

        price = _get_price(symbol)
        if price is None:
            continue

        state = json.loads(grid["state"])
        levels = state["levels"]
        open_buys = state.get("open_buys", {})
        profit_delta = 0
        trades_delta = 0
        msgs = []

        for i in range(len(levels) - 1):
            buy_level  = levels[i]
            sell_level = levels[i + 1]
            key = str(i)

            # Price dropped to this buy level — place buy
            if price <= buy_level and key not in open_buys:
                invest = min(cash * 0.05, 200)  # max 5% cash or $200 per grid level
                if invest < 5:
                    continue
                qty = invest / price
                open_buys[key] = {"price": price, "qty": qty, "cost": invest}
                db.set_cash(cash - invest)
                cash -= invest
                msgs.append(
                    f"📉 *Grid Buy — {symbol}*\n"
                    f"Level {i+1}/{len(levels)-1} @ ${price:,.4f}\n"
                    f"Qty: {qty:.6f} | Cost: ${invest:.2f}\n"
                    f"Will sell at ${sell_level:,.4f} (+${sell_level-price:,.4f})"
                )
                trades_delta += 1

            # Price rose to sell level — close the buy below it
            elif price >= sell_level and key in open_buys:
                buy = open_buys.pop(key)
                revenue = buy["qty"] * price
                profit  = revenue - buy["cost"]
                db.set_cash(cash + revenue)
                cash += revenue
                state["realized_profit"] = state.get("realized_profit", 0) + profit
                profit_delta += profit
                msgs.append(
                    f"📈 *Grid Sell — {symbol}*\n"
                    f"Level {i+1}/{len(levels)-1} @ ${price:,.4f}\n"
                    f"Profit: +${profit:.4f} | Total: +${state['realized_profit']:.2f}"
                )
                trades_delta += 1

        state["open_buys"] = open_buys
        db.update_grid(grid_id, price, json.dumps(state), profit_delta, trades_delta)

        for m in msgs:
            notifications.append((chat_id, m))

    return notifications


def get_grid_status(chat_id: int) -> str:
    import database as db
    from trading.auto_trader import _get_price

    grids = db.get_active_grids(chat_id)
    if not grids:
        return "No active grid bots. Start one with `/grid SYMBOL LOW HIGH LEVELS`"

    lines = ["*Active Grid Bots*", ""]
    for g in grids:
        g = dict(g)
        state = json.loads(g["state"])
        open_count = len(state.get("open_buys", {}))
        price = _get_price(g["symbol"])
        in_range = "in range" if g["low"] <= (price or 0) <= g["high"] else "OUT OF RANGE"
        lines += [
            f"*{g['symbol']}* — {in_range}",
            f"Range: ${g['low']:,.4f} — ${g['high']:,.4f} ({g['levels']} levels)",
            f"Current price: ${price:,.4f}" if price else "Price unavailable",
            f"Open buys: {open_count} | Trades: {g['trades_done']}",
            f"Total profit: +${g['total_profit']:.4f}",
            "",
        ]
    return "\n".join(lines)
