"""
DCA (Dollar Cost Averaging) Bot
Automatically buys a fixed $ amount of an asset at regular intervals.
Reduces the impact of volatility by spreading purchases over time.

Usage: /dca BTC 50 daily     → buy $50 of BTC every day
       /dca ETH 100 weekly   → buy $100 of ETH every week
       /dca GOLD 200 monthly → buy $200 of GOLD every month
"""
import logging
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


def _next_run(interval: str) -> str:
    now = datetime.now(timezone.utc)
    if interval == "daily":
        nxt = now + timedelta(days=1)
    elif interval == "weekly":
        nxt = now + timedelta(weeks=1)
    elif interval == "monthly":
        nxt = now + timedelta(days=30)
    else:
        nxt = now + timedelta(days=1)
    return nxt.isoformat()


def create_dca_bot(chat_id: int, symbol: str, amount_usd: float, interval: str):
    """Create a new DCA schedule. Returns (dca_id, summary_text)."""
    import database as db
    from trading.auto_trader import _get_price

    interval = interval.lower()
    if interval not in ("daily", "weekly", "monthly"):
        return None, "Interval must be: daily, weekly, or monthly"
    if amount_usd < 1:
        return None, "Minimum amount is $1"

    price = _get_price(symbol)
    price_str = f"${price:,.4f}" if price else "price unavailable"

    next_run = _next_run(interval)
    dca_id = db.create_dca(chat_id, symbol, amount_usd, interval, next_run)

    interval_map = {"daily": "every day", "weekly": "every week", "monthly": "every month"}
    lines = [
        f"DCA Bot Started — {symbol}",
        "",
        f"Buying ${amount_usd:.2f} of {symbol} {interval_map[interval]}",
        f"Current price: {price_str}",
        f"First purchase: now (immediately)",
        f"Next purchase: {interval} from now",
        "",
        f"Over time your average entry price smooths out,",
        f"reducing the risk of buying at the wrong moment.",
        "",
        f"Use /dcastop {symbol} to stop.",
        f"Use /dcaview to see all active DCA bots.",
    ]

    # Execute first buy immediately
    msg = _execute_dca_buy(dca_id, symbol, amount_usd, price)
    if msg:
        lines += ["", "First buy executed:", msg]

    return dca_id, "\n".join(lines)


def _execute_dca_buy(dca_id: int, symbol: str, amount_usd: float, price: float | None) -> str | None:
    import database as db
    from trading.auto_trader import _get_price

    if price is None:
        price = _get_price(symbol)
    if price is None or price <= 0:
        return None

    cash = db.get_cash()
    invest = min(amount_usd, cash * 0.95)
    if invest < 1:
        return None

    qty = invest / price
    db.set_cash(cash - invest)

    # Recalculate average price
    row = db.get_active_dcas()
    this_dca = next((dict(r) for r in row if r["id"] == dca_id), None)
    if this_dca:
        old_qty    = this_dca["total_qty"]
        old_inv    = this_dca["total_invested"]
        new_qty    = old_qty + qty
        new_inv    = old_inv + invest
        new_avg    = new_inv / new_qty if new_qty > 0 else price
    else:
        new_qty = qty
        new_avg = price
        invest  = invest

    next_run = _next_run("daily")  # placeholder, overridden by caller
    db.update_dca(dca_id, next_run, qty, invest, new_avg)

    current_val = new_qty * price
    pnl = current_val - (this_dca["total_invested"] + invest if this_dca else invest)

    return (f"Bought {qty:.6f} {symbol} @ ${price:,.4f} — spent ${invest:.2f}\n"
            f"Avg entry: ${new_avg:,.4f} | Total invested: ${(this_dca['total_invested'] if this_dca else 0) + invest:.2f}")


def run_dca_cycle(app=None):
    """
    Called every 15 minutes. Checks if any DCA is due and executes it.
    Returns list of (chat_id, message) notifications.
    """
    import database as db
    from trading.auto_trader import _get_price

    notifications = []
    now = datetime.now(timezone.utc).isoformat()
    dcas = db.get_active_dcas()

    for dca in dcas:
        dca = dict(dca)
        if dca["next_run"] > now:
            continue

        symbol     = dca["symbol"]
        amount_usd = dca["amount_usd"]
        dca_id     = dca["id"]
        chat_id    = dca["chat_id"]
        interval   = dca["interval_type"]

        price = _get_price(symbol)
        if price is None:
            continue

        cash   = db.get_cash()
        invest = min(amount_usd, cash * 0.95)
        if invest < 1:
            continue

        qty = invest / price

        old_qty = dca["total_qty"]
        old_inv = dca["total_invested"]
        new_qty = old_qty + qty
        new_inv = old_inv + invest
        new_avg = new_inv / new_qty if new_qty > 0 else price

        db.set_cash(cash - invest)
        next_run = _next_run(interval)
        db.update_dca(dca_id, next_run, qty, invest, new_avg)

        current_val = new_qty * price
        unrealised  = current_val - new_inv
        pnl_pct     = (unrealised / new_inv * 100) if new_inv > 0 else 0

        msg = (
            f"💰 *DCA Buy — {symbol}*\n"
            f"Bought {qty:.6f} @ ${price:,.4f}\n"
            f"Spent: ${invest:.2f} | Run #{dca['runs_done']+1}\n"
            f"Avg entry: ${new_avg:,.4f}\n"
            f"Total invested: ${new_inv:.2f} | P&L: {pnl_pct:+.2f}%\n"
            f"Next buy: {interval}"
        )
        notifications.append((chat_id, msg))

    return notifications


def get_dca_status(chat_id: int) -> str:
    import database as db
    from trading.auto_trader import _get_price

    dcas = db.get_active_dcas(chat_id)
    if not dcas:
        return "No active DCA bots.\n\nStart one with:\n`/dca BTC 50 weekly`"

    lines = ["*Active DCA Bots*", ""]
    for d in dcas:
        d = dict(d)
        price = _get_price(d["symbol"])
        if price and d["total_qty"] > 0:
            current_val = d["total_qty"] * price
            pnl = current_val - d["total_invested"]
            pnl_pct = (pnl / d["total_invested"] * 100) if d["total_invested"] > 0 else 0
            pnl_str = f"${pnl:+.2f} ({pnl_pct:+.2f}%)"
        else:
            pnl_str = "n/a"

        lines += [
            f"*{d['symbol']}* — ${d['amount_usd']:.2f} {d['interval_type']}",
            f"Runs: {d['runs_done']} | Total invested: ${d['total_invested']:.2f}",
            f"Avg entry: ${d['avg_price']:,.4f} | Holdings: {d['total_qty']:.6f}",
            f"Unrealised P&L: {pnl_str}",
            f"Next buy: {d['next_run'][:16].replace('T',' ')} UTC",
            "",
        ]
    return "\n".join(lines)
