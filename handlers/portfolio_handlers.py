from telegram import Update
from telegram.ext import ContextTypes
from telegram.constants import ParseMode

from utils.portfolio import add_holding, remove_holding, get_portfolio, calculate_pnl
from utils.formatters import fmt_price, fmt_large
from handlers.price_handlers import _fetch_price_message


async def cmd_buy(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """
    Usage: /buy BTC 0.5 65000
           /buy AAPL 10 190
    """
    if len(ctx.args) < 3:
        await update.message.reply_text(
            "*Add to Portfolio*\n\n"
            "Usage: `/buy <SYMBOL> <QUANTITY> <BUY_PRICE>`\n\n"
            "Examples:\n"
            "  `/buy BTC 0.5 65000`\n"
            "  `/buy ETH 2 3200`\n"
            "  `/buy AAPL 10 190`",
            parse_mode=ParseMode.MARKDOWN
        )
        return

    symbol = ctx.args[0].upper()
    try:
        quantity = float(ctx.args[1])
        buy_price = float(ctx.args[2].replace(",", ""))
    except ValueError:
        await update.message.reply_text("Invalid quantity or price.", parse_mode=ParseMode.MARKDOWN)
        return

    from config import CRYPTO_IDS, COMMODITY_SYMBOLS
    if symbol in CRYPTO_IDS:
        asset_type = "crypto"
    elif symbol in COMMODITY_SYMBOLS:
        asset_type = "commodity"
    elif len(symbol) == 6:
        asset_type = "forex"
    else:
        asset_type = "stock"

    chat_id = update.effective_chat.id
    holding = add_holding(chat_id, symbol, asset_type, quantity, buy_price)
    cost = holding.buy_price * holding.quantity

    await update.message.reply_text(
        f"Added to portfolio!\n\n"
        f"*{symbol}*\n"
        f"Quantity: {holding.quantity}\n"
        f"Avg Buy Price: {fmt_price(holding.buy_price)}\n"
        f"Total Cost: {fmt_price(cost)}\n\n"
        f"Use /portfolio to see your full portfolio.",
        parse_mode=ParseMode.MARKDOWN
    )


async def cmd_sell(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Remove holding: /sell BTC"""
    if not ctx.args:
        await update.message.reply_text("Usage: `/sell <SYMBOL>`", parse_mode=ParseMode.MARKDOWN)
        return

    symbol = ctx.args[0].upper()
    chat_id = update.effective_chat.id

    if remove_holding(chat_id, symbol):
        await update.message.reply_text(f"*{symbol}* removed from your portfolio.", parse_mode=ParseMode.MARKDOWN)
    else:
        await update.message.reply_text(f"*{symbol}* not found in your portfolio.", parse_mode=ParseMode.MARKDOWN)


async def cmd_portfolio(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Show full portfolio with P&L."""
    chat_id = update.effective_chat.id
    holdings = get_portfolio(chat_id)

    if not holdings:
        await update.message.reply_text(
            "Your portfolio is empty.\n\nAdd assets with `/buy SYMBOL QUANTITY PRICE`",
            parse_mode=ParseMode.MARKDOWN
        )
        return

    await update.message.reply_text("Calculating portfolio P&L...")

    total_cost = 0
    total_value = 0
    lines = ["*Your Portfolio*\n"]

    for h in holdings:
        current_price = _get_current_price(h.symbol, h.asset_type)
        if current_price is None:
            lines.append(f"*{h.symbol}* — price unavailable")
            continue

        pnl = calculate_pnl(h, current_price)
        total_cost += pnl["cost"]
        total_value += pnl["value"]

        arrow = "▲" if pnl["pnl"] >= 0 else "▼"
        lines.append(
            f"*{h.symbol}*  {fmt_price(current_price)}\n"
            f"  Qty: {h.quantity}  |  Avg: {fmt_price(h.buy_price)}\n"
            f"  P&L: {arrow} {fmt_price(abs(pnl['pnl']))} ({pnl['pnl_pct']:+.2f}%)"
        )

    total_pnl = total_value - total_cost
    total_pct = ((total_value - total_cost) / total_cost * 100) if total_cost > 0 else 0
    arrow = "▲" if total_pnl >= 0 else "▼"

    lines.append(f"\n*Total Portfolio*")
    lines.append(f"  Cost: {fmt_price(total_cost)}")
    lines.append(f"  Value: {fmt_price(total_value)}")
    lines.append(f"  P&L: {arrow} {fmt_price(abs(total_pnl))} ({total_pct:+.2f}%)")

    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)


def _get_current_price(symbol: str, asset_type: str) -> float | None:
    try:
        if asset_type == "crypto":
            from data.crypto import get_crypto_price
            data = get_crypto_price(symbol)
            return data["price"] if data else None
        elif asset_type in ("stock", "commodity"):
            from data.stocks import get_stock_price, get_commodity_price
            if asset_type == "commodity":
                data = get_commodity_price(symbol)
            else:
                data = get_stock_price(symbol)
            return data["price"] if data else None
        elif asset_type == "forex":
            from data.forex import get_forex_price
            data = get_forex_price(symbol)
            return data["price"] if data else None
    except Exception:
        return None
