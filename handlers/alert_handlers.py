from telegram import Update
from telegram.ext import ContextTypes
from telegram.constants import ParseMode

from utils.alerts import add_alert, get_user_alerts, remove_alert
from utils.formatters import fmt_price


async def cmd_alert(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """
    Usage: /alert BTC above 70000
           /alert ETH below 3000
           /alert AAPL above 200
    """
    if len(ctx.args) < 3:
        await update.message.reply_text(
            "*Price Alert Setup*\n\n"
            "Usage: `/alert <SYMBOL> <above|below> <PRICE>`\n\n"
            "Examples:\n"
            "  `/alert BTC above 70000`\n"
            "  `/alert ETH below 3000`\n"
            "  `/alert AAPL above 200`\n"
            "  `/alert EURUSD above 1.1`",
            parse_mode=ParseMode.MARKDOWN
        )
        return

    symbol = ctx.args[0].upper()
    condition = ctx.args[1].lower()
    if condition not in ("above", "below"):
        await update.message.reply_text("Condition must be `above` or `below`.", parse_mode=ParseMode.MARKDOWN)
        return

    try:
        target = float(ctx.args[2].replace(",", ""))
    except ValueError:
        await update.message.reply_text("Invalid price. Use a number like `70000` or `1.05`.", parse_mode=ParseMode.MARKDOWN)
        return

    # Detect asset type
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
    alert = add_alert(chat_id, symbol, asset_type, condition, target)

    await update.message.reply_text(
        f"Alert set!\n\n"
        f"ID: `{alert.alert_id}`\n"
        f"Symbol: *{symbol}*\n"
        f"Condition: price goes *{condition}* {fmt_price(target)}\n\n"
        f"Use /alerts to see all your alerts.",
        parse_mode=ParseMode.MARKDOWN
    )


async def cmd_alerts(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """List active alerts for this user."""
    chat_id = update.effective_chat.id
    alerts = get_user_alerts(chat_id)

    if not alerts:
        await update.message.reply_text(
            "You have no active alerts.\n\nUse /alert to set one.\nExample: `/alert BTC above 70000`",
            parse_mode=ParseMode.MARKDOWN
        )
        return

    lines = ["*Your Active Alerts*\n"]
    for a in alerts:
        lines.append(
            f"`{a.alert_id}` — *{a.symbol}* {a.condition} {fmt_price(a.target)}"
        )

    lines.append("\nTo cancel: `/cancelalert <ID>`")
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)


async def cmd_cancel_alert(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Usage: /cancelalert <ID>"""
    if not ctx.args:
        await update.message.reply_text("Usage: `/cancelalert <ALERT_ID>`\n\nGet IDs from /alerts", parse_mode=ParseMode.MARKDOWN)
        return

    alert_id = ctx.args[0].lower()
    chat_id = update.effective_chat.id

    if remove_alert(alert_id, chat_id):
        await update.message.reply_text(f"Alert `{alert_id}` cancelled.", parse_mode=ParseMode.MARKDOWN)
    else:
        await update.message.reply_text(f"Alert `{alert_id}` not found.", parse_mode=ParseMode.MARKDOWN)
