from telegram import Update
from telegram.ext import ContextTypes
from telegram.constants import ParseMode

from data.crypto import get_crypto_price, get_top_coins, get_trending_coins, get_fear_greed
from data.stocks import get_stock_price, get_commodity_price
from data.forex import get_forex_price
from utils.formatters import format_crypto, format_stock, format_forex, fmt_change, fmt_price, fmt_large
from config import CRYPTO_IDS, COMMODITY_SYMBOLS, FOREX_PAIRS


def _detect_asset(symbol: str):
    """Return (asset_type, normalized_symbol)."""
    sym = symbol.upper().replace("/", "")
    if sym in CRYPTO_IDS or len(sym) <= 5:
        # Try crypto first
        return "auto", sym
    if sym in COMMODITY_SYMBOLS:
        return "commodity", sym
    if len(sym) == 6 and sym[:3].isalpha() and sym[3:].isalpha():
        return "forex", sym
    return "stock", sym


async def cmd_price(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Usage: /price BTC or /price AAPL or /price EURUSD"""
    if not ctx.args:
        await update.message.reply_text("Usage: /price <SYMBOL>\nExamples: /price BTC  /price AAPL  /price EURUSD  /price GOLD")
        return

    symbol = ctx.args[0].upper().replace("/", "")
    await update.message.reply_text(f"Fetching price for {symbol}...")

    msg = _fetch_price_message(symbol)
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN, disable_web_page_preview=True)


def _fetch_price_message(symbol: str) -> str:
    sym = symbol.upper()

    # Commodity check
    if sym in COMMODITY_SYMBOLS:
        data = get_commodity_price(sym)
        if data:
            return format_stock(data)
        return f"Could not fetch price for {sym}."

    # Forex check (6-char currency pair)
    if len(sym) == 6 and sym[:3].isalpha() and sym[3:].isalpha():
        data = get_forex_price(sym)
        if data:
            return format_forex(data)

    # Crypto check
    if sym in CRYPTO_IDS:
        data = get_crypto_price(sym)
        if data:
            return format_crypto(data)

    # Try crypto by search
    data = get_crypto_price(sym)
    if data:
        return format_crypto(data)

    # Try stock
    data = get_stock_price(sym)
    if data:
        return format_stock(data)

    return f"Could not find price data for *{sym}*. Check the symbol and try again."


async def cmd_top(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Show top 10 coins by market cap."""
    limit = 10
    if ctx.args:
        try:
            limit = min(int(ctx.args[0]), 20)
        except ValueError:
            pass

    await update.message.reply_text(f"Fetching top {limit} coins...")
    coins = get_top_coins(limit)
    if not coins:
        await update.message.reply_text("Could not fetch top coins right now.")
        return

    lines = [f"*Top {limit} Coins by Market Cap*\n"]
    for i, c in enumerate(coins, 1):
        chg = c.get("price_change_percentage_24h", 0) or 0
        arrow = "▲" if chg >= 0 else "▼"
        price = fmt_price(c["current_price"])
        lines.append(f"{i:2}. *{c['symbol'].upper()}* {price}  {arrow}{abs(chg):.1f}%")

    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)


async def cmd_trending(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Show trending coins on CoinGecko."""
    coins = get_trending_coins()
    if not coins:
        await update.message.reply_text("Could not fetch trending coins.")
        return

    lines = ["*Trending on CoinGecko*\n"]
    for i, item in enumerate(coins[:7], 1):
        c = item["item"]
        lines.append(f"{i}. *{c['symbol'].upper()}* — {c['name']} (Rank #{c.get('market_cap_rank', 'N/A')})")

    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)


async def cmd_fear_greed(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Show crypto Fear & Greed index."""
    data = get_fear_greed()
    if not data:
        await update.message.reply_text("Could not fetch Fear & Greed Index.")
        return

    val = data["value"]
    cls = data["classification"]
    if val <= 25:
        emoji = "😱"
    elif val <= 45:
        emoji = "😨"
    elif val <= 55:
        emoji = "😐"
    elif val <= 75:
        emoji = "😊"
    else:
        emoji = "🤑"

    bar = "█" * (val // 10) + "░" * (10 - val // 10)
    msg = (
        f"*Crypto Fear & Greed Index* {emoji}\n\n"
        f"[{bar}] *{val}/100*\n"
        f"Classification: *{cls}*\n\n"
        f"_Extreme Fear (<25) = Buy opportunity_\n"
        f"_Extreme Greed (>75) = Potential sell signal_"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)


async def cmd_market(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Quick market overview."""
    await update.message.reply_text("Fetching market overview...")

    import asyncio
    from data.stocks import get_multiple_stocks
    from data.forex import get_multiple_forex
    from config import FOREX_PAIRS, STOCK_SYMBOLS

    # Fetch key assets
    btc = get_crypto_price("BTC")
    eth = get_crypto_price("ETH")
    spy = get_stock_price("SPY")
    fg = get_fear_greed()

    lines = ["*Market Overview*\n"]

    if btc:
        lines.append(f"BTC: {fmt_price(btc['price'])}  {fmt_change(btc['change_24h'])}")
    if eth:
        lines.append(f"ETH: {fmt_price(eth['price'])}  {fmt_change(eth['change_24h'])}")
    if spy:
        lines.append(f"S&P500 (SPY): {fmt_price(spy['price'])}  {fmt_change(spy['change_24h'])}")

    gold = get_commodity_price("GOLD")
    oil = get_commodity_price("OIL")
    if gold:
        lines.append(f"Gold: {fmt_price(gold['price'])}  {fmt_change(gold['change_24h'])}")
    if oil:
        lines.append(f"Oil (WTI): {fmt_price(oil['price'])}  {fmt_change(oil['change_24h'])}")

    eurusd = get_forex_price("EURUSD")
    if eurusd:
        lines.append(f"EUR/USD: {eurusd['price']:.4f}  {fmt_change(eurusd.get('change_24h'))}")

    if fg:
        lines.append(f"\nFear & Greed: *{fg['value']}* — {fg['classification']}")

    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)
