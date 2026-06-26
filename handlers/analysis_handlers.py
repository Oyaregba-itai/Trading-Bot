from telegram import Update
from telegram.ext import ContextTypes
from telegram.constants import ParseMode

from data.crypto import get_crypto_price, get_crypto_history
from data.stocks import get_stock_price, get_stock_history
from utils.indicators import full_analysis
from utils.formatters import format_analysis, fmt_price
from config import CRYPTO_IDS


async def cmd_analyze(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Usage: /analyze BTC [days]  — full technical analysis"""
    if not ctx.args:
        await update.message.reply_text(
            "Usage: /analyze <SYMBOL> [days]\n"
            "Examples:\n"
            "  /analyze BTC\n"
            "  /analyze ETH 90\n"
            "  /analyze AAPL"
        )
        return

    symbol = ctx.args[0].upper()
    days = 90
    if len(ctx.args) > 1:
        try:
            days = max(30, min(int(ctx.args[1]), 365))
        except ValueError:
            pass

    await update.message.reply_text(f"Running analysis for {symbol} ({days}d history)...")

    prices, current_price = _get_prices(symbol, days)

    if len(prices) < 30:
        await update.message.reply_text(f"Not enough historical data for {symbol}. Try a different symbol.")
        return

    analysis = full_analysis(prices)
    msg = format_analysis(symbol, current_price, analysis)
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)


def _get_prices(symbol: str, days: int) -> tuple[list, float]:
    """Returns (price_list, current_price)."""
    if symbol in CRYPTO_IDS:
        hist = get_crypto_history(symbol, days)
        if hist:
            prices = [p[1] for p in hist]
            return prices, prices[-1]
        # Fallback: get current price only
        data = get_crypto_price(symbol)
        return [], data["price"] if data else 0.0

    # Try as stock
    df = get_stock_history(symbol, period="3mo" if days <= 90 else "1y")
    if df is not None and not df.empty:
        prices = df["Close"].tolist()
        return prices, prices[-1]

    # Try crypto search
    hist = get_crypto_history(symbol, days)
    if hist:
        prices = [p[1] for p in hist]
        return prices, prices[-1]

    return [], 0.0


async def cmd_rsi(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Quick RSI check: /rsi BTC"""
    if not ctx.args:
        await update.message.reply_text("Usage: /rsi <SYMBOL>")
        return

    symbol = ctx.args[0].upper()
    prices, current = _get_prices(symbol, 60)

    if len(prices) < 15:
        await update.message.reply_text(f"Not enough data for {symbol}.")
        return

    from utils.indicators import rsi, interpret_rsi
    val = rsi(prices, 14)
    if val is None:
        await update.message.reply_text("Could not calculate RSI.")
        return

    interp = interpret_rsi(val)
    bar = "█" * int(val / 10) + "░" * (10 - int(val / 10))
    msg = (
        f"*RSI(14) — {symbol}*\n\n"
        f"[{bar}] *{val:.1f}*\n"
        f"Signal: {interp}\n\n"
        f"_< 30: Oversold (Buy zone)_\n"
        f"_> 70: Overbought (Sell zone)_"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)


async def cmd_macd(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Quick MACD check: /macd BTC"""
    if not ctx.args:
        await update.message.reply_text("Usage: /macd <SYMBOL>")
        return

    symbol = ctx.args[0].upper()
    prices, current = _get_prices(symbol, 90)

    if len(prices) < 35:
        await update.message.reply_text(f"Not enough data for {symbol}.")
        return

    from utils.indicators import macd, interpret_macd
    result = macd(prices)
    if not result:
        await update.message.reply_text("Could not calculate MACD.")
        return

    interp = interpret_macd(result)
    cross = "🟢 Bullish Cross" if result["histogram"] > 0 else "🔴 Bearish Cross"
    msg = (
        f"*MACD(12,26,9) — {symbol}*\n\n"
        f"MACD Line:    `{result['macd']:+.6f}`\n"
        f"Signal Line:  `{result['signal']:+.6f}`\n"
        f"Histogram:    `{result['histogram']:+.6f}`\n\n"
        f"{cross}\n"
        f"Signal: {interp}"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
