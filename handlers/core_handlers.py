from telegram import Update
from telegram.ext import ContextTypes
from telegram.constants import ParseMode

HELP_TEXT = """
*Trading Bot Commands*

*Prices & Market*
/price <SYMBOL> — live price for any asset
/top [N] — top coins by market cap
/trending — trending on CoinGecko
/market — full market snapshot
/fear — Fear & Greed Index
/movers — top gainers & losers right now
/dominance — BTC/ETH/stablecoin market share
/gas — Ethereum gas fees

*Technical Analysis*
/analyze <SYMBOL> — RSI, MACD, Bollinger Bands, SMA
/levels <SYMBOL> — support & resistance price levels
/rsi <SYMBOL> — RSI only
/macd <SYMBOL> — MACD only

*ML Predictions* 🤖
/train <SYMBOL|group|all> — train ML model
/predict <SYMBOL> — live prediction + sentiment
/accuracy — all model accuracy scores
/sources <SYMBOL> — sentiment from all 12 sources
/importance <SYMBOL> — top ML features

*Market Tools*
/compare <SYM1> <SYM2> — side-by-side comparison
/calc <SYMBOL> <AMOUNT> <ENTRY> — profit/loss calculator
/watchlist — all symbols with live signals
/report — full daily market briefing

*Auto Demo Trading* 📈
/autotrade start [group] — paper trade (crypto/meme/stocks/forex/commodities/all)
/autotrade stop — stop auto trading
/wallet — balance & open positions
/trades [N] — trade history
/performance — win rate, Sharpe, drawdown
/close <SYMBOL> — manually close a position
/reset — reset wallet to $10,000

*Price Alerts*
/alert <SYMBOL> <above|below> <PRICE>
/alerts — active alerts
/cancelalert <ID>

*News*
/news [SYMBOL|crypto|stocks|forex]

*Quick Start*
1. `/train all` — train models on all 48 symbols
2. `/predict BTC` — get a live signal
3. `/watchlist` — see signals for everything
4. `/autotrade start all` — let it trade automatically
5. `/performance` — check results
"""


async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    # Save this chat so autotrade can resume automatically after restarts
    try:
        from database import save_last_chat_id
        save_last_chat_id(update.effective_chat.id)
    except Exception:
        pass
    await update.message.reply_text(
        f"Welcome to *Trading Bot*!\n\n"
        f"Real-time prices, technical analysis, price alerts, and portfolio tracking for crypto, stocks, forex, and commodities.\n\n"
        f"Type /help to see all commands.",
        parse_mode=ParseMode.MARKDOWN
    )


async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TEXT, parse_mode=ParseMode.MARKDOWN)


async def cmd_myid(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    await update.message.reply_text(f"Your Chat ID: `{chat_id}`", parse_mode=ParseMode.MARKDOWN)


async def error_handler(update: object, ctx: ContextTypes.DEFAULT_TYPE):
    import logging
    logging.getLogger(__name__).error("Error: %s", ctx.error, exc_info=ctx.error)
    if isinstance(update, Update) and update.message:
        await update.message.reply_text("An error occurred. Please try again.")
