from telegram import Update
from telegram.ext import ContextTypes
from telegram.constants import ParseMode

from data.news import get_trading_news, get_rss_news
from utils.formatters import format_news


async def cmd_news(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """
    /news           — general trading news
    /news BTC       — news for specific symbol
    /news stocks    — stock market news
    /news forex     — forex news
    """
    if ctx.args:
        query = ctx.args[0].upper()
        category_map = {
            "STOCKS": "stocks", "STOCK": "stocks",
            "FOREX": "forex", "FX": "forex",
            "CRYPTO": "crypto", "GENERAL": "general",
        }
        if query in category_map:
            articles = get_rss_news(category_map[query], 5)
            title = f"{query} News"
        else:
            articles = get_trading_news(query, 5)
            title = f"News: {query}"
    else:
        articles = get_rss_news("crypto", 5)
        title = "Latest Crypto & Trading News"

    if not articles:
        await update.message.reply_text("Could not fetch news right now. Try again later.")
        return

    msg = format_news(articles, title)
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN, disable_web_page_preview=True)
