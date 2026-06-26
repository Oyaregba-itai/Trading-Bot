import asyncio
import threading
from telegram import Update
from telegram.ext import ContextTypes
from telegram.constants import ParseMode

from ml.trainer import train_symbol, train_multiple, predict_symbol
from ml.model import get_feature_importance, TradingModel
from data.sentiment import aggregate_sentiment, get_sentiment_label, sentiment_emoji
from data.crypto import get_fear_greed
from database import get_all_models, get_model_meta


# Grouped by asset class — used for /train <group> and /autotrade start <group>
SYMBOL_GROUPS = {
    "CRYPTO":      ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "AVAX", "DOT",
                    "MATIC", "LINK", "LTC", "TRX", "TON", "NEAR", "SUI", "ATOM"],
    "MEME":        ["DOGE", "SHIB", "PEPE", "WIF", "BONK", "FLOKI",
                    "BRETT", "MOG", "TURBO", "BABYDOGE", "POPCAT"],
    "STOCKS":      ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN", "GOOGL",
                    "META", "SPY", "QQQ", "AMD", "NFLX"],
    "FOREX":       ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"],
    "COMMODITIES": ["GOLD", "OIL", "SILVER", "NATGAS", "COPPER"],
    # Stablecoins: tracked for sentiment only, not traded
    "STABLECOINS": ["USDT", "USDC", "DAI"],
}

# Default set for /train all — excludes stablecoins (price doesn't move)
ALL_SYMBOLS = (
    SYMBOL_GROUPS["CRYPTO"] +
    SYMBOL_GROUPS["MEME"] +
    SYMBOL_GROUPS["STOCKS"] +
    SYMBOL_GROUPS["FOREX"] +
    SYMBOL_GROUPS["COMMODITIES"]
)


async def cmd_train(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """
    /train BTC          — train model for one symbol
    /train all          — train all default symbols
    /train BTC ETH SOL  — train multiple
    """
    if not ctx.args:
        await update.message.reply_text(
            "*ML Training*\n\n"
            "Usage:\n"
            "  `/train BTC` — one symbol\n"
            "  `/train all` — every symbol (crypto + meme + stocks + forex + commodities)\n"
            "  `/train crypto` — BTC ETH SOL BNB XRP ADA AVAX DOT LINK LTC TON...\n"
            "  `/train meme` — DOGE SHIB PEPE WIF BONK FLOKI BRETT MOG TURBO...\n"
            "  `/train stablecoins` — USDT USDC DAI (sentiment only)\n"
            "  `/train stocks` — AAPL TSLA NVDA MSFT AMZN GOOGL META SPY QQQ...\n"
            "  `/train forex` — EURUSD GBPUSD USDJPY AUDUSD USDCAD\n"
            "  `/train commodities` — GOLD OIL SILVER NATGAS COPPER\n"
            "  `/train crypto meme` — train multiple groups\n\n"
            "_Supports everything: coins • meme coins • stablecoins • stocks • forex • commodities_",
            parse_mode=ParseMode.MARKDOWN
        )
        return

    args_upper = [a.upper() for a in ctx.args]
    if args_upper[0] == "ALL":
        symbols = ALL_SYMBOLS
    elif args_upper[0] in SYMBOL_GROUPS:
        # Single group keyword: /train meme, /train forex, etc.
        symbols = SYMBOL_GROUPS[args_upper[0]]
    elif any(a in SYMBOL_GROUPS for a in args_upper):
        # Multiple group keywords or mix: /train crypto meme
        symbols = []
        for a in args_upper:
            symbols.extend(SYMBOL_GROUPS.get(a, [a]))
    else:
        symbols = args_upper

    # Capture the running event loop BEFORE spawning the thread
    main_loop = asyncio.get_event_loop()

    await update.message.reply_text(
        f"Starting ML training for *{len(symbols)} symbols*:\n"
        f"{', '.join(symbols)}\n\n"
        f"Training 1 symbol at a time. Updates will appear as each finishes.",
        parse_mode=ParseMode.MARKDOWN
    )

    def _send(coro):
        """Thread-safe fire-and-forget message sender."""
        asyncio.run_coroutine_threadsafe(coro, main_loop)

    def run_training():
        for i, sym in enumerate(symbols, 1):
            # Send "starting symbol N/total" message
            _send(update.message.reply_text(
                f"*[{i}/{len(symbols)}] Training {sym}…*\n"
                f"Fetching historical data…",
                parse_mode=ParseMode.MARKDOWN
            ))

            last_step = [None]

            def progress(msg, s=sym):
                # Only send if the message is different (avoid spam)
                if msg != last_step[0]:
                    last_step[0] = msg
                    _send(update.message.reply_text(
                        f"*{s}* — {msg}",
                        parse_mode=ParseMode.MARKDOWN
                    ))

            result = train_symbol(sym, progress_callback=progress)
            _send(update.message.reply_text(
                _format_train_result(result),
                parse_mode=ParseMode.MARKDOWN
            ))

        _send(update.message.reply_text(
            f"*All {len(symbols)} models trained!*\n\n"
            "Now try:\n"
            "• `/predict BTC` — live signal\n"
            "• `/accuracy` — model scores\n"
            "• `/autotrade start` — start demo trading",
            parse_mode=ParseMode.MARKDOWN
        ))

    thread = threading.Thread(target=run_training, daemon=True)
    thread.start()


def _format_train_result(result: dict) -> str:
    if "error" in result:
        return f"*{result['symbol']}* — Training failed: {result['error']}"
    lines = [
        f"*{result['symbol']} — Training Complete*",
        "",
        f"Accuracy:      `{result['accuracy']*100:.1f}%`",
        f"Precision:     `{result['precision']*100:.1f}%`",
        f"Recall:        `{result['recall']*100:.1f}%`",
        f"F1 Score:      `{result['f1']*100:.1f}%`",
        f"Training data: `{result['n_samples']} samples, {result['n_features']} features`",
        "",
        "_Evaluated with 5-fold time-series cross-validation_",
    ]
    acc = result["accuracy"]
    if acc >= 0.60:
        lines.append("\n Model confidence: HIGH — good for auto-trading")
    elif acc >= 0.55:
        lines.append("\n Model confidence: MEDIUM — use with caution")
    else:
        lines.append("\n Model confidence: LOW — needs more data or longer history")
    return "\n".join(lines)


async def cmd_predict(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """
    /predict BTC  — ML prediction with live sentiment from all 12 sources.
    Runs in a background thread to avoid Telegram timeout on slow scraping.
    """
    if not ctx.args:
        await update.message.reply_text(
            "Usage: `/predict <SYMBOL>`\n\nExamples: `/predict BTC` `/predict GOLD` `/predict EURUSD` `/predict TSLA`",
            parse_mode=ParseMode.MARKDOWN
        )
        return

    symbol = ctx.args[0].upper()
    main_loop = asyncio.get_event_loop()

    # Send initial message then countdown-edit it every 3s
    status_msg = await update.message.reply_text(
        f"🔍 Analysing *{symbol}*… ⏳ 15s\n"
        f"_Scraping: Google News · Reddit · YouTube · Telegram · CryptoPanic · Finviz · Bloomberg · Reuters · CNBC · Yahoo Finance · Hacker News_",
        parse_mode=ParseMode.MARKDOWN
    )

    def _countdown():
        import time
        steps = [
            (3,  f"🔍 Analysing *{symbol}*… ⏳ 12s\n_Collecting news & social media…_"),
            (3,  f"🔍 Analysing *{symbol}*… ⏳ 9s\n_Running ML model on live data…_"),
            (3,  f"🔍 Analysing *{symbol}*… ⏳ 6s\n_Scoring sentiment from 12 sources…_"),
            (3,  f"🔍 Analysing *{symbol}*… ⏳ 3s\n_Almost done…_"),
        ]
        for delay, text in steps:
            time.sleep(delay)
            try:
                asyncio.run_coroutine_threadsafe(
                    status_msg.edit_text(text, parse_mode=ParseMode.MARKDOWN), main_loop
                )
            except Exception:
                pass

    def _run():
        import threading as _t
        _t.Thread(target=_countdown, daemon=True).start()

        try:
            sentiment_result = aggregate_sentiment(symbol, log_to_db=True)
            sentiment_score = sentiment_result.composite
        except Exception:
            sentiment_score = 0.0
            sentiment_result = None

        try:
            fg = get_fear_greed()
            fear_greed = fg["value"] if fg else 50
        except Exception:
            fg = None
            fear_greed = 50

        pred = predict_symbol(symbol, sentiment_score, fear_greed)

        def _send(text):
            # Try Markdown first; if Telegram rejects it, fall back to plain text
            future = asyncio.run_coroutine_threadsafe(
                update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN),
                main_loop
            )
            try:
                future.result(timeout=15)
            except Exception:
                # Strip all markdown symbols and retry as plain text
                plain = (text.replace("*", "").replace("_", "")
                             .replace("`", "").replace("[", "").replace("]", ""))
                asyncio.run_coroutine_threadsafe(
                    update.message.reply_text(plain),
                    main_loop
                ).result(timeout=15)

        # Clear the countdown message
        try:
            asyncio.run_coroutine_threadsafe(
                status_msg.edit_text(f"✅ Analysis complete for *{symbol}*", parse_mode=ParseMode.MARKDOWN),
                main_loop
            )
        except Exception:
            pass

        if "error" in pred:
            _send(f"❌ *{symbol}* — {pred['error']}\n\nTrain it first: `/train {symbol}`")
            return

        try:
            signal     = pred["signal"]
            confidence = pred["confidence"]
            price      = pred["price"]
            conf_bar   = "█" * int(confidence * 10) + "░" * (10 - int(confidence * 10))
            fg_val     = fear_greed

            sig_line = "🟢 *BUY* — model thinks price will go UP" if signal == "BUY" else "🔴 *SELL* — model thinks price will go DOWN"

            if confidence >= 0.65:
                conf_explain = "Strong signal — bot will trade this automatically"
            elif confidence >= 0.60:
                conf_explain = "Good signal — above 60% threshold, bot WILL trade"
            elif confidence >= 0.55:
                conf_explain = "Weak signal — below 60% threshold, bot will SKIP"
            else:
                conf_explain = "Low confidence — bot is watching but NOT trading yet"

            if fg_val <= 25:
                fg_explain = f"😱 Extreme Fear ({fg_val}/100) — everyone is scared and selling. Historically a great time to buy."
            elif fg_val <= 45:
                fg_explain = f"😨 Fear ({fg_val}/100) — market is nervous. Prices may be undervalued."
            elif fg_val <= 55:
                fg_explain = f"😐 Neutral ({fg_val}/100) — no strong emotion in the market right now."
            elif fg_val <= 75:
                fg_explain = f"😊 Greed ({fg_val}/100) — people are optimistic. Watch for overbuying."
            else:
                fg_explain = f"🤑 Extreme Greed ({fg_val}/100) — everyone is buying. Often a warning sign."

            if sentiment_score > 0.1:
                sent_explain = f"People online are talking positively about {symbol}."
            elif sentiment_score < -0.1:
                sent_explain = f"People online are talking negatively about {symbol}."
            else:
                sent_explain = f"Mixed/neutral news — no strong crowd opinion on {symbol}."

            bd = sentiment_result.breakdown if sentiment_result else {}
            total_signals = sum(v["count"] for v in bd.values())

            lines = [
                f"*ML Prediction — {symbol}*",
                f"Current Price: `${price:,.4f}`",
                "",
                f"*Signal:* {sig_line}",
                f"*Confidence:* [{conf_bar}] *{confidence*100:.1f}%*",
                f"_{conf_explain}_",
                "",
                f"*What the crowd is saying* ({total_signals} signals from 12 sources):",
                f"{sentiment_emoji(sentiment_score)} {get_sentiment_label(sentiment_score)} — {sent_explain}",
            ]

            for src, info in bd.items():
                dot = "🟢" if info["avg"] > 0.05 else ("🔴" if info["avg"] < -0.05 else "🟡")
                lines.append(f"  {dot} {src}: `{info['avg']:+.3f}` ({info['count']} items)")

            lines += [
                "",
                f"*Market Mood:* {fg_explain}",
                "",
                "_Paper-trade bot. No real money used._",
                "_Below 60% confidence = bot watches but does not trade._",
            ]

            _send("\n".join(lines))

        except Exception as e:
            _send(f"❌ Error building result for *{symbol}*: {e}\n\nPlease try again.")

    threading.Thread(target=_run, daemon=True).start()


async def cmd_accuracy(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Show accuracy metrics for all trained models."""
    models = get_all_models()
    if not models:
        await update.message.reply_text(
            "No trained models yet.\n\nUse `/train BTC` or `/train all` to start.",
            parse_mode=ParseMode.MARKDOWN
        )
        return

    lines = ["*Trained Models*\n"]
    for m in models:
        acc     = m["accuracy"]    or 0
        prec    = m["precision_s"] or 0
        rec     = m["recall_s"]    or 0
        f1      = m["f1_s"]        or 0
        n       = m["n_samples"]   or 0
        bar     = "█" * int(acc * 10) + "░" * (10 - int(acc * 10))

        # Overfit: suspiciously high accuracy on tiny dataset
        is_overfit    = acc > 0.75 and n < 200
        # Degenerate: high acc but model always predicts one class
        is_degenerate = (acc > 0.75 and rec < 0.10) or (prec < 0.05 and rec < 0.05)
        # Reliable: enough data AND balanced metrics
        is_reliable   = n >= 200 and acc >= 0.57 and prec >= 0.50 and rec >= 0.50

        if is_overfit or is_degenerate:
            quality = "⚠️"
            note = "  _overfit risk — too few samples_" if is_overfit else "  _degenerate_"
        elif is_reliable:
            quality = "🟢"
            note = "  _reliable_"
        elif acc >= 0.53:
            quality = "🟡"
            note = ""
        else:
            quality = "🔴"
            note = ""

        lines.append(
            f"{quality} *{m['symbol']}* [{bar}] `{acc*100:.1f}%`{note}\n"
            f"  P={prec*100:.0f}% R={rec*100:.0f}% F1={f1*100:.0f}%  "
            f"|  {n} samples  |  {(m['trained_at'] or '')[:10]}"
        )

    lines.append(
        "\n*Legend:* 🟢 Reliable  🟡 Marginal  🔴 Weak  ⚠️ Overfit/degenerate\n"
        "_⚠️ models are blocked from auto-trading_"
    )

    lines.append(
        "\n_Accuracy is from time-series cross-validation (realistic, no look-ahead)_"
    )
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)


async def cmd_sources(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """
    /sources BTC — show live sentiment breakdown from all data sources
    """
    if not ctx.args:
        await update.message.reply_text("Usage: `/sources <SYMBOL>`", parse_mode=ParseMode.MARKDOWN)
        return

    symbol = ctx.args[0].upper()
    main_loop = asyncio.get_event_loop()

    await update.message.reply_text(
        f"🔍 Scraping all 12 sources for *{symbol}*…\n_Result in ~15 seconds._",
        parse_mode=ParseMode.MARKDOWN
    )

    def _run():
        result = aggregate_sentiment(symbol, log_to_db=True)
        bd = result.breakdown

        source_names = {
            "google_news":  "Google News RSS",
            "reddit":       "Reddit (crypto/stocks subreddits)",
            "youtube":      "YouTube Channels",
            "telegram":     "Telegram Channels",
            "cryptopanic":  "CryptoPanic",
            "reddit_extra": "Reddit Extra (WSB / r/investing)",
            "finviz":       "Finviz (US stock news)",
            "web_rss":      "Bloomberg/Reuters/CNBC/CoinDesk/+more",
            "yahoo_finance":"Yahoo Finance RSS",
            "newsapi":      "NewsAPI",
            "hacker_news":  "Hacker News",
            "rss_feeds":    "RSS Feeds",
        }

        lines = [
            f"*Live Sentiment — {symbol}*",
            f"Composite: {sentiment_emoji(result.composite)} `{result.composite:+.3f}` "
            f"— {get_sentiment_label(result.composite)}",
            "",
        ]

        for src, info in bd.items():
            name = source_names.get(src, src)
            dot = "🟢" if info["avg"] > 0.05 else ("🔴" if info["avg"] < -0.05 else "🟡")
            lines.append(
                f"{dot} *{name}*\n"
                f"  `{info['avg']:+.3f}` | +{info['positive']} 🟡{info['neutral']} -{info['negative']}"
            )

    if not bd:
        if not bd:
            lines.append("Could not collect data from any source right now. Try again.")

        asyncio.run_coroutine_threadsafe(
            update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN),
            main_loop
        )

    threading.Thread(target=_run, daemon=True).start()


async def cmd_importance(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """
    /importance BTC — show which features matter most to the model
    """
    if not ctx.args:
        await update.message.reply_text("Usage: `/importance <SYMBOL>`", parse_mode=ParseMode.MARKDOWN)
        return

    symbol = ctx.args[0].upper()
    model = TradingModel.load_for(symbol)
    if not model:
        await update.message.reply_text(
            f"No trained model for {symbol}. Run `/train {symbol}` first.",
            parse_mode=ParseMode.MARKDOWN
        )
        return

    imp = get_feature_importance(model)
    lines = [f"*Feature Importance — {symbol}*\n"]
    for i, (feat, val) in enumerate(imp.items(), 1):
        bar = "█" * max(1, int(val * 100))
        lines.append(f"{i:2}. `{feat:<20}` {bar} {val*100:.2f}%")

    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)
