"""
Extra feature handlers:
  /movers      — top gainers & losers right now
  /levels      — support & resistance levels
  /calc        — profit/loss calculator
  /compare     — side-by-side asset comparison
  /dominance   — crypto market dominance
  /gas         — Ethereum gas fees
  /watchlist   — all symbols with live price + signal
  /report      — full market briefing (also auto-sent daily)
"""
import asyncio
import threading
from telegram import Update
from telegram.ext import ContextTypes
from telegram.constants import ParseMode


# ── /movers ───────────────────────────────────────────────────────────────────

async def cmd_movers(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Top gainers and losers across crypto, stocks, forex, commodities."""
    await update.message.reply_text("📊 Fetching top movers across all markets…", parse_mode=ParseMode.MARKDOWN)
    main_loop = asyncio.get_event_loop()

    def _run():
        import yfinance as yf
        from config import CRYPTO_IDS, COMMODITY_SYMBOLS

        symbols = {
            # Crypto
            "BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD",
            "BNB": "BNB-USD", "XRP": "XRP-USD", "ADA": "ADA-USD",
            "DOGE": "DOGE-USD", "AVAX": "AVAX-USD", "LINK": "LINK-USD",
            "PEPE": "PEPE-USD", "SHIB": "SHIB-USD", "WIF": "WIF-USD",
            # Stocks
            "AAPL": "AAPL", "TSLA": "TSLA", "NVDA": "NVDA",
            "MSFT": "MSFT", "AMZN": "AMZN", "META": "META",
            # Forex
            "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X",
            # Commodities
            "GOLD": "GC=F", "OIL": "CL=F", "SILVER": "SI=F",
        }

        results = []
        try:
            tickers = yf.download(
                list(symbols.values()), period="2d", interval="1d",
                group_by="ticker", auto_adjust=True, progress=False, threads=True
            )
            for sym, ticker in symbols.items():
                try:
                    if len(symbols) > 1:
                        closes = tickers[ticker]["Close"].dropna()
                    else:
                        closes = tickers["Close"].dropna()
                    if len(closes) >= 2:
                        pct = (closes.iloc[-1] / closes.iloc[-2] - 1) * 100
                        results.append((sym, float(closes.iloc[-1]), float(pct)))
                except Exception:
                    pass
        except Exception as e:
            asyncio.run_coroutine_threadsafe(
                update.message.reply_text(f"Error fetching movers: {e}"), main_loop
            )
            return

        if not results:
            asyncio.run_coroutine_threadsafe(
                update.message.reply_text("Could not fetch price data right now. Try again."), main_loop
            )
            return

        results.sort(key=lambda x: x[2], reverse=True)
        gainers = results[:5]
        losers  = results[-5:][::-1]

        lines = ["📊 *Top Market Movers (24h)*\n"]
        lines.append("🟢 *Top Gainers*")
        for sym, price, pct in gainers:
            lines.append(f"  {sym}: +{pct:.2f}%  (${price:,.4f})")

        lines.append("\n🔴 *Top Losers*")
        for sym, price, pct in losers:
            lines.append(f"  {sym}: {pct:.2f}%  (${price:,.4f})")

        asyncio.run_coroutine_threadsafe(
            update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN), main_loop
        )

    threading.Thread(target=_run, daemon=True).start()


# ── /levels ───────────────────────────────────────────────────────────────────

async def cmd_levels(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Key support and resistance levels based on recent price history."""
    if not ctx.args:
        await update.message.reply_text("Usage: `/levels BTC`\n\nExamples: `/levels GOLD` `/levels TSLA`", parse_mode=ParseMode.MARKDOWN)
        return

    symbol = ctx.args[0].upper()
    main_loop = asyncio.get_event_loop()
    await update.message.reply_text(f"📐 Calculating support & resistance for *{symbol}*…", parse_mode=ParseMode.MARKDOWN)

    def _run():
        import numpy as np
        from ml.trainer import fetch_training_data

        df = fetch_training_data(symbol, days=180)
        if df is None or len(df) < 30:
            asyncio.run_coroutine_threadsafe(
                update.message.reply_text(f"Not enough data for {symbol}."), main_loop
            )
            return

        close  = df["Close"]
        high   = df["High"]
        low    = df["Low"]
        price  = float(close.iloc[-1])

        # Recent highs/lows as resistance/support
        recent = df.tail(90)
        r_high = float(recent["High"].max())
        r_low  = float(recent["Low"].min())

        # 20/50 day moving averages as dynamic S/R
        sma20 = float(close.rolling(20).mean().iloc[-1])
        sma50 = float(close.rolling(50).mean().iloc[-1])
        sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None

        # Swing highs/lows (local peaks over 5-candle windows)
        swing_highs = []
        swing_lows  = []
        h = high.values
        l = low.values
        for i in range(5, len(h) - 5):
            if h[i] == max(h[i-5:i+6]):
                swing_highs.append(h[i])
            if l[i] == min(l[i-5:i+6]):
                swing_lows.append(l[i])

        # Key resistances: swing highs above current price
        resistances = sorted(set([x for x in swing_highs if x > price]))[:3]
        # Key supports: swing lows below current price
        supports    = sorted(set([x for x in swing_lows  if x < price]), reverse=True)[:3]

        # 52-week high/low
        w52_high = float(df["High"].tail(252).max()) if len(df) >= 252 else r_high
        w52_low  = float(df["Low"].tail(252).min())  if len(df) >= 252 else r_low

        def fmt(v):
            return f"${v:,.4f}" if v < 100 else f"${v:,.2f}"

        lines = [f"📐 *Support & Resistance — {symbol}*", f"Current Price: {fmt(price)}", ""]

        lines.append("🔴 *Resistance Levels* (price needs to break through these to go higher)")
        if resistances:
            for r in resistances:
                pct = (r / price - 1) * 100
                lines.append(f"  {fmt(r)}  (+{pct:.1f}% from now)")
        else:
            lines.append(f"  {fmt(r_high)}  (90-day high)")

        lines.append("\n🟢 *Support Levels* (price tends to bounce up from these)")
        if supports:
            for s in supports:
                pct = (s / price - 1) * 100
                lines.append(f"  {fmt(s)}  ({pct:.1f}% from now)")
        else:
            lines.append(f"  {fmt(r_low)}  (90-day low)")

        lines.append("\n📈 *Moving Average Levels*")
        lines.append(f"  SMA20:  {fmt(sma20)}  ({'above' if price > sma20 else 'below'} — {'bullish' if price > sma20 else 'bearish'})")
        lines.append(f"  SMA50:  {fmt(sma50)}  ({'above' if price > sma50 else 'below'} — {'bullish' if price > sma50 else 'bearish'})")
        if sma200:
            lines.append(f"  SMA200: {fmt(sma200)}  ({'above — long-term bullish' if price > sma200 else 'below — long-term bearish'})")

        lines.append(f"\n📅 *52-Week Range*")
        lines.append(f"  Low: {fmt(w52_low)}  |  High: {fmt(w52_high)}")
        pos_in_range = (price - w52_low) / (w52_high - w52_low) * 100
        lines.append(f"  Current position: {pos_in_range:.0f}% of yearly range")

        asyncio.run_coroutine_threadsafe(
            update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN), main_loop
        )

    threading.Thread(target=_run, daemon=True).start()


# ── /calc ─────────────────────────────────────────────────────────────────────

async def cmd_calc(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """
    /calc BTC 0.1 59000        — P&L if you bought 0.1 BTC at $59,000
    /calc GOLD 2 1900          — P&L if you bought 2 oz gold at $1,900
    /calc TSLA 10 200          — P&L if you bought 10 TSLA shares at $200
    """
    if len(ctx.args) < 3:
        await update.message.reply_text(
            "*Profit/Loss Calculator*\n\n"
            "Usage: `/calc SYMBOL AMOUNT ENTRY_PRICE`\n\n"
            "Examples:\n"
            "  `/calc BTC 0.1 59000` — bought 0.1 BTC at $59,000\n"
            "  `/calc GOLD 2 1900` — bought 2 oz Gold at $1,900\n"
            "  `/calc TSLA 10 200` — bought 10 TSLA shares at $200",
            parse_mode=ParseMode.MARKDOWN
        )
        return

    symbol     = ctx.args[0].upper()
    try:
        amount     = float(ctx.args[1])
        entry      = float(ctx.args[2])
    except ValueError:
        await update.message.reply_text("Amount and price must be numbers. Example: `/calc BTC 0.1 59000`", parse_mode=ParseMode.MARKDOWN)
        return

    main_loop = asyncio.get_event_loop()

    def _run():
        from ml.trainer import fetch_training_data
        df = fetch_training_data(symbol, days=5)
        if df is None or df.empty:
            asyncio.run_coroutine_threadsafe(
                update.message.reply_text(f"Could not fetch current price for {symbol}."), main_loop
            )
            return

        current  = float(df["Close"].iloc[-1])
        invested = amount * entry
        value    = amount * current
        pnl      = value - invested
        pnl_pct  = (pnl / invested) * 100
        sl_price = entry * 0.95
        tp_price = entry * 1.12
        sl_loss  = (sl_price - entry) * amount
        tp_gain  = (tp_price - entry) * amount

        emoji = "🟢" if pnl >= 0 else "🔴"

        def fmt(v):
            return f"${v:,.4f}" if v < 100 else f"${v:,.2f}"

        lines = [
            f"*P&L Calculator — {symbol}*",
            "",
            f"You bought: {amount} {symbol} at {fmt(entry)}",
            f"Total invested: ${invested:,.2f}",
            "",
            f"*Current price:* {fmt(current)}",
            f"*Current value:* ${value:,.2f}",
            f"{emoji} *Profit/Loss: {'+' if pnl>=0 else ''}{pnl:.2f} ({pnl_pct:+.2f}%)*",
            "",
            f"*If bot's stop-loss hits* (-5% at {fmt(sl_price)}):",
            f"  Loss would be: ${sl_loss:.2f}",
            f"*If bot's take-profit hits* (+12% at {fmt(tp_price)}):",
            f"  Gain would be: ${tp_gain:.2f}",
            "",
            f"_Break-even price: {fmt(entry)}_",
        ]

        asyncio.run_coroutine_threadsafe(
            update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN), main_loop
        )

    threading.Thread(target=_run, daemon=True).start()


# ── /compare ──────────────────────────────────────────────────────────────────

async def cmd_compare(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """/compare BTC ETH — side-by-side comparison of two assets."""
    if len(ctx.args) < 2:
        await update.message.reply_text("Usage: `/compare BTC ETH`\nExamples: `/compare GOLD OIL` `/compare AAPL TSLA`", parse_mode=ParseMode.MARKDOWN)
        return

    sym1, sym2 = ctx.args[0].upper(), ctx.args[1].upper()
    main_loop  = asyncio.get_event_loop()
    await update.message.reply_text(f"⚖️ Comparing *{sym1}* vs *{sym2}*…", parse_mode=ParseMode.MARKDOWN)

    def _run():
        import numpy as np
        from ml.trainer import fetch_training_data

        d1 = fetch_training_data(sym1, days=90)
        d2 = fetch_training_data(sym2, days=90)

        if d1 is None or d2 is None:
            asyncio.run_coroutine_threadsafe(
                update.message.reply_text(f"Could not fetch data for one or both symbols."), main_loop
            )
            return

        def stats(df, sym):
            c = df["Close"]
            ret_1d  = (c.iloc[-1]/c.iloc[-2] - 1)*100 if len(c)>=2 else 0
            ret_7d  = (c.iloc[-1]/c.iloc[-8] - 1)*100 if len(c)>=8 else 0
            ret_30d = (c.iloc[-1]/c.iloc[-31]- 1)*100 if len(c)>=31 else 0
            vol     = float(c.pct_change().std()*100)
            high90  = float(df["High"].max())
            low90   = float(df["Low"].min())
            price   = float(c.iloc[-1])
            sma20   = float(c.rolling(20).mean().iloc[-1])
            trend   = "Bullish" if price > sma20 else "Bearish"
            return {"price":price,"ret_1d":ret_1d,"ret_7d":ret_7d,"ret_30d":ret_30d,
                    "vol":vol,"high":high90,"low":low90,"trend":trend}

        s1 = stats(d1, sym1)
        s2 = stats(d2, sym2)

        # Correlation
        c1 = d1["Close"].pct_change().dropna()
        c2 = d2["Close"].pct_change().dropna()
        min_len = min(len(c1), len(c2))
        corr = float(np.corrcoef(c1.tail(min_len).values, c2.tail(min_len).values)[0,1]) if min_len > 5 else 0

        def fmt(v): return f"${v:,.4f}" if v < 100 else f"${v:,.2f}"
        def pct(v): return f"{'+' if v>=0 else ''}{v:.2f}%"
        def win(v1,v2,higher=True):
            return ("✅","  ") if (v1>v2)==higher else ("  ","✅")

        lines = [
            f"⚖️ *{sym1} vs {sym2}*",
            "",
            f"{'Metric':<18} {sym1:<14} {sym2}",
            f"{'Price':<18} {fmt(s1['price']):<14} {fmt(s2['price'])}",
            f"{'24h Change':<18} {pct(s1['ret_1d']):<14} {pct(s2['ret_1d'])}",
            f"{'7d Change':<18} {pct(s1['ret_7d']):<14} {pct(s2['ret_7d'])}",
            f"{'30d Change':<18} {pct(s1['ret_30d']):<14} {pct(s2['ret_30d'])}",
            f"{'Volatility':<18} {s1['vol']:.2f}%{'':<9} {s2['vol']:.2f}%",
            f"{'Trend (SMA20)':<18} {s1['trend']:<14} {s2['trend']}",
            f"{'90d High':<18} {fmt(s1['high']):<14} {fmt(s2['high'])}",
            f"{'90d Low':<18} {fmt(s1['low']):<14} {fmt(s2['low'])}",
            "",
            f"*Correlation (90d):* {corr:.2f}",
        ]

        if corr > 0.7:
            lines.append(f"_These two move together strongly — when {sym1} goes up, {sym2} tends to go up too._")
        elif corr < -0.3:
            lines.append(f"_These two move opposite — when {sym1} goes up, {sym2} tends to go down._")
        else:
            lines.append(f"_Low correlation — these assets move independently._")

        winner_30d = sym1 if s1["ret_30d"] > s2["ret_30d"] else sym2
        lines.append(f"\n*Better performer (30d):* {winner_30d}")

        asyncio.run_coroutine_threadsafe(
            update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN), main_loop
        )

    threading.Thread(target=_run, daemon=True).start()


# ── /dominance ────────────────────────────────────────────────────────────────

async def cmd_dominance(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Crypto market dominance — BTC, ETH, stablecoins vs altcoins."""
    await update.message.reply_text("🌐 Fetching market dominance…")

    def _run():
        import requests
        try:
            r = requests.get("https://api.coingecko.com/api/v3/global", timeout=10)
            data = r.json()["data"]
            dom  = data["market_cap_percentage"]
            total_mcap = data["total_market_cap"].get("usd", 0)
            btc_d   = dom.get("btc", 0)
            eth_d   = dom.get("eth", 0)
            usdt_d  = dom.get("usdt", 0)
            usdc_d  = dom.get("usdc", 0)
            others  = 100 - btc_d - eth_d - usdt_d - usdc_d
            total_b = total_mcap / 1e9

            def bar(pct, w=10):
                filled = round(pct / 10)
                return "█" * filled + "░" * (w - filled)

            lines = [
                "*🌐 Crypto Market Dominance*",
                f"Total Market Cap: ${total_b:,.0f}B",
                "",
                f"₿ BTC  [{bar(btc_d)}] {btc_d:.1f}%",
                f"Ξ ETH  [{bar(eth_d)}] {eth_d:.1f}%",
                f"₮ USDT [{bar(usdt_d)}] {usdt_d:.1f}%",
                f"© USDC [{bar(usdc_d)}] {usdc_d:.1f}%",
                f"🔷 ALT  [{bar(others)}] {others:.1f}%",
                "",
            ]

            if btc_d > 55:
                lines.append("📊 *BTC dominance is HIGH* — money flowing into Bitcoin, altcoins underperforming. Consider focusing on BTC.")
            elif btc_d < 40:
                lines.append("📊 *BTC dominance is LOW* — altcoin season! Money flowing into smaller coins.")
            else:
                lines.append("📊 *Balanced market* — BTC and altcoins moving roughly together.")

            stable_d = usdt_d + usdc_d
            if stable_d > 10:
                lines.append(f"⚠️ *High stablecoin dominance ({stable_d:.1f}%)* — investors parked in cash. Market uncertainty.")

            asyncio.run_coroutine_threadsafe(
                update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN), main_loop
            )
        except Exception as e:
            asyncio.run_coroutine_threadsafe(
                update.message.reply_text(f"Could not fetch dominance data: {e}"), main_loop
            )

    main_loop = asyncio.get_event_loop()
    threading.Thread(target=_run, daemon=True).start()


# ── /gas ──────────────────────────────────────────────────────────────────────

async def cmd_gas(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Current Ethereum gas fees."""
    try:
        import requests
        r = requests.get("https://api.etherscan.io/api?module=gastracker&action=gasoracle", timeout=8)
        d = r.json()["result"]
        slow   = d.get("SafeGasPrice", "?")
        avg    = d.get("ProposeGasPrice", "?")
        fast   = d.get("FastGasPrice", "?")

        lines = [
            "*⛽ Ethereum Gas Fees*",
            "",
            f"🐢 Slow (may take 10+ min): *{slow} Gwei*",
            f"🚶 Average (2-3 min):        *{avg} Gwei*",
            f"🚀 Fast (under 30 sec):      *{fast} Gwei*",
            "",
            "_1 Gwei = 0.000000001 ETH_",
            "_Simple transfer costs ~21,000 gas units_",
        ]
        try:
            eth_price_r = requests.get("https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd", timeout=5)
            eth_usd = eth_price_r.json()["ethereum"]["usd"]
            transfer_cost_usd = float(avg) * 21000 * 1e-9 * eth_usd
            lines.append(f"_A simple ETH transfer costs ~${transfer_cost_usd:.3f} USD at average speed_")
        except Exception:
            pass

        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)
    except Exception:
        await update.message.reply_text(
            "⛽ Could not fetch gas data. Try: https://etherscan.io/gastracker"
        )


# ── /watchlist ────────────────────────────────────────────────────────────────

async def cmd_watchlist(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Quick overview of all trained models with live price + signal."""
    from database import get_all_models
    models = get_all_models()
    if not models:
        await update.message.reply_text("No trained models yet. Run `/train all` first.", parse_mode=ParseMode.MARKDOWN)
        return

    main_loop = asyncio.get_event_loop()
    await update.message.reply_text("📋 Building watchlist…")

    def _run():
        from ml.trainer import fetch_training_data, predict_symbol
        from config import STABLECOIN_SYMBOLS

        lines = ["*📋 Watchlist — Live Signals*\n"]
        groups = {
            "🪙 Crypto": [],
            "🐸 Meme": [],
            "📈 Stocks": [],
            "💱 Forex": [],
            "🥇 Commodities": [],
        }

        _meme = {"DOGE","SHIB","PEPE","WIF","BONK","FLOKI","BRETT","MOG","TURBO","BABYDOGE","POPCAT"}
        _forex = {"EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD","USDCHF"}
        _commod = {"GOLD","OIL","SILVER","NATGAS","COPPER","WHEAT"}
        _stocks = {"AAPL","TSLA","NVDA","MSFT","AMZN","GOOGL","META","SPY","QQQ","AMD","NFLX"}
        from config import CRYPTO_IDS

        for m in models:
            sym = m["symbol"]
            if sym in STABLECOIN_SYMBOLS:
                continue
            try:
                df = fetch_training_data(sym, days=5)
                price = float(df["Close"].iloc[-1]) if df is not None and not df.empty else None
                pred  = predict_symbol(sym, 0.0, 50) if price else None
                if price and pred and "error" not in pred:
                    sig  = pred["signal"]
                    conf = pred["confidence"]
                    will_trade = sig == "BUY" and conf >= 0.60
                    sig_icon = "🟢" if sig == "BUY" else "🔴"
                    trade_icon = "⚡" if will_trade else "👁"
                    entry = f"{sig_icon}{trade_icon} {sym}: ${price:,.4f}  {sig} {conf*100:.0f}%"
                else:
                    entry = f"⚪ {sym}: N/A"
            except Exception:
                entry = f"⚪ {sym}: error"

            if sym in _meme:             groups["🐸 Meme"].append(entry)
            elif sym in _forex:          groups["💱 Forex"].append(entry)
            elif sym in _commod:         groups["🥇 Commodities"].append(entry)
            elif sym in _stocks:         groups["📈 Stocks"].append(entry)
            elif sym in CRYPTO_IDS:      groups["🪙 Crypto"].append(entry)

        for group, entries in groups.items():
            if entries:
                lines.append(f"*{group}*")
                lines.extend(entries)
                lines.append("")

        lines.append("_⚡ = will auto-trade  👁 = watching, below 60%_")

        asyncio.run_coroutine_threadsafe(
            update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN), main_loop
        )

    threading.Thread(target=_run, daemon=True).start()


# ── /report ───────────────────────────────────────────────────────────────────

async def cmd_report(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Full market briefing — prices, signals, sentiment, movers."""
    await update.message.reply_text("📰 Generating full market report…")
    main_loop = asyncio.get_event_loop()
    chat_id = update.effective_chat.id
    await _send_report(chat_id, main_loop)


async def send_daily_report(app):
    """Called by APScheduler every morning at 8 AM."""
    from trading.auto_trader import _watched
    if not _watched:
        return
    loop = asyncio.get_event_loop()
    for chat_id in list(_watched.keys()):
        await _send_report(chat_id, loop, via_app=app)


async def _send_report(chat_id: int, loop, via_app=None):
    """Build and send the daily report to a chat."""
    import requests
    from data.crypto import get_fear_greed
    from utils.formatters import fmt_price

    def _run():
        try:
            lines = ["*📰 Daily Market Report*\n"]

            # Fear & Greed
            fg = get_fear_greed()
            if fg:
                fgv = fg["value"]
                fgc = fg["classification"]
                if fgv <= 25:   mood = "😱 Extreme Fear — historically good buying opportunity"
                elif fgv <= 45: mood = "😨 Fear — market cautious"
                elif fgv <= 55: mood = "😐 Neutral"
                elif fgv <= 75: mood = "😊 Greed — market optimistic"
                else:           mood = "🤑 Extreme Greed — be careful"
                lines.append(f"*Market Mood:* {mood} ({fgv}/100)")

            # Key crypto prices
            try:
                import yfinance as yf
                watch = ["BTC-USD","ETH-USD","SOL-USD","GOLD","GC=F","EURUSD=X"]
                names = {"BTC-USD":"BTC","ETH-USD":"ETH","SOL-USD":"SOL","GC=F":"GOLD","EURUSD=X":"EURUSD"}
                lines.append("\n*Key Prices (24h change)*")
                for ticker, name in names.items():
                    try:
                        df = yf.Ticker(ticker).history(period="2d")
                        if len(df) >= 2:
                            price = float(df["Close"].iloc[-1])
                            prev  = float(df["Close"].iloc[-2])
                            chg   = (price/prev - 1)*100
                            icon  = "🟢" if chg >= 0 else "🔴"
                            lines.append(f"  {icon} {name}: ${price:,.4f}  ({chg:+.2f}%)")
                    except Exception:
                        pass
            except Exception:
                pass

            # Performance summary
            try:
                from trading.performance import compute_metrics, bot_rating
                metrics = compute_metrics()
                if metrics and metrics.get("total_trades", 0) > 0:
                    rating = bot_rating(metrics)
                    lines.append(f"\n*Bot Performance:* {rating}")
                    lines.append(f"  Trades: {metrics['total_trades']}  |  Win rate: {metrics['win_rate']*100:.1f}%")
                    lines.append(f"  Total return: {metrics['total_return_pct']:+.1f}%")
            except Exception:
                pass

            lines.append("\n_Use /watchlist for all signals, /movers for top gainers/losers_")

            msg = "\n".join(lines)
            if via_app:
                asyncio.run_coroutine_threadsafe(
                    via_app.bot.send_message(chat_id=chat_id, text=msg, parse_mode=ParseMode.MARKDOWN), loop
                )
            else:
                asyncio.run_coroutine_threadsafe(
                    loop.create_task(
                        __import__("telegram").Bot(
                            __import__("os").getenv("TELEGRAM_BOT_TOKEN","")
                        ).send_message(chat_id=chat_id, text=msg, parse_mode=ParseMode.MARKDOWN)
                    ), loop
                )
        except Exception as e:
            pass

    threading.Thread(target=_run, daemon=True).start()


# ── /backtest ─────────────────────────────────────────────────────────────────

async def cmd_backtest(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """/backtest BTC 30 — simulate strategy on last N days of real data."""
    if not ctx.args:
        await update.message.reply_text(
            "*Backtest — simulate your strategy on historical data*\n\n"
            "Usage: `/backtest SYMBOL [days]`\n"
            "Examples:\n  `/backtest BTC 30`\n  `/backtest GOLD 60`\n  `/backtest TSLA 90`",
            parse_mode=ParseMode.MARKDOWN
        )
        return

    symbol = ctx.args[0].upper()
    days   = int(ctx.args[1]) if len(ctx.args) > 1 and ctx.args[1].isdigit() else 30
    days   = min(max(days, 10), 180)

    main_loop = asyncio.get_event_loop()
    await update.message.reply_text(
        f"Running backtest for *{symbol}* over last *{days} days*... (takes ~15 seconds)",
        parse_mode=ParseMode.MARKDOWN
    )

    def _run():
        try:
            from ml.trainer import fetch_training_data, predict_symbol
            from trading.demo_wallet import _asset_class

            df = fetch_training_data(symbol, days=days + 30)
            if df is None or len(df) < 20:
                asyncio.run_coroutine_threadsafe(
                    update.message.reply_text(f"Not enough data for {symbol}."), main_loop)
                return

            sl_map = {"meme": 0.10, "crypto": 0.07, "forex": 0.02, "stock": 0.05, "commodity": 0.05}
            tp_map = {"meme": 0.25, "crypto": 0.15, "forex": 0.05, "stock": 0.12, "commodity": 0.10}
            cls    = _asset_class(symbol)
            sl_pct = sl_map.get(cls, 0.05)
            tp_pct = tp_map.get(cls, 0.12)

            closes = df["Close"].values
            window = max(0, len(closes) - days - 1)
            equity = 10000.0
            in_trade = False
            entry_px = stop_loss = take_profit = 0.0
            trades = []

            for i in range(window, len(closes) - 1):
                price      = float(closes[i])
                next_price = float(closes[i + 1])

                if in_trade:
                    if next_price <= stop_loss:
                        pnl_pct = (stop_loss / entry_px - 1) * 100
                        pnl = equity * 0.20 * (pnl_pct / 100)
                        equity += pnl
                        trades.append({"result": "LOSS", "pnl_pct": pnl_pct, "pnl": pnl})
                        in_trade = False
                    elif next_price >= take_profit:
                        pnl_pct = (take_profit / entry_px - 1) * 100
                        pnl = equity * 0.20 * (pnl_pct / 100)
                        equity += pnl
                        trades.append({"result": "WIN", "pnl_pct": pnl_pct, "pnl": pnl})
                        in_trade = False
                else:
                    try:
                        pred = predict_symbol(symbol, 0.0, 50)
                        if pred.get("signal") == "BUY" and pred.get("confidence", 0) >= 0.60:
                            in_trade    = True
                            entry_px    = price
                            stop_loss   = price * (1 - sl_pct)
                            take_profit = price * (1 + tp_pct)
                    except Exception:
                        pass

            if not trades:
                asyncio.run_coroutine_threadsafe(
                    update.message.reply_text(f"No trades triggered for {symbol} in {days} days with current model."), main_loop)
                return

            wins     = [t for t in trades if t["result"] == "WIN"]
            losses   = [t for t in trades if t["result"] == "LOSS"]
            win_rate = len(wins) / len(trades) * 100
            total_ret = (equity / 10000 - 1) * 100
            avg_win  = sum(t["pnl_pct"] for t in wins) / len(wins) if wins else 0
            avg_loss = sum(t["pnl_pct"] for t in losses) / len(losses) if losses else 0

            grade = ("A" if win_rate >= 65 and total_ret > 5 else
                     "B" if win_rate >= 55 and total_ret > 0 else
                     "C" if win_rate >= 50 else "D")
            bar = "X" * int(win_rate / 10) + "." * (10 - int(win_rate / 10))

            lines = [
                f"Backtest Results -- {symbol} ({days}d)",
                "",
                f"Grade: {grade}",
                f"Trades: {len(trades)}  Wins: {len(wins)}  Losses: {len(losses)}",
                f"Win Rate: [{bar}] {win_rate:.1f}%",
                f"Total Return: {total_ret:+.2f}%",
                f"Avg Win: +{avg_win:.2f}%   Avg Loss: {avg_loss:.2f}%",
                f"Final equity: ${equity:,.2f} (started $10,000)",
                "",
                f"Settings: SL={sl_pct*100:.0f}%  TP={tp_pct*100:.0f}%  Size=20%",
                "",
                "Past performance does not guarantee future results.",
            ]

            asyncio.run_coroutine_threadsafe(
                update.message.reply_text("\n".join(lines)), main_loop)

        except Exception as e:
            asyncio.run_coroutine_threadsafe(
                update.message.reply_text(f"Backtest error: {e}"), main_loop)

    threading.Thread(target=_run, daemon=True).start()


# ── /export ───────────────────────────────────────────────────────────────────

async def cmd_export(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """/export — download all trade history as a CSV file."""
    import io, csv
    from database import get_all_closed_trades

    trades = get_all_closed_trades()
    if not trades:
        await update.message.reply_text("No completed trades yet to export.")
        return

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Symbol","Asset Type","Entry Price","Exit Price","Quantity",
                     "Cost","Revenue","P&L","P&L %","Result","Confidence",
                     "Reason","Opened At","Closed At"])
    for t in trades:
        t = dict(t)
        writer.writerow([
            t.get("symbol",""), t.get("asset_type",""),
            f"{t.get('entry_price',0):.6f}", f"{t.get('exit_price',0):.6f}",
            f"{t.get('quantity',0):.6f}", f"{t.get('cost',0):.2f}",
            f"{t.get('revenue',0):.2f}", f"{t.get('pnl',0):.2f}",
            f"{t.get('pnl_pct',0):.2f}", t.get("result",""),
            f"{t.get('confidence',0):.3f}", t.get("signal",""),
            t.get("opened_at",""), t.get("closed_at",""),
        ])

    output.seek(0)
    csv_bytes = io.BytesIO(output.getvalue().encode("utf-8"))
    csv_bytes.name = "trade_history.csv"

    await update.message.reply_document(
        document=csv_bytes,
        filename="trade_history.csv",
        caption=f"Trade history — {len(trades)} completed trades. Open in Excel or Google Sheets."
    )


# ── /funding ──────────────────────────────────────────────────────────────────

async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """/status — show live bot activity stats."""
    from trading.auto_trader import get_activity_status, is_watching
    import database as db

    chat_id = update.effective_chat.id
    s = get_activity_status()

    if s["last_cycle"] is None:
        last = "No cycle run yet — first cycle starts within 15 min"
    else:
        diff = (s["last_cycle"].replace(tzinfo=None) if s["last_cycle"].tzinfo else s["last_cycle"])
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        secs = int((now - s["last_cycle"]).total_seconds())
        if secs < 60:
            last = f"{secs}s ago"
        else:
            last = f"{secs // 60}m {secs % 60}s ago"

    active = is_watching(chat_id)
    wallet = db.get_wallet()
    total  = (wallet["cash"] + wallet["positions_value"]) if wallet else 10000
    ret    = ((total - 10000) / 10000) * 100

    lines = [
        f"*Bot Status*",
        f"",
        f"{'🟢 AUTO-TRADING ACTIVE' if active else '🔴 AUTO-TRADING INACTIVE'}",
        f"Watching: {s['watching']} symbols",
        f"",
        f"*Last Cycle:* {last}",
        f"Symbols scanned: {s['scanned']}",
        f"Trades executed: {s['trades']}",
        f"Filtered/skipped: {s['skipped']}",
        f"Total cycles run: {s['total_cycles']}",
        f"",
        f"*Open Positions:* {s['open_positions']}",
        f"*Wallet:* ${total:,.2f} ({ret:+.2f}%)",
        f"",
        f"Next full scan: within 15 min",
        f"Position monitor: every 2 min",
    ]
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)


async def cmd_funding(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """/funding BTC — show crypto futures funding rate."""
    if not ctx.args:
        await update.message.reply_text("Usage: `/funding BTC`", parse_mode=ParseMode.MARKDOWN)
        return
    symbol = ctx.args[0].upper()
    from trading.smart_features import get_funding_rate
    rate, interp = get_funding_rate(symbol)
    if rate == 0.0 and "unavailable" in interp:
        await update.message.reply_text(f"Could not fetch funding rate for {symbol}.")
        return
    lines = [
        f"Funding Rate -- {symbol}",
        "",
        f"Rate: {rate:+.4f}% per 8 hours",
        f"Status: {interp}",
        "",
        "Negative funding = shorts paying longs = market oversold (contrarian BUY).",
        "Positive funding = longs paying shorts = bullish but possibly overcrowded.",
    ]
    await update.message.reply_text("\n".join(lines))
