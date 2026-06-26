from telegram import Update
from telegram.ext import ContextTypes
from telegram.constants import ParseMode

from trading.auto_trader import (start_watching, stop_watching, get_watched,
                                  is_watching, DEFAULT_SYMBOLS)
from trading.demo_wallet import get_portfolio_value, execute_sell
from trading.performance import compute_metrics, bot_rating
from database import get_trade_history, reset_wallet, get_position
from utils.formatters import fmt_price, fmt_large


async def cmd_autotrade(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """
    /autotrade              — show status
    /autotrade start        — start auto-trading default symbols
    /autotrade start BTC ETH — start specific symbols
    /autotrade stop         — stop auto-trading
    """
    chat_id = update.effective_chat.id

    if not ctx.args or ctx.args[0].lower() == "status":
        symbols = get_watched(chat_id)
        if symbols:
            await update.message.reply_text(
                f"*Auto-Trading: ACTIVE*\n\n"
                f"Watching: {', '.join(sorted(symbols))}\n\n"
                f"Checks every 15 minutes.\n"
                f"Use `/autotrade stop` to halt.",
                parse_mode=ParseMode.MARKDOWN
            )
        else:
            await update.message.reply_text(
                "*Auto-Trading: INACTIVE*\n\n"
                "Start with:\n"
                "  `/autotrade start` — all asset classes\n"
                "  `/autotrade start crypto` — major coins (BTC ETH SOL BNB...)\n"
                "  `/autotrade start meme` — DOGE SHIB PEPE WIF BONK FLOKI...\n"
                "  `/autotrade start stocks` — AAPL TSLA NVDA MSFT AMZN...\n"
                "  `/autotrade start forex` — EURUSD GBPUSD USDJPY...\n"
                "  `/autotrade start commodities` — GOLD OIL SILVER NATGAS\n"
                "  `/autotrade start crypto meme` — combine groups\n"
                "  `/autotrade start all` — everything\n\n"
                "⚠️ Stablecoins (USDT/USDC/DAI) are always skipped — price never moves.\n"
                "_Train first: `/train all` or `/train meme`_",
                parse_mode=ParseMode.MARKDOWN
            )
        return

    action = ctx.args[0].lower()

    if action == "start":
        raw_args = [s.upper() for s in ctx.args[1:]]
        # Support group keywords: CRYPTO, STOCKS, FOREX, COMMODITIES, ALL
        _groups = {
            "CRYPTO":      ["BTC","ETH","SOL","BNB","XRP","ADA","AVAX","DOT",
                            "MATIC","LINK","LTC","TRX","TON","NEAR","SUI","ATOM"],
            "MEME":        ["DOGE","SHIB","PEPE","WIF","BONK","FLOKI",
                            "BRETT","MOG","TURBO","BABYDOGE","POPCAT"],
            "STOCKS":      ["AAPL","TSLA","NVDA","MSFT","AMZN","GOOGL","META","SPY","QQQ","AMD"],
            "FOREX":       ["EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD"],
            "COMMODITIES": ["GOLD","OIL","SILVER","NATGAS","COPPER"],
            # Stablecoins excluded from auto-trading (price never moves)
        }
        symbols = []
        for arg in raw_args:
            if arg in _groups:
                symbols.extend(_groups[arg])
            elif arg == "ALL":
                for v in _groups.values():
                    symbols.extend(v)
            else:
                symbols.append(arg)
        if not symbols:
            symbols = DEFAULT_SYMBOLS

        # Deduplicate while preserving order
        seen = set()
        symbols = [s for s in symbols if not (s in seen or seen.add(s))]

        start_watching(chat_id, symbols)

        # Group them for display
        from config import STABLECOIN_SYMBOLS, MEME_COIN_SYMBOLS, CRYPTO_IDS, COMMODITY_SYMBOLS
        # Block stablecoins — price never moves, no useful signal
        stable  = [s for s in symbols if s in STABLECOIN_SYMBOLS]
        symbols = [s for s in symbols if s not in STABLECOIN_SYMBOLS]

        crypto  = [s for s in symbols if s in CRYPTO_IDS and s not in MEME_COIN_SYMBOLS]
        meme    = [s for s in symbols if s in MEME_COIN_SYMBOLS]
        stocks  = [s for s in symbols if s in ["AAPL","TSLA","NVDA","MSFT","AMZN","GOOGL","META","SPY","QQQ","AMD","NFLX"]]
        forex   = [s for s in symbols if s in ["EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD","USDCHF"]]
        commod  = [s for s in symbols if s in COMMODITY_SYMBOLS]

        lines = ["*Auto-Trading STARTED*\n"]
        if crypto:  lines.append(f"🪙 Crypto: {', '.join(crypto)}")
        if meme:    lines.append(f"🐸 Meme coins: {', '.join(meme)}")
        if stocks:  lines.append(f"📈 Stocks: {', '.join(stocks)}")
        if forex:   lines.append(f"💱 Forex: {', '.join(forex)}")
        if commod:  lines.append(f"🥇 Commodities: {', '.join(commod)}")
        if stable:  lines.append(f"⚠️ Skipped (stablecoins, price pegged): {', '.join(stable)}")
        lines += [
            f"\nStarting wallet: $10,000",
            f"Position size: 20% per trade",
            f"Stop loss: -5% | Take profit: +12%",
            f"Min ML confidence: 60%\n",
            f"Checks signals every *15 minutes*. Notifies on every trade.\n",
            f"_Train new symbols with `/train GOLD`, `/train GBPUSD` etc._",
        ]
        await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)

    elif action == "stop":
        stop_watching(chat_id)
        await update.message.reply_text(
            "*Auto-Trading STOPPED*\n\n"
            "Your open positions remain open.\n"
            "Use /wallet to see your positions, /trades for history.",
            parse_mode=ParseMode.MARKDOWN
        )
    else:
        await update.message.reply_text(
            "Usage: `/autotrade start [symbols...]` or `/autotrade stop`",
            parse_mode=ParseMode.MARKDOWN
        )


async def cmd_wallet(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Show current demo wallet state."""
    # Try to get live prices
    prices = {}
    try:
        from database import get_all_positions
        positions = get_all_positions()
        for p in positions:
            from handlers.portfolio_handlers import _get_current_price
            price = _get_current_price(p["symbol"], p["asset_type"])
            if price:
                prices[p["symbol"]] = price
    except Exception:
        pass

    portfolio = get_portfolio_value(prices if prices else None)
    initial = __import__("database").get_initial()
    equity = portfolio["total_equity"]
    total_return = (equity / initial - 1) * 100

    arrow = "▲" if total_return >= 0 else "▼"
    lines = [
        f"*Demo Wallet*",
        f"",
        f"Cash:      {fmt_price(portfolio['cash'])}",
        f"Positions: {fmt_price(portfolio['positions_value'])}",
        f"Total:     *{fmt_price(equity)}*",
        f"Return:    {arrow} {abs(total_return):.2f}%",
        "",
    ]

    if portfolio["positions"]:
        lines.append("*Open Positions*")
        for p in portfolio["positions"]:
            arrow2 = "▲" if p["pnl"] >= 0 else "▼"
            lines.append(
                f"  *{p['symbol']}*  {fmt_price(p['current_price'])}\n"
                f"  Qty: {p['quantity']:.6f}  |  Entry: {fmt_price(p['entry_price'])}\n"
                f"  P&L: {arrow2} {fmt_price(abs(p['pnl']))} ({p['pnl_pct']:+.2f}%)\n"
                f"  SL: {fmt_price(p['stop_loss'])}  TP: {fmt_price(p['take_profit'])}"
            )
    else:
        lines.append("No open positions.")

    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)


async def cmd_trades(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Show recent demo trade history."""
    limit = 10
    if ctx.args:
        try:
            limit = min(int(ctx.args[0]), 30)
        except ValueError:
            pass

    trades = get_trade_history(limit)
    if not trades:
        await update.message.reply_text(
            "No completed trades yet.\n\nStart auto-trading with `/autotrade start`",
            parse_mode=ParseMode.MARKDOWN
        )
        return

    lines = [f"*Last {min(limit, len(trades))} Demo Trades*\n"]
    for t in trades:
        emoji = "✅" if t["result"] == "WIN" else "❌"
        lines.append(
            f"{emoji} *{t['symbol']}* {t['result']}\n"
            f"  Entry: {fmt_price(t['entry_price'])} → Exit: {fmt_price(t['exit_price'])}\n"
            f"  P&L: {t['pnl']:+.2f} ({t['pnl_pct']:+.2f}%)  |  {(t['closed_at'] or '')[:10]}"
        )

    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)


async def cmd_performance(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Full performance report with bot rating."""
    metrics = compute_metrics()
    rating = bot_rating(metrics)

    if metrics["total_trades"] == 0:
        await update.message.reply_text(
            "*Performance Report*\n\n"
            "No completed trades yet.\n\n"
            "Start with `/train all` then `/autotrade start`",
            parse_mode=ParseMode.MARKDOWN
        )
        return

    win_bar = "█" * int(metrics["win_rate"] / 10) + "░" * (10 - int(metrics["win_rate"] / 10))
    ret_arrow = "▲" if metrics["total_return_pct"] >= 0 else "▼"

    lines = [
        f"*Bot Performance Report*",
        f"",
        f"*Bot Rating: {rating}*",
        f"",
        f"*Overview*",
        f"Total Trades:  {metrics['total_trades']}",
        f"Wins / Losses: {metrics['wins']} / {metrics['losses']}",
        f"Win Rate:      [{win_bar}] *{metrics['win_rate']:.1f}%*",
        f"",
        f"*Returns*",
        f"Total P&L:     {metrics['total_pnl']:+.2f}",
        f"Total Return:  {ret_arrow} {abs(metrics['total_return_pct']):.2f}%",
        f"Avg per Trade: {metrics['avg_pct_per_trade']:+.2f}%",
        f"",
        f"*Risk Metrics*",
        f"Sharpe Ratio:  {metrics['sharpe_ratio']}",
        f"Max Drawdown:  -{metrics['max_drawdown_pct']:.2f}%",
        f"Profit Factor: {metrics['profit_factor']}",
        f"",
        f"*Best / Worst*",
        f"Best Trade:   +{metrics['best_trade']:.2f}",
        f"Worst Trade:  {metrics['worst_trade']:.2f}",
        f"Win Streak:    {metrics['max_win_streak']}  |  Loss Streak: {metrics['max_loss_streak']}",
        f"",
        f"*Wallet*",
        f"Start: {fmt_price(metrics['initial_balance'])}  →  Now: {fmt_price(metrics['current_equity'])}",
    ]

    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)


async def cmd_close(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Manually close a demo position: /close BTC"""
    if not ctx.args:
        await update.message.reply_text("Usage: `/close <SYMBOL>`", parse_mode=ParseMode.MARKDOWN)
        return

    symbol = ctx.args[0].upper()
    from handlers.portfolio_handlers import _get_current_price
    pos = get_position(symbol)

    if not pos:
        await update.message.reply_text(f"No open position for *{symbol}*.", parse_mode=ParseMode.MARKDOWN)
        return

    price = _get_current_price(symbol, pos["asset_type"]) or pos["entry_price"]
    result = execute_sell(symbol, price, reason="MANUAL")

    if result:
        emoji = "✅" if result["result"] == "WIN" else "❌"
        await update.message.reply_text(
            f"{emoji} *Position Closed: {symbol}*\n\n"
            f"Entry: {fmt_price(result['entry_price'])} → Exit: {fmt_price(result['exit_price'])}\n"
            f"P&L: {result['pnl']:+.2f} ({result['pnl_pct']:+.2f}%)\n"
            f"Result: *{result['result']}*",
            parse_mode=ParseMode.MARKDOWN
        )


async def cmd_reset(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Reset demo wallet back to $10,000."""
    reset_wallet()
    await update.message.reply_text(
        "*Demo wallet reset to $10,000*\n\nAll open positions cleared. Trade history preserved.\n\nStart again with `/autotrade start`",
        parse_mode=ParseMode.MARKDOWN
    )
