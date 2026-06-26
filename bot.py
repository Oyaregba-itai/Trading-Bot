import logging
import asyncio
from telegram import Update
from telegram.ext import Application, CommandHandler

from config import TELEGRAM_TOKEN, ALERT_CHECK_INTERVAL
from database import init_db
from handlers.core_handlers import cmd_start, cmd_help, cmd_myid, cmd_debug, error_handler
from handlers.price_handlers import cmd_price, cmd_top, cmd_trending, cmd_fear_greed, cmd_market
from handlers.analysis_handlers import cmd_analyze, cmd_rsi, cmd_macd
from handlers.alert_handlers import cmd_alert, cmd_alerts, cmd_cancel_alert
from handlers.portfolio_handlers import cmd_buy, cmd_sell, cmd_portfolio
from handlers.news_handlers import cmd_news
from handlers.ml_handlers import cmd_train, cmd_predict, cmd_accuracy, cmd_sources, cmd_importance
from handlers.trading_handlers import (cmd_autotrade, cmd_wallet, cmd_trades,
                                        cmd_performance, cmd_close, cmd_reset)
from handlers.extra_handlers import (cmd_movers, cmd_levels, cmd_calc,
                                      cmd_compare, cmd_dominance, cmd_gas,
                                      cmd_watchlist, cmd_report,
                                      cmd_backtest, cmd_export, cmd_funding, cmd_status,
                                      cmd_grid, cmd_gridstop, cmd_gridview,
                                      cmd_dca, cmd_dcastop, cmd_dcaview)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


async def check_alerts(app: Application):
    from utils.alerts import get_all_active_alerts, check_alert, mark_triggered
    from handlers.portfolio_handlers import _get_current_price
    from utils.formatters import fmt_price

    for alert in get_all_active_alerts():
        try:
            price = _get_current_price(alert.symbol, alert.asset_type)
            if price and check_alert(alert, price):
                mark_triggered(alert.alert_id)
                direction = "above" if alert.condition == "above" else "below"
                await app.bot.send_message(
                    chat_id=alert.chat_id,
                    text=(
                        f"*Alert Triggered!*\n\n"
                        f"*{alert.symbol}* is now {direction} {fmt_price(alert.target)}\n"
                        f"Current price: {fmt_price(price)}"
                    ),
                    parse_mode="Markdown"
                )
        except Exception as e:
            logger.error("Alert check error for %s: %s", alert.symbol, e)


async def run_auto_trading(app: Application):
    from trading.auto_trader import run_trading_cycle
    await run_trading_cycle(app)


async def run_dca_cycle(app: Application):
    from trading.dca_bot import run_dca_cycle as _dca
    from trading.auto_trader import _notify
    notifications = _dca(app)
    for chat_id, msg in notifications:
        await _notify(chat_id, msg)


async def run_position_monitor(app: Application):
    """
    Fast 2-minute cycle — only checks open positions.
    No ML, no sentiment, no new trades.
    Just: trailing stop update, SL/TP trigger, smart early exit.
    """
    from trading.auto_trader import _watched, _notify, _get_price, _format_exit_msg
    from trading.demo_wallet import check_stop_take, smart_position_review
    from data.crypto import get_fear_greed
    import database as db

    if not _watched:
        return

    positions = db.get_all_positions()
    if not positions:
        return

    fg_data    = get_fear_greed()
    fear_greed = fg_data["value"] if fg_data else 50

    for pos in positions:
        symbol = pos["symbol"]
        price  = _get_price(symbol)
        if price is None:
            continue

        # 1. Trailing stop + SL/TP
        result = check_stop_take(symbol, price)
        if result:
            msg = _format_exit_msg(result, price)
            for chat_id, syms in _watched.items():
                if symbol in syms:
                    await _notify(chat_id, msg)
            continue

        # 2. Smart early exit (uses cached ML — no new sentiment scrape)
        try:
            smart_result = smart_position_review(symbol, price, 0.0, fear_greed)
            if smart_result:
                msg = _format_exit_msg(smart_result, price)
                for chat_id, syms in _watched.items():
                    if symbol in syms:
                        await _notify(chat_id, msg)
        except Exception:
            pass

    # 3. Grid bot cycle
    try:
        from trading.grid_bot import run_grid_cycle
        from trading.auto_trader import _notify
        grid_notifications = run_grid_cycle(app)
        for chat_id, msg in grid_notifications:
            await _notify(chat_id, msg)
    except Exception:
        pass


async def auto_retrain(app: Application):
    """
    Weekly background retraining job.
    Runs every 7 days, retrains all models in a background thread,
    then notifies all active auto-trade users when done.
    """
    import threading
    from trading.auto_trader import _watched
    from ml.trainer import train_symbol
    from handlers.ml_handlers import ALL_SYMBOLS

    notify_chats = list(_watched.keys())
    loop = asyncio.get_event_loop()

    def _notify_chats(msg: str):
        for chat_id in notify_chats:
            asyncio.run_coroutine_threadsafe(
                app.bot.send_message(chat_id=chat_id, text=msg, parse_mode="Markdown"),
                loop,
            )

    def _run():
        logger.info("Auto-retraining all models…")
        if notify_chats:
            _notify_chats("*Auto-Retraining Started*\n\nWeekly model update — retraining all 9 symbols with latest data. This takes ~5 minutes.")
        results = []
        for sym in ALL_SYMBOLS:
            try:
                r = train_symbol(sym)
                results.append(r)
            except Exception as e:
                results.append({"symbol": sym, "error": str(e)})

        ok  = [r for r in results if "error" not in r]
        err = [r for r in results if "error" in r]
        summary = (
            f"*Auto-Retraining Complete*\n\n"
            f"✅ {len(ok)} models updated\n" +
            "".join(f"  • {r['symbol']}: {r['accuracy']*100:.1f}%\n" for r in ok) +
            (f"\n❌ {len(err)} failed: {', '.join(r['symbol'] for r in err)}" if err else "")
        )
        logger.info("Auto-retraining done: %d ok, %d failed", len(ok), len(err))
        if notify_chats:
            _notify_chats(summary)

    threading.Thread(target=_run, daemon=True).start()


async def _weekly_report(app: Application):
    """Every Sunday 9 AM — send a full week performance summary."""
    from trading.auto_trader import _watched
    from trading.performance import compute_metrics, bot_rating
    from database import get_all_closed_trades
    from datetime import datetime, timezone, timedelta

    if not _watched:
        return

    # Trades from last 7 days
    week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).date().isoformat()
    all_trades = get_all_closed_trades()
    week_trades = [t for t in all_trades if (t["closed_at"] or "") >= week_ago]

    metrics = compute_metrics()
    rating  = bot_rating(metrics)

    wins   = len([t for t in week_trades if t["result"] == "WIN"])
    losses = len([t for t in week_trades if t["result"] == "LOSS"])
    week_pnl = sum(t["pnl"] or 0 for t in week_trades)

    lines = [
        "*Weekly Performance Report*",
        "",
        f"This week: {len(week_trades)} trades | {wins}W / {losses}L | P&L: ${week_pnl:+.2f}",
        f"All-time rating: {rating}",
        f"Total return: {metrics.get('total_return_pct', 0):+.2f}%",
        f"Win rate: {metrics.get('win_rate', 0)*100:.1f}%",
        "",
        "Use /performance for full stats, /export for CSV download.",
    ]
    msg = "\n".join(lines)
    for chat_id in list(_watched.keys()):
        try:
            await app.bot.send_message(chat_id=chat_id, text=msg)
        except Exception:
            pass


def _setup_notify_callback(app: Application):
    from trading.auto_trader import set_notify_callback

    async def notify(chat_id: int, message: str):
        await app.bot.send_message(chat_id=chat_id, text=message, parse_mode="Markdown")

    set_notify_callback(notify)


def main():
    if not TELEGRAM_TOKEN:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN not set!\n"
            "1. Copy .env.example to .env\n"
            "2. Add your token from @BotFather"
        )

    # Initialize SQLite database
    init_db()

    # Restore autotrade sessions that were active before restart
    from trading.auto_trader import restore_sessions, _ensure_default_session
    restore_sessions()
    _ensure_default_session()

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    _setup_notify_callback(app)

    # Post-init: notify owner that bot started and trading is active
    async def _on_startup(application):
        import os
        from trading.auto_trader import _watched
        chat_id = int(os.environ.get("DEFAULT_CHAT_ID", "0"))
        if chat_id and _watched:
            sym_count = sum(len(v) for v in _watched.values())
            try:
                await application.bot.send_message(
                    chat_id=chat_id,
                    text=f"Bot restarted and is ACTIVE\nWatching {sym_count} symbols. Auto-trading ON."
                )
            except Exception:
                pass

    app.post_init = _on_startup

    # ── Core ──────────────────────────────────────────────────────────────────
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help",  cmd_help))
    app.add_handler(CommandHandler("myid",  cmd_myid))
    app.add_handler(CommandHandler("debug", cmd_debug))

    # ── Price ─────────────────────────────────────────────────────────────────
    app.add_handler(CommandHandler("price", cmd_price))
    app.add_handler(CommandHandler("p", cmd_price))
    app.add_handler(CommandHandler("top", cmd_top))
    app.add_handler(CommandHandler("trending", cmd_trending))
    app.add_handler(CommandHandler("market", cmd_market))
    app.add_handler(CommandHandler("fear", cmd_fear_greed))
    app.add_handler(CommandHandler("fg", cmd_fear_greed))

    # ── Technical Analysis ────────────────────────────────────────────────────
    app.add_handler(CommandHandler("analyze", cmd_analyze))
    app.add_handler(CommandHandler("ta", cmd_analyze))
    app.add_handler(CommandHandler("rsi", cmd_rsi))
    app.add_handler(CommandHandler("macd", cmd_macd))

    # ── Alerts ────────────────────────────────────────────────────────────────
    app.add_handler(CommandHandler("alert", cmd_alert))
    app.add_handler(CommandHandler("alerts", cmd_alerts))
    app.add_handler(CommandHandler("cancelalert", cmd_cancel_alert))
    app.add_handler(CommandHandler("delalert", cmd_cancel_alert))

    # ── Manual Portfolio ──────────────────────────────────────────────────────
    app.add_handler(CommandHandler("buy", cmd_buy))
    app.add_handler(CommandHandler("sell", cmd_sell))
    app.add_handler(CommandHandler("portfolio", cmd_portfolio))
    app.add_handler(CommandHandler("pf", cmd_portfolio))

    # ── News ──────────────────────────────────────────────────────────────────
    app.add_handler(CommandHandler("news", cmd_news))

    # ── ML ────────────────────────────────────────────────────────────────────
    app.add_handler(CommandHandler("train", cmd_train))
    app.add_handler(CommandHandler("predict", cmd_predict))
    app.add_handler(CommandHandler("accuracy", cmd_accuracy))
    app.add_handler(CommandHandler("sources", cmd_sources))
    app.add_handler(CommandHandler("importance", cmd_importance))

    # ── Demo Trading ──────────────────────────────────────────────────────────
    app.add_handler(CommandHandler("autotrade", cmd_autotrade))
    app.add_handler(CommandHandler("at", cmd_autotrade))
    app.add_handler(CommandHandler("wallet", cmd_wallet))
    app.add_handler(CommandHandler("trades", cmd_trades))
    app.add_handler(CommandHandler("performance", cmd_performance))
    app.add_handler(CommandHandler("perf", cmd_performance))
    app.add_handler(CommandHandler("close", cmd_close))
    app.add_handler(CommandHandler("reset", cmd_reset))

    # ── Extra Features ────────────────────────────────────────────────────────
    app.add_handler(CommandHandler("movers",    cmd_movers))
    app.add_handler(CommandHandler("levels",    cmd_levels))
    app.add_handler(CommandHandler("calc",      cmd_calc))
    app.add_handler(CommandHandler("compare",   cmd_compare))
    app.add_handler(CommandHandler("dominance", cmd_dominance))
    app.add_handler(CommandHandler("dom",       cmd_dominance))
    app.add_handler(CommandHandler("gas",       cmd_gas))
    app.add_handler(CommandHandler("watchlist", cmd_watchlist))
    app.add_handler(CommandHandler("wl",        cmd_watchlist))
    app.add_handler(CommandHandler("report",    cmd_report))
    app.add_handler(CommandHandler("backtest",  cmd_backtest))
    app.add_handler(CommandHandler("bt",        cmd_backtest))
    app.add_handler(CommandHandler("export",    cmd_export))
    app.add_handler(CommandHandler("funding",   cmd_funding))
    app.add_handler(CommandHandler("status",    cmd_status))
    app.add_handler(CommandHandler("grid",      cmd_grid))
    app.add_handler(CommandHandler("gridstop",  cmd_gridstop))
    app.add_handler(CommandHandler("gridview",  cmd_gridview))
    app.add_handler(CommandHandler("dca",       cmd_dca))
    app.add_handler(CommandHandler("dcastop",   cmd_dcastop))
    app.add_handler(CommandHandler("dcaview",   cmd_dcaview))

    app.add_error_handler(error_handler)

    # ── Background jobs ───────────────────────────────────────────────────────
    jq = app.job_queue
    # Price alert checker every 60s
    jq.run_repeating(
        lambda ctx: asyncio.ensure_future(check_alerts(app)),
        interval=ALERT_CHECK_INTERVAL,
        first=15,
    )
    # Fast position monitor every 2 minutes — SL/TP/trailing stop/smart exit + grid
    jq.run_repeating(
        lambda ctx: asyncio.ensure_future(run_position_monitor(app)),
        interval=120,   # 2 minutes
        first=30,
    )
    # DCA cycle every 15 minutes — check if any DCA buy is due
    jq.run_repeating(
        lambda ctx: asyncio.ensure_future(run_dca_cycle(app)),
        interval=900,
        first=120,
    )
    # Full trading cycle every 15 minutes — ML + sentiment + new trades
    jq.run_repeating(
        lambda ctx: asyncio.ensure_future(run_auto_trading(app)),
        interval=900,   # 15 minutes
        first=90,
    )
    # Weekly auto-retraining — keeps models fresh with latest price data
    jq.run_repeating(
        lambda ctx: asyncio.ensure_future(auto_retrain(app)),
        interval=7 * 24 * 3600,   # every 7 days
        first=7 * 24 * 3600,      # first run after 7 days (not on startup)
    )
    # Daily market report at 8:00 AM UTC
    from handlers.extra_handlers import send_daily_report
    import datetime
    jq.run_daily(
        lambda ctx: asyncio.ensure_future(send_daily_report(app)),
        time=datetime.time(hour=8, minute=0),
    )
    # Weekly performance report every Sunday at 9:00 AM UTC
    jq.run_daily(
        lambda ctx: asyncio.ensure_future(_weekly_report(app)),
        time=datetime.time(hour=9, minute=0),
        days=(6,),  # Sunday only
    )

    logger.info("Trading Bot with ML started. Press Ctrl+C to stop.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
