"""
Historical simulation backtester with Monte Carlo analysis.
Runs the full signal logic on 2 years of hourly data per symbol
to generate a synthetic trade record — gives "500+ trade" statistics
without waiting for months of live trading.
"""
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def _simulate_trades(df: pd.DataFrame, sl_pct: float, tp_pct: float,
                     signals: pd.Series, min_confidence: float = 0.65) -> list[dict]:
    """
    Given OHLCV data and a signal series, simulate all trades and return trade records.
    signals: Series of float confidence values (positive = BUY, 0 = no trade)
    """
    close  = df["Close"].values
    high   = df["High"].values
    low    = df["Low"].values
    n      = len(df)
    trades = []
    in_trade    = False
    entry_idx   = 0
    entry_price = 0.0

    for i in range(n - 1):
        if not in_trade:
            conf = float(signals.iloc[i]) if i < len(signals) else 0.0
            if conf >= min_confidence:
                entry_price = close[i]
                tp_price    = entry_price * (1 + tp_pct)
                sl_price    = entry_price * (1 - sl_pct)
                entry_idx   = i
                in_trade    = True
        else:
            tp_price = entry_price * (1 + tp_pct)
            sl_price = entry_price * (1 - sl_pct)
            hit_tp = high[i] >= tp_price
            hit_sl = low[i]  <= sl_price

            if hit_tp or hit_sl:
                exit_price = tp_price if (hit_tp and not hit_sl) else sl_price
                if hit_tp and hit_sl:
                    exit_price = sl_price  # SL hit first (conservative)
                pnl_pct = (exit_price - entry_price) / entry_price
                trades.append({
                    "entry_idx":   entry_idx,
                    "exit_idx":    i,
                    "entry_price": entry_price,
                    "exit_price":  exit_price,
                    "pnl_pct":     pnl_pct,
                    "result":      "WIN" if pnl_pct > 0 else "LOSS",
                    "bars_held":   i - entry_idx,
                })
                in_trade = False

    return trades


def _compute_metrics(trades: list[dict]) -> dict:
    if not trades:
        return {"n_trades": 0}

    pnls    = [t["pnl_pct"] for t in trades]
    wins    = [p for p in pnls if p > 0]
    losses  = [p for p in pnls if p <= 0]

    win_rate    = len(wins) / len(pnls)
    avg_win     = float(np.mean(wins))  if wins   else 0.0
    avg_loss    = float(np.mean(losses)) if losses else 0.0
    profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float("inf")

    # Equity curve
    equity = np.cumprod([1 + p for p in pnls])
    total_return = float(equity[-1] - 1)
    max_dd = 0.0
    peak   = equity[0]
    for e in equity:
        peak   = max(peak, e)
        max_dd = min(max_dd, (e - peak) / peak)

    # Sharpe (annualised, assuming 252 trading days)
    returns_arr = np.array(pnls)
    sharpe = 0.0
    if returns_arr.std() > 0:
        sharpe = round(float(returns_arr.mean() / returns_arr.std() * np.sqrt(252)), 4)

    return {
        "n_trades":      len(trades),
        "win_rate":      round(win_rate, 4),
        "avg_win_pct":   round(avg_win * 100, 3),
        "avg_loss_pct":  round(avg_loss * 100, 3),
        "profit_factor": round(profit_factor, 3),
        "total_return":  round(total_return * 100, 2),
        "max_drawdown":  round(max_dd * 100, 2),
        "sharpe":        sharpe,
        "expectancy":    round((win_rate * avg_win + (1 - win_rate) * avg_loss) * 100, 4),
    }


def monte_carlo(trades: list[dict], n_sims: int = 1000) -> dict:
    """
    Shuffle trade sequence n_sims times to estimate strategy robustness.
    Returns 5th/50th/95th percentile of final equity.
    """
    if len(trades) < 10:
        return {"p5": 0.0, "p50": 0.0, "p95": 0.0, "ruin_probability": 1.0}

    pnls = np.array([t["pnl_pct"] for t in trades])
    finals = []
    ruin_count = 0

    rng = np.random.default_rng(42)
    for _ in range(n_sims):
        shuffled = rng.permutation(pnls)
        equity   = np.cumprod(1 + shuffled)
        final    = float(equity[-1] - 1)
        finals.append(final)
        if np.min(equity) < 0.50:   # 50% drawdown = ruin
            ruin_count += 1

    finals = np.array(finals)
    return {
        "p5":               round(float(np.percentile(finals, 5)) * 100, 2),
        "p50":              round(float(np.percentile(finals, 50)) * 100, 2),
        "p95":              round(float(np.percentile(finals, 95)) * 100, 2),
        "ruin_probability": round(ruin_count / n_sims, 4),
    }


def run_full_backtest(symbol: str, timeframe: str = "1h") -> dict:
    """
    Full historical simulation on 2 years of data.
    Loads the trained model and replays signals on historical OHLCV.

    Returns a summary dict suitable for Telegram display.
    """
    try:
        import yfinance as yf
        from datetime import datetime, timedelta
        from ml.model import TradingModel
        from ml.features import build_feature_matrix

        from config import CRYPTO_IDS, COMMODITY_SYMBOLS

        sym = symbol.upper()
        # Fetch 2 years of data
        end   = datetime.utcnow()
        start = end - timedelta(days=730)

        if sym in CRYPTO_IDS:
            ticker = f"{sym}-USD"
        elif sym in COMMODITY_SYMBOLS:
            ticker = COMMODITY_SYMBOLS.get(sym, sym)
        elif len(sym) == 6 and sym.isalpha():
            ticker = sym + "=X"
        else:
            ticker = sym

        interval = {"5m": "1h", "1h": "1h", "1d": "1d"}.get(timeframe, "1h")
        df = yf.Ticker(ticker).history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval=interval,
        )
        if df is None or len(df) < 100:
            return {"error": f"Not enough data for {symbol}"}

        df = df[["Open", "High", "Low", "Close", "Volume"]]

        model = TradingModel.load_for(sym, timeframe)
        if model is None:
            return {"error": f"No trained model for {symbol}/{timeframe}"}

        # Generate signal confidence for each bar
        features = build_feature_matrix(df)
        signals  = pd.Series(index=features.index, dtype=float)
        for i in range(60, len(features)):
            try:
                row = features.iloc[[i]]
                sig, conf = model.predict(row)
                signals.iloc[i] = conf if sig == 1 else 0.0
            except Exception:
                signals.iloc[i] = 0.0

        # SL/TP based on timeframe defaults
        sl_map = {"5m": 0.003, "1h": 0.015, "1d": 0.04}
        sl = sl_map.get(timeframe, 0.015)
        tp = sl * 2.0

        trades  = _simulate_trades(df, sl_pct=sl, tp_pct=tp, signals=signals)
        metrics = _compute_metrics(trades)
        mc      = monte_carlo(trades)

        return {
            "symbol":    sym,
            "timeframe": timeframe,
            "period":    f"{start.strftime('%Y-%m-%d')} → {end.strftime('%Y-%m-%d')}",
            "metrics":   metrics,
            "monte_carlo": mc,
        }

    except Exception as e:
        logger.error("Backtest error %s/%s: %s", symbol, timeframe, e)
        return {"error": str(e)}


def format_backtest_report(result: dict) -> str:
    """Format a run_full_backtest() result for Telegram."""
    if "error" in result:
        return f"Backtest error: {result['error']}"

    m  = result["metrics"]
    mc = result["monte_carlo"]
    sym = result["symbol"]
    tf  = result["timeframe"]

    if m.get("n_trades", 0) == 0:
        return f"No trades generated for {sym}/{tf} in the backtest period."

    lines = [
        f"*Backtest: {sym} / {tf}*",
        f"Period: {result['period']}",
        "",
        f"Trades:        {m['n_trades']}",
        f"Win rate:      {m['win_rate']*100:.1f}%",
        f"Avg win:       +{m['avg_win_pct']:.2f}%",
        f"Avg loss:      {m['avg_loss_pct']:.2f}%",
        f"Profit factor: {m['profit_factor']:.2f}x",
        f"Total return:  {m['total_return']:+.1f}%",
        f"Max drawdown:  {m['max_drawdown']:.1f}%",
        f"Sharpe:        {m['sharpe']:.3f}",
        f"Expectancy:    {m['expectancy']:+.3f}% per trade",
        "",
        f"*Monte Carlo (1,000 simulations)*",
        f"Best case (95th):   {mc['p95']:+.1f}%",
        f"Median (50th):      {mc['p50']:+.1f}%",
        f"Worst case (5th):   {mc['p5']:+.1f}%",
        f"Ruin probability:   {mc['ruin_probability']*100:.1f}%",
    ]
    return "\n".join(lines)
