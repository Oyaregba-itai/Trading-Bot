"""
Performance metrics for the paper trading account.
Win rate, total return, Sharpe ratio, max drawdown, profit factor, etc.
"""
import math
from database import get_all_closed_trades, get_cash, get_initial


def get_portfolio_value_db() -> float:
    """Estimate total equity: cash + open positions at entry price (conservative)."""
    from database import get_cash, get_all_positions
    cash = get_cash()
    positions = get_all_positions()
    pos_value = sum(p["cost"] for p in positions)   # conservative: use cost basis
    return cash + pos_value


def compute_metrics() -> dict:
    trades = get_all_closed_trades()
    initial = get_initial()
    current_equity = get_portfolio_value_db()

    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "total_return_pct": round((current_equity / initial - 1) * 100, 2),
            "current_equity": current_equity,
            "initial_balance": initial,
        }

    wins   = [t for t in trades if t["result"] == "WIN"]
    losses = [t for t in trades if t["result"] == "LOSS"]
    pnls   = [t["pnl"] for t in trades if t["pnl"] is not None]
    pcts   = [t["pnl_pct"] for t in trades if t["pnl_pct"] is not None]

    total_trades = len(trades)
    win_rate     = len(wins) / total_trades if total_trades > 0 else 0
    total_pnl    = sum(pnls)
    avg_win      = sum(t["pnl"] for t in wins) / len(wins) if wins else 0
    avg_loss     = sum(t["pnl"] for t in losses) / len(losses) if losses else 0
    best_trade   = max(pnls) if pnls else 0
    worst_trade  = min(pnls) if pnls else 0
    avg_pct      = sum(pcts) / len(pcts) if pcts else 0

    # Profit factor: gross profit / gross loss
    gross_profit = sum(t["pnl"] for t in wins if t["pnl"])
    gross_loss   = abs(sum(t["pnl"] for t in losses if t["pnl"]))
    profit_factor = round(gross_profit / gross_loss, 3) if gross_loss > 0 else float("inf")

    # Max drawdown (sequential equity curve)
    equity_curve = [initial]
    running = initial
    for t in sorted(trades, key=lambda x: x["closed_at"] or ""):
        running += (t["pnl"] or 0)
        equity_curve.append(running)
    max_drawdown = _max_drawdown(equity_curve)

    # Simplified Sharpe ratio: avg_return / std_return * sqrt(252)
    sharpe = _sharpe(pcts)

    # Consecutive wins/losses
    streak_win, streak_loss = _streaks([t["result"] for t in sorted(trades, key=lambda x: x["closed_at"] or "")])

    return {
        "total_trades":       total_trades,
        "wins":               len(wins),
        "losses":             len(losses),
        "win_rate":           round(win_rate * 100, 2),
        "total_pnl":          round(total_pnl, 2),
        "total_return_pct":   round((current_equity / initial - 1) * 100, 2),
        "avg_win":            round(avg_win, 2),
        "avg_loss":           round(avg_loss, 2),
        "avg_pct_per_trade":  round(avg_pct, 2),
        "best_trade":         round(best_trade, 2),
        "worst_trade":        round(worst_trade, 2),
        "profit_factor":      profit_factor,
        "max_drawdown_pct":   round(max_drawdown * 100, 2),
        "sharpe_ratio":       sharpe,
        "max_win_streak":     streak_win,
        "max_loss_streak":    streak_loss,
        "current_equity":     round(current_equity, 2),
        "initial_balance":    round(initial, 2),
    }


def _max_drawdown(curve: list[float]) -> float:
    if len(curve) < 2:
        return 0.0
    peak = curve[0]
    max_dd = 0.0
    for val in curve:
        if val > peak:
            peak = val
        dd = (peak - val) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _sharpe(returns_pct: list[float], risk_free: float = 0.0) -> float:
    if len(returns_pct) < 2:
        return 0.0
    n = len(returns_pct)
    avg = sum(returns_pct) / n
    variance = sum((r - avg) ** 2 for r in returns_pct) / (n - 1)
    std = math.sqrt(variance) if variance > 0 else 0
    if std == 0:
        return 0.0
    return round((avg - risk_free) / std * math.sqrt(252), 3)


def _streaks(results: list[str]) -> tuple[int, int]:
    max_win = max_loss = cur_win = cur_loss = 0
    for r in results:
        if r == "WIN":
            cur_win += 1
            cur_loss = 0
        else:
            cur_loss += 1
            cur_win = 0
        max_win  = max(max_win, cur_win)
        max_loss = max(max_loss, cur_loss)
    return max_win, max_loss


def bot_rating(metrics: dict) -> str:
    """Convert metrics into a letter grade + percentage score."""
    score = 0
    total = 0

    # Win rate (max 30 pts)
    wr = metrics.get("win_rate", 0)
    if wr >= 65:
        score += 30
    elif wr >= 55:
        score += 20
    elif wr >= 45:
        score += 10
    total += 30

    # Return (max 25 pts)
    ret = metrics.get("total_return_pct", 0)
    if ret >= 20:
        score += 25
    elif ret >= 10:
        score += 18
    elif ret >= 5:
        score += 12
    elif ret >= 0:
        score += 5
    total += 25

    # Sharpe (max 20 pts)
    sharpe = metrics.get("sharpe_ratio", 0)
    if sharpe >= 2.0:
        score += 20
    elif sharpe >= 1.0:
        score += 14
    elif sharpe >= 0.5:
        score += 8
    total += 20

    # Profit factor (max 15 pts)
    pf = metrics.get("profit_factor", 1)
    if pf == float("inf") or pf >= 2.0:
        score += 15
    elif pf >= 1.5:
        score += 10
    elif pf >= 1.0:
        score += 5
    total += 15

    # Drawdown (max 10 pts — lower is better)
    dd = metrics.get("max_drawdown_pct", 100)
    if dd <= 5:
        score += 10
    elif dd <= 10:
        score += 7
    elif dd <= 20:
        score += 3
    total += 10

    pct = round(score / total * 100, 1)
    if pct >= 80:
        grade = "A+ (Excellent)"
    elif pct >= 65:
        grade = "B (Good)"
    elif pct >= 50:
        grade = "C (Average)"
    elif pct >= 35:
        grade = "D (Poor)"
    else:
        grade = "F (Needs Training)"

    return f"{pct}% — {grade}"
