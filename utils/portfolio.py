"""
In-memory portfolio store per user.
Each holding: {symbol, asset_type, quantity, buy_price, buy_date}
"""
from dataclasses import dataclass, field
from datetime import date


@dataclass
class Holding:
    symbol: str
    asset_type: str   # "crypto" | "stock" | "forex" | "commodity"
    quantity: float
    buy_price: float
    buy_date: str = field(default_factory=lambda: str(date.today()))


# {chat_id: {symbol: Holding}}
_portfolios: dict[int, dict[str, Holding]] = {}


def add_holding(chat_id: int, symbol: str, asset_type: str, quantity: float, buy_price: float) -> Holding:
    if chat_id not in _portfolios:
        _portfolios[chat_id] = {}
    key = symbol.upper()
    if key in _portfolios[chat_id]:
        # Average down / up
        existing = _portfolios[chat_id][key]
        total_qty = existing.quantity + quantity
        avg_price = (existing.buy_price * existing.quantity + buy_price * quantity) / total_qty
        existing.quantity = total_qty
        existing.buy_price = avg_price
        return existing
    holding = Holding(key, asset_type, quantity, buy_price)
    _portfolios[chat_id][key] = holding
    return holding


def remove_holding(chat_id: int, symbol: str) -> bool:
    key = symbol.upper()
    if chat_id in _portfolios and key in _portfolios[chat_id]:
        del _portfolios[chat_id][key]
        return True
    return False


def get_portfolio(chat_id: int) -> list[Holding]:
    return list(_portfolios.get(chat_id, {}).values())


def get_holding(chat_id: int, symbol: str) -> Holding | None:
    return _portfolios.get(chat_id, {}).get(symbol.upper())


def calculate_pnl(holding: Holding, current_price: float) -> dict:
    cost = holding.buy_price * holding.quantity
    value = current_price * holding.quantity
    pnl = value - cost
    pnl_pct = ((current_price - holding.buy_price) / holding.buy_price) * 100
    return {
        "symbol": holding.symbol,
        "quantity": holding.quantity,
        "buy_price": holding.buy_price,
        "current_price": current_price,
        "cost": cost,
        "value": value,
        "pnl": pnl,
        "pnl_pct": pnl_pct,
    }
