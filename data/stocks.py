import yfinance as yf
from config import COMMODITY_SYMBOLS


def get_stock_price(symbol: str) -> dict | None:
    try:
        ticker = yf.Ticker(symbol.upper())
        info = ticker.fast_info
        hist = ticker.history(period="2d")
        if hist.empty:
            return None
        prev_close = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else float(hist["Close"].iloc[-1])
        price = float(hist["Close"].iloc[-1])
        change = ((price - prev_close) / prev_close) * 100
        return {
            "symbol": symbol.upper(),
            "name": getattr(info, "display_name", symbol.upper()),
            "price": price,
            "change_24h": change,
            "volume": float(hist["Volume"].iloc[-1]),
            "high": float(hist["High"].iloc[-1]),
            "low": float(hist["Low"].iloc[-1]),
            "market_cap": getattr(info, "market_cap", None),
        }
    except Exception:
        return None


def get_commodity_price(name: str) -> dict | None:
    name = name.upper()
    ticker_sym = COMMODITY_SYMBOLS.get(name)
    if not ticker_sym:
        return None
    data = get_stock_price(ticker_sym)
    if data:
        data["symbol"] = name
    return data


def get_multiple_stocks(symbols: list[str]) -> list[dict]:
    results = []
    for sym in symbols:
        data = get_stock_price(sym)
        if data:
            results.append(data)
    return results


def get_stock_history(symbol: str, period: str = "1mo") -> object:
    """Returns a pandas DataFrame with OHLCV data."""
    try:
        ticker = yf.Ticker(symbol.upper())
        return ticker.history(period=period)
    except Exception:
        return None
