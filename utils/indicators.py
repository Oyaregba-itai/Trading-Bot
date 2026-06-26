import pandas as pd


def _to_series(prices: list) -> pd.Series:
    if isinstance(prices[0], (list, tuple)):
        return pd.Series([p[1] for p in prices], dtype=float)
    return pd.Series(prices, dtype=float)


def sma(prices: list, period: int = 20) -> float | None:
    s = _to_series(prices)
    if len(s) < period:
        return None
    return round(float(s.rolling(period).mean().iloc[-1]), 6)


def ema(prices: list, period: int = 20) -> float | None:
    s = _to_series(prices)
    if len(s) < period:
        return None
    return round(float(s.ewm(span=period, adjust=False).mean().iloc[-1]), 6)


def rsi(prices: list, period: int = 14) -> float | None:
    s = _to_series(prices)
    if len(s) < period + 1:
        return None
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi_val = 100 - (100 / (1 + rs))
    return round(float(rsi_val.iloc[-1]), 2)


def macd(prices: list, fast: int = 12, slow: int = 26, signal: int = 9) -> dict | None:
    s = _to_series(prices)
    if len(s) < slow + signal:
        return None
    ema_fast = s.ewm(span=fast, adjust=False).mean()
    ema_slow = s.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return {
        "macd": round(float(macd_line.iloc[-1]), 6),
        "signal": round(float(signal_line.iloc[-1]), 6),
        "histogram": round(float(histogram.iloc[-1]), 6),
    }


def bollinger_bands(prices: list, period: int = 20, std_dev: float = 2.0) -> dict | None:
    s = _to_series(prices)
    if len(s) < period:
        return None
    rolling = s.rolling(period)
    mid = rolling.mean()
    std = rolling.std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    price = float(s.iloc[-1])
    mid_val = float(mid.iloc[-1])
    return {
        "upper": round(float(upper.iloc[-1]), 6),
        "middle": round(mid_val, 6),
        "lower": round(float(lower.iloc[-1]), 6),
        "bandwidth": round(float((upper.iloc[-1] - lower.iloc[-1]) / mid_val * 100), 2),
        "price": round(price, 6),
    }


def stochastic(highs: list, lows: list, closes: list, k_period: int = 14, d_period: int = 3) -> dict | None:
    if len(closes) < k_period:
        return None
    df = pd.DataFrame({"high": highs, "low": lows, "close": closes})
    low_min = df["low"].rolling(k_period).min()
    high_max = df["high"].rolling(k_period).max()
    k = 100 * (df["close"] - low_min) / (high_max - low_min)
    d = k.rolling(d_period).mean()
    return {
        "k": round(float(k.iloc[-1]), 2),
        "d": round(float(d.iloc[-1]), 2),
    }


def interpret_rsi(val: float) -> str:
    if val >= 70:
        return "Overbought - potential sell signal"
    elif val <= 30:
        return "Oversold - potential buy signal"
    elif val >= 60:
        return "Bullish momentum"
    elif val <= 40:
        return "Bearish momentum"
    return "Neutral"


def interpret_macd(m: dict) -> str:
    if m["histogram"] > 0 and m["macd"] > m["signal"]:
        return "Bullish - MACD above signal"
    elif m["histogram"] < 0 and m["macd"] < m["signal"]:
        return "Bearish - MACD below signal"
    elif m["histogram"] > 0:
        return "Weakening bullish momentum"
    return "Weakening bearish momentum"


def full_analysis(prices: list) -> dict:
    """Run all indicators and return a summary."""
    result = {}
    result["sma_20"] = sma(prices, 20)
    result["sma_50"] = sma(prices, 50)
    result["ema_20"] = ema(prices, 20)
    result["rsi"] = rsi(prices, 14)
    result["macd"] = macd(prices)
    result["bb"] = bollinger_bands(prices, 20)

    if result["rsi"]:
        result["rsi_signal"] = interpret_rsi(result["rsi"])
    if result["macd"]:
        result["macd_signal"] = interpret_macd(result["macd"])

    # Overall signal
    signals = []
    if result["rsi"]:
        if result["rsi"] <= 30:
            signals.append("BUY")
        elif result["rsi"] >= 70:
            signals.append("SELL")
    if result["macd"]:
        if result["macd"]["histogram"] > 0:
            signals.append("BUY")
        else:
            signals.append("SELL")
    if result["sma_20"] and result["sma_50"]:
        if result["sma_20"] > result["sma_50"]:
            signals.append("BUY")
        else:
            signals.append("SELL")

    buy_count = signals.count("BUY")
    sell_count = signals.count("SELL")
    if buy_count > sell_count:
        result["overall_signal"] = "BUY"
    elif sell_count > buy_count:
        result["overall_signal"] = "SELL"
    else:
        result["overall_signal"] = "NEUTRAL"

    return result
