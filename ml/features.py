"""
Feature engineering for the ML trading model.
Input: pandas DataFrame with OHLCV columns or a list of close prices.
Output: feature matrix ready for XGBoost/RandomForest.
"""
import pandas as pd
import numpy as np
from utils.indicators import rsi, macd, bollinger_bands, sma, ema


def build_feature_matrix(df: pd.DataFrame, sentiment_score: float = 0.0,
                         fear_greed: int = 50) -> pd.DataFrame:
    """
    df must have columns: Open, High, Low, Close, Volume (standard yfinance format).
    Returns a DataFrame of features with one row per trading day.
    """
    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # ── Price returns ──────────────────────────────────────────────────────────
    df["ret_1d"] = close.pct_change(1)
    df["ret_3d"] = close.pct_change(3)
    df["ret_7d"] = close.pct_change(7)
    df["ret_14d"] = close.pct_change(14)
    df["ret_30d"] = close.pct_change(30)

    # ── RSI ────────────────────────────────────────────────────────────────────
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, min_periods=14).mean()
    avg_loss = loss.ewm(com=13, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    df["rsi_norm"] = (df["rsi_14"] - 50) / 50   # normalised to -1..+1

    # ── MACD ──────────────────────────────────────────────────────────────────
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = macd_line - signal_line
    df["macd_signal_cross"] = (macd_line > signal_line).astype(int)

    # ── Bollinger Bands ────────────────────────────────────────────────────────
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = sma20 + 2 * std20
    lower_bb = sma20 - 2 * std20
    band_width = (upper - lower_bb) / sma20
    bb_position = (close - lower_bb) / (upper - lower_bb + 1e-10)  # 0=at lower, 1=at upper
    df["bb_position"] = bb_position.clip(0, 1)
    df["bb_width"] = band_width

    # ── Moving Average signals ─────────────────────────────────────────────────
    sma20_s = close.rolling(20).mean()
    sma50_s = close.rolling(50).mean()
    ema20_s = close.ewm(span=20, adjust=False).mean()
    ema50_s = close.ewm(span=50, adjust=False).mean()
    df["sma_ratio"] = (sma20_s / sma50_s) - 1        # positive = uptrend
    df["ema_ratio"] = (ema20_s / ema50_s) - 1
    df["price_vs_sma20"] = (close / sma20_s) - 1
    df["price_vs_sma50"] = (close / sma50_s) - 1

    # ── Volume features ────────────────────────────────────────────────────────
    # Forex (EURUSD etc.) has zero volume — fill NaN with 0 so rows aren't dropped
    vol_nonzero = volume.replace(0, np.nan)
    df["vol_change_1d"] = vol_nonzero.pct_change(1).fillna(0)
    df["vol_change_7d"] = vol_nonzero.pct_change(7).fillna(0)
    vol_avg20 = vol_nonzero.rolling(20).mean()
    df["vol_vs_avg20"] = ((vol_nonzero / vol_avg20) - 1).fillna(0)

    # ── ATR (volatility) ──────────────────────────────────────────────────────
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    df["atr_norm"] = atr / close   # ATR as % of price

    # ── Higher-high / lower-low patterns ──────────────────────────────────────
    df["high_14d_pct"] = (close / close.rolling(14).max()) - 1
    df["low_14d_pct"]  = (close / close.rolling(14).min()) - 1
    df["high_30d_pct"] = (close / close.rolling(30).max()) - 1
    df["low_30d_pct"]  = (close / close.rolling(30).min()) - 1

    # ── Momentum & mean-reversion ──────────────────────────────────────────────
    df["momentum_10"]  = close / close.shift(10) - 1
    df["momentum_20"]  = close / close.shift(20) - 1
    df["momentum_60"]  = close / close.shift(60) - 1

    # ── Candlestick body size (proxy for conviction) ───────────────────────────
    df["body_size"]    = ((close - df["Open"]).abs() / close).fillna(0)
    df["upper_wick"]   = ((high - close.clip(lower=df["Open"])) / close).fillna(0)
    df["lower_wick"]   = ((close.clip(upper=df["Open"]) - low) / close).fillna(0)
    df["body_dir"]     = np.sign(close - df["Open"])   # +1 bullish, -1 bearish candle

    # ── RSI divergence levels ──────────────────────────────────────────────────
    df["rsi_7"]  = _rsi_series(close, 7)
    df["rsi_21"] = _rsi_series(close, 21)
    df["rsi_diff"] = df["rsi_7"] - df["rsi_21"]   # fast vs slow RSI

    # ── Stochastic RSI ────────────────────────────────────────────────────────
    rsi14 = _rsi_series(close, 14)
    rsi_min = rsi14.rolling(14).min()
    rsi_max = rsi14.rolling(14).max()
    df["stoch_rsi"] = ((rsi14 - rsi_min) / (rsi_max - rsi_min + 1e-10)).fillna(0.5)

    # ── Calendar effects ──────────────────────────────────────────────────────
    idx = df.index
    if hasattr(idx, 'dayofweek'):
        df["day_of_week"] = idx.dayofweek / 6   # 0=Mon, 1=Fri, normalised
        df["month"]       = (idx.month - 1) / 11
    else:
        df["day_of_week"] = 0.0
        df["month"]       = 0.0

    # ── Volatility regime ─────────────────────────────────────────────────────
    # Ratio of short-term vs long-term volatility: >1 = rising volatility
    vol_short = close.pct_change().rolling(7).std()
    vol_long  = close.pct_change().rolling(30).std()
    df["vol_regime"] = (vol_short / (vol_long + 1e-10)).fillna(1.0).clip(0, 5)

    # ── VWAP (Volume Weighted Average Price) ──────────────────────────────────
    # Key institutional reference level — price above VWAP = bullish bias
    typical_price = (high + low + close) / 3
    vol_safe = volume.replace(0, 1)
    vwap = (typical_price * vol_safe).rolling(20).sum() / vol_safe.rolling(20).sum()
    df["price_vs_vwap"] = ((close / vwap) - 1).fillna(0)
    df["vwap_slope"] = vwap.pct_change(3).fillna(0)

    # ── ADX (Average Directional Index — trend strength 0-100) ────────────────
    up_move   = high.diff()
    down_move = -low.diff()
    # Keep as pandas Series (preserves datetime index) — np.where strips the index
    plus_dm  = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr2 = pd.concat([high - low, (high - close.shift()).abs(),
                     (low - close.shift()).abs()], axis=1).max(axis=1)
    atr14 = tr2.rolling(14).mean().replace(0, 1e-9)
    pdi   = 100 * plus_dm.rolling(14).mean() / atr14
    mdi   = 100 * minus_dm.rolling(14).mean() / atr14
    dx    = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, 1e-9)
    adx   = dx.rolling(14).mean()
    df["adx"]       = (adx / 100).fillna(0.2)
    df["adx_trend"] = (pdi - mdi).fillna(0) / 100

    # ── Williams %R (overbought/oversold -100 to 0) ───────────────────────────
    h14 = high.rolling(14).max()
    l14 = low.rolling(14).min()
    df["williams_r"] = ((h14 - close) / (h14 - l14 + 1e-10) * -100).fillna(-50) / 100

    # ── CCI (Commodity Channel Index) ─────────────────────────────────────────
    mean_deviation = (typical_price - typical_price.rolling(20).mean()).abs().rolling(20).mean()
    cci = (typical_price - typical_price.rolling(20).mean()) / (0.015 * mean_deviation + 1e-10)
    df["cci_norm"] = cci.clip(-300, 300) / 300   # normalised -1..+1

    # ── Volume delta (buying pressure vs selling pressure) ────────────────────
    # Up candles: volume attributed to buyers; Down candles: to sellers
    candle_dir = np.sign(close - df["Open"])
    df["vol_delta"] = (vol_safe * candle_dir).rolling(10).sum() / (vol_safe.rolling(10).sum() + 1e-9)
    df["vol_delta"] = df["vol_delta"].fillna(0)

    # ── Price velocity (rate of change of momentum) ───────────────────────────
    df["rsi_velocity"]  = _rsi_series(close, 14).diff(3).fillna(0)
    df["price_accel"]   = close.pct_change(1).diff(3).fillna(0)

    # ── Support / Resistance proximity ────────────────────────────────────────
    # How far is price from its 52-week high/low? (proxy for key S/R levels)
    df["dist_52w_high"] = (close / close.rolling(min(252, len(close))).max() - 1).fillna(0)
    df["dist_52w_low"]  = (close / close.rolling(min(252, len(close))).min() - 1).fillna(0)

    # ── External signals ──────────────────────────────────────────────────────
    df["sentiment"]  = sentiment_score
    df["fear_greed"] = (fear_greed - 50) / 50

    feature_cols = [
        # Price returns
        "ret_1d", "ret_3d", "ret_7d", "ret_14d", "ret_30d",
        # Oscillators
        "rsi_norm", "rsi_7", "rsi_21", "rsi_diff", "stoch_rsi",
        # MACD
        "macd_hist", "macd_signal_cross",
        # Bands
        "bb_position", "bb_width",
        # Trend
        "sma_ratio", "ema_ratio", "price_vs_sma20", "price_vs_sma50",
        # Volume
        "vol_change_1d", "vol_change_7d", "vol_vs_avg20",
        # Volatility
        "atr_norm", "vol_regime",
        # Price position
        "high_14d_pct", "low_14d_pct", "high_30d_pct", "low_30d_pct",
        # Momentum
        "momentum_10", "momentum_20", "momentum_60",
        # Candlestick
        "body_size", "upper_wick", "lower_wick", "body_dir",
        # Calendar
        "day_of_week", "month",
        # External
        "sentiment", "fear_greed",
        # VWAP
        "price_vs_vwap", "vwap_slope",
        # ADX
        "adx", "adx_trend",
        # Oscillators
        "williams_r", "cci_norm",
        # Volume delta
        "vol_delta",
        # Velocity
        "rsi_velocity", "price_accel",
        # S/R proximity
        "dist_52w_high", "dist_52w_low",
    ]

    return df[feature_cols]


def _rsi_series(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, min_periods=period).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = (100 - (100 / (1 + rs))).fillna(50)
    return (rsi - 50) / 50   # normalise -1..+1


def build_labels(df: pd.DataFrame, horizon_days: int = 1, threshold: float = 0.01) -> pd.Series:
    """
    y = 1 (BUY) if close rises > threshold in `horizon_days`.
    y = 0 (SELL) if close falls > threshold.
    Rows between thresholds are dropped (HOLD zone too noisy to learn from).
    """
    future_ret = df["Close"].pct_change(horizon_days).shift(-horizon_days)
    labels = pd.Series(index=df.index, dtype="float64")
    labels[future_ret > threshold] = 1.0
    labels[future_ret < -threshold] = 0.0
    # NaN rows (HOLD zone) will be dropped in trainer
    return labels


def get_live_feature_row(prices_df: pd.DataFrame, sentiment: float = 0.0,
                         fear_greed: int = 50) -> pd.DataFrame | None:
    """
    Returns a single-row feature DataFrame for real-time prediction.
    prices_df must have at least 60 rows of OHLCV data.
    """
    if prices_df is None or len(prices_df) < 60:
        return None
    feat = build_feature_matrix(prices_df, sentiment, fear_greed)
    last = feat.iloc[[-1]].copy()
    if last.isnull().values.any():
        last = last.ffill().bfill()
    return last
