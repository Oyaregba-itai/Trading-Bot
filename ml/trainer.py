"""
Training pipeline:
  1. Fetch historical OHLCV data (yfinance for stocks/forex, CoinGecko for crypto)
  2. Build feature matrix + labels
  3. Train TradingModel
  4. Evaluate with time-series CV
  5. Save model + register in DB
"""
import pandas as pd
import numpy as np
from ml.features import build_feature_matrix, build_labels
from ml.model import TradingModel
from database import save_model_meta


# ── Data fetchers ─────────────────────────────────────────────────────────────

def _fetch_yfinance_crypto(symbol: str) -> pd.DataFrame | None:
    """
    Fetch crypto OHLCV via yfinance using SYMBOL-USD ticker.
    This is the PRIMARY crypto source — no rate limits, up to 10 years of data.
    """
    import yfinance as yf
    sym = symbol.upper()
    for ticker in [f"{sym}-USD", f"{sym}USD=X"]:
        try:
            df = yf.Ticker(ticker).history(period="max")
            if df is not None and not df.empty and len(df) >= 80:
                return df[["Open", "High", "Low", "Close", "Volume"]]
        except Exception:
            continue
    return None


def _fetch_coingecko_ohlcv(symbol: str, days: int = 365) -> pd.DataFrame | None:
    """CoinGecko OHLC — used as FALLBACK only (rate-limited to ~30 req/min on free tier)."""
    import time, requests
    try:
        from config import COINGECKO_BASE, CRYPTO_IDS
        coin_id = CRYPTO_IDS.get(symbol.upper())
        if not coin_id:
            r = requests.get(f"{COINGECKO_BASE}/search", params={"query": symbol}, timeout=10)
            coins = r.json().get("coins", [])
            if not coins:
                return None
            coin_id = coins[0]["id"]

        time.sleep(1.5)   # Respect free-tier rate limit
        params = {"vs_currency": "usd", "days": min(days, 365), "interval": "daily"}
        r = requests.get(f"{COINGECKO_BASE}/coins/{coin_id}/ohlc", params=params, timeout=15)
        if r.status_code == 429:   # Rate limited
            return None
        r.raise_for_status()
        data = r.json()
        if not data or len(data) < 80:
            return None
        df = pd.DataFrame(data, columns=["ts", "Open", "High", "Low", "Close"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        df = df.set_index("ts")

        time.sleep(1.5)
        r2 = requests.get(f"{COINGECKO_BASE}/coins/{coin_id}/market_chart",
                          params={"vs_currency": "usd", "days": min(days, 365), "interval": "daily"},
                          timeout=15)
        if r2.status_code != 429:
            chart = r2.json()
            vol_df = pd.DataFrame(chart.get("total_volumes", []), columns=["ts", "Volume"])
            vol_df["ts"] = pd.to_datetime(vol_df["ts"], unit="ms")
            vol_df = vol_df.set_index("ts")
            df = df.join(vol_df, how="left")
        df["Volume"] = df.get("Volume", pd.Series(0, index=df.index)).fillna(0)
        return df
    except Exception:
        return None


def _fetch_yfinance_ohlcv(symbol: str, period: str = "5y") -> pd.DataFrame | None:
    """Stocks, ETFs, forex, commodities via yfinance."""
    import yfinance as yf
    try:
        from config import COMMODITY_SYMBOLS
        sym = symbol.upper()
        ticker_sym = COMMODITY_SYMBOLS.get(sym, sym)
        # Forex: EURUSD → EURUSD=X
        if len(sym) == 6 and sym.isalpha() and ticker_sym == sym:
            ticker_sym = sym + "=X"
        df = yf.Ticker(ticker_sym).history(period=period)
        if df is None or df.empty:
            return None
        return df[["Open", "High", "Low", "Close", "Volume"]]
    except Exception:
        return None


def _fetch_hourly_ohlcv(symbol: str) -> pd.DataFrame | None:
    """Fetch 60 days of 1-hour candles for stocks/forex/commodities."""
    import yfinance as yf
    try:
        from config import COMMODITY_SYMBOLS
        sym = symbol.upper()
        ticker_sym = COMMODITY_SYMBOLS.get(sym, sym)
        if len(sym) == 6 and sym.isalpha() and ticker_sym == sym:
            ticker_sym = sym + "=X"
        df = yf.Ticker(ticker_sym).history(period="60d", interval="1h")
        if df is None or df.empty or len(df) < 50:
            return None
        return df[["Open", "High", "Low", "Close", "Volume"]]
    except Exception:
        return None


def _fetch_hourly_crypto(symbol: str) -> pd.DataFrame | None:
    """Fetch 60 days of 1-hour candles for crypto via yfinance."""
    import yfinance as yf
    sym = symbol.upper()
    for ticker in [f"{sym}-USD", f"{sym}USD=X"]:
        try:
            df = yf.Ticker(ticker).history(period="60d", interval="1h")
            if df is not None and not df.empty and len(df) >= 50:
                return df[["Open", "High", "Low", "Close", "Volume"]]
        except Exception:
            continue
    return None


def _fetch_5min_ohlcv(symbol: str) -> pd.DataFrame | None:
    """Fetch 60 days of 5-minute candles for stocks/forex/commodities."""
    import yfinance as yf
    try:
        from config import COMMODITY_SYMBOLS
        sym = symbol.upper()
        ticker_sym = COMMODITY_SYMBOLS.get(sym, sym)
        if len(sym) == 6 and sym.isalpha() and ticker_sym == sym:
            ticker_sym = sym + "=X"
        df = yf.Ticker(ticker_sym).history(period="60d", interval="5m")
        if df is None or df.empty or len(df) < 50:
            return None
        return df[["Open", "High", "Low", "Close", "Volume"]]
    except Exception:
        return None


def _fetch_5min_crypto(symbol: str) -> pd.DataFrame | None:
    """Fetch 60 days of 5-minute candles for crypto."""
    import yfinance as yf
    sym = symbol.upper()
    for ticker in [f"{sym}-USD"]:
        try:
            df = yf.Ticker(ticker).history(period="60d", interval="5m")
            if df is not None and not df.empty and len(df) >= 50:
                return df[["Open", "High", "Low", "Close", "Volume"]]
        except Exception:
            continue
    return None


def fetch_training_data(symbol: str, timeframe: str = "1h", days: int = 500) -> pd.DataFrame | None:
    """
    Fetch OHLCV training data for any symbol and timeframe.
      5m  — 5-minute candles (60 days max from yfinance)
      1h  — 1-hour candles  (60 days)
      1d  — daily candles   (5 years)
    """
    from config import CRYPTO_IDS
    sym = symbol.upper()

    if timeframe == "5m":
        if sym in CRYPTO_IDS:
            return _fetch_5min_crypto(sym)
        return _fetch_5min_ohlcv(sym)

    if timeframe == "1h":
        if sym in CRYPTO_IDS:
            df = _fetch_hourly_crypto(sym)
            if df is not None and len(df) >= 50:
                return df
            df = _fetch_yfinance_crypto(sym)
            if df is not None and len(df) >= 80:
                return df
            return _fetch_coingecko_ohlcv(sym, days)
        df = _fetch_hourly_ohlcv(sym)
        if df is not None and len(df) >= 50:
            return df
        return _fetch_yfinance_ohlcv(sym, "5y")

    # "1d" — daily candles
    if sym in CRYPTO_IDS:
        df = _fetch_yfinance_crypto(sym)
        if df is not None and len(df) >= 80:
            return df
        return _fetch_coingecko_ohlcv(sym, days)
    return _fetch_yfinance_ohlcv(sym, "5y")


# ── Main training function ────────────────────────────────────────────────────

def train_symbol(symbol: str, days: int = 500,
                 timeframe: str = "1h",
                 horizon_days: int = None,
                 threshold: float = 0.01,
                 progress_callback=None) -> dict:
    """
    Full training pipeline for one symbol.
    progress_callback(str) is called with status updates.

    Returns a metrics dict with keys:
        symbol, accuracy, precision, recall, f1, n_samples, n_features, model_path, error
    """
    # Default prediction horizon per timeframe
    if horizon_days is None:
        horizon_days = {"5m": 3, "1h": 4, "1d": 3}.get(timeframe, 4)

    def emit(msg):
        if progress_callback:
            progress_callback(msg)

    emit(f"[{timeframe}] Fetching OHLCV data for {symbol}…")
    df = fetch_training_data(symbol, timeframe, days)
    min_candles = 50 if timeframe in ("5m", "1h") else 80
    if df is None or len(df) < min_candles:
        return {"symbol": symbol, "error": f"Not enough historical data for {symbol} ({timeframe})."}

    emit(f"Got {len(df)} candles. Engineering feature rows…")

    # ── Adaptive threshold ────────────────────────────────────────────────────
    # Use the 25th-percentile of absolute daily returns as the signal threshold.
    # This self-adjusts per asset class:
    #   EURUSD (~0.3% daily)  → threshold ≈ 0.15%
    #   BTC    (~3.0% daily)  → threshold ≈ 1.5%
    #   AAPL   (~1.2% daily)  → threshold ≈ 0.6%
    # Guarantees ~75% of samples are labeled (top/bottom 37.5% each direction).
    abs_returns = df["Close"].pct_change().abs().dropna()
    adaptive_thr = float(abs_returns.quantile(0.25))
    adaptive_thr = max(0.0005, adaptive_thr)   # floor at 0.05%
    effective_threshold = min(threshold, adaptive_thr)   # never exceed caller's cap

    emit(f"Signal threshold: {effective_threshold*100:.3f}% (auto-calibrated for {symbol})")

    # Build features
    features = build_feature_matrix(df, sentiment_score=0.0, fear_greed=50)
    labels = build_labels(df, horizon_days=horizon_days, threshold=effective_threshold)

    # Align and drop NaN / HOLD rows
    combined = features.join(labels.rename("label"))
    combined = combined.dropna()

    X = combined.drop(columns=["label"])
    y = combined["label"].astype(int)

    if len(X) < 50:
        # Last resort: pure binary (all up/down, no HOLD zone) — never 0 samples
        emit("Low sample count — switching to pure directional labels…")
        pure_labels = (df["Close"].pct_change(horizon_days).shift(-horizon_days) > 0).astype(float)
        pure_labels.iloc[-horizon_days:] = np.nan
        combined2 = features.join(pure_labels.rename("label")).dropna()
        X = combined2.drop(columns=["label"])
        y = combined2["label"].astype(int)
        if len(X) < 50:
            return {"symbol": symbol, "error": f"Only {len(X)} usable samples after cleaning (need ≥50)."}

    emit(f"Training on {len(X)} samples with {X.shape[1]} features…")

    model = TradingModel(symbol, timeframe)
    try:
        metrics = model.train(X, y)
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}

    emit(f"Saving model…")
    model.save()

    save_model_meta(
        symbol,
        metrics["accuracy"], metrics["precision"],
        metrics["recall"], metrics["f1"],
        metrics["n_samples"], model.path,
        timeframe=timeframe,
    )

    metrics["symbol"]    = symbol
    metrics["timeframe"] = timeframe
    metrics["n_features"] = X.shape[1]
    metrics["model_path"] = model.path
    emit(f"[{timeframe}] Done! Accuracy: {metrics['accuracy']*100:.1f}%")
    return metrics


def train_multiple(symbols: list[str], timeframes: list[str] | None = None,
                   progress_callback=None) -> list[dict]:
    if timeframes is None:
        timeframes = ["5m", "1h", "1d"]
    results = []
    for sym in symbols:
        for tf in timeframes:
            r = train_symbol(sym, timeframe=tf, progress_callback=progress_callback)
            results.append(r)
    return results


# ── Live prediction ───────────────────────────────────────────────────────────

def predict_symbol(symbol: str, sentiment_score: float = 0.0, fear_greed: int = 50,
                   timeframe: str = "1h") -> dict:
    """
    Load trained model, fetch recent data, compute live prediction.
    Returns: {symbol, signal, confidence, price, sentiment, fear_greed, timeframe, error}
    """
    model = TradingModel.load_for(symbol, timeframe)
    if model is None:
        return {"symbol": symbol, "error": f"No trained model for {symbol} ({timeframe})."}

    df = fetch_training_data(symbol, timeframe)
    if df is None or len(df) < 30:
        return {"symbol": symbol, "error": "Not enough recent data for prediction."}

    from ml.features import get_live_feature_row
    X_live = get_live_feature_row(df, sentiment_score, fear_greed)
    if X_live is None:
        return {"symbol": symbol, "error": "Could not build feature row."}

    # Override sentiment and fear_greed in the feature row
    if "sentiment" in X_live.columns:
        X_live["sentiment"] = sentiment_score
    if "fear_greed" in X_live.columns:
        X_live["fear_greed"] = (fear_greed - 50) / 50

    try:
        signal, confidence = model.predict(X_live)
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}

    price = float(df["Close"].iloc[-1])
    signal_label = "BUY" if signal == 1 else "SELL"

    from database import log_prediction
    log_prediction(symbol, signal_label, confidence, price, sentiment_score)

    return {
        "symbol":    symbol,
        "signal":    signal_label,
        "confidence": confidence,
        "price":     price,
        "sentiment": sentiment_score,
        "fear_greed": fear_greed,
        "timeframe": timeframe,
    }
