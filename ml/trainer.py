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
    """
    Fetch up to 2 years of 1-hour candles for stocks/forex/commodities.
    yfinance supports hourly up to 730 days when using explicit date ranges.
    Falls back to 60-day period string if date-range fetch fails.
    """
    import yfinance as yf
    from datetime import datetime, timedelta
    try:
        from config import COMMODITY_SYMBOLS
        sym = symbol.upper()
        ticker_sym = COMMODITY_SYMBOLS.get(sym, sym)
        if len(sym) == 6 and sym.isalpha() and ticker_sym == sym:
            ticker_sym = sym + "=X"

        end   = datetime.utcnow()
        start = end - timedelta(days=728)   # ~2 years, yfinance max for 1h
        df = yf.Ticker(ticker_sym).history(start=start.strftime("%Y-%m-%d"),
                                           end=end.strftime("%Y-%m-%d"),
                                           interval="1h")
        if df is not None and not df.empty and len(df) >= 50:
            return df[["Open", "High", "Low", "Close", "Volume"]]
        # Fallback
        df = yf.Ticker(ticker_sym).history(period="60d", interval="1h")
        if df is not None and not df.empty and len(df) >= 50:
            return df[["Open", "High", "Low", "Close", "Volume"]]
        return None
    except Exception:
        return None


def _fetch_hourly_crypto(symbol: str) -> pd.DataFrame | None:
    """Fetch up to 2 years of 1-hour candles for crypto via yfinance."""
    import yfinance as yf
    from datetime import datetime, timedelta
    sym = symbol.upper()
    end   = datetime.utcnow()
    start = end - timedelta(days=728)
    for ticker in [f"{sym}-USD", f"{sym}USD=X"]:
        try:
            df = yf.Ticker(ticker).history(start=start.strftime("%Y-%m-%d"),
                                           end=end.strftime("%Y-%m-%d"),
                                           interval="1h")
            if df is not None and not df.empty and len(df) >= 50:
                return df[["Open", "High", "Low", "Close", "Volume"]]
        except Exception:
            continue
    # Fallback to 60d
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


# ── P&L-aware label builder (RL-inspired) ────────────────────────────────────

def build_pnl_labels(df: pd.DataFrame, sl_pct: float, tp_pct: float,
                     timeframe: str = "1h") -> pd.Series:
    """
    Labels based on whether a trade opened at each candle would hit TP or SL first.
    This directly mirrors the bot's actual trading logic, unlike horizon-based labels.

      label = 1 (BUY)  → TP hit before SL within lookahead window
      label = 0 (SELL) → SL hit before TP
      label = NaN      → neither hit (HOLD — dropped during training)

    lookahead candles per timeframe:
      5m → 48 candles (4 hours)
      1h → 48 candles (2 days)
      1d → 10 candles (2 weeks)
    """
    lookahead = {"5m": 48, "1h": 48, "1d": 10}.get(timeframe, 48)

    close_arr = df["Close"].values
    high_arr  = df["High"].values
    low_arr   = df["Low"].values
    n = len(df)

    labels = np.full(n, np.nan)

    for i in range(n - 1):
        entry = close_arr[i]
        if entry <= 0:
            continue
        tp_price = entry * (1 + tp_pct)
        sl_price = entry * (1 - sl_pct)

        end = min(i + 1 + lookahead, n)
        for j in range(i + 1, end):
            if high_arr[j] >= tp_price:
                labels[i] = 1.0   # TP hit first → good BUY
                break
            if low_arr[j] <= sl_price:
                labels[i] = 0.0   # SL hit first → bad trade
                break

    return pd.Series(labels, index=df.index)


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

    # Use P&L-aware labels when we have enough data (RL-inspired: directly optimises for profit)
    # Fall back to horizon labels for short datasets where lookahead would eat too many rows
    lookahead = {"5m": 48, "1h": 48, "1d": 10}.get(timeframe, 48)
    use_pnl_labels = len(df) >= (lookahead * 3)

    if use_pnl_labels:
        # Default SL/TP used for labeling — 2:1 R/R ratio
        label_sl = {"5m": 0.003, "1h": 0.015, "1d": 0.04}.get(timeframe, 0.015)
        label_tp = label_sl * 2.0
        labels = build_pnl_labels(df, sl_pct=label_sl, tp_pct=label_tp, timeframe=timeframe)
        emit(f"Using P&L-aware labels (SL={label_sl*100:.1f}%, TP={label_tp*100:.1f}%)")
    else:
        labels = build_labels(df, horizon_days=horizon_days, threshold=effective_threshold)
        emit(f"Using horizon labels (dataset too small for P&L labels)")

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


def walk_forward_backtest(symbol: str, timeframe: str = "1h",
                          n_splits: int = 4) -> dict:
    """
    Walk-forward validation: train on rolling windows, test on unseen data.
    Avoids overfitting by ensuring test data was never seen during training.

    Returns {symbol, timeframe, avg_accuracy, avg_precision, avg_recall,
             window_results, passed} where passed=True means avg_accuracy > 0.52
    and avg_recall > 0.10 (model generalises beyond random guessing).
    """
    from ml.features import build_feature_matrix, build_labels
    import numpy as np

    df = fetch_training_data(symbol, timeframe)
    if df is None or len(df) < 150:
        return {"symbol": symbol, "timeframe": timeframe,
                "error": "Not enough data for walk-forward test (need ≥150 candles)"}

    horizon = {"5m": 3, "1h": 4, "1d": 3}.get(timeframe, 4)
    abs_returns = df["Close"].pct_change().abs().dropna()
    thr = max(0.0005, float(abs_returns.quantile(0.25)))

    features = build_feature_matrix(df, sentiment_score=0.0, fear_greed=50)
    labels   = build_labels(df, horizon_days=horizon, threshold=thr)
    combined = features.join(labels.rename("label")).dropna()
    combined = combined[combined["label"] != 0]   # drop HOLD rows

    if len(combined) < 100:
        return {"symbol": symbol, "timeframe": timeframe,
                "error": f"Only {len(combined)} usable samples for walk-forward"}

    X = combined.drop(columns=["label"]).values
    y = combined["label"].astype(int).values

    window_size = len(X) // (n_splits + 1)
    results = []

    for i in range(n_splits):
        train_end  = window_size * (i + 1)
        test_start = train_end
        test_end   = min(test_start + window_size, len(X))

        X_train, y_train = X[:train_end], y[:train_end]
        X_test,  y_test  = X[test_start:test_end], y[test_start:test_end]

        if len(X_train) < 50 or len(X_test) < 20:
            continue

        from ml.model import TradingModel
        m = TradingModel(symbol, timeframe)
        try:
            import pandas as pd
            feat_names = combined.drop(columns=["label"]).columns.tolist()
            metrics = m.train(
                pd.DataFrame(X_train, columns=feat_names),
                pd.Series(y_train)
            )
            # Evaluate on held-out window
            from sklearn.metrics import accuracy_score, precision_score, recall_score
            X_test_df = pd.DataFrame(X_test, columns=feat_names)
            preds = m.model.predict(X_test_df)
            window_res = {
                "window": i + 1,
                "train_n": len(X_train),
                "test_n":  len(X_test),
                "accuracy":  float(accuracy_score(y_test, preds)),
                "precision": float(precision_score(y_test, preds, zero_division=0)),
                "recall":    float(recall_score(y_test, preds, zero_division=0)),
            }
            results.append(window_res)
        except Exception as e:
            results.append({"window": i + 1, "error": str(e)})

    if not results or all("error" in r for r in results):
        return {"symbol": symbol, "timeframe": timeframe, "error": "all windows failed"}

    valid = [r for r in results if "error" not in r]
    avg_acc = float(np.mean([r["accuracy"]  for r in valid]))
    avg_pre = float(np.mean([r["precision"] for r in valid]))
    avg_rec = float(np.mean([r["recall"]    for r in valid]))

    return {
        "symbol":        symbol,
        "timeframe":     timeframe,
        "avg_accuracy":  round(avg_acc, 4),
        "avg_precision": round(avg_pre, 4),
        "avg_recall":    round(avg_rec, 4),
        "window_results": results,
        "passed":        avg_acc > 0.52 and avg_rec > 0.10,
    }


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
