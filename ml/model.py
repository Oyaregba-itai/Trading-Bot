"""
ML model wrapper — XGBoost with:
  - Automatic class-imbalance correction (scale_pos_weight)
  - Adaptive CV folds (fewer folds for small datasets)
  - Ensemble vote: XGBoost + RandomForest + ExtraTrees
  - Saved per symbol under /models/
"""
import os
import joblib
import numpy as np
import pandas as pd

# Use Railway persistent volume if available, otherwise local models/ folder
_DATA_DIR  = "/app/data" if os.path.isdir("/app/data") else os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(_DATA_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def _class_weight_ratio(y: pd.Series) -> float:
    """Ratio of negative to positive samples for XGBoost scale_pos_weight."""
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    return float(n_neg / n_pos) if n_pos > 0 else 1.0


def _build_xgb(scale_pos_weight: float = 1.0):
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,   # fixes class imbalance
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )


def _build_rf(class_weight="balanced"):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1,
    )


def _build_et(class_weight="balanced"):
    from sklearn.ensemble import ExtraTreesClassifier
    return ExtraTreesClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1,
    )


def _build_mlp():
    from sklearn.neural_network import MLPClassifier
    return MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        learning_rate="adaptive",
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=42,
    )


class TradingModel:
    def __init__(self, symbol: str, timeframe: str = "1h"):
        self.symbol    = symbol.upper()
        self.timeframe = timeframe
        self.model     = None
        self.feature_names: list[str] = []
        self.path = os.path.join(MODELS_DIR, f"{self.symbol}_{timeframe}.joblib")

    def train(self, X: pd.DataFrame, y: pd.Series) -> dict:
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.ensemble import VotingClassifier

        self.feature_names = list(X.columns)
        spw = _class_weight_ratio(y)

        # Adaptive CV folds: 3 folds for small datasets, 5 for large
        n_splits = 3 if len(X) < 300 else 5
        tscv = TimeSeriesSplit(n_splits=n_splits)

        # Build ensemble (soft voting = use probabilities, more stable)
        # MLP gets lower weight — it's a different signal type, not a tiebreaker
        try:
            ensemble = VotingClassifier(
                estimators=[
                    ("xgb", _build_xgb(spw)),
                    ("rf",  _build_rf()),
                    ("et",  _build_et()),
                    ("mlp", _build_mlp()),
                ],
                voting="soft",
                weights=[2, 1, 1, 1],   # XGBoost gets 2x weight (most reliable for tabular)
                n_jobs=1,
            )
        except Exception:
            ensemble = VotingClassifier(
                estimators=[
                    ("rf",  _build_rf()),
                    ("et",  _build_et()),
                    ("mlp", _build_mlp()),
                ],
                voting="soft",
                n_jobs=1,
            )

        pipeline = Pipeline([("scaler", StandardScaler()), ("clf", ensemble)])

        accs, precs, recs, f1s = [], [], [], []
        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            # Skip fold if only 1 class present in training
            if len(y_tr.unique()) < 2:
                continue
            pipeline.fit(X_tr, y_tr)
            y_pred = pipeline.predict(X_val)
            accs.append(accuracy_score(y_val, y_pred))
            precs.append(precision_score(y_val, y_pred, zero_division=0, average="weighted"))
            recs.append(recall_score(y_val, y_pred, zero_division=0, average="weighted"))
            f1s.append(f1_score(y_val, y_pred, zero_division=0, average="weighted"))

        if not accs:
            return {"accuracy": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0, "n_samples": len(X)}

        # Final fit on all data
        pipeline.fit(X, y)
        self.model = pipeline

        return {
            "accuracy":  round(float(np.mean(accs)), 4),
            "precision": round(float(np.mean(precs)), 4),
            "recall":    round(float(np.mean(recs)), 4),
            "f1":        round(float(np.mean(f1s)), 4),
            "n_samples": len(X),
        }

    def predict(self, X: pd.DataFrame) -> tuple[int, float]:
        """Returns (signal, confidence). signal: 1=BUY, 0=SELL."""
        if self.model is None:
            raise RuntimeError(f"Model for {self.symbol} not trained yet.")
        X_aligned = X.reindex(columns=self.feature_names, fill_value=0)
        signal = int(self.model.predict(X_aligned)[0])
        proba = self.model.predict_proba(X_aligned)[0]
        confidence = round(float(max(proba)), 4)
        return signal, confidence

    def save(self):
        joblib.dump({"model": self.model, "features": self.feature_names}, self.path)

    def load(self) -> bool:
        if not os.path.exists(self.path):
            return False
        try:
            data = joblib.load(self.path)
            self.model = data["model"]
            self.feature_names = data["features"]
            return True
        except Exception:
            return False

    @classmethod
    def load_for(cls, symbol: str, timeframe: str = "1h") -> "TradingModel | None":
        m = cls(symbol, timeframe)
        return m if m.load() else None


def get_feature_importance(model: TradingModel) -> dict:
    """Return feature importances averaged across ensemble members."""
    try:
        ensemble = model.model.named_steps["clf"]
        importances = np.zeros(len(model.feature_names))
        count = 0
        for name, clf in ensemble.estimators:
            try:
                importances += clf.feature_importances_
                count += 1
            except AttributeError:
                pass
        if count > 0:
            importances /= count
        pairs = sorted(zip(model.feature_names, importances), key=lambda x: x[1], reverse=True)
        return {n: round(float(v), 4) for n, v in pairs[:12]}
    except Exception:
        return {}
