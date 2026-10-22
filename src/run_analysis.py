"""Directional forecasting for BTC-USD on real daily price data.

Task: predict whether the close 7 days from now is higher than today's
close. Everything is out of sample by time: the split date is fixed
before fitting, features only use past prices and volume, and the test
period is never touched during training.

Models: logistic regression and gradient boosting, plus a soft voting
ensemble that averages their probabilities. That ensemble is real: both
models feed it. Baselines are always predict up and a simple momentum
rule. There is no sentiment data in this version at all.
"""
import json
import pathlib
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
HORIZON = 7


def build_features(df):
    df = df.copy()
    close = df["close"]
    df["ret_1"] = close.pct_change(1)
    df["ret_7"] = close.pct_change(7)
    df["ret_14"] = close.pct_change(14)
    df["ret_30"] = close.pct_change(30)
    df["vol_14"] = df["ret_1"].rolling(14).std()
    df["vol_30"] = df["ret_1"].rolling(30).std()
    df["ma_ratio_20"] = close / close.rolling(20).mean() - 1
    df["ma_ratio_50"] = close / close.rolling(50).mean() - 1
    df["volume_change_7"] = df["volume"].pct_change(7)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    df["target_up_7d"] = (close.shift(-HORIZON) > close).astype(int)
    df["forward_return_7d"] = close.shift(-HORIZON) / close - 1
    return df


FEATURES = ["ret_1", "ret_7", "ret_14", "ret_30", "vol_14", "vol_30",
            "ma_ratio_20", "ma_ratio_50", "volume_change_7", "rsi_14"]


def scores(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_up": float(f1_score(y_true, y_pred)),
        "n": int(len(y_true)),
        "actual_up_rate": float(np.mean(y_true)),
    }


def main():
    OUT.mkdir(exist_ok=True)
    started = time.time()
    raw = pd.read_csv(ROOT / "data" / "btc_usd_daily.csv", parse_dates=["date"])
    n_raw = len(raw)
    df = build_features(raw).dropna().reset_index(drop=True)
    split_idx = int(len(df) * 0.7)
    split_date = str(df["date"].iloc[split_idx].date())
    train, test = df.iloc[:split_idx], df.iloc[split_idx:]

    logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    gbm = GradientBoostingClassifier(random_state=42, n_estimators=200, max_depth=3)
    logreg.fit(train[FEATURES], train["target_up_7d"])
    gbm.fit(train[FEATURES], train["target_up_7d"])

    p_lr = logreg.predict_proba(test[FEATURES])[:, 1]
    p_gbm = gbm.predict_proba(test[FEATURES])[:, 1]
    p_ens = (p_lr + p_gbm) / 2
    y = test["target_up_7d"].to_numpy()
    preds = {
        "logistic_regression": (p_lr >= 0.5).astype(int),
        "gradient_boosting": (p_gbm >= 0.5).astype(int),
        "ensemble_soft_vote": (p_ens >= 0.5).astype(int),
        "baseline_always_up": np.ones_like(y),
        "baseline_momentum_ret7": (test["ret_7"] > 0).astype(int),
    }
    results = {name: scores(y, p) for name, p in preds.items()}
    # One real significance check: is the ensemble better than a coin flip?
    correct = int((preds["ensemble_soft_vote"] == y).sum())
    binom = stats.binomtest(correct, len(y), 0.5, alternative="greater")
    results["ensemble_soft_vote"]["binom_p_vs_0.5"] = float(binom.pvalue)

    out_preds = pd.DataFrame({
        "date": test["date"].dt.strftime("%Y-%m-%d"),
        "close": test["close"],
        "actual_up_7d": y,
        "forward_return_7d": test["forward_return_7d"].round(4),
        "prob_up_logreg": p_lr.round(3),
        "prob_up_gbm": p_gbm.round(3),
        "prob_up_ensemble": p_ens.round(3),
        "pred_ensemble": preds["ensemble_soft_vote"],
    })
    out_preds.to_csv(OUT / "predictions.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(raw["date"], raw["close"], lw=0.8)
    ax.axvline(pd.Timestamp(split_date), color="red", lw=1)
    ax.set_ylabel("BTC-USD close")
    ax.set_title("Train / test split marked in red")
    fig.tight_layout()
    fig.savefig(OUT / "price_and_split.png", dpi=110)
    plt.close(fig)

    bh = float((1 + test["forward_return_7d"]).prod() ** (1 / max(len(test) // HORIZON, 1)) - 1)
    metrics = {
        "dataset": {
            "name": "BTC-USD daily candles, Coinbase Exchange public API",
            "source": "https://api.exchange.coinbase.com/products/BTC-USD/candles",
            "snapshot_file": "data/btc_usd_daily.csv",
            "candles_raw": int(n_raw),
            "rows_after_features": int(len(df)),
            "date_min": str(raw["date"].min().date()),
            "date_max": str(raw["date"].max().date()),
            "note": "Real exchange data. No sentiment data is used in this version.",
        },
        "task": f"predict whether close in {HORIZON} days is above today's close",
        "split": f"first 70% of usable rows train, last 30% test, split date {split_date}",
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "features": FEATURES,
        "results": results,
        "context": {
            "test_period_buy_and_hold_note": "Buy and hold return over the test window is context only, not a model score. It is in run_log.txt.",
            "buy_and_hold_compound_per_7d_block": bh,
        },
        "runtime_seconds": round(time.time() - started, 1),
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (OUT / "run_log.txt").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    print("split", split_date, "train", len(train), "test", len(test))


if __name__ == "__main__":
    main()
