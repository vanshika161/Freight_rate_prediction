"""Model comparison with leakage-safe, time-aware validation.

Why not a random K-fold?  The final validation file is the *future* (Nov-Dec 2025) relative to the
labelled data (Jan-Oct 2025), and it contains 8 cities never seen in training. A random split would
leak same-day market conditions into training and over-state accuracy. We therefore use:

  1. Forward-chaining (expanding window) folds  -> "train on the past, predict the next 2 months"
        fold A: train Jan-Jun, test Jul-Aug
        fold B: train Jan-Aug, test Sep-Oct   (closest analogue of the real Nov-Dec task)
  2. Unseen-city stress test: fold B again, but every load touching 6 held-out cities is removed
     from training and the model is scored on those loads only.

Usage: python src/validate.py
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from features import build_features, daily_market_index, route_distance_table

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
SEED = 42

LGB_BASE = dict(
    n_estimators=1500, learning_rate=0.03, num_leaves=31, min_child_samples=40,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
    random_state=SEED, n_jobs=-1, verbose=-1,
)


def metrics(y, p) -> dict:
    err = p - y
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE_%": float(np.mean(np.abs(err) / y) * 100),
        "MedAPE_%": float(np.median(np.abs(err) / y) * 100),
        "R2": float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)),
    }


def flag_label_outliers(X, y_log, mask_train, threshold=0.4):
    """Flag training labels whose log-residual from a robust first-stage model is huge.

    Out-of-fold residuals (5 folds, random within the *training window only*) so a corrupt label
    can't explain itself away. Huber loss keeps the first stage itself robust.
    """
    from sklearn.model_selection import KFold

    idx = np.where(mask_train)[0]
    resid = np.zeros(len(idx))
    params = {**LGB_BASE, "n_estimators": 400, "learning_rate": 0.08, "objective": "huber", "alpha": 0.3}
    for tr, te in KFold(5, shuffle=True, random_state=SEED).split(idx):
        m = lgb.LGBMRegressor(**params).fit(X.iloc[idx[tr]], y_log[idx[tr]])
        resid[te] = y_log[idx[te]] - m.predict(X.iloc[idx[te]])
    keep = np.abs(resid) <= threshold
    return idx[keep], int((~keep).sum())


def fit_predict(name, X, y_log, tr_idx, te_idx):
    Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
    ytr = y_log[tr_idx]
    if name == "baseline_median_rate_per_mile":
        rpm = np.exp(ytr) / X["distance"].iloc[tr_idx].to_numpy()
        return np.median(rpm) * X["distance"].iloc[te_idx].to_numpy()
    if name == "ridge_log_linear":
        cols = ["log_distance", "market_index", "quote_signal"]
        def mat(d):
            M = d[cols].copy()
            M["w"] = d["weight"].fillna(d["weight"].median()) / 1e4
            M["sq"] = (d["quote_signal"] - 2.05) ** 2
            return pd.concat([M, pd.get_dummies(d["equipment"], dtype=float)], axis=1)
        m = Ridge(alpha=1.0).fit(mat(Xtr), ytr)
        return np.exp(m.predict(mat(Xte)))
    kind = {
        "lgbm_l2_raw_labels": dict(objective="regression"),
        "lgbm_huber_raw_labels": dict(objective="huber", alpha=0.3),
        "lgbm_l1_raw_labels": dict(objective="regression_l1"),
    }[name]
    m = lgb.LGBMRegressor(**{**LGB_BASE, **kind}).fit(Xtr, ytr)
    return np.exp(m.predict(Xte))


MODELS = [
    "baseline_median_rate_per_mile", "ridge_log_linear",
    "lgbm_l2_raw_labels", "lgbm_huber_raw_labels", "lgbm_l1_raw_labels", "lgbm_l2_cleaned_labels",
]


def run_fold(label, X, y_log, y, train_mask, test_mask):
    rows = []
    # robust "clean" evaluation subset: rate-per-mile inside a plausible band
    rpm = y / X["distance"].to_numpy()
    clean = (rpm > 0.9) & (rpm < 4.0)
    for name in MODELS:
        if name == "lgbm_l2_cleaned_labels":
            keep, dropped = flag_label_outliers(X, y_log, train_mask)
            tr_idx = keep
            pred = fit_predict("lgbm_l2_raw_labels", X, y_log, tr_idx, np.where(test_mask)[0])
        else:
            tr_idx = np.where(train_mask)[0]
            pred = fit_predict(name, X, y_log, tr_idx, np.where(test_mask)[0])
            dropped = 0
        yt = y[test_mask]
        raw = metrics(yt, pred)
        cl = metrics(yt[clean[test_mask]], pred[clean[test_mask]])
        rows.append({"fold": label, "model": name, "dropped_train_labels": dropped,
                     **{f"raw_{k}": v for k, v in raw.items()}, **{f"clean_{k}": v for k, v in cl.items()}})
        print(f"[{label}] {name:32s} raw MAE={raw['MAE']:8.1f} RMSE={raw['RMSE']:8.1f} MAPE={raw['MAPE_%']:5.2f}% | "
              f"clean MAE={cl['MAE']:7.1f} MAPE={cl['MAPE_%']:5.2f}% R2={cl['R2']:.4f}")
    return rows


def main():
    OUT.mkdir(exist_ok=True)
    tr = pd.read_csv(ROOT / "data/train_test.csv", parse_dates=["date"])
    va = pd.read_csv(ROOT / "data/validation.csv", parse_dates=["date"])
    mi, rd = daily_market_index(tr, va), route_distance_table(tr, va)
    X = build_features(tr, mi, rd).reset_index(drop=True)
    y = tr["posted_rate"].to_numpy()
    y_log = np.log(y)
    month = tr["date"].dt.month.to_numpy()

    results = []
    results += run_fold("A: train Jan-Jun -> test Jul-Aug", X, y_log, y, month <= 6, (month >= 7) & (month <= 8))
    results += run_fold("B: train Jan-Aug -> test Sep-Oct", X, y_log, y, month <= 8, month >= 9)

    # unseen-city stress test (fold B window)
    rng = np.random.default_rng(SEED)
    cities = np.array(sorted(set(tr.pickup) | set(tr.delivery)))
    held = set(rng.choice(cities, 6, replace=False))
    touches = (tr.pickup.isin(held) | tr.delivery.isin(held)).to_numpy()
    results += run_fold(f"C: unseen cities {sorted(held)}", X, y_log, y,
                        (month <= 8) & ~touches, (month >= 9) & touches)

    pd.DataFrame(results).to_csv(OUT / "validation_metrics.csv", index=False)
    (OUT / "validation_metrics.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
