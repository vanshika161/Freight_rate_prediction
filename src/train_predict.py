"""Train the final model on ALL labelled data and write the submission files.

Outputs
  validation_predictions.csv        (load_id, predicted_rate)  -> 12,000 rows
  data/december_chart_inputs.csv    (predicted_rate filled in) -> 31 rows

Usage: python src/train_predict.py
"""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from features import build_features, daily_market_index, route_distance_table

ROOT = Path(__file__).resolve().parents[1]

BASE = dict(
    objective="regression_l1",  # conditional median of log-rate: robust to the ~1.3% corrupted labels
    n_estimators=1500, learning_rate=0.03, num_leaves=31, min_child_samples=40,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1, verbose=-1,
)
CONFIGS = [dict(), dict(num_leaves=63, min_child_samples=20)]  # two capacities x 3 seeds = 6 models
SEEDS = [0, 1, 2]


def fit_ensemble(X: pd.DataFrame, y_log: np.ndarray) -> list[lgb.LGBMRegressor]:
    return [
        lgb.LGBMRegressor(**{**BASE, **cfg, "random_state": seed}).fit(X, y_log)
        for cfg in CONFIGS for seed in SEEDS
    ]


def predict(models, X: pd.DataFrame) -> np.ndarray:
    return np.exp(np.mean([m.predict(X) for m in models], axis=0))  # average in log space


def december_frame(tr: pd.DataFrame, va: pd.DataFrame, dec: pd.DataFrame) -> pd.DataFrame:
    """Build full model inputs for the fixed December scenario.

    The chart file only has pickup/delivery/distance/equipment/weight/date. The model also needs
    city coordinates (looked up from the data), and the two market features, which we set to the
    *observed* per-day median market_index / quote_signal from validation.csv (features, not labels).
    """
    coords_p = pd.concat([tr, va]).groupby("pickup")[["pickup_lat", "pickup_lon"]].first()
    coords_d = pd.concat([tr, va]).groupby("delivery")[["delivery_lat", "delivery_lon"]].first()
    d = dec.drop(columns="predicted_rate").copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.join(coords_p, on="pickup").join(coords_d, on="delivery")
    day = va.groupby("date")[["market_index", "quote_signal"]].median()
    d["market_index"] = d["date"].map(day["market_index"])
    d["quote_signal"] = d["date"].map(day["quote_signal"])
    return d


def main():
    tr = pd.read_csv(ROOT / "data/train_test.csv", parse_dates=["date"])
    va = pd.read_csv(ROOT / "data/validation.csv", parse_dates=["date"])
    dec = pd.read_csv(ROOT / "data/december_chart_inputs.csv")

    daily_mi, route_dist = daily_market_index(tr, va), route_distance_table(tr, va)
    X_tr = build_features(tr, daily_mi, route_dist)
    models = fit_ensemble(X_tr, np.log(tr["posted_rate"].to_numpy()))

    # ---- 12,000 validation loads ----
    X_va = build_features(va, daily_mi, route_dist)
    pred = predict(models, X_va)
    tmpl = pd.read_csv(ROOT / "data/validation_predictions_template.csv")
    out = tmpl[["load_id"]].merge(pd.DataFrame({"load_id": va["load_id"], "predicted_rate": pred.round(2)}),
                                  on="load_id", how="left")
    assert out["predicted_rate"].notna().all() and len(out) == 12_000
    out.to_csv(ROOT / "validation_predictions.csv", index=False)

    # ---- fixed December scenario ----
    d = december_frame(tr, va, dec)
    X_dec = build_features(d, daily_mi, route_dist)
    dec_out = dec.copy()
    dec_out["predicted_rate"] = predict(models, X_dec).round(2)
    dec_out.to_csv(ROOT / "data/december_chart_inputs.csv", index=False)

    print(f"validation_predictions.csv: {len(out):,} rows, range ${out.predicted_rate.min():.0f}-${out.predicted_rate.max():.0f}")
    print(dec_out[["date", "predicted_rate"]].describe())


if __name__ == "__main__":
    main()
