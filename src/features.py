"""Data cleaning and feature engineering for the freight-rate model."""
from __future__ import annotations

import numpy as np
import pandas as pd

WEIGHT_CAP = 47_500.0
EQUIPMENT = ["Dry Van", "Flatbed", "Reefer"]
FEATURES = [
    "distance", "log_distance", "haversine", "road_factor", "route_distance",
    "equipment", "weight", "weight_missing", "weight_capped",
    "market_index", "market_index_missing", "quote_signal",
    "dow", "is_weekend", "day_of_month",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "d_lat", "d_lon", "bearing_sin", "bearing_cos",
]


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


def daily_market_index(*frames: pd.DataFrame) -> pd.Series:
    """Median market_index per calendar date, pooled over every file (features only, no labels).

    market_index is a market-wide daily level (within-day std ~0.02 vs between-day ~0.17),
    so a missing value is best filled with the same-day median from other loads.
    """
    allrows = pd.concat([f[["date", "market_index"]].assign(date=lambda x: pd.to_datetime(x["date"])) for f in frames])
    return allrows.groupby("date")["market_index"].median()


def route_distance_table(*frames: pd.DataFrame) -> pd.Series:
    """Mean recorded distance per directed (pickup, delivery) pair, pooled over all files.

    Per-load distance carries ~1.7% noise around the route mean, so the pooled route mean
    is a cleaner distance signal. Uses features only (no labels), so there is no leakage.
    """
    allrows = pd.concat([f[["pickup", "delivery", "distance"]] for f in frames])
    return allrows.groupby(["pickup", "delivery"])["distance"].mean()


def build_features(df: pd.DataFrame, daily_mi: pd.Series, route_dist: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    date = pd.to_datetime(df["date"])

    # --- weight: negative values are sign errors, 47,500 is a cap (censored) ---
    w = df["weight"].abs()
    out["weight_missing"] = w.isna().astype(int)
    out["weight_capped"] = (w >= WEIGHT_CAP).astype(int)
    out["weight"] = w  # NaN stays NaN: LightGBM routes missing values natively

    # --- market index: fill with same-day median ---
    mi = df["market_index"]
    out["market_index_missing"] = mi.isna().astype(int)
    out["market_index"] = mi.fillna(date.map(daily_mi))

    # --- distance / geography ---
    out["distance"] = df["distance"]
    out["log_distance"] = np.log(df["distance"])
    out["haversine"] = haversine(df.pickup_lat, df.pickup_lon, df.delivery_lat, df.delivery_lon)
    out["road_factor"] = df["distance"] / out["haversine"]
    idx = pd.MultiIndex.from_arrays([df["pickup"], df["delivery"]])
    out["route_distance"] = route_dist.reindex(idx).to_numpy()
    for c in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]:
        out[c] = df[c]
    out["d_lat"] = df.delivery_lat - df.pickup_lat
    out["d_lon"] = df.delivery_lon - df.pickup_lon
    ang = np.arctan2(out["d_lat"], out["d_lon"])
    out["bearing_sin"], out["bearing_cos"] = np.sin(ang), np.cos(ang)

    out["equipment"] = pd.Categorical(df["equipment"], categories=EQUIPMENT)
    out["quote_signal"] = df["quote_signal"]

    # --- calendar (no month / trend: validation months are unseen, market_index carries the time signal) ---
    out["dow"] = date.dt.dayofweek
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    out["day_of_month"] = date.dt.day
    return out[FEATURES]
