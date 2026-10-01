
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

SEED = 42
TEST_HOURS = 30 * 24
PEAK_Q = 0.90

def load_prepare(path):
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["dteday"]) + pd.to_timedelta(df["hr"], unit="h")
    df = df.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)

    # Target: next hour cnt, joined by exact datetime (safer than row shift).
    target = df[["datetime","cnt"]].copy()
    target["datetime"] = target["datetime"] - pd.Timedelta(hours=1)
    target = target.rename(columns={"cnt":"target"})
    work = df.merge(target, on="datetime", how="left")
    work["target_time"] = work["datetime"] + pd.Timedelta(hours=1)

    # Exact-time lag features.
    for h in [1,24,168]:
        lag = df[["datetime","cnt"]].copy()
        lag["datetime"] = lag["datetime"] + pd.Timedelta(hours=h)
        lag = lag.rename(columns={"cnt":f"lag_{h}"})
        work = work.merge(lag, on="datetime", how="left")

    # Past-only rolling features. shift(1) prevents current/future target leakage.
    s = df.set_index("datetime")["cnt"].sort_index()
    roll24 = s.shift(1).rolling("24h", min_periods=6).mean().rename("rolling_mean_24")
    roll168 = s.shift(1).rolling("168h", min_periods=24).mean().rename("rolling_mean_168")
    work = work.merge(roll24.reset_index(), on="datetime", how="left")
    work = work.merge(roll168.reset_index(), on="datetime", how="left")

    # Do not use casual / registered: cnt = casual + registered.
    drop_cols = ["casual","registered","instant","dteday","cnt"]
    work = work.drop(columns=[c for c in drop_cols if c in work.columns])
    work = work.dropna(subset=["target","lag_168"]).reset_index(drop=True)
    return work

def feature_columns(work):
    candidates = [
        "season","yr","mnth","hr","holiday","weekday","workingday","weathersit",
        "temp","atemp","hum","windspeed",
        "lag_1","lag_24","lag_168","rolling_mean_24","rolling_mean_168"
    ]
    return [c for c in candidates if c in work.columns]

def split_dev_test(work, test_hours=TEST_HOURS):
    test_start = work["target_time"].max() - pd.Timedelta(hours=test_hours-1)
    dev = work[work["target_time"] < test_start].copy()
    test = work[work["target_time"] >= test_start].copy()
    return dev, test

def rolling_origin_folds(dev, n_folds=3, val_hours=30*24):
    # Expanding training window, fixed chronological validation windows near end of DEV.
    end = len(dev)
    folds=[]
    for j in range(n_folds,0,-1):
        va_end = end - (j-1)*val_hours
        va_start = va_end - val_hours
        tr_end = va_start
        if tr_end <= 0: 
            continue
        folds.append((np.arange(0,tr_end), np.arange(va_start,va_end)))
    return folds

def seasonal_baseline(train, valid, k_weeks=4):
    # One baseline: mean of same weekday/hour from previous K weeks.
    hist = train[["target_time","target"]].copy().set_index("target_time")["target"]
    preds=[]
    for t in valid["target_time"]:
        vals=[]
        for k in range(1,k_weeks+1):
            tt=t-pd.Timedelta(hours=168*k)
            if tt in hist.index:
                vals.append(float(hist.loc[tt]))
        if vals:
            preds.append(float(np.mean(vals)))
        else:
            preds.append(float(train["target"].mean()))
    return np.array(preds)

def metrics(y, pred, peak_threshold):
    y=np.asarray(y, dtype=float); pred=np.asarray(pred, dtype=float)
    peak = y > peak_threshold
    return {
        "MAE": float(mean_absolute_error(y,pred)),
        "RMSE": float(mean_squared_error(y,pred)**0.5),
        "Peak_MAE": float(mean_absolute_error(y[peak],pred[peak])) if peak.any() else np.nan,
        "PUR": float(np.mean(pred[peak] < y[peak])) if peak.any() else np.nan,
        "N": int(len(y)),
        "N_peak": int(peak.sum())
    }

def leakage_report(work, dev, test, features):
    return {
        "chronological_sorted": bool(work["datetime"].is_monotonic_increasing),
        "target_after_feature_time": bool((work["target_time"] > work["datetime"]).all()),
        "casual_in_features": "casual" in features,
        "registered_in_features": "registered" in features,
        "dev_before_test": bool(dev["target_time"].max() < test["target_time"].min()),
        "test_rows": int(len(test)),
        "dev_rows": int(len(dev))
    }
