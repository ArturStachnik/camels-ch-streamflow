"""Gradient-boosted trees (XGBoost) as a comparison for the LSTM, in settings 1 and 2.

The model sees, for each basin and day, the same information the LSTM sees: the five
forcing variables over the past 365 days and the 45 static attributes. The forcing
history is summarised by hand-made features (recent values and trailing means, sums and
snow proxies over windows up to 365 days). Every window ends on the day being predicted,
so no future forcing enters. Discharge is never an input.

The loss matches the LSTM: squared error weighted by 1 / (std_b + 0.1)^2, where std_b is
the standard deviation of discharge of basin b in the training period (the NSE loss of
Kratzert et al., 2019). Hyperparameters are fixed in advance and not tuned; the number of
trees is fixed, without early stopping.

Results are written in the format of the LSTM runs, so that verification/verify_metrics.py
checks them in the same way. The metric file next to the results is computed with the
metric functions of NeuralHydrology, as for the LSTM.

Usage:
  python scripts/gbm.py --setting 1
  python scripts/gbm.py --setting 2 --fold 0      # 0..4
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[1]
FORCING = ["precipitation", "temperature_min", "temperature_mean", "temperature_max", "rel_sun_dur"]
TARGET = "discharge_spec"
PARAMS = dict(objective="reg:squarederror", tree_method="hist", max_depth=10, learning_rate=0.05,
              subsample=0.7, colsample_bytree=0.7, min_child_weight=50, reg_lambda=1.0, max_bin=256)
N_TREES = 1500
NSE_EPS = 0.1
SEED = {1: 301, 2: 311}


def features(df):
    """Feature table for one basin. All windows are trailing and include the day itself."""
    p, tmean = df.precipitation, df.temperature_mean
    out = {}
    for v in FORCING:
        for lag in (0, 1, 2):
            out[f"{v}_lag{lag}"] = df[v].shift(lag)
    for w in (3, 7, 15, 30, 60, 90, 180, 365):
        out[f"p_mean_{w}d"] = p.rolling(w, min_periods=w).mean()
    for w in (3, 7, 30, 90, 180):
        out[f"tmean_mean_{w}d"] = tmean.rolling(w, min_periods=w).mean()
    for w in (7, 30):
        out[f"sun_mean_{w}d"] = df.rel_sun_dur.rolling(w, min_periods=w).mean()
    snow = p.where(tmean < 0, 0.0)
    melt = tmean.clip(lower=0)
    for w in (30, 90, 180):
        out[f"snowfall_sum_{w}d"] = snow.rolling(w, min_periods=w).sum()
    for w in (7, 30, 90):
        out[f"degree_days_{w}d"] = melt.rolling(w, min_periods=w).sum()
    out["doy_sin"] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
    return pd.DataFrame(out, index=df.index)


def load_basin(b, attrs, statics):
    df = xr.open_dataset(ROOT / f"data/processed/time_series/{b}.nc").to_dataframe()
    X = features(df[FORCING])
    for s in statics:
        X[s] = attrs.loc[b, s]
    return X.astype("float32"), df[TARGET].astype("float32")


def stack(basins, data, start, end):
    Xs, ys, ids = [], [], []
    for b in basins:
        X, y = data[b]
        X, y = X.loc[start:end], y.loc[start:end]
        m = y.notna().values
        Xs.append(X[m]); ys.append(y[m]); ids += [b] * int(m.sum())
    return pd.concat(Xs), pd.concat(ys), np.array(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", type=int, choices=[1, 2], required=True)
    ap.add_argument("--fold", type=int)
    ap.add_argument("--nthread", type=int, default=8)
    args = ap.parse_args()
    yaml = YAML(typ="safe")
    periods = {k: [str(pd.Timestamp(d).date()) for d in v] for k, v in yaml.load((ROOT / "splits/periods.yml").read_text()).items()}
    base_cfg = yaml.load((ROOT / "configs/baseline_seed1.yml").read_text())
    statics = base_cfg["static_attributes"]
    attrs = pd.read_csv(ROOT / "data/processed/attributes/attributes.csv", dtype={"gauge_id": str}).set_index("gauge_id")

    if args.setting == 1:
        name = "gbm_setting1"
        train_file = test_file = "splits/basins_all.txt"
    else:
        name = f"gbm_setting2_fold{args.fold}"
        train_file, test_file = f"splits/setting2/fold{args.fold}_train.txt", f"splits/setting2/fold{args.fold}_test.txt"
    train_b = (ROOT / train_file).read_text().split()
    test_b = (ROOT / test_file).read_text().split()
    assert args.setting == 1 or not set(train_b) & set(test_b)

    t0 = time.time()
    data = {b: load_basin(b, attrs, statics) for b in sorted(set(train_b) | set(test_b))}
    X, y, ids = stack(train_b, data, *periods["train"])
    std = pd.Series({b: float(data[b][1].loc[periods["train"][0]:periods["train"][1]].std()) for b in train_b})
    w = 1.0 / (std.loc[ids].values + NSE_EPS) ** 2
    dtrain = xgb.DMatrix(X, label=y.values, weight=w, nthread=args.nthread)
    params = dict(PARAMS, seed=SEED[args.setting], nthread=args.nthread)
    booster = xgb.train(params, dtrain, num_boost_round=N_TREES)
    t_train = time.time() - t0

    from neuralhydrology.evaluation.metrics import calculate_metrics
    out = ROOT / "runs" / name
    res_dir = out / "test/model_epoch001"
    res_dir.mkdir(parents=True, exist_ok=True)
    results, rows = {}, []
    for b in test_b:
        Xb, yb = data[b]
        Xb, yb = Xb.loc[periods["test"][0]:periods["test"][1]], yb.loc[periods["test"][0]:periods["test"][1]]
        sim = np.clip(booster.predict(xgb.DMatrix(Xb, nthread=args.nthread)), 0, None).astype("float32")
        ds = xr.Dataset({f"{TARGET}_obs": (("date", "time_step"), yb.values[:, None]),
                         f"{TARGET}_sim": (("date", "time_step"), sim[:, None])},
                        coords={"date": Xb.index.values, "time_step": [0]})
        m = calculate_metrics(ds[f"{TARGET}_obs"].isel(time_step=0), ds[f"{TARGET}_sim"].isel(time_step=0),
                              metrics=["NSE", "KGE", "FHV", "FLV"], resolution="1D")
        results[b] = {"1D": dict(xr=ds, **m)}
        rows.append(dict(basin=b, **m))
    pickle.dump(results, open(res_dir / "test_results.p", "wb"))
    pd.DataFrame(rows).to_csv(res_dir / "test_metrics.csv", index=False)
    booster.save_model(out / "model.json")
    fmt = lambda s: pd.Timestamp(s).strftime("%d/%m/%Y")
    cfg = {"experiment_name": name, "model": "xgboost", "seed": params["seed"], "train_basin_file": train_file,
           "test_basin_file": test_file, "train_start_date": fmt(periods["train"][0]), "train_end_date": fmt(periods["train"][1]),
           "test_start_date": fmt(periods["test"][0]), "test_end_date": fmt(periods["test"][1]),
           "n_trees": N_TREES, "params": json.dumps(PARAMS), "features": json.dumps(list(X.columns)),
           "n_train_rows": int(len(y)), "train_seconds": round(t_train)}
    (out / "config.yml").write_text("".join(f"{k}: {v}\n" for k, v in cfg.items()), encoding="utf-8")
    print(f"{name}: {len(y)} training rows, {X.shape[1]} features, {t_train / 60:.1f} min; "
          f"median test NSE across {len(rows)} basins written to {res_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
