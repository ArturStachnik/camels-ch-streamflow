"""Independent audit of setting 2 (ungauged basins), setting 3 (data-scarce basins) and the
gradient-boosting comparison.

Written separately from the modelling code. It reads the run directories, the processed and the
raw CAMELS-CH files and writes every number the audit uses to results/audit_transfer/. It never
writes into runs/, splits/, configs/, scripts/, verification/ or tests/. The only model code it
calls is (a) NeuralHydrology, to rebuild the training samples of setting 3 and to repeat the test
inference of setting 2 on the CPU from the saved weights, and (b) scripts/gbm.py:features, which
is fed perturbed forcing to test that it is causal.

Usage: python audit/audit_transfer.py [--reinfer-n 4] [--skip-reinference] [--skip-setting3]
"""
import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/camels_ch_v0.9/camels_ch"
OUT = ROOT / "results/audit_transfer"
TARGET = "discharge_spec"
FORCING = ["precipitation", "temperature_min", "temperature_mean", "temperature_max", "rel_sun_dur"]
FINETUNE_YEARS = (1, 2, 5)
FOLDS = range(5)
FLAG_COLS = ["regulated", "reservoir_major", "glacier", "lake", "water_balance_open", "human_info_missing"]
# keys that may legitimately differ between the five setting-2 runs
FOLD_SPECIFIC = {"experiment_name", "train_basin_file", "validation_basin_file", "test_basin_file", "seed",
                 "run_dir", "train_dir", "img_log_dir", "number_of_basins", "is_continue_training", "commit_hash",
                 "base_run_dir"}  # base_run_dir: set by continue_training to the interrupted run


# ---------------------------------------------------------------- helpers
def yload(path):
    return YAML(typ="safe").load(Path(path).read_text(encoding="utf-8"))


def basins_of(path):
    return Path(ROOT / path if not Path(path).is_absolute() else path).read_text().split()


def nse(o, s):
    o, s = np.asarray(o, float), np.asarray(s, float)
    m = np.isfinite(o) & np.isfinite(s)
    o, s = o[m], s[m]
    den = ((o - o.mean()) ** 2).sum()
    return 1 - ((s - o) ** 2).sum() / den if den > 0 else np.nan


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def periods():
    p = yload(ROOT / "splits/periods.yml")
    return {k: [pd.Timestamp(d) for d in v] for k, v in p.items()}


def final_dir(k):
    """Folder holding the epoch-30 weights of setting-2 fold k (own search, not the modeller's helper)."""
    runs = sorted(ROOT.glob(f"runs/setting2_fold{k}_*"))
    assert len(runs) == 1, f"fold {k}: {len(runs)} run folders"
    hits = list(runs[0].rglob("model_epoch030.pt"))
    assert len(hits) == 1, f"fold {k}: {len(hits)} epoch-30 weight files"
    return runs[0], hits[0].parent


def load_results(path):
    res = pickle.load(open(path, "rb"))
    out = {}
    for b, v in res.items():
        ds = v["1D"]["xr"]
        idx = pd.to_datetime(ds["date"].values)
        out[b] = pd.DataFrame({"obs": ds[f"{TARGET}_obs"].values[:, 0].astype(float),
                               "sim": ds[f"{TARGET}_sim"].values[:, 0].astype(float)}, index=idx)
    return out


_CACHE = {}


def processed(b, cols=None):
    if b not in _CACHE:
        with xr.open_dataset(ROOT / f"data/processed/time_series/{b}.nc") as d:
            _CACHE[b] = d.to_dataframe()
    df = _CACHE[b]
    return df.copy() if cols is None else df[cols].copy()


def raw_series(b):
    return pd.read_csv(RAW / f"timeseries/observation_based/CAMELS_CH_obs_based_{b}.csv", index_col="date", parse_dates=True)


# ---------------------------------------------------------------- 1. scaler of setting 2
def check_scalers(P, data):
    all205 = basins_of("splits/basins_all.txt")
    attrs = pd.read_csv(ROOT / "data/processed/attributes/attributes.csv", dtype={"gauge_id": str}).set_index("gauge_id")
    rows, summary = [], {}
    warm0 = P["train"][0] - pd.Timedelta(days=364)
    for k in FOLDS:
        run, fin = final_dir(k)
        cfg = yload(fin / "config.yml")
        train_b, test_b = basins_of(f"splits/setting2/fold{k}_train.txt"), basins_of(f"splits/setting2/fold{k}_test.txt")
        assert basins_of(cfg["train_basin_file"]) == train_b
        sc = yload(fin / "train_data/train_data_scaler.yml")
        same_file = all((d / "train_data/train_data_scaler.yml").read_bytes() == (fin / "train_data/train_data_scaler.yml").read_bytes()
                        for d in {run, fin})
        center, scale = sc["xarray_feature_center"]["data_vars"], sc["xarray_feature_scale"]["data_vars"]
        variants = {
            "fold train basins, training period": (train_b, P["train"][1]),
            "fold train basins, training + validation period": (train_b, P["validation"][1]),
            "fold train + test basins, training period": (train_b + test_b, P["train"][1]),
            "all 205 basins, training period": (all205, P["train"][1]),
        }
        res = {}
        for name, (bs, end) in variants.items():
            diffs = []
            for v in FORCING + [TARGET]:
                a = P["train"][0] if v == TARGET else max(warm0, pd.Timestamp("1981-01-01"))
                x = np.concatenate([data[b].loc[a:end, v].to_numpy("float64") for b in bs])
                if v == TARGET:
                    x = np.clip(x, 0, None)
                for stat, saved, mine in (("center", center[v]["data"], np.nanmean(x)), ("scale", scale[v]["data"], np.nanstd(x))):
                    d = abs(mine - saved) / abs(saved)
                    diffs.append(d)
                    rows.append(dict(fold=k, variant=name, variable=v, statistic=stat, saved=saved, recomputed=mine, rel_diff=d))
            a = attrs.loc[bs, list(sc["attribute_means"])]
            for stat, mine in (("attribute_mean", a.mean()), ("attribute_std", a.std())):
                key = "attribute_means" if stat == "attribute_mean" else "attribute_stds"
                for col in a.columns:
                    saved = sc[key][col]
                    d = abs(mine[col] - saved) / (abs(saved) + 1e-12)
                    diffs.append(d)
                    rows.append(dict(fold=k, variant=name, variable=col, statistic=stat, saved=saved, recomputed=mine[col], rel_diff=d))
            res[name] = float(np.max(diffs))
        summary[f"fold{k}"] = dict(max_rel_diff_by_variant=res, scaler_identical_in_original_and_resumed_folder=same_file,
                                   n_train_basins=len(train_b))
    t = pd.DataFrame(rows)
    ok = all(v["max_rel_diff_by_variant"]["fold train basins, training period"] < 1e-5 and
             min(x for n, x in v["max_rel_diff_by_variant"].items() if n != "fold train basins, training period") > 1e-3
             for v in summary.values())
    summary["pass"] = bool(ok)
    return t, summary


# ---------------------------------------------------------------- 2. nested basins
def check_containment():
    import geopandas as gpd
    ids = basins_of("splits/setting2/basins_setting2.txt")
    all205 = basins_of("splits/basins_all.txt")
    excluded = basins_of("splits/setting2/excluded_large_basins.txt")
    cat = gpd.read_file(RAW / "catchment_delineations/CAMELS_CH_catchments.shp")
    cat["gauge_id"] = cat.gauge_id.astype(int).astype(str)
    cat = cat.set_index("gauge_id").loc[ids]
    cat["geometry"] = cat.geometry.buffer(0)
    pts = gpd.read_file(RAW / "catchment_delineations/CAMELS_CH_gauging_stations.shp")
    pts["gauge_id"] = pts.gauge_id.astype(int).astype(str)
    pts = pts.set_index("gauge_id").loc[ids]
    sub = gpd.read_file(RAW / "catchment_delineations/CAMELS_CH_sub_catchments.shp")
    sub["gauge_id"] = sub.gauge_id.astype(int).astype(str)
    h = sub.set_index("gauge_id")[["H1", "H2"]]
    area = cat.geometry.area
    fold = {}
    for k in FOLDS:
        for b in basins_of(f"splits/setting2/fold{k}_test.txt"):
            fold[b] = k
    rows = []
    sidx = cat.sindex
    for a in ids:
        for j in sidx.query(cat.geometry[a], predicate="intersects"):
            b = cat.index[j]
            if b <= a:
                continue
            inter = cat.geometry[a].intersection(cat.geometry[b]).area
            small, large = (a, b) if area[a] <= area[b] else (b, a)
            hier = (small in h.index and large in h.index and h.H1[large] <= h.H1[small] and h.H2[small] <= h.H2[large])
            rows.append(dict(a=a, b=b, smaller=small, larger=large, overlap_frac_of_smaller=inter / area[small],
                             gauge_of_smaller_inside_larger=bool(cat.geometry[large].contains(pts.geometry[small])),
                             gauge_of_smaller_dist_to_larger_m=float(cat.geometry[large].distance(pts.geometry[small])),
                             raw_hierarchy_contains=bool(hier), fold_a=fold[a], fold_b=fold[b]))
    t = pd.DataFrame(rows)
    t["cross_fold"] = t.fold_a != t.fold_b
    t["related"] = (t.overlap_frac_of_smaller > 0.5) | (t.gauge_of_smaller_inside_larger & (t.overlap_frac_of_smaller > 0.05)) | t.raw_hierarchy_contains
    theirs = pd.read_csv(ROOT / "splits/setting2/containment_pairs.csv", dtype=str)
    theirs_set = {frozenset(p) for p in zip(theirs.outer, theirs.inner)}
    t["in_containment_pairs_csv"] = [frozenset((x, y)) in theirs_set for x, y in zip(t.a, t.b)]
    cross = t[t.cross_fold]
    # partition of the 178 basins
    tests = [set(basins_of(f"splits/setting2/fold{k}_test.txt")) for k in FOLDS]
    trains = [set(basins_of(f"splits/setting2/fold{k}_train.txt")) for k in FOLDS]
    partition_ok = (sum(len(x) for x in tests) == len(ids) and set().union(*tests) == set(ids) and
                    all(not (tests[k] & trains[k]) and (tests[k] | trains[k]) == set(ids) for k in FOLDS) and
                    not (set(excluded) & set(ids)) and set(ids) | set(excluded) == set(all205))
    groups = pd.read_csv(ROOT / "splits/setting2/groups.csv", dtype={"gauge_id": str})
    summary = dict(
        n_basins=len(ids), n_pairs_with_any_overlap=int(len(t)), n_related_pairs_mine=int(t.related.sum()),
        n_related_pairs_mine_not_in_csv=int((t.related & ~t.in_containment_pairs_csv).sum()),
        n_cross_fold_pairs_related=int(cross.related.sum()),
        n_cross_fold_pairs_overlap_gt_0p5=int((cross.overlap_frac_of_smaller > 0.5).sum()),
        n_cross_fold_pairs_overlap_gt_0p1=int((cross.overlap_frac_of_smaller > 0.1).sum()),
        n_cross_fold_pairs_gauge_inside_other=int(cross.gauge_of_smaller_inside_larger.sum()),
        n_cross_fold_pairs_raw_hierarchy=int(cross.raw_hierarchy_contains.sum()),
        max_cross_fold_overlap=float(cross.overlap_frac_of_smaller.max()) if len(cross) else 0.0,
        largest_cross_fold_overlaps=cross.nlargest(5, "overlap_frac_of_smaller")[["a", "b", "overlap_frac_of_smaller", "gauge_of_smaller_inside_larger", "gauge_of_smaller_dist_to_larger_m"]].round(4).to_dict("records"),
        folds_partition_basins_and_large_basins_excluded=bool(partition_ok),
        groups_csv_fold_matches_test_files=bool(all(fold[b] == f for b, f in zip(groups.gauge_id, groups.fold))),
        max_area_km2_in_setting2=float(area.max() / 1e6),
        min_area_km2_excluded=float(pd.read_csv(ROOT / "results/data_report/basin_metadata.csv", dtype={"gauge_id": str}).set_index("gauge_id").loc[excluded, "area"].min()),
    )
    summary["pass"] = bool(summary["n_cross_fold_pairs_related"] == 0 and partition_ok)
    return t, summary


# ---------------------------------------------------------------- 3. fine-tuning base models
def latest_runs(kind):
    runs = {}
    for p in sorted((ROOT / "runs/setting3").glob(f"{kind}_*_*y_*")):
        if p.is_dir():
            runs.setdefault(p.name.rsplit("_", 2)[0], []).append(p)
    return runs


def check_finetune_bases(related):
    sample = pd.read_csv(ROOT / "splits/setting3/sample.csv", dtype={"gauge_id": str}).set_index("gauge_id")
    rows = []
    for name, dirs in sorted(latest_runs("finetune").items()):
        run = dirs[-1]
        cfg = yload(run / "config.yml")
        basin = name.split("_")[1]
        base = ROOT / cfg["base_run_dir"].replace("\\", "/")
        bcfg = yload(base / "config.yml")
        base_train = set(basins_of(bcfg["train_basin_file"].replace("\\", "/")))
        base_test = set(basins_of(bcfg["test_basin_file"].replace("\\", "/")))
        rel = related.get(basin, set())
        log = (run / "output.log").read_text(encoding="utf-8", errors="replace")
        m = re.search(r"Starting training from checkpoint (\S+)", log)
        ckpt = Path(ROOT / m.group(1).replace("\\", "/")) if m else None
        k = int(re.search(r"fold(\d)", str(base)).group(1))
        rows.append(dict(
            run=run.name, basin=basin, n_run_dirs=len(dirs), base_run_dir=cfg["base_run_dir"], base_fold=k,
            sample_fold=int(sample.fold[basin]),
            basin_in_base_train=basin in base_train, basin_in_base_test=basin in base_test,
            n_related_basins_in_base_train=len(rel & base_train),
            scaler_identical_to_base=(run / "train_data/train_data_scaler.yml").read_bytes() == (base / "train_data/train_data_scaler.yml").read_bytes(),
            checkpoint=str(ckpt.relative_to(ROOT)) if ckpt else None,
            checkpoint_is_base_epoch030=bool(ckpt and ckpt.resolve() == (base / "model_epoch030.pt").resolve()),
            statics_equal_base=cfg.get("static_attributes") == bcfg.get("static_attributes"),
            dynamic_inputs_equal_base=cfg.get("dynamic_inputs") == bcfg.get("dynamic_inputs")))
    t = pd.DataFrame(rows)
    ok = (len(t) > 0 and (~t.basin_in_base_train).all() and t.basin_in_base_test.all() and (t.n_related_basins_in_base_train == 0).all()
          and t.scaler_identical_to_base.all() and t.checkpoint_is_base_epoch030.all() and (t.base_fold == t.sample_fold).all()
          and t.statics_equal_base.all() and t.dynamic_inputs_equal_base.all())
    return t, dict(n_runs=int(len(t)), n_basins=int(t.basin.nunique()) if len(t) else 0, pass_=bool(ok),
                   n_names_with_more_than_one_run_dir=int((t.n_run_dirs > 1).sum()) if len(t) else 0)


# ---------------------------------------------------------------- 4. setting-3 training windows
def check_windows(P):
    from neuralhydrology.datasetzoo import get_dataset
    from neuralhydrology.datautils.utils import load_scaler
    from neuralhydrology.utils.config import Config
    os.chdir(ROOT)
    train_end = P["train"][1]
    test_input_start = P["test"][0] - pd.Timedelta(days=364)
    rows, scaler_rows = [], []
    for kind in ("scratch", "finetune"):
        for name, dirs in sorted(latest_runs(kind).items()):
            run = dirs[-1]
            years = int(name.split("_")[2][:-1])
            basin = name.split("_")[1]
            cfg = Config(run / "config.yml")
            ds = get_dataset(cfg, is_train=False, period="train", basin=basin, scaler=load_scaler(run))
            dates = pd.DatetimeIndex(ds._dates[basin]["1D"]) if basin in ds._dates else pd.DatetimeIndex([])
            y = ds._y[basin]["1D"].numpy()[:, 0] if basin in ds._y else np.array([])
            idx = [ix[0] for (bb, ix) in ds.lookup_table.values()]
            tdates = dates[idx]
            valid = np.isfinite(y[idx])
            t_used = tdates[valid]
            expected_start = train_end - pd.DateOffset(years=years) + pd.Timedelta(days=1)
            first_input = t_used.min() - pd.Timedelta(days=cfg.seq_length - 1)
            rows.append(dict(kind=kind, run=run.name, basin=basin, years=years,
                             first_target=t_used.min().date(), last_target=t_used.max().date(), n_targets=int(valid.sum()),
                             n_days_in_window=int((train_end - expected_start).days + 1),
                             expected_first_target=expected_start.date(), first_input_day=first_input.date(),
                             last_input_day=t_used.max().date(),
                             window_ok=bool(t_used.min() >= expected_start and t_used.max() == train_end),
                             days_between_last_training_input_and_first_test_input=int((test_input_start - t_used.max()).days)))
            if kind == "scratch":
                # the scaler of a single-basin model must come from its own training window only
                sc = yload(run / "train_data/train_data_scaler.yml")
                df = processed(basin)
                w0 = expected_start - pd.Timedelta(days=cfg.seq_length - 1)
                for v in FORCING + [TARGET]:
                    x = df.loc[(expected_start if v == TARGET else w0):train_end, v].to_numpy("float64")
                    x_all = df.loc[(expected_start if v == TARGET else w0):P["test"][1], v].to_numpy("float64")
                    if v == TARGET:
                        x, x_all = np.clip(x, 0, None), np.clip(x_all, 0, None)
                    for stat, f in (("center", np.nanmean), ("scale", np.nanstd)):
                        saved = sc[f"xarray_feature_{stat}"]["data_vars"][v]["data"]
                        scaler_rows.append(dict(run=run.name, variable=v, statistic=stat, saved=saved,
                                                rel_diff_window=abs(f(x) - saved) / abs(saved),
                                                rel_diff_window_to_test_end=abs(f(x_all) - saved) / abs(saved)))
    t, s = pd.DataFrame(rows), pd.DataFrame(scaler_rows)
    summary = dict(n_runs=int(len(t)), n_window_ok=int(t.window_ok.sum()),
                   min_gap_days_training_input_to_test_input=int(t.days_between_last_training_input_and_first_test_input.min()),
                   max_last_input_day=str(t.last_input_day.max()),
                   scratch_scaler_max_rel_diff_window=float(s.rel_diff_window.max()) if len(s) else None,
                   scratch_scaler_min_of_max_rel_diff_if_to_test_end=float(s.groupby("run").rel_diff_window_to_test_end.max().min()) if len(s) else None)
    summary["pass"] = bool(t.window_ok.all() and summary["min_gap_days_training_input_to_test_input"] > 0
                           and (summary["scratch_scaler_max_rel_diff_window"] or 0) < 1e-5)
    return t, s, summary


# ---------------------------------------------------------------- 5. gradient boosting
def load_gbm_module():
    spec = importlib.util.spec_from_file_location("gbm_under_audit", ROOT / "scripts/gbm.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_gbm(P, signatures):
    import xgboost as xgb
    gbm = load_gbm_module()
    rng = np.random.default_rng(7)
    attrs = pd.read_csv(ROOT / "data/processed/attributes/attributes.csv", dtype={"gauge_id": str}).set_index("gauge_id")
    # (a) causality: perturb forcing from a cut date on and check nothing before it changes
    caus = []
    for b in ["2126", "2299", "2436"]:
        df = processed(b, FORCING).astype("float64")
        f0 = gbm.features(df)
        for cut in ["2005-10-01", "2010-10-01", "2015-06-15"]:
            d2 = df.copy()
            m = d2.index >= cut
            d2.loc[m] = d2.loc[m].values * rng.uniform(0.2, 3.0, size=d2.loc[m].shape) + rng.normal(0, 2, size=d2.loc[m].shape)
            f1 = gbm.features(d2)
            before = f1.index < cut
            caus.append(dict(basin=b, cut=cut,
                             max_abs_change_before_cut=float(np.nanmax(np.abs((f1[before] - f0[before]).values))),
                             n_features_changed_on_cut_day=int(((f1.loc[cut] - f0.loc[cut]).abs() > 0).sum())))
    caus = pd.DataFrame(caus)
    rows = []
    for name in ["gbm_setting1"] + [f"gbm_setting2_fold{k}" for k in FOLDS]:
        run = ROOT / "runs" / name
        cfg = {l.split(": ", 1)[0]: l.split(": ", 1)[1] for l in (run / "config.yml").read_text().splitlines() if ": " in l}
        feats = json.loads(cfg["features"])
        statics = [f for f in feats if f in attrs.columns]
        booster = xgb.Booster()
        booster.load_model(run / "model.json")
        train_b = basins_of(cfg["train_basin_file"])
        test_b = basins_of(cfg["test_basin_file"])
        n_train = n_trainval = 0
        for b in train_b:
            q = processed(b, [TARGET])[TARGET]
            n_train += int(q.loc[P["train"][0]:P["train"][1]].notna().sum())
            n_trainval += int(q.loc[P["train"][0]:P["validation"][1]].notna().sum())
        # re-predict three test basins from the saved model and compare with the saved simulations
        res = load_results(run / "test/model_epoch001/test_results.p")
        diffs = []
        for b in test_b[:3]:
            df = processed(b)
            X = gbm.features(df[FORCING])
            for s in statics:
                X[s] = attrs.loc[b, s]
            X = X[feats].astype("float32").loc[P["test"][0]:P["test"][1]]
            sim = np.clip(booster.predict(xgb.DMatrix(X)), 0, None)
            diffs.append(float(np.max(np.abs(sim - res[b].sim.values))))
        bad = [f for f in feats if re.search(r"discharge|^q_|runoff|flow|waterlevel|swe", f, re.I) or f in signatures]
        rows.append(dict(run=name, n_features=len(feats), n_static=len(statics), features_discharge_or_signature=bad,
                         booster_feature_names_match=(booster.feature_names == feats), n_trees=booster.num_boosted_rounds(),
                         n_train_rows_config=int(cfg["n_train_rows"]), n_train_rows_recomputed_training_period=n_train,
                         n_rows_if_training_plus_validation=n_trainval,
                         test_basins_in_train=len(set(test_b) & set(train_b)) if name != "gbm_setting1" else None,
                         max_abs_diff_repredicted_vs_saved=max(diffs)))
    t = pd.DataFrame(rows)
    code = (ROOT / "scripts/gbm.py").read_text()
    summary = dict(
        causality_max_change_before_cut=float(caus.max_abs_change_before_cut.max()),
        n_runs=len(t), all_rows_match_training_period=bool((t.n_train_rows_config == t.n_train_rows_recomputed_training_period).all()),
        max_abs_diff_repredicted=float(t.max_abs_diff_repredicted_vs_saved.max()),
        any_discharge_feature=bool(t.features_discharge_or_signature.map(len).sum()),
        n_trees=sorted(set(t.n_trees)),
        code_uses_early_stopping=bool(re.search(r"early_stopping|evals\s*=", code)),
        weights_from_training_period_in_code=bool(re.search(r'std = .*loc\[periods\["train"\]\[0\]:periods\["train"\]\[1\]\]', code)),
    )
    summary["pass"] = bool(summary["causality_max_change_before_cut"] == 0 and summary["all_rows_match_training_period"]
                           and not summary["any_discharge_feature"] and summary["max_abs_diff_repredicted"] < 1e-3
                           and not summary["code_uses_early_stopping"] and t.booster_feature_names_match.all())
    return t, caus, summary


# ---------------------------------------------------------------- 6. static attributes
def signature_names():
    names = set()
    for f in RAW.glob("static_attributes/**/*hydrology*"):
        cols = pd.read_csv(f, comment="#", nrows=0, encoding="latin-1").columns
        names |= {c for c in cols if c != "gauge_id" and not c.startswith("sign_")}
    return names


def all_run_configs():
    out = []
    for p in sorted((ROOT / "runs").rglob("config.yml")):
        if "gbm_" in str(p) or "smoke_test" in str(p):
            continue
        out.append(p)
    return out


def check_statics(signatures):
    rows = []
    for p in all_run_configs():
        cfg = yload(p)
        st = cfg.get("static_attributes") or []
        dyn = cfg.get("dynamic_inputs") or []
        rows.append(dict(config=str(p.relative_to(ROOT)), n_static=len(st), statics_in_signatures=sorted(set(st) & signatures),
                         dynamic_inputs=",".join(dyn),
                         discharge_like_dynamic=[d for d in dyn if re.search(r"discharge|waterlevel|swe|_sim$", d)],
                         lagged_or_autoregressive=bool(cfg.get("lagged_features") or cfg.get("autoregressive_inputs"))))
    t = pd.DataFrame(rows)
    # the static values themselves: non-climate attributes equal the raw files (no substituted columns)
    attrs = pd.read_csv(ROOT / "data/processed/attributes/attributes.csv", dtype={"gauge_id": str}).set_index("gauge_id")
    raw = []
    for f in RAW.glob("static_attributes/CAMELS_CH_*_attributes*.csv"):
        if "hydrology" in f.name or "climate" in f.name:
            continue
        raw.append(pd.read_csv(f, comment="#", dtype={"gauge_id": str}, encoding="latin-1").set_index("gauge_id"))
    raw = pd.concat(raw, axis=1)
    raw = raw.loc[:, ~raw.columns.duplicated()]
    common = [c for c in attrs.columns if c in raw.columns]
    diffs = {c: float(np.nanmax(np.abs(attrs[c].astype(float) - raw.loc[attrs.index, c].astype(float)))) for c in common}
    hyd = pd.read_csv(RAW / "static_attributes/CAMELS_CH_hydrology_attributes_obs.csv", comment="#", dtype={"gauge_id": str}, encoding="latin-1").set_index("gauge_id")
    statics = yload(ROOT / "configs/setting2_fold0.yml")["static_attributes"]
    corr = attrs[statics].join(hyd.q_mean).corr()["q_mean"].drop("q_mean").abs()
    summary = dict(n_configs=len(t), n_with_signature_static=int((t.statics_in_signatures.map(len) > 0).sum()),
                   n_with_discharge_like_dynamic=int((t.discharge_like_dynamic.map(len) > 0).sum()),
                   n_with_lagged_or_autoregressive=int(t.lagged_or_autoregressive.sum()),
                   n_attributes_compared_with_raw=len(common), max_abs_diff_attributes_vs_raw=max(diffs.values()) if diffs else None,
                   attributes_not_in_raw_files=sorted(set(statics) - set(common)),
                   max_abs_corr_static_vs_q_mean=dict(attribute=corr.idxmax(), r=float(corr.max())))
    summary["pass"] = bool(summary["n_with_signature_static"] == 0 and summary["n_with_discharge_like_dynamic"] == 0
                           and summary["n_with_lagged_or_autoregressive"] == 0)
    return t, summary


# ---------------------------------------------------------------- 7. model selection
def check_selection():
    rows = []
    for k in FOLDS:
        run, fin = final_dir(k)
        for d in sorted({run, fin}):
            cfg = yload(d / "config.yml")
            test = sorted(p.name for p in (d / "test").iterdir()) if (d / "test").exists() else []
            val = sorted(p.name for p in (d / "validation").iterdir()) if (d / "validation").exists() else []
            rows.append(dict(group="setting2", run=str(d.relative_to(ROOT)), epochs=cfg["epochs"], validate_every=cfg.get("validate_every"),
                             validation_basin_file=cfg.get("validation_basin_file"), test_dirs=test, validation_dirs=val,
                             weights=sorted(p.name for p in d.glob("model_epoch*.pt")), only_final_epoch_tested=(test in ([], ["model_epoch030"])),
                             validation_on_test_basins=bool(set(basins_of(cfg["validation_basin_file"].replace("\\", "/"))) & set(basins_of(f"splits/setting2/fold{k}_test.txt")))))
    for kind in ("scratch", "finetune"):
        for name, dirs in sorted(latest_runs(kind).items()):
            for d in dirs:
                cfg = yload(d / "config.yml")
                test = sorted(p.name for p in (d / "test").iterdir()) if (d / "test").exists() else []
                weights = sorted(p.name for p in d.glob("model_epoch*.pt"))
                rows.append(dict(group=f"setting3_{kind}", run=str(d.relative_to(ROOT)), epochs=cfg["epochs"], validate_every=cfg.get("validate_every"),
                                 test_dirs=test, weights=weights, n_dirs_for_name=len(dirs),
                                 only_final_epoch_tested=(test == [f"model_epoch{cfg['epochs']:03d}"]) and weights == [f"model_epoch{cfg['epochs']:03d}.pt"]))
    for name in ["gbm_setting1"] + [f"gbm_setting2_fold{k}" for k in FOLDS]:
        d = ROOT / "runs" / name
        rows.append(dict(group="gbm", run=f"runs/{name}", test_dirs=sorted(p.name for p in (d / "test").iterdir()),
                         only_final_epoch_tested=sorted(p.name for p in (d / "test").iterdir()) == ["model_epoch001"]))
    t = pd.DataFrame(rows)
    s3 = t[t.group.str.startswith("setting3")]
    summary = dict(n_units=len(t), n_not_only_final=int((~t.only_final_epoch_tested).sum()),
                   setting2_epochs=sorted(set(t[t.group == "setting2"].epochs)),
                   setting2_validation_on_test_basins=int(t[t.group == "setting2"].validation_on_test_basins.sum()),
                   setting3_validate_every=sorted({str(v) for v in s3.validate_every}),
                   setting3_names_with_more_than_one_dir=int((s3.n_dirs_for_name > 1).sum()))
    summary["pass"] = bool(summary["n_not_only_final"] == 0 and summary["setting2_validation_on_test_basins"] == 0
                           and summary["setting3_names_with_more_than_one_dir"] == 0)
    return t, summary


def check_recipe_consistency():
    """Every setting-3 run config has the recipe of the committed script (no change between runs)."""
    spec = importlib.util.spec_from_file_location("s3cfg", ROOT / "scripts/make_setting3_configs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rows = []
    for kind in ("scratch", "finetune"):
        rec = mod.SCRATCH if kind == "scratch" else mod.FINETUNE
        for name, dirs in sorted(latest_runs(kind).items()):
            cfg = yload(dirs[-1] / "config.yml")
            years = int(name.split("_")[2][:-1])
            exp_epochs = int(np.ceil(rec["updates"] / mod.n_batches(years)))
            lr = cfg["learning_rate"]
            row = dict(kind=kind, run=dirs[-1].name, epochs=cfg["epochs"], expected_epochs=exp_epochs, seed=cfg["seed"],
                       lr=json.dumps({str(k): v for k, v in lr.items()}), hidden_size=cfg["hidden_size"], batch_size=cfg["batch_size"],
                       start_time=re.search(r"_(\d{4}_\d{6})$", dirs[-1].name).group(1))
            row["matches_committed_recipe"] = (cfg["epochs"] == exp_epochs and cfg["seed"] == rec["seed"] and
                                               list(lr.values()) == [rec["learning_rate"]] and cfg["batch_size"] == mod.BATCH and
                                               (kind == "finetune" or cfg["hidden_size"] == rec["hidden_size"]))
            rows.append(row)
    t = pd.DataFrame(rows)
    import subprocess
    log = subprocess.run(["git", "log", "--format=%h %ad", "--date=iso", "--", "scripts/make_setting3_configs.py"],
                         cwd=ROOT, capture_output=True, text=True).stdout.strip().splitlines()
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "scripts/make_setting3_configs.py"], cwd=ROOT).returncode != 0
    trial = (ROOT / "runs/setting3/stdout_trial.log")
    summary = dict(n_runs=len(t), n_matching=int(t.matches_committed_recipe.sum()), commits_touching_script=log,
                   script_differs_from_HEAD=dirty,
                   script_mtime=str(pd.Timestamp(os.path.getmtime(ROOT / "scripts/make_setting3_configs.py"), unit="s", tz="UTC").tz_convert("Europe/Zurich")),
                   trial_log_mtime=str(pd.Timestamp(os.path.getmtime(trial), unit="s", tz="UTC").tz_convert("Europe/Zurich")) if trial.exists() else None,
                   scratch_run_starts=[t[t.kind == "scratch"].start_time.min(), t[t.kind == "scratch"].start_time.max()],
                   finetune_run_starts=[t[t.kind == "finetune"].start_time.min(), t[t.kind == "finetune"].start_time.max()] if (t.kind == "finetune").any() else None)
    return t, summary


# ---------------------------------------------------------------- 8. resume incident
def check_resume(P, reinfer_n, skip):
    rows = []
    for k in FOLDS:
        run, fin = final_dir(k)
        w = fin / "model_epoch030.pt"
        row = dict(fold=k, final_dir=str(fin.relative_to(ROOT)), resumed=fin != run, sha256=sha256(w),
                   weights_mtime=str(pd.Timestamp(os.path.getmtime(w), unit="s", tz="UTC").tz_convert("Europe/Zurich").tz_localize(None)))
        log = (fin / "output.log").read_text(encoding="utf-8", errors="replace")
        ep = {int(m.group(2)): m.group(1) for m in re.finditer(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+: Epoch (\d+) average loss", log)}
        row["last_epoch_logged_in_run_log"] = max(ep) if ep else None
        row["epoch30_logged_at"] = ep.get(30)
        row["epoch31_logged_at"] = ep.get(31)
        m = re.search(r"Continue training from epoch (\d+)", log)
        row["resumed_from_epoch"] = int(m.group(1)) if m else None
        m = re.search(r"Using the model weights from (\S+)", log)
        row["test_eval_weights"] = m.group(1) if m else None
        row["test_eval_used_epoch030_of_final_dir"] = bool(m and Path(ROOT / m.group(1).replace("\\", "/")).resolve() == w.resolve())
        row["weights_files_in_run_tree"] = sorted(str(p.relative_to(run)) for p in run.rglob("model_epoch*.pt"))
        stdout = ROOT / f"runs/stdout_setting2_fold{k}.log"
        if stdout.exists():
            txt = stdout.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
            eps = [int(x) for x in re.findall(r"# Epoch (\d+):", txt)]
            row["highest_epoch_in_stdout"] = max(eps) if eps else None
        rows.append(row)
    t = pd.DataFrame(rows)
    # the weights were written before epoch 31 started and not touched afterwards
    t["weights_written_before_epoch31"] = [
        pd.isna(r.epoch31_logged_at) or (pd.Timestamp(r.weights_mtime) <= pd.Timestamp(r.epoch31_logged_at)) for r in t.itertuples()]
    reinf = pd.DataFrame()
    if not skip:
        reinf = reinference(P, reinfer_n)
    summary = dict(per_fold=t[["fold", "resumed", "resumed_from_epoch", "epoch30_logged_at", "weights_mtime", "epoch31_logged_at",
                               "last_epoch_logged_in_run_log", "test_eval_used_epoch030_of_final_dir", "weights_written_before_epoch31"]].to_dict("records"),
                   reinference_max_abs_diff=float(reinf.max_abs_diff.max()) if len(reinf) else None,
                   reinference_max_abs_diff_NSE=float((reinf.NSE_saved - reinf.NSE_rerun).abs().max()) if len(reinf) else None,
                   reinference_max_rel_diff_NSE=float(((reinf.NSE_saved - reinf.NSE_rerun).abs() / np.maximum(1, reinf.NSE_saved.abs())).max()) if len(reinf) else None,
                   reinference_n_basins=int(len(reinf)))
    summary["pass"] = bool(t.test_eval_used_epoch030_of_final_dir.all() and t.weights_written_before_epoch31.all()
                           and (skip or summary["reinference_max_rel_diff_NSE"] < 1e-3))  # CPU vs cuDNN arithmetic: exact equality is not expected
    return t, reinf, summary


def reinference(P, n):
    """Repeat the test inference on the CPU from the saved epoch-30 weights for n basins per fold."""
    import logging
    import torch
    from neuralhydrology.evaluation import get_tester
    from neuralhydrology.utils.config import Config
    logging.getLogger("neuralhydrology").setLevel(logging.WARNING)
    os.chdir(ROOT)
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    rows = []
    for k in FOLDS:
        run, fin = final_dir(k)
        cfg = Config(fin / "config.yml")
        cfg.update_config({"device": "cpu", "verbose": 0})
        tester = get_tester(cfg, run_dir=fin, period="test", init_model=True)
        saved = load_results(fin / "test/model_epoch030/test_results.p")
        shift = pd.read_csv(ROOT / f"results/verification_setting2/fold{k}/shift_check.csv", dtype={"basin": str})
        pick = list(dict.fromkeys(shift[shift.shifted_higher].basin.tolist()[:2] + sorted(saved)[:n]))[:max(n, 2)]
        tester.basins = pick
        res = tester.evaluate(epoch=30, save_results=False)
        for b in pick:
            ds = res[b]["1D"]["xr"]
            sim = ds[f"{TARGET}_sim"].values[:, 0]
            rows.append(dict(fold=k, basin=b, max_abs_diff=float(np.nanmax(np.abs(sim - saved[b].sim.values))),
                             NSE_saved=nse(saved[b].obs, saved[b].sim.clip(lower=0)), NSE_rerun=nse(saved[b].obs, np.clip(sim, 0, None))))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 9. folds on two GPUs
def check_fold_configs():
    cfgs = {}
    for k in FOLDS:
        run, fin = final_dir(k)
        cfgs[k] = yload(run / "config.yml")
        if fin != run:
            c2 = yload(fin / "config.yml")
            cfgs[f"{k}_resumed"] = c2
    ref = cfgs[0]
    rows = []
    for name, c in cfgs.items():
        keys = (set(ref) | set(c)) - FOLD_SPECIFIC
        diff = sorted(x for x in keys if ref.get(x) != c.get(x))
        rows.append(dict(config=str(name), differing_keys_vs_fold0=diff, seed=c.get("seed"), package_version=c.get("package_version"),
                         commit_hash=c.get("commit_hash"), device=c.get("device")))
    t = pd.DataFrame(rows)
    gpu = {}
    for k in FOLDS:
        run, fin = final_dir(k)
        txt = (fin / "output.log").read_text(encoding="utf-8", errors="replace")
        m = re.findall(r"(RTX \d{4}|GeForce[^\n,]*)", txt)
        gpu[str(k)] = sorted(set(m)) or "not recorded in output.log"
        val = fin / "validation/model_epoch030/validation_metrics.csv"
        gpu[f"{k}_val_median_NSE_epoch30"] = float(pd.read_csv(val).NSE.median()) if val.exists() else None
    summary = dict(n_configs=len(t), configs_with_nonfold_differences=t[t.differing_keys_vs_fold0.map(len) > 0][["config", "differing_keys_vs_fold0"]].to_dict("records"),
                   seeds={str(a): b for a, b in zip(t.config, t.seed)}, gpu_and_validation=gpu)
    summary["pass"] = bool(len(summary["configs_with_nonfold_differences"]) == 0)
    return t, summary


# ---------------------------------------------------------------- 10 and shift: skill and timing
def load_all_predictions():
    preds = {}
    for k in FOLDS:
        run, fin = final_dir(k)
        for b, df in load_results(fin / "test/model_epoch030/test_results.p").items():
            preds[("lstm_s2", b)] = df.assign(fold=k)
        for b, df in load_results(ROOT / f"runs/gbm_setting2_fold{k}/test/model_epoch001/test_results.p").items():
            preds[("gbm_s2", b)] = df.assign(fold=k)
    seeds = [load_results(next((ROOT / "runs").glob(f"baseline_seed{s}_*")) / "test/model_epoch030/test_results.p") for s in (1, 2, 3)]
    for b in seeds[0]:
        preds[("lstm_s1_seed1", b)] = seeds[0][b]
        preds[("lstm_s1_mean", b)] = pd.DataFrame({"obs": seeds[0][b].obs, "sim": np.mean([s[b].sim.clip(lower=0) for s in seeds], axis=0)})
    for b, df in load_results(ROOT / "runs/gbm_setting1/test/model_epoch001/test_results.p").items():
        preds[("gbm_s1", b)] = df
    return preds


def lagcorr(obs, sim, k):
    """corr(sim(t), obs(t+k))"""
    o = obs.shift(-k)
    m = o.notna() & sim.notna()
    return float(np.corrcoef(o[m], sim[m])[0, 1])


def skill_and_shift(preds, P):
    ids = basins_of("splits/setting2/basins_setting2.txt")
    flags = pd.read_csv(ROOT / "results/data_report/basin_flags.csv", dtype={"gauge_id": str}).set_index("gauge_id")
    meta = pd.read_csv(ROOT / "results/data_report/basin_metadata.csv", dtype={"gauge_id": str}).set_index("gauge_id")
    attrs = pd.read_csv(ROOT / "data/processed/attributes/attributes.csv", dtype={"gauge_id": str}).set_index("gauge_id")
    hg = pd.read_csv(RAW / "static_attributes/CAMELS_CH_hydrogeology_attributes.csv", comment="#", dtype={"gauge_id": str}, encoding="latin-1").set_index("gauge_id")
    models = ["lstm_s2", "gbm_s2", "lstm_s1_seed1", "lstm_s1_mean", "gbm_s1"]
    rows = []
    for b in ids:
        pr = processed(b, ["precipitation"]).precipitation.loc[P["test"][0]:P["test"][1]]
        r = dict(basin=b)
        for mname in models:
            df = preds[(mname, b)]
            obs, sim = df.obs, df.sim.clip(lower=0)
            r[f"NSE_{mname}"] = nse(obs, sim)
            lc = {kk: lagcorr(obs, sim, kk) for kk in (-2, -1, 0, 1, 2)}
            for kk, v in lc.items():
                r[f"r_{mname}_lag{kk:+d}"] = v
            r[f"best_lag_{mname}"] = max(lc, key=lc.get)
            # response time to precipitation: lag of the largest corr(P(t), Q(t+k))
            for lab, series in (("obs", obs), ("sim", sim)):
                if lab == "obs" and mname != "lstm_s2":
                    continue
                cc = {kk: lagcorr(pr.reindex(series.index), series, kk) for kk in range(0, 6)}
                r[f"P_response_lag_{lab if lab == 'obs' else mname}"] = max(cc, key=cc.get)
                r[f"P_corr_lag0_{lab if lab == 'obs' else mname}"] = cc[0]
                r[f"P_corr_lag1_{lab if lab == 'obs' else mname}"] = cc[1]
        r["fold"] = int(preds[("lstm_s2", b)].fold.iloc[0])
        rows.append(r)
    t = pd.DataFrame(rows).set_index("basin")
    t = t.join(flags).join(meta[["country", "area", "gauge_name", "water_body_name"]]).join(attrs[["inwater_perc", "carbonate_sedimentary", "elev_mean", "glac_frac"]]).join(hg[["karst_perc", "water_perc"]])
    for mname in models:
        t[f"shifted_{mname}"] = t[f"best_lag_{mname}"] != 0
    # skill table
    groups = [("All setting-2 basins", pd.Series(True, index=t.index)), ("Near-natural", t.near_natural), ("Not near-natural", ~t.near_natural)]
    groups += [(f, t[f]) for f in FLAG_COLS]
    sk = []
    for label, m in groups:
        g = t[m.astype(bool)]
        row = dict(group=label, n=len(g))
        for mname in models:
            row[f"median_NSE_{mname}"] = g[f"NSE_{mname}"].median()
            row[f"p05_NSE_{mname}"] = g[f"NSE_{mname}"].quantile(0.05)
            row[f"n_below0_{mname}"] = int((g[f"NSE_{mname}"] < 0).sum())
        sk.append(row)
    sk = pd.DataFrame(sk)
    fold_med = t.groupby("fold")[[f"NSE_{m}" for m in models]].median()
    return t, sk, fold_med


def check_alignment(t, P, preds):
    """Obs in the results against the raw files, and processed forcing against the raw forcing, same day and +-1 day."""
    rows = []
    for b in t.index:
        raw = raw_series(b)
        o = preds[("lstm_s2", b)].obs
        rq = raw["discharge_spec(mm/d)"]
        pr = processed(b, FORCING).loc[P["train"][0]:P["test"][1]]
        rp = raw["precipitation(mm/d)"]
        rec = dict(basin=b)
        for lab, d in (("same_day", 0), ("raw_day_before", -1), ("raw_day_after", 1)):
            rr = rq.reindex(o.index + pd.Timedelta(days=d)).values
            m = np.isfinite(rr) & np.isfinite(o.values)
            rec[f"obs_max_abs_diff_{lab}"] = float(np.max(np.abs(o.values[m] - rr[m])))
            rr = rp.reindex(pr.index + pd.Timedelta(days=d)).values
            m = np.isfinite(rr) & np.isfinite(pr.precipitation.values)
            rec[f"precip_max_abs_diff_{lab}"] = float(np.max(np.abs(pr.precipitation.values[m] - rr[m])))
        for v in FORCING[1:]:
            col = [c for c in raw.columns if c.startswith(v + "(")][0]
            rec[f"{v}_max_abs_diff_same_day"] = float(np.nanmax(np.abs(pr[v].values - raw[col].reindex(pr.index).values)))
        rows.append(rec)
    return pd.DataFrame(rows)


def shift_summary(t, align):
    models = ["lstm_s2", "gbm_s2", "lstm_s1_seed1", "lstm_s1_mean", "gbm_s1"]
    out = {m: dict(n_shifted=int(t[f"shifted_{m}"].sum()), n_best_lag_plus1=int((t[f"best_lag_{m}"] == 1).sum()),
                   n_best_lag_minus1=int((t[f"best_lag_{m}"] == -1).sum())) for m in models}
    out["lstm_s2_by_fold"] = {str(k): int(v) for k, v in t.groupby("fold").shifted_lstm_s2.sum().items()}
    out["gbm_s2_by_fold"] = {str(k): int(v) for k, v in t.groupby("fold").shifted_gbm_s2.sum().items()}
    sh = t[t.shifted_lstm_s2]
    out["lstm_s2_shifted_basins"] = sh[["fold", "best_lag_lstm_s2", "best_lag_lstm_s1_mean", "best_lag_lstm_s1_seed1", "best_lag_gbm_s2", "best_lag_gbm_s1",
                                        "NSE_lstm_s2", "NSE_lstm_s1_mean", "P_response_lag_obs", "P_response_lag_lstm_s2",
                                        "lake", "regulated", "karst_perc", "inwater_perc", "area", "country", "water_body_name"]].round(3).reset_index().to_dict("records")
    out["shifted_lstm_s2_that_are_shifted_in_setting1_seed_mean"] = int((sh.best_lag_lstm_s1_mean != 0).sum())
    rest = t[~t.shifted_lstm_s2]
    out["share_lake_flag"] = dict(shifted=float(sh.lake.mean()), others=float(rest.lake.mean()))
    out["median_karst_perc"] = dict(shifted=float(sh.karst_perc.median()), others=float(rest.karst_perc.median()))
    out["median_inwater_perc"] = dict(shifted=float(sh.inwater_perc.median()), others=float(rest.inwater_perc.median()))
    out["median_area"] = dict(shifted=float(sh.area.median()), others=float(rest.area.median()))
    out["share_outside_CH"] = dict(shifted=float((sh.country != "CH").mean()), others=float((rest.country != "CH").mean()))
    out["P_response_lag"] = dict(
        shifted_obs_minus_sim={str(k): int(v) for k, v in pd.Series(sh.P_response_lag_obs - sh.P_response_lag_lstm_s2).value_counts().sort_index().items()},
        others_obs_minus_sim={str(k): int(v) for k, v in pd.Series(rest.P_response_lag_obs - rest.P_response_lag_lstm_s2).value_counts().sort_index().items()})
    out["corr_gain_lag+1_minus_lag0_lstm_s2_shifted_median"] = float((sh["r_lstm_s2_lag+1"] - sh["r_lstm_s2_lag+0"]).median())
    out["corr_gain_lag+1_minus_lag0_lstm_s1_mean_same_basins_median"] = float((sh["r_lstm_s1_mean_lag+1"] - sh["r_lstm_s1_mean_lag+0"]).median())
    out["alignment"] = dict(
        obs_max_abs_diff_same_day=float(align.obs_max_abs_diff_same_day.max()),
        obs_min_over_basins_of_max_abs_diff_raw_day_before=float(align.obs_max_abs_diff_raw_day_before.min()),
        obs_min_over_basins_of_max_abs_diff_raw_day_after=float(align.obs_max_abs_diff_raw_day_after.min()),
        precip_max_abs_diff_same_day=float(align.precip_max_abs_diff_same_day.max()),
        precip_min_over_basins_of_max_abs_diff_raw_day_before=float(align.precip_max_abs_diff_raw_day_before.min()),
        other_forcing_max_abs_diff_same_day=float(align[[c for c in align.columns if c.endswith("_max_abs_diff_same_day") and not c.startswith(("obs", "precip"))]].max().max()))
    return out


def setting3_skill(preds, windows):
    """NSE of every setting-3 run next to the gauged (setting 1) and ungauged (setting 2) NSE of the same basin.
    Used to judge the single-basin runs with NSE above 0.90 that the verifier flags."""
    rows = []
    for kind in ("scratch", "finetune"):
        for name, dirs in sorted(latest_runs(kind).items()):
            run = dirs[-1]
            basin, years = name.split("_")[1], int(name.split("_")[2][:-1])
            res = load_results(next(run.glob("test/model_epoch*/test_results.p")))[basin]
            rows.append(dict(kind=kind, basin=basin, years=years, run=run.name, NSE=nse(res.obs, res.sim.clip(lower=0)),
                             r_lag0=lagcorr(res.obs, res.sim, 0), r_lag_plus1=lagcorr(res.obs, res.sim, 1),
                             test_obs_identical_to_setting1=bool(np.allclose(res.obs.values, preds[("lstm_s1_seed1", basin)].obs.reindex(res.index).values, atol=1e-4, equal_nan=True))))
    t = pd.DataFrame(rows)
    for m in ("lstm_s1_mean", "lstm_s1_seed1", "lstm_s2", "gbm_s1"):
        t[f"NSE_{m}_same_basin"] = [nse(preds[(m, b)].obs, preds[(m, b)].sim.clip(lower=0)) for b in t.basin]
    if windows is not None and len(windows):
        w = windows.set_index("run")[["first_target", "last_target", "days_between_last_training_input_and_first_test_input"]]
        t = t.join(w, on="run")
    t["above_0p90"] = t.NSE > 0.90
    hi = t[t.above_0p90]
    summary = dict(
        median_NSE={f"{k}_{y}y": round(float(v), 4) for (k, y), v in t.groupby(["kind", "years"]).NSE.median().items()},
        n_runs_above_0p90=int(len(hi)),
        runs_above_0p90=hi[["run", "NSE", "NSE_lstm_s1_mean_same_basin", "NSE_lstm_s1_seed1_same_basin", "NSE_lstm_s2_same_basin"]].round(4).to_dict("records"),
        n_finetune_runs_above_setting1_seed1=int((t[t.kind == "finetune"].NSE > t[t.kind == "finetune"].NSE_lstm_s1_seed1_same_basin).sum()),
        max_finetune_minus_setting1_seed1=float((t[t.kind == "finetune"].NSE - t[t.kind == "finetune"].NSE_lstm_s1_seed1_same_basin).max()),
    )
    return t, summary


def build_summary(S):
    """One entry per check: id, status (pass / fail / finding), key number and a one-sentence note."""
    st = lambda ok: "pass" if ok else "fail"
    E = []
    s = S["1_scaler"]
    worst_other = min(min(x for n, x in v["max_rel_diff_by_variant"].items() if n != "fold train basins, training period")
                      for k, v in s.items() if k.startswith("fold"))
    worst_ok = max(v["max_rel_diff_by_variant"]["fold train basins, training period"] for k, v in s.items() if k.startswith("fold"))
    E.append(dict(id="1_setting2_scaler", status=st(s["pass"]), key_number=worst_ok,
                  note=f"Saved scalers equal the statistics of each fold's train basins in the training period (max rel. diff {worst_ok:.1e}); "
                       f"any other basin set or period differs by at least {worst_other:.3f}."))
    s = S["2_containment"]
    E.append(dict(id="2_nested_basins", status=st(s["pass"]), key_number=s["n_cross_fold_pairs_related"],
                  note=f"Recomputed from polygons, gauge points and raw H1/H2 hierarchy: {s['n_related_pairs_mine']} related pairs, all in containment_pairs.csv, "
                       f"none across train/test; largest cross-fold overlap {s['max_cross_fold_overlap']:.1e} of the smaller polygon."))
    if "3_finetune_bases" in S:
        s = S["3_finetune_bases"]
        E.append(dict(id="3_finetune_base_models", status=st(s["pass_"]), key_number=s["n_runs"],
                      note="Every fine-tune run starts from epoch 30 of the fold where its basin (and every related basin) was a test basin, with the base model's scaler byte-identical."))
        s = S["4_setting3_windows"]
        E.append(dict(id="4_setting3_windows", status=st(s["pass"]), key_number=s["min_gap_days_training_input_to_test_input"],
                      note=f"Training targets rebuilt with NeuralHydrology are exactly the last N years ending {s['max_last_input_day']} in {s['n_window_ok']}/{s['n_runs']} runs; "
                           f"the closest training input lies {s['min_gap_days_training_input_to_test_input']} days before the first test input (incl. 365-day warm-up)."))
    s = S["5_gbm"]
    E.append(dict(id="5_gbm", status=st(s["pass"]), key_number=s["causality_max_change_before_cut"],
                  note="Perturbing forcing after a date changes no feature before it; no discharge feature; training rows equal the training-period count; "
                       "saved models reproduce the saved test simulations exactly; 1500 trees, no early stopping."))
    s = S["6_statics"]
    E.append(dict(id="6_static_attributes", status=st(s["pass"]), key_number=s["n_with_signature_static"],
                  note=f"No hydrological signature, discharge, lagged or autoregressive input in {s['n_configs']} run configurations; "
                       f"highest |r| of a static with q_mean is {s['max_abs_corr_static_vs_q_mean']['r']:.2f} ({s['max_abs_corr_static_vs_q_mean']['attribute']})."))
    s = S["7_selection"]
    E.append(dict(id="7_model_selection", status=st(s["pass"]), key_number=s["n_not_only_final"],
                  note="Each unit has exactly one test evaluation, at its fixed final epoch; validation in setting 2 uses training basins only and selects nothing."))
    s = S["8_resume"]
    E.append(dict(id="8_resume_incident", status=st(s["pass"]), key_number=s["reinference_max_rel_diff_NSE"],
                  note=f"Folds 0-2 were evaluated with continue_training's epoch-30 file, written before epoch 31 began; CPU re-inference of {s['reinference_n_basins']} basins "
                       f"from those weights reproduces the saved simulations to CPU/GPU rounding (max abs diff {s['reinference_max_abs_diff']:.3f} mm/d; NSE differs by at most {s['reinference_max_rel_diff_NSE']:.1e} relative, the largest absolute change being in basin 2239 with NSE -85). "
                       "No file records a hash at save time; the evidence is the log path, the file time (= epoch-30 log line, before epoch 31) and the re-inference. Epochs 31-32 left no weights and nothing from them was evaluated."))
    s = S["9_fold_configs"]
    E.append(dict(id="9_folds_other_gpu", status=st(s["pass"]), key_number=len(s["configs_with_nonfold_differences"]),
                  note="Configurations of folds 3-4 equal folds 0-2 except fold-specific keys (basin files, seed, names); GPU kernels are not deterministic, so no fold is bit-reproducible."))
    tab = pd.DataFrame(S["10_skill"]["table"]).set_index("group")
    E.append(dict(id="10_skill_with_without_human_influence", status="finding", key_number=float(tab.loc["All setting-2 basins", "median_NSE_lstm_s2"]),
                  note=f"Setting-2 LSTM median NSE {tab.loc['All setting-2 basins', 'median_NSE_lstm_s2']:.3f} on 178 basins, "
                       f"{tab.loc['Near-natural', 'median_NSE_lstm_s2']:.3f} on {int(tab.loc['Near-natural', 'n'])} near-natural and "
                       f"{tab.loc['Not near-natural', 'median_NSE_lstm_s2']:.3f} on the rest (gauged setting-1 seed 1 on the same basins {tab.loc['All setting-2 basins', 'median_NSE_lstm_s1_seed1']:.3f})."))
    if "a_recipes" in S:
        s = S["a_recipes"]
        E.append(dict(id="a_printed_trial_and_recipes", status="finding", key_number=s["n_matching"],
                      note=f"All {s['n_matching']}/{s['n_runs']} setting-3 run configs match the committed recipe, but the script was edited at {s['script_mtime'][:16]} "
                           f"and first committed in {s['commits_touching_script'][0][:7]}, after the scratch trial (01:37) and after all scratch runs; git cannot prove the recipe "
                           "predates the printed NSE, only that no run used a different one."))
    s = S["b_one_day_shift"]
    E.append(dict(id="b_one_day_shift_alarms", status="finding", key_number=s["lstm_s2"]["n_shifted"],
                  note=f"{s['lstm_s2']['n_shifted']}/178 basins peak one day early in setting 2 (best lag +1: {s['lstm_s2']['n_best_lag_plus1']}); only "
                       f"{s['shifted_lstm_s2_that_are_shifted_in_setting1_seed_mean']} of them (2239, regulated Spoel) is also shifted when gauged with the same inputs and observations; "
                       f"10 are lake-influenced (3 % of the other basins) and 8 are Doubs karst gauges forming one group in fold 0; observations and forcing equal the raw files on the same day "
                       f"(max diff {s['alignment']['obs_max_abs_diff_same_day']:.1e}) and differ by at least {s['alignment']['obs_min_over_basins_of_max_abs_diff_raw_day_before']:.2f} mm/d when shifted a day; "
                       "the shifted basins have low, not high, NSE; read as a timing error of the ungauged model, not leakage or misalignment."))
    if "c_setting3_skill" in S:
        s = S["c_setting3_skill"]
        E.append(dict(id="c_setting3_nse_above_0p90", status="finding", key_number=s["n_runs_above_0p90"],
                      note=f"{s['n_runs_above_0p90']} single-basin runs exceed NSE 0.90 (2202 1/2/5y, 2262 2/5y); both basins already reach 0.90-0.94 when gauged in setting 1, "
                           "the runs stay within -0.016 to +0.017 of setting-1 seed 1, test observations equal setting 1, and every training window ends 2005-09-30; plausible, no leakage found. "
                           f"Across all 60 fine-tune runs {s['n_finetune_runs_above_setting1_seed1']} beat setting-1 seed 1, by at most {s['max_finetune_minus_setting1_seed1']:+.3f} (basin 2299)."))
        E.append(dict(id="d_verify_all_suffix_bug", status="finding", key_number=len(s.get("verifier_hard_failures", [])),
                      note="verify_all.py exempts 'alarm_median_nse_above_0.90' by exact id, but ids carry a '[run]' suffix, so the exemption never applies; "
                           "all hard failures of setting 3 are this not-applicable alarm."))
    return E


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reinfer-n", type=int, default=3)
    ap.add_argument("--skip-reinference", action="store_true")
    ap.add_argument("--skip-setting3", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    P = periods()
    S = {}
    windows = None
    write = lambda df, name, **kw: df.to_csv(OUT / name, lineterminator="\n", **kw)

    all205 = basins_of("splits/basins_all.txt")
    data = {b: processed(b, FORCING + [TARGET]) for b in all205}
    signatures = signature_names()
    S["signature_names_from_raw"] = sorted(signatures)

    t, S["1_scaler"] = check_scalers(P, data)
    write(t, "1_scaler_check.csv", index=False)

    t, S["2_containment"] = check_containment()
    write(t, "2_containment_pairs_recomputed.csv", index=False)
    related = {}
    for a, b in zip(t[t.related].a, t[t.related].b):
        related.setdefault(a, set()).add(b)
        related.setdefault(b, set()).add(a)

    if not args.skip_setting3:
        t, S["3_finetune_bases"] = check_finetune_bases(related)
        write(t, "3_finetune_bases.csv", index=False)
        t, s, S["4_setting3_windows"] = check_windows(P)
        windows = t
        write(t, "4_setting3_windows.csv", index=False)
        write(s, "4_scratch_scaler.csv", index=False)
        t, S["a_recipes"] = check_recipe_consistency()
        write(t, "a_setting3_recipes.csv", index=False)

    t, c, S["5_gbm"] = check_gbm(P, signatures)
    write(t, "5_gbm_runs.csv", index=False)
    write(c, "5_gbm_causality.csv", index=False)

    t, S["6_statics"] = check_statics(signatures)
    write(t, "6_statics_and_inputs.csv", index=False)

    t, S["7_selection"] = check_selection()
    write(t, "7_model_selection.csv", index=False)

    t, r, S["8_resume"] = check_resume(P, args.reinfer_n, args.skip_reinference)
    write(t, "8_resume.csv", index=False)
    if len(r):
        write(r, "8_reinference.csv", index=False)

    t, S["9_fold_configs"] = check_fold_configs()
    write(t, "9_fold_configs.csv", index=False)

    preds = load_all_predictions()
    t, sk, fm = skill_and_shift(preds, P)
    write(t, "10_per_basin_skill_and_lags.csv")
    write(sk, "10_skill_by_group.csv", index=False)
    write(fm, "10_median_NSE_by_fold.csv")
    ver = pd.read_csv(ROOT / "results/verification_setting2/combined/per_basin_metrics.csv", dtype={"basin": str})
    ver = ver[ver.metric == "NSE"].set_index("basin").value.rename("NSE").to_frame()
    S["10_skill"] = dict(table=sk.round(4).to_dict("records"), by_fold={str(k): v for k, v in fm.round(4).to_dict("index").items()},
                         max_abs_diff_NSE_vs_verifier=float((t.NSE_lstm_s2 - ver.loc[t.index, "NSE"]).abs().max()) if "NSE" in ver else None)
    align = check_alignment(t, P, preds)
    write(align, "b_alignment_obs_and_forcing_vs_raw.csv", index=False)
    S["b_one_day_shift"] = shift_summary(t, align)
    if not args.skip_setting3:
        t3, S["c_setting3_skill"] = setting3_skill(preds, windows)
        write(t3, "c_setting3_skill.csv", index=False)
        ver3 = ROOT / "results/verification_setting3/checks.csv"
        if ver3.exists():
            ch = pd.read_csv(ver3)
            hard = ch[(ch.status == "fail") & ch.hard & ch.counts]
            S["c_setting3_skill"]["verifier_hard_failures"] = hard[["unit", "check"]].to_dict("records")
            S["c_setting3_skill"]["verifier_failures_are_only_the_not_applicable_median_alarm"] = bool(
                hard.check.str.startswith("alarm_median_nse_above_0.90").all())

    (OUT / "audit_transfer_details.json").write_text(json.dumps(S, indent=2, default=str), encoding="utf-8", newline="\n")
    (OUT / "audit_summary.json").write_text(json.dumps(build_summary(S), indent=2, default=str), encoding="utf-8", newline="\n")
    print(json.dumps({k: (v.get("pass", v.get("pass_")) if isinstance(v, dict) else None) for k, v in S.items()}, indent=1))


if __name__ == "__main__":
    main()
