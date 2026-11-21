"""Post-run audit of the baseline (setting 1: unseen period in gauged basins).

Reads the run directories, the processed and the raw data, and writes every number the
audit report uses to results/audit/. It does not train or evaluate any model and it does
not use the metric code of the modelling library.

Usage: python audit/audit_baseline.py --run-dirs runs/baseline_seed1_* runs/baseline_seed2_* runs/baseline_seed3_*
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/camels_ch_v0.9/camels_ch"
TARGET = "discharge_spec"
SIGNATURES = {"q_mean", "runoff_ratio", "stream_elas", "slope_fdc", "baseflow_index_landson", "hfd_mean",
              "Q5", "Q95", "high_q_freq", "high_q_dur", "low_q_freq", "low_q_dur", "zero_q_freq"}
SIM_FORCING = ["precipitation_sim", "wind_sim", "pet_sim"]


def yaml_load(path):
    return YAML(typ="safe").load(Path(path).read_text())


def nse(obs, sim):
    den = ((obs - obs.mean()) ** 2).sum()
    return 1 - ((sim - obs) ** 2).sum() / den if den > 0 else np.nan


def hydro_year(index):
    return index.year + (index.month >= 10)


def doy_key(index):
    return index.month * 100 + index.day


# ---------------------------------------------------------------- loading
def load_processed(basins):
    return {b: xr.open_dataset(ROOT / f"data/processed/time_series/{b}.nc").to_dataframe() for b in basins}


def load_results(run_dirs, basins):
    """Observed series and per-seed simulations of the test period, as saved by the library."""
    obs, sims = {}, {b: {} for b in basins}
    for run in run_dirs:
        res = pickle.load(open(run / "test/model_epoch030/test_results.p", "rb"))
        for b in basins:
            ds = res[b]["1D"]["xr"]
            index = pd.to_datetime(ds["date"].values)
            o = pd.Series(ds[f"{TARGET}_obs"].values[:, 0], index=index, dtype="float64")
            if b in obs:
                assert np.allclose(obs[b].values, o.values, equal_nan=True), b
            obs[b] = o
            sims[b][run.name] = pd.Series(ds[f"{TARGET}_sim"].values[:, 0], index=index, dtype="float64")
    return obs, sims


# ---------------------------------------------------------------- checks
def check_scaler(run_dirs, cfg, data, periods):
    """Recompute the normalisation statistics from candidate periods and compare with the saved scaler."""
    scalers = [yaml_load(r / "train_data/train_data_scaler.yml") for r in run_dirs]
    center = scalers[0]["xarray_feature_center"]["data_vars"]
    scale = scalers[0]["xarray_feature_scale"]["data_vars"]
    variables = cfg["dynamic_inputs"] + [TARGET]
    identical = all(
        np.isclose(s["xarray_feature_center"]["data_vars"][v]["data"], center[v]["data"], rtol=1e-12)
        and np.isclose(s["xarray_feature_scale"]["data_vars"][v]["data"], scale[v]["data"], rtol=1e-12)
        for s in scalers for v in variables)
    tr0, tr1 = periods["train"]
    # The library reads forcing from seq_length-1 days before the first training target day (clipped to
    # the start of the record) so that the first target day has a full input window.
    warmup0 = str((pd.Timestamp(tr0) - pd.Timedelta(days=cfg["seq_length"] - 1)).date())
    variants = {
        "training period (forcing incl. warm-up)": (warmup0, tr1, tr0, tr1),
        "training + validation period": (warmup0, periods["validation"][1], tr0, periods["validation"][1]),
        "training + validation + test period": (warmup0, periods["test"][1], tr0, periods["test"][1]),
    }
    rows = []
    for name, (f0, f1, t0, t1) in variants.items():
        for v in variables:
            a, e = (t0, t1) if v == TARGET else (f0, f1)
            x = np.concatenate([df.loc[a:e, v].to_numpy("float64") for df in data.values()])
            if v == TARGET:
                x = np.clip(x, 0, None)
            for stat, saved, mine in (("center", center[v]["data"], np.nanmean(x)), ("scale", scale[v]["data"], np.nanstd(x))):
                rows.append(dict(variant=name, variable=v, statistic=stat, saved=saved, recomputed=mine,
                                 rel_diff=abs(mine - saved) / abs(saved)))
    table = pd.DataFrame(rows)
    by_variant = table.groupby("variant", sort=False).rel_diff.max()
    summary = dict(
        identical_across_seeds=bool(identical),
        max_rel_diff_training=float(by_variant.iloc[0]),
        max_rel_diff_train_val=float(by_variant.iloc[1]),
        max_rel_diff_all_periods=float(by_variant.iloc[2]),
    )
    attrs = pd.read_csv(ROOT / "data/processed/attributes/attributes.csv", dtype={"gauge_id": str}).set_index("gauge_id")
    statics = cfg["static_attributes"]
    summary["static_means_max_rel_diff"] = float(max(
        abs(attrs[k].mean() - scalers[0]["attribute_means"][k]) / (abs(attrs[k].mean()) + 1e-12) for k in statics))
    return table, summary


def _run_lengths(mask):
    out, n = [], 0
    for x in mask:
        if x:
            n += 1
        elif n:
            out.append(n)
            n = 0
    if n:
        out.append(n)
    return out


def _climate_indices_from_raw(basin, start, end):
    """Climate indices written independently of scripts/prepare_data.py, from the raw CSV."""
    d = pd.read_csv(RAW / f"timeseries/observation_based/CAMELS_CH_obs_based_{basin}.csv",
                    index_col="date", parse_dates=True).loc[start:end]
    p, t = d["precipitation(mm/d)"], d["temperature_mean(degC)"]
    w = 2 * np.pi * d.index.dayofyear.values / 365.25
    X = np.c_[np.ones(len(w)), np.sin(w), np.cos(w)]
    cp = np.linalg.lstsq(X, p.values, rcond=None)[0]
    ct = np.linalg.lstsq(X, t.values, rcond=None)[0]
    high, low, years = (p >= 5 * p.mean()).values, (p < 1).values, len(d) / 365.25
    return dict(
        p_mean=p.mean(), t_mean=t.mean(),
        p_seasonality=np.hypot(cp[1], cp[2]) / p.mean() * np.cos(np.arctan2(cp[2], cp[1]) - np.arctan2(ct[2], ct[1])),
        frac_snow=p[t < 0].sum() / p.sum(),
        high_prec_freq=high.sum() / years, high_prec_dur=np.mean(_run_lengths(high)),
        low_prec_freq=low.sum() / years, low_prec_dur=np.mean(_run_lengths(low)))


def check_climate_indices(basins, periods):
    attrs = pd.read_csv(ROOT / "data/processed/attributes/attributes.csv", dtype={"gauge_id": str}).set_index("gauge_id")
    train = pd.DataFrame({b: _climate_indices_from_raw(b, *periods["train"]) for b in basins}).T
    full = pd.DataFrame({b: _climate_indices_from_raw(b, periods["train"][0], periods["test"][1]) for b in basins}).T
    rows = [dict(index=k,
                 max_abs_diff_training_period=float((attrs.loc[basins, k] - train[k]).abs().max()),
                 n_basins_closer_to_full_record=int(((attrs.loc[basins, k] - full[k]).abs() < (attrs.loc[basins, k] - train[k]).abs()).sum()),
                 median_abs_diff_training_vs_full=float((train[k] - full[k]).abs().median()))
            for k in train.columns]
    return pd.DataFrame(rows)


def check_inputs_and_selection(run_dirs):
    rows, summary = [], {}
    for run in run_dirs:
        cfg = yaml_load(run / "config.yml")
        test_dirs = sorted(p.name for p in (run / "test").iterdir())
        val = {}
        for d in sorted((run / "validation").iterdir()):
            f = d / "validation_metrics.csv"
            if f.exists():
                val[int(d.name[-3:])] = float(pd.read_csv(f).NSE.median())
        for epoch, v in val.items():
            rows.append(dict(run=run.name, epoch=epoch, validation_median_NSE=v, evaluated_on_test=f"model_epoch{epoch:03d}" in test_dirs))
        summary[run.name] = dict(
            seed=cfg["seed"], epochs=cfg["epochs"], dynamic_inputs=cfg["dynamic_inputs"],
            n_static_attributes=len(cfg["static_attributes"]),
            static_attributes_in_signatures=sorted(set(cfg["static_attributes"]) & SIGNATURES),
            lagged_or_autoregressive_inputs=bool(cfg.get("lagged_features") or cfg.get("autoregressive_inputs")),
            test_epochs_evaluated=test_dirs,
            best_validation_epoch=max(val, key=val.get) if val else None)
    return pd.DataFrame(rows), summary


def skill_tables(basins, obs, sims, data, periods):
    """Per-basin skill of the seed-mean simulation against the day-of-year climatology."""
    tr0, tr1 = periods["train"]
    rows, frames = [], {}
    for b in basins:
        df = pd.DataFrame({"obs": obs[b]})
        members = list(sims[b])
        for m in members:
            df[m] = sims[b][m]
        df["ens"] = df[members].clip(lower=0).mean(axis=1)
        q_train = data[b].loc[tr0:tr1, TARGET].astype("float64")
        clim = q_train.groupby(doy_key(q_train.index)).mean()
        if 229 not in clim.index or np.isnan(clim.get(229, np.nan)):
            clim.loc[229] = (clim.get(228, np.nan) + clim.get(301, np.nan)) / 2
        df["clim"] = clim.reindex(doy_key(df.index)).values
        df = df[df.obs.notna()]
        frames[b] = df
        o, e, c = df.obs.values, df.ens.values, df.clim.values
        c_test = df.obs.groupby(doy_key(df.index)).transform("mean").values
        sse = ((e - o) ** 2).sum()
        rows.append(dict(
            basin=b, n_valid=len(o), last_date=str(df.index.max().date()),
            NSE=nse(o, e), NSE_climatology=nse(o, c),
            skill_vs_climatology=1 - sse / ((o - c) ** 2).sum(),
            skill_vs_test_period_climatology=1 - sse / ((o - c_test) ** 2).sum(),
            NSE_anomalies=nse(o - c, e - c), r_anomalies=np.corrcoef(o - c, e - c)[0, 1],
            r=np.corrcoef(o, e)[0, 1], alpha=e.std() / o.std(), beta=e.mean() / o.mean(),
            NSE_bias_removed=nse(o, e - e.mean() + o.mean())))
    return pd.DataFrame(rows).set_index("basin"), frames


def group_table(per_basin):
    flags = pd.read_csv(ROOT / "results/data_report/basin_flags.csv", dtype={"gauge_id": str}).set_index("gauge_id")
    attrs = pd.read_csv(ROOT / "data/processed/attributes/attributes.csv", dtype={"gauge_id": str}).set_index("gauge_id")
    meta = pd.read_csv(ROOT / "results/data_report/basin_metadata.csv", dtype={"gauge_id": str}).set_index("gauge_id")
    X = per_basin.join(flags).join(attrs[["glac_frac", "elev_mean", "frac_snow"]]).join(meta[["country", "area"]])
    groups = [("All basins", pd.Series(True, index=X.index))]
    for flag, label in [("near_natural", "Near-natural"), ("regulated", "Regulated"), ("reservoir_major", "Major reservoir"),
                        ("lake", "Lake-influenced"), ("water_balance_open", "Open water balance"),
                        ("human_info_missing", "Outside Switzerland (human influence unknown)")]:
        groups.append((label, X[flag]))
        if flag == "near_natural":
            groups.append(("Not near-natural", ~X[flag]))
    groups += [("No glacier", X.glac_frac == 0), ("Glacier cover > 5 %", X.glac_frac > 0.05), ("Glacier cover > 20 %", X.glac_frac > 0.20),
               ("Snow fraction < 0.1", X.frac_snow < 0.1), ("Snow fraction > 0.5", X.frac_snow > 0.5),
               ("Mean elevation < 700 m", X.elev_mean < 700), ("Mean elevation 700-1200 m", X.elev_mean.between(700, 1200)),
               ("Mean elevation 1200-1800 m", X.elev_mean.between(1200, 1800, inclusive="right")),
               ("Mean elevation 1800-2400 m", X.elev_mean.between(1800, 2400, inclusive="right")),
               ("Mean elevation > 2400 m", X.elev_mean > 2400),
               ("Area < 50 km2", X.area < 50), ("Area 50-200 km2", X.area.between(50, 200)),
               ("Area 200-1000 km2", X.area.between(200, 1000, inclusive="right")), ("Area > 1000 km2", X.area > 1000)]
    rows = []
    for label, mask in groups:
        g = X[mask]
        rows.append(dict(group=label, n_basins=len(g), median_NSE=g.NSE.median(), p25_NSE=g.NSE.quantile(0.25), min_NSE=g.NSE.min(),
                         median_NSE_climatology=g.NSE_climatology.median(),
                         median_skill_vs_climatology=g.skill_vs_climatology.median(),
                         median_NSE_anomalies=g.NSE_anomalies.median()))
    return pd.DataFrame(rows), X


def per_year_tables(frames, flags):
    rows = []
    for b, df in frames.items():
        for hy, g in df.groupby(hydro_year(df.index)):
            if len(g) >= 300:
                rows.append(dict(basin=b, hydro_year=int(hy), n_valid=len(g), NSE=nse(g.obs.values, g.ens.values),
                                 obs_mean=g.obs.mean(), obs_std=g.obs.std()))
    py = pd.DataFrame(rows)
    py["near_natural"] = py.basin.map(flags.near_natural)
    n_years = py.hydro_year.nunique()
    complete = py.groupby("basin").size().pipe(lambda s: s[s == n_years].index)
    worst = py.loc[py.NSE.idxmin(), "basin"]
    summary = pd.DataFrame({
        "n_basins": py.groupby("hydro_year").size(),
        "median_NSE": py.groupby("hydro_year").NSE.median(),
        "mean_NSE": py.groupby("hydro_year").NSE.mean(),
        "mean_NSE_without_worst_basin": py[py.basin != worst].groupby("hydro_year").NSE.mean(),
        "median_NSE_basins_with_all_years": py[py.basin.isin(complete)].groupby("hydro_year").NSE.median(),
    }).reset_index()
    info = dict(worst_basin=worst, n_basins_with_all_years=int(len(complete)),
                max_abs_diff_median_all_vs_complete=float((summary.median_NSE - summary.median_NSE_basins_with_all_years).abs().max()),
                worst_basin_years=py[py.basin == worst].set_index("hydro_year")[["NSE", "obs_mean", "obs_std"]].round(4).to_dict("index"))
    return py, summary, info


def record_endings(per_basin, test_end):
    meta = pd.read_csv(ROOT / "results/data_report/basin_metadata.csv", dtype={"gauge_id": str}).set_index("gauge_id")
    early = per_basin[per_basin.last_date < test_end].join(meta.country)
    table = (early.assign(last_year=early.last_date.str[:4]).groupby(["last_year", "country"])
             .agg(n_basins=("NSE", "size"), median_NSE=("NSE", "median")).reset_index())
    return table, int(len(early))


def clipping_effect(basins, obs, sims):
    rows = []
    for b in basins:
        for member, s in sims[b].items():
            m = obs[b].notna()
            o, x = obs[b][m].values, s[m].values
            rows.append(dict(basin=b, member=member, n_negative=int((s < 0).sum()), min_sim=float(s.min()),
                             date_of_min=str(s.idxmin().date()), NSE_unclipped=nse(o, x), NSE_clipped=nse(o, np.clip(x, 0, None))))
    t = pd.DataFrame(rows)
    t["abs_diff"] = (t.NSE_clipped - t.NSE_unclipped).abs()
    med = t.groupby("member")[["NSE_unclipped", "NSE_clipped"]].median()
    w = t.loc[t.abs_diff.idxmax()]
    summary = dict(n_negative_values=int(t.n_negative.sum()), n_basin_member_pairs_with_negatives=int((t.n_negative > 0).sum()),
                   n_pairs_changed_by_more_than_0p01=int((t.abs_diff > 0.01).sum()),
                   largest_change=dict(basin=w.basin, member=w.member, NSE_unclipped=float(w.NSE_unclipped), NSE_clipped=float(w.NSE_clipped),
                                       min_sim=float(w.min_sim), date_of_min=w.date_of_min,
                                       other_members_same_day={m: float(s[pd.Timestamp(w.date_of_min)]) for m, s in sims[w.basin].items() if m != w.member}),
                   median_NSE_by_member=med.round(6).to_dict("index"))
    return t, summary


def forcing_homogeneity(data, cfg, periods):
    """Change of the annual statistics of each forcing variable between the training and the test period."""
    variables = cfg["dynamic_inputs"] + SIM_FORCING
    annual = {}
    for b, df in data.items():
        df = df.loc[periods["train"][0]:periods["test"][1]]
        hy = hydro_year(df.index)
        d = {(v, "mean"): df[v].groupby(hy).mean() for v in variables}
        d.update({(v, "std"): df[v].groupby(hy).std() for v in variables})
        d[("precipitation", "fraction of days in (0, 0.1) mm")] = ((df.precipitation > 0) & (df.precipitation < 0.1)).groupby(hy).mean()
        annual[b] = pd.DataFrame(d)
    P = pd.concat(annual, names=["basin", "hydro_year"])
    hy = P.index.get_level_values("hydro_year")
    last_train, first_test = pd.Timestamp(periods["train"][1]).year, pd.Timestamp(periods["test"][0]).year + 1
    train, test = P[hy <= last_train].groupby("basin").mean(), P[hy >= first_test].groupby("basin").mean()
    median_series = P.groupby("hydro_year").median()
    rows = []
    for col in P.columns:
        s = median_series[col]
        steps = {}
        for y in range(s.index.min() + 5, s.index.max() - 3):
            a, c = s.loc[y - 5:y - 1], s.loc[y:y + 4]
            steps[y] = (c.mean() - a.mean()) / np.sqrt((a.var() + c.var()) / 2 + 1e-12)
        steps = pd.Series(steps)
        absolute = col[0].startswith("temperature") and col[1] == "mean"
        change = test[col] - train[col] if absolute else 100 * (test[col] / train[col] - 1)
        rows.append(dict(variable=col[0], statistic=col[1], used_as_input=col[0] in cfg["dynamic_inputs"],
                         train_median=train[col].median(), test_median=test[col].median(),
                         change_unit="absolute" if absolute else "percent", median_change=change.median(),
                         largest_step_hydro_year=int(steps.abs().idxmax()), largest_step_statistic=float(steps.abs().max())))
    return pd.DataFrame(rows)


def worst_basins(X, data, periods, n=6):
    meta = pd.read_csv(ROOT / "results/data_report/basin_metadata.csv", dtype={"gauge_id": str}).set_index("gauge_id")
    flag_cols = ["regulated", "reservoir_major", "glacier", "lake", "water_balance_open", "human_info_missing"]
    rows = []
    for b, r in X.sort_values("NSE").head(n).iterrows():
        df = data[b]
        stats = {}
        for name in ("train", "test"):
            a, e = periods[name]
            q, p = df.loc[a:e, TARGET], df.loc[a:e, "precipitation"]
            v = q.notna()
            stats[name] = dict(qp=q[v].mean() / p[v].mean(), q_mean=q.mean(), q_std=q.std(), repeat=float((q.diff() == 0)[v].mean()))
        rows.append(dict(
            basin=b, gauge=meta.gauge_name[b], river=meta.water_body_name[b], country=meta.country[b], area_km2=meta.area[b],
            flags=", ".join(f for f in flag_cols if r[f]) or "none",
            NSE=r.NSE, r=r.r, alpha=r.alpha, beta=r.beta, NSE_bias_removed=r.NSE_bias_removed,
            QP_ratio_train=stats["train"]["qp"], QP_ratio_test=stats["test"]["qp"],
            Q_mean_train=stats["train"]["q_mean"], Q_mean_test=stats["test"]["q_mean"],
            Q_std_train=stats["train"]["q_std"], Q_std_test=stats["test"]["q_std"],
            frac_days_equal_to_previous_train=stats["train"]["repeat"], frac_days_equal_to_previous_test=stats["test"]["repeat"]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "results/audit")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    run_dirs = [r if r.is_absolute() else ROOT / r for r in args.run_dirs]

    basins = (ROOT / "splits/basins_all.txt").read_text().split()
    periods = {k: [str(pd.Timestamp(d).date()) for d in v] for k, v in yaml_load(ROOT / "splits/periods.yml").items()}
    cfg = yaml_load(run_dirs[0] / "config.yml")
    flags = pd.read_csv(ROOT / "results/data_report/basin_flags.csv", dtype={"gauge_id": str}).set_index("gauge_id")

    data = load_processed(basins)
    obs, sims = load_results(run_dirs, basins)
    summary = dict(runs=[r.name for r in run_dirs], n_basins=len(basins), periods=periods)

    table, summary["scaler"] = check_scaler(run_dirs, cfg, data, periods)
    table.to_csv(args.out / "scaler_check.csv", index=False)

    table = check_climate_indices(basins, periods)
    table.to_csv(args.out / "climate_indices_check.csv", index=False)
    summary["climate_indices"] = dict(max_abs_diff_training_period=float(table.max_abs_diff_training_period.max()),
                                      n_basins_closer_to_full_record=int(table.n_basins_closer_to_full_record.sum()))

    table, summary["runs_detail"] = check_inputs_and_selection(run_dirs)
    table.to_csv(args.out / "validation_by_epoch.csv", index=False)

    per_basin, frames = skill_tables(basins, obs, sims, data, periods)
    groups, X = group_table(per_basin)
    per_basin.to_csv(args.out / "skill_vs_climatology.csv")
    groups.to_csv(args.out / "skill_by_group.csv", index=False)
    q = per_basin[["NSE", "NSE_climatology", "skill_vs_climatology", "skill_vs_test_period_climatology", "NSE_anomalies", "r_anomalies"]]
    summary["seasonality"] = dict(
        median=q.median().to_dict(), p25=q.quantile(0.25).to_dict(), p05=q.quantile(0.05).to_dict(),
        n_basins_skill_vs_climatology_below_0=int((per_basin.skill_vs_climatology < 0).sum()),
        n_basins_skill_vs_climatology_below_0p5=int((per_basin.skill_vs_climatology < 0.5).sum()))
    low = X[X.NSE < 0.5]
    summary["basins_below_0p5"] = dict(n=int(len(low)), n_near_natural=int(low.near_natural.sum()),
                                       min_NSE_near_natural=float(X[X.near_natural].NSE.min()))

    py, table, summary["per_year"] = per_year_tables(frames, flags)
    py.to_csv(args.out / "per_basin_year.csv", index=False)
    table.to_csv(args.out / "per_year.csv", index=False)

    table, summary["n_basins_ending_before_test_end"] = record_endings(per_basin, periods["test"][1])
    table.to_csv(args.out / "record_endings.csv", index=False)

    table, summary["clipping"] = clipping_effect(basins, obs, sims)
    table.to_csv(args.out / "clipping_effect.csv", index=False)

    forcing_homogeneity(data, cfg, periods).to_csv(args.out / "forcing_homogeneity.csv", index=False)
    worst_basins(X, data, periods).to_csv(args.out / "worst_basins.csv", index=False)

    (args.out / "audit_summary.json").write_text(json.dumps(summary, indent=2, default=str), newline="\n")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
