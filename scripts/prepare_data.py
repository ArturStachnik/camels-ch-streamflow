"""Convert raw CAMELS-CH into the NeuralHydrology generic dataset layout.

Writes, under data/processed/ (not tracked by git):
    time_series/<gauge_id>.nc     one netCDF per basin, coordinate `date`
    attributes/attributes.csv     static attributes, no streamflow-derived columns

Writes, under splits/ and results/data_report/ (tracked by git):
    splits/basins_all.txt             basins used in the experiments
    splits/periods.yml                train / validation / test periods
    splits/SHA256SUMS                 hash of the split files (frozen after first write)
    results/data_report/*.csv         missing-data report, basin selection, dropped attributes

Split files are written once. If they already exist the script refuses to overwrite
them unless --force-splits is given.

Usage: python scripts/prepare_data.py
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

RAW = Path("data/raw/camels_ch_v0.9/camels_ch")
OUT = Path("data/processed")
SPLITS = Path("splits")
REPORT = Path("results/data_report")

# Hydrological years (1 Oct - 30 Sep).
PERIODS = {
    "train": ("1981-10-01", "2005-09-30"),
    "validation": ("2005-10-01", "2010-09-30"),
    "test": ("2010-10-01", "2020-09-30"),
}
MIN_TRAIN_DAYS = 5 * 365  # valid discharge days required in the training period
MIN_EVAL_DAYS = 2 * 365   # valid discharge days required in validation and in test

# Excluded on the recommendation of the dataset readme.
README_EXCLUSIONS = {
    "2446": "Zihlkanal: bidirectional channel between lakes, negative discharge",
    "2447": "Canal de la Broye: bidirectional channel between lakes, negative discharge",
    "2327": "Dischmabach: irregular discharge since 1999 (dataset readme)",
}

TARGET = "discharge_spec(mm/d)"
# Meteorological forcing. The "simulation_based" folder holds the gridded meteorology
# that drives PREVAH; it is complete for all 331 catchments, whereas the MeteoSwiss
# observation-based temperature and sunshine grids are missing for about 55 foreign
# catchments. PREVAH model output (discharge_*_sim, et_sim, intercept_*) is never used:
# PREVAH is calibrated against observed discharge.
FORCING = {
    "precipitation_sim(mm/d)": "precipitation",
    "temperature_sim(degC)": "temperature",
    "radiation_sim(W/m2)": "radiation",
    "rel_humidity_sim(%)": "rel_humidity",
    "wind_sim(m/s)": "wind",
    "pet_sim(mm/d)": "pet",
}
# Observation-based forcing is also stored, for a later sensitivity run on Swiss basins.
FORCING_OBS = {
    "precipitation(mm/d)": "precipitation_obs",
    "temperature_min(degC)": "temperature_min_obs",
    "temperature_mean(degC)": "temperature_mean_obs",
    "temperature_max(degC)": "temperature_max_obs",
    "rel_sun_dur(%)": "rel_sun_dur_obs",
}

# Attribute tables that are derived from observed or simulated streamflow. Never loaded.
STREAMFLOW_DERIVED_FILES = [
    "CAMELS_CH_hydrology_attributes_obs.csv",
    "simulation_based/CAMELS_CH_hydrology_attributes_sim.csv",
]
STATIC_SOURCES = {
    "CAMELS_CH_topographic_attributes.csv": [
        "area", "elev_mean", "slope_mean", "flat_area_perc", "steep_area_perc"],
    "CAMELS_CH_landcover_attributes.csv": [
        "crop_perc", "grass_perc", "scrub_perc", "dwood_perc", "mixed_wood_perc", "ewood_perc",
        "wetlands_perc", "inwater_perc", "ice_perc", "loose_rock_perc", "rock_perc", "urban_perc"],
    "CAMELS_CH_geology_attributes.csv": [
        "geo_porosity", "geo_log10_permeability", "unconsol_sediments", "siliciclastic_sedimentary",
        "mixed_sedimentary", "carbonate_sedimentary", "acid_plutonic", "metamorphics"],
    "CAMELS_CH_soil_attributes.csv": [
        "sand_perc", "silt_perc", "clay_perc", "organic_perc", "bulk_dens", "tot_avail_water",
        "root_depth", "coarse_fragm_perc", "porosity", "conductivity"],
    "CAMELS_CH_glacier_attributes.csv": ["glac_area"],
    "CAMELS_CH_humaninfluence_attributes.csv": ["dens_inhabitants", "hp_count", "num_reservoir", "reservoir_cap"],
}


def read_static(rel: str) -> pd.DataFrame:
    df = pd.read_csv(RAW / "static_attributes" / rel, comment="#", encoding="latin-1", dtype={"gauge_id": str})
    return df.set_index("gauge_id")


def _mean_run_length(mask: np.ndarray) -> float:
    """Mean length of runs of consecutive True values."""
    edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    return float(lengths.mean()) if len(lengths) else 0.0


def climate_indices(df: pd.DataFrame, period: tuple) -> dict:
    """Climate indices after Addor et al. (2017), from the forcing of `period` only.

    The indices shipped with CAMELS-CH cover 1981-2020, which includes the test period.
    They are recomputed here from the training period so that no static input depends
    on test-period data. Only forcing is used, never discharge.
    """
    d = df.loc[period[0]:period[1]]
    p, t, pet = d["precipitation"], d["temperature"], d["pet"]
    # Seasonality: least-squares sine fits to daily precipitation and temperature.
    w = 2 * np.pi * d.index.dayofyear.to_numpy() / 365.25
    X = np.column_stack([np.ones_like(w), np.sin(w), np.cos(w)])
    (_, ap, bp), (_, at, bt) = (np.linalg.lstsq(X, v.to_numpy(), rcond=None)[0] for v in (p, t))
    delta_p = np.hypot(ap, bp) / p.mean()
    phase_diff = np.arctan2(bp, ap) - np.arctan2(bt, at)
    high = (p >= 5 * p.mean()).to_numpy()
    low = (p < 1.0).to_numpy()
    years = len(d) / 365.25
    return {
        "p_mean": p.mean(),
        "pet_mean": pet.mean(),
        "aridity": pet.mean() / p.mean(),
        "p_seasonality": delta_p * np.cos(phase_diff),
        "frac_snow": p[t < 0].sum() / p.sum(),
        "high_prec_freq": high.sum() / years,
        "high_prec_dur": _mean_run_length(high),
        "low_prec_freq": low.sum() / years,
        "low_prec_dur": _mean_run_length(low),
    }


def valid_days(q: pd.Series, period: tuple) -> int:
    return int(q.loc[period[0]:period[1]].notna().sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-splits", action="store_true")
    args = parser.parse_args()
    for d in (OUT / "time_series", OUT / "attributes", SPLITS, REPORT):
        d.mkdir(parents=True, exist_ok=True)

    topo = read_static("CAMELS_CH_topographic_attributes.csv")
    supp = read_static("supplements/CAMELS_CH_topographic_attributes_supplement.csv")

    # ---- time series ----------------------------------------------------------------
    rows, climate = [], {}
    for gid in topo.index:
        obs = pd.read_csv(RAW / f"timeseries/observation_based/CAMELS_CH_obs_based_{gid}.csv",
                          index_col="date", parse_dates=True)
        sim = pd.read_csv(RAW / f"timeseries/simulation_based/CAMELS_CH_sim_based_{gid}.csv",
                          index_col="date", parse_dates=True)
        assert obs.index.equals(sim.index), gid
        df = pd.DataFrame(index=obs.index)
        df["discharge_spec"] = obs[TARGET]
        for src, name in FORCING.items():
            df[name] = sim[src]
        for src, name in FORCING_OBS.items():
            df[name] = obs[src]
        df.index.name = "date"
        xr.Dataset.from_dataframe(df.astype("float32")).to_netcdf(OUT / "time_series" / f"{gid}.nc")

        q = df["discharge_spec"]
        row = {
            "gauge_id": gid,
            "country": topo.loc[gid, "country"],
            "water_body_type": topo.loc[gid, "water_body_type"],
            "q_first_valid": q.first_valid_index(),
            "q_last_valid": q.last_valid_index(),
            "q_negative_days": int((q < 0).sum()),
            "forcing_nan_days": int(df[list(FORCING.values())].isna().any(axis=1).sum()),
            "forcing_obs_nan_frac": round(float(df[list(FORCING_OBS.values())].isna().any(axis=1).mean()), 4),
        }
        for name, period in PERIODS.items():
            row[f"q_valid_days_{name}"] = valid_days(q, period)
        rows.append(row)
        climate[gid] = climate_indices(df, PERIODS["train"])
    report = pd.DataFrame(rows).set_index("gauge_id")

    # ---- basin selection -------------------------------------------------------------
    reason = pd.Series("", index=report.index)
    reason[report.water_body_type == "lake"] = "lake (dataset readme)"
    for gid, why in README_EXCLUSIONS.items():
        reason[gid] = why
    short = ((report.q_valid_days_train < MIN_TRAIN_DAYS)
             | (report.q_valid_days_validation < MIN_EVAL_DAYS)
             | (report.q_valid_days_test < MIN_EVAL_DAYS))
    reason[(reason == "") & short] = (
        f"discharge record too short (<{MIN_TRAIN_DAYS} d train or <{MIN_EVAL_DAYS} d validation/test)")
    reason[(reason == "") & (report.forcing_nan_days > 0)] = "missing forcing"
    reason[(reason == "") & (report.q_negative_days > 0)] = "negative discharge"
    report["excluded_reason"] = reason
    report["selected"] = reason == ""
    report.to_csv(REPORT / "basin_report.csv")
    basins = sorted(report.index[report.selected])

    # ---- static attributes -----------------------------------------------------------
    parts = [read_static(rel)[cols] for rel, cols in STATIC_SOURCES.items()]
    attrs = pd.concat(parts + [pd.DataFrame.from_dict(climate, orient="index")], axis=1)
    attrs["glac_frac"] = attrs.pop("glac_area") / attrs["area"]
    attrs["reservoir_cap"] = attrs["reservoir_cap"].fillna(0.0)
    sel = attrs.loc[basins]
    nan_cols = sel.columns[sel.isna().any()].tolist()
    const_cols = [c for c in sel.columns if c not in nan_cols and sel[c].std() == 0]
    attrs = attrs.drop(columns=nan_cols + const_cols)
    attrs.index.name = "gauge_id"
    attrs.to_csv(OUT / "attributes" / "attributes.csv")
    dropped = (
        [{"attribute": f"all columns of {f}", "reason": "derived from streamflow"} for f in STREAMFLOW_DERIVED_FILES]
        + [{"attribute": c, "reason": "NaN in selected basins"} for c in nan_cols]
        + [{"attribute": c, "reason": "constant in selected basins"} for c in const_cols])
    pd.DataFrame(dropped).to_csv(REPORT / "dropped_attributes.csv", index=False)
    pd.Series(attrs.columns, name="attribute").to_csv(REPORT / "static_attributes_used.csv", index=False)

    # Metadata for the audit: nesting hierarchy and human influence, not used as inputs here.
    meta = supp[["hierarchy_lower", "hierarchy_upper", "river_basin", "sub_gauge_id", "sub_area"]].join(
        topo[["country", "gauge_name", "water_body_name", "gauge_lon", "gauge_lat", "area"]])
    meta.loc[basins].to_csv(REPORT / "basin_metadata.csv")

    # ---- splits (written once) -------------------------------------------------------
    basin_file = SPLITS / "basins_all.txt"
    periods_file = SPLITS / "periods.yml"
    if basin_file.exists() and not args.force_splits:
        existing = basin_file.read_text().split()
        if existing != basins:
            raise SystemExit("splits/basins_all.txt exists and differs from the new selection. "
                             "Splits are frozen; use --force-splits only with approval.")
        print("splits already exist and match; not rewritten")
    else:
        basin_file.write_text("\n".join(basins) + "\n")
        lines = [f"{k}: ['{a}', '{b}']" for k, (a, b) in PERIODS.items()]
        periods_file.write_text("# hydrological years, inclusive dates\n" + "\n".join(lines) + "\n")
        sums = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}" for p in (basin_file, periods_file)]
        (SPLITS / "SHA256SUMS").write_text("\n".join(sums) + "\n")

    print(f"basins in dataset: {len(report)}; selected: {len(basins)}")
    print(report.loc[~report.selected, "excluded_reason"].value_counts().to_string())
    print(f"static attributes used: {attrs.shape[1]}; dropped for NaN: {nan_cols}; constant: {const_cols}")
    print(report.loc[report.selected].groupby("country").size().to_string())


if __name__ == "__main__":
    main()
