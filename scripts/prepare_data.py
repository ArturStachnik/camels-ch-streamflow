"""Convert raw CAMELS-CH into the NeuralHydrology generic dataset layout.

Writes, under data/processed/ (not tracked by git):
    time_series/<gauge_id>.nc     one netCDF per basin, coordinate `date`
    attributes/attributes.csv     static attributes, no streamflow-derived columns

Writes, under splits/ and results/data_report/ (tracked by git):
    splits/basins_all.txt             basins used in the experiments
    splits/periods.yml                train / validation / test periods
    splits/SHA256SUMS                 hashes of the split files and of attributes.csv
    results/data_report/*.csv         basin report with quality-control columns, basin flags,
                                      dropped attributes, nesting metadata

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
MIN_VALID_DAYS = {"train": 5 * 365, "validation": 2 * 365, "test": 3 * 365}
MAX_QP_RATIO = 2.0  # mean discharge / mean precipitation in the training period

# Manual exclusions, each with a documented reason.
EXCLUSIONS = {
    "2446": "Zihlkanal: bidirectional channel between lakes (dataset readme)",
    "2447": "Canal de la Broye: bidirectional channel between lakes (dataset readme)",
    "2327": "Dischmabach: irregular discharge since 1999 (dataset readme)",
    "4008": "discharge series identical to gauge 4009 on a different river (design audit)",
    "4009": "discharge series identical to gauge 4008 on a different river (design audit)",
    "2613": "Weil: same discharge and forcing series as 2289 Basel, which is kept (design audit)",
}

TARGET = "discharge_spec(mm/d)"
# Primary forcing: MeteoSwiss observation-based grids, homogeneous over 1981-2020.
FORCING = {
    "precipitation(mm/d)": "precipitation",
    "temperature_min(degC)": "temperature_min",
    "temperature_mean(degC)": "temperature_mean",
    "temperature_max(degC)": "temperature_max",
    "rel_sun_dur(%)": "rel_sun_dur",
}
# Forcing of the "simulation_based" folder (meteorology that drives PREVAH). Kept only for a
# sensitivity run: the product changes on 2010-01-01 (wind +65 %, PET +19 %, different daily
# precipitation window), i.e. between the training and the test period. PREVAH model output
# (discharge_*_sim, et_sim, intercept_*) is never stored: PREVAH is calibrated on discharge.
FORCING_SIM = {
    "precipitation_sim(mm/d)": "precipitation_sim",
    "temperature_sim(degC)": "temperature_sim",
    "radiation_sim(W/m2)": "radiation_sim",
    "rel_humidity_sim(%)": "rel_humidity_sim",
    "wind_sim(m/s)": "wind_sim",
    "pet_sim(mm/d)": "pet_sim",
}

# Attribute tables that are derived from observed or simulated streamflow. Never loaded.
STREAMFLOW_DERIVED_FILES = [
    "CAMELS_CH_hydrology_attributes_obs.csv",
    "simulation_based/CAMELS_CH_hydrology_attributes_sim.csv",
]
# Shipped climate indices cover 1981-2020 (includes the test period): recomputed below instead.
TEST_PERIOD_DERIVED_FILES = [
    "CAMELS_CH_climate_attributes_obs.csv",
    "simulation_based/CAMELS_CH_climate_attributes_sim.csv",
]
# Hydropower and reservoir attributes are only populated inside Switzerland (SFOE data), so
# they are not used as model inputs. They are used to flag basins when reporting results.
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
    "CAMELS_CH_humaninfluence_attributes.csv": ["dens_inhabitants"],
}


def read_static(rel: str) -> pd.DataFrame:
    df = pd.read_csv(RAW / "static_attributes" / rel, comment="#", encoding="latin-1", dtype={"gauge_id": str})
    return df.set_index("gauge_id")


def _run_lengths(mask: np.ndarray) -> np.ndarray:
    """Lengths of runs of consecutive True values."""
    edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)


def _mean_run_length(mask: np.ndarray) -> float:
    lengths = _run_lengths(mask)
    return float(lengths.mean()) if len(lengths) else 0.0


def climate_indices(df: pd.DataFrame, period: tuple) -> dict:
    """Climate indices after Addor et al. (2017), from the forcing of `period` only.

    Uses observed precipitation and mean temperature. No PET-based index is included:
    the only PET in the dataset belongs to the forcing product that changes in 2010.
    Only forcing is used, never discharge.
    """
    d = df.loc[period[0]:period[1]]
    p, t = d["precipitation"], d["temperature_mean"]
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
        "t_mean": t.mean(),
        "p_seasonality": delta_p * np.cos(phase_diff),
        "frac_snow": p[t < 0].sum() / p.sum(),
        "high_prec_freq": high.sum() / years,
        "high_prec_dur": _mean_run_length(high),
        "low_prec_freq": low.sum() / years,
        "low_prec_dur": _mean_run_length(low),
    }


def quality_control(df: pd.DataFrame) -> dict:
    """Discharge quality indicators from the training period (no test data involved)."""
    d = df.loc[PERIODS["train"][0]:PERIODS["train"][1]]
    q = d["discharge_spec"]
    valid = q.notna()
    if valid.sum() == 0:
        return {"qp_ratio_train": np.nan, "q_zero_frac_train": np.nan, "q_max_constant_run_train": 0}
    same = (q.diff() == 0).to_numpy()
    runs = _run_lengths(same)
    return {
        "qp_ratio_train": round(float(q[valid].mean() / d.loc[valid, "precipitation"].mean()), 3),
        "q_zero_frac_train": round(float((q[valid] == 0).mean()), 4),
        "q_max_constant_run_train": int(runs.max()) + 1 if len(runs) else 1,
    }


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
            df[name] = obs[src]
        for src, name in FORCING_SIM.items():
            df[name] = sim[src]
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
        }
        for name, period in PERIODS.items():
            row[f"q_valid_days_{name}"] = int(q.loc[period[0]:period[1]].notna().sum())
        row.update(quality_control(df))
        rows.append(row)
        if row["forcing_nan_days"] == 0:
            climate[gid] = climate_indices(df, PERIODS["train"])
    report = pd.DataFrame(rows).set_index("gauge_id")

    # ---- basin selection (first matching reason wins) --------------------------------
    reason = pd.Series("", index=report.index)

    def exclude(mask, why):
        reason[(reason == "") & mask] = why

    exclude(report.water_body_type == "lake", "lake (dataset readme)")
    for gid, why in EXCLUSIONS.items():
        exclude(report.index == gid, why)
    exclude(report.forcing_nan_days > 0, "observation-based forcing incomplete")
    for name, days in MIN_VALID_DAYS.items():
        exclude(report[f"q_valid_days_{name}"] < days, f"fewer than {days} valid discharge days in {name} period")
    exclude(report.q_negative_days > 0, "negative discharge")
    exclude(report.qp_ratio_train > MAX_QP_RATIO,
            f"discharge exceeds {MAX_QP_RATIO:g} x precipitation in training period "
            "(contributing area does not match the topographic catchment)")
    report["excluded_reason"] = reason
    report["selected"] = reason == ""
    report.to_csv(REPORT / "basin_report.csv")
    basins = sorted(report.index[report.selected])

    # ---- static attributes -----------------------------------------------------------
    parts = [read_static(rel)[cols] for rel, cols in STATIC_SOURCES.items()]
    glac = read_static("CAMELS_CH_glacier_attributes.csv")
    attrs = pd.concat(parts + [pd.DataFrame.from_dict(climate, orient="index")], axis=1)
    attrs["glac_frac"] = (glac["glac_area"].fillna(0) + glac["glac_area_neighbours"].fillna(0)) / attrs["area"]
    attrs = attrs.loc[basins]
    nan_cols = attrs.columns[attrs.isna().any()].tolist()
    const_cols = [c for c in attrs.columns if c not in nan_cols and attrs[c].std() == 0]
    attrs = attrs.drop(columns=nan_cols + const_cols).round(6)
    attrs.index.name = "gauge_id"
    attr_file = OUT / "attributes" / "attributes.csv"
    attrs.to_csv(attr_file, lineterminator="\n")
    dropped = (
        [{"attribute": f"all columns of {f}", "reason": "derived from streamflow"} for f in STREAMFLOW_DERIVED_FILES]
        + [{"attribute": f"all columns of {f}", "reason": "computed over 1981-2020, includes test period; "
            "recomputed from training-period forcing"} for f in TEST_PERIOD_DERIVED_FILES]
        + [{"attribute": c, "reason": "hydropower/reservoir data only populated inside Switzerland"}
           for c in ("hp_count", "num_reservoir", "reservoir_cap")]
        + [{"attribute": c, "reason": "NaN in selected basins"} for c in nan_cols]
        + [{"attribute": c, "reason": "constant in selected basins"} for c in const_cols])
    pd.DataFrame(dropped).to_csv(REPORT / "dropped_attributes.csv", index=False, lineterminator="\n")
    pd.Series(attrs.columns, name="attribute").to_csv(
        REPORT / "static_attributes_used.csv", index=False, lineterminator="\n")

    # ---- flags for reporting results with and without influenced basins ------------------
    human = read_static("CAMELS_CH_humaninfluence_attributes.csv").loc[basins]
    land = read_static("CAMELS_CH_landcover_attributes.csv").loc[basins]
    area = topo.loc[basins, "area"].astype(float)
    flags = pd.DataFrame(index=pd.Index(basins, name="gauge_id"))
    flags["regulated"] = (human.hp_count > 0) | (human.num_reservoir > 0)
    # reservoir_cap in megalitres, area in km2: ML / km2 = mm
    flags["reservoir_major"] = (human.reservoir_cap.fillna(0) / area) > 50
    flags["glacier"] = attrs["glac_frac"] > 0.05
    flags["lake"] = land.inwater_perc > 2
    flags["water_balance_open"] = ~report.loc[basins, "qp_ratio_train"].between(0.25, 1.2)
    flags["human_info_missing"] = topo.loc[basins, "country"] != "CH"
    flags["near_natural"] = ~flags[["regulated", "glacier", "lake", "water_balance_open",
                                    "human_info_missing"]].any(axis=1)
    flags.to_csv(REPORT / "basin_flags.csv", lineterminator="\n")

    # Metadata for the nesting analysis (setting 2).
    meta = supp[["hierarchy_lower", "hierarchy_upper", "river_basin", "sub_gauge_id", "sub_area"]].join(
        topo[["country", "gauge_name", "water_body_name", "gauge_lon", "gauge_lat", "area"]])
    meta.loc[basins].to_csv(REPORT / "basin_metadata.csv", lineterminator="\n")

    # ---- splits (written once) -------------------------------------------------------
    basin_file = SPLITS / "basins_all.txt"
    periods_file = SPLITS / "periods.yml"
    if basin_file.exists() and not args.force_splits:
        if basin_file.read_text().split() != basins:
            raise SystemExit("splits/basins_all.txt exists and differs from the new selection. "
                             "Splits are frozen; use --force-splits only with approval.")
        print("split files already exist and match; not rewritten")
    else:
        basin_file.write_text("\n".join(basins) + "\n", newline="\n")
        lines = [f"{k}: ['{a}', '{b}']" for k, (a, b) in PERIODS.items()]
        periods_file.write_text("# hydrological years, inclusive dates\n" + "\n".join(lines) + "\n", newline="\n")
    # Recorded hashes are checked, never silently replaced: a rerun (on another machine, or
    # after a library update) must reproduce the frozen files byte for byte.
    sums_file = SPLITS / "SHA256SUMS"
    recorded = {}
    if sums_file.exists():
        for line in sums_file.read_text().splitlines():
            digest, name = line.split()
            recorded[name] = digest
    for p in (basin_file, periods_file, attr_file):
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        name = p.as_posix()
        if name in recorded and recorded[name] != digest and not args.force_splits:
            raise SystemExit(f"{name} does not reproduce its recorded SHA-256 ({recorded[name][:12]}... vs {digest[:12]}...). "
                             "Stop and compare before going on.")
        recorded[name] = digest
    sums_file.write_text("".join(f"{d}  {n}\n" for n, d in recorded.items()), newline="\n")

    print(f"basins in dataset: {len(report)}; selected: {len(basins)}")
    print(report.loc[~report.selected, "excluded_reason"].value_counts().to_string())
    print(f"static attributes used: {attrs.shape[1]}; dropped for NaN: {nan_cols}; constant: {const_cols}")
    print(report.loc[report.selected].groupby("country").size().to_string())
    print(flags.sum().to_string())


if __name__ == "__main__":
    main()
