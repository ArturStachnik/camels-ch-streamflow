#!/usr/bin/env python
"""Independent recomputation of evaluation metrics from saved model outputs.

This script does not import the modelling library and does not use any of the
modeller's code. It reads the result pickles written at evaluation time (they
unpickle to plain dicts holding xarray Datasets), recomputes every metric from
its published definition, compares against the metrics file written by the
library, checks the observations against the raw CAMELS-CH files, computes two
benchmarks from raw data, and writes a machine-readable report.

Usage (from the repository root):

    python verification/verify_metrics.py \
        --run-dirs runs/a runs/b runs/c --period test --epoch 30 \
        --out results/verification

Exit code is 0 only if no hard check failed. See verification/README.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Metric definitions. All functions take two 1-D float arrays of equal length
# that are already paired and free of NaN (see pair_valid).
# --------------------------------------------------------------------------

FHV_FRACTION = 0.02   # top 2 % of the flow duration curve (Yilmaz et al. 2008)
FLV_FRACTION = 0.30   # bottom 30 % of the flow duration curve (Yilmaz et al. 2008)
FLV_FLOOR = 1e-6      # mm/d; flows <= 0 are replaced by this before taking logs
NEAR_ZERO = 0.01      # mm/d; flows below this are reported as "near zero"
LOG_NSE_EPS = 0.01    # mm/d; added to obs and sim before the log in log-NSE


def pair_valid(obs, sim):
    """Return obs and sim restricted to the days where both are finite.

    Days with missing observations are excluded pairwise. Missing simulations
    are excluded as well so that the metrics stay finite, but they are counted
    and reported as a failed check elsewhere.
    """
    obs = np.asarray(obs, dtype=np.float64).ravel()
    sim = np.asarray(sim, dtype=np.float64).ravel()
    if obs.shape != sim.shape:
        raise ValueError("obs and sim must have the same length")
    keep = np.isfinite(obs) & np.isfinite(sim)
    return obs[keep], sim[keep]


def nse(obs, sim):
    """Nash-Sutcliffe efficiency (Nash & Sutcliffe 1970).

    NSE = 1 - sum((sim - obs)^2) / sum((obs - mean(obs))^2)
    """
    obs = np.asarray(obs, dtype=np.float64)
    sim = np.asarray(sim, dtype=np.float64)
    if obs.size < 2:
        return np.nan
    denom = np.sum((obs - obs.mean()) ** 2)
    if denom == 0:
        return np.nan
    return float(1.0 - np.sum((sim - obs) ** 2) / denom)


def pearson_r(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2:
        return np.nan
    da = a - a.mean()
    db = b - b.mean()
    denom = np.sqrt(np.sum(da ** 2) * np.sum(db ** 2))
    if denom == 0:
        return np.nan
    return float(np.sum(da * db) / denom)


def kge(obs, sim):
    """Kling-Gupta efficiency (Gupta et al. 2009), 2009 formulation.

    KGE = 1 - sqrt((r - 1)^2 + (alpha - 1)^2 + (beta - 1)^2)
    r     = linear correlation between sim and obs
    alpha = std(sim) / std(obs)
    beta  = mean(sim) / mean(obs)

    Returns (kge, r, alpha, beta). The ratio of standard deviations does not
    depend on the choice of N or N-1 in the denominator.
    """
    obs = np.asarray(obs, dtype=np.float64)
    sim = np.asarray(sim, dtype=np.float64)
    if obs.size < 2:
        return np.nan, np.nan, np.nan, np.nan
    r = pearson_r(obs, sim)
    so = obs.std()
    mo = obs.mean()
    alpha = float(sim.std() / so) if so != 0 else np.nan
    beta = float(sim.mean() / mo) if mo != 0 else np.nan
    value = 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
    return float(value), r, alpha, beta


def _segment_size(n, fraction):
    return max(int(np.round(fraction * n)), 1)


def fhv(obs, sim, fraction=FHV_FRACTION):
    """Percent bias of the high-flow segment of the flow duration curve.

    Yilmaz et al. (2008), eq. for %BiasFHV. Both series are sorted separately
    in descending order (two flow duration curves) and the top `fraction` of
    each is compared:

    FHV = 100 * sum(QS_h - QO_h) / sum(QO_h)
    """
    obs = np.asarray(obs, dtype=np.float64)
    sim = np.asarray(sim, dtype=np.float64)
    if obs.size < 1:
        return np.nan
    k = _segment_size(obs.size, fraction)
    o = np.sort(obs)[::-1][:k]
    s = np.sort(sim)[::-1][:k]
    denom = o.sum()
    if denom == 0:
        return np.nan
    return float(100.0 * np.sum(s - o) / denom)


def _low_segments(obs, sim, fraction):
    k = _segment_size(obs.size, fraction)
    o = np.sort(obs)[:k]   # the k lowest observed flows, ascending
    s = np.sort(sim)[:k]   # the k lowest simulated flows, ascending
    return o, s


def flv(obs, sim, fraction=FLV_FRACTION, floor=FLV_FLOOR):
    """Percent bias of the low-flow segment of the flow duration curve in log space.

    Yilmaz et al. (2008), eq. for %BiasFLV. Both series are sorted separately
    and the lowest `fraction` of each is used. With L the index of the minimum
    flow of each segment:

    FLV = -100 * ( sum(log QS_l - log QS_L) - sum(log QO_l - log QO_L) )
                 / sum(log QO_l - log QO_L)

    Zero handling: any flow <= 0 in either segment is replaced by `floor`
    (default 1e-6 mm/d) before the logarithm. The published definition does
    not say what to do with zeros, so this is a choice, and the result depends
    strongly on it: a single zero puts log(floor) = -13.8 into the reference
    term of every element of the sum. If the observed segment is constant the
    denominator is zero and NaN is returned.
    """
    obs = np.asarray(obs, dtype=np.float64)
    sim = np.asarray(sim, dtype=np.float64)
    if obs.size < 1:
        return np.nan
    o, s = _low_segments(obs, sim, fraction)
    o = np.log(np.where(o <= 0, floor, o))
    s = np.log(np.where(s <= 0, floor, s))
    qol = np.sum(o - o.min())
    qsl = np.sum(s - s.min())
    if qol == 0:
        return np.nan
    return float(-100.0 * (qsl - qol) / qol)


def flv_segment_info(obs, sim, fraction=FLV_FRACTION, near_zero=NEAR_ZERO):
    """Describe the low-flow segments so that unstable FLV values can be spotted."""
    obs = np.asarray(obs, dtype=np.float64)
    sim = np.asarray(sim, dtype=np.float64)
    if obs.size < 1:
        return {}
    o, s = _low_segments(obs, sim, fraction)
    info = {
        "FLV_n_segment": float(o.size),
        "FLV_n_obs_zero": float(np.sum(o <= 0)),
        "FLV_n_sim_zero": float(np.sum(s <= 0)),
        "FLV_n_obs_near_zero": float(np.sum(o < near_zero)),
        "FLV_n_sim_near_zero": float(np.sum(s < near_zero)),
        "FLV_min_obs": float(o.min()),
        "FLV_min_sim": float(s.min()),
    }
    info["FLV_segment_has_zero_or_near_zero"] = float(
        info["FLV_n_obs_near_zero"] > 0 or info["FLV_n_sim_near_zero"] > 0
    )
    return info


def pbias(obs, sim):
    """Percent bias of total volume: 100 * sum(sim - obs) / sum(obs)."""
    obs = np.asarray(obs, dtype=np.float64)
    sim = np.asarray(sim, dtype=np.float64)
    denom = obs.sum() if obs.size else 0.0
    if denom == 0:
        return np.nan
    return float(100.0 * np.sum(sim - obs) / denom)


def log_nse(obs, sim, eps=LOG_NSE_EPS):
    """NSE of log(q + eps). Negative flows are clipped to zero before the log."""
    obs = np.asarray(obs, dtype=np.float64)
    sim = np.asarray(sim, dtype=np.float64)
    return nse(np.log(np.clip(obs, 0, None) + eps), np.log(np.clip(sim, 0, None) + eps))


def all_metrics(obs, sim, log_eps=LOG_NSE_EPS, flv_floor=FLV_FLOOR, near_zero=NEAR_ZERO):
    """Compute every metric for one basin. Inputs may contain NaN."""
    o, s = pair_valid(obs, sim)
    k, r, a, b = kge(o, s)
    out = {
        "NSE": nse(o, s),
        "KGE": k,
        "KGE_r": r,
        "KGE_alpha": a,
        "KGE_beta": b,
        "FHV": fhv(o, s),
        "FLV": flv(o, s, floor=flv_floor),
        "PBIAS": pbias(o, s),
        "logNSE": log_nse(o, s, eps=log_eps),
    }
    out.update(flv_segment_info(o, s, near_zero=near_zero))
    return out, int(o.size)


SUMMARY_METRICS = ["NSE", "KGE", "KGE_r", "KGE_alpha", "KGE_beta", "FHV", "FLV", "PBIAS", "logNSE"]
EFFICIENCY_METRICS = {"NSE", "KGE", "logNSE"}   # metrics for which "< 0" is counted

# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_periods(path):
    """Parse splits/periods.yml (lines of the form  name: ['YYYY-MM-DD', 'YYYY-MM-DD'])."""
    periods = {}
    pat = re.compile(r"^\s*(\w+)\s*:\s*\[\s*['\"]?(\d{4}-\d{2}-\d{2})['\"]?\s*,\s*['\"]?(\d{4}-\d{2}-\d{2})['\"]?\s*\]")
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = pat.match(line)
        if m:
            periods[m.group(1)] = (pd.Timestamp(m.group(2)), pd.Timestamp(m.group(3)))
    return periods


def read_flat_config(path):
    """Read top-level 'key: value' pairs of a run config without a YAML library."""
    cfg = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Za-z_]\w*)\s*:\s*(.*?)\s*$", line)
        if m and m.group(2) != "":
            cfg[m.group(1)] = m.group(2).strip("'\"")
    return cfg


def read_basin_file(path):
    return [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]


def hydro_year(dates):
    """Hydrological year October-September, labelled by the year in which it ends."""
    dates = pd.DatetimeIndex(dates)
    return dates.year + (dates.month >= 10).astype(int)


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if np.isfinite(x) else None
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.isoformat()
    if isinstance(x, Path):
        # repository-relative where possible, so that reports do not record local directories
        repo = Path(__file__).resolve().parents[1]
        x = x.resolve()
        return x.relative_to(repo).as_posix() if x.is_relative_to(repo) else x.name
    return x


class Report:
    """Collects checks. status is 'pass', 'fail' (hard, sets exit code) or 'finding' (soft)."""

    def __init__(self):
        self.checks = []

    def add(self, check_id, description, ok, details=None, hard=True, alarm=False):
        status = "pass" if ok else ("fail" if hard else "finding")
        self.checks.append({
            "id": check_id, "description": description, "status": status,
            "hard": bool(hard), "alarm": bool(alarm), "details": details if details is not None else {},
        })
        tag = {"pass": "PASS   ", "fail": "FAIL   ", "finding": "FINDING"}[status]
        if alarm and not ok:
            print("!" * 78)
            print(f"!!! ALARM  {check_id}: {description}")
            print(f"!!!        {_short(details)}")
            print("!" * 78)
        else:
            print(f"[{tag}] {check_id}: {description}" + ("" if ok else f"\n          {_short(details)}"))
        return ok

    @property
    def n_failed(self):
        return sum(c["status"] == "fail" for c in self.checks)


def _short(details, limit=400):
    text = json.dumps(jsonable(details), default=str)
    return text if len(text) <= limit else text[:limit] + " ... (truncated, see verification_report.json)"


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_results(path, target):
    """Read a results pickle into {basin: DataFrame(index=date, columns=[obs, sim])}.

    Expected structure: dict basin -> dict frequency -> dict with key 'xr'
    holding an xarray Dataset with variables <target>_obs and <target>_sim on
    dims (date, time_step). The daily runs here have one frequency and a
    single time_step. Dates are returned exactly as stored (not de-duplicated).
    """
    with open(path, "rb") as fh:
        raw = pickle.load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a dict, got {type(raw)}")
    out, notes = {}, {"frequencies": set(), "time_steps": set(), "non_midnight_basins": []}
    for basin, per_freq in raw.items():
        freqs = list(per_freq.keys())
        freq = "1D" if "1D" in per_freq else freqs[0]
        notes["frequencies"].update(freqs)
        ds = per_freq[freq]["xr"]
        obs_name, sim_name = f"{target}_obs", f"{target}_sim"
        if obs_name not in ds or sim_name not in ds:
            raise ValueError(f"{path}: basin {basin} lacks {obs_name}/{sim_name}; has {list(ds.data_vars)}")
        if "time_step" in ds.dims:
            notes["time_steps"].add(int(ds.sizes["time_step"]))
            ds = ds.isel(time_step=-1)
        dates = pd.DatetimeIndex(ds["date"].values)
        if (dates != dates.normalize()).any():
            notes["non_midnight_basins"].append(str(basin))
        out[str(basin)] = pd.DataFrame(
            {"obs": np.asarray(ds[obs_name].values, dtype=np.float64).ravel(),
             "sim": np.asarray(ds[sim_name].values, dtype=np.float64).ravel()},
            index=dates.rename("date"),
        )
    notes["frequencies"] = sorted(notes["frequencies"])
    notes["time_steps"] = sorted(notes["time_steps"])
    return out, notes


class RawData:
    """Lazy reader for the raw CAMELS-CH per-basin daily files."""

    OBS_COL = "discharge_spec(mm/d)"          # observed specific discharge
    SIM_COL = "discharge_spec_sim(mm/d)"      # simulated specific discharge shipped with the dataset

    def __init__(self, raw_dir):
        self.raw_dir = Path(raw_dir)
        self.obs_dir = self._find("observation_based")
        self.sim_dir = self._find("simulation_based")
        self._obs, self._sim = {}, {}
        self.files_used = {}

    def _find(self, name):
        # several folders share these names (e.g. under static_attributes);
        # take the one that sits in the time series folder
        hits = sorted(p for p in self.raw_dir.rglob(name) if p.is_dir() and p.parent.name == "timeseries")
        return hits[0] if hits else None

    def _read(self, folder, pattern, basin, col):
        if folder is None:
            return None
        path = folder / pattern.format(basin=basin)
        if not path.exists():
            return None
        df = pd.read_csv(path, usecols=["date", col], parse_dates=["date"])
        self.files_used[path.as_posix()] = sha256_file(path)
        s = df.set_index("date")[col].astype(np.float64)
        if s.index.has_duplicates:
            raise ValueError(f"duplicated dates in raw file {path}")
        return s

    def obs(self, basin):
        if basin not in self._obs:
            self._obs[basin] = self._read(self.obs_dir, "CAMELS_CH_obs_based_{basin}.csv", basin, self.OBS_COL)
        return self._obs[basin]

    def sim(self, basin):
        if basin not in self._sim:
            self._sim[basin] = self._read(self.sim_dir, "CAMELS_CH_sim_based_{basin}.csv", basin, self.SIM_COL)
        return self._sim[basin]

    def combined_hash(self):
        h = hashlib.sha256()
        for name in sorted(self.files_used):
            h.update(f"{self.files_used[name]}  {name}\n".encode())
        return h.hexdigest()


# --------------------------------------------------------------------------
# Individual checks
# --------------------------------------------------------------------------


def integrity_rows(member, data, start, end):
    """Per-basin date and value checks for one member."""
    expected = len(pd.date_range(start, end, freq="D"))
    rows = []
    for basin, df in data.items():
        idx = df.index
        rows.append({
            "member": member, "basin": basin,
            "n_dates": len(idx),
            "first_date": idx.min().date().isoformat() if len(idx) else "",
            "last_date": idx.max().date().isoformat() if len(idx) else "",
            "n_outside_period": int(((idx < start) | (idx > end)).sum()),
            "n_duplicated_dates": int(idx.duplicated().sum()),
            "n_missing_days_in_period": int(expected - idx.normalize().unique().isin(pd.date_range(start, end, freq="D")).sum()),
            "n_obs_nan": int(df["obs"].isna().sum()),
            "n_sim_nan": int((~np.isfinite(df["sim"].values)).sum()),
            "n_sim_negative": int((df["sim"] < 0).sum()),
            "min_sim": float(np.nanmin(df["sim"].values)) if len(idx) else np.nan,
            "sim_identical_to_obs": bool(_identical(df)),
        })
    return rows


def _identical(df):
    o, s = pair_valid(df["obs"].values, df["sim"].values)
    return o.size > 0 and bool(np.max(np.abs(o - s)) < 1e-9)


def obs_vs_raw_rows(member, data, raw, tol):
    """Compare observations stored with the results against the raw files, date by date."""
    rows = []
    for basin, df in data.items():
        row = {"member": member, "basin": basin, "raw_file_found": False}
        series = raw.obs(basin)
        if series is None:
            rows.append(row)
            continue
        df = df[~df.index.duplicated()]
        dates = df.index.normalize()
        r0 = series.reindex(dates).values
        p = df["obs"].values
        both = np.isfinite(p) & np.isfinite(r0)
        row.update({
            "raw_file_found": True,
            "n_compared": int(both.sum()),
            "n_dates_not_in_raw": int((~dates.isin(series.index)).sum()),
            "max_abs_diff": float(np.max(np.abs(p[both] - r0[both]))) if both.any() else np.nan,
            "n_above_tol": int(np.sum(np.abs(p[both] - r0[both]) > tol)),
            # results have a value where the raw file has none: cannot be an observation
            "n_value_where_raw_nan": int(np.sum(np.isfinite(p) & ~np.isfinite(r0))),
            # results have no value where the raw file has one: data were dropped upstream
            "n_nan_where_raw_value": int(np.sum(~np.isfinite(p) & np.isfinite(r0))),
        })
        # The same comparison against the raw series moved by one day in each
        # direction. If one of these is near zero and the unshifted one is not,
        # the observations are misaligned by a day.
        for name, days in (("max_abs_diff_raw_day_before", -1), ("max_abs_diff_raw_day_after", 1)):
            rs = series.reindex(dates + pd.Timedelta(days=days)).values
            ok = np.isfinite(p) & np.isfinite(rs)
            row[name] = float(np.max(np.abs(p[ok] - rs[ok]))) if ok.any() else np.nan
        rows.append(row)
    return rows


def shift_rows(member, data):
    """Correlation of sim with obs on the same day and with obs one day earlier / later."""
    rows = []
    for basin, df in data.items():
        df = df[~df.index.duplicated()]
        if df.empty:
            continue
        full = pd.date_range(df.index.min(), df.index.max(), freq="D")
        obs = df["obs"].reindex(full)
        sim = df["sim"].reindex(full)

        def corr(o):
            a, b = pair_valid(o.values, sim.values)
            return pearson_r(a, b)

        r0 = corr(obs)
        r_next = corr(obs.shift(-1))   # sim on day t against obs on day t+1
        r_prev = corr(obs.shift(1))    # sim on day t against obs on day t-1
        rows.append({
            "member": member, "basin": basin, "r_same_day": r0,
            "r_sim_vs_obs_next_day": r_next, "r_sim_vs_obs_previous_day": r_prev,
            "shifted_higher": bool(np.nanmax([r_next, r_prev, -np.inf]) > r0) if np.isfinite(r0) else False,
        })
    return rows


def summarise(values, metric):
    v = np.asarray(values, dtype=np.float64)
    n_nan = int(np.sum(~np.isfinite(v)))
    v = v[np.isfinite(v)]
    row = {"n_basins": int(v.size), "n_nan": n_nan}
    if v.size == 0:
        row.update({k: np.nan for k in ["median", "mean", "p05", "p25", "p75", "p95", "min", "max", "n_below_zero"]})
        return row
    p05, p25, p75, p95 = np.percentile(v, [5, 25, 75, 95])
    row.update({
        "median": float(np.median(v)), "mean": float(v.mean()),
        "p05": float(p05), "p25": float(p25), "p75": float(p75), "p95": float(p95),
        "min": float(v.min()), "max": float(v.max()),
        "n_below_zero": int(np.sum(v < 0)) if metric in EFFICIENCY_METRICS else np.nan,
    })
    return row


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def parse_args(argv=None):
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run-dirs", nargs="+", required=True, help="one run directory per seed")
    p.add_argument("--period", required=True, choices=["train", "validation", "test"])
    p.add_argument("--epoch", required=True, type=int)
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--repo-root", default=str(here.parent), help="repository root (default: parent of this script's folder)")
    p.add_argument("--basin-file", default=None, help="expected basin list (default: splits/basins_all.txt)")
    p.add_argument("--allow-basin-mismatch", action="store_true",
                   help="report a basin-set mismatch as a finding instead of a failure (fixtures only)")
    p.add_argument("--periods-file", default=None, help="default: splits/periods.yml")
    p.add_argument("--sha256sums", default=None, help="default: splits/SHA256SUMS")
    p.add_argument("--raw-dir", default=None, help="default: data/raw/camels_ch_v0.9")
    p.add_argument("--flags-file", default=None, help="default: results/data_report/basin_flags.csv")
    p.add_argument("--target", default="discharge_spec", help="target variable name in the result files")
    p.add_argument("--tol-nse-kge", type=float, default=1e-3)
    p.add_argument("--tol-fhv", type=float, default=0.1, help="percentage points")
    p.add_argument("--tol-flv", type=float, default=0.1, help="percentage points; reported, not a hard check")
    p.add_argument("--tol-obs", type=float, default=5e-4,
                   help="mm/d; allowed |obs in results - raw obs| (raw files have 0.001 mm/d resolution)")
    p.add_argument("--log-eps", type=float, default=LOG_NSE_EPS, help="mm/d added before the log in log-NSE")
    p.add_argument("--flv-floor", type=float, default=FLV_FLOOR, help="mm/d replacing flows <= 0 in FLV")
    p.add_argument("--near-zero", type=float, default=NEAR_ZERO, help="mm/d below which a flow is reported as near zero")
    p.add_argument("--sim-clip", choices=["none", "zero"], default="none",
                   help="'none' (default) evaluates the simulations exactly as saved; 'zero' sets negative "
                        "simulated values to zero before any metric is computed")
    p.add_argument("--min-days-per-year", type=int, default=300,
                   help="minimum valid days for a basin to enter a hydrological-year NSE")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    root = Path(args.repo_root).resolve()

    def repo_rel(path):
        """Path relative to the repository root, so that reports do not record local directories."""
        path = Path(path).resolve()
        return path.relative_to(root).as_posix() if path.is_relative_to(root) else path.name

    def rp(value, default):
        path = Path(value) if value else root / default
        return path if path.is_absolute() else (Path.cwd() / path)

    basin_file = rp(args.basin_file, "splits/basins_all.txt")
    periods_file = rp(args.periods_file, "splits/periods.yml")
    sums_file = rp(args.sha256sums, "splits/SHA256SUMS")
    raw_dir = rp(args.raw_dir, "data/raw/camels_ch_v0.9")
    flags_file = rp(args.flags_file, "results/data_report/basin_flags.csv")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rep = Report()
    input_hashes = {}

    def record_hash(path):
        path = Path(path)
        if path.exists():
            input_hashes[repo_rel(path)] = sha256_file(path)

    # ---- frozen split definition --------------------------------------------------
    periods = read_periods(periods_file)
    record_hash(periods_file)
    record_hash(basin_file)
    record_hash(flags_file)
    record_hash(sums_file)
    if args.period not in periods:
        print(f"period '{args.period}' not found in {periods_file}", file=sys.stderr)
        return 2
    start, end = periods[args.period]
    print(f"Period '{args.period}': {start.date()} to {end.date()} (from {repo_rel(periods_file)})")

    sums_details, sums_ok = [], sums_file.exists()
    if sums_file.exists():
        for line in sums_file.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            expected, rel = parts[0].lower(), parts[-1].lstrip("*")
            target = root / rel
            actual = sha256_file(target) if target.exists() else None
            sums_details.append({"file": rel, "expected": expected, "actual": actual, "match": actual == expected})
            sums_ok &= actual == expected
    rep.add("split_sha256", f"files listed in {sums_file.name} match their SHA-256", sums_ok,
            {"files": sums_details} if sums_file.exists() else {"error": f"{sums_file} not found"})

    # ---- load results --------------------------------------------------------------
    members, lib_metrics, configs = {}, {}, {}
    for run in args.run_dirs:
        run = Path(run)
        name = run.name
        if name in members:
            print(f"duplicate run directory name: {name}", file=sys.stderr)
            return 2
        res_dir = run / args.period / f"model_epoch{args.epoch:03d}"
        res_file = res_dir / f"{args.period}_results.p"
        met_file = res_dir / f"{args.period}_metrics.csv"
        if not res_file.exists():
            print(f"missing results file: {res_file}", file=sys.stderr)
            return 2
        data, notes = load_results(res_file, args.target)
        members[name] = data
        record_hash(res_file)
        print(f"Loaded {name}: {len(data)} basins from {repo_rel(res_file)}")
        rep.add(f"structure[{name}]", "results hold one daily frequency with a single time step at midnight dates",
                notes["frequencies"] == ["1D"] and notes["time_steps"] in ([1], []) and not notes["non_midnight_basins"],
                notes, hard=False)
        if met_file.exists():
            lib = pd.read_csv(met_file, dtype={"basin": str}).set_index("basin")
            lib_metrics[name] = lib
            record_hash(met_file)
        cfg_file = run / "config.yml"
        if cfg_file.exists():
            configs[name] = read_flat_config(cfg_file)
            record_hash(cfg_file)

    seed_names = list(members)

    # ---- run configuration against the frozen split (context, soft) ----------------
    for name, cfg in configs.items():
        try:
            c_start = pd.to_datetime(cfg.get(f"{args.period}_start_date"), format="%d/%m/%Y")
            c_end = pd.to_datetime(cfg.get(f"{args.period}_end_date"), format="%d/%m/%Y")
            ok = (c_start == start) and (c_end == end)
            det = {"config_start": c_start, "config_end": c_end, "periods_start": start, "periods_end": end,
                   "seed": cfg.get("seed"), f"{args.period}_basin_file": cfg.get(f"{args.period}_basin_file")}
        except Exception as exc:  # unreadable config is reported, not fatal
            ok, det = False, {"error": repr(exc)}
        rep.add(f"config_period[{name}]", f"{args.period} dates in the run's config.yml equal periods.yml", ok, det, hard=False)
    seeds = {n: configs[n].get("seed") for n in configs}
    if len(seed_names) > 1:
        rep.add("distinct_seeds", "the runs use different random seeds",
                len(seeds) == len(seed_names) and len(set(seeds.values())) == len(seed_names), {"seeds": seeds}, hard=False)

    # ---- basin set -------------------------------------------------------------------
    expected_basins = read_basin_file(basin_file) if basin_file.exists() else []
    exp_set = set(expected_basins)
    for name in seed_names:
        got = set(members[name])
        missing, extra = sorted(exp_set - got), sorted(got - exp_set)
        rep.add(f"basin_set[{name}]", f"basins in results equal {basin_file.name} exactly",
                basin_file.exists() and not missing and not extra and len(expected_basins) == len(exp_set),
                {"n_expected": len(expected_basins), "n_in_results": len(got), "n_missing_from_results": len(missing),
                 "n_not_in_basin_file": len(extra), "missing_from_results": missing, "not_in_basin_file": extra,
                 "duplicates_in_basin_file": len(expected_basins) - len(exp_set)},
                hard=not args.allow_basin_mismatch)

    # ---- date and value integrity ------------------------------------------------------
    integ = pd.DataFrame([r for n in seed_names for r in integrity_rows(n, members[n], start, end)])
    integ.to_csv(out_dir / "integrity.csv", index=False)

    def bad(col):
        sub = integ[integ[col] > 0]
        return {"n_basin_member_pairs": int(len(sub)), "total": int(sub[col].sum()),
                "examples": sub[["member", "basin", col]].head(20).to_dict("records")}

    rep.add("dates_inside_period", f"every date lies inside {start.date()}..{end.date()}",
            (integ["n_outside_period"] == 0).all(), bad("n_outside_period"))
    rep.add("no_duplicated_dates", "no basin has duplicated dates", (integ["n_duplicated_dates"] == 0).all(), bad("n_duplicated_dates"))
    rep.add("period_fully_covered", "every basin has one row for every day of the period",
            (integ["n_missing_days_in_period"] == 0).all(), bad("n_missing_days_in_period"), hard=False)
    rep.add("no_nan_in_sim", "simulations contain no NaN or infinite value", (integ["n_sim_nan"] == 0).all(), bad("n_sim_nan"))
    neg = bad("n_sim_negative")
    neg["most_negative_sim"] = float(integ["min_sim"].min())
    rep.add("sim_non_negative", "simulations are non-negative", (integ["n_sim_negative"] == 0).all(), neg, hard=False)

    # Work on de-duplicated, date-sorted frames from here on (duplicates were reported above).
    for name in seed_names:
        members[name] = {b: df[~df.index.duplicated()].sort_index() for b, df in members[name].items()}

    # Negative simulated values. The result files can hold them even when the run
    # was configured to clip the target at zero, in which case the library's own
    # metrics were computed on clipped values. Nothing is clipped here unless asked
    # for; the comparison with the library reports both variants.
    if args.sim_clip == "zero":
        for name in seed_names:
            for df in members[name].values():
                df["sim"] = df["sim"].clip(lower=0)
        print("NOTE: --sim-clip zero: negative simulated values were set to zero before computing metrics")

    # ---- observations in results vs raw data -------------------------------------------
    raw = RawData(raw_dir)
    print(f"Raw observed discharge: column '{RawData.OBS_COL}' in {repo_rel(raw.obs_dir)}")
    ovr = pd.DataFrame([r for n in seed_names for r in obs_vs_raw_rows(n, members[n], raw, args.tol_obs)])
    ovr.to_csv(out_dir / "obs_vs_raw.csv", index=False)
    found = ovr[ovr["raw_file_found"]]
    n_no_raw = int((~ovr["raw_file_found"]).sum())
    if len(found):
        worst = found.sort_values("max_abs_diff", ascending=False).head(10)
        bad_rows = found[(found["n_above_tol"] > 0) | (found["n_value_where_raw_nan"] > 0) | (found["n_dates_not_in_raw"] > 0)]
        rep.add("obs_match_raw",
                f"observations in results equal raw '{RawData.OBS_COL}' on the same dates (tolerance {args.tol_obs} mm/d)",
                len(bad_rows) == 0 and n_no_raw == 0 and int(found["n_compared"].sum()) > 0,
                {"raw_column": RawData.OBS_COL, "tolerance_mm_per_day": args.tol_obs,
                 "max_abs_diff_overall": float(found["max_abs_diff"].max()),
                 "n_days_compared": int(found["n_compared"].sum()),
                 "n_basin_member_pairs_failing": int(len(bad_rows)), "n_without_raw_file": n_no_raw,
                 "n_values_where_raw_is_nan": int(found["n_value_where_raw_nan"].sum()),
                 "median_max_abs_diff_if_raw_shifted_one_day_back": float(found["max_abs_diff_raw_day_before"].median()),
                 "median_max_abs_diff_if_raw_shifted_one_day_forward": float(found["max_abs_diff_raw_day_after"].median()),
                 "worst": worst[["member", "basin", "max_abs_diff", "n_above_tol"]].to_dict("records"),
                 "failing": bad_rows[["member", "basin", "max_abs_diff", "n_above_tol", "n_value_where_raw_nan",
                                      "n_dates_not_in_raw"]].head(50).to_dict("records")})
        dropped = found[found["n_nan_where_raw_value"] > 0]
        rep.add("obs_not_dropped", "no observation present in the raw file is missing from the results",
                len(dropped) == 0,
                {"n_basin_member_pairs": int(len(dropped)), "total_days": int(dropped["n_nan_where_raw_value"].sum()),
                 "examples": dropped[["member", "basin", "n_nan_where_raw_value"]].head(20).to_dict("records")}, hard=False)
    else:
        rep.add("obs_match_raw", "observations in results equal the raw observed discharge", False,
                {"error": "no raw observation file found for any basin", "raw_dir": raw_dir})

    # ---- observations identical across seeds, ensemble mean ----------------------------
    reference = members[seed_names[0]]
    if len(seed_names) > 1:
        common = sorted(set.intersection(*[set(members[n]) for n in seed_names]))
        ens, obs_diff = {}, []
        for b in common:
            frames = [members[n][b] for n in seed_names]
            idx = frames[0].index
            for f in frames[1:]:
                idx = idx.intersection(f.index)
            o = np.vstack([f.loc[idx, "obs"].values for f in frames])
            if not np.allclose(o, o[0], rtol=0, atol=0, equal_nan=True) or any(len(f) != len(idx) for f in frames):
                obs_diff.append(b)
            ens[b] = pd.DataFrame({"obs": o[0], "sim": np.mean([f.loc[idx, "sim"].values for f in frames], axis=0)}, index=idx)
        members["ensemble"] = ens
        rep.add("obs_identical_across_seeds", "all runs hold the same dates and observations for every basin",
                not obs_diff and len(common) == max(len(members[n]) for n in seed_names),
                {"n_common_basins": len(common), "n_basins_differing": len(obs_diff), "basins_differing": obs_diff[:50]})
    member_names = list(members)

    # ---- metrics ---------------------------------------------------------------------
    metric_rows, wide = [], {}
    for name in member_names:
        table = {}
        for b, df in members[name].items():
            vals, n_valid = all_metrics(df["obs"].values, df["sim"].values, args.log_eps, args.flv_floor, args.near_zero)
            table[b] = vals
            valid_idx = df.index[np.isfinite(df["obs"].values) & np.isfinite(df["sim"].values)]
            first = valid_idx.min().date().isoformat() if len(valid_idx) else ""
            last = valid_idx.max().date().isoformat() if len(valid_idx) else ""
            for metric, value in vals.items():
                metric_rows.append({"basin": b, "member": name, "metric": metric, "value": value,
                                    "n_valid": n_valid, "first_date": first, "last_date": last})
        wide[name] = pd.DataFrame.from_dict(table, orient="index")
    pd.DataFrame(metric_rows).to_csv(out_dir / "per_basin_metrics.csv", index=False)

    for name in member_names:
        w = wide[name]
        if "FLV_segment_has_zero_or_near_zero" in w:
            flagged = w.index[w["FLV_segment_has_zero_or_near_zero"] > 0].tolist()
            rep.add(f"flv_low_segment[{name}]",
                    f"no basin has zero or near-zero (< {args.near_zero} mm/d) flows in the FLV low-flow segment",
                    not flagged,
                    {"n_basins_flagged": len(flagged), "n_with_obs_zero": int((w["FLV_n_obs_zero"] > 0).sum()),
                     "n_with_sim_zero": int((w["FLV_n_sim_zero"] > 0).sum()), "basins": flagged,
                     "note": "FLV is unstable for these basins; prefer PBIAS and logNSE there"}, hard=False)

    # ---- comparison with the library's metrics file -----------------------------------
    comp_rows = []
    tolerances = {"NSE": args.tol_nse_kge, "KGE": args.tol_nse_kge, "FHV": args.tol_fhv, "FLV": args.tol_flv}
    for name in seed_names:
        if name not in lib_metrics:
            rep.add(f"library_metrics_file[{name}]", f"{args.period}_metrics.csv exists next to the results", False, {})
            continue
        lib = lib_metrics[name]
        mine = wide[name]
        # the same metrics with negative simulated values set to zero, for diagnosis only
        clipped = pd.DataFrame.from_dict(
            {b: all_metrics(df["obs"].values, df["sim"].clip(lower=0).values, args.log_eps, args.flv_floor, args.near_zero)[0]
             for b, df in members[name].items()}, orient="index")
        for metric, tol in tolerances.items():
            if metric not in lib.columns:
                rep.add(f"library_vs_own[{name}][{metric}]", f"{metric} present in the library metrics file", False, {}, hard=False)
                continue
            basins = sorted(set(mine.index) | set(lib.index))
            a = mine[metric].reindex(basins).astype(float)
            l = lib[metric].reindex(basins).astype(float)
            diff = (a - l).abs()
            a_clip = clipped[metric].reindex(basins).astype(float)
            diff_clip = (a_clip - l).abs()
            one_nan = a.isna() ^ l.isna()
            above = (diff > tol) | one_nan
            for b in basins:
                comp_rows.append({"basin": b, "member": name, "metric": metric, "own": a[b], "library": l[b],
                                  "abs_diff": diff[b], "tolerance": tol, "above_tolerance": bool(above[b]),
                                  "own_sim_clipped_at_zero": a_clip[b], "abs_diff_sim_clipped_at_zero": diff_clip[b]})
            details = {"max_abs_diff": float(diff.max()) if diff.notna().any() else None, "tolerance": tol,
                       "n_compared": int(diff.notna().sum()), "n_above_tolerance": int(above.sum()),
                       "n_missing_on_one_side": int(one_nan.sum()),
                       "sim_clip_used": args.sim_clip,
                       "max_abs_diff_if_negative_sim_set_to_zero": float(diff_clip.max()) if diff_clip.notna().any() else None,
                       "basins_above_tolerance": [
                           {"basin": b, "own": a[b], "library": l[b], "abs_diff": diff[b]} for b in above.index[above]]}
            if metric == "FLV":
                details["note"] = ("Not a hard check. This script floors flows <= 0 at "
                                   f"{args.flv_floor} mm/d before the log and uses the exact Yilmaz et al. (2008) "
                                   "denominator; a different zero treatment or a small constant added to the "
                                   "denominator in the library changes FLV, most strongly for basins flagged in "
                                   "flv_low_segment.")
            rep.add(f"library_vs_own[{name}][{metric}]",
                    f"own {metric} agrees with the library's {args.period}_metrics.csv within {tol}",
                    not above.any(), details, hard=metric in ("NSE", "KGE"))
    comp = pd.DataFrame(comp_rows, columns=["basin", "member", "metric", "own", "library", "abs_diff", "tolerance", "above_tolerance",
                                            "own_sim_clipped_at_zero", "abs_diff_sim_clipped_at_zero"])
    comp.to_csv(out_dir / "comparison_with_library.csv", index=False)

    # ---- benchmarks from raw data --------------------------------------------------------
    bench_rows, bench_wide = [], {"benchmark_climatology_doy": {}, "benchmark_prevah": {}}
    t_start, t_end = periods.get("train", (None, None))
    n_no_clim, n_no_prevah = 0, 0
    for b, df in reference.items():
        robs = raw.obs(b)
        if robs is None:
            n_no_clim += 1
            n_no_prevah += 1
            continue
        # "same days": the days on which this basin has a valid observation in the results
        days = df.index[np.isfinite(df["obs"].values)].normalize()
        obs_eval = robs.reindex(days).values
        # (a) day-of-year climatology from the training period only
        if t_start is not None:
            tr = robs.loc[t_start:t_end].dropna()
            if len(tr):
                clim = tr.groupby([tr.index.month, tr.index.day]).mean()
                keys = [(m, d) if (m, d) in clim.index else (2, 28) for m, d in zip(days.month, days.day)]
                csim = clim.reindex(pd.MultiIndex.from_tuples(keys)).values if len(keys) else np.array([])
                vals, n = all_metrics(obs_eval, csim, args.log_eps, args.flv_floor, args.near_zero)
                bench_wide["benchmark_climatology_doy"][b] = vals
                for m in ("NSE", "KGE", "KGE_r", "KGE_alpha", "KGE_beta", "PBIAS", "logNSE"):
                    bench_rows.append({"basin": b, "benchmark": "climatology_doy_train", "metric": m, "value": vals[m],
                                       "n_valid": n, "n_train_days": int(len(tr))})
            else:
                n_no_clim += 1
        # (b) simulated discharge shipped with the dataset (PREVAH)
        rsim = raw.sim(b)
        if rsim is not None:
            vals, n = all_metrics(obs_eval, rsim.reindex(days).values, args.log_eps, args.flv_floor, args.near_zero)
            bench_wide["benchmark_prevah"][b] = vals
            for m in ("NSE", "KGE", "KGE_r", "KGE_alpha", "KGE_beta", "PBIAS", "logNSE"):
                bench_rows.append({"basin": b, "benchmark": "prevah_simulation_based", "metric": m, "value": vals[m],
                                   "n_valid": n, "n_train_days": np.nan})
        else:
            n_no_prevah += 1
    pd.DataFrame(bench_rows, columns=["basin", "benchmark", "metric", "value", "n_valid", "n_train_days"]).to_csv(
        out_dir / "benchmarks.csv", index=False)
    prevah_note = ("PREVAH ('simulation_based' in CAMELS-CH) is a calibrated process-based model whose calibration may "
                   "have used the evaluation period. It is context only, NOT a fair out-of-sample competitor.")
    print("NOTE: " + prevah_note)
    bench_frames = {k: pd.DataFrame.from_dict(v, orient="index") for k, v in bench_wide.items() if v}
    rep.add("benchmarks_computed", "climatology and PREVAH benchmarks could be computed for every basin",
            n_no_clim == 0 and n_no_prevah == 0,
            {"n_basins_without_climatology": n_no_clim, "n_basins_without_prevah": n_no_prevah,
             "climatology": f"mean observed discharge per calendar day over {t_start}..{t_end}, raw column {RawData.OBS_COL}",
             "prevah_column": RawData.SIM_COL, "prevah_note": prevah_note}, hard=False)

    # ---- summaries -----------------------------------------------------------------------
    near_natural = None
    if flags_file.exists():
        flags = pd.read_csv(flags_file, dtype={"gauge_id": str}).set_index("gauge_id")
        if "near_natural" in flags.columns:
            near_natural = flags["near_natural"].astype(str).str.lower().isin(["true", "1"])
    all_result_basins = sorted(set().union(*[set(members[n]) for n in seed_names]))
    unflagged = [b for b in all_result_basins if near_natural is None or b not in near_natural.index]
    rep.add("flags_cover_basins", "every basin in the results has a near_natural flag", not unflagged,
            {"n_without_flag": len(unflagged), "basins": unflagged[:300], "flags_file": flags_file}, hard=False)

    def subsets(index):
        out = {"all": list(index)}
        if near_natural is not None:
            known = [b for b in index if b in near_natural.index]
            out["near_natural"] = [b for b in known if near_natural[b]]
            out["not_near_natural"] = [b for b in known if not near_natural[b]]
        return out

    summary_rows = []
    for name, frame in list(wide.items()) + list(bench_frames.items()):
        metrics = SUMMARY_METRICS if name in wide else ["NSE", "KGE", "KGE_r", "KGE_alpha", "KGE_beta", "PBIAS", "logNSE"]
        for sub_name, basins in subsets(frame.index).items():
            for metric in metrics:
                row = {"member": name, "subset": sub_name, "metric": metric}
                row.update(summarise(frame.loc[basins, metric].values, metric))
                summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "summary.csv", index=False)

    # ---- NSE per hydrological year ---------------------------------------------------------
    year_rows = []
    for name in member_names:
        per_year = {}
        for b, df in members[name].items():
            hy = hydro_year(df.index)
            for year in np.unique(hy):
                sel = hy == year
                o, s = pair_valid(df["obs"].values[sel], df["sim"].values[sel])
                if o.size >= args.min_days_per_year:
                    per_year.setdefault(int(year), {})[b] = nse(o, s)
        for year in sorted(per_year):
            for sub_name, basins in subsets(list(per_year[year])).items():
                v = np.array([per_year[year][b] for b in basins], dtype=float)
                v = v[np.isfinite(v)]
                year_rows.append({"member": name, "hydro_year": year, "subset": sub_name, "n_basins": int(v.size),
                                  "median_NSE": float(np.median(v)) if v.size else np.nan,
                                  "mean_NSE": float(v.mean()) if v.size else np.nan,
                                  "n_NSE_below_zero": int(np.sum(v < 0))})
    pd.DataFrame(year_rows, columns=["member", "hydro_year", "subset", "n_basins", "median_NSE", "mean_NSE",
                                     "n_NSE_below_zero"]).to_csv(out_dir / "per_year.csv", index=False)

    # ---- sanity alarms ---------------------------------------------------------------------
    medians = {}
    shift_all = []
    for name in member_names:
        w = wide[name]
        high = w.index[w["NSE"] > 0.98].tolist()
        rep.add(f"alarm_nse_above_0.98[{name}]", "no basin has NSE > 0.98", not high,
                {"n_basins": len(high), "basins": [{"basin": b, "NSE": w.loc[b, "NSE"]} for b in high]}, alarm=True)
        med = float(np.nanmedian(w["NSE"].values)) if w["NSE"].notna().any() else np.nan
        medians[name] = med
        rep.add(f"alarm_median_nse_above_0.90[{name}]", "median NSE is not above 0.90", not (med > 0.90),
                {"median_NSE": med, "n_basins": int(w["NSE"].notna().sum())}, alarm=True)
        ident = [b for b, df in members[name].items() if _identical(df)]
        rep.add(f"alarm_sim_identical_to_obs[{name}]", "no simulation is identical to the observations", not ident,
                {"n_basins": len(ident), "basins": ident}, alarm=True)
        sh = pd.DataFrame(shift_rows(name, members[name]))
        shift_all.append(sh)
        n = int(sh["r_same_day"].notna().sum()) if len(sh) else 0
        k = int(sh["shifted_higher"].sum()) if len(sh) else 0
        frac = k / n if n else 0.0
        rep.add(f"alarm_one_day_shift[{name}]",
                "sim correlates better with obs shifted by one day than with same-day obs in at most 10 % of basins",
                frac <= 0.10,
                {"n_basins": n, "n_shifted_higher": k, "fraction": frac,
                 "n_higher_vs_obs_next_day": int((sh["r_sim_vs_obs_next_day"] > sh["r_same_day"]).sum()) if n else 0,
                 "n_higher_vs_obs_previous_day": int((sh["r_sim_vs_obs_previous_day"] > sh["r_same_day"]).sum()) if n else 0,
                 "basins": sh.loc[sh["shifted_higher"], "basin"].tolist() if n else []}, alarm=True)
    if shift_all:
        pd.concat(shift_all, ignore_index=True).to_csv(out_dir / "shift_check.csv", index=False)
    if len(seed_names) > 1:
        seed_med = {n: medians[n] for n in seed_names}
        spread = float(np.nanmax(list(seed_med.values())) - np.nanmin(list(seed_med.values())))
        rep.add("alarm_seed_spread", "median NSE differs by at most 0.03 between seeds", not (spread > 0.03),
                {"median_NSE_per_seed": seed_med, "max_minus_min": spread}, alarm=True)

    # ---- report ------------------------------------------------------------------------------
    input_hashes["raw_files_combined (sha256 over sorted 'hash  path' lines)"] = raw.combined_hash()
    headline = summary[(summary["subset"] == "all") & (summary["metric"].isin(["NSE", "KGE"]))]
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command_line": ["python"] + [repo_rel(a) if Path(a).exists() else a for a in sys.argv],
        "working_directory": repo_rel(Path.cwd()) or ".",
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "period": {"name": args.period, "start": start, "end": end}, "epoch": args.epoch,
        "members": member_names,
        "parameters": {"fhv_fraction": FHV_FRACTION, "flv_fraction": FLV_FRACTION, "flv_floor_mm_per_day": args.flv_floor,
                       "near_zero_mm_per_day": args.near_zero, "log_nse_eps_mm_per_day": args.log_eps,
                       "tol_nse_kge": args.tol_nse_kge, "tol_fhv": args.tol_fhv, "tol_flv": args.tol_flv,
                       "tol_obs_mm_per_day": args.tol_obs, "min_days_per_hydro_year": args.min_days_per_year,
                       "raw_obs_column": RawData.OBS_COL, "raw_sim_column": RawData.SIM_COL,
                       "allow_basin_mismatch": args.allow_basin_mismatch, "sim_clip": args.sim_clip, "basin_file": basin_file},
        "notes": {"prevah": prevah_note},
        "input_sha256": input_hashes,
        "raw_files_sha256": {repo_rel(k): v for k, v in raw.files_used.items()},
        "headline_summary": headline.to_dict("records"),
        "checks": rep.checks,
        "n_checks": len(rep.checks),
        "n_failed_hard": rep.n_failed,
        "n_findings": sum(c["status"] == "finding" for c in rep.checks),
        "n_alarms": sum(c["alarm"] and c["status"] != "pass" for c in rep.checks),
        "overall": "fail" if rep.n_failed else "pass",
    }
    with open(out_dir / "verification_report.json", "w", encoding="utf-8") as fh:
        json.dump(jsonable(report), fh, indent=2, default=str)

    print()
    print(headline.to_string(index=False))
    print()
    failed = [c for c in rep.checks if c["status"] == "fail"]
    if failed:
        print("=" * 78)
        print(f"VERIFICATION FAILED: {len(failed)} hard check(s) failed")
        for c in failed:
            print(f"  - {'ALARM ' if c['alarm'] else ''}{c['id']}: {c['description']}")
        print("=" * 78)
    else:
        print(f"All hard checks passed ({report['n_findings']} soft finding(s)).")
    print(f"Outputs written to {repo_rel(out_dir)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
