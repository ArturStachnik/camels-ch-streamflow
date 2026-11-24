"""Build tables, figures, RESULTS.md, AUDIT.md and the results block of README.md.

Every number in those files is read from results/verification/ (written by
verification/verify_metrics.py) and results/audit/ (written by audit/audit_baseline.py).
Nothing is typed by hand. The only constants are published values from the literature,
listed in LITERATURE with their source.

Usage: python scripts/make_report.py
"""
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import report_transfer as RT

ROOT = Path(__file__).resolve().parents[1]
VER = ROOT / "results/verification"
AUD = ROOT / "results/audit"
FIG = ROOT / "results/figures"
TAB = ROOT / "results/tables"

# Published medians of per-basin NSE, quoted for context only.
LITERATURE = {
    "lees2021": dict(value=0.88, text="Lees et al. (2021), LSTM, 669 catchments of CAMELS-GB, gauged basins, split in time",
                     ref="Lees, T., et al. (2021). Benchmarking data-driven rainfall-runoff models in Great Britain. Hydrology and Earth System Sciences 25, 5517. https://doi.org/10.5194/hess-25-5517-2021"),
    "kraft2025_lstm": dict(value=0.76, text="Kraft et al. (2025), LSTM, 98 Swiss catchments with little human influence, cross-validation in space and time",
                           ref="Kraft, B., et al. (2025). CH-RUN: a deep-learning-based spatially contiguous runoff reconstruction for Switzerland. Hydrology and Earth System Sciences 29, 1061. https://doi.org/10.5194/hess-29-1061-2025"),
    "kraft2025_prevah": dict(value=0.72, text="PREVAH in the same study"),
}

# Chart colours: fixed categorical order (model, process model, climatology) and one sequential hue.
C_MODEL, C_PREVAH, C_CLIM = "#2a78d6", "#eb6834", "#1baf7a"
C_SEED = "#9ec5f4"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.titlesize": 10,
    "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
})


# ---------------------------------------------------------------- helpers
def f(x, digits=2, sign=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}"


def sci(x):
    return f"{x:.0e}".replace("e-0", "e-")


def _numeric_text(col):
    return col.astype(str).str.fullmatch(r"[-+]?\d+(\.\d+)?( %)?|n/a|.* to .*").all()


def table(df):
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---:" if pd.api.types.is_numeric_dtype(df[c]) or _numeric_text(df[c]) else "---" for c in df.columns) + "|"
    return "\n".join([head, sep] + ["| " + " | ".join(str(v).replace("|", r"\|") for v in row) + " |" for row in df.itertuples(index=False)])


def member_label(name):
    m = re.search(r"seed(\d+)", name)
    return f"Seed {m.group(1)}" if m else {"ensemble": "Mean of the three seeds", "benchmark_climatology_doy": "Day-of-year climatology",
                                           "benchmark_prevah": "PREVAH (context only)"}[name]


# ---------------------------------------------------------------- inputs
def load():
    d = dict(
        summary=pd.read_csv(VER / "summary.csv"),
        per_basin=pd.read_csv(VER / "per_basin_metrics.csv", dtype={"basin": str}),
        bench=pd.read_csv(VER / "benchmarks.csv", dtype={"basin": str}),
        per_year_ver=pd.read_csv(VER / "per_year.csv"),
        report=json.loads((VER / "verification_report.json").read_text()),
        audit=json.loads((AUD / "audit_summary.json").read_text()),
        groups=pd.read_csv(AUD / "skill_by_group.csv"),
        skill=pd.read_csv(AUD / "skill_vs_climatology.csv", dtype={"basin": str}).set_index("basin"),
        per_year=pd.read_csv(AUD / "per_year.csv"),
        endings=pd.read_csv(AUD / "record_endings.csv"),
        forcing=pd.read_csv(AUD / "forcing_homogeneity.csv"),
        worst=pd.read_csv(AUD / "worst_basins.csv", dtype={"basin": str}),
        scaler=pd.read_csv(AUD / "scaler_check.csv"),
        val=pd.read_csv(AUD / "validation_by_epoch.csv"),
        flags=pd.read_csv(ROOT / "results/data_report/basin_flags.csv", dtype={"gauge_id": str}).set_index("gauge_id"),
        meta=pd.read_csv(ROOT / "results/data_report/basin_metadata.csv", dtype={"gauge_id": str}).set_index("gauge_id"),
    )
    d["seeds"] = [m for m in d["summary"].member.unique() if re.search(r"seed\d+", m)]
    return d


def stat(d, member, metric, column="median", subset="all"):
    s = d["summary"]
    return float(s[(s.member == member) & (s.metric == metric) & (s.subset == subset)][column].iloc[0])


def flv_unstable(d):
    out = {}
    for c in d["report"]["checks"]:
        if c["id"].startswith("flv_low_segment["):
            out[c["id"][len("flv_low_segment["):-1]] = c["details"]["n_basins_flagged"]
    return out


# ---------------------------------------------------------------- tables
def summary_table(d, subset):
    rows = []
    for member in d["seeds"] + ["ensemble", "benchmark_climatology_doy", "benchmark_prevah"]:
        bench = member.startswith("benchmark")
        rows.append({
            "Model": member_label(member),
            "Median NSE": f(stat(d, member, "NSE", subset=subset), 3),
            "NSE 5th-95th pct": f"{f(stat(d, member, 'NSE', 'p05', subset))} to {f(stat(d, member, 'NSE', 'p95', subset))}",
            "Basins NSE < 0": int(stat(d, member, "NSE", "n_below_zero", subset)),
            "Median KGE": f(stat(d, member, "KGE", subset=subset), 3),
            "Median log-NSE": f(stat(d, member, "logNSE", subset=subset), 3),
            "Median bias (%)": f(stat(d, member, "PBIAS", subset=subset), 1, sign=True),
            "Median FHV (%)": "n/a" if bench else f(stat(d, member, "FHV", subset=subset), 1, sign=True),
            "Median FLV (%)": "n/a" if bench else f(stat(d, member, "FLV", subset=subset), 1, sign=True),
        })
    return pd.DataFrame(rows)


def group_table(d):
    g = d["groups"]
    return pd.DataFrame({
        "Group": g.group, "Basins": g.n_basins, "Median NSE": g.median_NSE.map(lambda x: f(x, 3)),
        "25th pct NSE": g.p25_NSE.map(f), "Lowest NSE": g.min_NSE.map(f),
        "Median NSE of climatology": g.median_NSE_climatology.map(f),
        "Median skill over climatology": g.median_skill_vs_climatology.map(f)})


def year_table(d):
    y = d["per_year"]
    return pd.DataFrame({
        "Hydrological year": y.hydro_year, "Basins": y.n_basins, "Median NSE": y.median_NSE.map(lambda x: f(x, 3)),
        "Median NSE, basins with all years": y.median_NSE_basins_with_all_years.map(lambda x: f(x, 3)),
        "Mean NSE": y.mean_NSE.map(f), f"Mean NSE without basin {d['audit']['per_year']['worst_basin']}": y.mean_NSE_without_worst_basin.map(f)})


def worst_table(d):
    w = d["worst"]
    return pd.DataFrame({
        "Basin": w.basin, "Gauge": w.gauge.str.replace("_", " "), "River": w.river, "Area (km2)": w.area_km2.map(lambda x: f(x, 0)),
        "Flags": w["flags"].str.replace("_", " "), "NSE": w.NSE.map(f), "r": w.r.map(f),
        "Variability ratio": w.alpha.map(f), "Mean ratio": w.beta.map(f),
        "Q/P train": w.QP_ratio_train.map(f), "Q/P test": w.QP_ratio_test.map(f)})


# ---------------------------------------------------------------- figures
def cdf(ax, values, lo, **kw):
    v = np.sort(np.clip(np.asarray(values, dtype=float), lo, None))
    ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", **kw)


def fig_cdf(d):
    lo = -0.5
    pb, bench, flags = d["per_basin"], d["bench"], d["flags"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.9), sharey=True)
    for ax, (title, basins) in zip(axes, [(f"All {len(flags)} basins", flags.index),
                                         (f"{int(flags.near_natural.sum())} near-natural basins", flags.index[flags.near_natural])]):
        for i, seed in enumerate(d["seeds"]):
            v = pb[(pb.member == seed) & (pb.metric == "NSE") & pb.basin.isin(basins)].value
            cdf(ax, v, lo, color=C_SEED, lw=1, label="LSTM, single seeds" if i == 0 else None)
        v = pb[(pb.member == "ensemble") & (pb.metric == "NSE") & pb.basin.isin(basins)].value
        cdf(ax, v, lo, color=C_MODEL, lw=2, label="LSTM, mean of three seeds")
        for name, colour, label in [("prevah_simulation_based", C_PREVAH, "PREVAH (context only)"),
                                    ("climatology_doy_train", C_CLIM, "Day-of-year climatology")]:
            v = bench[(bench.benchmark == name) & (bench.metric == "NSE") & bench.basin.isin(basins)].value
            cdf(ax, v, lo, color=colour, lw=2, label=label)
        ax.set_title(title)
        ax.set_xlim(lo, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(f"NSE in the test period (values below {lo} drawn at {lo})")
    axes[0].set_ylabel("Fraction of basins with NSE below x")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "nse_cdf_setting1.png", dpi=200)
    plt.close(fig)


def fig_map(d):
    sk, meta = d["skill"], d["meta"]
    x, y, v = meta.gauge_lon[sk.index], meta.gauge_lat[sk.index], sk.NSE
    edges = [0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    shp = ROOT / "data/raw/camels_ch_v0.9/camels_ch/catchment_delineations/CAMELS_CH_catchments.shp"
    if shp.exists():  # outline of the modelled area, when the dataset is present locally
        import geopandas as gpd
        cat = gpd.read_file(shp).to_crs(4326)
        cat = cat[cat["gauge_id"].astype(int).astype(str).isin(sk.index)]
        gpd.GeoSeries([cat.geometry.union_all()]).boundary.plot(ax=ax, color=AXIS, linewidth=0.8, zorder=1)
    low = v < edges[0]
    for i in range(len(edges) - 1):
        m = (v >= edges[i]) & (v < edges[i + 1] if i < len(edges) - 2 else v <= edges[i + 1])
        ax.scatter(x[m], y[m], s=30, color=BLUES[i], edgecolor=MUTED, linewidth=0.4, zorder=3,
                   label=f"{edges[i]:.2f} to {edges[i + 1]:.2f}  ({int(m.sum())})")
    ax.scatter(x[low], y[low], s=46, marker="X", color=INK, edgecolor=SURFACE, linewidth=0.6, zorder=4,
               label=f"below {edges[0]:.2f}  ({int(low.sum())})")
    for b in v[low].index:
        ax.annotate(b, (x[b], y[b]), xytext=(5, 5), textcoords="offset points", fontsize=7, color=INK2)
    ax.set_aspect(1 / np.cos(np.deg2rad(46.8)))
    ax.set_xlabel("Longitude (degrees east)")
    ax.set_ylabel("Latitude (degrees north)")
    ax.set_title("NSE in the test period at each gauge, mean of three seeds")
    ax.legend(title="NSE (number of basins)", loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, title_fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "nse_map_setting1.png", dpi=200)
    plt.close(fig)


def fig_climatology(d):
    sk, flags = d["skill"], d["flags"]
    lo = -0.5
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    for mask, colour, label in [(~flags.near_natural[sk.index], C_PREVAH, "Not near-natural"), (flags.near_natural[sk.index], C_MODEL, "Near-natural")]:
        ax.scatter(sk.NSE_climatology[mask], np.clip(sk.NSE[mask], lo, None), s=22, color=colour, edgecolor=SURFACE, linewidth=0.5, label=label)
    ax.plot([lo, 1], [lo, 1], color=AXIS, lw=1, zorder=0)
    ax.annotate("equal skill", (0.72, 0.72), xytext=(0.62, 0.45), fontsize=8, color=INK2, arrowprops=dict(arrowstyle="-", color=AXIS))
    ax.set_xlim(-0.7, 1)
    ax.set_ylim(lo, 1)
    ax.set_xlabel("NSE of the day-of-year climatology (from the training period)")
    ax.set_ylabel(f"NSE of the LSTM (values below {lo} drawn at {lo})")
    ax.set_title("Model skill against the seasonal cycle alone")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "nse_vs_climatology_setting1.png", dpi=200)
    plt.close(fig)


def fig_years(d):
    py = d["per_year_ver"]
    py = py[py.member == "ensemble"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for subset, colour, label in [("all", C_MODEL, "All basins"), ("near_natural", C_PREVAH, "Near-natural basins")]:
        s = py[py.subset == subset]
        ax.plot(s.hydro_year, s.median_NSE, color=colour, lw=2, marker="o", ms=5, mec=SURFACE, mew=1, label=label)
    ax.set_ylim(0.7, 1.0)
    ax.set_xticks(sorted(py.hydro_year.unique()))
    ax.set_xlabel("Hydrological year (October to September)")
    ax.set_ylabel("Median NSE across basins")
    ax.set_title("Median NSE in each test year, mean of three seeds")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "nse_per_year_setting1.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- text
def facts(d):
    a, rep, g = d["audit"], d["report"], d["groups"].set_index("group")
    seed_medians = [stat(d, s, "NSE") for s in d["seeds"]]
    clip, year = a["clipping"], a["per_year"]
    forc = d["forcing"].set_index(["variable", "statistic"])
    val = d["val"]
    worst = d["worst"].set_index("basin")
    wy = pd.DataFrame(year["worst_basin_years"]).T
    wy.index = wy.index.astype(int)
    collapse = wy[wy.NSE < -1]
    seas = a["seasonality"]
    flv = flv_unstable(d)
    return dict(
        n=a["n_basins"], n_nn=int(d["flags"].near_natural.sum()), n_seeds=len(d["seeds"]),
        test=a["periods"]["test"], train=a["periods"]["train"],
        single=f(float(np.median(seed_medians)), 2), single_lo=f(min(seed_medians), 3), single_hi=f(max(seed_medians), 3),
        ens=f(stat(d, "ensemble", "NSE"), 3), ens2=f(stat(d, "ensemble", "NSE"), 2),
        ens_nn=f(stat(d, "ensemble", "NSE", subset="near_natural"), 3), ens_nnn=f(stat(d, "ensemble", "NSE", subset="not_near_natural"), 3),
        ens_min_nn=f(a["basins_below_0p5"]["min_NSE_near_natural"]), n_low=a["basins_below_0p5"]["n"], n_low_nn=a["basins_below_0p5"]["n_near_natural"],
        n_neg=int(stat(d, "ensemble", "NSE", "n_below_zero")),
        kge=f(stat(d, "ensemble", "KGE"), 3), kge_alpha=f(stat(d, "ensemble", "KGE_alpha")), kge_r=f(stat(d, "ensemble", "KGE_r")),
        fhv=f(stat(d, "ensemble", "FHV"), 0, sign=True), fhv_nn=f(stat(d, "ensemble", "FHV", subset="near_natural"), 0, sign=True),
        bias=f(stat(d, "ensemble", "PBIAS"), 1, sign=True), bias_nn=f(stat(d, "ensemble", "PBIAS", subset="near_natural"), 1, sign=True),
        lognse=f(stat(d, "ensemble", "logNSE"), 3),
        flv_min=min(v for k, v in flv.items() if k != "ensemble"), flv_max=max(v for k, v in flv.items() if k != "ensemble"), flv_ens=flv["ensemble"],
        clim=f(stat(d, "benchmark_climatology_doy", "NSE")), prevah=f(stat(d, "benchmark_prevah", "NSE")),
        prevah_nn=f(stat(d, "benchmark_prevah", "NSE", subset="near_natural")), prevah_neg=int(stat(d, "benchmark_prevah", "NSE", "n_below_zero")),
        ss=f(seas["median"]["skill_vs_climatology"]), ss_p25=f(seas["p25"]["skill_vs_climatology"]), ss_p05=f(seas["p05"]["skill_vs_climatology"]),
        anom=f(seas["median"]["NSE_anomalies"]), r_anom=f(seas["median"]["r_anomalies"]),
        ss_test=f(seas["median"]["skill_vs_test_period_climatology"]),
        ss_neg=seas["n_basins_skill_vs_climatology_below_0"], ss_half=seas["n_basins_skill_vs_climatology_below_0p5"],
        g=lambda name, col="median_NSE", digits=2: f(float(g.loc[name, col]), digits), gn=lambda name: int(g.loc[name, "n_basins"]),
        n_checks=rep["n_checks"], n_failed=rep["n_failed_hard"], n_findings=rep["n_findings"], n_alarms=rep["n_alarms"],
        tol=rep["parameters"]["tol_nse_kge"], tol_obs=rep["parameters"]["tol_obs_mm_per_day"],
        scaler=sci(a["scaler"]["max_rel_diff_training"]), scaler_tv=f(100 * a["scaler"]["max_rel_diff_train_val"], 1),
        scaler_all=f(100 * a["scaler"]["max_rel_diff_all_periods"], 0),
        clim_idx=sci(a["climate_indices"]["max_abs_diff_training_period"]), clim_idx_full=a["climate_indices"]["n_basins_closer_to_full_record"],
        best_epochs=", ".join(str(v["best_validation_epoch"]) for v in a["runs_detail"].values()),
        val_lo=f(val.validation_median_NSE[val.epoch >= 10].min(), 3), val_hi=f(val.validation_median_NSE[val.epoch >= 10].max(), 3),
        n_neg_values=clip["n_negative_values"], n_neg_pairs=clip["n_basin_member_pairs_with_negatives"], n_clip_changed=clip["n_pairs_changed_by_more_than_0p01"],
        spike_basin=clip["largest_change"]["basin"], spike_seed=member_label(clip["largest_change"]["member"]).lower(),
        spike=f(clip["largest_change"]["min_sim"], 1), spike_date=clip["largest_change"]["date_of_min"],
        spike_others=" and ".join(f(v, 1) for v in clip["largest_change"]["other_members_same_day"].values()),
        spike_unclipped=f(clip["largest_change"]["NSE_unclipped"], 3), spike_clipped=f(clip["largest_change"]["NSE_clipped"], 3),
        spike_seed_median_unclipped=f(clip["median_NSE_by_member"][clip["largest_change"]["member"]]["NSE_unclipped"], 3),
        spike_seed_median_clipped=f(clip["median_NSE_by_member"][clip["largest_change"]["member"]]["NSE_clipped"], 3),
        worst_basin=year["worst_basin"], worst_from=int(collapse.index.min()), worst_to=int(collapse.index.max()),
        worst_std_before=f(wy.obs_std[wy.index < collapse.index.min()].mean()), worst_std_after=f(collapse.obs_std.mean(), 3),
        worst_nse_lo=f(collapse.NSE.min(), 0), worst_nse_hi=f(collapse.NSE.max(), 0),
        n_complete=year["n_basins_with_all_years"], diff_complete=f(year["max_abs_diff_median_all_vs_complete"], 3),
        n_early=a["n_basins_ending_before_test_end"],
        year_lo=f(d["per_year"].median_NSE.min()), year_hi=f(d["per_year"].median_NSE.max()),
        p_change=f(forc.loc[("precipitation", "mean"), "median_change"], 0, sign=True),
        t_change=f(forc.loc[("temperature_mean", "mean"), "median_change"], 1, sign=True),
        sun_change=f(forc.loc[("rel_sun_dur", "mean"), "median_change"], 0, sign=True),
        step_obs=f(forc[forc.used_as_input].largest_step_statistic.max(), 1),
        step_wind=f(forc.loc[("wind_sim", "mean"), "largest_step_statistic"], 1), step_wind_year=int(forc.loc[("wind_sim", "mean"), "largest_step_hydro_year"]),
        drizzle_train=f(100 * forc.loc[("precipitation", "fraction of days in (0, 0.1) mm"), "train_median"], 1),
        drizzle_test=f(100 * forc.loc[("precipitation", "fraction of days in (0, 0.1) mm"), "test_median"], 1),
        w=lambda b, col, digits=2: f(float(worst.loc[b, col]), digits),
        lees=f(LITERATURE["lees2021"]["value"]), kraft=f(LITERATURE["kraft2025_lstm"]["value"]), kraft_prevah=f(LITERATURE["kraft2025_prevah"]["value"]),
    )


def results_block(d, x, prefix=""):
    """Results and limitations of setting 1. Used in RESULTS.md and, with prefix, in README.md."""
    g = x["g"]
    return f"""### Setting 1. Unseen period in gauged basins

One model is trained on all {x['n']} basins with data from {x['train'][0]} to {x['train'][1]} and evaluated in the same
basins from {x['test'][0]} to {x['test'][1]}. Three seeds, 30 epochs each, no early stopping, no hyperparameter search.
The last epoch is evaluated once on the test period. Metrics are computed per basin with simulated values below zero set
to zero, and then summarised across basins.

**All {x['n']} basins**

{table(summary_table(d, 'all'))}

**{x['n_nn']} near-natural basins** (no regulation, reservoir, glacier, lake, open water balance or unknown human influence flag)

{table(summary_table(d, 'near_natural'))}

NSE and KGE are 1 for a perfect simulation. FHV is the bias of the highest 2 % of flows and FLV the bias of the lowest
30 % in log space (Yilmaz et al., 2008). The climatology is the mean observed flow of each calendar day in the training
period. PREVAH is the calibrated process-based simulation shipped with CAMELS-CH. Its calibration may have used the
test period and its forcing changes on 2010-01-01, so it is shown for context and not as a competitor.

![Cumulative distribution of per-basin NSE]({prefix}results/figures/nse_cdf_setting1.png)

![Map of per-basin NSE]({prefix}results/figures/nse_map_setting1.png)

**What the numbers say**

- A single model reaches a median NSE of {x['single']} (seeds from {x['single_lo']} to {x['single_hi']}). Averaging the three
  simulations gives {x['ens']}. The first number describes one model and the second an ensemble of three.
- The near-natural subset has a median of {x['ens_nn']}, the remaining basins {x['ens_nnn']}. Human influence does not
  lower the median, but it accounts for every poor basin. All {x['n_low']} basins with NSE below 0.5 are regulated or
  have an open water balance, and the lowest NSE among near-natural basins is {x['ens_min_nn']}.
- {x['n_neg']} basins have a negative NSE. Both are described under failures below.
- High flows are underestimated. The median FHV is {x['fhv']} % for all basins and {x['fhv_nn']} % for near-natural
  basins, and the simulated variability is {x['kge_alpha']} of the observed one in the median basin.
- Total volume is underestimated by a median of {x['bias'].lstrip('-+')} % ({x['bias_nn'].lstrip('-+')} % in near-natural basins).
- Small basins are harder. Basins below 50 km2 have a median NSE of {g('Area < 50 km2')}, basins between 200 and 1000 km2
  {g('Area 200-1000 km2')}.
- FLV is not a usable summary here. Between {x['flv_min']} and {x['flv_max']} basins per seed have observed or simulated
  flows near zero in the low-flow segment, where the log bias takes values of several thousand percent. The median
  log-NSE of {x['lognse']} is the more reliable low-flow number.

**Is the skill only the seasonal cycle?**

Many of these basins are fed by snow and ice and have a regular annual cycle that a model can reproduce without
predicting anything. The table compares the model with the day-of-year climatology of each basin. The skill over
climatology is 1 minus the ratio of the squared errors of the model and of the climatology.

{table(group_table(d))}

Across all basins the median skill over climatology is {x['ss']} and the median NSE of the anomalies from the seasonal
cycle is {x['anom']}. In basins without glacier the climatology has no skill (median NSE {g('No glacier', 'median_NSE_climatology')})
and the model NSE of {g('No glacier')} is all predictive. In basins with more than 20 % glacier cover the climatology
alone reaches {g('Glacier cover > 20 %', 'median_NSE_climatology')}, and the model improves on it by
{g('Glacier cover > 20 %', 'median_skill_vs_climatology')} of the remaining error. A high NSE in a glacier basin means
less than the same NSE in a lowland basin.

![Model NSE against climatology NSE]({prefix}results/figures/nse_vs_climatology_setting1.png)

**Stability over the test years**

{table(year_table(d))}

The median stays between {x['year_lo']} and {x['year_hi']}. The mean is not usable. Basin {x['worst_basin']} has an almost
constant observed flow from {x['worst_from']} on, which gives yearly NSE values between {x['worst_nse_lo']} and
{x['worst_nse_hi']} and pulls the mean of all basins below zero. {x['n_early']} records, most of them outside Switzerland,
end before the test period does. Restricting the table to the {x['n_complete']} basins with all years changes the yearly
median by at most {x['diff_complete']}.

![Median NSE per hydrological year]({prefix}results/figures/nse_per_year_setting1.png)

**Failures**

{table(worst_table(d))}

The variability ratio is the standard deviation of the simulation over that of the observation, the mean ratio is the
same for the mean, and Q/P is discharge over precipitation.

- {d['worst'].basin[0]} ({d['worst'].gauge[0]}). Discharge exceeds precipitation in the training period and more so in
  the test period, so the catchment receives water that the forcing does not contain. With the bias removed the NSE
  would be {x['w'](d['worst'].basin[0], 'NSE_bias_removed')}. The share of days that repeat the previous value falls from
  {x['w'](d['worst'].basin[0], 'frac_days_equal_to_previous_train')} to {x['w'](d['worst'].basin[0], 'frac_days_equal_to_previous_test')},
  which points to a change in how the record was produced.
- {x['worst_basin']} ({d['worst'].gauge[1].replace('_', ' ')}). Below a reservoir. The yearly standard deviation of the
  observed flow drops from {x['worst_std_before']} to {x['worst_std_after']} mm/d in {x['worst_from']}. The model cannot
  know about a change in dam operation.
- The other four are regulated rivers or have a Q/P far from a closed water balance. The model reproduces the mean and
  a fraction of the variability, and the timing of releases is not predictable from weather.

**How this compares**

- {LITERATURE['lees2021']['text']}. Median NSE {x['lees']}. This is the same kind of experiment and the closest reference.
- {LITERATURE['kraft2025_lstm']['text']}. Median NSE {x['kraft']}, and {x['kraft_prevah']} for {LITERATURE['kraft2025_prevah']['text']}.
  That experiment predicts basins the model has not seen, which is setting 2 of this project and not this one.
- Values reported for CAMELS-US (Kratzert et al., 2019) are lower. That dataset includes arid basins, where NSE is
  low for every model. CAMELS-CH has none.

### Limitations

- The result holds for gauged basins. Ungauged basins are setting 2 below.
- {x['n'] - x['n_nn']} of {x['n']} basins carry at least one flag for human influence, glacier, lake or open water balance.
  Outside Switzerland the dataset has no information on hydropower, so those {x['gn']('Outside Switzerland (human influence unknown)')} basins are flagged as unknown.
- The highest flows are underestimated (median FHV {x['fhv']} %). For flood forecasting this is the number that matters
  most, and NSE does not show it.
- Hyperparameters are those of Kratzert et al. (2019) and were not tuned. Validation data were not used for any choice.
- The test period has now been looked at. Any later change to this model that is judged on the same ten years is no
  longer a clean test.
- The forcing is a gridded observation product. The test period is {x['t_change'].lstrip('+')} degrees warmer than the
  training period and has {x['p_change'].lstrip('-')} % less precipitation, so the test includes a moderate shift in climate.
- One seed produced a single simulated value of {x['spike']} mm/d in basin {x['spike_basin']} on {x['spike_date']}, while
  the other two seeds gave {x['spike_others']} mm/d. Setting negative values to zero hides it. The cause is not diagnosed.
"""


def transfer_blocks(t, prefix=""):
    parts = [RT.gbm_block(t, table), RT.setting2_block(t, table, prefix, kraft=LITERATURE["kraft2025_lstm"]["value"]),
             RT.setting3_block(t, table, prefix)]
    return "\n".join(parts)


def write_results(d, x, t):
    text = f"""# Results

Generated by `scripts/make_report.py` from the verified outputs in `results/verification*/` and the audits in
`results/audit*/`. Do not edit by hand.

{results_block(d, x)}
{transfer_blocks(t)}
### References

- {LITERATURE['lees2021']['ref']}
- {LITERATURE['kraft2025_lstm']['ref']}
- Kratzert, F., et al. (2019). Towards learning universal, regional, and local hydrological behaviors via machine learning applied to large-sample datasets. Hydrology and Earth System Sciences 23, 5089. https://doi.org/10.5194/hess-23-5089-2019
- Yilmaz, K. K., Gupta, H. V., Wagener, T. (2008). A process-based diagnostic approach to model evaluation. Water Resources Research 44, W09417.

### Files

- `results/verification/per_basin_metrics.csv` has every metric for every basin and seed.
- `results/verification/verification_report.json` lists every check with its outcome.
- `results/audit/` has the tables behind the audit.
- `results/verification_setting2/`, `results/verification_gbm/` and `results/verification_setting3/` hold the same
  for settings 2 and 3 and for gradient boosting, and `results/audit_transfer/` the audit of those settings.
- `results/tables/` has the tables of this page as CSV.
"""
    (ROOT / "RESULTS.md").write_text(text, encoding="utf-8", newline="\n")


def write_audit(d, x, t):
    forc = d["forcing"]
    forcing_table = pd.DataFrame({
        "Variable": forc.variable, "Statistic": forc.statistic, "Model input": forc.used_as_input.map({True: "yes", False: "no"}),
        "Training period": forc.train_median.map(f), "Test period": forc.test_median.map(f),
        "Change": [f(v, 1, sign=True) + (" %" if u == "percent" else "") for v, u in zip(forc.median_change, forc.change_unit)],
        "Largest step": forc.largest_step_statistic.map(lambda v: f(v, 1)), "Year of largest step": forc.largest_step_hydro_year})
    val = d["val"].assign(seed=d["val"].run.map(member_label)).pivot(index="epoch", columns="seed", values="validation_median_NSE")
    val_table = val.map(lambda v: f(v, 3)).reset_index().rename(columns={"epoch": "Epoch"})
    end = d["endings"]
    end_table = pd.DataFrame({"Last year of record": end.last_year, "Country": end.country, "Basins": end.n_basins, "Median NSE": end.median_NSE.map(f)})
    text = f"""# Audit

Generated by `scripts/make_report.py` from `results/verification/` and `results/audit/`. Do not edit by hand.

This file records what was checked before any number was reported, what was found, and what could not be checked.
The metric verification (`verification/verify_metrics.py`) and the audit (`audit/audit_baseline.py`) are separate from
the modelling code and do not import the metric functions of the modelling library.

## Verdict for setting 1

The baseline can be reported with the caveats listed in `RESULTS.md`. No leakage was found. The median NSE of
{x['single']} for a single model is consistent with published results for the same kind of experiment
({LITERATURE['lees2021']['text']}, median {x['lees']}).

## Verdict for settings 2 and 3 and gradient boosting

They can be reported with the caveats in `RESULTS.md` and section 8. No leakage was found. Nothing looks too good. The
ungauged LSTM is weak, and that result is reported as it is. The verifier raised two kinds of alarm. The one-day shift
in lake and karst basins was examined and is a property of the ungauged models. The alarm on single-basin runs came
from a bug in the verification wrapper, which was fixed with the owner's approval (section 8).

## 1. Design audit, before training

Done on the first version of the pipeline, before any 30-epoch run.

- **Blocking, fixed.** The first design used the forcing in the dataset's `simulation_based` folder. That product
  changes on 2010-01-01, between the training and the test period. The step is still visible in the table of section 4
  (wind, largest step {x['step_wind']} in {x['step_wind_year']}). The runs reported here use the observation-based forcing.
- **Fixed.** The climate indices shipped with the dataset cover the whole record, which includes the test period. They
  are recomputed from the training period.
- **Fixed.** Basins with discharge above twice the precipitation, duplicated records and catchments without complete
  observation-based forcing are excluded. The list and the reasons are in `results/data_report/basin_report.csv`.
- **Addressed in setting 2.** Almost all basins are nested with another one. Random folds over basins would put a gauge
  and its upstream gauge in different folds. Setting 2 uses grouped folds (section 9).
- **Carried into the report.** FLV is unstable with flows near zero, the NSE loss gives more weight to basins with low
  variance, and the human-influence attributes exist only inside Switzerland.

## 2. Metric verification

{x['n_checks']} checks, {x['n_failed']} hard failures, {x['n_alarms']} alarms, {x['n_findings']} findings that do not block.

- NSE, KGE, FHV and FLV were recomputed from the saved simulations with code written from the published definitions.
  They agree with the library within {x['tol']} for NSE and KGE in every basin and seed.
- The observed discharge inside the result files equals the raw CAMELS-CH files on the same dates within {x['tol_obs']} mm/d.
  This rules out unit errors and shifts by one day.
- Every date lies in the test period, no date is duplicated, the basins are exactly the {x['n']} of the frozen split, and
  the split files match their recorded SHA-256 sums.
- Alarms that would have stopped the work and did not trigger. Any basin with NSE above 0.98, a median above 0.90, a
  simulation identical to the observation, a simulation that correlates better with the observation shifted by one day
  in more than 10 % of basins, and seeds whose medians differ by more than 0.03.
- Findings. The saved simulations contain {x['n_neg_values']} negative values in {x['n_neg_pairs']} of {x['n'] * x['n_seeds']} basin and seed pairs
  (section 5). FLV is flagged as unstable in {x['flv_min']} to {x['flv_max']} basins per seed and {x['flv_ens']} for the mean of the seeds.

## 3. Leakage checks after the runs

| Check | Result |
|---|---|
| Normalisation statistics use only the training period | Recomputed from the training period they match the saved scaler within a relative difference of {x['scaler']}. Recomputed from training plus validation they differ by up to {x['scaler_tv']} %, and from the whole record by up to {x['scaler_all']} %, so the check can tell the cases apart. The scaler is identical in the three seeds. |
| Normalisation statistics use only training basins | In setting 1 all basins are training basins. For the folds of setting 2 see section 8. |
| Static attributes derived from streamflow | None of the 45 attributes in the run configuration is a hydrological signature. |
| Climate indices use only the training period | Recomputed from the raw files for all {x['n']} basins with separate code. Largest difference to the attribute table {x['clim_idx']}. Basins closer to the whole-record value than to the training-period value {x['clim_idx_full']}. |
| Observed discharge among the inputs | The run configuration has five meteorological inputs and no lagged or autoregressive input. In the library, `BaseDataset` sets the target to missing before the start of each period and passes only the dynamic inputs to the model. |
| Temporal overlap, including warm-up | The periods do not share target days. The 365-day input window of the first test days reaches into the validation period and contains forcing only. |
| Basin overlap, including nested gauges | Does not apply to setting 1, where the same basins are used for training and testing by design. |
| Model selection on the test period | Only epoch 30 was evaluated on the test period in each run. Validation was logged every five epochs and not used. The best validation epochs would have been {x['best_epochs']} for the three seeds, and the validation median from epoch 10 on stays between {x['val_lo']} and {x['val_hi']}. |
| Basin selection uses test-period information | Only through the requirement of three years of valid test discharge, which removes two basins. The water-balance filter uses the training period. |
| Human-influenced basins reported with and without | Yes, in every table of `RESULTS.md`. |

Median validation NSE by epoch, for the record

{table(val_table)}

## 4. Homogeneity of the forcing

Annual statistics of each variable, median over basins, training period against test period. The step statistic
compares the five years before and after each year, in units of their pooled standard deviation.

{table(forcing_table)}

The observation-based inputs show a warming of {x['t_change'].lstrip('+')} degrees, {x['p_change'].lstrip('-')} % less precipitation and
{x['sun_change'].lstrip('+')} % more sunshine. Their largest step statistic is {x['step_obs']}, against {x['step_wind']} for the wind of the product
that was dropped. The share of days with precipitation between 0 and 0.1 mm rises from {x['drizzle_train']} % to {x['drizzle_test']} %.
This may be a change in how the gridded product handles very light precipitation. It is not explained.

## 5. Negative simulated values

The library computes its metrics with negative simulated values set to zero, as the configuration states, and saves the
simulations as they come out of the network. The report uses the same convention.

- {x['n_neg_values']} negative values in {x['n_neg_pairs']} basin and seed pairs.
- Setting them to zero changes the NSE by more than 0.01 in {x['n_clip_changed']} pair. In basin {x['spike_basin']}, {x['spike_seed']}
  simulates {x['spike']} mm/d on {x['spike_date']}, where the other two seeds give {x['spike_others']} mm/d. The NSE of that pair goes from
  {x['spike_unclipped']} to {x['spike_clipped']} and the median of that seed from {x['spike_seed_median_unclipped']} to {x['spike_seed_median_clipped']}.
- The cause of that value is not diagnosed.

## 6. Is the result too good?

- The alarms of section 2 did not trigger.
- The skill is not the seasonal cycle. Median skill over the day-of-year climatology {x['ss']} (25th percentile {x['ss_p25']},
  5th percentile {x['ss_p05']}), median NSE of the anomalies {x['anom']}, median correlation of the anomalies {x['r_anom']}. Against a climatology
  computed from the test period itself, which no forecast could have, the median skill is still {x['ss_test']}.
  It is negative in {x['ss_neg']} basin and below 0.5 in {x['ss_half']}.
- The mean of three seeds scores higher than each single seed, as expected from averaging. It is reported separately
  from the single-model number.
- The comparison with the literature is in `RESULTS.md`.

## 7. Records that end early

{x['n_early']} basins have no observations up to the end of the test period.

{table(end_table)}

Their absence from the last years changes the yearly median by at most {x['diff_complete']}.

{RT.audit_block(t, table)}
## 9. Not verified

- Inference was not repeated from the saved weights. That the result files belong to the weights of epoch 30 rests on
  the training log and on the file dates.
- The runs started a few seconds before the commit that contains their configuration, so the run configuration has no
  commit hash. The configuration files in the run directories were compared with the committed ones and differ only in paths.
- Internal changes of the gridded forcing product, such as changes in the station network, are only visible indirectly.
- Whether PREVAH was calibrated on years of the test period.
- The cause of the isolated negative value in section 5.
- That the setting-3 recipe was fixed before the printed trial result (section 8).
- How much of the spread between setting-2 folds and between setting-3 runs is due to the seed. Settings 2 and 3 use
  one seed per model.
- Whether the pipeline test with three epochs, run before the design audit, influenced any decision. It was evaluated
  on the validation period only and nothing was tuned afterwards, but a log cannot prove an absence.
"""
    (ROOT / "AUDIT.md").write_text(text, encoding="utf-8", newline="\n")


def update_readme(d, x, t):
    path = ROOT / "README.md"
    text = path.read_text(encoding="utf-8")
    start, end = "<!-- results:start -->", "<!-- results:end -->"
    block = f"""{start}
<!-- Generated by scripts/make_report.py. Do not edit by hand. -->

{results_block(d, x)}
{transfer_blocks(t)}{end}"""
    assert start in text and end in text, "README.md lacks the results markers"
    text = text[:text.index(start)] + block + text[text.index(end) + len(end):]
    path.write_text(text, encoding="utf-8", newline="\n")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    d = load()
    x = facts(d)
    t = RT.load_transfer()
    for name, tb in [("setting2_summary_all_basins", RT.setting2_tables(t, "all")), ("setting2_summary_near_natural", RT.setting2_tables(t, "near_natural")),
                     ("setting2_by_fold", RT.fold_table(t)), ("setting2_by_group", RT.regime_table(t)), ("setting2_one_day_early", RT.shift_table(t)[0]),
                     ("setting3_summary", RT.setting3_table(t)), ("setting3_by_regime", RT.setting3_regime_table(t)), ("setting3_by_basin", RT.setting3_basin_table(t))]:
        tb.to_csv(TAB / f"{name}.csv", index=False, encoding="utf-8", lineterminator="\n")
    colours = (C_MODEL, C_PREVAH, C_CLIM, C_SEED, SURFACE)
    RT.fig_setting2(t, colours, cdf)
    RT.fig_setting3(t, colours)
    for name, tb1 in [("setting1_summary_all_basins", summary_table(d, "all")), ("setting1_summary_near_natural", summary_table(d, "near_natural")),
                    ("setting1_skill_by_group", group_table(d)), ("setting1_per_year", year_table(d)), ("setting1_worst_basins", worst_table(d))]:
        tb1.to_csv(TAB / f"{name}.csv", index=False, encoding="utf-8", lineterminator="\n")
    fig_cdf(d)
    fig_map(d)
    fig_climatology(d)
    fig_years(d)
    write_results(d, x, t)
    write_audit(d, x, t)
    update_readme(d, x, t)
    print("wrote RESULTS.md, AUDIT.md, README.md results block, results/tables, results/figures")


if __name__ == "__main__":
    main()
