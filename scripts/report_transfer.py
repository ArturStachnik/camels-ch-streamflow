"""Tables, figures and text for setting 2 (ungauged), setting 3 (data-scarce) and the
gradient-boosting comparison. Called by scripts/make_report.py. Reads only verified outputs:
results/verification/ (setting 1), results/verification_setting2/, results/verification_gbm/
and results/verification_setting3/, plus the audit outputs in results/audit_transfer/.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results/figures"
METRICS = ["NSE", "KGE", "logNSE", "PBIAS", "FHV"]


def _f(x, digits=2, sign=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}"


def _wide(pb, members=None):
    """basin x metric table; with several members, the per-basin mean over members."""
    if members is not None:
        pb = pb[pb.member.isin(members)]
    return pb[pb.metric.isin(METRICS)].groupby(["basin", "metric"]).value.mean().unstack()


def load_transfer():
    read = lambda p: pd.read_csv(ROOT / p, dtype={"basin": str})
    t = dict(
        s1=read("results/verification/per_basin_metrics.csv"),
        s1_bench=read("results/verification/benchmarks.csv"),
        s2=read("results/verification_setting2/combined/per_basin_metrics.csv"),
        g1=read("results/verification_gbm/setting1_combined/per_basin_metrics.csv"),
        g2=read("results/verification_gbm/setting2_combined/per_basin_metrics.csv"),
        s3=read("results/verification_setting3/per_basin_metrics.csv"),
        sum_s2=json.loads((ROOT / "results/verification_setting2/combined/summary.json").read_text()),
        sum_g1=json.loads((ROOT / "results/verification_gbm/setting1_combined/summary.json").read_text()),
        sum_g2=json.loads((ROOT / "results/verification_gbm/setting2_combined/summary.json").read_text()),
        sum_s3=json.loads((ROOT / "results/verification_setting3/summary.json").read_text()),
        sample=pd.read_csv(ROOT / "splits/setting3/sample.csv", dtype={"gauge_id": str}).set_index("gauge_id"),
        groups=pd.read_csv(ROOT / "splits/setting2/groups.csv", dtype={"gauge_id": str}).set_index("gauge_id"),
        excluded=(ROOT / "splits/setting2/excluded_large_basins.txt").read_text().split(),
        flags=pd.read_csv(ROOT / "results/data_report/basin_flags.csv", dtype={"gauge_id": str}).set_index("gauge_id"),
        attrs=pd.read_csv(ROOT / "data/processed/attributes/attributes.csv", dtype={"gauge_id": str}).set_index("gauge_id"),
    )
    seeds = sorted(m for m in t["s1"].member.unique() if "seed" in m)
    t["seeds"] = seeds
    basins2 = sorted(t["s2"].basin.unique())
    t["basins2"] = basins2
    bench = t["s1_bench"].pivot_table(index="basin", columns=["benchmark", "metric"], values="value")
    t["W"] = {
        "LSTM, gauged (setting 1)": _wide(t["s1"], seeds),
        "LSTM, ungauged (setting 2)": _wide(t["s2"]),
        "Gradient boosting, gauged (setting 1)": _wide(t["g1"]),
        "Gradient boosting, ungauged (setting 2)": _wide(t["g2"]),
        "Day-of-year climatology of the basin itself": bench["climatology_doy_train"].reindex(columns=METRICS),
        "PREVAH (context only)": bench["prevah_simulation_based"].reindex(columns=METRICS),
    }
    s3 = t["s3"].copy()
    parts = s3.unit.str.split("_", expand=True)
    s3["kind"], s3["years"] = parts[0], parts[2].str[:-1].astype(int)
    t["s3"] = s3
    return t


def _row(name, W, basins):
    w = W.reindex(basins)
    n = w.NSE.notna().sum()
    return {"Model": name, "Basins": int(n), "Median NSE": _f(w.NSE.median(), 3),
            "NSE 5th-95th pct": f"{_f(w.NSE.quantile(.05))} to {_f(w.NSE.quantile(.95))}",
            "Basins NSE < 0": int((w.NSE < 0).sum()), "Median KGE": _f(w.KGE.median(), 3),
            "Median log-NSE": _f(w.logNSE.median(), 3), "Median bias (%)": _f(w.PBIAS.median(), 1, sign=True),
            "Median FHV (%)": _f(w.FHV.median(), 1, sign=True) if w.FHV.notna().any() else "n/a"}


def setting2_tables(t, subset):
    basins = t["basins2"] if subset == "all" else [b for b in t["basins2"] if t["flags"].near_natural[b]]
    return pd.DataFrame([_row(name, W, basins) for name, W in t["W"].items()])


def fold_table(t):
    s2 = t["s2"][t["s2"].metric == "NSE"]
    g2 = t["g2"][t["g2"].metric == "NSE"]
    s1 = _wide(t["s1"], t["seeds"]).NSE
    rows = []
    for k in range(5):
        b = t["groups"].index[t["groups"].fold == k]
        rows.append({"Fold": k, "Test basins": len(b),
                     "Median NSE, LSTM gauged": _f(s1.reindex(b).median(), 3),
                     "Median NSE, LSTM ungauged": _f(s2[s2.unit == f"fold{k}"].value.median(), 3),
                     "Median NSE, boosting ungauged": _f(g2[g2.unit == f"setting2_fold{k}"].value.median(), 3)})
    return pd.DataFrame(rows)


def regime_table(t):
    a, fl = t["attrs"].loc[t["basins2"]], t["flags"].loc[t["basins2"]]
    W1, W2 = t["W"]["LSTM, gauged (setting 1)"].NSE, t["W"]["LSTM, ungauged (setting 2)"].NSE
    G2 = t["W"]["Gradient boosting, ungauged (setting 2)"].NSE
    groups = [("All setting-2 basins", a.index), ("Near-natural", fl.index[fl.near_natural]),
              ("Regulated", fl.index[fl.regulated]), ("Outside Switzerland", fl.index[fl.human_info_missing]),
              ("Glacier cover > 5 %", a.index[a.glac_frac > 0.05]), ("Mean elevation < 700 m", a.index[a.elev_mean < 700]),
              ("Mean elevation > 1800 m", a.index[a.elev_mean > 1800]), ("Area < 50 km2", a.index[a.area < 50])]
    rows = []
    for name, idx in groups:
        d = (W2 - W1).reindex(idx)
        rows.append({"Group": name, "Basins": len(idx), "Median NSE gauged": _f(W1.reindex(idx).median(), 3),
                     "Median NSE ungauged": _f(W2.reindex(idx).median(), 3),
                     "Median change": _f(d.median(), 3, sign=True),
                     "Basins losing > 0.2": int((d < -0.2).sum()),
                     "Median NSE ungauged, boosting": _f(G2.reindex(idx).median(), 3)})
    return pd.DataFrame(rows)


def setting3_frame(t):
    """basin x (kind, years) NSE, with zero-shot (setting 2) and the full regional model (setting 1)."""
    s3 = t["s3"][t["s3"].metric == "NSE"]
    F = s3.pivot_table(index="basin", columns=["kind", "years"], values="value")
    basins = t["sample"].index
    zero = t["W"]["LSTM, ungauged (setting 2)"].NSE.reindex(basins)
    full = t["W"]["LSTM, gauged (setting 1)"].NSE.reindex(basins)
    return F.reindex(basins), zero, full


def setting3_table(t):
    F, zero, full = setting3_frame(t)
    years = sorted(F.columns.get_level_values(1).unique())
    rows = [{"Training data from the target basin": "none (ungauged model of setting 2)",
             "Median NSE, fine-tuned": _f(zero.median(), 3), "Median NSE, from scratch": "n/a",
             "Fine-tuned better than scratch": "n/a", "Fine-tuned better than no fine-tuning": "n/a"}]
    for y in years:
        ft, sc = F[("finetune", y)], F[("scratch", y)]
        rows.append({"Training data from the target basin": f"{y} year{'s' if y > 1 else ''}",
                     "Median NSE, fine-tuned": _f(ft.median(), 3), "Median NSE, from scratch": _f(sc.median(), 3),
                     "Fine-tuned better than scratch": f"{int((ft > sc).sum())} of {int((ft.notna() & sc.notna()).sum())}",
                     "Fine-tuned better than no fine-tuning": f"{int((ft > zero).sum())} of {int((ft.notna() & zero.notna()).sum())}"})
    rows.append({"Training data from the target basin": "24 years, with all other basins (setting 1)",
                 "Median NSE, fine-tuned": _f(full.median(), 3), "Median NSE, from scratch": "n/a",
                 "Fine-tuned better than scratch": "n/a", "Fine-tuned better than no fine-tuning": "n/a"})
    return pd.DataFrame(rows)


def setting3_regime_table(t):
    F, zero, full = setting3_frame(t)
    years = sorted(F.columns.get_level_values(1).unique())
    rows = []
    for regime, g in t["sample"].groupby("regime", sort=False):
        b = g.index
        r = {"Regime": regime, "Basins": len(b), "No target data": _f(zero.reindex(b).median())}
        for y in years:
            r[f"Fine-tuned, {y} y"] = _f(F[("finetune", y)].reindex(b).median())
            r[f"Scratch, {y} y"] = _f(F[("scratch", y)].reindex(b).median())
        r["Setting 1"] = _f(full.reindex(b).median())
        rows.append(r)
    return pd.DataFrame(rows)


def setting3_basin_table(t):
    F, zero, full = setting3_frame(t)
    years = sorted(F.columns.get_level_values(1).unique())
    rows = []
    for b, r in t["sample"].iterrows():
        row = {"Basin": b, "Regime": r.regime.split(" (")[0], "No target data": _f(zero[b])}
        for y in years:
            row[f"FT {y} y"] = _f(F.loc[b, ("finetune", y)])
            row[f"Scratch {y} y"] = _f(F.loc[b, ("scratch", y)])
        row["Setting 1"] = _f(full[b])
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures
def fig_setting2(t, colours, cdf):
    C_MODEL, C_PREVAH, C_CLIM, C_SEED, SURFACE = colours
    lo = -0.5
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    W, b = t["W"], t["basins2"]
    for name, colour, ls in [("LSTM, gauged (setting 1)", C_MODEL, "-"), ("LSTM, ungauged (setting 2)", C_MODEL, "--"),
                             ("Gradient boosting, gauged (setting 1)", C_PREVAH, "-"), ("Gradient boosting, ungauged (setting 2)", C_PREVAH, "--")]:
        cdf(ax, W[name].NSE.reindex(b).dropna(), lo, color=colour, lw=2, ls=ls, label=name)
    ax.set_xlim(lo, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(f"NSE in the test period (values below {lo} drawn at {lo})")
    ax.set_ylabel("Fraction of basins with NSE below x")
    ax.set_title(f"Gauged against ungauged, {len(b)} basins")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "nse_cdf_setting2.png", dpi=200)
    plt.close(fig)


def fig_setting3(t, colours):
    C_MODEL, C_PREVAH, C_CLIM, C_SEED, SURFACE = colours
    F, zero, full = setting3_frame(t)
    years = sorted(F.columns.get_level_values(1).unique())
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    xs = [0] + years
    for kind, colour, label in [("finetune", C_MODEL, "Fine-tuned from the ungauged model"), ("scratch", C_PREVAH, "Trained from scratch on the basin")]:
        if kind == "finetune":
            cols = [zero] + [F[(kind, y)] for y in years]
            x = xs
        else:
            cols = [F[(kind, y)] for y in years]
            x = years
        med = [c.median() for c in cols]
        q1 = [c.quantile(.25) for c in cols]
        q3 = [c.quantile(.75) for c in cols]
        ax.fill_between(x, q1, q3, color=colour, alpha=0.15, lw=0)
        ax.plot(x, med, color=colour, lw=2, marker="o", ms=6, mec=SURFACE, mew=1, label=label)
    ax.axhline(full.median(), color=C_CLIM, lw=1.5, ls="--", label="Regional model with 24 years of every basin (setting 1)")
    ax.set_xticks(xs)
    ax.set_xticklabels(["0"] + [str(y) for y in years])
    ax.set_xlabel("Years of discharge from the target basin used for training")
    ax.set_ylabel("NSE in the test period")
    ax.set_title(f"Data-scarce basins, median and interquartile range of {len(F)} basins")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "finetuning_curve_setting3.png", dpi=200)
    plt.close(fig)
