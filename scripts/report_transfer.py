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


# ---------------------------------------------------------------- one-day timing alarm
def shift_table(t):
    """Basins whose simulation correlates better with the next day's observation than with the
    same day's, for the two ungauged models, with the same basins in the gauged LSTM runs."""
    import glob
    meta = pd.read_csv(ROOT / "results/data_report/basin_metadata.csv", dtype={"gauge_id": str}).set_index("gauge_id")
    s1 = pd.read_csv(ROOT / "results/verification/shift_check.csv", dtype={"basin": str})
    s1 = s1[s1.member.isin(t["seeds"])].groupby("basin").shifted_higher.sum()
    out = {}
    for key, pattern in [("lstm", "results/verification_setting2/fold*/shift_check.csv"),
                         ("gbm", "results/verification_gbm/setting2_fold*/shift_check.csv")]:
        out[key] = pd.concat([pd.read_csv(p, dtype={"basin": str}) for p in sorted(glob.glob(str(ROOT / pattern)))])
    lstm = out["lstm"].set_index("basin").shifted_higher
    gbm = out["gbm"].set_index("basin").shifted_higher
    basins = sorted(set(lstm.index[lstm]) | set(gbm.index[gbm]), key=lambda b: (t["groups"].fold[b], b))
    rows = []
    for b in basins:
        rows.append({"Basin": b, "Gauge": meta.gauge_name[b].replace("_", " "), "River": meta.water_body_name[b].replace("_", " "),
                     "Fold": int(t["groups"].fold[b]), "Lake flag": "yes" if t["flags"].lake[b] else "no",
                     "Carbonate rock (%)": _f(t["attrs"].carbonate_sedimentary[b], 0),
                     "LSTM ungauged": "shifted" if lstm.get(b, False) else "-",
                     "Boosting ungauged": "shifted" if gbm.get(b, False) else "-",
                     "LSTM gauged, seeds shifted": f"{int(s1.get(b, 0))} of {len(t['seeds'])}"})
    facts = dict(n_lstm=int(lstm.sum()), n_gbm=int(gbm.sum()), n_both=int((lstm & gbm.reindex(lstm.index, fill_value=False)).sum()),
                 n_rows=len(basins), n_lake=int(sum(t["flags"].lake[b] for b in basins)),
                 lake_share=t["flags"].lake.reindex(t["basins2"]).mean() * 100,
                 n_gauged=int(sum(s1.get(b, 0) > 0 for b in basins)), n_basins=len(lstm))
    return pd.DataFrame(rows), facts


# ---------------------------------------------------------------- text
def _med(t, name, basins, metric="NSE"):
    return t["W"][name][metric].reindex(basins).median()


def setting2_block(t, table, prefix="", kraft=None):
    b = t["basins2"]
    nn = [x for x in b if t["flags"].near_natural[x]]
    L1, L2 = "LSTM, gauged (setting 1)", "LSTM, ungauged (setting 2)"
    G1, G2 = "Gradient boosting, gauged (setting 1)", "Gradient boosting, ungauged (setting 2)"
    P = "PREVAH (context only)"
    m = {k: _med(t, k, b) for k in (L1, L2, G1, G2, P)}
    mn = {k: _med(t, k, nn) for k in (L1, L2, G1, G2, P)}
    folds = fold_table(t)
    f_lo, f_hi = folds["Median NSE, LSTM ungauged"].astype(float).min(), folds["Median NSE, LSTM ungauged"].astype(float).max()
    n_neg2 = int((t["W"][L2].NSE.reindex(b) < 0).sum())
    n_negg = int((t["W"][G2].NSE.reindex(b) < 0).sum())
    better = int((t["W"][G2].NSE.reindex(b) > t["W"][L2].NSE.reindex(b)).sum())
    reg = regime_table(t).set_index("Group")
    sh, sf = shift_table(t)
    fhv2, fhv2nn = _med(t, L2, b, "FHV"), _med(t, L2, nn, "FHV")
    # the sentences below state an order between the models; stop if the numbers no longer support it
    assert m[G2] > m[L2] and mn[G2] > mn[L2] and m[L1] > m[G1] and m[L1] > m[L2], "text of setting 2 no longer matches the numbers"
    kraft_line = (f"- Kraft et al. (2025) report a median NSE of {_f(kraft)} for an LSTM on 98 near-natural Swiss basins with\n"
                  f"  cross-validation in space and time. Here the near-natural subset reaches {_f(mn[L2])}. Their folds, inputs\n"
                  f"  and training differ, so the gap is a fact to explain and not a ranking.\n") if kraft is not None else ""
    return f"""### Setting 2. Ungauged basins

The {len(b)} basins below 2000 km2 are split into five folds. A gauge and every gauge upstream or downstream of it
always fall in the same fold, so no test basin contains or lies inside a training basin. {len(t['excluded'])} larger basins are left
out because they contain too many other gauges to be kept apart. For each fold a model with the setting-1 configuration
is trained on the other four folds over the training period and evaluated on the held-out basins over the test period.
One seed per fold. The gradient-boosting model of the same table uses the same folds, forcing and attributes.

**All {len(b)} basins**

{table(setting2_tables(t, 'all'))}

**{len(nn)} near-natural basins**

{table(setting2_tables(t, 'near_natural'))}

The gauged rows are the setting-1 results for the same basins, with the LSTM averaged over its three seeds. The
day-of-year climatology needs discharge from the basin itself and is not available for an ungauged basin. It is shown
as a reference point only.

**By fold**

{table(folds)}

**By group**

{table(regime_table(t))}

![Cumulative distribution of per-basin NSE, gauged and ungauged]({prefix}results/figures/nse_cdf_setting2.png)

**What the numbers say**

- Removing the basin from training lowers the median NSE of the LSTM from {_f(m[L1], 3)} to {_f(m[L2], 3)}.
  {n_neg2} of {len(b)} basins have a negative NSE.
- In this setting gradient boosting does better than the LSTM. Its median is {_f(m[G2], 3)}, it has {n_negg} basins
  below zero, and it is better than the LSTM in {better} of {len(b)} basins. In gauged basins the order is the other way
  round ({_f(m[L1], 3)} against {_f(m[G1], 3)}).
- The median of a fold ranges from {_f(f_lo, 3)} to {_f(f_hi, 3)}. With 35 or 36 basins per fold, the result depends
  on which basins are held out together, and a single fold says little.
- Near-natural basins lose less ({_f(mn[L1], 3)} to {_f(mn[L2], 3)}). Basins outside Switzerland lose most (median change
  {reg.loc['Outside Switzerland', 'Median change']}), as do basins above 1800 m ({reg.loc['Mean elevation > 1800 m', 'Median change']}) and below 50 km2
  ({reg.loc['Area < 50 km2', 'Median change']}).
- High flows are underestimated more than in gauged basins. The median FHV is {_f(fhv2, 1, sign=True)} % for all basins and
  {_f(fhv2nn, 1, sign=True)} % for near-natural basins.
- PREVAH, a calibrated process model, has a median of {_f(m[P], 3)} on these basins. It was calibrated on gauged data, so it
  is not an ungauged prediction and is shown for context only.
{kraft_line}- No hyperparameter was changed for this setting. Settings that generalise better to new basins, such as fewer
  static attributes or stronger regularisation, were not tried, because choosing them would reuse the test period.

**Simulations one day early in lake and karst basins**

The verifier raises an alarm when, in more than 10 % of the basins of a run, the simulation correlates better with the
observation of the next day than with that of the same day. It fired for LSTM folds 0 and 2 and boosting fold 0. Over
all folds the LSTM is a day early in {sf['n_lstm']} of {sf['n_basins']} basins and boosting in {sf['n_gbm']}, {sf['n_both']} of them the same.

{table(sh)}

{sf['n_lake']} of these {sf['n_rows']} basins carry the lake flag, against {_f(sf['lake_share'], 0)} % of all setting-2 basins, and the Doubs
basins drain karst. In both, water is stored and released slowly. With the basin in training, the same basins are on
time in {sf['n_rows'] - sf['n_gauged']} of {sf['n_rows']} cases. An ungauged model has not seen this delay and responds to rain a day too soon.
The models receive no discharge as input, so a shift of the simulation towards future observations cannot come from
leakage. The audit tests this separately (`AUDIT.md`).
"""


def setting3_block(t, table, prefix=""):
    F, zero, full = setting3_frame(t)
    years = sorted(F.columns.get_level_values(1).unique())
    ft = {y: F[("finetune", y)] for y in years}
    sc = {y: F[("scratch", y)] for y in years}
    y0, yN = years[0], years[-1]
    win = {y: int((ft[y] > sc[y]).sum()) for y in years}
    up = {y: int((ft[y] > zero).sum()) for y in years}
    worst_ft = ft[y0].idxmin()
    n = len(F)
    assert abs(ft[y0].median() - sc[yN].median()) < 0.02 and ft[y0].median() > zero.median(), "text of setting 3 no longer matches the numbers"
    return f"""### Setting 3. Basins with a short record

{n} basins, four per regime, are taken from the test basins of setting 2. For each one, only the last 1, 2 or 5 years
of the training period are used. Two ways of using them are compared.

- Fine-tuning. Start from the setting-2 model of the fold in which the basin was held out, so the starting model has
  never seen this basin or any basin on the same river, and train it further on the short record. 300 weight updates
  at a learning rate of 1e-4.
- From scratch. A smaller LSTM (64 units) trained on the short record of this basin alone, with the five forcing
  variables. 1500 weight updates at a learning rate of 1e-3.

Both recipes are the same for every basin and record length and are written in `scripts/make_setting3_configs.py`.
No validation data are used. Each run is evaluated once on the ten test years. One seed per run. The NSE of one trial
run was seen before the other runs started; `AUDIT.md` (section 8) says what this means.

{table(setting3_table(t))}

**By regime** (median NSE)

{table(setting3_regime_table(t))}

**By basin** (NSE)

{table(setting3_basin_table(t))}

![NSE against the length of the record used for training]({prefix}results/figures/finetuning_curve_setting3.png)

**What the numbers say**

- One year of discharge is worth a lot when it is added to a regional model. The median NSE goes from
  {_f(zero.median(), 3)} without any data from the basin to {_f(ft[y0].median(), 3)} with one year and {_f(ft[yN].median(), 3)} with {yN}.
  Fine-tuning improves on the ungauged model in {up[y0]} of {n} basins with one year.
- A model trained from scratch needs more data. Its median is {_f(sc[y0].median(), 3)} with one year and {_f(sc[yN].median(), 3)} with {yN}.
  One year of fine-tuning ({_f(ft[y0].median(), 3)}) does about as well as {yN} years from scratch.
- Fine-tuning is better than training from scratch in {win[y0]} of {n} basins with one year, {win[years[1]]} with {years[1]} and {win[yN]} with {yN}.
  The difference between the medians narrows as the record grows ({', '.join(_f(ft[y].median() - sc[y].median(), 2, sign=True) for y in years)}).
- With {yN} years the fine-tuned models reach a median of {_f(ft[yN].median(), 3)}, against {_f(full.median(), 3)} for the regional model
  of setting 1 trained on 24 years of every basin.
- Single runs vary. Basin {worst_ft} has the lowest one-year fine-tuned NSE ({_f(ft[y0][worst_ft])}). Twenty basins and one seed are
  enough to show the direction of the effect, not to give a precise number for any regime.
- The training years are the last years of the training period. Which years make up a short record matters, and only
  one choice was tested.
"""


def gbm_block(t, table):
    allb = sorted(t["W"]["LSTM, gauged (setting 1)"].index)
    rows = [_row(name, t["W"][name], allb) for name in ("LSTM, gauged (setting 1)", "Gradient boosting, gauged (setting 1)")]
    L, G = t["W"]["LSTM, gauged (setting 1)"].NSE.reindex(allb), t["W"]["Gradient boosting, gauged (setting 1)"].NSE.reindex(allb)
    assert L.median() > G.median(), "text of the boosting comparison no longer matches the numbers"
    return f"""### Gradient boosting

An XGBoost regressor predicts each day's discharge from 83 features. 38 describe the forcing: the five variables on
the day and the two days before, trailing means of precipitation, temperature and sunshine over windows of up to one
year, snowfall and degree-day sums as simple snow proxies, and the day of the year. The other 45 are the static
attributes of the LSTM. Only past and same-day forcing is used, which a test checks. The loss is the squared error weighted per basin
in the same way as the NSE loss of the LSTM. The parameters (depth 10, learning rate 0.05, 1500 trees) were fixed before
the first run and not tuned. In setting 1 it is trained on the same basins and years as the LSTM.

**Setting 1, all {len(allb)} basins** (LSTM averaged over three seeds)

{table(pd.DataFrame(rows))}

- In gauged basins the LSTM is ahead by a small margin ({_f(L.median(), 3)} against {_f(G.median(), 3)} median NSE) and better
  in {int((L > G).sum())} of {len(allb)} basins. The gap is largest at low flows (median log-NSE {_f(t['W']['LSTM, gauged (setting 1)'].logNSE.reindex(allb).median(), 3)} against {_f(t['W']['Gradient boosting, gauged (setting 1)'].logNSE.reindex(allb).median(), 3)}).
- In ungauged basins the order reverses (setting 2 above). In this experiment a feature model with a year of hand-built
  memory transfers to new basins better than the LSTM.
"""


def audit_block(t, table):
    """Section of AUDIT.md for settings 2 and 3 and gradient boosting, from the verifier summaries and
    from results/audit_transfer/audit_summary.json (written by audit/audit_transfer.py)."""
    aud = json.loads((ROOT / "results/audit_transfer/audit_summary.json").read_text())
    A = {a["id"]: a for a in aud}
    ver = [("LSTM, setting 2", t["sum_s2"]), ("Boosting, setting 1", t["sum_g1"]),
           ("Boosting, setting 2", t["sum_g2"]), ("Setting 3, 120 single-basin runs", t["sum_s3"])]
    vt = pd.DataFrame([{"Group": n, "Units": v["n_units"], "Checks": v["n_checks"], "Hard failures": v["n_hard_failures"],
                        "Of which alarms": v["n_alarms"], "Findings": v["n_findings"]} for n, v in ver])
    names = {"1_setting2_scaler": "Normalisation from the training basins and period of each fold",
             "2_nested_basins": "No nested pair of basins split across training and test",
             "3_finetune_base_models": "Fine-tuning starts from a model that never saw the basin",
             "4_setting3_windows": "Setting-3 training windows, including warm-up",
             "5_gbm": "Gradient boosting uses no future forcing and no discharge",
             "6_static_attributes": "No streamflow-derived input",
             "7_model_selection": "No model selection on the test period",
             "8_resume_incident": "Folds 0 to 2 were interrupted and resumed",
             "9_folds_other_gpu": "Folds 3 and 4 trained on another GPU",
             "10_skill_with_without_human_influence": "Skill with and without human-influenced basins",
             "a_printed_trial_and_recipes": "A test result seen before the setting-3 runs",
             "b_one_day_shift_alarms": "One-day-shift alarms",
             "c_setting3_nse_above_0p90": "Single-basin runs above NSE 0.90",
             "d_verify_all_suffix_bug": "Exemption in verify_all.py that does not apply"}
    rows = [{"Check": names.get(a["id"], a["id"]), "Outcome": a["status"], "What was found": a["note"]} for a in aud]
    n_fail = sum(a["status"] == "fail" for a in aud)
    s3_fail = [f["check"].split("[")[0] for f in t["sum_s3"].get("failed", [])]
    return f"""## 8. Settings 2 and 3 and gradient boosting

The metric verification was run on every unit with `verification/verify_all.py`, and a separate audit,
`audit/audit_transfer.py`, was written and run by a reviewer who did not build the models. Its outputs are in
`results/audit_transfer/`.

**Metric verification**

{table(vt)}

- The alarms of setting 2 are the one-day shift in LSTM folds 0 and 2 and boosting fold 0. They are explained in
  `RESULTS.md` and examined below.
- All {len(s3_fail)} hard failures of setting 3 are the alarm on the median NSE across basins ({', '.join(sorted(set(s3_fail)))}).
  With one basin per run the median is the NSE of that basin. `verify_all.py` was meant to exempt this alarm for
  single-basin runs, but it compares the check name exactly, and the names carry the run name as a suffix, so the
  exemption never applied. The verification code is frozen and was not changed. The failures are left as they are.

**Audit**

{n_fail} of {len(aud)} entries failed.

{table(pd.DataFrame(rows))}

**What this means**

- No leakage was found in settings 2 and 3 or in the boosting models.
- The recipes of setting 3 used in all 120 runs are those in the committed script. The NSE of the first scratch run
  (basin 2126, one year) and of one fine-tuning run were printed while the remaining runs were being prepared, and the
  script was committed after the runs. Git cannot show that the recipe was fixed before those numbers were seen. It
  was not changed after them, but this rests on the modeller's word.
- The one-day shift is a timing error of the ungauged models in lake and karst basins. Inputs and observations are
  aligned with the raw files, and the same basins are on time when gauged.
- Folds 3 and 4 ran on a different GPU. GPU training is not bit-reproducible, so no fold can be repeated exactly on
  any machine.
"""
