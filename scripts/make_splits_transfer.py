"""Write the basin splits for setting 2 (ungauged basins) and setting 3 (data-scarce transfer).

Runs once, before any model of these settings is trained. Uses only catchment geometry,
the catchment hierarchy and static attributes, never discharge in the validation or test
period and never a model result. Refuses to overwrite existing files.

Setting 2
- Two basins are related if one drains into the other (catchment hierarchy of CAMELS-CH,
  or more than 90 % of the smaller polygon inside the larger one). Related basins always
  fall in the same fold, so no test basin has a gauge upstream or downstream in training.
- Gauges draining more than MAX_AREA_KM2 contain many other gauges and would merge most of
  the country into one group. They are left out of setting 2 entirely.
- Groups of related basins are assigned to N_FOLDS folds, largest group first, each to the
  fold with the fewest basins so far. Ties are broken with a fixed random seed.

Setting 3
- A sample of setting-2 basins in five regimes, defined from static attributes only, with
  complete discharge in the five years at the end of the training period and in the test
  period. Within a regime, near-natural basins are preferred.

Usage: python scripts/make_splits_transfer.py
"""
from pathlib import Path

import geopandas as gpd
import hashlib
import networkx as nx
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "splits"
OUT2, OUT3 = SPLITS / "setting2", SPLITS / "setting3"
SHAPES = ROOT / "data/raw/camels_ch_v0.9/camels_ch/catchment_delineations/CAMELS_CH_catchments.shp"

MAX_AREA_KM2 = 2000
N_FOLDS = 5
SEED = 20261121
CONTAINED_FRACTION = 0.9
N_PER_REGIME = 4
FINETUNE_YEARS = (1, 2, 5)
REGIMES = {  # name: rule on static attributes
    "lowland (mean elevation < 700 m)": lambda a: (a.elev_mean < 700) & (a.glac_frac == 0),
    "hills (700-1200 m)": lambda a: a.elev_mean.between(700, 1200) & (a.glac_frac == 0),
    "pre-alpine (1200-1800 m)": lambda a: a.elev_mean.between(1200, 1800, inclusive="right") & (a.glac_frac < 0.01),
    "alpine, little glacier (> 1800 m, glacier < 5 %)": lambda a: (a.elev_mean > 1800) & (a.glac_frac < 0.05),
    "glacierised (glacier > 10 %)": lambda a: a.glac_frac > 0.10,
}


def write(path, text):
    if path.exists():
        raise SystemExit(f"{path} exists; splits are written once")
    path.write_text(text, encoding="utf-8", newline="\n")


def containment_pairs(ids, meta):
    lo, up, area = meta.hierarchy_lower, meta.hierarchy_upper, meta.area
    pairs = {(a, b, "hierarchy") for a in ids for b in ids
             if a != b and lo[a] <= lo[b] and up[b] <= up[a] and area[a] > area[b]}
    shapes = gpd.read_file(SHAPES)
    shapes["gauge_id"] = shapes.gauge_id.astype(int).astype(str)
    geom = shapes.set_index("gauge_id").loc[ids].geometry.buffer(0)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if geom[a].intersects(geom[b]):
                small = min(geom[a].area, geom[b].area)
                if geom[a].intersection(geom[b]).area / small > CONTAINED_FRACTION:
                    outer, inner = (a, b) if geom[a].area >= geom[b].area else (b, a)
                    pairs.add((outer, inner, "polygon"))
    df = pd.DataFrame(sorted(pairs), columns=["outer", "inner", "source"])
    return df.groupby(["outer", "inner"]).source.agg(lambda s: "+".join(sorted(s))).reset_index()


def make_folds(ids, meta, pairs):
    keep = [b for b in ids if meta.area[b] <= MAX_AREA_KM2]
    graph = nx.Graph()
    graph.add_nodes_from(keep)
    graph.add_edges_from((a, b) for a, b in zip(pairs.outer, pairs.inner) if a in keep and b in keep)
    rng = np.random.default_rng(SEED)
    groups = [sorted(c) for c in nx.connected_components(graph)]
    order = rng.permutation(len(groups))
    groups = sorted((groups[i] for i in order), key=len, reverse=True)  # stable sort keeps the random order within a size
    folds = [[] for _ in range(N_FOLDS)]
    rows = []
    for g_id, members in enumerate(groups):
        k = int(np.argmin([len(f) for f in folds]))
        folds[k] += members
        rows += [dict(gauge_id=b, group=g_id, fold=k) for b in members]
    return pd.DataFrame(rows).sort_values("gauge_id"), [b for b in ids if b not in keep]


def sample_setting3(assign, attrs, flags, ids_ok):
    rng = np.random.default_rng(SEED)
    a = attrs.loc[assign.gauge_id]
    rows, used = [], set()
    for regime, rule in REGIMES.items():
        cand = [b for b in a.index[rule(a)] if b in ids_ok and b not in used]
        natural = sorted(b for b in cand if flags.near_natural[b])
        other = sorted(b for b in cand if not flags.near_natural[b] and not flags.reservoir_major[b])
        pick = list(rng.permutation(natural)[:N_PER_REGIME])
        if len(pick) < N_PER_REGIME:
            pick += list(rng.permutation(other)[:N_PER_REGIME - len(pick)])
        used |= set(pick)
        rows += [dict(gauge_id=b, regime=regime, near_natural=bool(flags.near_natural[b])) for b in sorted(pick)]
    return pd.DataFrame(rows)


def complete_windows(ids, periods):
    """Basins with discharge on at least 98 % of the days of the longest fine-tuning window and of the test period."""
    train_end = pd.Timestamp(periods["train"][1])
    start = train_end - pd.DateOffset(years=max(FINETUNE_YEARS)) + pd.Timedelta(days=1)
    ok = []
    for b in ids:
        q = xr.open_dataset(ROOT / f"data/processed/time_series/{b}.nc").discharge_spec.to_series()
        w, t = q.loc[start:train_end], q.loc[periods["test"][0]:periods["test"][1]]
        if w.notna().mean() >= 0.98 and t.notna().mean() >= 0.98:
            ok.append(b)
    return ok


def main():
    from ruamel.yaml import YAML
    ids = (SPLITS / "basins_all.txt").read_text().split()
    periods = YAML(typ="safe").load((SPLITS / "periods.yml").read_text())
    periods = {k: [str(pd.Timestamp(d).date()) for d in v] for k, v in periods.items()}
    meta = pd.read_csv(ROOT / "results/data_report/basin_metadata.csv", dtype={"gauge_id": str}).set_index("gauge_id").loc[ids]
    attrs = pd.read_csv(ROOT / "data/processed/attributes/attributes.csv", dtype={"gauge_id": str}).set_index("gauge_id")
    flags = pd.read_csv(ROOT / "results/data_report/basin_flags.csv", dtype={"gauge_id": str}).set_index("gauge_id")

    OUT2.mkdir(exist_ok=True)
    OUT3.mkdir(exist_ok=True)
    pairs = containment_pairs(ids, meta)
    assign, dropped = make_folds(ids, meta, pairs)
    write(OUT2 / "containment_pairs.csv", pairs.to_csv(index=False, lineterminator="\n"))
    write(OUT2 / "groups.csv", assign.to_csv(index=False, lineterminator="\n"))
    write(OUT2 / "excluded_large_basins.txt", "\n".join(dropped) + "\n")
    write(OUT2 / "basins_setting2.txt", "\n".join(assign.gauge_id) + "\n")
    new = [OUT2 / n for n in ("containment_pairs.csv", "groups.csv", "excluded_large_basins.txt", "basins_setting2.txt")]
    for k in range(N_FOLDS):
        test = assign.gauge_id[assign.fold == k].tolist()
        train = assign.gauge_id[assign.fold != k].tolist()
        write(OUT2 / f"fold{k}_test.txt", "\n".join(test) + "\n")
        write(OUT2 / f"fold{k}_train.txt", "\n".join(train) + "\n")
        new += [OUT2 / f"fold{k}_test.txt", OUT2 / f"fold{k}_train.txt"]

    ok = complete_windows(assign.gauge_id.tolist(), periods)
    sample = sample_setting3(assign, attrs, flags, set(ok)).merge(assign[["gauge_id", "fold"]], on="gauge_id")
    write(OUT3 / "sample.csv", sample.to_csv(index=False, lineterminator="\n"))
    new.append(OUT3 / "sample.csv")
    for b in sample.gauge_id:
        write(OUT3 / f"basin_{b}.txt", b + "\n")
        new.append(OUT3 / f"basin_{b}.txt")

    with open(SPLITS / "SHA256SUMS", "a", encoding="utf-8", newline="\n") as f:
        for p in new:
            f.write(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT).as_posix()}\n")

    sizes = assign.groupby("fold").size().tolist()
    print(f"containment pairs {len(pairs)}; setting 2: {len(assign)} basins in {assign.group.nunique()} groups, "
          f"fold sizes {sizes}, largest group {assign.groupby('group').size().max()}, left out {len(dropped)} gauges > {MAX_AREA_KM2} km2")
    print(f"setting 3: {len(ok)} basins with complete windows; sample")
    print(sample.to_string(index=False))


if __name__ == "__main__":
    main()
