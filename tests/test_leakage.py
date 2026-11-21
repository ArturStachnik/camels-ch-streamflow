"""Checks that must pass before any training run. Run with: pytest -q"""
import hashlib
import importlib.util
from pathlib import Path

import pandas as pd
import pytest
import xarray as xr
from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / "configs").glob("*.yml"))
ATTRIBUTES = ROOT / "data/processed/attributes/attributes.csv"

# Hydrological signatures in CAMELS-CH (hydrology_attributes_obs / _sim): derived from streamflow.
SIGNATURES = {"q_mean", "runoff_ratio", "stream_elas", "slope_fdc", "baseflow_index_landson", "hfd_mean",
              "Q5", "Q95", "high_q_freq", "high_q_dur", "low_q_freq", "low_q_dur", "zero_q_freq"}
# Whitelists: any dynamic input outside these sets fails.
FORCING_OBS = {"precipitation", "temperature_min", "temperature_mean", "temperature_max", "rel_sun_dur"}
FORCING_SIM = {"precipitation_sim", "temperature_sim", "radiation_sim", "rel_humidity_sim", "wind_sim", "pet_sim"}


def load(path):
    return YAML(typ="safe").load(path.read_text())


def date(s):
    return pd.to_datetime(s, format="%d/%m/%Y")


def prepare_module():
    spec = importlib.util.spec_from_file_location("prepare_data", ROOT / "scripts/prepare_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_files_unchanged():
    """Split files and the attribute table match the recorded hashes."""
    for line in (ROOT / "splits/SHA256SUMS").read_text().splitlines():
        digest, name = line.split()
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name


def test_no_signatures_in_attribute_table():
    cols = set(pd.read_csv(ATTRIBUTES, nrows=1).columns)
    assert not cols & SIGNATURES


def test_attribute_table_covers_exactly_the_selected_basins():
    basins = (ROOT / "splits/basins_all.txt").read_text().split()
    table = pd.read_csv(ATTRIBUTES, dtype={"gauge_id": str})
    assert sorted(table.gauge_id) == sorted(basins)
    assert not table.isna().any().any()


def test_climate_attributes_come_from_training_period():
    """Recompute the climate indices of three basins from training-period forcing only."""
    prep = prepare_module()
    table = pd.read_csv(ATTRIBUTES, dtype={"gauge_id": str}).set_index("gauge_id")
    basins = (ROOT / "splits/basins_all.txt").read_text().split()
    for basin in (basins[0], basins[len(basins) // 2], basins[-1]):
        df = xr.open_dataset(ROOT / f"data/processed/time_series/{basin}.nc").to_dataframe()
        expected = prep.climate_indices(df, prep.PERIODS["train"])
        other = prep.climate_indices(df, ("1981-10-01", "2020-09-30"))
        for name, value in expected.items():
            assert table.loc[basin, name] == pytest.approx(value, abs=1e-5), (basin, name)
        # the check must be able to tell the two periods apart
        assert abs(other["p_mean"] - expected["p_mean"]) > 1e-4


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_config(path):
    cfg = load(path)
    table_cols = set(pd.read_csv(ATTRIBUTES, nrows=1).columns)
    statics = set(cfg.get("static_attributes", []))
    assert not statics & SIGNATURES
    assert statics <= table_cols
    inputs = set(cfg["dynamic_inputs"])
    assert inputs <= FORCING_OBS or inputs <= FORCING_SIM, inputs
    assert cfg["target_variables"] == ["discharge_spec"]
    # train < validation < test, no shared target days
    assert date(cfg["train_end_date"]) < date(cfg["validation_start_date"])
    assert date(cfg["validation_end_date"]) < date(cfg["test_start_date"])
    # periods match the frozen split definition
    periods = load(ROOT / "splits/periods.yml")
    for key in ("train", "validation", "test"):
        assert date(cfg[f"{key}_start_date"]) == pd.Timestamp(periods[key][0])
        assert date(cfg[f"{key}_end_date"]) == pd.Timestamp(periods[key][1])
    # basins come from the frozen split files, no duplicates
    for key in ("train", "validation", "test"):
        f = ROOT / cfg[f"{key}_basin_file"]
        assert ROOT / "splits" in f.parents
        assert cfg[f"{key}_basin_file"] in (ROOT / "splits/SHA256SUMS").read_text(), "basin file has no frozen hash"
        basins = f.read_text().split()
        assert len(basins) == len(set(basins))
    # no model selection: fixed number of epochs
    assert not cfg.get("early_stopping", False)


# ---------------------------------------------------------------- setting 2: ungauged basins
SETTING2 = ROOT / "splits/setting2"


def _basins(path):
    return path.read_text().split()


def test_setting2_folds_partition_the_basins():
    """Every setting-2 basin is tested in exactly one fold and never trained on in that fold."""
    allb = _basins(SETTING2 / "basins_setting2.txt")
    tested = []
    for k in range(5):
        test, train = _basins(SETTING2 / f"fold{k}_test.txt"), _basins(SETTING2 / f"fold{k}_train.txt")
        assert not set(test) & set(train)
        assert sorted(test + train) == sorted(allb)
        tested += test
    assert sorted(tested) == sorted(allb)
    assert set(allb) <= set(_basins(ROOT / "splits/basins_all.txt"))
    assert not set(allb) & set(_basins(SETTING2 / "excluded_large_basins.txt"))


def test_setting2_no_nested_pair_crosses_train_and_test():
    pairs = pd.read_csv(SETTING2 / "containment_pairs.csv", dtype=str)
    assert len(pairs) > 0
    for k in range(5):
        test, train = set(_basins(SETTING2 / f"fold{k}_test.txt")), set(_basins(SETTING2 / f"fold{k}_train.txt"))
        crossing = [(a, b) for a, b in zip(pairs.outer, pairs.inner) if (a in test and b in train) or (a in train and b in test)]
        assert not crossing, (k, crossing[:5])


def test_setting2_containment_pairs_match_the_catchment_hierarchy():
    """Recompute the hierarchy pairs and check that none is missing from the frozen file."""
    meta = pd.read_csv(ROOT / "results/data_report/basin_metadata.csv", dtype={"gauge_id": str}).set_index("gauge_id")
    ids = _basins(ROOT / "splits/basins_all.txt")
    lo, up, area = meta.hierarchy_lower, meta.hierarchy_upper, meta.area
    expected = {(a, b) for a in ids for b in ids if a != b and lo[a] <= lo[b] and up[b] <= up[a] and area[a] > area[b]}
    pairs = pd.read_csv(SETTING2 / "containment_pairs.csv", dtype=str)
    assert expected <= set(zip(pairs.outer, pairs.inner))


@pytest.mark.parametrize("path", sorted((ROOT / "configs").glob("setting2_fold*.yml")), ids=lambda p: p.name)
def test_setting2_config_uses_its_own_fold(path):
    cfg = load(path)
    k = path.stem[-1]
    assert cfg["train_basin_file"] == f"splits/setting2/fold{k}_train.txt"
    assert cfg["validation_basin_file"] == f"splits/setting2/fold{k}_train.txt"
    assert cfg["test_basin_file"] == f"splits/setting2/fold{k}_test.txt"


# ---------------------------------------------------------------- setting 3: data-scarce basins
SETTING3_CONFIGS = sorted((ROOT / "configs/setting3").glob("*.yml"))


def test_setting3_sample_comes_from_setting2_with_the_right_fold():
    sample = pd.read_csv(ROOT / "splits/setting3/sample.csv", dtype={"gauge_id": str})
    groups = pd.read_csv(SETTING2 / "groups.csv", dtype={"gauge_id": str}).set_index("gauge_id")
    for b, k in zip(sample.gauge_id, sample.fold):
        assert groups.fold[b] == k
        assert b in _basins(SETTING2 / f"fold{k}_test.txt")


@pytest.mark.parametrize("path", SETTING3_CONFIGS, ids=lambda p: p.name)
def test_setting3_config(path):
    cfg = load(path)
    kind, basin, years = path.stem.split("_")
    years = int(years[:-1])
    periods = load(ROOT / "splits/periods.yml")
    for key in ("train", "validation", "test"):
        assert _basins(ROOT / cfg[f"{key}_basin_file"]) == [basin]
        assert cfg[f"{key}_basin_file"] in (ROOT / "splits/SHA256SUMS").read_text()
    # the training window is the last `years` years of the frozen training period
    assert date(cfg["train_end_date"]) == pd.Timestamp(periods["train"][1])
    start = pd.Timestamp(periods["train"][1]) - pd.DateOffset(years=years) + pd.Timedelta(days=1)
    assert date(cfg["train_start_date"]) == start >= pd.Timestamp(periods["train"][0])
    assert date(cfg["test_start_date"]) == pd.Timestamp(periods["test"][0])
    assert date(cfg["test_end_date"]) == pd.Timestamp(periods["test"][1])
    assert cfg.get("validate_every") is None and not cfg.get("early_stopping", False)
    if kind == "scratch":
        assert not cfg.get("static_attributes")
        assert "base_run_dir" not in cfg
        assert set(cfg["dynamic_inputs"]) <= FORCING_OBS and cfg["target_variables"] == ["discharge_spec"]
    else:
        sample = pd.read_csv(ROOT / "splits/setting3/sample.csv", dtype={"gauge_id": str}).set_index("gauge_id")
        k = sample.fold[basin]
        base = ROOT / cfg["base_run_dir"]
        assert base.name.startswith(f"setting2_fold{k}_") or base.parent.name.startswith(f"setting2_fold{k}_")
        assert (base / "model_epoch030.pt").exists()
        base_cfg = load(base / "config.yml")
        assert basin not in _basins(ROOT / base_cfg["train_basin_file"])
        assert basin in _basins(ROOT / base_cfg["test_basin_file"])
        assert set(cfg["finetune_modules"]) == {"lstm", "head"}


# ---------------------------------------------------------------- gradient boosting
def test_gbm_features_use_no_future_forcing():
    spec = importlib.util.spec_from_file_location("gbm", ROOT / "scripts/gbm.py")
    gbm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gbm)
    basin = (ROOT / "splits/basins_all.txt").read_text().split()[0]
    df = xr.open_dataset(ROOT / f"data/processed/time_series/{basin}.nc").to_dataframe()[gbm.FORCING]
    changed = df.copy()
    changed.loc["2000-01-01":] = changed.loc["2000-01-01":] * 3 + 5
    a, b = gbm.features(df), gbm.features(changed)
    past = slice(None, "1999-12-31")
    pd.testing.assert_frame_equal(a.loc[past], b.loc[past])
    assert not a.loc["2000-01-01":].equals(b.loc["2000-01-01":])
    assert not any("discharge" in c for c in a.columns)
