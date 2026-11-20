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
        assert f.parent == ROOT / "splits"
        basins = f.read_text().split()
        assert len(basins) == len(set(basins))
    # no model selection: fixed number of epochs
    assert not cfg.get("early_stopping", False)
