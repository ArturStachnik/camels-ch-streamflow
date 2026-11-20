"""Checks that must pass before any training run. Run with: pytest -q"""
import hashlib
from pathlib import Path

import pandas as pd
import pytest
from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / "configs").glob("*.yml"))
# Hydrological signatures in CAMELS-CH (hydrology_attributes_obs / _sim): derived from streamflow.
SIGNATURES = {"q_mean", "runoff_ratio", "stream_elas", "slope_fdc", "baseflow_index_landson", "hfd_mean",
              "Q5", "Q95", "high_q_freq", "high_q_dur", "low_q_freq", "low_q_dur", "zero_q_freq"}
# PREVAH output and anything measured at the gauge: never an input.
FORBIDDEN_DYNAMIC = {"discharge", "waterlevel", "et_sim", "intercept", "swe"}


def load(path):
    return YAML(typ="safe").load(path.read_text())


def date(s):
    return pd.to_datetime(s, format="%d/%m/%Y")


def test_split_files_unchanged():
    for line in (ROOT / "splits/SHA256SUMS").read_text().splitlines():
        digest, name = line.split()
        assert hashlib.sha256((ROOT / "splits" / name).read_bytes()).hexdigest() == digest, name


def test_no_signatures_in_attribute_table():
    cols = set(pd.read_csv(ROOT / "data/processed/attributes/attributes.csv", nrows=1).columns)
    assert not cols & SIGNATURES


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_config(path):
    cfg = load(path)
    assert not set(cfg.get("static_attributes", [])) & SIGNATURES
    for name in cfg["dynamic_inputs"]:
        assert not any(bad in name for bad in FORBIDDEN_DYNAMIC), name
    assert not set(cfg["dynamic_inputs"]) & set(cfg["target_variables"])
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
