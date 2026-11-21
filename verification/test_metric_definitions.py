"""Unit tests for the metric definitions in verify_metrics.py.

Every expected value is derived by hand from the published formula.
Run with:  python -m pytest verification/test_metric_definitions.py
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_metrics as vm  # noqa: E402

OBS = np.array([1.0, 2.0, 3.0, 4.0, 5.0])


def test_perfect_simulation():
    assert vm.nse(OBS, OBS) == 1.0
    k, r, alpha, beta = vm.kge(OBS, OBS)
    assert (k, alpha, beta) == (1.0, 1.0, 1.0)
    assert r == pytest.approx(1.0)
    assert vm.fhv(OBS, OBS) == 0.0
    assert vm.flv(OBS, OBS) == 0.0
    assert vm.pbias(OBS, OBS) == 0.0
    assert vm.log_nse(OBS, OBS) == 1.0


def test_mean_of_obs_gives_nse_zero():
    sim = np.full_like(OBS, OBS.mean())
    assert vm.nse(OBS, sim) == pytest.approx(0.0)


def test_nse_hand_value():
    # obs = 1..5, sim = obs + 1: numerator 5, denominator 4+1+0+1+4 = 10
    assert vm.nse(OBS, OBS + 1.0) == pytest.approx(0.5)


def test_kge_scaled_simulation():
    # sim = 2 * obs: r = 1, alpha = 2, beta = 2, KGE = 1 - sqrt(0 + 1 + 1)
    k, r, alpha, beta = vm.kge(OBS, 2.0 * OBS)
    assert r == pytest.approx(1.0)
    assert alpha == pytest.approx(2.0)
    assert beta == pytest.approx(2.0)
    assert k == pytest.approx(1.0 - math.sqrt(2.0))


def test_kge_shifted_simulation():
    # sim = obs + 3: r = 1, alpha = 1, beta = (3 + 3) / 3 = 2, KGE = 0
    k, r, alpha, beta = vm.kge(OBS, OBS + 3.0)
    assert alpha == pytest.approx(1.0)
    assert beta == pytest.approx(2.0)
    assert k == pytest.approx(0.0)


def test_kge_anticorrelated():
    # sim is obs reversed: r = -1, alpha = 1, beta = 1, KGE = 1 - 2
    k, r, alpha, beta = vm.kge(OBS, OBS[::-1])
    assert r == pytest.approx(-1.0)
    assert k == pytest.approx(-1.0)


def test_pbias_scaled():
    assert vm.pbias(OBS, 1.1 * OBS) == pytest.approx(10.0)
    assert vm.pbias(OBS, 0.5 * OBS) == pytest.approx(-50.0)


def test_fhv_uses_top_two_percent_of_each_curve():
    obs = np.arange(1.0, 101.0)            # top 2 % = {100, 99}, sum 199
    sim = obs.copy()
    sim[-2:] = [108.9, 110.0]              # top 2 % of sim = {110, 108.9}, sum 218.9
    assert vm.fhv(obs, sim) == pytest.approx(100.0 * (218.9 - 199.0) / 199.0)
    # a uniform 10 % overestimate is a 10 % high-flow bias
    assert vm.fhv(obs, 1.1 * obs) == pytest.approx(10.0)
    # the curves are sorted separately, so the order in time does not matter
    assert vm.fhv(obs, obs[::-1]) == pytest.approx(0.0)


def test_flv_scale_invariant_and_hand_value():
    obs = np.exp(np.arange(10.0))          # logs 0..9; bottom 30 % has logs 0, 1, 2
    # multiplying by a constant shifts the log curve and leaves its shape unchanged
    assert vm.flv(obs, 3.0 * obs) == pytest.approx(0.0)
    # sim logs 0, 2, 4 in the low segment: sum above minimum 6, obs 3 -> -100 * (6 - 3) / 3
    sim = np.exp(2.0 * np.arange(10.0))
    assert vm.flv(obs, sim) == pytest.approx(-100.0)
    # a flat simulated low-flow segment: -100 * (0 - 3) / 3
    assert vm.flv(obs, np.full(10, 5.0)) == pytest.approx(100.0)


def test_flv_zero_flow_is_floored_and_flagged():
    obs = np.exp(np.arange(10.0))
    sim = obs.copy()
    sim[0] = 0.0                           # becomes log(1e-6) in the simulated segment
    floor_log = math.log(vm.FLV_FLOOR)
    qsl = (1.0 - floor_log) + (2.0 - floor_log)
    assert vm.flv(obs, sim) == pytest.approx(-100.0 * (qsl - 3.0) / 3.0)
    info = vm.flv_segment_info(obs, sim)
    assert info["FLV_n_sim_zero"] == 1
    assert info["FLV_n_obs_zero"] == 0
    assert info["FLV_n_segment"] == 3
    assert info["FLV_segment_has_zero_or_near_zero"] == 1


def test_flv_constant_observed_segment_is_nan():
    obs = np.array([1.0, 1.0, 1.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
    assert math.isnan(vm.flv(obs, obs + 1.0))


def test_log_nse_hand_value():
    # with eps = 0 and obs = e^0..e^4, sim = e * obs shifts the logs by one: same as test_nse_hand_value
    obs = np.exp(np.arange(5.0))
    assert vm.log_nse(obs, math.e * obs, eps=0.0) == pytest.approx(0.5)


def test_nan_observations_are_excluded_pairwise():
    obs = np.array([1.0, np.nan, 3.0, 4.0, np.nan, 5.0, 2.0])
    sim = np.array([1.0, 99.0, 3.0, 4.0, -99.0, 5.0, 2.0])
    o, s = vm.pair_valid(obs, sim)
    assert o.tolist() == [1.0, 3.0, 4.0, 5.0, 2.0]
    assert s.tolist() == [1.0, 3.0, 4.0, 5.0, 2.0]
    values, n_valid = vm.all_metrics(obs, sim)
    assert n_valid == 5
    assert values["NSE"] == 1.0
    assert values["KGE"] == pytest.approx(1.0)


def test_degenerate_inputs_return_nan():
    assert math.isnan(vm.nse(np.array([2.0, 2.0, 2.0]), np.array([1.0, 2.0, 3.0])))
    assert math.isnan(vm.nse(np.array([]), np.array([])))
    assert math.isnan(vm.pbias(np.zeros(3), np.ones(3)))


def test_hydrological_year():
    import pandas as pd
    dates = pd.to_datetime(["2010-09-30", "2010-10-01", "2011-09-30", "2011-10-01"])
    assert list(vm.hydro_year(dates)) == [2010, 2011, 2011, 2012]
