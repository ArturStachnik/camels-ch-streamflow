# Independent verification of evaluation metrics

`verify_metrics.py` recomputes the evaluation of saved model outputs without
importing the modelling library and without using any of the modelling code.
It needs only numpy, pandas and xarray (xarray is required to unpickle the
result files).

## Run

From the repository root, once the three test evaluations exist:

```
python verification/verify_metrics.py \
    --run-dirs runs/baseline_seed1_2011_132323 runs/baseline_seed2_2011_132325 runs/baseline_seed3_2011_132328 \
    --period test --epoch 30 --out results/verification
```

Format fixture (older design, 259 basins, so the basin-set check is downgraded):

```
python verification/verify_metrics.py --run-dirs runs/smoke_test_2011_121338 \
    --period validation --epoch 3 --allow-basin-mismatch --out <scratch folder>
```

Unit tests of the metric definitions:

```
python -m pytest verification/test_metric_definitions.py
```

The exit code is 0 only if no hard check failed.

## Inputs

| Input | Default | Use |
|---|---|---|
| `<run>/<period>/model_epochNNN/<period>_results.p` | from `--run-dirs` | observed and simulated series per basin |
| `<run>/<period>/model_epochNNN/<period>_metrics.csv` | same folder | the library's own metric values, compared against |
| `<run>/config.yml` | | period dates and seed, compared with the frozen split (soft) |
| `splits/periods.yml`, `splits/basins_all.txt`, `splits/SHA256SUMS` | | frozen split definition |
| `data/raw/camels_ch_v0.9/.../timeseries/observation_based/CAMELS_CH_obs_based_<id>.csv` | | ground truth for observed discharge, column `discharge_spec(mm/d)` |
| `data/raw/camels_ch_v0.9/.../timeseries/simulation_based/CAMELS_CH_sim_based_<id>.csv` | | PREVAH benchmark, column `discharge_spec_sim(mm/d)` |
| `results/data_report/basin_flags.csv` | | `near_natural` subset |

The processed NetCDF inputs are not read at all.

## Metric definitions

All metrics use the days on which the observation is not missing (pairwise
exclusion). Computation is in float64.

- **NSE** (Nash & Sutcliffe 1970): `1 - sum((sim-obs)^2) / sum((obs-mean(obs))^2)`.
- **KGE** (Gupta et al. 2009): `1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)` with
  `r` the Pearson correlation, `alpha = std(sim)/std(obs)`, `beta = mean(sim)/mean(obs)`.
  The three components are reported as `KGE_r`, `KGE_alpha`, `KGE_beta`.
- **FHV** (Yilmaz et al. 2008): observed and simulated flows are sorted
  separately; with the top 2 % of each (`round(0.02 n)` values),
  `100 * sum(QS_h - QO_h) / sum(QO_h)`.
- **FLV** (Yilmaz et al. 2008): the lowest 30 % of each sorted series
  (`round(0.30 n)` values), in natural-log space, each relative to its own minimum:
  `-100 * (sum(log QS_l - log QS_min) - sum(log QO_l - log QO_min)) / sum(log QO_l - log QO_min)`.
- **PBIAS**: `100 * sum(sim - obs) / sum(obs)`.
- **logNSE**: NSE of `log(q + eps)` with `eps = 0.01 mm/d` (`--log-eps`); negative
  values are set to zero before the log.

### Zero flows in FLV

The published definition does not cover zero flow. Here every value `<= 0` in
either low-flow segment is replaced by `1e-6 mm/d` (`--flv-floor`) before the
logarithm. One such value places `log(1e-6) = -13.8` in the reference term of
the whole segment, so FLV can reach hundreds or thousands of percent and
depends on the floor chosen. If the observed segment is constant the
denominator is zero and FLV is NaN (no constant is added to the denominator).
For each basin the output holds `FLV_n_obs_zero`, `FLV_n_sim_zero`,
`FLV_n_obs_near_zero`, `FLV_n_sim_near_zero` (below 0.01 mm/d, `--near-zero`),
`FLV_min_obs`, `FLV_min_sim` and the flag `FLV_segment_has_zero_or_near_zero`.
For flagged basins use PBIAS and logNSE instead of FLV.

### Negative simulated values

Result files can contain negative simulated discharge even when the run was
configured with `clip_targets_to_zero`: in the fixture the library's metrics
are reproduced to about 2e-7 only if negative values are set to zero first,
and to about 1e-4 (NSE) and 2.5e-4 (KGE) otherwise. By default this script
evaluates the series exactly as saved. `--sim-clip zero` sets negative values
to zero first. `comparison_with_library.csv` always gives both variants, and
the number of negative values per basin is in `integrity.csv`.

## Checks

Status is `pass`, `fail` (hard, non-zero exit) or `finding` (soft, reported only).

| Check | Hard |
|---|---|
| SHA-256 of every file listed in `splits/SHA256SUMS` | yes |
| Basin set in each run equals the basin file exactly (`--basin-file` to use another list, `--allow-basin-mismatch` to downgrade) | yes |
| Every date inside the period from `periods.yml` | yes |
| No duplicated dates | yes |
| No NaN or infinite simulated value | yes |
| Observations in the results equal the raw observed discharge on the same dates (default tolerance 0.0005 mm/d; raw resolution is 0.001); also no value where the raw file has none. `obs_vs_raw.csv` additionally gives the difference against the raw series moved by one day each way | yes |
| All runs hold identical dates and observations | yes |
| Own NSE and KGE agree with the library file within 1e-3 | yes |
| Own FHV within 0.1 percentage points; own FLV within 0.1 (see zero handling) | no |
| Period fully covered; simulations non-negative; no observation dropped relative to raw; config dates equal `periods.yml`; seeds distinct; flags cover all basins; FLV low-flow segment free of zeros | no |
| Alarm: any basin NSE > 0.98 | yes |
| Alarm: median NSE > 0.90 | yes |
| Alarm: a simulation identical to the observations | yes |
| Alarm: for more than 10 % of basins the correlation of sim with obs moved by one day (either direction) exceeds the same-day correlation | yes |
| Alarm: median NSE differs by more than 0.03 between seeds | yes |

Alarms are computed for every seed and for the ensemble.

## Benchmarks

Both use raw data only and the days on which the basin has an observation in
the results.

- `climatology_doy_train`: mean observed discharge per calendar day (month, day)
  over the training period only. 29 February falls back to 28 February if absent.
- `prevah_simulation_based`: the simulated discharge shipped with CAMELS-CH.
  PREVAH is a calibrated process-based model whose calibration may have used
  the evaluation period. It is context, not a fair out-of-sample competitor.

## Outputs (`--out`)

- `per_basin_metrics.csv`: long format `basin, member, metric, value, n_valid, first_date, last_date`;
  `member` is the run directory name or `ensemble`.
- `summary.csv`: per member (seeds, ensemble, both benchmarks), subset (`all`,
  `near_natural`, `not_near_natural`) and metric: number of basins, median, mean,
  5/25/75/95 percentiles, min, max, number below zero (NSE, KGE, logNSE).
- `comparison_with_library.csv`: own value, library value, absolute difference
  and tolerance per basin, seed and metric, plus the variant with negative
  simulated values set to zero.
- `benchmarks.csv`: per-basin benchmark metrics.
- `per_year.csv`: median NSE across basins per hydrological year (October to
  September, labelled by the year in which it ends). NSE is computed within
  each year with that year's observed mean; a basin enters a year with at
  least 300 valid days (`--min-days-per-year`).
- `integrity.csv`, `obs_vs_raw.csv`, `shift_check.csv`: per-basin detail behind the checks.
- `verification_report.json`: every check with status and details, the command
  line, parameters, SHA-256 of all input files, and a UTC timestamp.

The ensemble is the mean of the simulated series across runs on their common
dates, for basins present in every run.
