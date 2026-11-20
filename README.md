# Multi-basin streamflow modelling on CAMELS-CH

**Status: work in progress. No verified results yet.** Only the data pipeline and a
pipeline smoke test exist so far. Numbers will be added here only after they have been
recomputed independently from saved predictions.

## Question

How well does a single regional LSTM, trained on many catchments of hydrologic
Switzerland at once, predict daily streamflow in three settings?

1. Unseen period in gauged basins.
2. Ungauged basins (k-fold cross-validation over basins).
3. Data-scarce basins (pre-train on other basins, fine-tune with 1, 2 and 5 years).

## Data

CAMELS-CH (Höge et al., 2023, ESSD 15, 5755-5784), version 0.9 from Zenodo
(concept DOI [10.5281/zenodo.7784632](https://doi.org/10.5281/zenodo.7784632), CC-BY-4.0).
The dataset is not redistributed here. `scripts/download_data.py` downloads it and
checks the MD5 sums published by Zenodo.

- Target: observed specific discharge, `discharge_spec` (mm/d), daily.
- Forcing: the gridded meteorology in the dataset's `simulation_based` folder
  (precipitation, temperature, radiation, relative humidity, wind, PET). It is used
  because it is complete for all catchments; the MeteoSwiss observation-based temperature
  and sunshine grids are missing for about 55 catchments outside Switzerland.
  PREVAH model output (simulated discharge, ET, interception) is never used as input.
- Static attributes: 49 attributes from topography, climate, land cover, geology, soil,
  glacier and human-influence tables. The hydrological signature tables are excluded
  because they are computed from streamflow. See `results/data_report/`.

### Basin selection

Of 331 catchments, 259 are used. Excluded: 33 lakes and 3 gauges that the dataset
readme recommends excluding (2446, 2447, 2327), and 36 gauges with fewer than 5 years
of valid discharge in the training period or fewer than 2 years in the validation or
test period. The per-basin table is `results/data_report/basin_report.csv`.

### Periods (hydrological years)

| Period | From | To |
|---|---|---|
| Train | 1981-10-01 | 2005-09-30 |
| Validation | 2005-10-01 | 2010-09-30 |
| Test | 2010-10-01 | 2020-09-30 |

Split files live in `splits/` with SHA-256 sums and are checked by `tests/test_leakage.py`.

## Known open points

- The climate attributes shipped with CAMELS-CH are computed over 1981-2020, which
  includes the test period. They use forcing only, not discharge. To be decided before
  the baseline: recompute them from the training period.
- Nested gauges on the same river are not yet handled; this matters for setting 2.
- Regulated, glacierised and lake-influenced basins are included; results will be
  reported with and without them.

## Reproduce

```
conda create -n camelsch -c conda-forge python=3.11 pip
conda activate camelsch
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python scripts/download_data.py
python scripts/prepare_data.py
pytest -q
nh-run train --config-file configs/smoke_test.yml
```

## Licence

Code: MIT. Dataset: CC-BY-4.0, by its authors.
