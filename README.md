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
- Forcing: MeteoSwiss observation-based grids shipped with CAMELS-CH (precipitation,
  minimum, mean and maximum temperature, relative sunshine duration).
- The forcing in the dataset's `simulation_based` folder is not used for the main runs.
  A design audit found that this product changes on 2010-01-01 (mean wind +65 %, PET
  +19 %, different daily precipitation window), which falls between the training and
  the test period. It is kept for one sensitivity run. PREVAH model output (simulated
  discharge, ET, interception) is never used as input.
- Static attributes: 45 attributes from topography, land cover, geology, soil, glacier
  cover, population density, and eight climate indices recomputed from training-period
  forcing (the indices shipped with the dataset cover 1981-2020, which includes the test
  period). The hydrological signature tables are excluded because they are computed from
  streamflow. Hydropower and reservoir attributes are not inputs because they are only
  populated inside Switzerland; they are used to flag basins. See `results/data_report/`.

### Basin selection

Of 331 catchments, 205 are used (175 CH, 14 AT, 13 FR, 2 DE, 1 IT). Excluded:

| Reason | Basins |
|---|---|
| Observation-based forcing incomplete (catchments outside the MeteoSwiss grids) | 53 |
| Lake (dataset readme) | 33 |
| Fewer than 5 years of valid discharge in the training period | 26 |
| Fewer than 2 years of valid discharge in the validation period | 3 |
| Fewer than 3 years of valid discharge in the test period | 2 |
| Discharge more than twice precipitation in the training period | 3 |
| Dataset readme recommendation (2446, 2447, 2327) | 3 |
| Duplicated or identical discharge series (2613, 4008, 4009) | 3 |

The per-basin table with quality-control columns is `results/data_report/basin_report.csv`.
`results/data_report/basin_flags.csv` flags regulated (81), major reservoir (48),
glacierised (32), lake-influenced (34), open water balance (18) and basins outside
Switzerland with unknown human influence (30). 75 basins carry no flag. Results will be
reported for all basins and for the unflagged subset.

### Periods (hydrological years)

| Period | From | To |
|---|---|---|
| Train | 1981-10-01 | 2005-09-30 |
| Validation | 2005-10-01 | 2010-09-30 |
| Test | 2010-10-01 | 2020-09-30 |

Split files live in `splits/` with SHA-256 sums and are checked by `tests/test_leakage.py`.

## Known open points

- 203 of the 205 basins are nested with at least one other basin. This does not affect
  setting 1 but requires grouped folds in setting 2.
- Input sequences for the first 364 days of the validation and test periods use forcing
  from the preceding period. No discharge from another period is used.
- Requiring three years of valid test-period discharge is a selection on data
  availability in the test period; it removes two basins.

## Reproduce

```
conda create -n camelsch -c conda-forge python=3.11 pip
conda activate camelsch
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python scripts/download_data.py
python scripts/prepare_data.py
pytest -q
nh-run train --config-file configs/baseline_seed1.yml
```

## Licence

Code: MIT. Dataset: CC-BY-4.0, by its authors.
