# Grades-Historical v5 Rejection

**Status:** rejected; not production.
**Date:** 2026-08-07
**Live model retained:** `phase_g_v4_timing_correct`

## Experiment

Added point-in-time FMP `/grades-historical` aggregate-rating features to the
v4 timing-correct matrix:

- 9 aggregate rating features
- `gh_data_available`
- `gh_n_missing`
- AMC cutoff: latest snapshot on or before T-1
- BMO cutoff: latest snapshot on or before T-2

The feature matrix contained 16,789 rows and 34 features. Historical rating
coverage was approximately 63.8%; missing history was retained as NaN with
explicit availability indicators.

## Strict v4/v5 comparison using v4 HP

Both feature sets used the same four walk-forward folds, theta=0.20, XGBoost
training, XLF exclusion, four-slot portfolio selection, pre-gap entry,
five-day hold, and delayed stop logic.

| Metric | v4 | v5 | Delta |
|---|---:|---:|---:|
| Executed trades | 99 | 106 | +7 |
| Win rate | 57.6% | 55.7% | -1.9 pp |
| Average trade | +2.78% | +2.83% | +0.05 pp |
| PEAD precision | ~24.3% | ~24.3% | flat |
| Compounded NAV | +89.7% | +100.7% | +11.0 pp |
| Maximum drawdown | -12.46% | -11.41% | slightly better |

The earlier temporary result of 108 trades, 59.3% win rate, +3.73% average
trade, and +158.3% NAV was not reproducible and is discarded.

## HP-tuned comparison

A 60-combination grid was run separately for v4 and v5:

- gamma: 1, 3, 5, 8, 12
- min_child_weight: 20, 50, 100, 200
- max_depth: 2, 3, 4
- n_estimators: 300

Selecting directly on all held-out folds is descriptive, not a promotion-grade
selection procedure, because it uses test results. Even under this favorable
comparison, v5 did not establish a robust improvement.

### Best by PEAD F1

- v4: gamma=1, mcw=20, depth=4; F1 24.12, NAV +115.8%, win rate 58.2%.
- v5: gamma=1, mcw=100, depth=4; F1 23.91, NAV +58.6%, win rate 56.7%.

### Best by NAV

- v4: gamma=5, mcw=20, depth=2; NAV +155.4%, win rate 60.8%, precision 25.7%.
- v5: gamma=5, mcw=100, depth=2; NAV +153.4%, win rate 56.6%, precision 24.2%.

Thus the grades-historical features were rejected: they changed rankings but
did not improve PEAD detection, win rate, or NAV robustly after feature-specific
HP tuning. No v5 artifact is used by live inference.

## Retained data

The raw point-in-time FMP grades-historical nodes remain in
`/analyst/grades_historical/{permaTicker}` for possible future research. The
fetcher is retained at `01_data/08_fmp_grades_historical_gathering.py`.
The derived v5 matrix, frozen v5 classifier, and v5-specific scripts are
removed from the active workspace after this archival record.

## Next experiment boundary

Start from the v4 timing-correct matrix and preserve its information cutoff,
walk-forward protocol, and execution assumptions. Do not use the rejected v5
feature set or artifact unless a new experiment explicitly reopens this
finding.
