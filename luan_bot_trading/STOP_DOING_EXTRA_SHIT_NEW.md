
## Env quirk (2026-09-16)
- trading env pandas 3.0.3/numpy 2.4.6: Series.corr/.cov NATIVELY CRASHES (rc=127, no traceback, intermittent onset). NEVER call .corr()/.cov() — use manual numpy pearson (see _rc18_threshold_probe2.py:pearson_np) or rank+manual-pearson for spearman. pandas-3 also: fillna(DatetimeIndex) rejected; Series.zfill gone (.str.zfill).
