# phase_g_v3_v4_era/ — gated-events era experiments (superseded, evidence chain retained)

One-time experiments from the Phase G v3 (honest 24-feature) → v4
(timing-correct 23-feature) era. Each script's conclusion is cited by
`Design.md` (§17.A/§17.C) and/or `strategy_v2_synthesis.md`; scripts are kept
so those citations remain executable/reproducible. None of these are part of
the current pipeline — the live chain is 51–65 in `../../`.

| Script | What it decided (citation) |
|---|---|
| `17_theta_sweep.py` | v3 operating point theta=0.20 (§17.A.5) |
| `19_practical_trade_stats.py` | v3 trade statistics |
| `22_bmo_amc_pregap.py` | THE timing contract: BMO entry Close[T-1], AMC Close[T], exit T+5 (§17.C.2; reused by 63/67 sims) |
| `24_delayed_stop.py` | −10% delayed stop (§17.A.5.8) |
| `30_hold_comparison_bootstrap.py` | 5d beats 10d hold (§17.A.5.7) |
| `35_macro_ab_test.py` | macros excluded — PEAD is stock-specific (§17.A.7) |
| `37_wider_stop_test.py` | wider stops rejected |
| `38_precision_investigation.py` | precision levers hurt total PnL (§17.A.11) |
| `40_false_positive_analysis.py` | FP anatomy |
| `41_exclude_xlf_test.py` | XLF inference exclusion (§17.A.5.3) |
| `42_xlf_excluded_detailed_stats.py` | XLF-excluded reference stats (§17.A.9) |
| `43_slot_utilization_analysis.py` | slot utilization |
| `44_slot_sweep_nav_sizing.py` | 4-slot sizing |
| `46_analyst_revision_probe.py` | analyst-revision-as-feature probe (see `../findings/analyst_revision_findings.md`) |
| `47_pre_event_19feature_backtest.py` | pre-event feature ladder 1 |
| `48_expanded_pre_event_features.py` | pre-event feature ladder 2 |
| `49_pre_event_with_macros.py` | pre-event + macros |
| `50_macro_feature_selection.py` | top-3 macro selection (fed the 23-feature set) |
| `52_hold_period_comparison.py` | hold-period comparison (v4 era) |
