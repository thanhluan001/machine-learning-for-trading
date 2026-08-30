# archive/ — superseded experiments and findings

This folder holds all Phase G v1 scripts, superseded Phase G v2 experiments,
private diagnostic scripts, and historical findings docs. The current
deployable model scripts and docs are in the parent `04_backtest/` folder.

For the live strategy spec, see
[`../strategy_v2_synthesis.md`](../strategy_v2_synthesis.md).
For the current folder index, see [`../README.md`](../README.md).

---

## Folder structure

```
archive/
├── phase_g_v1/              # 13 scripts — Phase G v1 (gap filter era, all superseded)
├── phase_g_v2_superseded/   # 15 scripts — Phase G v2 experiments (rejected/superseded)
├── phase_g_v3_v4_era/       # 19 scripts — gated/v3/v4-era one-time experiments
│                            #   (theta sweeps, timing contract, stops, XLF,
│                            #   slot studies, pre-event feature ladder)
├── edge_search_2026/        # 6 scripts — 2026-08 slow-week edge search (closed):
│                            #   sp500 studies, analyst upgrades, insiders,
│                            #   dividend run-up, index additions
├── megatrend_wave_2026/     # 14 scripts — megatrend/RC-4 research wave (closed:
│                            #   advisory-only program; partial exposure failed
│                            #   every stress gate)
├── rc_programs_2026/        # 8 scripts — RC research-program wave Aug-Sep 2026
│                            #   (RC-9 PROMOTED to panel §[13]; RC-1/RC-11 and
│                            #   the T+4 exit test closed with evidence)
├── private_scripts/         # 7 scripts — private diagnostics and utilities
├── experiments/             # Output data (CSVs, JSONs, NPZs) from prior runs
└── findings/                # Findings docs (Docs §0, B–K + 2026 edge search;
                             #   authoritative summary: edge_landscape_memo.md)
```

---

## archive/phase_g_v1/ — Phase G v1 (13 scripts, ALL SUPERSEDED)

These scripts implement the Phase G v1 strategy with the `opening_gap_t1 ∈
[-15%, -2%]` gap filter. **The gap filter was DELETED in Phase G v2** (blocked
99.5% of PEAD events). These scripts are kept for historical reference only.

| Script | Role | Status |
|---|---|---|
| `01_val_backtest.py` | Stage 4 single-OOS backtest (Phase F era, leaky) | SUPERSEDED |
| `04_phase_g_portfolio.py` | Shared portfolio sim library (overlapping 10-day holds) | SUPERSEDED |
| `05_phase_g_oos_validation.py` | App C OOS forward-shifted validation | SUPERSEDED |
| `06_phase_g_nested_cv.py` | App D 4-fold nested CV (produces per-fold POS-tuned HP) | SUPERSEDED |
| `07_phase_g_ensemble.py` | App E multi-rule ensemble (POS+NEG) | SUPERSEDED |
| `08_phase_g_neg_tuned.py` | NEG-tuned retrain (harmful, do NOT use) | RESCINDED |
| `09_phase_g_neg_theta_sweep.py` | Doc F theta sweep → theta=0.20 | SUPERSEDED |
| `10_phase_g_neg_gap_sweep.py` | Doc G gap range sweep → [-15%, -2%] | SUPERSEDED |
| `11_phase_g_bootstrap_ci.py` | Doc H bootstrap CI | SUPERSEDED by `30_hold_comparison_bootstrap.py` |
| `12_phase_g_deadzone_skip.py` | Doc I dead-zone skip (rescinded by Doc J) | RESCINDED |
| `13_phase_g_deadzone_nested_cv.py` | Doc J nested CV of dead-zone selection | SUPERSEDED |
| `14_phase_g_trade_stats.py` | Trade-level win/loss analysis | SUPERSEDED by `42_xlf_excluded_detailed_stats.py` |
| `16_pead_capture_diagnostic.py` | PEAD capture diagnostic | SUPERSEDED |

---

## archive/phase_g_v2_superseded/ — Phase G v2 experiments (15 scripts)

These scripts tested specific hypotheses during Phase G v2 development. Each
was run once, the decision was made, and the script is no longer needed.
Kept for reproducibility — the findings are documented in
[`../strategy_v2_synthesis.md`](../strategy_v2_synthesis.md).

| Script | What it tested | Decision |
|---|---|---|
| `18_confirm_theta025.py` | theta=0.25 confirmation | SUPERSEDED (theta=0.20 chosen) |
| `20_path_analysis.py` | Intra-hold max drawdown by win/loss | Reference only |
| `21_pead_loser_deep_dive.py` | Why PEAD-labeled trades lose (gap eats drift) | Led to pre-gap entry (22) |
| `23_pregap_stop_loss.py` | Pre-gap entry + 3% stop test | SUPERSEDED by 24/37 |
| `25_theta_sweep_pregap.py` | Theta sweep with pre-gap entry | SUPERSEDED by `17_theta_sweep.py` |
| `26_three_class_classifier.py` | 3-class softprob classifier sweep | **REJECTED** (degenerate argmax) |
| `27_binary_vs_3class_head2head.py` | Binary vs 3-class head-to-head | **REJECTED** (binary wins) |
| `28_3class_lower_theta.py` | 3-class with lower theta sweep | **REJECTED** |
| `29_3class_detailed_stats.py` | Detailed 3-class stats | SUPERSEDED |
| `31_verify_large_pead.py` | Verify the 19 large PEAD tickers | One-off verification |
| `32_calibrated_sizing.py` | Calibrated position sizing test | **REJECTED** (marginal benefit) |
| `33_two_stage_model.py` | 2-stage (binary + CAR regression) | **REJECTED** (Stage 2 has no signal) |
| `34_binary_vs_3class_deep.py` | Deep binary vs 3-class comparison | **REJECTED** (binary wins) |
| `36_binary_detailed_stats.py` | Detailed binary model stats | SUPERSEDED by `42_xlf_excluded_detailed_stats.py` |
| `39_eps_filter_test.py` | eps_surprise_pct secondary filter | **REJECTED** (hurts total PnL) |

---

## archive/private_scripts/ — diagnostics and utilities (7 scripts)

| Script | Role | Status |
|---|---|---|
| `_phase_g_random_baseline.py` | Random baseline simulation (100 trials) | Reference (result documented) |
| `_phase_g_portfolio_sweep.py` | n_slots sweep helper | SUPERSEDED by `44_slot_sweep_nav_sizing.py` |
| `_pead_classifier.py` | PEAD-target binary classifier + threshold sweep | Historical (Doc §0) |
| `_pead_gap_strategy.py` | Model-free gap-driven backtest | Historical (Doc §0) |
| `_pead_exploration.py` | Gate statistics + gap stratification | Historical (Doc §0) |
| `_neg_gap_bucket_diag.py` | Gap-bucket contribution diagnostic | Historical (Doc G) |
| `_neg_gap_per_fold_diag.py` | Per-fold gap-bucket distribution | Historical (Doc G) |

> **Note:** `_pead_target_retrain.py` stays in the parent folder — it exports
> `compute_pead_gates_full` which is imported by current scripts (40-44, 46).

---

## archive/experiments/ — output data from v1 runs

| Folder/File | What's inside |
|---|---|
| `phase_g_v1_1_oos_20241231_n4/` | Single-OOS diagnostic (trades, equity curve, summary) |
| `phase_g_v1_1_nested_cv_n4/` | Per-fold POS-tuned HP source (`fold_results.csv`) |
| `phase_g_v1_1_ensemble_n4/` | Multi-rule ensemble run |
| `phase_g_v1_1_neg_tuned_n4/` | NEG-only HP re-tuning (do NOT use) |
| `phase_g_v1_1_neg_theta_sweep_n4/` | Theta sweep artifacts |
| `phase_g_v1_1_neg_gap_sweep_n4/` | Gap range sweep artifacts |
| `phase_g_v1_1_bootstrap_ci_n4/` | Bootstrap distributions + summary |
| `phase_g_v1_1_deadzone_skip_n4/` | Dead-zone skip run (rescinded) |
| `phase_g_v1_1_deadzone_nested_cv_n4/` | Nested-CV of dead-zone selection |
| `phase_g_portfoliosim_v1_1_two_stage_n4/` | Original v1.1 portfolio sim (circular, rescinded) |
| `phase_g_v1_1_trade_stats_n4/` | Per-fold trade-level analysis |
| `phase_g_portfolio_sweep.csv` | n_slots sweep across variants (legacy) |
| `phase_g_random_baseline_dist_n4.csv` | Random baseline distribution |

---

## archive/findings/ — 9 historical markdown docs

| Doc | File | Status |
|---|---|---|
| §0 | `pead_target_findings.md` | AUTHORITATIVE (foundational PEAD study) |
| B | `phase_g_findings.md` | SUPERSEDED |
| (neg-tuned) | `phase_g_neg_tuned_findings.md` | RESCINDED |
| F | `phase_g_neg_theta_sweep_findings.md` | SUPERSEDED |
| G | `phase_g_neg_gap_sweep_findings.md` | SUPERSEDED |
| H | `phase_g_bootstrap_ci_findings.md` | AUTHORITATIVE on CI methodology |
| I | `phase_g_deadzone_skip_findings.md` | RESCINDED |
| J | `phase_g_deadzone_nested_cv_findings.md` | AUTHORITATIVE (final word on dead-zone) |
| K | `phase_g_pos_vs_neg_findings.md` | AUTHORITATIVE (final word on POS-vs-NEG) |
