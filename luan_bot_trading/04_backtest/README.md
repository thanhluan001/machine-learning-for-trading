# 04_backtest/ -- PEAD Strategy Backtest

> The current paper-executable research candidate is **phase_g_v6_gate_decomposition**.
> V4 remains the frozen comparison baseline. Historical Phase G/v3 sections and
> archived experiments are retained for provenance only.

---

## 1. Current V6 paper policy

V6 uses three independently trained binary classifiers on the same 23-feature,
timing-correct matrix:

```text
pass_g1: CAR > +3%
pass_g2: event volume ratio > 2x baseline
pass_g3: market-adjusted MaxDD > -1.5%
```

Executable score:

```python
v6_score = min(p_pass_g1, p_pass_g2, p_pass_g3)
accept if v6_score >= 0.33
```

| Parameter | Value |
|---|---|
| Model | `phase_g_v6_gate_decomposition` |
| Matrix | `/features/train_matrix_v4_timing_correct` |
| Features | 23 timing-correct honest features |
| Gate models | `pass_g1`, `pass_g2`, `pass_g3` |
| Ensemble | `min(p_pass_g1, p_pass_g2, p_pass_g3)` |
| Threshold | `0.33` (raised from 0.30 2026-08-13; bootstrap-validated, `61_v6_threshold_bootstrap.py`) |
| Entry | Close[T-1] BMO / Close[T] AMC |
| Hold | Close[T+5] |
| Stop | -10% delayed, skip gap day (sim/backtest convention) |
| Sector exclusion | XLF at inference |
| Slots | 4, equal-weight 1/4 NAV |
| Slot policy | Weekly slot-refresh (force-refresh, mh=4 guard) — displaces oldest prior-week position held >= 4 trading days when full (`63`/`64`; deployed in `05b_alpaca_live/02_paper_trade.py`) |
| Order type | Immediate DAY market orders (buys AND sells) in paper; run Script 02 near close (~3:45 PM ET); live buy type deferred to promotion |

The historical V6 policy does not apply the V4 LEG-like filter. The current
paper pipeline keeps V4 comparison output separate from V6 execution.

---

## 2. V6 validation reference

Frozen final holdout: train through 2025-06-30, sweep/refit through 2025-12-31,
untouched final test in 2026 H1.

| Metric | Frozen V6 final holdout |
|---|---:|
| Executed trades | 47 |
| Win rate | 63.8% |
| Average trade | +3.88% |
| Compounded NAV | +53.4% |
| Average-trade bootstrap 95% CI | +0.19% to +7.93% |
| Four-week block-NAV bootstrap 95% CI | -5.9% to +175.1% |

V6 remains a controlled paper-trading candidate, not a live-capital promotion.
The full validation records are in:

```text
archive/experiments/gate_decomposition_v6/policy.json
archive/experiments/gate_decomposition_v6/nested_results.json
archive/experiments/gate_decomposition_v6/stability_validation.json
archive/experiments/gate_decomposition_v6/final_holdout.json
archive/findings/gate_decomposition_v6_findings.md
```

---

## 3. V4 comparison baseline

V4 is the single timing-correct classifier:

```text
phase_g_v4_timing_correct
P(PEAD) >= 0.20
XLF excluded
LEG-like filter applied in V4 comparison plan
```

Reference four-fold result:

```text
99 executed trades
57.6% win rate
+2.78% average trade
+89.7% compounded NAV
```

V4 is generated into `05b_alpaca_live/v4_plan.json` and recorded in
`v4_shadow_trades.json`; it is not read by the execution script.

---

## 4. Live paper comparison workflow

```bash
python 05b_alpaca_live/01_fetch_and_predict.py --weeks 2
python 05b_alpaca_live/02_paper_trade.py
```

Script 01 writes:

```text
plan.json              V6 executable plan
v4_plan.json           V4 comparison-only plan
v4_shadow_trades.json  V4 hypothetical entry/exit ledger
```

Script 02 reads only `plan.json`, verifies that its model is V6, and executes
only V6 candidates. Entry selection is the weekly slot-refresh policy
(force-refresh, mh=4 guard): each ISO week's top-4 threshold-passers by score
enter; if all 4 slots are full, the oldest position from a PRIOR week held
>= 4 trading days is force-sold to make room. No V4 orders are placed.

---

## 5. Research scripts

| Script | Role |
|---|---|
| `51_hp_theta_sweep_23feat.py` | Shared library (bt): DEPLOY_FEATURES, folds, macro joins — imported by 60-65 |
| `_pead_target_retrain.py` | Shared library: the 3 gate definitions (`compute_pead_gates_full`) |
| `53_gate_decomposition_v6.py` | Fixed-HP/descriptive V6 gate experiment |
| `54_gate_decomposition_v6_nested.py` | Nested V6 walk-forward validation |
| `55_validate_gate_decomposition_v6.py` | V6 trade/week bootstrap |
| `56_validate_gate_decomposition_v6_stability.py` | Fixed policy and dependence validation |
| `57_validate_v6_final_holdout.py` | Frozen-policy final holdout |
| `60_v6_threshold_sensitivity.py` | Threshold sensitivity (0.30-0.35) |
| `61_v6_threshold_bootstrap.py` | Threshold bootstrap -> 0.33 decision |
| `62_v6_per_fold_stats.py` | Per-fold execution stats at 0.33 |
| `63_force_refresh_backtest.py` | Force-refresh vs conviction-priority slot sim (shared simulator) |
| `64_force_refresh_guard_bootstrap.py` | mh guard sweep + bootstrap -> mh=4 decision |
| `65_shap_pick.py` | Live diagnostic: per-pick gate SHAP attribution (native TreeSHAP) |

Historical scripts and findings are under `archive/` (see `archive/README.md`;
the 2026-08 edge search lives in `archive/edge_search_2026/` with the
authoritative summary in `archive/findings/edge_landscape_memo.md`).

The RC-4 megatrend watcher is no longer a backtest script. Its operational
manual monthly panel report lives in:

```text
05c_megatrend_watcher/monthly_panel_report.py
05c_megatrend_watcher/README.md
```

Research scripts and findings remain here under `archive/` for provenance.
