# 03_model/ — Model training

**This folder trains the deployable PEAD detection model.**

The strategy in 1 line (see `04_backtest/README.md` for the full
concise summary):

> XGBoost Sunday classifier on 17 Sunday-safe features → `P(PEAD)`
> per upcoming-week earnings event. On T+1 morning, ENTER iff
> `P(PEAD) >= 0.20` AND realized `opening_gap_t1 ∈ [-15%, -2%]`.

---

## 1. Scripts (in execution order)

| # | Script | Role | Outputs |
|---|---|---|---|
| 01 | `01_train_model.py` | **(OBSOLETE main(), kept for shared utility API)** The original Phase F v2 listwise-ranker trainer. Its `main()` trains the leaky `rank:ndcg` ranker (Sharpe 4.31 edge was entirely from forward-looking `opening_gap_t1`; NaN-test → -0.14). DO NOT RUN. Kept because it exports the load-bearing helper API: `load_train_matrix()`, `apply_priming_cutoff()`, `DB_FILE`, `PRIMING_RUNWAY_START`, `split_walk_forward()`, etc., imported by every Phase G backtest script. | (no longer produces stable artifacts) |
| 02 | `02_phase_g_sunday_classifier.py` | **DEPLOYABLE MODEL TRAINER**. Trains `XGBClassifier` (objective `binary:logistic`, target = `pead_pass` from the 3 PEAD verification gates) on the 17 Sunday-safe features (no leak features). Imports `tm` (helpers) + `v3` (gate logic, from `04_backtest/_pead_target_retrain.py`). Single-VAL SINGLE-FOLD preview + threshold sweep. | `03_model/models/phase_g_v1_sunday_classifier/{classifier.json, calibrator.pkl, meta.json, threshold_sweep.csv}` |
| 03 | `03_phase_g_sweep.py` | **HYPERPARAMETER SELECTION SWEEP** (Phase G v1.1). 72-config grid sweep over `gamma ∈ {3, 5, 10, 100}`, `max_depth ∈ {2, 3}`, `min_child_weight ∈ {50, 100, 1000}`, `n_estimators ∈ {200, 300, 500}`. Ranks configs by `sweep_val_pnl` and saves the best classifier artifact. | `03_model/models/phase_g_v1_1_sunday_sweep/{classifier.json, calibrator.pkl, leaderboard*, meta.json}` |

**For the OOS generalization analysis (4-fold anchored walk-forward
nested CV)** that produces the per-fold POS-tuned gamma used in the
deployable rule (gamma = 10/5/3/3), see:

- `04_backtest/06_phase_g_nested_cv.py` (App D — the NESTED-CV runner).
  Its output `phase_g_v1_1_nested_cv_n4/fold_results.csv` is the
  load-bearing HP source used by `04_backtest/11_phase_g_bootstrap_ci.py`
  and 7 other diagnostic backtest scripts.
- See `04_backtest/README.md` for the headline OOS statistics.

---

## 2. Producing the FINAL deployable model

The final-deployable artifact is **already produced and saved at**
`03_model/models/phase_g_v1_sunday_classifier/`. To reproduce:

```bash
conda run -n trading python luan_bot_trading/03_model/02_phase_g_sunday_classifier.py
```

Or to reproduce the HP-sweep-selected phase_g_v1.1 variant:

```bash
conda run -n trading python luan_bot_trading/03_model/03_phase_g_sweep.py
```

Each runs ~30-180 seconds (single TRAIN+SWEEP fit on the current
20,265-row deduped train_matrix).

DO NOT run `01_train_model.py` main() -- it is the OBSOLETE Phase F v2
leaky `rank:ndcg` ranker. See the deprecation banner at the top of that
script for the longer explanation.

---

## 3. Saved model artifacts (`models/`)

| Folder | Producer | Status | Notes |
|---|---|---|---|
| `phase_f_baseline_v1/` | (OBSOLETE Phase F v1) | OBSOLETE | Trained on contaminated EODHD price data. DO NOT USE. |
| `phase_f_v2_baseline_ndcg/` | OBSOLETE Phase F v2 | LEAKY | The "Sharpe 4.31" leaky model. NaN-test confirmed forward-looking-leak. |
| `phase_f_v2_baseline_pairwise/` | OBSOLETE Phase F v2 | LEAKY | Pairwise variant of above. |
| `phase_f_v2_pead_classifier/` | OBSOLETE Phase F v2 | LEAKY | PEAD-target binary classifier w/ leak feature included. |
| `phase_f_v2_pead_target/` | OBSOLETE Phase F v2 | n/a | PEAD-target ranker variant. Diagnostic only. |
| `phase_g_v1_sunday_classifier/` | `02_phase_g_sunday_classifier.py` | **CANDIDATE** | Sunday-safe 17-feature v1 classifier (gamma=5). |
| `phase_g_v1_1_sunday_sweep/` | `03_phase_g_sweep.py` | **CANDIDATE** | Sunday-safe 17-feature v1.1 (gamma=10, HP-sweep winner). |

**NOTE**: Phase G nested-CV runs (App D -- `phase_g_v1_1_nested_cv_n4/`,
now archived at `04_backtest/archive/experiments/`) do **not** save a
single persistent model artifact — they fit a NEW model per fold with
that fold's POS-tuned gamma, and the per-fold HP table
(`fold_results.csv`) is the load-bearing product. They are diagnostic
runs, not deployable artifacts.

---

## 4. Folder layout

```
03_model/
├── README.md                       ← THIS FILE
├── 01_train_model.py               ← OBSOLETE main() (don't run), but
│                                     load-bearing shared helper API
├── 02_phase_g_sunday_classifier.py ← DEPLOYABLE model trainer
├── 03_phase_g_sweep.py             ← 72-config HP sweep for v1.1
├── models/
│   ├── phase_f_*/                  ← OBSOLETE Phase F artifacts (kept
│   │                                  for reference / smoke testing)
│   ├── phase_g_v1_sunday_classifier/    ← v1 saved candidate model
│   └── phase_g_v1_1_sunday_sweep/       ← v1.1 saved candidate model
└── __pycache__/
```

---

## 5. Cross-folder dependency map

```
                       01_train_model.py
                            (helpers)
                                ▲
                                │ imports tm
                            ┌───┴───┐
            02_phase_g_sunday    │       │  03_phase_g_sweep
                _classifier.py  │       │
                       │        │       │
                       │        │       │
                       └────────┘       │
                                         │
                      imports pg         │
                          ▼              │
                     (04_backtest scripts that run OOS backtests of
                     the saved model from 03_model/models/)
```

- `04_backtest/_pead_target_retrain.py` exports `compute_pead_gates_full`
  + the `v3` PEAD-gate logic; it's stored in `04_backtest/` because it's
  a feature-computation/diagnostic library for the backtest scripts.

---

End of README.
