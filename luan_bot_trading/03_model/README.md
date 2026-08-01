# 03_model/ — Model training

**This folder trains the deployable PEAD detection model.**

The strategy in 1 line (see `04_backtest/README.md` for the full
concise summary):

> **Binary Sunday classifier** (`binary:logistic`) on 24 Sunday-safe
> features (is_bmo removed, 8 FMP revision momentum added). Target:
> `pead_pass` (0/1 from 3 PEAD gates). Accept if **P(PEAD) >= 0.20**.
> Enter **pre-gap** (Close[T-1] BMO / Close[T] AMC), exit Close[T+5],
> no stop-loss. See `04_backtest/README.md` for the full summary.

---

## 1. Scripts (in execution order)

| # | Script | Role | Outputs |
|---|---|---|---|
| 01 | `01_train_model.py` | **(OBSOLETE main(), kept for shared utility API)** The original Phase F v2 listwise-ranker trainer. Its `main()` trains the leaky `rank:ndcg` ranker (Sharpe 4.31 edge was entirely from forward-looking `opening_gap_t1`; NaN-test → -0.14). DO NOT RUN. Kept because it exports the load-bearing helper API: `load_train_matrix()`, `apply_priming_cutoff()`, `DB_FILE`, `PRIMING_RUNWAY_START`, `split_walk_forward()`, etc., imported by every Phase G backtest script. | (no longer produces stable artifacts) |
| 02 | `02_phase_g_sunday_classifier.py` | **DEPLOYABLE MODEL TRAINER**. Trains `XGBClassifier` (objective `binary:logistic`, target = `pead_pass` from the 3 PEAD verification gates) on the 17 Sunday-safe features (no leak features). Imports `tm` (helpers) + `v3` (gate logic, from `04_backtest/_pead_target_retrain.py`). Single-VAL SINGLE-FOLD preview + threshold sweep. | `03_model/models/phase_g_v1_sunday_classifier/{classifier.json, calibrator.pkl, meta.json, threshold_sweep.csv}` |
| 03 | `03_phase_g_sweep.py`  *(v1.1, superseded by v2 below)* | **HYPERPARAMETER SELECTION SWEEP** (Phase G v1.1). 72-config grid sweep over `gamma`, `max_depth`, `min_child_weight`, `n_estimators`. Ranks configs by `sweep_val_pnl` and saves the best classifier artifact. | `03_model/models/phase_g_v1_1_sunday_sweep/{classifier.json, calibrator.pkl, leaderboard*, meta.json}` |
| 04 | `03_freeze_3class_model.py`  *(superseded by 05 below)* | **DEPRECATED 3-class trainer**. Trained `XGBClassifier` (objective `multi:softprob`, 3 classes: no/small/large PEAD). Superseded: 2-stage test proved CAR magnitude is unpredictable, 3-class split adds no value. | `03_model/models/phase_g_v2_3class/{classifier.json, meta.json}` |
| 05 | `04_freeze_binary_model.py` | **DEPLOYABLE MODEL TRAINER (v2 final)**. Trains `XGBClassifier` (objective `binary:logistic`, target = `pead_pass`) on the 24 Sunday-safe features. Saves the FROZEN artifact for live paper-trading. | `03_model/models/phase_g_v2_binary/{classifier.json, meta.json}` | | **HYPERPARAMETER SELECTION SWEEP** (Phase G v1.1). 72-config grid sweep over `gamma ∈ {3, 5, 10, 100}`, `max_depth ∈ {2, 3}`, `min_child_weight ∈ {50, 100, 1000}`, `n_estimators ∈ {200, 300, 500}`. Ranks configs by `sweep_val_pnl` and saves the best classifier artifact. | `03_model/models/phase_g_v1_1_sunday_sweep/{classifier.json, calibrator.pkl, leaderboard*, meta.json}` |

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

The final-deployable artifact is **`phase_g_v2_binary`** (the binary
classifier). To reproduce:

```bash
conda run -n trading python luan_bot_trading/03_model/04_freeze_binary_model.py
```

Runs ~2 seconds (single TRAIN+VAL fit on the current 16,789-row
train_matrix with 24 Sunday-safe features).

The 3-class model (`phase_g_v2_3class`) is kept for reference but is
**superseded**. The deep comparison (`34_binary_vs_3class_deep.py`)
showed binary beats 3-class on total return and win rate. The 2-stage
test (`33_two_stage_model.py`) proved the CAR magnitude split adds
no value.

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
| `phase_g_v1_1_sunday_sweep/` | `03_phase_g_sweep.py` | SUPERSEDED | Sunday-safe 17-feature v1.1 (gamma=10, HP-sweep winner). |
| `phase_g_v2_3class/` | `03_freeze_3class_model.py` | SUPERSEDED | 3-class softprob. Beaten by binary in deep comparison. |
| `phase_g_v2_binary/` | `04_freeze_binary_model.py` | **DEPLOYABLE** | Binary logistic on 24 Sunday-safe features (gamma=3). The FROZEN artifact for live paper-trading. |

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

## 6. Why binary over 3-class (model selection rationale)

The 3-class model (`phase_g_v2_3class`, now superseded) exhibited a **degenerate
argmax prediction** on the held-out VAL set: it predicts class 0 (no
PEAD) for 100% of events. This is a known consequence of the extreme
class imbalance (89.3% no PEAD / 6.3% small / 4.3% large).

**This does NOT affect live trading** because:

1. We threshold on **P(any PEAD) = P(small) + P(large)**, not on argmax.
2. Even when argmax is class 0, the model assigns nonzero probability to
   classes 1 and 2, and those probabilities carry the signal.
3. The VAL P(any PEAD) AUC is 0.63 -- modest but above random.
4. The 4-fold nested CV backtest (99 trades, +607% total PnL) validated
   that the probability thresholding works despite the degenerate argmax.

The 3-class model was **usable for live paper-trading** (probability thresholding worked despite the degenerate argmax), but the deep comparison (`34_binary_vs_3class_deep.py`) showed the binary model is superior. If non-degenerate
per-class predictions become needed in the future, options include:
class weighting (`scale_pos_weight`), SMOTE oversampling, or a 2-stage
binary-then-regression approach. These are deferred to a future
iteration.

---

End of README.
