# architecture.md — how the pieces fit

```text
                       ┌──────────────── 01_data (build once, refresh nightly) ───────────────┐
 Tiingo ──prices──────▶│ /sp400/{pt}  /index/*                                             │
 Tiingo ──identity────▶│ /metadata/sp400_permatickers (+ membership intervals)               │
 FMP ────calendar─────▶│ /earnings/fmp  (future rows seeded, actuals back-filled lazily)     │
 FMP ────grades───────▶│ /grades/*     (report_date cutoffs, incremental)                    │
 FRED ──macros────────▶│ /macro/*                                                           │
                       └────────────────────────────┬───────────────────────────────────────┘
                                                    ▼ db.h5 (HDF5)
                       ┌──────────────── 02_features ──────────────────┐
                       │ gate events (universe + quality passes)       │
                       │ build matrix: 24 pre-event features + labels   │
                       └────────────────────────┬──────────────────────┘
                                /features/train_matrix
                                                    ▼
                       ┌──────────────── 03_model ──────────────────────┐
                       │ 01_train_model.py  (the generic pattern)       │
                       │ 08_freeze_v6_gate_models.py  (the real thing)  │
                       │   -> pass_g1 / pass_g2 / pass_g3 classifiers   │
                       │      min-gate >= 0.33 = eligible               │
                       └────────────────────────┬──────────────────────┘
                                frozen model dir
                                                    ▼
                       ┌──────────────── 04_backtest (gauntlet) ────────┐
                       │ 53 gates -> 54 nested CV -> 55 validate ->     │
                       │ 57 final holdout -> 61 bootstrap CI ->         │
                       │ 63 force-refresh sim (the live-trading rules)  │
                       └────────────────────────┬──────────────────────┘
                                policy + evidence
                                                    ▼
                       ┌──────────────── 05_live (paper) ───────────────┐
                       │ 01_fetch_and_predict.py  (nightly, plan.json) │
                       │ 02_paper_trade.py  (intraday, Alpaca paper)    │
                       │ 03_portfolio_report / 04_tax_report            │
                       └───────────────────────────────────────────────┘
```

Key invariants:
1. The model never sees the event's own earnings result (labels only).
2. The trade script trusts broker fills over local state (reconcile
   step) and refuses double-entries.
3. The frozen model directory is immutable; new versions get new names.
