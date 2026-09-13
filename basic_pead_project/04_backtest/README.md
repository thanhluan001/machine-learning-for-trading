# 04_backtest — the validation gauntlet

Run order (artifacts chain): 51 (base lib, imported not run) →
53_gate_decomposition_v6 → 54_gate_decomposition_v6_nested →
55_validate_gate_decomposition_v6 → 57_validate_v6_final_holdout →
61_v6_threshold_bootstrap → 63_force_refresh_backtest.

Outputs land in archive/experiments/gate_decomposition_v6/ (created on
first run). 63 validates the live trading rules themselves (4 slots,
T+5, -10% stop, force-refresh) — it is the bridge between the model
and 05_live.

Doctrine: docs/validation_doctrine.md. Do not iterate parameters here;
run, read, and either accept or kill.
