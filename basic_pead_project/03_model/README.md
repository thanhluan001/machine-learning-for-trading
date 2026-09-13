# 03_model — training and freezing

01_train_model.py is the generic trainer pattern (read its header).
08_freeze_v6_gate_models.py builds the production V6: three XGBoost
gate classifiers (pass_g1/g2/g3), deployed as min(gates) >= 0.33.

models/phase_g_v6_gate_decomposition/ contains the FROZEN reference
model (classifiers + meta.json). It runs as-is; rebuild only after
re-running the full 04_backtest gauntlet on your own matrix.

Note on meta.json: the "status" field ("shadow_candidate_not_live")
records the model's freeze lineage — in the parent project V6 passed
its shadow period and runs the live paper book. Treat the meta as
provenance, not deployment state: in a fresh rebuild the model is a
candidate until YOUR gauntlet and paper period say otherwise.
