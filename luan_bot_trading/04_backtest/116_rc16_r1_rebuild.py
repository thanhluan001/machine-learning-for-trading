"""RC-16 R1 — corrected-contract matrix rebuilds (v6c, v7c).

Chain (all steps subprocess, env-keyed so frozen lineage is never touched):
  [1] 02_features/02_build_feature_matrix.py
        RC16_BASE_OUT=/features/train_matrix_v6c_base
        (F2 car_drift 45+mask, F3 label_end)
  [2] 03_model/06_freeze_timing_correct_model.py
        reads RC16_BASE_OUT, RC16_V6C_OUT=/features/train_matrix_v6c,
        RC16_MODEL_DIR=archive/experiments/rc16_r1/v6c_model_probe
        (F1 pub-aware macros in _prepare, F4 gates via compute_pead_gates_full)
  [3] 04_backtest/103_rc12b_phase3_matrix.py
        RC16_SP600_OUT=/features/train_matrix_sp600_pt_v7c
        (F1/F2/F3 ported to the SP600 builder)
  [4] 04_backtest/106_rc13_combined_matrix.py
        RC16_V6C_OUT + RC16_SP600_OUT -> RC16_COMBINED_OUT=/features/train_matrix_v7c
        (F4 unlabelled filter both sides, F3 label_end carried)

Delta reports (pre-specified in rc16_information_integrity_pre_registration.md,
written before any R2/R3 result is seen):
  base:      /features/train_matrix            vs v6c_base
  timing:    /features/train_matrix_v4_timing_correct vs v6c
  combined:  /features/train_matrix_combined   vs v7c
Metrics are contract/coverage only (row counts, NaN rates, base rates,
macro feature shifts on joined keys). No performance numbers.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DB = ROOT / "01_data" / "db.h5"
DB_SP600 = ROOT / "01_data" / "db_sp600.h5"
OUT_DIR = ROOT / "04_backtest" / "archive" / "experiments" / "rc16_r1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_OUT = "/features/train_matrix_v6c_base"
V6C_OUT = "/features/train_matrix_v6c"
SP600_OUT = "/features/train_matrix_sp600_pt_v7c"
COMBINED_OUT = "/features/train_matrix_v7c"

STEPS = [
    ("[1/4] base builder -> v6c_base",
     [sys.executable, str(ROOT / "02_features" / "02_build_feature_matrix.py")],
     {"RC16_BASE_OUT": BASE_OUT}),
    ("[2/4] timing-correct freeze -> v6c",
     [sys.executable, str(ROOT / "03_model" / "06_freeze_timing_correct_model.py")],
     {"RC16_BASE_OUT": BASE_OUT,
      "RC16_V6C_OUT": V6C_OUT,
      "RC16_MODEL_DIR": str(OUT_DIR / "v6c_model_probe")}),
    ("[3/4] SP600 pt matrix -> v7c",
     [sys.executable, str(HERE / "103_rc12b_phase3_matrix.py")],
     {"RC16_SP600_OUT": SP600_OUT}),
    ("[4/4] combined matrix -> v7c",
     [sys.executable, str(HERE / "106_rc13_combined_matrix.py")],
     {"RC16_V6C_OUT": V6C_OUT,
      "RC16_SP600_OUT": SP600_OUT,
      "RC16_COMBINED_OUT": COMBINED_OUT}),
]


def run_chain() -> list[dict]:
    log = []
    for name, cmd, extra_env in STEPS:
        env = {**os.environ, **extra_env}
        t0 = time.time()
        print(f"\n{'=' * 88}\n{name}\n{'=' * 88}", flush=True)
        with open(OUT_DIR / (name.split("]")[0].strip("[/") + ".log"), "w") as f:
            proc = subprocess.run(cmd, env=env, stdout=f,
                                  stderr=subprocess.STDOUT, cwd=str(ROOT))
        dt = time.time() - t0
        rec = {"step": name, "rc": proc.returncode, "seconds": round(dt)}
        print(f"  -> rc={proc.returncode} ({dt:.0f}s)", flush=True)
        log.append(rec)
        if proc.returncode != 0:
            print("CHAIN STOPPED — see step log", flush=True)
            break
    return log


# ----------------------------------------------------------------------------
# Delta reports
# ----------------------------------------------------------------------------
def _summ(s: pd.Series) -> dict:
    s = pd.to_numeric(s, errors="coerce")
    return {"n": int(len(s)), "non_nan": int(s.notna().sum()),
            "nan_pct": round(float(s.isna().mean()) * 100, 2),
            "mean": round(float(s.mean()), 6) if s.notna().any() else None,
            "std": round(float(s.std()), 6) if s.notna().sum() > 1 else None}


def delta_base() -> dict:
    old = pd.read_hdf(DB, "/features/train_matrix")
    new = pd.read_hdf(DB, BASE_OUT)
    out = {"old_rows": len(old), "new_rows": len(new),
           "old_tickers": int(old.permaTicker.nunique()),
           "new_tickers": int(new.permaTicker.nunique()),
           "old_range": [str(old.report_date.min().date()), str(old.report_date.max().date())],
           "new_range": [str(new.report_date.min().date()), str(new.report_date.max().date())],
           "car_drift_old_60d_unmasked": _summ(old.car_drift_historical_q1),
           "car_drift_new_45d_masked": _summ(new.car_drift_historical_q1)}
    # maturity effect by year (new NaN that old had filled = window past cutoff)
    o = old.set_index(pd.to_datetime(old.report_date)).car_drift_historical_q1.notna()
    n = new.set_index(pd.to_datetime(new.report_date)).car_drift_historical_q1.notna()
    yy = pd.DataFrame({"old_filled": o.values, "new_filled": n.values},
                      index=o.index).groupby(lambda d: d.year).mean()
    out["car_drift_filled_rate_by_year"] = {
        str(y): {"old": round(float(r.old_filled), 4), "new": round(float(r.new_filled), 4)}
        for y, r in yy.iterrows()}
    out["label_end_col_present"] = "label_end" in new.columns
    if "label_end" in new.columns:
        le = pd.to_datetime(new.label_end)
        out["label_end_na"] = int(le.isna().sum())
    return out


def delta_timing() -> dict:
    old = pd.read_hdf(DB, "/features/train_matrix_v4_timing_correct")
    new = pd.read_hdf(DB, V6C_OUT)
    out = {"old_rows": len(old), "new_rows": len(new),
           "row_delta_pct": round((len(new) - len(old)) / max(len(old), 1) * 100, 2),
           "old_pead_rate": round(float(old.pead_pass.mean()), 4),
           "new_pead_rate": round(float(new.pead_pass.mean()), 4),
           "old_gate_rates": {g: round(float(old[g].mean()), 4) for g in ("pass_g1", "pass_g2", "pass_g3")},
           "new_gate_rates": {g: round(float(new[g].mean()), 4) for g in ("pass_g1", "pass_g2", "pass_g3")},
           "old_bmo": int(old.is_bmo.sum()), "new_bmo": int(new.is_bmo.sum())}
    # F1 effect: macro features shifted on the same (permaTicker, report_date)
    key = ["permaTicker", "report_date"]
    m = old[key + ["vix", "fed_funds", "unemployment_roc21"]].merge(
        new[key + ["vix", "fed_funds", "unemployment_roc21"]],
        on=key, suffixes=("_old", "_new"))
    out["macro_joined_rows"] = len(m)
    out["macro_deltas"] = {}
    for c in ("vix", "fed_funds", "unemployment_roc21"):
        d = (pd.to_numeric(m[f"{c}_new"], errors="coerce")
             - pd.to_numeric(m[f"{c}_old"], errors="coerce")).abs()
        out["macro_deltas"][c] = {
            "pct_rows_changed": round(float((d > 1e-9).mean()) * 100, 2),
            "mean_abs_delta": round(float(d[d.notna()].mean()), 8) if d.notna().any() else None}
    # F2 effect on the frozen chain's own column
    out["car_drift_old"] = _summ(old.car_drift_historical_q1)
    out["car_drift_new"] = _summ(new.car_drift_historical_q1)
    if "label_end" in new.columns:
        le = pd.to_datetime(new.label_end)
        rd = pd.to_datetime(new.report_date)
        out["label_end_na"] = int(le.isna().sum())
        out["label_end_lag_days_median"] = float((le - rd).dt.days.median()) if le.notna().any() else None
    return out


def delta_combined() -> dict:
    old = pd.read_hdf(DB, "/features/train_matrix_combined")
    new = pd.read_hdf(DB, COMBINED_OUT)
    out = {"old_rows": len(old), "new_rows": len(new),
           "old_sp400": int((old.is_sp400 == 1).sum()), "new_sp400": int((new.is_sp400 == 1).sum()),
           "old_sp600": int((old.is_sp400 == 0).sum()), "new_sp600": int((new.is_sp400 == 0).sum()),
           "old_pead_rate": round(float(old.pead_pass.mean()), 4),
           "new_pead_rate": round(float(new.pead_pass.mean()), 4),
           "old_range": [str(pd.to_datetime(old.report_date).min().date()),
                         str(pd.to_datetime(old.report_date).max().date())],
           "new_range": [str(pd.to_datetime(new.report_date).min().date()),
                         str(pd.to_datetime(new.report_date).max().date())],
           "label_end_col_present": "label_end" in new.columns}
    return out


def main() -> int:
    t0 = time.time()
    run_log = run_chain()

    report = {"chain": run_log}
    try:
        report["delta_base"] = delta_base()
    except Exception as e:  # noqa: BLE001
        report["delta_base_error"] = repr(e)
    try:
        report["delta_timing"] = delta_timing()
    except Exception as e:  # noqa: BLE001
        report["delta_timing_error"] = repr(e)
    try:
        report["delta_combined"] = delta_combined()
    except Exception as e:  # noqa: BLE001
        report["delta_combined_error"] = repr(e)

    report["total_seconds"] = round(time.time() - t0)
    with open(OUT_DIR / "report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("\n" + "=" * 88)
    print("RC-16 R1 REPORT (contract deltas only — no performance numbers)")
    print("=" * 88)
    for k in ("delta_base", "delta_timing", "delta_combined"):
        if k in report:
            print(f"\n--- {k} ---")
            print(json.dumps(report[k], indent=2, default=str))
    print(f"\nwrote {OUT_DIR / 'report.json'}")
    print(f"total: {report['total_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
