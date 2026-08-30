"""91_rc1_phase2_univariate.py — RC-1 Phase 2: univariate screen.

Pre-registration: 04_backtest/rc1_pre_registration.md (2026-08-30).

KILL GATE 1 (pre-registered): each feature survives only with
same-sign rank-IC in >= 3 of 4 folds AND pooled |t| >= 2 (vs the
gate labels pass_g1/g2/g3 and the CAR5 outcome). All four die ->
close RC-1.

Folds: the same DEFAULT_FOLDS used by the V6 machinery (script 51).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
DB = HERE.parent / "01_data" / "db.h5"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

FEATS = ["insider_net_buy_90d", "insider_cluster_90d",
         "insider_sell_pressure_30d", "insider_days_since_last_buy"]
TARGETS = ["pass_g1", "pass_g2", "pass_g3", "car_5d"]

# fold windows from script 51's DEFAULT_FOLDS (te, sweep_start, test_end)
FOLDS = [(1, "2024-07-01", "2024-12-31"),
         (2, "2025-01-01", "2025-06-30"),
         (3, "2025-07-01", "2025-12-31"),
         (4, "2026-01-01", "2026-06-30")]


def main() -> None:
    print("=" * 100)
    print("RC-1 PHASE 2 — univariate rank-IC screen (kill gate 1)")
    print("=" * 100)

    mx = pd.read_hdf(DB, "/features/train_matrix_v4_timing_correct")
    rc1 = pd.read_hdf(DB, "/features/rc1_insider_features")
    rc1["report_date"] = pd.to_datetime(rc1["report_date"])
    mx["report_date"] = pd.to_datetime(mx["report_date"])
    d = mx.merge(rc1, on=["permaTicker", "report_date"], how="left")
    print(f"joined: {len(d):,} rows | targets present: "
          f"{[t for t in TARGETS if t in d.columns]}")

    # find the CAR5 outcome column name if not exact
    if "car_5d" not in d.columns:
        cand = [c for c in d.columns if "car" in c.lower() or "ret" in c.lower()][:8]
        print(f"  car_5d missing; candidates: {cand}")
        # use pregap_return (the CAR-style outcome the simulator uses)
        if "pregap_return" in d.columns:
            d["car_5d"] = d["pregap_return"]
            print("  -> using pregap_return as CAR5 outcome")

    rd = d.report_date
    fold_masks = {i: (rd >= pd.Timestamp(s)) & (rd <= pd.Timestamp(e))
                  for i, s, e in FOLDS}

    results = {}
    for f in FEATS:
        print(f"\n--- {f} ---")
        rows = []
        for tgt in TARGETS:
            if tgt not in d.columns:
                continue
            ics, ns = [], []
            for fi, m in fold_masks.items():
                sub = d.loc[m, [f, tgt]].dropna()
                if len(sub) < 50:
                    ics.append(np.nan); ns.append(len(sub)); continue
                ic, _ = stats.spearmanr(sub[f], sub[tgt])
                ics.append(ic); ns.append(len(sub))
            ics_a = np.array(ics, dtype=float)
            valid = ~np.isnan(ics_a)
            # pooled IC (all folds concatenated)
            pooled_mask = fold_masks[1] | fold_masks[2] | fold_masks[3] | fold_masks[4]
            sub = d.loc[pooled_mask, [f, tgt]].dropna()
            pic, _ = stats.spearmanr(sub[f], sub[tgt])
            se = 1.0 / np.sqrt(len(sub))
            t = pic / se
            signs = np.sign(ics_a[valid])
            same = int((signs == np.sign(pic)).sum())
            survive = (same >= 3) and (abs(t) >= 2)
            rows.append({"target": tgt, "fold_ics": np.round(ics_a, 3).tolist(),
                         "pooled_ic": round(pic, 4), "t": round(t, 2),
                         "same_sign_folds": f"{same}/{valid.sum()}",
                         "verdict": "SURVIVES" if survive else "dies"})
            print(f"  {tgt:10s} folds {np.round(ics_a, 3).tolist()}  pooled {pic:+.4f} "
                  f"(t={t:+.2f}, n={len(sub):,})  same-sign {same}/{valid.sum()}  "
                  f"{'SURVIVES' if survive else 'dies'}")
        results[f] = rows

    # quartile CAR5 table for the dollar features
    print("\n--- CAR5 by feature quartile (pooled folds, NaN excluded) ---")
    pooled_mask = fold_masks[1] | fold_masks[2] | fold_masks[3] | fold_masks[4]
    for f in FEATS:
        sub = d.loc[pooled_mask, [f, "car_5d"]].dropna()
        if len(sub) < 200 or sub[f].nunique() < 4:
            continue
        try:
            q = pd.qcut(sub[f], 4, labels=False, duplicates="drop")
        except ValueError:
            continue
        m = sub.groupby(q)["car_5d"].agg(["mean", "count"])
        mono = np.all(np.diff(m["mean"].values) >= 0) or np.all(np.diff(m["mean"].values) <= 0)
        print(f"  {f:32s} " + " | ".join(f"Q{int(i)+1} {r['mean']:+.3%} (n={int(r['count'])})"
                                         for i, r in m.iterrows())
              + ("  [monotone]" if mono else ""))

    n_survive = sum(any(r["verdict"] == "SURVIVES" for r in rows) for rows in results.values())
    print("\n" + "=" * 100)
    print(f"PHASE 2 VERDICT: {n_survive}/4 features survive kill gate 1 "
          f"-> {'PROCEED to Phase 3 with survivors' if n_survive else 'CLOSE RC-1'}")
    print("=" * 100)


if __name__ == "__main__":
    main()
