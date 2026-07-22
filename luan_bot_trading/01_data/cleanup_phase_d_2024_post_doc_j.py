"""
Cleanup script: Post-Phase-D-and-doc-J maintenance.

Actions:
1. Drop `legacy_perm_id` column from /metadata/sp400_permatickers
   (962 x 11 -> 962 x 10). Functional references are zero (only
   docstrings mention it); the column was a Phase-A->D bridge that
   is no longer needed after the permaTicker migration completed.
2. Dedup /earnings/raw by (permaTicker, report_date), keeping the
   row with the alphabetically-smallest cik per dup group.
   This removes 626 dup groups (1,252 – 626 = 626 redundant rows).
   Because ALL 626 dup groups have IDENTICAL non-cik columns (only
   cik differs), this loses no information -- it just removes the
   duplicate rows.

Downstream features (/features/gated_events, /features/train_matrix)
will need RE-GENERATION to remove the 349 propagated dup groups in
train_matrix. This script NOTE: only the DB level cleanup is done
here. The user must re-run:
    02_features/01_features_gate_events.py
    02_features/02_build_feature_matrix.py
to refresh the downstream tables.

NO EXTERNAL API calls.
"""
from __future__ import annotations
import sys, pandas as pd, numpy as np
from pathlib import Path

try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

DB = Path("C:/Users/thanh/Projects/machine-learning-for-trading/luan_bot_trading/01_data/db.h5")
PERMATICKERS_KEY = "/metadata/sp400_permatickers"
EARNINGS_KEY = "/earnings/raw"

def _backup_size(path: Path, suffix: str) -> str:
    p = path.with_name(path.name + suffix)
    if p.exists():
        return f"{p.stat().st_size / 1e6:.1f} MB"
    return "not found"

print("=" * 70)
print("CLEANUP — Phase D / Doc-J / legacy_perm_id")
print("=" * 70)
print(f"DB: {DB}")
print(f"DB size before: {DB.stat().st_size / 1e6:.2f} MB")

# Safety: confirm key uniqueness assumptions
with pd.HDFStore(DB, mode="r") as s:
    permatickers = s[PERMATICKERS_KEY]
    earnings = s[EARNINGS_KEY]
print(f"\n[1] Verifying current state:")
print(f"  /metadata/sp400_permatickers: {len(permatickers):,} rows x "
      f"{len(permatickers.columns)} cols")
print(f"  Columns: {permatickers.columns.tolist()}")
assert "legacy_perm_id" in permatickers.columns, \
    "Expected legacy_perm_id in /metadata/sp400_permatickers -- abort."

print(f"\n  /earnings/raw: {len(earnings):,} rows x {len(earnings.columns)} cols")
print(f"  Columns: {earnings.columns.tolist()}")
print(f"  (permaTicker, report_date) dup rows: "
      f"{int(earnings.duplicated(subset=['permaTicker', 'report_date']).sum())}")

# === Step 1: Drop legacy_perm_id from /metadata/sp400_permatickers ===
print(f"\n[2] Dropping 'legacy_perm_id' column from "
      f"/metadata/sp400_permatickers ...")
n_with_legacy = int(permatickers["legacy_perm_id"].notna().sum())
print(f"  legacy_perm_id non-null count: {n_with_legacy} of "
      f"{len(permatickers)} rows -- informational only at this point")
permatickers_new = permatickers.drop(columns=["legacy_perm_id"]).copy()
print(f"  new shape: {permatickers_new.shape}")
assert permatickers_new.shape[1] == permatickers.shape[1] - 1, \
    f"shape mismatch: {permatickers_new.shape} vs {permatickers.shape}"

# Write back via HDFStore mode='a' + store.remove() (NEVER mode='w')
print(f"  Writing back to DB ...")
with pd.HDFStore(DB, mode="a") as s:
    if PERMATICKERS_KEY in s.keys():
        s.remove(PERMATICKERS_KEY)
    s.put(PERMATICKERS_KEY, permatickers_new, format="table")
print(f"  OK: /metadata/sp400_permatickers now "
      f"{permatickers_new.shape[0]:,} x {permatickers_new.shape[1]}")

# === Step 2: Dedup /earnings/raw ===
print(f"\n[3] Deduping /earnings/raw by (permaTicker, report_date) ...")
print(f"  Before: {len(earnings):,} rows")

# Verifying the assumption (all dup groups have identical non-(key,cik) cols)
dup_key = ["permaTicker", "report_date"]
dup_mask = earnings.duplicated(subset=dup_key, keep=False)
dup_df = earnings[dup_mask].copy()
non_cik_non_key_cols = [c for c in earnings.columns
                        if c not in dup_key and c != "cik"]
groups = dup_df.groupby(dup_key)
n_diff_groups = 0
for _, sub in groups:
    if sub[non_cik_non_key_cols].drop_duplicates().shape[0] > 1:
        n_diff_groups += 1
if n_diff_groups > 0:
    print(f"  WARNING: {n_diff_groups} dup groups have NON-IDENTICAL non-cik cols -- skipping dedup!")
    # Skip Step 2 if assumption violated -- we are not willing to merge rows with discordant values.
    sys.exit(2)
print(f"  Verified: all {len(groups)} dup groups have identical non-cik cols (only cik differs)")

# Build dedup: keep row with smallest cik per (permaTicker, report_date) group
# Sorting with stable sort on [permaTicker, report_date, cik] + drop_duplicates(keep='first')
dedup_sort_cols = dup_key + ["cik"]
earnings_sorted = earnings.sort_values(
    dedup_sort_cols, kind="mergesort"
).reset_index(drop=True)
deduped = earnings_sorted.drop_duplicates(
    subset=dup_key, keep="first"
).copy()
print(f"  After: {len(deduped):,} rows ({len(earnings) - len(deduped):,} removed)")
print(f"  Resultant (permaTicker, report_date) dup rows: "
      f"{int(deduped.duplicated(subset=dup_key).sum())} (should be 0)")
assert deduped.duplicated(subset=dup_key).sum() == 0

# Write back via mode='a' + store.remove()
print(f"  Writing back to DB ...")
with pd.HDFStore(DB, mode="a") as s:
    if EARNINGS_KEY in s.keys():
        s.remove(EARNINGS_KEY)
    s.put(EARNINGS_KEY, deduped, format="table")
print(f"  OK: /earnings/raw now {len(deduped):,} rows")

# === Verification read ===
print(f"\n[4] Verification -- read-back:")
with pd.HDFStore(DB, mode="r") as s:
    permatickers_check = s[PERMATICKERS_KEY]
    earnings_check = s[EARNINGS_KEY]
print(f"  /metadata/sp400_permatickers: {permatickers_check.shape}  "
      f"cols: {permatickers_check.columns.tolist()}")
print(f"  /earnings/raw:              {earnings_check.shape}")
print(f"  /earnings/raw dup (permaTicker, report_date): "
      f"{int(earnings_check.duplicated(subset=dup_key).sum())}")
assert "legacy_perm_id" not in permatickers_check.columns
assert earnings_check.duplicated(subset=dup_key).sum() == 0

print(f"\nDB size after: {DB.stat().st_size / 1e6:.2f} MB")
