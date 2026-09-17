"""Diagnostic (non-gating): raw expectancy of the model's picks — no portfolio.

Question: for all events scoring >= 0.33 (the only filter), is the mean
return distinguishable from 0? And how does it compare with the
UNCONDITIONAL mean of all events in the same OOS test windows (which also
lets us back out the mean of the REJECTED group = score < 0.33)?

Returns measured two ways: pregap_return (5-session hold, 10% stop) and
car_10d (10-session CAR, no stop). Filters identical across groups:
EXCLUDE_SECTORS and non-null return. OOS test windows only (label_end in
(sve, tse] per DEFAULT_FOLDS) — picks are out-of-sample by construction.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
import os

os.chdir(ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bt = load("bt_raw", HERE / "51_hp_theta_sweep_23feat.py")
DB = bt.DB
CAND = HERE / "archive" / "experiments" / "rc22" / "candidates.h5"

# ---- test windows ----
def in_test(rd):
    m = np.zeros(len(rd), dtype=bool)
    for te, sve, tse in bt.DEFAULT_FOLDS:
        m |= (rd > pd.Timestamp(sve)) & (rd <= pd.Timestamp(tse))
    return m


def summarize(df, label):
    r = pd.to_numeric(df["pregap_return"], errors="coerce")
    ok = r.notna()
    if "sector" in df.columns:
        ok &= ~df["sector"].isin(bt.EXCLUDE_SECTORS)
    r = r[ok].to_numpy(dtype=float)
    if len(r) == 0:
        return None
    se = r.std(ddof=1) / np.sqrt(len(r)) if len(r) > 1 else np.nan
    out = {"group": label, "n": int(len(r)), "mean_pct": round(float(r.mean() * 100), 3),
           "se_pct": round(float(se * 100), 3),
           "t": round(float(r.mean() / se), 2) if se and np.isfinite(se) else None,
           "win_pct": round(float((r > 0).mean() * 100), 1),
           "median_pct": round(float(np.median(r) * 100), 3)}
    if "car_10d" in df.columns:
        c = pd.to_numeric(df.loc[ok, "car_10d"], errors="coerce").to_numpy(dtype=float)
        c = c[np.isfinite(c)]
        if len(c) > 1:
            cse = c.std(ddof=1) / np.sqrt(len(c))
            out["car10_mean_pct"] = round(float(c.mean() * 100), 3)
            out["car10_se_pct"] = round(float(cse * 100), 3)
            out["car10_t"] = round(float(c.mean() / cse), 2)
    return out


# ---- universes (all events in test windows) ----
v6c = pd.read_hdf(DB, "/features/train_matrix_v6c")
v6c["report_date"] = pd.to_datetime(v6c["report_date"])
v6c["label_end"] = pd.to_datetime(v6c["label_end"])
old = pd.read_hdf(DB, "/features/train_matrix_v4_timing_correct")
old["report_date"] = pd.to_datetime(old["report_date"])
ko = set(zip(old.permaTicker, old.report_date))
v6c = v6c[v6c.set_index(["permaTicker", "report_date"]).index.isin(ko)].copy()
v6c_t = v6c[in_test(v6c["label_end"])]

v7 = pd.read_hdf(DB, "/features/train_matrix_v7c")
v7["report_date"] = pd.to_datetime(v7["report_date"])
v7["label_end"] = pd.to_datetime(v7["label_end"])
v7_t = v7[in_test(v7["label_end"])]
v7_600 = v7_t[v7_t["is_sp400"] == 0]
v7_400 = v7_t[v7_t["is_sp400"] == 1]

print("=" * 100)
print("UNCONDITIONAL (all events in the same OOS test windows)")
rows = []
for df, lab in ((v6c_t, "SP400 all (common keys)"), (v7_400, "v7c SP400 all"),
                (v7_600, "v7c SP600 all"), (v7_t, "v7c COMBINED all")):
    s = summarize(df, lab)
    if s:
        rows.append(s)
print(pd.DataFrame(rows).to_string(index=False))

print()
print("=" * 100)
print("PICKS >= 0.33 (no portfolio layer, no slots, no weeks)")
st = pd.HDFStore(CAND, "r")
rows = []
for nm, lab, uni in (("rc19", "rc19 SP400 24feat", "SP400"),
                     ("rc20", "rc20 COMBINED 24feat", "COMBINED"),
                     ("rc21", "rc21 SP600 transfer", "SP600"),
                     ("v6n", "v6n SP400 3-gate 22feat", "SP400"),
                     ("v7n", "v7n COMBINED 3-gate 22feat", "COMBINED")):
    c = st[f"/{nm}"].copy()
    s = summarize(c, lab)
    if s:
        rows.append(s)
print(pd.DataFrame(rows).to_string(index=False))
st.close()

print()
print("=" * 100)
print("REJECTED GROUP (score < 0.33), backed out of universe minus picks")
uni_map = {"SP400": v6c_t, "COMBINED": v7_t, "SP600": v7_600}
st = pd.HDFStore(CAND, "r")
rows = []
for nm, lab, uni in (("rc19", "rc19 rejected (<0.33)", "SP400"),
                     ("rc20", "rc20 rejected (<0.33)", "COMBINED"),
                     ("rc21", "rc21 rejected (<0.33)", "SP600")):
    all_df = uni_map[uni]
    a = summarize(all_df, "all")
    c = summarize(st[f"/{nm}"], "picks")
    if a and c:
        n_rej = a["n"] - c["n"]
        mean_rej = (a["mean_pct"] * a["n"] - c["mean_pct"] * c["n"]) / n_rej
        rows.append({"group": lab, "n": n_rej, "mean_pct": round(mean_rej, 3)})
st.close()
print(pd.DataFrame(rows).to_string(index=False))
