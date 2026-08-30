"""90_rc1_phase1_features.py — RC-1 Phase 1: build the 4 frozen insider features.

Pre-registration: 04_backtest/rc1_pre_registration.md (2026-08-30).
Phase 0 results appended there (coverage 13.2%, value cap $50M, 63
truncated tickers -> NaN).

Features (FROZEN — no additions after seeing results):

  insider_net_buy_90d        log1p(net P-purchase $ in [T-90d, T))
  insider_cluster_90d        flag: >=2 distinct reportingNames AND >=$50k
                             combined in [T-90d, T)
  insider_sell_pressure_30d  log1p(net S-sale $ in [T-30d, T))
  insider_days_since_last_buy days since most recent P-purchase filing in
                             [T-180d, T); NaN if none

Rules (all pre-registered):
  - Point-in-time: filingDate strictly < report_date. transactionDate
    never used.
  - Per-filing value cap $50M (FMP glitch guard); rows with invalid
    value (NaN/negative) contribute NOTHING to dollars but still count
    for buyer counts / recency (their transactionType is real).
  - Tickers at FMP's 5000-row page cap (truncated history): ALL features
    NaN — a missing early history must never read as "no insider
    activity".
  - Output: side table keyed (permaTicker, report_date) -> written to
    /features/rc1_insider_features in db.h5. The frozen V6 matrix is
    NOT touched.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DB = HERE.parent / "01_data" / "db.h5"
INSIDER_DB = HERE.parent / "01_data" / "db_insider.h5"
OUT_KEY = "/features/rc1_insider_features"
OUT_JSON = Path(__import__("os").environ.get("TEMP", "/tmp")) / "_rc1_phase1.json"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def main() -> None:
    print("=" * 100)
    print("RC-1 PHASE 1 — build 4 frozen insider features (side table, matrix untouched)")
    print("=" * 100)

    mx = pd.read_hdf(DB, "/features/train_matrix_v4_timing_correct")
    mx["report_date"] = pd.to_datetime(mx["report_date"])
    ev = mx[["permaTicker", "report_date"]].drop_duplicates()
    print(f"events: {len(ev):,}")

    frames, capped = [], set()
    with pd.HDFStore(INSIDER_DB, "r") as s:
        for k in s.keys():
            df = s[k]
            if len(df) >= 5000:
                capped.add(k.split("/")[-1])
            frames.append(df)
    ins = pd.concat(frames, ignore_index=True)
    ins["filingDate"] = pd.to_datetime(ins["filingDate"], errors="coerce")
    ins["price"] = pd.to_numeric(ins["price"], errors="coerce")
    ins["qty"] = pd.to_numeric(ins["securitiesTransacted"], errors="coerce")
    ins["value"] = ins["price"] * ins["qty"]
    ins.loc[~np.isfinite(ins["value"]) | (ins["value"] > 50e6) | (ins["value"] < 0), "value"] = np.nan
    ins = ins.dropna(subset=["filingDate"])

    with pd.HDFStore(DB, "r") as s:
        pt = s["/metadata/sp400_permatickers"][["permaTicker", "canonical_ticker"]]
    ins["permaTicker"] = ins["symbol"].map(dict(zip(pt.canonical_ticker, pt.permaTicker)))
    ins = ins.dropna(subset=["permaTicker"])

    buys = ins[ins.transactionType.astype(str).str.startswith("P")]
    sales = ins[ins.transactionType.astype(str).str.startswith("S")]
    buys_by = {t: g[["filingDate", "value", "reportingName"]] for t, g in buys.groupby("permaTicker")}
    sales_by = {t: g[["filingDate", "value"]] for t, g in sales.groupby("permaTicker")}
    print(f"buys: {len(buys):,} | sales: {len(sales):,} | capped tickers: {len(capped)}")

    rows = []
    for pt_id, T in zip(ev.permaTicker.values, ev.report_date.values):
        rec = {"permaTicker": pt_id, "report_date": T}
        if pt_id in capped:
            rec.update({"insider_net_buy_90d": np.nan, "insider_cluster_90d": np.nan,
                        "insider_sell_pressure_30d": np.nan, "insider_days_since_last_buy": np.nan})
            rows.append(rec)
            continue
        b = buys_by.get(pt_id)
        if b is not None:
            fd = b.filingDate.values
            m90 = (fd >= T - np.timedelta64(90, "D")) & (fd < T)
            w = b.value.values
            net90 = np.nansum(w[m90])
            rec["insider_net_buy_90d"] = float(np.log1p(max(net90, 0.0)))
            buyers = pd.unique(b.reportingName.values[m90])
            rec["insider_cluster_90d"] = float(
                (len(buyers) >= 2) and (np.nansum(w[m90]) >= 50_000))
            lb = fd[(fd >= T - np.timedelta64(180, "D")) & (fd < T)]
            rec["insider_days_since_last_buy"] = (
                float((T - lb.max()) / np.timedelta64(1, "D")) if len(lb) else np.nan)
        else:
            rec.update({"insider_net_buy_90d": 0.0, "insider_cluster_90d": 0.0,
                        "insider_days_since_last_buy": np.nan})
        s_ = sales_by.get(pt_id)
        if s_ is not None:
            fd = s_.filingDate.values
            m30 = (fd >= T - np.timedelta64(30, "D")) & (fd < T)
            net30 = np.nansum(s_.value.values[m30])
            rec["insider_sell_pressure_30d"] = float(np.log1p(max(net30, 0.0)))
        else:
            rec["insider_sell_pressure_30d"] = 0.0
        rows.append(rec)

    d = pd.DataFrame(rows)
    d["report_date"] = pd.to_datetime(d["report_date"])

    print("\nFeature summary:")
    for c in ["insider_net_buy_90d", "insider_cluster_90d",
              "insider_sell_pressure_30d", "insider_days_since_last_buy"]:
        x = d[c]
        nz = (x.fillna(0) > 0)
        print(f"  {c:32s} NaN {x.isna().mean():5.1%} | >0 {nz.mean():5.1%} | "
              f"mean {x.mean():8.3f} | p99 {x.quantile(0.99):8.3f}")
    x = d.insider_net_buy_90d
    print(f"  net_buy_90d quartiles (nonzero): "
          f"{x[x > 0].quantile([.25, .5, .75]).round(2).tolist()}")

    with pd.HDFStore(DB, "a") as s:
        if OUT_KEY in s.keys():
            s.remove(OUT_KEY)
        s.put(OUT_KEY, d, format="table", data_columns=["permaTicker", "report_date"])
    print(f"\nwrote {OUT_KEY} ({len(d):,} rows) — frozen matrix untouched")
    d.to_json(OUT_JSON, orient="records", date_format="iso")


if __name__ == "__main__":
    main()
