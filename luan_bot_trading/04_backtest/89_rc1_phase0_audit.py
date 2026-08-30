"""89_rc1_phase0_audit.py — RC-1 Phase 0: point-in-time audit & coverage.

Pre-registration: 04_backtest/rc1_pre_registration.md (2026-08-30).

Measures, for each of the 16,789 matrix earnings events:
  - pre-print windows  [T-90d, T) and [T-30d, T): open-market P-purchase
    dollars, distinct buyers, sell dollars (filingDate-based: public info)
  - post-print control [T, T+30d): same counts (blackout asymmetry)
  - days since last buy (180d lookback)

KILL GATE 0 (pre-registered): if <10% of events have any pre-event
P-purchase >= $10k in [T-90d, T) -> close RC-1 at Phase 0.

Also flags permaTickers whose FMP fetch hit the 5000-row page cap
(truncated history -> Phase 1 features must be NaN, never zero).
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
OUT = Path(__import__("os").environ.get("TEMP", "/tmp")) / "_rc1_phase0.json"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def main() -> None:
    print("=" * 100)
    print("RC-1 PHASE 0 — insider pre-event coverage audit (point-in-time, filingDate)")
    print("=" * 100)

    mx = pd.read_hdf(DB, "/features/train_matrix_v4_timing_correct")
    mx["report_date"] = pd.to_datetime(mx["report_date"])
    print(f"\nmatrix events: {len(mx):,} | tickers: {mx.permaTicker.nunique()} "
          f"| {mx.report_date.min().date()} .. {mx.report_date.max().date()}")

    # ---- load insider cache into one frame ---------------------------------
    frames, capped = [], []
    with pd.HDFStore(INSIDER_DB, "r") as s:
        for k in s.keys():
            df = s[k]
            if len(df) >= 5000:
                capped.append(k.split("/")[-1])
            frames.append(df)
    ins = pd.concat(frames, ignore_index=True)
    ins["filingDate"] = pd.to_datetime(ins["filingDate"], errors="coerce")
    ins["transactionDate"] = pd.to_datetime(ins["transactionDate"], errors="coerce")
    ins["price"] = pd.to_numeric(ins["price"], errors="coerce")
    ins["securitiesTransacted"] = pd.to_numeric(ins["securitiesTransacted"], errors="coerce")
    ins["value"] = ins["price"] * ins["securitiesTransacted"]
    # Pre-registered data-cleaning (added after Phase 0 first pass found FMP
    # price glitches: rows with value in the $billions/trillions, ~1.5% of
    # rows carrying ~100% of raw dollar mass). Per-filing value cap $50M;
    # count-based features are immune but dollar features need this.
    ins.loc[~np.isfinite(ins["value"]) | (ins["value"] > 50e6) | (ins["value"] < 0), "value"] = np.nan
    ins["is_buy"] = ins["transactionType"].astype(str).str.startswith("P")
    ins["is_sale"] = ins["transactionType"].astype(str).str.startswith("S")
    ins = ins.dropna(subset=["filingDate"])
    print(f"insider cache: {len(ins):,} filings | {ins.permaTicker.nunique() if 'permaTicker' in ins.columns else ins.symbol.nunique()} tickers "
          f"| {ins.filingDate.min().date()} .. {ins.filingDate.max().date()}")
    print(f"  P-purchases: {int(ins.is_buy.sum()):,} | S-sales: {int(ins.is_sale.sum()):,} "
          f"| 5000-cap truncated: {len(capped)} tickers")

    # key on permaTicker if present else symbol->matrix canonical map
    key_col = "permaTicker" if "permaTicker" in ins.columns else None
    if key_col is None:
        # map canonical ticker -> permaTicker via metadata
        with pd.HDFStore(DB, "r") as s:
            pt = s["/metadata/sp400_permatickers"][["permaTicker", "canonical_ticker"]]
        m = dict(zip(pt.canonical_ticker, pt.permaTicker))
        ins["permaTicker"] = ins["symbol"].map(m)
        key_col = "permaTicker"
        print(f"  symbol->permaTicker mapped: {ins.permaTicker.notna().mean():.0%} of filings")

    # ---- per-event windows --------------------------------------------------
    ev = mx[["permaTicker", "report_date"]].drop_duplicates()
    buys = ins[ins.is_buy & (ins.value > 0)]
    sales = ins[ins.is_sale & (ins.value > 0)]

    # group filings per ticker once, then window-search per event
    buys_by_pt = {pt: g[["filingDate", "value", "reportingName"]]
                  for pt, g in buys.groupby(key_col)}
    sales_by_pt = {pt: g[["filingDate", "value"]] for pt, g in sales.groupby(key_col)}

    rows = []
    for pt, T in zip(ev.permaTicker.values, ev.report_date.values):
        b = buys_by_pt.get(pt)
        s_ = sales_by_pt.get(pt)
        rec = {"pt": pt, "T": T}
        if b is not None:
            fd = b.filingDate.values
            pre90 = (fd >= T - np.timedelta64(90, "D")) & (fd < T)
            pre30 = (fd >= T - np.timedelta64(30, "D")) & (fd < T)
            post30 = (fd >= T) & (fd < T + np.timedelta64(30, "D"))
            w = b.value.values
            rec["buy90_val"] = float(w[pre90].sum())
            rec["buy30_val"] = float(w[pre30].sum())
            rec["buy90_n"] = int(pre90.sum())
            rec["buy90_buyers"] = int(pd.unique(b.reportingName.values[pre90]).size) if pre90.any() else 0
            rec["post30_val"] = float(w[post30].sum())
            rec["post30_events_flag"] = bool(post30.any())
            lb = fd[(fd >= T - np.timedelta64(180, "D")) & (fd < T)]
            rec["days_since_buy"] = float((T - lb.max()) / np.timedelta64(1, "D")) if len(lb) else np.nan
        else:
            rec.update({"buy90_val": 0.0, "buy30_val": 0.0, "buy90_n": 0,
                        "buy90_buyers": 0, "post30_val": 0.0,
                        "post30_events_flag": False, "days_since_buy": np.nan})
        if s_ is not None:
            fd = s_.filingDate.values
            pre30 = (fd >= T - np.timedelta64(30, "D")) & (fd < T)
            rec["sell30_val"] = float(s_.value.values[pre30].sum())
        else:
            rec["sell30_val"] = 0.0
        rows.append(rec)

    d = pd.DataFrame(rows)
    d["capped_pt"] = d.pt.isin(set(capped))

    # ---- report -------------------------------------------------------------
    n = len(d)
    print("\n" + "-" * 100)
    print(f"KILL GATE 0 — pre-event material P-purchases ($>=10k, [T-90d,T)):")
    mat = d.buy90_val >= 10_000
    print(f"  events with >= $10k pre-print buys : {mat.sum():,} ({mat.mean():.1%})")
    print(f"  events with ANY pre-print buy      : {(d.buy90_n > 0).sum():,} ({(d.buy90_n > 0).mean():.1%})")
    print(f"  cluster (>=2 buyers, >=$50k)       : {((d.buy90_buyers >= 2) & (d.buy90_val >= 50_000)).sum():,} "
          f"({((d.buy90_buyers >= 2) & (d.buy90_val >= 50_000)).mean():.1%})")
    print(f"  pre-30d any buy                    : {(d.buy30_val > 0).sum():,} ({(d.buy30_val > 0).mean():.1%})")
    print(f"  pre-30d sell activity              : {(d.sell30_val > 0).sum():,} ({(d.sell30_val > 0).mean():.1%})")

    print(f"\nBlackout asymmetry (post-print control [T,T+30d)):")
    print(f"  post30 buy dollars total : ${d.post30_val.sum() / 1e6:,.0f}M")
    print(f"  pre90  buy dollars total : ${d.buy90_val.sum() / 1e6:,.0f}M")
    print(f"  ratio post30/pre90 (per-day normalized 1:3): "
          f"{(d.post30_val.sum() / 30) / (d.buy90_val.sum() / 90):.2f}x")

    print(f"\nDays-since-last-buy distribution (events with a buy in 180d):")
    x = d.days_since_buy.dropna()
    print(f"  n={len(x):,} ({len(x) / n:.0%}) | median {x.median():.0f}d | "
          f"<=30d: {(x <= 30).mean():.0%} | 31-90d: {((x > 30) & (x <= 90)).mean():.0%}")

    print(f"\nTruncated-history tickers (5000-cap): {len(capped)}")
    print(f"  events on those tickers: {d.capped_pt.sum():,} ({d.capped_pt.mean():.1%}) — Phase 1 -> NaN")

    # ---- year-by-year coverage stability (regime check) ---------------------
    d["year"] = pd.to_datetime(d["T"]).dt.year
    g = d.groupby("year").agg(events=("pt", "size"), mat10k=("buy90_val", lambda v: (v >= 10_000).mean()))
    print("\nCoverage by year (fraction with >=$10k pre-print buys):")
    for y, r in g.iterrows():
        print(f"  {y}: {r.events:>6,} events  {r.mat10k:>5.1%}")

    verdict = "PASS (>=10%)" if mat.mean() >= 0.10 else "FAIL — CLOSE RC-1 AT PHASE 0"
    print("\n" + "=" * 100)
    print(f"PHASE 0 VERDICT: {verdict}  —  coverage {mat.mean():.1%}")
    print("=" * 100)

    d.to_json(OUT, orient="records", date_format="iso")
    print(f"saved per-event counts -> {OUT}")


if __name__ == "__main__":
    main()
