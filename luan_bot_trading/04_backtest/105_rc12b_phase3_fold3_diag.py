"""RC-12b Phase 3 diagnostic — fold 3 (2025H2) decomposition.

Question: why did fold 3 flip from +1.56% (current-members-backward) to
-0.24% (pt-in-time)?  Decomposes the fold-3 trade ledger into:

  1. provenance: current-member trades vs dead/graduated-name trades
  2. competition: trades the pt-in-time slate DISPLACED (in old, not new;
     in new, not old)
  3. timing: month-by-month, biggest winners/losers
  4. delisting-edge: trades whose exit lands near the END of a dead
     ticker's price series (the classic delisting-return zone)

Diagnostic only — no policy change. Reuses 104's machinery by import.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


v104 = _load("v104", HERE / "104_rc12b_phase3_transfer.py")
fr = v104.fr

DB_SP600 = v104.DB_SP600
W0, W1 = "2025-07-01", "2025-12-31"


def ledger(matrix_key: str) -> pd.DataFrame:
    d = pd.read_hdf(DB_SP600, matrix_key)
    d = v104.score_v6(d)
    slate = v104.build_cands(d, W0, W1)
    trades = v104.simulate(fr.weekly_slate(slate))
    out = pd.DataFrame(trades)
    return out


def provenance_of(sym_or_pt) -> str:
    return "?"


def main() -> None:
    pt3 = pd.read_hdf(DB_SP600, "/features/train_matrix_sp600_pt")
    # provenance map: permaTicker -> dead/new/current/graduate
    with pd.HDFStore(DB_SP600, "r") as s:
        cur = set(s["/metadata/sp600_ptmap"].permaTicker)
        new = set(s["/metadata/sp600_ptmap3"].permaTicker) - cur
    with pd.HDFStore(v104.DB_PROD, "r") as s:
        grads = set(s["/metadata/sp400_permatickers"].permaTicker)

    def prov(pt):
        if pt in new:
            return "dead(new)"
        if pt in grads and pt not in cur:
            return "graduate"
        return "current"

    la = ledger("/features/train_matrix_sp600")     # old, survivorship-biased
    lb = ledger("/features/train_matrix_sp600_pt")  # pt-in-time
    for df in (la, lb):
        df["month"] = pd.to_datetime(df.entry_date).dt.to_period("M").astype(str)
    la["prov"] = la.permaTicker.map(prov)
    lb["prov"] = lb.permaTicker.map(prov)

    def summ(df, label):
        r = df["return"].astype(float)
        print(f"\n[{label}] n={len(df)} win={(r>0).mean():.0%} "
              f"raw={r.mean():+.3%} spread-adj={r.mean()-0.003:+.3%}")
        for p, g in df.groupby("prov"):
            g = g["return"].astype(float)
            print(f"   {p:10s} n={len(g):3d}  mean {g.mean():+.3%}  "
                  f"win {(g>0).mean():.0%}  worst {g.min():+.2%}")
        for m, g in df.groupby("month"):
            g = g["return"].astype(float)
            print(f"   {m}: n={len(g):3d}  mean {g.mean():+.3%}")

    summ(la, "OLD fold3 (current-members-backward)")
    summ(lb, "NEW fold3 (pt-in-time)")

    # displacement analysis
    ka = set(zip(la.permaTicker, la.entry_date.dt.strftime("%Y-%m-%d")))
    kb = set(zip(lb.permaTicker, lb.entry_date.dt.strftime("%Y-%m-%d")))
    only_old = la[[k in (ka - kb) for k in zip(la.permaTicker, la.entry_date.dt.strftime('%Y-%m-%d'))]]
    only_new = lb[[k in (kb - ka) for k in zip(lb.permaTicker, lb.entry_date.dt.strftime('%Y-%m-%d'))]]
    print(f"\nshared trades: {len(ka & kb)} | old-only: {len(only_old)} | new-only: {len(only_new)}")
    if len(only_old):
        r = only_old["return"].astype(float)
        print(f"OLD-ONLY (displaced): mean {r.mean():+.3%} — these were REMOVED from the pt slate")
    if len(only_new):
        r = only_new["return"].astype(float)
        print(f"NEW-ONLY (added):     mean {r.mean():+.3%} — these ENTERED via pt-in-time")
        print("\nnew-only detail (worst 12):")
        for _, t in only_new.sort_values("return").head(12).iterrows():
            print(f"   {str(t.permaTicker)[-6:]} {t.entry_date.date()} {t.prov:10s} "
                  f"{t['return']:+.2%}  score={t.score:.3f}")

    # delisting-edge: exit within 10 bars of series end
    v104._bulk_load()
    ends = {}
    lb["series_end"] = lb.permaTicker.map(lambda pt: (v104._PCACHE[pt][0][-1]
                                                      if pt in v104._PCACHE else pd.NaT))
    lb["days_from_end"] = (pd.to_datetime(lb.series_end) - pd.to_datetime(lb.exit_date)).dt.days
    near = lb[lb.days_from_end <= 14]
    if len(near):
        r = near["return"].astype(float)
        print(f"\nDELISTING-EDGE trades (exit within 14d of series end): n={len(near)} "
              f"mean {r.mean():+.3%}")
        for _, t in near.sort_values('return').head(8).iterrows():
            print(f"   {str(t.permaTicker)[-6:]} exit {t.exit_date.date()} "
                  f"end {t.series_end.date()} {t['return']:+.2%} {t.prov}")
    else:
        print("\nDELISTING-EDGE trades: none (no exits within 14d of series end)")


if __name__ == "__main__":
    main()
