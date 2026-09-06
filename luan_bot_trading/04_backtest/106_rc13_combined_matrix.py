"""RC-13 — combined pt-in-time SP400+SP600 matrix for V7 training.

Pre-registered (rc13_combined_pre_registration.md, 2026-09-06):
- is_sp400: membership AT EVENT DATE (pt-in-time, both universes)
- gate labels NET-OF-COST thresholds: g1 car_10d > 3% + cost;
  g3 maxdd_ma > -1.5% + cost; g2 unadjusted. cost = 10bp (SP400),
  30bp (SP600) per RC-12b Phase-0 measured proxies.
- no liquidity feature in the feature set (keeps is_sp400 importance
  readable); ADV >= $10M stays a hard execution filter downstream.

Sources:
- SP400: /features/train_matrix_v4_timing_correct (labels + continuous
  gate signals already present; re-threshold with cost shift)
- SP600: /features/train_matrix_sp600_pt (compute car_10d,
  inst_vol_ratio, maxdd_ma from prices; benchmark IJR)

Output: /features/train_matrix_combined in db.h5
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "01_data" / "db.h5"
DB_SP600 = ROOT / "01_data" / "db_sp600.h5"
COST_SP400 = 0.0010
COST_SP600 = 0.0030
G1, G2, G3 = 0.03, 2.0, -0.015

import importlib.util
import sys
spec = importlib.util.spec_from_file_location(
    "bt", Path(__file__).resolve().parent / "51_hp_theta_sweep_23feat.py")
bt = importlib.util.module_from_spec(spec)
sys.modules["bt"] = bt
spec.loader.exec_module(bt)
FEATURES = bt.DEPLOY_FEATURES


def main() -> None:
    # ---------- SP400 side ----------
    v4 = pd.read_hdf(DB, "/features/train_matrix_v4_timing_correct")
    v4 = v4[v4.pregap_return.notna()].copy()
    sp400 = pd.DataFrame({
        "permaTicker": v4.permaTicker,
        "canonical_ticker": v4.canonical_ticker,
        "report_date": pd.to_datetime(v4.report_date),
        "sector": v4.sector,
        "entry_date": pd.to_datetime(v4.pregap_entry_date),
        "exit_date": pd.to_datetime(v4.pregap_exit_date),
        "pregap_return": v4.pregap_return.astype(float),
        "car_10d": v4.car_10d.astype(float),
        "inst_vol_ratio": v4.inst_vol_ratio.astype(float),
        "maxdd_ma": v4.maxdd_ma.astype(float),
        "is_sp400": 1,
        "cost": COST_SP400,
    })
    for f in FEATURES:
        sp400[f] = v4[f]
    sp400["pass_g1"] = (sp400.car_10d.fillna(-9) > G1 + COST_SP400).astype(int)
    sp400["pass_g2"] = (sp400.inst_vol_ratio > G2).astype(int)
    sp400["pass_g3"] = (sp400.maxdd_ma.fillna(-9) > G3 + COST_SP400).astype(int)
    sp400["pead_pass"] = (sp400.pass_g1 & sp400.pass_g2 & sp400.pass_g3).astype(int)
    print(f"SP400 side: {len(sp400):,} events | g1 {sp400.pass_g1.mean():.3f} "
          f"g2 {sp400.pass_g2.mean():.3f} g3 {sp400.pass_g3.mean():.3f} "
          f"pead {sp400.pead_pass.mean():.3f}")

    # ---------- SP600 side ----------
    s6 = pd.read_hdf(DB_SP600, "/features/train_matrix_sp600_pt").copy()
    s6["report_date"] = pd.to_datetime(s6.report_date)
    with pd.HDFStore(DB_SP600, "r") as st:
        ijr = st["/sp600/benchmark_IJR"].copy()
        ijr["Date"] = pd.to_datetime(ijr["Date"]).dt.tz_localize(None).dt.normalize()
        px = {}
        for k in st.keys():
            if k.startswith("/sp600/") and k.count("/") == 2:
                d = st[k].copy()
                d["Date"] = pd.to_datetime(d["Date"]).dt.tz_localize(None).dt.normalize()
                px[k.split("/")[-1]] = d.sort_values("Date").reset_index(drop=True)
    with pd.HDFStore(DB, "r") as st:
        for k in st.keys():
            if k.startswith("/sp400/") and k.count("/") == 2:
                pt = k.split("/")[-1]
                if pt not in px:
                    d = st[k].copy()
                    d["Date"] = pd.to_datetime(d["Date"]).dt.tz_localize(None).dt.normalize()
                    px[pt] = d.sort_values("Date").reset_index(drop=True)
    ijr_s = pd.Series(ijr["Adj_Close"].to_numpy(float), index=ijr["Date"])

    car10 = np.full(len(s6), np.nan)
    volr = np.full(len(s6), np.nan)
    mdd = np.full(len(s6), np.nan)
    n_missing = 0
    for i, row in enumerate(s6.itertuples(index=False)):
        p = px.get(row.permaTicker)
        if p is None or len(p) < 35:
            n_missing += 1
            continue
        dates = p["Date"].to_numpy()
        close = p["Adj_Close"].to_numpy(float)
        vol = p["Adj_Volume"].to_numpy(float)
        rd = np.datetime64(pd.Timestamp(row.report_date))
        t = int(np.searchsorted(dates, rd, side="left"))
        if t < 20 or t + 12 >= len(close):
            continue
        vma20 = float(np.nanmean(vol[t - 20:t]))
        if vma20 and vma20 > 0:
            volr[i] = float(np.nanmean(vol[t:t + 3])) / vma20
        # car_10d: cumulative abnormal log return vs IJR, [t+1, t+11]
        al = pd.Series(close, index=p["Date"])
        b = ijr_s.reindex(al.index, method="ffill").to_numpy(float)
        slr = np.diff(np.log(np.where(close > 0, close, np.nan)))
        blr = np.diff(np.log(np.where(b > 0, b, np.nan)))
        car10[i] = float(np.nansum(slr[t + 1:t + 11] - blr[t + 1:t + 11]))
        # maxdd_ma: simple-return paths vs IJR over T+1..T+11 (label convention)
        sp_path = close[t + 1:t + 12] / close[t] - 1.0
        it = int(np.searchsorted(ijr["Date"].to_numpy(), dates[t]))
        if it + 12 < len(ijr):
            ijh_path = ijr["Adj_Close"].to_numpy(float)[it + 1:it + 12] / ijr["Adj_Close"].to_numpy(float)[it] - 1.0
            n = min(len(sp_path), len(ijh_path))
            mdd[i] = float(np.min(sp_path[:n] - ijh_path[:n]))
    s6["car_10d"], s6["inst_vol_ratio"], s6["maxdd_ma"] = car10, volr, mdd
    print(f"SP600 signals: car10 nan={np.isnan(car10).sum()}, volr nan={np.isnan(volr).sum()}, "
          f"mdd nan={np.isnan(mdd).sum()}, no-price={n_missing}")

    sp600 = pd.DataFrame({
        "permaTicker": s6.permaTicker,
        "canonical_ticker": s6.ticker,
        "report_date": s6.report_date,
        "sector": s6.sector,
        "entry_date": pd.to_datetime(s6.entry_date),
        "exit_date": pd.to_datetime(s6.exit_date),
        "pregap_return": s6.pregap_return.astype(float),
        "car_10d": s6.car_10d,
        "inst_vol_ratio": s6.inst_vol_ratio,
        "maxdd_ma": s6.maxdd_ma,
        "is_sp400": 0,
        "cost": COST_SP600,
    })
    for f in FEATURES:
        sp600[f] = s6[f]
    sp600["pass_g1"] = (sp600.car_10d.fillna(-9) > G1 + COST_SP600).astype(int)
    sp600["pass_g2"] = (sp600.inst_vol_ratio.fillna(0) > G2).astype(int)
    sp600["pass_g3"] = (sp600.maxdd_ma.fillna(-9) > G3 + COST_SP600).astype(int)
    sp600["pead_pass"] = (sp600.pass_g1 & sp600.pass_g2 & sp600.pass_g3).astype(int)
    print(f"SP600 side: {len(sp600):,} events | g1 {sp600.pass_g1.mean():.3f} "
          f"g2 {sp600.pass_g2.mean():.3f} g3 {sp600.pass_g3.mean():.3f} "
          f"pead {sp600.pead_pass.mean():.3f}")

    # ---------- combine + dedupe ----------
    both = pd.concat([sp400, sp600], ignore_index=True)
    dup = both.duplicated(subset=["permaTicker", "report_date"], keep=False)
    print(f"\ncollision (pt,report_date) pairs: {int(dup.sum())}")
    both = both.sort_values(["permaTicker", "report_date", "is_sp400"],
                            ascending=[True, True, False])  # SP400 wins boundary ties
    both = both[~both.duplicated(subset=["permaTicker", "report_date"], keep="first")]
    both = both.sort_values(["report_date", "permaTicker"]).reset_index(drop=True)

    # feature NaN audit (the phase-1 lesson: check signatures BEFORE trusting)
    nan_sig = both[FEATURES].isna().mean().sort_values(ascending=False)
    print("\nfeature NaN signature (top 8):")
    for k, v in nan_sig.head(8).items():
        print(f"   {k:38s} {v:.1%}")
    print(f"\nCOMBINED: {len(both):,} events | SP400 {int((both.is_sp400==1).sum()):,} "
          f"| SP600 {int((both.is_sp400==0).sum()):,} | "
          f"{both.report_date.min().date()} .. {both.report_date.max().date()}")

    with pd.HDFStore(DB, "a") as st:
        if "/features/train_matrix_combined" in st.keys():
            st.remove("/features/train_matrix_combined")
        st.put("/features/train_matrix_combined", both, format="table",
               data_columns=["permaTicker", "report_date"])
    print("wrote /features/train_matrix_combined")


if __name__ == "__main__":
    main()
