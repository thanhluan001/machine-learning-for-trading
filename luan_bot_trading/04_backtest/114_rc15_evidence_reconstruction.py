"""114_rc15_evidence_reconstruction.py — RC-15 pre-registration exhibits.

Reproduces every probe that informed the RC-15 lifecycle design, on two
datasets:

A. 2000-02 leader-death exam — 8-name era basket (CSCO ORCL QCOM MU INTC
   MSFT JDSU NVDA), Tiingo daily EOD 1996-2002, equal-weight daily
   rebalanced:
     1. ungated basket
     2. two-cadence alarm (W + monthly bear-bloc) + flat 50% exposure
     3. two-cadence alarm + user depth rule (cash = |dd|, clip 80%)
     4. band-integrated schedule (misreading, kept for the record),
        rolling-104-week anchor
     5. CPPI user spec (cash = |dd| x ratio, ratios 0/1/1.5/2 at
        15/25/40%), MA-ATH anchor (63d MA high)
     6. CPPI user spec, raw-ATH anchor  <-- the frozen RC-15 rule

B. Theme universe 2014-2026 (equal-weight AI/clean/crypto composites,
   db_megatrend.h5): CPPI raw-ATH, gross — probe-grade estimate of the
   modern-era protection/premium (the official evaluation run measures
   the share-weighted mirror index with costs; see registration).

All rules frozen before running; numbers are reported, not selected.
"""
from __future__ import annotations

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))   # luan_bot_trading/
load_dotenv(os.path.join(ROOT, ".env"))
KEY = os.getenv("TIINGO_API_KEY")
CACHE = os.path.join(HERE, "archive", "experiments", "rc15_2000_basket.csv")

WANT = ["CSCO", "ORCL", "QCOM", "MU", "INTC", "MSFT", "JDSU", "NVDA"]


# ---------- data ----------
def resolve(t: str):
    try:
        r = requests.get(
            f"https://api.tiingo.com/tiingo/utilities/search/{t}",
            params={"token": KEY, "includeDelisted": "true",
                    "exactTickerMatch": "false"}, timeout=30)
        for c in r.json():
            if c.get("ticker") == t:
                return c.get("permaTicker") or t
    except Exception:
        pass
    return None


def fetch_basket() -> pd.DataFrame:
    if os.path.exists(CACHE):
        P = pd.read_csv(CACHE, index_col=0)
        P.index = pd.to_datetime(P.index)
        print(f"  loaded cached basket ({P.shape[1]} names, {P.index.min().date()}..{P.index.max().date()})")
        return P
    import time
    series = {}
    for t in WANT:
        pt = resolve(t) or t
        d = None
        for attempt in range(3):
            try:
                r = requests.get(
                    f"https://api.tiingo.com/tiingo/daily/{pt}/prices",
                    params={"token": KEY, "startDate": "1996-01-01",
                            "endDate": "2002-12-31", "columns": "date,adjClose"},
                    timeout=60)
                if r.status_code == 200:
                    d = pd.DataFrame(r.json())
                    break
            except Exception:
                pass
            time.sleep(5 * (attempt + 1))
        if d is None or d.empty:
            print(f"  !! {t} unavailable")
            continue
        d["date"] = pd.to_datetime(d["date"]).dt.tz_localize(None)
        series[t] = d.set_index("date")["adjClose"].astype(float)
        time.sleep(1.0)
    P = pd.DataFrame(series).dropna(how="all")
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    P.to_csv(CACHE)
    return P


# ---------- gate machinery (2000 exam variants) ----------
def cppi_cash(dd: float) -> float:
    d = -min(dd, 0.0)
    if d < 0.15:
        return 0.0
    if d < 0.25:
        return min(1.0, d * 1.0)
    if d < 0.40:
        return min(1.0, d * 1.5)
    return min(1.0, d * 2.0)


def band_cash(dd: float) -> float:
    d = -min(dd, 0.0)
    if d < 0.15:
        return 0.0
    if d < 0.25:
        return d - 0.15
    if d < 0.40:
        return 0.10 + (d - 0.25) * 1.5
    return min(1.0, 0.325 + (d - 0.40) * 2.0)


def weekly_schedule_exposure(nav_m, cash_fn, dd_of):
    wkdd = dd_of(nav_m).resample("W-FRI").last()
    exposure = pd.Series(1.0, index=nav_m.index)
    for w, v in wkdd.items():
        if pd.isna(v):
            continue
        e = 1.0 - cash_fn(v)
        nxt = nav_m.index.searchsorted(w + pd.Timedelta(days=1))
        end = nav_m.index.searchsorted(w + pd.Timedelta(weeks=1))
        exposure.iloc[nxt:max(end, nxt + 1)] = e
    return exposure


def alarm_flat_depth(P, basket, mode):
    """Two-cadence alarm (W + monthly bear-bloc) with flat-50% or depth rule."""
    M = P.resample("ME").last()
    B = (M > M.rolling(10).mean()).mean(axis=1)
    R = P.pct_change().dropna(how="all").fillna(0)
    dates, arr = R.index, R.to_numpy()
    Cv = []
    for i in range(59, len(dates)):
        w = arr[i - 59:i + 1]
        Z = (w - w.mean(0)) / (w.std(0) + 1e-12)
        c = (Z.T @ Z) / 59.0
        n = c.shape[0]
        Cv.append((dates[i], float(c[np.triu_indices(n, 1)].mean())))
    C = pd.Series(dict(Cv))
    hist, Cp = [], {}
    for d, c in C.items():
        hist.append(c)
        if len(hist) >= 756:
            Cp[d] = float((np.array(hist[:-1]) <= c).mean() * 100)
    Cp = pd.Series(Cp)

    def m_reg(m_end):
        if not (Cp.index <= m_end).any() or m_end not in B.index:
            return False
        return (Cp[Cp.index <= m_end].iloc[-1] >= 60.0) and (B.loc[m_end] < 0.5)

    Mreg = pd.Series({m: m_reg(m) for m in M.index})
    Wk = P.resample("W-FRI").last()
    nb = P.shape[1]
    below_n = nb - (Wk > Wk.rolling(10).mean()).sum(axis=1)
    W_on = (below_n >= 6) & (below_n.shift(1) >= 6)
    nav = (1 + basket.fillna(0)).cumprod()
    dd = nav / nav.cummax() - 1.0
    exposure = pd.Series(1.0, index=basket.index)
    state, last = "NORMAL", 1.0
    for w, w_on in W_on.items():
        if w < pd.Timestamp("1999-01-04") or pd.isna(w_on):
            continue
        mm = Mreg[Mreg.index <= w]
        m = bool(mm.iloc[-1]) if len(mm) else False
        streak = 0
        for me in reversed(mm.index):
            if not Mreg[me]:
                streak += 1
            else:
                break
        Midx = M.mean(axis=1)
        u10 = Midx.loc[mm.index[-1]] > Midx.rolling(10).mean().loc[mm.index[-1]] \
            if len(mm) else False
        if state == "NORMAL" and bool(w_on) and m:
            state = "ALARM"
        elif state == "ALARM" and streak >= 2 and u10:
            state = "NORMAL"
        if state == "ALARM":
            dd_w = dd.loc[:w].iloc[-1]
            last = 0.5 if mode == "flat" else float(np.clip(1 + dd_w, 0.20, 1.0))
        else:
            last = 1.0 if last >= 1.0 else min(1.0, last + 0.20)
        nxt = basket.index.searchsorted(w + pd.Timedelta(days=1))
        end = basket.index.searchsorted(w + pd.Timedelta(weeks=1))
        exposure.iloc[nxt:max(end, nxt + 1)] = last
    return exposure


def report(basket, exposure, label):
    port = basket.fillna(0) * exposure.shift(1).fillna(1.0)
    nav_p = (1 + port).cumprod()
    nav_m = (1 + basket.fillna(0)).cumprod()
    ddp = nav_p / nav_p.cummax() - 1
    ddm = nav_m / nav_m.cummax() - 1
    print(f"  {label:44s} maxDD {ddp.min()*100:6.1f}%  total {nav_p.iloc[-1]*100-100:+7.0f}%"
          f"  avg cash {(1-exposure).mean()*100:3.0f}%")
    return ddp.min()


def dd_raw_ath(nav):
    return nav / nav.cummax() - 1.0


def dd_ma_ath(nav):
    return nav / nav.rolling(63, min_periods=20).mean().cummax() - 1.0


def dd_roll104w(nav):
    return nav / nav.rolling(104 * 5, min_periods=60).max() - 1.0


def main() -> None:
    print("=" * 100)
    print("RC-15 EVIDENCE RECONSTRUCTION (pre-registration exhibits)")
    print("=" * 100)
    print("[1] fetching 2000 basket ...")
    P = fetch_basket()
    basket = P.pct_change().mean(axis=1)
    nav_m = (1 + basket.fillna(0)).cumprod()
    ddm = nav_m / nav_m.cummax() - 1
    print(f"\n[A] 2000-02 leader-death exam — basket maxDD {ddm.min()*100:.1f}%")
    print(f"  {'(1) ungated basket':44s} maxDD {ddm.min()*100:6.1f}%  total {nav_m.iloc[-1]*100-100:+7.0f}%")
    report(basket, alarm_flat_depth(P, basket, "flat"),
           "(2) alarm + flat 50%")
    report(basket, alarm_flat_depth(P, basket, "depth"),
           "(3) alarm + depth rule (cash=dd, clip 80%)")
    report(basket, weekly_schedule_exposure(nav_m, band_cash, dd_roll104w),
           "(4) band schedule (misreading), roll-104w anchor")
    report(basket, weekly_schedule_exposure(nav_m, cppi_cash, dd_ma_ath),
           "(5) CPPI user spec, MA-ATH anchor")
    report(basket, weekly_schedule_exposure(nav_m, cppi_cash, dd_raw_ath),
           "(6) CPPI user spec, RAW-ATH anchor  [RC-15 rule]")

    print(f"\n[B] theme universe 2014-2026 — CPPI raw-ATH (probe, gross)")
    with pd.HDFStore(os.path.join(ROOT, "01_data", "db_megatrend.h5"), "r") as s:
        px = {k.split("/")[-1]: s[k].set_index("date")["adjClose"] for k in s.keys()}
    frames = []
    for t, syms in {"AI": ["SMH"], "clean": ["ICLN", "TAN"],
                    "crypto": ["MSTR", "COIN"]}.items():
        comps = [(px[x] / px[x].dropna().iloc[0]).rename(x) for x in syms if x in px]
        frames.append(pd.concat(comps, axis=1).mean(axis=1).rename(t))
    TH = pd.concat(frames, axis=1).mean(axis=1).dropna()
    navb = (1 + TH.pct_change().fillna(0)).cumprod()
    print(f"  {'(B0) ungated equal-weight index':44s} maxDD "
          f"{(navb/navb.cummax()-1).min()*100:6.1f}%  total {navb.iloc[-1]*100-100:+7.0f}%")
    report(TH.pct_change(), weekly_schedule_exposure(navb, cppi_cash, dd_raw_ath),
           "(B1) CPPI raw-ATH (probe)")


if __name__ == "__main__":
    main()
