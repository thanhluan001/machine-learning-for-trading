#!/usr/bin/env python3
"""70_preex_dividend_runup.py — Event study: pre-ex-dividend run-up (S&P 400).

RESEARCH QUESTION (slow-week edge candidate)
--------------------------------------------
Do stocks drift UP in the days before their ex-dividend date? Specifically —
following the PEAD playbook — what fraction of ex-div events have a "big"
run-up (CAR > 1% vs IJH over the pre-ex window)? If ~10-20% of events are
"interesting" AND the tail is predictable ex-ante (e.g., by dividend yield),
a filtered variant might be worth pursuing.

EXECUTION ASSUMPTIONS
---------------------
- Ex-date and amount are public knowledge well in advance (announced with the
  dividend declaration, typically weeks ahead) — so entering N days before the
  ex-date is look-ahead-free.
- Entry: Close[T-N] (N trading days before ex-date). Exit: Close[T-1] (last
  close before ex-date). We do NOT hold through the ex-date (avoids the
  mechanical price drop + ordinary-income conversion).
- Returns relative to IJH (same convention as pipeline).

ANALYSES
--------
1. Run-up CAR by window: N in {3, 5, 10} trading days before ex.
2. Tail frequency: P(CAR > +1%) and P(CAR > +2%) per window — the PEAD-style
   "how often is it interesting" question.
3. Conditioning on dividend yield (ex-ante observable!): quartile buckets.
   Does yield predict the tail?
4. Supply: events per month (slow-week fit), events per year (stability).

DATA
----
- Ex-dividend dates/amounts: Tiingo daily bars `divCash` field (subscription
  includes it), fetched fresh into 01_data/db_div.h5 (production db.h5
  untouched). Back to 2014-06.
- Prices: /sp400/{pt} from db.h5 (Adj_Close).
- Benchmark: /macros/IJH from db.h5.
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # luan_bot_trading/ (2 up from archive/edge_search_2026)
load_dotenv(ROOT / ".env")
TIINGO_API_KEY = os.getenv("TIINGO_API_KEY")

DB_PROD = ROOT / "01_data" / "db.h5"
DB_DIV = ROOT / "01_data" / "db_div.h5"
START = "2014-06-01"
WINDOWS = [3, 5, 10]      # entry N trading days before ex-date
TAILS = [0.01, 0.02]      # "interesting" thresholds


def fetch_div_series(ticker: str) -> pd.DataFrame | None:
    """Fetch date/close/divCash for one ticker (light payload)."""
    url = f"https://api.tiingo.com/tiingo/daily/{requests.utils.quote(ticker)}/prices"
    params = {"token": TIINGO_API_KEY, "startDate": START,
              "columns": "date,close,divCash"}
    for _ in range(2):
        try:
            r = requests.get(url, params=params, timeout=60)
            if r.status_code == 200:
                df = pd.DataFrame(r.json())
                if df.empty or not {"date", "close", "divCash"}.issubset(df.columns):
                    return None
                df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
                df["close"] = pd.to_numeric(df["close"], errors="coerce")
                df["divCash"] = pd.to_numeric(df["divCash"], errors="coerce").fillna(0.0)
                return df.sort_values("date").reset_index(drop=True)
            elif r.status_code == 429:
                time.sleep(1.0)
        except Exception:
            time.sleep(1.0)
    return None


def ensure_div_data(meta: pd.DataFrame) -> None:
    mode = "a" if DB_DIV.exists() else "w"
    with pd.HDFStore(DB_DIV, mode=mode) as store:
        keys = set(store.keys())
        needed = [(r["permaTicker"], r["canonical_ticker"])
                  for _, r in meta.iterrows() if f"/div/{r['permaTicker']}" not in keys]
        print(f"[1] Fetching divCash series: {len(needed)} tickers needed "
              f"({len(keys)} cached) ...")
        n = 0
        for i, (pt, sym) in enumerate(needed, 1):
            df = fetch_div_series(sym)
            if df is not None:
                store.put(f"/div/{pt}", df, format="table")
                n += 1
            time.sleep(0.06)
            if i % 100 == 0:
                print(f"    [{i}/{len(needed)}] fetched {n}")
        print(f"    done: {n} new series, {len(store.keys())} total")


def main():
    print("=" * 86)
    print("PRE-EX-DIVIDEND RUN-UP EVENT STUDY — S&P 400 (2015–2026)")
    print("=" * 86)

    meta = pd.read_hdf(DB_PROD, "/metadata/sp400_permatickers")
    ensure_div_data(meta)

    print("[2] Computing run-up CARs ...")
    with pd.HDFStore(DB_PROD, mode="r") as ps:
        ijh = ps["/macros/IJH"].copy()
        ijh["Date"] = pd.to_datetime(ijh["Date"]).dt.tz_localize(None).dt.normalize()
        ijh = ijh.sort_values("Date").reset_index(drop=True)
        i_dates = ijh["Date"].to_numpy()
        i_close = ijh["Close"].to_numpy(float)
        prod_keys = set(ps.keys())
        px_cache = {}
        for _, r in meta.iterrows():
            pt = r["permaTicker"]
            if f"/sp400/{pt}" in prod_keys:
                p = ps[f"/sp400/{pt}"].copy()
                p["Date"] = pd.to_datetime(p["Date"]).dt.tz_localize(None).dt.normalize()
                p = p.sort_values("Date").reset_index(drop=True)
                px_cache[pt] = (p["Date"].to_numpy(), p["Adj_Close"].to_numpy(float),
                                p["Adj_Volume"].to_numpy(float))

    events = []
    with pd.HDFStore(DB_DIV, mode="r") as store:
        for _, r in meta.iterrows():
            pt = r["permaTicker"]
            dk = f"/div/{pt}"
            if dk not in store.keys():
                continue
            d = store[dk]
            divs = d[d["divCash"] > 0]
            if divs.empty:
                continue
            pl = px_cache.get(pt)
            if pl is None:
                continue
            pdates, pclose, pvol = pl
            div_dates = divs["date"].to_numpy()
            div_amts = divs["divCash"].to_numpy(float)
            for _, dv in divs.iterrows():
                ex = dv["date"]
                if ex < pd.Timestamp("2015-01-01"):
                    continue
                # index of first trading day ON/AFTER ex-date; T-1 = last close before ex
                x = int(np.searchsorted(pdates, np.datetime64(ex), side="left"))
                if x < 1 or x >= len(pclose):
                    continue
                t1_close = pclose[x - 1]           # last close before ex-date
                if not np.isfinite(t1_close) or t1_close <= 0:
                    continue
                dyield = float(dv["divCash"]) / tclose_safe(dv, t1_close)
                # TTM annualized yield (sum of dividends in trailing 365d / raw close)
                lo = np.datetime64(ex - pd.Timedelta(days=365))
                hi = np.datetime64(ex)
                i_lo = int(np.searchsorted(div_dates, lo, side="left"))
                i_hi = int(np.searchsorted(div_dates, hi, side="right"))
                ttm_yield = float(div_amts[i_lo:i_hi].sum()) / float(dv["close"]) if dv["close"] > 0 else np.nan
                # benchmark alignment at T-1
                bt = int(np.searchsorted(i_dates, pdates[x - 1], side="left"))
                if bt < 1:
                    continue
                row = {"pt": pt, "ex_date": ex, "year": ex.year, "month": ex.month,
                       "yield_pct": dyield * 100, "ttm_yield_pct": ttm_yield * 100}
                # entry-index diagnostics for the filter stack (T-5 entry)
                e5 = x - 1 - 5
                be5 = bt - 5
                if e5 >= 49 and be5 >= 49:
                    row["stk_above_sma50"] = bool(pclose[e5] > np.mean(pclose[e5 - 49:e5 + 1]))
                    row["ijh_above_sma50"] = bool(i_close[be5] > np.mean(i_close[be5 - 49:be5 + 1]))
                else:
                    row["stk_above_sma50"] = np.nan
                    row["ijh_above_sma50"] = np.nan
                if e5 >= 19:
                    row["adv20_usd"] = float(np.mean(pvol[e5 - 19:e5 + 1]) * pclose[e5])
                else:
                    row["adv20_usd"] = np.nan
                ok = True
                for N in WINDOWS:
                    e = x - 1 - N               # entry index N td before ex
                    be = bt - N
                    if e < 0 or be < 0:
                        row[f"car_{N}d"] = np.nan
                        continue
                    stk = pclose[x - 1] / pclose[e] - 1.0
                    bmk = i_close[bt] / i_close[be] - 1.0
                    row[f"car_{N}d"] = stk - bmk
                events.append(row)
    df = pd.DataFrame(events)
    print(f"    ex-dividend events (2015+): {len(df):,}")

    def stats(d, col):
        v = d[col].to_numpy(float)
        v = v[~np.isnan(v)]
        if len(v) == 0:
            return None
        return {"n": len(v), "mean": v.mean(), "med": np.median(v),
                "win": (v > 0).mean(),
                "p1": (v > 0.01).mean(), "p2": (v > 0.02).mean()}

    print("\n[3] Run-up CAR by window (entry T-N close -> exit T-1 close, vs IJH)")
    print(f"  {'window':>8} {'n':>7} {'mean':>8} {'median':>8} {'win%':>6} "
          f"{'P(>1%)':>8} {'P(>2%)':>8}")
    for N in WINDOWS:
        s = stats(df, f"car_{N}d")
        if s:
            print(f"  T-{N:>6} {s['n']:>7,} {s['mean']*100:>+7.2f}% {s['med']*100:>+7.2f}% "
                  f"{s['win']*100:>5.1f}% {s['p1']*100:>7.1f}% {s['p2']*100:>7.1f}%")

    # yield conditioning (ex-ante observable)
    print("\n[4] Conditioning on DIVIDEND YIELD (ex-ante observable), window T-5")
    col = "car_5d"
    qs = df["yield_pct"].quantile([0.25, 0.5, 0.75]).to_numpy()
    print(f"    yield quartiles: <{qs[0]:.2f}% | {qs[0]:.2f}-{qs[1]:.2f}% | "
          f"{qs[1]:.2f}-{qs[2]:.2f}% | >{qs[2]:.2f}%")
    buckets = [
        ("Q1 smallest", df[df.yield_pct <= qs[0]]),
        ("Q2", df[(df.yield_pct > qs[0]) & (df.yield_pct <= qs[1])]),
        ("Q3", df[(df.yield_pct > qs[1]) & (df.yield_pct <= qs[2])]),
        ("Q4 largest", df[df.yield_pct > qs[2]]),
        (">=0.5% yield", df[df.yield_pct >= 0.5]),
        (">=0.75% yield", df[df.yield_pct >= 0.75]),
    ]
    print(f"  {'bucket':>14} {'n':>7} {'mean':>8} {'win%':>6} {'P(>1%)':>8} {'P(>2%)':>8}")
    for nm, d in buckets:
        s = stats(d, col)
        if s:
            print(f"  {nm:>14} {s['n']:>7,} {s['mean']*100:>+7.2f}% {s['win']*100:>5.1f}% "
                  f"{s['p1']*100:>7.1f}% {s['p2']*100:>7.1f}%")

    # conditional expectancy: among Q4 (big dividends), mean CAR of the >1% tail
    print("\n[5] Tail expectancy — if we could pick only >1% events (upper bound)")
    for nm, d in [("all", df), (">=0.5% yield", df[df.yield_pct >= 0.5])]:
        v = d[col].to_numpy(float); v = v[~np.isnan(v)]
        tail = v[v > 0.01]
        print(f"  {nm:>14}: P(>1%)={len(tail)/len(v)*100:5.1f}%  "
              f"mean of tail={tail.mean()*100:+.2f}%  "
              f"unconditional mean={v.mean()*100:+.2f}%")

    print("\n[6] Monthly supply (all events)")
    mo = df.groupby("month").size()
    mnames = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
    for m in range(1, 13):
        print(f"  {mnames[m-1]:3s}  {mo.get(m,0):5,}")

    print("\n[7] Year-by-year (window T-5, all events)")
    for y in sorted(df.year.unique()):
        s = stats(df[df.year == y], col)
        if s:
            print(f"  {y}: n={s['n']:5,}  mean={s['mean']*100:+.2f}%  "
                  f"P(>1%)={s['p1']*100:5.1f}%")

    # ------------------------------------------------------------------
    # [8] USER FILTER STACK (all ex-ante observable):
    #   F1  annualized TTM dividend yield > 2.5%
    #   F2  + stock AND IJH both above their 50-day SMA at entry (T-5 close)
    #   F3  + ADV20 dollar volume >= $50M   (F3s: stricter $100M)
    # ------------------------------------------------------------------
    print("\n[8] FILTER STACK (window T-5)")
    f1 = df[df.ttm_yield_pct > 2.5]
    f2 = f1[(f1.stk_above_sma50 == True) & (f1.ijh_above_sma50 == True)]  # noqa: E712
    f3 = f2[f2.adv20_usd >= 50e6]
    f3s = f2[f2.adv20_usd >= 100e6]
    layers = [
        ("all events (reference)", df),
        ("F1 yield>2.5% ann.", f1),
        ("F2 + both>SMA50", f2),
        ("F3 + ADV20>=$50M", f3),
        ("F3s + ADV20>=$100M", f3s),
    ]
    print(f"  {'layer':>22} {'n':>7} {'mean':>8} {'median':>8} {'win%':>6} "
          f"{'P(>1%)':>8} {'P(>2%)':>8}")
    for nm, d in layers:
        s = stats(d, "car_5d")
        if s:
            print(f"  {nm:>22} {s['n']:>7,} {s['mean']*100:>+7.2f}% {s['med']*100:>+7.2f}% "
                  f"{s['win']*100:>5.1f}% {s['p1']*100:>7.1f}% {s['p2']*100:>7.1f}%")
    if len(f3):
        med_adv = f3.adv20_usd.median()
        print(f"\n  F3 composition: median ADV20 = ${med_adv/1e6:.0f}M, "
              f"median TTM yield = {f3.ttm_yield_pct.median():.2f}%")
        # how much does the yield>trend+liquidity set OVERLAP with prior finding
        # (high yield anti-predictive)? split F2 by trend direction:
        dn = f1[(f1.stk_above_sma50 == False) | (f1.ijh_above_sma50 == False)]  # noqa: E712
        s_dn = stats(dn, "car_5d")
        if s_dn:
            print(f"  (contrast) F1 + BELOW SMA50 either side: n={s_dn['n']:,} "
                  f"mean={s_dn['mean']*100:+.2f}%  P(>1%)={s_dn['p1']*100:.1f}%")
        print("\n  F3 by year:")
        for y in sorted(f3.year.unique()):
            s = stats(f3[f3.year == y], "car_5d")
            if s:
                print(f"    {y}: n={s['n']:4,}  mean={s['mean']*100:+.2f}%  "
                      f"win={s['win']*100:.1f}%  P(>1%)={s['p1']*100:.1f}%")
        print("\n  F3 monthly supply:")
        mo3 = f3.groupby("month").size()
        mnames = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
        rowstr = "  "
        for m in range(1, 13):
            rowstr += f"{mnames[m-1]}:{mo3.get(m, 0):<5d}"
        print(rowstr)


def tclose_safe(dv, fallback):
    return fallback


if __name__ == "__main__":
    main()
