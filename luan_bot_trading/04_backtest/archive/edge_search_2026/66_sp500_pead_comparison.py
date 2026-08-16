#!/usr/bin/env python3
"""66_sp500_pead_comparison.py — PEAD event frequency: S&P 500 vs S&P 400.

QUESTION
--------
Are PEAD events as common in the S&P 500 as in the S&P 400? (Prior belief:
large-cap PEAD has eroded.) This script measures it with the SAME gate
definition the pipeline uses for the S&P 400:

    pass_g1 : car_10d  > +3%        (log CAR vs benchmark, T+1..T+11)
    pass_g2 : vol_ratio > 2.0       ((Vol_T..T+2)/3) / vma20
    pass_g3 : maxdd_ma > -1.5%      (worst relative drawdown, T+1..T+11)
    pead_pass = all three

METHOD (mirrors 04_backtest/_pead_target_retrain.py exactly)
------------------------------------------------------------
- S&P 500 constituents parsed from Wikipedia (current members — survivorship
  caveat: current members' histories only; slightly FLATTERS sp500 results if
  anything, since dropped members underperformed).
- Earnings dates: FMP /stable/earnings per ticker (full history).
- Prices: Tiingo daily EOD (Adj_Close/Adj_Volume), 2014-06-01 -> today.
- Benchmark for sp500 events: SPY. S&P 400 side uses the persisted
  train_matrix_v4_timing_correct (gates already computed vs IJH).
- T = first trading close on/after report_date; event kept if
  t_idx >= 20 (vma20 priming) and t_idx+12 < len(series).
- Both sides filtered to report_date >= 2015-01-01.

DATA / CACHE
------------
Fetched data cached in luan_bot_trading/01_data/db_sp500.h5
(/sp500/earnings per ticker, /sp500/prices/{TICKER}, /sp500/benchmark_SPY).
Re-run with --refresh to force re-download. db.h5 (production) is untouched.

USAGE
-----
    conda run -n trading python luan_bot_trading/04_backtest/66_sp500_pead_comparison.py
    ... --refresh          # ignore cache, re-download everything
"""
from __future__ import annotations
import io
import sys
import time
from pathlib import Path

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
import os

HERE = Path(__file__).resolve().parent          # archive/edge_search_2026/
ROOT = HERE.parents[2]                             # luan_bot_trading/ (2 up from archive/edge_search_2026)
load_dotenv(ROOT / ".env")
TIINGO_API_KEY = os.getenv("TIINGO_API_KEY")
FMP_API_KEY = os.getenv("FMP_API_KEY")

DB_PROD = ROOT / "01_data" / "db.h5"
DB_SP500 = ROOT / "01_data" / "db_sp500.h5"
SP500_MATRIX_KEY = "/features/train_matrix_v4_timing_correct"

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
FMP_BASE = "https://financialmodelingprep.com/stable"
START_DATE = "2014-06-01"          # vma20 priming runway before 2015 events
EVENT_START = pd.Timestamp("2015-01-01")
BENCH_TICKER = "SPY"

# Gates — identical to _pead_target_retrain.py
GATE1_CAR_MIN = 0.03
GATE2_VOL_RATIO_MIN = 2.0
GATE3_MAXDD_MIN = -0.015


def _wiki_symbol_to_api(sym: str) -> str:
    """Wikipedia uses dots for class shares (BRK.B); Tiingo/FMP use dashes."""
    return sym.strip().replace(".", "-")


def fetch_sp500_symbols() -> list[str]:
    print("[1] Parsing S&P 500 constituents from Wikipedia ...")
    r = requests.get(WIKI_URL, headers={"User-Agent": "research-script"}, timeout=60)
    r.raise_for_status()
    tables = pd.read_html(io.StringIO(r.text))
    # Constituents table is the first one with a 'Symbol' column.
    for t in tables:
        if "Symbol" in t.columns:
            syms = sorted({_wiki_symbol_to_api(s) for s in t["Symbol"].dropna().astype(str)})
            print(f"    {len(syms)} current constituents")
            return syms
    raise RuntimeError("Wikipedia constituents table (Symbol column) not found")


def fetch_tiingo(ticker: str) -> pd.DataFrame | None:
    url = f"https://api.tiingo.com/tiingo/daily/{requests.utils.quote(ticker)}/prices"
    params = {"token": TIINGO_API_KEY, "startDate": START_DATE}
    for _ in range(2):
        try:
            resp = requests.get(url, params=params, timeout=60)
            if resp.status_code == 200:
                df = pd.DataFrame(resp.json())
                # Tiingo returns lowercase keys when no `columns` param is set.
                ren = {"date": "Date", "adjClose": "Adj_Close", "adjVolume": "Adj_Volume"}
                df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})
                need = ["Date", "Adj_Close", "Adj_Volume"]
                if df.empty or not set(need).issubset(df.columns):
                    return None
                df = df[need].dropna().copy()
                df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
                return df.sort_values("Date").reset_index(drop=True)
        except Exception:
            time.sleep(2)
    return None


def fetch_fmp_earnings(ticker: str) -> pd.DataFrame | None:
    url = f"{FMP_BASE}/earnings"
    params = {"symbol": ticker, "apikey": FMP_API_KEY, "includeReportTimes": "true"}
    for _ in range(2):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                data = r.json()
                if not isinstance(data, list) or not data:
                    return None
                df = pd.DataFrame(data)
                if "date" not in df.columns:
                    return None
                return pd.DataFrame({"report_date": pd.to_datetime(df["date"]).dt.normalize()})
        except Exception:
            time.sleep(2)
    return None


def gather(symbols: list[str], refresh: bool) -> tuple[dict, pd.DataFrame]:
    """Fetch + cache earnings dates and prices for all symbols; return
    (prices_by_ticker, benchmark_df)."""
    mode = "a" if DB_SP500.exists() else "w"
    if refresh and DB_SP500.exists():
        DB_SP500.unlink()
        mode = "w"
    fetched_earn, fetched_px = 0, 0
    bench = None
    with pd.HDFStore(DB_SP500, mode=mode) as store:
        keys = set(store.keys())
        bench_key = f"/sp500/benchmark_{BENCH_TICKER}"
        if bench_key in keys:
            bench = store[bench_key]
        else:
            print(f"    fetching benchmark {BENCH_TICKER} ...")
            bench = fetch_tiingo(BENCH_TICKER)
            if bench is None:
                raise RuntimeError("could not fetch SPY benchmark")
            store.put(bench_key, bench, format="table")
        keys = set(store.keys())
        for i, sym in enumerate(symbols, 1):
            ek, pk = f"/sp500/earnings/{sym}", f"/sp500/prices/{sym}"
            need_e, need_p = ek not in keys, pk not in keys
            if not (need_e or need_p):
                continue
            if need_e:
                e = fetch_fmp_earnings(sym)
                if e is not None:
                    store.put(ek, e, format="table")
                    fetched_earn += 1
                time.sleep(0.12)
            if need_p:
                p = fetch_tiingo(sym)
                if p is not None:
                    store.put(pk, p, format="table")
                    fetched_px += 1
                time.sleep(0.12)
            if i % 50 == 0:
                print(f"    [{i}/{len(symbols)}] fetched this run: {fetched_earn} earnings, {fetched_px} price series")
    print(f"    fetched this run: {fetched_earn} earnings, {fetched_px} price series (cache: {DB_SP500.name})")
    # reload prices into memory
    prices = {}
    with pd.HDFStore(DB_SP500, mode="r") as store:
        for sym in symbols:
            pk = f"/sp500/prices/{sym}"
            if pk in store.keys():
                prices[sym] = store[pk]
    return prices, bench


def compute_events(symbols: list[str], prices: dict, bench: pd.DataFrame) -> pd.DataFrame:
    """Compute the 3 gates for every earnings event (S&P 500 side)."""
    print("[3] Computing PEAD gates for S&P 500 events ...")
    bench_idx = bench["Date"].values
    bench_close = bench["Adj_Close"].values.astype(float)
    bench_logret = np.diff(np.log(bench_close))
    rows = []
    n_earn = 0
    with pd.HDFStore(DB_SP500, mode="r") as store:
        for i, sym in enumerate(symbols, 1):
            ek = f"/sp500/earnings/{sym}"
            if ek not in store.keys():
                continue
            ev = store[ek]
            px = prices.get(sym)
            if px is None or px.empty:
                continue
            p_idx = px["Date"].values
            p_close = px["Adj_Close"].values.astype(float)
            p_vol = px["Adj_Volume"].values.astype(float)
            p_logret = np.diff(np.log(p_close))
            for rd in ev["report_date"]:
                rd = pd.Timestamp(rd)
                if rd < EVENT_START:
                    continue
                n_earn += 1
                t_mask = p_idx >= rd.to_datetime64()
                if not t_mask.any():
                    continue
                t = int(np.argmax(t_mask))
                if t < 20 or t + 12 >= len(p_close):
                    continue
                vma20 = float(np.mean(p_vol[t - 20:t]))
                if vma20 <= 0:
                    continue
                vol_ratio = float(np.mean(p_vol[t:t + 3])) / vma20
                # CAR T+1..T+11 (log, vs SPY)
                n = min(11, len(p_logret) - t, len(bench_logret) - 0)
                bt = int(np.searchsorted(bench_idx, p_idx[t]))
                if bt + 12 >= len(bench_close):
                    continue
                car10 = float(np.sum(p_logret[t:t + 11] - bench_logret[bt:bt + 11]))
                # maxdd relative, T+1..T+11
                s_path = p_close[t + 1:t + 12] / p_close[t] - 1.0
                b_path = bench_close[bt + 1:bt + 12] / bench_close[bt] - 1.0
                m = min(len(s_path), len(b_path))
                maxdd = float(np.min(s_path[:m] - b_path[:m]))
                rows.append({"ticker": sym, "report_date": rd, "year": rd.year,
                             "car_10d": car10, "vol_ratio": vol_ratio, "maxdd_ma": maxdd,
                             "pass_g1": int(car10 > GATE1_CAR_MIN),
                             "pass_g2": int(vol_ratio > GATE2_VOL_RATIO_MIN),
                             "pass_g3": int(maxdd > GATE3_MAXDD_MIN)})
            if i % 100 == 0:
                print(f"    [{i}/{len(symbols)}] events so far: {len(rows):,}")
    df = pd.DataFrame(rows)
    df["pead_pass"] = (df.pass_g1 & df.pass_g2 & df.pass_g3).astype(int)
    print(f"    {n_earn:,} earnings events since {EVENT_START.date()}; "
          f"{len(df):,} computable")
    return df


def yearly_table(df: pd.DataFrame, name: str) -> pd.DataFrame:
    g = df.groupby("year").agg(
        events=("pead_pass", "size"),
        g1=("pass_g1", "mean"), g2=("pass_g2", "mean"),
        g3=("pass_g3", "mean"), pead=("pead_pass", "mean"),
        mean_car=("car_10d", "mean"),
    )
    g.name = name
    return g


def main():
    refresh = "--refresh" in sys.argv
    print("=" * 78)
    print("S&P 500 vs S&P 400 — PEAD event frequency comparison")
    print("=" * 78)
    symbols = fetch_sp500_symbols()

    print(f"[2] Gathering data (cache {DB_SP500.name}, refresh={refresh}) ...")
    prices, bench = gather(symbols, refresh)

    sp5 = compute_events(symbols, prices, bench)

    print("[4] Loading S&P 400 persisted matrix (gates vs IJH) ...")
    sp4 = pd.read_hdf(DB_PROD, SP500_MATRIX_KEY)
    sp4["year"] = pd.to_datetime(sp4["report_date"]).dt.year
    sp4 = sp4[sp4.report_date >= EVENT_START]
    print(f"    {len(sp4):,} sp400 events since {EVENT_START.date()}")

    t5 = yearly_table(sp5, "sp500")
    t4 = yearly_table(sp4, "sp400")
    cmp = pd.concat({"S&P 500": t5, "S&P 400": t4}, axis=1)
    print("\n" + "=" * 78)
    print("Year-by-year gate pass rates  (g1=CAR>3%, g2=Vol>2x, g3=MaxDD>-1.5%)")
    print("=" * 78)
    pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
    print(cmp.to_string())

    print("\n" + "=" * 78)
    print("OVERALL (2015+)")
    print("=" * 78)
    for nm, d in [("S&P 500", sp5), ("S&P 400", sp4)]:
        print(f"  {nm:8s} n={len(d):6,}  "
              f"PEAD rate={d.pead_pass.mean()*100:5.2f}%  "
              f"g1={d.pass_g1.mean()*100:5.1f}%  "
              f"g2={d.pass_g2.mean()*100:5.1f}%  "
              f"g3={d.pass_g3.mean()*100:5.1f}%  "
              f"mean car_10d={d.car_10d.mean()*100:+5.2f}%")
    # erosion slope: rate in first 3 vs last 3 full years
    yrs = sorted(sp5.year.unique())
    for nm, d in [("S&P 500", sp5), ("S&P 400", sp4)]:
        early = d[d.year.isin([y for y in yrs if 2015 <= y <= 2017])]
        late = d[d.year.isin([y for y in yrs if 2023 <= y <= 2025])]
        print(f"  {nm} PEAD rate 2015-17: {early.pead_pass.mean()*100:.2f}% (n={len(early):,})   "
              f"2023-25: {late.pead_pass.mean()*100:.2f}% (n={len(late):,})")


if __name__ == "__main__":
    main()
