"""RC-12b Phase 3, step 3 — point-in-time SP600 matrix rebuild.

Differences vs corrected Phase 1 (98_rc12b_phase1_matrix --build):
1. Universe = current members (ptmap) + 320 newly mapped removed/graduated
   names (ptmap3) + 161 SP400-graduate tickers sourced from db.h5.
2. Membership filter: an event at report_date t is included ONLY if the
   ticker's /metadata/sp600 intervals cover t (added <= t < removed).
   This cuts BOTH ways: dead names' in-window events come IN, current
   members' pre-addition events go OUT.
3. Prices for graduates fall back to db.h5 /sp400/{pt}; earnings for
   graduates from db.h5 /earnings/fmp (canonical_ticker keyed, loaded
   once); grades from db.h5 /analyst/grades/{pt}.

Feature code copied VERBATIM from the corrected Phase-1 builder
(reset-after-null-filter ordering preserved — the 2026-08-31 indexing
bug lesson).

Output: /features/train_matrix_sp600_pt in db_sp600.h5.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB_SP600 = ROOT / "01_data" / "db_sp600.h5"
DB_PROD = ROOT / "01_data" / "db.h5"
WIN_START = pd.Timestamp("2018-09-01")
WIN_END = pd.Timestamp("2026-06-30")

GICS_ETF = {
    "Energy": "XLE", "Materials": "XLB", "Industrials": "XLI",
    "Consumer Discretionary": "XLY", "Consumer Staples": "XLP",
    "Health Care": "XLV", "Financials": "XLF", "Information Technology": "XLK",
    "Communication Services": "XLC", "Real Estate": "XLRE", "Utilities": "XLU",
}
GRADE_ORDINAL = {
    "strong sell": 1, "sell": 1, "underweight": 1, "underperform": 1,
    "reduce": 1, "reduce in price": 1,
    "negative": 2, "below average": 2, "below market": 2,
    "market underperform": 2, "market underperformer": 2,
    "hold": 3, "neutral": 3, "sector weight": 3, "sector perform": 3,
    "in-line": 3, "in-line market perform": 3, "sector weight (equal-weight)": 3,
    "market perform": 3, "equal-weight": 3, "equal weight": 3,
    "positive": 4, "above average": 4, "outperform": 4, "market outperform": 4,
    "add": 4, "accumulate": 4, "overweight": 4, "market overweight": 4,
    "sector outperform": 4, "strong buy": 5, "buy": 5,
}


def grade_to_ordinal(g):
    if g is None or (isinstance(g, float) and np.isnan(g)):
        return None
    return GRADE_ORDINAL.get(str(g).strip().lower())


def revision_features(gdf, rd):
    """8 revision features. NaN when no grades node; zeros when node exists
    but nothing before rd (verbatim Phase-1 semantics)."""
    nan8 = {k: np.nan for k in
            ["revision_momentum_30d", "revision_momentum_60d", "revision_momentum_90d",
             "revision_ordinal_momentum_90d", "revision_intensity_90d",
             "grade_dispersion_90d", "n_analysts_covering",
             "last_action_days_before_earnings"]}
    if gdf is None or gdf is None or len(gdf) == 0:
        return nan8
    gdf = gdf.copy()
    gdf["date"] = pd.to_datetime(gdf["date"], errors="coerce")
    pre = gdf[gdf["date"] < rd]
    if pre.empty:
        return {**{k: 0 for k in nan8 if k != "last_action_days_before_earnings"},
                "last_action_days_before_earnings": np.nan}
    pre["prev_o"] = pre["previous_grade"].apply(grade_to_ordinal)
    pre["new_o"] = pre["new_grade"].apply(grade_to_ordinal)
    pre["delta"] = pre.apply(
        lambda r: (r["new_o"] - r["prev_o"])
        if (r["prev_o"] is not None and r["new_o"] is not None) else None, axis=1)

    def net(df):
        if df.empty:
            return 0
        return int((df["action"] == "upgrade").sum() - (df["action"] == "downgrade").sum())

    def ordinal_mom(df):
        if df.empty:
            return 0.0
        d = df["delta"].dropna()
        return float(d.sum()) if len(d) else 0.0

    def intensity(df):
        if df.empty:
            return 0
        return int(((df["action"] == "upgrade") | (df["action"] == "downgrade")).sum())

    def dispersion(df):
        if df.empty:
            return 0
        o = df["new_o"].dropna()
        return int(o.nunique()) if len(o) else 0

    w90 = pre[pre["date"] >= rd - pd.Timedelta(days=90)]
    w60 = pre[pre["date"] >= rd - pd.Timedelta(days=60)]
    w30 = pre[pre["date"] >= rd - pd.Timedelta(days=30)]
    return {
        "revision_momentum_30d": net(w30),
        "revision_momentum_60d": net(w60),
        "revision_momentum_90d": net(w90),
        "revision_ordinal_momentum_90d": ordinal_mom(w90),
        "revision_intensity_90d": intensity(w90),
        "grade_dispersion_90d": dispersion(w90),
        "n_analysts_covering": int(w90["grading_company"].nunique()) if not w90.empty else 0,
        "last_action_days_before_earnings": float((rd - pre["date"].max()).days),
    }


def member_at(intervals: str, t: pd.Timestamp) -> bool:
    try:
        ivs = json.loads(str(intervals).replace("'", '"').replace("None", "null"))
    except Exception:
        return False
    for iv in ivs or []:
        a = iv.get("added")
        rm = iv.get("removed")
        try:
            a = pd.Timestamp(a) if a else None
            rm = pd.Timestamp(rm) if rm else None
        except Exception:
            continue
        if a is not None and a <= t and (rm is None or t < rm):
            return True
    return False


def main() -> None:
    with pd.HDFStore(DB_SP600, "r") as s:
        pm = s["/metadata/sp600_ptmap"]
        pm3 = s["/metadata/sp600_ptmap3"]
        md = s["/metadata/sp600"]
        iv_of = dict(zip(md.ticker.astype(str), md.intervals))
        sector_of = dict(zip(md.ticker.astype(str), md.gics_sector.map(GICS_ETF)))
    with pd.HDFStore(DB_PROD, "r") as s:
        pt400 = s["/metadata/sp400_permatickers"]
    grad_pt = dict(zip(pt400.canonical_ticker.astype(str), pt400.permaTicker))
    sp400_tickers = set(grad_pt)

    # ---- BULK LOAD (one pass per store; per-ticker opens are the
    # ---- timeout killer — lesson from the archived 98/99 bulk loaders)
    with pd.HDFStore(DB_SP600, "r") as store:
        keys = set(store.keys())
        bench = store["/sp600/benchmark_IJR"].copy()
        etf_px = {}
        for etf in set(sector_of.values()):
            k = f"/sp600/etf_{etf}"
            if k in keys:
                d = store[k].copy()
                d["Date"] = pd.to_datetime(d["Date"]).dt.tz_localize(None).dt.normalize()
                etf_px[etf] = d
        ekeys = {k.split("/")[-1] for k in store.keys() if k.startswith("/sp600/earnings_full")}
        gkeys = {k.split("/")[-1] for k in store.keys() if k.startswith("/sp600/grades")}
        ev_cache = {sym: store[f"/sp600/earnings_full/{sym}"].copy() for sym in ekeys}
        gr_cache = {sym: store[f"/sp600/grades/{sym}"].copy() for sym in gkeys}
        px600 = {k.split("/")[-1]: store[k].copy()
                 for k in store.keys() if k.startswith("/sp600/") and len(k.split("/")) == 3}
    bench["Date"] = pd.to_datetime(bench["Date"]).dt.tz_localize(None).dt.normalize()
    bench_s = pd.Series(bench["Adj_Close"].to_numpy(float), index=bench["Date"])

    # graduates' earnings + prices: one bulk load each
    grad_earn, px400, gr400 = {}, {}, {}
    with pd.HDFStore(DB_PROD, "r") as s:
        ef = s["/earnings/fmp"]
        ef = ef[(ef.report_date >= WIN_START - pd.Timedelta(days=400))]
        for sym, grp in ef.groupby("canonical_ticker"):
            grad_earn[sym] = pd.DataFrame({
                "report_date": pd.to_datetime(grp["report_date"]),
                "actual": pd.to_numeric(grp["eps_actual"], errors="coerce"),
                "estimate": pd.to_numeric(grp["eps_estimated"], errors="coerce"),
                "time": grp["before_after_market"].map(
                    lambda v: "bmo" if str(v).lower().startswith("b") else "amc"),
            }).sort_values("report_date").reset_index(drop=True)
        for pt in set(grad_pt.values()):
            k = f"/sp400/{pt}"
            if k in s.keys():
                px400[pt] = s[k].copy()
            k2 = f"/analyst/grades/{pt}"
            if k2 in s.keys():
                gr400[pt] = s[k2].copy()

    def load_price(pt: str):
        d = px600.get(pt)
        if d is None or len(d) == 0:
            d = px400.get(pt)
        if d is None or len(d) == 0:
            return None
        d = d.copy()
        d["Date"] = pd.to_datetime(d["Date"]).dt.tz_localize(None).dt.normalize()
        return d.sort_values("Date").reset_index(drop=True)

    # unified ticker->(permaTicker, earnings_source) registry
    registry = {}
    for r in pm.itertuples():
        registry[r.ticker] = (r.permaTicker, "sp600")
    for r in pm3.itertuples():
        if r.permaTicker and r.ticker not in registry:
            registry[r.ticker] = (r.permaTicker, "sp600")
    for sym in sp400_tickers:
        if sym in iv_of and sym not in registry:
            registry[sym] = (grad_pt[sym], "db400")

    print(f"registry: {len(registry)} tickers "
          f"(current+new={sum(1 for v in registry.values() if v[1]=='sp600')}, "
          f"graduates={sum(1 for v in registry.values() if v[1]=='db400')})")

    rows = []
    stats = {"no_member": 0, "no_px": 0, "no_ev": 0}
    for i, (sym, (pt, src)) in enumerate(sorted(registry.items()), 1):
        ivs = iv_of.get(sym)
        if ivs is None:
            continue
        if src == "sp600":
            if sym not in ev_cache:
                stats["no_ev"] += 1
                continue
            ev = ev_cache[sym]
            gdf = gr_cache.get(sym)
        else:
            ev = grad_earn.get(sym)
            gdf = gr400.get(pt)
            if ev is None:
                stats["no_ev"] += 1
                continue
        px = load_price(pt)
        if px is None or px.empty or len(px) < 60:
            stats["no_px"] += 1
            continue
        dates = px["Date"].to_numpy()
        close = px["Adj_Close"].to_numpy(float)
        vol = px["Adj_Volume"].to_numpy(float)
        s_ = pd.Series(close, index=px["Date"])
        bench_al = bench_s.reindex(s_.index, method="ffill")
        etf = sector_of.get(sym)
        etf_al = (pd.Series(etf_px[etf]["Adj_Close"].to_numpy(float),
                            index=etf_px[etf]["Date"]).reindex(s_.index, method="ffill")
                  if etf in etf_px else None)
        slr = np.diff(np.log(np.where(close > 0, close, np.nan)))
        blr = np.diff(np.log(bench_al.to_numpy(float)))
        ev = ev.sort_values("report_date").reset_index(drop=True)
        # reset AFTER the null filter (indexing-bug lesson, verbatim Phase-1)
        ev = ev[ev.actual.notna() & ev.estimate.notna()].reset_index(drop=True)
        if ev.empty:
            continue
        diff = ev["actual"] - ev["estimate"]
        roll = diff.rolling(12, min_periods=12).std(ddof=1)
        sue = diff / roll
        beat = (ev.actual > ev.estimate)
        consec = np.zeros(len(ev), dtype=int)
        run = 0
        for k in range(len(ev)):
            run = run + 1 if beat.iloc[k] else 0
            consec[k] = run
        ev["sue_lag_1"] = sue.shift(1)
        ev["sue_lag_2"] = sue.shift(2)
        ev["consec_pre"] = pd.Series(consec).shift(1)
        # RC-16 F2: prior event's 45-session CAR, NaN unless the 45th session
        # completed by the consuming event's cutoff (T-1 AMC / T-2 BMO).
        CAR_DRIFT_WINDOW = 45
        car45 = np.full(len(ev), np.nan)
        tpos = np.searchsorted(dates, ev["report_date"].to_numpy().astype("datetime64[D]").astype(dates.dtype), side="left")
        for k in range(len(ev)):
            t = int(tpos[k])
            if t + CAR_DRIFT_WINDOW + 1 < len(close):
                car45[k] = float(np.nansum(slr[t + 1:t + CAR_DRIFT_WINDOW + 1] - blr[t + 1:t + CAR_DRIFT_WINDOW + 1]))
        drift = pd.Series(car45).shift(1)
        mature = np.zeros(len(ev), dtype=bool)
        _bam = ev["time"].astype(str).str.lower().values
        for k in range(1, len(ev)):
            tk, tp_ = int(tpos[k]), int(tpos[k - 1])
            if tk + 11 >= len(dates):
                continue
            back = 1 if _bam[k] == "amc" else 2
            if (tk - back) - tp_ >= CAR_DRIFT_WINDOW:
                mature[k] = True
        drift[~mature] = np.nan
        ev["car_drift_q1"] = drift
        for k, rd in enumerate(ev["report_date"]):
            if not (WIN_START - pd.Timedelta(days=120) <= rd <= WIN_END):
                continue
            if not member_at(ivs, rd):          # <<< POINT-IN-TIME GATE
                stats["no_member"] += 1
                continue
            t = int(tpos[k])
            if t < 31 or t + 6 >= len(close):
                continue
            f = {"ticker": sym, "permaTicker": pt, "report_date": rd, "sector": etf}
            # RC-16 F3: label completion date (T+11 session) for
            # maturity-aware fold splits.
            f["label_end"] = pd.Timestamp(dates[t + 11]) if t + 11 < len(dates) else pd.NaT
            f["sue_lag_1"] = ev["sue_lag_1"].iloc[k]
            f["sue_lag_2"] = ev["sue_lag_2"].iloc[k]
            f["consecutive_surprises_pre"] = ev["consec_pre"].iloc[k]
            f["car_drift_historical_q1"] = ev["car_drift_q1"].iloc[k]
            idio = slr[t - 20:t] - blr[t - 20:t]
            f["pre_event_idiosyncratic_vol"] = float(np.std(idio, ddof=1)) if len(idio) == 20 else np.nan
            lv = np.log(vol[t - 10:t])
            if len(lv) == 10 and np.all(np.isfinite(lv)):
                f["pre_event_volume_trend"] = float(np.polyfit(np.arange(10, dtype=float), lv, 1)[0])
            else:
                f["pre_event_volume_trend"] = np.nan
            for h in (3, 5, 10, 20, 30):
                lo = t - 1 - h
                f[f"rel_ret_{h}d"] = (np.log(close[t - 1] / close[lo])
                                      - np.log(bench_al.iloc[t - 1] / bench_al.iloc[lo])) if lo >= 0 else np.nan
            if etf_al is not None and t - 21 >= 0:
                f["sector_adjusted_ret_20d"] = (np.log(close[t - 1] / close[t - 21])
                                                - np.log(etf_al.iloc[t - 1] / etf_al.iloc[t - 21]))
            else:
                f["sector_adjusted_ret_20d"] = np.nan
            f["adv20"] = float(np.nanmean(close[t - 20:t] * vol[t - 20:t]))
            f.update(revision_features(gdf, rd))
            tm = (ev["time"].iloc[k] or "").lower() if isinstance(ev["time"].iloc[k], str) else ""
            entry_idx = exit_idx = None
            if tm == "bmo" and t >= 1:
                entry_idx, exit_idx = t - 1, t + 5
            elif tm == "amc":
                entry_idx, exit_idx = t, t + 5
            if entry_idx is not None and exit_idx < len(close):
                ep, xp = close[entry_idx], close[exit_idx]
                if ep > 0 and np.isfinite(ep) and np.isfinite(xp):
                    f["entry_date"] = pd.Timestamp(dates[entry_idx])
                    f["exit_date"] = pd.Timestamp(dates[exit_idx])
                    f["pregap_return"] = float(xp / ep - 1.0)
            rows.append(f)
        if i % 100 == 0:
            print(f"  [{i}/{len(registry)}] events so far: {len(rows):,}")

    d = pd.DataFrame(rows)
    print(f"\nraw events: {len(d):,}  (membership-excluded: {stats['no_member']:,})")
    d = d[d.pregap_return.notna()].copy()
    with pd.HDFStore(DB_PROD, "r") as sp:
        def series(key, col):
            df = sp[key]
            df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
            return df.set_index("Date")[col].astype(float)

        vix = series("/macros/fred_vix_close", "vix_close")
        ff = series("/macros/fred_fed_funds_rate", "fed_funds_rate")
        un = series("/macros/fred_unemployment_rate", "unemployment_rate")
        # RC-16 F1: join each observation at its FIRST-RELEASE date (ALFRED
        # vintages), not its observation date. roc21 is computed on the macro
        # calendar BEFORE joining (the old code pct_change'd across event rows).
        rel = (sp["/macros/fred_release_dates"]
               if "/macros/fred_release_dates" in sp.keys() else None)

        def _avail_obs(s: pd.Series, series_id: str) -> pd.Series:
            """s indexed by obs date -> reindexed by first-release date."""
            if rel is None:
                return s
            r = rel[rel.series == series_id][["obs_date", "first_release"]].copy()
            r["obs_date"] = pd.to_datetime(r["obs_date"]).dt.normalize()
            m = s.to_frame("val").join(r.set_index("obs_date"), how="left")
            _idx_series = pd.Series(m.index, index=m.index)
            m["_avail"] = m["first_release"].fillna(_idx_series)
            return m.groupby("_avail")["val"].last().sort_index()

        vix_a = _avail_obs(vix, "VIXCLS")
        ff_a = _avail_obs(ff, "DFF")
        un_a = _avail_obs(un, "UNRATE")
        un_roc = un_a.pct_change(21).replace([np.inf, -np.inf], np.nan)
    rd = pd.to_datetime(d.report_date)
    d["vix"] = vix_a.reindex(rd, method="ffill").values
    d["fed_funds"] = ff_a.reindex(rd, method="ffill").values
    d["unemployment_roc21"] = un_roc.reindex(rd, method="ffill").values

    out_key = os.environ.get("RC16_SP600_OUT", "/features/train_matrix_sp600_pt")
    with pd.HDFStore(DB_SP600, "a") as store:
        if out_key in store.keys():
            store.remove(out_key)
        store.put(out_key, d, format="table",
                  data_columns=["permaTicker", "report_date"])
    print(f"wrote {out_key}: {len(d):,} events, "
          f"{d.permaTicker.nunique()} tickers, "
          f"{d.report_date.min().date()} .. {d.report_date.max().date()}")
    print(f"adv20 >= $10M share: {(d.adv20 >= 1e7).mean():.0%}")
    print(f"stats: {stats}")


if __name__ == "__main__":
    main()
