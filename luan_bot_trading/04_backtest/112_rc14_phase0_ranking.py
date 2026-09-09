"""RC-14 Phase 0 — ranking diagnostics (pre-registered
rc14_ranking_pre_registration.md). No training beyond the fold-OOS gate
scoring (identical to 108/110). Diagnostics:

  a) weekly eligible-count distribution (capacity; eligible > 4 weeks)
  b) within-week Spearman(score, ret) among eligible
  c) trivial alternatives (g1-only, mean-of-gates) + top-1 vs all vs
     oracle top-1 weekly means

Kill gates per pre-registration, evaluated on DEV 1-3 + holdout pooled.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


v108 = load("v108", HERE / "108_rc13_v7_full_bar.py")
v104 = v108.v104
fr = v108.fr
bt = v108.bt

DB = bt.DB
MATRIX_KEY = "/features/train_matrix_combined"
FEATURES = v108.FEATURES
GATES = v108.GATES
XLF = v108.XLF

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def rets_for(c, cache):
    rows = []
    for r in c.itertuples(index=False):
        pl = v104.get_prices(r.permaTicker)
        if pl is None:
            continue
        dates, closes = pl
        e = int(np.searchsorted(dates, np.datetime64(r.entry_date), side="left"))
        x = int(np.searchsorted(dates, np.datetime64(pd.Timestamp(r.exit_date)), side="left"))
        if e >= len(closes) or x >= len(closes) or x <= e:
            continue
        ret = fr.ret_with_stop(closes, e, x, stop=v104.STOP)
        if ret is None:
            continue
        rows.append({"entry_date": r.entry_date, "ret_cost": ret - float(r.cost),
                     "score": r.score, "g1": r.p_pass_g1,
                     "gmean": np.mean([r.p_pass_g1, r.p_pass_g2, r.p_pass_g3]),
                     "score_noflag": r.score_noflag})
    return pd.DataFrame(rows)


def spear(x, y):
    if len(x) < 2:
        return np.nan
    rx = pd.Series(x).rank().values
    ry = pd.Series(y).rank().values
    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def diagnostics(df, label):
    print(f"\n[{label}] weeks={df.wk.nunique()}")
    counts = df.groupby("wk").size()
    print(f"  eligible/week: mean {counts.mean():.1f} median {counts.median():.0f} "
          f"max {counts.max()} | weeks eligible>4: {(counts > 4).mean():.0%} "
          f"(KILL1 if <=25%) | zero-eligible weeks excluded")
    sp = df.groupby("wk").apply(lambda g: spear(g.score.values, g.ret_cost.values)
                                if len(g) >= 2 else np.nan).dropna()
    sp_g1 = df.groupby("wk").apply(lambda g: spear(g.g1.values, g.ret_cost.values)
                                   if len(g) >= 2 else np.nan).dropna()
    sp_gm = df.groupby("wk").apply(lambda g: spear(g.gmean.values, g.ret_cost.values)
                                   if len(g) >= 2 else np.nan).dropna()
    print(f"  mean weekly Spearman: min-gate {sp.mean():+.3f} (n={len(sp)}) | "
          f"g1-only {sp_g1.mean():+.3f} (n={len(sp_g1)}) | "
          f"mean-gates {sp_gm.mean():+.3f} (n={len(sp_gm)})")
    # top-1 economics on weeks with >=2 eligible (competition weeks)
    top1 = df.sort_values(["wk", "score"], ascending=[True, False]).groupby("wk").head(1)
    top1_g1 = df.sort_values(["wk", "g1"], ascending=[True, False]).groupby("wk").head(1)
    top1_gm = df.sort_values(["wk", "gmean"], ascending=[True, False]).groupby("wk").head(1)
    oracle = df.sort_values(["wk", "ret_cost"], ascending=[True, False]).groupby("wk").head(1)
    print(f"  top-1 mean: min-gate {top1.ret_cost.mean():+.2%} | g1 {top1_g1.ret_cost.mean():+.2%} | "
          f"mean-gates {top1_gm.ret_cost.mean():+.2%}")
    print(f"  all-eligible mean {df.ret_cost.mean():+.2%} | oracle top-1 {oracle.ret_cost.mean():+.2%}")
    return counts, sp.mean()


def main() -> None:
    d = pd.read_hdf(DB, MATRIX_KEY)
    s6 = pd.read_hdf(v104.DB_SP600, "/features/train_matrix_sp600_pt",
                     columns=["permaTicker", "report_date", "adv20"])
    s6["report_date"] = pd.to_datetime(s6.report_date)
    d = d.merge(s6, on=["permaTicker", "report_date"], how="left")
    d["adv_pass"] = ~(d.adv20.notna() & (d.adv20 < 1e7))

    frames = []
    for fi, tr, sv, ts in v108.folds_for(d):
        train = pd.concat([tr, sv], ignore_index=True)
        for g in GATES:
            y = train[g].astype(int).to_numpy()
            yt = ts[g].astype(int).to_numpy()
            clf = v108.fit(train[FEATURES], y, ts[FEATURES], yt)
            ts[f"p_{g}"] = clf.predict_proba(ts[FEATURES])[:, 1]
            Xcf = ts[FEATURES].copy()
            Xcf["is_sp400"] = 0.0
            ts[f"p0_{g}"] = clf.predict_proba(Xcf)[:, 1]
        ts["fold"] = fi
        ts["score"] = ts[[f"p_{g}" for g in GATES]].min(axis=1)
        ts["score_noflag"] = ts[[f"p0_{g}" for g in GATES]].min(axis=1)
        frames.append(ts)
        print(f"  fold {fi} scored")
    oos = pd.concat(frames, ignore_index=True)
    v104._bulk_load()

    m = ((oos.score >= 0.33) & (~oos.sector.isin(XLF)) & oos.pregap_return.notna()
         & oos.adv_pass)
    elig = oos[m].copy()
    elig["entry_date"] = pd.to_datetime(elig.entry_date)
    df = rets_for(elig, v104)
    iso = df.entry_date.dt.isocalendar()
    df["wk"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)

    counts, sp = diagnostics(df, "ELIGIBLE @0.33 real score (current policy), DEV+HOLDOUT")

    # B-eligibility capacity only
    mB = m & ((oos.is_sp400 == 0) | (oos.score_noflag >= 0.33))
    eligB = oos[mB].copy()
    eligB["entry_date"] = pd.to_datetime(eligB.entry_date)
    dfB = rets_for(eligB, v104)
    iso = dfB.entry_date.dt.isocalendar()
    dfB["wk"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
    countsB = dfB.groupby("wk").size()
    print(f"\n[B-eligibility capacity] weeks={dfB.wk.nunique()} | "
          f"mean/week {countsB.mean():.1f} | eligible>4 weeks {(countsB > 4).mean():.0%}")

    # KILL gate evaluation (pre-registered)
    pct_comp = (counts > 4).mean()
    top1 = df.sort_values(["wk", "score"], ascending=[True, False]).groupby("wk").head(1)
    top1_g1 = df.sort_values(["wk", "g1"], ascending=[True, False]).groupby("wk").head(1)
    top1_gm = df.sort_values(["wk", "gmean"], ascending=[True, False]).groupby("wk").head(1)
    best_triv = max(top1_g1.ret_cost.mean(), top1_gm.ret_cost.mean())
    kill1 = pct_comp <= 0.25
    kill2 = (top1.ret_cost.mean() >= best_triv - 0.005) and (sp >= 0.05)
    print(f"\nKILL1 (eligible>4 <= 25%): {pct_comp:.0%} -> {'KILL' if kill1 else 'no'}")
    print(f"KILL2 (min-gate top-1 {top1.ret_cost.mean():+.2%} >= best-trivial {best_triv:+.2%} - 0.5pp "
          f"AND Spearman {sp:+.3f} >= 0.05): -> {'KILL' if kill2 else 'no'}")
    verdict = "CLOSED (ranking does not need a dedicated model)" if (kill1 or kill2) \
        else "PASS to Phase 1 (LambdaMART program)"
    print(f"\nRC-14 PHASE 0 VERDICT: {verdict}")


if __name__ == "__main__":
    main()
