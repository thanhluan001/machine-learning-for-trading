"""RC-16 R3 — the pre-registered revalidation gauntlet.

Registered comparisons ONLY (rc16_information_integrity_pre_registration.md §3):
  (a) v6c vs frozen-V6 mechanics on identical folds   (contract-fix effect)
  (b) v6c vs V4c                                        (decomposition survival)
  (c) v7c vs v6c                                        (universe expansion)

Gates (no thresholds tuned here; 10k resamples, seed 20260807, same standard
as the frozen-V6 validation scripts):
  G1  v6c mean trade > 0 AND week-block bootstrap CI excludes 0
  G2  v6c > V4c on mean trade AND paired week-block CI of the difference
      excludes 0
  G3  v7c satisfies the G1/G2 analogues
  G4  one-factor (F1/F2/F3/F4) decomposition of the v6c-vs-baseline delta —
      in-memory column/split swaps on the OLD matrix restricted to the
      common row keys, so run differences are contract effects only.
      REPORTING REQUIREMENT, not a selection step.

All runs use FROZEN HPs and FROZEN thresholds (V6 trio HPs, V4 uniform HP,
min-gate 0.33, V4 theta 0.20). Simulator = matrix pregap_return (stop embedded)
for every arm; v7c uses its own entry/exit/pregap_return with side costs in
labels (no resim-with-stop — consistent across arms, noted in the report).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os_target = Path(__file__).resolve().parents[2]
import os

os.chdir(os_target)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "archive" / "experiments" / "rc16_r3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_BOOT = 10_000
SEED = 20260807
THRESH = 0.33          # frozen V6/V7 min-gate
V4_THETA = 0.20        # frozen V4 single-model threshold
GATES = ["pass_g1", "pass_g2", "pass_g3"]
V6_HP = {
    "pass_g1": {"gamma": 8, "min_child_weight": 20, "max_depth": 3},
    "pass_g2": {"gamma": 12, "min_child_weight": 50, "max_depth": 3},
    "pass_g3": {"gamma": 1, "min_child_weight": 50, "max_depth": 3},
}
V4_HP = {"gamma": 3, "min_child_weight": 100, "max_depth": 2}
N_TREES = 300


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bt = load("bt_rc16r3", HERE / "51_hp_theta_sweep_23feat.py")
DB = bt.DB
FEATURES = list(bt.DEPLOY_FEATURES)
FEATURES_V7 = FEATURES + ["is_sp400"]

OLD_KEY = "/features/train_matrix_v4_timing_correct"
V6C_KEY = "/features/train_matrix_v6c"
V7C_KEY = "/features/train_matrix_v7c"
SP600_KEY = "/features/train_matrix_sp600_pt_v7c"


# ----------------------------------------------------------------------------
# machinery (mirrors 53/55 semantics exactly)
# ----------------------------------------------------------------------------
def fit_gate(X, y, Xe, ye, hp):
    import xgboost as xgb
    return xgb.XGBClassifier(
        objective="binary:logistic", eval_metric=["logloss", "auc"],
        n_estimators=N_TREES, learning_rate=0.05,
        max_depth=hp["max_depth"], min_child_weight=hp["min_child_weight"],
        gamma=hp["gamma"], reg_lambda=1.0, subsample=0.7,
        colsample_bytree=0.7, random_state=42, n_jobs=-1,
    ).fit(X, y, eval_set=[(Xe, ye)], verbose=False)


def folds_for(df, split_key: str):
    rd = (df["label_end"] if split_key == "label_end" and "label_end" in df.columns
          else pd.to_datetime(df["report_date"]))
    for fi, (te, sve, tse) in enumerate(bt.DEFAULT_FOLDS, 1):
        tr = df[rd <= pd.Timestamp(te)]
        sv = df[(rd > pd.Timestamp(te)) & (rd <= pd.Timestamp(sve))]
        ts = df[(rd > pd.Timestamp(sve)) & (rd <= pd.Timestamp(tse))].copy()
        yield fi, tr, sv, ts


def run_gates(df, hp_map, feats, split_key):
    """Frozen-HP gate trio walk-forward -> executed trades (min-gate policy)."""
    out = []
    for fi, tr, sv, ts in folds_for(df, split_key):
        train = pd.concat([tr, sv], ignore_index=True)
        for i, g in enumerate(GATES, 1):
            clf = fit_gate(train[feats], train[g].astype(int).to_numpy(),
                           ts[feats], ts[g].astype(int).to_numpy(), hp_map[g])
            ts[f"p{i}"] = clf.predict_proba(ts[feats])[:, 1]
        ts["score"] = ts[["p1", "p2", "p3"]].min(axis=1)
        ts["fold"] = fi
        out.append(ts)
    pred = pd.concat(out, ignore_index=True)
    mask = (pred["score"] >= THRESH) & (~pred["sector"].isin(bt.EXCLUDE_SECTORS))
    raw = pred[mask & pred["pregap_return"].notna()].copy()
    raw["p"] = raw["score"]                      # select_weekly ranks on `p`
    raw["entry_date"] = pd.to_datetime(raw["pregap_entry_date"])
    raw["exit_date"] = pd.to_datetime(raw["pregap_exit_date"])
    pieces = [bt.select_weekly(raw[raw["fold"] == fi], bt.N_SLOTS) for fi in range(1, 5)
              if not raw[raw["fold"] == fi].empty]
    ex = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    return pred, raw, ex


def run_single(df, theta, split_key):
    """V4-style single pead_pass classifier walk-forward -> executed trades."""
    out = []
    for fi, tr, sv, ts in folds_for(df, split_key):
        train = pd.concat([tr, sv], ignore_index=True)
        clf = fit_gate(train[FEATURES], train["pead_pass"].astype(int).to_numpy(),
                       ts[FEATURES], ts["pead_pass"].astype(int).to_numpy(), V4_HP)
        ts["p"] = clf.predict_proba(ts[FEATURES])[:, 1]
        ts["fold"] = fi
        out.append(ts)
    pred = pd.concat(out, ignore_index=True)
    mask = (pred["p"] >= theta) & (~pred["sector"].isin(bt.EXCLUDE_SECTORS))
    raw = pred[mask & pred["pregap_return"].notna()].copy()
    raw["entry_date"] = pd.to_datetime(raw["pregap_entry_date"])
    raw["exit_date"] = pd.to_datetime(raw["pregap_exit_date"])
    pieces = [bt.select_weekly(raw[raw["fold"] == fi], bt.N_SLOTS) for fi in range(1, 5)
              if not raw[raw["fold"] == fi].empty]
    ex = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    return pred, raw, ex


def weekly_returns(ex):
    if isinstance(ex, pd.DataFrame) and ex.empty:
        return pd.Series(dtype=float)
    z = ex.copy()
    z["entry_date"] = pd.to_datetime(z["entry_date"])
    iso = z["entry_date"].dt.isocalendar()
    z["week"] = iso.year.astype(str) + "-W" + iso.week.astype(str).str.zfill(2)
    return z.groupby("week", sort=True)["pregap_return"].sum() / bt.N_SLOTS


def stats(ex):
    if ex is None or len(ex) == 0:
        return {"executed": 0}
    r = ex["pregap_return"].astype(float)
    w = weekly_returns(ex)
    nav = float((1 + w).prod() - 1) if len(w) else 0.0
    return {"executed": int(len(r)), "win_rate_pct": round(float((r > 0).mean() * 100), 1),
            "avg_trade_pct": round(float(r.mean() * 100), 3), "weeks": int(len(w)),
            "weekly_mean_pct": round(float(w.mean() * 100), 3) if len(w) else None,
            "nav_pct": round(nav * 100, 2),
            "raw_picks_precision": round(float(ex["pead_pass"].mean() * 100), 1)}


def boot_weekly_mean(w: pd.Series, rng):
    v = w.to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return {"mean_pct": None, "ci95_pct": [None, None]}
    idx = rng.integers(0, len(v), size=(N_BOOT, len(v)))
    means = v[idx].mean(axis=1)
    return {"mean_pct": round(float(v.mean() * 100), 4),
            "ci95_pct": [round(float(np.percentile(means, 2.5) * 100), 4),
                         round(float(np.percentile(means, 97.5) * 100), 4)]}


def paired_week_diff(wa: pd.Series, wb: pd.Series, rng):
    j = pd.concat({"a": wa, "b": wb}, axis=1, join="inner").dropna()
    d = (j["a"] - j["b"]).to_numpy(dtype=float)
    if not len(d):
        return {"paired_weeks": 0, "mean_diff_pct": None, "ci95_pct": [None, None]}
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    means = d[idx].mean(axis=1)
    return {"paired_weeks": int(len(d)), "mean_diff_pct": round(float(d.mean() * 100), 4),
            "ci95_pct": [round(float(np.percentile(means, 2.5) * 100), 4),
                         round(float(np.percentile(means, 97.5) * 100), 4)],
            "prob_pos": round(float((d > 0).mean()), 4)}


def ci_excludes_zero(ci):
    return ci is not None and ci[0] is not None and (ci[0] > 0 or ci[1] < 0)


# ----------------------------------------------------------------------------
# v7c arm (own row set; uniform-HP gates; entry/exit from the matrix)
# ----------------------------------------------------------------------------
def run_v7c(df, adv):
    d = df.copy()
    if adv is not None:
        d = d.merge(adv, on=["permaTicker", "report_date"], how="left")
        d["adv_pass"] = ~(d["adv20"].notna() & (d["adv20"] < 1e7))
    else:
        d["adv_pass"] = True
    out = []
    for fi, tr, sv, ts in folds_for(d, "label_end"):
        train = pd.concat([tr, sv], ignore_index=True)
        for i, g in enumerate(GATES, 1):
            clf = fit_gate(train[FEATURES_V7], train[g].astype(int).to_numpy(),
                           ts[FEATURES_V7], ts[g].astype(int).to_numpy(), V4_HP)
            ts[f"p{i}"] = clf.predict_proba(ts[FEATURES_V7])[:, 1]
        ts["score"] = ts[["p1", "p2", "p3"]].min(axis=1)
        ts["fold"] = fi
        out.append(ts)
    pred = pd.concat(out, ignore_index=True)
    mask = ((pred["score"] >= THRESH) & (~pred["sector"].isin(bt.EXCLUDE_SECTORS))
            & pred["pregap_return"].notna() & pred["adv_pass"])
    raw = pred[mask].copy()
    raw["p"] = raw["score"]                      # select_weekly ranks on `p`
    raw["entry_date"] = pd.to_datetime(raw["entry_date"])
    raw["exit_date"] = pd.to_datetime(raw["exit_date"])
    pieces = [bt.select_weekly(raw[raw["fold"] == fi], bt.N_SLOTS) for fi in range(1, 5)
              if not raw[raw["fold"] == fi].empty]
    ex = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    return pred, raw, ex


# ----------------------------------------------------------------------------
def main() -> int:
    rng = np.random.default_rng(SEED)
    print("=" * 100)
    print("RC-16 R3 GAUNTLET — pre-registered comparisons, frozen HPs/thresholds")
    print("=" * 100)

    old = pd.read_hdf(DB, OLD_KEY)
    v6c = pd.read_hdf(DB, V6C_KEY)
    v7c = pd.read_hdf(DB, V7C_KEY)
    for d in (old, v6c):
        d["report_date"] = pd.to_datetime(d["report_date"])

    # ---- common row keys for the V6 family (run diffs = contract only) ----
    ko = set(zip(old.permaTicker, old.report_date))
    kc = set(zip(v6c.permaTicker, v6c.report_date))
    common = ko & kc
    old_c = old[old.set_index(["permaTicker", "report_date"]).index.isin(common)].copy()
    v6c_c = v6c[v6c.set_index(["permaTicker", "report_date"]).index.isin(common)].copy()
    print(f"common keys: {len(common):,} (old {len(old):,} / v6c {len(v6c):,})")

    # label_end for OLD matrix rows: exact T+11 sessions on the IJH calendar
    ijh = pd.read_hdf(DB, "/macros/IJH")
    ijh_dates = pd.to_datetime(pd.Series(ijh.index if ijh.index.dtype.kind == "M"
                                         else ijh["Date"])).sort_values().to_numpy()
    tpos = np.searchsorted(ijh_dates, pd.to_datetime(old_c["T"]).to_numpy(), side="left")
    le = np.array([ijh_dates[p + 11] if p + 11 < len(ijh_dates) else np.datetime64("NaT")
                   for p in tpos])
    old_c["label_end"] = pd.to_datetime(le)

    # corrected columns for the one-factor arms (join from v6c)
    swap_cols = ["vix", "fed_funds", "unemployment_roc21", "car_drift_historical_q1"]
    v6c_cols = v6c_c[["permaTicker", "report_date"] + swap_cols]
    f1_arm = old_c.drop(columns=["vix", "fed_funds", "unemployment_roc21"]).merge(
        v6c_cols[["permaTicker", "report_date", "vix", "fed_funds", "unemployment_roc21"]],
        on=["permaTicker", "report_date"], how="left")
    f2_arm = old_c.drop(columns=["car_drift_historical_q1"]).merge(
        v6c_cols[["permaTicker", "report_date", "car_drift_historical_q1"]],
        on=["permaTicker", "report_date"], how="left")
    f4_arm = old_c[old_c["car_10d"].notna() & old_c["inst_vol_ratio"].notna()
                   & old_c["maxdd_ma"].notna()].copy()
    print(f"F4-only: dropped {len(old_c) - len(f4_arm)} unlabelled rows of {len(old_c)}")

    runs = {}

    def gate_run(name, df, hp_map, split_key, feats=None):
        print(f"\n--- {name} ({split_key}) ---", flush=True)
        pred, raw, ex = run_gates(df, hp_map, feats or FEATURES, split_key)
        w = weekly_returns(ex)
        runs[name] = {"stats": stats(ex), "boot": boot_weekly_mean(w, rng),
                      "weekly": w, "exec": ex}
        print(json.dumps(runs[name]["stats"], default=str))
        print("week-block CI:", json.dumps(runs[name]["boot"]))
        return pred, raw, ex

    # (a) contract-fix effect + G4 one-factor arms
    gate_run("base_v6", old_c, V6_HP, "report")          # frozen-V6 mechanics
    gate_run("f1_only", f1_arm, V6_HP, "report")
    gate_run("f2_only", f2_arm, V6_HP, "report")
    gate_run("f3_only", old_c, V6_HP, "label_end")
    gate_run("f4_only", f4_arm, V6_HP, "report")
    gate_run("v6c", v6c_c, V6_HP, "label_end")

    # (b) V4c comparator (same corrected matrix, same splits)
    _, _, ex_v4c = run_single(v6c_c, V4_THETA, "label_end")
    runs["v4c"] = {"stats": stats(ex_v4c), "boot": boot_weekly_mean(weekly_returns(ex_v4c), rng),
                   "weekly": weekly_returns(ex_v4c), "exec": ex_v4c}
    print("\n--- v4c ---"); print(json.dumps(runs["v4c"]["stats"], default=str))
    print("week-block CI:", json.dumps(runs["v4c"]["boot"]))

    _, _, ex_v4b = run_single(old_c, V4_THETA, "report")
    runs["v4_base"] = {"stats": stats(ex_v4b)}
    print("\n--- v4_base ---"); print(json.dumps(runs["v4_base"]["stats"], default=str))

    # (c) v7c
    s6 = pd.read_hdf(DB_SP600 := (HERE.parent / "01_data" / "db_sp600.h5"), SP600_KEY)
    adv400 = v6c_c[["permaTicker", "report_date", "adv20"]] if "adv20" in v6c_c.columns else None
    adv600 = s6[["permaTicker", "report_date", "adv20"]]
    adv = pd.concat([x for x in (adv400, adv600) if x is not None], ignore_index=True)
    print(f"\n--- v7c ---")
    _, _, ex_v7 = run_v7c(v7c, adv)
    runs["v7c"] = {"stats": stats(ex_v7), "boot": boot_weekly_mean(weekly_returns(ex_v7), rng),
                   "weekly": weekly_returns(ex_v7), "exec": ex_v7}
    print(json.dumps(runs["v7c"]["stats"], default=str))
    print("week-block CI:", json.dumps(runs["v7c"]["boot"]))

    # ---- gates ----
    print("\n" + "=" * 100)
    g1 = (runs["v6c"]["stats"].get("avg_trade_pct", 0) > 0
          and ci_excludes_zero(runs["v6c"]["boot"]["ci95_pct"]))
    d_g2 = paired_week_diff(runs["v6c"]["weekly"], runs["v4c"]["weekly"], rng)
    g2 = (runs["v6c"]["stats"]["avg_trade_pct"] > runs["v4c"]["stats"]["avg_trade_pct"]
          and ci_excludes_zero(d_g2["ci95_pct"]))
    d_g3 = paired_week_diff(runs["v7c"]["weekly"], runs["v6c"]["weekly"], rng)
    g3a = (runs["v7c"]["stats"].get("avg_trade_pct", 0) > 0
           and ci_excludes_zero(runs["v7c"]["boot"]["ci95_pct"]))
    g3b = (d_g3["mean_diff_pct"] is not None
           and runs["v7c"]["stats"]["avg_trade_pct"] > runs["v6c"]["stats"]["avg_trade_pct"]
           and ci_excludes_zero(d_g3["ci95_pct"]))

    # G4: one-factor attribution (reporting only)
    base_avg = runs["base_v6"]["stats"]["avg_trade_pct"]
    full_avg = runs["v6c"]["stats"]["avg_trade_pct"]
    g4 = {"baseline_avg_pct": base_avg, "v6c_avg_pct": full_avg,
          "total_delta_pp": round(full_avg - base_avg, 3),
          "one_factor": {}}
    for arm in ("f1_only", "f2_only", "f3_only", "f4_only"):
        a = runs[arm]["stats"]["avg_trade_pct"]
        g4["one_factor"][arm] = {"avg_pct": a, "delta_vs_base_pp": round(a - base_avg, 3),
                                 "share_of_total_pct": round((a - base_avg) / (full_avg - base_avg) * 100, 1)
                                 if abs(full_avg - base_avg) > 1e-9 else None}

    verdict = {
        "G1_contract_survival": {"pass": bool(g1),
                                 "v6c_boot": runs["v6c"]["boot"]},
        "G2_decomposition": {"pass": bool(g2), "paired_weekly_v6c_minus_v4c": d_g2},
        "G3_universe": {"pass": bool(g3a and g3b), "g1_analogue_pass": bool(g3a),
                        "g2_analogue_pass": bool(g3b),
                        "paired_weekly_v7c_minus_v6c": d_g3},
        "G4_damage_attribution": g4,
    }
    print(json.dumps(verdict, indent=2, default=str))

    payload = {"runs": {k: {kk: vv for kk, vv in v.items() if kk not in ("weekly", "exec")}
                        for k, v in runs.items()},
               "verdict": verdict,
               "config": {"n_boot": N_BOOT, "seed": SEED, "thresh": THRESH,
                          "v4_theta": V4_THETA, "v6_hp": V6_HP, "v4_hp": V4_HP,
                          "common_keys": len(common),
                          "f4_unlabelled_old": int(len(old_c) - len(f4_arm))}}
    with open(OUT_DIR / "gauntlet.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)
    # executed trades ledger for later audits
    with pd.HDFStore(OUT_DIR / "executed.h5", "w") as st:
        for k, v in runs.items():
            if "exec" in v and isinstance(v["exec"], pd.DataFrame) and len(v["exec"]):
                keep = [c for c in ("permaTicker", "fold", "entry_date", "pregap_exit_date",
                                    "pregap_return", "score", "pead_pass", "is_sp400") if c in v["exec"].columns]
                st.put(f"/{k}", v["exec"][keep], format="table")
    print(f"\nwrote {OUT_DIR / 'gauntlet.json'} and executed.h5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
