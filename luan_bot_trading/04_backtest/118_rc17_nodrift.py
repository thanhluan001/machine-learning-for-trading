"""RC-17 — no-drift baseline gauntlet (pre-registered 2026-09-15).

Arms: v6n / v4n / v7n on the RC-16 corrected matrices with
car_drift_historical_q1 REMOVED from the feature set. Frozen HPs,
frozen thresholds (min-gate 0.33, V4 theta 0.20), label-mature folds,
10k week-block bootstrap seed 20260807, identical machinery and row-key
restriction as RC-16 R3 (direct comparability).

G4 reporting: paired week-block diffs vs RC-16's v6c/v4c/v7c executed
ledgers (archive/experiments/rc16_r3/executed.h5).
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.chdir(Path(__file__).resolve().parents[2])
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "archive" / "experiments" / "rc17_nodrift"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RC16_EXEC = HERE / "archive" / "experiments" / "rc16_r3" / "executed.h5"

N_BOOT = 10_000
SEED = 20260807
THRESH = 0.33
V4_THETA = 0.20
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


bt = load("bt_rc17", HERE / "51_hp_theta_sweep_23feat.py")
DB = bt.DB
FEATURES22 = [f for f in bt.DEPLOY_FEATURES if f != "car_drift_historical_q1"]
FEATURES_V7 = FEATURES22 + ["is_sp400"]

OLD_KEY = "/features/train_matrix_v4_timing_correct"
V6C_KEY = "/features/train_matrix_v6c"
V7C_KEY = "/features/train_matrix_v7c"
SP600_KEY = "/features/train_matrix_sp600_pt_v7c"


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


def _exec_from_raw(raw):
    raw = raw.copy()
    raw["p"] = raw["score"]
    raw["entry_date"] = pd.to_datetime(raw["pregap_entry_date"])
    raw["exit_date"] = pd.to_datetime(raw["pregap_exit_date"])
    pieces = [bt.select_weekly(raw[raw["fold"] == fi], bt.N_SLOTS) for fi in range(1, 5)
              if not raw[raw["fold"] == fi].empty]
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def run_gates(df, hp_map, feats, split_key):
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
    return pred, raw, _exec_from_raw(raw)


def run_single(df, theta, feats, split_key):
    out = []
    for fi, tr, sv, ts in folds_for(df, split_key):
        train = pd.concat([tr, sv], ignore_index=True)
        clf = fit_gate(train[feats], train["pead_pass"].astype(int).to_numpy(),
                       ts[feats], ts["pead_pass"].astype(int).to_numpy(), V4_HP)
        ts["p"] = clf.predict_proba(ts[feats])[:, 1]
        ts["score"] = ts["p"]
        ts["fold"] = fi
        out.append(ts)
    pred = pd.concat(out, ignore_index=True)
    mask = (pred["p"] >= theta) & (~pred["sector"].isin(bt.EXCLUDE_SECTORS))
    raw = pred[mask & pred["pregap_return"].notna()].copy()
    return pred, raw, _exec_from_raw(raw)


def run_v7n(df, adv):
    d = df.copy()
    d = d.merge(adv, on=["permaTicker", "report_date"], how="left")
    d["adv_pass"] = ~(d["adv20"].notna() & (d["adv20"] < 1e7))
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
    raw["p"] = raw["score"]
    raw["entry_date"] = pd.to_datetime(raw["entry_date"])
    raw["exit_date"] = pd.to_datetime(raw["exit_date"])
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


def boot_weekly_mean(w, rng):
    v = w.to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return {"mean_pct": None, "ci95_pct": [None, None]}
    idx = rng.integers(0, len(v), size=(N_BOOT, len(v)))
    means = v[idx].mean(axis=1)
    return {"mean_pct": round(float(v.mean() * 100), 4),
            "ci95_pct": [round(float(np.percentile(means, 2.5) * 100), 4),
                         round(float(np.percentile(means, 97.5) * 100), 4)]}


def paired_week_diff(wa, wb, rng):
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


def main() -> int:
    rng = np.random.default_rng(SEED)
    print("=" * 100)
    print("RC-17 NO-DRIFT GAUNTLET — 22 features, frozen HPs/thresholds, pre-registered")
    print("=" * 100)
    print(f"features: {len(FEATURES22)} (removed car_drift_historical_q1)")

    v6c = pd.read_hdf(DB, V6C_KEY)
    v7c = pd.read_hdf(DB, V7C_KEY)
    old = pd.read_hdf(DB, OLD_KEY)
    for d in (old, v6c):
        d["report_date"] = pd.to_datetime(d["report_date"])
    ko = set(zip(old.permaTicker, old.report_date))
    kc = set(zip(v6c.permaTicker, v6c.report_date))
    common = ko & kc
    v6c_c = v6c[v6c.set_index(["permaTicker", "report_date"]).index.isin(common)].copy()
    print(f"common keys: {len(common):,}")

    runs = {}

    def record(name, raw, ex):
        w = weekly_returns(ex)
        runs[name] = {"stats": stats(ex), "boot": boot_weekly_mean(w, rng),
                      "weekly": w, "exec": ex}
        print(f"--- {name} ---")
        print(json.dumps(runs[name]["stats"], default=str))
        print("week-block CI:", json.dumps(runs[name]["boot"]))

    _, _, ex = run_gates(v6c_c, V6_HP, FEATURES22, "label_end")
    record("v6n", None, ex)
    _, _, ex = run_single(v6c_c, V4_THETA, FEATURES22, "label_end")
    record("v4n", None, ex)

    s6 = pd.read_hdf(HERE.parent / "01_data" / "db_sp600.h5", SP600_KEY)
    adv400 = (v6c_c[["permaTicker", "report_date", "adv20"]]
              if "adv20" in v6c_c.columns else None)
    adv600 = s6[["permaTicker", "report_date", "adv20"]]
    adv = pd.concat([x for x in (adv400, adv600) if x is not None], ignore_index=True)
    _, _, ex = run_v7n(v7c, adv)
    record("v7n", None, ex)

    # G4: paired vs RC-16 c-companions
    g4 = {}
    with pd.HDFStore(RC16_EXEC, "r") as st:
        for new_arm, old_arm in (("v6n", "v6c"), ("v4n", "v4c"), ("v7n", "v7c")):
            key = f"/{old_arm}"
            if key in st.keys():
                wold = weekly_returns(st[key])
                g4[f"{new_arm}_minus_{old_arm}"] = paired_week_diff(
                    runs[new_arm]["weekly"], wold, rng)

    gates = {}
    for arm, g in (("v6n", "G1"), ("v4n", "G2"), ("v7n", "G3")):
        st_ = runs[arm]["stats"]
        gates[g] = {"arm": arm, "pass": bool(st_.get("avg_trade_pct", 0) > 0
                                             and ci_excludes_zero(runs[arm]["boot"]["ci95_pct"])),
                    "boot": runs[arm]["boot"]}
        print(f"{g} ({arm}): {'PASS' if gates[g]['pass'] else 'FAIL'}")

    verdict = {"gates": gates, "G4_vs_rc16_c": g4}
    print(json.dumps(verdict, indent=2, default=str))

    payload = {"runs": {k: {kk: vv for kk, vv in v.items() if kk not in ("weekly", "exec")}
                        for k, v in runs.items()},
               "verdict": verdict,
               "config": {"n_boot": N_BOOT, "seed": SEED, "thresh": THRESH,
                          "v4_theta": V4_THETA, "features": len(FEATURES22),
                          "removed": "car_drift_historical_q1",
                          "common_keys": len(common)}}
    with open(OUT_DIR / "gauntlet.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)
    with pd.HDFStore(OUT_DIR / "executed.h5", "w") as st:
        for k, v in runs.items():
            if len(v["exec"]):
                keep = [c for c in ("permaTicker", "fold", "entry_date", "pregap_exit_date",
                                    "pregap_return", "score", "pead_pass", "is_sp400") if c in v["exec"].columns]
                st.put(f"/{k}", v["exec"][keep], format="table")
    print(f"\nwrote {OUT_DIR / 'gauntlet.json'} and executed.h5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
