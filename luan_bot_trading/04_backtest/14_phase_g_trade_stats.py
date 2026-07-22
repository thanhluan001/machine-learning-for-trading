"""
Per-fold trade-level analysis at the HONEST deployable operating point
(theta=0.20, gap[-15%,-2%], n_slots=4, per-fold POS-tuned HP).

Mirrors the fold-iteration loop in `11_phase_g_bootstrap_ci.py` but
instead of bootstrapping, captures `trades_done` per fold and
emits:
  - cross-fold aggregate trade-level CSV (every trade, every fold)
  - per-fold win-rate / avg_win / avg_loss / etc.
  - cross-fold summary (allocated-weighted vs equal-weight per-trade PnL)

NO API calls. Reads only db.h5 + App D fold_results.csv.
Does NOT modify /features/* or any persisted pipeline tables.
"""
from __future__ import annotations
import sys, importlib.util, json, time
from pathlib import Path
import numpy as np
import pandas as pd

try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "tm", HERE.parent / "03_model" / "01_train_model.py")
tm = importlib.util.module_from_spec(spec); spec.loader.exec_module(tm)
pg_spec = importlib.util.spec_from_file_location(
    "pg", HERE.parent / "03_model" / "02_phase_g_sunday_classifier.py")
pg = importlib.util.module_from_spec(pg_spec); pg_spec.loader.exec_module(pg)
ps_spec = importlib.util.spec_from_file_location(
    "ps", HERE / "04_phase_g_portfolio.py")
ps = importlib.util.module_from_spec(ps_spec); ps_spec.loader.exec_module(ps)
rb_spec = importlib.util.spec_from_file_location(
    "rb", HERE / "_phase_g_random_baseline.py")
rb = importlib.util.module_from_spec(rb_spec); rb_spec.loader.exec_module(rb)

DB = tm.DB_FILE
SUNDAY_SAFE_FEATURES = pg.SUNDAY_SAFE_FEATURES

APPD_RESULTS_CSV = HERE / "archive" / "experiments" / "phase_g_v1_1_nested_cv_n4" / "fold_results.csv"

# Recommended operating point (App F / App G -- per Doc H baseline)
NEG_THETA = 0.20
NEG_GAP_LO = -0.15
NEG_GAP_HI = -0.02

DEFAULT_FOLDS = [
    ("2023-12-31", "2024-06-30", "2024-12-31"),
    ("2024-06-30", "2024-12-31", "2025-06-30"),
    ("2024-12-31", "2025-06-30", "2025-12-31"),
    ("2025-06-30", "2025-12-31", "2026-06-30"),
]

N_SLOTS = 4
INITIAL_NAV = 100000.0
OUT_DIR = HERE / "phase_g_v1_1_trade_stats_n4"


def fit_classifier(X_train, y_train, X_val, y_val, hp):
    import xgboost as xgb
    params = dict(
        objective="binary:logistic",
        eval_metric=["logloss", "auc"],
        n_estimators=hp["n_estimators"],
        learning_rate=0.05,
        max_depth=hp["max_depth"],
        min_child_weight=hp["min_child_weight"],
        gamma=hp["gamma"],
        reg_lambda=1.0,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42,
        n_jobs=-1,
    )
    clf = xgb.XGBClassifier(**params)
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return clf


def select_neg_picks(test_df, proba):
    """Apply theta + negative-gap criterion. Mirrors 11_phase_g_bootstrap_ci.py
    exactly: inclusive gap upper bound, requires path_pnl_t11_pct to not be NaN
    (drops T-match failures so they don't enter the slot simulation)."""
    df2 = test_df.copy()
    df2["pead_proba"] = proba
    mask = (
        (df2["pead_proba"] >= NEG_THETA) &
        (df2["opening_gap_t1"] >= NEG_GAP_LO) &
        (df2["opening_gap_t1"] <= NEG_GAP_HI) &
        (df2["path_pnl_t11_pct"].notna())
    )
    return df2[mask].copy().reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    print("=" * 78)
    print("PHASE G v1.1 -- TRADE-LEVEL STATS AT RECOMMENDED OP POINT")
    print(f"  Theta={NEG_THETA}, gap [{NEG_GAP_LO:+.2f}, {NEG_GAP_HI:+.2f}]")
    print(f"  n_slots={N_SLOTS}, folds={len(DEFAULT_FOLDS)}")
    print("=" * 78)

    # Load App D HP
    appd = pd.read_csv(APPD_RESULTS_CSV)
    fold_hp = []
    for _, row in appd.iterrows():
        fold_hp.append({
            "gamma": int(row["sel_gamma"]),
            "min_child_weight": int(row["sel_mcw"]),
            "max_depth": int(row["sel_md"]),
            "n_estimators": int(row["sel_n_est"]),
        })

    # Load train_matrix + apply cutoff
    df = tm.load_train_matrix()
    df = tm.apply_priming_cutoff(df, tm.PRIMING_RUNWAY_START)
    print(f"rows after §12 cut: {len(df)}")
    df = pg.v3.compute_pead_gates_full(df)
    df = pg.compute_entry_pnl(df)
    df = ps.compute_trade_paths(df)
    print(f"coverage: {int(df['path_pnl_t11_pct'].notna().sum())}/{len(df)}")

    # Precache trading calendar
    with pd.HDFStore(DB, mode="r") as sstore:
        all_dates = set()
        for key in sstore.keys():
            if not key.startswith("/sp400/"):
                continue
            d = sstore[key]
            all_dates.update(pd.to_datetime(d["Date"]).tolist())
    calendar = sorted(all_dates)
    print(f"calendar: {len(calendar)} trading days")

    fold_data = []
    for fold_idx, (train_end, sweep_end, test_end) in enumerate(DEFAULT_FOLDS, 1):
        print("\n" + "=" * 60)
        print(f"  FOLD {fold_idx}/{len(DEFAULT_FOLDS)}: TEST {sweep_end}->{test_end}")
        print("=" * 60)
        train_ts = pd.Timestamp(train_end)
        sweep_ts = pd.Timestamp(sweep_end)
        test_ts  = pd.Timestamp(test_end)
        rd = pd.to_datetime(df["report_date"])
        train_df = df[rd <= train_ts].copy().reset_index(drop=True)
        sweep_df = df[(rd > train_ts) & (rd <= sweep_ts)].copy().reset_index(drop=True)
        test_df  = df[(rd > sweep_ts) & (rd <= test_ts)].copy().reset_index(drop=True)
        print(f"  TRAIN={len(train_df)}  SWEEP={len(sweep_df)}  TEST={len(test_df)}")

        hp = fold_hp[fold_idx - 1]
        print(f"  HP (App D fold): gamma={hp['gamma']}, mcw={hp['min_child_weight']}, "
              f"md={hp['max_depth']}, n_est={hp['n_estimators']}")

        # Doc-D nested CV: train FINAL model on TRAIN+SWEEP, TEST is heldout OOS
        # The VAL set passed to xgboost for early-stopping is the TEST set
        # (same as bootstrap_ci.py). Refit-on-(train+sweep) canonical.
        feat_cols = SUNDAY_SAFE_FEATURES
        X_tr = train_df[feat_cols]; y_tr = train_df["pead_pass"].astype(int).values
        X_sv = sweep_df[feat_cols]; y_sv = sweep_df["pead_pass"].astype(int).values
        X_ts = pd.concat([X_tr, X_sv], axis=0).reset_index(drop=True)
        y_ts = np.concatenate([y_tr, y_sv])
        X_te = test_df[feat_cols]; y_te = (test_df["pead_pass"].astype(int).values
                if "pead_pass" in test_df.columns else np.zeros(len(test_df)))
        clf = fit_classifier(X_ts, y_ts, X_te, y_te, hp)
        proba_te = clf.predict_proba(X_te)[:, 1]
        picks = select_neg_picks(test_df, proba_te)
        print(f"  Picks after theta+gap filter: n={len(picks)}")

        # Simulate the portfolio over the test period
        result = rb._simulate_with_cached_calendar(
            picks, N_SLOTS, INITIAL_NAV, calendar)
        eq = result["equity_curve"]
        td = result["trades_done"]
        sumS = result["summary"]
        if td is None or len(td) == 0:
            print("  [!] no trades -- skipping")
            fold_data.append(None)
            continue
        fold_data.append({
            "fold": fold_idx,
            "test_slice": f"{sweep_end}->{test_end}",
            "hp": hp,
            "trades_done": td.copy(),
            "summary": sumS,
        })
        print(f"  n_trades={sumS.get('n_trades_executed', 0)}, "
              f"Sharpe={sumS.get('sharpe_liq_annualized', float('nan')):+.3f}, "
              f"IRR={sumS.get('irr_pct', float('nan')):+.2f}%, "
              f"MaxDD={sumS.get('max_drawdown_pct', float('nan')):+.2f}%, "
              f"hit_rate={sumS.get('hit_rate_pct', float('nan')):.2f}%, "
              f"avg_pnl={sumS.get('avg_trade_pnl_pct', float('nan')):+.3f}%")

    # === Aggregate trade-level stats ===
    print("\n" + "=" * 78)
    print("AGGREGATE TRADE-LEVEL STATS (across all folds)")
    print("=" * 78)
    all_trades = []
    for fd in fold_data:
        if fd is None:
            continue
        td = fd["trades_done"].copy()
        td["fold"] = fd["fold"]
        all_trades.append(td)
    if not all_trades:
        print("[ERR] No trades produced in any fold")
        return 2
    big = pd.concat(all_trades, ignore_index=True)
    OUT_DIR.mkdir(exist_ok=True)
    big.to_csv(OUT_DIR / "all_trades.csv", index=False)
    print(f"\n  all trades saved: {OUT_DIR / 'all_trades.csv'}  ({len(big)} rows)")

    pnls = pd.to_numeric(big["realized_arith_pct"], errors="coerce").dropna().to_numpy() * 100.0
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    zeros = pnls[pnls == 0]
    n = len(pnls)
    n_w = len(wins)
    n_l = len(losses)
    n_z = len(zeros)
    print(f"\n  Total trades (4-fold OOS):     {n}")
    print(f"  Wins (>0):                     {n_w} ({n_w/n*100:.1f}%)")
    print(f"  Losses (<0):                   {n_l} ({n_l/n*100:.1f}%)")
    print(f"  Flat (=0):                     {n_z} ({n_z/n*100:.1f}%)")
    print(f"\n  Avg win:    +{wins.mean():.2f}%   (median +{np.median(wins):.2f}%)")
    print(f"  Best win:   +{wins.max():.2f}%")
    print(f"\n  Avg loss:   {losses.mean():.2f}%   (median {np.median(losses):.2f}%)")
    print(f"  Worst loss: {losses.min():.2f}%")
    print(f"\n  Win/Loss ratio (avg_win / |avg_loss|):  {wins.mean() / abs(losses.mean()):.3f}")
    print(f"  Expectancy per trade (mean pnl):       {pnls.mean():+.2f}%")
    print(f"  Std of pnl per trade:                  {pnls.std():.2f}%")
    print(f"  T-test H0: mean=0   t-stat = {pnls.mean() / (pnls.std()/np.sqrt(n)):.3f}  "
          f"(df={n-1})")

    # Per-fold breakdown
    print("\n" + "=" * 78)
    print("PER-FOLD BREAKDOWN")
    print("=" * 78)
    pf_rows = []
    for fd in fold_data:
        if fd is None:
            continue
        td = fd["trades_done"]
        fold_pnls = pd.to_numeric(td["realized_arith_pct"], errors="coerce").dropna().to_numpy() * 100.0
        fw = fold_pnls[fold_pnls > 0]
        fl = fold_pnls[fold_pnls < 0]
        row = {
            "fold": fd["fold"],
            "test_slice": fd["test_slice"],
            "n_trades": len(fold_pnls),
            "n_wins": len(fw),
            "n_losses": len(fl),
            "hit_rate_pct": len(fw) / max(len(fold_pnls), 1) * 100.0,
            "avg_win_pct": fw.mean() if len(fw) else float('nan'),
            "avg_loss_pct": fl.mean() if len(fl) else float('nan'),
            "best_pct": fold_pnls.max() if len(fold_pnls) else float('nan'),
            "worst_pct": fold_pnls.min() if len(fold_pnls) else float('nan'),
            "mean_pct": fold_pnls.mean() if len(fold_pnls) else float('nan'),
            "sharpe": fd["summary"].get("sharpe_liq_annualized", float('nan')),
            "irr_pct": fd["summary"].get("irr_pct", float('nan')),
            "max_dd_pct": fd["summary"].get("max_drawdown_pct", float('nan')),
        }
        pf_rows.append(row)
    pf_df = pd.DataFrame(pf_rows)
    pf_df.to_csv(OUT_DIR / "per_fold_stats.csv", index=False)
    print("\n" + pf_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Cross-fold Sharpe sanity check (should match Doc H +1.31)
    cross_sharpe_mean = pf_df["sharpe"].mean()
    cross_sharpe_std = pf_df["sharpe"].std(ddof=1)
    print("\n" + "=" * 78)
    print(f"Cross-fold Sharpe: mean={cross_sharpe_mean:+.3f}  std={cross_sharpe_std:.3f}")
    print(f"  (Doc H baseline reference: +1.31, std 0.17)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
