#!/usr/bin/env python3
"""
Stage 1: Gated Events + Weekly Distribution Profile  (DESIGN / SKELETON ONLY)
================================================================================

STATUS: DESIGN ONLY — implementation deferred until plan is approved.
        This file currently contains the full functional spec, algorithm, and
        function signatures with docstrings; the bodies are NOT yet
        implemented. After approval, fill in the bodies per the docstrings
        and the pseudocode blocks.

PURPOSE
-------
Before building the listwise-ranker feature matrix (Stage 2:
`02_features/build_feature_matrix.py`), we need to know the **shape** of the
eligible-event population along the LTR group axis: `calendar_week_group`
(ISO calendar week of `report_date`, format `YYYY-Www` — see `Design.md` §17
and `features.md` §0).

The listwise objective `rank:ndcg` with `eval_metric="ndcg@3"` only produces
useful gradients when weekly cohort groups are large enough to support
top-of-funnel ranking comparisons. The exact cohort distribution is unknown
because raw `/earnings/raw` (44,637 rows) contains events that MUST be pruned
before they can become ranker rows:

  * earnings dated BEFORE the company's `added` date (pre-addition quarters);
  * earnings dated INSIDE the 90-day stabilization buffer (the "first quarter
    post-addition" — already excluded by the buffer expansion; no separate
    per-event skip is required);
  * earnings dated AFTER a `removed` date (delistings / removals);
  * earnings from `price_unavailable=True` companies (no `/sp400/{canonical}`
    price node exists for them — they can never produce feature rows).

This script gates those out and emits two things:

  1. A persistent elites table at `/features/gated_events` in `db.h5` —
     the row-pool Stage 2 will expand into a full feature matrix.
  2. A weekly-distribution profile printed to stdout — used to decide:
       (a) the ranker's `ndcg@k` (do cohorts reliably reach size k?)
       (b) whether a sparse-week strategy (merge adjacent weeks vs. drop
           sparse weeks) is necessary.

WHAT THIS SCRIPT DOES (inputs and outputs)
------------------------------------------
INPUTS (from `01_data/db.h5`; READ-ONLY — no external API calls):
  * `/metadata/sp400_companies`  — columns:
        canonical_ticker, cik, aliases (JSON), name, sic, index_ref,
        combined_intervals (JSON list of {"added", "removed"} dicts),
        per_ticker_intervals (JSON audit), price_unavailable (bool).
  * `/earnings/raw`              — columns:
        report_date, fiscal_period_end, code, canonical_ticker, cik,
        actual, estimate, difference, percent, before_after_market, currency.

OUTPUTS:
  * `/features/gated_events` in `01_data/db.h5` (HDF5 table,
    `HDFStore(mode='a')`, overwrite via `store.remove` if the key already
    exists — NEVER `mode='w'` on the whole DB; see STOP_DOING_EXTRA_SHIT.md
    and the team's HDF5 safety convention).
    Columns (exactly six):
        canonical_ticker : str                    — joins back to sp400_companies
        cik              : str                    — SEC CIK anchor (audit / dedup)
        report_date      : datetime64[us]         — the earnings announcement date
        added            : datetime64[us]         — interval.added (audit only)
        removed          : datetime64[us]         — interval.removed or today (audit only)
        calendar_week_group : str                — ISO week "YYYY-Www" (LTR group key)

  * stdout profile (no disk):
        - total eligible events
        - min / max report_date (date range of gated universe)
        - per-week summary stats (count / mean / median / std / min / max)
        - week-size histogram buckets  (<3, 3-5, 6-10, 11-20, 21-50, 51+)
        - #weeks with <3 events   (ndcg@3 feasibility flag)
        - #weeks with <5 events   (ndcg@5 feasibility flag)
        - #weeks with <10 events  (ndcg@10 feasibility flag)

WHAT THIS SCRIPT DOES NOT DO (deferred to Stage 2)
-------------------------------------------------
  * No feature computation (no `sue_score`, no Block 2-5 features).
  * No `/sp400/{canonical}` price-series loading.
  * No `T` (matched trading day) computation.
  * No `car_10d` (the LTR target) — requires T+1..T+11 abnormal return window.
  * No sector ETF or macro joins.
  * No model code.

ALGORITHM (locked; bodies populated after approval)
---------------------------------------------------
    today = pd.Timestamp.now().normalize()      # `removed = null` -> today
    companies = load_sp400_companies()           # /metadata/sp400_companies
    earnings   = load_earnings_raw()             # /earnings/raw
    gated_rows = []

    for company in companies:
        if company.price_unavailable:
            continue                              # zero-row contributors

        intervals   = parse_json(company.combined_intervals)
        comp_earn   = earnings[earnings.canonical_ticker == company.canonical_ticker]

        for itv in intervals:
            added_dt   = pd.to_datetime(itv['added'])
            removed_dt = pd.to_datetime(itv['removed']) if itv['removed'] else today
            buf_start  = added_dt + pd.Timedelta(days=90)     # 90d = 1Q buffer

            mask = (comp_earn.report_date >= buf_start) & (comp_earn.report_date <= removed_dt)
            for ev_date in comp_earn[mask].report_date:
                iso = ev_date.isocalendar()
                gated_rows.append({
                    canonical_ticker: company.canonical_ticker,
                    cik:              company.cik,
                    report_date:      ev_date,
                    added:            added_dt,
                    removed:          removed_dt,
                    calendar_week_group: f"{iso.year}-W{iso.week:02d}",
                })

    gated_df = DataFrame(gated_rows).sort_values(['calendar_week_group', 'canonical_ticker'])

    write_gated_events(gated_df)        # /features/gated_events in db.h5
    profile = profile_weekly_distribution(gated_df)
    print_profile(profile)

KEY DESIGN DECISIONS (already locked in prior thread)
----------------------------------------------------
  * `combined_intervals` is already backfilled to 2012-01-01 for all tickers
    (including removed constituents) — no NaT handling needed for `added`.
  * 90-day buffer = first-quarter exclusion. Applied as window start
    `[added + 90d, removed]`. NO ADDITIONAL per-event "skip first in window"
    rule. (See features.md §0.)
  * `price_unavailable=True` companies produce ZERO gated events.
  * Feature lookback MAY cross interval boundaries (irrelevant to Stage 1,
    relevant to Stage 2 only).
  * `calendar_week_group` uses `isocalendar()` — ISO calendar week
    (format `YYYY-Www`). NOT a custom trading-week cluster on T+1 alignment.
    This matches the explicit `2026_W27` example in Design.md §17.
  * If an event falls in MULTIPLE overlapping intervals of the same company,
    it is emitted once per qualifying interval (audit trail). In practice
    `combined_intervals` are merged so overlaps should not occur
    (02b_build_company_map.py merges abutting spans), but we keep this
    behavior explicit and document it.

RANKER IMPLICATION (what the profile decides)
---------------------------------------------
After running this and inspecting the printed profile, we will know:
  * Eligible event count vs. raw 44,637 (sanity check; expected to be much
    smaller since most events fall outside SP400 windows).
  * Off-season sparsity — how sparse are weeks outside the Jan/Apr/Jul/Oct
    earnings-cluster peaks?
  * `ndcg@k` feasibility — what fraction of weeks have  >= k events?
  * Whether a sparse-week strategy is needed:
        - if  >50%  of weeks have  <3 events  ->  consider dropping sparse
          weeks from the ranker training set (use them only at inference),
          OR clustering adjacent weeks.

CLI USAGE (after implementation)
--------------------------------
    python luan_bot_trading/02_features/01_features_gate_events.py            # gate + write + profile
    python luan_bot_trading/02_features/01_features_gate_events.py --dry-run # profile only, no write


HDF5 WRITE SAFETY
-----------------
Per STOP_DOING_EXTRA_SHIT.md and the team convention: NEVER open `db.h5` with
`mode='w'` — it would destroy every existing node (`/sp400/*`, `/earnings/raw`,
`/macros/*`, `/metadata/*`). Always use `HDFStore(mode='a')` and, if the target
node already exists, `store.remove('/features/gated_events')` before
`store.put(...)`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


# ==============================================================================
# CONFIGURATION
# ==============================================================================
DB_FILE = Path(__file__).resolve().parent.parent / "01_data" / "db.h5"
COMPANIES_KEY = "/metadata/sp400_companies"
EARNINGS_KEY = "/earnings/raw"
GATED_EVENTS_KEY = "/features/gated_events"

BUFFER_DAYS = 90  # 1 quarter post-addition stabilization window (= first-qtr exclusion)

# Week-size histogram buckets (inclusive lower/upper for readability).
_EPS_BUCKETS = [
    ("lt_3", lambda n: n < 3),
    ("3_to_5", lambda n: 3 <= n <= 5),
    ("6_to_10", lambda n: 6 <= n <= 10),
    ("11_to_20", lambda n: 11 <= n <= 20),
    ("21_to_50", lambda n: 21 <= n <= 50),
    ("gt_50", lambda n: n > 50),
]


# ==============================================================================
# PART 1 — LOADERS
# ==============================================================================
def load_sp400_companies() -> pd.DataFrame:
    """
    Load /metadata/sp400_companies from DB_FILE.

    Return columns (existing schema):
        canonical_ticker, cik, aliases (JSON str), name, sic, index_ref,
        combined_intervals (JSON str), per_ticker_intervals (JSON str),
        price_unavailable (bool)

    Raises:
        FileNotFoundError if DB_FILE or the /metadata/sp400_companies key
            is missing (caller must run 02b_build_company_map.py first).
    """
    if not DB_FILE.exists():
        raise FileNotFoundError(
            f"DB file not found: {DB_FILE} — run 02b_build_company_map.py first."
        )
    with pd.HDFStore(DB_FILE, mode="r") as store:
        if COMPANIES_KEY not in store.keys():
            raise FileNotFoundError(
                f"Key {COMPANIES_KEY} not in {DB_FILE} — run 02b_build_company_map.py first."
            )
        return store.get(COMPANIES_KEY)


def load_earnings_raw() -> pd.DataFrame:
    """
    Load /earnings/raw from DB_FILE.

    Return columns (existing schema):
        report_date, fiscal_period_end, code, canonical_ticker, cik,
        actual, estimate, difference, percent, before_after_market, currency

    Only `report_date` and `canonical_ticker` are strictly needed by the
    gate; other columns are not loaded for memory efficiency in Stage 1.

    Raises:
        FileNotFoundError if DB_FILE or /earnings/raw is missing (caller
            must run 06_earnings_gathering.py first).
    """
    if not DB_FILE.exists():
        raise FileNotFoundError(
            f"DB file not found: {DB_FILE} — run 06_earnings_gathering.py first."
        )
    with pd.HDFStore(DB_FILE, mode="r") as store:
        if EARNINGS_KEY not in store.keys():
            raise FileNotFoundError(
                f"Key {EARNINGS_KEY} not in {DB_FILE} — run 06_earnings_gathering.py first."
            )
        return store.get(EARNINGS_KEY)


# ==============================================================================
# PART 2 — GATING LOGIC
# ==============================================================================
def gate_events(
    companies: pd.DataFrame,
    earnings: pd.DataFrame,
    today: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Apply interval gating + 90-day buffer to produce the eligible-events pool.

    Args:
        companies: output of load_sp400_companies()
        earnings  : output of load_earnings_raw()
        today     : override for "now" (testing). Defaults to pd.Timestamp.now().normalize()

    Returns:
        DataFrame with columns:
            canonical_ticker, cik, report_date, added, removed, calendar_week_group
        Sorted by ['calendar_week_group', 'canonical_ticker'].

    Per-event duplication rule:
        An event qualifying under MULTIPLE overlapping intervals of the same
        company is emitted once per qualifying interval. (Only relevant if a
        future rebuild of /metadata/sp400_companies stops merging abutting
        spans; today they are merged by 02b, so this is audit-safety only.)
    """
    if today is None:
        today = pd.Timestamp.now().normalize()
    else:
        today = pd.Timestamp(today).normalize()

    buffer = pd.Timedelta(days=BUFFER_DAYS)

    earnings = earnings.copy()
    if not pd.api.types.is_datetime64_any_dtype(earnings["report_date"]):
        earnings["report_date"] = pd.to_datetime(earnings["report_date"])

    # Pre-group earnings by canonical_ticker to avoid the per-company full-frame scan.
    earnings_by_ticker: dict[str, pd.Series] = {
        t: g["report_date"].sort_values().reset_index(drop=True)
        for t, g in earnings.groupby("canonical_ticker")
    }

    rows: list[dict] = []
    for _, company in companies.iterrows():
        if bool(company.get("price_unavailable", False)):
            continue

        canonical = company["canonical_ticker"]
        cik = company.get("cik", None)

        comp_earn = earnings_by_ticker.get(canonical)
        if comp_earn is None or comp_earn.empty:
            continue

        for itv in _parse_intervals(company["combined_intervals"]):
            added_at = itv.get("added")
            removed_at = itv.get("removed")

            if added_at is None or pd.isna(added_at):
                # Defensive: a null-added interval should not occur post-02b
                # backfill, but if it does we cannot anchor the 90d window —
                # treat that interval as non-contributing for SAFETY rather
                # than collapsing it into pre-history.
                continue

            added_dt = pd.Timestamp(added_at)
            removed_dt = today if removed_at is None or pd.isna(removed_at) else pd.Timestamp(removed_at)
            buf_start = added_dt + buffer

            in_window = comp_earn[(comp_earn >= buf_start) & (comp_earn <= removed_dt)]
            for ev_date in in_window:
                iso = pd.Timestamp(ev_date).isocalendar()
                rows.append({
                    "canonical_ticker": canonical,
                    "cik": cik,
                    "report_date": pd.Timestamp(ev_date),
                    "added": added_dt,
                    "removed": removed_dt,
                    "calendar_week_group": f"{iso.year}-W{iso.week:02d}",
                })

    df = pd.DataFrame(rows, columns=[
        "canonical_ticker", "cik", "report_date",
        "added", "removed", "calendar_week_group",
    ])
    if not df.empty:
        df = df.sort_values(["calendar_week_group", "canonical_ticker"]).reset_index(drop=True)
    return df


def _parse_intervals(combined_intervals) -> list[dict]:
    """
    Normalize the `combined_intervals` cell into a list of
    {"added": str|None, "removed": str|None} dicts. Accepts either a
    JSON string (HDF read) or an already-parsed list (in-memory caller).

    Returns [] if the field is empty / unparsable, so the caller's
    `for itv in intervals:` loop simply contributes no rows.
    """
    import json
    if combined_intervals is None:
        return []
    if isinstance(combined_intervals, str):
        s = combined_intervals.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
        except (json.JSONDecodeError, ValueError):
            return []
        return list(parsed) if isinstance(parsed, list) else []
    if isinstance(combined_intervals, (list, tuple)):
        return list(combined_intervals)
    return []


# ==============================================================================
# PART 3 — STORAGE
# ==============================================================================
def write_gated_events(df: pd.DataFrame, key: str = GATED_EVENTS_KEY) -> None:
    """
    Persist the gated events frame to DB_FILE under /features/gated_events.

    Safety:
        NEVER open with mode='w'  — would truncate the entire db.h5.
        Open HDFStore(mode='a'); if the key exists, store.remove(key) first.
    """
    with pd.HDFStore(DB_FILE, mode="a") as store:
        if key in store.keys():
            store.remove(key)
        store.put(
            key,
            df,
            format="table",
            data_columns=["calendar_week_group", "canonical_ticker"],
        )


# ==============================================================================
# PART 4 — WEEKLY DISTRIBUTION PROFILE
# ==============================================================================
def profile_weekly_distribution(df: pd.DataFrame) -> dict:
    """
    Compute the weekly-distribution profile used to decide:
        (a) the ranker's `ndcg@k`
        (b) whether sparse-week clustering/dropping is required.
    """
    if df.empty:
        return {
            "total_events": 0,
            "date_min": None,
            "date_max": None,
            "date_span_days": 0,
            "n_weeks": 0,
            "events_per_week": pd.Series(dtype=int),
            "eps_summary": {},
            "histogram": {name: 0 for name, _ in _EPS_BUCKETS},
            "weeks_lt_3": 0,
            "weeks_lt_5": 0,
            "weeks_lt_10": 0,
            "frac_weeks_lt_3": 0.0,
            "frac_weeks_lt_5": 0.0,
            "frac_weeks_lt_10": 0.0,
        }

    eps = df.groupby("calendar_week_group").size()
    eps = eps.sort_index()

    histogram = {name: int((eps.apply(pred)).sum()) for name, pred in _EPS_BUCKETS}
    n_weeks = int(eps.size)

    return {
        "total_events": int(df.shape[0]),
        "date_min": pd.Timestamp(df["report_date"].min()),
        "date_max": pd.Timestamp(df["report_date"].max()),
        "date_span_days": int((df["report_date"].max() - df["report_date"].min()).days),
        "n_weeks": n_weeks,
        "events_per_week": eps,
        "eps_summary": {
            "count": int(eps.count()),
            "mean": float(eps.mean()),
            "median": float(eps.median()),
            "std": float(eps.std(ddof=1)) if eps.count() > 1 else 0.0,
            "min": int(eps.min()),
            "max": int(eps.max()),
            "p10": float(eps.quantile(0.10)),
            "p90": float(eps.quantile(0.90)),
        },
        "histogram": histogram,
        "weeks_lt_3": int((eps < 3).sum()),
        "weeks_lt_5": int((eps < 5).sum()),
        "weeks_lt_10": int((eps < 10).sum()),
        "frac_weeks_lt_3": float((eps < 3).mean()),
        "frac_weeks_lt_5": float((eps < 5).mean()),
        "frac_weeks_lt_10": float((eps < 10).mean()),
    }


def print_profile(profile: dict) -> None:
    """
    Human-readable stdout dump of profile_weekly_distribution().
    """
    bar = "=" * 64
    print(bar)
    print("STAGE 1 — GATED EVENTS WEEKLY DISTRIBUTION PROFILE")
    print(bar)

    if profile["total_events"] == 0:
        print("(no gated events — nothing to profile)")
        print(bar)
        return

    s = profile["eps_summary"]
    print(f"Total gated events:        {profile['total_events']:,}")
    print(f"Distinct weeks (groups):   {profile['n_weeks']:,}")
    print(f"Date range:                {profile['date_min'].date()} -> {profile['date_max'].date()} "
          f"({profile['date_span_days']:,} days)")
    print()
    print("Events per week (EPS) summary:")
    print(f"  count = {s['count']}, mean = {s['mean']:.2f}, median = {s['median']:.1f}, "
          f"std = {s['std']:.2f}")
    print(f"  min = {s['min']}, max = {s['max']}, p10 = {s['p10']:.1f}, p90 = {s['p90']:.1f}")
    print()
    print("Week-size histogram (count of weeks falling in each bucket):")
    h = profile["histogram"]
    print(f"  {'lt_3':<10} (<3)       : {h['lt_3']}")
    print(f"  {'3_to_5':<10} (3-5)      : {h['3_to_5']}")
    print(f"  {'6_to_10':<10} (6-10)     : {h['6_to_10']}")
    print(f"  {'11_to_20':<10} (11-20)    : {h['11_to_20']}")
    print(f"  {'21_to_50':<10} (21-50)    : {h['21_to_50']}")
    print(f"  {'gt_50':<10} (51+)      : {h['gt_50']}")
    print()
    print("ndcg@k feasibility flags (of all weeks, this many have <k events):")
    print(f"  weeks with <3 events  : {profile['weeks_lt_3']:>4} ({profile['frac_weeks_lt_3']*100:5.1f}%)  <- ndcg@3")
    print(f"  weeks with <5 events  : {profile['weeks_lt_5']:>4} ({profile['frac_weeks_lt_5']*100:5.1f}%)  <- ndcg@5")
    print(f"  weeks with <10 events : {profile['weeks_lt_10']:>4} ({profile['frac_weeks_lt_10']*100:5.1f}%)  <- ndcg@10")
    print(bar)


# ==============================================================================
# MAIN
# ==============================================================================
def main(dry_run: bool = False) -> int:
    """
    Orchestrates Stage 1 end to end:

        1. print header (config + DB_FILE)
        2. companies = load_sp400_companies()
        3. earnings   = load_earnings_raw()
        4. gated_df   = gate_events(companies, earnings)
        5. if not dry_run: write_gated_events(gated_df)
           else:           print "(--dry-run: not writing)"
        6. profile    = profile_weekly_distribution(gated_df)
        7. print_profile(profile)

    Exit codes:
        0 — success (profile printed; wrote gated_events unless --dry-run)
        1 — missing DB_FILE or required keys (load_* raise FileNotFoundError
            with helpful "run 02b_build_company_map.py / 06_earnings_gathering.py first" message)
        2 — empty gated set (DB still readable but nothing to do)
    """
    bar = "=" * 64
    print(bar)
    print("STAGE 1 — Gated Events Builder")
    print(f"DB file: {DB_FILE}")
    print(f"Output key: {GATED_EVENTS_KEY}  {'(dry-run, no write)' if dry_run else ''}")
    print(bar)

    companies = load_sp400_companies()
    earnings = load_earnings_raw()
    print(f"Loaded {len(companies):,} companies, {len(earnings):,} raw earnings events.")

    gated_df = gate_events(companies, earnings)
    n = len(gated_df)
    print(f"Gated events: {n:,}")

    if n == 0:
        print("(empty gated set — nothing to do)")
        print(bar)
        return 2

    if dry_run:
        print("(--dry-run: not writing to db.h5)")
    else:
        write_gated_events(gated_df)
        print(f"Wrote {n:,} rows to {GATED_EVENTS_KEY}.")

    profile = profile_weekly_distribution(gated_df)
    print_profile(profile)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 1: gate S&P 400 earnings events to eligible weekly cohorts "
                    "for the listwise ranker, then profile the weekly distribution."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only profile; do NOT write /features/gated_events to db.h5.",
    )
    args = parser.parse_args()
    raise SystemExit(main(dry_run=args.dry_run))
