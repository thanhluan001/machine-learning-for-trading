#!/usr/bin/env python3
"""
Stage 1: Gated Events + Weekly Distribution Profile  (Phase E rewrite v2)
==========================================================================

PURPOSE
-------
Before building the listwise-ranker feature matrix (Stage 2:
`02_features/02_build_feature_matrix.py`), we need to know the shape of the
eligible-event population along the LTR group axis: `calendar_week_group`
(ISO calendar week of `report_date`, format `YYYY-Www` -- see `Design.md` 17
and `features.md` 0).

The listwise objective `rank:ndcg` with `eval_metric="ndcg@3"` only produces
useful gradients when weekly cohort groups are large enough. Raw
`/earnings/raw` (44,308 rows post Phase D migration) contains events that MUST
be pruned before becoming ranker rows:

  * earnings dated BEFORE the permaTicker's `added` date (pre-addition quarters);
  * earnings dated INSIDE the 90-day stabilization buffer (first quarter
    post-addition; already excluded by the buffer expansion; no separate
    per-event skip is required);
  * earnings dated AFTER a `removed` date (delistings / removals);
  * earnings from `price_unavailable=True` permaTickers (no
    `/sp400/{permaTicker}` price node exists for them -- they can never
    produce feature rows).

The legacy v1 §7.7 disambiguation rule (loser-perm_id overlap drop) is GONE in
this v2 rewrite -- under the permaTicker identity model, each permaTicker IS
the storage key, so no two active permaTickers can collide on a single
canonical_ticker. (Per Phase-A audit: the only 2 canonical_ticker collisions
have price_unavailable=True, producing 0 gated events.)

INPUTS (from `01_data/db.h5`; READ-ONLY -- no external API calls):
  * `/metadata/sp400_permatickers` -- columns:
        permaTicker, canonical_ticker, name, isActive, openfigi, cik,
        sic, index_ref, wikipedia_intervals (JSON str),
        price_unavailable (bool).
  * `/earnings/raw`              -- Phase D schema (post-migration):
        report_date, fiscal_period_end, code, canonical_ticker,
        cik, actual, estimate, difference, percent, before_after_market,
        currency, permaTicker.

OUTPUTS:
  * `/features/gated_events` in `01_data/db.h5` (HDF5 table,
    `HDFStore(mode='a')`, overwrite via `store.remove` if key exists -- NEVER
    `mode='w'` on the whole DB).
    Columns (exactly seven; Phase E v2 KEYED BY permaTicker):
        permaTicker         : str                    -- Tiingo identity anchor (PRIMARY row key)
        canonical_ticker     : str                    -- audit only; joins to /sp400/{canonical_ticker via pt map}
        cik                   : str                    -- SEC CIK (audit / dedup)
        report_date          : datetime64[us]         -- the earnings announcement date
        added                : datetime64[us]         -- interval.added (audit only)
        removed              : datetime64[us]         -- interval.removed or today (audit only)
        calendar_week_group  : str                    -- ISO week "YYYY-Www" (LTR group key)

  * stdout profile (no disk): per-week summary stats, week-size histogram
    buckets, ndcg@k feasibility flags.

WHAT THIS SCRIPT DOES NOT DO (deferred to Stage 2)
-------------------------------------------------
  * No feature computation (no `sue_score`, no Block 2-5 features).
  * No `/sp400/{permaTicker}` price-series loading.
  * No `T` (matched trading day) computation.
  * No `car_10d` (the LTR target) -- requires T+1..T+11 abnormal return window.
  * No sector ETF or macro joins.
  * No model code.

ALGORITHM
---------
    today = pd.Timestamp.now().normalize()      # `removed = null` -> today
    permatickers = load_permatickers()          # /metadata/sp400_permatickers
    earnings = load_earnings_raw()              # /earnings/raw (post Phase D)
    gated_rows = []

    for pt_ in permatickers:
        if pt_.price_unavailable: continue     # zero-row contributors
        intervals = pt_.wikipedia_intervals
        comp_earn = earnings[earnings.permaTicker == pt_.permaTicker]

        for itv in intervals:
            added_dt   = pd.to_datetime(itv['added'])
            removed_dt = pd.to_datetime(itv['removed']) if itv['removed'] else today
            buf_start  = added_dt + pd.Timedelta(days=90)  # 90d = 1Q buffer

            mask = (comp_earn.report_date >= buf_start) & (comp_earn.report_date <= removed_dt)
            for ev_date in comp_earn[mask].report_date:
                iso = ev_date.isocalendar()
                gated_rows.append({...})

    gated_df = DataFrame(gated_rows).sort_values(['calendar_week_group', 'permaTicker'])

KEY DESIGN DECISIONS (carried over from v1; v2 changes per Phase D/E report)
-----------------------------------------------------------------------------
  * 90-day buffer = first-quarter exclusion. Applied as window start
    `[added + 90d, removed]`. NO ADDITIONAL per-event skip.
  * `price_unavailable=True` permaTicakers produce ZERO gated events.
  * `calendar_week_group` uses `isocalendar()` -- ISO calendar week.
  * No §7.7 disambiguation rule (deleted in v2 -- permaTicker is the
    storage key, no collision possible).
  * Multi-Wikipedia-interval permaTicakers (e.g. ALK has 2 add/remove cycles):
    each interval's [added+90d, removed] window is applied independently.
    An earnings event from either interval survives if it falls in either
    window. Standard multi-interval application; no special aggregation.

CLI USAGE
---------
    python luan_bot_trading/02_features/01_features_gate_events.py            # gate + write + profile
    python luan_bot_trading/02_features/01_features_gate_events.py --dry-run # profile only, no write

HDF5 WRITE SAFETY
-----------------
Per STOP_DOING_EXTRA_SHIT.md: NEVER open `db.h5` with `mode='w'` -- it would
destroy every existing node. Always use `HDFStore(mode='a')` and, if the
target node already exists, `store.remove('/features/gated_events')` before
`store.put(...)`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


# ==============================================================================
# CONFIGURATION
# ==============================================================================
DB_FILE = Path(__file__).resolve().parent.parent / "01_data" / "db.h5"
PERMATICKERS_KEY = "/metadata/sp400_permatickers"
EARNINGS_KEY = "/earnings/fmp"
GATED_EVENTS_KEY = "/features/gated_events"

BUFFER_DAYS = 90  # 1 quarter post-addition stabilization window (= first-qtr exclusion)

# Sentinel for "+infinity" effective end (permaTicker still LIVE / removed=null)
# -- retained for consistency though no longer used for overlap disambiguation.
_INF_END = pd.Timestamp("2099-12-31")

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
# PART 1 -- LOADERS
# ==============================================================================
def load_permatickers() -> pd.DataFrame:
    """Load /metadata/sp400_permatickers from DB_FILE.

    Required per-row fields (Phase A v2 output schema, post-cleanup_2024):
        permaTicker, canonical_ticker, name, isActive, openfigi, cik,
        sic, index_ref, wikipedia_intervals (JSON str),
        price_unavailable (bool).
    """
    if not DB_FILE.exists():
        raise FileNotFoundError(
            f"DB file not found: {DB_FILE} -- run 02b_build_company_map.py (Phase A) first."
        )
    with pd.HDFStore(DB_FILE, mode="r") as store:
        if PERMATICKERS_KEY not in store.keys():
            raise FileNotFoundError(
                f"Key {PERMATICKERS_KEY} not in {DB_FILE} -- run "
                "02b_build_company_map.py (Phase A v2) first."
            )
        return store.get(PERMATICKERS_KEY)


def load_earnings_raw() -> pd.DataFrame:
    """Load /earnings/raw from DB_FILE (Phase D migrated schema).

    Stage 1 uses `report_date` and `permaTicker`; other columns are not
    required here.
    """
    if not DB_FILE.exists():
        raise FileNotFoundError(
            f"DB file not found: {DB_FILE} -- run 06_earnings_gathering.py +"
            "phase_d_migrate_earnings_keys.py first."
        )
    with pd.HDFStore(DB_FILE, mode="r") as store:
        if EARNINGS_KEY not in store.keys():
            raise FileNotFoundError(
                f"Key {EARNINGS_KEY} not in {DB_FILE}"
            )
        return store.get(EARNINGS_KEY)


# ==============================================================================
# PART 2 -- INTERVAL PARSING
# ==============================================================================
def _parse_intervals(wikipedia_intervals) -> list[dict]:
    """Normalize the `wikipedia_intervals` cell into a list of
    {"added": str|None, "removed": str|None} dicts.
    Accepts either a JSON string (HDF read) or an already-parsed list."""
    if wikipedia_intervals is None:
        return []
    if isinstance(wikipedia_intervals, str):
        s = wikipedia_intervals.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
        except (json.JSONDecodeError, ValueError):
            return []
        return list(parsed) if isinstance(parsed, list) else []
    if isinstance(wikipedia_intervals, (list, tuple)):
        return list(wikipedia_intervals)
    return []


# ==============================================================================
# PART 3 -- GATING LOGIC
# ==============================================================================
def gate_events(
    permatickers_df: pd.DataFrame,
    earnings: pd.DataFrame,
    today: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Apply interval gating + 90-day buffer to permaTicker-keyed earnings.

    Returns:
        gated_df : DataFrame with columns
            permaTicker, canonical_ticker, cik, report_date,
            added, removed, calendar_week_group
            sorted by ['calendar_week_group', 'permaTicker'].
        audit    : dict with key counts.
    """
    if today is None:
        today = pd.Timestamp.now().normalize()
    else:
        today = pd.Timestamp(today).normalize()

    buffer = pd.Timedelta(days=BUFFER_DAYS)

    earnings = earnings.copy()
    if not pd.api.types.is_datetime64_any_dtype(earnings["report_date"]):
        earnings["report_date"] = pd.to_datetime(earnings["report_date"])

    # Pre-group earnings by permaTicker for O(1) lookup per permaTicker.
    earnings_by_pt: dict[str, pd.DataFrame] = {
        pt: g.sort_values("report_date").reset_index(drop=True)
        for pt, g in earnings.groupby("permaTicker")
    }

    audit = {
        "n_permatickers_total": len(permatickers_df),
        "n_permatickers_skipped_unavailable": 0,
        "n_permatickers_zero_earnings": 0,
        "n_permatickers_gated_zero": 0,
        "n_buffer_drops": 0,
        "n_events_dup_collapsed": 0,
        "total_events_processed": 0,
        "total_events_gated": 0,
    }

    rows: list[dict] = []
    for _, rec in permatickers_df.iterrows():
        if bool(rec.get("price_unavailable", False)):
            audit["n_permatickers_skipped_unavailable"] += 1
            continue

        pt = rec["permaTicker"]
        if pt is None or (isinstance(pt, float) and pd.isna(pt)):
            continue

        canonical = rec["canonical_ticker"]
        cik = rec.get("cik", None)
        if isinstance(cik, float) and pd.isna(cik):
            cik = None

        comp_earn = earnings_by_pt.get(pt)
        if comp_earn is None or comp_earn.empty:
            audit["n_permatickers_zero_earnings"] += 1
            continue
        audit["total_events_processed"] += len(comp_earn)

        intervals = _parse_intervals(rec["wikipedia_intervals"])

        n_gated_for_this_pt = 0
        for itv in intervals:
            added_at = itv.get("added")
            removed_at = itv.get("removed")

            if added_at is None or (isinstance(added_at, float) and pd.isna(added_at)):
                # Defensive: null-added interval should not occur post-Phase A backfill.
                continue

            added_dt = pd.Timestamp(added_at)
            removed_dt = (
                today
                if removed_at is None or (isinstance(removed_at, float) and pd.isna(removed_at))
                else pd.Timestamp(removed_at)
            )
            buf_start = added_dt + buffer

            in_window = (comp_earn["report_date"] >= buf_start) & (comp_earn["report_date"] <= removed_dt)
            audit["n_buffer_drops"] += int((~in_window).sum())

            for ev_date in comp_earn.loc[in_window, "report_date"]:
                iso = pd.Timestamp(ev_date).isocalendar()
                rows.append({
                    "permaTicker": pt,
                    "canonical_ticker": canonical,
                    "cik": cik,
                    "report_date": pd.Timestamp(ev_date),
                    "added": added_dt,
                    "removed": removed_dt,
                    "calendar_week_group": f"{iso.year}-W{iso.week:02d}",
                })
                n_gated_for_this_pt += 1

        if n_gated_for_this_pt == 0:
            audit["n_permatickers_gated_zero"] += 1
            # If this permaTicker had events but NONE survived gating.

    df = pd.DataFrame(
        rows,
        columns=[
            "permaTicker", "canonical_ticker", "cik", "report_date",
            "added", "removed", "calendar_week_group",
        ],
    )
    if not df.empty:
        # Dedup by (permaTicker, report_date) after interval application.
        # Why: A permaTicker can have multiple `wikipedia_intervals` entries,
        # and when two intervals both contain the same earnings event
        # (overlap zone, e.g. GME has [{added: 2016-04-22, removed: null}
        # AND {added: 2021-08-04, removed: null}] -- both windows still open
        # today so all 2021-11+ events qualify twice), the loop above emits
        # 2 duplicate gated rows for one underlying earnings event. We keep
        # the row whose `added` is EARLIEST (preserves the historical
        # provenance of the first S&P 400 addition that owns this event).
        # Sort stably on [permaTicker, report_date, added] so drop_duplicates
        # deterministically picks the earliest-added interval's row.
        n_pre_dupmerge = len(df)
        df = (
            df.sort_values(["permaTicker", "report_date", "added"], kind="mergesort")
              .drop_duplicates(subset=["permaTicker", "report_date"], keep="first")
        )
        n_dropped_by_merge = n_pre_dupmerge - len(df)
        if n_dropped_by_merge > 0:
            audit["n_events_dup_collapsed"] = n_dropped_by_merge
            # Re-sort to the consumer's expected order.
            df = df.sort_values(["calendar_week_group", "permaTicker"], kind="mergesort").reset_index(drop=True)

    audit["total_events_gated"] = len(df)
    return df, audit


# ==============================================================================
# PART 4 -- STORAGE
# ==============================================================================
def write_gated_events(df: pd.DataFrame, key: str = GATED_EVENTS_KEY) -> None:
    """Persist the gated events frame to DB_FILE under /features/gated_events.

    Safety:
        NEVER open with mode='w' -- would truncate the entire db.h5.
        Open HDFStore(mode='a'); if the key exists, store.remove(key) first.
    """
    with pd.HDFStore(DB_FILE, mode="a") as store:
        if key in store.keys():
            store.remove(key)
        store.put(
            key,
            df,
            format="table",
            data_columns=["calendar_week_group", "permaTicker", "canonical_ticker"],
        )


# ==============================================================================
# PART 5 -- WEEKLY DISTRIBUTION PROFILE
# ==============================================================================
def profile_weekly_distribution(df: pd.DataFrame) -> dict:
    """Compute the weekly-distribution profile used to decide:
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
    """Human-readable stdout dump of profile_weekly_distribution()."""
    bar = "=" * 64
    print(bar)
    print("STAGE 1 -- GATED EVENTS WEEKLY DISTRIBUTION PROFILE")
    print(bar)

    if profile["total_events"] == 0:
        print("(no gated events -- nothing to profile)")
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


def print_audit(audit: dict) -> None:
    bar = "-" * 64
    print(bar)
    print("STAGE 1 -- GATING AUDIT (Phase E v2: permaTicker-keyed)")
    print(bar)
    print(f"  permaTickers total                : {audit['n_permatickers_total']}")
    print(f"  permaTickers skipped (unavailable) : {audit['n_permatickers_skipped_unavailable']}")
    print(f"  permaTickers zero raw earnings     : {audit['n_permatickers_zero_earnings']}")
    print(f"  permaTickers gated-to-zero         : {audit['n_permatickers_gated_zero']}")
    print(f"  events processed                   : {audit['total_events_processed']}")
    print(f"  events dropped by buffer/window    : {audit['n_buffer_drops']}")
    if audit.get('n_events_dup_collapsed', 0) > 0:
        print(f"  events collapsed (overlap dups)     : {audit['n_events_dup_collapsed']} (multi-interval overlap dedup)")
    print(f"  events gated (survived)            : {audit['total_events_gated']}")
    print(bar)


# ==============================================================================
# MAIN
# ==============================================================================
def main(dry_run: bool = False) -> int:
    bar = "=" * 64
    print(bar)
    print("STAGE 1 -- Gated Events Builder  [Phase E v2 rewrite: permaTicker-keyed]")
    print(f"DB file: {DB_FILE}")
    print(f"Output key: {GATED_EVENTS_KEY}  {'(dry-run, no write)' if dry_run else ''}")
    print(bar)

    permatickers_df = load_permatickers()
    earnings = load_earnings_raw()
    print(
        f"Loaded {len(permatickers_df):,} permaTickers, {len(earnings):,} raw earnings events."
    )

    gated_df, audit = gate_events(permatickers_df, earnings)
    print(f"Gated events: {len(gated_df):,}")

    print_audit(audit)

    if len(gated_df) == 0:
        print("(empty gated set -- nothing to do)")
        print(bar)
        return 2

    if dry_run:
        print("(--dry-run: not writing to db.h5)")
    else:
        write_gated_events(gated_df)
        print(f"Wrote {len(gated_df):,} rows to {GATED_EVENTS_KEY}.")

    profile = profile_weekly_distribution(gated_df)
    print_profile(profile)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 1 (v2): gate S&P 400 permaTicker earnings events to "
                    "eligible weekly cohorts for the listwise ranker, then "
                    "profile the weekly distribution. (Phase E v2 rewrite: "
                    "permaTicker-keyed, section-7.7 disambiguation REMOVED.)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only profile; do NOT write /features/gated_events to db.h5.",
    )
    args = parser.parse_args()
    raise SystemExit(main(dry_run=args.dry_run))
