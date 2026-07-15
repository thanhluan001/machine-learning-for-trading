#!/usr/bin/env python3
"""
Stage 1: Gated Events + Weekly Distribution Profile  (Phase E rewrite)
=======================================================================

PURPOSE
-------
Before building the listwise-ranker feature matrix (Stage 2:
`02_features/02_build_feature_matrix.py`), we need to know the shape of the
eligible-event population along the LTR group axis: `calendar_week_group`
(ISO calendar week of `report_date`, format `YYYY-Www` — see `Design.md` §17
and `features.md` §0).

The listwise objective `rank:ndcg` with `eval_metric="ndcg@3"` only produces
useful gradients when weekly cohort groups are large enough. Raw
`/earnings/raw` (44,897 rows post Phase D) contains events that MUST be
pruned before becoming ranker rows:

  * earnings dated BEFORE the perm_id's `added` date (pre-addition quarters);
  * earnings dated INSIDE the 90-day stabilization buffer (first quarter
    post-addition; already excluded by the buffer expansion; no separate
    per-event skip is required);
  * earnings dated AFTER a `removed` date (delistings / removals);
  * earnings from `price_unavailable=True` perm_ids (no `/sp400/{canonical}`
    price node exists for them — they can never produce feature rows);
  * NEW (Phase E §7.7): earnings of a LOSER perm_id whose report_date falls
    inside the OVERLAP ZONE shared with the WINNER perm_id (which shares the
    same `canonical_ticker`). These belong to the LIVE-winning perm_id's
    underlying asset, not the stale loser phantom; NaN-dropping them at the
    gate removes them at the earliest possible layer.

INPUTS (from `01_data/db.h5`; READ-ONLY — no external API calls):
  * `/metadata/sp400_perm_ids` — columns:
        perm_id, cik, canonical_ticker, aliases (JSON), name, sic, index_ref,
        combined_intervals (JSON list of {"added", "removed"} dicts),
        per_ticker_intervals (JSON audit), price_unavailable (bool).
  * `/earnings/raw`              — Phase D schema:
        report_date, fiscal_period_end, code, perm_id, canonical_ticker,
        cik, actual, estimate, difference, percent, before_after_market,
        currency.

OUTPUTS:
  * `/features/gated_events` in `01_data/db.h5` (HDF5 table,
    `HDFStore(mode='a')`, overwrite via `store.remove` if key exists — NEVER
    `mode='w'` on the whole DB).
    Columns (exactly seven; Phase E adds `perm_id`):
        perm_id              : str                    — Phase-A asset-track anchor (PRIMARY row key)
        canonical_ticker     : str                    — joins to /sp400/{canonical}
        cik                   : str                    — SEC CIK (audit / dedup)
        report_date          : datetime64[us]         — the earnings announcement date
        added                : datetime64[us]         — interval.added (audit only)
        removed              : datetime64[us]         — interval.removed or today (audit only)
        calendar_week_group  : str                     — ISO week "YYYY-Www" (LTR group key)

  * stdout profile (no disk): per-week summary stats, week-size histogram
    buckets, ndcg@k feasibility flags.

WHAT THIS SCRIPT DOES NOT DO (deferred to Stage 2)
-------------------------------------------------
  * No feature computation (no `sue_score`, no Block 2-5 features).
  * No `/sp400/{canonical}` price-series loading.
  * No `T` (matched trading day) computation.
  * No `car_10d` (the LTR target) — requires T+1..T+11 abnormal return window.
  * No sector ETF or macro joins.
  * No model code.

ALGORITHM
---------
    today = pd.Timestamp.now().normalize()      # `removed = null` -> today
    perm_ids = load_perm_ids()                 # /metadata/sp400_perm_ids
    earnings = load_earnings_raw()             # /earnings/raw
    gated_rows = []

    # Pre-compute §7.7 disambiguation: per canonical, identify winner and
    # overlap-zone loser-perm_id. Loser events in overlap zone get NaN-dropped.
    loser_overlap_filter = _build_loser_overlap_filter(perm_ids, today)

    for perm_id in perm_ids:
        if perm_id.price_unavailable:
            continue                            # zero-row contributors

        intervals   = perm_id.combined_intervals
        comp_earn   = earnings[earnings.perm_id == perm_id.perm_id]
        comp_earn   = comp_earn[~loser_overlap_filter[perm_id.perm_id]]  # §7.7 filter

        for itv in intervals:
            added_dt   = pd.to_datetime(itv['added'])
            removed_dt = pd.to_datetime(itv['removed']) if itv['removed'] else today
            buf_start  = added_dt + pd.Timedelta(days=90)     # 90d = 1Q buffer

            mask = (comp_earn.report_date >= buf_start) & (comp_earn.report_date <= removed_dt)
            for ev_date in comp_earn[mask].report_date:
                iso = ev_date.isocalendar()
                gated_rows.append({...})

    gated_df = DataFrame(gated_rows).sort_values(['calendar_week_group', 'perm_id'])

KEY DESIGN DECISIONS (carried over from v1 + Phase E §7.7)
----------------------------------------------------------
  * 90-day buffer = first-quarter exclusion. Applied as window start
    `[added + 90d, removed]`. NO ADDITIONAL per-event skip.
  * `price_unavailable=True` perm_ids produce ZERO gated events.
  * `calendar_week_group` uses `isocalendar()` — ISO calendar week.
  * §7.7 disambiguation: loser-perm_id events in the overlap zone with the
    winner perm_id (sharing the same canonical_ticker) are dropped AT THE
    GATE — not later at Stage 2. Empirical impact: 105 events dropped
    (0.23% of all /earnings/raw rows).

CLI USAGE
---------
    python luan_bot_trading/02_features/01_features_gate_events.py            # gate + write + profile
    python luan_bot_trading/02_features/01_features_gate_events.py --dry-run # profile only, no write

HDF5 WRITE SAFETY
-----------------
Per STOP_DOING_EXTRA_SHIT.md: NEVER open `db.h5` with `mode='w'` — it would
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
PERM_IDS_KEY = "/metadata/sp400_perm_ids"
EARNINGS_KEY = "/earnings/raw"
GATED_EVENTS_KEY = "/features/gated_events"

BUFFER_DAYS = 90  # 1 quarter post-addition stabilization window (= first-qtr exclusion)

# Sentinel for "+infinity" effective end (perm_id still LIVE / removed=null).
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
# PART 1 — LOADERS
# ==============================================================================
def load_perm_ids() -> pd.DataFrame:
    """Load /metadata/sp400_perm_ids from DB_FILE.

    Required per-row fields (Phase A output schema):
        perm_id, cik, canonical_ticker, aliases (JSON str), name,
        sic, index_ref, combined_intervals (JSON str),
        per_ticker_intervals (JSON str), price_unavailable (bool)
    """
    if not DB_FILE.exists():
        raise FileNotFoundError(
            f"DB file not found: {DB_FILE} — run 02b_build_company_map.py (Phase A) first."
        )
    with pd.HDFStore(DB_FILE, mode="r") as store:
        if PERM_IDS_KEY not in store.keys():
            raise FileNotFoundError(
                f"Key {PERM_IDS_KEY} not in {DB_FILE} — run 02b_build_company_map.py (Phase A) first."
            )
        return store.get(PERM_IDS_KEY)


def load_earnings_raw() -> pd.DataFrame:
    """Load /earnings/raw from DB_FILE (Phase D schema).

    Stage 1 uses only `report_date` and `perm_id`; other columns are not
    required here.

    Raises FileNotFoundError if DB_FILE or /earnings/raw is missing (caller
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
# PART 2 — §7.7 DISAMBIGUATION BUILDERS
# ==============================================================================
def _parse_intervals(combined_intervals) -> list[dict]:
    """Normalize the `combined_intervals` cell into a list of
    {"added": str|None, "removed": str|None} dicts.
    Accepts either a JSON string (HDF read) or an already-parsed list."""
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


def _effective_end(intervals: list[dict]) -> pd.Timestamp:
    """Effective end = +infinity sentinel if any interval has null `removed`,
    else max `removed` date. Used for §7.7 winner/loser determination."""
    max_removed = pd.Timestamp("1970-01-01")
    for itv in intervals:
        r = itv.get("removed")
        if r is None or (isinstance(r, float) and pd.isna(r)):
            return _INF_END
        try:
            d = pd.Timestamp(r)
            if d > max_removed:
                max_removed = d
        except Exception:
            continue
    # Defensive: if all intervals had null `added` somehow, we still want
    # 1970 (always-loses per §7.7). Realistically this return is unreachable.
    return max_removed


def _spans(intervals: list[dict]):
    """Yield (added_ts, removed_ts_or_INF) for each interval."""
    for itv in intervals:
        a = itv.get("added")
        r = itv.get("removed")
        a_ts = (
            pd.Timestamp(a)
            if a and not (isinstance(a, float) and pd.isna(a))
            else pd.Timestamp("1970-01-01")
        )
        if r is None or (isinstance(r, float) and pd.isna(r)):
            r_ts = _INF_END
        else:
            r_ts = pd.Timestamp(r)
        yield a_ts, r_ts


def _overlap_span(itvs_a: list[dict], itvs_b: list[dict]):
    """Return (lo, hi) of earliest contiguous overlap, or None if no overlap.
    For Phase E we only need a YES/NO + first overlap span; per-pair overlap
    is simpler than computing the union region."""
    for a_a, a_r in _spans(itvs_a):
        for b_a, b_r in _spans(itvs_b):
            lo = max(a_a, b_a)
            hi = min(a_r, b_r)
            if lo <= hi:
                return (lo, hi)
    return None


def build_loser_overlap_masks(
    perm_ids_df: pd.DataFrame,
) -> dict[str, tuple[pd.Timestamp, pd.Timestamp] | None]:
    """For each perm_id, determine the §7.7 "loser overlap zone" span if any.

    Returns a dict mapping perm_id -> (overlap_lo_ts, overlap_hi_ts) for losers,
    or perm_id -> None for non-losers (no overlap drop applies).

    Algorithm:
        1. Group by canonical_ticker.
        2. For each group with >= 2 perm_ids, compute each perm_id's effective
           end (max(removed) over all intervals; +inf if any interval is open).
        3. The LATER effective-end perm_id is the WINNER; the OTHER is the LOSER.
        4. Compute the overlap span between the WINNER and LOSER (earliest).
        5. Record (lo, hi) for the LOSER; mark WINNER as None (no drop).
        6. If there is no overlap (NONE), both are None.
        7. For groups with > 2 perm_ids (defensive in case of future data), we
           use the SAME rule pairwise against the max-end one (the rest all lose
           and may get their own overlap span recorded).

    Notes:
        * This is a one-shot O(P^2 * ~3 intervals) computation; cheap.
        * `_INF_END` sentinel is the 'live' marker. Comparison `pd.Timestamp`
          math is preserved because the sentinel is a real Timestamp far
          in the future.
    """
    loser_overlap: dict[str, tuple[pd.Timestamp, pd.Timestamp] | None] = {}

    for canon, group in perm_ids_df.groupby("canonical_ticker"):
        pids = group["perm_id"].tolist()
        if len(pids) < 2:
            for pid in pids:
                loser_overlap[pid] = None
            continue

        # Effective end per perm_id in this canonical group.
        end_by_pid = {
            pid: _effective_end(_parse_intervals(intervals_json))
            for pid, intervals_json in zip(group["perm_id"], group["combined_intervals"])
        }
        # The winner = max effective end. (Tie: by perm_id lexicographic, to be
        # deterministic; in practice ties only happen if both have identical
        # `removed` dates: §7.7 says later-removed wins; if exactly equal,
        # 00the question is undefined and we just pick one.)
        winner_pid = max(pids, key=lambda p: (end_by_pid[p], p))

        for pid in pids:
            if pid == winner_pid:
                loser_overlap[pid] = None
                continue
            ov = _overlap_span(
                _parse_intervals(group.loc[group["perm_id"] == winner_pid, "combined_intervals"].iloc[0]),
                _parse_intervals(group.loc[group["perm_id"] == pid, "combined_intervals"].iloc[0]),
            )
            loser_overlap[pid] = ov  # may be None if no overlap

    return loser_overlap


# ==============================================================================
# PART 3 — GATING LOGIC
# ==============================================================================
def gate_events(
    perm_ids_df: pd.DataFrame,
    earnings: pd.DataFrame,
    today: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Apply interval gating + 90-day buffer + §7.7 loser-overlap drop.

    Returns:
        gated_df : DataFrame with columns
            perm_id, canonical_ticker, cik, report_date,
            added, removed, calendar_week_group
            sorted by ['calendar_week_group', 'perm_id'].
        audit    : dict with key counts:
            n_perm_ids_total      — total perm_ids iterated
            n_perm_ids_skipped    — price_unavailable=True
            n_perm_ids_zero_earnings — perm_id absent from /earnings/raw
            n_perm_ids_gated_zero — perm_id had earnings, but all pruned by buffer/interval
            n_loser_overlap_propsed_drop — events dropped by §7.7
            n_buffer_drops        — events pruned by [added+90d, removed] window
            total_events_processed — total events examined
            total_events_gated    — events that survived
    """
    if today is None:
        today = pd.Timestamp.now().normalize()
    else:
        today = pd.Timestamp(today).normalize()

    buffer = pd.Timedelta(days=BUFFER_DAYS)

    earnings = earnings.copy()
    if not pd.api.types.is_datetime64_any_dtype(earnings["report_date"]):
        earnings["report_date"] = pd.to_datetime(earnings["report_date"])

    # Pre-group earnings by perm_id for O(1) lookup per perm_id.
    earnings_by_pid: dict[str, pd.DataFrame] = {
        pid: g.sort_values("report_date").reset_index(drop=True)
        for pid, g in earnings.groupby("perm_id")
    }

    # Build §7.7 loser-overlap spans.
    loser_overlap = build_loser_overlap_masks(perm_ids_df)

    audit = {
        "n_perm_ids_total": len(perm_ids_df),
        "n_perm_ids_skipped": 0,
        "n_perm_ids_zero_earnings": 0,
        "n_perm_ids_gated_zero": 0,
        "n_loser_overlap_drops": 0,
        "n_buffer_drops": 0,
        "total_events_processed": 0,
        "total_events_gated": 0,
    }

    rows: list[dict] = []
    for _, rec in perm_ids_df.iterrows():
        if bool(rec.get("price_unavailable", False)):
            audit["n_perm_ids_skipped"] += 1
            continue

        pid = rec["perm_id"]
        if pid is None or (isinstance(pid, float) and pd.isna(pid)):
            # Defensive: __nocik_ perm_ids have a string pid; only the
            # legacy fallback case could have NaN, which we skip.
            continue

        canonical = rec["canonical_ticker"]
        cik = rec.get("cik", None)
        if isinstance(cik, float) and pd.isna(cik):
            cik = None

        comp_earn = earnings_by_pid.get(pid)
        if comp_earn is None or comp_earn.empty:
            audit["n_perm_ids_zero_earnings"] += 1
            continue
        audit["total_events_processed"] += len(comp_earn)

        # §7.7 loser-overlap drop (if applicable).
        ov = loser_overlap.get(pid)
        comp_earn_pre_drop = len(comp_earn)
        if ov is not None:
            ov_lo, ov_hi = ov
            in_overlap = (comp_earn["report_date"] >= ov_lo) & (comp_earn["report_date"] <= ov_hi)
            audit["n_loser_overlap_drops"] += int(in_overlap.sum())
            comp_earn = comp_earn.loc[~in_overlap]

        intervals = _parse_intervals(rec["combined_intervals"])

        n_gated_for_this_pid = 0
        for itv in intervals:
            added_at = itv.get("added")
            removed_at = itv.get("removed")

            if added_at is None or (isinstance(added_at, float) and pd.isna(added_at)):
                # Defensive: null-added interval should not occur post-02b backfill.
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
                    "perm_id": pid,
                    "canonical_ticker": canonical,
                    "cik": cik,
                    "report_date": pd.Timestamp(ev_date),
                    "added": added_dt,
                    "removed": removed_dt,
                    "calendar_week_group": f"{iso.year}-W{iso.week:02d}",
                })
                n_gated_for_this_pid += 1

        if n_gated_for_this_pid == 0:
            audit["n_perm_ids_gated_zero"] += 1
            # If this perm_id had events but NONE survived gating, count it
            # separately so the audit can distinguish "no raw earnings at
            # all" from "earnings existed but all pruned".

    df = pd.DataFrame(
        rows,
        columns=[
            "perm_id", "canonical_ticker", "cik", "report_date",
            "added", "removed", "calendar_week_group",
        ],
    )
    if not df.empty:
        df = df.sort_values(["calendar_week_group", "perm_id"]).reset_index(drop=True)

    audit["total_events_gated"] = len(df)
    return df, audit


# ==============================================================================
# PART 4 — STORAGE
# ==============================================================================
def write_gated_events(df: pd.DataFrame, key: str = GATED_EVENTS_KEY) -> None:
    """Persist the gated events frame to DB_FILE under /features/gated_events.

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
            data_columns=["calendar_week_group", "perm_id", "canonical_ticker"],
        )


# ==============================================================================
# PART 5 — WEEKLY DISTRIBUTION PROFILE
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


def print_audit(audit: dict) -> None:
    bar = "-" * 64
    print(bar)
    print("STAGE 1 — GATING AUDIT (Phase E)")
    print(bar)
    print(f"  perm_ids total               : {audit['n_perm_ids_total']}")
    print(f"  perm_ids skipped (unavail)    : {audit['n_perm_ids_skipped']}")
    print(f"  perm_ids zero raw earnings    : {audit['n_perm_ids_zero_earnings']}")
    print(f"  perm_ids gated-to-zero        : {audit['n_perm_ids_gated_zero']}")
    print(f"  events processed              : {audit['total_events_processed']}")
    print(f"  events dropped by §7.7 overlap : {audit['n_loser_overlap_drops']}")
    print(f"  events dropped by buffer/window: {audit['n_buffer_drops']}")
    print(f"  events gated (survived)        : {audit['total_events_gated']}")
    print(bar)


# ==============================================================================
# MAIN
# ==============================================================================
def main(dry_run: bool = False) -> int:
    bar = "=" * 64
    print(bar)
    print("STAGE 1 — Gated Events Builder  [Phase E rewrite]")
    print(f"DB file: {DB_FILE}")
    print(f"Output key: {GATED_EVENTS_KEY}  {'(dry-run, no write)' if dry_run else ''}")
    print(bar)

    perm_ids_df = load_perm_ids()
    earnings = load_earnings_raw()
    print(
        f"Loaded {len(perm_ids_df):,} perm_ids, {len(earnings):,} raw earnings events."
    )

    gated_df, audit = gate_events(perm_ids_df, earnings)
    print(f"Gated events: {len(gated_df):,}")

    print_audit(audit)

    if len(gated_df) == 0:
        print("(empty gated set — nothing to do)")
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
        description="Stage 1: gate S&P 400 perm_id earnings events to eligible "
                    "weekly cohorts for the listwise ranker, then profile the "
                    "weekly distribution. (Phase E rewrite: perm_id-keyed, "
                    "§7.7 disambiguation applied.)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only profile; do NOT write /features/gated_events to db.h5.",
    )
    args = parser.parse_args()
    raise SystemExit(main(dry_run=args.dry_run))
