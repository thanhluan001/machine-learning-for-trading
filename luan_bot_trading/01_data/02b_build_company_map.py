#!/usr/bin/env python3
"""02b - Build Company-Level Map (CIK-anchored canonical ticker merge)
====================================================================

Produces a company-level view of the S&P 400 historical universe so the
PEAD feature builder (and `03_data_gathering.py`) can treat each **company**
as one entity, regardless of ticker renames/rebrands/bankruptcy-Q-suffixes.

Inputs (from db.h5):
    /metadata/sp400   per-ticker rows with columns:
                      ticker, name, gics_sector, gics_sub_industry,
                      intervals (JSON list of {added, removed|None}),
                      sic, index_ref

Outputs (back into db.h5):
    /metadata/sp400   EXTENDED with two new columns:
                      cik              SEC CIK (10-digit string) or None
                      canonical_ticker the canonical ticker for this alias's company
    /metadata/sp400_companies   NEW table, one row per company (CIK):
        canonical_ticker        str
        cik                     str or None (None for singletons)
        aliases                 JSON list[str]  (canonical first, then verified,
                                                then unverified)
        name                    str (best-available)
        sic                     str (from canonical ticker's row)
        index_ref               str
        combined_intervals      JSON list[{added, removed|None}] merged spans
                                (overlapping/abutting -> span; >7d gap -> 2 spans)
        per_ticker_intervals    JSON dict[ticker -> list[{added,removed|None}]]
                                (audit trail of original per-ticker intervals)
        price_unavailable       bool  (True if no alias verified on Tiingo)

Also resets `stock_offset.txt` to 0 (the next `03_data_gathering.py` run
iterates the new per-company space; the checkpoint switches from per-ticker
to per-company index space).

See `luan_bot_trading/01_data/company_merge_design.md` for the full design.
"""

import json
import os
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

_load_env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(_load_env_path)

# Load 02_SEC_sector_gathering utilities via runpy (the leading-digit filename
# makes normal `import` impossible). This exposes `build_ticker_to_cik_history`
# without re-running its `main()`.
import runpy

_SEC_MODULE_NS = runpy.run_path(
    str(Path(__file__).parent / "02_SEC_sector_gathering.py"),
    run_name="_02_module_only",
)
build_ticker_to_cik_history = _SEC_MODULE_NS["build_ticker_to_cik_history"]

DB_FILE = Path(__file__).parent / "db.h5"
OFFSET_FILE = Path(__file__).parent / "stock_offset.txt"
META_KEY = "/metadata/sp400"
COMPANIES_KEY = "/metadata/sp400_companies"

TIINGO_API_KEY = os.getenv("TIINGO_API_KEY")
EODHD_API_KEY = os.getenv("EODHD_API_KEY")
if not EODHD_API_KEY:
    raise ValueError(
        "EODHD_API_KEY not found in .env. The 02b availability probe uses "
        "EODHD (paid subscription, 100k/day, 1000/min) so we don't consume "
        "the 50 req/hour Tiingo free-tier limit during probing."
    )
# Abutting intervals within this many days are merged into one span.
ABUT_DAYS = 7
# EODHD probe throttle. EODHD limit is 1000/min ~= 16/sec, so 0.05s sleep
# keeps us comfortably under that and finishes ~993 tickers in <1 minute.
EODHD_PROBE_DELAY = 0.05
# Per-ticker probe window length (days). Each ticker is probed over a
# window ending at its latest-known-active date (removed date from
# /metadata/sp400, or today if removed is null). A small window keeps
# responses minimal; 30 days is enough to ensure at least a few trading
# days exist even for tickers that had irregular daily data.
EODHD_PROBE_WINDOW_DAYS = 30

# Full-history window for the second-pass fallback probe. Used only for the
# small set of companies marked `price_unavailable=True` after the initial
# per-ticker-interval probe. Wikipedia sometimes misses a company's name
# change or exit (incomplete `removed` date, or wrong `removed` date), so the
# interval window can land on a date where the ticker had no trading data.
# Re-probing with the full 15-year window catches these cases. Trivial cost:
# 1 EODHD call per unavailable company (typically 30-60 calls), within the
# 100k/day EODHD subscription limit.
EODHD_FALLBACK_FROM = "2012-01-01"
EODHD_FALLBACK_TO = pd.Timestamp.now().strftime("%Y-%m-%d")

# Hardcoded overrides for residual cases where:
#  - CIK lookup fails entirely (singleton), or
#  - aliases of one company get split across multiple CIKs (rare).
# Values may be either a single canonical ticker (for a no-CIK singleton that
# we know maps to a Tiingo-available ticker), or a list of aliases to add to a
# company group. Seeded empty; populated iteratively after each audit run.
KNOWN_RENAMES: dict[str, object] = {}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _parse_intervals(raw) -> list[dict]:
    """Parse the JSON interval string stored in /metadata/sp400.

    Returns a list of dicts, each with keys ``added`` (str date or None) and
    ``removed`` (str date, None, or the string "None").
    """
    if raw is None:
        return []
    s = str(raw).strip()
    if s in {"", "nan", "None", "[]", "{}"}:
        return []
    try:
        data = json.loads(s)
        if not isinstance(data, list):
            return []
        out = []
        for item in data:
            if not isinstance(item, dict):
                continue
            added = item.get("added")
            removed = item.get("removed")
            # Normalize: "None" string / NaN -> None
            if isinstance(removed, str) and removed.title() == "None":
                removed = None
            if removed is not None and isinstance(removed, str) and removed.lower() in {"nan", ""}:
                removed = None
            out.append({"added": added, "removed": removed})
        return out
    except Exception:
        return []


def _to_ts(d) -> pd.Timestamp | None:
    if d is None:
        return None
    try:
        ts = pd.Timestamp(d)
        if pd.isna(ts):
            return None
        return ts
    except Exception:
        return None


def _ts_to_str(ts) -> str | None:
    if ts is None or pd.isna(ts):
        return None
    return ts.strftime("%Y-%m-%d")


def _latest_removed(intervals: list[dict]) -> pd.Timestamp | None:
    best = None
    for iv in intervals:
        r = _to_ts(iv.get("removed"))
        if r is None:
            # null removed = currently in index = treat as "open ended" = latest
            return pd.Timestamp.now().normalize()
        if best is None or r > best:
            best = r
    return best


def _latest_added(intervals: list[dict]) -> pd.Timestamp | None:
    best = None
    for iv in intervals:
        a = _to_ts(iv.get("added"))
        if a is None:
            continue
        if best is None or a > best:
            best = a
    return best


def _gap_days(left_removed: pd.Timestamp | None, right_added: pd.Timestamp | None) -> float | None:
    """Returns the gap in days between two non-null endpoints."""
    if left_removed is None or right_added is None:
        return None
    delta = (right_added - left_removed).total_seconds() / 86400.0
    return delta


def merge_intervals(intervals: list[dict]) -> list[dict]:
    """Merge overlapping or abutting interval spans into a single list.

    Abutting: gap <= ABUT_DAYS counts as continuous (rebrand continuity).
    Real gaps (> ABUT_DAYS) preserved as separate spans.
    """
    norm = []
    for iv in intervals:
        a = _to_ts(iv.get("added"))
        r = _to_ts(iv.get("removed"))
        if a is None and r is None:
            continue
        norm.append((a, r))
    if not norm:
        return []

    # Sort by added. A null 'added' (NaT) represents a historical terminal span
    # ("company was in the index from some unknown pre-history until 'removed'")
    # and should sort to the FRONT of the timeline (treated as the very
    # earliest date), so subsequent re-adds abut against it.
    norm.sort(key=lambda x: (pd.Timestamp.min if x[0] is None else x[0]))

    merged = []
    cur_a, cur_r = norm[0]
    for a, r in norm[1:]:
        # Extend current if overlap or abut (a <= cur_r + ABUT).
        if cur_r is None:
            # Open-ended current: cannot extend further.
            merged.append((cur_a, cur_r))
            cur_a, cur_r = a, r
            continue
        if a is None:
            # Next interval has no added; only its removed is known. It is a
            # legacy terminal span that should be merged into the current span
            # by using max(removed). Don't advance the cursor.
            if cur_r is None:
                # current is open-ended; nothing to extend.
                continue
            if r is None:
                cur_r = None
            elif r > cur_r:
                cur_r = r
            continue
        gap = (a - cur_r).total_seconds() / 86400.0
        if a <= cur_r or gap <= ABUT_DAYS:
            # overlap or abutting -> extend the span
            if r is None:
                cur_r = None  # extension makes span open-ended
            elif cur_r is not None and r > cur_r:
                cur_r = r
        else:
            merged.append((cur_a, cur_r))
            cur_a, cur_r = a, r
    merged.append((cur_a, cur_r))

    out = []
    for a, r in merged:
        out.append({"added": _ts_to_str(a), "removed": _ts_to_str(r)})
    return out


# ------------------------------------------------------------------
# Tiingo availability probe
# ------------------------------------------------------------------

def _latest_active_date(intervals_raw) -> pd.Timestamp | None:
    """Extract the latest active (removed or currently-in-index) date from a
    ticker's interval list in /metadata/sp400.

    Returns:
        - pd.Timestamp of latest `removed` date if any intervals have one
        - pd.Timestamp.now() if any interval has `removed` == None
          (currently in index -> probe recent dates)
        - pd.Timestamp of latest `added` date as fallback if only added known
        - None if no parseable intervals
    """
    if intervals_raw is None or (isinstance(intervals_raw, float) and pd.isna(intervals_raw)):
        return None
    try:
        data = intervals_raw
        if isinstance(data, str):
            data = json.loads(data)
        if not isinstance(data, list) or not data:
            return None
    except Exception:
        return None

    has_open = False
    best_removed = None
    best_added = None
    for iv in data:
        if not isinstance(iv, dict):
            continue
        r = iv.get("removed")
        a = iv.get("added")
        if r is None or (isinstance(r, str) and r.lower() in {"null", "none", "nan", ""}):
            has_open = True
        else:
            try:
                rts = pd.Timestamp(r)
                if not pd.isna(rts) and (best_removed is None or rts > best_removed):
                    best_removed = rts
            except Exception:
                pass
        if a is not None and not (isinstance(a, float) and pd.isna(a)):
            try:
                ats = pd.Timestamp(a)
                if not pd.isna(ats) and (best_added is None or ats > best_added):
                    best_added = ats
            except Exception:
                pass

    if has_open:
        return pd.Timestamp.now().normalize()
    if best_removed is not None:
        return best_removed
    return best_added


def _probe_window_for(latest_active: pd.Timestamp | None) -> tuple[str, str] | None:
    """Compute the (from, to) probe window ending at `latest_active`.

    Returns None if latest_active is None (no usable date).
    """
    if latest_active is None or pd.isna(latest_active):
        return None
    to_ts = latest_active
    from_ts = to_ts - pd.Timedelta(days=EODHD_PROBE_WINDOW_DAYS)
    # Clamp to not exceed today
    if to_ts > pd.Timestamp.now().normalize():
        to_ts = pd.Timestamp.now().normalize()
    if from_ts > pd.Timestamp.now().normalize():
        from_ts = pd.Timestamp.now().normalize() - pd.Timedelta(days=EODHD_PROBE_WINDOW_DAYS)
    return (from_ts.strftime("%Y-%m-%d"), to_ts.strftime("%Y-%m-%d"))


def verify_tickers_on_tiingo(tickers: list[str], progress_every: int = 50, ticker_intervals: dict[str, object] | None = None) -> dict[str, bool]:
    """Probe ticker availability via EODHD /api/eod/{TICKER}.US.

    Function name is retained for backward compatibility with the rest of
    the module; the actual probe now uses EODHD's End-Of-Day endpoint, NOT
    Tiingo. This avoids burning Tiingo's 50 req/hour free-tier quota on
    ~993 existence probes (which alone would lock us out for ~20 hours).

    Per-ticker probe window (live-verified fix):
        Delisted rebrand tickers (e.g. AAXN, APY, OZRK) returned empty arrays
        when probed over a recent window (2024-01), even though they have
        data during their actual trading era. So we compute each ticker's
        probe window from its /metadata/sp400 interval data:
          - if any interval has removed==None -> probe recent dates
            (latest_active = today)
          - else latest_active = max(removed) across intervals
          - probe window = [latest_active - 30d, latest_active]
        This catches both currently-trading and delisted tickers in one pass.

    EODHD contract (live-verified on multiple tickers):
        HTTP 404               -> ticker does not exist on EODHD -> False
        HTTP 200 + "[]" (empty)-> no historical data in window -> False
        HTTP 200 + non-empty   -> True

    Returns:
        dict[ticker -> bool] of availability.
    """
    results: dict[str, bool] = {}
    for i, t in enumerate(tickers, 1):
        if ticker_intervals is None:
            latest_active = None
        else:
            latest_active = _latest_active_date(ticker_intervals.get(t))
        window = _probe_window_for(latest_active)
        if window is None:
            # No usable date in metadata; fall back to a recent window
            # (likely catches currently-trading cases only).
            window = ("2024-01-02", "2024-01-05")
        url = f"https://eodhd.com/api/eod/{t}.US"
        try:
            r = requests.get(
                url,
                params={
                    "from": window[0],
                    "to": window[1],
                    "api_token": EODHD_API_KEY,
                    "fmt": "json",
                    "period": "d",
                },
                timeout=20,
            )
            if r.status_code == 200:
                try:
                    body = r.json()
                    results[t] = isinstance(body, list) and len(body) > 0
                except Exception:
                    results[t] = False
            elif r.status_code == 404:
                results[t] = False
            else:
                print(f"   {t}: unexpected probe status {r.status_code}: {r.text[:80]}")
                results[t] = False
        except Exception as e:
            print(f"   {t}: probe error: {e}")
            results[t] = False
        time.sleep(EODHD_PROBE_DELAY)
        if i % progress_every == 0:
            print(f"   probe progress: {i}/{len(tickers)} (avail={sum(1 for v in results.values() if v)})")
    return results


def reprobe_unavailable_canonicals(
    companies: list[dict],
    tiingo_available: dict[str, bool],
) -> tuple[list[dict], dict[str, bool]]:
    """Second-pass probe for companies marked price_unavailable=True.

    For each such company, query EODHD with the full 15-year window
    ([EODHD_FALLBACK_FROM, EODHD_FALLBACK_TO]) on the canonical ticker.
    If data comes back, flip the company's `price_unavailable` flag to
    False and update `tiingo_available` so the data-gathering downstream
    step will include the company in its iteration.

    Rationale (per user, doctored after audit):
        Wikipedia sometimes misses a company's name change / exit date.
        The initial probe uses the per-ticker interval window, which can
        land on a date range where no trading data exists even though the
        ticker was clearly trading at some point in the 15-year window.
        With EODHD's 100k/day quota, doing one more call per unavailable
        company is trivial and recovers these cases.

    Args:
        companies: list of company-row dicts from build_company_map().
        tiingo_available: ticker -> bool availability map (mutated in place
            for recovered canonicals).

    Returns:
        (companies, tiingo_available) -- same objects, mutated in place
        for convenience.
    """
    unavail = [c for c in companies if c.get("price_unavailable")]
    if not unavail:
        return companies, tiingo_available

    print(
        f"\n[3b/4] Re-probing {len(unavail)} unavailable canonicals with the full "
        f"15-year window ({EODHD_FALLBACK_FROM}..{EODHD_FALLBACK_TO})..."
    )
    recovered = 0
    for i, c in enumerate(unavail, 1):
        canonical = c["canonical_ticker"]
        url = f"https://eodhd.com/api/eod/{canonical}.US"
        try:
            r = requests.get(
                url,
                params={
                    "from": EODHD_FALLBACK_FROM,
                    "to": EODHD_FALLBACK_TO,
                    "api_token": EODHD_API_KEY,
                    "fmt": "json",
                    "period": "d",
                },
                timeout=30,
            )
            ok = False
            if r.status_code == 200:
                try:
                    body = r.json()
                    ok = isinstance(body, list) and len(body) > 0
                except Exception:
                    pass
            if ok:
                c["price_unavailable"] = False
                tiingo_available[canonical] = True
                recovered += 1
                print(f"   recovered: {canonical}")
        except Exception as e:
            print(f"   {canonical}: fallback probe error: {e}")
        time.sleep(EODHD_PROBE_DELAY)
    print(
        f"   fallback recovered {recovered} / {len(unavail)} companies. "
        f"Still unavailable: {len(unavail) - recovered}."
    )
    return companies, tiingo_available


# ------------------------------------------------------------------
# Core algorithm
# ------------------------------------------------------------------

def build_company_map(meta_df: pd.DataFrame, tiingo_available: dict[str, bool]) -> tuple[pd.DataFrame, list[dict]]:
    """Group tickers by CIK, pick canonical, merge intervals.

    Returns:
        extended_meta: per-ticker view, with cik + canonical_ticker columns added
        companies: list of company-row dicts (will become /metadata/sp400_companies)
    """
    active_sec_tickers = build_ticker_to_cik_history()

    # Step 1 - assign CIK per ticker
    ticker_to_cik: dict[str, str | None] = {}
    for ticker in meta_df["ticker"].tolist():
        ck = active_sec_tickers.get(ticker.upper().strip())
        if not ck:
            # KNOWN_RENAMES fallback (singleton canonical merge)
            kn = KNOWN_RENAMES.get(ticker.upper().strip()) if isinstance(KNOWN_RENAMES, dict) else None
            # KNOWN_RENAMES entries may be {canonical_or_aliases}; not affecting CIK lookup here.
            if kn is not None:
                # If the entry tells us the canonical ticker, allow it to inherit that
                # canonical's CIK (looked-up below). Otherwise None.
                cl = kn if isinstance(kn, str) else (kn[0] if isinstance(kn, list) and kn else None)
                ck = active_sec_tickers.get(cl) if cl else None
        ticker_to_cik[ticker] = ck if ck else None

    # Step 2 - group by CIK (singletons share a synthetic key)
    groups: dict[str, list[str]] = defaultdict(list)
    singleton_counter = 0
    for ticker in meta_df["ticker"].tolist():
        cik = ticker_to_cik.get(ticker)
        if cik is None:
            groups[f"__single_{singleton_counter}"].append(ticker)
            singleton_counter += 1
        else:
            groups[cik].append(ticker)

    # Step 3 & 4 - per group: pick canonical + merge intervals
    meta_indexed = meta_df.set_index("ticker", drop=False)
    companies: list[dict] = []
    ticker_to_canonical: dict[str, str] = {}

    n_with_cik = sum(1 for t in meta_df["ticker"] if ticker_to_cik.get(t))
    n_merged_groups = 0

    for cik, aliases in groups.items():
        # Sort aliases alphabetically for determinism (canonical selection picks one out)
        aliases_sorted = sorted(aliases, key=str.upper)

        # Compute per-alias parsed intervals
        per_ticker_intervals = {}
        for t in aliases_sorted:
            row = meta_indexed.loc[t]
            iv = _parse_intervals(row.get("intervals"))
            per_ticker_intervals[t] = iv

        # All intervals across aliases
        all_intervals = [iv for ivs in per_ticker_intervals.values() for iv in ivs]
        combined = merge_intervals(all_intervals)

        # Step 3: canonical-selection priority
        # Priority 1: ticker in active_sec_tickers AND Tiingo-verified
        # Priority 2: ticker with most-recent removed (or open-ended) AND Tiingo-verified
        # Priority 3: most-recently-added ticker regardless of Tiingo
        def _info(t):
            iv = per_ticker_intervals[t]
            return {
                "active_sec": t.upper().strip() in active_sec_tickers,
                "latest_removed": _latest_removed(iv),
                "latest_added": _latest_added(iv),
                "tiingo_ok": bool(tiingo_available.get(t, False)),
                "ticker": t,
            }

        infos = [_info(t) for t in aliases_sorted]

        tiingo_verified = [i for i in infos if i["tiingo_ok"]]
        canonical = None
        if tiingo_verified:
            # Tier 1: active_sec AND tiingo Ok
            tier1 = [i for i in tiingo_verified if i["active_sec"]]
            if tier1:
                canonical = sorted(tier1, key=lambda i: (i["ticker"].upper()))[0]["ticker"]
            else:
                # Tier 2: most-recent removed AND tiingo Ok
                tier2 = sorted(tier2 := tiingo_verified, key=lambda i: (
                    i["latest_removed"] if i["latest_removed"] is not None else pd.Timestamp.min
                ), reverse=True)
                canonical = tier2[0]["ticker"]

        price_unavailable = False
        if canonical is None:
            # Tier 3: most-recently-added ticker among ALL aliases regardless of Tiingo
            sorted_added = sorted(infos, key=lambda i: (
                i["latest_added"] if i["latest_added"] is not None else pd.Timestamp.min
            ), reverse=True)
            canonical = sorted_added[0]["ticker"]
            price_unavailable = not bool(tiingo_available.get(canonical, False))

        # Alias ordering for fetching: canonical first, then other Tiingo-verified,
        # then unverified (alphabetical within each block).
        other_verified = sorted(
            [i["ticker"] for i in infos if i["tiingo_ok"] and i["ticker"] != canonical],
            key=str.upper,
        )
        unverified = sorted(
            [i["ticker"] for i in infos if not i["tiingo_ok"] and i["ticker"] != canonical],
            key=str.upper,
        )
        ordered_aliases = [canonical] + other_verified + unverified

        # Pick name from the canonical ticker's row, falling back to any alias.
        name = meta_indexed.loc[canonical, "name"]
        if pd.isna(name) or name == "":
            for t in aliases_sorted:
                nm = meta_indexed.loc[t, "name"]
                if not pd.isna(nm) and nm != "":
                    name = nm
                    break

        canonical_row = meta_indexed.loc[canonical]
        companies.append({
            "canonical_ticker": canonical,
            "cik": None if cik.startswith("__single") else cik,
            "aliases": ordered_aliases,
            "name": name,
            "sic": canonical_row.get("sic") if not pd.isna(canonical_row.get("sic")) else None,
            "index_ref": canonical_row.get("index_ref") if not pd.isna(canonical_row.get("index_ref")) else None,
            "combined_intervals": combined,
            "per_ticker_intervals": per_ticker_intervals,
            "price_unavailable": bool(price_unavailable),
        })
        for t in aliases_sorted:
            ticker_to_canonical[t] = canonical
        if len(aliases_sorted) > 1:
            n_merged_groups += 1

    # Build extended per-ticker view
    extended = meta_df.copy()
    extended["cik"] = extended["ticker"].map(lambda t: ticker_to_cik.get(t))
    extended["canonical_ticker"] = extended["ticker"].map(lambda t: ticker_to_canonical[t])

    print(f"   CIK-found tickers: {n_with_cik} / {len(meta_df)}")
    print(f"   Company groups:    {len(companies)}")
    print(f"   Multi-alias groups: {n_merged_groups}")
    return extended, companies


# ------------------------------------------------------------------
# Output writing
# ------------------------------------------------------------------

def _json_dumps(obj) -> str:
    # NaN-safe JSON serialization; ints/bools preserved.
    return json.dumps(obj, default=str)


def write_outputs(extended_meta: pd.DataFrame, companies: list[dict]) -> None:
    """Persist both tables to db.h5 using the HDFStore('a') + remove() pattern.

    NEVER use mode='w' on an existing database (it truncates the whole file).
    """
    # Normalize JSON-friendly fields on the companies table.
    companies_df = pd.DataFrame(companies)
    if not companies_df.empty:
        companies_df["aliases"] = companies_df["aliases"].apply(lambda v: _json_dumps(list(v) if v else []))
        companies_df["combined_intervals"] = companies_df["combined_intervals"].apply(
            lambda v: _json_dumps(v if v else [])
        )
        companies_df["per_ticker_intervals"] = companies_df["per_ticker_intervals"].apply(
            lambda v: _json_dumps(v if v else {})
        )
        # cik can be None -> store as appropriate dtype
        companies_df["cik"] = companies_df["cik"].astype(object)

    with pd.HDFStore(DB_FILE, mode="a") as store:
        # Replace /metadata/sp400
        if META_KEY in store:
            store.remove(META_KEY)
        store.put(META_KEY, extended_meta, format="table")
        print(f" Wrote {META_KEY} ({len(extended_meta)} rows)")

        # Replace /metadata/sp400_companies
        if COMPANIES_KEY in store:
            store.remove(COMPANIES_KEY)
        if not companies_df.empty:
            store.put(COMPANIES_KEY, companies_df, format="table")
            print(f" Wrote {COMPANIES_KEY} ({len(companies_df)} rows)")


def reset_offset() -> None:
    """Reset stock_offset.txt to 0 (the per-company iteration starts fresh)."""
    with open(OFFSET_FILE, "w") as f:
        f.write("0")
    print(f" Reset {OFFSET_FILE.name} to 0 (per-company checkpoint space)")


# ------------------------------------------------------------------
# Audit report
# ------------------------------------------------------------------

def print_audit(meta_df: pd.DataFrame, companies: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("  COMPANY MERGE AUDIT")
    print("=" * 70)

    n_tickers = len(meta_df)
    n_cik = sum(1 for _, r in meta_df.iterrows() if r.get("cik"))
    print(f" Total tickers in /metadata/sp400 : {n_tickers}")
    print(f" Tickers with CIK found          : {n_cik} / {n_tickers}")
    print(f" Companies (groups)              : {len(companies)}")
    multi = [c for c in companies if len(c["aliases"]) > 1]
    print(f" Multi-alias companies            : {len(multi)}")

    if multi:
        print("\n --- Merged companies (canonical <- aliases) ---")
        for c in multi:
            cik_disp = c["cik"] if c["cik"] is not None else "<singleton>"
            print(f"   {c['canonical_ticker']:6}  (CIK {cik_disp}, {len(c['aliases'])} aliases) <-  "
                  f"{', '.join(c['aliases'])}")

    unavailable = [c for c in companies if c["price_unavailable"]]
    print(f"\n Companies with price_unavailable=True: {len(unavailable)}")
    if unavailable:
        print("   These will be skipped by 03_data_gathering.py. Add to KNOWN_RENAMES")
        print("   in 02b if a Tiingo-available alias should be preferred.")
        for c in unavailable:
            cik_disp = c["cik"] if c["cik"] is not None else "<singleton>"
            print(f"   - {c['canonical_ticker']:6} (CIK {cik_disp}) aliases={c['aliases']} sic={c['sic']}")

    print("=" * 70 + "\n")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  02b - Build Company Map (CIK-anchored canonical ticker merge)")
    print("=" * 70)

    if not DB_FILE.exists():
        print(f"[FATAL] {DB_FILE} not found. Run 01_metadata_gathering.py and")
        print("        02_SEC_sector_gathering.py first.")
        return

    print("\n[1/4] Loading /metadata/sp400 ...")
    meta_df = pd.read_hdf(DB_FILE, key=META_KEY)
    if "ticker" not in meta_df.columns:
        meta_df = meta_df.reset_index()
    print(f"   Loaded {len(meta_df)} tickers")

    print("\n[2/4] Building ticker -> CIK history map (current SEC + DERA snapshots)...")
    # done inside build_company_map; just declare intent here
    print("   (will use union of ticker.txt + cached DERA sub_*.txt snapshots)")

    print("\n[3/4] Probing ticker availability via EODHD (not Tiingo; saves 50/hr quota)...")
    all_tickers = sorted(set(meta_df["ticker"].astype(str).tolist()), key=str.upper)
    # Pass per-ticker interval data so the probe window is computed from each
    # ticker's latest-active date (critical for delisted tickers like ASNA,
    # OZRK, GMCR which would falsely return [] if probed against recent dates).
    ticker_intervals = dict(zip(meta_df["ticker"].astype(str), meta_df["intervals"]))
    tiingo_av = verify_tickers_on_tiingo(all_tickers, ticker_intervals=ticker_intervals)
    avail_ct = sum(1 for v in tiingo_av.values() if v)
    print(f"   {avail_ct} / {len(all_tickers)} tickers available on EODHD")

    print("\n[4/4] Building company groups, merging intervals ...")
    extended, companies = build_company_map(meta_df, tiingo_av)

    # Second-pass fallback: re-probe unavailable canonicals with the full
    # 15-year window. Recovers companies whose per-ticker interval probe
    # window landed on an empty-data range due to incomplete Wikipedia
    # metadata (missing/wrong removed date).
    reprobe_unavailable_canonicals(companies, tiingo_av)

    print("\n Writing outputs to db.h5 ...")
    write_outputs(extended, companies)
    reset_offset()

    print_audit(extended, companies)


if __name__ == "__main__":
    main()
