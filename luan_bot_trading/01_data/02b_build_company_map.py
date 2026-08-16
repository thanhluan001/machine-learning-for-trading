#!/usr/bin/env python3
"""02b - Build Company-Level Map (Tiingo permaTicker-anchored)
============================================================

Phase A rewrite per `luan_bot_trading/01_data/tiingo_permaTicker_audit.md`.

SUPersedes the previous `perm_id`-anchored, CIK-synthesis-based algorithm
(that ~1300-line Wikipedia+DERA+point-in-time-CIK code is GONE). The new
primary entity identifier is Tiingo's `permaTicker`, which is identity-stable
across rebrands, mergers, delistings, and same-CIK reorgs / spinoffs (verified
via ~50 live probe API calls; see the audit doc for evidence). Tiingo
back-merges a company's full rebrand-covered history under the permaTicker
key, so we no longer need alias-concatenation, §7.7 disambiguation rules, or
point-in-time CIK lookup.

Inputs (from db.h5):
    /metadata/sp400   Wikipedia per-ticker rows (already populated):
                      ticker, name, gics_sector, gics_sub_industry,
                      intervals (JSON list of {added, removed|None}),
                      sic, index_ref, + legacy cols (cik, perm_id, ...)
                      -- legacy cols carried forward ONLY as cross-reference
                         for Phase D re-keying (legacy_perm_id column).

External (Tiingo paid tier):
    GET /tiingo/utilities/search/{ticker}?includeDelisted=true&exactTickerMatch=true
        Returns list of permaTickers historically held by that ticker code,
        with fields: ticker, name, permaTicker, openFIGIComposite, isActive,
        assetType, countryCode, startDate, endDate.
    GET /tiingo/daily/{permaTicker}/prices?startDate=...&endDate=...
        Name-sanity check: verifies real price rows exist for this permaTicker
        in our Wikipedia interval window. Empty -> price_unavailable=True.

Outputs (back into db.h5):
    /metadata/sp400_permatickers   NEW (replaces /metadata/sp400_perm_ids):
        permaTicker         str     PRIMARY KEY (Tiingo's identity-stable ID)
        canonical_ticker    str     result.ticker from the chosen search hit,
                                    used by EODHD calendar join downstream.
        name                str     from search response
        isActive            bool    from search response (currently trading?)
        openfigi            str     Bloomberg OpenFIGI (defensive redundancy)
        cik                 str     carried from old /metadata/sp400 row
                                    (informational only; SIC sector lookup
                                    uses canonical_ticker, NOT cik).
        sic                 str     carried from old /metadata/sp400 row
        index_ref           str     carried from old /metadata/sp400 row
        wikipedia_intervals JSON list[{added, removed|None}]
                            (point-in-time S&P 400 residency spans tracked
                            by Wikipedia; permaTicker tracks entity identity)
        price_unavailable   bool    True if Tiingo /prices returns 0 rows in
                                    the interval window (name-sanity failure;
                                    surfaces Class-W/S variants organically)

See `luan_bot_trading/01_data/tiingo_permaTicker_audit.md` for the full
design + probe evidence (also the AUTHORITATIVE reference doc; the other
.md files carry deprecation banners pointing back to it).
"""

import json
import sys
import io
import time
from pathlib import Path

# Windows console defaults to cp1252 -- force UTF-8 so any non-ASCII ticker
# names in Tiingo search responses don't crash the print() statements below.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import pandas as pd
import requests
from dotenv import load_dotenv

_load_env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(_load_env_path)

import os

DB_FILE = Path(__file__).parent / "db.h5"
META_KEY = "/metadata/sp400"
PERMATICKERS_KEY = "/metadata/sp400_permatickers"
# Keys removed during this run: legacy perm_id table + legacy companies table.
LEGACY_PERM_IDS_KEY = "/metadata/sp400_perm_ids"
LEGACY_COMPANIES_KEY = "/metadata/sp400_companies"

TIINGO_API_KEY = os.getenv("TIINGO_API_KEY")
if not TIINGO_API_KEY:
    raise ValueError(
        "TIINGO_API_KEY not found in .env. Phase A's permaTicker discovery "
        "uses the Tiingo search + /prices endpoints (paid tier 10k/hr)."
    )
EODHD_API_KEY = os.getenv("EODHD_API_KEY")  # informational; not strictly required here.

# Tiingo rate limit on paid tier: 10000 req/hour, 100000 req/day. We use
# a tiny polite delay to avoid request bursts but it is NOT load-bearing
# (deliberately -- we have plenty of headroom on the paid plan).
TIINGO_DELAY = 0.02

# Name-sanity probe window: when we fetch /prices head for a permaTicker,
# we use [added - 7d, added + 30d] inside the Wikipedia interval. A 30-day
# window is more than enough to detect whether the permaTicker has real
# historical data for this interval (Class-W/S cases produce 0 rows).
PROBE_BEFORE_DAYS = 7
PROBE_AFTER_DAYS = 30

# When the narrow probe returns 0 rows, we re-probe with a wider 1-year
# window around the Wikipedia added_date -- the permaTicker may own data
# starting some months after added_date (e.g. GOOG's Class C permaTicker
# starts in 2014 even though Wikipedia tracks GOOG back to 2010; the
# added_date与现实 active start can differ by weeks or months).
WIDE_PROBE_BEFORE_DAYS = 90
WIDE_PROBE_AFTER_DAYS = 275

# Max number of permaTicker candidates we inspect per search call to
# disambiguate. (Tiingo search returns multiple permaTickers for ticker
# codes that have been recycled -- typically 1-10. We never need to scan
# far; cap at 50 to bound log noise if a ticker code has many holders.)
MAX_CANDIDATES_PER_SEARCH = 50


# ------------------------------------------------------------------
# Interval helpers (carried over from previous implementation -- still
# useful for parsing the JSON intervals column in /metadata/sp400)

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


def _parse_intervals(raw) -> list[dict]:
    """Parse the JSON interval string stored in /metadata/sp400.

    Returns a list of dicts, each with keys ``added`` (str date or None) and
    ``removed`` (str date, None, or the string "None").
    """
    if raw is None:
        return []
    s = raw
    if isinstance(s, float) and pd.isna(s):
        return []
    if not isinstance(s, str):
        try:
            s = json.dumps(s, default=str)
        except Exception:
            return []
    s = s.strip()
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
            if isinstance(removed, str) and removed.title() == "None":
                removed = None
            if removed is not None and isinstance(removed, str) and removed.lower() in {"nan", ""}:
                removed = None
            out.append({"added": added, "removed": removed})
        return out
    except Exception:
        return []


# ------------------------------------------------------------------
# Tiingo API helpers
# ------------------------------------------------------------------

_TIINGO_HEADERS = {"Content-Type": "application/json"}


def tiingo_search_ticker(ticker: str, *, verbose: bool = True, retries: int = 4) -> list[dict]:
    """Call Tiingo /utilities/search/{ticker} with includeDelisted + exactTicker.

    Returns the list of result dicts (typically a few, sometimes up to ~50 for
    ticker codes that have been recycled many times). Each dict carries:
        ticker, name, permaTicker, openFIGIComposite, isActive, assetType,
        countryCode, startDate, endDate, ...

    On HTTP error returns []. Caller logs + skips. Retries transient 5xx
    (Tiingo's search endpoint intermittently returns 502 Bad Gateway).
    """
    url = f"https://api.tiingo.com/tiingo/utilities/search/{requests.utils.quote(ticker)}"
    params = {
        "token": TIINGO_API_KEY,
        "includeDelisted": "true",
        "exactTickerMatch": "true",
    }
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=_TIINGO_HEADERS, timeout=20)
            if r.status_code >= 500 and attempt < retries - 1:
                if verbose:
                    print(f"   [search] {ticker}: HTTP {r.status_code}, retry {attempt+1}/{retries}")
                time.sleep(2.0 * (attempt + 1))
                continue
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, list):
                return []
            return data[:MAX_CANDIDATES_PER_SEARCH]
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                if verbose:
                    print(f"   [search] {ticker}: error {e}, retry {attempt+1}/{retries}")
                time.sleep(2.0 * (attempt + 1))
            else:
                if verbose:
                    print(f"   [search] {ticker}: error {e} (gave up after {retries})")
    return []


def tiingo_prices_head(permaTicker: str, start: str, end: str, *, verbose: bool = True) -> list[dict]:
    """Fetch a narrow /prices window for a permaTicker and return the raw rows.

    Used as the name-sanity check: confirms the chosen permaTicker actually
    has real historical rows covering our Wikipedia interval window. A
    permaTicker with 0 rows in the [added-ProbeBefore, added+ProbeAfter]
    window flags the row as price_unavailable=True (Class-W-style case
    where the chosen permaTicker doesn't actually own price history for
    the historical interval we expect it to). Additionally, the rows are
    used for the "physical-data sanity" heuristic during disambiguation:`
    back-adjusted blast-through rows (where adjClose / close is huge --
    see Class-S SUNE case for US000000002062's 2015 adjClose=$3M) are
    considered NOT the real historical owner of that interval.
    """
    url = f"https://api.tiingo.com/tiingo/daily/{requests.utils.quote(permaTicker)}/prices"
    params = {"token": TIINGO_API_KEY, "startDate": start, "endDate": end}
    try:
        r = requests.get(url, params=params, headers=_TIINGO_HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, list) else []
    except Exception as e:
        if verbose:
            print(f"   [prices] {permaTicker}: error {e}")
        return []


def _prices_sanity_score(rows: list[dict]) -> tuple[bool, float]:
    """Score a /prices row-list for "physical-data" plausibility.

    Returns (is_sane, score). is_sane=True means the rows look like real
    market quotes (not back-adjusted blast-through junk). score is a
    sortable penalty; lower = more physical.

    Heuristic rationale (from SUNE Class-S probe):
      - Real Sunedison 2015 rows (US000000002709): close=$19.71,
        adjClose=$19.71, ratio=1.0.
      - Real-but-back-adjusted SUNation 2022 rows (US000000002062):
        close=$2.03, adjClose=$304,500 (the SUNation permaTicker
        carries CSII->PEGY->SUNE back-merged serial splits; the close
        is real, the adjClose is Tiingo's accumulated back-adjustment).

    IMPORTANT: This is the OVERALL sanity of the entire row batch (close
    bounded). The PER-row "physical ratio" heuristic for adrClose/close
    is computed separately by "_physical_row_count" below -- it is the
    real disambiguator for cases like SUNE 2012 (SunEdison rows are
    all ratio=1.0, SUNation rows have ratio~150000x in 2012 era).
    """
    if not rows:
        return False, float("inf")
    import math
    import statistics
    closes = []
    max_close = 0.0
    for r in rows:
        try:
            c = float(r.get("close", 0) or 0)
        except Exception:
            continue
        if c <= 0:
            continue
        max_close = max(max_close, c)
        closes.append(c)
    if not closes:
        return False, float("inf")
    try:
        med_close = statistics.median(closes)
    except Exception:
        return False, float("inf")
    sane = (0 < med_close < 100_000) and (max_close < 100_000)
    score = abs(math.log10(max(med_close, 1e-9)) - 1.5)  # 1.5 ~= log10($31.6)
    if max_close >= 100_000:
        score += 10.0
    if not sane:
        score += 100.0
    return sane, score

def _physical_row_count(rows: list[dict]) -> int:
    """Count rows whose adjClose/close ratio is in a "physical" range.

    This is the disambiguator for cases where two permaTickers BOTH probed
    non-empty in the Wikipedia interval, but ONE of them has normal
    close-adjClose relationships (real-era data) while the other has
    back-adjusted blast-through (adjClose exploded by accumulated reverse
    splits through the permaTicker's chain). Returns the count of rows in
    the probe window whose adjClose/close ratio is in [0.001, 1000] -- the
    one with MORE such rows is the real owner of the historical era.

    Heuristic source: SUNE 2012 probe. SunEdison rows have ratio=1.0
    (24/24 physical), SUNation 2011-2012 rows have ratio=3310919/13 =
    ~150000x (0/24 physical). The Wikipedia interval was 2012-2016 --
    SunEdison is the real owner for that interval.
    """
    if not rows:
        return 0
    n = 0
    for r in rows:
        try:
            c = float(r.get("close", 0) or 0)
            ac = float(r.get("adjClose", 0) or 0)
        except Exception:
            continue
        if c <= 0 or ac <= 0:
            continue
        ratio = ac / c
        if 0.001 <= ratio <= 1000.0:
            n += 1
    return n


# ------------------------------------------------------------------
# PermaTicker disambiguation
# ------------------------------------------------------------------

def _safe_iso(v) -> str | None:
    """Coerce a search-response date field to a YYYY-MM-DD string, or None."""
    if v is None or v == "":
        return None
    ts = _to_ts(v)
    return _ts_to_str(ts)


def _to_ts_or_none(v) -> pd.Timestamp | None:
    return _to_ts(v)


def _us_stock_candidate(hit: dict) -> bool:
    """Filter search hits to US-listed equities (not funds, not foreign)."""
    if str(hit.get("assetType", "")).strip().lower() not in {"stock", ""}:
        # Allow missing assetType but reject clearly-fund/ETF hits.
        at = str(hit.get("assetType", "")).strip().lower()
        if at and at not in {"stock"}:
            return False
    cc = str(hit.get("countryCode", "")).strip().upper()
    return cc == "US"


def disambiguate_permaTicker(
    ticker: str,
    candidates: list[dict],
    added_date: pd.Timestamp | None,
    removed_date: pd.Timestamp | None,
) -> tuple[dict | None, str | None, int, list[dict]]:
    """Pick the single permaTicker that owns the (added_date, removed_date)
    Wikipedia interval window for the given ticker code.

    Strategy (probe-based; search responses don't include startDate/endDate
    so we MUST probe /prices for each candidate): For each US-stock candidate
    permaTicker, fetch /prices for the Wikipedia-interval probe window
    [added - PROBE_BEFORE_DAYS, added + PROBE_AFTER_DAYS] and score the
    rows for "physical-data" plausibility. Choose the candidate whose
    rows are SANE and most physical.

    Tie-breakers when multiple candidates probe as sane:
        a) Prefer isActive=False (typical for the historical holder --
           the modern recycler is isActive=True). For ticker codes shared
           by distinct entities (SUNE, NSR, SAI), the historical entity
           is typically inactive and the modern recycler is active.
        b) Lower penalty score (rows whose adjClose/close is close to 1.0,
           max(close) bounded).

    Returns (chosen_hit, reason_str, n_probe_rows, probe_rows).
    chosen_hit is None only if candidates is empty after filtering.
    """
    if not candidates:
        return None, "no_candidates", 0, []
    us_hits = [h for h in candidates if _us_stock_candidate(h)]
    if not us_hits:
        return None, "no_us_candidates", 0, []

    # Probe prices for each US-stock candidate in the Wikipedia window.
    # First pass: narrow window.
    start_iso, end_iso = _probe_window(added_date)
    wide_start_iso, wide_end_iso = _probe_window(added_date, wide=True)
    scored = []
    for h in us_hits:
        pt = h.get("permaTicker")
        if pt is None or not isinstance(pt, str) or pt.strip() == "":
            continue
        rows = tiingo_prices_head(pt, start_iso, end_iso, verbose=False)
        time.sleep(TIINGO_DELAY)
        # If narrow probe returns 0 rows, try wide probe before scoring.
        if not rows:
            rows = tiingo_prices_head(pt, wide_start_iso, wide_end_iso, verbose=False)
            time.sleep(TIINGO_DELAY)
        sane, score = _prices_sanity_score(rows)
        if h.get("isActive") is True:
            score += 0.5
        scored.append((h, sane, score, rows))

    # Keep only candidates that actually returned sane rows.
    sane_hits = [x for x in scored if x[1]]
    if sane_hits:
        # PRIMARY TIE-BREAKER: count of "physical" rows (adjClose/close ratio
        # in [0.001, 1000]). This is crucial for cases like SUNE 2012 where
        # both SunEdison AND SUNation returned sane-typed rows in the probe
        # window, but SUNation's historical rows are back-adjusted
        # blast-through (adjClose=$3M from CSII->PEGY->SUNE chain) while
        # SunEdison's are truly era-physical (ratio=1.0). The candidate
        # with MORE physical-era rows is the real owner of that interval.
        def _sane_sort_key(x):
            h, sane, score, rows = x
            n_physical = _physical_row_count(rows)
            # Higher physical count is better -> use -n_physical so sort ascending
            # gives lowest -n_physical first.
            # If the Wikipedia interval has a 'removed' date that is in the past,
            # prefer isActive=False (delisted / graduated out / bankrupt).
            inactive_pref = 0 if h.get("isActive") is not True else 1
            return (-n_physical, inactive_pref, score)
        sane_hits.sort(key=_sane_sort_key)
        chosen, sane, score, rows = sane_hits[0]
        n_phys = _physical_row_count(rows)
        reason = f"sane_probe(phys={n_phys}, score={score:.2f}, n={len(rows)})"
        return chosen, reason, len(rows), rows

    # Fallback: no candidate had sane rows in the probe window. If exactly
    # one candidate isActive=False (the typical "historical, now defunct"
    # case) -> return that one (and flag later as price_unavailable).
    inactive_hits = [x for x in scored if x[0].get("isActive") is not True]
    if inactive_hits and len(inactive_hits) == 1:
        chosen, sane, score, rows = inactive_hits[0]
        reason = f"inactive_only_fallback(score={score:.2f}, n={len(rows)})"
        return chosen, reason, len(rows), rows

    # Last fallback: just pick the inactive one (or first US-stock candidate).
    if inactive_hits:
        chosen, sane, score, rows = inactive_hits[0]
        return chosen, f"inactive_fallback(n={len(rows)})", len(rows), rows
    # No probe data; no inactive hint -> take first US-stock candidate.
    chosen = us_hits[0]
    return chosen, "no_probe_fallback", 0, []


# ------------------------------------------------------------------
# Per-ticker processing
# ------------------------------------------------------------------

def _probe_window(added_ts: pd.Timestamp | None, *, wide: bool = False) -> tuple[str, str]:
    """Compute the /prices probe window for the permaTicker name-sanity check.

    wide=False (default) is the narrow window [added - 7d, added + 30d] used
    for primary disambiguation. wide=True is the backup window
    [added - 90d, added + 275d] used when the narrow probe returns 0 rows
    to detect permaTickers that exist but started trading some months after
    the Wikipedia added_date.
    """
    today = pd.Timestamp.now().normalize()
    before = WIDE_PROBE_BEFORE_DAYS if wide else PROBE_BEFORE_DAYS
    after = WIDE_PROBE_AFTER_DAYS if wide else PROBE_AFTER_DAYS
    if added_ts is None:
        # No added_date: probe recent 30 days as a weak liveness check.
        end = today
        start = today - pd.Timedelta(days=after)
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    start = added_ts - pd.Timedelta(days=before)
    end = added_ts + pd.Timedelta(days=after)
    if end > today:
        end = today
        if start > today:
            start = today - pd.Timedelta(days=after)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def process_ticker_row(
    row,
    *,
    legacy_perm_id_map: dict[str, str] | None = None,
    verbose: bool = True,
) -> list[dict]:
    """Process one /metadata/sp400 row (one ticker, multiple intervals).

    `row` can be a pandas Series (preferred) OR an itertuples namedtuple
    from /metadata/sp400. Either form exposes the named columns via
    .get() / attribute access. We convert to dict to normalize access.

    Returns a list of permaTicker-row dicts. Each dict corresponds to ONE
    distinct permaTicker that owned at least one of this ticker code's
    Wikipedia intervals. If the same permaTicker owns multiple Wikipedia
    intervals (rebrand-series like CSII -> PEGY -> SUNE under one
    permaTicker), they are aggregated into ONE row with a multi-element
    wikipedia_intervals list.

    Returns [] if Tiingo has no US-stock candidate for this ticker code at
    all (rare -- typically means Wikipedia has a stale ticker code that
    Tiingo never tracked).
    """
    # Normalize input: accept both pd.Series and itertuples namedtuple.
    if hasattr(row, "_asdict"):
        row = dict(row._asdict())
    elif isinstance(row, pd.Series):
        row = row.to_dict()
    elif not isinstance(row, dict):
        # Try a plain attribute fallback.
        row = {k: getattr(row, k, None) for k in ("ticker", "name", "intervals", "sic", "index_ref", "cik")}

    def get(name, default=None):
        v = row.get(name, default)
        if isinstance(v, float) and pd.isna(v):
            return default
        return v

    ticker = str(get("ticker", "")).upper().strip()
    if not ticker:
        return []

    intervals = _parse_intervals(get("intervals"))
    if not intervals:
        if verbose:
            print(f"  {ticker}: no Wikipedia intervals, skipping")
        return []

    # One search call per ticker code (NOT per interval -- the search
    # returns ALL historical holders of that ticker code; we then
    # disambiguate positionally against each Wikipedia interval).
    if verbose:
        print(f"  {ticker}: searching Tiingo ...")
    candidates = tiingo_search_ticker(ticker, verbose=verbose)
    time.sleep(TIINGO_DELAY)

    # Aggregate permaTicker -> list of (interval_dict, chosen_hit, reason, probe_rows) tuples.
    permaticker_to_intervals: dict[str, list[tuple[dict, dict, str, list[dict]]]] = {}

    skipped_intervals = 0
    for iv in intervals:
        added_ts = _to_ts(iv.get("added"))
        removed_ts = _to_ts(iv.get("removed"))
        chosen, reason, _, probe_rows = disambiguate_permaTicker(
            ticker, candidates, added_ts, removed_ts
        )
        if chosen is None:
            if verbose:
                print(f"    {ticker}: {iv} -> NO permaTicker ({reason}); skipped")
            skipped_intervals += 1
            continue
        pt = chosen.get("permaTicker")
        if pt is None or not isinstance(pt, str) or pt.strip() == "":
            if verbose:
                print(f"    {ticker}: {iv} -> chosen hit missing permaTicker; skipped")
            skipped_intervals += 1
            continue
        permaticker_to_intervals.setdefault(pt, []).append((iv, chosen, reason, probe_rows))
        if verbose:
            print(
                f"    {ticker}: {iv} -> permaTicker={pt} name={chosen.get('name')!r} "
                f"active={chosen.get('isActive')} reason={reason}"
            )

    # Build per-permaTicker rows. Aggregate intervals.
    rows = []
    for pt, hits in permaticker_to_intervals.items():
        # The chosen hit metadata from the FIRST interval is representative
        # for the permaTicker-level fields (name, isActive, etc.). But we
        # keep the per-interval chosen hits in an audit-log list for the
        # disambiguation log file; not stored in db.h5.
        chosen_first = hits[0][1]
        # Each permaTicker's aggregated Wikipedia intervals list.
        wiki_intervals = [h[0] for h in hits]
        # Reuse the probe rows already fetched for the FIRST interval as
        # the name-sanity probe (don't re-fetch).
        first_probe_rows = hits[0][3] if hits else []
        n_rows = len(first_probe_rows)
        # If first probe returned 0 rows (e.g. inactive_fallback path),
        # try probing against the next interval's window as a backup.
        sane, _ = _prices_sanity_score(first_probe_rows)
        if n_rows == 0:
            for hit_tuple in hits[1:]:
                cand_rows = hit_tuple[3]
                if cand_rows:
                    first_probe_rows = cand_rows
                    n_rows = len(cand_rows)
                    sane, _ = _prices_sanity_score(first_probe_rows)
                    break
        price_unavailable = (n_rows == 0) or (not sane)
        if verbose and price_unavailable:
            print(
                f"    >> {ticker} permaTicker={pt}: /prices probe returned {n_rows} rows, sane={sane} "
                f"-> price_unavailable=True"
            )

        # Legacy perm_id mapping no longer needed (Phase D migration
        # complete). The field was a Phase A->D bridge column for one-time
        # re-keying of /earnings/raw, and is no longer written to the DB
        # (see cleanup_phase_d_2024_post_doc_j.py).

        # Carry SIC/index_ref/cik from the original /metadata/sp400 row.
        sic = get("sic")
        index_ref = get("index_ref")
        cik = get("cik")

        row_out = {
            "permaTicker": pt,
            "canonical_ticker": chosen_first.get("ticker", ticker),
            "name": chosen_first.get("name"),
            "isActive": bool(chosen_first.get("isActive")),
            "openfigi": chosen_first.get("openFIGIComposite"),
            "cik": cik,
            "sic": sic,
            "index_ref": index_ref,
            "wikipedia_intervals": json.dumps(wiki_intervals, default=str),
            "price_unavailable": price_unavailable,
        }
        rows.append(row_out)
        if verbose:
            print(
                f"  {ticker} -> permaTicker={pt} "
                f"({len(wiki_intervals)} interval(s), prices_probe={n_rows} rows, "
                f"price_unavailable={price_unavailable})"
            )
    if skipped_intervals and verbose:
        print(f"  {ticker}: {skipped_intervals} interval(s) skipped (no permaTicker)")
    return rows


# ------------------------------------------------------------------
# Output / persistence
# ------------------------------------------------------------------

_PERMATICKER_COLS = [
    "permaTicker",
    "canonical_ticker",
    "name",
    "isActive",
    "openfigi",
    "cik",
    "sic",
    "index_ref",
    "wikipedia_intervals",
    "price_unavailable",
]


def write_outputs(permatickers: list[dict]) -> pd.DataFrame:
    """Persist /metadata/sp400_permatickers to db.h5 using the
    HDFStore('a') + remove() pattern (never mode='w' on existing DB).

    Also removes the legacy /metadata/sp400_perm_ids and
    /metadata/sp400_companies keys -- they are superseded by this table.

    Returns the written DataFrame.
    """
    df = pd.DataFrame(permatickers, columns=_PERMATICKER_COLS)
    if not df.empty:
        df["isActive"] = df["isActive"].astype(bool)
        df["price_unavailable"] = df["price_unavailable"].astype(bool)
        df["cik"] = df["cik"].astype(object)
        df["openfigi"] = df["openfigi"].astype(object)

    with pd.HDFStore(DB_FILE, mode="a") as store:
        if PERMATICKERS_KEY in store:
            store.remove(PERMATICKERS_KEY)
        if not df.empty:
            store.put(PERMATICKERS_KEY, df, format="table")
            print(f"\nWrote {PERMATICKERS_KEY} ({len(df)} rows, {len(df.columns)} cols)")
        # Purge legacy perm_id table.
        if LEGACY_PERM_IDS_KEY in store:
            store.remove(LEGACY_PERM_IDS_KEY)
            print(f"Purged legacy {LEGACY_PERM_IDS_KEY} (superseded by {PERMATICKERS_KEY})")
        if LEGACY_COMPANIES_KEY in store:
            store.remove(LEGACY_COMPANIES_KEY)
            print(f"Purged legacy {LEGACY_COMPANIES_KEY} (superseded)")
    return df


def write_outputs_merge(permatickers: list[dict]) -> pd.DataFrame:
    """Upsert new/updated permaTicker rows into the existing table.

    Used by the --tickers + --merge incremental mode: process only a few
    tickers (e.g. new constituents from refresh_sp400_membership.py) and
    merge their rows into /metadata/sp400_permatickers by permaTicker,
    preserving all other existing rows. Never rewrites the whole table.
    """
    if not permatickers:
        print("\nNo new rows to merge.")
        with pd.HDFStore(DB_FILE, mode="r") as store:
            if PERMATICKERS_KEY in store:
                return store[PERMATICKERS_KEY]
        return pd.DataFrame()
    new_df = pd.DataFrame(permatickers, columns=_PERMATICKER_COLS)
    new_df["isActive"] = new_df["isActive"].astype(bool)
    new_df["price_unavailable"] = new_df["price_unavailable"].astype(bool)
    new_df["cik"] = new_df["cik"].astype(object)
    new_df["openfigi"] = new_df["openfigi"].astype(object)
    new_pts = set(new_df["permaTicker"])

    with pd.HDFStore(DB_FILE, mode="a") as store:
        if PERMATICKERS_KEY in store:
            existing = store[PERMATICKERS_KEY]
            kept = existing[~existing["permaTicker"].isin(new_pts)].copy()
            combined = pd.concat([kept, new_df], ignore_index=True)
            store.remove(PERMATICKERS_KEY)
            store.put(PERMATICKERS_KEY, combined, format="table")
            print(f"\nMerged {len(new_df)} row(s) into {PERMATICKERS_KEY} "
                  f"(replaced {len(existing) - len(kept)}; total now {len(combined)})")
            return combined
        else:
            store.put(PERMATICKERS_KEY, new_df, format="table")
            print(f"\nWrote {PERMATICKERS_KEY} ({len(new_df)} rows)")
            return new_df


# ------------------------------------------------------------------
# Audit report
# ------------------------------------------------------------------

def print_audit(df: pd.DataFrame) -> None:
    if df.empty:
        print("  EMPTY")
        return
    n_total = len(df)
    n_unavailable = int(df["price_unavailable"].sum())
    n_active = int(df["isActive"].sum())
    n_openfigi = int(df["openfigi"].notna().sum())
    n_multi_interval = 0
    for s in df["wikipedia_intervals"]:
        try:
            ivs = json.loads(s) if isinstance(s, str) else s
            if isinstance(ivs, list) and len(ivs) > 1:
                n_multi_interval += 1
        except Exception:
            pass

    print("\n=== /metadata/sp400_permatickers audit ===")
    print(f"  Total permaTicker rows:        {n_total}")
    print(f"  Active (currently trading):     {n_active}")
    print(f"  With openFIGI field:            {n_openfigi}")
    print(f"  Multi-interval permaTickers:    {n_multi_interval}")
    print(f"  price_unavailable (0 rows):     {n_unavailable}")
    if n_unavailable:
        # Show the first few unavailable permaTickers for manual inspection.
        sub = df[df["price_unavailable"] == True].head(20)
        print(f"\n  First {len(sub)} price_unavailable entries (Class-W/S suspects):")
        print(sub[["permaTicker", "canonical_ticker", "name", "isActive", "wikipedia_intervals"]].to_string(index=False))


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

def load_legacy_perm_id_map() -> dict[str, str]:
    """Load the tickers-to-perm_id mapping from the existing legacy table.

    Used to populate the `legacy_perm_id` column in the new table so Phase D
    re-keying can rename perm_id references in /earnings/raw in a single
    rename pass.

    Returns: dict[uppercase ticker -> legacy perm_id]
    """
    if LEGACY_PERM_IDS_KEY is None:
        return {}
    try:
        with pd.HDFStore(DB_FILE, mode="r") as s:
            if LEGACY_PERM_IDS_KEY not in s:
                return {}
            df = s[LEGACY_PERM_IDS_KEY]
    except Exception:
        return {}
    out = {}
    for canon, pid in zip(df["canonical_ticker"], df["perm_id"]):
        if pd.notna(canon) and pd.notna(pid):
            out[str(canon).upper().strip()] = str(pid)
    return out


def main(tickers_filter: set | None = None, merge: bool = False):
    print("=" * 70)
    print("02b - Build Company-Level Map (Tiingo permaTicker-anchored)")
    print("=" * 70)
    print(f"DB file: {DB_FILE}")
    if tickers_filter is not None:
        print(f"Mode: TARGETED ({len(tickers_filter)} ticker(s): {sorted(tickers_filter)})")
    if merge:
        print("Mode: MERGE (upsert into existing table)")

    if not DB_FILE.exists():
        raise SystemExit(f"db.h5 not found at {DB_FILE}")

    # Load /metadata/sp400 source rows.
    with pd.HDFStore(DB_FILE, mode="r") as store:
        if META_KEY not in store:
            raise SystemExit(f"{META_KEY} not found in {DB_FILE} -- run 01_metadata_gathering first.")
        meta = store[META_KEY]
    print(f"Loaded {META_KEY}: {len(meta)} ticker rows")
    if tickers_filter is not None:
        meta = meta[meta["ticker"].astype(str).str.upper().isin(tickers_filter)].copy()
        print(f"After --tickers filter: {len(meta)} ticker rows to process")

    # Load legacy perm_id -> ticker mapping for the legacy_perm_id column.
    # (DEPRECATED after cleanup_phase_d_2024; the legacy_perm_id column
    # is no longer written to the DB but the legacy mapping function is
    # kept for audit / historical reference.
    legacy_map = load_legacy_perm_id_map()
    print(f"Loaded {len(legacy_map)} legacy perm_id -> ticker mappings "
          f"(DEPRECATED after cleanup_phase_d_2024; legacy_perm_id column removed)")

    # Persist a per-ticker disambiguation log to help review ambiguous
    # decisions after the run (e.g. ticker codes with multiple permaTickers).
    log_path = Path(__file__).parent / "permaticker_disambiguation.log"
    log_file = log_path.open("w", encoding="utf-8")
    print(f"Disambiguation log -> {log_path.name}")

    all_rows: list[dict] = []
    n_tickers = len(meta)
    t0 = time.time()
    for i, row in enumerate(meta.itertuples(index=False), 1):
        ticker = str(row.ticker)
        elapsed = time.time() - t0
        rate = i / max(elapsed, 0.001)
        eta = (n_tickers - i) / max(rate, 0.001)
        log_file.write(f"\n[{i}/{n_tickers}] {ticker} (elapsed {elapsed:.1f}s, eta {eta:.1f}s)\n")
        # Redirect this row's progress prints to both stdout (live) and the
        # log file. The Tiingo probe prints inside process_ticker_row go
        # through sys.stdout (which we wrapped with utf-8 errors=replace),
        # so to capture them in the log we tee-prinlate via a custom helper.
        # Simpler: just call process_ticker_row with verbose=True and re-emit
        # the row summary line to the log.
        print(f"[{i}/{n_tickers}] {ticker} ...", flush=True)
        rows = process_ticker_row(
            row, legacy_perm_id_map=legacy_map, verbose=False
        )
        for r in rows:
            log_file.write(
                f"   permaTicker={r['permaTicker']} ticker={r['canonical_ticker']} "
                f"name={r.get('name')!r} isActive={r['isActive']} "
                f"price_unavailable={r['price_unavailable']} "
                f"openfigi={r.get('openfigi')}\n"
            )
            try:
                ivs = json.loads(r["wikipedia_intervals"]) if isinstance(r["wikipedia_intervals"], str) else r["wikipedia_intervals"]
                for iv in ivs:
                    log_file.write(f"     interval: {iv}\n")
            except Exception:
                pass
        all_rows.extend(rows)
        log_file.flush()
    log_file.close()
    print(f"\nDone processing. Total permaTicker rows: {len(all_rows)}")
    print(f"Wall time: {time.time() - t0:.1f}s")

    if merge:
        df = write_outputs_merge(all_rows)
    else:
        df = write_outputs(all_rows)
    print_audit(df)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Build company-level permaTicker map")
    p.add_argument("--tickers", type=str, default=None,
                   help="Comma-separated tickers to process only (incremental mode)")
    p.add_argument("--merge", action="store_true",
                   help="Upsert processed rows into the existing table (use with --tickers)")
    args = p.parse_args()
    tf = None
    if args.tickers:
        tf = {t.strip().upper() for t in args.tickers.split(",") if t.strip()}
    main(tickers_filter=tf, merge=args.merge)
