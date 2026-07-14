#!/usr/bin/env python3
"""02b - Build Company-Level Map (perm_id-anchored, interval-forked merge)
=========================================================================

Phase A rewrite per `luan_bot_trading/01_data/merger_identity_patch.md`.

Replaces the old CIK-anchor-and-merge approach (which wrongly collapsed
acquirer + target pre-merger histories into one canonical because SEC
retroactively reassigns the absorbed company's CIK into the surviving
parent's CIK) with a ``perm_id`` anchor:

    perm_id = "{cik_at_added}_{start_ticker}"

Each point-in-time "tradable asset track" gets its own immutable ``perm_id``.
When two SP-400 entries share the same CIK but their residency windows
**overlap**, they cannot be the same continuously-listed asset, so they are
forked into separate ``perm_id``s (preserving Company B's independent
pre-merger history aka the "Survivor-CIK Collision Bug").

CIK lookup is POINT-IN-TIME, not present-day:
For each Wikipedia (ticker, added) interval entry in /metadata/sp400 we look
up the CIK that ticker reported under at that ``added`` year via the cached
DERA `sub_{YYYY}.txt` snapshots. This decouples the legal entity tracked at
index-addition time from whatever CIK the SEC has consolidated into today.

Inputs (from db.h5):
    /metadata/sp400   per-ticker rows with columns:
                      ticker, name, gics_sector, gics_sub_industry,
                      intervals (JSON list of {added, removed|None}),
                      sic, index_ref

Cache files used (no network required for these):
    sec_cache/dera/sub_{YYYY}.txt       DERA Q4 filings; CIK per (ticker, year)
    sec_cache/company_tickers_exchange.json   Current SEC active tickers
                                              (used for canonical alias selection
                                               only -- NOT for fork decisions)
    sec_cache/ticker.txt                Older cached SEC ticker map (fallback)

Outputs (back into db.h5):
    /metadata/sp400_perm_ids   NEW table, one row per perm_id:
        perm_id                str   PRIMARY KEY: f"{cik_at_added}_{start_ticker}"
        cik                    str or None (point-in-time CIK at first interval's added)
        canonical_ticker       str   alias used to look up price series in /sp400/
        aliases                JSON list[str]  all tickers seen on this track,
                                                 ordered [canonical, then alpha]
        name                   str
        sic                    str
        index_ref              str
        combined_intervals     JSON list[{added, removed|None}] merged spans
                               (overlap or abut <= ABUT_DAYS -> one span; gap -> 2)
        per_ticker_intervals   JSON dict[ticker -> list[{added, removed|None}]]
                               (audit trail)
        price_unavailable      bool  True if no alias verified on EODHD

    /metadata/sp400            EXTENDED with point-in-time `cik_at_added` and
                               `perm_id` columns (audit trail only; canonical
                               still stored for backwards-compat with downstream
                               stages that haven't been refactored yet). The
                               OLD `cik`/`canonical_ticker` columns are
                               deprecated and replaced by these new columns.

See `luan_bot_trading/01_data/company_merge_design.md` and
`luan_bot_trading/01_data/merger_identity_patch.md` for the full design.
"""

import json
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

_load_env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(_load_env_path)

import os

DB_FILE = Path(__file__).parent / "db.h5"
META_KEY = "/metadata/sp400"
# NEW: replaces /metadata/sp400_companies. One row per perm_id (CIK "track"
# of a tradable asset, forked on overlap).
PERM_IDS_KEY = "/metadata/sp400_perm_ids"
# Legacy key kept for backwards-compat during the staged refactor. We DO NOT
# write to it after Phase A -- downstream stages will be migrated in
# Phases B-E.
LEGACY_COMPANIES_KEY = "/metadata/sp400_companies"

EODHD_API_KEY = os.getenv("EODHD_API_KEY")
if not EODHD_API_KEY:
    raise ValueError(
        "EODHD_API_KEY not found in .env. The 02b availability probe uses "
        "EODHD (paid subscription, 100k/day, 1000/min) so we don't consume "
        "the 50 req/hour Tiingo free-tier limit during probing."
    )
# Abutting intervals within this many days are merged into one span.
ABUT_DAYS = 7
# EODHD probe throttle. EODHD limit is 1000/min ~= 16/sec, 0.05s sleep
# keeps us comfortably under that. Finishes ~993 tickers in <1 minute.
EODHD_PROBE_DELAY = 0.05
EODHD_PROBE_WINDOW_DAYS = 30

# Full-history window for second-pass fallback probe. Used only for the
# small set of perm_ids marked `price_unavailable=True` after the initial
# targeted probe. Recovers companies whose per-ticker-interval probe
# window landed on a date range with no trading data because Wikipedia's
# metadata had a missing/wrong `removed` date.
EODHD_FALLBACK_FROM = "2012-01-01"
EODHD_FALLBACK_TO = pd.Timestamp.now().strftime("%Y-%m-%d")

# DERA snapshot year range we cache.
DERA_MIN_YEAR = 2010
DERA_MAX_YEAR = 2025

# Year range over which we resolve point-in-time CIK. Anything outside
# gets clamped to these bounds.
PIT_YEAR_MIN = DERA_MIN_YEAR
PIT_YEAR_MAX = DERA_MAX_YEAR


# ----------------------------------------------------------------------
# Manual overrides: SEC DERA's `instance` column can carry a STALE ticker
# symbol post-rebrand when NYSE/Nasdaq has reassigned that symbol to a
# different company. In those cases the Wikipedia interval entry is an
# ADDED record for the new company, but DERA still maps the ticker symbol
# to the previous owner's CIK. Override here.
#
# Authoritative source for these overrides: SEC's current
# `company_tickers_exchange.json`, which reflects actively-traded NYSE
# /Nasdaq tickers. For each override we pin the CIK to the correct legal
# entity for every (ticker, year) pair that DERA gets wrong.
#
# Map (ticker_upper, year_added) -> cik_zfilled_10digit_str
# ----------------------------------------------------------------------
MANUAL_TAS_OVERRIDE: dict[tuple[str, int], str] = {
    # NYSE re-tasks the "RBC" ticker from Regal Beloit / Regal Rexnord
    # (CIK 82811) to RBC Bearings (CIK 1324948) starting Sept 2023.
    # Wikipedia recorded SP400 RBC addition 2023-09-18, which is RBC
    # Bearings; DERA still reports ticker "RBC" -> 82811 in sub_2023/
    # 2024/2025 because Regal Rexnord's filings continue under the
    # legacy instance label "RBC" (instance column = RBC-...)
    ("RBC", 2023): "0001324948",
    ("RBC", 2024): "0001324948",
    ("RBC", 2025): "0001324948",
}


# ------------------------------------------------------------------
# Interval helpers (unchanged from previous implementation)
# ------------------------------------------------------------------

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
        # Already-parsed list/dict -- coerce through JSON normalization.
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

    # Sort by added. A null 'added' (NaT) represents a historical terminal
    # span ("company was in the index from some unknown pre-history until
    # 'removed'") and should sort to the FRONT of the timeline, so
    # subsequent re-adds abut against it.
    norm.sort(key=lambda x: (pd.Timestamp.min if x[0] is None else x[0]))

    merged = []
    cur_a, cur_r = norm[0]
    for a, r in norm[1:]:
        if cur_r is None:
            # Current span is open-ended ([cur_a, +inf)). ALL subsequent
            # intervals are contained within it by definition -- skip
            # without advancing the cursor. Don't append; the open-ended
            # span collapses everything that follows.
            continue
        if a is None:
            # Next interval has no added; only its removed is known. It is a
            # legacy terminal span that should be merged into the current
            # span by using max(removed). Don't advance the cursor.
            if cur_r is None:
                continue
            if r is None:
                cur_r = None
            elif r > cur_r:
                cur_r = r
            continue
        gap = (a - cur_r).total_seconds() / 86400.0
        if a <= cur_r or gap <= ABUT_DAYS:
            if r is None:
                cur_r = None
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


def _latest_active_date(intervals_raw) -> pd.Timestamp | None:
    """Latest date the ticker had any index residency (removed or open)."""
    ivs = _parse_intervals(intervals_raw)
    if not ivs:
        return None
    has_open = False
    best_removed = None
    best_added = None
    for iv in ivs:
        r = iv.get("removed")
        a = iv.get("added")
        if r is None:
            has_open = True
        else:
            rts = _to_ts(r)
            if rts is not None and (best_removed is None or rts > best_removed):
                best_removed = rts
        ats = _to_ts(a)
        if ats is not None and (best_added is None or ats > best_added):
            best_added = ats
    if has_open:
        return pd.Timestamp.now().normalize()
    if best_removed is not None:
        return best_removed
    return best_added


def _latest_added(intervals: list[dict]) -> pd.Timestamp | None:
    best = None
    for iv in intervals:
        a = _to_ts(iv.get("added"))
        if a is None:
            continue
        if best is None or a > best:
            best = a
    return best


def _probe_window_for(latest_active: pd.Timestamp | None) -> tuple[str, str] | None:
    if latest_active is None or pd.isna(latest_active):
        return None
    to_ts = latest_active
    from_ts = to_ts - pd.Timedelta(days=EODHD_PROBE_WINDOW_DAYS)
    today = pd.Timestamp.now().normalize()
    if to_ts > today:
        to_ts = today
    if from_ts > today:
        from_ts = today - pd.Timedelta(days=EODHD_PROBE_WINDOW_DAYS)
    return (from_ts.strftime("%Y-%m-%d"), to_ts.strftime("%Y-%m-%d"))


# ------------------------------------------------------------------
# Point-in-time CIK lookup
# ------------------------------------------------------------------

def load_active_sec_ciks() -> dict[str, str]:
    """Load current SEC active ticker->CIK map from
    `sec_cache/company_tickers_exchange.json`. Used as a secondary
    fallback for tickers absent from DERA, and for canonical-alias
    selection within perm_ids (NOT for fork decisions).

    Returns:
        dict[uppercase ticker -> 10-digit zero-padded CIK]
    """
    cache_path = Path(__file__).parent / "sec_cache" / "company_tickers_exchange.json"
    if not cache_path.exists():
        return {}
    try:
        bundle = json.loads(cache_path.read_text())
        fields = bundle["fields"]
        out: dict[str, str] = {}
        for entry in bundle["data"]:
            d = dict(zip(fields, entry))
            t = str(d.get("ticker", "")).upper().strip()
            cik = d.get("cik")
            if t and cik:
                out[t] = str(cik).zfill(10)
        return out
    except Exception as e:
        print(f"   [load_active_sec_ciks] failed to parse {cache_path}: {e}")
        return {}


def load_cached_ticker_txt() -> dict[str, str]:
    """Load cached `sec_cache/ticker.txt` (older SEC ticker->CIK listing).
    Last-resort fallback for tickers absent from both DERA and the active
    exchange.json.
    """
    cache_path = Path(__file__).parent / "sec_cache" / "ticker.txt"
    if not cache_path.exists():
        return {}
    out: dict[str, str] = {}
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        parts = [p.strip() for p in line.split()]
        if len(parts) < 2:
            continue
        t, cik = parts[0], parts[1]
        if t and cik:
            out[t.upper()] = str(cik).zfill(10)
    return out


def build_pit_cik_index() -> dict[tuple[str, int], str]:
    """Build a point-in-time index ``{(ticker_upper, year_added) -> cik}``
    from cached DERA `sub_{YYYY}.txt` snapshots.

    Each `sub_{year}.txt` is parsed once: rows give ticker (derived from
    the `instance` field) -> cik. If the same ticker appears under multiple
    CIKs in a single year (rare; usually same-legal-entity re-filings), we
    keep the most-frequent CIK to reduce noise.

    Years read: DERA_MIN_YEAR..DERA_MAX_YEAR (only those present in cache).
    """
    dera_dir = Path(__file__).parent / "sec_cache" / "dera"
    pit: dict[tuple[str, int], str] = {}
    if not dera_dir.exists():
        print(f"   [build_pit_cik_index] DERA cache dir not found: {dera_dir}")
        return pit

    n_years = 0
    for year in range(DERA_MIN_YEAR, DERA_MAX_YEAR + 1):
        sub_path = dera_dir / f"sub_{year}.txt"
        if not sub_path.exists():
            continue
        try:
            df = pd.read_csv(
                sub_path, sep="\t", dtype=str,
                usecols=["cik", "instance", "sic"],
            )
        except Exception as e:
            print(f"   [build_pit_cik_index] skip {sub_path.name}: {e}")
            continue
        df = df.dropna(subset=["instance", "cik"])
        df["ticker"] = (
            df["instance"].astype(str).str.split("-").str[0].str.upper().str.strip()
        )
        df["cik"] = df["cik"].astype(str).str.zfill(10)
        df = df[df["ticker"].astype(bool)]
        # Most-frequent CIK per (ticker, year): reduces noise from refilings.
        counts = df.groupby(["ticker", "cik"]).size().reset_index(name="n")
        counts = counts.sort_values(["ticker", "n"], ascending=[True, False])
        counts = counts.drop_duplicates(subset=["ticker"], keep="first")
        for t, cik in zip(counts["ticker"].values, counts["cik"].values):
            pit[(t, year)] = cik
        n_years += 1

    print(f"   [build_pit_cik_index] parsed {n_years} DERA snapshots; "
          f"index has {len(pit)} (ticker, year) entries")
    return pit


def lookup_cik_at(
    ticker: str,
    added_year: int,
    pit_index: dict[tuple[str, int], str],
    active_sec: dict[str, str],
    ticker_txt: dict[str, str],
) -> str | None:
    """Point-in-time CIK lookup for a (ticker, year) tuple.

    Resolution order (most-authoritative first):
       1. MANUAL_TAS_OVERRIDE  -- explicit, audited overrides for DERA
          instance-column staleness.
       2. DER A point-in-time at added_year, with year-walkback/forward.
       3. Active SEC `company_tickers_exchange.json` (cache).
       4. Cached `ticker.txt` (older cache).
       5. None.

    Walkback/forward: if DERA sub_{year}.txt has the ticker -> cik, use it.
    Otherwise try year-1, year-2, ..., DERA_MIN_YEAR; then year+1, ...,
    DERA_MAX_YEAR. Capping at DERA boundaries.

    Args:
        ticker: NYSE/Nasdaq ticker symbol, case-insensitive.
        added_year: Year of the Wikipedia-added date for this interval.
        pit_index: Pre-built {(ticker_upper, year) -> cik} from DERA.
        active_sec: Active SEC ticker.json (current).
        ticker_txt: Cached ticker.txt (older).

    Returns:
        10-digit zero-padded CIK string, or None if no source resolves.
    """
    t = ticker.upper().strip()
    if not t:
        return None

    # 1. Manual override (top priority).
    ov = MANUAL_TAS_OVERRIDE.get((t, added_year))
    if ov:
        return ov

    # Clamp year to DERA range for the walkback/forward loop.
    cy = max(PIT_YEAR_MIN, min(PIT_YEAR_MAX, added_year))

    # 2. DERA point-in-time. Walk back first, then forward.
    # Walk-back: prefer matching near added_year going back to PIT_YEAR_MIN.
    for off in range(0, cy - PIT_YEAR_MIN + 1):
        y = cy - off
        if y < PIT_YEAR_MIN:
            break
        cik = pit_index.get((t, y))
        if cik:
            return cik
    # Walk-forward from cy + 1 up to PIT_YEAR_MAX.
    for off in range(1, PIT_YEAR_MAX - cy + 1):
        y = cy + off
        if y > PIT_YEAR_MAX:
            break
        cik = pit_index.get((t, y))
        if cik:
            return cik

    # 3. Active SEC `company_tickers_exchange.json`.
    active = active_sec.get(t)
    if active:
        return active

    # 4. Cached ticker.txt.
    txt = ticker_txt.get(t)
    if txt:
        return txt

    # 5. None.
    return None


# ------------------------------------------------------------------
# EODHD availability probe (preserved from prior implementation;
# renamed internally to `eodhd_` for clarity)
# ------------------------------------------------------------------

def probe_tickers_on_eodhd(
    tickers: list[str],
    progress_every: int = 50,
    ticker_intervals: dict[str, object] | None = None,
) -> dict[str, bool]:
    """Probe ticker availability via EODHD /api/eod/{TICKER}.US.

    Returns dict[ticker -> bool]. A True means EODHD returned a non-empty
    daily-EOD list inside the per-ticker probe window.

    Per-ticker probe window: computed from each ticker's
    /metadata/sp400 interval data (latest removed/open added date),
    which catches both currently-trading and delisted tickers in one pass.
    """
    results: dict[str, bool] = {}
    for i, t in enumerate(tickers, 1):
        if ticker_intervals is None:
            latest_active = None
        else:
            latest_active = _latest_active_date(ticker_intervals.get(t))
        window = _probe_window_for(latest_active)
        if window is None:
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


def reprobe_unavailable_perm_ids(
    perm_ids: list[dict],
    eodhd_avail: dict[str, bool],
) -> tuple[list[dict], dict[str, bool]]:
    """Second-pass probe for perm_ids marked price_unavailable=True.

    Uses the full 15-year EODHD window on each alias in priority order
    (canonical first, then alphabetical). Recovers perm_ids whose
    targeted probe window landed on an empty-data range due to
    incomplete Wikipedia `removed` dates.

    Mutates `perm_ids` (sets price_unavailable=False for recoveries) and
    `eodhd_avail` in place. Returns both for convenience.
    """
    unavail = [c for c in perm_ids if c.get("price_unavailable")]
    if not unavail:
        return perm_ids, eodhd_avail

    print(
        f"\n[3b/4] Re-probing {len(unavail)} unavailable perm_ids with the full "
        f"15-year window ({EODHD_FALLBACK_FROM}..{EODHD_FALLBACK_TO})..."
    )
    recovered = 0
    for c in unavail:
        aliases_in_priority = [c["canonical_ticker"]] + [
            a for a in c["aliases"] if a != c["canonical_ticker"]
        ]
        got = False
        for alias in aliases_in_priority:
            url = f"https://eodhd.com/api/eod/{alias}.US"
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
                    # If canonical was unavailable but an alias worked, keep
                    # canonical_ticker as-is (price data lives under canonical). If
                    # the canonical itself now probes successfully, just flip flag.
                    c["price_unavailable"] = False
                    eodhd_avail[alias] = True
                    got = True
                    if alias != c["canonical_ticker"]:
                        # Note: we keep canonical_ticker pointing at the original
                        # canonical alias (per latest-name-rule from
                        # company_tickers_exchange.json). The alias just confirms
                        # EODHD has SOME data; downstream Stage B will probe the
                        # canonical specifically.
                        c["_fallback_alias_for_price"] = alias
                    recovered += 1
                    print(f"   recovered: {c['perm_id']} via alias {alias}")
                    break
            except Exception as e:
                print(f"   {alias}: fallback probe error: {e}")
            time.sleep(EODHD_PROBE_DELAY)
        if not got:
            pass
    print(
        f"   fallback recovered {recovered} / {len(unavail)} perm_ids. "
        f"Still unavailable: {len(unavail) - recovered}."
    )
    return perm_ids, eodhd_avail


# ------------------------------------------------------------------
# Core algorithm -- perm_id forking per merger_identity_patch.md
# ------------------------------------------------------------------

def _intervals_overlap(a_added, a_removed, b_added, b_removed) -> bool:
    """Half-open-style overlap test, but with None treated as the latest
    possible endpoint. Returns True if intervals A and B overlap.

    Convention for None:
      - removed = None -> open-ended interval (extends to today).
      - added = None -> pre-history terminal span (extends to earliest).
    """
    today = pd.Timestamp.now().normalize()
    a_added_ts = (a_added if a_added is not None else pd.Timestamp.min)
    a_removed_ts = (a_removed if a_removed is not None else today)
    b_added_ts = (b_added if b_added is not None else pd.Timestamp.min)
    b_removed_ts = (b_removed if b_removed is not None else today)
    if isinstance(a_added_ts, str):
        a_added_ts = _to_ts(a_added_ts)
    if isinstance(a_removed_ts, str):
        a_removed_ts = _to_ts(a_removed_ts)
    if isinstance(b_added_ts, str):
        b_added_ts = _to_ts(b_added_ts)
    if isinstance(b_removed_ts, str):
        b_removed_ts = _to_ts(b_removed_ts)
    # Safe fallbacks if anything came back None after coercion.
    a_added_ts = a_added_ts if a_added_ts is not None else pd.Timestamp.min
    b_added_ts = b_added_ts if b_added_ts is not None else pd.Timestamp.min
    a_removed_ts = a_removed_ts if a_removed_ts is not None else today
    b_removed_ts = b_removed_ts if b_removed_ts is not None else today
    # No overlap iff A entirely before B, or B entirely before A.
    if a_removed_ts <= b_added_ts or b_removed_ts <= a_added_ts:
        return False
    return True


def _make_track(entry: dict, pid: str | None = None) -> dict:
    """Build a fresh track dict for a single entry.

    Args:
        entry: dict with keys ticker, added, removed, cik, sic, index_ref, name.
        pid: if given, use this perm_id (supports suffixed keys on collisions);
             if None, derive a default pid from entry's (cik, ticker).
    """
    ticker = entry["ticker"]
    cik = entry["cik"]
    if pid is None:
        pid = (f"{cik}_{ticker.upper()}" if cik else f"__nocik_{ticker.upper()}")
    new_iv = {"added": _ts_to_str(entry["added"]),
              "removed": _ts_to_str(entry["removed"])}
    return {
        "perm_id": pid,
        "cik": cik,
        "start_ticker": ticker.upper(),
        "aliases": [ticker.upper()],
        "intervals": [new_iv],
        "per_ticker_intervals": {ticker.upper(): [new_iv]},
        "sic": entry["sic"],
        "index_ref": entry["index_ref"],
        "name": entry["name"],
    }


def _extend_track(track: dict, entry: dict) -> None:
    """Append a non-overlapping interval entry to an existing track."""
    ticker = entry["ticker"].upper()
    new_iv = {"added": _ts_to_str(entry["added"]),
              "removed": _ts_to_str(entry["removed"])}
    track["intervals"].append(new_iv)
    if ticker not in track["aliases"]:
        track["aliases"].append(ticker)
    track["per_ticker_intervals"].setdefault(ticker, []).append(new_iv)
    # Promote name if it was empty and now we have one.
    if (not track["name"]) and entry["name"]:
        track["name"] = entry["name"]


def _make_track_with_pid(entry: dict, pid: str) -> dict:
    """Build a fresh track dict using a specific (suffixed) perm_id."""
    return _make_track(entry, pid=pid)


def build_perm_id_map(
    meta_df: pd.DataFrame,
    pit_index: dict[tuple[str, int], str],
    active_sec: dict[str, str],
    ticker_txt: dict[str, str],
) -> tuple[pd.DataFrame, list[dict]]:
    """Build the perm_id table by interval-forking on overlap.

    Returns:
        extended_meta: per-ticker view of /metadata/sp400 with new columns:
            cik_at_added ^ point-in-time CIK at the ticker's first interval
                         added-date (single value per ticker; informational)
            perm_id      ^ comma-joined list of all perm_ids this ticker
                         contributes to (a ticker may fork across multiple
                         perm_ids if its CIK changes between intervals)
        perm_ids:     list of perm_id-row dicts, each with keys:
            perm_id, cik, start_ticker, canonical_ticker, aliases, name,
            sic, index_ref, combined_intervals, per_ticker_intervals,
            price_unavailable=placeholder False (set in probe step)
    """
    meta_indexed = meta_df.set_index("ticker", drop=False)

    # Step 1: Expand /metadata/sp400 rows into per-interval "entries".
    # Each entry = one Wikipedia (ticker, added, removed) span with
    # point-in-time CIK looked up at the added year.
    entries: list[dict] = []
    for _, row in meta_df.iterrows():
        ticker = str(row["ticker"]).strip()
        if not ticker:
            continue
        ivs = _parse_intervals(row.get("intervals"))
        for iv in ivs:
            added_ts = _to_ts(iv.get("added"))
            removed_ts = _to_ts(iv.get("removed"))
            # Backfill rule: per Design.md, missing added backfills to 2012-01-01.
            if added_ts is None and removed_ts is not None:
                added_ts = pd.Timestamp("2012-01-01")
            if added_ts is None:
                # utterly unknown span -- skip entirely
                continue
            year = added_ts.year
            cik = lookup_cik_at(ticker, year, pit_index, active_sec, ticker_txt)
            entries.append({
                "ticker": ticker,
                "added": added_ts,
                "removed": removed_ts,
                "cik": cik,
                "sic": row.get("sic") if not pd.isna(row.get("sic")) else None,
                "index_ref": row.get("index_ref") if not pd.isna(row.get("index_ref")) else None,
                "name": row.get("name") if not pd.isna(row.get("name")) else None,
            })

    # Sort entries by added asc so re-entries extend an existing track if non-
    # overlapping. (Tiebreak: ticker alphabetical for determinism.)
    entries.sort(key=lambda e: (e["added"], e["ticker"]))

    n_entries = len(entries)
    n_with_cik = sum(1 for e in entries if e["cik"])

    # Step 2: Fork-merge.
    # perm_id_table maps perm_id_key -> track dict.
    perm_id_table: dict[str, dict] = {}

    for entry in entries:
        ticker_u = entry["ticker"].upper()
        cik = entry["cik"]
        # Per mergers_identity_patch.md §2 and our Phase A refinement:
        # the `perm_id = {cik}_{start_ticker}` decouples legal-entity (CIK,
        # which SEC retroactively consolidates post-merger) from tradable
        # asset track. The point-in-time CIK lookup at each interval's
        # added-year already produces DIFFERENT CIKs for distinct pre-merger
        # tradable assets (HR pre-merger = 899749 vs HTA pre-merger =
        # 1360604; new CR = 1944013 vs CXT = 25445; etc.). Therefore:
        #
        #   - Entries with the SAME point-in-time CIK *always* belong to the
        #     same legal entity, and any ticker-alias differences are rebrands.
        #     ALWAYS merge them into the same perm_id (extend aliases).
        #   - Entries with DIFFERENT CIKs are never merged; they are forked
        #     into separate perm_ids (preserving Company B's independent
        #     pre-merger history).
        #
        # The patch doc's "overlap -> fork" rule was a defence against
        # survivor-CIK consolidation (two pre-merger CIKs collapsing into
        # one post-merger CIK), but our point-in-time CIK lookup at the
        # `added` year returns those pre-merger CIKs and naturally forks.
        merged_into: str | None = None
        if cik:
            # Find the existing track for this CIK (there can be only one
            # since same-CIK merges by definition; suffixed #n only arises
            # for the __nocik_* placeholder case where CIK is None and a
            # single ticker produced multiple entries).
            for pid, track in perm_id_table.items():
                if track["cik"] != cik:
                    continue
                # Same CIK -> merge (extend aliases, possibly overlapping).
                _extend_track(track, entry)
                merged_into = pid
                break
        if merged_into is not None:
            continue

        # No merge possible -> start a new perm_id. If the candidate key
        # already exists (e.g., same cik+ticker overlapping collision), append
        # a numeric suffix to disambiguate.
        if cik:
            base_pid = f"{cik}_{ticker_u}"
        else:
            base_pid = f"__nocik_{ticker_u}"

        pid = base_pid
        suffix = 2
        while pid in perm_id_table:
            # Check if this collision is actually same-cik+ticker overlap.
            # If so, suffix the perm_id. Else just fresh-start.
            pid = f"{base_pid}#{suffix}"
            suffix += 1
        perm_id_table[pid] = _make_track_with_pid(entry, pid)

    # Step 3: Compute combined_intervals and finalize aliases ordering.
    for pid, track in perm_id_table.items():
        track["combined_intervals"] = merge_intervals(track["intervals"])

    perm_ids = list(perm_id_table.values())

    # Step 4a: Extend perm_ids with post-Wikipedia active aliases.
    #
    # When a perm_id's CIK is currently live in SEC's `company_tickers_exchange.json`
    # under a ticker symbol that's NOT in the perm_id's aliases, that ticker is
    # typically a rebrand/rename that happened AFTER Wikipedia's most recent
    # SP400 changes log entry for this company. To fetch post-rebrand prices we
    # must extend the perm_id with the new ticker alias.
    #
    # Safety guards (all three must hold to add the alias):
    #   (a) perm_id's combined_intervals must be OPEN (last removed=None), AND
    #   (b) NO OTHER perm_id exists that already has the active alias as its
    #       start_ticker AND has an OPEN combined interval (removed=None) --
    #       that perm_id owns the active ticker's tradable track; we must not
    #       claim it, AND
    #   (c) any CONFLICTING perm_id that has the active alias as start_ticker
    #       and a CLOSED interval must have been CLOSED BEFORE this perm_id's
    #       own most-recent combined interval started -- i.e. it must not
    #       temporally overlap our perm_id's open interval. (If they overlap,
    #       both tickers were live simultaneously on the exchange, meaning
    #       they were different companies that SEC consolidated under the same
    #       CIK post-merger/spinoff -- adding the alias would contaminate our
    #       perm_id's track.)
    #
    # Known limitation: cases like SAI->LDOS (post-split Leidos pre-2019 +
    # post-2019 newSAIC-bound LDOS-treated-as-CIK-1336920 currently) won't be
    # resolved by this guard and may need a manual override in Phase B.
    secu_active_alias_of_cik: dict[str, str] = {}
    for t, c in active_sec.items():
        secu_active_alias_of_cik.setdefault(c, t)
    start_ticker_owners: dict[str, list[str]] = defaultdict(list)
    for pid, track in perm_id_table.items():
        start_ticker_owners[track["start_ticker"]].append(pid)

    extended_count = 0
    extended_log: list[str] = []
    for pid, track in perm_id_table.items():
        cik = track.get("cik")
        if not cik:
            continue
        active_alias = secu_active_alias_of_cik.get(cik)
        if not active_alias or active_alias in track["aliases"]:
            continue
        ci = track.get("combined_intervals", [])
        if not ci:
            continue
        # Guard (a): perm_id's combined_intervals must be OPEN.
        last_ci = ci[-1]
        if last_ci.get("removed") is not None:
            # perm_id's most recent SP400 interval is CLOSED -- the company
            # has left the index, so post-Wikipedia rebrand is not relevant
            # for our event study (we only fetch prices during SP400 residency).
            extended_log.append(
                f"  skip {pid} cik={cik} active={active_alias} "
                f"(perm_id's last interval is closed -- no post-Wiki rebrand needed)"
            )
            continue
        # Guards (b)+(c): check conflicting perm_ids. The DIFFERENT-CIK CLOSED
        # case is the genuine acquirer-rebrand scenario (acquirer keeps its CIK,
        # retires target's ticker symbol but takes target's ticker name). In
        # this case our perm_id is the surviving entity and SHOULD adopt the
        # active alias post-target-delisting. So we only SKIP when:
        #   - conflicting perm_id is ALSO OPEN (both still live -> two
        #     different companies trading simultaneously; adopting the alias
        #     would contaminate), OR
        #   - conflicting perm_id's CIK == ours AND IS CLOSED (defensive --
        #     with our same-CIK-merges-to-one-perm_id invariant this should
        #     never happen, but skip to be safe).
        # We ALSO log the SAI/LDOS category of cases (DIFF-CIK CLOSED with
        # temporal overlap) so they can be audited separately.
        skip = False
        for other_pid in start_ticker_owners.get(active_alias, []):
            if other_pid == pid:
                continue
            other = perm_id_table[other_pid]
            oci = other.get("combined_intervals", [])
            if not oci:
                continue
            olast = oci[-1]
            ocik = other.get("cik")
            if olast.get("removed") is None:
                # Conflicting perm_id is ALSO OPEN.
                skip = True
                extended_log.append(
                    f"  skip {pid} cik={cik} active={active_alias} "
                    f"(conflict OPEN perm_id {other_pid} also owns that ticker)"
                )
                break
            if ocik == cik:
                # Conflicting perm_id has SAME CIK as us but is CLOSED; defensive.
                skip = True
                extended_log.append(
                    f"  skip {pid} cik={cik} active={active_alias} "
                    f"(conflict {other_pid} same-CIK but CLOSED; defensive skip)"
                )
                break
            # Conflicting perm_id is CLOSED with DIFFERENT CIK than us:
            # genuine acquirer-rebrand scenario -> we DO add the alias.
            # (The temporal overlap in Wikipedia intervals is just pre-merger
            #  co-listing in SP400; post-target-delisting the ticker belongs
            #  to us.)
        if skip:
            continue
        # Add active alias. Compute the alias's start date -- this is the
        # earliest plausible date the survivor trading under this alias began.
        # Cases:
        #   (i)  No conflicting perm_id: wiki-interval churn; use latest
        #        combined_intervals' added date.
        #  (ii)  Conflicting perm_id is CLOSED with DIFF-CIK acquirer-rebrand:
        #        use the max(target's removed date, our perm_id's latest
        #        combined interval's added date) as start.
        conflict_removed = None
        for other_pid in start_ticker_owners.get(active_alias, []):
            if other_pid == pid:
                continue
            other = perm_id_table[other_pid]
            oci = other.get("combined_intervals", [])
            if not oci:
                continue
            olast = oci[-1]
            if olast.get("removed") is None:
                continue
            if other.get("cik") == cik:
                continue
            r_ts = _to_ts(olast.get("removed"))
            if r_ts is None:
                continue
            conflict_removed = r_ts
        last_added_ts = _to_ts(last_ci.get("added"))
        if conflict_removed is None:
            start_str = last_ci.get("added")
        else:
            # prefer the later of conflict_removed and our last_added.
            if last_added_ts is None or conflict_removed > last_added_ts:
                start_str = conflict_removed.strftime("%Y-%m-%d")
            else:
                start_str = last_ci.get("added")
        new_iv = {"added": start_str, "removed": None}
        track["aliases"].append(active_alias)
        track["per_ticker_intervals"].setdefault(active_alias, []).append(new_iv)
        extended_count += 1
        extended_log.append(
            f"  add  {pid} cik={cik} active={active_alias} start={start_str}"
        )
    if extended_log:
        print(f"   [Step 4a] Post-Wiki active-alias extensions: {extended_count}")
        for line in extended_log[:120]:
            print(line)
        if len(extended_log) > 120:
            print(f"   ... and {len(extended_log) - 120} more (see downstream logs)")

    # Step 4b: Canonical-ticker selection -- "latest name wins".
    # Prefer the alias that appears in active SEC `company_tickers_exchange.json`.
    # Otherwise pick the alias with the most recent latest-added across this
    # perm_id's per_ticker_intervals (a rebrand's "newer" ticker symbol
    # tends to have the latest entry in /metadata/sp400).
    # Fallback: start_ticker (first alias of perm_id).
    for track in perm_ids:
        track["canonical_ticker"] = _select_canonical_ticker(track, active_sec, meta_indexed)

    # Step 5: extended per-ticker view (audit trail only). For each /metadata/sp400
    # ticker row, set cik_at_added to the CIK of its first interval's added year
    # (so analysts can see what was determined), and perm_id to a comma-joined
    # list of perm_ids this ticker contributed to (since a single ticker can
    # fork across multiple perm_ids via CIK change over time).
    extended = meta_df.copy()
    # Map ticker -> list of perm_ids (could be multiple if CIK differs across
    # intervals) and ticker -> first-interval CIK for the cik_at_added preview.
    ticker_to_perm_ids: dict[str, list[str]] = defaultdict(list)
    ticker_to_first_cik: dict[str, str | None] = {}
    for e in entries:
        ticker = e["ticker"]
        cik = e["cik"]
        if cik:
            pid = f"{cik}_{ticker.upper()}"
        else:
            pid = f"__nocik_{ticker.upper()}"
        # If the same ticker + cik pair had multiple overlapping intervals
        # (suffix disambiguation), record the one that actually matches
        # via ticker + entry counters is nontrivial. For audit purposes, we
        # simply list ALL perm_ids this ticker's entries contributed to:
        actual_pids = _find_actual_pids_for_entry(e, perm_id_table)
        for ap in actual_pids:
            if ap not in ticker_to_perm_ids[ticker]:
                ticker_to_perm_ids[ticker].append(ap)
        if ticker not in ticker_to_first_cik:
            ticker_to_first_cik[ticker] = cik

    extended["cik_at_added"] = extended["ticker"].map(lambda t: ticker_to_first_cik.get(t))
    extended["perm_id"] = extended["ticker"].map(
        lambda t: ",".join(ticker_to_perm_ids.get(t, [])) or None
    )

    print(f"   Wiki interval entries: {n_entries}")
    print(f"   Entries with CIK:      {n_with_cik} / {n_entries}")
    print(f"   perm_ids:              {len(perm_ids)}")
    multi = [c for c in perm_ids if len(c["aliases"]) > 1]
    print(f"   Multi-alias perm_ids:  {len(multi)}")
    return extended, perm_ids


def _find_actual_pids_for_entry(entry: dict, perm_id_table: dict[str, dict]) -> list[str]:
    """Best-effort reverse-lookup: find perm_ids whose per_ticker_intervals
    actually contain this entry's (ticker, added, removed).
    """
    matches: list[str] = []
    t = entry["ticker"].upper()
    a = _ts_to_str(entry["added"])
    r = _ts_to_str(entry["removed"])
    for pid, track in perm_id_table.items():
        ivs = track.get("per_ticker_intervals", {}).get(t, [])
        for iv in ivs:
            if iv.get("added") == a and iv.get("removed") == r:
                matches.append(pid)
                break
    return matches


def _select_canonical_ticker(
    track: dict,
    active_sec: dict[str, str],
    meta_indexed: pd.DataFrame,
) -> str:
    """Latest-name-rule: prefer the alias that's currently active in SEC.

    Tie breaks:
       - the alias whose /metadata/sp400 row has the most recent latest-added
         date (newer rebrand), and
       - finally alphabetical.
    Fallback: the perm_id's start_ticker.
    """
    aliases = list(track["aliases"])
    if not aliases:
        return track["start_ticker"]

    # 1. Active-SEC aliases -- only consider an alias that has the SAME CIK in
    # active_sec as our perm_id's CIK. This prevents the well-known
    # Covista/Adtalem + INCR/Syneos-style bug where SEC's
    # `company_tickers_exchange.json` has a stale ticker reassignment
    # mapping our CIK to a different company's ticker (or to a phantom).
    our_cik = track.get("cik")
    aliases_active_our_cik = [a for a in aliases if our_cik and active_sec.get(a.upper()) == our_cik]
    if aliases_active_our_cik:
        if len(aliases_active_our_cik) == 1:
            return aliases_active_our_cik[0]
        def _rich(a: str) -> tuple[pd.Timestamp, str]:
            try:
                row = meta_indexed.loc[a]
            except Exception:
                return (pd.Timestamp.min, a)
            ivs = _parse_intervals(row.get("intervals"))
            latest = _latest_added(ivs)
            la = latest if latest is not None else pd.Timestamp.min
            return (la, a)
        return sorted(aliases_active_our_cik, key=_rich, reverse=True)[0]
    # 2. No alias matches our perm_id CIK in active SEC: fall back to the
    # alias with the most recent latest-added date in /metadata/sp400
    # (a rebrand's newer ticker tends to have the most recent latest-added).

    # 2. Most recent latest-added date (with alpha tiebreak).
    def _rich(a: str) -> tuple[pd.Timestamp, str]:
        try:
            row = meta_indexed.loc[a]
        except Exception:
            return (pd.Timestamp.min, a)
        ivs = _parse_intervals(row.get("intervals"))
        latest = _latest_added(ivs)
        la = latest if latest is not None else pd.Timestamp.min
        return (la, a)
    return sorted(aliases, key=_rich, reverse=True)[0]


# ------------------------------------------------------------------
# Output writing
# ------------------------------------------------------------------

def _json_dumps(obj) -> str:
    return json.dumps(obj, default=str)


def _serialize_perm_id_row(row: dict) -> dict:
    out = dict(row)
    # Drop scratch fields used during construction.
    out.pop("intervals", None)
    out.pop("_fallback_alias_for_price", None)
    out["aliases"] = _json_dumps(list(row.get("aliases", [])))
    out["combined_intervals"] = _json_dumps(row.get("combined_intervals", []))
    out["per_ticker_intervals"] = _json_dumps(row.get("per_ticker_intervals", {}))
    # Normalize column order for table writing.
    cols = ["perm_id", "cik", "canonical_ticker", "aliases", "name",
            "sic", "index_ref", "combined_intervals",
            "per_ticker_intervals", "price_unavailable"]
    return {c: out.get(c) for c in cols}


def write_outputs(extended_meta: pd.DataFrame, perm_ids: list[dict]) -> None:
    """Persist both tables to db.h5 using the HDFStore('a') + remove() pattern.

    NEVER use mode='w' on an existing database (it truncates the whole file).
    """
    perm_ids_rows = [_serialize_perm_id_row(p) for p in perm_ids]
    perm_ids_df = pd.DataFrame(perm_ids_rows)
    if not perm_ids_df.empty:
        perm_ids_df["cik"] = perm_ids_df["cik"].astype(object)
        perm_ids_df["price_unavailable"] = perm_ids_df["price_unavailable"].astype(bool)

    with pd.HDFStore(DB_FILE, mode="a") as store:
        # Replace /metadata/sp400 (extended with cik_at_added + perm_id audit cols)
        if META_KEY in store:
            store.remove(META_KEY)
        store.put(META_KEY, extended_meta, format="table")
        print(f" Wrote {META_KEY} ({len(extended_meta)} rows)")

        # Write /metadata/sp400_perm_ids (NEW)
        if PERM_IDS_KEY in store:
            store.remove(PERM_IDS_KEY)
        if not perm_ids_df.empty:
            store.put(PERM_IDS_KEY, perm_ids_df, format="table")
            print(f" Wrote {PERM_IDS_KEY} ({len(perm_ids_df)} rows)")

        # Purge legacy /metadata/sp400_companies: it is superseded by
        # /metadata/sp400_perm_ids. Removing prevents downstream stages
        # from accidentally reading stale data during the Phase B-E refactor.
        if LEGACY_COMPANIES_KEY in store:
            store.remove(LEGACY_COMPANIES_KEY)
            print(f" Purged legacy {LEGACY_COMPANIES_KEY} (superseded by {PERM_IDS_KEY})")


# ------------------------------------------------------------------
# Audit report
# ------------------------------------------------------------------

def print_audit(extended_meta: pd.DataFrame, perm_ids: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("  PERM_ID FORK BUILD AUDIT")
    print("=" * 70)

    n_rows = len(extended_meta)
    n_cik_at_added = extended_meta["cik_at_added"].notna().sum() if "cik_at_added" in extended_meta.columns else 0
    print(f" /metadata/sp400 ticker rows      : {n_rows}")
    print(f"   with cik_at_added (CIK found)  : {n_cik_at_added} / {n_rows}")
    print(f" perm_ids built                    : {len(perm_ids)}")
    no_cik = [p for p in perm_ids if not p.get("cik")]
    print(f"   without CIK (__nocik_*)         : {len(no_cik)}")
    multi = [p for p in perm_ids if len(p["aliases"]) > 1]
    print(f"   multi-alias (rebrand)           : {len(multi)}")

    if multi:
        # Group multi-alias perm_ids by CIK to expose the M&A patterns.
        print("\n --- Multi-alias perm_ids (rebrand / overlap-merged) ---")
        for c in sorted(multi, key=lambda p: (-len(p["aliases"]), p["perm_id"])):
            n_iv = len(c["intervals"])
            print(f"   {c['perm_id']:25}  can={c['canonical_ticker']:6} "
                  f"[{len(c['aliases'])} alias, {n_iv} iv] <- {', '.join(c['aliases'])}")

    # List perm_ids that SHARE a CIK but are forked (the M&A split cases).
    print("\n --- CIK shared across perm_ids (M&A fork pattern) ---")
    cik_groups: dict[str, list[str]] = defaultdict(list)
    for p in perm_ids:
        if p.get("cik"):
            cik_groups[p["cik"]].append(p["perm_id"])
    shared = {k: v for k, v in cik_groups.items() if len(v) > 1}
    print(f"   {len(shared)} CIK(s) forked across {sum(len(v) for v in shared.values())} perm_ids")
    for cik, pids in sorted(shared.items(), key=lambda kv: -len(kv[1])):
        print(f"     CIK {cik}: {pids}")

    # List perm_ids whose aliases SHARE a ticker symbol (start_ticker) across
    # perm_ids — the M&A-survivor fork pattern (e.g. old Coherent Inc. CIK
    # 21510 + new Coherent Corp CIK 820318 both use ticker COHR at different
    # times). This is the inverse of the CIK-shared list above.
    print("\n --- Ticker aliases shared across DIFFERENT-CIK perm_ids ---")
    print("     (sym-swap case: old CIK retires ticker, new CIK adopts it)")
    start_ticker_owners: dict[str, list[str]] = defaultdict(list)
    for p in perm_ids:
        start_ticker_owners[p["start_ticker"]].append(p["perm_id"])
    ticker_shared = {t: pids for t, pids in start_ticker_owners.items() if len(pids) > 1}
    if ticker_shared:
        print(f"   {len(ticker_shared)} tickers shared across multiple perm_ids by start_ticker")
        for t, pids in sorted(ticker_shared.items(), key=lambda kv: -len(kv[1]))[:20]:
            ciks = []
            for pid in pids:
                trk = next((x for x in perm_ids if x["perm_id"] == pid), None)
                ciks.append(trk.get("cik") if trk else "-")
            print(f"     {t}: {list(zip(pids, ciks))}")
    else:
        print("   none")

    unavailable = [p for p in perm_ids if p["price_unavailable"]]
    print(f"\n perm_ids with price_unavailable=True: {len(unavailable)}")
    if unavailable:
        print("   These will be skipped by 03_data_gathering.py (Phase B).")
        for p in sorted(unavailable, key=lambda x: x["perm_id"]):
            fb = p.get("_fallback_alias_for_price", "")
            extra = f"  (alt alias OK: {fb})" if fb else ""
            print(f"   - {p['perm_id']:25} (can={p['canonical_ticker']}, "
                  f"aliases={p['aliases']}){extra}")
    print("=" * 70 + "\n")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  02b - Build Perm-ID Map (interval-forked, point-in-time CIK)")
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

    print("\n[2/4] Loading point-in-time CIK sources ...")
    pit_index = build_pit_cik_index()
    active_sec = load_active_sec_ciks()
    ticker_txt = load_cached_ticker_txt()
    print(f"   Active SEC tickers (current): {len(active_sec)}")
    print(f"   Cached ticker.txt       : {len(ticker_txt)}")

    print("\n[3/4] Building perm_id table (fork M&A, point-in-time CIK lookup) ...")
    extended, perm_ids = build_perm_id_map(meta_df, pit_index, active_sec, ticker_txt)

    print("\n[3b/4] Probing EODHD availability per perm_id canonical ticker ...")
    # Probe each perm_id's canonical ticker (one EODHD call per perm_id).
    # For multi-alias perm_ids, probe the canonical first; if unavailable,
    # the reprobe step walks the remaining aliases with the 15-year window.
    ticker_intervals = dict(zip(meta_df["ticker"].astype(str), meta_df["intervals"]))
    canonicals = [p["canonical_ticker"] for p in perm_ids]
    eodhd_avail = probe_tickers_on_eodhd(
        canonicals,
        progress_every=100,
        ticker_intervals=ticker_intervals,
    )
    for p in perm_ids:
        p["price_unavailable"] = not bool(eodhd_avail.get(p["canonical_ticker"], False))
    avail_ct = sum(1 for v in eodhd_avail.values() if v)
    print(f"   {avail_ct} / {len(canonicals)} canonicals available on EODHD")

    # Second-pass fallback: walk other aliases with the full 15-year window
    # for perm_ids still marked price_unavailable.
    reprobe_unavailable_perm_ids(perm_ids, eodhd_avail)

    print("\n[4/4] Writing outputs to db.h5 ...")
    write_outputs(extended, perm_ids)

    print_audit(extended, perm_ids)


if __name__ == "__main__":
    main()
