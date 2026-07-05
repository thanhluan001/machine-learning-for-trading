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

See `luan_bot_trading/company_merge_design.md` for the full design.
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
# Abutting intervals within this many days are merged into one span.
ABUT_DAYS = 7
# Tiingo verification throttle (the /daily/{ticker} metadata endpoint).
TIINGO_PROBE_DELAY = 0.4

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

def verify_tickers_on_tiingo(tickers: list[str], progress_every: int = 50) -> dict[str, bool]:
    """Probe Tiingo /daily/{ticker} metadata endpoint for each ticker.

    Cheap probe: returns 200 if Tiingo has the ticker, 404 otherwise. Does NOT
    consume the price-history request quota.

    Returns:
        dict[ticker -> bool] of availability.
    """
    if not TIINGO_API_KEY:
        print("  [verify_tickers_on_tiingo] TIINGO_API_KEY missing; marking all unknown")
        return {t: False for t in tickers}

    results: dict[str, bool] = {}
    for i, t in enumerate(tickers, 1):
        try:
            r = requests.get(
                f"https://api.tiingo.com/tiingo/daily/{t}",
                params={"token": TIINGO_API_KEY},
                timeout=15,
            )
            results[t] = (r.status_code == 200)
        except Exception as e:
            print(f"   {t}: probe error: {e}")
            results[t] = False
        time.sleep(TIINGO_PROBE_DELAY)
        if i % progress_every == 0:
            print(f"   probe progress: {i}/{len(tickers)}")
    return results


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

    print("\n[3/4] Probing Tiingo availability for ALL tickers in /metadata/sp400...")
    all_tickers = sorted(set(meta_df["ticker"].astype(str).tolist()), key=str.upper)
    tiingo_av = verify_tickers_on_tiingo(all_tickers)
    avail_ct = sum(1 for v in tiingo_av.values() if v)
    print(f"   {avail_ct} / {len(all_tickers)} tickers available on Tiingo")

    print("\n[4/4] Building company groups, merging intervals ...")
    extended, companies = build_company_map(meta_df, tiingo_av)

    print("\n Writing outputs to db.h5 ...")
    write_outputs(extended, companies)
    reset_offset()

    print_audit(extended, companies)


if __name__ == "__main__":
    main()
