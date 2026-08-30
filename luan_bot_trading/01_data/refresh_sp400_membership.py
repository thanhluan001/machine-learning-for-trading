#!/usr/bin/env python3
"""refresh_sp400_membership.py — Monthly S&P 400 membership refresh.

Re-parses Wikipedia S&P 400 constituents + changes (free, no paid API),
updates /metadata/sp400 with fresh membership intervals, then propagates the
updates into /metadata/sp400_permatickers by matching on the permaTicker's
`added` date — WITHOUT the expensive per-ticker Tiingo disambiguation in
02b_build_company_map.py.

WHY
---
The graduated-stock case is the dangerous one: a stock moved to the S&P 500
but its persisted `wikipedia_intervals` still shows the last interval open,
so live inference (_get_current_sp400_members) wrongly treats it as a current
mid-cap. This is exactly the AMD/isActive bug. Wikipedia is the source of
truth for add/remove dates; refresh it regularly.

SAFETY
------
A permaTicker only stays "current" if a fresh interval with a MATCHING
`added` date is still open. If the ticker is now held by a different company
(recycled ticker code), the old permaTicker's added date won't match and its
last interval is CLOSED (never wrongly re-opened). Genuinely new ticker codes
with no permaTicker mapping are flagged for a one-time 02b_build_company_map.py
re-run; they are not silently attached to an unrelated permaTicker.

USAGE
-----
    conda run -n trading python luan_bot_trading/01_data/refresh_sp400_membership.py
    conda run -n trading python luan_bot_trading/01_data/refresh_sp400_membership.py --dry-run

Run monthly (e.g. first inference run of each month). Dry-run reports the
diff without writing.
"""
from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

import pandas as pd

DB_FILE = Path(__file__).resolve().parent / "db.h5"
META_KEY = "/metadata/sp400"
PERMATICKERS_KEY = "/metadata/sp400_permatickers"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Reuse the exact Wikipedia parsing from 01_metadata_gathering.py so the
# refresh never diverges from the original interval-build logic.
_meta = _load("meta_gather", Path(__file__).resolve().parent / "01_metadata_gathering.py")


def parse_intervals(raw) -> list[dict]:
    """Parse the JSON interval string into a list of {added, removed} dicts."""
    if raw is None:
        return []
    if isinstance(raw, float) and pd.isna(raw):
        return []
    s = raw
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


def is_current(intervals: list[dict]) -> bool:
    """Match live inference's current-membership test exactly."""
    if not intervals:
        return True  # defensive; matches _get_current_sp400_members
    last = intervals[-1]
    removed = last.get("removed")
    return removed is None or (isinstance(removed, str) and removed.strip() == "")


def serialize_intervals(intervals: list[dict]) -> str:
    return json.dumps(
        [{"added": iv.get("added"), "removed": iv.get("removed")} for iv in intervals],
        default=str,
    )


def fetch_fresh_meta() -> pd.DataFrame:
    """Re-parse Wikipedia -> fresh ticker-level metadata with intervals."""
    df = _meta.build_unified_metadata()
    if df is None or df.empty:
        raise RuntimeError("Wikipedia parse returned no rows")
    df["ticker"] = df["ticker"].astype(str)
    return df


def load_existing():
    with pd.HDFStore(DB_FILE, mode="r") as store:
        meta = store[META_KEY] if META_KEY in store else pd.DataFrame()
        pt = store[PERMATICKERS_KEY] if PERMATICKERS_KEY in store else pd.DataFrame()
    return meta, pt


def merge_fresh_into_meta(existing: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    """Build the new /metadata/sp400 from fresh Wikipedia data, preserving
    SEC-derived columns (sic, index_ref, cik) for tickers we already had.
    """
    sec_cols = [c for c in ("sic", "index_ref", "cik") if c in existing.columns]
    if sec_cols:
        sec = existing[["ticker"] + sec_cols].drop_duplicates("ticker").set_index("ticker")
        out = fresh.set_index("ticker").join(sec, how="left").reset_index().rename(columns={"index": "ticker"})
        out["ticker"] = out["ticker"].astype(str)
        return out
    return fresh


def propagate_to_permatickers(pt: pd.DataFrame, fresh_by_ticker: dict) -> tuple[pd.DataFrame, list, list, list]:
    """Update wikipedia_intervals in the permaTicker table.

    Safety rule: match permaTicker intervals to fresh intervals by `added`
    date. A permaTicker stays current only if a matching fresh interval is
    still open. Unmatched open intervals are CLOSED (graduated/replaced).
    Returns (updated_pt, graduated, reopened, ambiguous).
    """
    today_iso = pd.Timestamp.now().normalize().strftime("%Y-%m-%d")
    graduated, reopened, ambiguous = [], [], []
    new_intervals_col = []
    for row in pt.itertuples(index=False):
        intervals = parse_intervals(row.wikipedia_intervals)
        ticker = str(getattr(row, "canonical_ticker", "") or "")
        fresh = fresh_by_ticker.get(ticker, [])
        fresh_by_added = {iv["added"]: iv for iv in fresh if iv.get("added")}
        fresh_added_dates = set(fresh_by_added.keys())

        new_intervals = []
        for iv in intervals:
            added = iv.get("added")
            if added in fresh_by_added:
                # Owned by this permaTicker in fresh Wikipedia; use fresh
                # version so an updated removed date (graduation) flows in.
                new_intervals.append(fresh_by_added[added])
            else:
                # permaTicker interval not represented in fresh Wikipedia.
                # If still open, the company has graduated/been replaced.
                if iv.get("removed") is None:
                    new_iv = dict(iv)
                    new_iv["removed"] = today_iso
                    new_iv["_refresh_closing"] = today_iso
                    new_intervals.append(new_iv)
                    graduated.append(
                        {"permaTicker": getattr(row, "permaTicker", ""),
                         "canonical_ticker": ticker,
                         "added": added,
                         "closed_on": today_iso}
                    )
                else:
                    new_intervals.append(iv)

        # Detect re-entry: a fresh open interval whose added date matches one
        # of this permaTicker's historical added dates but was previously
        # closed. (append already handled because added matched fresh_by_added)
        owned_added = {iv.get("added") for iv in intervals}
        for iv in fresh:
            added = iv.get("added")
            if added in owned_added and iv.get("removed") is None:
                was_closed = all(
                    (x.get("added") != added) or (x.get("removed") is not None)
                    for x in intervals
                )
                if was_closed:
                    reopened.append({"permaTicker": getattr(row, "permaTicker", ""),
                                     "canonical_ticker": ticker, "added": added})
            elif added not in owned_added and iv.get("removed") is None:
                # Fresh open interval that this permaTicker never owned.
                # Likely a different company now holds this ticker code.
                ambiguous.append({"permaTicker": getattr(row, "permaTicker", ""),
                                  "canonical_ticker": ticker,
                                  "unmatched_added": added})

        new_intervals.sort(key=lambda iv: (iv.get("added") or ""))
        new_intervals_col.append(serialize_intervals(new_intervals))

    out = pt.copy()
    out["wikipedia_intervals"] = new_intervals_col
    return out, graduated, reopened, ambiguous


def main(dry_run: bool = False) -> None:
    bar = "=" * 78
    print(bar)
    print("  refresh_sp400_membership.py — monthly S&P 400 membership refresh")
    print(bar)
    print(f"  DB:      {DB_FILE}")
    print(f"  Dry run: {dry_run}")

    print("\n[1] Re-parsing Wikipedia S&P 400 constituents + changes ...")
    fresh = fetch_fresh_meta()

    # --- Defensive closure of stale changes-table rows ---------------------
    # Wikipedia's changes table sometimes never records a removal (ticker
    # renamed/acquired/absorbed). Such rows have an OPEN interval, an empty
    # name, and are NOT in the 400-constituents table. They inflate the
    # "current" set (432 vs 400) and can never map to a permaTicker.
    # NEVER delete (point-in-time ledger) — close the interval at refresh
    # date instead: removal happened sometime before today; exact date
    # unknown, recorded late. Same philosophy as the recycled-ticker close.
    constituents_tickers = {
        str(t) for t in _meta.fetch_constituents().index
    }
    today_str = pd.Timestamp.now().strftime("%Y-%m-%d")
    _closed_rows = []
    _out = []
    for r in fresh.itertuples(index=False):
        tk = str(r.ticker)
        name_val = getattr(r, "name", "")
        name_empty = (name_val is None) or (isinstance(name_val, float)) or (not str(name_val).strip())
        ivs = parse_intervals(r.intervals)
        if is_current(ivs) and tk not in constituents_tickers and name_empty:
            if ivs:
                ivs[-1]["removed"] = today_str
            r = r._replace(intervals=serialize_intervals(ivs))
            _closed_rows.append(tk)
        _out.append(r)
    if _closed_rows:
        fresh = pd.DataFrame(_out)
        print(f"  Defensively closed {len(_closed_rows)} stale nameless rows "
              f"(open interval, not in constituents table): "
              f"{', '.join(sorted(_closed_rows)[:20])}")

    fresh_by_ticker = {
        str(r.ticker): parse_intervals(r.intervals) for r in fresh.itertuples(index=False)
    }
    fresh_current_tickers = {
        str(r.ticker) for r in fresh.itertuples(index=False)
        if is_current(parse_intervals(r.intervals))
    }
    print(f"  Fresh tickers: {len(fresh)} | current constituents: {len(fresh_current_tickers)}")

    print("\n[2] Loading existing metadata ...")
    existing_meta, pt = load_existing()
    if pt.empty:
        raise RuntimeError(
            f"{PERMATICKERS_KEY} missing — run 02b_build_company_map.py first."
        )
    print(f"  /metadata/sp400: {len(existing_meta)} rows")
    print(f"  /metadata/sp400_permatickers: {len(pt)} rows")

    # Current membership BEFORE refresh.
    pt_current_before = {
        str(getattr(r, "permaTicker", "")): str(getattr(r, "canonical_ticker", ""))
        for r in pt.itertuples(index=False)
        if is_current(parse_intervals(r.wikipedia_intervals))
    }
    print(f"  Current permaTickers BEFORE: {len(pt_current_before)}")

    print("\n[3] Propagating intervals into permaTicker table ...")
    pt_new, graduated, reopened, ambiguous = propagate_to_permatickers(pt, fresh_by_ticker)

    pt_current_after = {
        str(getattr(r, "permaTicker", "")): str(getattr(r, "canonical_ticker", ""))
        for r in pt_new.itertuples(index=False)
        if is_current(parse_intervals(r.wikipedia_intervals))
    }
    print(f"  Current permaTickers AFTER:  {len(pt_current_after)}")

    # Graduated = current before but not current after (clearest definition).
    graduated_keys = set(pt_current_before) - set(pt_current_after)
    graduated = [
        {"permaTicker": ptid, "canonical_ticker": pt_current_before[ptid]}
        for ptid in graduated_keys
    ]
    graduated.sort(key=lambda g: g["canonical_ticker"])

    # New constituents in Wikipedia that have no permaTicker mapping at all.
    known_tickers = {str(getattr(r, "canonical_ticker", "")) for r in pt.itertuples(index=False)}
    new_constituents = sorted(fresh_current_tickers - known_tickers)

    print("\n[4] Change report")
    print(f"  Graduated / closed (no longer current): {len(graduated)}")
    for g in graduated[:50]:
        print(f"    - {g['canonical_ticker']} ({g['permaTicker']})")
    if len(graduated) > 50:
        print(f"    ... and {len(graduated) - 50} more")
    print(f"  Re-opened (re-entry of same company):   {len(reopened)}")
    for g in reopened[:50]:
        print(f"    + {g['canonical_ticker']} ({g['permaTicker']}) added={g['added']}")
    print(f"  Ambiguous (ticker possibly recycled):   {len(ambiguous)}")
    for g in ambiguous[:50]:
        print(f"    ? {g['canonical_ticker']} ({g['permaTicker']}) unmatched_added={g['unmatched_added']}")
    print(f"  New constituents with NO permaTicker (need 02b re-run): {len(new_constituents)}")
    for t in new_constituents[:80]:
        print(f"    * {t}")

    if dry_run:
        print("\n  [DRY] No writes performed.")
        return

    print("\n[5] Writing updates ...")
    new_meta = merge_fresh_into_meta(existing_meta, fresh) if not existing_meta.empty else fresh
    with pd.HDFStore(DB_FILE, mode="a") as store:
        if META_KEY in store:
            store.remove(META_KEY)
        store.put(META_KEY, new_meta, format="table")
        if PERMATICKERS_KEY in store:
            store.remove(PERMATICKERS_KEY)
        store.put(PERMATICKERS_KEY, pt_new, format="table")
    print(f"  Wrote {META_KEY} ({len(new_meta)} rows)")
    print(f"  Wrote {PERMATICKERS_KEY} ({len(pt_new)} rows)")

    if graduated or ambiguous or new_constituents:
        print("\n  NOTE:")
        if graduated:
            print("    - Graduated tickers are now correctly excluded from live inference.")
        if ambiguous:
            print("    - Ambiguous/recycled tickers were closed defensively. If a genuinely")
            print("      new company now holds one of these ticker codes, run")
            print("      02b_build_company_map.py to disambiguate the new permaTicker.")
        if new_constituents:
            print("    - New constituents without a permaTicker mapping are NOT yet tradable.")
            print("      Run 02b_build_company_map.py to map them (Tiingo search).")
    print(bar)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Monthly S&P 400 membership refresh")
    p.add_argument("--dry-run", action="store_true", help="Report the diff without writing")
    args = p.parse_args()
    main(dry_run=args.dry_run)
