#!/usr/bin/env python3
"""
Metadata Gathering - S&P 400 Company Info
Fetches sector/metadata + historical changes from Wikipedia.
Stores unified table in db.h5 under /metadata/sp400 with:
- ticker, name, gics_sector, gics_sub_industry
- intervals: list of {"added": date|None, "removed": date|None}
"""
import json
from pathlib import Path
import pandas as pd
import requests
from bs4 import BeautifulSoup

DB_FILE = Path(__file__).parent / "db.h5"
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"


def fetch_constituents() -> pd.DataFrame:
    """Fetch current S&P 400 constituents (Table 1)."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }
    resp = requests.get(WIKI_URL, headers=headers)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table", {"id": "constituents"})
    if not table:
        table = soup.find("table", {"class": "wikitable"})
    rows = []
    for row in table.find_all("tr")[1:]:
        cols = row.find_all("td")
        if len(cols) >= 4:
            rows.append(
                {
                    "ticker": cols[0].text.strip().replace(".", "-"),
                    "name": cols[1].text.strip(),
                    "gics_sector": cols[2].text.strip(),
                    "gics_sub_industry": cols[3].text.strip(),
                }
            )
    df = pd.DataFrame(rows).set_index("ticker")
    return df


def fetch_changes() -> pd.DataFrame:
    """Fetch historical component changes (Table 2)."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }
    resp = requests.get(WIKI_URL, headers=headers)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    tables = soup.find_all("table", {"class": "wikitable"})
    if len(tables) < 2:
        raise ValueError("Could not find changes table on Wikipedia page")
    table = tables[1]
    rows = []
    current_date = None
    for tr in table.find_all("tr")[2:]:
        cols = tr.find_all("td")
        if not cols:
            continue
        first_rowspan = cols[0].get("rowspan", "1") if cols else "1"
        try:
            rs = int(first_rowspan)
        except (ValueError, TypeError):
            rs = 1
        if rs > 1 or len(cols) >= 6:
            current_date = cols[0].text.strip()
            if len(cols) >= 6:
                rows.append(
                    {
                        "date": current_date,
                        "added_ticker": cols[1].text.strip().replace(".", "-"),
                        "added_security": cols[2].text.strip(),
                        "removed_ticker": cols[3].text.strip().replace(".", "-"),
                        "removed_security": cols[4].text.strip(),
                    }
                )
        elif len(cols) == 4 and current_date:
            rows.append(
                {
                    "date": current_date,
                    "added_ticker": cols[0].text.strip().replace(".", "-"),
                    "added_security": cols[1].text.strip(),
                    "removed_ticker": cols[2].text.strip().replace(".", "-"),
                    "removed_security": cols[3].text.strip(),
                }
            )
        elif len(cols) == 5 and current_date:
            rows.append(
                {
                    "date": current_date,
                    "added_ticker": cols[0].text.strip().replace(".", "-"),
                    "added_security": cols[1].text.strip(),
                    "removed_ticker": cols[2].text.strip().replace(".", "-"),
                    "removed_security": cols[3].text.strip(),
                }
            )
    df = pd.DataFrame(rows)
    df["added_ticker"] = df["added_ticker"].replace("", pd.NA)
    df["removed_ticker"] = df["removed_ticker"].replace("", pd.NA)
    return df


def build_unified_metadata() -> pd.DataFrame:
    """Merge constituents + changes into one table with interval arrays.

    Returns DataFrame with columns:
        ticker, name, gics_sector, gics_sub_industry, intervals
    where intervals is a list of dicts:
        [{"added": "YYYY-MM-DD", "removed": "YYYY-MM-DD"|None}, ...]
    """
    print("Fetching constituents...")
    constituents = fetch_constituents()
    print(f" Constituents: {len(constituents)}")

    print("Fetching changes...")
    changes = fetch_changes()
    print(f" Changes: {len(changes)}")

    changes["date"] = (
        changes["date"]
        .str.replace(r"\[\d+\]", "", regex=True)
        .str.strip()
    )
    changes["date"] = pd.to_datetime(changes["date"], errors="coerce")
    changes = changes.dropna(subset=["date"])

    changes = changes.sort_values("date").reset_index(drop=True)
    ticker_intervals = {}

    # Step 1: Build raw intervals from the Wikipedia changes table.
    # - Every "added_ticker" opens a new open-ended interval.
    # - Every "removed_ticker" closes the most recent open interval.
    # - If a ticker appears only as removed with no prior open interval,
    #   it means it was in the index before Wikipedia history starts.
    #   We record the removal as a terminal interval with added=NaT,
    #   and let the backfill step fill in the missing added date.
    for _, row in changes.iterrows():
        added_ticker = row["added_ticker"]
        removed_ticker = row["removed_ticker"]
        date = row["date"]

        if pd.notna(added_ticker):
            t = str(added_ticker)
            ticker_intervals.setdefault(t, []).append(
                {"added": date, "removed": pd.NaT}
            )

        if pd.notna(removed_ticker):
            rt = str(removed_ticker)
            ticker_intervals.setdefault(rt, [])
            if ticker_intervals[rt] and pd.isna(
                ticker_intervals[rt][-1]["removed"]
            ):
                ticker_intervals[rt][-1]["removed"] = date
            else:
                # Ticker appears only as removed with no prior open interval.
                # Record as: {"added": NaT, "removed": date}
                # This will be backfilled to 2012-01-01 later.
                ticker_intervals[rt].append(
                    {"added": pd.NaT, "removed": date}
                )

    # Step 2: Backfill missing added dates for ALL tickers.
    # Wikipedia history cuts off before 2012. Three cases need backfill:
    # 1) Ticker is a current constituent with NO intervals in ticker_intervals.
    # 2) Ticker has empty interval list in ticker_intervals.
    # 3) Earliest interval has added=NaT (removed-only ticker from Step 1).
    # For all cases, assign the conservative 2012-01-01 start date and
    # preserve any captured removed date.
    # Design.md says: backfill for ALL tickers, including removed ones.
    backfill_start = pd.Timestamp("2012-01-01")
    all_tickers = set(constituents.index) | set(ticker_intervals.keys())
    for t in all_tickers:
        intervals = ticker_intervals.get(t, [])
        if not intervals:
            ticker_intervals[t] = [
                {"added": backfill_start, "removed": pd.NaT}
            ]
        elif pd.isna(intervals[0].get("added")):
            # Preserve the captured removed date; only fill added.
            intervals[0]["added"] = backfill_start

    # Step 2b: Merge null-added intervals into the previous interval.
    # Some removed-only tickers from Step 1 created terminal intervals with
    # added=NaT. If such an interval is NOT the first interval, it means the
    # stock was in the index between the two captured removal events. We merge
    # by extending the previous interval's removed date to this one's removed.
    for t in list(ticker_intervals.keys()):
        intervals = ticker_intervals[t]
        merged = []
        for iv in intervals:
            added = iv.get("added")
            if pd.isna(added):
                if merged:
                    merged[-1]["removed"] = iv["removed"]
                else:
                    # Orphan at position 0 => backfill to 2012-01-01
                    iv["added"] = backfill_start
                    merged.append(iv)
            else:
                merged.append(iv)
        ticker_intervals[t] = merged

    def serialize(intervals):
        out = []
        for iv in intervals:
            out.append(
                {
                    "added": iv["added"].date().isoformat()
                    if pd.notna(iv["added"])
                    else None,
                    "removed": iv["removed"].date().isoformat()
                    if pd.notna(iv["removed"])
                    else None,
                }
            )
        return out

    # Step 3: Build one row per ticker for storage in /metadata/sp400.
    rows = []
    for t in sorted(set(list(constituents.index) + list(ticker_intervals.keys()))):
        name = (
            constituents.loc[t, "name"]
            if t in constituents.index
            else ""
        )
        gics_sector = (
            constituents.loc[t, "gics_sector"]
            if t in constituents.index
            else ""
        )
        gics_sub_industry = (
            constituents.loc[t, "gics_sub_industry"]
            if t in constituents.index
            else ""
        )
        intervals = ticker_intervals.get(t, [])
        rows.append(
            {
                "ticker": str(t),
                "name": name,
                "gics_sector": gics_sector,
                "gics_sub_industry": gics_sub_industry,
                "intervals": json.dumps(serialize(intervals)),
            }
        )

    return pd.DataFrame(rows)


def store_metadata(df: pd.DataFrame):
    h5_path = "/metadata/sp400"
    if df.index.name == "ticker" or "ticker" not in df.columns:
        df = df.reset_index()
    df["ticker"] = df["ticker"].astype(str)
    # Append mode + drop only the metadata node; never wipe the rest of db.h5
    # (the old `mode="w"` truncated the entire file, deleting /sp400, /macros, /earnings).
    with pd.HDFStore(DB_FILE, mode="a") as store:
        if h5_path in store:
            store.remove(h5_path)
        store.put(h5_path, df, format="table")
    print(f"Stored {len(df)} rows to {h5_path}")


def main():
    print("Building unified SP400 metadata...")
    df = build_unified_metadata()
    print(df.head(10).to_string())
    print(f"\nTotal: {len(df)} tickers")
    print(f"Sample intervals for ALK:")
    print(df[df["ticker"] == "ALK"]["intervals"].iloc[0])
    store_metadata(df)


if __name__ == "__main__":
    main()
