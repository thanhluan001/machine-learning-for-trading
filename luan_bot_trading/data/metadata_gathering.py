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
    # Replace empty strings with NaN so they do not create phantom tickers
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

    changes = changes.sort_values("date").reset_index(drop=True)
    ticker_intervals = {}

    for _, row in changes.iterrows():
        added_ticker = row["added_ticker"]
        removed_ticker = row["removed_ticker"]
        date = row["date"]
        if pd.isna(date):
            continue

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
                ticker_intervals[rt].append(
                    {"added": date, "removed": date}
                )

    now = pd.Timestamp.today().normalize()
    for t in constituents.index:
        intervals = ticker_intervals.setdefault(t, [])
        if not intervals or pd.notna(intervals[-1].get("removed")):
            intervals.append(
                {"added": now - pd.Timedelta(days=365), "removed": pd.NaT}
            )

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
        if not intervals:
            intervals = [{"added": pd.Timestamp("2012-01-01"), "removed": pd.NaT}]
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
    df.to_hdf(DB_FILE, key=h5_path, mode="w", format="table")
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

