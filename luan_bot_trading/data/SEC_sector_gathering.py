#!/usr/bin/env python3
"""SEC Sector Gathering
Retrieves SIC codes from SEC EDGAR for all tickers in /metadata/sp400 and updates
that same HDF5 node with two columns: sic, index_ref. Uses historical DERA Q4
snapshots as fallback for missing / delisted tickers.
"""

import io
import json
import math
import os
import re
import time
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

DB_FILE = Path(__file__).parent / "db.h5"
SEC_TICKERS_URL = "https://www.sec.gov/include/ticker.txt"
_EMAIL = os.getenv("email", "")
SEC_HEADERS = {
    "User-Agent": f"PEAD-Bot/1.0 ({_EMAIL})"
}
SEC_DELAY_SECONDS = 0.50  # Stay under 10 req/sec

# Hardcoded patch for legacy / renamed / acquired tickers that no automated
# SEC source resolves cleanly. Sourced manually from EDGAR company records.
SIC_PATCH = {
    'AINV': '6159', 'ALTM': '2819', 'CADE': '6022', 'CCP': '6798',
    'CLC': '3714', 'CNH': '3523', 'CNVR': '6770', 'FLG': '6331',
    'KLXI': '1389', 'LPT': '6798', 'OA': '3760', 'OZK': '6022',
    'SN': '1311', 'TCF': '6021', 'TXNM': '4911', 'WYND': '7011'
}


def load_tickers_from_metadata() -> pd.DataFrame:
    """Load tickers from /metadata/sp400 in db.h5."""
    df = pd.read_hdf(DB_FILE, key="/metadata/sp400")
    return df.copy()


def fetch_sec_ticker_map() -> dict:
    """Fetch SEC ticker-to-CIK mapping, with 30-day local cache.

    Uses current ticker.txt fallback for active tickers; delisted tickers will
    be recovered later via historical DERA snapshots.
    """
    cache_path = Path(__file__).parent / "sec_cache" / "ticker.txt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_age_days = None
    if cache_path.exists():
        cache_age_days = (datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)).days
    if cache_path.exists() and cache_age_days is not None and cache_age_days < 30:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = f.read()
        print(f" Loaded SEC ticker map from cache ({cache_age_days} days old).")
    else:
        resp = requests.get(SEC_TICKERS_URL, headers=SEC_HEADERS)
        resp.raise_for_status()
        data = resp.text
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(data)
        print(f" Downloaded SEC ticker map and cached to {cache_path}")

    mapping = {}
    for raw_line in data.splitlines():
        parts = [part.strip() for part in raw_line.split()]
        if len(parts) < 2:
            continue
        ticker, cik_raw = parts[0], parts[1]
        if not ticker or not cik_raw:
            continue
        mapping[ticker.upper()] = str(cik_raw).zfill(10)
    return mapping


def fetch_sic_code(cik: str) -> str | None:
    """Fetch the SIC code for a given CIK from SEC EDGAR."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    try:
        time.sleep(SEC_DELAY_SECONDS)
        resp = requests.get(url, headers=SEC_HEADERS)
        resp.raise_for_status()
        return resp.json().get("sic")
    except Exception:
        return None


import re


def clean_and_match_ticker(target_ticker: str, sec_data: dict) -> str | None:
    """Cleans up the bot's target ticker string and runs a multi-pass match
    against SEC records to catch preferred shares and structural classes."""
    target = target_ticker.upper().strip()
    # Pass 1: Exact Match (Best case scenario)
    if target in sec_data:
        return sec_data[target]
    # Pass 2: Base Ticker Normalization
    base_ticker = re.split(r"[-._/\s]", target)[0]
    if base_ticker in sec_data:
        return sec_data[base_ticker]
    # Pass 3: Reverse Suffix Match
    for sec_ticker, cik in sec_data.items():
        if sec_ticker.startswith(base_ticker + "-") or sec_ticker.startswith(base_ticker + "."):
            return cik
    return None


def get_index_ref(sic_code: str | None) -> str | None:
    """Map SIC code to the S&P 400 sector index ticker.

    Follows SIC_code_to_index.md: range buckets + structural overrides.
    Override precedence: 6770 -> IJH, 283x -> IJK, 357x -> IJK, 367x -> IJK.
    """
    s = None if sic_code is None else str(sic_code)
    if not s or s.lower() in {"nan", "none", "null", "nat"} or len(s) < 2:
        return "IJH"

    # --- Structural Classification Overrides (highest precedence) ---
    if s.startswith("6770"):
        return "IJH"          # Blank Checks / restructuring entities
    if s.startswith("283"):
        return "IJK"          # Pharmaceuticals bypass Materials
    if s.startswith("357"):
        return "IJK"          # Computing Hardware bypasses Industrials
    if s.startswith("367"):
        return "IJK"          # Semiconductors bypass Consumer Cyclicals

    prefix = int(s[:2])

    # --- Deterministic SIC-to-Index Component Map (ranges) ---
    if prefix in (13, 29):
        return "IJS"          # Energy (13xx, 29xx)
    if (10 <= prefix <= 14) or prefix in (24, 25, 26, 28, 33):
        return "XLB"         # Materials (10xx-14xx, 24xx-26xx, 28xx, 33xx) excl. 283x handled above
    if (15 <= prefix <= 17) or prefix in (34, 35, 37) or (40 <= prefix <= 47):
        return "IJJ"         # Industrials & Cyclicals (15xx-17xx, 34xx, 35xx, 37xx, 40xx-47xx)
    if prefix in (20, 21, 22, 23, 30, 31, 36, 39, 51, 54) or (52 <= prefix <= 59) or (70 <= prefix <= 72) or (75 <= prefix <= 77):
        return "IJJ"         # Consumer Staples & Discretionary (Value) excl. 367x handled above
    if (60 <= prefix <= 64) or prefix == 67:
        return "XLF"         # Financials (60xx-64xx, 67xx) excl. 6770 handled above
    if prefix == 65:
        return "XLRE"        # Real Estate
    if prefix in (38, 48, 73, 78, 79, 80, 27):
        return "IJK"         # Technology, Healthcare & Growth Services
    if prefix == 49:
        return "XLU"         # Utilities
    return "IJH"              # Default Mapping


def build_sic_map(tickers: list[str], ticker_to_cik: dict, meta_df: pd.DataFrame | None = None) -> dict:
    """Build ticker -> SIC code mapping from SEC EDGAR with historical DERA fallback."""
    sic_map = {}
    still_missing = []
    for i, ticker in enumerate(tickers, 1):
        cik = ticker_to_cik.get(ticker.upper().strip())
        if not cik:
            cik = clean_and_match_ticker(ticker, ticker_to_cik)
        if not cik:
            sic_map[ticker] = None
            still_missing.append(ticker)
            continue
        sic = fetch_sic_code(cik)
        # Treat empty-string SIC (some SEC submissions return "") as a miss so
        # the patch fallback can fill it.
        if not sic:
            sic_map[ticker] = None
            still_missing.append(ticker)
            continue
        sic_map[ticker] = sic
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(tickers)} tickers...")

    if not still_missing:
        return sic_map

    print(f" Attempting historical DERA recovery for {len(still_missing)} missing tickers...")
    meta_for_fallback = meta_df if meta_df is not None else load_tickers_from_metadata()
    hist_map = build_historical_sic_map(still_missing, meta_for_fallback)

    recovered = 0
    for ticker in still_missing:
        entry = hist_map.get(ticker)
        if not entry:
            continue
        cik = entry.get("CIK")
        sic = entry.get("SIC")
        if cik and not sic:
            sic = fetch_sic_code(cik)
        if sic:
            sic_map[ticker] = sic
            recovered += 1

    print(f" Recovered SIC via DERA: {recovered}/{len(still_missing)}")

    # Tier 3: hardcoded patch for legacy tickers that no automated source resolves.
    still_missing = [t for t in still_missing if not sic_map.get(t)]
    if not still_missing:
        return sic_map

    patched = 0
    for ticker in still_missing:
        sic = SIC_PATCH.get(ticker.upper().strip())
        if sic:
            sic_map[ticker] = sic
            patched += 1

    print(f" Recovered SIC via hardcoded patch: {patched}/{len(still_missing)}")
    return sic_map


def _parse_intervals(intervals_raw):
    """Parse stored JSON interval strings into Python objects."""
    if intervals_raw is None:
        return []
    raw = str(intervals_raw)
    if raw.strip() in {"", "nan", "None", "[]", "{}"}:
        return []
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []


def get_latest_active_year_from_intervals(ticker: str, meta_df: pd.DataFrame) -> int | None:
    """Return the latest calendar year the ticker was active in the SP400 based on metadata intervals."""
    row = meta_df.loc[meta_df["ticker"] == ticker]
    if row.empty:
        return None
    intervals = _parse_intervals(row.iloc[0].get("intervals"))
    years = []
    for item in intervals:
        added = item.get("added")
        if added is None:
            continue
        try:
            years.append(pd.Timestamp(added).year)
        except Exception:
            pass
    if not years:
        return None
    return int(max(years))


def _dera_url(year: int) -> str:
    return f"https://www.sec.gov/files/dera/data/financial-statement-data-sets/{year}q4.zip"


def _cache_paths_for_year(year: int, base: Path):
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{year}_q4.zip", base / f"sub_{year}.txt"


def _download_and_extract_sub(year: int) -> pd.DataFrame | None:
    zip_path, sub_path = _cache_paths_for_year(year, Path(__file__).parent / "sec_cache" / "dera")
    if sub_path.exists():
        try:
            return pd.read_csv(sub_path, sep="\t", dtype=str, usecols=["cik", "instance", "sic"])
        except Exception:
            sub_path.unlink(missing_ok=True)

    if not zip_path.exists():
        url = _dera_url(year)
        print(f" Downloading SEC DERA {year} Q4 snapshot...")
        try:
            resp = requests.get(url, headers=SEC_HEADERS, timeout=120)
            resp.raise_for_status()
            zip_path.write_bytes(resp.content)
            print(f" Cached DERA zip: {zip_path}")
        except Exception as e:
            print(f" Failed to download {year} Q4 DERA: {e}")
            return None

    try:
        with zipfile.ZipFile(zip_path) as z:
            if "sub.txt" not in z.namelist():
                raise FileNotFoundError(f"sub.txt not found in {zip_path}")
            with z.open("sub.txt") as f:
                df = pd.read_csv(f, sep="\t", dtype=str, usecols=["cik", "instance", "sic"], on_bad_lines="skip")
                df.to_csv(sub_path, sep="\t", index=False)
                return df
    except Exception as e:
        print(f" Failed to extract {year} Q4 sub.txt: {e}")
        return None


def build_historical_sic_map(target_tickers: list[str], meta_df: pd.DataFrame, years: list[int] | None = None) -> dict:
    """Build ticker -> {cik, sic} from historical SEC DERA snapshots.
    Uses the latest active year from metadata intervals to order lookups.
    """
    if years is None:
        current_year = pd.Timestamp.now().year
        years = [current_year, current_year - 1, current_year - 2, current_year - 3]

    ticker_years = {}
    for ticker in target_tickers:
        latest = get_latest_active_year_from_intervals(ticker, meta_df)
        if latest is None:
            ticker_years[ticker] = years
        else:
            base = [latest, latest - 1, latest - 2]
            seen = set()
            ordered = []
            for y in base + years:
                if y not in seen and y > 1990:
                    seen.add(y)
                    ordered.append(y)
            ticker_years[ticker] = ordered

    needed_years = sorted({y for ys in ticker_years.values() for y in ys})
    year_to_df = {}

    for year in needed_years:
        year_to_df[year] = _download_and_extract_sub(year)

    # Normalize each loaded snapshot: derive ticker from the instance column, clean it.
    # The 'instance' field has the format "{ticker}-{period}.xml" (e.g. "logi-20100930.xml"),
    # so the leading token (split on '-') is the historical ticker used by the company.
    for year, df in year_to_df.items():
        if df is None:
            continue
        df = df.dropna(subset=["instance", "cik"])
        df["ticker"] = (
            df["instance"]
            .astype(str)
            .str.split("-")
            .str[0]
            .str.upper()
            .str.strip()
        )
        df["ticker_clean"] = df["ticker"].str.replace(r"[-._/]", "-", regex=True)
        df["cik"] = df["cik"].astype(str).str.zfill(10)
        year_to_df[year] = df

    cleaned_targets = {t.upper().strip().replace(".", "-"): t for t in target_tickers}
    historical_master = {}

    for ticker in target_tickers:
        cand_years = ticker_years[ticker]
        for year in cand_years:
            df = year_to_df.get(year)
            if df is None:
                continue
            base_ticker = ticker.upper().strip().replace(".", "-")
            match = df[(df["ticker"] == base_ticker) | (df["ticker_clean"] == base_ticker)]
            if match.empty:
                continue
            row = match.iloc[0]
            historical_master[ticker] = {
                "CIK": str(row["cik"]),
                "SIC": None if pd.isna(row["sic"]) else str(row["sic"]),
            }
            break

    return historical_master


def main():
    print("Loading /metadata/sp400 from db.h5...")
    meta_df = load_tickers_from_metadata()
    tickers = meta_df["ticker"].tolist()
    print(f" {len(tickers)} tickers loaded.")

    print("Fetching SEC ticker->CIK map...")
    ticker_to_cik = fetch_sec_ticker_map()
    print(f" {len(ticker_to_cik)} CIK mappings loaded.")

    print("Fetching SIC codes from SEC EDGAR...")
    sic_map = build_sic_map(tickers, ticker_to_cik, meta_df=meta_df)
    meta_df["sic"] = meta_df["ticker"].map(sic_map)
    meta_df["index_ref"] = meta_df["sic"].apply(get_index_ref)

    found = meta_df["sic"].notna().sum()
    print(f"\nDone. Found SIC codes for {found}/{len(tickers)} tickers.")

    print("Storing back to /metadata/sp400 with new columns...")
    # Append mode + drop the existing node first so we never wipe the rest of db.h5
    # (the old `mode="w"` truncated the entire file, deleting /sp400, /macros, /earnings).
    with pd.HDFStore(DB_FILE, mode="a") as store:
        if "/metadata/sp400" in store:
            store.remove("/metadata/sp400")
        store.put("/metadata/sp400", meta_df, format="table")
    print("Done.")

    print("\nSample:")
    print(meta_df[["ticker", "sic", "index_ref"]].dropna(subset=["sic"]).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
