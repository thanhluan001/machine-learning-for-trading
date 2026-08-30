"""tax_log.py — French tax trade ledger for Alpaca fills.

Spec: NOTES/French_Tax_Reporting_Alpaca.md
  - every fill logged immediately as a CSV row (doc §2A schema)
  - EUR conversion at the official ECB daily reference rate (doc §6C)
    exchange_rate column = USD->EUR multiplier (= 1 / ECB USD-per-EUR quote)
  - FIFO gains computed at year-end by 04_tax_report.py (doc §8)
  - SHORT/DIVIDEND rows supported by schema (strategy is long-only today)

Failure policy: if the ECB rate cannot be fetched, the row is still
logged with rate/total_eur EMPTY and a warning is printed — the fill
record (the legally required part) is never lost, and 04_tax_report.py
backfills missing rates later from the same ECB series.
"""
from __future__ import annotations

import csv
import json
import os
import time
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

TAX_LOG = Path(__file__).resolve().parents[1] / "tax" / "trade_log.csv"
RATE_CACHE = Path(os.environ.get("TEMP", "/tmp")) / "_ecb_usd_eur.json"
ECB_URL = ("https://data-api.ecb.europa.eu/service/data/EXR/D.USD.EUR.SP00.A"
           "?startPeriod={d}&endPeriod={d}&format=csvdata")
FIELDS = ["date", "symbol", "action", "qty", "price_usd", "fees_usd",
          "exchange_rate", "total_eur"]


# ---------------------------------------------------------------- rates

def _load_cache() -> dict:
    try:
        return json.loads(RATE_CACHE.read_text())
    except Exception:
        return {}


def _save_cache(c: dict) -> None:
    try:
        RATE_CACHE.write_text(json.dumps(c))
    except Exception:
        pass


def ecb_usd_to_eur(day: str, retries: int = 2) -> float | None:
    """USD->EUR multiplier for `day` (YYYY-MM-DD), ECB official fixing.

    Walks back up to 5 calendar days for weekends/ECB holidays.
    Returns None if unavailable (caller logs the row without conversion).
    """
    cache = _load_cache()
    if day in cache and cache[day]:          # '' cached = known-missing
        return cache[day] or None
    d0 = datetime.strptime(day, "%Y-%m-%d").date()
    for back in range(6):                    # day itself .. 5 days back
        d = (d0 - timedelta(days=back)).isoformat()
        if d in cache:
            if cache[d]:
                return cache[d]
            continue                         # known-missing marker
        url = ECB_URL.format(d=d)
        for attempt in range(retries + 1):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 tax-reporting research"})
                with urllib.request.urlopen(req, timeout=15) as r:
                    text = r.read().decode("utf-8")
                lines = [ln for ln in text.splitlines() if ln.strip()]
                if len(lines) >= 2 and "OBS_VALUE" in lines[0]:
                    hdr = lines[0].split(",")
                    vi, ti = hdr.index("OBS_VALUE"), hdr.index("TIME_PERIOD")
                    last = lines[-1].split(",")
                    if last[ti] == d:  # only accept a fixing FOR this day
                        usd_per_eur = float(last[vi])
                        mult = round(1.0 / usd_per_eur, 6)
                        cache[d] = mult
                        _save_cache(cache)
                        return mult
                break                        # valid response, no fixing this day
            except Exception:
                if attempt < retries:
                    time.sleep(1.5 * (attempt + 1))
        cache[d] = ""                        # mark missing (holiday/weekend)
        _save_cache(cache)
    return None


# ---------------------------------------------------------------- ledger

def _existing_keys() -> set:
    keys = set()
    if TAX_LOG.exists():
        with open(TAX_LOG, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                keys.add((row["date"], row["symbol"], row["action"],
                          row["qty"], row["price_usd"]))
    return keys


def log_fill(day: str, symbol: str, action: str, qty: float, price_usd: float,
             fees_usd: float = 0.0) -> bool:
    """Append one fill to NOTES/trade_log.csv (idempotent per doc §2A schema).

    action: BUY | SELL | SHORT | COVER | DIVIDEND
    Returns True if a new row was written.
    """
    day = str(day)[:10]
    qty_s = f"{qty:g}"
    price_s = f"{price_usd:.4f}".rstrip("0").rstrip(".")
    key = (day, symbol, action, qty_s, price_s)
    if key in _existing_keys():
        return False

    rate = ecb_usd_to_eur(day)
    total_eur = ""
    if rate is not None:
        gross = qty * price_usd
        signed = gross + fees_usd if action in ("SELL", "COVER") else gross - fees_usd
        total_eur = f"{round(signed * rate, 2):.2f}"
    else:
        print(f"    [TAX] WARN: ECB rate unavailable for {day}; "
              f"row logged without EUR conversion (backfill later)")

    new_file = not TAX_LOG.exists()
    TAX_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(TAX_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(FIELDS)
        w.writerow([day, symbol, action, qty_s, price_s, f"{fees_usd:.2f}",
                    rate if rate is not None else "", total_eur])
    tag = "TAX" if rate is not None else "TAX(no-fx)"
    print(f"    [{tag}] {action} {qty_s} {symbol} @ ${price_s} on {day}"
          + (f" -> EUR {total_eur}" if total_eur else ""))
    return True
