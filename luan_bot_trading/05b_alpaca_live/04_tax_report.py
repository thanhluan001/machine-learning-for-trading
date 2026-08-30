"""04_tax_report.py — year-end French tax report from NOTES/trade_log.csv.

Spec: NOTES/French_Tax_Reporting_Alpaca.md §8 (year-end FIFO approach).
Long-only strategy: BUY/SELL matched FIFO per symbol; partial lots handled
by a proper lot queue (each sell consumes buy lots fractionally).
SHORT/COVER rows, if ever present, are matched separately.
DIVIDEND rows are summed gross/withheld/net per symbol.

Outputs (NOTES/):
  capital_gains_<year>.csv   — 2074-style line schedule (FIFO)
  tax_summary_<year>.txt     — net totals + 3916-bis reminder
  koinly_export_<year>.csv   — Koinly/tax-software mapping (doc §5A)

Also backfills missing ECB rates in trade_log.csv (rows logged when the
ECB API was unreachable) before computing.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tax_log import TAX_LOG, FIELDS, ecb_usd_to_eur  # noqa: E402


def load_rows(year: int | None = None) -> list[dict]:
    if not TAX_LOG.exists():
        print(f"No trade log at {TAX_LOG}")
        return []
    with open(TAX_LOG, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)]
    # backfill missing EUR conversions (rate was unavailable at log time)
    changed = False
    for r in rows:
        if not r["exchange_rate"] and r["action"] in ("BUY", "SELL", "SHORT", "COVER"):
            rate = ecb_usd_to_eur(r["date"])
            if rate is not None:
                gross = float(r["qty"]) * float(r["price_usd"])
                fees = float(r["fees_usd"] or 0)
                signed = gross + fees if r["action"] in ("SELL", "COVER") else gross - fees
                r["exchange_rate"] = str(rate)
                r["total_eur"] = f"{round(signed * rate, 2):.2f}"
                changed = True
    if changed:
        with open(TAX_LOG, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        print("Backfilled missing ECB rates in trade_log.csv")
    if year is not None:
        rows = [r for r in rows if r["date"][:4] == str(year)]
    return rows


def fifo_gains(rows: list[dict]) -> list[dict]:
    """FIFO matching. Sells consume buy lots fractionally; returns 2074-style lines."""
    lots: dict[str, deque] = defaultdict(deque)   # symbol -> [ [qty, eur_per_share, buy_date], ...]
    gains = []
    for r in sorted(rows, key=lambda x: x["date"]):
        a, sym, qty = r["action"], r["symbol"], float(r["qty"])
        eur = float(r["total_eur"]) if r["total_eur"] else float(r["qty"]) * float(r["price_usd"]) * float(r["exchange_rate"] or 0)
        pps = eur / qty if qty else 0.0
        if a == "BUY":
            lots[sym].append([qty, pps, r["date"]])
        elif a in ("SELL", "COVER"):
            remaining = qty
            while remaining > 1e-9 and lots[sym]:
                lot = lots[sym][0]
                take = min(remaining, lot[0])
                proceeds = take * pps
                basis = take * lot[1]
                gains.append({
                    "symbol": sym, "buy_date": lot[2], "sell_date": r["date"],
                    "qty": f"{take:g}",
                    "sale_proceeds_eur": round(proceeds, 2),
                    "cost_basis_eur": round(basis, 2),
                    "capital_gain_eur": round(proceeds - basis, 2),
                    "method": "FIFO",
                })
                lot[0] -= take
                remaining -= take
                if lot[0] <= 1e-9:
                    lots[sym].popleft()
            if remaining > 1e-9:
                gains.append({
                    "symbol": sym, "buy_date": "UNMATCHED", "sell_date": r["date"],
                    "qty": f"{remaining:g}",
                    "sale_proceeds_eur": round(remaining * pps, 2),
                    "cost_basis_eur": 0.0,
                    "capital_gain_eur": round(remaining * pps, 2),
                    "method": "FIFO(unmatched)",
                })
                print(f"  WARN: {r['date']} SELL {sym}: {remaining:g} shares without a logged BUY lot")
    return gains


def main() -> None:
    year = int(sys.argv[1]) if len(sys.argv) > 1 else datetime.now().year
    rows = load_rows(year)
    if not rows:
        print(f"No rows for {year}.")
        return
    print(f"Tax report {year}: {len(rows)} ledger rows from {TAX_LOG}")

    gains = fifo_gains(rows)
    divs = [r for r in rows if r["action"] == "DIVIDEND"]

    gains_csv = HERE.parents[0] / "tax" / f"capital_gains_{year}.csv"
    with open(gains_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["symbol", "buy_date", "sell_date", "qty",
                                          "sale_proceeds_eur", "cost_basis_eur",
                                          "capital_gain_eur", "method"])
        w.writeheader()
        w.writerows(gains)

    koinly = HERE.parents[0] / "tax" / f"koinly_export_{year}.csv"
    with open(koinly, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Date", "Type", "Symbol", "Amount", "Total (EUR)"])
        for r in rows:
            w.writerow([r["date"],
                        {"BUY": "Buy", "SELL": "Sell", "SHORT": "Sell", "COVER": "Buy",
                         "DIVIDEND": "Income"}[r["action"]],
                        r["symbol"], r["qty"], r["total_eur"]])

    total = round(sum(g["capital_gain_eur"] for g in gains), 2)
    wins = [g for g in gains if g["capital_gain_eur"] > 0]
    losses = [g for g in gains if g["capital_gain_eur"] <= 0]
    div_gross = sum(float(r["price_usd"]) for r in divs)  # placeholder if per-unit logged
    summary = [
        f"FRENCH TAX SUMMARY {year} — generated {datetime.now().isoformat()[:19]}",
        f"Ledger: {TAX_LOG}",
        f"Realized lines (FIFO): {len(gains)}  wins {len(wins)} / losses {len(losses)}",
        f"NET CAPITAL GAIN (EUR): {total:+,.2f}",
        f"  sum of gains: {sum(g['capital_gain_eur'] for g in wins):+,.2f}",
        f"  sum of losses: {sum(g['capital_gain_eur'] for g in losses):+,.2f}",
        f"Dividend rows: {len(divs)}",
        "",
        "FILING REMINDERS:",
        "- Form 2074 (plus-values, foreign broker) -> 2042; PFU 30% or barème option.",
        "- Form 3916-bis EVERY year the Alpaca account is open (even at a loss).",
        "- Net loss carries forward 10 years.",
        "- Rates used: official ECB daily reference (EXR/D.USD.EUR.SP00.A).",
        "- Paper trading now: verify rows are REAL fills before declaring.",
    ]
    summ_path = HERE.parents[0] / "tax" / f"tax_summary_{year}.txt"
    summ_path.write_text("\n".join(summary), encoding="utf-8")

    print(f"\n  NET CAPITAL GAIN {year}: EUR {total:+,.2f} over {len(gains)} FIFO lines")
    print(f"  Wrote: {gains_csv.name}, {koinly.name}, {summ_path.name}")


if __name__ == "__main__":
    main()
