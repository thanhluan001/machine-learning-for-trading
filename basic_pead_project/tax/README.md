# Tax Reporting (French) — Alpaca trades

Spec / user notes: `NOTES/French_Tax_Reporting_Alpaca.md` (user-maintained).

## Files here

| File | What |
|---|---|
| `trade_log.csv` | **Live ledger** — one row per confirmed fill (BUY/SELL, doc §2A schema, ECB rate at trade date). Appended automatically by `05b_alpaca_live/02_paper_trade.py` via `tax_log.log_fill()`. This is the legally required record — keep 10+ years. |
| `capital_gains_<year>.csv` | Generated: 2074-style FIFO line schedule (doc §8) |
| `koinly_export_<year>.csv` | Generated: Koinly/tax-software mapping (doc §5A) |
| `tax_summary_<year>.txt` | Generated: net totals + filing reminders |

## Pipeline

```text
Script 02 (3:45pm run)  ->  log_fill() appends rows to trade_log.csv
                                 (ECB daily fixing EXR/D.USD.EUR.SP00.A,
                                  weekend walk-back; on API failure the row
                                  is logged without FX and backfilled later)
December (year-end)     ->  python luan_bot_trading/05b_alpaca_live/04_tax_report.py <year>
                            (FIFO matching incl. partial lots, dividend annex,
                             writes the three generated files above)
April/May (filing)      ->  2074 + 2042 from capital_gains_<year>.csv
                            3916-bis EVERY year the Alpaca account is open
```

## Conventions

- `exchange_rate` = USD→EUR multiplier = 1 / ECB "USD per EUR" quote.
- Fees default to 0.00 (Alpaca paper); pull real fees from Alpaca
  activities API before declaring on a funded account.
- DIVIDEND rows: schema supported (`action=DIVIDEND`, gross in
  `price_usd`, withheld in `fees_usd`); strategy is long-only 5-9 day
  holds, dividends not expected but report handles them.
- Ledger rows are never rewritten (append-only), except the documented
  FX backfill of empty `exchange_rate`/`total_eur` cells.

⚠️ Currently **paper trading**: verify rows are real fills before
declaring. Rates/rules change — confirm with a fiscaliste.
