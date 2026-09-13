# 05_live — paper trading

01_fetch_and_predict.py — nightly, >=1h after close. Discovers upcoming
SP400 earnings, refreshes scored+held data, computes features, scores
with frozen V6 gates, writes plan.json + candidates.json. The optional
V4 comparison model (not included) is skipped gracefully if absent.

02_paper_trade.py — intraday, market open (~3:45pm ET in the parent
project). Stops, T+5 exits, force-refresh, entries due today. Alpaca
PAPER only. Broker fills are reconciliation truth.

03_portfolio_report.py — book summary. 04_tax_report.py — French tax
ledger (see ../tax/README.md); tax_log.py + alpaca_client.py are the
support libraries.

Never point this at a live account. The scripts hardcode paper
endpoints by design.
