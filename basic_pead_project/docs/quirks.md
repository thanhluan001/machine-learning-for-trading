# quirks.md — data-provider traps this pipeline learned the hard way

Every item below caused a real bug or a real false signal. The fixes are
already embedded in the scripts; this file exists so you recognize the
symptoms if they recur.

## Tiingo
- Delisted/renamed tickers: the plain daily-meta endpoint omits
  `permaTicker`. Resolve identity via
  `GET /tiingo/utilities/search/{ticker}?includeDelisted=true` and use
  the returned permaTicker as the storage key. Tickers get REUSED
  (SUNW, EMC-era names) — always join on permaTicker, never ticker.
- EOD consolidation lag: same-day bars may not be available until
  ~1-3 h after close. The nightly script runs >=1 h after close and
  hard-fails scoring on lagged bars (freshness contract).

## FMP (earnings + grades)
- The calendar seeds future events with estimated report dates and
  REVISES them (observed shifts of 1-4 days, e.g. 09-23 -> 09-22).
  Any matcher on report_date must tolerate +/-4 days and adopt FMP's
  corrected date. Exact-date matching leaves phantom NaN quarters.
- `/stable/earnings-calendar` covers from YESTERDAY — request the
  window starting today-1 or you silently drop same-day events.
- Actuals appear on `/stable/earnings?symbol=X` hours-to-days after
  the print, keyed to the fiscal period. The back-fill (step [4.7] in
  the nightly script) is lazy: scored set + held book only. Prints of
  names that dropped out of candidacy stay unfilled until a sweep —
  run a periodic null-actual sweep for hygiene.
- Grades rows carry `report_date`-ish action timestamps that can be
  T-1 relative to the print. Cut features at REPORT date, not fetch
  date, or you leak print-eve analyst actions (this cost a live
  mis-score once).
- `apikey` goes in the QUERY STRING, not a header.

## Alpaca (paper)
- `get_orders` needs the `GetOrdersRequest` enum values; the orders
  API returns newest-first — paginate.
- DAY market orders placed on a market holiday queue for the next
  open. The trade script has NO market-calendar gate by design;
  interpret queued orders accordingly.
- Reconciliation truth is the FILL (price + fill date), never the
  order or the run date.

## Python / Windows
- Never import the base backtest module (51_*.py) twice in one
  process under different names — its module-level code crashes
  stdout handling. The gauntlet scripts each load it once.
- Long data builds: use one bulk HDFStore pass, not per-ticker opens.
- If you shell out to conda-run scripts, capture output to a file;
  the parent process may not share the child's encoding.
