# data_contracts.md — freshness and point-in-time rules

## Point-in-time
- Universe membership: interval history (when a name ENTERED the index),
  never "today's list". The matrix builder marks each event with the
  membership state at the event date.
- Features: 24 inputs, all computable before the print (see
  docs/features.md for the knowability audit of each).
- Labels: T+5 return with a -10% stop path, computed from prices only,
  attributed to the event's fiscal period.

## Freshness (per nightly run)
- Prices: every scored + held name needs a bar from the LAST COMPLETED
  session; lagged bars hard-fail the name (no silent stale scoring).
- Benchmarks/macros: as-of joins only; a missing macro row degrades the
  feature to NaN (xgboost handles natively) but the run logs it.
- Grades: incremental refresh for scored names; cutoff at report_date.
- Earnings actuals: lazy back-fill for scored + held; +/-4 day date
  shift tolerance with adoption of FMP's corrected date.

## Storage
- Single HDF5 store `01_data/db.h5` with a stable key layout (see
  database_layout in the parent project; keys are documented in each
  gathering script's header). The store is append/refresh in place;
  never rebuild from scratch without re-running the freshness audit.
