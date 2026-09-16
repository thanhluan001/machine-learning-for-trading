# RC-18 amendment prep — peer read-across spec arithmetic (2026-09-16)

Read-only diagnostic (`_rc18_peer_spec_arithmetic.py`). NO feature-vs-outcome
relationship measured (that is P1, gated behind the P0 firewall).

## 1. Coverage: share of SP400 events with >=1 same-group peer reporting in
##    the trailing K sessions (v6c, 16,665 events; PIT permaTicker->SIC map, 99.8%)

| grain      | groups | med tickers/grp | K=10  | K=15  | K=20  | completed-11 @K=15 | @K=20 |
|------------|-------:|----------------:|------:|------:|------:|-------------------:|------:|
| SIC-4      |    257 |               2 | 43.9% | 46.1% | 48.0% |               5.6% |  9.7% |
| SIC-3      |    181 |               2 | 53.6% | 56.4% | 59.0% |               8.3% | 14.4% |
| SIC-2      |     58 |               7 | 76.5% | 80.1% | 82.6% |              18.1% | 30.4% |
| sector ETF |     11 |             ~35 | 96.5% |    -- |    -- |                 -- |    -- |

GOTCHA (found during the check): joining the *current* metadata table on
canonical_ticker silently produced a blank-string bucket with 386 tickers
(40% of history: delisted/renamed names) and inflated coverage to 67.8%.
Correct join = permaTicker -> /metadata/sp400_permatickers.sic (string dtype:
coerce with to_numeric).

## 2. Maturity arithmetic (the binding constraint)

A peer's COMPLETED 10-session drift needs gap >= 13 sessions (AMC entry) /
14 (BMO) between peer report and my cutoff. Therefore:

- K=10 (the proposed 1-2 week window): completed-drift coverage = 0.0% at
  EVERY grain. The 1-2 week window can never contain a completed peer drift.
- Fast flavor (peer day-0 reaction): 43.9% (SIC-4) .. 96.5% (sector) at K=10.
- 3-session partial: available from K>=5.

## 3. GWRE case mechanics (the motivating anecdote, checked)

- GWRE 2026-06-04 AMC: SUE 0.39, car_10d -39.7%, pregap_return -15.9%
  (the live stop-loss case). Relative price: -11.8% into the print
  (Jun 1->Jun 4), a further -8.8pp on T+1.
- Peer OKTA (same GICS sub-industry "Application Software", SIC 7372)
  reported 2026-05-28, five sessions earlier: SUE 1.88, car_10d +18.4%,
  pead_pass=1. At GWRE's cutoff only ~4 sessions of OKTA's drift existed
  (partial, strongly positive).
- VERDICT: the naive peer read-across signal was POSITIVE while GWRE fell
  -40%. The motivating anecdote is a COUNTEREXAMPLE at taxonomy grain —
  evidence for finer/learned similarity (OKTA identity security vs GWRE
  insurance software are not comparable businesses) rather than GICS/SIC.

## 4. Implications for the amendment

1. Fast flavor is the practical primary (gap / 1-3 session partial).
2. Slow flavor (completed drift) requires a 3-4 week lookback and is sparse
   at fine grains (3-10% SIC-4; 18-30% SIC-2/sector).
3. Grain trade-off is real: coarse = coverage, poor similarity; fine =
   similarity, ~44-54% coverage (must be declared; consider a no-peer flag).
4. Trailing-correlation / embedding similarity (user's original instinct)
   has near-100% coverage by construction and can separate OKTA from GWRE —
   test it as a variant, not a replacement.
