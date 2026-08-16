# Senate-Trading Drift Probe — PARKED (real signal, insufficient supply)

**Endpoint:** FMP `/stable/senate-latest` (probe only; no study script persisted)
**Date:** 2026-08-15
**Question:** Can senator trades (disclosed via STOCK Act PTRs) fill slow-week slots?

## Probe findings

- Endpoint viable on our plan: 10,100 records, 2017-10 → present, includes BOTH
  `disclosureDate` and `transactionDate` (honest backtesting possible).
- sp400 overlap: 520 purchase events over ~8.8 years (Stock assetType only).
- Amount buckets: 77% in smallest range ($1k–$15k); ~6/yr at $50k+.
- Disclosure lag: median 28 days (STOCK Act window).
- Monthly supply in slow months: Sep ≈ 1.7/month, Jun ≈ 1.7/month, Mar ≈ 2.6/month.

## Event study (from disclosure close, vs IJH, n=466, 2018–2026)

```text
 5d CAR: +0.61% mean, -0.06% median, 47.6% win   (flat at our horizon)
10d CAR: +0.73% mean, +0.48% median, 53.4% win
20d CAR: +2.13% mean, +1.26% median, 53.0% win   (genuinely positive)
$50k+ subset (n=54): 20d +2.71%, 53.7% win
```

**Not dead-flat** — first candidate of the six with measurable drift (consistent
with the political-information underreaction literature). Caveats: mean/median
gap = outlier dependence; survivorship (current members backward); senator
trades cluster within single disclosures.

## Verdict

**PARKED — real anomaly, wrong business model.**
- Supply: ~4.5 events/month universe-wide; ~1.7/month in September — cannot
  fill 4 slots.
- Signal is 28 days stale at median and already scraped by retail crowds
  (Unusual Whales et al.); measured drift is what survives AFTER them.
- Horizon mismatch: drift lives at 10–20d; architecture is 5-day.
- 77% smallest-size bucket — not conviction capital.

Revisit trigger: if the model ever reopens for research and a 10–20d horizon
variant is considered, or if disclosure rules shorten (real-time filing bills).

## Also probed

- `/fundraisers`, `/fundraisers-latest`: 404 — wrong path. Correct paths probed
  2026-08-15: `/fundraising-latest`, `/fundraising?cik=`, `/fundraising-search`,
  `/crowdfunding-offerings-latest` — all live on our plan.
- **FUNDRAISERS = SEC Form D (Reg D) + Form C (Reg CF): private-market fundraising.**
  Sample issuers: CodeRabbit Inc., Classic Coffee and Tea LLC — private companies,
  no tickers/listings/prices. Events untradeable directly.
- Only testable angle (public sp400 companies filing Form D): 4 of 40 sampled CIKs
  have ANY history, all stale one-offs 2011–2016 (CNC/ALB/HE/CBT). Modern sp400
  supply ≈ 0; and real public-company raises disclose via 8-K anyway.
- Pre-IPO pipeline angle: Form D can't be acted on until listing; IPO pops not
  capturable at offer; aftermarket chasing has negative long-run expectancy.
- **Verdict: fundraisers CLOSED — wrong asset class (private market), zero
  modern sp400 event supply.**
- `/mergers-acquisitions-latest`: exists (100-record feed) — noted; merger arb
  is a different strategy class, not a slow-week slot-filler.
