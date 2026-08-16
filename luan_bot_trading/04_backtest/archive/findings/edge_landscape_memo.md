# Edge Landscape Memo — The 2026-08 Slow-Week Search and the Analysis-Depth Doctrine

**Date:** 2026-08-15 (compiled); investigations 2026-08-13 → 2026-08-15
**Companion to:** Design.md §18 (Research Backlog); findings files in this directory
**Status:** Search closed. Doctrine adopted. Nothing here is live; nothing touches
the frozen V6 paper-executable model.

---

## 1. Origin and question

The PEAD book has seasonal dead zones (mid-August–September, and the Mar/Jun/Dec
shoulders) where earnings supply collapses. The original plan was a portfolio of
3–4 independent edges for defense against single-edge corrosion. This memo
documents the systematic search for a second edge, the results, and the strategic
doctrine that emerged.

**Method discipline (same bar for every candidate):** execution-honest event
studies — entry at the first close after information becomes public (never the
transaction date), CAR vs IJH benchmark, tail-vs-dispersion test (a fat tail with
zero mean is vol, not alpha), ex-ante conditioning (can anything observable
predict the tail?), and regime stability across years.

## 2. Candidates tested and their verdicts

| # | Candidate | Data | Result (headline) | Verdict |
|---|---|---|---|---|
| 1 | Analyst upgrade drift | FMP grades, 31,318 events (script 68) | Day-0 move +3.27% (untradeable); post-announcement drift 5d/10d ≈ +0.09% (coin flip). Edge is entirely in the announcement instant. | **CLOSED** |
| 2 | S&P 400 index additions | Wikipedia intervals, 501 events (script 45) | Post-2020 abnormal return −0.00% vs IJH (t=−0.01). Front-run and crowded out. | **CLOSED** |
| 3 | Insider cluster buying | FMP Form 4, 71,663 records / 14,222 P-purchases (script 69) | 5d/10d CAR +0.11% (49% win) even for ≥2-insider clusters ≥$50k; only 60d shows drift (+4.2%), which conflicts with 5-day slot architecture. September 20d CAR −0.28%. | **CLOSED (as edge; see RC-1 as feature)** |
| 4 | Pre-ex-dividend run-up | Tiingo divCash, 20,015 events (script 70) | Mean ≈ 0 in every window/year. P(>1%)=35% is pure dispersion (zero-mean 3σ 5-day returns produce it mechanically). Yield is ANTI-predictive: biggest dividends drift −0.29% (capture desks shorting in). | **CLOSED** |
| 5 | Dividend run-up + classic filters (yield>2.5% ann., stock&IJH>SMA50, ADV≥$50M/$100M) | same | Each filter made it equal or worse (−0.05%→−0.18%→−0.10%→−0.22%). Filters concentrate into the most-arbitraged corner. | **CLOSED** |
| 6 | Senate (congressional) trades | FMP senate-latest, 10,100 records; 466 sp400 purchases, disclosure-date entry | **Real drift**: 20d CAR +2.13% (53.0% win), $50k+ subset +2.71%. But supply ~4.5 events/mo universe-wide, ~1.7/mo in September; 28-day median disclosure staleness; drift lives at 10–20d vs our 5d architecture. | **PARKED (real anomaly, wrong business model)** |
| 7 | S&P 500 PEAD expansion | FMP+Tiingo, 22,216 events; frozen V6 transfer (scripts 66/67) | Transfer WORKS: +663% NAV 3y (out-of-universe, robustness proof). But home universe wins: 61.4% vs 55.5% win, +821% vs +663% NAV, better DD. 2026-H1 holdout: sp500 +135% vs sp400 +78% (small n, survivorship-flattered — watch, don't act). | **PARKED, low priority** |
| 8 | FMP fundraisers (Form D/C private placements) | FMP endpoints | Private-market data (CodeRabbit, LLCs) — untradeable directly. Public sp400 overlap: 4/40 sampled CIKs, all stale one-offs 2011–2016. | **CLOSED** |
| 9 | FMP menu remainder: COT, DCF, TipRanks add-on | reasoning, not tested | COT = regime feature, not event edge; macros frozen + low importance. DCF = vendor value factor, not an edge. TipRanks = repackaged dead signals (ratings/insiders/13F lag). | **NOT PURSUED** |
| 10 | Polymarket whale-lifecycle tracking (design-stage only, no data pulled) | on-chain Polygon/Polymarket (public) | Mechanism sound: identify emerging skilled wallets (CLV-based validation), copy while their follower crowd is below a measured threshold, retire at the threshold. Competes on analysis depth, not speed. Killed at Phase 0 legal gate: France (ANJ) ISP-blocks the venue; operator risk; gray zone not worth it. 13D small-activist coattailing noted as the legal-market analog, parked. | **DESIGN NOTES KEPT, PARKED (venue-gated)** |

## 3. The empirical pattern (the "why" behind every verdict)

Every closed candidate shares one fingerprint — and it is worth stating as a
law for this universe:

> **Edge survives only where information requires slow interpretation. It dies
> wherever the event is scheduled or instantly public.**

Layer-by-layer attribution of the kills (institutional advantage map):

```text
edge that died              which layer ate it
analyst upgrades        L2 — channel checks front-ran the publication; note was last domino
insider Form 4s         L1/L3 — 2-day filing is scraped instantly; priced at disclosure
dividend run-up         L3/tax — capture desks with near-zero financing costs
index additions         L1 — announced days ahead, arb'd to zero pre-effective
senate trades           survived partially — 28-day-old info, sub-institutional scale
PEAD (mid-caps)         survives — between the layers: too slow for HFT, too small
                        for capacity-constrained funds, too under-covered for
                        alt-data pre-pricing, diffuses over days
```

The S&P 400 is the habitat where the layers leave cracks: mid-caps sit below
institutional minimum position sizes (capacity), below alt-data coverage
breadth (attention), and above retail-effort floors. PEAD is the anomaly that
lives in those cracks. Script 66 additionally measured PEAD frequency *rising*
in the S&P 400 (12.3% in 2023–25, best on record) while eroding in the S&P 500.

## 4. The Analysis-Depth Doctrine (adopted)

The search closed with a strategic conclusion that now governs all strategy
selection:

> **We compete on depth of analysis, not on speed.**
>
> Speed tiers (HFT, sub-second scraping) and information-purchase tiers
> (alt-data feeds, expert networks, management access) are structurally closed
> to us — and structurally irrelevant at multi-day horizons. What remains open
> is the horizon the fast tiers ignore and the slow crowd under-analyzes:
> patient measurement, honest validation, and richer state than the
> competition holds.

Operational corollaries:

1. **Our edges are horizon-natives.** PEAD at 5 days is native to our horizon;
   a whale-copy at 2 seconds is native to someone else's. A candidate must
   profit at the horizon we can actually operate (daily decisions, manual
   execution) — this is why insider 60d drift and senate 10–20d drift were
   parked despite being real.
2. **Depth beats latency when the state is richer.** The Polymarket lifecycle
   design (per-wallet crowd curves, CLV validation, threshold bands) is an
   example: hold more state than the copy-crowd dashboards, update slowly,
   decide rarely. The same principle animates the V6 gates (three independent
   classifiers, min() over them) versus any single fast signal.
3. **Crowd size is a state variable, not a race.** Where a crowd exists
   (copy-trading, whale watching, dividend capture), its size/acceleration is
   measurable ex ante and backtestable — usable as a feature, a filter, or an
   exit trigger. Enter where the crowd is thin, exit on its acceleration.
4. **Filters cannot create an edge — only concentrate one that exists.**
   The dividend study's lesson: conditioning on "quality" (yield, trend,
   liquidity) in a zero-edge event just selects the most-arbitraged corner.
   Candidate selection is mechanism-first, always.
5. **Composite sparse events are pre-rejected.** Sparse ∩ sparse ≈ nothing
   (senate 0.23%/stock-wk × insider clusters 1.2% → ~1 event/yr universe-wide),
   and same-mechanism signals substitute rather than compound. Beware the
   multiple-testing trap: 7 dead candidates imply 21 pairs, each a ~5%
   false-positive machine.
6. **Dead as an edge ≠ dead as a feature — with a bar.** Analyst grades:
   standalone dead, weak-but-nonzero as conditioning features. Insiders:
   standalone dead at 5d, mechanism-backed as future PEAD features (RC-1,
   Piotroski-Roulstone). Features need conditional information value, proven
   by walk-forward + bootstrap against the frozen baseline — never assumed.

## 5. What survives, where it lives

```text
LIVE            PEAD on S&P 400 (V6 gates, theta 0.33, force-refresh mh=4) —
                frozen, paper-executing. The one edge this universe sells us.
PARKED          S&P 500 transfer (validated, home wins; revisit if holdout repeats)
PARKED          Senate drift (real signal; supply + staleness + horizon)
PARKED          Polymarket lifecycle design (venue-gated; revive on legal change
                or an EU-licensed public-ledger equivalent; 13D coattailing is
                the legal-market analog)
PARKED          Megatrend watcher (RC-4) — design notes only; long-horizon trend
                overlay for the core 90% book, NOT the PEAD sleeve; premise
                verified (NVDA/TSM Tiingo 2026-08); kill tests defined
                (survivorship across dead megatrends, regime table, trivial
                200d-MA bar)
BACKLOG (RC-1)  Insider-accumulation PEAD features — mechanism-backed, data
                cached (db_insider.h5), gated behind freeze + research cycle
CLOSED          upgrades, index adds, insider-as-edge, dividend run-up (all
                variants), fundraisers, COT/DCF/TipRanks
```

The diversification goal was achieved in an unexpected form: not a portfolio
of parallel edges, but (a) one deeply-validated edge with kill-criteria
monitoring, (b) a validated second universe on the bench, (c) a mechanism-
backed feature path (RC-1), and (d) a design-pattern bank of parked strategies
with explicit revive triggers. Idle-slot cash through dead zones is the
optimum, not a placeholder — the calendar itself is the position.

## 6. Files and artifacts

- Scripts: `45`, `66`, `67`, `68`, `69`, `70` in `04_backtest/archive/edge_search_2026/` (all cached,
  re-runnable, production db.h5 untouched)
- Data caches: `db_sp500.h5`, `db_insider.h5`, `db_div.h5` in `01_data/`
- Findings: `analyst_upgrade_drift_findings.md`, `insider_buying_drift_findings.md`,
  `preex_dividend_runup_findings.md`, `senate_trading_findings.md`,
  `sp500_vs_sp400_pead_findings.md`, `sp500_model_transfer_findings.md` (this dir)
- Design.md §18: research backlog with RC-1/R C-2 and the composite pre-rejection
- Diagnostics: `65_shap_pick.py` (per-pick gate attribution, native TreeSHAP)
