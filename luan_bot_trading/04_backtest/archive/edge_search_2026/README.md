# edge_search_2026/ — The 2026-08 slow-week edge search (CLOSED)

One-time event studies hunting for a second edge to fill earnings dead zones
(Sep/Dec/Mar/Jun). All conclusions are recorded and authoritative in:

- `../findings/edge_landscape_memo.md` — the master record (all 10 candidates,
  verdicts, institutional-advantage map, the analysis-depth doctrine)
- `../../Design.md` §18 — research backlog (RC-1..RC-4)

Scripts are preserved re-runnable (all read cached data; production db.h5 is
never written). Data caches live in `01_data/`: `db_sp500.h5`, `db_insider.h5`,
`db_div.h5`.

| Script | Candidate | Verdict | Findings doc |
|---|---|---|---|
| `45_index_rebalance_probe.py` | S&P 400 index additions (earlier probe, reused) | CLOSED (−0.00% vs IJH post-2020) | memo §2 #2 |
| `66_sp500_pead_comparison.py` | S&P 500 PEAD frequency study | PARKED (sp400 rate 10.7% vs 8.5%, gap widening) | `../findings/sp500_vs_sp400_pead_findings.md` |
| `67_sp500_model_transfer.py` | Frozen V6 applied to S&P 500, 3y | PARKED (transfer works +663%; home wins 61.4% vs 55.5% win) | `../findings/sp500_model_transfer_findings.md` |
| `68_analyst_upgrade_drift.py` | Analyst upgrade drift | CLOSED (Day-0 +3.27%, then zero) | `../findings/analyst_upgrade_drift_findings.md` |
| `69_insider_cluster_drift.py` | SEC Form 4 insider cluster buying | CLOSED as edge (5d ≈ 0); RC-1 as future PEAD feature | `../findings/insider_buying_drift_findings.md` |
| `70_preex_dividend_runup.py` | Pre-ex-dividend run-up (incl. yield/SMA/ADV filter stack) | CLOSED (zero mean; tail = dispersion; yield anti-predictive) | `../findings/preex_dividend_runup_findings.md` |

Note: 66/67 path logic patched for this archive location (ROOT =
HERE.parents[2]; 67 imports 51/63 from `../../` = the live policy chain).
Senate-trades and fundraisers probes (temp scripts only, no persisted code)
are summarized in `../findings/senate_trading_findings.md`.

Related earlier-phase probes moved to `../phase_g_v3_v4_era/`:
`46_analyst_revision_probe.py` (revision-as-feature probe) — see
`../findings/analyst_revision_findings.md`.
