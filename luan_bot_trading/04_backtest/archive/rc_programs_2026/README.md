# rc_programs_2026/ — the RC research-program wave (Aug-Sep 2026)

Systematic edge-candidate testing for slow-week filling and model
augmentation. Every program closed with a named gate and cause of
death; one promotion (RC-9 → advisory panel). Full ledger:
Design.md §18.

| Script | Program | Outcome |
|---|---|---|
| `86_rc9_undecided_state_detector.py` | RC-9 regime-state detector | **PROMOTED** — logic reimplemented inline in `05c_megatrend_watcher/monthly_panel_report.py` §[13] (this script is provenance, not a runtime dep) |
| `87_rc9_state_portfolio_sim.py` | RC-9 portfolio simulation | regime says WHEN not WHO; leader-chasing destroys value |
| `88_t4_vs_t5_exit_test.py` | exit-horizon A/B (user paper observation) | REJECT — T+5 stays frozen; reuses script-63 machinery |
| `89-92_rc1_phase*` | RC-1 insider pre-event features | REJECT at promotion bar (Phase 0 pass, Phase 2 inverted, Phase 3 fail ×3) |
| `93_rc11_phase0_audit.py` | RC-11 ex-div neglect pricing | CLOSED at kill gate 0 — attention gradient real, saturates at zero |

Dependencies: 88 and 92 import `../63_force_refresh_backtest.py` and
`../51_hp_theta_sweep_23feat.py` **at their archived paths' parents** —
if re-running, run from `04_backtest/` with
`importlib.util.spec_from_file_location` pointing at the top-level
scripts, or copy back into `04_backtest/` first.

Findings: `../findings/rc9_undecided_state_findings.md`,
`../findings/hold_path_day_slices_findings.md`,
`../findings/rc1_insider_pre_event_findings.md`,
`../findings/rc11_exdiv_neglect_findings.md`.
