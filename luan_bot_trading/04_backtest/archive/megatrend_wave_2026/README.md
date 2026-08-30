# megatrend_wave_2026/ — megatrend / RC-4 research wave (2026)

Phased build-out of the manual megatrend advisory program: trend death
diagnostics, theme clustering, broad-regime breadth, capex warnings,
insider/news normalization, false-reentry tests, and the partial-exposure
stress-gate chain (cycle 1 → cycle 2 → short gate → final stress gate).

**Outcome:** program settled as a MANUAL month-end advisory watcher —
no allocation automation (user decision, sustained under all tests:
partial exposure failed every stress gate; the value is in the panels,
not in mechanical exposure rules). Operational output lives in
`05c_megatrend_watcher/monthly_panel_report.py` (panels + advisory
ratios + §[13] RC-9 state detector, which reimplements
`rc_programs_2026/86` inline).

| Scripts | Phase |
|---|---|
| `71_megatrend_phase1_dead_trends.py` | trend-death diagnostics |
| `72,75,76` | clustering, pivot structure, capex blend |
| `73_megatrend_phase3_broad_regime.py` | broad regime/breadth |
| `77-79` | relative-capex warning + quality v1/v2 |
| `80_megatrend_normalize_insider_news.py` | insider/news normalization |
| `81_megatrend_false_reentry_test.py` | false-reentry stress |
| `82-85` | partial-exposure gate chain (all failed → advisory-only) |

Findings: `../findings/megatrend_*.md` (phase1/2/2b2c/3, relative_capex,
warning_quality v1/v2, false_reentry, partial_exposure cycle1/cycle2/
short_gate/final_stress_gate) + `../experiments/` outputs.
