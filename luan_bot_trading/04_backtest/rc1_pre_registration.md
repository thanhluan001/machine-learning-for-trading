# RC-1: Insider Pre-Event Features for PEAD — Pre-Registration

**Status:** APPROVED research cycle (user, 2026-08-30). Runs in the
September dead zone. Target resolution: before October earnings season.
**Hypothesis (frozen before any results):** Piotroski & Roulstone (2005)
— insider trades predict future earnings surprises. PEAD after a beat
should be LARGER when the print was preceded by net insider buying, and
weaker/hazardous under net insider selling.

**Relationship to prior work:** Script 69 closed insider buying as a
*standalone* signal (zero 5-10d edge standalone). RC-1 tests insider
activity as *features on the 16.8k earnings-event matrix* — the machine
does the selection. Precedent cuts both ways: grades failed standalone
and failed as features (V5); the mechanism here is stronger (informed
capital front-running fundamentals vs repackaged opinions).

---

## Phase 0 — Point-in-time audit & coverage (`89_rc1_phase0_audit.py`)

~2h. Decides everything downstream.

1. **Leak rule (frozen):** features use `filingDate` (EDGAR public
   dissemination), window `[T−90d, T)` strictly before `report_date`.
   `transactionDate` is NOT public information — never used for joins.
2. For each of the 16,789 matrix events count, in `[T−90d,T)` and
   `[T−30d,T)`: total P-Purchase dollars (price>0, open-market),
   distinct buyers, sell dollars, days since last buy.
3. **Blackout control:** same counts for `[T, T+30d)` (post-print).
   Insider activity concentrates AFTER prints (script 69: March/May
   peaks). The pre-print window is legally sparse; the *rare pre-print
   buy* is the informed one. Quantify the asymmetry.
4. **5000-row FMP caps** (~10 tickers: US…35360, 74181, 96453, 108620,
   116622): histories truncated → features = NaN for those tickers'
   early windows (never zero-filled). Optionally re-paginate FMP for
   the affected tickers if Phase 2 shows signal.
5. **KILL GATE 0 (pre-registered):** if <10% of events have any
   pre-event P-purchase ≥$10k, the features are mostly-zero → close
   RC-1 at Phase 0 with the coverage number as cause of death.
   (§18 estimate: 15-25%.)

## Phase 1 — Feature build (`90_rc1_phase1_features.py`)

Side table keyed `(permaTicker, report_date)` → join at train time.
The frozen 4-feature set (NO additions after seeing results — the V5
multiplicity lesson):

| feature | definition |
|---|---|
| `insider_net_buy_90d` | log1p(net P-purchase $ in [T−90d, T)) |
| `insider_cluster_90d` | ≥2 distinct reportingNames, ≥$50k combined |
| `insider_sell_pressure_30d` | log1p(net S-sale $ in [T−30d, T)) |
| `insider_days_since_last_buy` | days; NaN if no buy in 180d |

Missing → NaN (native XGBoost handling). No role-weighting, no
60d/180d extra horizons, no per-officer-type splits.

## Phase 2 — Univariate screen (`91_rc1_phase2_univariate.py`)

Rank-IC of each feature vs the three V6 gate labels + CAR5, per fold
(4 folds) + pooled. Quartile CAR5 tables.
**KILL GATE 1 (pre-registered):** a feature survives only with
same-sign IC in ≥3/4 folds and pooled |t| ≥ 2. All four die → close.

## Phase 3 — Nested walk-forward A/B (`92_rc1_phase3_ab.py`)

Reuse script-54 nested protocol + script-63/88 simulator machinery.
- Arm A: frozen V6 (23 features, policy.json HPs verbatim).
- Arm B: 27 features (23 + surviving insider set), gates RETRAINED,
  same HP grids, fold-legitimate, θ=0.33 unchanged.
- Per-trade PAIRED bootstrap (10k, seed fixed) on common trades; NAV,
  win rate, precision; 2026 H1 holdout evaluated LAST, once.
- **No HP fishing on insider features** — single pre-registered config.

**PROMOTION BAR (pre-registered, same as V5 faced):**
DEV pooled paired-diff CI excludes 0 in Arm B's favor AND no fold
directionally negative AND holdout not negative. Anything less →
rejected with findings doc (the grades_historical_v5_rejection.md path).

## Phase 4 — Branch

- **Promoted:** `phase_g_v7_insider` freeze; ≥4 weeks paper shadow
  beside V6 before any swap; Script 01 gains an insider fetch for the
  ~14 weekly candidates only (FMP by-symbol, cached to db_insider.h5).
- **Rejected:** findings doc + §18 closure with cause of death.
  Either way RC-1 resolves before October.

## Budget & logistics

4 evenings, all-local compute, zero new data cost (cache covers
99.6% of matrix events — verified 2026-08-30: 775/777 tickers,
16,715/16,789 events). Live fetch needed only if promoted.
