# Future Implementation — Phase v2 Roadmap [SUPERSEDED 2026-08-15]

> **STATUS: HISTORICAL DOCUMENT — superseded by `Design.md` §18 (Research
> Backlog) and `04_backtest/archive/findings/edge_landscape_memo.md`.**
>
> This roadmap was written at the end of the Phase G v1/v2 era (July 2026)
> and much of it has since been overtaken by events. It is retained in
> `archive/` for provenance: it documents the Phase F leak discovery, the
> scope cuts taken for Phase G, and the prioritization logic of that era.
>
> **How each item was resolved (2026-08 audit):**
>
> | Roadmap item | Resolution |
> |---|---|
> | §3.6 dedup retrain (P0) | **Done** — superseded by the v4 timing-correct matrix (16,789 rows, 68 cols) and the V6 gate models; dups are long gone |
> | §3.3 live fold #5 (P0) | **Done differently** — superseded by `05b_alpaca_live/` live paper execution (V6 @ 0.33, force-refresh mh=4), which is the ongoing forward-looking test |
> | §3.2 nested-CV theta re-sweep (P1) | **Done** — V6 threshold raised 0.30→0.33 via `61_v6_threshold_bootstrap.py` (bootstrap CI excluded zero only at 0.33) |
> | §3.1 3-class classifier (P1) | **Closed — rejected** (§17.A.8: degenerate argmax; binary dominates) |
> | §2.1 Kelly sizing (P2) | Superseded by current policy: equal-weight 1/4 slots; revisit only via a research cycle |
> | §2.2 leak-clean XGBRanker (P2) | Superseded — the gate-decomposition path (V6, three independent classifiers) became the chosen architecture instead |
> | §2.3 sector-ETF CAR (P2) | Not pursued; IJH-relative CAR retained everywhere (§17.B) |
> | §2.4 tier hurdles / Perfect-Beat sim (P3) | Closed — superseded by the V6 gate architecture |
> | §2.5 Day-T re-rank (P3) | Superseded by pre-gap entry timing contract (§17.C.2) |
> | §2.6 short side / pair trading (P3) | Not pursued; long-only retained |
> | §2.7 transaction costs (P3) | Still open — must be modeled before live capital (§17.A.11) |
> | §3.4 confidence sizing (P2) | Not pursued (equal-weight retained) |
> | §3.5 regime probe (P3) | Superseded — macros tested and mostly excluded (§17.A.7); top-3 macros live inside the 23-feature set |
>
> **For current research priorities, see `Design.md` §18** (RC-1 insider
> features, RC-2 senate, RC-3 Polymarket design, RC-4 megatrend watcher)
> and the **Analysis-Depth Doctrine** in
> `04_backtest/archive/findings/edge_landscape_memo.md`.

---

(Original document follows, unchanged, for historical reference.)

---
# Future Implementation — Phase v2 Roadmap

> **Status**: Architectural roadmap and research priorities for the
> NEXT iteration of the PEAD trading bot. Compiles the design choices
> that were intentionally **downgraded or removed during Phase G** so
> Phase G could produce a defensible, deployable baseline. Most of these
> will be brought back in Phase v2 once they pass honest OOS testing.
> See `Design.md` (current architecture) and `04_backtest/strategy_v2_synthesis.md`
> (current deployable strategy) for what we have today; this doc covers
> what we WAIVED to get there.

---

## 1. Why we made compromises for Phase G

Phase G's mandate was to produce a **defensible, Sunday-safe
deployable baseline** after discovering the Phase F v2 XGBRanker's
"edge" was an artifact of forward-looking feature leakage
(`opening_gap_t1` uses `Open[T+1]` — see Phase F leak test, Phase G
Doc A §A.1 + `pead_target_findings.md §8`).

To isolate the leak, we made these deliberate scope cuts:

| Compromise | What it replaced | Why cut for Phase G |
|---|---|---|
| **Equal-weight 1/4 NAV** | Continuous-Time Kelly (Merton's Fraction) sizing engine §5 | A contaminated (leaky) sizing layer would have amplified the leak artifact. Equal-weighting also keeps cross-study comparability across Phase G docs. Will be re-introduced in v2 once the base classifier is leak-cleaned and robustly alpha-producing on Sunday features alone. |
| **`XGBClassifier` (`binary:logistic`)** | `XGBRanker` (`rank:ndcg`) cross-sectional listwise LTR §17 | The ranker's edge was the leak. Reverted to a simpler binary classifier on the 3 PEAD gates (`pead_pass`). The listwise architecture (which neutralizes cross-sectional macro noise structurally) is the right long-term design and will return in v2 with Sunday-safe features ONLY. |
| **Flat IJH benchmark for all stocks** | Per-stock sector-ETF matching (`index_ref` from SIC codes) | Sector-matched CAR neutralizes sector rotation noise but adds a look-up layer that confounded the leak-test isolation. Once Phase G's NEG_only alpha is reproduced with sector-matched CAR, the sector ETFs in `/macros/` (IJJ, IJK, IJS, XLF, XLB, XLRE, XLU, XLK, XLI, XLY, XLP, XLV, XLE, XLC) will be wired in. |
| **Single 2-stage rule (P(PEAD) ≥ θ AND gap ∈ [−15%, −2%])** | Sunday "Perfect Beat Baseline" simulation + Tier-1/2/3 dynamic weekday hurdles (1.5% / 2.5% / 4.0% CAR thresholds) | The simulator trick had no Sunday-equivalent of "real live data"; we discovered that real Sunday features have ~zero OOS signal on no-PEAD weeks (per `pead_target_findings.md` finding 1) — the estimator was a placeholder. Tier-1/2/3 threshold laddering was untested. v2: dynamic hurdle tiering may return if Phase G's NEG_only alpha fails in live-fold paper trading. |
| **2-stage Sunday pre-screen + T+1 confirm only** | Sunday ranker + weekday re-rank (two-pass full inference) | A two-pass full-recompute is more complex than the T+1 gap filter. v2: the re-rank pass becomes useful as a comma-2-sigma-cap protective layer when the ranker returns, to detect late-flowing price anomalies (e.g., pre-close gaps). |
| **No short side / hedged book** | Long-only | Long-only is the simplest OOS-testing config; the empirical claim in `pead_target_findings.md §0 finding 4` is that shorting on no-PEAD weeks is a wash-out, while long on NEG-gap-PEAD is the actual alpha engine. v2: revisit once we have live evidence that NEG_only long survives fold #5. |
| **No transaction cost model** | Idealized backtest | Keep simple for cross-study comparability. v2: ISG+per-event commissions impact quantifiable; estimate as a Sharpe penalty (`-0.08` to `-0.15` likely). |

Each of these is captured in more detail below.

---

## 2. Architectural features DEFERRED to v2

### 2.1 Continuous-Time Kelly (Merton's Fraction) Position Sizing

**Originally**: §5 of `Design.md` —
$$K^* = \frac{1}{\gamma} \cdot \frac{\mu}{\sigma^2}$$
where $\mu$ = weekday final predicted CAR from the calibrator, $\sigma^2$ = scaled historical daily return variance matching the 10-day horizon, $\gamma = 2$ (Half-Kelly).

**Compromise in Phase G**: equal-weight `1 / 4 NAV` per slot. No Kelly.

**Why deferred**:
1. Kelly requires **a calibrated expected-return point estimate ($\mu$)** as input. The XGBRanker's per-stock `mu` was wrong because edge was leak-driven; the XGBClassifier outputs `P(PEAD)` which is **not a $\mu$ prediction** — it's a probability-of-class-membership.
2. The non-monotonic proba→PnL relationship (`pead_target_findings.md §6.6`) — high-confidence picks do NOT have high PnL — means a naive $\mu = a + b \cdot \text{proba}$ mapping is misleading. Kelly would over-allocate to the high-proba picks that empirically had middling PnL.

**v2 plan**:
1. **First**: train a `multi:softprob` 3-class classifier on `{no PEAD, small PEAD, large PEAD}` (§3.1 below) — the softprob probabilities naturally yield a magnitude-shaped confidence.
2. Then derive a calibrated $\mu$ via isotonic regression on `P(large PEAD)` → `mean(car_10d | bucket)`, fit on the SWEEP-VAL set per fold (nested CV).
3. Pass $\mu$ through Kelly with $\sigma^2$ from the rolling 20-day pre-event variance already in `train_matrix` as `pre_event_idiosyncratic_vol` (squared).
4. Validate against the leak test: NaN the Kelly input (`mu`) — the strategy's alpha must come from the **ranking** of picks, not from the size assigned to each.

**Implementation debt**:
- Add `04_backtest/_v2_kelly_sizing.py` mirroring `04_phase_g_portfolio.py`'s `simulate_portfolio` but with `kelly_size(mu, var)` per slot.
- Cross-study: confirm `v2_kelly` vs `v2_equal` gives a HIGHER mean IRR but acceptable mean MaxDD. Stop-loss `-10%` per-trade (Doc H §H.7.3) is the boundary.

### 2.2 XGBRanker `rank:ndcg` (Cross-Sectional Listwise LTR)

**Originally**: §17 of `Design.md` — `XGBRanker` with `objective="rank:ndcg"`, `eval_metric="ndcg@3"`, group = `calendar_week_group` (cross-sectional weekly groups), Isotonic calibration bridge → $\mu$ → Kelly sizing.

**Compromise in Phase G**: `XGBClassifier (binary:logistic)` on 17 Sunday-safe features, target = `pead_pass` (the 3 PEAD gates combined). No listwise LTR, no isotonic calibrator, no NDCG optimization, no cross-sectional weekly groups except as the **nested-CV fold structure** (TRAIN + SWEEP rolled into SWEEP-VAL for per-fold POS-tuned HP).

**Why deferred**:
1. The XGBRanker's `main()` in `03_model/01_train_model.py` still trains this model and CARRIES the leak. UNUSABLE.
2. NDCG-optimized ranker needs a CARDINAL gain target; `car_10d` early-Phase F used log CAR directly. Tests showed NDCG@3 = 0.776 on TRAIN but only 0.086 on VAL — the ranker overfit the cross-sectional ordering on the leak-contaminated TRAIN and couldn't generalize OOS.
3. A leak-clean Sunday ranker targeting `pead_pass` (binary gate label, not cardinal CAR) would require bucketing into NDCG-relevant ranks. The bucket count (N_BUCKETS=10, integer-bucketed `car_10d` quantiles per `01_train_model.py`) is an untested HP.

**v2 plan**:
1. **First**: confirm the Phase G XGBClassifier on Sunday-safe features survives fold #5 live paper-trading (+1.31 ± CI cross-fold Sharpe).
2. Research: train a leak-clean ranker on the 17 Sunday-safe features. Compare NDCG@3 cross-fold vs. the classifier's AUC: which delivers a better top-k recall on `pead_pass`?
3. If ranker wins, the calibration bridge becomes: `ranker.predict_proba` equivalent → isotonic calibrated $\mu$ → Kelly.
4. If classifier wins (likely simpler and more interpretable), stay with classifier and add Kelly separately (§2.1).

**Implementation debt**:
- `03_model/04_v2_ranker_train.py` — train the ranker on `{SUNDAY_SAFE_FEATURES}` only, NO leak features.
- `03_model/05_v2_ranker_eval.py` — nested-CV NDCG@3 vs. classifier AUC@3 on identical split.
- Drop the `01_train_model.py main()` deprecated block once the ranker is leak-clean v2.

### 2.3 Per-Stock Sector-ETF Relative-Return Benchmark

**Originally**: §14 of `Design.md` — replace IJH flat benchmark with per-stock sector ETF (using `/macros/{index_ref}` from `SIC_code_to_index.md`).

**Compromise in Phase G**: flat IJH benchmark for all stocks. `car_10d` and `relative-return` features computed against `/macros/IJH` only.

**Why deferred**:
1. Sector ETFs in `/macros/` (XLC, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLY, XLY, IJJ, IJK, IJS, XLB, XLU) have DIFFERENT start dates (XLC starts 2018, XLRE starts 2019). Sector-matched CAR for pre-2018 events would intermittently fall back to SPY or IJH anyway, creating a regime-dependent benchmark.
2. The leak-test isolation needed a clean CAR definition — sector-matching would have folded sector rotation into the same noise group that masked the leak.

**v2 plan**:
1. Implement `02_features/03_v2_sector_car.py` computing `car_10d_sector` using `/macros/{index_ref}` per permaTicker.
2. Run dual-CAR columns for cross-validation: `car_10d_flat_IJH` vs `car_10d_sector`. Compare the cross-fold Sharpe of the recommended operating point on each.
3. Validate against a leak-test analog: if sector-matched CAR differences are stable across folds but flat-CAR differences collapse, sector matching is alpha — keep it.

**Implementation debt**:
- New column `car_10d_sector` in `/features/train_matrix`.
- New meta-feature `rel_ret_sector_{3,5,10,20,30}d` (mirror of `rel_ret_*d` but vs sector ETF).
- Verify `/macros/{index_ref}` is fully populated for all permaTickers in `/metadata/sp400_permatickers`.

### 2.4 Sunday "Perfect Beat Baseline" Simulation + Tier-1/2/3 Dynamic Weekday Hurdles

**Originally**: §2 + §3 + §8 of `Design.md` — Sunday screening ranks candidates by a calibrated "Simulated Predicted CAR / Prediction Standard Error" ratio with `SUE_current = 2.0, SUV_day_1 = 3.0` (Perfect Beat Baseline) substituted for unrealized catalyst features, then assigns each candidate to a Tier (1-3, 4-7, 8-10) and the weekday engine enforces per-Tier CAR thresholds (1.5%, 2.5%, 4.0%).

**Compromise in Phase G**: Single Sunday classifier → `P(PEAD)`, single threshold `theta = 0.20`. No tiering.

**Why deferred**:
1. `pead_target_findings.md §0 finding 1` established that 17 Sunday-safe features have **~zero OOS signal alone** — the "Perfect Beat Baseline" simulation would substitute static values for features that contribute nothing empirically. The tier-splitting was structured on top of a noise estimator.
2. The (1.5%, 2.5%, 4.0%) threshold ladder was untested design intent — it had no Phase F or Phase G empirical justification. The number of tiers (3) and the threshold split-points were arbitrary.

**v2 plan**:
1. **First**: add at least one fundamental pre-event feature whose OOS signal we can validate (analyst revision momentum is a likely candidate now; short interest if a 15-year-equivalent source can be found).
2. If and only if new fundamental features lift Sunday-classifier AUC >0.7 (current ~0.66), reconsider tiering: split Sunday picks into top-quartile `P(PEAD)` and run **multiple theta thresholds per quartile** (data-driven via nested CV). Multiple-threshold tiering might lift the Sharpe ceiling by tightening the trade-access gate on high-variance picks.
3. If new features don't lift AUC substantially, tier-laddering's value is probably not worth the complexity (Doc J pattern: in-sample rule-tuning rarely survives OOS).

**Implementation debt**:
- New fundamental feature(s) in db.h5 (analyst-revision momentum, short interest).
- New `_v2_quartile_hurdle.py` in `04_backtest/` that sweeps theta per quartile-of-proba, IF the new-feature AUC threshold is met.

### 2.5 Two-Pass Sunday-Plan + Weekday Re-Rank Inference

**Originally**: §17.5 of `Design.md` — Sunday produces a ranked plan, weekday** re-runs the ranker** against live-updated features (`SUE_current`, `SUV_day_1`, `VWAP_Coherence`) to produce a fresh final-rank list, executes sequentially.

**Compromise in Phase G**: Single Sunday-plan, weekday only does **filter** (`P(PEAD) ≥ θ` AND `gap ∈ [−15%, −2%]`), no re-rank. The 4 Day-T features (`intraday_range_t`, `volume_vma20_ratio_pre_event`, `suv_day_1`, `opening_gap_t1` post-realization) are NOT re-fed to the model; only `opening_gap_t1` enters as a confirmation rule.

**Why deferred**:
1. A re-rank pass requires the Day-T features (currently incomplete pre-event — `suv_day_1` is volume-shock-on-day-T, only knowable after T-close, not at T+1 morning which is the entry point).
2. The re-rank is more aggressive than necessary; the empirically-strongest alpha is the NEG-only filter, NOT the rank order.

**v2 plan**:
1. **First**: A 5-feature Day-T confirmation sub-model trained ONLY on post-T-close data (Day-T features available after market close of T, before T+1 entry) could become a re-rank layer.
2. `04_backtest/_v2_day_t_rerank.py` — sweeps a Day-T confirmation model with `theta_t1` threshold; the existing `opening_gap_t1` filter becomes the simplest case (`gate = opening_gap_t1 ∈ [-15, -2]`).
3. If the Day-T model lifts per-trade expectancy beyond the current `+2.59%`, the re-rank layer is justified.

**Implementation debt**:
- `02_features/04_v2_day_t_confirmation.py` — derive a post-T-close "Day-T deck" of features for the weekday re-rank pass.

### 2.6 Short Side + Hedged Book

**Originally**: implicit in §1 (drift goes both ways for negative-gap events that fell short — could short) + §3 (the original Phase F design assumed long-only but contemplated hedge variants).

**Compromise in Phase G**: long-only NEG_only. The empirical claim in `pead_target_findings.md §0 finding 4` is that shorting on no-PEAD weeks is a "wash-out" while long on NEG-gap-PEAD is the alpha engine.

**Why deferred**:
1. Scope limitation; short-side borrow is hard to backtest honestly (we don't have borrow-availability timeseries).
2. The negative-momentum-side requires a fully separate model (POS_only — the mirror). Per `strategy_v2_synthesis.md §2.2`, POS_only had mean Sharpe +0.86 but only 58% random exceedance — regime-fragile, not the winner.

**v2 plan**:
1. **First**: survive live fold #5 for NEG_only with the long-only baseline.
2. If live data shows the POS_only side of 2024 H2 was the regime anomaly, develop a **POS_only mirror** as a second model — short the high-probability POS-gap events (i.e., those with `opening_gap_t1 ∈ [+2%, +8%]` per `14_phase_g_trade_stats.py` probe section).
3. Pair-trade: long NEG_only simultaneously with short POS_only — net approximately market-neutral.

**Implementation debt**:
- New `_v2_pos_only_classifier.py` trained on the POS-gap events (specifically: bucket `opening_gap_t1 ∈ [+2%, +8%]`, where `14_phase_g_trade_stats.py` probe showed hit rate +60% with avg PnL +0.435%).
- New `_v2_pair_trading.py` portfolio simulator handling simultaneous long + short.

### 2.7 Transaction Cost + Slippage Model

**Originally**: implicit baseline flag ("no transaction costs" in `04_backtest/README.md §1`).

**Compromise in Phase G**: idealized backtest, no commission, no slippage, no borrow fee.

**Why deferred**:
- For the cross-study comparability across Phase G docs (B-J), keeping the eval frictionless made regression-discovery clean.
- The realistic impact is now estimable: mean trades per fold = 7.2 × 4 folds = 28.8 trades/year on a 4-slot portfolio. On a 15.57% IRR headline, broker commissions + bid-ask spread on mid-caps could shave ~0.5% per trade → ~1.4% drag → realized IRR ~14.2%.

**v2 plan**:
1. Model commission: $0.005/share (IBKR Pro is a reasonable baseline).
2. Model slippage as 0.1-0.2% of trade-notional for mid-caps.
3. Compute annualized IRR drag: validate against today's +15.57% baseline; if v2-with-costs drops below ~10%, prioritize a longer-hold variant (T+1→T+21) to trade fewer times per year.

**Implementation debt**:
- Add `transaction_costs` parameter to `simulate_portfolio` in `04_phase_g_portfolio.py`; default backtest remains free, but live-model variant enables it.

---

## 3. Research extensions ordered by expected value

These come directly from `strategy_v2_synthesis.md §5.5` and
`pead_target_findings.md §7.2`. They are research experiments
rather than architectural-rollback items, but several overlap with
the deferred architecture (e.g., §3.1's 3-class classifier is a
Kelly-sizing precondition; §3.3's regime probe overlaps with §2.6
pair-trading).

### 3.1 Magnitude-aware 3-class `multi:softprob` classifier — HIGHEST CEILING LIFT

`pead_target_findings.md §7.2 (a)` — target `{no PEAD, small PEAD, large PEAD}` instead of binary `pead_pass`. Why:
- The Phase G v1.1 binary classifier has a non-monotonic proba→PnL relationship (`strategy_v2_synthesis.md §6.6`: high-proba picks have only middling PnL). Confidence calibration requires knowing WHY a pick has high `P(PEAD)` — is it "high confidence small PEAD" or "low confidence large PEAD"?
- A 3-class softprob yields a 3-vector per pick that contains both a confidence and magnitude signal.
- The magnitude component unlocks **calibrated Kelly sizing** (§2.1).

Implementation target: `03_model/06_v2_3class_classifier.py`. Compare per-fold Sharpe vs. baseline. If 3-class lifts Sharpe > +1.5 with CI lower bound > +1.2, become the v2 default.

### 3.2 Re-sweep theta + gap under proper nested CV — CLOSES RESIDUAL CIRCULARITY

`strategy_v2_synthesis.md §5.5 item 2` (highest-priority honesty closure). The current theta=0.20 + gap=[−15%, −2%] operating point was selected FROM the 2024–2026 OOS data; Doc H's +1.31 Sharpe CI lower bound contains this selection bias.

Implementation: `04_backtest/15_v2_nested_cv_theta_gap_resweep.py` — mirror Doc J's dead-zone nested-CV procedure applied to the 2D (theta, gap) grid.

Expected outcome: the +1.31 cross-fold Sharpe regresses to +1.0–+1.2 if the theta+gap operating point loses nested-CV convergence across folds. We accept the regression — it provides a more honest deployable baseline.

### 3.3 Live paper-trading fold #5 — FIRST FORWARD-LOOKING OOS DATA POINT

`strategy_v2_synthesis.md §5.5 item 1` (highest-priority validation). Live 6-month tracking of the +1.31 baseline operating point produces the first OOS fold that is uncontaminated by ANY of the 5 in-sample rule-tunings (`theta=0.20`, `gap=[-15,-2]`, per-fold POS-tuned gamma=10/5/3/3, dead-zone rescindment, model-fit param choices).

Implementation: `04_backtest/16_v2_live_tracker.py` — dailyBAR-CSV live-tracking harness that logs actual trades and realized PnL against the model's predicted probability + realized opening_gap_t1.

### 3.4 Confidence-calibrated sizing — DEPENDS ON §3.1

`strategy_v2_synthesis.md §5.5 item 4` (medium-priority). Once a 3-class classifier is available (§3.1), the `P(large PEAD)` softprob can power a confidence-weighted equal-weight variant:

```python
w = 0.5 + 1.5 * P_large   # baseline 0.5 weight, scaled up by P_large
```

Compare cross-fold Sharpe vs. equal-weight. If higher, becomes the v2 sizing baseline (Kelly (§2.1) is the long-term aspiration, but the confidence-weighted variant is a low-effort intermediate).

### 3.5 Regime probe feature — POS- vs NEG-favorable regimes

`strategy_v2_synthesis.md §5.5 item 5`. Phase G App E (NEG_only winner) and Doc G §G.8 (gap buckets flip sign across folds) both indicate that the alpha engine is regime-dependent. Adding a regime feature to the Sunday-safe feature set could enable regime-aware dispatch:

- Compute a rolling 60-90 day signal e.g., "fraction of earnings events that produced positive `car_10d` over the last 60d" or "macro state (yield curve, VIX)".
- Add to `SUNDAY_SAFE_FEATURES` if it lifts OOS AUC.

Implementation: `04_backtest/17_v2_regime_probe.py`.

### 3.6 Retraining on deduped `train_matrix` — CLOSURE OF PHASE G CLEANUP

`strategy_v2_synthesis.md §6 + §5.6 item 1` (high-priority reproducibility). Today's Phase G artifacts (in `03_model/models/phase_g_v1_1_*` and `04_backtest/archive/experiments/`) were trained on the pre-dedup `train_matrix` of 22,607 rows with 1,342 dup rows. The dedup-confirmation run on the current 20,265-row `train_matrix` (`14_phase_g_trade_stats.py` today) produced a cross-fold Sharpe +1.23 (vs Doc H +1.31 — within noise).

Implementation: produce a `phase_g_v2` artifact folder trained on the deduped matrix. Roadmap:
1. Retrain classifier via `03_model/02_phase_g_sunday_classifier.py` → `03_model/models/phase_g_v2_sunday_classifier/`.
2. Re-run 4-fold nested CV via `04_backtest/06_phase_g_nested_cv.py` — confirm per-fold gamma=10/5/3/3 POS-tuned HP still wins or find new HP.
3. Re-run bootstrap CI via `04_backtest/11_phase_g_bootstrap_ci.py` against the new artifacts.

If cross-fold Sharpe shifts by >0.2 pp vs. the current +1.23, the v2 baselines should adopt the new artifacts; otherwise keep the existing for cross-study comparability.

---

## 4. Folder layout for v2 work (proposed)

```
luan_bot_trading/
├── 02_features/
│   ├── 01_features_gate_events.py        # current (no v2 change)
│   ├── 02_build_feature_matrix.py        # current (no v2 change)
│   ├── 03_v2_sector_car.py               # §2.3 — sector-matched CAR
│   └── 04_v2_day_t_confirmation.py       # §2.5 — Day-T re-rank deck
├── 03_model/
│   ├── 01_train_model.py                 # legacy (helper API still used)
│   ├── 02_phase_g_sunday_classifier.py   # current deployable
│   ├── 03_phase_g_sweep.py               # current
│   ├── 04_v2_ranker_train.py             # §2.2 — leak-clean ranker
│   ├── 05_v2_ranker_eval.py              # §2.2
│   ├── 06_v2_3class_classifier.py        # §3.1 — magnitude-aware target
│   └── models/
│       ├── phase_g_v1_sunday_classifier/   # current deployable
│       └── phase_g_v2_sunday_classifier/   # §3.6 — retrained on deduped
└── 04_backtest/
    ├── _v2_kelly_sizing.py               # §2.1 — Continuous-Time Kelly
    ├── _v2_quartile_hurdle.py            # §2.4 — theta-laddering per quartile
    ├── _v2_day_t_rerank.py               # §2.5 — re-rank pass
    ├── _v2_pos_only_classifier.py        # §2.6 — POS_only mirror for pair trading
    ├── _v2_pair_trading.py               # §2.6 — simultaneous L+S
    ├── 14_phase_g_trade_stats.py         # current
    ├── 15_v2_nested_cv_theta_gap.py      # §3.2 — closes circularity
    ├── 16_v2_live_tracker.py             # §3.3 — live-fold tracking
    └── 17_v2_regime_probe.py             # §3.5
```

Naming convention: `_v2_*.py` files are **shared libraries** (mirroring the
existing `_phase_g_random_baseline.py` pattern); `15_v2_*.py` and onwards are
**numbered pipeline scripts**, continuing from `14_phase_g_trade_stats.py`.

The existing Phase G scripts (01-14) are LEFT AS-IS for cross-study reproducibility — we keep comparing all v2 results back to the same Doc A-J baselines.

---

## 5. Prioritization and acceptance gates

The honest priority ordering, with rough effort estimates and "do not start until" gates:

| Priority | Item | Effort | Gate before starting |
|---|---|---|---|
| **P0** (validity) | §3.6 — Retrain on deduped `train_matrix` | 1-2 days | None — start immediately. The deduped matrix exists today (`/features/train_matrix`: 20,265 rows × 30 cols, 0 dups). |
| **P0** (validity) | §3.3 — Live paper-trading fold #5 | live tracking, no engineer effort | None — start immediately. The `+1.31` baseline is deployable today. |
| **P1** (honesty) | §3.2 — Nested CV re-sweep of (theta, gap) | 3-5 days | None — mirrors Doc J procedure. |
| **P1** (ceiling lift) | §3.1 — 3-class `multi:softprob` classifier | 5-7 days | After §3.6 confirms the binary classifier's +1.31 is stable on deduped data. |
| **P2** (sizing) | §2.1 — Continuous-Time Kelly sizing | 3-5 days | After §3.1 (needs calibrated $\mu$ that the 3-class softprob provides). |
| **P2** (cross-sectional design) | §2.2 — Leak-clean XGBRanker | 5-7 days | After §3.1 + §3.6 (so v2 baseline exists to compare against). |
| **P2** (price-data refinement) | §2.3 — Per-stock sector-ETF benchmark | 2-3 days | Independent of others; can parallelize. |
| **P3** (control structure) | §2.4 — "Perfect Beat" simulator + tier hurdles | unbounded | AFTER new fundamental features from §3.5 lift AUC >0.7. Complex; unlikely to pay off until features improve. |
| **P3** (post-T data model) | §2.5 — Day-T confirmation re-rank | 5-7 days | Independent of others; can parallelize with P2s. |
| **P3** (multi-side strategy) | §2.6 — POS_only mirror + pair trading | 7-14 days | After live fold #5 (§3.3) confirms NEG_only survives regime-modification. |
| **P3** (live realisms) | §2.7 — Transaction cost + slippage | 1-2 days | Should be done BEFORE live deployment; otherwise academic-only. |
| **P3** (regime) | §3.5 — Regime probe feature | 3-5 days | After §3.6 + §3.2 (so we have a stable +1.0–1.2 floor to evaluate against). |

### Acceptance gates per item

- **§3.6 retrain**: pass if cross-fold Sharpe ∈ [1.0, 1.5] on the deduped matrix; FAIL if Sharpe <0.9 (the cleanup perturbed the model-state in unexpected ways — debug).
- **§3.3 live fold**: pass if 6-month live Sharpe ≥ 0.5 AND ≥ 5 trades (small-sample confidence); "success" is accumulating evidence over folds 5-8, not a single fold.
- **§3.2 theta+gap nested-CV**: pass if 3 of 4 folds AGREE on theta range and gap range (within a tolerance); FAIL if boundaries scatter regime-dependently (Doc J pattern for dead-zone). Either way, we get a defensible honest Gap-Sharpe range.
- **§3.1 3-class classifier**: pass if cross-fold Sharpe > +1.5 AND CI lower bound > +1.2; FAIL = the binary classifier + Kelly remains the simpler choice.
- **§2.1 Kelly**: pass if mean IRR rises ≥3pp AND mean MaxDD stays within -10%; FAIL if Kelly doesn't outperform equal-weight, the realizability cost of mixing doesn't pay.
- **§2.2 ranker**: pass if cross-fold NDCG@3 vs classifier's AUC@3 shows >5pp edge on the SAME features; FAIL if classifier wins (simpler, keep).
- **§2.3 sector CAR**: pass if cross-fold Sharpe using sector-CAR > flat-CAR baseline by ≥0.05; FAIL = sector rotation is not alpha for this strategy.
- **§2.6 pair trading**: pass if combined long+short mean Sharpe > NEG_only long-only mean Sharpe by ≥0.2 AND drawdown is reduced; FAIL = short borrow + borrow fees exceed the alpha gain.

---

## 6. Acceptance philosophy: what makes V2 an upgrade vs V1

The Phase-v2 investment is worthwhile ONLY IF any one of these:

1. **Honesty floor**: the +1.31 cross-fold Sharpe survives §3.2's nested-CV re-sweep at +1.0 or higher (and §3.3's live-fold confirms it). This is the floor below which v2 is a small win. The +1.31 ceiling of Phase G after-the-fact was inflated by in-sample rule-tuning; v2's honestly-reswept baseline is the genuine benchmark.

2. **Ceiling lift**: any one of §3.1 (3-class classifier), §2.1 (Kelly sizing), or §2.6 (pair trading) lifts the cross-fold Sharpe by ≥0.3 pp OR the IRR by ≥5 pp, without a parallel collapse in MaxDD beyond −10%.

If neither is met, v2 is just a correctness exercise on Phase G's baseline, and the architectural revolutions (Kelly, ranker, tier hurdles, "Perfect Beat" simulator) should be MODELS, not production code. That is still a worthwhile contribution to the codebase architecture (cleaner abstractions for sizing and inference), but it isn't the "kill the Phase G baseline" win we want.

---

End of Future Implementation roadmap. The live Phase G baseline stays deployed; the v2 improvements described here are RESEARCH extensions, not productionized until their acceptance gates clear.

The current `Design.md` and `04_backtest/strategy_v2_synthesis.md` remain the AUTHORITATIVE production references.
