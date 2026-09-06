# RC-13 Pre-registration — Combined-universe V7 (2026-09-06)

## Decision context

RC-12b Phase 3 PASSED (pt-in-time SP600 transfer: DEV +1.90%, holdout
+3.68%). User decision 2026-09-06: skip the native-vs-transfer 2x2
comparison (no decision value — combined universe is the candidate
either way); `is_sp400` feature importance is the SUFFICIENT information
probe (significant -> real; null -> inconclusive, no follow-up).

## Hypothesis

A single gate-decomposition model (V7) trained on the pooled
point-in-time SP400+SP600 universe (33,604 events; is_sp400 flag;
net-of-cost gate labels) matches or beats both V6-on-SP400 and the
V6-transfer-on-SP600 on the untouched 2026H1 holdout.

## Locked design decisions

1. `is_sp400` = membership AT EVENT DATE (pt-in-time both universes;
   SP400 wins boundary-day collisions — 98 such events logged).
2. Gate labels net-of-cost: g1 car_10d > 3% + cost; g3 maxdd_ma >
   -1.5% + cost; g2 unadjusted. cost = 10bp SP400 / 30bp SP600
   (RC-12b Phase-0 measured proxies).
3. NO liquidity feature in the feature set (keeps is_sp400 importance
   readable); ADV >= $10M remains a hard execution filter downstream.
4. Own threshold via sweep + bootstrap (do NOT inherit V6's 0.33 —
   new score distribution needs its own calibration).
5. Same fold windows as V6: DEV folds 2024H2 / 2025H1 / 2025H2,
   holdout 2026H1. Per-universe (SP400/SP600 split) fold reporting is
   MANDATORY in all validation output.

## Matrix (built 2026-09-06, /features/train_matrix_combined)

33,604 events = SP400 16,696 + SP600 16,908; 2015-01..2026-07.
Cost-shifted gate base rates: SP400 pead_pass 10.5% vs SP600 5.5%
(g1 .297/.274, g2 .370/.372, g3 .397/.355). NaN signature sane
(sector_adjusted_ret_20d 19.4% top — dead names lack sector ETF
coverage; sue lags ~9-10% = known FMP coverage; no mass-corruption
signature).

## Promotion gates (pre-registered)

- G1: V7 DEV cost-adj mean > 0 AND holdout cost-adj mean > 0 with
     bootstrap CI excluding 0.
- G2: SP400-subset holdout mean >= +2.1% (= half of V6 home +4.15%;
     no material home degradation).
- G3: SP600-subset holdout mean >= 0.
- KILL: any gate fails -> V7 rejected; V6 (paper) + transfer numbers
  stand; closure documented in findings + Design.md §18.

## Explicitly out of scope

- Native-vs-transfer 2x2 comparison (dropped: no decision value).
- log(adv20) feature arm (dropped: would confound the is_sp400 probe).
- Any change to the live V6 paper pipeline before V7 passes ALL gates.
