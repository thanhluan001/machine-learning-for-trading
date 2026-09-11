# RC-15 Pre-Registration — Theme-Universe Lifecycle Allocation

**Status:** REGISTERED (2026-09-11). Official evaluation (script 115) awaits
run approval. Design by operator; probes and this document by the session
doctrine (pre-register → validate → freeze → shadow-first).
**Seat:** high-risk thematic sleeve only. The ~90% core book, the PEAD
sleeve, and all live models are untouched. Advisory/shadow until a paper
evidence period closes.

---

## 1. Objective

A complete, signal-free lifecycle allocation for the theme universe,
synthesizing everything the RC-4 program validated or killed:

```text
WHO          invested portion mirrors point-in-time capex share
             (parameter-free)
HOW MUCH     CPPI drawdown schedule on absolute exposure
             (3 thresholds + 3 ratios, chosen a priori by the operator)
WHEN THE     new-theme registration at 2-5% capex share, fed by the
UNIVERSE     watcher §7 price-RS candidate list; anchor resets on
CHANGES      registration
```

The design premise, stated by the operator and accepted as an axiom:
**no predictive signal exists in this universe** — capex lags, ratios lag
or lead-too-early, MAs lag — so price is the only truth, and protection
must be a function of price, not a forecast of it.

## 2. Frozen specification

```text
SCHEDULE (the exit mechanism; always-on, no alarm legs):
    dd        = drawdown of the mirror NAV from its anchor
    cash frac = |dd| x ratio,   ratio = 0    |dd| < 15%
                              = 1    15-25%
                              = 1.5  25-40%
                              = 2    >= 40%,  cash capped 100%
    reads     = weekly (Friday close), applied next trading day
    ANCHOR    = raw all-time high of the mirror NAV
                RESET when a new theme registers (see below)
    Rationale for raw vs 3m-MA anchor: the MA-ATH anchor trails the true
    peak by 10-15pp and engages the schedule 1-2 bands late — the
    difference between 18.8pp and 13.2pp of protection on the 2000 exam.
    The ATH's one pathology (perma-cash after a leader death) is solved
    by the registration reset, not by smoothing.

MIRROR (the allocation of the invested fraction):
    weights   = point-in-time TTM capex shares (available_date aligned),
                raw shares — no floor, no cap, no step, no momentum half.
                (Full-sample: raw mirror beat every engineered variant on
                total AND maxDD; D's momentum half held 35.6% crypto at
                the 2021 top, its floors dragged 2022.)

ONBOARDING (how the universe changes):
    watchlist = watcher §7 price-RS candidates (>=10m history, above own
                10m mean, rising 12m RS vs SPY)
    register  = a candidate's capex share crosses 2-5% (crypto shows ~2%
                is the natural emergence scale). Price discovers, capex
                confirms — each instrument in its native timescale.
    on        = registration: theme added to the mirror, ANCHOR RESET,
                deliberate manual act (proxy mapping, category check),
                at most quarterly per watcher doctrine.

DELIBERATELY EXCLUDED (each killed earlier in the program):
    momentum half (procyclical at tops), absolute-capex triggers (never
    move), Sahm recession overlay (2022 was non-recessionary), two-cadence
    alarm (converged at -67/-68% — outperformed by the always-on CPPI;
    survives only as an advisory log line), floors/caps on mirror weights.
```

## 3. Evidence (script 114, reproducible from cached basket)

2000-02 leader-death exam — 6-name era basket (Tiingo, cached at
`archive/experiments/rc15_2000_basket.csv`; across fetch variants the
final rule's protection ranged 18.8-21.6pp):

```text
                                              maxDD    total   avg cash
(1) ungated basket                            -81.0%   +459%
(2) two-cadence alarm + flat 50%              -75.4%   +580%      7%
(3) two-cadence alarm + depth (cash=dd)       -73.8%   +588%     10%
(4) band schedule, roll-104w anchor           -68.3%   +686%     18%
(5) CPPI spec, MA-ATH anchor                  -66.2%   +750%     22%
(6) CPPI spec, RAW-ATH anchor  [RC-15]        -62.2%   +801%     25%
                                              (18.8pp protection; PASSES
                                               the >=15pp bar)
```

Modern-era probe (2014-2026, equal-weight index, gross):
`-63.0% -> -56.5%` (6.5pp protection); terminal +1182% -> +596%
(avg cash 17%). **This is the premium: ~4-6pp/yr in bull-dominated
regimes, collected as +340pp of relative terminal wealth in the
leader-death regime.**

Softening profile achieved (2000 exam, RC-15 rule): market -40% ->
portfolio -31.5%; -60% -> -48%; -82% -> -61%. The first ~15-35% is the
deductible, eaten at full exposure by design.

## 4. Official evaluation protocol (script 115, run on approval)

```text
Portfolio:  mirror weights = monthly point-in-time TTM capex shares;
            dd measured on the MIRROR NAV (insure what you hold), weekly
            reads; costs 50bp per unit traded (cash-fraction changes +
            share drift); anchor resets only on (hypothetical historical)
            registration events — none exist in-sample, so the run is
            single-era: state this in the output.
Sample:     2015-01 .. 2026-08 (capex panel availability).
Benchmarks: mirror-only (no schedule), D reference (50/50, 10/70/10),
            SPY.
Gates (all must pass; no re-tuning on any outcome):
  G1 PROTECTION  maxDD(schedule+mirror) <= maxDD(mirror-only) - 5pp
  G2 PREMIUM     annualized >= mirror-only - 5pp
  G3 SHOCKS      no fast-shock episode (2016-01, 2018-12, 2020-03,
                 2024-04, 2024-08, 2026-07) with give-back > 8pp
  G4 REGRESSION  2000 reconstruction from cached basket reproduces
                 >= 15pp protection
Verdict:     all pass -> freeze as advisory reference, shadow-track in
             watcher; any fail -> RC-15 closed with cause named.
```

## 5. Known limitations (the honest contract)

1. **The first leg is the deductible.** No post-hoc rule protects the
   first 15-35%; four mechanisms tested converge at -67/-68% for that
   reason — the raw-ATH CPPI escapes the convergence only by engaging
   from truth (not a smoothed anchor) at -15%.
2. **Fast shocks under 15% are unprotected by design; boundary dips**
   (2024-04 class) cost ~5-8pp. Documented, accepted.
3. **100% cash at market -50/-60** means partial recovery participation
   until drawdown shrinks — the consolidation-regime re-accumulation is
   automatic (cash releases weekly as dd shrinks) but partial.
4. **The leader-transition leg is observationally empty.** No theme ever
   registered mid-sample; the robot scenario is mechanism-plausible, not
   tested. Failure signatures to watch when live: registration cycling,
   anchor staleness, cash that never re-deploys. The 2-5% threshold is
   the operator's a priori choice, untested in-sample.
5. **n=1 era.** The entire capex ledger is the AI era; the design is a
   concentration bet on "the best-funded theme keeps winning" with a
   price-based exit. The 2000 exam is the only leader-death observation.
6. **Bellwether panel**: 18 hand-picked companies; crypto aggregate
   unstable (few names distort it). Broad-panel rebuild is a listed
   improvement, not a blocker.

## 6. Operational path (if gates pass)

```text
Seat:        05c watcher extension — weekly CPPI line (dd, cash target),
             monthly mirror-share print, quarterly onboarding check.
Orders:      NONE. Advisory reads only, identical firewall to RC-4.
Promotion:   hypothetical tracking first (mirror+schedule NAV series in
             the monthly log); any live sizing decision is separate,
             manual, and sized for a -60% sleeve drawdown to be survivable.
```

## 7. Closure conditions (named in advance)

- G1-G4 any fail: closed, cause recorded (probe evidence says the most
  likely failure is G2 premium in the modern bull-heavy sample — that
  would mean the premium exceeded the contract, not that the mechanism
  is wrong; the closure note must say which).
- Live failure signatures (§5.4) observed over a tracking period.
- Operator decision that a -61%-class sleeve outcome is unsuitable for
  the risk budget at any available sizing.
