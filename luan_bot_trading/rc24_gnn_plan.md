# RC-24 Plan — Peer-Propagation GNN for Earnings Surprise (DRAFT FOR REVIEW)

**Status:** DRAFT — not registered. Decisions marked `[DECISION]` are yours.
**Predecessors:** everything in RC-16 → RC-23 + the beat/payoff/market-adjustment
diagnostics (2026-09-16). This plan bakes in their lessons.

---

## 0. Why this, in one paragraph

The thesis (yours, validated): edges survive only on triggers that cannot be
front-run. Pre-known events (ex-dividend, insider filings) are dead — we
measured that. Earnings *surprises* are un-front-runnable by definition; the
only question is whether our surprise prior is sharper than the market's.
Individual-stock features failed (feature audit: median AUC 0.499). But peer
propagation is real and *under-priced*: among low-confidence events, drift
given a beat rises +0.28% per +1% of peer drift (t=2.3), concentrated in
low-attention SP600 names; the market's read-across is slowest exactly there.
Our hand-built peer feature (F1_sb_h3) was beat-anchored and one-sided. A GNN
with signed outcomes aggregates many weak connections (attention over
industry/correlation/ETF edges) and carries BOTH branches — the miss branch is
the genuinely new, untested part.

**Anchors from this session (fixed facts the plan must respect):**

```text
beat base rate 68%; XGBoost-22 AUC(beat) = 0.673 (SP400 0.694 / SP600 0.649)
  → but entirely priced: excess EV of its picks ≈ +0.02% (the raw edge was beta)
payoff structure: E[excess|beat] = +1.7% , E[excess|miss] = −4.3% (excess terms)
low-confidence region: breakeven P(beat) ≈ 60% at current payoffs;
  current stack reaches 57% → the gap a GNN must close (or cushion the miss)
miss branch: untested for peer information (our peer pool was beaters-only)
measurement law: EXCESS returns everywhere; splits on label_end maturity;
  PIT edges; truncation-equivalence for the graph builder
```

---

## 1. Scope

```text
IN  v1:  unified SP400+SP600 graph; edges = sector-ETF + fine industry +
         trailing return correlation; outcomes = signed EPS surprise % +
         1d/3d post-print returns (horizon-masked); GAT 2 layers; label =
         beat/miss; economic evaluation = excess-EV decomposition
OUT v1:  supply-chain edges (no data — vendor acquisition later, if ever)
         valuation features (P/E, EV/EBITDA — no data yet)
         options-implied anything (no data; would price the miss branch)
         transcripts (separate future program)
         any live trading — research only; shadow book only after gates pass
```

---

## 2. Data plan — the initial embedding inputs, source by source

### 2.1 What already exists (inventory verified this session)

```text
prices             /sp400/{pt} in db.h5 (932 tickers), /sp600/{pt} in
                   db_sp600.h5 — OHLC + volume, enough for returns, adv20,
                   trailing correlations
benchmarks         /macros/IJH (SP400), /sp600/benchmark_IJR (SP600) — for
                   EXCESS returns
events             train_matrix_v7c (33,598 rows) with the 22 deploy features,
                   entry/exit dates, pass_g1, label_end, sector (ETF), is_sp400
beat labels        SP400: sue_score (v6c join). SP600: actual vs estimate
                   from /sp600/earnings_full/{ticker}
industry           SP400: SIC via /metadata/sp400_permatickers (PIT caveat:
                   current-day values; SIC changes are rare — disclosed)
                   SP600: gics_sector + gics_sub_industry in /metadata/sp600
peer machinery     rc18_p0_firewall.py: drift computation, maturity masks,
                   embedding checkpoints, truncation-equivalence harness
```

### 2.2 Node features for `CompanyEmbeddingLayer` (continuous block)

```text
from v7c (already PIT):  pre_event_idiosyncratic_vol, rel_ret_5d/10d/20d/30d,
                         sector_adjusted_ret_20d, revision_momentum_30/60/90d,
                         revision_intensity_90d, grade_dispersion_90d,
                         n_analysts_covering, sue_lag_1/2,
                         consecutive_surprises_pre, vix, fed_funds,
                         unemployment_roc21
compute from prices:     log(adv20)  — 20-session mean dollar volume from the
                         existing price frames (Close × Volume; fallback to
                         Close if Volume dirty). Serves as the size/liquidity
                         proxy for v1.
[DECISION] market cap:   v1 uses log(adv20) as the size proxy — no new data
                         needed. TRUE market cap (shares × price, PIT by
                         report date) needs FMP income-statements
                         (weightedAverageShsOutDiluted) for ~2k tickers —
                         ~1 day of API pulls, deferrable to v1.1.
[DECISION] P/E, EV/EBITDA, IV rank: deferred (no data; IV is the most
                         interesting future addition — it prices the miss
                         branch — but it's an options-data acquisition).
```

### 2.3 Categorical IDs for the industry embedding

```text
common layer (both universes):  sector ETF (11 values) — already a column
fine layer (within universe):   SP400: SIC-2 division / SIC-4 group
                                SP600: gics_sub_industry (~150 values)
v1 design:  nn.Embedding over the FINE taxonomy per universe + a shared
            sector-ETF embedding added on top (the spec's 16-dim industry
            embedding; fine taxonomy clipped to categories with >= 30
            events, remainder → "other")
[DECISION] accept the SIC-vs-GICS taxonomy split, or map both to sector-ETF
           only for v1 (simpler, coarser)? Recommendation: fine-per-universe
           + shared sector layer, as above.
```

### 2.4 Edge types (v1: three, all computable today)

```text
E1 same fine industry (within universe; weight 1)
E2 same sector ETF (cross-universe; weight 1)
E3 return-correlation: trailing 120-session return correlation, computed
   PIT at MONTHLY checkpoints (same discipline as the SVD embedding
   checkpoints). Keep top-k = 10 correlations per stock with |rho| >= 0.35.
   This is the edge that must pass truncation-equivalence — it is the
   likeliest leak site (a full-history correlation = the RC-16 leak reborn).
[DECISION] supply chain: out of scope v1 (no vendor). If ever added:
           FMP supply-chain endpoint or FactSet Revere — priced separately.
```

### 2.5 Peer outcome features (the message payload — the core novelty)

Per reporting peer j (reported within T_target − 10 sessions):

```text
signed EPS surprise %   SP400: from the SUE pipeline's actual/estimate
                        (verify raw fields exist in the earnings tables;
                        fallback = z-scored SUE itself)
                        SP600: (actual − estimate)/|estimate| from
                        earnings_full (already used, 88.9% time coverage)
1-day post-print return r1 (excess vs own benchmark)     [always observable]
3-day partial r3 (excess)  [only if fully printed before cutoff — masked]
horizon mask: per-peer observable-window flag (the two-clock rule from the
              peer work; the spec's single reported_mask is NOT enough)
direction completeness: missers are INCLUDED (this is the point — the miss
              branch finally gets information)
```

### 2.6 Data acquisition checklist (effort)

```text
D1  verify EPS actual/estimate fields for SP400 earnings tables      0.5 h
D2  compute adv20 from existing price frames (both universes)        1 h
D3  reconcile SIC-4 (SP400) / GICS sub-industry (SP600) into the
    category index, with >= 30-event clipping                        1 h
D4  [v1.1 only] FMP income-statements pull → shares × price →
    PIT market cap (batch, ~2k tickers, rate-limited)                1 day
D5  nothing else — v1 runs on data we own
```

---

## 3. Model & training protocol

```text
architecture      exactly your spec: CompanyEmbeddingLayer (industry embed
                  16-d + continuous proj 32-d → hidden 64) + signed outcome
                  encoder + GATConv(64, 4 heads) → GATConv(64, 1) → MLP head
sample            one target event = one graph: target node + all reporters
                  in [T−10 sessions, T); edges E1/E2/E3 restricted to the
                  subgraph; unreported/late peers masked (no leakage)
label             beat = 1 (definitions as in §2.5)
loss              BCEWithLogitsLoss (68/32 imbalance is mild; no reweighting
                  unless calibration checks fail)
splits            label_end maturity folds — the SAME DEFAULT_FOLDS as every
                  baseline this session, for comparability
training          CPU-only (trading env), early stopping on the fold's
                  validation slice, dropout as specced; 3 seeds, report mean
                  and spread; HP search LIMITED to one pass over
                  {hidden 32/64, heads 2/4, dropout 0.1/0.2} — frozen after
environment       pip install torch (CPU) torch-geometric into the trading
                  env; version-pin in environment.yml; note in
                  ENVIRONMENT_INCIDENTS.md if any BLAS conflict appears
                  (torch has its own — verify no MKL GEMM issue resurfaces)
```

### Baselines (the bar, all on identical splits)

```text
B0  XGBoost-22                        AUC(beat) ≈ 0.673  (known)
B1  XGBoost-22 + F1_sb_h3 + A2        (the existing peer features)
B2  GNN with outcomes zeroed          (graph structure alone, no spillover —
                                       isolates the peer-outcome contribution)
```

**B2 is the critical ablation**: if the GNN beats B0/B1 but not B2, the gain
is the node features, not propagation — the thesis fails.

---

## 4. Evaluation — the part that killed everything else, applied from day 1

### 4.1 Prediction metrics (secondary)

```text
OOS AUC(beat) overall + by universe + by coverage tercile (n_analysts)
increment: GNN − B1 AUC, with the low-coverage SP600 slice as the PRIMARY
slice (that is where the +0.28 slope evidence lives)
miss-side AUC: AUC(1−beat) — the branch our F1 never carried
calibration: predicted vs actual beat rate by decile (XGBoost's compression
  to [0.39, 0.88] limited its usable thresholds)
```

### 4.2 Economic metrics (primary — the gates)

```text
EXCESS returns only (stock − own benchmark, same windows as the matrix)
EV decomposition per prediction decile:
    n, P(beat), E[excess|beat], E[excess|miss], EV
MISS-BRANCH experiments (the new ground):
    a. E[excess|miss] by GNN miss-confidence        — does it rank the miss?
    b. E[excess|miss] by peer-support tercile       — miss cushioning test
       (if cushioned to −2.5%, breakeven drops 60% → 48%: game-changer)
week-block bootstrap 10k, seed 20260807, for every reported interval
```

### 4.3 Pre-registered gates (draft — for your review)

```text
G0  FIREWALL: graph builder passes truncation-equivalence (>= 100 events,
    rebuilt with prices truncated at each cutoff; predictions identical;
    negative control = deliberately leak h+5 and require detection)
G1  INCREMENT: GNN OOS AUC(beat) ≥ B1 + 0.02 in the low-coverage slice,
    AND B2 ablation confirms the gain comes from peer outcomes
G2  ECONOMIC:  some pre-specified slice (defined by the model's own
    calibration, frozen before evaluation) has EV > 0 with bootstrap CI
    excluding 0 on EXCESS returns
G3  MISS BRANCH: either miss-side AUC > 0.55 or significant miss cushioning
    (report regardless of pass — this is the thesis's novelty)

PASS → shadow-book decision (forward OOS only; no capital)
FAIL at G1 → the propagation thesis is dead; close RC-24, publish the map
FAIL at G2 only → mechanism partially real; park; no threshold archaeology
```

**Power statement (written now, before any run):** a slice of n≈500 trades
with per-trade excess SD ≈ 13% has a detection floor of ≈1.2pp EV at t=2.
If the true edge is smaller, an honest inconclusive FAIL is likely — same
caveat as RC-23. The mitigations: (a) evaluate on ALL OOS events (~15–30k
graphs, not just a slice) for the AUC gates; (b) the miss-cushioning test
has ~7,700 misses as its sample — decent power for a −1pp cushion effect.

---

## 5. Sequence & effort

```text
step 1  data: D1–D3 (inventory + adv20 + taxonomy index)          ~0.5 day
step 2  graph builder + firewall (G0, incl. negative control)     ~2 days
step 3  baselines B0/B1 + B2                                      ~0.5 day
step 4  GNN training, seeds, HP pass, freeze                      ~2–3 days
step 5  evaluation battery (4.1/4.2), findings memo               ~1 day
        total ≈ 6–7 working days; CPU-only
step 6  (only after gates) shadow-book design                     separate
```

---

## 6. Risks (honest)

```text
R1  the market reads across too — sell-side routing is fast in covered
    names. Mitigation: primary slice = low-coverage SP600; the +0.28 slope
    lived exactly there.
R2  overfitting: a 2-layer GAT on ~30k events can memorize. Mitigation:
    B2 ablation, 3 seeds, frozen single HP pass, early stopping.
R3  beat-label trap: optimizing toward the priced serial component.
    Mitigation: economic metrics are the gates; AUC is diagnostic only.
R4  leak in graph construction (correlation edges, outcome horizons).
    Mitigation: G0 is a hard gate; no evaluation before it passes.
R5  miss branch stays blind (peers don't inform misses at all) → EV stuck
    at ~0. This is a real possibility; G3 tests it explicitly, and a clean
    negative there is a publishable result about read-across asymmetry.
R6  SP600 estimate coverage is 88.9% (time known) — the signed-surprise
    payload has holes; disclosed, and the mask handles it.
```

---

## 7. What I need from you

```text
[DECISION] A1  taxonomy choice (§2.3): fine-per-universe + shared sector
            layer (recommended) vs sector-only
[DECISION] A2  market cap: log(adv20) for v1 (recommended) vs 1-day FMP
            pull for true PIT market cap
[DECISION] A3  gates: accept G0–G3 as written (incl. the +0.02 AUC bar and
            the excess-EV CI rule), or amend before registration
[DECISION] A4  sequencing: run steps 1–5 as one program (RC-24) or split
            into RC-24a (graph + firewall + baselines) and RC-24b (GNN)?
```

On your sign-off, this file becomes the RC-24 pre-registration (frozen),
and execution starts at step 1.
