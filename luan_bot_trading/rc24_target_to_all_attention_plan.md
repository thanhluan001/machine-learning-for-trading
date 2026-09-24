# RC-24 Plan — Target-to-All Cross-Attention for Earnings Surprise

**Status: DRAFT FOR REVIEW — not registered; no training or deployment authorized.**

**Architecture decision:** replace the two-layer, similarity-filtered GAT with
one small target-to-all cross-attention block. Each sample contains one target
earnings event and all eligible recent reporters. The same model is trained
across samples; its parameters are frozen at inference.

This document supersedes the architecture in `GNN_idea.md` for RC-24 v1. That
file remains the original proposal. This plan is named
`rc24_target_to_all_attention_plan.md` to reflect the new model. Although the
sample can be drawn as a star graph, the implementation
is a target-conditioned set model, not a multi-hop GNN or a full Transformer.

---

## 0. Research question and evidence boundary

**Hypothesis:** many recent reporting companies, including weakly related ones,
contain incremental information about a target's upcoming earnings. Learned,
target-specific weighting might use that information better than simple averages.

Prior research motivates the question but does not establish a trading edge:

- The existing beat model achieved approximately 0.673 OOS AUC, but the tested
  high-confidence long slice had approximately zero benchmark-excess return.
- The earlier peer feature showed a conditional relationship between peer drift
  and target returns among low-confidence beats (slope approximately 0.28,
  t approximately 2.3). That is not proof of exploitable mispricing or causality.
- Including signed outcomes from both beating and missing reporters broadens
  the information set. It does not guarantee that misses become predictable.
- SP600 and low analyst coverage are research slices, not established locations
  of an edge. The prior conditional finding does not by itself establish a
  low-coverage mechanism.

Public peer reports can already be reflected in prices. Neither surprises nor
attention provide automatic protection against competition. The program must
separately demonstrate incremental prediction and positive net excess expectancy.

---

## 1. Scope

**IN v1**

- SP400 + SP600 target events and eligible reporters, subject to an audited
  historical universe and identity mapping.
- All eligible reporters in the preceding ten trading sessions, anchored to
  the actual prediction/trading cutoff, not the target's release time.
- Signed EPS surprise, observable one-/three-session benchmark-excess reactions,
  availability masks, reporter age, and company features.
- One shared company encoder, one cross-attention block, and a small prediction
  head. The supervised label is beat/miss; economic evaluation is separate.
- Industry/sector relationship and trailing correlation as attention inputs,
  **not eligibility filters**.
- Research only. Existing live/paper models remain unchanged.

**OUT v1**

- Reporter-to-reporter message passing, multi-hop GAT, or everyone-to-everyone
  self-attention; learned top-k graph construction.
- Ticker-ID embeddings, separately trained stock-pair weights, per-event models,
  or inference-time fine-tuning.
- Supply-chain data, options, transcripts, valuation expansion, and new vendors.
- Short selling or changing the trading/holding policy to rescue a result.

---

## 2. Data plan — what creates the initial embeddings?

### 2.1 Inventory leads and required verification

The following are starting points from the existing research pipeline, not a
newly verified schema or coverage guarantee. Phase 0 must produce a field-level
inventory: exact key, column, units, identity, availability timestamp, coverage
by year/universe, and missing-data treatment.

| Input | Existing source to inspect | Required check |
|---|---|---|
| Prices and volume | SP400 price tables in `db.h5`; `/sp600/{permaTicker}` in `db_sp600.h5` | Actual SP400 keys, OHLCV fields, adjustment conventions, delistings, bar availability |
| Benchmarks | `/macros/IJH`; `/sp600/benchmark_IJR` | Coverage and identical stock/benchmark return windows |
| Event features | v6c/v7c research matrices and their builders | Exact 22-feature contract, feature cutoffs, label maturity, dates |
| SP400 beat/outcomes | v6c SUE join and underlying earnings tables | Raw actual/estimate availability and whether consensus was known before release |
| SP600 beat/outcomes | `/sp600/earnings_full/{ticker}` | Actual, estimate, report_date, bmo/amc, identity joins, timestamp uncertainty |
| Company identity | Existing permaTicker maps, including `/metadata/sp600_ptmap` | Historical ticker changes and membership; avoid current-survivor selection |
| Industry/sector | Existing SIC, GICS, and sector mappings | Exact fields, taxonomy differences, effective dates |
| Clock helpers | `04_backtest/rc18_p0_firewall.py` under this project | Reuse only after confirming its clocks match the new sample contract |

A sector-ETF label is a sector proxy, **not evidence of actual ETF co-ownership**.
Current metadata is not automatically point-in-time. Historical classifications
must have effective dates; unavailable history must be omitted or explicitly
isolated in a sensitivity analysis, not silently assumed correct.

### 2.2 Continuous company features

Start with the existing 22-feature contract after audit rather than inventing a
new feature list. Candidate blocks already discussed include:

- Relative momentum and pre-event volatility.
- Lagged earnings surprises and consecutive-surprise history.
- Analyst coverage, grades/revision proxies, and dispersion where available.
- Macro context already supported by the pipeline.
- Optional log average dollar volume computed from existing price/volume bars.

**Do not assume grades-based revision proxies are timestamped EPS estimate
revisions.** Preserve the source's actual meaning and availability rules.

For dollar volume, use compatible price/volume units and corporate-action
conventions. Missing or invalid volume means a missing liquidity feature plus
its mask—never substitute price alone and call it dollar volume. Liquidity is
not market capitalization.

**Snapshot contract:** target features are frozen at its prediction cutoff.
Reporter company features use an audited pre-release snapshot for that reporter;
its subsequently disclosed result and observable reaction enter through the
separate outcome block. All reporter fields must also be available by the target
cutoff. Do not accidentally reuse target label columns or the reporter's latest
future feature row.

Fit imputers, scaling, clipping bounds, and any feature selection only within
the training partition. Apply frozen transformations to validation/test samples.
Provide numerical missingness indicators alongside imputed values.

### 2.3 Categorical features and learned embeddings

Proposed inputs:

- Shared coarse sector category.
- Fine industry category, namespaced by taxonomy: SIC and GICS codes are not
  interchangeable, even if their numeric/string values happen to match.
- An explicit unknown category for missing or unseen classifications.

Build vocabulary and rare-category grouping from training data only. Recommended
starting design: fine-industry embedding of 16 dimensions and sector embedding
of 8 dimensions, concatenated with the numerical projection. Final vocabulary
sizes come from the inventory, not assumed counts.

An `nn.Embedding` here is a trainable lookup table for **categories**, not one
permanent vector for each stock. The full company representation is recomputed
from its features at each event. No ticker lookup is used in v1.

**[DECISION A1]** Fine-industry plus sector, where historical coverage supports
it, versus sector-only for the first run. Gemini's proposed >95% coverage rule
is an operational suggestion, not an accepted gate. Measure historically valid,
as-of classifications separately by universe and period; populated current-day
fields do not count as point-in-time coverage. Freeze any threshold, denominator,
and fallback policy after inventory but before outcome evaluation. If fine
industry history is inadequate, prefer sector-only, provided the sector history
itself passes the availability audit.

### 2.4 Relationship features — preferences, not connection filters

For every reporter j and target i, supply:

| Feature | Construction | Constraint |
|---|---|---|
| Same sector | Compare valid coarse categories | Unknown is not a confirmed match |
| Same fine industry | Compare categories within the same taxonomy | Include comparability/availability flags |
| Signed return correlation | Trailing 120-session correlation at the latest completed monthly checkpoint before the cutoff | Keep the sign; use only available prices; report observation count and missingness |
| Information age | Trading-session age since public release, measured at target prediction cutoff | Normalize using a fixed scale such as age/10; keep precise timestamps for eligibility |

No `|rho| >= 0.35` filter and no top-10 restriction in the primary model.
Missing industry/correlation does not exclude an otherwise eligible reporter.
Historical earnings-reaction similarity is a possible later feature, not assumed
to be supplied by ordinary return correlation.

### 2.5 Reporter outcomes and the two clocks

Include beats, misses, and ties; freeze label/tie conventions before training.
Each outcome component has its own availability timestamp and mask.

- Signed EPS surprise: `(actual - estimate) / abs(estimate)` only when the
  denominator is usable. Freeze near-zero handling and training-only clipping.
  Do not mix standardized SUE and percentage surprise as though they share units;
  use separate channels/masks if both are needed.
- One-session excess reaction `r1`, only after its complete price window is
  observable. It is **not** immediately available at announcement time.
- Three-session excess reaction `r3`, likewise masked until complete.
- Component availability flags, plus reporting age.

Sanitize unavailable raw values **before** encoding: zero-filled transformed
values plus explicit masks, never an unavailable value passed through an encoder
and multiplied by zero afterward (`NaN * 0` remains NaN).

Fit each outcome transformation on observed training values only, transform
available components, and fill unavailable transformed components with `0.0`.
For example, an EPS result and r1 may be available while r3 is not:

```text
transformed outcomes: [EPS surprise, r1, 0.0]
component masks:      [           1,  1,   0]
```

Concatenate values and component masks into the reporter encoder; do not multiply
the entire encoded reporter by one `(N, 1)` outcome mask. That would conflate
partial availability with an entirely unavailable reporter. Company features,
age, and other available outcomes must remain usable.

`torch.nan_to_num` is not inherently forbidden or an autograd hazard. The policy
is to sanitize declared missing inputs before learned arithmetic, not repair
nonfinite activations or gradients afterward. Assert finite preprocessed values,
encoder outputs, unmasked attention logits, final logits, loss, and gradients.
Unexpected nonfinite values in fields declared observed are data errors to
investigate, not silently zero-fill.

**Clock 1: information age.** How old is the public earnings report?
**Clock 2: reaction maturity.** Which return windows have completed?

An EPS result may be available while both reaction channels are masked. Older
information is not forced to carry less weight: the model learns age effects.
Earlier three-day versus ten-day return-horizon findings did not establish
information-age decay; those are different questions.

If only dates or uncertain BMO/AMC timing exist, use a conservative documented
availability rule. No simultaneous or ambiguously timed release may sneak into
a pre-entry sample.

### 2.6 Acquisition and preprocessing deliverables

1. Inventory existing fields and source/availability timestamps for both universes.
2. Reconcile event identities and report missingness separately for actuals,
   estimates, release time, industries, and prices. Do not reuse a time-coverage
   percentage as an estimate-coverage percentage.
3. Audit reporter pre-release snapshots; build outcomes with per-field clocks.
4. Build training-only category vocabularies and numerical preprocessing.
5. Compute optional liquidity and historical correlation using existing prices.
6. Save a sample manifest with target cutoff, reporter IDs, source timestamps,
   masks, preprocessing/checkpoint versions, and exclusions.

V1 aims to use existing data, but the inventory may reduce scope or block fields.
No new subscription or API pull is authorized by this plan.

**[DECISION A2]** Omit market cap for v1; optionally use audited log dollar volume
as liquidity. If historical market cap is later acquired, require shares
outstanding and publication dates. Weighted-average diluted shares are not an
exact point-in-time shares-outstanding series; current profiles cannot backfill
historical cap. Revenue surprise is deferred unless both universes have audited
historical actual/consensus fields.

---

## 3. Model architecture — beginner-friendly specification

### 3.1 One event = one separate target-to-all sample

Let `c_i` be the actual prediction cutoff for target event i. Construct a sample
from the target plus all eligible other-company reports publicly available in
the preceding ten trading sessions. Enforce `available_at < c_i` with conservative
handling of boundary timestamps. Freeze duplicate/restatement handling in Phase 0.

```text
Reporter A ──┐
Reporter B ──┤
Reporter C ──┼──► ONE TARGET ──► Beat probability
Reporter D ──┤
Reporter E ──┘
```

All eligible reporters have a direct potential influence, regardless of sector
or similarity. There are no reporter-to-reporter edges or indirect passes.
With N reporters, each attention head calculates N scores, not N-squared scores.
Never truncate the set by realized outcomes; any memory cap would require a
predeclared, outcome-independent sampling policy and coverage audit.

### 3.2 What is learned across samples?

We learn a **shared recipe for producing representations and predictions**:

1. A company encoder represents each target/reporter from its available features.
2. An outcome encoder represents a reporter's disclosed result, observable
   reactions, masks, and age.
3. Attention learns which reporter information matters for the current target.
4. A prediction head combines that information with the target's own features.

One event supplies one supervised target label. Training batches contain many
separate event samples. Gradient updates improve the same parameters across all
samples; there is no separately fine-tuned model or pair-weight table per stock.
A company can have a different representation next quarter because its inputs
change. A new company can be represented without having a trained ticker ID.

### 3.3 Proposed compact architecture

```text
SHARED COMPANY ENCODER (target and reporters)
  Fine industry ID ─► Embedding(16) ─┐
  Sector ID ────────► Embedding(8) ──┼─► Concatenate ─► Linear ─► LayerNorm
  Numerical values + masks ─►       │                         │
       Linear(32) + LayerNorm + ReLU┘                    company h (32)

TARGET PATH
  Target h_i ─► query projection ─► q_i: "What information matters to me?"

REPORTER PATH (same encoder weights for every reporter)
  h_j + signed outcomes + outcome masks + age
      ─► Small MLP ─► reporter z_j (32)
      ├─► key projection ─► k_j: "When am I relevant?"
      └─► value projection ─► v_j: "What information do I carry?"

PAIR CONTEXT
  Same-sector / same-industry / signed correlation / age + masks
      ─► Small shared bias network ─► attention-score adjustment

ONE CROSS-ATTENTION BLOCK (2 heads, 16 dimensions per head)
  q_i attends to every eligible reporter's k_j
      ─► weighted sums of v_j ─► concatenate heads ─► context c_i (32)

PREDICTION HEAD
  [target h_i | context c_i | counts and agreement summaries]
      ─► Linear(32) + ReLU + Dropout(0.1) ─► Linear(1) ─► logit
      ─► sigmoid for reported beat probability
```

These dimensions are proposed starting settings, not tuned results. Count actual
parameters after implementing the audited input schema. Do not rely on the old
GAT parameter estimate or assume a small network cannot memorize data.

### 3.4 Attention, without the jargon

For each head, the model computes:

```text
score_ij = dot(q_i, k_j) / sqrt(head_dimension) + bias(pair_features_ij)
weight_ij = softmax_j(score_ij)                 # valid reporters only
context_i = sum_j(weight_ij * v_j)
```

Queries, keys, and values are learned projections, using the same mechanism as
Transformer attention. But this is **one target reading a set**, not a deep
Transformer processing every pair of stocks.

The shared bias network can prefer a recent related reporter without requiring
such a relationship. Attention may also recognize useful cross-sector reporters.
Weights describe model routing, not proven causal influence or reliable feature
attributions by themselves.

**Many weak signals:** softmax-normalized pooling forms a weighted average;
ten agreeing reporters can look like two identical agreeing reporters. This is
not general "scale invariance of softmax": multiplying logits changes attention
weights. Rather, replicating the complete reporter set can leave the pooled
context unchanged because the weights renormalize. Therefore give the prediction
head explicit `log1p(reporter_count)`, observed-outcome counts, beat/miss/tie
fractions, and signed-surprise mean/dispersion with availability flags. These
summaries use the same observable pool and also go to simple baselines. More
reporters are not assumed to be independent evidence.

### 3.5 Missing reporters, masking, and batching

- Batch as padded reporter sets, with a boolean padding mask—not PyG disjoint
  graphs. Padded tokens must contribute neither scores nor values.
- If the eligible set is empty, bypass attention and return a zero context with
  a no-reporters flag. Do not softmax an all-masked row.
- Retain the target's own representation in the head so prediction need not
  depend on reporter information. This fallback is a neural target-only path,
  not identical to the existing XGBoost model.
- Missing r3 must not suppress an available EPS surprise or r1. Masks are per
  component, independent of the padding/eligibility mask.
- Reporter order must not change the prediction: no arbitrary list-position
  embedding; time enters through age features.

#### 3.5.1 Explicit tensor and mask contract

Use `B` = batch size, `N` = padded reporter count, `H` = attention heads,
`D` = dimensions per head, `F` = outcome components, and `P` = pair features.
Boolean mask convention throughout this model: **True means valid/available**.
An API adapter must explicitly translate this if a library uses the opposite
convention (for example, PyTorch `key_padding_mask` uses True for padding).

| Tensor | Shape | Meaning |
|---|---|---|
| Outcome values | `[B, N, F]` | Finite transformed values, unavailable components zero-filled |
| Outcome mask | `[B, N, F]` | Separate availability of EPS, r1, r3, etc. |
| Reporter-valid mask | `[B, N]` | True for eligible real reporters; False for padding |
| Query | `[B, H, 1, D]` | One target per event, split into heads |
| Keys and values | `[B, H, N, D]` | Shared reporter projections split into heads |
| Pair features | `[B, N, P]` | Finite relationship values plus their masks and age |
| Pair bias | `[B, H, 1, N]` | Shared pair MLP outputs H biases per reporter, explicitly rearranged |
| Attention weights | `[B, H, 1, N]` | Normalized across valid reporters only |
| Context | `[B, H * D]` | Concatenated weighted values, zero for an empty set |

The pair MLP operates on each pair independently; it has shared parameters,
not a table of stock-pair weights. Its H outputs permit a different relationship
preference per head. Add its bias **before** softmax. Do not rely on implicit
broadcasting of a `[B, N]` tensor against multi-head scores.

#### 3.5.2 Reference attention core (illustrative, not implemented here)

The following assumes finite encoded inputs, finite zero-filled padded inputs,
and correctly shaped projections. It excludes all-empty samples from softmax
entirely and uses float32 attention arithmetic for numerical stability.

```python
# q: [B, H, 1, D]; k, v: [B, H, N, D]
# pair_features: [B, N, P]; reporter_valid: bool [B, N], True = valid
B, H, _, D = q.shape
N = k.shape[2]
assert reporter_valid.dtype == torch.bool
assert reporter_valid.shape == (B, N)
assert k.shape == v.shape == (B, H, N, D)

weights = q.new_zeros((B, H, 1, N), dtype=torch.float32)
context_heads = q.new_zeros((B, H, 1, D), dtype=torch.float32)
nonempty = reporter_valid.any(dim=-1)  # [B]

if nonempty.any():
    qn = q[nonempty].float()
    kn = k[nonempty].float()
    vn = v[nonempty].float()
    # pair_mlp: [..., P] -> [..., H]
    bias = pair_mlp(pair_features[nonempty]).float()  # [B_nonempty, N, H]
    bias = bias.permute(0, 2, 1).unsqueeze(2)        # [B_nonempty, H, 1, N]
    scores = (qn @ kn.transpose(-1, -2)) / (D ** 0.5) + bias
    assert torch.isfinite(scores).all()
    valid = reporter_valid[nonempty, None, None, :]  # [B_nonempty, 1, 1, N]
    scores = scores.masked_fill(~valid, float('-inf'))
    wn = torch.softmax(scores, dim=-1)
    weights[nonempty] = wn
    context_heads[nonempty] = wn @ vn

context = context_heads.squeeze(2).reshape(B, H * D)
assert torch.isfinite(context).all()
# Concatenate context with the target representation and valid-set summaries.
# Preserve the no-reporters flag; do not add a context-only bias to empty rows.
```

With no valid reporters, even filling every logit with `-1e9` can produce
uniform rather than zero weights; all `-inf` logits can produce NaNs. Neither
is an empty-set solution. The explicit bypass handles both mixed batches and
`N = 0` batches. Masking scores also does not repair NaNs in value vectors:
`0 * NaN` can still contaminate a matrix product, so padded values must be finite.

This is a contract sketch, not permission to skip input/shape validation. If
mixed precision is later introduced, explicitly disable autocast around the
float32 attention computation and rerun the numerical tests.

#### 3.5.3 Required implementation tests

- Mixed empty/nonempty batches and an all-empty batch, including `N = 0`:
  finite logits/loss, zero empty contexts/weights, and a working target path.
- Partial outcome availability: absent r3 does not erase observed EPS/r1;
  perturbations to masked raw fields before preprocessing do not affect output.
- Padding invariance: add padding or change finite padded payloads; prediction
  remains unchanged. Padded reporters are excluded from counts/summaries too.
- Reporter permutation invariance: reorder reporter, pair, and mask tensors
  together; prediction stays equal within the declared numerical tolerance.
- Valid nonempty attention weights sum to one per head; padding weights are zero.
- Explicit shape checks with batch size different from head count and reporter
  count, plus a single-reporter case, to expose accidental broadcasting.
- Backpropagation through nonempty samples produces finite gradients; empty
  samples train the target/head path without requiring reporter-branch gradients.
- Unexpected nonfinite observed inputs/activations fail validation instead of
  being hidden by a late `nan_to_num` call.

Run deterministic invariance comparisons in evaluation mode with dropout off.
These tests supplement, not replace, the source-timestamp firewall in G0.

### 3.6 Training versus inference

**Training:** predict the target's beat/miss label using binary cross-entropy
with logits. Backpropagate the error through the head, attention, projections,
company/category encoders, and outcome encoder. The objective is prediction;
embeddings are learned as a means to that objective, not a separate unsupervised
target. Signed inputs do not turn the binary head into a surprise-magnitude or
return predictor.

**Inference:** freeze all parameters and preprocessing; switch to evaluation
mode (dropout off). Build a new sample at its cutoff, compute fresh embeddings
and attention weights, aggregate the available information, and predict. Input-
dependent attention weights change; learned model parameters do not. No labels,
backpropagation, or fine-tuning are used during inference.

### 3.7 Training protocol and overfitting controls

- Retain the existing outer chronological folds and purge training labels that
  are not mature before each relevant boundary. Use the economic label-end
  discipline for comparability; validate exact split dates in Phase 0.
- Within each outer training history, reserve chronological validation and
  calibration portions with the same availability/maturity discipline. Outer
  test data never selects epochs, hyperparameters, thresholds, or calibration.
- Proposed fixed configuration: hidden 32, two heads, one block, dropout 0.1;
  AdamW, learning rate 1e-3, weight decay 1e-4, maximum 100 epochs, early stopping
  on validation log loss with patience 10. No architecture sweep in v1.
- Three fixed training seeds; report all and use a predeclared probability
  average, not the best seed. Freeze seed values at registration.
- Proposed batch size 64 event samples; lower only for memory without altering
  reporter eligibility. Use unweighted BCE; do not adapt class weighting after
  seeing economic outcomes.
- Report raw probabilities; if calibration is used, freeze a logistic calibration
  procedure on the separate historical calibration portion. Calibration cannot
  manufacture discrimination or economic edge.
- Native PyTorch is sufficient; PyTorch Geometric is not required. Plan a separate
  version-pinned research environment rather than modifying live dependencies.
  Record numerical-library compatibility and reproducibility checks.

Historical events overlap in reporter sets and economic conditions. More edges
are not more labeled samples; small parameter counts do not guarantee sufficient
data. Report training/validation gaps and seed variability.

### 3.8 Baselines and ablations

All comparisons use the same target events, cutoffs, labels, folds, and available
information; missingness exclusions must not silently improve one model's sample.

| ID | Model | Question |
|---|---|---|
| B0 | Existing XGBoost target-feature model | Does the expanded model improve on individual-company prediction? |
| B1 | XGBoost plus existing F1/A2 features | Does it improve on the earlier peer stack? |
| B1+ | XGBoost plus simple aggregates of the same signed reporter payloads, counts, age bins, sector and correlation summaries | Does learned weighting add value over inexpensive aggregation? |
| B2 | Target-only neural model | Is the reporter branch useful beyond the neural target encoder? |
| B3 | Cross-attention trained without outcome information | Do signed reported outcomes add information beyond reporter descriptors? |
| B4 | Same encoders/head with uniform reporter averaging | Is target-conditioned weighting better than an unweighted summary? |

For B3, remove outcomes **and all outcome-derived summaries**; retain ordinary
reporter-count and availability information. Train ablations independently rather
than zeroing inputs only at test time.

**Diagnostic:** permute complete outcome payloads among compatible reporters
within the same cutoff, respecting availability patterns; recompute dependent
summaries. This tests reporter/outcome alignment. It preserves aggregate season
information, so performance need not collapse to B2/B3. Persistence is not by
itself evidence of leakage. Separately permuting reporter order should leave
predictions unchanged and is an implementation test.

Inspect attention alongside leave-one-reporter-out sensitivity. Neither alone
establishes an economic mechanism. Sparse GAT can be a later registered comparison;
it is not a second primary model or an implicit architecture search in v1.

---

## 4. Evaluation and draft gates

### 4.1 Prediction quality

Report OOS ROC-AUC, log loss, Brier score, calibration, and miss precision/recall
at thresholds selected without outer-test data. Include overall results, each
universe, and training-defined coverage strata.

**Correction:** `AUC(1-y, 1-p)` equals `AUC(y, p)`. A separate "miss-side AUC"
is not independent evidence and is not a gate. Assess miss identification through
precision/recall and the payoff decomposition instead.

The primary population and coverage cutoffs must be selected before execution.
Low-coverage SP600 remains a proposed diagnostic slice, not a proven location of
the earlier effect. Report counts and comparisons on matched samples.

### 4.2 Economic quality

For fixed diagnostic probability bins and a trading rule frozen using training/
validation/calibration data, report:

```text
n, P(beat), E[excess | beat], E[excess | non-beat], total mean excess,
transaction costs, and uncertainty

EV = p * gain + (1-p) * loss - costs
breakeven p = (costs - loss) / (gain - loss), when gain > loss
```

There is no universal 60% hit-rate gate: conditional payoffs change with the
selected population. A better miss-avoidance signal can help even if the return
conditional on a miss is unchanged. A binary classifier does not directly learn
miss-loss severity; any cushioning analysis is secondary, not guaranteed.

Use benchmark-excess returns (IJH for SP400, IJR for SP600) with identical stock
and benchmark windows. The intended hold remains **five sessions**, not the
separate ten-session label horizon; preserve and disclose the audited BMO/AMC
entry/exit conventions. For stopped trades, the benchmark window ends at the
actual exit as well. Report costs explicitly.

Portfolio evaluation uses causal continuous slot allocation, four slots, and no
ranking against future candidates from the rest of the week. Selection is per
fold before concatenation. Freeze entry, exit, costs, threshold/selection rule,
and overlap policy before evaluating test outcomes.

Use week-block bootstrap, 10,000 draws, seed 20260807; use paired resampling for
model comparisons. Dependence across weeks/sectors may require additional
predeclared sensitivity checks. Do not treat attention edges as observations.

### 4.3 Gates — proposals, not registered pass conditions

- **G0 / integrity:** pass the sample and model firewall before outcome evaluation.
  Rebuild at least 100 representative cutoffs from truncated source data; compare
  sample membership, features, outcomes/masks, historical correlations, and frozen
  predictions (declared numerical tolerance). Cover BMO/AMC, simultaneous releases,
  missing timestamps, empty sets, and immature reactions. A deliberate future-data
  injection must be detected. The tensor/mask, empty-set, padding, permutation,
  masked-value perturbation, and numerical/backpropagation tests in §3.5 must
  also pass. Passing these implementation tests does not itself prove the source
  data was point-in-time.
- **G1 / incremental prediction:** proposed practical target is OOS AUC at least
  0.02 above B1+, paired uncertainty reported, with a predeclared outcome ablation
  and uniform-aggregation comparison supporting the claimed improvement. Freeze
  the primary population and exact joint pass rule before execution. Failure closes
  this specification; it does not refute all possible peer-information mechanisms.
- **G2 / economics:** the single predeclared policy has positive mean net excess
  return with its week-block confidence interval excluding zero. Also report the
  paired result against the baseline. No post-hoc slice search or threshold rescue.
- **Miss diagnostics (not G3):** report miss precision/recall and conditional
  payoffs. The old separate miss-AUC gate is removed as mathematically redundant.

G0/G1/G2 passing permits only a separate shadow-book proposal, not live deployment.
A G1 pass with G2 failure is predictive evidence without a certified trade.

**Power:** the old approximation of 500 independent observations with 13% return
SD gives roughly a 1.2 percentage-point two-standard-error scale, not a reliable
power guarantee. Week dependence and selected-slice sizes can substantially worsen
it. Compute sample counts and a pre-run power assessment for the chosen policy;
do not promise thousands of usable OOS misses from total matrix row counts.

---

## 5. Execution sequence (after registration)

1. **Inventory and clocks:** verify sources, labels, historical universe, taxonomy,
   and reporter/target snapshots; freeze schema and exclusions.
2. **Sample builder and firewall:** implement reporter sets, pair context, masks,
   counts, training-only preprocessing, and G0 tests. Save sample manifests.
3. **Baselines:** reproduce B0/B1 and build same-information B1+; establish common
   event sets and split artifacts.
4. **Compact cross-attention model:** implement the shared encoders, one attention
   block, empty-set handling, head, and frozen inference path; count parameters.
5. **Training and ablations:** use the fixed settings/seeds and nested chronological
   protocol; save every model/preprocessor version and selection decision.
6. **Locked evaluation:** run prediction, payoff, and causal portfolio evaluations;
   publish all baseline/ablation results and the cause of any closure.
7. **Only if gates pass:** propose independent forward shadow evaluation.

Re-estimate work after the inventory. The previous fixed six-to-seven-day estimate
was provisional; data availability and temporal reconstruction may dominate effort.
No dependency installation, data purchase, training, or live-system change occurs
merely because this draft is updated.

---

## 6. Main risks and explicit corrections to the earlier draft

- Public read-across can already be priced. Conditional correlations do not prove
  that this architecture will deliver a tradable edge.
- All-to-target attention allows more irrelevant candidates; it is not a free
  improvement over sparse GAT. Shared parameters and strict validation reduce,
  but do not remove, overfitting risk.
- Temporal availability at the prediction cutoff is the firewall. Chronological
  message arrows are not themselves proof of causality; v1 has no peer-to-peer
  messages to order.
- Return-horizon comparisons do not establish freshness decay. Age is an input
  whose usefulness must be tested.
- Weighted averaging alone does not measure the amount of corroborating evidence;
  counts/dispersion are supplied explicitly and can themselves be noisy.
- Broader signed inputs can help miss avoidance; they do not guarantee smaller
  losses conditional on a miss. The estimated breakeven is slice-specific.
- Historical classifications, analyst snapshots, and consensus revision timestamps
  may be inadequate. Missing point-in-time evidence limits claims even if a
  current-data reconstruction looks predictive.

---

## 7. Decisions remaining before registration

**Architecture agreed:** target-to-all cross-attention, shared learned encoders,
one target label per event, and frozen parameters at inference. No GAT/PyG primary
model; no everyone-to-everyone graph.

- **A1:** audited fine-industry + sector versus sector-only categories.
- **A2:** optional liquidity feature; defer market cap/new data acquisition.
- **A3:** primary population, exact G1 comparison/uncertainty rule, calibration and
  economic policy/costs, seed values, and numeric firewall tolerances. These must
  be written explicitly before registration; an unspecified "some slice" is not
  a valid gate.
- **A4:** authorize inventory/firewall only first (recommended), or register the
  full phased program with a hard stop if the inventory fails.

Review/sign-off should lead to a completed frozen registration—not automatically
convert unresolved draft choices into an executable research program.
