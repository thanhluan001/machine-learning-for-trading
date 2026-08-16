# Phase B Price-Data Contamination Audit

> **⛔ RESOLUTION NOTICE (2026-07-14).** ALL three bug classes documented below
> (Class U/V/W/S) are natively eliminated by switching the primary identity
> key from synthetic `perm_id` to Tiingo `permaTicker`. The Phase B alias-
> concatenation machinery (`fetch_concatenated_aliases_from_eodhd`) is being
> **removed entirely**. Per the audit results in
> [`01_data/tiingo_permaTicker_audit.md`](tiingo_permaTicker_audit.md):
> - Class U/V (alias-intertwining): permaTicker-keyed single `/prices` fetch
>   per canonical returns the full rebrand-covered history with zero alias-
>   concat or non-stable sort involved.
> - Class W (NSR retroactive rewiring): permaTicker separates Neustar
>   perma `US000000006945` from Nomad-Royalty perma `US000000062716`. Fetch by
>   Neustar's perma cleanly returns the 2015 Neustar prices (verified).
> - Class S (SUNE confusion): permaTicker separates SunEdison perma
>   `US000000002709` (delisted) from Sunation perma `US000000002062` (active).
>   Both histories are cleanly fetchable separately.
>
> The original remediation plan documented in the §”Original TL;DR“ section
> (EODHD alias-fetch hygiene fixes) is **OBSOLETE**. Read this audit as the
> forensic record of WHY we abandoned the synthetic `perm_id` approach.
> The migration plan is in `tiingo_permaTicker_audit.md`.

**Status**: Discovered 2026-07-14 during Stage 3 v1 audit pass.
**Scope**: `/sp400/{canonical}` price nodes built by `01_data/03_data_gathering.py`
(Phase B v2.1). Affects `train_matrix` features that depend on `stock_Adj_Close`
(i.e. all CAR labels + all Block 2/3 features).

---

## Historical correction (2026-07-14)

After a deeper probe of CHK / EXE I discovered my characterization was
incorrect: CHK and EXE had OVERLAPPING active trading dates (CHK continued
on OTC post-NYSE-delist through 2024, EXE started trading publicly in
2021). They are NOT a clean rebrand, so the CHK/EXE case shouldn't be
used as evidence for "Tiingo easily solves rebrand merging".

The simple case I should have used from the start is **POL → AVNT**
(PolyOne → Avient 2019, same CIK). Probing EODHD directly:
- `AVNT.US` response starts 2010-01-04 at close=$7.95 (real PolyOne
  pre-rename prices).
- `POL.US` response starts 2018-06 at close=$42 (real PolyOne 2018)  AND
  contains modern-day close=$0.0003 rows around 2024 (a modern unrelated
  $0.0003 POL holder's retro-applied data).

So in this case **EODHD already merged PolyOne's pre-rename history
under AVNT at the database level**. Phase B's fetch of `POL` as an extra
alias was redundant — we'd have had clean data by fetching AVNT alone.
All three of our Class-U cases (ENOV/CFX, LDOS/SAI, AVNT/POL) follow the
same pattern: EODHD's modern canonical_ticker code already contains
the full pre-rebrand history back-merged.

**The actual root cause of all our contamination: Phase B's
`fetch_concatenated_aliases_from_eodhd` needlessly fetched the LEGACY
alias ticker, and that legacy ticker symbol has been re-issued to a
DIFFERENT modern security, allowing contamination into our concat.
Canceling that legacy fetch eliminates Class U + Class V symptoms.
Going forward we should fetch ONLY canonical_ticker, never aliases.
**(This applies whether we use EODHD or Tiingo.)

Class W (NSR) is the only one that genuinely needs a different solution
because Neustar got taken private and its ticker was never inherited by
a modern canonical holder — the modern NSR belongs to Nomad Royalty
and Tiingo cleanly separates those entities (confirmed: Tiingo NSR
metadata `name=Nomad Royalty`, `startDate=2021-06-03`; 2015 NSR prices
return zero rows), but Tiingo doesn't recover Neustar's pre-2017
history under any ticker symbol — Tiingo just "doesn't have it".

For both EODHD and Tiingo, the path forward looks identical:
  1. Fetch only canonical_ticker.
  2. Add a `name`-sanity check against the expected perm_id name to catch
     Class W cases (NSR / SUNE / other).

See `04_backtest/archive/docs/eodhd_vs_tiingo.md` (archived) for the full provider-comparison review.

---

## Original TL;DR (kept for tracing decisions made earlier)

Of the 958 canonical-ticker price nodes in `db.h5` (table below retained
for cultural context; the corrected understanding above supersedes
several of its claims):

Of the 958 canonical-ticker price nodes in `db.h5`:

| Bug class | # Nodes | Responsible party | Severity |
|---|---|---|---|
| **Class U** — Phase-B alias range contamination. Phase B fetches every alias over the FULL 15y window, then non-stable sort + drop_duplicates(keep='last') randomly picks an alias's adjusted_close per date. Causes (a) different-company ticker-recycle contamination (LDOS polluted by modern Saia under SAI.US) and (b) same-company different-adjustment-chain contamination (ENOV polluted by CFX's split-factored adjusted_close) | 3 confirmed + 5 suspicious | **OUR BUG** (Phase B fetch + pandas sort_values default quicksort) | HIGH |
| **Class V (was thought to be EODHD adj_close bug — actually a Class U manifestation)** — Visible pattern: Adj_Close alternates wildly between two regimes on consecutive days though Close is stable. Hidden root cause: same as Class U — two aliases with different adjustment chains random-picked by non-stable sort_values | 5 nodes (ENOV, COHR, AVNT, VIAV, HR) with >=5 spike rows | **OUR BUG (same as U)** | HIGH |
| **Class W** — EODHD ticker-code aliases distinct company periods under same ticker (retroactively mismaps modern security's price to legacy date) | 1 confirmed (NSR). Tiingo has same problem for NSR. | **EODHD upstream** (partly; for NSR specifically, no easy external fallback) | MEDIUM/HIGH |

A Stage-2 outlier clip is **NOT** a sufficient fix because Class U/V
contaminates all features (volatility, range, momentum) at contaminated
dates, not just the CAR label. A 3500% CAR event is only the visible
extremity; the malignancy is silent across all contaminated-but-not-
extreme rows.

**Tiingo swap resolves Class U/V natively** (point-in-time ticker
resolution; legacy aliases return zero pre-rebrand data under old ticker;
modern ticker contains the full consolidated history). Tiingo swap
DOES NOT resolve Class W (NSR) — Neustar is simply missing from
both databases; no ticker in either provider gives us pre-2017
Neustar NSR prices.

**For all 3 verified Class-U cases** (ENOV, LDOS, AVNT) the same fix
applies: drop the legacy alias fetch and use only the canonical ticker.
Both EODHD and Tiingo give us clean full-history under canonical_ticker
alone. No alias-merging needed.

---

## Discovery path

1. Trained Stage 3 `XGBRanker (rank:ndcg)` (Phase F v1); got
   `TRAIN NDCG@3 = 0.776` vs `VAL NDCG@3 = 0.086` — huge train/val gap.

2. Per-year training/NDCG analysis showed TRAIN NDCG was uniformly high
   (0.69 - 0.87) for 2015-2023 but VAL NDCG collapsed to 0.34/0.31/-0.79
   for 2024/2025/2026 — characteristic of memorizing a small number of
   "winner knows winner" outlier events in TRAIN.

3. `|arith CAR| > 50%` counted 88 train rows; top-15 positive and negative
   outliers showed up to +363,862% and -98.7% CAR — physically impossible
   for S&P 400 mid-caps.

4. Tracing the +363,862% (`NSR 2015-11-26`) and -98.7% (`LDOS 2023-02-14`)
   events back to raw `/sp400/{TICKER}` rows revealed the contamination
   patterns documented below.

---

## Class U — Phase-B alias concatenation contamination

### Root cause

`01_data/03_data_gathering.py` function `fetch_concatenated_aliases_from_eodhd()`
fetches EVERY alias ticker in `combined_aliases` over the SAME full
15-year window (`START_DATE = 2011-...`, `END_DATE = yesterday`), then
`pd.concat(..., sort_values("Date"), drop_duplicates(Date, keep="last"))`.

**The problem**: when a perm_id's `combined_aliases` includes a ticker
code that was RELINQUISHED and later RECYCLED to a different company by
the exchange (e.g., `SAI` = Science Applications Intl -> freed -> reassigned
to `Saia Inc.` post 2021), EODHD's modern `SAI.US` response now points
to the new company's history. Phase B's concat merges this unrelated
modern price data into the perm_id's price node.

The `per_ticker_intervals` field in `/metadata/sp400_perm_ids` tracks when
each alias was held by the perm_id as an **S&P 400 constituent**, not as a
US-exchange-listed ticker. These are different concepts:
"SAI was in S&P 400 from 2013-09-20" tells us nothing about when
`SAI.US` (the ticker code) was actively traded by Leidos, vs. today when
`SAI.US` is Saia Inc.

### Concrete evidence — `LDOS` (perm_id `0001336920_SAI`)

`per_ticker_intervals`:
```
SAI:  [{"added": "2013-09-20", "removed": null}]
LDOS: [{"added": "2019-08-09", "removed": null}]
```

Phase B fetches `SAI.US` from 2011 to yesterday AND `LDOS.US` from 2011
to yesterday, then concatenates. Modern EODHD `SAI.US` is now Saia Inc.
(a ground-logistics company, mid-tier US-listed, current price ~$9.65).
Its history starts ~2021-06. So:

- Pre-2021: `SAI.US` EODHD rows are dead/delisted legacy Leidos prices
  (real Leidos pre-2013 SAI history + EODHD-preserved wide coverage gap
  until ~ 2021).
- Post-2021-06: `SAI.US` EODHD rows = Saia Inc. prices (~$9.65/share).
- Phase B sorts + `drop_duplicates(Date, keep="last")` → on overlapping
  dates 2021-06 onwards, keep="last" randomly interleaves Saia-$9.65
  vs. LDOS-$100 rows.

Result in `/sp400/LDOS`:
- 352 rows with |single-day Close return| > 50% (vs 1-3 for clean mid-caps).
- Contiguous regime analysis shows $9.65 vs $100 alternating on adjacent
  dates for 2021-06-2022 entire period (~87 contaminated weeks).

Equivalently `AVNT` (`0001122976_AVNT`, aliases `["AVNT", "POL"]`) —
POL.US in EODHD currently = a different instrument post de-list.

### Other Class-U suspects (need verification)

Snapback-detector flagged (>=20% of big Close jumps snap back to
recent-previous-level): `AVNT`, `LDOS`, `VVC`.

Verify-but-not-confirm (5-20% snapback): `CPWR`, `SBNY`, `CHRD`, `EXE`,
`SPN`. Each needs manual eye + Tiingo cross-reference.

### Fix location

`01_data/03_data_gathering.py::fetch_concatenated_aliases_from_eodhd()`.
The fix is to add a per-alias date-window parameter derived from a new
field on each alias in `/metadata/sp400_perm_ids` — the SEC EDGAR
active-ticker history (described below), which is a different dataset
than `per_ticker_intervals` (the S&P 400 membership history).

We will need to build a "fetch this alias ONLY between dates X and Y"
capability in Phase B, then call it with the correct per-alias bound.

---

## Class V — (was thought to be "EODHD adjusted_close factor corruption";
actsuelly a HIDDEN Class U manifestation)

### Original diagnosis — WRONG

Initial suspicion: EODHD's `adjusted_close` is sometimes computed with a
wrong dividend/split adjustment factor, causing Adj_Close to alternate
between two regimes on consecutive days where Close is identical.

### Actual root cause — confirmed 2026-07-14

Direct EODHD probe for `ENOV.US` in Aug-Sep 2011 returns CLEAN data: every
row's Adj_Close is `Close * constant_factor`, with normal daily returns.
But our `/sp400/ENOV` shows Adj_Close alternation on those same dates.

The discrepancy is **the per-alias adjusted_close divergence** between
the two aliases of the same underlying security, COMBINED with
non-stable-sort in Phase B's dedup step:

1. ENOV's `aliases = ["ENOV", "CFX"]` (Colfax renamed to Enovis in 2022;
EODHD keeps the legacy CFX.US archive entombed through the 2022-04-05
event while the modern ENOV.US series starts fresh).

2. The 2022-04-05 1:3 split is encoded in the **legacy** `CFX.US`
adjusted_close series (so CFX pre-2022 adjusted_close ~ $27.44 × 3.0 =
$82.32) but NOT in the **modern** `ENOV.US` series (so ENOV pre-2022
adjusted_close ~ $27.44 × 1.72 = $47.23, no split factor).

3. Phase B fetches BOTH, concatenates, sorts by Date, then
`drop_duplicates(subset="Date", keep="last")`.

4. **`pandas.DataFrame.sort_values("Date")` default `kind="quicksort"`
is NON-STABLE** — within ties, the order is randomized per
instantiation. Then `keep="last"` picks one alias's adjusted_close
randomly per date. Result: /sp400/ENOV alternates Adj_Close between the
ENOV-regime (~$42) and CFX-regime (~$76) in no consistent pattern.

### Verified via direct EODHD + Tiingo probes

For 2011-08-02 ENOV/CFX as compared to our stored `/sp400/ENOV` row:
| Source                                            | Adj_Close |
|---------------------------------------------------|-----------|
| EODHD `ENOV.US` raw response                       | 47.2289   |
| EODHD `CFX.US` raw response                        | 82.32     |
| Tiingo `ENOV` daily (clean point-in-time resolver) | 47.228915662 (matches EODHD ENOV) |
| Our `/sp400/ENOV` stored                           | 82.32 (randomly landed on CFX) |

Both ENOV and CFX feeds return IDENTICAL `Open/High/Low/Close/Volume`
(same underlying security Colfax). The divergence is only in
adjusted_close (because the split factor is encoded differently between
EODHD-ID and Tiingo-style fresh-start).

### True bug classification

**Class V is NOT a separate EODHD upstream bug. It is a HIDDEN Phase B
Class U manifestation that happens to produce Adj_Close alternation.**
ENOV, COHR, AVNT, VIAV, HR (the 5 "Class-V" flagged canonicals) all
have legacy-vs-modern alias pairs (rebrand events within the 15-year
window) where EODHD returned different adjusted_close series for the
two aliases even though the underlying raw price was identical.

### Fix options

Same as Class U — the contamination goes away if Phase B does ONE of:
- (a) Uses per-alias date bounds (no overlapping aliases fetched).
- (b) Uses stable sort + canonical-alias preference (drop_duplicates
  with a custom key that prefers canonical_ticker).
- (c) Switch primary data source to Tiingo (which resolves rebrands
  natively via point-in-time ticker resolution — see Tiingo probe
  results in `AB-SwitchData-FB-META` below).

### Verification audit

Phase B bug fix verification will require re-deriving Class-V spike
counts on the rebuilt nodes. The previous "807 spike rows for ENOV"
detection will yield zero if the per-alias date bounds (or stable
sort + canonical-preference) are applied correctly. The audit pass
needs to verify this for COHR/AVNT/VIAV/HR too.

---

## Class W — EODHD ticker-code time-machine (NSR)

### Root cause

EODHD's `NSR.US` symbol database is not point-in-time. It returns
data for different securities under the same ticker code without
separating them. Specifically:

- `NSR` was the ticker for **Neustar Inc.** until ~2017 (take-private
  deal completed 2017-circa). Real Neustar traded $20-35 mid-cap range.
- `NSR` was reassigned to `Nomad Royalty Company Ltd` (an NYSE-listed
  gold-streaming royalty vehicle), which IPO'd in June 2021 and traded
  $5-15 until its 2022-08 acquisition.

Our `/sp400/NSR` node (1,677 rows, 2015-02-11 .. 2022-08-15) shows:
- Run #1 (2015-02-11 .. 2015-12-30, 11 rows): Close 0.01-0.06 (modern
  Nomad-Royalty data apparently retro-mapped to 2015 timestamps).
- Run #2 (2016-01-04 .. 2021-08-30, 1425 rows): Close $22-50 (real
  Neustar mid-cap prices).
- Run #3 (2021-08-31 .. 2022-08-15, 241 rows): Close $7-15 (real
  Nomad-Royalty prices, in correct time).

EODHD's response itself contains the same artifacts (verified by
direct API call to `https://eodhd.com/api/eod/NSR.US`):
```
2015-04-07   close=0.0399
2015-12-10   close=0.0065
2015-12-30   close=0.011
2016-01-04   close=23.10     <- jump (real Neustar)
2016-01-05   close=22.47
2021-08-30   ...
```

So Phase B faithfully translated EODHD's bug into our node. This is the
1-of-4 outlier cases that is purely downstream from EODHD.

### Tiingo equiv

Tiingo metadata for `NSR`: `{"ticker": "NSR", "name": "Nomad Royalty Company Ltd",
"startDate": "2021-06-03", "endDate": "2022-08-15"}`.
Tiingo does NOT have Neustar in 2015 under NSR (its coverage of the NSR
ticker begins with Nomad Royalty in 2021). This makes sense — Tiingo
applies point-in-time ticker resolution: when a ticker code is freed and
later re-used by a different security, Tiingo's `NSR` only ever points
to the most-recent holder; nothing is back-filled.

Implication: For delisted/reassigned legacy tickers (Neustar, taken
private in 2017, ticker freed → reassigned to Nomad Royalty in 2021),
Tiingo's free tier also doesn't have the legacy history. This means
switching to Tiingo as a sole data source doesn't recover Class-W cases.
We'd need a third historical provider (e.g., Yahoo Finance free) or
mark `NSR` permanently missing.

### Fix location

For NSR specifically, no fix is simple. Options:
- (a) Mark `NSR` perm_id (`0001265888_NSR`) `price_unavailable=True`
  in `/metadata/sp400_perm_ids`, so Phase E Stage 1 silently drops its
  events.
- (b) Backfill NSR from Yahoo Finance / Alpha Vantage free historical.

Until fixed, our 6 NSR earnings events (2015-11-26 = +363,862% CAR) are
wildly misleading the ranker.

---

## Fix prioritization

Suggested execution order (smallest scope → largest scope):

1. **Class W (NSR)** — mark `price_unavailable`, no other code changes.
   Effort: ~5 min code + verify drop count in Stage 1/2.

2. **Class V (5 canonicals)** — investigate each separately:
   - Probe each via EODHD splits+dividends endpoint to see if local
     Adj_Close derivation works.
   - Probe each via Tiingo (paid coverage?) to compare alt adj_close.
   - Or drop the canonical from training set if all alt providers are bad.
   Effort: ~half day probe + 0.5-2 days fix.

3. **Class U (alias contamination)** — re-engineer Phase B's fetch:
   - Build `SEC EDGAR active-ticker` history per perm_id (when each
     ticker code was held by this CIK, by exchange-trade dates).
   - Modify `fetch_concatenated_aliases_from_eodhd` to fetch each alias
     over its per-alias date window ONLY.
   - Re-fetch 3-8 affected canonicals.
   - Verify with snapback audit (~0 expected post-fix).
   Effort: ~1-2 days.

4. After 1-3 complete, re-run Stage 1 → Stage 2 → Stage 3.

---

## Suspect canonicals — detailed table

(updated as audit progresses; see `merger_identity_patch.md` §7.11 commit
cross-reference)

| Canon | Class U | Class V (= Class U) | Class W | Rows | Action pending |
|---|---|---|---|---|---|
| NSR  | — | — (2 spikes from same root) | YES | 1677 | mark price_unavailable |
| LDOS | YES (SAI-modern Saia interleaved) | — | — | 3768 | rebuild after Phase B fix (per-alias date bounds) OR switch to Tiingo |
| AVNT | YES (POL-or-CFX-style alias) + CFX wrong-adj | (was Class V) | — | 3768 | rebuild after Phase B fix OR switch to Tiingo |
| VVC  | maybe (snapback-detector) | — | — | 2146 | verify |
| CPWR | maybe | — | — | 3769 | verify |
| SBNY | maybe | — | — | 3758 | verify (likely silent Class-W) |
| CHRD | maybe | (1 spike from same) | — | 3768 | verify |
| EXE  | maybe | — | — | 2645 | verify (this is the CHK -> Expand Energy canonical!) |
| SPN  | maybe | — | — | 2334 | verify |
| ENOV | YES (CFX-modern alias) | (was Class V; re-classified as U) | — | 3768 | rebuild after Phase B fix OR switch to Tiingo |
| COHR | YES (II-VI legacy alias) | (was Class V; re-classified as U) | — | 3768 | rebuild after Phase B fix OR switch to Tiingo |
| VIAV | YES (likely VIAS-modern alias) | (was Class V; re-classified as U) | — | 3768 | rebuild after Phase B fix OR switch to Tiingo |
| HR   | maybe | (was Class V; re-classified as U) | — | 3768 | rebuild after Phase B fix OR switch to Tiingo |

---

## Open questions for user decision

1. **Data source strategy**: **Tiingo probes show Tiingo resolves Classes U + V natively**
   via point-in-time ticker resolution. User-suggested comparison:
   - **META** (Meta Platforms, formerly Facebook): Tiingo metadata gives full
     Facebook-era history (startDate=2012-05-18, IPO date) under `META`.
     2013 META prices = real Facebook IPO-era quotes (~$25). Verified ✓
   - **FB** (legacy ticker): Tiingo metadata points to a brand-new ProShares
     ETF (startDate=2025-06-26). Asking FB/prices for 2013 returns zero rows
     (Tiingo doesn't backfill reused tickers with unrelated-history data).
     Verified ✓
   - **ENOV**: Tiingo metadata startDate=2008-05-08 — full Colfax + Enovis
     history under one canonical ticker; no alias-merge needed; clean
     adjClose matching EODHD's ENOV.US raw response.
   - The only downside: **Tiingo free tier doesn't have been used; we
     don't have a Tiingo subscription", so a full ~958-ticker re-fetch
     might exceed monthly quota.

2. **Rebuild strategy options** (need user decision):
   - **Option A (switch to Tiingo)**: Rewrite `03_data_gathering.py` to
     fetch from Tiingo with point-in-time ticker resolution.
     No more alias concatenation; no more Class U/V bug fixes by hand.
     Cost: needs Tiingo subscription or careful free-tier pacing.
   - **Option B (fix Phase B in place)**: Add per-alias date bounds to
     `fetch_concatenated_aliases_from_eodhd` AND replace
     `sort_values("Date")` with `sort_values("Date", kind="stable")`
     AND implement custom dedup that prefers canonical_ticker alias on
     overlap. Re-fetch all 958 nodes. Cost: ~1 day of coding.
   - **Option C (hybrid)**: Keep EODHD primary; adopt Tiingo as fallback
     for rebrand-modeled canonicals only (the ~5-10 contaminated nodes).
     Most robust; cost: minimal new Tiingo queries (only for problem
     cases).
   - **Option D (surgical / minimal)**: Fix the quicksort randomness
     alone (use stable sort + canonical-preference dedup); don't do
     per-alias date bounds. This eliminates random alias-vs-alias
     alternation but leaves systematic alias-vs-alias mismatches when
     both contribute over the same window with different adj chains
     (still wrong but at least DETERMINISTIC).

3. **Phase-B re-fetch policy**: how much wall-time is acceptable?
   Re-fetching 1 Class-W canonical = trivial. Re-fetching 3-8 Class-U
   canonicals = ~1 hour. Re-fetching all 958 with a tightened Phase B
   = ~hour. Re-deriving Adj_Close from splits/dividends = ~hour or
   two.

---

## Sort-values stability bug (smoking gun)

The `pandas.DataFrame.sort_values` default `kind="quicksort"` is
NON-STABLE. Within ties (rows with same "Date"), the original concat
order is NOT preserved; quicksort randomly reorders ties.

This was confirmed by direct reproduction:
```
ENOV piece: 44 rows  CFX piece: 44 rows
After concat: 88 rows
After sort: 88 rows
```
On overlapping dates the sort order randomly placed ENOV or CFX first,
and `drop_duplicates(subset="Date", keep="last")` therefore randomly
chose the alias whose adjusted_close survived into `/sp400/ENOV`.

This is the mechanism for ALL Class-V alternations, not just ENOV.

(Pandas demands `kind="stable"` or `kind="mergesort"` for guaranteed
tie-order-preservation. Phase B's current code does NOT specify a
`kind=`.)

A stable sort alone would not entirely fix ENOV's data (the ties would
still need a deterministic alias-preference rule because the two aliases
have DIFFERENT valid adjusted_close series). But it would expose the
underlying semantic ambiguity that the proper Phase-B rewrite must
resolve explicitly.
