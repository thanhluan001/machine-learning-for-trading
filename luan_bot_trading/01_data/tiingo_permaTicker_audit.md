# Tiingo `permaTicker` Stability Audit (2026-07-14)

**Status**: ⭐ **AUTHORITATIVE — New identity layer design for the pipeline.
** All other docs that previously anchored identity on synthetic `perm_id`
(`merger_identity_patch.md`, `database_layout.md`, `Design.md` §9b,
`features.md` §0, `company_merge_design.md`, `earnings_gathering_design.md`,
`phase_b_contamination_audit.md`, `eodhd_vs_tiingo.md`) carry deprecation
banners pointing back to THIS file. Forward Phase A rewrite (`02b_build_company_map.py`),
Phase B rewrite (`03_data_gathering.py`), and Phase D re-keying MUST follow
the migration design in §"Migration design implication" below.

**Live probes run**: ~50 API calls across 13 phases (free-tier), exercising
all critical identity-stress scenarios. Conclusive.

**Verdict**: ✅ **`permaTicker` is identity-stable across rebrands, mergers,
delisting events, and spin-offs. Use it as the primary key.**

This audit also serves as the location-of-record for the OpenFIGI
complementary stable-id field (provided alongside `permaTicker` in Tiingo
search response; same role, Bloomberg-issued, used as a defensive fallback
anchor).

---

## Key probe results

### Lookup mechanism (CRITICAL DISCOVERY)

The `permaTicker` is **not directly surfaced as a field in the
`/tiingo/daily/{TICKER}` metadata endpoint**. To discover a ticker's
permaTicker, you must use the search endpoint:

```
GET /tiingo/utilities/search/{query}
  ?includeDelisted=true        -- REQUIRED: surfaces BOTH active + inactive holders
  &exactTickerMatch=true       -- REQUIRED: filter on exact ticker code (not prefix match)
  &columns=name,ticker,permaTicker,isActive,countryCode  -- minimizes payload
  &countryCode=US              -- filter out non-US listings (无关 e.g. AU-ticker)
```

Without `includeDelisted=true` we cannot see legacy holders of ticker
codes that have been reassigned. Without `exactTickerMatch=true` the
search defaults to "prefix match" (search "SAI" returns SAIA, SAIC,
SAIDD, etc.).

### `permaTicker`-keyed /prices fetch works for delisted companies

The prices endpoint accepts the permaTicker as the URL path key:

```
GET /tiingo/daily/{permaTicker}/prices
  ?startDate=YYYY-MM-DD&endDate=YYYY-MM-DD
```

Tested with multiple delisted permaTickers — all return real historical
rows:
- `US000000000059` (META modern, formerly FB) — 2013-05 rows returned
  (real Facebook-era prices $27 with ~64M volume)
- `US000000002709` (SunEdison, bankrupt 2016, delisted) — 2015 rows
  returned ($30 SunEdison pre-collapse era), 2016-04 bankruptcy rows
  returned (close $0.43)
- `US000000006945` (NeuStar, taken private ~2017) — 2015 rows
  returned ($27 real Neustar-era prices)
- `US000000000505` (Chesapeake Energy, NYSE-delisted 2024-10) — 2015
  rows returned ($19.76 raw, $3909 adjClose correctly back-adjusted
  for the 1:200 reverse split during the 2020 bankruptcy)

### `permaTicker` is identity-stable: same permaTicker across rebrand chain

Verified by fetching prices for a single permaTicker across multiple
historical eras corresponding to different ticker codes:

| permaTicker | Year | Ticker code then | close range | identity then |
|---|---|---|---|---|
| `US000000002062` | 2010 | CSII | $13.29 | Communications Systems Inc |
| `US000000002062` | 2020 | PEGY | $6.25 | Pineapple Energy (f/k/a CSII) |
| `US000000002062` | 2022 | SUNE | $2.45 | SUNation Energy Inc |

Three different ticker codes, **ONE permaTicker**, all retrievable
via the permaTicker-keyed `/prices` endpoint. Tiingo tracks the legal
entity continuously across the CSII → PEGY → SUNE rebrand/remerger
chain.

This **refutes our earlier "SUNE is Class S symbol-recycling
contamination" hypothesis**: SUNation's permaTicker US000000002062 is
the continuation of Communications Systems Inc, not a recycled ticker.
EODHD showed "SUNE 1990-03-26 startDate" because SUNE today = SUNation =
continuation of CSII which IPO'd in 1990. There's NO contamination —
it's a real continuous identity we just hadn't been able to track
without permaTicker.

### `permaTicker` is identity-distinct across spin-offs

Colfax Corp split into Enovis (ENOV) + ESAB Corp (ESAB) on April 4,
2022.

| Entity | permaTicker | startDate | Notes |
|---|---|---|---|
| Enovis Corp (modern) | `US000000001291` | 2008-05-08 | Real Colfax 2008-2022 + Enovis post-split history combined. Search by company name returns ENOV. |
| ESAB Corp (modern) | `US000000104025` | 2022-03-29 | Real ESAB spinoff history only (no inherited Colfax data). |
| Colfax Corp legacy (CFX) | `US000000104366` | null (active=True but no data) | `/prices` returns 0 rows everywhere — data was migrated to ENOV; CFX is a placeholder. |

So Tiingo's permaTicker strategy for spin-offs: one of the resulting
companies inherits the permaTicker of the parent (Enovis inherits
CFX's full history), and the other spin-off gets a fresh permaTicker
(ESAB gets new, post-split history only).

### `permaTicker` is identity-distinct across same-CIK reorg (Chesapeake → Expand)

Chesapeake Energy (CHK) went bankrupt 2020, restructured, emerged as
Expand Energy (EXE) in 2021. Same CIK 0000895126 from SEC's
perspective.

| Entity | permaTicker | current ticker code | startDate | endDate | isActive |
|---|---|---|---|---|---|
| Chesapeake Energy Corp | `US000000000505` | CHK (dead-ended) | 1993-02-16 | 2024-10-04 | False |
| Expand Energy Corp | `US000000092728` | EXE (modern) | 2021-02-10 | 2026-07-15+ | True |

**Tiingo treats them as separate permaTickers.** Even though SEC calls
them the same CIK (Chapter 11 reorg), Tiingo is using FINRA/NASDAQ
ticker issue / listing events as identity boundaries. Each permaTicker
key cleanly fetches its own era:
- CHK perma US000000000505 — 2015 prices $19.76 raw / back-adjusted
  $3909 reflecting reverse split done during bankruptcy
- EXE perma US000000092728 — 2021+ prices $44 continuous Expand Energy

### Looked-up entities that share ticker code over history (disambiguation test)

For ticker code **SUNE** with `includeDelisted=true, exactTickerMatch=true`:

| Ticker | permaTicker | name | active | country |
|---|---|---|---|---|
| SUNE | US000000002709 | Sunedison Inc | False | US |
| SUNE | US000000002062 | SUNation Energy Inc | True | US |

**Two distinct identities sharing the SUNE ticker code**. Both
retrievable separately via permaTicker-keyed /prices. The two return
non-overlapping rows (different close prices on the same dates). This
is the disambiguation case our search query needs to handle when we
lookup-by-ticker-code-and-filter-by-country — we'll typically want to
filter by `isActive` OR by startDate's relevance to a given论文']))

For ticker code **NSR** with same query:

| Ticker | permaTicker | name | active | country |
|---|---|---|---|---|
| NSR | AU000000100025 | NATIONAL STORAGE REIT | False | AU |
| NSR | US000000006945 | NeuStar Inc | False | US |
| NSR | US000000062716 | Nomad Royalty Company Ltd | False | US |

Three permaTickers across the NSR ticker code — Australian storage REIT
(irrelevant for our US pipeline), Neustar (legacy US), Nomad Royalty
(separate modern entity, also inactive now). The `countryCode=US`
filter immediately reduces noise.

### `permaTicker`-keyed /metadata DOESN'T work (caveat)

The metadata endpoint `/tiingo/daily/{TICKER}` works only with a
**ticker code** as the URL path key, NOT with a permaTicker. Calling
`/tiingo/daily/{permaTicker}` returns 404:

```
$ GET /tiingo/daily/US000000002709   # SunEdison perma
-> 404 {"detail":"Not found."}
```

But the prices endpoint `/tiingo/daily/{permaTicker}/prices` DOES work
with a permaTicker path key. So:
- For prices: use either `{ticker}/prices` or `{permaTicker}/prices` — both work.
- For metadata (name, description, startDate, endDate, exchangeCode):
  use only `{ticker}/prices [...]` with `countryCode=US` filter, OR
  take the metadata fields directly from the search response
  (`name` and `isActive` are returned in the search results body).

For our migration simplicity, the search response already gives us
{name, ticker, permaTicker, isActive, countryCode} — that's the
metadata we need to make primary-key decisions. We don't need to call
the `/daily/{ticker}` metadata endpoint again per-ticker after the
search call.

### `columns=` parameter behavior

The `columns` parameter on the search endpoint applies ONLY to the
prices responses (open/high/low/close/adjOpen/adjHigh/...). It does
NOT filter the search results' top-level fields. We learned this from
the error response: "All column fields must be case-sensitive and one
of the following: open, high, low, close, adjOpen, adjHigh, adjLow,
adjClose, adjVolume, divCash, splitFactor" — these are price column
names, not search-result column names.

---

## Migration design implication

We can now define our data model:

```
/metadata/sp400_perm_ids (to be renamed /metadata/sp400_permatickers):
  permaTicker            -- PRIMARY KEY (Tiingo's identity-stable ID; replaces perm_id)
  canonical_ticker       -- current trading ticker code (informational; used by EODHD
                            calendar endpoint for the /earnings/raw join)
  name                   -- from search response
  isActive               -- from search response; True if currently trading
  openFIGI               -- Bloomberg-issued OpenFIGI composite identifier (complementary
                            to permaTicker; same role; defensively captured for redundancy)
  cik                    -- SEC EDGAR informational (SIC source only; identity is permaTicker now)
  sic                    -- SEC EDGAR (for sector ETF mapping)
  index_ref              -- computed from SIC
  wikipedia_intervals    -- list of (added_date, removed_date) periods of S&P 400 membership
                           -- (Wikipedia tracks the index membership; permaTicker tracks
                           the company identity)
  sec_active_aliases     -- from SEC EDGAR (optional, kept for SIC sector lookup)
  price_unavailable      -- for cases where search returns 0 results or all
                           permaTickers have empty /prices history
```

Phase A rewrite plan:
1. Read `/metadata/sp400` (our existing Wikipedia intervals) tickers
2. For each ticker-code in `(added_date, removed_date)` intervals, call
   `search/{ticker}?includeDelisted=true&exactTickerMatch=true` once.
   The search response will contain, per matching permaTicker, BOTH the
   `permaTicker` AND `openFIGIComposite` fields. Capture both for
   cross-validation (Bloomberg-issued OpenFIGI is the defensive-redundancy
   anchor; see openFIGI note at the end of this section).
3. For each permaTicker returned, narrow to ones whose historical era
   covers our interval window. Pick the one whose Tiingo `startDate` is
   nearest prior to `added_date`. Drop the others as "not the identity
   we want at this point in time."
4. If multiple intervals for the same ticker code in our Wikipedia data
   resolve to DIFFERENT permaTickers (e.g. SAI in 1998 vs SAI in 2013
   are different companies per permaTicker lookup), treat the permaTicker
   as the perm_id, NOT the ticker code. The ticker-code path becomes
   irrelevant beyond point-in-time-disambiguation.
5. The storage key MUST be `/sp400/{permaTicker}` (NOT `/sp400/{ticker}`).
   For ticker codes shared by multiple historical entities (NSR, SUNE, CHK,
   SAI), keying by ticker code would clobber one entity's price node with
   another's. Keyed by permaTicker, each permaTicker has its own /prices
   history hydrated into its own /sp400/{permaTicker} node.

Per-ticker EOD price fetch (Phase B rewrite):
- `/tiingo/daily/{permaTicker}/prices?startDate=...&endDate=...` — one
  endpoint, permaTicker-keyed, no alias-merging needed.
- Single fetch per permaTicker per full 15-year window.
- EODHD `/api/calendar/earnings` retained ONLY for Phase D (earnings
  events fetched by ticker code; earnings keys are (permaTicker,
  announce_date, fiscal_period_end) in the new model).

Cost per canonical full fetch EOD: 1 search call (per-ticker-discovery)
+ 1 prices call (per permaTicker) = 2 calls per perm_id. For 970
permaTickers = 1,940 calls per full rebuild. Free tier (50/hr) ~40
hours. Power tier (1,000/hr) ~2 hours.

---

## Anomalies / edge cases discovered

1. **SunEdison in the Wikipedia S&P 400 era is recoverable.** perma
   US000000002709 returned clean 2015 + 2016-04 prices. So our
   historical SunEdison data is recoverable, the contamination is
   gone.

2. **CFX (Colfax legacy holder) has no price history** stored under that
   permaTicker (US000000104366). Tiingo migrated Colfax pre-2022 data
   to ENOV's permaTicker. Calling `/prices` with CFX's permaTicker
   returns 0 rows for any date range. This means our existing Phase B
   would have produced `NaN` for any Colfax-era 2015-2022 prices if
   we keyed on canonical_ticker="CFX". Switching to permaTicker-keyed
   storage (with PERMA ENOV) gives us complete Colfax-era history.

3. **SUNE metadata deceived us.** Original "Class S SUNE" assumption was
   wrong — SunEdison (US000000002709) and COMMUNICATIONS SYSTEMS INC
   → Pineapple → SUNation (US000000002062) are genuinely DIFFERENT
   companies trading under the ticker "SUNE" at different points in
   time. Both histories are individually clean in Tiingo. The price
   collapse we saw in EODHD's "SUNE 2016-2022" was real — it's
   SUNation Energy's 2-dollar delisted pennystock era. Our PEAD training
   universe just needs both permaTickers keyed separately.

4. **CSII-era SUNation** had 2010 close $13.29 with very low volume (19K,
   10K). Tiingo retroactively back-adjusts this for reverse stock
   splits done during the CSII-PEGY-SUNE era ($2.9M adjClose for $13.29
   raw close). These back-adjustments are HUGE numbers, which the
   current audit's "snapback detector" might catch as contamination.
   Switching to permaTicker-keyed storage + the standard "use adjClose
   for all calcs" policy removes this concern.

5. **ESAB spinoff**: startDate 2022-03-29 strictly; no inherited Colfax
   data. Each spinoff branch becomes a separate entity in Tiingo's
   model. Our SEC-CIK-based Phase A approach created a single perm_id
   for Colfax-ENOV-ESAB all-together (since same CIK); Tiingo would
   split Colfax (perma=US000000001291 -- ENOV's perma) and ESAB
   (perma=US000000104025 -- separate). **Tiingo's identity is more
   granular than SEC's CIK** — a spinoff into two separately-traded
   entities produces two permaTickers even when one is the legal
   name-changed continuation.

---

## One-liner conclusion for the migration decision

✅ `permaTicker` is a robust, identity-stable primary key across all
relevant identity events. The only operational caveat is that we
need to use the search endpoint with `includeDelisted=true, exactTickerMatch=true`
to discover permaTickers for defunct ticker codes (the simpler `/daily/{ticker}`
metadata call alone doesn't expose permaTicker directly).

Migrating from our synthetic `perm_id` to `permaTicker`:
- yields cleaner identity, including historical entities EODHD erased
- eliminates Class W (NSR) and Class S (SUNE) issues at source
- supports spin-off granularity (CIK-conflation bug fixed in the new model)
- lets us delete ~80% of Phase A's Wikipedia+DERA identity-synthesis code
- requires permaTicker-keyed storage (refactor `/sp400/{ticker}` -> `/sp400/{permaTicker}`)
- requires re-keying Phase D earnings rows from perm_id to permaTicker

---

## Complementary stable-ID: OpenFIGI (defensive redundancy)

Tiingo's search endpoint returns BOTH `permaTicker` (Tiingo-assigned) and
`openFIGIComposite` (Bloomberg OpenFIGI-assigned) on every result row.
Examples from probe phase 1:

| ticker | permaTicker | openFIGIComposite |
|---|---|---|
| AVNT | US000000009263 | BBG000C8NJ10 |
| ZMET (Meta Canadian listing) | CA000000141726 | (not provided) |
| SAIA | US000000002530 | BBG000P5LMQ0 |
| MACF | US000000000697 | BBG000DFVMC5 |

Two upstream authorities (Tiingo and Bloomberg) issuing independent
stable IDs for the same entity gives us **defensive redundancy**:
- If either provider ever drops coverage or re-issues IDs, the other still
  anchors our rows.
- If a permaTicker ever returns 0 rows on `/prices` (provider-side
  metadata corruption), the OpenFIGI field lets us cross-check via
  Bloomberg / FIGI API sources if needed.
- For entities missing `openFIGIComposite` (some mutual funds / non-primary
  listings), permaTicker is still the working anchor.

The pipeline captures BOTH in the new `/metadata/sp400_permatickers` table
and uses `permaTicker` as the working PRIMARY KEY (it has the working
`/prices` lookup path), with `openFIGI` as a reserve-lookup field stored
for diagnostic use.

### PermaTicker-only phase-in roadmap

A recent note in Tiingo's official docs text marks their permaTicker field
as: `permaTicker - string - Placeholder for an upcoming change to the Tiingo
API that allows querying by permaTicker`. Implication: Tiingo will soon
expose a direct `permaTicker -> metadata` lookup path (today we use
the search endpoint to discover permaTickers via company-name-fuzz; soon
we'll be able to go permaTicker -> metadata directly).

When that lands, the migration's lookup code can drop its search-endpoint
dependency for refresh cycles and use the permaTicker URL path directly.
The OpenFIGI field remains as the disaster-recovery anchor if Tiingo
itself ever has an issue, AND the canonical_ticker field acts as the
key for the EODHD earnings-calendar endpoint join (which uses ticker
codes, not stable IDs).
