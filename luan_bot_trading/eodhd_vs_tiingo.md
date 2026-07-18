# EODHD vs Tiingo — Data Source Comparison v2 (corrected)

> **⛔ SUPERSEDED (2026-07-14).** This document's final recommendation
> section ("Option C as planned: EODHD primary + Tiingo free-tier validation")
> is **OBSOLETE**. Decisions made later in the conversation flip the
> recommendation to **Option C v2: Tiingo primary (paid $30/mo) + EODHD
> calendar-only**. Live probe evidence that drove the flip is captured in
> [`01_data/tiingo_permaTicker_audit.md`](01_data/tiingo_permaTicker_audit.md),
> where probing Tiingo's `permaTicker` field proved it is identity-stable
> across all bug classes from this audit (Class U/V/W/S eliminated at
> source).
>
> The body of this v2 doc (API comparison table, bug-class analysis, observations
> about EODHD's alias-merge behavior) remains useful context for why we
> decided to use BOTH providers, with Tiingo as the identity + price source
> of truth and EODHD retained for the calendar endpoint. But the explicit
> "Option C as planned" recommendation at the bottom is reversed by the
> permaTicker audit. Read the audit doc for the migration plan / forward design.

**Status**: Probe-based comparison to inform the buy/no-buy Tiingo
subscription decision. v2 incorporates corrections after deeper probes
revealed some of my first-round claims were sloppy. ⛔ SUPERSEDED; see banner above.

**Key correction in v2**: My initial "CHK → EXE is a clean rebrand
where Tiingo simplifies merging" claim was WRONG. CHK and EXE have
overlapping trading windows (CHK kept trading OTC post-2021, EXE
started trading 2021-02). I should not use that case to argue rebrand
handling — neither provider simplifies the CHK/EXE case cleanly because
it's NOT a rebrand — they're two separately-active tickers trading at
the same time.

The actual simple rebrand case we've verified is **POL → AVNT**
(PolyOne Corp renamed to Avient Corp ~June 2019, same CIK 0001122976).
That's the only case I'll argue from confidently.

---

## What our Phase-B bug actually was — refined

After deeper probing I have a more accurate picture:

### Case POL → AVNT (PolyOne renamed Avient 2019)

EODHD ALREADY handles this rebrand correctly:
- `AVNT.US` EODHD response starts **2010-01-04** at close=$7.95 (= real PolyOne 2010 prices).
- EODHD has backfilled the full PolyOne pre-rename history into AVNT.US.

So in theory **Phase B doesn't even need to fetch POL.US** — it can just fetch AVNT.US alone and get the full history.

But our `/metadata/sp400_perm_ids` has `aliases = ["AVNT", "POL"]` for the Avient perm_id.
Phase B faithfully fetched BOTH aliases, then concat + sort + `drop_duplicates
(keep="last")`. So:

- `AVNT.US`: clean, single adjusted_close chain, full history 2010-present.
- `POL.US`: messy — first contains PolyOne pre-rename 2018-2019 close $42,
  then dead-ends, then around 2024 has $0.0003 modern POL holder's data
  (an unrelated modern instrument at the bottom of the price heap).
- The DIRTY POL.US data leaked into /sp400/AVNT.

This is **NOT an EODHD bug** — EODHD's AVNT.US was correctly assembled at the source. It's OUR Phase B bug for including POL as a fetch source when it didn't need to.

### Case LDOS ↔ SAI (Leidos renamed to LDOS in 2013)

Similar — but here LDOS.US is the modern canonical and SAI.US is the
legacy ticker. EODHD has LDOS.US spanning pre-2013 (real Leidos as SAI
history) AND post-2013 onwards (LDOS as Leidos). It's all one clean
chain under LDOS. **Phase B's fetch of "SAI" as an alias** is what
introduces the modern $9.65 Saia Inc. contamination — EODHD doesn't
need any more "modern SAI" history because LDOS.US has it.

### Case ENOV ↔ CFX (Colfax renamed Enovis Apr 2022; spin-off split)

Identical pattern: ENOV.US has the full Colfax + Enovis history under
one adjusted_close chain. Phase B's fetch of CFX.US introduced
the legacy 2022-04-05 1:3 split-factored adjusted_close which alternated
randomly into our stored series.

**In ALL three CLASS U cases, EODHD's modern canonical_ticker
alone would have given us CLEAN data — Phase B's fetch of the legacy
alias (CFX, POL, SAI) was OSP every time.**

### Class W (NSR) — only true EODHD upstream bug

NSR is the only case where EODHD itself has the corruption in the
canonical_ticker. NSR.US as EODHD returns it today contains the modern
Nomad-Royalty-2021 prices retro-mapped back into 2015 timestamps, then
jumps to real-Neustar Era 2016+ at $23/share. There's no EODHD-side
"clean NSR ticker that contains Neustar history" — Neustar got taken
private and there's no descendant ticker symbol. So Phase B can't fix
this by fetching the canonical alone.

Tiingo allows us to detect this via `metadata.name`. Asking Tiingo
"NSR" metadata: name="Nomad Royalty Company Ltd", startDate=2021-06-03
→ that means modern NSR is Nomad, not Neustar. Doing the same probe in
EODHD is harder because EODHD's metadata for NSR is unreliable by design.

### Class S (SUNE) — both providers contaminated

Tiingo's SUNE metadata `name=SUNation Energy Inc startDate=1990-03-26`
is incorrect (SUNation Energy didn't exist in 1990). Tiingo intermixes
SunEdison defunct data (2016 $7 close with $2.2M adjClose) and SUNation
modern data ($0.20 close). This is a Tiingo Class-W-like bug, equally
present in EODHD. A `name`-sanity check catches neither if the metadata
itself is contaminated.

This is the case where **neither provider is clean** — and where we
genuinely need either a third source (Yahoo Finance / Alpha Vantage
free) or a "drop SUNE from training universe" decision.

---

## What's happening at EODHD alias-merging under the hood

When I look at `AVNT.US` EODHD response and notice it starts 2010 with
$7.95 close (real PolyOne pre-2019 rename), I infer EODHD has internal
logic that **port historical data fetched under older ticker codes into
the modern canonical ticker at the database level**.

But the merge is imperfect:

| Issue | EODHD quality | Examples |
|---|---|---|
| Backfills pre-rename history under modern ticker | Mostly ✓ | AVNT (pre-2019 PolyOne included), LDOS (pre-2013 SAI Leidos included), ENOV (pre-2022 Colfax included), META (pre-2022 Facebook included — same pattern as Tiingo) |
| Old ticker symbol dead-ends cleanly at rename | Mostly NO | POL doesn't dead-end (modern $0.0003 instrument uses same POL code feeding same POL.US response); NSR doesn't dead-end (modern Nomad-Royalty data leaks back into 2015 rows); SUNE doesn't dead-end (modern SUNation data leaks into same series) |

So EODHD's "port old history into new ticker" is fine; its "kill the
old ticker" is broken. The poison-arrow comes from Phase B then
fetching the messy-old-ticker series and contaminating the new one.

## What's happening at Tiingo alias-merging under the hood

From probing FB and META we know:

| Issue | Tiingo quality | Examples |
|---|---|---|
| Backfills pre-rebrand history under modern ticker | ✓ | META startDate=2012-05-18 (= Facebook IPO), full pre-FB-rename data available under META; asking for ENOV pre-2022 (Colfax) returns real Colfax prices. |
| Old ticker dead-ends cleanly at rename | Mostly ✓ | FB metadata reports "ProShares ETF (2025+)"; FB prices pre-2025 = zero rows returned. NSR metadata reports "Nomad Royalty" startDate=2021-06; NSR pre-2021 prices = zero rows returned. |
| Other symbol recycling cases (SUNE) | Sometimes confused | SUNE metadata says "SUNation Energy Inc startDate=1990-03-26"; SUNE prices intermix SunEdison + SUNation data. |

Tiingo's pattern: ticker symbols are **point-in-time bound** — once a
ticker code is de-listed today, it's "dead" in Tiingo's DB; if it's
later re-IPO'd by a different company, Tiingo metadata's `name` and
`startDate` reflect the new entity. Asking for prices outside the
modern holder's `startDate..endDate` returns zero rows. This avoids
Phase B's contamination when fetching by canonical_ticker alone.

## "Mergers & acquisitions" — what we actually need

The PEAD trading bot doesn't care about CHK-style bankruptcy + reorg
edge cases. What we DO have in /metadata/sp400_perm_ids is:
- Many perm_ids have `aliases = [canon, legacy_pre_rename]` representing
  the same-CIK rebrand events Wikipedia tracks.
  Examples: CFX→ENOV, SAI→LDOS, POL→AVNT.

For each: EODHD's modern canonical_ticker ALREADY contains the legacy
pre-rename history back-merged at the DB level. So we don't need to
fetch the legacy alias at all. Phase B's alias-concat was redundant —
🏷️ EODHD does the merge for us.

If we switch to Tiingo, the same is true (verified META + ENOV).

## What ChatGPT simplified/mythologized as "Tiingo easily solves CHK↔EXE"

The reality is:

If we had a same-CIK rename A → B (verified cases: PolyOne→Avient,
Colfax→Enovis, Science-Applications→Leidos) where the modern ticker
alone has the full legacy history, then:

- **Using EODHD**: fetch `B.US` alone — contains A's history back-merged. DO NOT fetch `A.US` — that has the legacy content interspersed with the modern-$0.0003 holder's contamination.

- **Using Tiingo**: fetch `B` alone — contains A's history back-merged. DO NOT fetch `A` — either returns "no data" (FB-style) or returns the modern-$0.0003-holder's data (SUNE-style).

**Both providers give us the same fix path: drop the legacy-alias
fetch and just use the canonical.**

Tiingo doesn't get bonus points for rebrand detection here — EODHD also
detects it (just at the database level rather than the metadata level).
Tiingo's win is being SAFER if we screw up and fetch a recycled
ticker, because Tiingo's response is "no data" rather than "garbage
data". EODHD can give us garbage data.

## True Tiingo wins, ranked

1. **Symbol-recycle safety** (Class W bug avoidance): if we accidentally
   fetch `NSR` (modern Nomad-Royalty) expecting to get Neustar's 2015
   EODHD-style contamination, Tiingo gives 0 rows. This is the
   single most important reason to switch if you're worried about
   unseen bugs.

2. **Metadata clean `name`** for sanity check: we can compare Tiingo's
   metadata `name` for every ticker to our expected name in
   /metadata/sp400_perm_ids and verify they match. Mismatches flag
   contamination candidates. EODHD's metadata DOES exist but is way
   less reliable.

3. **No `old-ticker-name contamination`** via over-fetching: if we
   mis-include SAI as an alias of LDOS and ask Tiingo for SAI prices in
   2011-2018, we get 0 rows (modern SAI不存在 back then). Asking EODHD
   gets us modern Saia Inc. data retro-mapped.

## EODHD's true wins, ranked

1. **Earnings calendar**: Tiingo has no equivalent to EODHD
   `/api/calendar/earnings`. Critical for Phase D forward event-fetching.

2. **Possibly cheaper**: EODHD ~$20/mo unlimited requests; Tiingo paid tier
   ~$10-30/mo depending on quota (5k-50k calls/mo). User wants subscription
   on the order of $10/mo anyway.

3. **More exchanges / non-US**: not relevant for S&P 400 (US-only).

## Probe results (2026-07-14): What Tiingo free tier ACTUALLY gives us

Live API probe with our current Tiingo key (~16 calls used, conservative):

| Endpoint | Free tier access | Notes |
|---|---|---|
| `/tiingo/daily/{TICKER}` metadata | ✓ FREE, unlimited tickers | Returns `{ticker, name, description, startDate, endDate, exchangeCode}`. The `description` field is prose: contains rebrand history in natural language. Used for name-sanity check (one call per canonical). |
| `/tiingo/daily/{TICKER}/prices` | ✓ FREE | Returns full OHLC + adj OHLC + divCash + splitFactor. Same quality as EODHD /api/eod but with splitFactor + divCash broken out. Our EODHD round-trip derivation gives identical adj chains. |
| `/tiingo/utilities/search/{KEYWORD}` | ✓ FREE | Searches by company name keyword. Returns `{name, ticker, permaTicker, openFIGIComposite, assetType, isActive, countryCode}`. The `permaTicker` is Tiingo's permanent-id field -- could be used for rebrand tracking. |
| `/tiingo/iex/{TICKER}` | ✓ FREE (intraday) | Intraday quote. Not relevant for our pipeline (we use EOD). |
| `/tiingo/daily/{TICKER}` metadata | ✓ FREE, unlimited tickers | Returns `{ticker, name, description, startDate, endDate, exchangeCode}`. The `description` field is full prose: contains rebrand history in natural language -- e.g. ENOV's description mentions "In March of 2021, Colfax announced its intention to separate into two independent and public companies... Enovis Corporation (NYSE: ENOV) will be an innovation-driven medical technology growth company". Used for name-sanity check (one call per canonical). The `description` field can be text-mined for rebrand events (search for "formerly known as", separation phrases, etc.). |
| `/tiingo/fundamentals/{TICKER}/daily` | ❌ DOW 30 only on free+Power | Returns 5 GTE metrics (marketCap/enterpriseVal/peRatio/pbRatio/trailingPEG1Y). Free tier hard-limited to DOW 30 -- AAPL works (763 rows back to 2023-07), LDOS/AMD/AER/CUBE all 400 with the explicit error: **"Free and Power plans are limited to the DOW 30. If you would like access to all supported tickers, then please E-mail support@tiingo.com to get the Fundamental Data API added as an add-on service."** Full coverage requires a Fundamental Data add-on subscription ABOVE the Power tier. |
| `/tiingo/news` (general + per-ticker) | ❌ 403 "You do not have permission to access the News API" in probe | News is a separate add-on. NOT included in Power tier. |
| `/tiingo/fx`, `/tiingo/crypto` | (not probed) | Not relevant for our US equity pipeline. |

### What the paid Tiingo tier ($30/mo) gives us over free

- **Higher rate limit**: free = 50 calls/hour; paid = 1000 calls/hour (per their docs). This is the ONLY meaningful upgrade. All endpoints (daily metadata + prices + search + IEX) are already free -- they're just rate-limited.
- **No fundamentals upgrade** -- the fundamental data add-on is a SEPARATE charge ON TOP of the $30/mo paid tier.
- **No news upgrade** -- same, separate add-on.

### Implication for our pipeline

**We do NOT need the Tiingo paid tier at all** for the use-cases we care about:

- **Daily price history**: free works (just rate-limited)
- **Daily metadata + description + startDate/endDate**: free works
- **Search utility** (name → ticker mapping): free works

The only thing the $30/mo paid tier buys us is **rate-limit removal**.

### Rate-limit math: does free tier cover our 958-canonical validation?

For the Phase B rewrite validation (post-cleanup cross-check):
- Need: 1 metadata call per canonical = 958 calls.
- Free tier: 50 calls/hour → 958/50 = ~20 hours of pacing.

Alternatively:
- Need: 1 metadata call + 3 sample price rows per canonical = 4 × 958 = 3,832 calls.
- Free tier: 50/hr → 77 hours of pacing (~3.2 days).

With paid tier: 1,000/hr → all 4 calls × 958 = 3,832 calls in ~4 hours. Single metadata only = 958 calls in 1 hour.

### Updated recommendation

The free Tiingo tier is sufficient for occasional metadata validation / name-sanity cross-checks. For one-shot post-rewrite validation of all 958 canonicals, you would pace ~3 hours with the paid tier or ~20 hours with the free tier (just metadata calls; 50/hour). Not a strong reason to subscribe for one-time validation.

For ongoing use (e.g., if we shift Phase B from EODHD to Tiingo for daily fetch): paid tier is necessary. Full fetch of 958 canonicals × ~4000 rows × 2 (metadata + prices) = ~3,800 calls per full rebuild. Even for monthly refreshes of just-canonicals metadata, free tier (50/hr) handles 950 calls in a day -- adequate for monthly work, marginal for daily.

### Final recommendation (v3, post-probe)

1. **For Phase B rewrite**: Use EODHD as primary (we already have it, unlimited). Drop the legacy-alias-fetch bug (Class U/V fix at source).

2. **For Class W/S name-sanity**: Use Tiingo free tier. Pace 50/hr; for 958 canonicals that's ~20 hrs of metadata calls (one-time, post-rewrite). NO PAID TIER NEEDED.

3. **Persist EODHD for earnings calendar**: Phase D `/api/calendar/earnings` has no Tiingo equivalent.

4. **Do NOT subscribe to Tiingo paid tier** unless we plan to switch Phase B entirely to Tiingo (which would let us drop EODHD subscription, total cost: $30 vs $20+optional). Not worth it now.

---

## Open questions for user to decide

1. **Proceed with Option C as planned (EODHD primary + Tiingo free-tier validation)?** My probe confirms free tier is sufficient for one-time post-rewrite validation; you'd pace ~20 hours manually.

2. **OR switch Phase B to Tiingo primary (free tier is enough since we only refresh monthly)?** Would let us drop EODHD entirely EXCEPT for earnings calendar (Phase D still needs /api/calendar/earnings). Net: keep both EODHD + Tiingo-free, no new subscription cost.

3. **Manual Class-W/S pre-flagging**: Should I hard-fail NSR + SUNE in `02b_build_company_map.py` (set price_unavailable=True) up-front, OR rely on the post-rewrite Tiingo metadata validation to discover them empirically?

4. **Tiingo `permaTicker` field**: Do you want to capture it as part of our metadata schema? Could be useful for cross-ticker identity tracking (rebrand robust), alternative to our current `perm_id` derivation.
