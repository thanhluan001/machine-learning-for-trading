# 05b_alpaca_live/ — Live Paper Trading Pipeline

PEAD trading bot live execution. Two scripts, strict separation of concerns:

- **Script 01** (`01_fetch_and_predict.py`): Stateless V6 prediction engine → `plan.json`; V4 comparison → `v4_plan.json`
- **Script 02** (`02_paper_trade.py`): Stateful slot manager + Alpaca executor → `positions.json`

```
01_fetch_and_predict.py                02_paper_trade.py
       │                                     │
       │ FMP calendar + Tiingo refresh       │ Reads plan.json
       │ + 23 timing-correct features + v4    │ Manages 4 slots
       │                                     │ Talks to Alpaca
       ▼                                     ▼
   plan.json  ──────────────────►  positions.json
   (4 picks)                       (active/pending/closed)
```

## Script 01 — `01_fetch_and_predict.py` (V6 default; run whenever a fresh actionable list is needed)

**Freshness contract (every feature input refreshed per run):**

| Step | Input | Mechanism |
|---|---|---|
| [3] | Tiingo prices (91 safety tickers) | hard-fail gate — any lag aborts the run |
| [4] | benchmarks + FRED macros | daily refresh |
| [4.6] | analyst grades (8 revision features) | incremental FMP `/stable/grades` append for scored set (fix 2026-08-31; was frozen since the one-time 07/08 gather) |
| [4.7] | earnings actuals (Block-1 features) | **lazy** back-fill of passed null-actual quarters — scored set + held book only; runs BEFORE feature computation, so any ticker being scored is fresh by construction. Whole-table sweep for research/matrix rebuilds: `01_data/09_earnings_backfill_sweep.py` |
| monthly | SP400 membership | `01_data/refresh_sp400_membership.py` + defensive closure of stale rows |

**Stateless**: no slot management, no position tracking. V6 produces the executable `plan.json`; V4 is recorded separately for comparison.

```bash
python 05b_alpaca_live/01_fetch_and_predict.py              # V6 executable plan + V4 comparison plan
python 05b_alpaca_live/01_fetch_and_predict.py --weeks 3    # 3-week window
python 05b_alpaca_live/01_fetch_and_predict.py --dry-run    # no writes
```

**Flow**:
1. Fetch FMP earnings calendar (future dates, BMO/AMC timing)
2. Filter to CURRENT S&P 400 members (`canonical_ticker` → `permaTicker`, via `wikipedia_intervals`)
3. Refresh Tiingo prices for concerned tickers (~3s/ticker, writes to DB)
4. Drop events whose entry window has already passed
5. **Refresh benchmark ETFs + FRED macros** from source before feature
   computation — stale benchmarks (IJH/sector ETFs) and stale macros (VIX,
   fed funds, unemployment) silently corrupt `rel_ret` / `car_drift` and the
   macro features respectively, so both refresh every invocation
6. Compute 23 timing-correct features (no look-ahead)
   - Use the freshest fully completed daily close available now
   - At the final decision point this equals AMC T-1 / BMO T-2
7. Load frozen V4 comparison classifier and V6 gate classifiers
8. Apply V6 min-gate threshold and XLF filter; apply V4 comparison filters separately
9. Sort V6 by minimum gate score, output the ranked actionable bench to `plan.json`
10. Score the frozen V4 comparison classifier; write the V4
   comparison plan to `v4_plan.json` and record V4 hypothetical trades in
   `v4_shadow_trades.json`.

**Runtime**: ~17 min (Tiingo refresh ~9 min + features ~8 min for 169 events)

**Output**: `plan.json` — V6 executable ranked bench with entry/exit dates, min gate score, and full features

V4 comparison output is `v4_plan.json`; its hypothetical events are persisted in
`v4_shadow_trades.json` and never submitted.

## Comparison filters and V6 execution filters

The V4 comparison plan applies its historical P(PEAD) >= 0.20, XLF, and
LEG-like filters. V6 execution uses the min-gate score >= 0.33 and XLF
exclusion; it does not apply the V4 LEG-like filter.

The following describes the retained V4 comparison safety filters:

Script 01 applies three filters on top of P(PEAD) >= 0.20. These are
backtest-validated activations that protect against known model flaws.

### 1. Current S&P 400 membership (via `wikipedia_intervals`)

- **Problem**: The `isActive` metadata flag is stale — 294 graduated stocks
  (AMD, ETSY, ENPH, +291 others) wrongly flagged active. They'd enter the
  trading universe even though they left the S&P 400 (became mega-caps).
- **Fix**: Use `wikipedia_intervals` (add/remove dates) instead. Only 419 of
  962 historical members are current.
- **Why matters**: The model trades mid-caps, not mega-caps. AMD returned a
  pick (P=0.24) until this filter excluded it.

### 2. XLF sector exclusion (Financials)

- **Problem**: Financials have 13% PEAD precision vs 41% for the rest.
  Structural — bank earnings are macro-driven, not surprise-driven.
- **Fix**: Exclude XLF at inference only (model still trains on it).

### 3. LEG-like exclusion (negative SUE + no streak + oversold)

- **Problem**: Beaten-down stocks that pass theta lose money.
  Backtest >0.20 picks: LEG-like = **-1.31% avg, 38% win** vs
  strong = +3.2%-4.1%, 60-66% win.
- **Definition**: prior SUE < -0.5 AND no beat streak
  (`consecutive_surprises_pre == 0`) AND oversold (`rel_ret_20d < -5%`).
- **Fix**: Drop any pick matching this profile regardless of P(PEAD).
- **Why matters**: Model reverse-selects within beaten-down stocks — the
  ones it "likes" for PEAD actually go the wrong way.

### Filter order (applied in script)

```
V4: P(PEAD) >= 0.20 → sector not XLF → not LEG-like → top V4 picks
V6: min(P_gate1, P_gate2, P_gate3) >= 0.33 → sector not XLF → top V6 picks
```

## V6 default paper-trading mode

V6 is now the default executable candidate with policy:

```text
min(p_pass_g1, p_pass_g2, p_pass_g3) >= 0.33
```

Run Script 01 normally to score V6 from the same fresh Tiingo data, calendar,
and feature rows used by V4:

```bash
python 05b_alpaca_live/01_fetch_and_predict.py --weeks 2
```

It writes the V6 executable `plan.json`, which Script 02 reads and trades. It
also writes `v4_plan.json` and updates `v4_shadow_trades.json`; those are
comparison-only and never executed.

V6 artifacts:

```text
03_model/models/phase_g_v6_gate_decomposition/{pass_g1,pass_g2,pass_g3}/classifier.json
03_model/models/phase_g_v6_gate_decomposition/meta.json
```

## Script 02 — `02_paper_trade.py` (run daily, ~3:45 PM ET)

**Stateful**: manages the 4-slot portfolio, syncs with Alpaca.

```bash
python 05b_alpaca_live/02_paper_trade.py              # normal daily run
python 05b_alpaca_live/02_paper_trade.py --status     # check status only
python 05b_alpaca_live/02_paper_trade.py --dry-run    # show what would happen
```

**Daily flow**:
1. **[VERIFY]** Connection, market status, buying power
2. **[SYNC]** Verify pending entry orders → move filled quantities to ACTIVE
3. **[SYNC-EXIT]** Verify prior sell orders → filled exits to CLOSED, failed exits back to ACTIVE
4. **[CHECK]** Active positions: -10% delayed stop? T+5 exit? → submit sell order
5. **[VERIFY-EXIT]** Verify same-run sell orders before replacement entries
6. **[ENTRY]** Reserve only V6 picks with entry dates today through today + 7
   calendar days; place due-today MOC/market buys
7. **[SAVE]** Write `positions.json`

**Verify-check pattern** (per user requirement): Always verify with Alpaca
that yesterday's positions were filled before doing anything. Five verify
functions in `alpaca_client.py`:
- `[V1]` verify_connection — API reachable?
- `[V2]` verify_market_status — Market open?
- `[V3]` verify_order_filled — Did order actually fill?
- `[V4]` verify_position — Does Alpaca actually hold this?
- `[V5]` verify_buying_power — Enough cash?

## Slot management

- Max **4 simultaneous positions** (4 slots)
- Each position = **1/4 NAV** (paper account: $4,000 → $1,000/position)
- **5-day hold** (exit at Close[T+5])
- **Order types:** buys and sells are both immediate DAY market orders — reliable fills, same-day slot/buying-power turnover, no dependence on Alpaca MOC/CLS support. (This is the paper config: paper simulates CLS with injected partial fills and no auction fidelity. Live buy type is deferred to promotion time and may revert to MOC if elite smart-router routing is available — see Design.md §17.C.6.)
- **-10% delayed stop** (skip gap day = day 0, check days 1+)
- Sell orders remain in `pending_exits` until actual broker fill confirmation.
- Same-day replacement buys require broker verification of the sell order.
- **Weekly slot-refresh (force-refresh, mh=4 guard):** each ISO week the top-4 V6 picks by score occupy the slots. A due-today pick fills a free slot, or — if all slots are full — force-sells the oldest slot held from last week >= 4 business days (near-T+5) to make room. Front-loaded PEAD => last-week positions have banked most drift, so refreshing is EV-positive (validated: 04_backtest/63, 64)
- If all slots are full of fresh (this-week / <4-day) positions, a due-today pick is skipped

## Entry/exit timing

| Event | Entry | v4 feature cutoff | When to run Script 02 |
|-------|-------|-------------------|----------------------|
| BMO on day T | Market buy near T-1 close | Daily data through T-2 | Near close (~3:45 PM ET) on T-1 |
| AMC on day T | Market buy near T close | Daily data through T-1 | Near close (~3:45 PM ET) on T |

**Run-timing note (post market-order change):** buys are DAY market orders
that fill at script-run time, so run Script 02 near the close (~3:45 PM ET) to
approximate the Close[T] entry the backtest assumes. Under the prior MOC buys,
run time didn't matter (MOC filled at the close regardless); now it does.

**Actionable-list rule:** Script 01 is an as-of-now list, not a list of every
future event. During Aug 6 market hours it includes AMC Aug 6+ and BMO Aug 7+;
after the Aug 6 close it includes AMC Aug 7+ and BMO Aug 8+. Events whose
entry date has passed are removed before inference. This prevents stale BMO
candidates from appearing after their entry window.

**Daily-only feature rule:** Script 01 uses the latest fully completed daily close
available now and never uses a partial intraday bar. At the final decision point,
this is the v4 training contract: AMC T-1 / BMO T-2. For farther-future events the
ranking is provisional and is refreshed again as the entry date approaches.

## Files

| File | Description |
|------|-------------|
| `01_fetch_and_predict.py` | V6 default / V4 comparison prediction engine (stateless) |
| `02_paper_trade.py` | Slot manager + executor (stateful) |
| `alpaca_client.py` | Alpaca SDK wrapper with verify-checks |
| `plan.json` | Script 01 output: ranked V6 executable bench with min gate score + full features |
| `v4_plan.json` | V4 comparison plan; never executed |
| `v4_shadow_trades.json` | Persistent V4 hypothetical entry/exit ledger |
| `positions.json` | Active, pending-entry, pending-exit, and closed positions |
| `v4_shadow_trades.json` | Persistent V4 hypothetical entry/exit ledger |
| `positions.json` | Script 02 state: active/pending/closed positions |

## Current generated plan (2026-08-07)

The current V6 executable `plan.json` contains:

| # | Ticker | Event | Entry | V6 min-gate score |
|---:|---|---|---|---:|
| 1 | HRB | Aug 11 AMC | Aug 11 | 0.450 |
| 2 | BILL | Aug 19 AMC | Aug 19 | 0.363 |
| 3 | PFGC | Aug 12 BMO | Aug 11 | 0.349 |
| 4 | MIDD | Aug 11 BMO | Aug 10 | 0.343 |
| 5 | ENS | Aug 12 AMC | Aug 12 | 0.316 |
| 6 | ACM | Aug 10 AMC | Aug 10 | 0.309 |

Script 02 does not enter BILL yet because its entry (Aug 19) is in a later ISO
week, outside this week's top-4 slate. It will be reconsidered when its week
arrives.
