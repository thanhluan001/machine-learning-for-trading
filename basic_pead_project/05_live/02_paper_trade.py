#!/usr/bin/env python3
"""
02_paper_trade.py — Slot Manager + Alpaca Executor (STATEFUL)
==============================================================

PURPOSE
-------
Reads the V6 executable plan.json (from Script 01), manages the 4-slot
portfolio, and executes trades on Alpaca paper trading. V4 comparison picks
are stored separately in v4_plan.json and are never read.

DAILY FLOW (run ~3:45 PM ET before market close):
  1. VERIFY: connection, market status, buying power
  2. SYNC: verify all PENDING entry orders filled on Alpaca → move to ACTIVE
  3. SYNC EXITS: verify prior sell orders; filled exits → CLOSED, rejected exits → ACTIVE
  4. RECONCILE: sync local state with Alpaca (handle manual closes while away)
  5. CHECK ACTIVE: any hit -10% delayed stop? Any at T+5 exit? → submit sell
  6. VERIFY EXITS: verify same-run sell submission before considering replacement slots
  7. COUNT free slots (4 - active - pending entries)
  8. PLACE new entries: weekly slot-refresh (top-4 picks fill slots, force-sell last-week laggards if full)
  7. WRITE positions.json

RESILIENCE: If you close positions manually on Alpaca while away from the
computer, this script reconciles local positions.json with Alpaca's actual
state on your next launch. Manually-closed positions are detected (local
ACTIVE but not on Alpaca) and moved to CLOSED as 'manual_close'.

VERIFY-CHECK PATTERN (per user requirement):
  ALWAYS verify with Alpaca that yesterday's positions were filled
  before doing anything. More verify-checks when talking to Alpaca.
  Never trust local JSON — always sync with broker state.

SLOT MANAGEMENT:
  - Max 4 simultaneous positions (4 slots)
  - Each position = 1/4 of current NAV
  - 5-day hold (exit at Close[T+5])
  - -10% delayed stop (skip gap day = day 0, check days 1+)
  - When a slot frees up (stop/exit), the week's top V6 picks fill it
  - Weekly slot-refresh: if all slots are full, force-sell the oldest slot held
    from last week >= MIN_FORCE_HOLD business days (near-T+5) to make room for
    a fresh due-today pick. Front-loaded PEAD => last-week positions have banked
    most of their drift, so refreshing is EV-positive (validated 63/64 backtest).

USAGE
-----
    python 05b_alpaca_live/02_paper_trade.py            # normal daily run
    python 05b_alpaca_live/02_paper_trade.py --status   # just check status, no trades
    python 05b_alpaca_live/02_paper_trade.py --dry-run  # show what would happen
"""
from __future__ import annotations

import io
import json
import sys
from datetime import datetime, date, timedelta
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, RuntimeError):
    pass

import pandas as pd

from alpaca_client import AlpacaClient
from tax_log import log_fill

HERE = Path(__file__).resolve().parent
PLAN_JSON = HERE / "plan.json"
POSITIONS_JSON = HERE / "positions.json"

MAX_SLOTS = 4
STOP_LOSS_PCT = 0.10  # -10% delayed stop
# Force-refresh guard: only force-sell a slot held >= MIN_FORCE_HOLD business
# days (i.e. near its T+5), so negligible drift is sacrificed. mh=4 selected by
# 04_backtest/64_force_refresh_guard_bootstrap.py (best on the 2026 H1 holdout).
MIN_FORCE_HOLD = 4


# ==============================================================================
# STATE MANAGEMENT
# ==============================================================================
def load_positions() -> dict:
    """Load positions.json, or create empty state."""
    if POSITIONS_JSON.exists():
        with open(POSITIONS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"last_updated": None, "active": [], "pending": [], "closed": []}


def save_positions(positions: dict):
    positions["last_updated"] = datetime.now().isoformat()
    with open(POSITIONS_JSON, "w", encoding="utf-8") as f:
        json.dump(positions, f, indent=2, ensure_ascii=False, default=str)
    print(f"  → Saved {POSITIONS_JSON.name}")


def load_plan() -> dict | None:
    """Load the executable V6 plan.json from Script 01."""
    if not PLAN_JSON.exists():
        return None
    with open(PLAN_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def count_active(positions: dict) -> int:
    """Count currently active (filled) positions."""
    return len(positions.get("active", []))


def free_slots(positions: dict) -> int:
    """How many slots are available after active/pending entries.

    A broker-verified pending exit has been removed from active and therefore
    frees its slot for same-run replacement. The exit remains in
    ``pending_exits`` until the actual sell fill is confirmed.
    """
    return MAX_SLOTS - count_active(positions) - len(positions.get("pending", []))


# ==============================================================================
# DAILY EXECUTION
# ==============================================================================
def reconcile_positions(trader: AlpacaClient, positions: dict, dry_run: bool = False):
    """RECONCILE: sync local positions.json with ACTUAL Alpaca state.

    Handles the case where the user was away and manually closed positions
    on the Alpaca app/web. Local state would still show them as ACTIVE.

    Logic:
      - Local ACTIVE but symbol NOT on Alpaca (or qty 0) → user closed it
        manually → move to CLOSED with exit_reason='manual_close'.
        Exit price from Alpaca's most recent SELL fill (fallback current price).
      - Local ACTIVE but qty mismatch → update local qty to actual.
      - Alpaca holds a symbol we don't track → warn (could be a manual buy).

    This runs BEFORE stop/exit checks so we never act on stale local state.
    """
    # Get actual Alpaca positions as {symbol: dict}
    actual = {p['symbol']: p for p in trader.get_all_positions()}

    active = positions.get('active', [])
    if not active and not actual:
        print("  Local and Alpaca both empty. No reconciliation needed.")
        return

    print(f"  Local active: {len(active)} | Alpaca positions: {len(actual)}")

    reconciled = []
    for pos in active:
        sym = pos['canonical_ticker']
        alp = actual.get(sym)

        if alp is None or alp['qty'] == 0:
            # Manually closed while we were away
            exit_price, fill_date = (None, None)
            if not dry_run:
                exit_price, fill_date = trader.get_last_sell_fill(sym)
            if exit_price is None:
                # Fallback: current price (approximation)
                exit_price = trader.get_current_price(sym)
            if exit_price is None:
                exit_price = pos.get('entry_price', 0)
            pos['exit_reason'] = 'manual_close'
            # The TRADE date is the Alpaca fill date, not the run date —
            # the user may have closed it days before we reconciled
            # (found 2026-09-02: NTNX closed Sep-1 logged as Sep-2).
            actual_exit_date = fill_date or date.today().isoformat()
            # For CLOSED records, exit_date is the actual close date. Preserve
            # the original planned T+5 date separately for audit/history.
            pos['planned_exit_date'] = pos.get('exit_date')
            pos['exit_date'] = actual_exit_date
            pos['exit_date_actual'] = actual_exit_date
            pos['exit_price'] = exit_price
            pos['return_pct'] = round(
                (exit_price - pos.get('entry_price', 0)) / pos.get('entry_price', 1) * 100, 2)
            try:
                log_fill(actual_exit_date, sym, "SELL", pos.get('qty', 0), exit_price)
            except Exception as e:
                print(f"    [TAX] WARN: manual close not logged to tax ledger: {e}")
            positions['closed'].append(pos)
            print(f"    [RECON] {sym}: manual_close @ ${exit_price:.2f} "
                  f"({pos['return_pct']:+.1f}%) → CLOSED")
        elif abs(alp['qty'] - pos.get('qty', 0)) > 0.001:
            # Qty mismatch — trust Alpaca
            old = pos.get('qty')
            pos['qty'] = alp['qty']
            print(f"    [RECON] {sym}: qty {old} → {alp['qty']} (synced to Alpaca)")
            reconciled.append(pos)
        else:
            reconciled.append(pos)

    positions['active'] = reconciled

    # Warn about Alpaca positions we don't track (possible manual buy)
    tracked_syms = {p['canonical_ticker'] for p in active}
    for sym in actual:
        if sym not in tracked_syms:
            print(f"    [WARN] Alpaca holds {sym} but local doesn't track it "
                  f"(possible manual buy — will NOT auto-sell)")


def sync_pending(trader: AlpacaClient, positions: dict, dry_run: bool = False):
    """VERIFY: reconcile every pending order with Alpaca.

    Alpaca can return a terminal status (expired/canceled) with a nonzero
    filled_qty. Always process the fill quantity first; never classify such an
    order as wholly failed. Zero-fill terminal orders are recorded as failed
    and free the slot.
    """
    pending = positions.get("pending", [])
    if not pending:
        print("  No pending orders to verify.")
        return

    print(f"\n  [VERIFY] Checking {len(pending)} pending orders against Alpaca ...")
    still_pending = []
    for p in pending:
        order_id = p.get("order_id")
        if not order_id or dry_run:
            print(f"    [DRY] Would verify {p['canonical_ticker']} order {order_id}")
            still_pending.append(p)
            continue

        # [V3] Verify order status AND actual fill quantity.
        fill = trader.verify_order_filled(order_id)
        filled_qty = float(fill.get("filled_qty") or 0.0)
        planned_qty = float(p.get("planned_qty") or p.get("qty") or 0.0)
        terminal = bool(fill.get("is_canceled"))

        # A terminal order may still have filled shares (e.g. DBX: expired,
        # 9/29 filled). The shares are real and must become an active position.
        if filled_qty > 0 and fill.get("filled_avg_price") is not None:
            p["entry_price"] = float(fill["filled_avg_price"])
            p["qty"] = filled_qty
            p["filled_qty"] = filled_qty
            p["planned_qty"] = planned_qty
            p["unfilled_qty"] = max(0.0, planned_qty - filled_qty)
            p["fill_status"] = "partial" if p["unfilled_qty"] > 0 else "filled"
            p["order_status_final"] = fill.get("status")
            p["stop_price"] = round(p["entry_price"] * (1 - STOP_LOSS_PCT), 2)
            p["fill_time"] = datetime.now().isoformat()
            try:
                # DAY order placed ~3:45pm on entry_date -> fill is same session
                log_fill(str(p.get("entry_date") or date.today().isoformat())[:10],
                         p["canonical_ticker"], "BUY", filled_qty, p["entry_price"])
            except Exception as e:
                print(f"        [TAX] WARN: entry not logged to tax ledger: {e}")
            positions["active"].append(p)
            if p["unfilled_qty"] > 0:
                print(f"    → {p['canonical_ticker']} PARTIAL: {filled_qty:g}/{planned_qty:g} "
                      f"@ ${p['entry_price']:.2f}; {p['unfilled_qty']:g} unfilled "
                      f"(order {fill['status']}) → ACTIVE")
            else:
                print(f"    → {p['canonical_ticker']} FILLED @ ${p['entry_price']:.2f} "
                      f"(stop=${p['stop_price']:.2f}) → ACTIVE")
        elif terminal:
            # No shares exist at Alpaca. Record the failed attempt for audit,
            # but do not treat it as a position or consume a slot.
            p["filled_qty"] = 0.0
            p["unfilled_qty"] = planned_qty
            p["fill_status"] = "unfilled"
            p["order_status_final"] = fill.get("status")
            p["exit_reason"] = f"order_{fill['status']}"
            p["exit_date_actual"] = date.today().isoformat()
            positions["closed"].append(p)
            print(f"    → {p['canonical_ticker']} {fill['status'].upper()}, 0/{planned_qty:g} filled "
                  "→ FAILED/EXPIRED (slot freed)")
        else:
            still_pending.append(p)
            print(f"    → {p['canonical_ticker']} still {fill['status']} "
                  f"(filled={filled_qty:g})")

    positions["pending"] = still_pending


def _finalize_exit_fill(pos: dict, fill: dict, positions: dict) -> None:
    """Move a fully filled exit from pending_exits to closed."""
    exit_price = float(fill.get("filled_avg_price") or pos.get("exit_price") or 0.0)
    filled_qty = float(fill.get("filled_qty") or pos.get("qty") or 0.0)
    pos["filled_exit_qty"] = filled_qty
    pos["exit_price"] = exit_price
    pos["exit_date_actual"] = date.today().isoformat()
    pos["exit_date"] = pos["exit_date_actual"]
    pos["return_pct"] = round((exit_price - pos.get("entry_price", 0)) / pos.get("entry_price", 1) * 100, 2)
    pos["exit_order_status_final"] = fill.get("status")
    pos["exit_fill_status"] = "filled"
    try:
        log_fill(pos["exit_date_actual"], pos["canonical_ticker"], "SELL",
                 filled_qty, exit_price)
    except Exception as e:
        print(f"    [TAX] WARN: exit not logged to tax ledger: {e}")
    positions.setdefault("closed", []).append(pos)


def _submit_sell_and_finalize(trader: AlpacaClient, positions: dict, pos: dict,
                              reason: str, today, planned_price: float) -> None:
    """Place a market sell, poll up to ~10s, fast-path clean full fills to CLOSED.

    Market sells typically fill in 1-3s. If the order reaches a clean full
    fill within the poll window, finalize it same-run (move straight to
    closed). Partial or not-filled-after-10s orders are left in pending_exits
    for the existing next-run reconciliation (sync_exit_orders) — i.e. the old
    behavior. Either way the slot is freed once the sell is confirmed.
    """
    ticker = pos["canonical_ticker"]
    order_id = trader.place_market_sell(
        ticker, pos["qty"], f"pead_{reason}_{ticker}_{today.isoformat()}")
    pos["exit_reason"] = reason
    pos["planned_exit_date"] = pos.get("exit_date")
    pos["planned_exit_qty"] = pos["qty"]
    pos["planned_exit_price"] = planned_price
    pos["exit_order_id"] = order_id
    fill = trader.poll_order_fill(order_id, timeout=10.0)
    planned = float(pos["planned_exit_qty"])
    fq = float(fill.get("filled_qty") or 0.0)
    if fill.get("is_filled") and fq >= planned and fill.get("filled_avg_price") is not None:
        _finalize_exit_fill(pos, fill, positions)
        print(f"    → {ticker} SELL FILLED {fq:g}/{planned:g} "
              f"@ ${float(fill['filled_avg_price']):.2f} → CLOSED (same-run)")
    else:
        positions.setdefault("pending_exits", []).append(pos)
        print(f"    → {ticker} sell pending after 10s (status={fill.get('status')}, "
              f"filled={fq:g}/{planned:g}); reconcile next run")


def sync_exit_orders(trader: AlpacaClient, positions: dict, dry_run: bool = False):
    """Verify prior sell orders and reconcile their actual fill quantities."""
    pending = positions.get("pending_exits", [])
    if not pending:
        print("  No pending sell orders to verify.")
        return
    print(f"\n  [VERIFY-EXIT] Checking {len(pending)} sell orders against Alpaca ...")
    still_pending = []
    for pos in pending:
        order_id = pos.get("exit_order_id")
        if not order_id or dry_run:
            print(f"    [DRY] Would verify sell {pos.get('canonical_ticker')} {order_id}")
            still_pending.append(pos)
            continue
        fill = trader.verify_order_filled(order_id)
        filled_qty = float(fill.get("filled_qty") or 0.0)
        planned_qty = float(pos.get("planned_exit_qty") or pos.get("qty") or 0.0)
        terminal = bool(fill.get("is_canceled"))
        if filled_qty >= planned_qty and fill.get("filled_avg_price") is not None:
            _finalize_exit_fill(pos, fill, positions)
            print(f"    → {pos['canonical_ticker']} SELL FILLED {filled_qty:g}/{planned_qty:g} "
                  f"@ ${float(fill['filled_avg_price']):.2f} → CLOSED")
        elif filled_qty > 0 and terminal and fill.get("filled_avg_price") is not None:
            # Partial sell: close the sold quantity and restore the residual.
            sold = dict(pos)
            sold["qty"] = filled_qty
            _finalize_exit_fill(sold, fill, positions)
            residual = dict(pos)
            residual["qty"] = planned_qty - filled_qty
            residual["planned_exit_qty"] = residual["qty"]
            residual.pop("exit_order_id", None)
            residual["exit_fill_status"] = "partial_residual"
            positions.setdefault("active", []).append(residual)
            print(f"    → {pos['canonical_ticker']} SELL PARTIAL {filled_qty:g}/{planned_qty:g}; "
                  f"residual {residual['qty']:g} restored ACTIVE")
        elif terminal:
            # Rejected/canceled/expired with no fill: restore the position.
            restored = dict(pos)
            restored.pop("exit_order_id", None)
            restored.pop("planned_exit_qty", None)
            restored["exit_fill_status"] = "unfilled_restored"
            positions.setdefault("active", []).append(restored)
            print(f"    → {pos['canonical_ticker']} SELL {fill.get('status','unknown').upper()}, "
                  "0 filled → restored ACTIVE")
        else:
            pos["exit_order_verified_status"] = fill.get("status")
            still_pending.append(pos)
            print(f"    → {pos['canonical_ticker']} sell still {fill.get('status')} "
                  f"(filled={filled_qty:g}); remains pending exit")
    positions["pending_exits"] = still_pending


def verify_same_run_exits(trader: AlpacaClient, positions: dict, dry_run: bool = False) -> bool:
    """Verify newly submitted sell orders before allowing replacement buys.

    Sells are immediate DAY market orders, so they typically fill within
    seconds of submission; ``accepted``/``filled`` is sufficient to establish
    that the sell is broker-verified and the slot/buying power is freed for a
    same-run (MOC) replacement buy. The actual fill quantity/price is
    reconciled by ``sync_exit_orders`` on the next run.
    """
    pending = positions.get("pending_exits", [])
    if not pending:
        return True
    print(f"\n  [VERIFY-EXIT] Sell barrier for {len(pending)} pending exit(s) ...")
    verified = True
    for pos in pending:
        order_id = pos.get("exit_order_id")
        if dry_run or not order_id:
            continue
        fill = trader.verify_order_filled(order_id)
        if fill.get("is_canceled") and float(fill.get("filled_qty") or 0.0) <= 0:
            restored = dict(pos)
            restored.pop("exit_order_id", None)
            restored.pop("planned_exit_qty", None)
            positions.setdefault("active", []).append(restored)
            positions["pending_exits"].remove(pos)
            verified = False
            print(f"    → {pos['canonical_ticker']} sell rejected/terminal with no fill; "
                  "restored ACTIVE, replacement blocked")
        else:
            pos["exit_order_verified_status"] = fill.get("status")
            print(f"    → {pos['canonical_ticker']} sell broker-verified "
                  f"({fill.get('status')}); replacement slot may be used")
    return verified


def check_active_positions(trader: AlpacaClient, positions: dict, dry_run: bool = False):
    """Check active positions for -10% stop triggers and T+5 exits.

    Delayed stop: skip day 0 (entry/gap day), check days 1+.
    Exit: sell at exit_date (report_date + 5 trading days).
    """
    active = positions.get("active", [])
    if not active:
        print("  No active positions to check.")
        return

    today = date.today()
    print(f"\n  [CHECK] {len(active)} active positions (today={today})")

    still_active = []
    for pos in active:
        ticker = pos["canonical_ticker"]
        entry_date = datetime.strptime(pos["entry_date"], "%Y-%m-%d").date()
        exit_date = datetime.strptime(pos["exit_date"], "%Y-%m-%d").date()
        days_held = (today - entry_date).days
        stop_price = pos.get("stop_price", 0)

        if dry_run:
            print(f"    [DRY] {ticker}: held {days_held}d, "
                  f"exit={exit_date}, stop=${stop_price:.2f}")
            still_active.append(pos)
            continue

        # Get current price
        price = trader.get_current_price(ticker)
        if price is None:
            print(f"    [WARN] {ticker}: can't get price — keeping position")
            still_active.append(pos)
            continue

        # [V4] Verify position actually exists on Alpaca
        alpaca_pos = trader.verify_position(ticker)

        # Check T+5 exit first (takes priority over stop)
        if today >= exit_date:
            print(f"    [EXIT] {ticker}: T+5 reached (exit_date={exit_date})")
            _submit_sell_and_finalize(trader, positions, pos, "t5_exit", today, price)
            continue

        # Check delayed stop (skip day 0 = gap day)
        if days_held >= 1 and price <= stop_price:
            print(f"    [STOP] {ticker}: STOP HIT (price=${price:.2f} <= stop=${stop_price:.2f})")
            _submit_sell_and_finalize(trader, positions, pos, "stop_loss", today, price)
            continue

        # Position still active
        pnl_pct = (price - pos["entry_price"]) / pos["entry_price"] * 100
        print(f"    [HOLD] {ticker}: day {days_held}, price=${price:.2f} "
              f"({pnl_pct:+.1f}%), exit={exit_date}, stop=${stop_price:.2f}")
        still_active.append(pos)

    positions["active"] = still_active


def _iso_week(d):
    """ISO (year, week) tuple for a date or 'YYYY-MM-DD' string (None-safe)."""
    if d is None:
        return None
    if isinstance(d, str):
        d = date.fromisoformat(d)
    iso = d.isocalendar()
    return (iso.year, iso.week)


def _biz_days_held(entry, today):
    """Mon-Fri day count from entry to today (entry exclusive, today inclusive)."""
    d0 = date.fromisoformat(entry) if isinstance(entry, str) else entry
    days = 0
    cur = d0
    while cur < today:
        cur += timedelta(days=1)
        if cur.weekday() < 5:
            days += 1
    return days


def _pick_force_victim(positions: dict, today):
    """Oldest active slot that is force-sellable: from a PRIOR ISO week AND held
    >= MIN_FORCE_HOLD business days (mh guard). None if no candidate."""
    tw = _iso_week(today)
    cands = []
    for pos in positions.get("active", []):
        if _iso_week(pos.get("entry_date")) >= tw:
            continue  # entered this week/later -> protected (fresh)
        if _biz_days_held(pos.get("entry_date"), today) < MIN_FORCE_HOLD:
            continue  # too fresh to cut (mh guard)
        cands.append(pos)
    if not cands:
        return None
    return min(cands, key=lambda p: p.get("entry_date", "9999"))


def _enter_pick(trader: AlpacaClient, positions: dict, pick: dict,
                per_position: float, today, plan: dict, dry_run: bool) -> bool:
    """Enter one pick into a free slot, or force-refresh the oldest eligible
    slot if none free. Returns True if a buy order was placed."""
    today_str = today.isoformat()
    ticker = pick["canonical_ticker"]
    price = trader.get_current_price(ticker)
    if price is None or price <= 0:
        print(f"    [SKIP] {ticker}: can't get price")
        return False

    # DAY market orders fill at ~the current ask; whole-share sizing (no buffer
    # needed since execution tracks the sizing price). See Design.md §17.C.6.
    qty = int(per_position / price)
    if qty <= 0:
        print(f"    [SKIP] {ticker}: qty=0 (price ${price:.2f} too high for ${per_position:.0f})")
        return False
    cost = qty * price

    # Find a slot: free first, else force-refresh oldest eligible slot.
    victim = None
    if free_slots(positions) <= 0:
        victim = _pick_force_victim(positions, today)
        if victim is None:
            print(f"    [SKIP] {ticker}: no free slot and no force-sellable slot "
                  f"(all this-week or < {MIN_FORCE_HOLD} biz days)")
            return False
        vheld = _biz_days_held(victim["entry_date"], today)
        print(f"    [FORCE-REFRESH] no free slot -> force-selling oldest prior-week "
              f"slot {victim['canonical_ticker']} (held {vheld} biz days, entry "
              f"{victim['entry_date']}) to make room for {ticker}")
        if not dry_run:
            _submit_sell_and_finalize(trader, positions, victim, "force_refresh", today, price)

    if not trader.verify_buying_power(cost):
        print(f"    [SKIP] {ticker}: insufficient buying power")
        return False

    if dry_run:
        tag = f" [would force-refresh {victim['canonical_ticker']}]" if victim else ""
        print(f"    [DRY] Would BUY {ticker} × {qty} @ ~${price:.2f} (${cost:,.0f}){tag}")
        return False

    order_id = trader.place_market_buy(ticker, qty, f"pead_entry_{ticker}_{today_str}")
    pick_pending = {
        "canonical_ticker": ticker,
        "permaTicker": pick.get("permaTicker"),
        "entry_date": pick["entry_date"],
        "report_date": pick["report_date"],
        "time": pick["time"],
        "exit_date": pick["exit_date"],
        "p_pead": pick["p_pead"],
        "model": plan.get("model", "phase_g_v6_gate_decomposition"),
        "sector": pick.get("sector"),
        "order_id": order_id,
        "order_placed_at": datetime.now().isoformat(),
        "planned_qty": qty,
        "planned_entry_price": price,
    }
    # Same-run honesty pass (mirrors the sell path): poll up to ~10s so the
    # printed status reflects the broker's actual state. A clean full fill
    # goes straight to ACTIVE with the real entry price; partial fills are
    # recorded with the unfilled remainder; rejected/expired zero-fill orders
    # are recorded as failed and free the slot. Anything still working after
    # the poll stays pending for next-run reconciliation (sync_pending).
    fill = trader.poll_order_fill(order_id, timeout=10.0)
    fq = float(fill.get("filled_qty") or 0.0)
    tag = f" [force-refreshed {victim['canonical_ticker']}]" if victim else ""
    if fq > 0 and fill.get("filled_avg_price") is not None:
        pick_pending["entry_price"] = float(fill["filled_avg_price"])
        pick_pending["qty"] = fq
        pick_pending["filled_qty"] = fq
        pick_pending["unfilled_qty"] = max(0.0, qty - fq)
        pick_pending["fill_status"] = "partial" if pick_pending["unfilled_qty"] > 0 else "filled"
        pick_pending["order_status_final"] = fill.get("status")
        pick_pending["stop_price"] = round(pick_pending["entry_price"] * (1 - STOP_LOSS_PCT), 2)
        pick_pending["fill_time"] = datetime.now().isoformat()
        try:
            log_fill(today.isoformat(), ticker, "BUY", fq, pick_pending["entry_price"])
        except Exception as e:
            print(f"        [TAX] WARN: entry not logged to tax ledger: {e}")
        positions["active"].append(pick_pending)
        if pick_pending["unfilled_qty"] > 0:
            print(f"    → {ticker} BUY PARTIAL {fq:g}/{qty:g} @ ${pick_pending['entry_price']:.2f} "
                  f"(order {order_id[:8]}..., status={fill.get('status')}) → ACTIVE{tag}")
        else:
            print(f"    → {ticker} BUY FILLED {fq:g}/{qty:g} @ ${pick_pending['entry_price']:.2f} "
                  f"→ ACTIVE (stop=${pick_pending['stop_price']:.2f}){tag}")
    elif fill.get("is_canceled"):
        pick_pending["filled_qty"] = 0.0
        pick_pending["unfilled_qty"] = float(qty)
        pick_pending["fill_status"] = "unfilled"
        pick_pending["order_status_final"] = fill.get("status")
        pick_pending["exit_reason"] = f"order_{fill['status']}"
        pick_pending["exit_date_actual"] = date.today().isoformat()
        positions["closed"].append(pick_pending)
        print(f"    → {ticker} BUY {fill['status'].upper()} 0/{qty:g} filled → FAILED (slot freed){tag}")
    else:
        positions["pending"].append(pick_pending)
        print(f"    → {ticker} BUY {qty:g} @ ~${price:.2f} pending after 10s "
              f"(status={fill.get('status')}, filled={fq:g}) → reconcile next run{tag}")
    return True


def place_new_entries(trader: AlpacaClient, plan: dict, positions: dict,
                      dry_run: bool = False):
    """Weekly slot-refresh entry policy (force-refresh, mh=4 guard).

    Each ISO week, the top-MAX_SLOTS V6 picks (by p_pead) should occupy the
    slots. For each pick DUE TODAY (in priority order):
      - free slot -> buy into it;
      - no free slot -> force-sell the oldest slot held from a PRIOR week for
        >= MIN_FORCE_HOLD business days (mh=4: only near-T+5 positions, so
        negligible drift is sacrificed), then buy;
      - else skip.
    Rationale: front-loaded PEAD means last-week positions have banked most of
    their drift; refreshing slots with fresh weekly picks is EV-positive.
    Validated by 04_backtest/63_force_refresh_backtest.py and
    64_force_refresh_guard_bootstrap.py (force-refresh mh=4 best on the
    untouched 2026 H1 holdout: NAV +87.2%, DD -8.6%, win 69.4%, reliable NAV
    CI; statistical edge over conviction-skip is suggestive, not decisive).
    """
    today = date.today()
    today_str = today.isoformat()
    picks = plan.get("picks", [])

    held = {p['canonical_ticker'] for p in positions.get('active', [])} | \
           {p['canonical_ticker'] for p in positions.get('pending', [])}

    # BROKER-TRUTH DOUBLE-BUY GUARD: never buy a ticker the broker already
    # holds, regardless of local state. Protects against: (a) re-running the
    # script after a crash between order fill and positions.json save;
    # (b) a stale/lost local file; (c) manual buys of the same name. A
    # partial broker fill is NOT topped up here - it reconciles via the
    # pending-order sync flow on the next run instead.
    try:
        broker_held = {p["symbol"] for p in trader.get_all_positions() if p["qty"] > 0}
    except Exception as exc:
        broker_held = set()
        print(f"  [GUARD] broker positions unavailable ({exc}); local state only")
    stray = broker_held - held
    if stray:
        print(f"  [GUARD] broker holds but local does not: {sorted(stray)} "
              f"-> excluded from buys (reconcile with reconcile_positions)")
    held |= broker_held

    # Weekly slate: this ISO week's picks (not held), top-MAX_SLOTS by p_pead.
    tw = _iso_week(today)
    week_picks = [p for p in picks if _iso_week(p.get("entry_date")) == tw
                  and p['canonical_ticker'] not in held]
    week_picks.sort(key=lambda p: p.get("p_pead", 0), reverse=True)
    slate = week_picks[:MAX_SLOTS]
    due_today = [p for p in slate if p.get("entry_date") == today_str]

    print(f"\n  [ENTRY] weekly slot-refresh (force-refresh, mh={MIN_FORCE_HOLD} guard)")
    print(f"    free slots {free_slots(positions)}/{MAX_SLOTS} | "
          f"this-week picks={len(week_picks)} | slate(top {MAX_SLOTS})="
          f"{[p['canonical_ticker'] for p in slate]} | due today="
          f"{[p['canonical_ticker'] for p in due_today]}")

    if not due_today:
        print(f"    No slate picks due today ({today_str}).")
        return

    account = trader.verify_connection()
    nav = account["portfolio_value"]
    per_position = nav / MAX_SLOTS
    print(f"    NAV=${nav:,.0f}, per-position=${per_position:,.0f}")

    for pick in due_today:
        _enter_pick(trader, positions, pick, per_position, today, plan, dry_run)


# ==============================================================================
# STATUS REPORT
# ==============================================================================
def print_status(positions: dict, plan: dict | None):
    """Print current portfolio status."""
    bar = "-" * 60
    print(f"\n{bar}")
    print("  PORTFOLIO STATUS")
    print(bar)

    active = positions.get("active", [])
    pending = positions.get("pending", [])
    closed = positions.get("closed", [])

    print(f"  Active:   {len(active)}/{MAX_SLOTS} slots")
    for p in active:
        print(f"    {p['canonical_ticker']:6s} | entry=${p.get('entry_price',0):.2f} "
              f"| stop=${p.get('stop_price',0):.2f} | exit={p['exit_date']} "
              f"| P={p.get('p_pead',0):.3f}")

    print(f"  Pending:  {len(pending)}")
    for p in pending:
        print(f"    {p['canonical_ticker']:6s} | entry_date={p['entry_date']} "
              f"| order={p.get('order_id','?')[:8]}...")

    if closed:
        wins = [c for c in closed if c.get("return_pct", 0) > 0]
        losses = [c for c in closed if c.get("return_pct", 0) <= 0]
        total_ret = sum(c.get("return_pct", 0) for c in closed)
        print(f"  Closed:   {len(closed)} trades "
              f"({len(wins)}W / {len(losses)}L, total={total_ret:+.1f}%)")
        for c in closed[-5:]:  # last 5
            print(f"    {c['canonical_ticker']:6s} | {c.get('return_pct',0):+.1f}% "
                  f"| {c.get('exit_reason','?')}")

    if plan:
        today_str = date.today().isoformat()
        due = [p for p in plan.get("picks", []) if p.get("entry_date") == today_str]
        if due:
            print(f"  Due today: {len(due)} picks")
            for p in due:
                print(f"    {p['canonical_ticker']:6s} | P={p['p_pead']:.3f} "
                      f"| {p['time'].upper()} | {p['report_date']}")
    print(bar)


# ==============================================================================
# MAIN
# ==============================================================================
def main(status_only: bool = False, dry_run: bool = False):
    bar = "=" * 70
    print(bar)
    print("  02_paper_trade.py — Slot Manager + Alpaca Executor")
    print(bar)

    # Load state
    plan = load_plan()
    positions = load_positions()

    if plan:
        print(f"  Plan: {plan.get('generated_at','?')[:10]}, "
              f"{len(plan.get('picks',[]))} picks, model={plan.get('model','?')}")
        if plan.get("model") != "phase_g_v6_gate_decomposition":
            raise RuntimeError("plan.json is not the frozen V6 executable plan; refusing to place orders")
    else:
        print("  Plan: NONE (run 01_fetch_and_predict.py first)")

    # Print current status
    print_status(positions, plan)

    if status_only:
        print("\n  [--status] Status only, no trades.")
        return

    # --- VERIFY: init + connection ---
    print(f"\n  [VERIFY] Connecting to Alpaca paper trading ...")
    try:
        trader = AlpacaClient()
        trader.verify_connection()
        trader.verify_market_status()
    except Exception as e:
        print(f"\n  ERROR: Cannot connect to Alpaca: {e}")
        return

    # --- STEP 1: SYNC pending entry orders ---
    print(f"\n[1] SYNC: Verify pending entry orders with Alpaca ...")
    sync_pending(trader, positions, dry_run)

    # --- STEP 1.25: SYNC prior exit orders ---
    print(f"\n[1.25] SYNC: Verify prior sell orders with Alpaca ...")
    sync_exit_orders(trader, positions, dry_run)

    # --- STEP 1.5: RECONCILE local state with Alpaca (manual closes while away) ---
    print(f"\n[1.5] RECONCILE: Sync local state with Alpaca (manual closes) ...")
    reconcile_positions(trader, positions, dry_run)

    # --- STEP 2: CHECK active positions ---
    print(f"\n[2] CHECK: Active positions for stops + exits ...")
    check_active_positions(trader, positions, dry_run)

    # --- STEP 2.5: VERIFY same-run exits before replacements ---
    print(f"\n[2.5] VERIFY: Confirm same-run sell orders before replacement entries ...")
    exits_ok = verify_same_run_exits(trader, positions, dry_run)

    # --- STEP 3: PLACE new entries ---
    print(f"\n[3] ENTRY: Place new entries if slots available ...")
    if plan and exits_ok:
        place_new_entries(trader, plan, positions, dry_run)
    elif not exits_ok:
        print("  Replacement entries blocked because a sell order was not broker-verified.")
    else:
        print("  No plan.json — skipping entry placement.")

    # --- SAVE ---
    # A dry run must not mutate local state, including last_updated.
    if dry_run:
        print("  [DRY] State not saved.")
    else:
        save_positions(positions)

    # --- FINAL STATUS ---
    print_status(positions, plan)
    print(bar)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Slot manager + Alpaca paper executor")
    parser.add_argument("--status", action="store_true", help="Just check status, no trades")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen")
    args = parser.parse_args()
    main(status_only=args.status, dry_run=args.dry_run)
