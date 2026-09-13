#!/usr/bin/env python3
"""03_portfolio_report.py — Full portfolio PnL report from Alpaca transactions.

Replaces the thin Alpaca dashboard: reconstructs the entire trade ledger from
broker-side account activities (FILL / FEE / JNLC cash journal), computes
round-trip realized PnL per trade via FIFO lot matching, unrealized PnL for
open positions, and whole-account PnL against net deposits.

WHAT IT ANSWERS
---------------
- How much money was deposited / withdrawn?
- Realized PnL per completed trade (win/loss, holding days)
- Aggregate stats: win rate, avg win/loss, payoff, profit factor, expectancy
- Unrealized PnL of open positions (mark to live quote)
- Fees paid, trading volume, turnover
- Total account return = (equity - net deposits) / net deposits
- Cross-check: broker-derived closed trades vs local positions.json
- Daily equity + PnL curve from portfolio history

SOURCE OF TRUTH
---------------
Everything is computed from Alpaca's own activity records — not positions.json
— so manual trades, expired orders, and partial fills are all captured. The
local ledger is shown only as a reconciliation line.

USAGE
-----
    python 05b_alpaca_live/03_portfolio_report.py            # full report
    python 05b_alpaca_live/03_portfolio_report.py --days 30  # daily PnL, last 30d

No orders are placed. Read-only.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
except (AttributeError, RuntimeError):
    pass

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from alpaca_client import AlpacaClient  # noqa: E402

from alpaca.trading.requests import GetPortfolioHistoryRequest, GetCalendarRequest  # noqa: E402

POSITIONS_FILE = HERE / "positions.json"


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------

def fetch_activities(client) -> list[dict]:
    """All account activities, oldest first, paginated via page_token."""
    out: list[dict] = []
    token = None
    while True:
        params = {"direction": "desc", "page_size": 100}
        if token:
            params["page_token"] = token
        batch = client.get("/account/activities", data=params)
        if not batch:
            break
        out.extend(batch if isinstance(batch, list) else batch)
        if len(batch) < 100:
            break
        token = batch[-1].get("id")
    out.reverse()  # oldest first for chronological FIFO matching
    return out


def parse_time(s) -> datetime:
    if isinstance(s, datetime):
        return s
    return pd.Timestamp(s).to_pydatetime()


def latest_price(client, symbol: str) -> float | None:
    """Latest trade or quote price for an open position."""
    try:
        r = client.get(f"/v2/stocks/{symbol}/trades/latest",
                       data={"feed": "iex"}, base_url="https://data.alpaca.markets")
        return float(r["trade"]["p"])
    except Exception:
        pass
    try:
        q = client.get(f"/v2/stocks/{symbol}/quotes/latest",
                       data={"feed": "iex"}, base_url="https://data.alpaca.markets")
        return float(q["quote"]["ap"] or q["quote"]["bp"] or 0) or None
    except Exception:
        return None


# ----------------------------------------------------------------------
# Ledger construction
# ----------------------------------------------------------------------

def build_ledger(activities: list[dict]) -> dict:
    fills: list[dict] = []
    fees = 0.0
    deposits = withdrawals = 0.0

    for a in activities:
        kind = a.get("activity_type")
        if kind == "FILL":
            fills.append({
                "time": parse_time(a.get("transaction_time") or a.get("timestamp")),
                "symbol": a.get("symbol"),
                "side": a.get("side"),
                "qty": float(a.get("qty") or 0),
                "price": float(a.get("price") or 0),
                "notional": float(a.get("qty") or 0) * float(a.get("price") or 0),
            })
        elif kind == "FEE":
            fees += abs(float(a.get("amount") or 0))
        elif kind in ("JNLC", "TRANS", "ACATC", "ACATS", "CASH"):
            net = float(a.get("net_amount") or 0)
            if net > 0:
                deposits += net
            else:
                withdrawals += abs(net)

    # FIFO lot matching per symbol -> round-trip trades + open lots
    lots: dict[str, list] = defaultdict(list)   # open buy lots: {time, qty, price}
    shorts: dict[str, list] = defaultdict(list) # open sell lots (if ever short)
    trades: list[dict] = []
    gross_volume = 0.0

    for f in sorted(fills, key=lambda x: x["time"]):
        gross_volume += f["notional"]
        if f["side"] == "buy":
            # first close any open short
            remaining = f["qty"]
            while remaining > 1e-9 and shorts[f["symbol"]]:
                s = shorts[f["symbol"]][0]
                take = min(remaining, s["qty"])
                trades.append(round_trip(f["symbol"], "cover", s, f, take))
                s["qty"] -= take
                remaining -= take
                if s["qty"] <= 1e-9:
                    shorts[f["symbol"]].pop(0)
            if remaining > 1e-9:
                lots[f["symbol"]].append({**f, "qty": remaining})
        else:  # sell
            remaining = f["qty"]
            while remaining > 1e-9 and lots[f["symbol"]]:
                b = lots[f["symbol"]][0]
                take = min(remaining, b["qty"])
                trades.append(round_trip(f["symbol"], "sell", b, f, take))
                b["qty"] -= take
                remaining -= take
                if b["qty"] <= 1e-9:
                    lots[f["symbol"]].pop(0)
            if remaining > 1e-9:  # naked short (shouldn't happen in this bot)
                shorts[f["symbol"]].append({**f, "qty": remaining})

    open_lots = [dict(l, symbol=sym) for sym, ls in lots.items() for l in ls]
    return {"fills": fills, "trades": trades, "open_lots": open_lots,
            "fees": fees, "deposits": deposits, "withdrawals": withdrawals,
            "gross_volume": gross_volume}


def aggregate_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """Merge multi-fill FIFO fragments into one row per trade.

    Broker fills arrive one row per partial fill, so a single logical trade
    (e.g. 7 shares filled 6+1) becomes several round-trips with identical
    entry/exit prices. Aggregate by symbol + entry day + exit day.
    """
    if trades.empty:
        return trades
    trades = trades.copy()
    trades["entry_time"] = pd.to_datetime(trades["entry_time"])
    trades["exit_time"] = pd.to_datetime(trades["exit_time"])
    trades["entry_day"] = trades["entry_time"].dt.normalize()
    trades["exit_day"] = trades["exit_time"].dt.normalize()
    agg = (trades.groupby(["symbol", "kind", "entry_day", "exit_day"], as_index=False)
           .agg(qty=("qty", "sum"), entry_price=("entry_price", "first"),
                exit_price=("exit_price", "first"), pnl=("pnl", "sum"),
                entry_time=("entry_time", "min"), exit_time=("exit_time", "max")))
    agg["pnl_pct"] = agg.apply(
        lambda r: r.pnl / (r.entry_price * r.qty) * 100 if r.qty else 0.0, axis=1)
    agg["hold_days"] = (agg["exit_time"] - agg["entry_time"]).dt.total_seconds() / 86400
    return agg.sort_values("exit_time").reset_index(drop=True)


def round_trip(symbol: str, kind: str, open_lot: dict, close_fill: dict, qty: float) -> dict:
    pnl = (close_fill["price"] - open_lot["price"]) * qty * (1 if kind == "sell" else -1)
    days = (close_fill["time"] - open_lot["time"]).total_seconds() / 86400
    return {"symbol": symbol, "kind": kind, "qty": qty,
            "entry_time": open_lot["time"], "exit_time": close_fill["time"],
            "entry_price": open_lot["price"], "exit_price": close_fill["price"],
            "pnl": pnl, "pnl_pct": pnl / (open_lot["price"] * qty) * 100 if qty else 0.0,
            "hold_days": days}


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------

def fmt_money(x): return f"{'+' if x >= 0 else ''}${x:,.2f}"
def fmt_pct(x):   return f"{'+' if x >= 0 else ''}{x:.2f}%"


def print_header(title):
    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)


def fetch_trading_days(client, start, end) -> set:
    """Exchange trading dates (excludes weekends AND holidays) as a set."""
    days = set()
    cursor = start
    # calendar API caps range length; walk in 1-year chunks
    while cursor <= end:
        chunk_end = min(end, cursor + pd.Timedelta(days=360))
        cal = client.get_calendar(filters=GetCalendarRequest(
            start=cursor.strftime("%Y-%m-%d"), end=chunk_end.strftime("%Y-%m-%d")))
        days.update(pd.Timestamp(d.date).normalize() for d in cal)
        cursor = chunk_end + pd.Timedelta(days=1)
    return days


def hold_trading_days(trading_days: set, entry, exit) -> int:
    """Trading dates strictly after the entry day, up to and including exit.

    Matches the T+5 contract: enter at Close[T], exit at Close[T+5] -> 5.
    """
    e = pd.Timestamp(entry).tz_localize(None).normalize()
    x = pd.Timestamp(exit).tz_localize(None).normalize()
    return sum(1 for d in trading_days if e < d <= x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=0, help="show daily PnL for last N days")
    args = ap.parse_args()

    t = AlpacaClient()
    print_header("PORTFOLIO REPORT — Alpaca Paper (broker-derived, read-only)")
    acct = t.verify_connection()

    acts = fetch_activities(t.client)
    L = build_ledger(acts)
    trades = aggregate_trades(pd.DataFrame(L["trades"]))
    fees, net_deposits = L["fees"], L["deposits"] - L["withdrawals"]

    # trading calendar for honest hold durations (calendar days overstate
    # 5-trading-day holds by weekends/holidays)
    all_times = [f["time"] for f in L["fills"]] or [datetime.now(timezone.utc)]
    trading_days = fetch_trading_days(t.client, min(all_times), max(all_times))
    if len(trades):
        trades["hold_td"] = trades.apply(
            lambda r: hold_trading_days(trading_days, r.entry_time, r.exit_time), axis=1)

    # fetch open positions first so section 1 can report open PnL correctly
    positions = t.client.get_all_positions()
    total_open = sum(float(p.unrealized_pl) for p in positions)
    market_value = sum(float(p.market_value) for p in positions)

    # ---- account PnL vs deposits --------------------------------------
    print_header("[1] ACCOUNT P&L vs DEPOSITS")
    equity = acct["portfolio_value"]
    cash = acct["cash"]
    realized = trades["pnl"].sum() if len(trades) else 0.0
    print(f"  Net deposits:          {fmt_money(net_deposits)}")
    print(f"  Current equity:        {fmt_money(equity)}  (cash {fmt_money(cash)} + positions {fmt_money(market_value)})")
    print(f"  Total PnL:             {fmt_money(equity - net_deposits)}"
          f"   ({fmt_pct((equity / net_deposits - 1) * 100) if net_deposits else 'n/a'})")
    print(f"  Realized PnL (FIFO):   {fmt_money(realized)}")
    print(f"  Unrealized PnL (mark): {fmt_money(total_open)}")
    print(f"  Fees paid:             ${fees:,.2f}")
    print(f"  Fills: {len(L['fills'])} | Trades: {len(trades)} | "
          f"Gross volume: {fmt_money(L['gross_volume'])} | Activities: {len(acts)}")

    # ---- closed trades -------------------------------------------------
    print_header("[2] CLOSED TRADES (FIFO round-trips from broker fills)")
    if len(trades):
        trades_show = trades.sort_values("exit_time")
        print(f"  {'symbol':<7}{'side':<7}{'qty':>6}{'entry':>9}{'exit':>9}"
              f"{'PnL$':>10}{'PnL%':>8}{'hold':>7}{'(cal)':>7}  exit_date")
        for _, r in trades_show.iterrows():
            print(f"  {r.symbol:<7}{r.kind:<7}{r.qty:>6g}{r.entry_price:>9.2f}"
                  f"{r.exit_price:>9.2f}{r.pnl:>10.2f}{r.pnl_pct:>7.2f}%"
                  f"{int(r.hold_td):>6}d{r.hold_days:>6.1f}d  {r.exit_time:%Y-%m-%d}")
        # aggregate stats
        wins = trades[trades.pnl > 0]
        losses = trades[trades.pnl <= 0]
        gross_win = wins.pnl.sum()
        gross_loss = abs(losses.pnl.sum())
        print(f"\n  Trades: {len(trades)} | W {len(wins)} / L {len(losses)}"
              f"  → win rate {len(wins)/len(trades)*100:.1f}%")
        print(f"  Total realized:  {fmt_money(realized)}")
        if len(wins):
            print(f"  Avg win:         {fmt_money(gross_win/len(wins))}"
                  f" (best {fmt_money(trades.pnl.max())} {trades.loc[trades.pnl.idxmax(),'symbol']})")
        if len(losses):
            print(f"  Avg loss:        {fmt_money(-gross_loss/len(losses))}"
                  f" (worst {fmt_money(trades.pnl.min())} {trades.loc[trades.pnl.idxmin(),'symbol']})")
        payoff = (gross_win/len(wins)) / (gross_loss/len(losses)) if len(wins) and len(losses) and gross_loss else float('nan')
        pf = gross_win / gross_loss if gross_loss else float('inf')
        print(f"  Payoff ratio:    {payoff:.2f} | Profit factor: {pf:.2f} "
              f"| Expectancy/trade: {fmt_money(realized/len(trades))}")
        by_sym = trades.groupby("symbol").agg(n=("pnl","size"), pnl=("pnl","sum"),
                                              win=("pnl", lambda s:(s>0).mean()*100))
        print("\n  By symbol:")
        for sym, r in by_sym.sort_values("pnl", ascending=False).iterrows():
            print(f"    {sym:<7} n={int(r.n):<3} PnL {fmt_money(r.pnl):>12}  win {r.win:.0f}%")
    else:
        print("  (no completed round-trips yet)")

    # ---- open positions ------------------------------------------------
    print_header("[3] OPEN POSITIONS (marked to latest trade)")
    if positions:
        print(f"  {'symbol':<7}{'qty':>6}{'avg':>9}{'last':>9}{'mkt val':>11}"
              f"{'unrl PnL$':>11}{'unrl PnL%':>10}  lots(FIFO)")
        for p in positions:
            last = float(p.current_price) or float(p.avg_entry_price)
            upnl = float(p.unrealized_pl)
            sym_lots = [l for l in L["open_lots"] if l["symbol"] == p.symbol]
            lot_str = ",".join(f"{l['qty']:g}@{l['price']:.2f}" for l in sym_lots) or "broker-only"
            print(f"  {p.symbol:<7}{float(p.qty):>6g}{float(p.avg_entry_price):>9.2f}"
                  f"{last:>9.2f}{float(p.market_value):>11.2f}{upnl:>11.2f}"
                  f"{float(p.unrealized_plpc)*100:>9.2f}%  {lot_str}")
    else:
        print("  (flat — no open positions)")
    print(f"\n  Total unrealized: {fmt_money(total_open)}")

    # ---- reconcile with local ledger ------------------------------------
    print_header("[4] RECONCILIATION vs LOCAL positions.json")
    if POSITIONS_FILE.exists():
        local = json.loads(POSITIONS_FILE.read_text())
        local_closed = local.get("closed", [])
        local_active = local.get("active", [])
        local_filled = [c for c in local_closed
                        if (c.get("fill_status") in ("filled", "partial") or c.get("entry_price"))]
        local_expired = [c for c in local_closed if c not in local_filled]
        print(f"  Broker round-trips: {len(trades)} | Local closed (filled): {len(local_filled)}"
              f" | Local closed (expired, 0 fills): {len(local_expired)}"
              f" | Local active: {len(local_active)}")
        local_pnl = sum((c.get("return_pct") or 0) for c in local_filled)
        print(f"  Local filled-return_pct sum: {local_pnl:+.2f}% "
              f"(percent units; broker PnL is dollars — sanity check only)")
        if len(trades) != len(local_filled):
            print("  NOTE: counts differ — check for manual trades / pre-ledger fills.")
    else:
        print("  positions.json not found.")

    # ---- equity curve ----------------------------------------------------
    days = args.days or 30
    print_header(f"[5] DAILY EQUITY & PnL (last {days} calendar days)")
    try:
        ph = t.client.get_portfolio_history(
            history_filter=GetPortfolioHistoryRequest(period=f"{days}D", timeframe="1D"))
        if ph and getattr(ph, "timestamp", None):
            df = pd.DataFrame({"ts": pd.to_datetime(ph.timestamp, unit="s"),
                               "equity": ph.equity,
                               "cum_pnl": ph.profit_loss})
            df = df[df.equity.notna() & (df.equity > 0)].reset_index(drop=True)
            if len(df):
                df["day_pnl"] = df.equity.diff().fillna(df.cum_pnl)
                base = df.equity.iloc[0]
                print(f"  {'date':<12}{'equity':>11}{'day PnL':>11}{'since-start':>13}")
                for _, r in df.iterrows():
                    print(f"  {r.ts:%Y-%m-%d} {r.equity:>11,.2f}{r.day_pnl:>+11.2f}"
                          f"{(r.equity/base-1)*100:>+12.2f}%")
                peak = df.equity.cummax()
                mdd = ((df.equity / peak - 1).min()) * 100
                print(f"\n  Period return: {fmt_pct((df.equity.iloc[-1]/base-1)*100)}"
                      f" | Max drawdown: {mdd:.2f}%")
    except Exception as e:
        print(f"  (portfolio history unavailable: {e})")

    print_header("END — read-only report; no orders placed")


if __name__ == "__main__":
    main()
