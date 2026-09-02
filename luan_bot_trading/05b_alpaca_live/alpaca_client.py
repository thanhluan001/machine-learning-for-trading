#!/usr/bin/env python3
"""
alpaca_client.py — Alpaca Paper Trading Client Wrapper
=======================================================

Thin wrapper around alpaca-py TradingClient with built-in VERIFY-CHECKS.
Every interaction with Alpaca goes through a verify function that confirms
the API response before proceeding.

VERIFY-CHECK PATTERN (per user requirement):
  Before doing ANYTHING, sync local state with Alpaca's actual state.
  Never trust local JSON — always verify against the broker.

VERIFY CHECKS:
  [V1] verify_connection()    — API reachable? Get account.
  [V2] verify_market_status() — Is market open? Next close?
  [V3] verify_order_filled()  — Did this order actually fill?
  [V4] verify_position()      — Does Alpaca actually hold this position?
  [V5] verify_buying_power()  — Enough cash for the next order?
"""
from __future__ import annotations

import io
import sys
import os
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, RuntimeError):
    pass

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")


class AlpacaClient:
    """Paper trading client with verify-checks."""

    def __init__(self):
        if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
            raise ValueError("ALPACA_API_KEY / ALPACA_SECRET_KEY not in .env")
        self.client = TradingClient(
            api_key=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY,
            paper=True,
        )

    # ==================================================================
    # VERIFY-CHECKS
    # ==================================================================

    def verify_connection(self) -> dict:
        """[V1] Confirm API is reachable. Returns account dict.

        Raises RuntimeError if connection fails.
        """
        acct = self.client.get_account()
        info = {
            "id": str(acct.id),
            "status": acct.status,
            "cash": float(acct.cash),
            "portfolio_value": float(acct.portfolio_value),
            "buying_power": float(acct.buying_power),
            "equity": float(acct.equity),
        }
        print(f"  [V1] Connection OK — account={info['id'][:8]}... "
              f"NAV=${info['portfolio_value']:,.0f} cash=${info['cash']:,.0f}")
        return info

    def verify_market_status(self) -> dict:
        """[V2] Check if market is open. Returns {is_open, next_close}."""
        clock = self.client.get_clock()
        info = {
            "is_open": clock.is_open,
            "timestamp": str(clock.timestamp),
            "next_open": str(clock.next_open) if clock.next_open else None,
            "next_close": str(clock.next_close) if clock.next_close else None,
        }
        status = "OPEN" if info["is_open"] else "CLOSED"
        print(f"  [V2] Market {status}"
              + (f" (next close: {info['next_close']})" if info["is_open"] else ""))
        return info

    def verify_order_filled(self, order_id: str, quiet: bool = False) -> dict:
        """[V3] Check if an order actually filled. Returns fill details.

        Returns:
            {status, filled_qty, filled_avg_price, is_filled}
        """
        order = self.client.get_order_by_id(order_id)
        # alpaca-py returns OrderStatus enum; use .value for clean string
        raw_status = order.status
        status = raw_status.value if hasattr(raw_status, 'value') else str(raw_status)
        status_lower = status.lower()
        is_filled = status_lower == "filled"
        info = {
            "order_id": order_id,
            "status": status,
            "is_filled": is_filled,
            "filled_qty": float(order.filled_qty) if order.filled_qty else 0.0,
            "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else None,
            "is_canceled": status_lower in ("canceled", "rejected", "expired", "cancelled"),
        }
        if not quiet:
            if is_filled:
                print(f"  [V3] Order {order_id[:8]}... FILLED "
                      f"@ ${info['filled_avg_price']:.2f} × {info['filled_qty']}")
            elif info["is_canceled"]:
                print(f"  [V3] Order {order_id[:8]}... {status.upper()}")
            else:
                print(f"  [V3] Order {order_id[:8]}... {status}")
        return info

    def poll_order_fill(self, order_id: str, timeout: float = 10.0,
                        interval: float = 2.0) -> dict:
        """Poll an order until it reaches a terminal state or timeout (~10s).

        Market sells usually fill in 1-3s; this lets the caller fast-path a
        same-run finalization for the common full-fill case instead of waiting
        for the next run. Polls quietly (no per-poll [V3] lines) and returns
        the final fill dict from verify_order_filled.
        """
        import time
        fill = self.verify_order_filled(order_id, quiet=True)
        deadline = time.time() + timeout
        while time.time() < deadline and not (fill.get("is_filled") or fill.get("is_canceled")):
            time.sleep(interval)
            fill = self.verify_order_filled(order_id, quiet=True)
        return fill

    def verify_position(self, symbol: str) -> dict | None:
        """[V4] Check actual Alpaca position for a symbol.

        Returns None if no position exists.
        """
        try:
            pos = self.client.get_open_position(symbol)
            info = {
                "symbol": symbol,
                "qty": float(pos.qty),
                "side": str(pos.side),
                "avg_entry_price": float(pos.avg_entry_price),
                "market_value": float(pos.market_value),
                "unrealized_pl": float(pos.unrealized_pl),
                "unrealized_plpc": float(pos.unrealized_plpc),
                "current_price": float(pos.current_price),
            }
            print(f"  [V4] Position {symbol}: {info['qty']} shares "
                  f"@ ${info['avg_entry_price']:.2f}, "
                  f"P&L ${info['unrealized_pl']:+.2f} ({info['unrealized_plpc']*100:+.1f}%)")
            return info
        except Exception:
            print(f"  [V4] Position {symbol}: NONE")
            return None

    def verify_buying_power(self, required: float = 0) -> bool:
        """[V5] Check if we have enough buying power."""
        acct = self.client.get_account()
        bp = float(acct.buying_power)
        ok = bp >= required
        print(f"  [V5] Buying power: ${bp:,.0f} "
              f"({'OK' if ok else 'INSUFFICIENT'} for ${required:,.0f})")
        return ok

    # ==================================================================
    # ACTIONS
    # ==================================================================

    def get_current_price(self, symbol: str) -> float | None:
        """Get latest price for a symbol.

        Tries open position first, then latest trade fallback.
        get_stock_latest_trade returns a DICT keyed by symbol, not a Trade.
        """
        # Try from position first (has current_price)
        try:
            pos = self.client.get_open_position(symbol)
            return float(pos.current_price)
        except Exception:
            pass
        # Fallback: latest trade
        try:
            from alpaca.data.requests import StockLatestTradeRequest
            from alpaca.data.historical.stock import StockHistoricalDataClient
            dc = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
            req = StockLatestTradeRequest(symbol_or_symbols=symbol)
            trade = dc.get_stock_latest_trade(req)
            # alpaca-py returns dict {symbol: Trade} for a single string too
            if isinstance(trade, dict):
                t = trade.get(symbol)
                return float(t.price) if t is not None else None
            return float(trade.price)
        except Exception:
            return None

    def place_market_buy(self, symbol: str, qty: int, client_order_id: str = None) -> str | None:
        """Place an immediate DAY market BUY order (TimeInForce.DAY).

        This is the PAPER-trading configuration. Alpaca paper simulates CLS as a
        market order at the close with intentionally injected partial fills and
        delivers no auction-price fidelity, so DAY market is correct for paper
        (reliable fills, no fidelity lost). For LIVE, the buy order type is
        deferred to promotion time: real MOC may be preferable (rare partials,
        true auction price) if the account has elite smart-router routing. See
        Design.md §17.C.6.

        IMPORTANT timing difference (under DAY-market execution): unlike MOC
        (which fills at the close regardless of submission time), a DAY market
        order fills at script-run time. To approximate the Close[T] entry the
        backtest assumes, Script 02 should run near the close (~3:45 PM ET),
        NOT at market open — otherwise entries fill in the morning, ~6h before
        the intended close entry. If live reverts to MOC, this run-timing
        discipline relaxes.
        """
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,  # immediate market order
            client_order_id=client_order_id,
        )
        order = self.client.submit_order(req)
        order_id = str(order.id)
        print(f"  [BUY] {symbol} × {qty} → MARKET (DAY) order {order_id[:8]}...")
        return order_id

    def place_market_sell(self, symbol: str, qty: int, client_order_id: str = None) -> str | None:
        """Place an immediate DAY market SELL order (TimeInForce.DAY).

        Sells use an immediate market order rather than MOC for two reasons:
          1. Fill certainty / downside protection — an unfilled sell exposes
             us to continued downside and locks the slot; an unfilled buy only
             costs upside. Sells must not linger unfilled.
          2. Capital + slot release — a same-day sell+buy pair needs the sell
             to free buying power before the (MOC) buy fires at the closing
             auction. A market sell fills now; an MOC sell would compete for
             the same closing cross as the buy and risk the buy failing.

        The exit is still SCHEDULED at T+5 (stop checked on days 1+); only the
        order type changes. The script runs near the close, so the market fill
        tracks close closely. Returns order_id or None on failure.
        """
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,  # immediate market order
            client_order_id=client_order_id,
        )
        order = self.client.submit_order(req)
        order_id = str(order.id)
        print(f"  [SELL] {symbol} × {qty} → MARKET (DAY) order {order_id[:8]}...")
        return order_id

    def cancel_order(self, order_id: str):
        """Cancel an order."""
        self.client.cancel_order_by_id(order_id)
        print(f"  [CANCEL] order {order_id[:8]}...")

    def get_all_positions(self) -> list[dict]:
        """Get all open positions as list of dicts."""
        positions = self.client.get_all_positions()
        out = []
        for p in positions:
            out.append({
                "symbol": p.symbol,
                "qty": float(p.qty),
                "side": str(p.side),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc),
            })
        return out

    def get_last_sell_fill(self, symbol: str):  # -> (price, fill_date) or (None, None)
        """Get the filled average price of the most recent SELL order for a symbol.

        Used when reconciling a manually-closed position — we need the actual
        exit price the user sold at (they closed it on the Alpaca app while
        we were away). Returns None if no sell fill is found.
        """
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus, OrderSide
            req = GetOrdersRequest(
                status=QueryOrderStatus.CLOSED,
                symbols=[symbol],
                side=OrderSide.SELL,
                limit=10,
            )
            orders = self.client.get_orders(req)
            # Find most recent FILLED sell
            for order in orders:
                raw = order.status
                st = raw.value if hasattr(raw, 'value') else str(raw)
                if st.lower() == 'filled' and order.filled_avg_price:
                    ts = getattr(order, 'filled_at', None) or getattr(order, 'created_at', None)
                    fd = str(ts)[:10] if ts else None
                    return float(order.filled_avg_price), fd
            return None, None
        except Exception:
            return None, None
