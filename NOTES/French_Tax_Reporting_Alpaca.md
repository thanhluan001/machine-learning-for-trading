# French Tax Reporting for Alpaca Trades

## **1. Overview**
- **Requirement**: French tax law mandates **detailed records** of all trades, capital gains/losses, dividends, and corporate actions.
- **Penalties**: Failure to report accurately can result in **fines or audits**.
- **Solution**: Automate trade logging in your Alpaca codebase and export to tax software (e.g., Koinly).

---

## **2. What Must Be Recorded?**

### **A. Trade Log (Mandatory)**
| **Field**         | **Example**       | **Purpose**                                                                 |
|-------------------|-------------------|-----------------------------------------------------------------------------|
| Date              | `2024-05-20`      | Determines tax year and holding period.                                    |
| Symbol            | `NVDA`            | Identifies the traded asset.                                               |
| Action            | `BUY`/`SELL`/`SHORT` | Distinguishes between purchase, sale, and short positions.               |
| Quantity          | `10`              | Number of shares/units traded.                                             |
| Price (USD)       | `850.00`          | Cost basis for capital gains calculation.                                  |
| Fees (USD)        | `1.50`            | Reduces taxable gain (deducted from proceeds).                             |
| Exchange Rate     | `0.93`            | Converts USD-denominated trades to EUR (official ECB rate recommended).   |
| Total (EUR)       | `€7,905.00`       | Final taxable amount in EUR.                                               |

### **B. Dividends**
| **Field**             | **Example**       | **Purpose**                                                                 |
|-----------------------|-------------------|-----------------------------------------------------------------------------|
| Date                  | `2024-05-15`      | Date dividend was received.                                                |
| Symbol                | `AAPL`            | Stock paying the dividend.                                                 |
| Gross Amount (USD)    | `$50.00`          | Total dividend before withholding tax.                                     |
| Withholding Tax (US)  | `$7.50` (15%)     | US tax deducted at source (creditable in France).                         |
| Net Amount (EUR)      | `€39.53`          | Dividend amount after US tax and FX conversion.                           |

### **C. Corporate Actions**
- **Stock splits** (e.g., NVDA 4:1 split).
- **Mergers/spin-offs** (e.g., ARM spin-off from NVDA).
- **Adjust cost basis** for existing positions.

---

## **3. How to Automate Logging in Alpaca**

### **A. Trade Logging Function**
```python
import pandas as pd
from datetime import datetime

# Path to trade log CSV
TRADE_LOG_FILE = "C:/Users/thanh/Projects/machine-learning-for-trading/NOTES/trade_log.csv"

def log_trade(symbol, action, qty, price_usd, fees_usd, exchange_rate):
    """Log a trade to CSV for tax reporting."""
    total_eur = (qty * price_usd + (fees_usd if action == "SELL" else -fees_usd)) * exchange_rate
    trade_data = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "symbol": symbol,
        "action": action,
        "qty": qty,
        "price_usd": price_usd,
        "fees_usd": fees_usd,
        "exchange_rate": exchange_rate,
        "total_eur": round(total_eur, 2),
    }
    
    # Append to CSV
    df = pd.DataFrame([trade_data])
    df.to_csv(
        TRADE_LOG_FILE,
        mode="a",
        header=not pd.io.common.file_exists(TRADE_LOG_FILE),
        index=False,
    )
```

### **B. Example Usage**
```python
# Buy 10 NVDA shares @ $850 with $1.50 fees
log_trade(
    symbol="NVDA",
    action="BUY",
    qty=10,
    price_usd=850.00,
    fees_usd=1.50,
    exchange_rate=0.93,
)

# Short 5 GME shares @ $15 with $0.50 fees
log_trade(
    symbol="GME",
    action="SHORT",
    qty=5,
    price_usd=15.00,
    fees_usd=0.50,
    exchange_rate=0.93,
)
```

### **C. Output CSV**
| date       | symbol | action | qty | price_usd | fees_usd | exchange_rate | total_eur |
|------------|--------|--------|-----|-----------|----------|---------------|-----------|
| 2024-05-20 | NVDA   | BUY    | 10  | 850.00    | 1.50     | 0.93          | 7905.00   |
| 2024-05-21 | GME    | SHORT  | 5   | 15.00     | 0.50     | 0.93          | 70.13     |

---

## **4. Tax Calculations**

### **A. Capital Gains/Losses**
**Formula (per trade):**
```
Gain/Loss (EUR) = (Sell Price - Buy Price) * Qty * Exchange Rate - Fees
```
**Example:**
- Buy 10 NVDA @ $850, Sell @ $900:
```
Gain = (900 - 850) * 10 * 0.93 - 1.50 = €463.50
```

### **B. Dividend Income**
**Formula:**
```
Net Dividend (EUR) = Gross Dividend (USD) * (1 - US Withholding Tax) * Exchange Rate
```

### **C. Cost Basis Methods**
- **FIFO (First-In-First-Out)**: Default method (easiest to automate).
- **Average Cost Basis**: Useful for frequent traders (requires more bookkeeping).

---

## **5. Integration with Tax Software**

### **A. Export to Koinly/CoinTracking**
1. Generate CSV from your trade log.
2. Map columns to tax software fields:
   - `date` → "Date"
   - `action` → "Type" (Buy/Sell/Short)
   - `symbol` → "Symbol"
   - `qty` → "Amount"
   - `total_eur` → "Total (EUR)"

### **B. Direct API Connection**
- **Koinly** supports Alpaca API → automatic sync.

---

## **6. Compliance Checklist**

### **A. Record-Keeping**
- Save all trade logs **for 10+ years** (French tax authorities can audit retroactively).
- Store files securely (e.g., Google Drive, encrypted backup).

### **B. Reporting**
- **Annual Tax Return (Form 2042)**: Report capital gains/losses and dividends.
- **Zero Gains**: Must still be reported if you traded.

### **C. Currency Exchange Rates**
- Use **ECB’s daily EUR/USD rate** for consistency (avoid disputes).

### **D. Corporate Actions**
- Adjust cost basis for **splits, mergers, spin-offs** (Alpaca API provides this data).

---

## **7. Full Automation Workflow**
1. **Trade Execution** → Log trade instantly via `log_trade()`.
2. **End of Month** → Generate **realized gains/losses report**.
3. **December** → Run **tax-loss harvesting** and export full-year CSV.
4. **April/May** → Import into **Koinly** and file taxes.

---

## **8. Gain Calculation: Year-End Approach**

### **A. Why Calculate Gains at Year-End?**
- **Simpler Code**: No need to track cost basis in real-time.
- **French Tax Compliance**: Report **annual** capital gains/losses (not per-trade).
- **Audit-Friendly**: All trades and calculations documented in CSV.

---

### **B. How to Calculate Gains at Year-End**

#### **Step 1: Trade Logging (Real-Time)**
- Log **every trade** (buy/sell/short) to `trade_log.csv` immediately after execution.
- Example CSV:
```csv
date,symbol,action,qty,price_usd,fees_usd,exchange_rate,total_eur
2024-01-15,NVDA,BUY,5,500.00,1.50,0.93,2326.58
2024-02-20,NVDA,BUY,3,550.00,1.50,0.93,1530.18
2024-03-10,NVDA,SELL,4,600.00,2.00,0.93,2230.80
```

#### **Step 2: Year-End Script (December)**
- Run a Python script to:
  1. **Group trades by symbol** (e.g., all NVDA trades).
  2. **Apply FIFO or average cost basis** to calculate gains.
- Example script:
```python
import pandas as pd

# Load trade log
trades = pd.read_csv("trade_log.csv")

# Initialize results
capital_gains = []

# Group by symbol and process sells
for symbol, group in trades.groupby("symbol"):
    buys = group[group["action"] == "BUY"]
    sells = group[group["action"] == "SELL"]
    
    # Skip if no sells
    if len(sells) == 0:
        continue
    
    # Apply FIFO: match sells to oldest buys first
    buy_queue = buys.copy().reset_index(drop=True)
    sell_queue = sells.copy().reset_index(drop=True)
    
    for _, sell in sell_queue.iterrows():
        remaining_qty = sell["qty"]
        
        for _, buy in buy_queue.iterrows():
            if remaining_qty <= 0:
                break
            
            if buy["qty"] <= remaining_qty:
                # Full buy lot sold
                gain = sell["total_eur"] - buy["total_eur"]
                capital_gains.append({
                    "symbol": symbol,
                    "sell_date": sell["date"],
                    "qty": buy["qty"],
                    "sale_proceeds_eur": sell["total_eur"],
                    "cost_basis_eur": buy["total_eur"],
                    "capital_gain_eur": gain,
                    "method": "FIFO"
                })
                remaining_qty -= buy["qty"]
                buy_queue.loc[buy.name, "qty"] = 0  # Mark buy as fully sold
            else:
                # Partial buy lot sold
                gain = (remaining_qty / buy["qty"]) * (sell["total_eur"] - buy["total_eur"])
                capital_gains.append({
                    "symbol": symbol,
                    "sell_date": sell["date"],
                    "qty": remaining_qty,
                    "sale_proceeds_eur": (remaining_qty / buy["qty"]) * sell["total_eur"],
                    "cost_basis_eur": (remaining_qty / buy["qty"]) * buy["total_eur"],
                    "capital_gain_eur": gain,
                    "method": "FIFO"
                })
                buy_queue.loc[buy.name, "qty"] -= remaining_qty
                remaining_qty = 0

# Export capital gains report
gains_df = pd.DataFrame(capital_gains)
gains_df.to_csv("capital_gains_2024.csv", index=False)
print(gains_df)
```

#### **Step 3: Output (Capital Gains Report)**
Example `capital_gains_2024.csv`:
```csv
symbol,sell_date,qty,sale_proceeds_eur,cost_basis_eur,capital_gain_eur,method
NVDA,2024-03-10,4,2230.80,2326.58,-95.78,FIFO
``` *(Adjust for **average cost basis** by averaging buy prices before sells.)*

---

### **C. Handling Short Sales**
- Log shorts as **negative buys**:
```csv
date,symbol,action,qty,price_usd,fees_usd,exchange_rate,total_eur
2024-01-20,GME,SHORT,100,15.00,1.50,0.93,-1396.50
```
- Calculate gains/losses when **covering** (buying back):
```python
shorts = group[group["action"] == "SHORT"]
covers = group[group["action"] == "COVER"]
```
- Formula: `(COVER price - SHORT price) * qty`.

---

### **D. Handling Dividends**
- Add a `DIVIDEND` row to `trade_log.csv`:
```csv
date,symbol,action,qty,price_usd,fees_usd,exchange_rate,total_eur
2024-03-15,AAPL,DIVIDEND,0,0.22,0.03,0.93,19.22
```
- Sum dividends at year-end for **Form 2042** (French tax return).

---

## **9. When to Run the Script?**
| **Action**          | **Timing**               | **Why?**                                      |
|----------------------|--------------------------|-----------------------------------------------|
| Record trades        | Immediately after trade  | Avoid missing trades.                        |
| Calculate gains      | December 31 (year-end)   | French tax year = calendar year.             |
| File taxes           | April/May                | French tax deadline.                          |

---

## **10. Key Takeaways**
- **Log all trades** to CSV immediately (use `log_trade()` function).
- **Calculate gains/losses at year-end** (December) using FIFO or average cost basis.
- **Export a capital gains report** for French tax filing (Form 2042).
- **Use ECB exchange rates** for EUR conversions (audit-proof).
- **Keep records for 10+ years** (French tax authorities may audit retroactively).

**Next Steps:**
- Extend the script to handle **shorts and dividends**.
- Add **error handling** (e.g., missing data, partial fills).
- Integrate with **Koinly API** for direct sync.

**Next Steps:**
- Extend the script to track **dividends** and **corporate actions**.
- Add **FIFO/Average cost basis** calculations.
- Integrate with **Koinly API** for direct sync.