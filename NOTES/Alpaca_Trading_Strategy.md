# Alpaca Trading Strategy: Aggressive Growth + Tactical Hedging

## **1. Overview**
- **Purpose**: High-conviction, growth-focused trading strategy for **9% of net worth** (outside PEA account).
- **Account Type**: Alpaca (US stocks, shorting allowed, API automation).
- **Risk Tolerance**: Aggressive but controlled (max 30% annual drawdown).
- **Tax Considerations**: French tax rules (30% flat tax on gains, loss carry-forward for 10 years).

---

## **2. Portfolio Construction**
### **A. Allocation Breakdown**
| **Bucket**               | **Weight** | **Strategy**                          | **Examples (2024)**               |
|--------------------------|------------|---------------------------------------|-----------------------------------|
| **High-Conviction Longs** | 60%        | Kelly-based sizing on 3–5 stocks      | NVDA, SMCI, META, CRWD, AMD       |
| **Sector Momentum**       | 20%        | Equal weight in top sector            | TSLA, ASML, ARM, AVGO, GOOGL      |
| **Short Positions**       | 10%        | Overvalued/weak stocks                | GME, BBBY, LCID                  |
| **Hedging**              | 10%        | SPY puts or gold (GLD)                | SPY 450 Puts, GLD                |

---

### **B. Position Sizing Rules**
- **Kelly Criterion**: Calculate edge (e.g., 60% win probability, 2:1 payout → Kelly = 20%).
- **Half-Kelly**: Use **50% of Kelly fraction** to reduce risk.
- **Max Position**: **20% of portfolio** (even if Kelly says 30%).

**Example**:
- If Kelly says 25% for NVDA → Allocate **12.5%**.
- If Kelly says -15% for GME → Short **7.5%**.

---

## **3. Tactical Execution**

### **A. High-Conviction Longs (60%)**
- **Selection Criteria**:
  - Strong growth trends (e.g., AI, cloud computing).
  - Momentum (RSI > 70, 50-day MA > 200-day MA).
  - Catalysts (earnings, FDA approvals, product launches).
- **Hold Period**: 6–12 months (unless thesis breaks).
- **Trim Winners**: If a stock exceeds **20% of portfolio**, trim back to **15%**.

---

### **B. Sector Momentum (20%)**
- **Strategy**: Pick the **top-performing sector** (e.g., semiconductors, fintech).
- **Equal Weights**: 4 stocks = **5% each**.
- **Rebalance**: Swap sectors every **3–6 months** based on performance.

**Example Sectors**:
| **Sector**   | **Top Stocks**                  |
|--------------|---------------------------------|
| AI           | NVDA, SMCI, AMD, ASML          |
| Fintech      | SQ, PYPL, COIN, UPST           |
| Biotech      | CRISPR, MRNA, REGN             |

---

### **C. Short Positions (10%)**
- **Selection**: Overvalued or weak stocks (e.g., meme stocks, failing businesses).
- **Risk Management**: Use **stop-losses** (e.g., cover if short rises 20%).
- **Avoid**: Crowded shorts (e.g., TSLA, AMZN).

**Example Shorts (2024)**:
- GME (fading meme stock hype)
- BBBY (bankruptcy risk)
- LCID (weak fundamentals)

---

### **D. Hedging (10%)**
- **Purpose**: Protect against market crashes.
- **Tools**:
  - **SPY puts** (e.g., SPY January 2025 450 Puts).
  - **Gold (GLD)** (inflation hedge).
  - **Inverse ETFs** (e.g., SQQQ for tech crashes).

---

## **4. Risk Management**

### **A. Drawdown Limits**
| **Scope**         | **Limit** | **Action**                          |
|-------------------|-----------|-------------------------------------|
| Total Portfolio   | -30%      | Reduce risk, reassess strategy      |
| Single Position   | -50%      | Sell half, reassess                 |
| Short Positions   | +20%      | Cover immediately                   |

### **B. Stop-Loss Rules**
- **Longs**: Sell **50% of position at -25%**, exit fully at -50%.
- **Shorts**: Cover if position rises **20%**.
- **Hedges**: Hold until expiry or profit target reached.

---

## **5. Rebalancing & Tax-Loss Harvesting**

### **A. Rebalancing Frequency**
- **Quarterly** (March, June, September, December).
- **Threshold-Based**: If a stock grows to **>20% of portfolio**, trim back to **15%**.

### **B. Tax-Loss Harvesting (France)**
- **Timing**: December (align with French tax year).
- **How**:
  1. Sell losing positions to realize losses.
  2. Offset gains (reduce taxable income).
  3. Repurchase similar assets (no wash-sale rule in France).

**Example**:
- Sell **AMD** (tech), buy **NVDA** (similar exposure).

---

## **6. Expected Outcomes**
| **Scenario** | **Annual Return** | **Volatility** | **Max Drawdown** |
|--------------|-------------------|----------------|------------------|
| Best Case    | +40–60%           | High           | -25%             |
| Base Case    | +20–30%           | Medium         | -15%             |
| Worst Case   | -20–30%           | High           | -50%             |

---

## **7. Tools to Execute This Strategy**
| **Tool**            | **Purpose**                          |
|---------------------|---------------------------------------|
| Alpaca API          | Automate trades                      |
| TradingView         | Momentum screeners, technical analysis|
| Portfolio Visualizer| Backtest strategies                  |
| Koinly              | Track trades for French tax reporting|

---

## **8. Final Notes**
- **Adaptability**: Adjust allocations based on market conditions (e.g., switch momentum sectors).
- **Flexibility**: Replace underperforming stocks quarterly.
- **Discipline**: Stick to stop-losses and drawdown limits.

**Python Script**: Automate strategy using Alpaca’s API (see separate file).

---

### **Key Takeaways**
1. **Kelly Criterion**: Optimize position sizes based on edge.
2. **Momentum + Growth**: Focus on high-performing sectors and stocks.
3. **Shorts + Hedging**: Protect against downturns.
4. **Rebalance Quarterly**: Lock in gains and harvest losses.
5. **Tax Efficiency**: TLH in December to offset gains.