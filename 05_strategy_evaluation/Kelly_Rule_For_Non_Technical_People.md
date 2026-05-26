# The Kelly Rule: A Simple Explanation for Non-Technical People

---

## **1. The Gambling Story That Started It All**

In 1956, a scientist named **John Kelly** working at Bell Labs (where the telephone was invented) was thinking about a curious problem:

*"If someone has a betting system that wins in the long run, but each individual bet is still uncertain, how much should they bet each time to make the most money possible without going broke?"*

This wasn't just theoretical. Kelly was inspired by a **quiz show scandal** in the 1950s called "The $64,000 Question." Someone on the West Coast was using the 3-hour time delay to get insider information about who would win, then placing huge bets. They were making a **fortune** because they had an **edge**—they knew more than the bookmakers.

Kelly wondered: *If you have an edge, what's the perfect bet size?*

---

## **2. The Core Insight: Growth vs. Safety**

Here's the key **trade-off**:

**Scenario A: Bet everything**
- If you win → you get rich FAST
- If you lose → you're broke (can't bet anymore)
- Problem: Even with a 60% win rate, you'll eventually lose 4 times in a row sometime and go bankrupt

**Scenario B: Bet tiny amounts**
- You'll never go broke
- But you'll make almost no money
- Your wealth grows at a snail's pace

**Kelly's breakthrough question**: What's the **sweet spot** that makes your money grow **as fast as possible** while still **avoiding bankruptcy**?

---

## **3. The Simple Intuition (No Math)**

Imagine you have a **magic coin** that lands heads 55% of the time (you have a 5% edge).

**What happens if you bet different amounts?**

| Bet % | Win 10 times in a row → Result | Lose 10 times in a row → Result |
|-------|-------------------------------|--------------------------------|
| 100% (all-in) | $1 becomes $2¹⁰ = $1,024 | $1 becomes $0 (ruined) |
| 50% | $1 becomes $1.5¹⁰ = $57.66 | $1 becomes $0.5¹⁰ = $0.001 (almost ruined) |
| 25% | $1 becomes $1.25¹⁰ = $9.31 | $1 becomes $0.75¹⁰ = $0.056 (survived but hurt) |
| **10%** | $1 becomes $1.1¹⁰ = $2.59 | $1 becomes $0.9¹⁰ = $0.35 (survived, grew modestly) |

**Kelly's insight**: If you bet too big, one bad streak wipes you out. If you bet too small, you leave money on the table. **There's an optimal middle**.

---

## **4. Kelly's Simple Formula (Conceptual Version)**

**For a simple bet with two outcomes:**

```
Edge = (Probability of winning × Payout) - (Probability of losing × Loss)

Optimal Bet % = Edge ÷ Payout
```

**Example 1: Coin flip**
- Heads wins 2-to-1 (you bet $1, get $2 profit + $1 back = $3 total)
- Your chance of winning = 60% (edge = 0.6×2 - 0.4×1 = 0.8)
- Kelly bet = 0.8 ÷ 2 = **40% of your money**

If you have 60% chance to double your money, bet 40% each time.

**Example 2: Stock with edge**
- You think stock goes up 12% on average, down 10% if wrong (so "loss" = 10%)
- You're right 55% of the time
- Edge = (0.55 × 12%) - (0.45 × 10%) = 6.6% - 4.5% = 2.1%
- Kelly bet = 2.1% ÷ 10% = **21% of your money**

---

## **5. The Million-Dollar Question: Where Does the "Edge" Come From?**

This is the **hard part** of investing. If you knew every stock's true edge, you'd be rich.

**Kelly betting requires you to know your edge accurately.** If you *think* you have a 10% edge but actually have a 0% edge (or negative edge), you'll bet too much and lose money!

**How do investors get an edge?**
1. **Better information**: You know something others don't (illegal insider info)
2. **Better analysis**: You've found a pattern that predicts future returns
3. **Better models**: Quantitative factors that have historically worked
4. **Faster execution**: You can trade before others do

For most individual investors: **finding a reliable edge is extremely hard.**

---

## **6. What Kelly Actually Looks Like in Investing**

When you have **many investments** (like a portfolio of 20 stocks), the Kelly formula becomes:

```
Bet weight for each stock ∝ (Expected return) ÷ (Risk/variance)
```

Or simply: **Invest more in stocks that have higher expected returns relative to their risk.**

**Example**:
- Stock A: Expected return 15%, volatility 30% → ratio = 15/30 = 0.5
- Stock B: Expected return 10%, volatility 15% → ratio = 10/15 = 0.67
- Stock C: Expected return 8%, volatility 10% → ratio = 8/10 = 0.8

Kelly says: **Weight them by these ratios** → C gets most, B less, A least.

But **wait!** That's if you **know** these expected returns. In reality, you're guessing!

---

## **7. The Reality Check: Why Kelly is Dangerous in Practice**

### **7.1 Overestimation of Edge**

Research shows that **individual investors** are TERRIBLE at estimating their edge:
- Overconfident: think they have skill when they're just lucky
- Sample size too small: 3 winning trades doesn't mean you have an edge
- Survivorship bias: only remember winners, forget losers

If you **overestimate your edge by 20%**, Kelly will tell you to bet **20% too much** → higher chance of ruin.

### **7.2 Kelli Is Ruthless**

Kelly gives you the **maximum growth rate possible**, but:
- It accepts **huge drawdowns** along the way (30-50% is normal)
- If you have a bad year early, recovery is mathematically much harder
- Most people **can't stomach** the volatility

### **7.3 Transaction Costs**

Kelly suggests rebalancing frequently as your estimates change. Each trade costs money (commissions, spreads, taxes). These costs can eat up most of the edge.

### **7.4 Leverage Trap**

Kelly often says: **"Bet more than 100% of your money!"** (use leverage/borrowing)
- If your edge is high enough, mathematically you should borrow to invest more
- But borrowing costs money, and margin calls can force you to sell at the worst time
- **2008**: Many leveraged investors were ruined (Long-Term Capital Management)

---

## **8. What Smart Money Actually Does: "Fractional Kelly"**

Since full Kelly is too aggressive and risky, most professional investors use **Fractional Kelly**:

```
Actual bet = Half-Kelly or Quarter-Kelly or Some fraction f of Full Kelly
```

**Why?**
- If Full Kelly gives 15% annual growth and 40% drawdowns
- Half-Kelly gives ~7.5% annual growth (growth rate ≈ f²) but with 20% drawdowns (volatility ≈ f)
- Quarter-Kelly gives ~4% annual growth with 10% drawdowns

**Many pros prefer** Half-Kelly or even less because:
1. **Less stress** (smaller swings)
2. **Lower ruin risk**
3. **Still very good returns** (4-7% vs 15% is still excellent)
4. **More realistic** given estimation error

---

## **9. The Famous Hack: The "1/N" Strategy**

Researchers found something shocking: **A simple equal-weight portfolio** (split money equally among N assets) often **outperforms** Kelly-optimized portfolios **out-of-sample**.

Why?
- Kelly requires good estimates of expected returns (almost impossible)
- Equal-weight makes **no estimates** → no estimation error
- Diversification alone does most of the work
- Transaction costs lower (rebalance less frequently)

**Example**: If you want to own 20 stocks:
- Kelly: tries to predict which 3 are best, put 50% in them
- 1/N: 5% in each, no predictions needed

Turns out the market is so efficient that making predictions is really hard. **Simplicity wins** often.

---

## **10. The "Kelly Criterion" vs. "Modern Portfolio Theory"**

They're **almost the same idea**:

**Modern Portfolio Theory (Markowitz, 1952)**: Choose portfolio weights to maximize expected return for given level of risk (or minimize risk for given return).

**Solution**: `w ∝ Σ⁻¹μ` (weights proportional to **precision matrix times expected returns**)

**Kelly Criterion**: Choose weights to maximize long-term growth rate.

**Solution**: `w ∝ Σ⁻¹μ` (the same formula!)

So: **Maximum Sharpe portfolio = Full Kelly portfolio**

The difference is in **motivation**, not mathematics:
- Markowitz: "I want the best risk-adjusted return"
- Kelly: "I want my money to grow as fast as possible without ruin"

Same math, different objective functions.

---

## **11. Real-World Examples**

### **Example 1: Blackjack**
- Edward Thorp calculated the exact edge in blackjack when using card counting
- Kelly told him to bet ≈ 5-10% of bankroll when count was favorable
- He made millions, wrote "Beat the Dealer"
- **Key**: He had a **real, measurable, high edge** (the card count gave ~2% advantage)

### **Example 2: Warren Buffett**
- Does Warren Buffett use Kelly? Probably not explicitly, but his approach is **Kelly-like**:
  - Concentrated bets (large % of portfolio in a few great companies)
  - Only invests when he has high confidence (large edge)
  - Holds long-term (low turnover)
- But he'd probably use **Half-Kelly** or less because:
  - His estimates of intrinsic value have error
  - He hates big drawdowns
  - He has multi-billion dollar portfolio → position limits, market impact

### **Example 3: Quantitative Hedge Funds**
- Renaissance Technologies: Yes, they use Kelly (or fractional Kelly)
- Why? They have **very high statistical edges** (short-term patterns that repeat)
- Their data and models give them **high-confidence** predictions
- They trade frequently (transaction costs low because of scale)
- They accept volatility (investors know what they're getting)

### **Example 4: Your 401(k)**
**Do not use full Kelly!** Here's why:
- You're investing in broad index funds (S&P 500)
- The "edge" of owning the S&P 500 is just the equity risk premium (maybe 4-6% per year)
- But that's not a **reliable** edge—it disappears for decades sometimes
- Use **1/N** or **risk parity** instead:
  - 60% stocks, 40% bonds (simple)
  - Rebalance annually
  - Never lever up
  - Never bet more than you can afford to lose

---

## **12. The Simple Takeaway for You**

### **If you're a retail investor:**
1. **Don't bet the farm** on any single stock or idea
2. **Diversify** across many assets (index funds are fine)
3. **Use equal weights** or **risk-based weights** (less in more volatile assets)
4. **Never use leverage** (borrowing)
5. **Rebalance** once or twice a year (not monthly)
6. **Accept that you probably don't have an edge**—the market is efficient
7. **Your edge is simply taking risk** (owning stocks and bonds) and diversifying
8. If you think you have a special edge, **cut your bet in half** from what Kelly says (or more)

### **If you're a professional with a real edge:**
1. Estimate your expected return and volatility **accurately** (use proper statistical methods)
2. Use **Half-Kelly** as a starting point
3. Add **constraints**: sector caps, single-name caps, max leverage
4. Use **shrinkage** on your return estimates (they're noisy)
5. Test **out-of-sample** for years before trusting your edge
6. Account for **transaction costs** in backtest
7. Accept that **drawdowns will be large** (if you're using real Kelly)

---

## **13. The Bottom Line in Plain English**

**Kelly Criterion tells you:**
"Given how sure you are about your bet paying off, here's the **maximum** percentage of your total money you should put on it."

**But remember:**
- The formula only works if you **accurately know your edge**
- Most people **overestimate** their edge
- Overbetting by a little bit can wipe you out over time
- **Underbetting** just means you grow slower—that's safer!

**Practical rule of thumb**:
```
Estimated Kelly bet = (Your perceived edge) ÷ (Payout odds)
Actual bet you should use = 1/2 to 1/4 of that
```

---

## **14. The One-Sentence Summary**

**Kelly Criterion**: A mathematical formula that tells you the **maximum** amount to bet on a sure thing—but since nothing is ever truly sure, smart people only bet a **fraction** of that amount to avoid going broke when they're wrong.

---

## **15. Final Thought: The Tale of Two Bettors**

In 1998, two hedge fund managers started with similar capital:

**Manager A** (uses Kelly):
- Bets big when confident
- Makes 40% one year, loses 50% the next
- After 10 years: up 200%, but investors fled after the 50% drawdown
- Retired rich but stressed

**Manager B** (uses Half-Kelly):
- Bets half as big
- Makes 20% one year, loses 25% the next
- After 10 years: up 120%, but smooth ride
- Investors happy, grew assets massively

**Which is better?** Depends on your psychology and your investors'. But both made money. The market rewards **survival** as much as it rewards **aggression**.

---

## **16. Your Action Plan**

1. **If picking individual stocks**: Never bet more than 5% of portfolio on one stock (this is already ~Half-Kelly for a typical stock edge)
2. **If running a portfolio of strategies**: Allocate to each strategy based on its Sharpe ratio (higher Sharpe → larger allocation)
3. **If just buying index funds**: Don't bother with Kelly—just use 60/40 or risk parity
4. **Always leave room for error**: You're wrong more often than you think
5. **Remember**: The goal is to **stay in the game** forever. A slow, steady 8% with no blowups beats 20% with a 50% drawdown that ends your career.

---

**Remember**: Kelly is like **nitrous oxide** for your portfolio—powerful, but use the wrong amount and you crash. Most people are better off with a **regular engine** that reliably gets them to their destination.

# Appendix: Kelly Rule for Negative Edge

---

## **What If the Edge Is Negative?**

### **Scenario:**
- You analyze an asset and find a **55% chance of losing money** (negative edge).
- Example: A stock has a **45% chance of going up** and a **55% chance of going down**.

---

## **How the Kelly Rule Handles Negative Edge**

### **1. Kelly Formula Recap**
The Kelly formula is:
```
Kelly % = (p × b - q) / b
```
- **p**: Probability of winning
- **q**: Probability of losing (`q = 1 - p`)
- **b**: Payout (profit per dollar bet)

---

### **2. Negative Edge Calculation**
- If **p = 0.45** (45% win rate) and **q = 0.55** (55% lose rate), the formula gives:
  ```
  Kelly % = (0.45 × 1 - 0.55) / 1 = -0.10 → -10%
  ```
- A **negative Kelly %** means you should **bet against the asset** (short it).

---

## **What to Do with a Negative Edge?**

### **1. If You Can Short the Asset**
- **Flip the sign**: Short **10% of your portfolio** (since Kelly % = -10%).
- **Why?** A 55% chance of losing on a long position = **55% chance of winning on a short position**.

### **2. If You Cannot Short**
- **Avoid the trade entirely** (Kelly % = 0).
- Look for other opportunities where you have a **positive edge**.

---

## **Example: Shorting a Stock**

### **Scenario:**
- **Stock X** has:
  - **45% chance of going up** (win for longs, lose for shorts).
  - **55% chance of going down** (win for shorts, lose for longs).
  - If it goes down, it drops by **10%**.
  - If it goes up, it rises by **10%**.

### **Kelly Calculation for Shorting:**
- **Edge for shorts**: 55% win rate (`p = 0.55`).
- **Payout (b)**: If you short \$1 and win, you gain \$0.10 (10% of \$1).
- **Formula:**
  ```
  Kelly % = (0.55 × 0.10 - 0.45 × 0.10) / 0.10 = 0.10 → 10%
  ```

### **Action:**
- **Short 10% of your portfolio** in Stock X.

---

## **Key Takeaways**

| Scenario               | Kelly %  | Action                          |
|------------------------|----------|---------------------------------|
| Positive Edge (55% win) | +10%    | Buy 10% of portfolio            |
| Negative Edge (55% lose) | -10%    | Short 10% of portfolio          |
| Cannot Short            | Negative | Avoid the bet                   |

---

## **Final Advice**
- **Negative edge?** Short the asset if possible.
- **Cannot short?** Avoid the trade.
- **Always manage risk**: Use **Fractional Kelly** (e.g., half-Kelly) to reduce volatility.