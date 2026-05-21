# Kelly Rule Notebook - Detailed Explanation

## **1. Purpose & Context**

This notebook demonstrates the **Kelly Criterion** - a mathematical formula for optimal position sizing to maximize long-term wealth growth. It shows:

- **Single-asset Kelly**: How much to bet on one investment
- **Multi-asset Kelly**: How to allocate across multiple investments simultaneously
- **Connection to MVO**: Multi-asset Kelly is equivalent to the Maximum Sharpe Ratio portfolio
- **Practical application**: Computing Kelly allocations for S&P 500 stocks

---

## **2. Historical Background**

John Kelly (1956) at Bell Labs solved the problem: *"Given favorable odds but uncertain probability of success, what fraction of current wealth should I bet to maximize long-term wealth?"*

**Key insight**: You don't just maximize expected return - you maximize the **growth rate of wealth** (log wealth) to avoid ruin.

Ed Thorp applied Kelly to:
- Blackjack (in "Beat the Dealer")
- Later founded Princeton/Newport Partners hedge fund
- Made $80 billion trading using Kelly-inspired position sizing

---

## **3. The Mathematics**

### **3.1 Single Bet - Binary Outcome**

For a bet with:
- `p` = probability of winning
- `b` = odds (e.g., b=5 means win $5 for every $1 bet)
- `f` = fraction of wealth to bet

**Final wealth after N bets:**
```
W_N = W_0 * (1 + b*f)^(N*p) * (1 - f)^(N*(1-p))
```

**Growth rate G:**
```
G = lim(N→∞) (1/N) * log(W_N/W_0)
  = p * log(1 + b*f) + (1-p) * log(1 - f)
```

Maximize G with respect to f. Derivative:
```
dG/df = p*b/(1+b*f) - (1-p)/(1-f) = 0
```

**Kelly formula:**
```
f* = (p*b + p - 1) / b = (p*(b+1) - 1) / b
```

For **fair odds** where b = (1-p)/p (break-even), simplifies to:
```
f* = 2p - 1
```

**Example**: If you have 60% edge (p=0.6) on fair odds:
```
f* = 2*0.6 - 1 = 0.2 (20% of wealth)
```

### **3.2 Solver Derivation (from notebook)**

Using symbolic math with SymPy:

**General odds case**:
```python
share, odds, probability = symbols('share odds probability')
Value = probability * log(1 + odds * share) + (1 - probability) * log(1 - share)
solve(diff(Value, share), share)
⇒ (odds*probability + probability - 1)/odds
```

**Even money odds (b=1)**:
```python
f, p = symbols('f p')
y = p * log(1 + f) + (1 - p) * log(1 - f)
solve(diff(y, f), f)
⇒ 2*p - 1
```

Both match the Kelly formula.

---

## **4. Single-Asset Kelly Application**

The notebook applies Kelly to the **S&P 500 index**:

### **4.1 Data Preparation**

```python
with pd.HDFStore(DATA_STORE) as store:
    sp500 = store['sp500/stooq'].close

annual_returns = sp500.resample('A').last().pct_change().dropna().to_frame('sp500')
return_params = annual_returns.sp500.rolling(25).agg(['mean', 'std']).dropna()
```

- Uses annual S&P 500 returns (1928-2020 ~ 92 years)
- Rolling 25-year window to estimate **μ** and **σ** (approx. generation length)
- Result: time series of mean and standard deviation estimates

### **4.2 Continuous Distribution Integration**

Financial returns are continuous, not binary. Kelly's formula generalizes to:

Maximize expected log growth:
```
G(f) = ∫ log(1 + f * r) * p(r) dr
```

Where `r` is the random return, `p(r)` is the probability density (assumed normal).

**Implementation**:
```python
def norm_integral(f, mean, std):
    # Integrate log(1+f*r) * N(r|mean,std) from mean-3σ to mean+3σ
    val, er = quad(lambda s: np.log(1 + f * s) * norm.pdf(s, mean, std),
                   mean - 3*std, mean + 3*std)
    return -val  # negative for minimization
```

Uses numerical integration (`scipy.integrate.quad`) over ±3 standard deviations.

**Alternative derivative approach**:
```python
def norm_dev_integral(f, mean, std):
    # First derivative = 0 at optimum
    return quad(lambda s: (s / (1 + f * s)) * norm.pdf(s, mean, std),
                mean-3*std, mean+3*std)[0]
```

Then use Newton's method to find root:
```python
x0 = newton(norm_dev_integral, .1, args=(m, s))
```

### **4.3 Computing Kelly Fraction**

```python
def get_kelly_share(data):
    solution = minimize_scalar(norm_integral,
                               args=(data['mean'], data['std']),
                               bounds=[0, 2],  # f between 0 and 2
                               method='bounded')
    return solution.x

annual_returns['f'] = return_params.apply(get_kelly_share, axis=1)
```

For each 25-year window, computes optimal Kelly fraction `f*`.

### **4.4 Results for S&P 500**

The plot shows:
- **Blue line**: Rolling 25-year mean return (μ)
- **Light blue/shading**: ±2σ confidence band (95% interval)
- **Green line**: Kelly fraction f* (right axis)

**Interpretation**:
- Kelly fraction varies over time (typically 0.5 to 1.5)
- During high-return, low-vol periods (e.g., 1990s) → higher f*
- During crisis periods (2008) → lower f*
- Many periods: **f* > 1**, meaning **leveraged position** (borrow to invest more than 100% of wealth)

**Why f* often > 1 for S&P 500?**
- Long-term average annual return ≈ 8-10%
- Volatility ≈ 15-20%
- Kelly fraction ≈ μ/σ² (for normal distribution)
  - f* ≈ 0.08 / (0.20²) = 0.08 / 0.04 = **2.0** (200% of wealth, i.e., 2x leverage)

So Kelly suggests using leverage!

---

## **5. Single-Asset Performance Evaluation**

```python
(annual_returns[['sp500']]
 .assign(kelly=annual_returns.sp500.mul(annual_returns.f.shift()))
 .dropna()
 .loc['1900':]
 .add(1)
 .cumprod()
 .sub(1)
 .plot(lw=2))
```

**Methodology**:
- Each year bet fraction `f*` (determined from previous 25 years of data)
- Return = f* × S&P 500 return
- Cumulative growth from $1 initial

**Results show**:
- Kelly strategy massively outperforms simple buy-and-hold
- But this is **in-sample** (using future-knowledge in rolling window)
- In reality, you wouldn't know future μ, σ with such precision

**Critical issue**: The plot is **misleadingly optimistic** because:
1. Uses same data to estimate parameters AND test strategy
2. Rolling 25-year window provides smooth estimates (low estimation error)
3. No transaction costs for annual rebalancing
4. Unlimited leverage at risk-free rate (unrealistic)

---

## **6. Multi-Asset Kelly (The Main Point)**

### **6.1 Why Multiple Assets?**

Single-asset Kelly works if you have ONE investment opportunity. But we have **many stocks** - how to allocate across them?

The optimal multi-asset Kelly allocation turns out to be:

**Weights proportional to:**
```
w ∝ Σ⁻¹ μ
```

Where:
- μ = vector of expected returns (n×1)
- Σ = covariance matrix (n×n)
- Σ⁻¹ = precision matrix

This is **exactly the same** as the **maximum Sharpe ratio portfolio** from mean-variance optimization!

### **6.2 Derivation Sketch**

Let:
- w = vector of fractions of total wealth (sum(w) = 1)
- r = vector of random returns (mean μ, covariance Σ)

Growth rate:
```
G(w) = E[log(1 + w·r)]
```

Using Taylor expansion and ignoring higher moments:
```
G(w) ≈ log(1 + w·μ) - 0.5 * wᵀ Σ w  (for small returns)
```

Maximize wᵀ μ - 0.5 wᵀ Σ w (ignoring constant log(1) and scaling) subject to sum(w)=1.

Lagrangian:
```
L = wᵀ μ - 0.5 wᵀ Σ w - λ(1ᵀw - 1)
```

First order condition:
```
∂L/∂w = μ - Σ w - λ·1 = 0
⇒ Σ w = μ - λ·1
⇒ w = Σ⁻¹ μ - λ·Σ⁻¹ 1
```

With constraint 1ᵀw = 1:
```
1ᵀΣ⁻¹ μ - λ·1ᵀΣ⁻¹ 1 = 1
⇒ λ = (1ᵀΣ⁻¹ μ - 1) / (1ᵀΣ⁻¹ 1)
```

This gives Kelly weights that sum to 1.

**Note**: This is the **full Kelly** (sum of weights = 1). There's also **fractional Kelly** (scaled by factor < 1) to reduce volatility.

### **6.3 Implementation**

```python
# Load S&P 500 stock prices
with pd.HDFStore(DATA_STORE) as store:
    sp500_stocks = store['sp500/stocks'].index
    prices = store['quandl/wiki/prices'].adj_close.unstack('ticker').filter(sp500_stocks)

# Monthly returns for 1988-2017 (30 years)
monthly_returns = prices.loc['1988':'2017'].resample('M').last().pct_change().dropna(how='all').dropna(axis=1)
stocks = monthly_returns.columns
```

Then:
```python
cov = monthly_returns.cov()
precision_matrix = pd.DataFrame(inv(cov), index=stocks, columns=stocks)

# Kelly allocation: w ∝ Σ⁻¹μ
kelly_allocation = monthly_returns.mean().dot(precision_matrix)
```

**Important**: This gives **unscaled** weights (sum may not equal 1). Need to normalize:
```python
kelly_allocation.div(kelly_allocation.sum())
```

---

## **7. Multi-Asset Kelly Results**

### **7.1 Weight Distribution**

```python
kelly_allocation.describe()
```

Shows:
- Mean: positive (because stocks have positive expected returns)
- Std: high (large dispersion)
- Min/Max: Some large positive, some negative weights (short positions)
- Sum: Not necessarily 1 (needs normalization)

### **7.2 Extreme Allocations**

```python
kelly_allocation[kelly_allocation.abs() > 5].sort_values(ascending=False).plot.barh(figsize=(8, 10))
```

Shows tickers with absolute weight > 5 (i.e., 5x their "fair share" or extreme shorts).

**Observation**: A handful of stocks get **very large allocations** (10-20x), while most get small or negative.

This is a **feature of Kelly**: it's extremely concentrated. If you estimate μ and Σ with error, this leads to **catastrophic overbetting** on spurious signals.

### **7.3 Normalized Weights**

After dividing by sum, total allocation = 1 (or -1 if net short). The portfolio is typically:
- **Net long** (since μ > 0 on average)
- **Highly concentrated** in top 5-10 names
- May have **short positions** in many stocks (if they have negative expected return after covariance adjustment)

---

## **8. Performance Evaluation**

```python
ax = monthly_returns.loc['2010':].mul(kelly_allocation.div(kelly_allocation.sum())).sum(1).to_frame('Kelly').add(1).cumprod().sub(1).plot(figsize=(14,4))
sp500.filter(monthly_returns.loc['2010':].index).pct_change().add(1).cumprod().sub(1).to_frame('SP500').plot(ax=ax)
```

**Test**: Use the Kelly weights computed from 1988-2009 data, then apply to 2010-2017 out-of-sample.

**Result shown**: Kelly portfolio **dramatically outperforms** S&P 500 during this period.

**But major caveats**:
1. **Estimation error**: We used 21 years of monthly data (252 months) to estimate μ and Σ for ~500 stocks. This is highly unstable.
2. **No regularization**: Plain sample mean/covariance matrix is extremely noisy with N≈500, T≈252 → N > T problem
3. **No transaction costs**: Kelly requires frequent rebalancing → high turnover
4. **No leverage constraint**: Kelly weights may exceed 1 (need to scale or use leverage)
5. **Data mining**: The whole dataset is used both to compute and test - not true OOS

---

## **9. Kelly vs. Mean-Variance Optimization**

The notebook notes (from E. Chan 2008): *"the multi-asset Kelly Rule is equivalent to the (potentially levered) maximum Sharpe ratio portfolio from the mean-variance optimization."*

**Proof sketch**:
- MVO with risk-free rate: maximize (wᵀμ - r_f) / √(wᵀΣw)
- Without risk-free rate (or setting to 0): maximize wᵀμ / √(wᵀΣw)
- Equivalent to maximizing wᵀμ subject to wᵀΣw = 1 (normalize variance)
- Solution: w ∝ Σ⁻¹μ

**Exactly Kelly!**

So:
- **Single-period MVO** (max Sharpe) = **Full Kelly**
- **Fractional Kelly** (bet fraction < 1) = **scaled-down MVO weights**

---

## **10. Practical Implementation Issues**

### **10.1 Estimation Error - The Biggest Problem**

Kelly weights are **hyper-sensitive** to estimates of μ and Σ.

**Example**: If μ has standard error σ_μ = 2% annually (realistic for monthly data), then:
- True μ might be 8% ± 2%
- Kelly f* ∝ μ, so a 1σ error in μ → 25% error in f*
- With 500 stocks, some μ estimates will be off by 3σ → Kelly suggests huge bets on noise

**Solutions**:
1. **Shrinkage**: Regress μ toward global mean
   ```
   μ_shrunk = α * μ_sample + (1-α) * μ_global
   ```
2. **Factor model**: Estimate μ from factor exposures instead of individual asset means
3. **Bayesian**: Use prior that all assets have same expected return (CAPM equilibrium)
4. **Black-Litterman**: Combine equilibrium returns with investor views
5. **Use longer lookback**: But parameters change over time (non-stationarity)

### **10.2 Covariance Estimation**

Σ⁻¹ amplifies estimation errors when assets are highly correlated.

**Solutions**:
1. **Shrinkage** (Ledoit-Wolf): Shrink sample covariance toward diagonal or constant correlation
2. **Factor model**: Σ = BFBᵀ + D (factor covariance + specific variance)
3. **Exponential weighting**: Give more weight to recent observations
4. **Regularization**: Add diagonal ridge penalty

### **10.3 Leverage**

Kelly often suggests f* > 1 (use leverage). In practice:
- **Margin costs**: Borrowing rate > risk-free
- **Margin calls**: Volatility can cause forced liquidation
- **Limited leverage**: Retail investors may only get 2x; institutions maybe 4-6x

**Practical adaptation**: Scale down Kelly to max leverage constraint.

### **10.4 Turnover**

Kelly weights change dramatically with new μ, Σ estimates → high turnover → transaction costs.

**Mitigation**:
- Rebalance quarterly or annually, not monthly
- Use **damped updates**: w_new = θ * w_kelly + (1-θ) * w_old
- Add turnover penalty to optimization

### **10.5 Bankruptcy Risk**

Kelly maximizes long-term growth but **accepts high short-term volatility**. A bad streak can blow up the account.

**Fractional Kelly**: Bet fraction γ×f* where γ ∈ [0.5, 0.75] often recommended. This:
- Reduces volatility by factor γ
- Reduces growth rate by ~γ² (not linear)
- Much more robust to estimation error

### **10.6 Discrete Adjustments & Market Impact**

- Cannot trade fractional Kelly precisely due to minimum lot sizes
- Large positions move prices (market impact)
- Illiquid assets cannot hold large positions

---

## **11. When to Use Kelly**

**Appropriate**:
- You have **high-confidence estimates** of μ and Σ (e.g., statistical arbitrage with strong signal)
- You can **rebalance frequently** and costs are low
- You have **sufficient data** (T >> N) for stable estimates
- You understand and accept **large drawdowns** (Kelly can have 50%+ drawdowns)

**Inappropriate**:
- Individual stock picking (μ very noisy)
- Low-frequency trading (high turnover costs)
- Small capital base (can't absorb volatility)
- Retail investor (can't access leverage or shorting)

---

## **12. Code Structure & Functions**

### **12.1 Single-Asset Integration Functions**

```python
def norm_integral(f, mean, std):
    """Negative expected log growth for normal distribution"""
    val, er = quad(lambda s: np.log(1 + f * s) * norm.pdf(s, mean, std),
                   mean - 3*std, mean + 3*std)
    return -val

def norm_dev_integral(f, mean, std):
    """First derivative - used by Newton's method"""
    val, er = quad(lambda s: (s / (1 + f * s)) * norm.pdf(s, mean, std),
                   mean-3*std, mean+3*std)
    return val

def get_kelly_share(data):
    """Compute optimal f for given mean, std estimates"""
    solution = minimize_scalar(norm_integral,
                               args=(data['mean'], data['std']),
                               bounds=[0, 2],
                               method='bounded')
    return solution.x
```

### **12.2 Multi-Asset Kelly**

```python
# Covariance matrix
cov = monthly_returns.cov()

# Precision matrix (inverse)
precision_matrix = pd.DataFrame(inv(cov), index=stocks, columns=stocks)

# Unscaled Kelly weights: w_i ∝ Σ⁻¹μ
kelly_allocation = monthly_returns.mean().dot(precision_matrix)

# Normalize to sum to 1
kelly_weights = kelly_allocation / kelly_allocation.sum()
```

**Vectorized computation**: O(N³) for matrix inverse (slow for large N), but manageable for N=500.

---

## **13. Comparison with Alternatives**

| Approach | Formula | Pros | Cons |
|----------|---------|------|------|
| **Kelly** | w ∝ Σ⁻¹μ | Growth-optimal theoretically | Extreme weights, sensitive to μ |
| **Fractional Kelly** | w = γ·Kelly | More robust, less volatile | Lower growth if γ too small |
| **Risk Parity** | w ∝ 1/σ_i | Only needs volatilities, robust | Ignores correlations/returns |
| **Equal-Weight** | w = 1/N | Extremely robust, simple | No optimization |
| **MVO (Max Sharpe)** | Same as Kelly | Mean-variance efficient | Same sensitivity issues |
| **Black-Litterman** | w ∝ Σ⁻¹[(τΣ)⁻¹μ_equil + Ω⁻¹p] | Bayesian, incorporates views | Requires equilibrium weights |

**Key insight**: Kelly is theoretically optimal **if and only if** μ and Σ are known exactly. Since they're not, fractional Kelly or risk parity often perform better in practice.

---

## **14. Notebook Critical Assessment**

### **14.1 What's Good**

- Clear mathematical derivation with SymPy
- Shows both integral and derivative approaches
- Connects single-asset to multi-asset formulation
- Visualizes results (Kelly fraction over time, extreme weights)
- Compares to benchmark

### **14.2 What's Missing/Problematic**

1. **No regularization**: Using sample mean/cov directly is dangerous
2. **No cross-validation**: All on single in-sample period
3. **Overfitting**: Using same data to compute and evaluate Kelly weights
4. **No transaction costs**: Major omission for monthly rebalancing of 500 stocks
5. **No leverage constraints**: Doesn't scale to realistic leverage limits
6. **No turnover analysis**: Would show extreme turnover
7. **Small N in multi-asset**: Only shows 500 stocks but also uses long time series → should use shrinkage
8. **No comparison to fractionally rescaled Kelly or risk parity**
9. **Performance plot is in-sample**: Uses entire 1988-2017 to estimate, then tests subset - still data mining
10. **Unrealistic data frequency**: Monthly rebalancing for 500 stocks would incur huge transaction costs

### **14.3 Correct Approach Would Be**

1. Split into **estimation period** (1988-2000) and **testing period** (2001-2017)
2. Apply **shrinkage** to μ and Σ
3. Compare **full Kelly**, **half Kelly**, **quarter Kelly**, **risk parity**, **1/N**
4. Account for **0.5% transaction costs per turnover**
5. Cap **leverage at 2x**
6. Report:
   - CAGR
   - Volatility
   - Sharpe ratio
   - Maximum drawdown
   - Turnover percentage
   - Worst-case wealth (for ruin risk)

---

## **15. Key Takeaways**

1. **Kelly maximizes long-term growth** but is **extremely aggressive** (often suggests >100% allocation)
2. **Multi-asset Kelly** = **Maximum Sharpe Ratio portfolio** from MVO
3. **Biggest danger**: Estimation error causes extreme overbetting → ruin
4. **Fractional Kelly** (50-75% of full Kelly) is **much more robust** with small loss in growth rate
5. **For individual stock picking**, Kelly is generally **not recommended** - μ too noisy
6. **For statistical arbitrage** with high SNR, Kelly can be appropriate
7. **Regularization is essential**: Always shrink μ toward prior, use factor models
8. **In-sample outperformance is meaningless**: Must do proper walk-forward testing
9. **Kelly doesn't account for drawdown utility**: Many investors prefer smoother path even if terminal wealth lower
10. **Practical use**: Institutions use Kelly as an **upper bound**; actual allocations fractionally scaled

---

## **16. Connection to Previous Notebooks**

### **16.1 Mean-Variance Optimization (04)**

- This notebook shows **Kelly is equivalent to Maximum Sharpe MVO**
- Both use w ∝ Σ⁻¹μ
- MVO derived from utility maximization, Kelly from log utility growth maximization
- Same optimization problem, different motivation

### **16.2 Pyfolio Evaluation (03)**

If you wanted to evaluate Kelly:
1. Generate Kelly weights from historical data
2. Create Zipline backtest with those weights
3. Run Pyfolio analysis to check out-of-sample performance
4. Compare to equal-weight benchmark

### **16.3 Strategy Design**

Kelly can be used to **combine alpha factors**:
- Each factor has expected return (α) and covariance with others
- Kelly gives optimal combination weights
- But again, μ estimation error fatal

---

## **17. Code Summary**

```
Cells 1-4: Imports and setup
Cells 5-11: Single-asset Kelly for S&P 500
  - Load data
  - Compute rolling μ, σ
  - Define integral functions
  - Compute f* for each window
  - Plot results
Cells 12-17: Single-asset performance evaluation
Cells 18-37: Multi-asset Kelly for S&P 500 stocks
  - Load stock prices (1988-2017)
  - Compute monthly returns
  - Calculate covariance and precision matrix
  - Compute Kelly weights: w ∝ Σ⁻¹μ
  - Normalize
Cells 38-51: Results visualization
  - Show extreme allocations (>5x)
  - Compare Kelly vs. SP500 performance (2010-2017)
```

---

## **18. References**

- Kelly, J. L. (1956). "A New Interpretation of Information Rate." *Bell System Technical Journal*.
- Thorp, E. O. (1962). "Beat the Dealer."
- Thorp, E. O. (1969). "Optimal Gambling Systems for Favorable Games."
- Chan, E. P. (2008). "Quantitative Trading: How to Build Your Own Algorithmic Trading Business."
- MacLean, L. C., Thorp, E. O., & Ziemba, W. T. (Eds.). (2010). "The Kelly Capital Growth Investment Criterion: Theory and Practice."

---

## **Appendix: Mathematical Details**

### **A.1 Expected Log Growth Derivation**

For discrete returns with probability p of +b*f and (1-p) of -f:

```
G(f) = p * log(1 + b*f) + (1-p) * log(1 - f)
```

First derivative:
```
G'(f) = p*b/(1+b*f) - (1-p)/(1-f)
```

Set = 0:
```
p*b*(1-f) = (1-p)*(1+b*f)
p*b - p*b*f = 1 - p + (1-p)*b*f
p*b - 1 + p = (p*b + 1 - p) * f
f = (p*(b+1) - 1) / (p*b + 1 - p) = (p*b + p - 1) / b
```

For **even-money bet** (b=1):
```
f* = p*2 - 1
```

### **A.2 Continuous Approximation**

When returns are continuous with mean μ and variance σ²:

```
G(f) = E[log(1 + f*r)]
      ≈ log(1 + f*μ) - 0.5 * f² * σ²  (Taylor expansion around r=μ)
```

Maximize:
```
G'(f) = μ/(1+fμ) - f*σ² = 0
μ = f*σ²*(1+fμ) ≈ f*σ² (if fμ small)
f* ≈ μ/σ²
```

This is the **continuous approximation** to Kelly. The notebook's numerical integration solves the exact integral.

### **A.3 Multi-Asset Kelly Derivation (Multivariate)**

For vector of returns r ~ N(μ, Σ), portfolio weights w with sum(w)=1:

```
G(w) = E[log(1 + w·r)]
      ≈ log(1 + w·μ) - 0.5 * wᵀ Σ w  (for small returns)
```

Objective: maximize wᵀ μ - 0.5 wᵀ Σ w

Using Lagrange multiplier λ for constraint 1ᵀw = 1:

```
L = wᵀ μ - 0.5 wᵀ Σ w - λ(1ᵀw - 1)

∂L/∂w = μ - Σ w - λ·1 = 0
Σ w = μ - λ·1
w = Σ⁻¹μ - λ·Σ⁻¹1

Constraint: 1ᵀw = 1ᵀΣ⁻¹μ - λ·1ᵀΣ⁻¹1 = 1
λ = (1ᵀΣ⁻¹μ - 1) / (1ᵀΣ⁻¹1)

Substitute back:
w = Σ⁻¹μ - [(1ᵀΣ⁻¹μ - 1) / (1ᵀΣ⁻¹1)] · Σ⁻¹1
```

This is the **full Kelly portfolio** with weights summing to 1.

If we allow **leverage** (no constraint on sum of absolute weights), the formula simplifies to:
```
w ∝ Σ⁻¹μ
```
and then scaled by a constant to reach desired leverage.

---

## **Appendix: Code Deep Dive**

### **A.1 Numerical Integration Approach**

```python
def norm_integral(f, mean, std):
    val, er = quad(lambda s: np.log(1 + f * s) * norm.pdf(s, mean, std),
                   mean - 3*std, mean + 3*std)
    return -val
```

- `quad` performs adaptive quadrature integration
- Integrates over ±3σ (99.7% of normal distribution)
- Returns negative because `minimize_scalar` minimizes
- `er` is error estimate (ignored)

This computes:
```
G(f) = ∫_{-∞}^{∞} log(1+f*r) · φ(r|μ,σ) dr
```
where φ is normal PDF. The integral has no closed form, so numerical integration needed.

### **A.2 Newton's Method Alternative**

```python
def norm_dev_integral(f, mean, std):
    return quad(lambda s: (s / (1 + f * s)) * norm.pdf(s, mean, std),
                mean-3*std, mean+3*std)[0]

x0 = newton(norm_dev_integral, .1, args=(m, s))
```

Newton's method finds root of G'(f)=0:
```
G'(f) = E[r/(1+f*r)] = 0
```

Numerically same result as direct maximization.

### **A.3 Precision Matrix Inversion**

```python
precision_matrix = pd.DataFrame(inv(cov), index=stocks, columns=stocks)
kelly_allocation = monthly_returns.mean().dot(precision_matrix)
```

This is:
```
kelly_allocation_i = Σ_j μ_j · (Σ⁻¹)_{ji}
```

or in vector form: `w = μᵀ Σ⁻¹` (row vector) or `w = Σ⁻¹μ` (column vector).

Both equivalent since Σ symmetric.

**Warning**: Matrix inversion is O(N³). For N=500, this is ~125 million operations - doable but slow. For N=5000, would be problematic.

---

## **End of Documentation**
