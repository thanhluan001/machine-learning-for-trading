# Mean-Variance Optimization Notebook - Detailed Explanation

## **1. Purpose & Context**

This notebook demonstrates **Modern Portfolio Theory (MPT)** - Harry Markowitz's mean-variance optimization framework for constructing optimal portfolios. It shows how to:

- Compute the **efficient frontier** (set of optimal portfolios)
- Find the **Maximum Sharpe Ratio portfolio** (tangency portfolio)
- Find the **Minimum Volatility portfolio**
- Compare optimization results with heuristics (equal-weight, risk parity, Kelly)

---

## **2. Theoretical Background**

### **2.1 The Mean-Variance Objective**

The optimization problem finds portfolio weights (ω) that either:

1. **Minimize variance** for a target return: `min ω^T Σ ω`
2. **Maximize Sharpe ratio**: `max (ω^T μ - r_f) / sqrt(ω^T Σ ω)`

**Where:**
- ω = portfolio weights (vector)
- Σ = covariance matrix of returns
- μ = expected returns (means)
- r_f = risk-free rate

**Constraints:**
- Sum of absolute weights = 1 (allows short selling if `short=True`)
- Optionally: each weight bounded between 0-1 (no shorting) or -1 to 1 (with shorting)

### **2.2 Why Optimization Works**

Portfolio variance depends on **covariances** between assets, not just individual volatilities:

```
σ_p² = ω^T Σ ω = Σ_i ω_i²σ_i² + Σ_i≠j ω_iω_jσ_iσ_jρ_ij
```

By including **uncorrelated assets**, you can reduce portfolio variance below the weighted average of individual variances - this is **diversification**.

---

## **3. Notebook Workflow**

### **3.1 Data Preparation**

```python
# Load S&P 500 constituent data
with pd.HDFStore('../data/assets.h5') as store:
    sp500_stocks = store['sp500/stocks']

# Load price data for 30 randomly selected stocks
prices = (store['quandl/wiki/prices']
          .adj_close
          .unstack('ticker')
          .filter(sp500_stocks.index)
          .sample(n=30, axis=1))
```

- Uses **adjusted close prices** from Quandl Wiki dataset
- Randomly samples 30 stocks from S&P 500
- Period: 2008-2017 (10 years)

### **3.2 Compute Weekly Returns**

```python
weekly_returns = prices.loc['2008':'2017'].resample('W').last().pct_change().dropna(how='all')
```

- Converts daily prices to **weekly frequency** (faster computation)
- Drops dates with all NaN values
- Result: 25 assets × 521 weeks (some stocks may have been dropped due to missing data)

### **3.3 Compute Inputs**

#### **Mean Returns & Covariance**
```python
mean_returns = weekly_returns.mean()  # 25×1 vector
cov_matrix = weekly_returns.cov()     # 25×25 matrix
```

#### **Precision Matrix**
```python
precision_matrix = pd.DataFrame(inv(cov_matrix), index=stocks, columns=stocks)
```

- Σ⁻¹ used in **Kelly portfolio** and **Black-Litterman** models
- Inverse of covariance matrix (warning: can be unstable if assets highly correlated)

#### **Annualization Factor**
```python
periods_per_year = round(weekly_returns.resample('A').size().mean())  # ≈ 52 weeks
```
Converts weekly returns/volatility to annualized figures.

#### **Risk-Free Rate**
```python
treasury_10yr_monthly = web.DataReader('DGS10', 'fred', start, end)
rf_rate = treasury_10yr_monthly.mean()  # ~2% annual
```

Loads historical 10-year Treasury rate and averages over the period.

---

## **4. Portfolio Simulation (Monte Carlo)**

### **4.1 Random Portfolio Generation**

```python
def simulate_portfolios(mean_ret, cov, rf_rate=rf_rate, short=True):
    alpha = np.full(shape=n_assets, fill_value=.05)
    weights = dirichlet(alpha=alpha, size=NUM_PF)  # Generate random weights summing to 1

    if short:
        weights *= choice([-1, 1], size=weights.shape)  # Randomly assign long/short

    # Annualize returns
    returns = weights @ mean_ret.values + 1
    returns = returns ** periods_per_year - 1

    # Annualize volatility
    std = (weights @ weekly_returns.T).std(1)
    std *= np.sqrt(periods_per_year)

    # Sharpe ratio
    sharpe = (returns - rf_rate) / std

    return pd.DataFrame({'Annualized Standard Deviation': std,
                         'Annualized Returns': returns,
                         'Sharpe Ratio': sharpe}), weights
```

**Key points:**
- Uses **Dirichlet distribution** to generate weights that sum to 1 (concentration parameter α=0.05 creates roughly uniform distribution)
- `short=True` randomly assigns long/short positions (50% chance each)
- **100,000** random portfolios simulated (NUM_PF constant)
- Performance metrics annualized from weekly data

### **4.2 Simulation Results**

The weights DataFrame statistics show:
- **Count**: 100,000 portfolios
- **Mean weight**: ~0.04 (4% per stock on average across all simulated portfolios)
- **Std**: ~0.13 (standard deviation of weights across portfolios, indicating high variability)
- **Min/Max**: Near 0 and 1 (some portfolios have tiny weights, some are highly concentrated)
- **Quartiles**: 50% median ~7e-07 (very small), 75% ~0.002, showing most portfolios have small weights with few large allocations

---

## **5. Optimization Functions**

### **5.1 Portfolio Performance Metrics**

```python
def portfolio_std(wt, rt=None, cov=None):
    """Annualized portfolio standard deviation"""
    return np.sqrt(wt @ cov @ wt * periods_per_year)

def portfolio_returns(wt, rt=None, cov=None):
    """Annualized portfolio returns"""
    return (wt @ rt + 1) ** periods_per_year - 1

def portfolio_performance(wt, rt, cov):
    """Annualized portfolio returns & standard deviation"""
    r = portfolio_returns(wt, rt=rt)
    sd = portfolio_std(wt, cov=cov)
    return r, sd
```

These functions compute the core metrics used throughout.

### **5.2 Maximum Sharpe Ratio Optimization**

```python
def neg_sharpe_ratio(weights, mean_ret, cov):
    r, sd = portfolio_performance(weights, mean_ret, cov)
    return -(r - rf_rate) / sd  # Negative for minimization

def max_sharpe_ratio(mean_ret, cov, short=False):
    return minimize(fun=neg_sharpe_ratio,
                    x0=x0,  # Initial guess
                    args=(mean_ret, cov),
                    method='SLSQP',
                    bounds=((-1 if short else 0, 1),) * n_assets,
                    constraints={'type': 'eq',
                                 'fun': lambda x: np.sum(np.abs(x))-1})
```

**Optimization details:**
- Uses `scipy.optimize.minimize`
- Method: **SLSQP** (Sequential Least Squares Programming) - handles bounds and constraints
- Objective: Minimize negative Sharpe (equivalent to maximizing Sharpe)
- Constraint: weights sum to 1 in absolute value (allows long/short)
- Bounds: `(0,1)` for long-only, `(-1,1)` for long-short

### **5.3 Minimum Volatility Portfolio**

```python
def min_vol(mean_ret, cov, short=False):
    bounds = ((-1 if short else 0, 1),) * n_assets
    return minimize(fun=portfolio_std,
                    x0=x0,
                    args=(mean_ret, cov),
                    method='SLSQP',
                    bounds=bounds,
                    constraints=weight_constraint)
```

- Minimizes portfolio standard deviation
- Subject to weights summing to 1 (in absolute value if allowing shorts)

### **5.4 Minimum Volatility for Target Return**

```python
def min_vol_target(mean_ret, cov, target, short=False):
    def ret_(wt):
        return portfolio_returns(wt, mean_ret)

    constraints = [
        {'type': 'eq', 'fun': lambda x: ret_(x) - target},  # Hit target return
        weight_constraint
    ]

    bounds = ((-1 if short else 0, 1),) * n_assets
    return minimize(portfolio_std,
                    x0=x0,
                    args=(mean_ret, cov),
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints)
```

- For a given target return, finds minimum variance portfolio
- This is the building block for the efficient frontier

### **5.5 Efficient Frontier**

```python
def efficient_frontier(mean_ret, cov, ret_range, short=False):
    return [min_vol_target(mean_ret, cov, ret) for ret in ret_range]
```

- Solves min volatility for each target return in a specified range
- The **upper envelope** of these solutions forms the efficient frontier
- `ret_range` typically spans from min to max achievable returns

---

## **6. Alternative Portfolio Strategies**

The notebook compares optimized results with several heuristics:

### **6.1 Maximum Sharpe Portfolio**
- **Result**: ~19.42% annual return, ~25.70% annual volatility
- Optimized weights maximizing risk-adjusted returns
- Theoretically optimal if inputs are correct and investor only cares about Sharpe

### **6.2 Minimum Volatility Portfolio**
- Lowest possible variance given historical data
- Often more robust in practice - reduces estimation error risk
- Attractive to risk-averse investors

### **6.3 Kelly Portfolio**
```python
kelly_wt = precision_matrix.dot(mean_returns).clip(lower=0).values
kelly_wt /= np.sum(np.abs(kelly_wt))
```

**Kelly Criterion** for portfolio optimization:
- Formula: `w ∝ Σ⁻¹ μ` (weights proportional to precision matrix times expected returns)
- Maximizes long-run growth rate
- `clip(lower=0)` sets negative weights to 0 → long-only Kelly
- Often produces **extreme concentrations** - can be very risky

### **6.4 Risk Parity Portfolio**
```python
std = weekly_returns.std()
std /= std.sum()  # Weight ∝ 1/volatility
```

- Allocates so each asset contributes **equally** to portfolio risk
- More robust to return estimation errors (only uses volatilities, not correlations)
- Simple but effective heuristic

### **6.5 Equal-Weight (1/n) Portfolio**
```python
np.full(n_assets, 1/n_assets)
```

- Naive 1/N allocation
- Surprisingly competitive benchmark (DeMiguel et al. 2007 showed it often outperforms MVO out-of-sample)
- Eliminates estimation risk entirely

---

## **7. Results Visualization**

### **7.1 Scatter Plot: Random Portfolios + Efficient Frontier**

![Efficient Frontier Plot]

**Elements:**
- **Blue points**: 100,000 randomly generated portfolios (Monte Carlo simulation)
- **Black dashed line**: Efficient frontier (optimized portfolios for each return level)
- **Black star (★)**: Maximum Sharpe Ratio portfolio
- **Black triangle (▼)**: Minimum Volatility portfolio
- **Black diamond (◆)**: Kelly portfolio
- **Black X (✖)**: Risk Parity portfolio
- **Black circle (●)**: Equal-weight (1/n) portfolio

**Interpretation:**
- The **efficient frontier** is the **upper-left envelope** of all feasible portfolios
- Portfolios below the frontier are **inefficient** (lower return for same risk)
- Maximum Sharpe is the **tangency point** where frontier meets the Capital Allocation Line (from risk-free rate)
- Minimum Volatility is the **leftmost point** on the frontier
- Random portfolios cluster around but mostly below the frontier
- The Kelly portfolio is often far to the right (high return, high risk) and can be very concentrated

---

## **8. Critical Considerations & Limitations**

### **8.1 The Markowitz (or Curse of Dimensionality)**

The notebook warns: "The more diversification is required (by correlated investment opportunities), the more unreliable the weights produced by the algorithm."

**Why does this happen?**

1. **Estimation error in expected returns (μ)**:
   - Expected returns are extremely noisy (typically only 10-20% of variation is predictable)
   - Historical mean has high standard error: σ_μ ≈ σ/√T where T is sample size
   - For monthly returns with 10 years data (T=120), standard error is ~√(monthly variance/120)
   - Small differences in μ estimates lead to huge differences in optimal weights

2. **Covariance matrix instability**:
   - Correlation matrices with high asset correlations have **high condition number**
   - Near-singular matrices make inverse unstable numerically
   - Small changes in data cause large swings in Σ⁻¹
   - This amplifies errors in μ when computing Kelly weights: Σ⁻¹μ

3. **Extreme weights**:
   - Unconstrained optimization often produces:
     - Very long/short positions (leveraged)
     - Concentrated bets in a few assets
     - Weights far from 1/N
   - These extreme weights are highly sensitive to estimation error

4. **Turnover & transaction costs**:
   - MVO weights can change dramatically each period
   - High turnover would incur costs that aren't accounted for

**Empirical evidence**: Studies show that out-of-sample, MVO often underperforms simple 1/N or risk parity strategies.

### **8.2 Forward-Looking Inputs Problem**

"Markowitz's mean-variance frontier relies on **in-sample, backward-looking** optimization. In practice, portfolio optimization requires **forward-looking** input."

**The fundamental circularity:**

1. **Goal**: Build portfolio that maximizes future risk-adjusted returns
2. **Inputs needed**: Expected future returns (μ_future), future covariance (Σ_future)
3. **What we have**: Historical data (μ_historical, Σ_historical)
4. **Assumption**: μ_historical ≈ μ_future
5. **Problem**: Historical μ is a noisy estimate of μ_future
6. **Consequence**: Optimizing on noisy estimates leads to overfitting

The optimization is essentially **data-mining** - we're finding the best portfolio that would have worked in the past, but there's no guarantee it will work in the future.

### **8.3 All-Weather Portfolio Challenge**

The efficient frontier assumes all investors want the same thing: maximum return for given risk, or minimum risk for given return. But different investors have different horizons, liquidity needs, and risk tolerances. The 1/N portfolio is surprisingly robust precisely because it makes **no assumptions** about μ.

### **8.4 Practical Issues Not Addressed**

1. **No out-of-sample testing**: All inputs and optimization use same in-sample period (2008-2017). Should test on holdout period.
2. **No transaction costs**: In reality, turnover costs eat 1-2% annually for active strategies.
3. **No liquidity constraints**: Can't trade infinitely large positions in small-cap stocks.
4. **No sector/factor constraints**: Real portfolios often limit sector exposure, avoid single-name concentration.
5. **Static weights**: No dynamic rebalancing as new information arrives.
6. **Gaussian assumption**: MVO assumes returns are normally distributed (or at least quadratic utility). Real returns have fat tails and skew.

---

## **9. Code Structure & Best Practices**

### **9.1 Vectorization**
- Uses `@` operator (matrix multiplication) instead of loops
- Efficient computation: `weights @ cov_matrix @ weights.T` for portfolio variance
- All major operations are vectorized across portfolios in simulation

### **9.2 Separation of Concerns**
- `portfolio_performance()`: central function reused everywhere
- `simulate_portfolios()`: separate simulation logic
- Optimization functions: pure, deterministic, easy to test

### **9.3 Proper Use of Constraints & Bounds**
- `weight_constraint`: sum of absolute weights = 1 (allows both long/short)
- `bounds`: tuple of (lower, upper) for each asset
- Properly handles **no-short** (`(0,1)`) vs **short-allowed** (`(-1,1)`)
- Constraints defined as dictionary with `type: 'eq'` (equality) or `'ineq'` (inequality)

### **9.4 Numerical Settings**
```python
options={'tol': 1e-10, 'maxiter': 1e4}
```
- Tight tolerance (1e-10) ensures accurate solution
- Max iterations allows convergence on difficult problems

---

## **10. Extensions & Improvements**

If you were to extend this notebook for real-world use:

### **10.1 Walk-Forward Analysis**
Instead of single-period optimization:
```python
for train_end in rolling_windows:
    # Estimate μ, Σ on training period
    # Optimize weights
    # Test on next period (out-of-sample)
    # Record out-of-sample performance
```
This gives realistic estimate of strategy performance.

### **10.2 Regularization of Covariance**
```python
from sklearn.covariance import LedoitWolf
lw = LedoitWolf()
cov_shrunk = lw.fit(weekly_returns).covariance_
```
- **Shrinkage**: Pulls extreme correlations toward average
- Reduces condition number, improves numerical stability
- Especially important with many assets or short samples

### **10.3 Black-Litterman Model**
- Start with **market equilibrium weights** (inverse volatility weighted, or market cap)
- Implied returns: `μ_implied = λ Σ w_mkt`
- Add investor **views** (absolute or relative)
- Blend: `μ_post = [(τΣ)⁻¹ + Ω⁻¹]⁻¹ [(τΣ)⁻¹ μ_implied + Ω⁻¹ p]`
- More robust than historical μ alone

### **10.4 Factor-Based MVO**
Instead of 25 individual stocks, work with **factor portfolios**:
- Use PCA or pre-defined factors (Fama-French, Barra)
- Reduces dimensionality from N assets to K << N factors
- Covariance of factors better estimated (fewer parameters)
- Translate factor optimizations back to asset weights via factor exposures

### **10.5 Resampling Method (Michaud)**
- Bootstrap resample returns many times (e.g., 1000 bootstrap samples)
- For each sample, compute efficient frontier
- Average weights across all resampled frontiers
- Creates **more diversified, robust** portfolios
- Essentially a Monte Carlo approach to incorporate estimation error

### **10.6 Robust Optimization (Minimax)**
Instead of optimizing point estimates:
```python
min_w max_θ L(w, θ) where θ ∈ uncertainty set
```
- Accounts for uncertainty in μ and Σ
- Common approach: assume μ lies in ellipsoid around estimate
- Solve min-max problem that hedges against worst-case μ within ellipse

### **10.7 Transaction Cost Penalty**
Add turnover penalty to objective:
```python
def objective_with_cost(weights, prev_weights, mean_ret, cov, cost_per_turnover=0.001):
    r, sd = portfolio_performance(weights, mean_ret, cov)
    turnover = np.sum(np.abs(weights - prev_weights))
    cost = cost_per_turnover * turnover
    return -(r/sd - cost)  # or add cost to risk: -(r - γ(cost + sd))
```

### **10.8 Multi-Period Optimization**
Instead of single-period, consider multiple periods:
- Dynamic programming approach
- Or approximate with certainty equivalence (single-period with updated μ, Σ each period)
- Often leads to more conservative allocations

---

## **11. Connection to Previous Notebooks**

### **11.1 Pyfolio (03_pyfolio_demo)**
- Pyfolio is for **evaluation** of a given strategy
- This notebook is for **constructing** the strategy via optimization
- Typical workflow:
  1. Use MVO (this notebook) to determine weights
  2. Run backtest with Zipline using those weights
  3. Evaluate backtest with Pyfolio
  4. Check if out-of-sample performance holds

### **11.2 Strategy Development**
- The equal-weight momentum strategy from earlier notebooks serves as a **benchmark**
- MVO could be used to combine multiple alpha factors optimally
- Or to combine multiple strategies into a meta-portfolio

---

## **12. Practical Implementation Checklist**

If you want to implement MVO in production:

- [ ] **Estimate μ and Σ robustly**:
  - Use shrinkage (Ledoit-Wolf) for Σ
  - Consider Bayesian or Black-Litterman for μ
  - Or use factor models to reduce dimensionality

- [ ] **Add constraints**:
  - Sector caps (e.g., max 20% in any sector)
  - Single-name caps (e.g., max 5% in any stock)
  - Turnover limits
  - Liquidity constraints (volume-based position limits)

- [ ] **Optimization method**:
  - SLSQP works for moderate N (< 100)
  - For large N, use **quadratic programming** with dedicated solver (OSQP, CVXOPT, MOSEK)
  - Consider **convex formulation** to guarantee global optimum

- [ ] **Walk-forward backtest**:
  - Rolling window estimation (e.g., 3 years data, optimize, hold 1 month, repeat)
  - Compare to naive benchmarks (1/N, risk parity)
  - Measure turnover and estimate transaction costs

- [ ] **Out-of-sample validation**:
  - Reserve at least 20-30% of data for final validation
  - Or use **cross-validation** with expanding window

- [ ] **Monitor stability**:
  - Track turnover (high = likely overfitting)
  - Check if weights change dramatically with small data changes
  - Sensitivity analysis: how much do weights change if μ perturbed by 1σ?

---

## **13. Key Takeaways**

1. **MPT is elegant but fragile**: The math is beautiful, the practice is difficult due to estimation error
2. **Benchmark against simple heuristics**: Always compare MVO to 1/N and risk parity
3. **Regularization is essential**: Unconstrained MVO almost always overfits
4. **Forward testing is mandatory**: In-sample efficient portfolios may fail out-of-sample
5. **Simple often beats sophisticated**: 1/N portfolio is hard to beat consistently
6. **Consider the Kelly alternative**: But Kelly is aggressive and can lead to ruin if μ estimation error is large
7. **Use MVO as a tool, not a solution**: It's one approach among many for portfolio construction

---

## **14. References**

- Markowitz, H. (1952). "Portfolio Selection." Journal of Finance.
- Michaud, R. (1989). "The Markowitz Optimization Enigma: Is 'Optimized' Optimal?"
- DeMiguel, V., Garlappi, L., & Uppal, R. (2007). "Optimal versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy?"
- Litterman, R. (1999). "The Logic of the Latzis-Litterman Model."
- Kelly, J. L. (1956). "A New Interpretation of Information Rate."

---

## **Appendix: Complete Code Structure**

```
Notebook Organization:

1. Imports & Settings
   - Libraries: numpy, pandas, scipy.optimize, pandas_datareader
   - Visualization: matplotlib, seaborn
   - Random seed for reproducibility

2. Data Preparation
   - Load S&P 500 constituents
   - Sample 30 stocks
   - Get price data (adjusted close)
   - Compute weekly returns

3. Compute Inputs
   - Mean returns
   - Covariance matrix
   - Precision matrix (inverse)
   - Annualization factor
   - Risk-free rate

4. Portfolio Simulation
   - Function: simulate_portfolios()
   - Generate 100,000 random portfolios
   - Compute annualized returns, volatility, Sharpe

5. Optimization Functions
   - portfolio_performance()
   - neg_sharpe_ratio()
   - max_sharpe_ratio()
   - min_vol()
   - min_vol_target()
   - efficient_frontier()

6. Calculate Key Portfolios
   - Max Sharpe
   - Min Volatility
   - Kelly
   - Risk Parity
   - Equal-Weight

7. Visualization
   - Scatter plot with efficient frontier
   - Markers for all portfolio types

```

---

**End of Documentation**
