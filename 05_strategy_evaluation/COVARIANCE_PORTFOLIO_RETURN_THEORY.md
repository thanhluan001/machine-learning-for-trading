# Covariance and Portfolio Return Theory

## **1. Portfolio Return: The Basics**

### **1.1 Definition**

For a portfolio with **weights** `w_i` (fraction of capital in asset i) and **returns** `r_i`:

**Portfolio return** (single period):
```
R_p = Σ_i w_i * r_i
```

In vector notation:
```
R_p = wᵀ r
```

Where:
- `w` = weight vector (n×1)
- `r` = return vector (n×1)
- `wᵀ` = transpose of w

**Expected portfolio return**:
```
μ_p = E[R_p] = wᵀ μ
```

Where `μ` is the vector of expected returns.

---

## **2. Portfolio Variance: The Core Theory**

### **2.1 The Fundamental Formula**

Portfolio **variance** depends on **covariances** between assets:

```
σ_p² = Var(R_p) = Var(Σ_i w_i r_i)
     = Σ_i w_i² Var(r_i) + Σ_i≠j w_i w_j Cov(r_i, r_j)
```

In matrix notation:
```
σ_p² = wᵀ Σ w
```

Where:
- `Σ` = covariance matrix (n×n)
- Diagonal elements: `σ_ii = Var(r_i) = σ_i²`
- Off-diagonal elements: `σ_ij = Cov(r_i, r_j)`

---

### **2.2 Expanded Form**

Breaking down `wᵀΣw`:

```
σ_p² = w₁²σ₁² + w₂²σ₂² + ... + wₙ²σₙ²       (variance terms)
     + 2w₁w₂σ₁₂ + 2w₁w₃σ₁₃ + ...           (covariance terms)
```

The factor of 2 appears because:
- σ₁₂ and σ₂₁ are the same (symmetric matrix)
- w₁w₂σ₁₂ + w₂w₁σ₂₁ = 2w₁w₂σ₁₂

**In terms of correlations**:

Let `σ_i` = standard deviation of asset i, `ρ_ij` = correlation coefficient.

```
σ_p² = Σ_i w_i² σ_i² + Σ_i≠j w_i w_j σ_i σ_j ρ_ij
```

---

## **3. The Power of Diversification**

### **3.1 Single Asset Case**

If you hold only one asset (say asset k):
```
w_k = 1, all other w_i = 0
σ_p² = 1² * σ_k² = σ_k²
```

No diversification possible - you inherit all the volatility.

---

### **3.2 Two-Asset Portfolio**

For two assets with weights w and (1-w):

```
σ_p² = w²σ₁² + (1-w)²σ₂² + 2w(1-w)σ₁₂
```

**Perfect correlation** (ρ₁₂ = 1, so σ₁₂ = σ₁σ₂):
```
σ_p² = w²σ₁² + (1-w)²σ₂² + 2w(1-w)σ₁σ₂
     = (wσ₁ + (1-w)σ₂)²
```

Portfolio standard deviation = weighted average of individual σ's. **No diversification benefit**.

---

**Zero correlation** (ρ₁₂ = 0, σ₁₂ = 0):
```
σ_p² = w²σ₁² + (1-w)²σ₂²
```

Now σ_p² < (wσ₁ + (1-w)σ₂)² because cross-term is zero. **Diversification benefit exists**.

**Example**:
- w = 0.5, σ₁ = 20%, σ₂ = 10%, ρ = 0
- Weighted avg σ = 0.5*0.2 + 0.5*0.1 = 0.15 = 15%
- σ_p = √(0.25*0.04 + 0.25*0.01) = √0.0125 = 11.18%

We reduced volatility from 15% to 11.18% simply by diversification!

---

**Perfect negative correlation** (ρ₁₂ = -1, σ₁₂ = -σ₁σ₂):
```
σ_p² = w²σ₁² + (1-w)₂σ₂² - 2w(1-w)σ₁σ₂
     = (wσ₁ - (1-w)σ₂)²
```

Can achieve zero variance if wσ₁ = (1-w)σ₂ → w = σ₂/(σ₁+σ₂).

**Example**:
- σ₁ = 20%, σ₂ = 10%, ρ = -1
- Perfect hedge weight: w = 0.1/(0.2+0.1) = 1/3 ≈ 0.333
- σ_p = |0.333*0.2 - 0.667*0.1| = |0.0667 - 0.0667| = 0
- **Zero risk portfolio exists!**

---

### **3.3 Equal-Weight n-Asset Portfolio**

For **n assets with equal weights** (w_i = 1/n) and **equal variances** σ² and **average covariance** C̄:

```
σ_p² = Σ_i (1/n)² σ² + Σ_i≠j (1/n)(1/n) σ_ij
     = n * (1/n² σ²) + n(n-1) * (1/n²) * C̄
     = (σ² / n) + ((n-1)/n) * C̄
```

**As n → ∞**:
```
σ_p² → C̄
```

**Interpretation**:
- Portfolio variance approaches the **average covariance** between assets
- The **idiosyncratic variance** (σ²/n) disappears as we add more assets
- **Systematic/common risk** (C̄) remains regardless of diversification

**Example**:
- n = 50 stocks, each σ = 30%, average ρ = 0.2
- C̄ = ρ × σ_i × σ_j (assuming equal σ) = 0.2 × 0.3² = 0.018
- σ_p² ≈ 0.018 → σ_p ≈ 13.4%
- Single asset σ = 30%
- **Diversification reduced risk by more than half!**

---

## **4. The Two-Fund Separation Theorem**

### **4.1 Statement**

Any efficient portfolio can be constructed as a combination of:
1. The **Global Minimum Variance (GMV)** portfolio
2. Any other **efficient** portfolio

Or equivalently: All efficient portfolios lie on a **straight line** in mean-variance space when you allow leverage at the risk-free rate.

This is the foundation of **Capital Market Theory**.

---

### **4.2 Implications**

1. **All investors hold the same risky portfolio** (the tangency portfolio)
2. Only difference is how much to **mix with risk-free asset**
3. Optimal risky portfolio = **Maximum Sharpe Ratio** portfolio

---

## **5. The Efficient Frontier (Markowitz, 1952)**

### **5.1 Definition**

The set of portfolios that are **Pareto optimal**: you cannot increase expected return without increasing variance, and vice versa.

Mathematically:
```
Efficient Frontier = { w : σ_p² minimized subject to wᵀμ = μ_target, wᵀ1 = 1 }
```

---

### **5.2 Quadratic Optimization Problem**

```
min  wᵀ Σ w
s.t. wᵀ μ = μ_p      (target return)
     wᵀ 1 = 1        (weights sum to 1)
```

**Solution** (using Lagrange multipliers):

Let:
- `A = 1ᵀΣ⁻¹1`
- `B = 1ᵀΣ⁻¹μ`
- `C = μᵀΣ⁻¹μ`
- `D = AC - B²`

Then weights for target return μ_p:
```
w = (A*(μ_p*Σ⁻¹1) - B*(Σ⁻¹μ)) / D
```

The **global minimum variance portfolio** (μ_p varies) is:
```
w_GMV = Σ⁻¹1 / (1ᵀΣ⁻¹1)
```

**Portfolio variances on frontier**:
```
σ_p² = (A*μ_p² - 2B*μ_p + C) / D
```

This is a **quadratic function** in μ_p → frontier is a parabola in (σ, μ) space.

---

### **5.3 Maximum Sharpe Ratio Portfolio**

When risk-free rate r_f is available:

```
max  (wᵀμ - r_f) / √(wᵀΣw)
s.t. wᵀ1 = 1
```

**Solution**:
```
w* ∝ Σ⁻¹(μ - r_f·1)
```

Normalized to sum to 1:
```
w* = [Σ⁻¹(μ - r_f·1)] / [1ᵀΣ⁻¹(μ - r_f·1)]
```

**Important**: This is **identical** to the Kelly portfolio (when sum(w) = 1 constraint).

---

## **6. Why Covariance Matters: Numerical Example**

### **6.1 Three Assets**

| Asset | μ (E[return]) | σ (vol) | Weights |
|-------|--------------|---------|---------|
| A     | 10%          | 20%     | 50%     |
| B     | 8%           | 15%     | 30%     |
| C     | 6%           | 10%     | 20%     |

**Correlation matrix**:
```
      A     B     C
A   1.00  0.30  0.10
B   0.30  1.00  0.20
C   0.10  0.20  1.00
```

**Step 1**: Covariance matrix
```
σ_A = 0.20, σ_B = 0.15, σ_C = 0.10

Σ = [0.04   0.009   0.002
     0.009  0.0225  0.003
     0.002  0.003   0.01]
```

**Step 2**: Portfolio variance
```
wᵀ = [0.5, 0.3, 0.2]

wᵀΣw = 0.5²*0.04 + 0.3²*0.0225 + 0.2²*0.01
      + 2*0.5*0.3*0.009
      + 2*0.5*0.2*0.002
      + 2*0.3*0.2*0.003
      = 0.01 + 0.002025 + 0.0004
      + 0.0027 + 0.0002 + 0.00036
      = 0.015685
σ_p = √0.015685 = 12.53%
```

**Step 3**: Compare to weighted average variance
```
Weighted avg variance = 0.5²*0.04 + 0.3²*0.0225 + 0.2²*0.01
                     = 0.01 + 0.002025 + 0.0004 = 0.012425
σ_avg_var = √0.012425 = 11.45%
```

Wait! Our portfolio variance (0.015685) is **higher** than weighted average variance (0.012425). That's because correlations are positive!

Diversification benefit (σ_p² < weighted avg of variances) is **not guaranteed**. What matters is the **covariance term**:
```
Covariance contribution = 2w_i w_j σ_ij = 0.00326
```

This pushed variance up because correlations are positive.

---

### **6.2 What if correlations were zero?**

If all ρ_ij = 0 → all σ_ij = 0:
```
σ_p² = 0.01 + 0.002025 + 0.0004 = 0.012425
σ_p = 11.45%
```

Now we have diversification benefit:
- Weighted avg σ from individual assets = √(0.5²*0.04 + ...) = 11.45%
- Wait, that's the same as before? Let's recalc:

Actually weighted average of variances (not of σ):
```
Σ w_i²σ_i² = 0.012425
```

With zero correlation, σ_p² = 0.012425 exactly.

With positive correlation (ρ=0.3,0.2,0.1), σ_p² = 0.015685 > 0.012425.

**So positive correlations increase portfolio variance above the sum of weighted variances**.

---

### **6.3 Intuition via Diversification Ratio**

Define **Diversification Ratio (DR)**:
```
DR = (weighted avg σ) / portfolio σ
```

For our example:
- Weighted avg σ = Σ w_i σ_i = 0.5*0.2 + 0.3*0.15 + 0.2*0.1 = 0.155 = 15.5%
- Portfolio σ = 12.53%
- DR = 15.5% / 12.53% = 1.237

**DR > 1**: diversification benefit (portfolio less risky than asset-weighted average)

If correlations were all 1:
- Portfolio would be perfect combination → σ_p = weighted avg σ = 15.5%
- DR = 1.0 → no diversification

If correlations were all -1:
- Can construct zero-variance portfolio → σ_p < weighted avg σ
- DR → ∞ (if perfect hedge possible)

So DR measures how much **correlation structure** reduces risk.

---

## **7. The Covariance Matrix Properties**

### **7.1 Positive Semi-Definiteness**

For any weight vector w:
```
wᵀΣw ≥ 0
```

That is, portfolio variance is **always non-negative** (zero only if w is in null space of Σ).

**Why?** Because Σ = Cov(r) and Var(wᵀr) ≥ 0 always.

This implies:
- All eigenvalues of Σ are ≥ 0
- Σ can be decomposed: Σ = PDPᵀ where P orthogonal, D diagonal with λ_i ≥ 0

---

### **7.2 Conditioning & The "Curse"**

**Condition number** κ = λ_max / λ_min

If assets are highly correlated, Σ has:
- λ₁ large (market factor)
- λ₂,...,λ_n very small (near-zero eigenvalues)

High condition number (κ >> 1) means Σ is **nearly singular** → Σ⁻¹ highly unstable.

**Example**: S&P 500 stocks
- N = 500 assets
- First eigenvalue explains ~40% of variance (market)
- Remaining 499 eigenvalues share 60% → many are tiny (0.1% each)
- Σ⁻¹ will have huge entries due to division by small λ_i

This is why **plain sample covariance** fails for large N.

---

### **7.3 Factor Structure**

In reality, Σ has a **factor structure**:

```
Σ = B * F * Bᵀ + D
```

Where:
- B = matrix of factor loadings (n×k)
- F = factor covariance matrix (k×k)
- D = diagonal matrix of idiosyncratic variances (n×n)

Typical factors:
- Market (CAPM β)
- Size, Value, Momentum (Fama-French)
- Industry/sector dummies
- Quality, Low-volatility

**Why factor models help**:
- Reduce number of parameters from O(n²) to O(nk + k²)
- More stable estimation (k << n, and time series for factors longer)
- Captures most of Σ's eigenvalue structure

---

## **8. Estimation Error in Covariance**

### **8.1 Sampling Distribution**

Given T observations of n assets, sample covariance:

```
Σ̂ = (1/(T-1)) Σ_t (r_t - μ̂)(r_t - μ̂)ᵀ
```

**Standard error** of individual covariance estimates:
```
SE(σ̂_ij) ≈ √((σ_i²σ_j² + σ_ij²) / T)
```

For typical stock: σ_i ≈ 0.20 annually = 0.06 monthly
If ρ_ij = 0.3, σ_ij = 0.3*0.06*0.06 = 0.00108
SE ≈ √((0.0036*0.0036 + 0.00000117) / T) ≈ √(1.3e-5 / T)

For T=60 months (5 years), SE ≈ 0.0047
For T=120 months (10 years), SE ≈ 0.0033

Relative error: 0.0047/0.00108 ≈ 435% !!

**Conclusion**: Covariances are **extremely noisy** with typical data lengths.

---

### **8.2 Shrinkage**

Ledoit-Wolf shrinkage:
```
Σ_shrunk = (1-α) * Σ̂ + α * F
```

Where:
- F = shrinkage target (often diagonal matrix of variances, or constant correlation matrix)
- α ∈ [0,1] chosen to minimize MSE

**Effect**:
- Pulls extreme covariances toward target
- Reduces condition number
- Makes Σ⁻¹ more stable
- Small bias, large variance reduction → lower overall MSE

---

## **9. Mathematical Relationships**

### **9.1 Correlation vs Covariance**

```
ρ_ij = σ_ij / (σ_i σ_j)
σ_ij = ρ_ij * σ_i σ_j
```

So variance formula in terms of correlations:
```
σ_p² = Σ_i w_i² σ_i² + Σ_i≠j w_i w_j ρ_ij σ_i σ_j
```

If we standardize weights by volatility:
```
z_i = w_i / σ_i   (weight per unit risk)
```

Then:
```
σ_p² = (Σ_i z_i σ_i)² ??? No, better approach:
```

Actually:
```
σ_p² = Σ_i σ_i² z_i² + 2 Σ_i<j σ_i σ_j ρ_ij z_i z_j
```

If ρ_ij = 0:
```
σ_p² = Σ_i σ_i² z_i²
```

We can choose z_i to equalize contribution to variance → **risk parity**.

---

### **9.2 Beta Representation**

Portfolio beta to some benchmark (e.g., market):
```
β_p = Cov(R_p, R_m) / Var(R_m)
     = (wᵀΣv) / σ_m²
```

where v = vector of asset betas to market: β_i = Cov(r_i, r_m)/σ_m²

Then:
```
σ_p² = β_p² σ_m² + σ_{residual}²
```

Total variance = systematic + residual

Diversification reduces **residual variance** but systematic variance remains.

---

## **10. The Relation to the Efficient Frontier**

### **10.1 Two-Fund Theorem**

Any efficient portfolio can be written as:
```
w = θ w₁ + (1-θ) w₂
```

where w₁ and w₂ are any two **distinct** efficient portfolios.

In practice:
- w₁ = GMV portfolio
- w₂ = maximum Sharpe portfolio

Then all efficient portfolios = combinations of these two.

---

### **10.2 The Capital Market Line**

With risk-free rate r_f, the efficient frontier becomes the **Capital Market Line**:

```
μ_p = r_f + Sharpe_max * σ_p
```

Where:
```
Sharpe_max = (w_Tangencyᵀ μ - r_f) / (w_Tangencyᵀ Σ w_Tangency)^(1/2)
```

And tangent portfolio weights:
```
w_Tangency ∝ Σ⁻¹(μ - r_f·1)
```

---

## **11. Practical Issues with Covariance Estimation**

### **11.1 Sample Size Requirement**

To estimate Σ accurately for n assets, need:
```
T >> n
```

Rule of thumb: T ≥ 5n or T ≥ 10n for reasonable stability.

For S&P 500 (n≈500):
- Monthly data: need 5*500 = 250 months = 20+ years
- But within 20 years, market dynamics change (non-stationarity)
- Daily data: 5*500 = 2500 days = 10 years → maybe sufficient

**Conclusion**: Accurate covariance estimation requires much more data than many practitioners have.

---

### **11.2 Eigenvalue Analysis**

Empirical covariance matrix Σ̂ has:
- **First few eigenvalues**: capture common factors (market, sectors)
- **Remaining eigenvalues**: noise + idiosyncratic variation

For S&P 500:
- λ₁ (market): 40-50% of total variance
- Next 10-20 eigenvalues: industry factors
- Remaining 480+ eigenvalues: noise (should be roughly equal, but sample noise creates variation)

**Problem**: Small eigenvalues have huge relative error → Σ̂⁻¹ has huge eigenvalues (1/λ_small) → unstable weights.

---

### **11.3 Constant Correlation Model**

Shrink all correlations to a constant ρ̄:
```
σ_ij = ρ̄ * σ_i σ_j
```

Then:
```
Σ_shrink = diag(σ_i²) + ρ̄ * (σσᵀ - diag(σ_i²))
```

where σ = vector of volatilities.

- Only 2 parameters to estimate: all σ_i from univariate data (stable), and single ρ̄ (can use average of pairwise correlations)
- Much more stable than full sample covariance
- Works surprisingly well in practice

---

### **11.4 Exponentially Weighted Covariance**

Give more weight to recent observations:
```
Σ̂_EW = (1-δ) Σ_{t=1}^∞ δ^{t-1} (r_{T-t} - μ̂)(r_{T-t} - μ̂)ᵀ
```

- δ ∈ (0,1) decay factor (e.g., δ=0.94 for monthly, half-life ~12 months)
- Estimates adapt to changing market conditions
- But more noisy than longer lookback

---

## **12. Summary of Key Formulas**

### **12.1 Portfolio Moments**
```
μ_p = wᵀ μ
σ_p² = wᵀ Σ w
```

### **12.2 Two-Asset Special Case**
```
σ_p² = w₁²σ₁² + w₂²σ₂² + 2w₁w₂ρ₁₂σ₁σ₂
```

With w₂ = 1-w₁:
```
σ_p² = w₁²σ₁² + (1-w₁)²σ₂² + 2w₁(1-w₁)ρ₁₂σ₁σ₂
```

Min variance weight:
```
w₁* = (σ₂² - ρ₁₂σ₁σ₂) / (σ₁² + σ₂² - 2ρ₁₂σ₁σ₂)
```

---

### **12.3 Equal-Weight n-Asset Portfolio**
```
σ_p² = (1/n) * avg(σ_i²) + (1 - 1/n) * avg(σ_ij)
     ≈ C̄    as n → ∞
```

where C̄ = average covariance.

---

### **12.4 Efficient Frontier Weights**
```
A = 1ᵀΣ⁻¹1
B = 1ᵀΣ⁻¹μ
C = μᵀΣ⁻¹μ
D = AC - B²

For target μ_p:
w = [ (Aμ_p - B)Σ⁻¹1 + (C - Bμ_p)Σ⁻¹μ ] / D
```

---

### **12.5 Maximum Sharpe (Tangency)**
```
w* = Σ⁻¹(μ - r_f·1) / [1ᵀΣ⁻¹(μ - r_f·1)]
```

Sharpe_max = √(μᵀΣ⁻¹μ - 2r_f·1ᵀΣ⁻¹μ + r_f²·1ᵀΣ⁻¹1) = √(C - 2r_fB + r_f²A)

---

## **13. Common Misconceptions**

1. **"Diversification always reduces risk"** → False! If correlations are high (near 1), adding assets may not help much. Only benefit if some correlations < 1.

2. **"Portfolio variance equals weighted average of variances"** → False! That's only true if all correlations are zero OR weights are perfectly aligned with principal components.

3. **"More assets always better"** → Only up to a point. After capturing all systematic factors, adding more assets just adds idiosyncratic noise (which averages out anyway with equal weighting).

4. **"Out-of-sample, MVO beats 1/N"** → Often false! Due to estimation error, MVO underperforms naive equal-weight.

5. **"Covariance matrix can be ignored"** → No, it's essential. Risk parity uses only volatilities, but equal-weight implicitly diversifies away covariance through many assets.

---

## **14. The Bottom Line**

The formula `σ_p² = wᵀΣw` is **deceptively simple** but has profound implications:

1. **Diversification benefits come from the off-diagonal elements** of Σ
2. **Covariance estimation is the hardest part** of portfolio optimization
3. **High correlations → limited diversification** (e.g., during crises)
4. **Low/negative correlations → strong diversification**
5. **The efficient frontier** shows the trade-off: for any given level of risk, what's the maximum achievable return
6. **The global minimum variance portfolio** is stable (only needs Σ, not μ)
7. **The maximum Sharpe portfolio** requires both μ and Σ and is extremely sensitive to estimation error

**Practical wisdom**:
- Use **shrinkage** on Σ
- Use **factor models** to reduce dimensionality
- Consider **risk parity** or **equal-weight** as robust alternatives
- If using MVO, apply **regularization** and **constraints** (sector caps, single-name caps)
- Always test **out-of-sample** with walk-forward analysis

---

## **Appendix: Derivation of Portfolio Variance**

Starting from:
```
R_p = Σ_i w_i r_i
```

```
Var(R_p) = E[(R_p - μ_p)²]
         = E[(Σ_i w_i (r_i - μ_i))²]
         = E[Σ_i w_i² (r_i - μ_i)² + Σ_i≠j w_i w_j (r_i - μ_i)(r_j - μ_j)]
         = Σ_i w_i² E[(r_i - μ_i)²] + Σ_i≠j w_i w_j E[(r_i - μ_i)(r_j - μ_j)]
         = Σ_i w_i² σ_i² + Σ_i≠j w_i w_j σ_ij
```

QED.

---

## **References**

- Markowitz, H. (1952). "Portfolio Selection." Journal of Finance.
- Markowitz, H. (1959). "Portfolio Selection: Efficient Diversification of Investments."
- Elton, E. J., Gruber, M. J., Brown, S. J., & Goetzmann, W. N. (2009). "Modern Portfolio Theory and Investment Analysis."
- Michaud, R. (1989). "The Markowitz Optimization Enigma: Is 'Optimized' Optimal?"
- Ledoit, O., & Wolf, M. (2004). "Honey, I Shrunk the Sample Covariance Matrix."
- Chan, E. P. (2008). "Quantitative Trading: How to Build Your Own Algorithmic Trading Business."

---

**End of Documentation**
