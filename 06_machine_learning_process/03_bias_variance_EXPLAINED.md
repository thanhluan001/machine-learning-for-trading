# Bias-Variance Tradeoff — Explained

## Overview

This notebook demonstrates one of the most fundamental concepts in machine learning: **the bias-variance tradeoff**. It shows visually and quantitatively how model complexity affects prediction error, and how learning curves diagnose whether your model suffers from bias or variance problems.

---

## Section 1: Generate Sample Data (Cells 5–7)

### What it does

Creates a **synthetic true function** using a Taylor series approximation of `sin(x)`:

```python
def f(x, max_degree=9):
    taylor = [(-1)**i * x**e / factorial(e)
              for i, e in enumerate(range(1, max_degree, 2))]
    return np.sum(taylor, axis=0)
```

**Taylor series of sin(x):**

$$\sin(x) = x - \frac{x^3}{3!} + \frac{x^5}{5!} - \frac{x^7}{7!} + \frac{x^9}{9!} - \cdots$$

With `max_degree=5`, the function approximates sin(x) with 3 terms:
- $x$ (degree 1)
- $-x^3/6$ (degree 3)
- $x^5/120$ (degree 5)

The plot shows both the Taylor approximation and the true `sin(x)` function — they're close but not identical, which is intentional. This gap represents **irreducible error** (the true function we're trying to learn).

### Why synthetic data?

Using a known function (sin) lets us:
1. **Know the ground truth** — we can measure exactly how wrong our models are
2. **Control the noise** — we add noise deliberately to simulate real-world messiness
3. **Repeat experiments** — we can draw many samples and study the *distribution* of errors

### Key insight

The Taylor series with `max_degree=5` is our **true data-generating process**. Any model we fit is trying to *recover* this function from noisy samples. A polynomial of degree 5 should be the "right fit."

---

## Section 2: Underfitting vs Overfitting — Visual Example (Cells 8–10)

### What it does

Fits **three polynomial models** to 30 noisy samples of sin(x):

| Model | Polynomial Degree | What it does |
|-------|------------------|--------------|
| **Underfit** | Degree 1 | Straight line — too simple to capture the curve |
| **Right Fit** | Degree 5 | Matches the true function's complexity |
| **Overfit** | Degree 15 | Wiggles wildly to hit every training point |

The code:
```python
n = 30                  # 30 sample points
noise = .2              # 20% noise relative to std of y
degrees = [1, 5, 15]   # polynomial degrees to try

for degree in degrees:
    fit = np.poly1d(np.polyfit(x=x_, y=y_, deg=degree))
    mse = mean_squared_error(fit(x), np.sin(x))
```

### What the plot shows

**3 side-by-side panels:**

1. **Degree 1 (Underfit)**: The model draws a straight line through curved data. It misses the wave shape entirely. High MSE — the model is **too simple**.

2. **Degree 5 (Right Fit)**: The model captures the sinusoidal pattern well. Low MSE — it matches the true function's complexity.

3. **Degree 15 (Overfit)**: The model wiggles wildly between data points. It hits the training dots perfectly but deviates enormously elsewhere. The MSE explodes — the model is **too complex** for the data.

### Key insight

- **Underfitting** = model can't capture the pattern → **high bias**
- **Overfitting** = model memorizes noise, not pattern → **high variance**
- **Right fit** = model captures pattern but ignores noise → **low bias + low variance**

---

## Section 3: Bias-Variance Tradeoff — Quantitative Analysis (Cells 11–16)

### What it does

Runs a **Monte Carlo simulation** (100 trials) to decompose prediction error into bias and variance components.

**Setup:**
- **Train region**: x ∈ [-1, 1] (the model sees this data)
- **Test region**: x ∈ [1, 2] (the model is evaluated here — out-of-sample)
- **Models**: Degree 1 (Underfit), Degree 5 (Right Fit), Degree 9 (Overfit)
- **Each trial**: Sample 25 random points from train region, add noise, fit model, evaluate on both train and test

```python
for i in range(100):  # 100 random training sets
    x_ = {d: choice(X[d], size=25, replace=False) for d in datasets}
    y_ = {d: f(x_[d], max_degree=5) for d in datasets}
    y_['Train'] += normal(loc=0, scale=..., size=25)  # Add noise

    trained_models = {
        fit: np.poly1d(np.polyfit(x=x_['Train'], y=y_['Train'], deg=deg))
        for fit, deg in models.items()
    }
```

### Why 100 trials?

A single train/test split gives you **one number**. But the error depends on which random sample you happened to draw. With 100 trials, you see the **distribution** of errors — this is what reveals bias vs variance:

- **Bias**: Does the model systematically miss the truth? → Average error across trials
- **Variance**: How much do predictions change with different training data? → Spread of errors across trials

### What the plots show

**Left panel — Box plots of errors:**
- **Underfit**: Both train and test errors are large and similar → model can't learn the pattern regardless of data → **high bias, low variance**
- **Right Fit**: Both train and test errors are small and similar → model learns well and generalizes → **low bias, low variance**
- **Overfit**: Train error is tiny, test error is enormous → model memorizes training noise but fails on new data → **low bias (on train), high variance (on test)**

**Right panel — Scatter of out-of-sample predictions:**
- Underfit predictions cluster around a simple shape (miss the curve)
- Right Fit predictions track the dashed true function closely
- Overfit predictions are wildly scattered around the truth

### The Mathematical Framework

Total prediction error decomposes as:

$$\text{Total Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Noise}$$

| Component | What it measures | Underfit | Overfit |
|-----------|-----------------|----------|---------|
| **Bias²** | Systematic error from wrong model class | **HIGH** (wrong shape) | **LOW** (flexible enough) |
| **Variance** | Sensitivity to training data randomness | **LOW** (simple model is stable) | **HIGH** (complex model wiggles) |
| **Noise** | Inherent randomness in data | Same | Same |

**The tradeoff**: As model complexity increases:
- Bias² **decreases** (model can represent more complex patterns)
- Variance **increases** (model becomes more sensitive to training noise)
- Somewhere in between is the **sweet spot** (minimum total error)

---

## Section 4: Learning Curves (Cells 17–22)

### What it does

Plots **train RMSE vs test RMSE** as a function of **training set size** for each model complexity.

This answers a different question: "If I get more data, will my model improve?"

**Method:**
1. Start with very few training samples (10% of data)
2. Gradually increase to 100%
3. At each size, do 5-fold cross-validation
4. Record train RMSE and test RMSE (mean ± std)

```python
def folds(train, test, nfolds):
    shuffle(train); shuffle(test)
    steps = (np.array([len(train), len(test)]) / nfolds).astype(int)
    for fold in range(nfolds):
        i, j = fold * steps
        yield train[i:i+steps[0]], test[j:j+steps[1]]

def rmse(y, x, model):
    return np.sqrt(mean_squared_error(y_true=y, y_pred=model.predict(x)))

def create_poly_data(data, degree):
    return np.hstack((data.reshape(-1,1)**i) for i in range(degree+1))
```

### What the plots show

**3 panels (one per model), log-scale RMSE vs training size:**

**1. Underfit model (Degree 1):**
- Train RMSE and test RMSE **converge quickly** to a **high value**
- Both curves plateau — adding more data doesn't help much
- **Diagnosis**: High bias problem → **need a more complex model**, not more data
- The gap between train and test is small (low variance) but the level is high (high bias)

**2. Right Fit model (Degree 5):**
- Train and test curves converge to a **low value**
- Test RMSE decreases as training size increases
- **Diagnosis**: Good model — more data continues to help up to a point
- Both curves are low and close together

**3. Overfit model (Degree 9):**
- Train RMSE is very low (near zero) but test RMSE is very high
- **Large gap** between train and test that persists even with more data
- **Diagnosis**: High variance problem → **need more data OR less complex model**
- The gap slowly closes as data increases (more data helps overfitting somewhat)

### How to read learning curves

```
RMSE
  |
  |  Train ----___           Train ----___
  |              ---___       Test  ----====---___
  |  Test  ----===----___         \
  |                    ---___     gap = variance
  |                          ---
  |    gap closes quickly         gap persists
  |    = low variance             = high variance
  |
  |  plateau at high error    plateau at low error  
  |  = high bias              = low bias (good!)
  +----------------------------------------------> Training Size
```

**Decision framework from learning curves:**

| What you see | Problem | Solution |
|-------------|---------|----------|
| Both curves plateau at high error | **High bias** (underfitting) | Use more complex model, add features, reduce regularization |
| Large persistent gap between curves | **High variance** (overfitting) | Get more data, simplify model, add regularization |
| Both curves low and converging | **Good fit** | You're in the sweet spot! |

---

## Key Takeaways

### 1. The Fundamental Tradeoff

```
Model Complexity ──────────────────────────────►

  UNDERFITTING              SWEET SPOT            OVERFITTING
  ┌──────────┐          ┌──────────────┐       ┌──────────────┐
  │ High Bias │          │ Low Bias     │       │ Low Bias     │
  │ Low Var   │   ───►   │ Low Variance │  ───► │ High Variance│
  │           │          │ = BEST ERROR │       │ = BAD ERROR  │
  └──────────┘          └──────────────┘       └──────────────┘
```

### 2. Why More Data Isn't Always the Answer

- If your model has **high bias**, adding data barely helps (the model is too simple)
- If your model has **high variance**, adding data helps (averages out the noise)
- **Learning curves tell you which problem you have** before you waste resources collecting more data

### 3. Practical Implications for Quant Finance

| Scenario | Diagnosis | Action |
|----------|-----------|--------|
| Linear model on stock returns → poor train AND test | High bias | Add nonlinear features (interactions, polynomials) |
| Deep neural net on 500 samples → great train, terrible test | High variance | Simplify model, get more data, add regularization |
| Random forest with 50 features → moderate train, bad test | High variance | Feature selection, reduce tree depth, more data |
| Ridge regression → moderate train AND test | Possible good fit | Try slightly more complex model (lower alpha) |

### 4. Connections to Other Notebook Topics

- **Mutual Information (Notebook 02)**: Feature selection reduces variance by removing noisy features
- **Cross-Validation (Notebook 04)**: CV estimates out-of-sample error to detect overfitting
- **Regularization (Notebook 05)**: Lasso/Ridge directly control the bias-variance tradeoff by penalizing complexity
- **Learning Curves**: Tell you whether to invest in **more data** or **more features/complexity**

### 5. The Taylor Series Connection

The notebook cleverly uses a Taylor polynomial of sin(x) as the true function. This is appropriate because:

- The true function is degree 5 (3 non-zero Taylor terms)
- A degree 1 model **underfits** (misses the curvature)
- A degree 5 model is the **right fit** (matches the truth)
- A degree 9+ model **overfits** (fits noise because it has more parameters than the truth requires)

The polynomial degree **IS the model complexity knob** that directly controls the bias-variance tradeoff.

---

## Quick Reference: Code Functions

| Function | Purpose |
|----------|---------|
| `f(x, max_degree)` | True function (Taylor approximation of sin) |
| `folds(train, test, nfolds)` | K-fold cross-validation data splitter |
| `rmse(y, x, model)` | Root mean squared error metric |
| `create_poly_data(data, degree)` | Transform x into polynomial features [1, x, x², ..., x^d] |

---

## For Non-Technical Readers

**Imagine you're learning to draw a portrait:**

- **Underfitting** = You can only draw stick figures. No matter how many photos you study, your drawings always look like stick figures. **Problem: Your technique is too simple. Solution: Learn more advanced techniques.**

- **Overfitting** = You memorize every shadow in one specific photo. When asked to draw a different person, you draw the original photo's shadows on the wrong face. **Problem: You memorized instead of learning general principles. Solution: Study more diverse photos (get more data) or simplify your technique.**

- **Right fit** = You learn the general principles of human faces. You can draw any person reasonably well. **This is the sweet spot.**

**Learning curves** answer: "Should I study more photos, or should I improve my drawing technique?" If your technique is too simple (stick figures), more photos won't help. If you're over-memorizing, more diverse photos WILL help.
