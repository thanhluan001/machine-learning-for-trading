# Additional Section: Checking Mutual Information Stability via Bootstrapping

## **Why Check Stability?**

When we compute mutual information (MI) from a finite dataset, the estimate contains **sampling noise**. If we had different data (or resampled the same data), we'd get slightly different MI values.

**Key questions:**
- Is a feature's high MI score **real** or just **random luck**?
- Which features have **reliable** predictive information?
- Which features are **noise** that happen to correlate by chance?

**Bootstrap stability analysis** answers these by:
1. Resampling the data many times (with replacement)
2. Computing MI each time
3. Looking at the **distribution** of MI scores for each feature
4. Features with **low variance** across bootstraps → stable, reliable
5. Features with **high variance** → unstable, likely spurious

---

## **Step-by-Step Implementation**

### **Step 1: Setup and Prepare Data**

```python
# We'll use the same setup as before
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns

# Load data (same as original notebook)
with pd.HDFStore('../data/assets.h5') as store:
    data = store['engineered_features']

# Prepare features and targets
target_labels = [f'target_{i}m' for i in [1, 2, 3, 6, 12]]
targets = data.dropna().loc[:, target_labels]
features = data.dropna().drop(target_labels, axis=1)

# Factorize categorical variables (simpler than dummies for stability analysis)
features.sector = pd.factorize(features.sector)[0]
cat_cols = ['year', 'month', 'msize', 'age', 'sector']
discrete_features = [features.columns.get_loc(c) for c in cat_cols]

# Choose a target horizon (e.g., 12-month forward return)
target_horizon = 'target_12m'
y_binary = (targets[target_horizon] > 0).astype(int)
```

---

### **Step 2: Bootstrap Function**

```python
def bootstrap_mi(X, y, discrete_features, n_bootstrap=100, random_state=42):
    """
    Compute mutual information scores across bootstrap samples.

    Parameters:
    -----------
    X : DataFrame, features
    y : Series, binary target
    discrete_features : list of indices for discrete features
    n_bootstrap : number of bootstrap iterations
    random_state : random seed for reproducibility

    Returns:
    --------
    mi_bootstrap : DataFrame of shape (n_bootstrap, n_features)
    """
    np.random.seed(random_state)
    n_samples = len(X)
    n_features = X.shape[1]

    # Storage for bootstrap results
    mi_bootstrap = np.zeros((n_bootstrap, n_features))

    # For each bootstrap sample
    for i in range(n_bootstrap):
        # Sample with replacement
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = X.iloc[idx]
        y_boot = y.iloc[idx]

        # Compute MI for this bootstrap sample
        mi_boot = mutual_info_classif(
            X_boot, y_boot,
            discrete_features=discrete_features,
            random_state=random_state + i
        )
        mi_bootstrap[i, :] = mi_boot

    # Convert to DataFrame for easier analysis
    mi_bootstrap_df = pd.DataFrame(
        mi_bootstrap,
        columns=X.columns,
        index=[f'bootstrap_{i}' for i in range(n_bootstrap)]
    )

    return mi_bootstrap_df
```

---

### **Step 3: Run Bootstrap**

```python
# Run bootstrap (this may take a few minutes with 100+ features)
# Use 200-500 bootstrap samples for more reliable estimates
mi_bootstrap = bootstrap_mi(features, y_binary, discrete_features, n_bootstrap=200)

print(f"Bootstrap shape: {mi_bootstrap.shape}")
mi_bootstrap.head()
```

---

### **Step 4: Analyze Stability**

#### **4.1 Summary Statistics**

```python
# Compute mean and standard deviation of MI across bootstraps
mi_mean = mi_bootstrap.mean()
mi_std = mi_bootstrap.std()
mi_cv = mi_std / (mi_mean + 1e-10)  # Coefficient of variation (avoid div by 0)

# Create summary DataFrame
mi_summary = pd.DataFrame({
    'mean_mi': mi_mean,
    'std_mi': mi_std,
    'cv_mi': mi_cv,
    'q25': mi_bootstrap.quantile(0.25),
    'q75': mi_bootstrap.quantile(0.75)
}).sort_values('mean_mi', ascending=False)

mi_summary.head(20)
```

**Interpretation:**
- **mean_mi**: Average MI across bootstraps → signal strength
- **std_mi**: How much MI varies across bootstraps → instability
- **cv_mi**: Relative variability (std/mean). Lower = more stable
- **q25, q75**: Interquartile range → robust spread measure

---

#### **4.2 Stability Plot: Mean vs Std**

```python
fig, ax = plt.subplots(figsize=(12, 8))

# Scatter: mean MI (x) vs std MI (y)
ax.scatter(mi_summary['mean_mi'], mi_summary['std_mi'], alpha=0.6)

# Label top features
top_features = mi_summary.head(10).index
for feat in top_features:
    ax.annotate(feat,
                (mi_summary.loc[feat, 'mean_mi'], mi_summary.loc[feat, 'std_mi']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=9)

ax.set_xlabel('Mean Mutual Information (across bootstraps)')
ax.set_ylabel('Standard Deviation of MI')
ax.set_title('Feature Stability: Mean MI vs. Bootstrap Variability')
ax.grid(True, alpha=0.3)

# Add guidance lines
ax.axhline(y=mi_summary['std_mi'].median(), color='red', linestyle='--',
           label=f'Median std: {mi_summary["std_mi"].median():.4f}', alpha=0.5)
ax.axvline(x=mi_summary['mean_mi'].median(), color='green', linestyle='--',
           label=f'Median mean: {mi_summary["mean_mi"].median():.4f}', alpha=0.5)
ax.legend()

plt.tight_layout()
plt.show()
```

**How to read this plot:**
- **Top-right corner**: High mean MI + high std → strong signal but unstable (maybe overfitting)
- **Top-left corner**: High mean MI + low std → **GREAT** - strong, stable signal ✓
- **Bottom-right**: Low mean MI + high std → noise, ignore
- **Bottom-left**: Low mean MI + low std → reliably weak (maybe true independence)

---

#### **4.3 Coefficient of Variation Plot**

```python
# Sort by CV (lowest = most stable)
mi_summary_cv = mi_summary.sort_values('cv_mi')

fig, ax = plt.subplots(figsize=(10, 6))
mi_summary_cv['cv_mi'].head(30).plot(kind='barh', ax=ax, color='steelblue')
ax.set_xlabel('Coefficient of Variation (std/mean)')
ax.set_title('Most Stable Features (lowest CV)')
ax.axvline(x=0.5, color='red', linestyle='--', label='CV = 0.5')
ax.legend()
plt.tight_layout()
plt.show()
```

**Rule of thumb:**
- CV < 0.3: Very stable
- CV < 0.5: Acceptable
- CV > 1.0: Unstable, high variance

---

### **Step 5: Select Stable Features**

```python
# Define criteria for selection:
# 1. Mean MI above threshold (e.g., 0.01 or top 20 features)
# 2. CV below threshold (e.g., 0.5)
# 3. Maybe also: lower bound of confidence interval > some minimum

mean_threshold = 0.01  # Minimum average MI
cv_threshold = 0.5     # Maximum coefficient of variation

stable_features = mi_summary[
    (mi_summary['mean_mi'] > mean_threshold) &
    (mi_summary['cv_mi'] < cv_threshold)
].sort_values('mean_mi', ascending=False)

print(f"Selected {len(stable_features)} stable features out of {len(features.columns)}")
print("\nTop stable features:")
print(stable_features.head(15)[['mean_mi', 'std_mi', 'cv_mi']])
```

---

### **Step 6: Compare with Original MI Selection**

```python
# Original MI (without stability check)
original_mi = mutual_info_classif(
    features, y_binary,
    discrete_features=discrete_features,
    random_state=42
)
original_mi_series = pd.Series(original_mi, index=features.columns).sort_values(ascending=False)

# Top 15 from original one-shot estimation
original_top15 = original_mi_series.head(15).index

# Compare
print("Features in original top-15 but NOT in stable selection:")
print(set(original_top15) - set(stable_features.index))

print("\nFeatures in stable selection but NOT in original top-15:")
print(set(stable_features.index) - set(original_top15))
```

---

### **Step 7: Visualize Bootstrap Distributions**

```python
def plot_mi_distribution(mi_bootstrap, feature_name, ax=None):
    """Plot histogram of bootstrap MI estimates for a given feature."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    values = mi_bootstrap[feature_name]
    ax.hist(values, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    ax.axvline(values.mean(), color='red', linestyle='--',
               label=f'Mean: {values.mean():.4f}')
    ax.axvline(values.quantile(0.025), color='orange', linestyle=':',
               label=f'2.5%: {values.quantile(0.025):.4f}', alpha=0.7)
    ax.axvline(values.quantile(0.975), color='orange', linestyle=':',
               label=f'97.5%: {values.quantile(0.975):.4f}', alpha=0.7)

    ax.set_xlabel('Mutual Information')
    ax.set_ylabel('Count (bootstrap samples)')
    ax.set_title(f'Bootstrap Distribution: {feature_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax

# Example: compare a stable vs unstable feature
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# Most stable feature (lowest CV)
stable_feature = mi_summary_cv.index[0]
plot_mi_distribution(mi_bootstrap, stable_feature, ax=axes[0])
axes[0].set_title(f'STABLE: {stable_feature} (CV={mi_summary_cv.iloc[0]["cv_mi"]:.3f})')

# Most unstable feature from top 20 (highest CV)
unstable_candidates = mi_summary[mi_summary['mean_mi'] > 0.01].sort_values('cv_mi', ascending=False)
if len(unstable_candidates) > 0:
    unstable_feature = unstable_candidates.index[0]
    plot_mi_distribution(mi_bootstrap, unstable_feature, ax=axes[1])
    axes[1].set_title(f'UNSTABLE: {unstable_feature} (CV={mi_summary.loc[unstable_feature, "cv_mi"]:.3f})')

plt.tight_layout()
plt.show()
```

**What to look for:**
- **Stable feature**: Narrow distribution, tight confidence interval
- **Unstable feature**: Wide distribution, overlapping zero (MI might be zero in some samples)
- If 95% CI includes zero → MI might not be significantly different from 0

---

### **Step 8: Optional - Statistical Significance Test**

We can compute a **simple signal-to-noise ratio**:

```python
# Signal-to-noise: mean MI / std MI (similar to CV but inverted)
mi_summary['signal_to_noise'] = mi_summary['mean_mi'] / (mi_summary['std_mi'] + 1e-10)

# Sort by S/N ratio
mi_summary_snr = mi_summary.sort_values('signal_to_noise', ascending=False)

print("Top features by Signal-to-Noise ratio:")
mi_summary_snr[['mean_mi', 'std_mi', 'signal_to_noise']].head(15)
```

**Rule of thumb**:
- S/N > 2: Strong, stable signal
- 1 < S/N < 2: Moderate signal, use with caution
- S/N < 1: Weak or noisy signal

---

### **Step 9: Apply to Dummy-Encoded Features (Optional)**

If you want to check stability of MI with one-hot encoded features:

```python
# Create dummy variables (from earlier)
dummy_data = pd.get_dummies(data,
                            columns=['year','month', 'msize', 'age',  'sector'],
                            prefix=['year','month', 'msize', 'age', ''],
                            prefix_sep=['_', '_', '_', '_', ''])

dummy_features = dummy_data.dropna().drop(target_labels, axis=1)
cat_cols_dummy = [c for c in dummy_features.columns if c not in features.columns]
discrete_dummy = [dummy_features.columns.get_loc(c) for c in cat_cols_dummy]

# Run bootstrap on dummy features (this will be slower due to many columns)
# You might want to use fewer bootstrap samples or select features first
# mi_bootstrap_dummy = bootstrap_mi(dummy_features, y_binary, discrete_dummy, n_bootstrap=100)
```

**Note**: With many dummy variables (120+ columns), especially for categoricals with many levels:
- Variance will be higher for rare categories
- MI estimates less reliable
- Consider grouping rare categories into "Other"

---

### **Step 10: Using Stability Analysis for Feature Selection**

```python
# Combine stability criteria with MI magnitude
def select_stable_features(mi_summary, mean_threshold=0.01, cv_threshold=0.5, min_bootstrap_pct=0.05):
    """
    Select features that are both informative (mean MI) and stable (low CV).

    Parameters:
    -----------
    mi_summary : DataFrame with 'mean_mi', 'std_mi', 'cv_mi', 'q25' columns
    mean_threshold : minimum average MI
    cv_threshold : maximum coefficient of variation
    min_bootstrap_pct : optional: minimum fraction of bootstrap samples where MI > 0

    Returns:
    --------
    stable_features : list of selected feature names
    """
    # Filter by mean MI and CV
    stable = mi_summary[
        (mi_summary['mean_mi'] > mean_threshold) &
        (mi_summary['cv_mi'] < cv_threshold)
    ]

    # Optionally: ensure MI is positive in most bootstrap samples
    if 'q25' in stable.columns:
        stable = stable[stable['q25'] > 0]  # 75% of bootstraps have positive MI

    return stable.index.tolist()

# Apply selection
selected_features = select_stable_features(mi_summary, mean_threshold=0.01, cv_threshold=0.5)

print(f"Selected {len(selected_features)} stable features")
print("Selected features:", selected_features)

# Compare with naive top-20 selection
naive_top20 = mi_summary.head(20).index.tolist()
print(f"\nNaive top-20 had {len(naive_top20)} features")
print(f"Stable selection kept {len(set(naive_top20) & set(selected_features))} of those")
```

---

## **Interpretation Guidelines**

### **What is "stable" enough?**

| CV Range | Interpretation | Recommendation |
|----------|----------------|----------------|
| **0.0 - 0.3** | Very stable | Keep, strong signal |
| **0.3 - 0.5** | Moderately stable | Keep, but monitor |
| **0.5 - 1.0** | Somewhat unstable | Consider dropping or using regularization |
| **> 1.0** | Very unstable | Likely noise, drop |

### **Bootstrap Sample Size**

- **Minimum**: 100 bootstrap samples (quick, rough)
- **Good**: 200-500 (reliable, still fast for <200 features)
- **Comprehensive**: 1000+ (for critical applications, may be slow)

**Trade-off**: More bootstraps = more precise stability estimates, but slower.

---

## **Why This Matters**

### **Scenario 1: High MI, High Variance**
```
Feature X: MI = 0.08 (good), but CV = 1.2 (high variance)
```
**Interpretation**: Sometimes appears informative, sometimes not. Might be overfitting to noise.
**Action**: Drop this feature even though its single-shot MI looks good.

### **Scenario 2: Moderate MI, Low Variance**
```
Feature Y: MI = 0.025 (moderate), CV = 0.25 (very stable)
```
**Interpretation**: Consistently provides some information, reliably.
**Action**: Keep this feature - it's robust.

### **Scenario 3: Low MI, Low Variance**
```
Feature Z: MI = 0.005 (low), CV = 0.3 (stable)
```
**Interpretation**: Reliably provides almost no information.
**Action**: Drop - not worth the complexity.

### **Scenario 4: High MI, Low Variance**
```
Feature W: MI = 0.12 (high), CV = 0.2 (stable)
```
**Interpretation**: Strong, reliable signal.
**Action**: Definitely keep, maybe give more weight in model.

---

## **Common Pitfalls**

1. **Not enough bootstrap samples**: Using too few (e.g., 20) gives unreliable CV estimates. Use at least 100.

2. **Ignoring the target definition**: We binarized returns. If you used continuous target with `mutual_info_regression`, stability patterns might differ.

3. **No across-horizon analysis**: MI varies by prediction horizon (1m vs 12m). Consider stability **for each horizon separately**, then look for features that are stable **across horizons** (more robust).

4. **Assuming stationarity**: If your data spans different market regimes (bull/bear), MI stability might vary across time. Consider **time-based block bootstrapping** instead of simple row resampling.

5. **Looking only at mean MI**: The **distribution** matters. A feature with mean MI=0.03 but 50% of bootstraps have MI<0.01 is unstable.

---

## **Advanced: Time Series Block Bootstrapping**

Financial data has **temporal correlation** (today's return correlated with yesterday's). Simple row resampling breaks this structure.

**Block bootstrap**:
```python
def block_bootstrap_mi(X, y, block_size=20, n_bootstrap=100, random_state=42):
    """
    Block bootstrap for time series data.
    Randomly samples contiguous blocks instead of individual rows.
    """
    np.random.seed(random_state)
    n_samples = len(X)
    n_features = X.shape[1]
    mi_bootstrap = np.zeros((n_bootstrap, n_features))

    # Number of blocks needed
    n_blocks = int(np.ceil(n_samples / block_size))

    for i in range(n_bootstrap):
        # Randomly sample blocks
        block_starts = np.random.choice(
            n_samples - block_size,
            size=n_blocks,
            replace=True
        )
        idx = np.concatenate([np.arange(s, min(s+block_size, n_samples)) for s in block_starts])
        idx = idx[:n_samples]  # truncate to original size

        X_boot = X.iloc[idx]
        y_boot = y.iloc[idx]

        mi_boot = mutual_info_classif(
            X_boot, y_boot,
            discrete_features=discrete_features,
            random_state=random_state + i
        )
        mi_bootstrap[i, :] = mi_boot

    return pd.DataFrame(mi_bootstrap, columns=X.columns)
```

Use block size of 20-60 (1-3 months of weekly data) to preserve autocorrelation.

---

## **Putting It All Together**

**Complete workflow with stability check**:

```python
# 1. Compute one-shot MI (quick screen)
mi_initial = mutual_info_classif(features, y_binary, discrete_features=discrete_features)
mi_initial_series = pd.Series(mi_initial, index=features.columns)

# 2. Take top 30-50 features from initial screen (reduce computation)
top_candidates = mi_initial_series.sort_values(ascending=False).head(50).index
X_candidates = features[top_candidates]

# Update discrete_features indices for reduced set
discrete_candidates = [X_candidates.columns.get_loc(c) for c in cat_cols if c in X_candidates.columns]

# 3. Bootstrap only on candidates (faster)
mi_boot_candidates = bootstrap_mi(
    X_candidates, y_binary, discrete_candidates,
    n_bootstrap=200
)

# 4. Analyze stability
mi_boot_summary = pd.DataFrame({
    'mean_mi': mi_boot_candidates.mean(),
    'std_mi': mi_boot_candidates.std(),
    'cv_mi': mi_boot_candidates.std() / mi_boot_candidates.mean()
}).sort_values('mean_mi', ascending=False)

# 5. Select stable features
stable_selection = mi_boot_summary[
    (mi_boot_summary['mean_mi'] > 0.01) &
    (mi_boot_summary['cv_mi'] < 0.5)
].index.tolist()

print(f"From 50 candidates, selected {len(stable_selection)} stable features")
```

---

## **Summary Table: Feature Selection with Stability**

| Feature | One-shot MI | Bootstrap Mean MI | Bootstrap Std | CV | Decision |
|---------|-------------|-------------------|---------------|----|----------|
| momentum_12 | 0.045 | 0.044 | 0.008 | 0.18 | ✓ Keep (high, stable) |
| return_1m | 0.038 | 0.039 | 0.015 | 0.38 | ✓ Keep (stable) |
| sector_Unknown | 0.025 | 0.008 | 0.012 | 1.50 | ✗ Drop (unstable, maybe spurious) |
| year_2008 | 0.022 | 0.006 | 0.011 | 1.83 | ✗ Drop (high variance, likely noise) |
| sqft_living | 0.015 | 0.016 | 0.005 | 0.31 | ✓ Keep (consistent) |

---

## **Conclusion**

**Bootstrap stability analysis** helps you distinguish **true signal** from **sampling noise** in feature selection.

**Key benefits**:
- Identifies features with **reliable** mutual information
- Filters out **spurious correlations** that only appear due to random chance
- Gives **confidence intervals** for MI estimates
- Enables **data-driven threshold** for feature selection (instead of arbitrary cutoff)

**Recommended practice**:
1. Always check stability when selecting features based on any statistical measure (MI, correlation, etc.)
2. Use at least 200 bootstrap samples
3. Set thresholds for both mean MI (signal strength) and CV (stability)
4. Consider block bootstrap for time series data
5. Document which features are stable vs unstable for reproducibility

---

**Add to notebook** after the "Dummy Data" section and before any final feature selection or modeling steps.
