# Cross-Validation Techniques Explained

This document explains the cross-validation techniques demonstrated in the `04_cross_validation.py` script. Cross-validation is a fundamental technique for evaluating machine learning models by splitting the dataset into training and validation sets in various ways to ensure robust performance estimation.

---

## 1. **Import Libraries**

The script uses the following cross-validation strategies from `scikit-learn`:

```python
from sklearn.model_selection import (
    train_test_split, 
    KFold, 
    LeaveOneOut, 
    LeavePOut, 
    ShuffleSplit, 
    TimeSeriesSplit
)
```

| Method               | Purpose                                                                                     |
|----------------------|---------------------------------------------------------------------------------------------|
| `train_test_split`   | Splits data into random training and validation subsets.                                   |
| `KFold`              | Splits data into *k* folds for *k*-fold cross-validation.                                   |
| `LeaveOneOut`        | Leaves one sample out for validation in each iteration.                                     |
| `LeavePOut`          | Leaves *p* samples out for validation in each iteration.                                    |
| `ShuffleSplit`       | Randomly shuffles and splits data into training and validation sets multiple times.         |
| `TimeSeriesSplit`    | Splits time-series data while preserving temporal order.                                    |

---

## 2. **Sample Data**

The script uses a simple list of numbers for demonstration:

```python
data = list(range(1, 11))  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```

---

## 3. **`train_test_split`**

### **Purpose**
Splits data into **80% training** and **20% validation** subsets randomly.

### **Code**
```python
print(train_test_split(data, train_size=.8))
```

### **Output Example**
```python
([2, 3, 5, 6, 7, 8, 9, 10], [1, 4])  # Training set: 8 samples, Validation set: 2 samples
```

### **Key Points**
- Simple and fast for quick model evaluation.
- Not robust for small datasets due to high variance in performance estimates.

---

## 4. **`KFold` (Standard *k*-Fold Cross-Validation)**

### **Purpose**
Splits data into *k* folds (groups) for cross-validation, ensuring each sample is used for validation exactly once.

### **Code**
```python
kf = KFold(n_splits=5)
for train, validate in kf.split(data):
    print(train, validate)
```

### **Output Example**
```python
[2 3 4 5 6 7 8 9] [0 1]  # Train on indices 2-9, validate on 0-1
[0 1 4 5 6 7 8 9] [2 3]  # Train on indices 0-1,4-9, validate on 2-3
[0 1 2 3 6 7 8 9] [4 5]
[0 1 2 3 4 5 8 9] [6 7]
[0 1 2 3 4 5 6 7] [8 9]
```

### **Key Points**
- **Robust**: Reduces variance in performance estimates compared to a single train-test split.
- **Use Case**: Works well for most datasets, especially those with balanced distributions.

---

## 5. **`KFold` with Shuffling**

### **Purpose**
Same as `KFold`, but **shuffles the data** before splitting to avoid order bias.

### **Code**
```python
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train, validate in kf.split(data):
    print(train, validate)
```

### **Output Example**
```python
[0 1 3 4 5 6 8 9] [2 7]
[0 1 2 4 5 6 7 9] [3 8]
[0 1 2 3 5 6 7 8] [4 9]
[1 2 3 4 5 6 7 8] [0 9]
[0 2 3 4 5 6 7 8] [1 9]
```

### **Key Points**
- **Avoids Order Bias**: Useful if the data is ordered (e.g., sorted by target value).
- **Reproducibility**: `random_state` ensures the same shuffle every time.

---

## 6. **`LeaveOneOut`**

### **Purpose**
Leaves **one sample out** for validation in each iteration, maximizing the use of training data.

### **Code**
```python
loo = LeaveOneOut()
for train, validate in loo.split(data):
    print(train, validate)
```

### **Output Example**
```python
[1 2 3 4 5 6 7 8 9] [0]  # Validate on index 0 (value 1)
[0 2 3 4 5 6 7 8 9] [1]  # Validate on index 1 (value 2)
[0 1 3 4 5 6 7 8 9] [2]
...
[0 1 2 3 4 5 6 7 8] [9]  # Validate on index 9 (value 10)
```

### **Key Points**
- **Maximizes Training Data**: Uses all but one sample for training.
- **Computationally Expensive**: Requires *n* model fits (where *n* is the number of samples).
- **Use Case**: Best for **small datasets** where every sample counts.

---

## 7. **`LeavePOut`**

### **Purpose**
Leaves **p samples out** for validation in each iteration, testing all possible combinations.

### **Code**
```python
lpo = LeavePOut(p=2)
for train, validate in lpo.split(data):
    print(train, validate)
```

### **Output Example**
```python
[2 3 4 5 6 7 8 9] [0 1]  # Validate on indices 0 and 1 (values 1, 2)
[1 3 4 5 6 7 8 9] [0 2]  # Validate on indices 0 and 2 (values 1, 3)
[1 2 4 5 6 7 8 9] [0 3]
...
[0 1 2 3 4 5 6 7] [8 9]  # Validate on indices 8 and 9 (values 9, 10)
```

### **Key Points**
- **Exhaustive Validation**: Tests all combinations of *p* samples.
- **Computationally Expensive**: Total iterations = *n choose p* (here, `10 choose 2` = 45).
- **Use Case**: Useful for **small datasets** where you want to test all possible combinations.

---

## 8. **`ShuffleSplit`**

### **Purpose**
Randomly shuffles and splits data into training and validation sets **multiple times**.

### **Code**
```python
ss = ShuffleSplit(n_splits=3, test_size=2, random_state=0)
for train, validate in ss.split(data):
    print(train, validate)
```

### **Output Example**
```python
[1 2 4 8 9 3 6 7] [5 0]  # First split (random)
[1 2 5 0 8 3 7 6] [4 9]  # Second split
[7 6 1 0 8 4 5 3] [2 9]  # Third split
```

### **Key Points**
- **Flexible**: Generates multiple independent splits.
- **Use Case**: Useful when you want **multiple random splits** without strict folding.

---

## 9. **`TimeSeriesSplit`**

### **Purpose**
Splits **time-series data** while preserving temporal order to avoid lookahead bias.

### **Code**
```python
tscv = TimeSeriesSplit(n_splits=5)
for train, validate in tscv.split(data):
    print(train, validate)
```

### **Output Example**
```python
[0 1 2] [3]      # Train on 1,2,3; validate on 4
[0 1 2 3] [4]     # Train on 1,2,3,4; validate on 5
[0 1 2 3 4] [5]    # Train on 1,2,3,4,5; validate on 6
[0 1 2 3 4 5] [6]
[0 1 2 3 4 5 6] [7]
```

### **Key Points**
- **Preserves Temporal Order**: Ensures validation data comes after training data.
- **Use Case**: Essential for **time-series forecasting** (e.g., stock prices).

---

## **Comparison of Cross-Validation Methods**

| Method               | Use Case                          | Pros                                      | Cons                                      |
|----------------------|-----------------------------------|-------------------------------------------|-------------------------------------------|
| `train_test_split`   | Quick model evaluation            | Fast, simple                              | High variance in performance estimates    |
| `KFold`              | General-purpose cross-validation  | Robust, low variance                      | Assumes data is unordered                 |
| `KFold (shuffle)`    | Ordered or imbalanced data        | Avoids order bias                         | Slightly slower due to shuffling          |
| `LeaveOneOut`        | Small datasets                    | Uses all data for training                | Computationally expensive                 |
| `LeavePOut`          | Exhaustive validation             | Tests all combinations                    | Very slow for large datasets              |
| `ShuffleSplit`       | Flexible splits                   | Multiple random splits                    | Not as structured as *k*-fold             |
| `TimeSeriesSplit`    | Time-series data                  | Preserves temporal order                  | Cannot shuffle data                       |

---

## **When to Use Which?**
- **For most machine learning tasks**: Use `KFold` (or `KFold` with `shuffle=True`).
- **For small datasets**: Use `LeaveOneOut` or `LeavePOut`.
- **For time-series data**: Use `TimeSeriesSplit`.
- **For flexible, multiple random splits**: Use `ShuffleSplit`.

---