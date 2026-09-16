# Environment Incidents — permanent log

Each entry: full postmortem, the diagnosis technique that worked, the fix,
and the verification. Purpose: future silent crashes get diagnosed in
minutes, and environment rebuilds never resurrect a fixed bug.

---

## Incident 001 — MKL 2026.0.0 GEMM delay-load crash (2026-09-16, RESOLVED)

**Impact window:** unknown start (first exercise of 2-D BLAS on this env) —
2026-09-16. Surfaced during the RC-18 threshold probe (`pandas.Series.corr`
on OOS scores). Likely latent since env creation 2026-09-01: nothing in the
nightly stack (XGBoost trains on its own libs; features use element-wise
numpy) ever calls GEMM. `np.dot` on 1-D also avoids it.

### Symptom

- `pandas.Series.corr/.cov`, `np.corrcoef`, `np.cov`, and ANY 2-D matmul
  (`A @ B`, `np.matmul`) hard-crash the process: **rc=127, zero output,
  no traceback**. Block-buffered stdout makes it look like the script
  "ran and produced nothing."
- 1-D `np.dot(a, b)` works. `mean/var/rank` work. xgboost fits work.

### Diagnosis chain (the reproducible path)

```text
1. python -X faulthandler -u -c "<crash repro>"
     -> "Windows fatal exception: code 0xc06d007f"
        stack: np.cov -> numpy _function_base_impl -> pandas nancorr
        THE key move: faulthandler prints the C-level stack that the
        default silent death hides. Use this FIRST on any native crash.

2. Exception code 0xc06d007f = delayimp raise = ERROR_PROC_NOT_FOUND
   (0x7F=127): a delay-loaded DLL loaded, but a SYMBOL it exports-
   imports was missing. Not a file-not-found.

3. Bypass numpy: ctypes.WinDLL(libcblas.dll); call cblas_dgemm directly
   -> same exception. numpy/pandas innocent; the BLAS backend itself.

4. Ruled out:
   - mixed DLL versions (all mkl_*.dll = 2026.0.0, same build date)
   - DLL search path (os.add_dll_directory + PATH prepend: no effect)
   - kernel dispatch choice (MKL_ENABLE_INSTRUCTIONS=AVX2/SSE4_2/AVX512:
     no effect — failure fires before dispatch matters)

5. Root cause: conda-forge mkl 2026.0.0's GEMM kernel-dispatch delay-load
   is broken on this machine (CPU-dispatch -> delay-load -> proc-missing).
```

### Fix

```bash
conda install -n trading "libblas=*=*openblas"
# surgical: mkl removed, libopenblas 0.3.34 installed; numpy/pandas/
# xgboost untouched; numexpr rebuilt against openblas.
```

`environment.yml` re-exported from the fixed env (it previously PINNED the
broken packages — `mkl=2026.0.0`, `libblas=3.11.0=8_*_mkl` — so a rebuild
would have recreated the crash).

### Verification (run after any BLAS-touching env change)

```bash
python -u -c "
import numpy as np, pandas as pd
a = np.random.rand(300, 300); print('gemm', (a @ a.T)[0, 0])
s1 = pd.Series(np.random.rand(3000)); s2 = pd.Series(np.random.rand(3000))
print('corr', s1.corr(s2), 'spearman', s1.corr(s2, method='spearman'))
import xgboost as xgb
X = np.random.rand(2000, 10); y = (np.random.rand(2000) > .8).astype(int)
xgb.XGBClassifier(n_estimators=50).fit(X, y, verbose=False)
print('ALL VERIFIED')"
```

Cross-check performed: native spearman on the probe data reproduced the
manual-workaround value exactly (-0.0205).

### Residual notes

- BLAS swap (MKL -> OpenBLAS) may shift floating-point results in the last
  ulp; nothing in the nightly stack used GEMM, so no behavioral change
  expected. If an archived number differs microscopically from a rerun,
  this swap is the reason.
- pandas-3 gotchas that remain (not BLAS-related):
    `fillna(DatetimeIndex)` rejected (use a Series);
    `Series.zfill` gone (use `.str.zfill`);
    string-dtype columns break Timestamp comparisons (coerce first).
- If MKL is ever wanted again: pin `mkl<2026` and re-verify with the
  snippet above before trusting the env.

---

## Template for future entries

```text
## Incident NNN — <one-line title> (date, STATUS)
Impact window / Symptom / Diagnosis chain (commands + what ruled out) /
Fix (exact commands) / Verification snippet / Residual notes
```
