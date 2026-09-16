
## Env quirk (2026-09-16) — RESOLVED SAME DAY
- SYMPTOM: pandas Series.corr/.cov and ANY 2-D matmul (np gemm) hard-crashed the
  process (Windows fatal exception 0xc06d007f = delayimp ERROR_PROC_NOT_FOUND,
  rc=127, no traceback). 1-D np.dot was fine. -X faulthandler pinned the crash
  to numpy np.cov -> libcblas -> MKL; direct ctypes cblas_dgemm reproduced it,
  so numpy/pandas were innocent.
- ROOT CAUSE: conda mkl 2026.0.0's GEMM kernel-dispatch delay-load is broken on
  this machine (coherent single-version install; MKL_ENABLE_INSTRUCTIONS could
  not bypass; add_dll_directory/PATH irrelevant).
- FIX: conda install -n trading "libblas=*=*openblas" (mkl removed,
  libopenblas 0.3.34 in; numpy/pandas/xgboost untouched). Verified: gemm,
  pearson, spearman, cov, xgboost fit, HDF read — and native spearman on the
  probe data reproduces the manual-workaround value (-0.0205).
- Note: BLAS swap (MKL->OpenBLAS) may shift floating-point last-ulp results;
  nothing in the nightly stack used gemm, so no behavioral change expected.
- pandas-3 gotchas that remain: fillna(DatetimeIndex) rejected; Series.zfill
  gone (.str.zfill).
