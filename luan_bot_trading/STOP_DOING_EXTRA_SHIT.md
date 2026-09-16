# RULE: Do exactly what is asked. Nothing more.

When the user asks for X, only do X.
Do not run additional diagnostics, checks, code, or logs.
Do not "quickly verify" or "while we're at it" or "let me also check".
Do not make assumptions about what else might be wrong.
Do not investigate unless explicitly asked.

If asked "where are the 2 rows?", answer that.
Do not go and run more queries to see what else is broken.

If unsure whether something is needed, ASK before doing it.

## On Running Code

**Never execute user scripts without explicit permission.** "User scripts" =
any full pipeline / production / side-effect-bearing script in this repo
(e.g. `01_*.py` through `06_*.py`, `02b_*.py`, anything that writes a full
node to `db.h5`, runs a full batch through Tiingo/SEC/FRED/yahooquery,
downloads DERA zip files, or resets `*_offset.txt`). Getting timeouts or
aborts on these is costly and disrupts rate-limit batches.

State that the script is ready and ask before launching. Wait for the
user's "yes" / "go ahead" before running.

Verification code is OK and does not require permission. This includes:
- importing a module and/or checking a function exists
- running a function on tiny synthetic data
- one-off small/quick network calls that probe a few endpoints (e.g.
  checking whether a handful of tickers exist on Tiingo) — small request
  counts, fast
Use judgement: if a snippet hits a network endpoint many times, downloads
large files, or writes to persistent state, it is NOT verification code.
Apply the spirit of the rule: quick local checks fine; anything that could
burn your Tiingo hourly quota, write to `db.h5`, or take minutes → ASK first.

## On Edits

If an `edit` operation fails due to text-matching issues, do NOT retry it
repeatedly with ever-changing exact-match blocks. After one or two failed
edits, write the updated file to a new path and ask the user to overwrite
the original. Do not burn turns on brittle exact-match churn.

## On Tables

Always put a blank line before a markdown table. Without a preceding blank
line, the table may not render correctly.

## On Discrepancies

If you notice a data discrepancy (e.g. removed_date before added_date, suggesting
a ticker was removed then re-added), DO NOT launch an investigation.

Instead, raise a brief note in your response:
- State what you observed
- Suggest the likely cause in one sentence
- Wait for the user to decide next steps

## Environment Quirks (operational one-liners; full postmortems in ENVIRONMENT_INCIDENTS.md)

### Env quirk (2026-09-16) — RESOLVED SAME DAY (Incident 001)
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
- DIAGNOSIS TECHNIQUE for any future silent native crash: run with
  `python -X faulthandler` FIRST; then ctypes-call the suspected DLL directly
  to separate interpreter libraries from the native layer.
