# Environment Setup — `trading` conda env

Reproducing the Python environment for this project.

## Quick path (exact, recommended)

```bash
conda env create -f environment.yml        # recreates `trading` with exact conda-forge builds
conda activate trading
```

`environment.yml` is a full `conda env export` snapshot (platform:
win-64, created 2026-09-01). If it fights your platform, use the pip
pins instead:

```bash
conda create -n trading python=3.11
conda activate trading
pip install -r requirements_frozen.txt
```

`requirements_frozen.txt` is `pip list --format=freeze` — 99 packages
with exact versions.

## Project-critical pins (what actually matters)

| Package | Version | Why it matters |
|---|---|---|
| `python` | 3.11 | env base |
| `pandas` | 3.0.3 | everything; note: pandas-3 behavior (string dtypes, `string_` arrays) bit us once — see features.md history |
| `tables` (PyTables) | 3.11.1 | HDF5 stores: db.h5, db_insider.h5, db_div.h5, db_sp500.h5, db_sp600.h5 |
| `xgboost` | 3.2.0 | frozen V6/V4 gate models (saved as classifier.json — Booster API) |
| `alpaca-py` | 0.43.5 | paper trading (TradingClient; note: SIP data is subscription-blocked, IEX works) |
| `fredapi` | 0.5.2 | FRED macros |
| `beautifulsoup4` | 4.15.0 | Wikipedia membership parsing |
| `requests` / `python-dotenv` | 2.34.2 / 1.2.2 | FMP/Tiingo/EDGAR APIs + `.env` keys |

## Keys needed in `luan_bot_trading/.env`

- `TIINGO_API_KEY` (paid tier — prices)
- `FMP_API_KEY` ($49/mo tier — earnings/grades/calendar/sec-filings)
- `FRED_API_KEY`
- `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` (paper)

## Known quirks (save yourself an evening)

- Run scripts via `conda run -n trading --no-capture-output python <script>`
  from the repo root; long fetches must be chunked (~890s tool timeout).
- Inline multiline python in `conda run` fails on Windows — write temp
  scripts to `%TEMP%` and run those.
- Script 51 replaces `sys.stdout` at import — never import it twice
  via `importlib` in one process (closed-buffer crash); load script 63
  which loads 51 as its own submodule.
