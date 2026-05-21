# Project Memory: ML for Trading

**Last Updated:** 2026-05-14

## Quick Summary
This is the code repository for Stefan Jansen's "Machine Learning for Trading" (2nd Edition). It contains ~150 Jupyter notebooks demonstrating ML applications in algorithmic trading across 24 chapters.

## Project Root Structure
```
machine-learning-for-trading/
├── 01_machine_learning_for_trading/ - 24 numbered chapter dirs with notebooks
├── data/                          # Data files (~1.6GB)
│   ├── assets.h5 (1.6GB)          # HDF5 asset database
│   ├── wiki_prices.csv (1.8GB)    # Wikipedia stock prices
│   ├── spx_d.csv                  # S&P 500 daily data
│   ├── stock_data.csv             # Stock price data
│   ├── us_equities_meta_data.csv  # Equity metadata
│   ├── bbc.zip, earnings_calls.zip # Text data
│   ├── mnist/, fashion_mnist/     # ML datasets
│   └── create_*.ipynb             # Data creation notebooks
├── figures/                       # Output figures
├── assets/                        # Static assets
├── installation/                  # Setup instructions
├── README.md                      # Full documentation
├── utils.py                       # Utility functions
├── stock_data.csv                 # Stock data
├── .env                           # Environment variables
└── package.json                   # Minimal Node.js setup
```

## Book Structure (4 Parts + Appendix)

### Part 1: Data to Strategy (01-05)
- 01: ML for Trading intro
- 02: Market & Fundamental Data
- 03: Alternative Data
- 04: Financial Feature Engineering
- 05: Portfolio Optimization

### Part 2: ML Fundamentals (06-13)
- 06: ML Process
- 07: Linear Models
- 08: ML4T Workflow / Backtesting
- 09: Time Series Models
- 10: Bayesian ML
- 11: Random Forests
- 12: Gradient Boosting (XGBoost, LightGBM, CatBoost)
- 13: Unsupervised Learning (PCA, Clustering)

### Part 3: NLP for Trading (14-16)
- 14: Text Data / Sentiment Analysis
- 15: Topic Modeling (LDA)
- 16: Word Embeddings (Word2Vec, BERT)

### Part 4: Deep Learning (17-22)
- 17: Deep Learning Intro
- 18: CNN (time series, satellite images)
- 19: RNN/LSTM (multivariate, sentiment)
- 20: Autoencoders (risk factors)
- 21: GANs (synthetic time series)
- 22: Deep Reinforcement Learning

### Appendix
- 24: Alpha Factor Library (100+ factors, TA-Lib, WorldQuant alphas)

## Key Libraries Used
- pandas, numpy, scipy
- scikit-learn, XGBoost, LightGBM, CatBoost
- TensorFlow, Keras, PyTorch
- statsmodels, pymc3
- spaCy, NLTK, gensim
- Zipline, backtrader, pyfolio
- TA-Lib, Alphalens
- BeautifulSoup, requests

## Data Sources Mentioned
- Quandl, Algoseek, Yahoo Finance
- SEC filings (XBRL)
- Bloomberg, FactSet
- Reuters, Compustat
- Twitter, news APIs

## Important Notes
- Zipline backtesting uses custom fork (zipline-reloaded)
- Data creation notebooks in `data/` directory
- Installation instructions in `installation/` directory
- Community forum at exchange.ml4trading.io

## Common Tasks
- Backtesting: Zipline, backtrader
- Factor evaluation: Alphalens, pyfolio
- Data storage: HDF5 (.h5 files), CSV
- Research notebooks: Jupyter

## File: utils.py
Contains shared utility functions for the project.

## File: .env
Environment variables (API keys, paths, etc.)