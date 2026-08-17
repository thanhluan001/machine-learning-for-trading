

## Scope correction: full theme-company insider coverage (2026-08-16)

The first run incorrectly reused `db_insider.h5`, which is keyed to S&P 400
permaTickers. That produced zero insider observations for the AI and crypto
panels. The study was corrected to fetch FMP Form 4 data directly by every
company ticker in the theme panels, including mega-cap leaders:

```text
AI:    MSFT GOOGL AMZN META NVDA AVGO ORCL
clean: FSLR ENPH SEDG NEE RUN PLUG
crypto:MSTR COIN RIOT MARA CLSK
```

News was already fetched directly by all theme tickers (not from the S&P 400
cache), although FMP coverage is uneven: AI news history was sparse in the
current response window, while clean-energy and crypto coverage was broader.

### Corrected coverage

```text
                 capex months   insider months   news months
AI/hyperscale          140           140              7
clean_energy           140           140            133
crypto                 140           140             88
```

### Corrected ablation

```text
price                         n=121   mean fwd6m= +1.4%   win 53.7%
capex                         n= 56   mean fwd6m=+18.9%   win 71.4%
price + capex                 n= 23   mean fwd6m=+11.7%   win 73.9%
price + capex + insider       n=  9   mean fwd6m=+20.0%   win 66.7%
price + capex + news          n=  4   mean fwd6m=+21.4%   win 75.0%
all four                      n=  4   mean fwd6m=+21.4%   win 75.0%
```

The larger insider panel does not change the architectural conclusion: the
joint samples remain too small for a production rule, and positive post-warning
returns show that capex warnings are often early. Full-market coverage fixes
the data error, but it does not yet validate insider/news as timing signals.

Latest context after correction:

```text
AI/hyperscale: price false, capex false, insider true, news true
clean energy : price true,  capex false, insider true, news true
crypto       : price true,  capex true,  insider true, news true
```

Use these only as monthly research context. No automatic exit or allocation
change is approved.
