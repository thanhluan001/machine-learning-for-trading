# SIC-to-Index Deterministic Mapping Architecture

This file contains the definitive structural rules for mapping individual equities to their respective pure factor or sector sub-indexes based on their Standard Industrial Classification (SIC) code. 

## Deterministic SIC-to-Index Component Map

| SIC Code Range / Prefix | Standard Industry Sector | Target Regression Benchmark Ticker | Structural Justification |
| :--- | :--- | :--- | :--- |
| `13xx`, `29xx` | Energy | **IJS** | Pure iShares S&P Small/MidCap Energy representation. |
| `10xx-14xx`, `24xx-26xx`, `28xx`, `33xx` *(excl. 283x)* | Materials | **XLB** | Uses the SPDR Materials Select Sector ETF for core industrial commodity pricing. |
| `15xx-17xx`, `34xx`, `35xx`, `37xx`, `40xx-47xx` *(excl. 357x)* | Industrials & Cyclicals | **IJJ** | Captures asset-heavy, value-driven industrial risk profiles via iShares S&P MidCap 400 Value. |
| `20xx-23xx`, `30xx-31xx`, `36xx`, `39xx`, `51xx`, `54xx`, `52xx-59xx`, `70xx-72xx`, `75xx-77xx` *(excl. 367x)* | Consumer Staples & Discretionary (Value) | **IJJ** | Consolidates asset-heavy consumer value and cyclical risk exposures into a liquid proxy. |
| `60xx-64xx`, `67xx` *(excl. 6770)* | Financials | **XLF** | Employs the Financial Select Sector SPDR to ensure deep historical data liquidity on Tiingo. |
| `65xx` | Real Estate | **XLRE** | Isolates pure-play real estate and REIT performance via the Real Estate Select Sector SPDR. |
| `38xx`, `48xx`, `73xx`, `78xx`, `79xx`, `80xx`, `27xx` *(incl. overrides)* | Technology, Healthcare & Growth Services | **IJK** | Captures systemic high-beta growth momentum, software, tech-platforms, and high-multiple entertainment services. |
| `49xx` | Utilities | **XLU** | Employs the Utilities Select Sector SPDR to ensure robust historical data continuity on Tiingo. |

## Structural Classification Overrides
*The following specific rules take execution precedence over the broad range mappings above:*

* **The Restructuring Exception (`6770`):** Any asset displaying an SIC code matching exactly `6770` (Blank Checks / Non-Operating Entities undergoing restructuring) must bypass the **Financials (XLF)** mapping block and route directly to the **Default Mapping (IJH)** to prevent sector distortion.
* **The Pharmaceutical Exception (`283x`):** Any SIC code starting with `283` (e.g., `2834` - Pharmaceutical Preparations) must bypass the **Materials (XLB)** mapping and route into the **Growth Cluster (IJK)**.
* **The Computing Hardware Exception (`357x`):** Any SIC code starting with `357` (e.g., `3571` - Electronic Computers) must bypass the **Industrials (IJJ)** mapping and route into the **Growth Cluster (IJK)**.
* **The Semiconductor Exception (`367x`):** Any SIC code starting with `367` (e.g., `3674` - Semiconductors) must bypass the **Consumer Cyclicals (IJJ)** mapping and route into the **Growth Cluster (IJK)**.

## Default Mapping
Any SIC code that is not explicitly matched by any range or override rule above defaults to the core mid-cap market blend: **IJH**.