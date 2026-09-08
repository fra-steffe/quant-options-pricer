# Quantitative Options Pricing Engine

A Python-based modular engine for options pricing and Implied Volatility (IV) calculation. The system handles live market data extraction, ETL processing, and numerical inversion of the Black-Scholes model.

## Core Architecture

The project follows Object-Oriented Programming (OOP) principles, separating data ingestion from the mathematical solvers:

* **Data Connectors (`scrapers.py`):** Abstract base classes for data providers. Currently implements `YahooProvider` for live price and option chain extraction via `yfinance`.
* **ETL Pipeline (`transformers.py`):** Cleans raw market data. Drops illiquid contracts (volume = 0), handles `NaN` values, and normalizes expiration dates into time-to-maturity ($T$).
* **Instrument Abstraction (`instruments.py`):** Encapsulates financial logic into `EuropeanCall` and `EuropeanPut` objects.
* **Pricing & Solvers (`models.py`, `solvers.py`):** Implements the Black-Scholes pricing model. Uses the Newton-Raphson method with analytical Vega to compute the Implied Volatility.
* **Pipeline Orchestration (`main.py`):** Automates the workflow from dynamic spot price fetching to generating a clean `.csv` output of the IV surface.

## Known Limitations & Assumptions

* **Dividend Yield:** The current Black-Scholes implementation assumes a continuous dividend yield of $q = 0$. For dividend-paying underlying assets (e.g., AAPL), this mathematically generates an IV step across the ATM strike. The engine exports labeled Call/Put datasets to allow frontend separation rather than forcing continuous curve interpolation.
* **Interest Rates:** Assumes a flat, constant risk-free rate across all maturities.

## Quick Start

1. **Clone the repository and install dependencies:**

    git clone [https://github.com/fra-steffe/quant-options-pricer.git](https://github.com/fra-steffe/quant-options-pricer.git)
    cd quant-options-pricer
    pip install -r requirements.txt

2. **Run the engine:**

    python main.py

3. **Expected Output:**
   The terminal will display the live data extraction process and numerical inversion logs. Upon completion, a CSV file (e.g., `superficie_iv_AAPL_2024-12-31.csv`) containing the computed IV data will be generated in the root directory.

## Tech Stack

* **Language:** Python 3.10+ (tested on 3.13.5)
* **Data Processing & Math:** Pandas, NumPy, SciPy (Optimization)
* **Market Data:** yfinance

## Future Roadmap

- [ ] **Merton Model Integration:** Incorporate continuous dividend yields ($q$) to geometrically align Call and Put IV curves.
- [ ] **Interactive Dashboard:** Build a UI via `Streamlit` for dynamic 3D Volatility Surface rendering.
- [ ] **Advanced Models:** Expand the pricing engine to include stochastic volatility models (e.g., Heston Model).
- [ ] **Rate Limiting Defenses:** Implement request caching for multi-expiry extractions to prevent IP bans.