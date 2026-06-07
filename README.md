# quant-portfolio-optimization

Quantitative portfolio optimization using fundamentalist screening, Markowitz mean-variance and Monte Carlo simulation

### Overview

This is a quantitative finance tool that optimizes an equity portfolio through a systemized, two-stage pipeline. The script automates data ingestion, computes fundamentalist factors for the selected asset pool and calculates optimal allocations using Markowitz's Modern Portfolio Theory. Through a Monte Carlo simulation of 10,000 randomized weight combinations, the model generates a scatter plot of the Efficient Frontier, highlighting the optimal investment boundaries.

### Features
- Fetches real-time, 5-year historical close prices and financial statements directly from the Yahoo Finance API ('yfinance').
- Ranks assets systematically by combining operational efficiency (Return on Assets - ROA) and market valuation multiples (EV/EBITDA).
- Saves raw data and simulation results locally using compressed Parquet files to ensure data persistence and fast executions.
- Runs a Monte Carlo simulation across 10,000 randomized portfolios, computing annualized returns, expected volatilities, and covariances.
- Plots a complete risk-return scatter map using Matplotlib, color-coded by the Sharpe ratio, highlighting the Maximum Sharpe and Minimum Variance portfolios.
