# quant-portfolio-optimization

Quantitative portfolio optimization using fundamentalist screening, Markowitz mean-variance and Monte Carlo simulation

**Overview**

This is a quantitative finance tool that optimizes an equity portfolio through a systemized, two-stage pipeline. The script automates data ingestion from the Yahoo Finance API, computes fundamentalist factors (ROA and EV/EBITDA) for the selected asset pool and calculates optimal allocations using Markowitz's Modern Portfolio Theory. Through a Monte Carlo simulation of 10,000 randomized weight combinations, the model processes financial metrics locally via Parquet files and generates a Matplotlib scatter plot of the Efficient Frontier, explicitly highlighting the maximum Sharpe and minimum variance portfolios as the optimal investment boundaries.
