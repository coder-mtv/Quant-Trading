# Pairs Trading Strategy with Cointegration & Risk Management

This project implements a **quantitative pairs trading strategy** using **cointegration testing** to identify stock pairs with mean-reverting relationships. The strategy generates **long/short trading signals** when the spread deviates from its rolling mean and incorporates a **stop-loss mechanism** to enhance risk-adjusted returns.

## Key Features
- **Cointegration Testing:** Uses **OLS regression + Augmented Dickey-Fuller (ADF) test** to identify stationary spreads.
- **Signal Generation:** Trades triggered when the spread moves **±1.5 standard deviations** from its rolling mean (10-day window).
- **Backtesting:** Evaluates performance over **KO & PEP (2018–2023)** with **Sharpe ratio, max drawdown, and cumulative returns**.
- **Risk Management:** Implements a **2% stop-loss**, improving strategy robustness.

## Performance Metrics

| Metric | Before Stop-Loss | After Stop-Loss |
|--------|-----------------|----------------|
| **Sharpe Ratio** | 0.87 | **1.44** |
| **Max Drawdown** | 17% | **5%** |