import yfinance as yf
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
import matplotlib.pyplot as plt

# 1. Data Download and Preparation
tickers = ['KO', 'PEP']
# Download daily data from 2018-01-01 to 2023-01-01
stock = yf.download(tickers, start='2018-01-01', end='2023-01-01')
# Select the Close prices
data = stock['Close']

data = data.copy()

# 2. Cointegration Testing
# Run OLS regression: KO ~ const + PEP
model = sm.OLS(data['KO'], sm.add_constant(data['PEP'])).fit()
data.loc[:, 'residuals'] = model.resid

# Perform the Augmented Dickey-Fuller (ADF) test on the residuals
adf_result = ts.adfuller(data['residuals'])
print("ADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])
# A p-value below 0.05 (e.g., 0.01) indicates stationarity and supports cointegration


# 3. Signal Generation
# Parameters for rolling window and trade execution thresholds
window = 10
base_threshold = 1.5

# Calculate rolling statistics of the residuals
data.loc[:, 'spread_mean'] = data['residuals'].rolling(window=window).mean()
data.loc[:, 'spread_std'] = data['residuals'].rolling(window=window).std()

data.loc[:, 'signal'] = 0

# Generate signals based on the spread deviating from its rolling mean:
# If the residual is below (spread_mean - threshold * spread_std) --> expect reversion upward --> long signal
data.loc[data['residuals'] < data['spread_mean'] - base_threshold * data['spread_std'], 'signal'] = 1
# If the residual is above (spread_mean + threshold * spread_std) --> expect reversion downward --> short signal
data.loc[data['residuals'] > data['spread_mean'] + base_threshold * data['spread_std'], 'signal'] = -1

# Shift the signal by one day to simulate entering a trade the next day
data.loc[:, 'signal_shifted'] = data['signal'].shift(1)

# 4. Backtesting the Base Strategy
# Calculate daily returns for each stock
data.loc[:, 'return_KO'] = data['KO'].pct_change()
data.loc[:, 'return_PEP'] = data['PEP'].pct_change()

# Compute the strategy return: assuming a dollar-neutral trade (long one and short the other)
data.loc[:, 'strategy_return'] = data['signal_shifted'] * (data['return_KO'] - data['return_PEP'])
# Compute the cumulative return over time
data.loc[:, 'cumulative_return'] = (1 + data['strategy_return'].fillna(0)).cumprod()

# Calculate performance metrics for the base strategy
avg_daily_return = data['strategy_return'].mean()
std_daily_return = data['strategy_return'].std()
sharpe_ratio = (avg_daily_return / std_daily_return) * np.sqrt(252) if std_daily_return != 0 else np.nan

running_max = data['cumulative_return'].cummax()
drawdown = (running_max - data['cumulative_return']) / running_max
max_drawdown = drawdown.max()

print("Base Strategy Sharpe Ratio:", sharpe_ratio)
print("Base Strategy Maximum Drawdown:", max_drawdown)

# 5. Incorporating a Stop-Loss Mechanism
# Define a stop-loss threshold (e.g., if the trade loses more than 2% from its entry, exit the trade)
stop_loss_threshold = -0.02

# We'll add a new column for adjusted strategy returns that applies the stop-loss rule
data.loc[:, 'adjusted_strategy_return'] = data['strategy_return']

# Initialize variables to track trade status
in_trade = False  # Are we currently in a trade?
cumulative_trade_return = 1.0  # Reset cumulative return for a trade

# Loop through the DataFrame row by row
for i in range(len(data)):
    # If not in a trade and a trade signal is generated, mark trade entry
    if not in_trade and data['signal_shifted'].iloc[i] != 0:
        in_trade = True
        cumulative_trade_return = 1.0  # Reset for new trade

    if in_trade:
        # Update the cumulative return for the current trade
        cumulative_trade_return *= (1 + data['strategy_return'].iloc[i])
        
        # If the cumulative return falls below the stop-loss threshold, exit the trade
        if cumulative_trade_return - 1 < stop_loss_threshold:
            data.loc[data.index[i], 'adjusted_strategy_return'] = 0  # Simulate exit on this day
            in_trade = False  # End the trade

    # Optionally, if the signal goes to 0, consider that as an exit (if desired)
    if in_trade and data['signal_shifted'].iloc[i] == 0:
        in_trade = False

# Calculate adjusted cumulative returns based on the stop-loss
data.loc[:, 'adjusted_cumulative_return'] = (1 + data['adjusted_strategy_return'].fillna(0)).cumprod()

# Compute performance metrics for the adjusted strategy
avg_daily_return_adj = data['adjusted_strategy_return'].mean()
std_daily_return_adj = data['adjusted_strategy_return'].std()
sharpe_ratio_adj = (avg_daily_return_adj / std_daily_return_adj) * np.sqrt(252) if std_daily_return_adj != 0 else np.nan

running_max_adj = data['adjusted_cumulative_return'].cummax()
drawdown_adj = (running_max_adj - data['adjusted_cumulative_return']) / running_max_adj
max_drawdown_adj = drawdown_adj.max()

print("Adjusted Strategy Sharpe Ratio:", sharpe_ratio_adj)
print("Adjusted Strategy Maximum Drawdown:", max_drawdown_adj)