import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

# DATA COLLECTING

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
prices = {}
info = {}
for ticker in tickers:
    try:
        data = yf.Ticker(ticker)
        eps = data.info.get('epsTrailingTwelveMonths')
        ev_ebitda = data.info.get('enterpriseToEbitda')
        ev_revenue = data.info.get('enterpriseToRevenue')
        roa = data.info.get('returnOnAssets')
        firm_data = {
            'eps': eps,
            'ev_ebitda': ev_ebitda,
            'ev_revenue': ev_revenue,
            'roa': roa
        }
        info[ticker] = firm_data
        close_prices = data.history(period='5y')['Close']
        prices[ticker] = close_prices
    except Exception as e:
        print(f'{ticker} failed.')
        continue
df_info = pd.DataFrame(info).T
df_info.to_parquet('df_info.parquet')
df_prices = pd.DataFrame(prices)
df_prices.to_parquet('df_prices.parquet')


# FUNDAMENTALIST ANALYSIS

df_info = pd.read_parquet('df_info.parquet')

df_info['roa_rank'] = df_info['roa'].rank(ascending=False)
df_info['value_rank'] = df_info['ev_ebitda'].rank()
df_info['final_rank'] = df_info['roa_rank'] + df_info['value_rank']
df_info = df_info.sort_values(by=['final_rank'])


# QUANTITATIVE ANALYSIS

df_prices = pd.read_parquet('df_prices.parquet')
df_returns = df_prices.pct_change().dropna()

annual_std = df_returns.std() * np.sqrt(252)  # standard deviation per year
corr_matrix = df_returns.corr()  # matrix of correlation between companies


# VISUAL ANALYSIS

# Return

df_cumulative = (1 + df_returns).cumprod()
df_cumulative.plot(figsize=(10, 6), linewidth=0.7)

plt.title('Cumulative Return (5Y)')
plt.ylabel('Wealth Index (Base $1)')
plt.grid(alpha=0.3)
plt.show()

# Risk

plt.figure()
annual_std.plot(kind='bar', title='Annual Standard Deviation (5Y)', grid=True)
plt.grid(alpha=0.3)
plt.show()


# MONTE CARLO SIMULATION

num_portfolios = 10000
port_returns = []
port_volatility = []
port_weights = []

mean_return_yr = df_returns.mean() * 252
cov_matrix_yr = df_returns.cov() * 252

for i in range(num_portfolios):
    weights = np.random.random(len(df_returns.columns))  # generates random weights for the companies
    weights /= np.sum(weights)  # divides each number by the sum for total to be always 1
    p_return = np.sum(mean_return_yr * weights)  # annual return of portfolio
    p_volat = np.sqrt(np.dot(weights.T, np.dot(cov_matrix_yr, weights)))  # annual volatility of portfolio

    port_returns.append(p_return)
    port_volatility.append(p_volat)
    port_weights.append(weights)

df_portfolios = pd.DataFrame(port_weights, columns=df_returns.columns)
df_portfolios['Expected_Return'] = port_returns
df_portfolios['Expected_Volatility'] = port_volatility
df_portfolios.to_parquet('df_monte_carlo.parquet')

df_monte_carlo = pd.read_parquet('df_monte_carlo.parquet')

rf_rate = 0.04
sharpe = (df_monte_carlo['Expected_Return'] - rf_rate) / df_monte_carlo['Expected_Volatility']
df_monte_carlo['Sharpe_Ratio'] = sharpe


# FINDING BEST PORTFOLIOS

# By Sharpe Ratio

index_sharpe = df_monte_carlo['Sharpe_Ratio'].idxmax()
port_sharpe = df_monte_carlo.loc[index_sharpe]

print('\n' + '='*30)
print('   MAXIMUM SHARPE PORTFOLIO     ')
print('='*30)
print(f"AAPL:                 {port_sharpe['AAPL']*100:>6.2f}%")
print(f"MSFT:                 {port_sharpe['MSFT']*100:>6.2f}%")
print(f"GOOGL:                {port_sharpe['GOOGL']*100:>6.2f}%")
print(f"AMZN:                 {port_sharpe['AMZN']*100:>6.2f}%")
print('-'*30)
print(f"Expected Return:      {port_sharpe['Expected_Return']*100:>6.2f}%")
print(f"Annual Volatility:    {port_sharpe['Expected_Volatility']*100:>6.2f}%")
print(f"Sharpe Ratio:         {port_sharpe['Sharpe_Ratio']:>7.2f}")
print('='*30)

sharpe_xcor = port_sharpe['Expected_Volatility']
sharpe_ycor = port_sharpe['Expected_Return']

print('\n')

# By Minimum Variance

index_variance = df_monte_carlo['Expected_Volatility'].idxmin()
port_variance = df_monte_carlo.loc[index_variance]

print('\n' + '='*30)
print('  MINIMUM VARIANCE PORTFOLIO   ')
print('='*30)
print(f"AAPL:                 {port_variance['AAPL']*100:>6.2f}%")
print(f"MSFT:                 {port_variance['MSFT']*100:>6.2f}%")
print(f"GOOGL:                {port_variance['GOOGL']*100:>6.2f}%")
print(f"AMZN:                 {port_variance['AMZN']*100:>6.2f}%")
print('-'*30)
print(f"Expected Return:      {port_variance['Expected_Return']*100:>6.2f}%")
print(f"Annual Volatility:    {port_variance['Expected_Volatility']*100:>6.2f}%")
print(f"Sharpe Ratio:         {port_variance['Sharpe_Ratio']:>7.2f}")
print('='*30)

variance_xcor = port_variance['Expected_Volatility']
variance_ycor = port_variance['Expected_Return']


# PLOTTING GRAPH

plt.figure(figsize=(10, 6))
plt.title('Efficient Frontier')
plt.ylabel('Expected Return')
plt.xlabel('Expected Volatility')
plt.scatter(
    x=df_monte_carlo['Expected_Volatility'],
    y=df_monte_carlo['Expected_Return'],
    c=df_monte_carlo['Sharpe_Ratio'],
    cmap='viridis',
    s=2,
    alpha=0.6
)
plt.colorbar(label='Sharpe Ratio')
plt.scatter(
    x=sharpe_xcor,
    y=sharpe_ycor,
    color='red',
    marker='*',
    s=50,
    label='Max Sharpe Portfolio'
)
plt.scatter(
    x=variance_xcor,
    y=variance_ycor,
    color='blue',
    marker='*',
    s=50,
    label='Min Variance Portfolio'
)
plt.legend()
plt.show()
