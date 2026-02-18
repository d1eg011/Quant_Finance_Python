import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, chi2
import pandas as pd

#MXN=X ^IXIC DBK.DE ^VIX ^S&P500
    

def load_time_series(ric):
    #get market data
    
    table_raw = pd.read_csv('data/' + ric + '.csv')
    
    #create table of returns
    t = pd.DataFrame()
    t['date'] = pd.to_datetime(table_raw['Date'], dayfirst = True)
    t['close'] = table_raw['Close']
    t['close_previous'] = table_raw['Close'].shift(1)
    t['return_close'] = t['close'] / t['close_previous'] - 1
    t.sort_values(by='date', ascending = True)
    t = t.dropna()
    t = t.reset_index(drop=True)

    #input 
    ric = '^VIX' #MXN=X ^IXIC DBK.DE ^VIX ^S&P500
    
    #get market data
    
    table_raw = pd.read_csv('data/' + ric + '.csv')
    
    #create table of returns
    t = pd.DataFrame()
    t['date'] = pd.to_datetime(table_raw['Date'], dayfirst = True)
    t['close'] = table_raw['Close']
    t['close_previous'] = table_raw['Close'].shift(1)
    t['return_close'] = t['close'] / t['close_previous'] - 1
    t.sort_values(by='date', ascending = True)
    t = t.dropna()
    t = t.reset_index(drop=True)

    #input for Jarque-Bera test
    x = t['return_close'].values
    
    return t, x

def plot_time_series_price(t, ric):
    #plot time series
    plt.figure()
    plt.plot(t['date'], t['close'])
    plt.title('Time series real prices ' + ric)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.show()

def plot_histogram(x, ric, n_bin = 100):

    x_mean = np.mean(x)
    x_stdev = np.std(x) 
    x_skew = skew(x)
    x_kurt = kurtosis(x) #excess kurtosis is kurtosis - 3, so 0 for a standard normal distribution
    x_sharpe = x_mean / x_stdev * np.sqrt(252) # expected return per risk unit, annualized
    x_var_95 = np.percentile(x, 5)
    x_cvar_95 = np.mean(x[x <= x_var_95])
    jb = len(x)/6 * (x_skew**2 + 1/4*x_kurt**2)
    p_value = 1 - chi2.cdf(jb, df = 2)
    is_normal = (p_value > 0.05) # equivalently jb < 6

    round_digits = 2
    
    str1 = 'Mean: ' + str(np.round(x_mean, round_digits)) +\
    ' | Std Dev: ' + str(np.round(x_stdev, round_digits)) +\
    ' | Skewness: ' + str(np.round(x_skew, round_digits)) +\
    ' | Kurtosis: ' + str(np.round(x_kurt, round_digits)) +\
    ' | Sharpe: ' + str(np.round(x_sharpe, round_digits))

    str2 = 'VaR 95%: ' + str(np.round(x_var_95, round_digits)) +\
    ' | CVaR 95%: ' + str(np.round(x_cvar_95, round_digits)) +\
    ' | Jarque-Bera: ' + str(np.round(jb, round_digits)) +\
    ' | p_value: ' + str(np.round(p_value, round_digits)) +\
    ' | is_normal: ' + str(is_normal)

    x_str = 'Real returns ' + ric
    plt.figure()
    plt.hist(x, bins = n_bin, edgecolor = 'black', histtype='step', density = True)
    plt.title('Histogram ' + x_str)
    plt.xlabel(str1 + '\n' + str2)
    plt.show()