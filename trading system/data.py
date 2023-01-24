import yfinance as yf
import pandas as pd
import numpy as np
start = '2022-5-1'
end ='2022-7-1'
AAPL = yf.Ticker('AAPL').history(period = '5y', interval = '1d')
IXIC = yf.Ticker('^IXIC').history(period = '5y', interval = '1d')
EURUSD = yf.Ticker('EURUSD=X').history(period = '5y', interval = '1d')
AMZN = yf.Ticker('AMZN').history(period = '5y', interval = '1d')
SYY = yf.Ticker('SYY').history(period = '5y', interval = '1d')
TSLA = yf.Ticker('TSLA').history(period = '5y', interval = '1d')
GOOG = yf.Ticker('GOOG').history(period = '5y', interval = '1d')
META = yf.Ticker('META').history(period = '5y', interval = '1d')
MSFT = yf.Ticker('MSFT').history(period = '5y', interval = '1d')
GSPC = yf.Ticker('^GSPC').history(period = '5y', interval = '1d')

def data_adj (data, ind, name_, loc_):
    llen = len(data)
    llen_ = len(ind)
    diff=llen- llen_
    data = data.assign(indicator = 0)
    data.rename(columns={'indicator': name_}, inplace = True)
    data.iloc[diff:,data.columns.get_loc(name_)]= ind
    return data

