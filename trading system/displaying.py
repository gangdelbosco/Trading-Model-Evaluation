from bokeh.plotting import figure
from bokeh.io import show
from bokeh.resources import INLINE
from bokeh.layouts import column
import yfinance as yf
new_data = yf.Ticker('^IXIC').history(period="6mo", interval = '1h')
inc = new_data.Close > new_data.Open
dec = new_data.Open > new_data.Close

w = 6*30*30*250

## Candlestick chart
candlestick = figure(x_axis_type="datetime", width=1400, height= 300, x_range=(new_data.index.min(), new_data.index.max()))
candlestick.line(new_data.index, new_data.Close, width=1, line_color="lime", alpha=0.8)
## Volume Chart
indicator = figure(x_axis_type="datetime", width=1400, height=  250, x_range=(new_data.index.min(), new_data.index.max()))
indicator.line(new_data.index, new_data.High, width=1, line_color="fuchsia", alpha=0.8)
#0000
close = figure(x_axis_type="datetime", width=1400, height= 300, x_range=(new_data.index.min(), new_data.index.max()))
close.line(new_data.index, new_data.Close, width=1, line_color="fuchsia", alpha=0.8)

show(column(candlestick, indicator, close))