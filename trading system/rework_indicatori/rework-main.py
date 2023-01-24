from __future__ import (absolute_import, division, print_function, unicode_literals)
import indicator as ind
import backtrader as bt
from math import pi
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import os.path
import sys
frame = yf.Ticker('^IXIC').history(period="1y", interval = '1h')

class TestStrategy(bt.Strategy):
    
    def log(self, txt, dt=None): #logging functions for the strategy
        dt = dt or self.datas[0].datetime.date(0)
#        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        self.dataclose = self.datas[0].close #keep reference

    def next(self):
        # Simply log the closing price of the series from the reference
        self.log('Close, %.2f' % self.dataclose[0])

        if self.dataclose[0] < self.dataclose[-1]:
            # current close less than previous close

            if self.dataclose[-1] < self.dataclose[-2]:
                # previous close less than the previous close

                # BUY, BUY, BUY!!! (with all possible default parameters)
                self.log('BUY CREATE, %.2f' % self.dataclose[0])
                self.buy()

if __name__ =='__main__':

    cerebro = bt.Cerebro()#crea cerebro
    
    cerebro.addstrategy(TestStrategy)


    data = bt.feeds.PandasData(dataname = frame)

    # Add the Data Feed to Cerebro
    cerebro.adddata(data)

    # Set our desired cash start
    cerebro.broker.setcash(100000.0)

    # Print out the starting conditions
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    cerebro.run()

    # Print out the final result
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())






new_data1 = ind.crv(frame, ind.adx(frame, 14), ind.adx.__name__)

new_data2 = ind.crv(frame, ind.cci(frame, 14), ind.cci.__name__)

new_data3 = ind.crv(frame, ind.z_score(frame, 14), ind.z_score.__name__)

new_data4 = ind.crv(frame, ind.stochastic_oscillator(frame, 14), ind.stochastic_oscillator.__name__)

new_data5 = ind.crv(frame, ind.rsi(frame, 14), ind.rsi.__name__)

new_data6 = ind.crv(frame, ind.williams_r(frame, 14), ind.williams_r.__name__)

new_data7 = ind.crv(frame, ind.atr(frame, 14), ind.atr.__name__)

new_data8 = ind.crv(frame, ind.cmf(frame, 14), ind.cmf.__name__)

new_data9 = ind.crv(frame, ind.mf_index(frame, 14), ind.mf_index.__name__)

new_data10 = ind.crv(frame, ind.eom(frame, 14), ind.eom.__name__)

new_data11 = ind.crv(frame, ind.cvi(frame, 14), ind.cvi.__name__)

new_data12 = ind.crv(frame, ind.vwma(frame, 14), ind.vwma.__name__)

new_data13 = ind.crv(frame, ind.asi(frame, 14), ind.asi.__name__)

new_data14 = ind.crv(frame, ind.vroc(frame, 14), ind.vroc.__name__)

new_data15 = ind.crv(frame, ind.dmi_stochastic(frame, 14), ind.dmi_stochastic.__name__)

new_data16 = ind.crv(frame, ind.mfi(frame, 14), ind.mfi.__name__)

new_data17 = ind.crv(frame, ind.ultima(frame, 14), ind.ultima.__name__)

new_data18 = ind.crv(frame, ind.ac(frame, 14), ind.ac.__name__)

new_data19 = ind.crv(frame, ind.di(frame, 14), ind.di.__name__)

new_data20 = ind.crv(frame, ind.gap(frame, 14), ind.gap.__name__)

new_data21 = ind.crv(frame, ind.wma(frame, 14), ind.wma.__name__)

new_data22 = ind.crv(frame, ind.tma(frame, 14), ind.tma.__name__)

new_data23 = ind.crv(frame, ind.zlema(frame, 14), ind.zlema.__name__)

new_data24 = ind.crv(frame, ind.hma(frame, 14), ind.hma.__name__)

new_data25 = ind.crv(frame, ind.ama(frame, 14), ind.ama.__name__)

new_data26 = ind.crv(frame, ind.dema(frame, 14), ind.dema.__name__)

new_data27 = ind.crv(frame, ind.tema(frame, 14), ind.tema.__name__)

new_data28 = ind.crv(frame, ind.frama(frame, 14), ind.frama.__name__)

new_data29 = ind.crv(frame, ind.vortex(frame, 14), ind.vortex.__name__)

new_data30 = ind.crv(frame, ind.trix(frame, 14), ind.trix.__name__)

new_data31 = ind.crv(frame, ind.stochastic_rsi(frame, 14), ind.stochastic_rsi.__name__)

new_data32 = ind.crv(frame, ind.roc(frame, 14), ind.roc.__name__)

new_data33 = ind.crv(frame, ind.momentum(frame, 14), ind.momentum.__name__)

print()