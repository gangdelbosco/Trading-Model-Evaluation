import numpy as np
import math
import pandas as pd

def moving_average(data, period):
    ma = []
    for i in range(len(data) - period + 1):
        sum = 0
        for j in range(i, i + period):
            sum += data[j]
        ma.append(sum / period)
    return ma
def macd(data, fast_period, slow_period):
    close = data['Close']
    fast_ma = moving_average(close, fast_period)
    slow_ma = moving_average(close, slow_period)
    macd = []
    for i in range(len(fast_ma)):
        macd.append(fast_ma[i] - slow_ma[i])
    return macd
def adx(data, period):
    high = data['High']
    low = data['Low']
    close = data['Close']
    tr = []
    for i in range(len(high)):
        tr.append(max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1])))
    atr = moving_average(tr, period)
    plus_di = []
    minus_di = []
    for i in range(len(high) - period + 1):
        up = 0
        down = 0
        for j in range(i, i + period):
            if high[j] - high[j-1] > low[j-1] - low[j]:
                up += high[j] - high[j-1]
            if low[j-1] - low[j] > high[j] - high[j-1]:
                down += low[j-1] - low[j]
        plus_di.append(up / atr[i])
        minus_di.append(down / atr[i])
    adx = []
    for i in range(len(plus_di)):
        adx.append(100 * (abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])))
    return adx
def cci(data, period):
    high = data['High']
    low = data['Low']
    close = data['Close']
    tp = []
    for i in range(len(high)):
        tp.append((high[i] + low[i] + close[i]) / 3)
    ma = moving_average(tp, period)
    md = []
    for i in range(len(tp)):
        md.append(abs(tp[i] - ma[i % period]))
    mean_deviation = moving_average(md, period)
    cci = []
    for i in range(len(tp) - period + 1):
        cci.append((tp[i] - ma[i]) / (0.015 * mean_deviation[i]))
    return cci
def z_score(data_, period):
    data =data_['Close']
    ma = moving_average(data, period)
    std_dev = []
    for i in range(len(data) - period + 1):
        sum = 0
        for j in range(i, i + period):
            sum += (data[j] - ma[i]) ** 2
        std_dev.append((sum / period) ** 0.5)
    z_score = []
    for i in range(len(data) - period + 1):
        z_score.append((data[i] - ma[i]) / std_dev[i])
    return z_score
def stochastic_oscillator(data, period):
    high = data['High']
    low = data['Low']
    close = data['Close']
    k = []
    for i in range(len(high) - period + 1):
        highest_high = max(high[i:i+period])
        lowest_low = min(low[i:i+period])
        k.append((close[i+period-1] - lowest_low) / (highest_high - lowest_low) * 100)
    return k
def bollinger_bands(data, period, std_devs):
    close = data['Close']
    ma = moving_average(close, period)
    std_dev = []
    for i in range(len(close) - period + 1):
        sum = 0
        for j in range(i, i + period):
            sum += (close[j] - ma[i]) ** 2
        std_dev.append((sum / period) ** 0.5)
    upper_band = []
    lower_band = []
    for i in range(len(ma)):
        upper_band.append(ma[i] + std_devs * std_dev[i])
        lower_band.append(ma[i] - std_devs * std_dev[i])
    return (ma, upper_band, lower_band)
def rsi(data, period):
    close = data['Close']
    change = close.diff()
    change.dropna(inplace=True)
# Create two copies of the Closing price Series
    change_up = change.copy()
    change_down = change.copy()

# 
    change_up[change_up<0] = 0
    change_down[change_down>0] = 0

# Verify that we did not make any mistakes
    change.equals(change_up+change_down)

# Calculate the rolling average of average up and average down
    avg_up = change_up.rolling(period).mean()
    avg_down = change_down.rolling(period).mean().abs()
    rsi = 100 * avg_up / (avg_up + avg_down)

# Take a look at the 20 oldest datapoints
    rsi.head(period)
    return rsi
def williams_r(data, period):
    high = data['High']
    low = data['Low']
    close = data['Close']
    williams_r = []
    for i in range(len(high) - period + 1):
        highest_high = max(high[i:i+period])
        lowest_low = min(low[i:i+period])
        williams_r.append((highest_high - close[i+period-1]) / (highest_high - lowest_low) * -100)
    return williams_r
def atr(data, period):
    high = data['High']
    low = data['Low']
    close = data['Close']
    tr = []
    for i in range(len(high)):
        tr.append(max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1])))
    return moving_average(tr, period)
def donchian_channel(data, period):
    high = data['High']
    low = data['Low']
    upper_channel = []
    lower_channel = []
    for i in range(len(high) - period + 1):
        upper_channel.append(max(high[i:i+period]))
        lower_channel.append(min(low[i:i+period]))
    return (upper_channel, lower_channel)
def tsi(data, long, short, signal):
    close = data['Close']
    
    diff = close - close.shift(1)
    abs_diff = abs(diff)
    
    diff_smoothed = diff.ewm(span = long, adjust = False).mean()
    diff_double_smoothed = diff_smoothed.ewm(span = short, adjust = False).mean()
    abs_diff_smoothed = abs_diff.ewm(span = long, adjust = False).mean()
    abs_diff_double_smoothed = abs_diff_smoothed.ewm(span = short, adjust = False).mean()
    
    tsi = (diff_double_smoothed / abs_diff_double_smoothed) * 100
    signal = tsi.ewm(span = signal, adjust = False).mean()
    tsi = tsi[tsi.index >= '2020-01-01'].dropna()
    signal = signal[signal.index >= '2020-01-01'].dropna()
    
    return tsi
def obv(data):
    close = data['Close']
    volume = data['Volume']
    obv = [volume[0]]
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            obv.append(obv[i-1] + volume[i])
        elif close[i] < close[i-1]:
            obv.append(obv[i-1] - volume[i])
        else:
            obv.append(obv[i-1])
    return obv
def cmf(data, period):
    high = data['High']
    low = data['Low']
    close = data['Close']
    volume = data['Volume']
    cmf = []
    for i in range(len(high) - period + 1):
        mf = 0
        for j in range(i, i + period):
            mf += ((close[j] - low[j]) - (high[j] - close[j])) / (high[j] - low[j]) * volume[j]
        cmf.append(mf / sum(volume[i:i+period]))
    return cmf
def adl(data):
    high = data['High']
    low = data['Low']
    close = data['Close']
    volume = data['Volume']
    adl = [volume[0]]
    for i in range(1, len(close)):
        mf = ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i]) * volume[i]
        adl.append(adl[i-1] + mf)
    return adl
def vwap(data):
    high = data['High']
    low = data['Low']
    close = data['Close']
    volume = data['Volume']
    vwap = [((high[0] + low[0] + close[0]) / 3) * volume[0]]
    for i in range(1, len(close)):
        vwap.append(vwap[i-1] + (((high[i] + low[i] + close[i]) / 3) * volume[i]))
    total_volume = sum(volume)
    for i in range(len(vwap)):
        vwap[i] = vwap[i] / total_volume
    return vwap
def mf_index(data, period):
    high = data['High']
    low = data['Low']
    close = data['Close']
    volume = data['Volume']
    typical_price = []
    for i in range(len(high)):
        typical_price.append((high[i] + low[i] + close[i]) / 3)
    positive_flow = []
    negative_flow = []
    for i in range(1, len(typical_price)):
        if typical_price[i] > typical_price[i-1]:
            positive_flow.append(typical_price[i] * volume[i])
            negative_flow.append(0)
        else:
            positive_flow.append(0)
            negative_flow.append(typical_price[i] * volume[i])
    mfi = []
    for i in range(len(positive_flow) - period + 1):
        pos_flow = sum(positive_flow[i:i+period])
        neg_flow = sum(negative_flow[i:i+period])
        mfi.append(100 - (100 / (1 + (pos_flow / neg_flow))))
    return mfi
def eom(data, ndays): 
    dm = ((data['High'] + data['Low'])/2) - ((data['High'].shift(1) + data['Low'].shift(1))/2)
    br = (data['Volume'] / 100000000) / ((data['High'] - data['Low']))
    EMV = dm / br 
    EMV_MA = pd.Series(EMV.rolling(ndays).mean(), name = 'EMV') 
    return EMV_MA
def vpt(data):
    close = data['Close']
    volume = data['Volume']
    vpt = [0]
    for i in range(1, len(close)):
        vpt.append(vpt[i-1] + (close[i] - close[i-1]) * volume[i])
    return vpt
def cvi(data, period):
    close = data['Close']
    volume = data['Volume']
    cvi = [50]
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            cvi.append(cvi[i-1] + (volume[i] / 1000000) * (close[i] - close[i-1]) / close[i-1])
        elif close[i] < close[i-1]:
            cvi.append(cvi[i-1] - (volume[i] / 1000000) * (close[i-1] - close[i]) / close[i-1])
        else:
            cvi.append(cvi[i-1])
    return moving_average(cvi, period)
def vwma(data, period):
    close = data['Close']
    volume = data['Volume']
    sum_vwma = 0
    sum_volume = 0
    vwma = []
    for i in range(len(close) - period + 1):
        for j in range(i, i+period):
            sum_vwma += close[j] * volume[j]
            sum_volume += volume[j]
        vwma.append(sum_vwma / sum_volume)
        sum_vwma = 0
        sum_volume = 0
    return vwma
def vwmacd(data, fast_period, slow_period, signal_period):
    vwma_fast = vwma(data, fast_period)
    vwma_slow = vwma(data, slow_period)
    macd = []
    for i in range(len(vwma_fast)):
        macd.append(vwma_fast[i] - vwma_slow[i])
    signal = moving_average(macd, signal_period)
    return (macd, signal)
def asi(data, period):
    high = data['High']
    low = data['Low']
    close = data['Close']
    open = data['Open']
    swing_index = []
    for i in range(2, len(high)):
        hc = high[i]-close[i-1]
        hl = high[i]-low[i]
        lc =low[i]-close[i-1]
        hyc = high[i-1]-close[i]
        lyc = low[i-1]-close[i]
        cyoy = (close[i-1]- open[i-1])
        if lyc>hyc:
            k = lyc
        else:
            k=hyc
        t=period
        if hc > hl :
            if hc > lc:
                r= hc-lc/2
        elif hl >hc :
            if hl >lc:
                r = hl +cyoy/4
        else:
            r =lc - hc/2
        
        swing = 50*((close[i-1]-close[i]+((close[i-1]-open[i-1])/2)+((close[i]-open[i])/4))/r)*(k/t)
        swing_index.append(swing)
    return swing_index
def vroc(data, period):
    volume = data['Volume']
    vroc = []
    for i in range(len(volume) - period):
        vroc.append((volume[i + period] - volume[i]) / volume[i])
    return vroc
def dmi_stochastic(data, period):
    high = data['High']
    low = data['Low']
    close = data['Close']
    dm_plus = []
    dm_minus = []
    for i in range(1, len(high)):
        dm_plus.append(max(0, high[i] - high[i-1]))
        dm_minus.append(max(0, low[i-1] - low[i]))
    dm_plus_ma = moving_average(dm_plus, period)
    dm_minus_ma = moving_average(dm_minus, period)
    stochastic_k = []
    for i in range(len(dm_plus_ma)):
        stochastic_k.append(100 * dm_plus_ma[i] / (dm_plus_ma[i] + dm_minus_ma[i]))
    return stochastic_k
def bop(data):
    high = data['High']
    low = data['Low']
    close = data['Close']
    bop = []
    for i in range(len(high)):
        bop.append(close[i] - (high[i] + low[i]) / 2)
    return bop
def awesome_oscillator(data, fast_period, slow_period):
    fast_ma = moving_average(data['Close'], fast_period)
    slow_ma = moving_average(data['Close'], slow_period)
    ao = []
    for i in range(len(fast_ma)):
        ao.append(fast_ma[i] - slow_ma[i])
    return ao
def fractal(data):
    high = data['High']
    low = data['Low']
    fractal_high = []
    fractal_low = []
    for i in range(1, len(high) - 1):
        if high[i] > high[i-1] and high[i] > high[i+1]:
            fractal_high.append(high[i])
        if low[i] < low[i-1] and low[i] < low[i+1]:
            fractal_low.append(low[i])
    return (fractal_high, fractal_low)
def mfi(data, period):
    high = data['High']
    low = data['Low']
    close = data['Close']
    volume = data['Volume']
    typical_price = []
    money_flow = []
    for i in range(len(high)):
        typical_price.append((high[i] + low[i] + close[i]) / 3)
        money_flow.append(typical_price[i] * volume[i])
    pos_money_flow = []
    neg_money_flow = []
    for i in range(1, len(typical_price)):
        if typical_price[i] > typical_price[i-1]:
            pos_money_flow.append(money_flow[i])
            neg_money_flow.append(0)
        else:
            pos_money_flow.append(0)
            neg_money_flow.append(money_flow[i])
    pos_money_flow_ma = moving_average(pos_money_flow, period)
    neg_money_flow_ma = moving_average(neg_money_flow, period)
    mfi = []
    for i in range(len(pos_money_flow_ma)):
        mfi.append(100 - (100 / (1 + pos_money_flow_ma[i] / neg_money_flow_ma[i])))
    return mfi
def ultima(data, period):
    high = data['High']
    low = data['Low']
    close = data['Close']
    typical_price = []
    ultima = []
    for i in range(len(high)):
        typical_price.append((high[i] + low[i] + close[i]) / 3)
        ultima.append(4 * typical_price[i] - 2 * high[i] - 2 * low[i])
    ultima_ma = moving_average(ultima, period)
    return ultima_ma
def ac(data, period):
    close = data['Close']
    five_ema = moving_average(close,5)
    five_ema_of_five_ema = moving_average(five_ema,5)
    ac = []
    for i in range(5, len(five_ema_of_five_ema)):
        ac.append(five_ema[i-5]-five_ema_of_five_ema[i])
    return moving_average(ac,period)
def di(data, period):
    high = data['High']
    low = data['Low']
    close = data['Close']
    di = []
    for i in range(len(high)):
        di.append(((close[i]-low[i])-(high[i]-close[i])) / (high[i]-low[i]))
    return moving_average(di,period)
def gap(data, period):
    high = data['High']
    low = data['Low']
    gap = []
    for i in range(len(high) - period + 1):
        gap.append(max(high[i:i+period]) - min(low[i:i+period]))
    return gap
def median_price(data):
    high = data['High']
    low = data['Low']
    median_price = []
    for i in range(len(high)):
        median_price.append((high[i] + low[i]) / 2)
    return median_price
def wma(df, period):
    n = period
    k = (n * (n + 1)) / 2.0
    wmas = []
    for i in range(0, len(df) - n + 1):
        product = [df['Close'][i + n_i] * (n_i + 1) for n_i in range(0, n)]
        wma = sum(product) / k
        wmas.append(wma)
    return wmas
def tma(data, period):
    ma = moving_average(data['Close'], period)
    return moving_average(ma, math.ceil(period/2))
def zlema(data, period):
    close = data['Close']
    alpha = 2 / (period + 1)
    ema = [close[0]]
    for i in range(1, len(close)):
        ema.append(alpha * close[i] + (1 - alpha) * ema[i-1])
    lag = (1 - alpha) ** period
    zlema = []
    for i in range(len(ema)):
        zlema.append((1 - lag) * ema[i] + lag * close[i])
    return zlema
def hma(data, period):
    wma1 = []
    close = data['Close']
    for i in range(len(close)):
        wma1.append(sum(close[i-period+1:i+1]) / period)
    wma2 = []
    for i in range(len(wma1)):
        wma2.append(sum(wma1[i-period+1:i+1]) / period)
    diff = []
    for i in range(len(wma1)):
        diff.append(2 * wma1[i] - wma2[i])
    return moving_average(diff, int(math.sqrt(period)))
def ama(data, period):
    er = []
    close = data['Close']
    for i in range(1, len(close)):
        er.append(abs(close[i] - close[i-1]))
    sc = 2 / (period + 1)
    ema = [close[0]]
    for i in range(1, len(close)):
        ema.append(sc * er[i-1] + (1 - sc) * ema[i-1])
    ama = [close[0]]
    for i in range(1, len(close)):
        ama.append(close[i] - ema[i] + ama[i-1])
    return ama
def t3(data, period, vFactor):
    close = data['Close']
    t3 = moving_average(close, period)
    for i in range(len(t3)):
        t3[i] = (vFactor * (close[i] - t3[i]) + t3[i])
    return t3
def dema(data, period):
    close = data['Close']
    ema1 = moving_average(close, period)
    ema2 = moving_average(ema1, period)
    dema = []
    for i in range(1, len(ema1)-period):
        dema.append(2 * ema1[i+period] - ema2[i])
    return dema
def tema(data, period):
    close = data['Close']
    ema1 = moving_average(close, period)
    ema2 = moving_average(ema1, period)
    ema3 = moving_average(ema2, period)
    tema = []
    for i in range(len(ema1)-2*period):
        tema.append(3 * ema1[i+2*period] - 3 * ema2[i+period] + ema3[i])
    return tema
def frama(data, period):
    close = data['Close']
    ema1 = moving_average(close, period)
    ema2 = moving_average(ema1, period)
    ema3 = moving_average(ema2, period)
    frama = []
    for i in range(len(ema1)-2*period):
        frama.append((ema1[i+2*period] + ema2[i]+period * 2 + ema3[i]) / 4)
    return frama
def vortex(data, period):
    high = data['High']
    low = data['Low']
    close = data['Close']
    tr = []
    for i in range(len(high)):
        tr.append(max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1])))
    tr_period = moving_average(tr, period)
    vm = []
    for i in range(len(high)):
        vm.append(abs(high[i] - low[i]))
    vm_period = moving_average(vm, period)
    vmi = []
    for i in range(len(vm) - period + 1):
        vmi.append(vm_period[i] / tr_period[i])
    vp = []
    for i in range(1, len(vmi)):
        vp.append(vmi[i] - vmi[i-1])
    vp_period = moving_average(vp, period)
    vi = []
    for i in range(len(vp) - period + 1):
        vi.append(vp_period[i] / vmi[i])
    return vi
def trix(data, period):
    close = data['Close']
    ema1 = moving_average(close, period)
    ema2 = moving_average(ema1, period)
    ema3 = moving_average(ema2, period)
    trix = []
    for i in range(len(ema3)):
        trix.append((ema3[i] - ema3[i-1]) / ema3[i-1] * 100)
    return trix
def stochastic_rsi(data, period):
    close = data['Close']
    rsi_ = rsi(data, period)
    k = []
    for i in range(len(rsi_) - period + 1):
        highest_rsi = max(rsi_[i:i+period])
        lowest_rsi = min(rsi_[i:i+period])
        k.append((rsi_[i+period-1] - lowest_rsi) / (highest_rsi - lowest_rsi) * 100)
    return k
def roc(data, period):
    roc = []
    close = data['Close']
    for i in range(len(data['Close']) - period):
        roc.append((close[i+period] - close[i]) / close[i] * 100)
    return roc
def momentum(data, period):
    close = data['Close']
    momentum = []
    for i in range(len(close) - period):
        momentum.append(close[i+period] - close[i])
    return momentum
def envelope(data, period, upper_multiplier, lower_multiplier):
    close = data['Close']
    ma = moving_average(close, period)
    upper = []
    lower = []
    for i in range(len(ma)):
        upper.append(ma[i] * (1 + upper_multiplier))
        lower.append(ma[i] * (1 - lower_multiplier))
    return (upper, lower)

def ind_series(df, len , i):

    name_mapping = {
        1: 'adx',
        2: 'cci',
        3: 'z_score',
        4: 'stochastic_oscillator',
        5: 'rsi',
        6: 'williams_r',
        7: 'atr',
        8: 'cmf',
        9: 'mf_index',
        10: 'eom',
        11: 'cvi',
        12: 'vwma',
        13: 'asi',
        14: 'vroc',
        15: 'dmi_stochastic',
        16: 'mfi',
        17: 'ultima',
        18: 'ac',
        19: 'di',
        20: 'gap',
        21: 'wma',
        22: 'tma',
        23: 'zlema',
        24: 'hma',
        25: 'ama',
        26: 'dema',
        27: 'tema',
        28: 'frama',
        29: 'vortex',
        30: 'trix',
        31: 'stochastic_rsi',
        32: 'roc',
        33: 'momentum'

    }
    array_mapping = {
        1: adx(df, len),
        2: cci(df, len),
        3: z_score(df, len),
        4: stochastic_oscillator(df, len),
        5: rsi(df, len),
        6: williams_r(df, len),
        7: atr(df, len),
        8: cmf(df, len),
        9: mf_index(df, len),
        10: eom(df, len),
        11: cvi(df, len),
        12: vwma(df, len),
        13: asi(df, len),
        14: vroc(df, len),
        15: dmi_stochastic(df, len),
        16: mfi(df, len),
        17: ultima(df, len),
        18: ac(df, len),
        19: di(df, len),
        20: gap(df, len),
        21: wma(df, len),
        22: tma(df, len),
        23: zlema(df, len),
        24: hma(df, len),
        25: ama(df, len),
        26: dema(df, len),
        27: tema(df, len),
        28: frama(df, len),
        29: vortex(df, len),
        30: trix(df, len),
        31: stochastic_rsi(df, len),
        32: roc(df, len),
        33: momentum(df, len)
    }
    return name_mapping[i], array_mapping[i]

