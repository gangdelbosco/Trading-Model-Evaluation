import indicator as ind
import pattern_finder as pf
from pattern_finder import crv
from math import pi
import yfinance as yf
import pandas as pd
import numpy as np
from bokeh.plotting import figure
from bokeh.io import show
from bokeh.resources import INLINE
from bokeh.layouts import column

start = '2021-1-20'
end ='2023-1-20'
start_test = '2022-1-24'
end_test ='2023-1-24'

frame = yf.Ticker('^IXIC').history(period="2y", interval = '1h')
test = yf.Ticker('^IXIC').history(start = start_test, end = end_test, interval = '1h')
base_test = yf.Ticker('^IXIC').history(start = start_test, end = end_test, interval = '1h')

c_index = []
f_index = []

new_data1 = crv(frame, ind.adx(frame, 14), ind.adx.__name__)
adx_data =pf.scan_data_v4(new_data1, ind.adx.__name__)
new_data2 = crv(frame, ind.cci(frame, 14), ind.cci.__name__)
cci_data =pf.scan_data_v4(new_data2, ind.cci.__name__)
new_data3 = crv(frame, ind.z_score(frame, 14), ind.z_score.__name__)
z_score_data =pf.scan_data_v4(new_data3, ind.z_score.__name__)
new_data4 = crv(frame, ind.stochastic_oscillator(frame, 14), ind.stochastic_oscillator.__name__)
stochastic_oscillator_data =pf.scan_data_v4(new_data4, ind.stochastic_oscillator.__name__)
new_data5 = crv(frame, ind.rsi(frame, 14), ind.rsi.__name__)
rsi_data =pf.scan_data_v4(new_data5, ind.rsi.__name__)
new_data6 = crv(frame, ind.williams_r(frame, 14), ind.williams_r.__name__)
williams_r_data =pf.scan_data_v4(new_data6, ind.williams_r.__name__)
new_data7 = crv(frame, ind.atr(frame, 14), ind.atr.__name__)
atr_data =pf.scan_data_v4(new_data7, ind.atr.__name__)
new_data8 = crv(frame, ind.cmf(frame, 14), ind.cmf.__name__)
cmf_data =pf.scan_data_v4(new_data8, ind.cmf.__name__)
new_data9 = crv(frame, ind.mf_index(frame, 14), ind.mf_index.__name__)
mf_index_data =pf.scan_data_v4(new_data9, ind.mf_index.__name__)
new_data10 = crv(frame, ind.eom(frame, 14), ind.eom.__name__)
#eom_data =pf.scan_data_v4(new_data10, ind.eom.__name__)
new_data11 = crv(frame, ind.cvi(frame, 14), ind.cvi.__name__)
cvi_data =pf.scan_data_v4(new_data11, ind.cvi.__name__)
new_data12 = crv(frame, ind.vwma(frame, 14), ind.vwma.__name__)
vwma_data =pf.scan_data_v4(new_data12, ind.vwma.__name__)
new_data13 = crv(frame, ind.asi(frame, 14), ind.asi.__name__)
asi_data =pf.scan_data_v4(new_data13, ind.asi.__name__)
new_data14 = crv(frame, ind.vroc(frame, 14), ind.vroc.__name__)
vroc_data =pf.scan_data_v4(new_data14, ind.vroc.__name__)
new_data15 = crv(frame, ind.dmi_stochastic(frame, 14), ind.dmi_stochastic.__name__)
dmi_stochastic_data =pf.scan_data_v4(new_data15, ind.dmi_stochastic.__name__)
new_data16 = crv(frame, ind.mfi(frame, 14), ind.mfi.__name__)
mfi_data =pf.scan_data_v4(new_data16, ind.mfi.__name__)
new_data17 = crv(frame, ind.ultima(frame, 14), ind.ultima.__name__)
ultima_data =pf.scan_data_v4(new_data17, ind.ultima.__name__)
new_data18 = crv(frame, ind.ac(frame, 14), ind.ac.__name__)
ac_data =pf.scan_data_v4(new_data18, ind.ac.__name__)
new_data19 = crv(frame, ind.di(frame, 14), ind.di.__name__)
di_data =pf.scan_data_v4(new_data19, ind.di.__name__)
new_data20 = crv(frame, ind.gap(frame, 14), ind.gap.__name__)
gap_data =pf.scan_data_v4(new_data20, ind.gap.__name__)
new_data21 = crv(frame, ind.wma(frame, 14), ind.wma.__name__)
wma_data =pf.scan_data_v4(new_data21, ind.wma.__name__)
new_data22 = crv(frame, ind.tma(frame, 14), ind.tma.__name__)
tma_data =pf.scan_data_v4(new_data22, ind.tma.__name__)
new_data23 = crv(frame, ind.zlema(frame, 14), ind.zlema.__name__)
zlema_data =pf.scan_data_v4(new_data23, ind.zlema.__name__)
new_data24 = crv(frame, ind.hma(frame, 14), ind.hma.__name__)
hma_data =pf.scan_data_v4(new_data24, ind.hma.__name__)
new_data25 = crv(frame, ind.ama(frame, 14), ind.ama.__name__)
ama_data =pf.scan_data_v4(new_data25, ind.ama.__name__)
new_data26 = crv(frame, ind.dema(frame, 14), ind.dema.__name__)
dema_data =pf.scan_data_v4(new_data26, ind.dema.__name__)
new_data27 = crv(frame, ind.tema(frame, 14), ind.tema.__name__)
tema_data =pf.scan_data_v4(new_data27, ind.tema.__name__)
new_data28 = crv(frame, ind.frama(frame, 14), ind.frama.__name__)
frama_data =pf.scan_data_v4(new_data28, ind.frama.__name__)
new_data29 = crv(frame, ind.vortex(frame, 14), ind.vortex.__name__)
vortex_data =pf.scan_data_v4(new_data29, ind.vortex.__name__)
new_data30 = crv(frame, ind.trix(frame, 14), ind.trix.__name__)
trix_data =pf.scan_data_v4(new_data30, ind.trix.__name__)
new_data31 = crv(frame, ind.stochastic_rsi(frame, 14), ind.stochastic_rsi.__name__)
stochastic_rsi_data =pf.scan_data_v4(new_data31, ind.stochastic_rsi.__name__)
new_data32 = crv(frame, ind.roc(frame, 14), ind.roc.__name__)
roc_data =pf.scan_data_v4(new_data32, ind.roc.__name__)
new_data33 = crv(frame, ind.momentum(frame, 14), ind.momentum.__name__)
momentum_data =pf.scan_data_v4(new_data33, ind.momentum.__name__)

momentum_increase, momentum_decrease = pf.patterns(momentum_data, 9999999)
momentum_increasing_val_ = momentum_increase[2]-momentum_increase[0]
momentum_decreasing_val_ = momentum_decrease[2]-momentum_decrease[0]
a,b,c,d,e,f = pf.inc_prtg(new_data33, ind.momentum.__name__, momentum_increase, 20, round(momentum_increasing_val_, 4))

c_index.append(c)
f_index.append(f)

roc_increase, roc_decrease = pf.patterns(roc_data, 9999999)
roc_increasing_val_ = roc_increase[2]-roc_increase[0]
roc_decreasing_val_ = roc_decrease[2]-roc_decrease[0]
a,b,c,d,e,f = pf.inc_prtg(new_data32, ind.roc.__name__, roc_increase, 20,round(roc_increasing_val_, 4))

c_index.append(c)
f_index.append(f)


stochastic_rsi_increase, stochastic_rsi_decrease = pf.patterns(stochastic_rsi_data, 9999999)
stochastic_rsi_increasing_val_ = stochastic_rsi_increase[2]-stochastic_rsi_increase[0]
stochastic_rsi_decreasing_val_ = stochastic_rsi_decrease[2]-stochastic_rsi_decrease[0]
a,b,c,d,e,f = pf.inc_prtg(new_data31, ind.stochastic_rsi.__name__, stochastic_rsi_increase, 20, round(stochastic_rsi_increasing_val_, 4))

c_index.append(c)
f_index.append(f)

trix_increase, trix_decrease = pf.patterns(trix_data, 9999999)
trix_increasing_val_ = trix_increase[2]-trix_increase[0]
trix_decreasing_val_ = trix_decrease[2]-trix_decrease[0]
a,b,c,d,e,f = pf.inc_prtg(new_data30, ind.trix.__name__, trix_increase, 20, round(trix_increasing_val_, 4))

c_index.append(c)
f_index.append(f)


vortex_increase, vortex_decrease = pf.patterns(vortex_data, 9999999)
vortex_increasing_val_ = vortex_increase[2]-vortex_increase[0]
vortex_decreasing_val_ = vortex_decrease[2]-vortex_decrease[0]
a,b,c,d,e,f = pf.inc_prtg(new_data29, ind.vortex.__name__, vortex_increase, 20, round(vortex_increasing_val_, 4))

c_index.append(c)
f_index.append(f)

frama_increase, frama_decrease = pf.patterns(frama_data, 9999999)
frama_increasing_val_ = frama_increase[2]-frama_increase[0]
frama_decreasing_val_ = frama_decrease[2]-frama_decrease[0]
a,b,c,d,e,f = pf.inc_prtg(new_data28, ind.frama.__name__, frama_increase, 20, round(frama_increasing_val_, 4))

c_index.append(c)
f_index.append(f)


tema_increase, tema_decrease = pf.patterns(tema_data, 9999999)
tema_increasing_val_ = tema_increase[2]-tema_increase[0]
tema_decreasing_val_ = tema_decrease[2]-tema_decrease[0]
a,b,c,d,e,f = pf.inc_prtg(new_data27, ind.tema.__name__, tema_increase, 20, round(tema_increasing_val_, 4))

c_index.append(c)
f_index.append(f)


dema_increase, dema_decrease = pf.patterns(dema_data, 9999999)
dema_increasing_val_ = dema_increase[2]-dema_increase[0]
dema_decreasing_val_ = dema_decrease[2]-dema_decrease[0]
a,b,c,d,e,f = pf.inc_prtg(new_data26, ind.dema.__name__, dema_increase, 20, round(dema_increasing_val_, 4))

c_index.append(c)
f_index.append(f)


ama_increase, ama_decrease = pf.patterns(ama_data, 9999999)
ama_increasing_val_ = ama_increase[2]-ama_increase[0]
a,b,c,d,e,f = pf.inc_prtg(new_data25, ind.ama.__name__, ama_increase, 20, round(ama_increasing_val_, 4))

c_index.append(c)
f_index.append(f)


hma_increase, hma_decrease = pf.patterns(hma_data, 9999999)
hma_increasing_val_ = hma_increase[2]-hma_increase[0]
hma_decreasing_val_ = hma_decrease[2]-hma_decrease[0]
a,b,c,d,e,f = pf.inc_prtg(new_data24, ind.hma.__name__, hma_increase, 20, round(hma_increasing_val_, 4))

c_index.append(c)
f_index.append(f)


zlema_increase, zlema_decrease = pf.patterns(zlema_data, 9999999)
zlema_increasing_val_ = zlema_increase[2]-zlema_increase[0]
zlema_decreasing_val_ = zlema_decrease[2]-zlema_decrease[0]
a,b,c,d,e,f = pf.inc_prtg(new_data23, ind.zlema.__name__, zlema_increase, 20, round(zlema_increasing_val_, 4))

c_index.append(c)
f_index.append(f)


tma_increase, tma_decrease = pf.patterns(tma_data, 9999999)
tma_increasing_val_ = tma_increase[2]-tma_increase[0]
tma_decreasing_val_ = tma_decrease[2]-tma_decrease[0]
a,b,c,d,e,f = pf.inc_prtg(new_data22, ind.tma.__name__, tma_increase, 20, round(tma_increasing_val_, 4))

c_index.append(c)
f_index.append(f)


wma_increase,wma_decrease = pf.patterns(wma_data, 9999999)
wma_increasing_val_ = wma_increase[2]-wma_increase[0]
wma_decreasing_val_ = wma_decrease[2]-wma_decrease[0]
a,b,c,d,e,f = pf.inc_prtg(new_data21, ind.wma.__name__, wma_increase, 20, round(wma_increasing_val_, 4))

c_index.append(c)
f_index.append(f)


gap_increase, gap_decrease = pf.patterns(gap_data, 9999999)
gap_increasing_val_ = gap_increase[2]-gap_increase[0]
gap_decreasing_val_ = gap_decrease[2]-gap_decrease[0]
a,b,c,d,e,f = pf.inc_prtg(new_data20, ind.gap.__name__, gap_increase, 20, round(gap_increasing_val_, 4))

c_index.append(c)
f_index.append(f)


di_increase, di_decrease = pf.patterns(di_data, 9999999)
di_increasing_val_ = di_increase[2]-di_increase[0]
di_decreasing_val_ = di_decrease[2]-di_decrease[0]
a,b,c,d,e,f = pf.inc_prtg(new_data19, ind.di.__name__, di_increase, 20, round(di_increasing_val_, 4))

c_index.append(c)
f_index.append(f)


ac_increase, ac_decrease = pf.patterns(ac_data, 9999999)
ac_increasing_val_ = ac_increase[2]-ac_increase[0]
ac_decreasing_val_ = ac_decrease[2]-ac_decrease[0]
a,b,c,d,e,f = pf.inc_prtg(new_data18, ind.ac.__name__, ac_increase, 20, round(ac_increasing_val_, 4))

c_index.append(c)
f_index.append(f)


ultima_increase, ultima_decrease = pf.patterns(ultima_data, 9999999)
ultima_increasing_val_ = ultima_increase[2]-ultima_increase[0]
ultima_decreasing_val_ = ultima_decrease[2]-ultima_decrease[0]
a,b,c,d,e,f = pf.inc_prtg(new_data17, ind.ultima.__name__, ultima_increase, 20, round(ultima_increasing_val_, 4))

c_index.append(c)
f_index.append(f)


mfi_increase, mfi_decrease = pf.patterns(mfi_data, 9999999)
mfi_increasing_val_ = mfi_increase[2]-mfi_increase[0]
mfi_decreasing_val_ = mfi_decrease[2]-mfi_decrease[0]
a, b, c, d, e, f = pf.inc_prtg(new_data16, ind.mfi.__name__, mfi_increase, 20, round(mfi_increasing_val_, 4))

c_index.append(c)
f_index.append(f)


dmi_stochastic_increase, dmi_stochastic_decrease = pf.patterns(dmi_stochastic_data, 9999999)
dmi_stochastic_increasing_val_ = dmi_stochastic_increase[2]-dmi_stochastic_increase[0]
dmi_stochastic_decreasing_val_ = dmi_stochastic_decrease[2]-dmi_stochastic_decrease[0]
a,b,c,d,e,f = pf.inc_prtg(new_data15, ind.dmi_stochastic.__name__, dmi_stochastic_increase, 20, round(dmi_stochastic_increasing_val_, 4))

c_index.append(c)
f_index.append(f)


vroc_increase, vroc_decrease = pf.patterns(vroc_data, 9999999)
vroc_increasing_val_ = vroc_increase[2]-vroc_increase[0]
vroc_decreasing_val_ = vroc_decrease[2]-vroc_decrease[0]
a,b,c,d,e,f = pf.inc_prtg(new_data14, ind.vroc.__name__, vroc_increase, 20, round(vroc_increasing_val_, 4))

c_index.append(c)
f_index.append(f)


asi_increase, asi_decrease = pf.patterns(asi_data, 9999999)
asi_increasing_val_ = asi_increase[2]-asi_increase[0]
asi_decreasing_val_ = asi_decrease[2]-asi_decrease[0]
a,b,c,d,e,f = pf.inc_prtg(new_data13, ind.asi.__name__, asi_increase, 20, round(asi_increasing_val_, 4))

c_index.append(c)
f_index.append(f)


vwma_increase, vwma_decrease = pf.patterns(vwma_data, 9999999)
vwma_increasing_val_ = vwma_increase[2]-vwma_increase[0]
vwma_decreasing_val_ = vwma_decrease[2]-vwma_decrease[0]
a,b,c,d,e,f = pf.inc_prtg(new_data12, ind.vwma.__name__, vwma_increase, 20, round(vwma_increasing_val_, 4))

c_index.append(c)
f_index.append(f)


cvi_increase, cvi_decrease = pf.patterns(cvi_data, 9999999)
cvi_increasing_val_ = cvi_increase[2]-cvi_increase[0]
cvi_decreasing_val_ = cvi_decrease[2]-cvi_decrease[0]
a,b,c,d,e,f = pf.inc_prtg(new_data11, ind.cvi.__name__, cvi_increase, 20, round(cvi_increasing_val_, 4))

c_index.append(c)
f_index.append(f)


""""eom_increase, eom_decrease = pf.patterns(eom_data, 9999999)
eom_increasing_val_ = eom_increase[2]-eom_increase[0]
eom_decreasing_val_ = eom_decrease[2]-eom_decrease[0]
a,b,c,d,e,f = pf.inc_prtg(new_data10, ind.eom.__name__, eom_increase, 20, round(eom_increasing_val_, 4))
"""""
c_index.append(c)
f_index.append(f)


mf_index_increase, mf_index_decrease = pf.patterns(mf_index_data, 9999999)
mf_index_increasing_val_ = mf_index_increase[2]-mf_index_increase[0]
mf_index_decreasing_val_ = mf_index_decrease[2]-mf_index_decrease[0]
a,b,c,d,e,f = pf.inc_prtg(new_data9, ind.mf_index.__name__, mf_index_increase, 20, round(mf_index_increasing_val_, 4))

c_index.append(c)
f_index.append(f)


cmf_increase, cmf_decrease = pf.patterns(cmf_data, 9999999)
cmf_increasing_val_ = cmf_increase[2]-cmf_increase[0]
cmf_decreasing_val_ = cmf_decrease[2]-cmf_decrease[0]
a,b,c,d,e,f = pf.inc_prtg(new_data8, ind.cmf.__name__, cmf_increase, 20, round(cmf_increasing_val_, 4))

c_index.append(c)
f_index.append(f)


atr_increase, atr_decrease = pf.patterns(atr_data, 9999999)
atr_increasing_val_ = atr_increase[2]-atr_increase[0]
atr_decreasing_val_ = atr_decrease[2]-atr_decrease[0]
a,b,c,d,e,f = pf.inc_prtg(new_data7, ind.atr.__name__, atr_increase, 20, round(atr_increasing_val_, 4))

c_index.append(c)
f_index.append(f)


williams_r_increase, williams_r_decrease = pf.patterns(williams_r_data, 9999999)
williams_r_increasing_val_ = williams_r_increase[2]-williams_r_increase[0]
williams_r_decreasing_val_ = williams_r_decrease[2]-williams_r_decrease[0]
a,b,c,d,e,f = pf.inc_prtg(new_data6, ind.williams_r.__name__, williams_r_increase, 20, round(williams_r_increasing_val_, 4))

c_index.append(c)
f_index.append(f)


rsi_increase, rsi_decrease = pf.patterns(rsi_data, 9999999)
rsi_increasing_val_ = rsi_increase[2]-rsi_increase[0]
rsi_decreasing_val_ = rsi_decrease[2]-rsi_decrease[0]
a,b,c,d,e,f = pf.inc_prtg(new_data5, ind.rsi.__name__, rsi_increase, 20, round(rsi_increasing_val_, 4))

c_index.append(c)
f_index.append(f)


stochastic_oscillator_increase, stochastic_oscillator_decrease = pf.patterns(stochastic_oscillator_data, 9999999)
stochastic_oscillator_increasing_val_ = stochastic_oscillator_increase[2]-stochastic_oscillator_increase[0]
stochastic_oscillator_decreasing_val_ = stochastic_oscillator_decrease[2]-stochastic_oscillator_decrease[0]
a,b,c,d,e,f = pf.inc_prtg(new_data4, ind.stochastic_oscillator.__name__, stochastic_oscillator_increase, 20, round(stochastic_oscillator_increasing_val_, 4))

c_index.append(c)
f_index.append(f)


z_score_increase, z_score_decrease = pf.patterns(z_score_data, 9999999)
z_score_increasing_val_ = z_score_increase[2]-z_score_increase[0]
z_score_decreasing_val_ = z_score_decrease[2]-z_score_decrease[0]
a,b,c,d,e,f = pf.inc_prtg(new_data3, ind.z_score.__name__, z_score_increase, 20, round(z_score_increasing_val_, 4))

c_index.append(c)
f_index.append(f)


cci_increase, cci_decrease = pf.patterns(cci_data, 9999999)
cci_increasing_val_ = cci_increase[2]-cci_increase[0]
cci_decreasing_val_ = cci_decrease[2]-cci_decrease[0]
a,b,c,d,e,f = pf.inc_prtg(new_data2, ind.cci.__name__, cci_increase, 20, round(cci_increasing_val_, 4))

c_index.append(c)
f_index.append(f)


adx_increase, adx_decrease = pf.patterns(adx_data, 9999999)
adx_increasing_val_ = adx_increase[2]-adx_increase[0]
adx_decreasing_val_ = adx_decrease[2]-adx_decrease[0]
a,b,c,d,e,f = pf.inc_prtg(new_data1, ind.adx.__name__, adx_increase, 20, round(adx_increasing_val_, 4))

c_index.append(c)
f_index.append(f)


new_data = []
for i in range(33):
    name, ind_fr =ind.ind_series(test, 14, i+1)
    new_data.append(crv(test, ind_fr, name))

combo_frame = []

for i in range(len(new_data)):
    new_data[i].drop(['Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits', 'Close'] ,axis = 1, inplace = True)
frtm = []

adx = new_data[0]
cci = new_data[1]
z_score = new_data[2]
stochastic_oscillator = new_data[3]
rsi = new_data[4]
williams_r = new_data[5]
atr = new_data[6]
cmf = new_data[7]
mf_index = new_data[8]
eom = new_data[9]
cvi = new_data[10]
vwma = new_data[11]
asi = new_data[12]
vroc = new_data[13]
dmi_stochastic = new_data[14]
mfi = new_data[15]
ultima = new_data[16]
ac = new_data[17]
di = new_data[18]
gap = new_data[19]
wma = new_data[20]
tma = new_data[21]
zlema = new_data[22]
hma = new_data[23]
ama = new_data[24]
dema = new_data[25]
tema = new_data[26]
frama = new_data[27]
vortex = new_data[28]
trix = new_data[29]
stochastic_rsi = new_data[30]
roc = new_data[31]
momentum = new_data[32]

signal_array = []
pf.sign_model(test, adx, f_index[0], 'adx')
pf.sign_model(test, cci, f_index[1], 'cci')
pf.sign_model(test, z_score, f_index[2], 'z_score')
pf.sign_model(test, stochastic_oscillator, f_index[3], 'stochastic_oscillator')
pf.sign_model(test, rsi, f_index[4], 'rsi')
pf.sign_model(test, williams_r, f_index[5], 'williams_r')
pf.sign_model(test, atr, f_index[6], 'atr')
pf.sign_model(test, cmf, f_index[7], 'cmf')
pf.sign_model(test, mf_index, f_index[8], 'mf_index')
pf.sign_model(test, eom, f_index[9], 'eom')
pf.sign_model(test, cvi, f_index[9], 'cvi')
pf.sign_model(test, vwma, f_index[10], 'vwma')
pf.sign_model(test, asi, f_index[11], 'asi')
pf.sign_model(test, vroc, f_index[12], 'vroc')
pf.sign_model(test, dmi_stochastic, f_index[13], 'dmi_stochastic')
pf.sign_model(test, mfi, f_index[14], 'mfi')
pf.sign_model(test, ultima, f_index[15], 'ultima')
pf.sign_model(test, ac, f_index[16], 'ac')
pf.sign_model(test, di, f_index[17], 'di')
pf.sign_model(test, gap, f_index[18], 'gap')
pf.sign_model(test, wma, f_index[19], 'wma')
pf.sign_model(test, tma, f_index[20], 'tma')
pf.sign_model(test, zlema, f_index[21], 'zlema')
pf.sign_model(test, hma, f_index[22], 'hma')
pf.sign_model(test, ama, f_index[23], 'ama')
pf.sign_model(test, dema, f_index[24], 'dema')
pf.sign_model(test, tema, f_index[25], 'tema')
pf.sign_model(test, frama, f_index[26], 'frama')
pf.sign_model(test, vortex, f_index[27], 'vortex')
pf.sign_model(test, trix, f_index[28], 'trix')
pf.sign_model(test, stochastic_rsi, f_index[29], 'stochastic_rsi')
pf.sign_model(test, roc, f_index[30], 'roc')
pf.sign_model(test, momentum, f_index[31], 'momentum')

combinations = pf.generate_combinations(800)
func_combinations = [] 
for combination in combinations:
    name_list = []
    for num in combination:
        name_list.append(pf.name(num+1))
    func_combinations.append(name_list)

last_check = []
for combination_name in func_combinations:
    base_test['final_confirmation'] = 0
    for name in combination_name:    
        for i in range(len(test)):
            base_test['final_confirmation'][i] += test[name][i]
    last_check.append(base_test['final_confirmation'])

def check_threshold(last_ch, df, threshold):
    array_ratio = []
    array_win = []
    accuracy_indices = []
    win_treshold =[]
    capital_ = []
    win_av = []
    for check in last_ch:
        capital= 100
        percentage = []
        incremental = []

        win = 0
        loss = 0
        
        for i in range(len(check)-7):
            incremental.append(capital)
            if check[i] >= threshold:
                change = (((df['Close'][i]-df['Close'][i-7])/df['Close'][i-7])*100)
                capital=capital+(capital/100*(change/100))
                if change > 0:
                    win+=1
                else:
                    loss+=1
                percentage.append(change)
            else:
                capital = capital
        if win+loss>0:    
            win_ratio = win/(win+loss)
            win_av.append(sum(percentage)/(win+loss))
        else:
            win_ratio = 0
            win_av.append(0)
        win_treshold.append(win+loss)
        capital_.append(incremental)
        total_win = sum(percentage)
        array_ratio.append(win_ratio)
        array_win.append(total_win)
        
        accuracy_indices.append((win_ratio*100)*total_win)
    return array_ratio, array_win, accuracy_indices, win_treshold, capital_, win_av

above_ratio = []
above_win = []
above_confirm=[]
above_best_win = []
above_best_ratio = []
above_accuracy = []
above_n_win = []
above_capital = []
win_av_ =[]
for i in range(2, 6):
    ratio, win_, accuracy, n_win, capital, win_av=check_threshold(last_check, base_test, i)
    above_confirm.append(i)
    above_ratio.append(ratio)
    above_win.append(win_)
    above_accuracy.append(accuracy)
    above_n_win.append(n_win)
    above_capital.append(capital)
    win_av_.append(win_av)

param_folder = []
r_1 =[]
p_1 = []
i_1 = []
w_1 = []
c_1 =[]
wa_1 =[]
for i in range(4):
    best_combination_accuracy =above_accuracy[i].index(max(above_accuracy[i]))
    r_1.append(above_ratio[i][best_combination_accuracy])
    p_1.append(above_win[i][best_combination_accuracy])
    w_1.append(above_n_win[i][best_combination_accuracy])
    i_1.append(best_combination_accuracy)
    c_1.append(above_capital[i][best_combination_accuracy])
    wa_1.append(win_av_[i][best_combination_accuracy])
    param_folder.append(best_combination_accuracy)
print(r_1)
print(p_1)
print(i_1)
print(w_1)
print(c_1)
for i in range(4):
    candlestick = figure(x_axis_type="datetime", width=1400, height= 300, x_range=(base_test.index.min(), base_test.index.max()))
    candlestick.line(base_test.index, base_test.Close, width=1, line_color="black", alpha=0.8)## Volume Chart
    
    indicator = figure(x_axis_type="datetime", width=1400, height=  250, x_range=(base_test.index.min(), base_test.index.max()))
    indicator.line(base_test.index, c_1[i], width=1, line_color="fuchsia", alpha=0.8)

    show(column(candlestick, indicator))
print(4)