def crv (data, ind, name_, loc_):
    llen = len(data)
    llen_ = len(ind)
    diff=llen- llen_
    data = data.assign(indicator = 0)
    data.rename(columns={'indicator': name_}, inplace = True)
    data.iloc[diff:,data.columns.get_loc(name_)]= ind
    return data

def scan_data_v4(df,name_):
    # Create an empty list to store the indicator values
    indicator_values = []
    # Create a variable to store the state of the market
    is_bull = False
    
    # Iterate through the rows of the dataframe
    for i in range(len(df) - 7):
        # Check if the market is in a bullish state
        if is_bull:
            # Check if the market slows down to a 0.25% increase in 7 days
            if df.Close[i + 7] <= df.Close[i] * 1.0025:
                is_bull = False
        else:
            # Check if the price has risen by at least 1% in the next 7 days
            if df.Close[i + 7] > df.Close[i] * 1.01:
                # Append the indicator values of the 5 days before the price rise to the list
                indicator_values.append(df[name_][i-5:i].tolist())
                is_bull = True
    
    return indicator_values

def patterns(indicator_values, max_change):
    grid_1 = []
    grid_2 = []
    positive_grid = []
    negative_grid = []
    
    for i in range(2, len(indicator_values)):
        change_grid_1 = max(indicator_values[i])-min(indicator_values[i])
        if change_grid_1<=max_change:
            grid_1.append(indicator_values[i])
    first_values = [sublist[0] for sublist in grid_1]
    last_values = [sublist[-1] for sublist in grid_1]
    
    for i in range( len(first_values)):
        change_grid_2 = ((last_values[i]-first_values[i])/(first_values[i]+1))*100
        grid_2.append(change_grid_2)
    
    for i in range(len(grid_2)):
        if grid_2[i] > 0:
            positive_grid.append(grid_2[i])
        if grid_2[i] < 0:
            negative_grid.append(grid_2[i])
    positive_grid.sort()
    negative_grid.sort(reverse = True)
    return positive_grid, negative_grid

def by_prtg_win(df, percentage,name_):
    # Create an empty list to store the indicator values
    indicator_values = []
    win = []
    loss = []
    rin = 0
    ren = 0
    for i in range(len(df) - 7):
        if (((df[name_][i + 7] - df[name_][i])/df[name_][i])*100) > percentage:
            indicator_values.extend([df.Close[i-7], df.Close[i]])
            if df.Close[i-7]< df.Close[i]:
                change = (((df.Close[i]-df.Close[i-7])/df.Close[i-7])*100)
                rin +=1
                win.append(change)
            if df.Close[i-7]> df.Close[i]:
                change = (((df.Close[i]-df.Close[i-7])/df.Close[i-7])*100)
                ren +=1
                loss.append(change)
    win_ratio = rin/(rin+ren)
    total_win = sum(win)-sum(loss)

    return win_ratio, win, loss, total_win

def dec_prtg_win(df, percentage,name_):
    # Create an empty list to store the indicator values
    indicator_values = []
    win = []
    loss = []
    rin = 1
    ren = 1
    for i in range(len(df) - 7):
        if (((df[name_][i + 7] - df[name_][i])/df[name_][i])*100) < percentage:
            indicator_values.extend([df.Close[i-7], df.Close[i]])
            if df.Close[i-7]< df.Close[i]:
                change = (((df.Close[i]-df.Close[i-7])/df.Close[i-7])*100)
                rin +=1
                win.append(change)
            if df.Close[i-7]> df.Close[i]:
                change = (((df.Close[i]-df.Close[i-7])/df.Close[i-7])*100)
                ren +=1
                loss.append(change)
    win_ratio = rin/(rin+ren)
    total_win = sum(win)-sum(loss)

    return win_ratio, win, loss, total_win

def inc_prtg (data, name_, increase_, num_it, inc_val):
    perc=round(increase_[0], 4)
    ratio_eval= []
    win_eval= []
    perc_eval= []
    for i in range(num_it):
        ratio, win, loss, total_win= by_prtg_win(data,perc,name_)
        perc_eval.append(perc)
        ratio_eval.append(ratio)
        win_eval.append(total_win)
        perc+=inc_val
    bwb = max(win_eval)
    for i in range(len(win_eval)):
        if bwb == win_eval[i]:
            brw=ratio_eval[i]
            bpw = round(perc_eval[i], 4)
    brb = max(ratio_eval)
    for i in range(len(ratio_eval)):
        if brb == ratio_eval[i]:
            brr=win_eval[i]
            bpr = round(perc_eval[i], 4) 

    return bwb, brw, bpw, brr, brb,  bpr

def dec_prtg (data, name_, decrease_, num_it, inc_val):
    perc=round(decrease_[0], 4)
    ratio_eval= []
    win_eval= []
    perc_eval= []
    for i in range(num_it):
        ratio, win, loss, total_win= dec_prtg_win(data,perc,name_)
        perc_eval.append(perc)
        ratio_eval.append(ratio)
        win_eval.append(total_win)
        perc-=inc_val
    bwb = max(win_eval)
    for i in range(len(win_eval)):
        if bwb == win_eval[i]:
            brw=ratio_eval[i]
            bpw = round(perc_eval[i], 4)
    brb = max(ratio_eval)
    for i in range(len(ratio_eval)):
        if brb == ratio_eval[i]:
            brr=win_eval[i]
            bpr = round(perc_eval[i], 4) 

    return bwb, brw, bpw, brr, brb,  bpr

import random
import numpy as np

def eval_model(df, df_ind, percentage,name_):
    # Create an empty list to store the indicator values
    indicator_values = []
    win = []
    loss = []
    rin = 0
    ren = 0
    for i in range(len(df_ind) - 7):
        if (((df_ind[name_][i + 7] - df_ind[name_][i])/df_ind[name_][i])*100) > percentage:
            indicator_values.extend([df.Close[i-7], df.Close[i]])
            if df.Close[i-7]< df.Close[i]:
                change = (((df.Close[i]-df.Close[i-7])/df.Close[i-7])*100)
                rin +=1
                win.append(change)
            if df.Close[i-7]> df.Close[i]:
                change = (((df.Close[i]-df.Close[i-7])/df.Close[i-7])*100)
                ren +=1
                loss.append(change)
    win_ratio = rin/(rin+ren)
    total_win = sum(win)-sum(loss)

    return win_ratio, win, loss, total_win

def sign_model(df, df_ind, percentage,name_):
    # Create an empty list to store the indicator values
    indicator_values = []
    df[name_] = False
    for i in range(len(df_ind) - 7):
        if (((df_ind[i + 7] - df_ind[i])/df_ind[i])*100) > percentage:
            df[name_][i-1] = True        
    return df

def combination (data_frame, ind_1, ind_2, ind_3, ind_4, ind_5, ind_6, ind_7, ind_8, ind_9, ind_10, ind_11, ind_12, ind_13, ind_14, ind_15, ind_16, ind_17, ind_18, ind_19, ind_20, ind_21, ind_22, ind_23, ind_24, ind_25, ind_26, ind_27, ind_28, ind_29, ind_30, ind_31, ind_32, ind_33, change_index,  n_confirm, n_indicator, n_iteration):
    data_frame['confirm'] = False
    combinations = []
    # Iterate for the desired number of combinations
    for i in range(n_iteration):
        # Generate a random combination of numbers from 1 to 33
        combination = random.sample(range(1, 34), n_indicator)
        # Append the combination to the list
        combinations.append(combination)
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
        1: ind_1,
        2: ind_2,
        3: ind_3,
        4: ind_4,
        5: ind_5,
        6: ind_6,
        7: ind_7,
        8: ind_8,
        9: ind_9,
        10: ind_10,
        11: ind_11,
        12: ind_12,
        13: ind_13,
        14: ind_14,
        15: ind_15,
        16: ind_16,
        17: ind_17,
        18: ind_18,
        19: ind_19,
        20: ind_20,
        21: ind_21,
        22: ind_22,
        23: ind_23,
        24: ind_24,
        25: ind_25,
        26: ind_26,
        27: ind_27,
        28: ind_28,
        29: ind_29,
        30: ind_30,
        31: ind_31,
        32: ind_32,
        33: ind_33
    }
    val_cross = []
    name_iter =[]
    array_combinations = []
    for combination in combinations:
        array_combination = []
        val_tcross = []
        i = 0
        full_combo = []
        for number in combination:
            array_combination.append(array_mapping[number])
            val_tcross.append(change_index[number-1])
            name_iter.append(name_mapping[number])
            df = sign_model(data_frame, array_combination[i], val_tcross[i], name_mapping[number])
            df['final_confirm']= 0
            full_combo.append(df)
            i+=1
        array_combinations.append(array_combination)
        val_cross.append(val_tcross)
    profit_array =[]
    ratio_array = []
    for i in range(len(full_combo)):
        df_i =full_combo[i]
        open_pos = []
        close_pos = []
        for i in range(len(df_i)):
            for k in range(n_indicator):
                if df_i[name_iter[k]][i] == True:
                    df_i['final_confirm'][i]+=1
                    if df_i['final_confirm'][i]>=n_confirm:
                        open_pos.append(df_i['Close'][i])
                        close_pos.append(df_i['Close'][i+7])
        win_percent = []
        for i in range(len(close_pos)):
            win_percent.append(((close_pos[i]-open_pos[i])/open_pos[i])*100)
        win = 0
        loss = 0
        profit_array.append(sum(win_percent))
        for i in range(len(win_percent)):
            if win_percent[i]>0:
                win += 1
            if win_percent[i]<0:
                loss += 1
        ratio_array.append(win/(win+loss))

    
    return array_combinations, full_combo, profit_array, ratio_array