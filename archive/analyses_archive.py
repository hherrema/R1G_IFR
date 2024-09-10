### Analysis code (analyses not in paper)

# imports
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import scipy.stats

# probability of 2nd recall following intrusion
def p2r_intr_sess(data, ll):
    p2r = np.zeros(ll)
    intr = 0                     # number of lists initiated with intrusion
    r2_intr = 0                  # second recall intrusions
    rec_evs = data[data['type'] == 'REC_WORD']
    
    for i in data.list.unique():
        sp = rec_evs[rec_evs['list'] == i].serial_position.to_numpy()      # current list serial positions
        if len(sp) > 2:          # always null recall at end of list, so need at least 3 recalls
            if sp[0] == 88:      # first recall intrusion
                intr += 1
                if sp[1] > 0 and sp[1] <= ll:
                    p2r[int(sp[1] - 1)] += 1
                elif sp[1] == 88:
                    r2_intr += 1
                    
    if intr > 0:
        p2r /= intr
        r2_intr /= intr
    else:
        return np.full(ll + 1, np.nan)
    
    return np.concatenate([np.array([r2_intr]), p2r])
    
def p2r_intr(df, condition_map):
    p2r_intr_data = []
    for (sub, strat, sess, c, ll, pr), data in tqdm(df.groupby(['worker_id', 'strategy', 'session', 'condition', 'l_length', 'pres_rate'])):
        p2r = p2r_intr_sess(data, int(ll))
        p2r_intr_data.append((sub, strat, sess, condition_map.get(c), ll, pr,) + tuple(p2r))
        
    # store results in dataframe
    cols = np.array(['subject', 'strategy', 'session', 'condition', 'l_length', 'pres_rate', 'intrusion'] + [f'sp_{x}' for x in range(1, 41)])
    p2r_intr_data = pd.DataFrame(p2r_intr_data, columns=cols)
    
    return p2r_intr_data

def p2r_intr_btwn_subj_avg(p2r_intr_data):
    p2r_intr_data_bsa = p2r_intr_data.groupby(['subject', 'strategy', 'condition', 'l_length', 'pres_rate'])[['intrusion'] + [f'sp_{x}' for x in range(1, 41)]].mean().reset_index()
    
    # sort by condition
    p2r_intr_data_bsa = sort_by_condition(p2r_intr_data_bsa)

    return p2r_intr_data_bsa


# final N recalls
# lists with mu +/- sig correct recalls
def irt_final_sess(data, ll, lb, ub):
    irt = np.full((len(data.list.unique()), lb), np.nan)
    rec_evs = data[data['type'] == 'REC_WORD']
    
    for idx, i in enumerate(data.list.unique()):
        sp = rec_evs[rec_evs['list'] == i].serial_position.to_numpy()    # current list serial positions
        sp = sp[sp != 99]                                                # remove last null recall
        rt = rec_evs[rec_evs['list'] == i].rt.to_numpy()                 # current list response times
        rt = rt[:len(sp)]                                                # remove last null recall
        
        sp = mark_repetitions(sp)             # change repetitions to serial position 77
        
        ncr = len(sp[(sp!=88) & (sp!=77)])    # number of correct recalls
        if ncr <= lb or ncr > ub:              # only lists with ncr within 1 standard deviation of condition MWR
            continue
        
        irts = []
        for j in range(len(sp) - 1):
            # transition between correct recalls
            if sp[j]!=88 and sp[j]!=77 and sp[j+1]!=88 and sp[j+1]!=77:
                irts.append(rt[j+1])
                
            # transition from intrusion/repetition to correct recall (not first recall)
            elif (sp[j]==88 or sp[j]==77) and sp[j+1]!=88 and sp[j+1]!=77 and np.any((sp[:j] >= 1) & (sp[:j] <= ll)):
                irts.append(np.nan)

        irt[idx, :] = np.array(irts)[-lb:]

    return np.nanmean(irt, axis=0)

def irt_final(df, condition_map, mwr_md):
    irt_data = []
    for (sub, strat, sess, c, ll, pr), data in tqdm(df.groupby(['worker_id', 'strategy', 'session', 'condition', 'l_length', 'pres_rate'])):
        c_ = condition_map.get(c)
        lb = mwr_md.query("condition == @c_").iloc[0].lb
        ub = mwr_md.query("condition == @c_").iloc[0].ub
        irt = irt_final_sess(data, int(ll), lb, ub)
        irt = [(sub, strat, sess, c_, ll, pr,) + tuple(irt)]
        irt = pd.DataFrame(irt, columns=['subject', 'strategy', 'session', 'condition', 'l_length', 'pres_rate'] +
                           [f'rot_{x}' for x in np.arange(lb-1, -1, -1)])
        irt_data.append(irt)
        
    irt_data = pd.concat(irt_data, ignore_index=True)
    irt_data = irt_data[['subject', 'strategy', 'session', 'condition', 'l_length', 'pres_rate'] + [f'rot_{x}' for x in np.arange(max(mwr_md.lb)-1, -1, -1)]]

    return irt_data

def irt_final_btwn_subj_avg(irt_final_data):
    irt_final_data_bsa = irt_final_data.groupby(['subject', 'strategy', 'condition', 'l_length', 'pres_rate'])[[f'rot_{x}' for x in range(7, -1, -1)]].mean().reset_index()
    
    # sort by condition
    irt_final_data_bsa = sort_by_condition(irt_final_data_bsa)

    return irt_final_data_bsa

# conditioned on serial position
def lag_crp_sp_sess(data, ll, buffer):
    ac = np.zeros((ll, 2*ll - 1))
    poss = np.zeros_like(ac)
    word_evs = data[data['type'] == 'WORD']; rec_evs = data[data['type'] == 'REC_WORD']
    
    for i in data.list.unique():
        sp = rec_evs[rec_evs['list'] == i].serial_position.to_numpy()       # current list serial positions
        sp = sp[sp != 99]                                                   # remove last null recall
        
        sp = mark_repetitions(sp)
        
        # exclude first 'buffer' output positions
        excludeOutputs = sp[:buffer]
        sp = sp[buffer:]
        sps_left = [x for x in range(1, ll+1)]        # array with remaining serial positions, subtract from
        
        for exOut in excludeOutputs:
            try:
                sps_left.remove(exOut)               # remove first outputs from possible transitions
            except:
                pass                                  # item already removed or intrusion
        
        for j in range(len(sp) - 1):
            if sp[j]!=88 and sp[j]!=77:             # correct recall
                sps_left.remove(sp[j])                # can't transition to already recalled serial position
                
                if sp[j+1]!=88 and sp[j+1]!=77:       # transition between correct recalls
                    lag = sp[j+1] - sp[j]             # actual transition
                    ac[int(sp[j])-1, int(lag)+ll-1] += 1
                    for l in sps_left:                # find all possible transitions
                        p_lag = l - sp[j]
                        poss[int(sp[j])-1, int(p_lag)+ll-1] += 1
    
    crp = ac / poss       # we actually want the NaNs from zero division
    crp = np.delete(crp, crp.shape[1]//2, axis=1)        # remove lag = 0
    
    return pd.DataFrame(crp, columns=[f'ln_{x}' for x in np.arange(ll-1, 0, -1)] + [f'lp_{x}' for x in range(1, ll)])

def lag_crp_sp(df, condition_map, buffer):
    lcrp_sp_data = []
    for (sub, strat, sess, c, ll, pr), data in tqdm(df.groupby(['worker_id', 'strategy', 'session', 'condition', 'l_length', 'pres_rate'])):
        lcrp = lag_crp_sp_sess(data, int(ll), buffer)
        lcrp['subject'] = sub; lcrp['strategy'] = strat; lcrp['session'] = sess
        lcrp['condition'] = condition_map.get(c); lcrp['l_length'] = ll; lcrp['pres_rate'] = pr
        lcrp['serial_position'] = np.arange(1, int(ll)+1)
        
        lcrp_sp_data.append(lcrp)
        
    lcrp_sp_data = pd.concat(lcrp_sp_data, ignore_index=True)
    lcrp_sp_data = lcrp_sp_data[['subject', 'strategy', 'session', 'condition', 'l_length', 'pres_rate', 'serial_position'] + 
                                [f'ln_{x}' for x in np.arange(39, 0, -1)] + [f'lp_{x}' for x in range(1, 40)]]
    return lcrp_sp_data

def lag_crp_sp_btwn_subj_avg(lcrp_sp_data):
    lcrp_sp_data_bsa = lcrp_sp_data.groupby(['subject', 'strategy', 'condition', 'l_length', 'pres_rate', 'serial_position'])[[f'ln_{x}' for x in range(39, 0, -1)] + [f'lp_{x}' for x in range(1, 40)]].mean().reset_index()

    # sort by condition
    lcrp_sp_data_bsa = sort_by_condition(lcrp_sp_data_bsa)

    return lcrp_sp_data_bsa

# lag +1 v. -1 asymmetry (both transitions available)
def lag_crp_l1_sess(data, ll, buffer):
    p1 = 0; n1 = 0                 # +1, -1 lag transition counters
    poss = 0                       # +1, -1 lag transition available
    rec_evs = data[data['type'] == 'REC_WORD']
    
    for i in data.list.unique():
        sp = rec_evs[rec_evs['list'] == i].serial_position.to_numpy()      # current list serial positions
        sp = sp[sp != 99]                                                  # remove last null recall
        
        sp = mark_repetitions(sp)
        
        # exclude first 'buffer' output positions
        excludeOutputs = sp[:buffer]
        sp = sp[buffer:]
        sps_left = [x for x in range(1, ll+1)]        # array with remaining serial positions, subtract from
        
        for exOut in excludeOutputs:
            try:
                sps_left.remove(exOut)                # remove first outputs from possible transitions
            except:
                pass                                  # item already removed or intrusion
            
        for j in range(len(sp) - 1):
            if sp[j]!=88 and sp[j]!=77:               # correct recall
                sps_left.remove(sp[j])                # can't transition to already recalled serial position
                
                if sp[j+1]!=88 and sp[j+1]!=77:       # transition between correct recalls
                    possList = []
                    lag = sp[j+1] - sp[j]             # actual transition
                    
                    for l in sps_left:                # find all possible transitions
                        possList.append(l - sp[j])
                        
                    if 1 in possList and -1 in possList:    # both +1, -1 lag transitions available
                        poss += 1
                        if lag == 1:
                            p1 += 1
                        elif lag == -1:
                            n1 += 1
    
    return p1/poss, n1/poss, (p1/poss) - (n1/poss)     # we actually want the NaNs from zero division

def lag_crp_l1(df, condition_map, buffer):
    lcrp_l1_data = []
    for (sub, strat, sess, c, ll, pr), data in tqdm(df.groupby(['worker_id', 'strategy', 'session', 'condition', 'l_length', 'pres_rate'])):
        crp_p1, crp_n1, crp_delta = lag_crp_l1_sess(data, int(ll), buffer)
        lcrp_l1_data.append((sub, strat, sess, condition_map.get(c), ll, pr, crp_p1, crp_n1, crp_delta))
        
    # store results in dataframe
    lcrp_l1_data = pd.DataFrame(lcrp_l1_data, columns=['subject', 'strategy', 'session', 'condition', 'l_length', 'pres_rate', 'crp_p1', 'crp_n1', 'crp_delta'])
    
    return lcrp_l1_data

def lag_crp_l1_btwn_subj_avg(lcrp_l1_data):
    lcrp_l1_data_bsa = lcrp_l1_data.groupby(['subject', 'strategy', 'condition', 'l_length', 'pres_rate'])[['crp_p1', 'crp_n1', 'crp_delta']].mean().reset_index()
    
    # sort by condition
    lcrp_l1_data_bsa = sort_by_condition(lcrp_l1_data_bsa)
    
    return lcrp_l1_data_bsa