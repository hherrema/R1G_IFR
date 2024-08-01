### Analysis code
### Single session, all sessions, between-subject average

# imports
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import scipy.stats

# ---------- Utility ----------

# sort dataframe by condition
# list length and presentation rate or total time
def sort_by_condition(df, style='total_time'):
    if style=='total_time':
        conds = ['10-2', '20-1', '15-2', '30-1', '20-2', '40-1']
    elif style=='ll_pr':
        conds = ['10-2', '15-2', '20-2', '20-1', '30-1', '40-1']
    else:
        raise ValueError(f"Invalid category sort style: {style}.")
        
    df['condition'] = pd.Categorical(df['condition'], categories = conds)
    df = df.sort_values(by=['condition', 'subject'], ignore_index=True)
    
    return df

# change repetitions serial position to 77
def mark_repetitions(sp):
    used = []
    for u in range(len(sp)):
        count = 0
        if sp[u] != 88 and sp[u] != 77:
            used.append(sp[u])
        for v in range(len(used)):
            if sp[u] == used[v]:
                count += 1
        if count > 1:
            sp[u] = 77
        else:
            sp[u] = sp[u]
        
    return sp

# ---------- All Data Together ----------

# serial position curve
def spc_sess(data, ll):
    spc = np.zeros(ll)
    rec_evs = data[data['type'] == 'REC_WORD']
    
    for i in data.list.unique():
        sp = rec_evs[rec_evs['list'] == i].serial_position.to_numpy()        # current list serial positions
        srpos = np.unique(sp[(sp != 99) & (sp != 88)])                       # remove intrusions and repetitions

        for j in range(1, ll + 1):
            spc[j-1] += np.count_nonzero(srpos == j)
            
    spc /= len(data.list.unique())
    
    return spc

def spc_all(df, condition_map):
    spc_data_all = []
    for (sub, sess, c, ll, pr), data in tqdm(df.groupby(['worker_id', 'session', 'condition', 'l_length', 'pres_rate'])):
        spc = spc_sess(data, int(ll))
        spc_data_all.append((sub, sess, condition_map.get(c), ll, pr,) + tuple(spc))
    
    # store results in dataframe
    cols = np.array(['subject', 'session', 'condition', 'l_length', 'pres_rate'] + [f'sp_{x}' for x in range(1, 41)])
    spc_data_all = pd.DataFrame(spc_data_all, columns=cols)
    
    return spc_data_all

def spc_btwn_subj_avg_all(spc_data_all):
    spc_data_bsa_all = []
    for (sub, c, ll, pr), dat in tqdm(spc_data_all.groupby(['subject', 'condition', 'l_length', 'pres_rate'])):
        sp_precall = dat.iloc[:,5:].mean(axis=0).to_numpy()
        spc_data_bsa_all.append((sub, c, ll, pr,) + tuple(sp_precall))
        
    # store results in dataframe
    cols = np.array(['subject', 'condition', 'l_length', 'pres_rate'] + [f'sp_{x}' for x in range(1, 41)])
    spc_data_bsa_all = pd.DataFrame(spc_data_bsa_all, columns=cols)
    
    # sort by condition
    spc_data_bsa_all = sort_by_condition(spc_data_bsa_all)
    
    return spc_data_bsa_all


# linear regression for primacy ad recency effects
def prim_rec_lr(row):
    # primacy effect
    prim_spc = row[['sp_1', 'sp_2', 'sp_3', 'sp_4', 'sp_5']].astype(float)
    prim_slope, prim_intercept, _, _, _ = scipy.stats.linregress(np.arange(1, 6), prim_spc)
    
    # recency effect
    if row.l_length == 10.0:
        rec_spc = row[['sp_6', 'sp_7', 'sp_8', 'sp_9', 'sp_10']].astype(float)
    elif row.l_length == 15.0:
        rec_spc = row[['sp_11', 'sp_12', 'sp_13', 'sp_14', 'sp_15']].astype(float)
    elif row.l_length == 20.0:
        rec_spc = row[['sp_16', 'sp_17', 'sp_18', 'sp_19', 'sp_20']].astype(float)
    elif row.l_length == 30.0:
        rec_spc = row[['sp_26', 'sp_27', 'sp_28', 'sp_29', 'sp_30']].astype(float)
    else:
        rec_spc = row[['sp_36', 'sp_37', 'sp_38', 'sp_39', 'sp_40']].astype(float)
        
    rec_slope, rec_intercept, _, _, _ = scipy.stats.linregress(np.arange(1, 6), rec_spc)
    
    return pd.Series({'subject': row.subject, 'condition': row.condition, 
                        'l_length': row.l_length, 'pres_rate': row.pres_rate, 
                        'prim_slope': prim_slope, 'prim_intercept': prim_intercept, 
                        'rec_slope': rec_slope, 'rec_intercept': rec_intercept})


# probability of first recall
# toogle = True, only trials initiated with correct recall
def pfr_sess(data, ll, toggle):
    pfr = np.zeros(ll)
    rec_evs = data[data['type'] == 'REC_WORD']
    
    subtract = 0            # lists with no recalls (toggle=False) or lists with R1 != correct (toggle=True)
    for i in data.list.unique():
        sp = rec_evs[rec_evs['list'] == i].serial_position.to_numpy()      # current list serial positions
        
        # only lists with correct first recall
        if toggle:
            if len(sp) > 1 and sp[0] > 0 and sp[0] <= ll:
                pfr[int(sp[0]) - 1] += 1
            else:
                subtract += 1
                
        # all lists with at least 1 recall
        else:
            if len(sp) > 1:
                if sp[0] > 0 and sp[0] <= ll:
                    pfr[int(sp[0]) - 1] += 1
            else:
                subtract += 1
                
    pfr /= (len(data.list.unique()) - subtract)         # divide by valid lists in session
    
    return pfr

def pfr_all(df, condition_map, toggle):
    pfr_data_all = []
    for (sub, sess, c, ll, pr), data in tqdm(df.groupby(['worker_id', 'session', 'condition', 'l_length', 'pres_rate'])):
        pfr = pfr_sess(data, int(ll), toggle)
        pfr_data_all.append((sub, sess, condition_map.get(c), ll, pr,) + tuple(pfr))
    
    # store results in dataframe
    cols = np.array(['subject', 'session', 'condition', 'l_length', 'pres_rate'] + [f'sp_{x}' for x in range(1, 41)])
    pfr_data_all = pd.DataFrame(pfr_data_all, columns=cols)
    
    return pfr_data_all

def pfr_btwn_subj_avg_all(pfr_data_all):
    pfr_data_bsa_all = []
    for (sub, c, ll, pr), dat in tqdm(pfr_data_all.groupby(['subject', 'condition', 'l_length', 'pres_rate'])):
        sp_pfr = dat.iloc[:,5:].mean(axis=0).to_numpy()
        pfr_data_bsa_all.append((sub, c, ll, pr,) + tuple(sp_pfr))
        
    # store results in dataframe
    cols = np.array(['subject', 'condition', 'l_length', 'pres_rate'] + [f'sp_{x}' for x in range(1, 41)])
    pfr_data_bsa_all = pd.DataFrame(pfr_data_bsa_all, columns=cols)
    
    # sort by condition
    pfr_data_bsa_all = sort_by_condition(pfr_data_bsa_all)

    return pfr_data_bsa_all


# primacy and recency initiation bias
def pfr_primacy_recency_bias(pfr_data_bsa_all):
    # probability of first recall for primacy, middle, recency items
    prim_rec_pfr = []
    prim_cols = ['sp_1', 'sp_2', 'sp_3', 'sp_4']
    for _, row in tqdm(pfr_data_bsa_all.iterrows()):
        if row.l_length == 10:
            rec_cols = [f'sp_{x}' for x in range(7, 11)]
            middle_cols = [f'sp_{x}' for x in range(5, 7)]
        elif row.l_length == 15:
            rec_cols = [f'sp_{x}' for x in range(12, 16)]
            middle_cols = [f'sp_{x}' for x in range(5, 12)]
        elif row.l_length == 20:
            rec_cols = [f'sp_{x}' for x in range(17, 21)]
            middle_cols = [f'sp_{x}' for x in range(5, 17)]
        elif row.l_length == 30:
            rec_cols = [f'sp_{x}' for x in range(27, 31)]
            middle_cols = [f'sp_{x}' for x in range(5, 27)]
        elif row.l_length == 40:
            rec_cols = [f'sp_{x}' for x in range(37, 41)]
            middle_cols = [f'sp_{x}' for x in range(5, 37)]
    
        # sum of PFR (integral)
        prim_pfr = row[prim_cols].sum()
        middle_pfr = row[middle_cols].sum()
        rec_pfr = row[rec_cols].sum()
        
        # interested in recency - primacy bias
        prim_rec_pfr.append((row.subject, row.condition, row.l_length, row.pres_rate, 
                             prim_pfr, middle_pfr, rec_pfr, prim_pfr > middle_pfr, rec_pfr > middle_pfr, 
                             rec_pfr - prim_pfr, rec_pfr > prim_pfr))
        
    # store results in dataframe
    prim_rec_pfr = pd.DataFrame(prim_rec_pfr, columns=['subject', 'condition', 'l_length', 'pres_rate', 
                                                       'prim_pfr', 'middle_pfr', 'rec_pfr', 'prim_effect', 'rec_effect', 
                                                       'rec_prim_bias', 'rec_prim_bias_bool'])
    
    # condition mean for plotting color
    prim_rec_pfr['condition_mu'] = ['rec' if prim_rec_pfr[prim_rec_pfr['condition'] == row.condition].rec_prim_bias.mean() > 0
                                     else 'prim' for _, row in prim_rec_pfr.iterrows()]
    
    return prim_rec_pfr


# ---------- Motivation for R1 Groups ----------

# correlation of PFR and SPC (each serial position)
def pfr_spc_correlation(pfr_data_bsa_all, spc_data_bsa_all):
    pfr_spc_corrs = []
    for c in pfr_data_bsa_all.condition.unique():
        pfr_cond = pfr_data_bsa_all[pfr_data_bsa_all.condition == c]
        spc_cond = spc_data_bsa_all[spc_data_bsa_all.condition == c]
        ll = pfr_cond.l_length.unique()[0]
        pr = pfr_cond.pres_rate.unique()[0]
        for i, sp in enumerate([f'sp_{x}' for x in range(1, int(ll)+1)]):
            res = scipy.stats.pearsonr(pfr_cond[sp], spc_cond[sp], alternative='two-sided')
            pfr_spc_corrs.append((c, ll, pr, i+1, res.statistic, res.pvalue))


    pfr_spc_corrs = pd.DataFrame(pfr_spc_corrs, columns=['condition', 'l_length', 'pres_rate', 'serial_position', 'pearson_r', 'p_value'])
    
    return pfr_spc_corrs


# within session recall initiation variance
def r1_variance_sess(data, ll, cond_pfr):
    r1_list = []                  # serial positions of first recall
    r1_permutation = []           # serial positions of random recalls
    rec_evs = data[data['type'] == 'REC_WORD']
    
    for i in data.list.unique():
        sp = rec_evs[rec_evs['list'] == i].serial_position.to_numpy()     # current list serial positions
        if len(sp) > 1:     # always null recall at end of list
            if sp[0] > 0 and sp[0] <= ll:
                r1_list.append(sp[0])
                r1_permutation.append(np.random.choice(np.arange(1, ll+1), p=cond_pfr))        # random serial position
                
    r1_list = np.array(r1_list)                                          # array for mathematical operations
    r1_permutation = np.array(r1_permutation)
    denom = np.sqrt(len(r1_list))                                        # sqrt(n)
    if len(r1_list) > 0:
        r1_list_dec = r1_list / (ll/10)                                  # normalize to deciles
        r1_permutation_dec = r1_permutation / (ll/10)
    else:
        return np.nan, np.nan, np.nan, np.nan
    
    return np.std(r1_list) / denom, np.std(r1_list_dec) / denom, np.std(r1_permutation) / denom, np.std(r1_permutation_dec) / denom


def r1_variance(df, condition_map, pfr_data_only_cr_bsa_all):
    r1_var_data = []
    for (sub, sess, c, ll, pr), data in tqdm(df.groupby(['worker_id', 'session', 'condition', 'l_length', 'pres_rate'])):
        c_ = condition_map.get(c)                                                                          # condition string ("ll-pr")
        cond_data = pfr_data_only_cr_bsa_all.query("condition == @c_")
        cond_pfr = cond_data[[f'sp_{x}' for x in range(1, int(ll)+1)]].mean().to_numpy()                   # use PFR as pdf for permutation (only lists initiated with correct recall)
        r1_sem, r1_dec_sem, r1_perm_sem, r1_dec_perm_sem = r1_variance_sess(data, int(ll), cond_pfr)
        r1_var_data.append((sub, sess, c_, ll, pr, r1_sem, r1_dec_sem, r1_perm_sem, r1_dec_perm_sem))
        
    # store results in dataframe
    r1_var_data = pd.DataFrame(r1_var_data, columns=['subject', 'session', 'condition', 'l_length', 'pres_rate', 
                                                     'sp_sem', 'dec_sem', 'permutation_sp_sem', 'permutation_dec_sem'])
    
    return r1_var_data

def r1_variance_btwn_subj_avg(r1_var_data):
    r1_var_data_bsa = []
    for (sub, c, ll, pr), dat in tqdm(r1_var_data.groupby(['subject', 'condition', 'l_length', 'pres_rate'])):
        r1_var_data_bsa.append((sub, c, ll, pr, np.mean(dat.sp_sem), np.mean(dat.dec_sem), 
                                np.mean(dat.permutation_sp_sem), np.mean(dat.permutation_dec_sem)))
        
    # store results in dataframe
    r1_var_data_bsa = pd.DataFrame(r1_var_data_bsa, columns=['subject', 'condition', 'l_length', 'pres_rate', 
                                                             'sp_sem', 'dec_sem', 'permutation_sp_sem', 'permutation_dec_sem'])
    
    # sort by condition
    r1_var_data_bsa = sort_by_condition(r1_var_data_bsa)

    return r1_var_data_bsa


# change in recall initiation serial position across sessions
# only lists initiated with correct recall
def r1_sp_dec_sess(data, ll):
    r1_list = []               # serial positions of first recall
    rec_evs = data[data['type'] == 'REC_WORD']
    
    for i in data.list.unique():
        sp = rec_evs[rec_evs['list'] == i].serial_position.to_numpy()     # current list serial positions
        if len(sp) > 1:     # always null recall at end of list
            if sp[0] > 0 and sp[0] <= ll:
                r1_list.append(sp[0])
                
    r1_list = np.array(r1_list)     # array for mathematical operations
    if len(r1_list) > 0:
        r1_list_dec = r1_list / (ll/10)          # normalize to deciles
    else:
        return np.nan, np.nan
    
    return np.mean(r1_list), np.mean(r1_list_dec)

def r1_sp_dec(df, condition_map):
    r1_sp_dec_data = []
    for (sub, sess, c, ll, pr), data in tqdm(df.groupby(['worker_id', 'session', 'condition', 'l_length', 'pres_rate'])):
        r1_sp, r1_dec = r1_sp_dec_sess(data, ll)
        r1_sp_dec_data.append((sub, sess, condition_map.get(c), ll, pr, r1_sp, r1_dec))
        
    # store results in dataframe
    r1_sp_dec_data = pd.DataFrame(r1_sp_dec_data, columns=['subject', 'session', 'condition', 'l_length', 'pres_rate', 'r1_sp', 'r1_decile'])
    
    return r1_sp_dec_data

def r1_sp_dec_btwn_subj_avg(r1_sp_dec_data):
    r1_sp_dec_data_bsa = []
    for (sub, c, ll, pr), dat in tqdm(r1_sp_dec_data.groupby(['subject', 'condition', 'l_length', 'pres_rate'])):
        res_sp_lr = scipy.stats.linregress(dat.session, dat.r1_sp)                # linear regression of change in r1_sp over sessions
        res_dec_lr = scipy.stats.linregress(dat.session, dat.r1_decile)           # linear regression of change in r1_decile over sessions
        r1_sp_dec_data_bsa.append((sub, c, ll, pr, np.mean(dat.r1_sp), np.mean(dat.r1_decile), res_sp_lr.slope, res_dec_lr.slope))
        
    # store results in dataframe
    r1_sp_dec_data_bsa = pd.DataFrame(r1_sp_dec_data_bsa, columns=['subject', 'condition', 'l_length', 'pres_rate', 'r1_sp', 'r1_decile', 'r1_sp_slope', 'r1_dec_slope'])
    
    # sort by condition
    r1_sp_dec_data_bsa = sort_by_condition(r1_sp_dec_data_bsa)
    return r1_sp_dec_data_bsa


# ---------- Recall Initiation Groups ----------

# apply group labels
# probability of first recall, only lists initiated with correct recall
def r1_groups_partition(row):
    # primacy = 1st 4 serial positions
    prim_pfr = row[['sp_1', 'sp_2', 'sp_3', 'sp_4']].astype(float).sum()
    
    # recency = last 4 serial positions
    if row.l_length == 10.0:
        rec_pfr = row[['sp_7', 'sp_8', 'sp_9', 'sp_10']].astype(float).sum()
    elif row.l_length == 15.0:
        rec_pfr = row[['sp_12', 'sp_13', 'sp_14', 'sp_15']].astype(float).sum()
    elif row.l_length == 20.0:
        rec_pfr = row[['sp_17', 'sp_18', 'sp_19', 'sp_20']].astype(float).sum()
    elif row.l_length == 30.0:
        rec_pfr = row[['sp_27', 'sp_28', 'sp_29', 'sp_30']].astype(float).sum()
    else:
        rec_pfr = row[['sp_37', 'sp_38', 'sp_39', 'sp_40']].astype(float).sum()
        
    if prim_pfr >= 2/3:
        strat = 'prim'
    elif rec_pfr >= 2/3:
        strat = 'rec'
    else:
        strat = 'ns'
        
    return pd.Series({'subject': row.subject, 'condition': row.condition, 'l_length': row.l_length, 'pres_rate': row.pres_rate, 
                      'prim_pfr': prim_pfr, 'rec_pfr': rec_pfr, 'strat': strat})

# apply recall initiation group to dataframe
def apply_recall_initiation_labels(r1_groups, dataframe):        
    # dictionary mapping subject to group
    sub_strat = dict(zip(r1_groups.subject, r1_groups.strat))
    
    # add recall initiation group as field in dataframe
    dataframe['strategy'] = [sub_strat.get(row.worker_id) for _, row in dataframe.iterrows()]
    
    return dataframe


# serial position curve
def spc(df, condition_map):
    spc_data = []
    for (sub, strat, sess, c, ll, pr), data in tqdm(df.groupby(['worker_id', 'strategy', 'session', 'condition', 'l_length', 'pres_rate'])):
        spc = spc_sess(data, int(ll))
        spc_data.append((sub, strat, sess, condition_map.get(c), ll, pr,) + tuple(spc))
    
    # store results in dataframe
    cols = np.array(['subject', 'strategy', 'session', 'condition', 'l_length', 'pres_rate'] + [f'sp_{x}' for x in range(1, 41)])
    spc_data = pd.DataFrame(spc_data, columns=cols)
    
    return spc_data

def spc_btwn_subj_avg(spc_data):
    spc_data_bsa = []
    for (sub, strat, c, ll, pr), dat in tqdm(spc_data.groupby(['subject', 'strategy', 'condition', 'l_length', 'pres_rate'])):
        sp_precall = dat.iloc[:,6:].mean(axis=0).to_numpy()
        spc_data_bsa.append((sub, strat, c, ll, pr,) + tuple(sp_precall))
        
    # store results in dataframe
    cols = np.array(['subject', 'strategy', 'condition', 'l_length', 'pres_rate'] + [f'sp_{x}' for x in range(1, 41)])
    spc_data_bsa = pd.DataFrame(spc_data_bsa, columns=cols)
    
    # sort by condition
    spc_data_bsa = sort_by_condition(spc_data_bsa)

    return spc_data_bsa


# probability of first recall
def pfr(df, condition_map, toggle):
    pfr_data = []
    for (sub, strat, sess, c, ll, pr), data in tqdm(df.groupby(['worker_id', 'strategy', 'session', 'condition', 'l_length', 'pres_rate'])):
        pfr = pfr_sess(data, int(ll), toggle)
        pfr_data.append((sub, strat, sess, condition_map.get(c), ll, pr,) + tuple(pfr))
    
    # store results in dataframe
    cols = np.array(['subject', 'strategy', 'session', 'condition', 'l_length', 'pres_rate'] + [f'sp_{x}' for x in range(1, 41)])
    pfr_data = pd.DataFrame(pfr_data, columns=cols)
    
    return pfr_data

def pfr_btwn_subj_avg(pfr_data):
    pfr_data_bsa = []
    for (sub, strat, c, ll, pr), dat in tqdm(pfr_data.groupby(['subject', 'strategy', 'condition', 'l_length', 'pres_rate'])):
        sp_pfr = dat.iloc[:,6:].mean(axis=0).to_numpy()
        pfr_data_bsa.append((sub, strat, c, ll, pr,) + tuple(sp_pfr))
        
    # store results in dataframe
    cols = np.array(['subject', 'strategy', 'condition', 'l_length', 'pres_rate'] + [f'sp_{x}' for x in range(1, 41)])
    pfr_data_bsa = pd.DataFrame(pfr_data_bsa, columns=cols)

    # sort by condition
    pfr_data_bsa = sort_by_condition(pfr_data_bsa)

    return pfr_data_bsa


# mean words recalled
def mwr_sess(data):
    rec_evs = data[data['type'] == 'REC_WORD']
    list_recalls = []

    for i in data.list.unique():
        sp = rec_evs[rec_evs['list'] == i].serial_position.to_numpy()     # current list serial positions
        list_recalls.append(len(np.unique(sp[(sp != 99) & (sp != 88)])))  # number of correct recalls on list (no intrusions/repetitions)

    return np.mean(list_recalls)            # average over lists

def mwr(df, condition_map):
    mwr_data = []
    for (sub, strat, sess, c, ll, pr), data in tqdm(df.groupby(['worker_id', 'strategy', 'session', 'condition', 'l_length', 'pres_rate'])):
        mwr_ = mwr_sess(data)
        mwr_data.append((sub, strat, sess, condition_map.get(c), ll, pr, mwr_))
        
    # store results in dataframe
    mwr_data = pd.DataFrame(mwr_data, columns=['subject', 'strategy', 'session', 'condition', 'l_length', 'pres_rate', 'mwr'])
    
    return mwr_data

def mwr_btwn_subj_avg(mwr_data):
    mwr_data_bsa = []
    for (sub, strat, c, ll, pr), dat in tqdm(mwr_data.groupby(['subject', 'strategy', 'condition', 'l_length', 'pres_rate'])):
        mwr_data_bsa.append((sub, strat, c, ll, pr, np.mean(dat.mwr)))
        
    # store results in dataframe
    mwr_data_bsa = pd.DataFrame(mwr_data_bsa, columns=['subject', 'strategy', 'condition', 'l_length', 'pres_rate', 'mwr'])
    
    # sort by condition
    mwr_data_bsa = sort_by_condition(mwr_data_bsa)

    return mwr_data_bsa


# proportion of lists initiated with an intrusion
def r1_intrusion_sess(data):
    intr = 0                     # lists initiated with intrusion
    subtract = 0                 # lists with no recalls
    rec_evs = data[data['type'] == 'REC_WORD']
    
    for i in data.list.unique():
        sp = rec_evs[rec_evs['list'] == i].serial_position.to_numpy()
        
        if len(sp) > 1:          # always null recall at end of lust
            if sp[0] == 88:      # intrusion
                intr += 1
        else:
            subtract += 1
            
    return intr / (len(data.list.unique()) - subtract)

def r1_intrusion(df, condition_map):
    r1_intr_data = []
    for (sub, strat, sess, c, ll, pr), data in tqdm(df.groupby(['worker_id', 'strategy', 'session', 'condition', 'l_length', 'pres_rate'])):
        prop_wrong = r1_intrusion_sess(data)
        r1_intr_data.append((sub, strat, sess, condition_map.get(c), ll, pr, prop_wrong))
        
    # store results in dataframe
    r1_intr_data = pd.DataFrame(r1_intr_data, columns=['subject', 'strategy', 'session', 'condition', 'l_length', 'pres_rate', 'prop_wrong'])
    
    return r1_intr_data

def r1_intr_btwn_subj_avg(r1_intr_data):
    r1_intr_data_bsa = []
    for (sub, strat, c, ll, pr), dat in tqdm(r1_intr_data.groupby(['subject', 'strategy', 'condition', 'l_length', 'pres_rate'])):
        r1_intr_data_bsa.append((sub, strat, c, ll, pr, np.mean(dat.prop_wrong)))
        
    # store results in dataframe
    r1_intr_data_bsa = pd.DataFrame(r1_intr_data_bsa, columns=['subject', 'strategy', 'condition', 'l_length', 'pres_rate', 'prop_wrong'])
    
    # sort by condition
    r1_intr_data_bsa = sort_by_condition(r1_intr_data_bsa)

    return r1_intr_data_bsa


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
    p2r_intr_data_bsa = []
    for (sub, strat, c, ll, pr), dat in tqdm(p2r_intr_data.groupby(['subject', 'strategy', 'condition', 'l_length', 'pres_rate'])):
        sp_p2r = dat.iloc[:,6:].mean(axis=0).to_numpy()
        p2r_intr_data_bsa.append((sub, strat, c, ll, pr,) + tuple(sp_p2r))
        
    # store results in dataframe
    cols = np.array(['subject', 'strategy', 'condition', 'l_length', 'pres_rate', 'intrusion'] + [f'sp_{x}' for x in range(1, 41)])
    p2r_intr_data_bsa = pd.DataFrame(p2r_intr_data_bsa, columns=cols)
    
    # sort by condition
    p2r_intr_data_bsa = sort_by_condition(p2r_intr_data_bsa)

    return p2r_intr_data_bsa


# initial response times
# toggle = True, only trials initiated with correct recall
def rt_init_all_trials_sess(data, sub, strat, sess, c, ll, pr, toggle):
    rts = []                                       # initial response times
    rec_evs = data[data['type'] == 'REC_WORD']
    
    for i in data.list.unique():
        sp = rec_evs[rec_evs['list'] == i].serial_position.to_numpy()     # current list serial positions
        rt = rec_evs[rec_evs['list'] == i].rt.to_numpy()                  # current list response times
        if len(sp) > 1:                                                   # always null recall at end of list
            # only trials initiated with correct recall
            if toggle:
                if sp[0] > 0 and sp[0] <= ll:
                    rts.append((sub, strat, sess, c, ll, pr, sp[0], rt[0]))      
            # include R1 intrusions
            else:
                rts.append((sub, strat, sess, c, ll, pr, sp[0], rt[0]))       
            
    rts = pd.DataFrame(rts, columns=['subject', 'strategy', 'session', 'condition', 'l_length', 'pres_rate', 'serial_position', 'rt'])
    rts['rt'] = np.array(rts.rt) - np.min(rts.rt)                         # subtract (session) minimum to approximate typing time
    rts['min_rt'] = [True if x == np.min(rts.rt) else False for x in rts.rt]     # mark baseline correction minimum trial
    
    return rts

def rt_init_all_trials(df, condition_map, toggle):
    rti_at_data = []
    for (sub, strat, sess, c, ll, pr), data in tqdm(df.groupby(['worker_id', 'strategy', 'session', 'condition', 'l_length', 'pres_rate'])):
        rts = rt_init_all_trials_sess(data, sub, strat, sess, condition_map.get(c), ll, pr, toggle)
        rti_at_data.append(rts)
        
    # store results in dataframe
    rti_at_data = pd.concat(rti_at_data)

    return rti_at_data

# average of trials
# toggle = True, only lists initiated with correct recall
def rt_init_sess(data, ll, toggle):
    rts = []                                                              # initial response times
    rec_evs = data[data['type'] == 'REC_WORD']
    
    for i in data.list.unique():
        sp = rec_evs[rec_evs['list'] == i].serial_position.to_numpy()     # current list serial positions
        rt = rec_evs[rec_evs['list'] == i].rt.to_numpy()                  # current list response times
        if len(sp) > 1:                                                   # always null recall at end of list
            if toggle:                                                    # only include lists initiated with correct recall
                if sp[0] > 0 and sp[0] <= ll:
                    rts.append(rt[0])
            else:                                                         # include lists initiated with intrusion
                rts.append(rt[0])
                
    if len(rts) > 0:
        rts = np.array(rts) - np.min(rts)                                 # subtract (session) mimimum to approximate typing time
        return np.mean(rts)
    else:
        return np.nan

def rt_init(df, condition_map, toggle):
    rti_data = []
    for (sub, strat, sess, c, ll, pr), data in tqdm(df.groupby(['worker_id', 'strategy', 'session', 'condition', 'l_length', 'pres_rate'])):
        rt_mu = rt_init_sess(data, ll, toggle)
        rti_data.append((sub, strat, sess, condition_map.get(c), ll, pr, rt_mu))
        
    # store results in dataframe
    rti_data = pd.DataFrame(rti_data, columns=['subject', 'strategy', 'session', 'condition', 'l_length', 'pres_rate', 'rt'])
    
    return rti_data

def rti_btwn_subj_avg(rti_data):
    rti_data_bsa = []
    for (sub, strat, c, ll, pr), dat in tqdm(rti_data.groupby(['subject', 'strategy', 'condition', 'l_length', 'pres_rate'])):
        rti_data_bsa.append((sub, strat, c, ll, pr, np.mean(dat.rt)))
        
    # store results in dataframe
    rti_data_bsa = pd.DataFrame(rti_data_bsa, columns=['subject', 'strategy', 'condition', 'l_length', 'pres_rate', 'rt'])
    
    # sort by condition
    rti_data_bsa = sort_by_condition(rti_data_bsa)

    return rti_data_bsa

# intrusion rates (ELI, PLI)
# toggle = True, only lists initiated with correct recall
# only lists 4-11 (omit 0-3) for PLIs
def intrusion_rates_sess(data, toggle):
    tot_eli = 0; tot_pli = 0      # intrusion counters
    subtract = 4        # lists to not include for PLI rates
    
    word_evs = data[data['type'] == 'WORD']
    rec_evs = data[data['type'] == 'REC_WORD']
    wl_dict = dict(zip([w.upper() for w in word_evs.word], word_evs.list))      # dictionary mapping words to list

    for i in data.list.unique():
        sp = rec_evs[rec_evs['list'] == i].serial_position.to_numpy()    # current list serial positions
        sp = sp[sp != 99]                                                # remove last null recall
        r = rec_evs[rec_evs['list'] == i].rec_word.tolist()              # current list recalls
        r = r[:len(sp)]                                                  # remove last null recall
        
        sp = mark_repetitions(sp)         # change repetition serial position to 77
        
        # only lists initiated with correct recall
        if toggle and (len(sp) == 0 or sp[0] == 88):
            continue
        
        for j in range(len(sp)):
            if sp[j] == 88:                             # intrusion
                if r[j].upper() in wl_dict.keys():      # presented word
                    l = wl_dict.get(r[j].upper())       # list of word presentation
                    if l < i and i >= subtract:         # PLI
                        tot_pli += 1
                    elif l > i:                         # ELI (presented on future list)
                        tot_eli += 1
                else:
                    tot_eli += 1                        # ELI (not presented)
                    
    eli_rate = tot_eli / len(data.list.unique())
    pli_rate = tot_pli / len(data.list.unique() - subtract)
    
    return eli_rate, pli_rate

def intrusion_rates(df, condition_map, toggle):
    intr_data = []
    for (sub, strat, sess, c, ll, pr), data in tqdm(df.groupby(['worker_id', 'strategy', 'session', 'condition', 'l_length', 'pres_rate'])):
        eli_rate, pli_rate = intrusion_rates_sess(data, toggle)
        intr_data.append((sub, strat, sess, condition_map.get(c), ll, pr, eli_rate, pli_rate))
        
    # store results in dataframe
    intr_data = pd.DataFrame(intr_data, columns=['subject', 'strategy', 'session', 'condition', 'l_length', 'pres_rate', 
                                                 'eli_rate', 'pli_rate'])
    
    return intr_data

def intrusion_rates_btwn_subj_avg(intr_data):
    intr_data_bsa = []
    for (sub, strat, c, ll, pr), dat in tqdm(intr_data.groupby(['subject', 'strategy', 'condition', 'l_length', 'pres_rate'])):
        intr_data_bsa.append((sub, strat, c, ll, pr, np.mean(dat.eli_rate), np.mean(dat.pli_rate)))
        
    # store results in dataframe
    intr_data_bsa = pd.DataFrame(intr_data_bsa, columns=['subject', 'strategy', 'condition', 'l_length', 'pres_rate',
                                                         'eli_rate', 'pli_rate'])
    
    # sort by condition
    intr_data_bsa = sort_by_condition(intr_data_bsa)

    return intr_data_bsa


# inter-response times
# joint function of number of correct recalls and output position
def irt_sess(data, ll):
    irt = []               # irt data for each list
    max_ncr = 0            # maximum number of correct recalls on list
    rec_evs = data[data['type'] == 'REC_WORD']
    
    for i in data.list.unique():
        sp = rec_evs[rec_evs['list'] == i].serial_position.to_numpy()    # current list serial positions
        sp = sp[sp != 99]                                                # remove last null recall
        rt = rec_evs[rec_evs['list'] == i].rt.to_numpy()                 # current list response times
        rt = rt[:len(sp)]                                                # remove last null recall
        
        sp = mark_repetitions(sp)             # change repetitions to serial position 77
        
        irts = []
        for j in range(len(sp) - 1):
            # transition between correct recalls
            if sp[j]!=88 and sp[j]!=77 and sp[j+1]!=88 and sp[j+1]!=77:
                irts.append(rt[j+1])
            
            # transition from intrusion/repetition to correct recall (not first correct recall)
            elif (sp[j]==88 or sp[j]==77) and sp[j+1]!=88 and sp[j+1]!=77 and np.any((sp[:j] >= 1) & (sp[:j] <= ll)):
                irts.append(np.nan)
                
        if len(irts) > 0:
            irt.append((len(irts)+1,) + tuple(irts))
            if len(irts)+1 > max_ncr:                       # update maximum number of correct recalls on list
                max_ncr = len(irts)+1
            
    irt = pd.DataFrame(irt, columns=['ncr'] + [f'tr_{x}' for x in range(1, max_ncr)])

    return irt.groupby('ncr').mean().reset_index()           # average within session

def irt(df, condition_map):
    irt_data = []
    for (sub, strat, sess, c, ll, pr), data in tqdm(df.groupby(['worker_id', 'strategy', 'session', 'condition', 'l_length', 'pres_rate'])):
        irt_df = irt_sess(data, ll)
        cols = list(irt_df.columns)
        irt_df['subject'] = sub; irt_df['strategy'] = strat; irt_df['session'] = sess
        irt_df['condition'] = condition_map.get(c); irt_df['l_length'] = ll; irt_df['pres_rate'] = pr
        irt_df = irt_df[['subject', 'strategy', 'session', 'condition', 'l_length', 'pres_rate'] + cols]
        irt_data.append(irt_df)
        
    return pd.concat(irt_data, ignore_index=True)

def irt_btwn_subj_avg(irt_data):
    irt_data_bsa = []
    for (sub, strat, c, ll, pr, ncr), dat in tqdm(irt_data.groupby(['subject', 'strategy', 'condition', 'l_length', 'pres_rate', 'ncr'])):
        tr_irt = dat.iloc[:,7:].mean(axis=0).to_numpy()
        irt_data_bsa.append((sub, strat, c, ll, pr, ncr, ) + tuple(tr_irt))
        
    # store results in dataframe
    irt_data_bsa = pd.DataFrame(irt_data_bsa, columns=['subject', 'strategy', 'condition', 'l_length', 'pres_rate', 
                                                       'ncr'] + [f'tr_{x}' for x in range(1, len(irt_data.columns) - 6)])
    
    # sort by condition
    irt_data_bsa = sort_by_condition(irt_data_bsa)

    return irt_data_bsa

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

def irt_final_btwn_subj_avg(irt_final_data, mwr_md):
    irt_final_data_bsa = []
    for (sub, strat, c, ll, pr), dat in tqdm(irt_final_data.groupby(['subject', 'strategy', 'condition', 'l_length', 'pres_rate'])):
        rot_irt = dat.iloc[:,6:].mean(axis=0).to_numpy()
        irt_final_data_bsa.append((sub, strat, c, ll, pr,) + tuple(rot_irt))
        
    # store results in dataframe
    irt_final_data_bsa = pd.DataFrame(irt_final_data_bsa, columns=['subject', 'strategy', 'condition', 'l_length', 'pres_rate'] + [f'rot_{x}' for x in np.arange(max(mwr_md.lb)-1, -1, -1)])
    
    # sort by condition
    irt_final_data_bsa = sort_by_condition(irt_final_data_bsa)

    return irt_final_data_bsa

# total time between first and middle, middle and last correct recall
def irt_tot_sess(data):
    irt_tot = []                # irt data for each list
    rec_evs = data[data['type'] == 'REC_WORD']
    
    for i in data.list.unique():
        sp = rec_evs[rec_evs['list'] == i].serial_position.to_numpy()    # current list serial positions
        sp = sp[sp != 99]                                                # remove last null recall
        rt = rec_evs[rec_evs['list'] == i].rt.to_numpy()                 # current list response times
        rt = rt[:len(sp)]                                                # remove last null recall
        
        # omit trials containing an IRT > 30 seconds
        if len(rt) == 0 or max(rt) > 30000:
            continue
        
        sp = mark_repetitions(sp)             # change repetitions to serial position 77
        
        tot_rt = np.cumsum(rt)                                           # response time relative to start of recall period
        cr_idx = np.where((sp != 88) & (sp != 77))[0]                    # correct recalls
        if len(cr_idx) >= 3:                                             # at least 3 correct recalls
            irt = tot_rt[cr_idx[-1]] - tot_rt[cr_idx[0]]                 # time between first and last correct recall
            
            # even number of correct recalls (split on middle recall)
            if len(cr_idx) % 2 == 0:
                mi1 = len(cr_idx) // 2
                mi2 = mi1
            else:
                mi2 = len(cr_idx) // 2
                mi1 = mi2 - 1
            
            irt_h1 = tot_rt[cr_idx[mi1]] - tot_rt[cr_idx[0]]            # first half of correct recalls
            irt_h2 = tot_rt[cr_idx[-1]] - tot_rt[cr_idx[mi2]]           # 2nd half of correct recalls
                
            irt_tot.append((len(cr_idx), irt, irt_h1, irt_h2, irt_h2 - irt_h1))
            
    irt_tot = pd.DataFrame(irt_tot, columns=['ncr', 'total_irt', 'irt_h1', 'irt_h2', 'irt_delta'])

    return irt_tot.groupby('ncr').mean().reset_index()          # average within session

def irt_tot(df, condition_map):
    irt_tot_data = []
    for (sub, strat, sess, c, ll, pr), data in tqdm(df.groupby(['worker_id', 'strategy', 'session', 'condition', 'l_length', 'pres_rate'])):
        irt_tot_ = irt_tot_sess(data)
        irt_tot_['subject'] = sub; irt_tot_['strategy'] = strat; irt_tot_['session'] = sess
        irt_tot_['condition'] = condition_map.get(c); irt_tot_['l_length'] = ll; irt_tot_['pres_rate'] = pr
        irt_tot_ = irt_tot_[['subject', 'strategy', 'session', 'condition', 'l_length', 'pres_rate', 'ncr', 
                             'total_irt', 'irt_h1', 'irt_h2', 'irt_delta']]
        irt_tot_data.append(irt_tot_)

    return pd.concat(irt_tot_data, ignore_index=True)

def irt_tot_btwn_subj_avg(irt_tot_data):
    irt_tot_data_bsa = []
    for (sub, strat, c, ll, pr, ncr), dat in tqdm(irt_tot_data.groupby(['subject', 'strategy', 'condition', 'l_length', 'pres_rate', 'ncr'])):
        irt_tot_data_bsa.append((sub, strat, c, ll, pr, ncr, np.mean(dat.total_irt), 
                                 np.mean(dat.irt_h1), np.mean(dat.irt_h2), np.mean(dat.irt_delta)))

    # store results in dataframe
    irt_tot_data_bsa = pd.DataFrame(irt_tot_data_bsa, columns=['subject', 'strategy', 'condition', 'l_length', 'pres_rate', 'ncr', 
                                                               'total_irt', 'irt_h1', 'irt_h2', 'irt_delta'])
    
    # ncr bins = middle correct recall
    irt_tot_data_bsa['ncr_bin'] = irt_tot_data_bsa['ncr'] // 2

    # sort by condition
    irt_tot_data_bsa = sort_by_condition(irt_tot_data_bsa)
    
    return irt_tot_data_bsa


# temporal clustering score
# temporal percentile rank
def percentile_rank_T(actual, possible):
    if len(possible) < 2:
        return None

    # sort possible transitions from largest to smallest lag
    possible.sort(reverse=True)

    # get indices of the one or more possible transitions with the same lag as the actual transition
    matches = np.where(possible == actual)[0]

    if len(matches) > 0:
        # get number of posible lags that were more distance than actual lag
        rank = np.mean(matches)
        # convert rank to proportion of possible lags that were greater than actual lag
        ptile_rank = rank / (len(possible) - 1.0)
    else:
        ptile_rank = None

    return ptile_rank

def tcl_sess(data, ll, buffer):
    tcl = []                        # temporal clustering scores for each list
    rec_evs = data[data['type'] == 'REC_WORD']
                
    for i in data.list.unique():
        sp = rec_evs[rec_evs['list'] == i].serial_position.to_numpy()         # current list serial positions
        sp = sp[sp != 99]                                                     # remove last null recall

        sp = mark_repetitions(sp)        # repetitions = serial position 77
        
        # exclude first 'buffer' output positions
        excludeOutputs = sp[:buffer]
        sp = sp[buffer:]
        sps_left = [x for x in range(1, ll+1)]        # array with remaining serial positions, subtract from
        
        for exOut in excludeOutputs:
            try:
                sps_left.remove(exOut)               # remove first outputs from possible transitions
            except:
                pass                                  # item already removed or intrusion
        
        tcl_list = []                         # temporal clustering scores for each recall
        for j in range(len(sp) - 1):
            if sp[j]!=88 and sp[j]!=77:             # correct recall
                sps_left.remove(sp[j])
                if sp[j+1]!=88 and sp[j+1]!=77:     # transition to correct recall
                    possList = []
                    lag = abs(sp[j+1] - sp[j])      # actual transition lag
                    for l in range(len(sps_left)):
                        poss_lag = abs(sps_left[l] - sp[j])
                        possList.append(poss_lag)   # list includes actual lag

                    ptile_rank = percentile_rank_T(lag, possList)
                    if ptile_rank is not None:
                        tcl_list.append(ptile_rank)
        
        # take average of scores on list  ### weight each list or each transition equally???
        if len(tcl_list) > 0:
            tcl.append(np.mean(tcl_list))
            
    # return average of list scores
    return np.mean(tcl)

def tcl(df, condition_map, buffer):
    tcl_data = []
    for (sub, strat, sess, c, ll, pr), data in tqdm(df.groupby(['worker_id', 'strategy', 'session', 'condition', 'l_length', 'pres_rate'])):
        tcl_ = tcl_sess(data, int(ll), buffer)
        tcl_data.append((sub, strat, sess, condition_map.get(c), ll, pr, tcl_))
        
    # store results in dataframe
    tcl_data = pd.DataFrame(tcl_data, columns=['subject', 'strategy', 'session', 'condition', 'l_length', 'pres_rate', 'tcl'])
    
    return tcl_data

def tcl_btwn_subj_avg(tcl_data):
    tcl_data_bsa = []
    for (sub, strat, c, ll, pr), dat in tqdm(tcl_data.groupby(['subject', 'strategy', 'condition', 'l_length', 'pres_rate'])):
        tcl_data_bsa.append((sub, strat, c, ll, pr, np.mean(dat.tcl)))
        
    # store results in dataframe
    tcl_data_bsa = pd.DataFrame(tcl_data_bsa, columns=['subject', 'strategy', 'condition', 'l_length', 'pres_rate', 'tcl'])
    
    # sort by condition
    tcl_data_bsa = sort_by_condition(tcl_data_bsa)

    return tcl_data_bsa

# conditioned on half of recall
# conditioned on half of recall
def tcl_h_sess(data, ll):
    tcl_h1 = []; tcl_h2 = []; tcl_delta = []                     # temporal clustering scores for each list
    rec_evs = data[data['type'] == 'REC_WORD']
    
    for i in data.list.unique():
        sp = rec_evs[rec_evs['list'] == i].serial_position.to_numpy()      # current list serial positions
        sp = sp[sp != 99]                                                  # remove last null recall
        
        sp = mark_repetitions(sp)                      # repetitions = serial position 77
        
        cr_idx = np.where((sp != 88) & (sp != 77))[0]           # correct recall indices
        if len(cr_idx) % 2 == 1:
            mi1 = len(cr_idx) // 2
            mi2 = mi1
        else:
            mi2 = len(cr_idx) // 2
            mi1 = mi2 - 1
        
        tcl_h1_list = []; tcl_h2_list = []             # temporal clustering scores for each recall
        toggle_h1 = False; toggle_h2 = False
        sps_left = [x for x in range(1, ll+1)]
        for j in range(len(sp) - 1):
            # determine half of correct recalls
            if j < mi1:
                toggle_h1 = True
                toggle_h2 = False
            elif j >= mi2:
                toggle_h1 = False
                toggle_h2 = True
            else:
                toggle_h1 = False
                toggle_h2 = False
                
            if sp[j]!=88 and sp[j]!=77:                       # correct recall
                sps_left.remove(sp[j])
                if sp[j+1]!=88 and sp[j+1]!=77:               # transition to correct recall
                    possList = []
                    lag = abs(sp[j+1] - sp[j])                # actual transition lag
                    for l in range(len(sps_left)):
                        poss_lag = abs(sps_left[l] - sp[j])
                        possList.append(poss_lag)             # list includes actual lag
                        
                    ptile_rank = percentile_rank_T(lag, possList)
                    if ptile_rank is not None:
                        if toggle_h1:
                            tcl_h1_list.append(ptile_rank)
                        elif toggle_h2:
                            tcl_h2_list.append(ptile_rank)
                            
        # take average of scofres on list (weight each list equally)
        if len(tcl_h1_list) > 0 and len(tcl_h2_list) > 0:
            tcl_h1.append(np.mean(tcl_h1_list))
            tcl_h2.append(np.mean(tcl_h2_list))
            tcl_delta.append(np.mean(tcl_h2_list) - np.mean(tcl_h1_list))
            
    # return average of list scores
    return np.mean(tcl_h1), np.mean(tcl_h2), np.mean(tcl_delta)

def tcl_h(df, condition_map):
    tcl_h_data = []
    for (sub, strat, sess, c, ll, pr), data in tqdm(df.groupby(['worker_id', 'strategy', 'session', 'condition', 'l_length', 'pres_rate'])):
        tcl_h1, tcl_h2, tcl_delta = tcl_h_sess(data, int(ll))
        tcl_h_data.append((sub, strat, sess, condition_map.get(c), ll, pr, tcl_h1, tcl_h2, tcl_delta))
        
    # store results in dataframe
    tcl_h_data = pd.DataFrame(tcl_h_data, columns=['subject', 'strategy', 'session', 'condition', 'l_length', 'pres_rate', 'tcl_h1', 'tcl_h2', 'tcl_delta'])
    
    return tcl_h_data

def tcl_h_btwn_subj_avg(tcl_h_data):
    tcl_h_data_bsa = []
    for (sub, strat, c, ll, pr), dat in tqdm(tcl_h_data.groupby(['subject', 'strategy', 'condition', 'l_length', 'pres_rate'])):
        tcl_h_data_bsa.append((sub, strat, c, ll, pr, np.mean(dat.tcl_h1), np.mean(dat.tcl_h2), np.mean(dat.tcl_delta)))
        
    # store results in dataframe
    tcl_h_data_bsa = pd.DataFrame(tcl_h_data_bsa, columns=['subject', 'strategy', 'condition', 'l_length', 'pres_rate', 'tcl_h1', 'tcl_h2', 'tcl_delta'])
    
    # sort by condition
    tcl_h_data_bsa = sort_by_condition(tcl_h_data_bsa)

    return tcl_h_data_bsa


# lag-CRP
def lag_crp_sess(data, ll, buffer):
    ac = np.zeros(2*ll - 1)
    poss = np.zeros_like(ac)
    rec_evs = data[data['type'] == 'REC_WORD']
    
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
                    ac[int(lag)+ll-1] += 1
                    for l in sps_left:                # find all possible transitions
                        p_lag = l - sp[j]
                        poss[int(p_lag)+ll-1] += 1
    
    crp = ac / poss       # we actually want the NaNs from zero division
    crp = np.delete(crp, len(crp)//2)        # remove lag = 0
    
    return crp

def lag_crp(df, condition_map, buffer):
    lcrp_data = []
    for (sub, strat, sess, c, ll, pr), data in tqdm(df.groupby(['worker_id', 'strategy', 'session', 'condition', 'l_length', 'pres_rate'])):
        crp = lag_crp_sess(data, int(ll), buffer)
        lcrp = [(sub, strat, sess, condition_map.get(c), ll, pr,) + tuple(crp)]
        lcrp = pd.DataFrame(lcrp, columns=['subject', 'strategy', 'session', 'condition', 'l_length', 'pres_rate'] + 
                            [f'ln_{x}' for x in np.arange(int(ll)-1, 0, -1)] + [f'lp_{x}' for x in range(1, int(ll))])
        lcrp_data.append(lcrp)
        
    # store results in dataframe
    lcrp_data = pd.concat(lcrp_data)
    lcrp_data = lcrp_data[['subject', 'strategy', 'session', 'condition', 'l_length', 'pres_rate'] + \
                          [f'ln_{x}' for x in np.arange(39, 0, -1)] + [f'lp_{x}' for x in range(1, 40)]]
    
    return lcrp_data

def lag_crp_btwn_subj_avg(lcrp_data):
    lcrp_data_bsa = []
    for (sub, strat, c, ll, pr), dat in tqdm(lcrp_data.groupby(['subject', 'strategy', 'condition', 'l_length', 'pres_rate'])):
        lcrp = dat.iloc[:, 6:].mean(axis=0).to_numpy()
        lcrp_data_bsa.append((sub, strat, c, ll, pr,) + tuple(lcrp))
        
    # store results in dataframe
    cols = ['subject', 'strategy', 'condition', 'l_length', 'pres_rate'] + [f'ln_{x}' for x in np.arange(39, 0, -1)] + [f'lp_{x}' for x in range(1, 40)]
    lcrp_data_bsa = pd.DataFrame(lcrp_data_bsa, columns=cols)
    
    # sort by condition
    lcrp_data_bsa = sort_by_condition(lcrp_data_bsa)

    return lcrp_data_bsa

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
    lcrp_sp_data_bsa = []
    for (sub, strat, c, ll, pr, sp), dat in tqdm(lcrp_sp_data.groupby(['subject', 'strategy', 'condition', 'l_length', 'pres_rate', 'serial_position'])):
        crp = dat.iloc[:, 7:].mean(axis=0).to_numpy()
        lcrp_sp_data_bsa.append((sub, strat, c, ll, pr, sp,) + tuple(crp))
        
     # store results in dataframe
    lcrp_sp_data_bsa = pd.DataFrame(lcrp_sp_data_bsa, columns=['subject', 'strategy', 'condition', 'l_length', 'pres_rate', 'serial_position'] + \
                                                              [f'ln_{x}' for x in np.arange(39, 0, -1)] + [f'lp_{x}' for x in range(1, 40)])

    # sort by condition
    lcrp_sp_data_bsa = sort_by_condition(lcrp_sp_data_bsa)

    return lcrp_sp_data_bsa


# semantic clustering score
# semantic percentile rank
def percentile_rank_S(actual, possible):
    if len(possible) < 2:
        return None

    # sort possible transitions from lowest to highest similarity
    possible.sort()

    # get indices of possible transitions with same similarity as actual transition
    matches = np.where(possible == actual)[0]
    if len(matches) > 0:
        # get number of possible transition that were less similar than the actual transition
        rank = np.mean(matches)
        # convert rank to proportion of possible transitions that were less similar than the actual transition
        ptile_rank = rank / (len(possible) - 1.0)
    else:
        ptile_rank = None

    return ptile_rank

# semantic clustering score single session
def scl_sess(data, buffer, wordpool, w2v_scores):
    scl = []                 # semantic clustering scores for eacl list
    word_evs = data[data['type'] == 'WORD']; rec_evs = data[data['type'] == 'REC_WORD']
    
    for i in data.list.unique():
        sp = rec_evs[rec_evs['list'] == i].serial_position.to_numpy()         # current list serial positions
        sp = sp[sp != 99]                                                     # remove last null recall
        w = word_evs[word_evs['list'] == i].word.tolist()                     # current list words
        r = rec_evs[rec_evs['list'] == i].rec_word.tolist()                   # current list recalls
        r = r[:len(sp)]                                                       # remove last null recall

        sp = mark_repetitions(sp)                     # repetitions = serial position 77
        
        # exclude first 'buffer' output words
        excludeOutputs = r[:buffer]
        r = r[buffer:]
        sp = sp[buffer:]
        words_left = [x.upper() for x in w]           # array with remaining words, subtract from
        
        for exOut in excludeOutputs:
            try:
                words_left.remove(exOut.upper())      # remove first outputs from possible transitions
            except:
                pass                                  # item already removed or intrusion
        
        scl_list = []                                 # semantic clustering scores for each recall
        for j in range(len(sp) - 1):
            if sp[j]!=88 and sp[j]!=77:               # correct recall
                words_left.remove(r[j].upper())       # can't transition to already recalled serial position
                if sp[j+1]!=88 and sp[j+1]!=77:
                    if r[j].upper() in wordpool and r[j+1].upper() in wordpool:
                        possList = []
                        wv1 = wordpool.index(r[j].upper())
                        wv2 = wordpool.index(r[j+1].upper())
                        ss = w2v_scores[wv1][wv2]                           # actual transition semantic similarity
                        for l in range(len(words_left)):                    # possible transition semantic similarities
                            wv3 = wordpool.index(words_left[l].upper())
                            poss_ss = w2v_scores[wv1][wv3]
                            possList.append(poss_ss)                        # list includes actual transition
                            
                            ptile_rank = percentile_rank_S(ss, possList)
                            if ptile_rank is not None:
                                scl_list.append(ptile_rank)
                                
        # take avearge of scores on list   ### weight each list or each transition equally?
        if len(scl_list) > 0:
            scl.append(np.mean(scl_list))
            
    # return average of list scores
    return np.mean(scl)

def scl(df, condition_map, buffer, wordpool, w2v_scores):
    scl_data = []
    for (sub, strat, sess, c, ll, pr), data in tqdm(df.groupby(['worker_id', 'strategy', 'session', 'condition', 'l_length', 'pres_rate'])):
        scl_ = scl_sess(data, buffer, wordpool, w2v_scores)
        scl_data.append((sub, strat, sess, condition_map.get(c), ll, pr, scl_))
        
    # store results in dataframe
    scl_data = pd.DataFrame(scl_data, columns=['subject', 'strategy', 'session', 'condition', 'l_length', 'pres_rate', 'scl'])
    
    return scl_data

def scl_btwn_subj_avg(scl_data):
    scl_data_bsa = []
    for (sub, strat, c, ll, pr), dat in tqdm(scl_data.groupby(['subject', 'strategy', 'condition', 'l_length', 'pres_rate'])):
        scl_data_bsa.append((sub, strat, c, ll, pr, np.mean(dat.scl)))
        
    # store results in dataframe
    scl_data_bsa = pd.DataFrame(scl_data_bsa, columns=['subject', 'strategy', 'condition', 'l_length', 'pres_rate', 'scl'])
    
    # sort by condition
    scl_data_bsa = sort_by_condition(scl_data_bsa)

    return scl_data_bsa


# ---------- Other Group Analyses ----------
# compare trials where recall is initiated with a primacy v. recency item

# mean words recalled
def mwr_ns_sess(data, ll):
    rec_evs = data[data['type'] == 'REC_WORD']
    list_recalls_prim = []
    list_recalls_rec = []
    
    for i in data.list.unique():
        sp = rec_evs[rec_evs['list'] == i].serial_position.to_numpy()     # current list serial positions
        
        # initiate recall with primacy item
        if sp[0] >= 1 and sp[0] <= 4:
            list_recalls_prim.append(len(np.unique(sp[(sp != 99) & (sp != 88)])))
        
        # initiate recall with recency item
        elif sp[0] >= ll-3 and sp[0] <= ll:
            list_recalls_rec.append(len(np.unique(sp[(sp != 99) & (sp != 88)])))
            
    return np.mean(list_recalls_prim), np.mean(list_recalls_rec)      # returns NaN if no data

def mwr_ns(df, condition_map):
    df_ns = df[df.strategy == 'ns']    # only no strategy group
    
    mwr_ns_data = []
    for (sub, strat, sess, c, ll, pr), data in tqdm(df_ns.groupby(['worker_id', 'strategy', 'session', 'condition', 'l_length', 'pres_rate'])):
        mwr_prim, mwr_rec = mwr_ns_sess(data, ll)
        mwr_ns_data.append((sub, strat, sess, condition_map.get(c), ll, pr, mwr_prim, mwr_rec))
        
    # store results in dataframe
    mwr_ns_data = pd.DataFrame(mwr_ns_data, columns=['subject', 'strategy', 'session', 'condition', 'l_length', 'pres_rate', 'mwr_prim', 'mwr_rec'])
    
    return mwr_ns_data

def mwr_ns_btwn_subj_avg(mwr_ns_data):
    mwr_ns_data_bsa = []
    for (sub, strat, c, ll, pr), dat in tqdm(mwr_ns_data.groupby(['subject', 'strategy', 'condition', 'l_length', 'pres_rate'])):
        mwr_ns_data_bsa.append((sub, strat, c, ll, pr, np.mean(dat.mwr_prim), np.mean(dat.mwr_rec)))   # omits NaN even without nanmean for sessions with no data
        
    # store results in dataframe
    mwr_ns_data_bsa = pd.DataFrame(mwr_ns_data_bsa, columns=['subject', 'strategy', 'condition', 'l_length', 'pres_rate', 'mwr_prim', 'mwr_rec'])
    
    # sort by condition
    mwr_ns_data_bsa = sort_by_condition(mwr_ns_data_bsa)

    return mwr_ns_data_bsa


# initial response times
def rt_init_ns(rti_at_data):
    rti_at_ns_data = rti_at_data.query("strategy == 'ns'")    # only no strategy group

    rti_ns_data = []
    for (sub, strat, sess, c, ll, pr), data in tqdm(rti_at_ns_data.groupby(['subject', 'strategy', 'session', 'condition', 'l_length', 'pres_rate'])):
        # initiate recall with primacy item
        data_prim = data[(data.serial_position >= 1) & (data.serial_position <= 4)]

        # initiate recall with recency item
        data_rec = data[(data.serial_position >= ll-3) & (data.serial_position <= ll)]

        rti_prim = np.mean(data_prim.rt) if len(data_prim) > 0 else np.nan
        rti_rec = np.mean(data_rec.rt) if len(data_rec) > 0 else np.nan

        rti_ns_data.append((sub, strat, sess, c, ll, pr, rti_prim, rti_rec))

    # store results in dataframe
    rti_ns_data = pd.DataFrame(rti_ns_data, columns=['subject', 'strategy', 'session', 'condition' , 'l_length', 'pres_rate',
                                                     'rt_prim', 'rt_rec'])
    
    return rti_ns_data

def rti_ns_btwn_subj_avg(rti_ns_data):
    rti_ns_data_bsa = []
    for (sub, strat, c, ll, pr), dat in tqdm(rti_ns_data.groupby(['subject', 'strategy', 'condition', 'l_length', 'pres_rate'])):
        rti_ns_data_bsa.append((sub, strat, c, ll, pr, np.mean(dat.rt_prim), np.mean(dat.rt_rec)))
        
    # store results in dataframe
    rti_ns_data_bsa = pd.DataFrame(rti_ns_data_bsa, columns=['subject', 'strategy', 'condition', 'l_length', 'pres_rate', 'rt_prim', 'rt_rec'])
    
    # sort by condition
    rti_ns_data_bsa = sort_by_condition(rti_ns_data_bsa)

    return rti_ns_data_bsa


# temporal clustering score
def tcl_ns_sess(data, ll):
    tcl_prim = []
    tcl_rec = []
    rec_evs = data[data['type'] == 'REC_WORD']
    
    for i in data.list.unique():
        sp = rec_evs[rec_evs['list'] == i].serial_position.to_numpy()         # current list serial positions
        sp = sp[sp != 99]                                                     # remove last null recall

        sp = mark_repetitions(sp)        # repetitions = serial position 77
        
        tcl_list = []                    # temporal clustering scores for each recall
        sps_left = [x for x in range(1, ll+1)]
        for j in range(len(sp) - 1):
            if sp[j]!=88 and sp[j]!=77:               # correct recall
                sps_left.remove(sp[j])
                if sp[j+1]!=88 and sp[j+1]!=77:       # transition to correct recall
                    possList = []
                    lag = abs(sp[j+1] - sp[j])        # actual transition lag
                    for l in range(len(sps_left)):
                        poss_lag = abs(sps_left[l] - sp[j])
                        possList.append(poss_lag)     # list includes actual lag
                        
                    ptile_rank = percentile_rank_T(lag, possList)
                    if ptile_rank is not None:
                        tcl_list.append(ptile_rank)
                        
        # take average of scores on list
        if len(tcl_list) > 0:
            try:
                # initiate recall with primacy item
                if sp[0] >= 1 and sp[0] <= 4:
                    tcl_prim.append(np.mean(tcl_list))

                # initiate recall with recency item
                if sp[0] >= ll-3 and sp[0] <= ll:
                    tcl_rec.append(np.mean(tcl_list))
                    
            except IndexError:                         # no recalls on list
                continue
                
    # return average of list scores
    return np.mean(tcl_prim), np.mean(tcl_rec)         # returns NaN if no data

def tcl_ns(df, condition_map):
    df_ns = df[df.strategy == 'ns']    # only no strategy group
    
    tcl_ns_data = []
    for (sub, strat, sess, c, ll, pr), data in tqdm(df_ns.groupby(['worker_id', 'strategy', 'session', 'condition', 'l_length', 'pres_rate'])):
        tcl_prim, tcl_rec = tcl_ns_sess(data, int(ll))
        tcl_ns_data.append((sub, strat, sess, condition_map.get(c), ll, pr, tcl_prim, tcl_rec))
        
    # store results in dataframe
    tcl_ns_data = pd.DataFrame(tcl_ns_data, columns=['subject', 'strategy', 'session', 'condition', 'l_length', 'pres_rate', 'tcl_prim', 'tcl_rec'])
    
    return tcl_ns_data

def tcl_ns_btwn_subj_avg(tcl_ns_data):
    tcl_ns_data_bsa = []
    for (sub, strat, c, ll, pr), dat in tqdm(tcl_ns_data.groupby(['subject', 'strategy', 'condition', 'l_length', 'pres_rate'])):
        tcl_ns_data_bsa.append((sub, strat, c, ll, pr, np.mean(dat.tcl_prim), np.mean(dat.tcl_rec)))
        
    # store results in dataframe
    tcl_ns_data_bsa = pd.DataFrame(tcl_ns_data_bsa, columns=['subject', 'strategy', 'condition', 'l_length', 'pres_rate', 'tcl_prim', 'tcl_rec'])
    
    # sort by condition
    tcl_ns_data_bsa = sort_by_condition(tcl_ns_data_bsa)

    return tcl_ns_data_bsa