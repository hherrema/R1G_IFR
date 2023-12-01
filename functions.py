### analysis functions for murdock-cmr
### functions should take full dataframe as argument, return dictionary mapping condition to results
### separate function for plotting, takes dictionary as argument
### treat subjects, not sessions, as independent
# --------------------------------------------------------------------------------------------------------- #

# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats

# ---------- Utility ---------- 

# access condition data
def condition_metadata(df, c):
    cond_data = df[df['condition'] == c]
    list_length = int(cond_data.l_length.unique()[0])
    presentation_rate = int(cond_data.pres_rate.unique()[0]/1000)
    cond_str = str(list_length) + '-' + str(presentation_rate)
    
    return cond_data, list_length, presentation_rate, cond_str


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

# ---------- Mean Words Recalled ----------

def MWR(df):
    mwr_dict = {}                           # dictionary mapping condition to mean words recalled
    se_dict = {}                            # dictionary mapping condition to standard error
    sub_dict = {}                           # dictionary mapping condition to subject scores
    
    # iterate over conditions
    for c in df.condition.unique():
        cond_data, list_length, presentation_rate, cond_str = condition_metadata(df, c)
        
        sub_recalls = []
        # iterate over subjects within condition
        for n in cond_data.worker_id.unique():
            sub_data = cond_data[cond_data['worker_id'] == n]
            
            sess_recalls = []
            # iterate over subject's sessions
            for s in sub_data.session.unique():
                sess_data = sub_data[sub_data['session'] == s]
                rec_evs = sess_data[sess_data['type'] == 'REC_WORD']
                list_recalls = []
                
                for i in sess_data.list.unique():
                    sp = rec_evs[rec_evs['list'] == i].serial_position.to_numpy()     # current list serial positions
                    list_recalls.append(len(np.unique(sp[(sp != 99) & (sp != 88)])))  # number of correct recalls on list
                
                
                sess_recalls.append(np.mean(list_recalls))            # average over lists
                
            sub_recalls.append(np.mean(sess_recalls))                 # average over sessions
        
        # calculate mean and standard error across subjects
        mwr_dict.update({cond_str: np.mean(sub_recalls)})
        se_dict.update({cond_str: np.std(sub_recalls, ddof=1)/(np.sqrt(len(sub_recalls)))})
        sub_dict.update({cond_str: sub_recalls})
        
    return mwr_dict, se_dict, sub_dict

def plot_MWR(mwr_dict, se_dict):
    conds = ['10-2', '20-1', '15-2', '30-1', '20-2', '40-1']
    colors = ['firebrick', 'darkorange', 'forestgreen', 'dodgerblue', 'midnightblue', 'darkviolet']
    mwr = []
    se = []
    for c in conds:
        mwr.append(mwr_dict.get(c))
        se.append(se_dict.get(c))
    
    fig, ax = plt.subplots(figsize=(7,5))
    ax.bar(conds, mwr, yerr=se, width=0.65, color=colors)
    ax.set(xlabel='Condition', ylabel='Mean Words Recalled')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()

def compare_MWR(mwr_prim_dict, mwr_prim_se, mwr_prim_sub, mwr_ns_dict, mwr_ns_se, mwr_ns_sub, mwr_rec_dict, mwr_rec_se, mwr_rec_sub):
    conds = ['10-2', '20-1', '15-2', '30-1', '20-2', '40-1']
    fig, ax = plt.subplots(2, 3, figsize=(9, 6), sharey=True, sharex=True)
    for i, c in enumerate(conds):
        mwr_prim = mwr_prim_dict.get(c)
        se_prim = mwr_prim_se.get(c)
        sub_prim = mwr_prim_sub.get(c)
        
        mwr_ns = mwr_ns_dict.get(c)
        se_ns = mwr_ns_se.get(c)
        sub_ns = mwr_ns_sub.get(c)

        mwr_rec = mwr_rec_dict.get(c)
        se_rec = mwr_rec_se.get(c)
        sub_rec = mwr_rec_sub.get(c)

        #ts, pvals = scipy.stats.ttest_ind(sub_prim, sub_rec, equal_var=False, alternative='two-sided')

        ax[i%2, i//2].bar([1,2,3], [mwr_prim, mwr_ns, mwr_rec], yerr=[se_prim, se_ns, se_rec], width=0.65, color=['orange', 'darkgray', 'purple'])
        ax[i%2, i//2].set(xticks=[1,2,3], xticklabels=['Primacy', 'No Strategy', 'Recency'], title=conds[i])
        ax[i%2, i//2].spines['right'].set_visible(False)
        ax[i%2, i//2].spines['top'].set_visible(False)

    #fig.supxlabel('Group')
    fig.supylabel('Mean Words Recalled')
    ax[1, 1].set(xlabel='Group')
    plt.tight_layout()
    plt.show()
    
def compare_MWR_stats(mwr_prim_dict, mwr_prim_se, mwr_prim_sub, mwr_ns_dict, mwr_ns_se, mwr_ns_sub, mwr_rec_dict, mwr_rec_se, mwr_rec_sub):
    conds = ['10-2', '20-1', '15-2', '30-1', '20-2', '40-1']
    fig, ax = plt.subplots(2, 3, figsize=(9, 6), sharey=True, sharex=True)
    for i, c in enumerate(conds):
        mwr_prim = mwr_prim_dict.get(c)
        se_prim = mwr_prim_se.get(c)
        sub_prim = mwr_prim_sub.get(c)
        
        mwr_ns = mwr_ns_dict.get(c)
        se_ns = mwr_ns_se.get(c)
        sub_ns = mwr_ns_sub.get(c)

        mwr_rec = mwr_rec_dict.get(c)
        se_rec = mwr_rec_se.get(c)
        sub_rec = mwr_rec_sub.get(c)

        ts_prim_rec, pval_prim_rec = scipy.stats.ttest_ind(sub_prim, sub_rec, equal_var=False, alternative='two-sided')
        ts_prim_ns, pval_prim_ns = scipy.stats.ttest_ind(sub_prim, sub_ns, equal_var=False, alternative='two-sided')
        ts_ns_rec, pval_ns_rec = scipy.stats.ttest_ind(sub_ns, sub_rec, equal_var=False, alternative='two-sided')
        
        h1 = np.max(np.concatenate((sub_prim, sub_ns, sub_rec))) + 2
        h2 = h1 + 2
        
        if pval_prim_rec < 0.05:
            ax[i%2, i//2].plot([1,3], [h2, h2], color='midnightblue')
        if pval_prim_ns < 0.05:
            ax[i%2, i//2].plot([1.1,1.9], [h1, h1], color='saddlebrown')
        if pval_ns_rec < 0.05:
            ax[i%2, i//2].plot([2.1,2.9], [h1, h1], color='saddlebrown')

        ax[i%2, i//2].bar([1,2,3], [mwr_prim, mwr_ns, mwr_rec], yerr=[1.96*se_prim, 1.96*se_ns, 1.96*se_rec], width=0.65, color=['orange', 'darkgray', 'purple'], alpha=0.6)
        ax[i%2, i//2].scatter(np.full(len(sub_prim), 1), sub_prim, s=5, color='orange', alpha=0.9)
        ax[i%2, i//2].scatter(np.full(len(sub_ns), 2), sub_ns, s=5, color='darkgray', alpha=0.9)
        ax[i%2, i//2].scatter(np.full(len(sub_rec), 3), sub_rec, s=5, color='purple', alpha=0.9)
        ax[i%2, i//2].set(xticks=[1,2,3], xticklabels=['Primacy', 'No Strategy', 'Recency'], title=conds[i])
        ax[i%2, i//2].spines['right'].set_visible(False)
        ax[i%2, i//2].spines['top'].set_visible(False)

    #fig.supxlabel('Group')
    fig.supylabel('Mean Words Recalled')
    ax[1, 1].set(xlabel='Group')
    plt.tight_layout()
    plt.show()
    
def MWR_fst(df,  strat):
    mwr_dict = {}                          # dictionary mapping condition to [mwr_strat, mwr_other]
    se_dict = {}                           # dictionary mapping condition to standard error
    sub_dict = {}                          # dictionary mapping condition to subject scores
    
    # iterate over conditions
    for c in df.condition.unique():
        cond_data, list_length, presentation_rate, cond_str = condition_metadata(df, c)
        
        sub_recalls_strat = []; sub_recalls_other = []
        # iterate over subjects within condition
        for n in cond_data.worker_id.unique():
            sub_data = cond_data[cond_data['worker_id'] == n]
            
            sess_recalls_strat = []; sess_recalls_other = []
            # iterate over subject's sessions
            for s in sub_data.session.unique():
                sess_data = sub_data[sub_data['session'] == s]
                rec_evs = sess_data[sess_data['type'] == 'REC_WORD']
                list_recalls_strat = []; list_recalls_other = []
                
                for i in sess_data.list.unique():
                    sp = rec_evs[rec_evs['list'] ==i].serial_position.to_numpy()        # current list serial positions
                    nrecs = len(np.unique(sp[(sp != 99) & (sp != 88)]))                 # number of correct recalls on list
                    if nrecs > 0:
                        if strat == 'prim' and sp[0] <= 4:
                            list_recalls_strat.append(nrecs)
                        elif strat == 'prim' and sp[0] > 4 and sp[0] <= list_length:
                            #print('other list')
                            list_recalls_other.append(nrecs)
                        elif strat == 'rec' and sp[0] > list_length - 4 and sp[0] <= list_length:
                            list_recalls_strat.append(nrecs)
                        elif strat == 'rec' and sp[0] <= list_length - 4:
                            list_recalls_other.append(nrecs)
                
                if len(list_recalls_strat) > 0:
                    sess_recalls_strat.append(np.mean(list_recalls_strat))          # average over lists
                if len(list_recalls_other) > 0:
                    sess_recalls_other.append(np.mean(list_recalls_other))
                    
            if len(sess_recalls_strat) > 0:
                sub_recalls_strat.append(np.mean(sess_recalls_strat))                   # average over sessions
            else: 
                sub_recalls_strat.append(np.nan)
            if len(sess_recalls_other) > 0:
                sub_recalls_other.append(np.mean(sess_recalls_other))
            else:
                sub_recalls_other.append(np.nan)
        
        # calculate mean and standard error across subjects
        mwr_dict.update({cond_str: [np.nanmean(sub_recalls_strat), np.nanmean(sub_recalls_other)]})
        se_dict.update({cond_str: [np.nanstd(sub_recalls_strat, ddof=1) / np.sqrt(np.count_nonzero(~np.isnan(sub_recalls_strat))), 
                                   np.nanstd(sub_recalls_other, ddof=1) / np.sqrt(np.count_nonzero(~np.isnan(sub_recalls_other)))]})
        sub_dict.update({cond_str: [sub_recalls_strat, sub_recalls_other]})
        
    return mwr_dict, se_dict, sub_dict

def plot_MWR_fst_stats(mwr_fst_prim_sub, mwr_fst_rec_sub):
    conds = ['10-2', '20-1', '15-2', '30-1', '20-2', '40-1']
    fig, ax = plt.subplots(2, 3, figsize=(10,6), sharey=True, sharex=True)
    for i, c in enumerate(conds):
        sub_prim = mwr_fst_prim_sub.get(c)
        sub_prim_diff = np.array(sub_prim[0]) - np.array(sub_prim[1])               # will include the NaN values
        mean_prim_diff = np.nanmean(sub_prim_diff)
        se_prim_diff = np.nanstd(sub_prim_diff, ddof=1) / np.sqrt(np.count_nonzero(~np.isnan(sub_prim_diff)))

        sub_rec = mwr_fst_rec_sub.get(c)
        sub_rec_diff = np.array(sub_rec[0]) - np.array(sub_rec[1])
        mean_rec_diff = np.nanmean(sub_rec_diff)
        se_rec_diff = np.nanstd(sub_rec_diff, ddof=1) / np.sqrt(np.count_nonzero(~np.isnan(sub_rec_diff)))

        # two-sample t-tests between strategy groups
        #ts_prim_rec, pval_prim_rec = scipy.stats.ttest_ind(sub_prim_diff, sub_rec_diff, nan_policy='omit', equal_var=False, alternative='two-sided')

        # onse sample t-test against no difference
        os_ts_prim, os_pval_prim = scipy.stats.ttest_1samp(sub_prim_diff, 0, nan_policy='omit', alternative='greater')
        os_ts_rec, os_pval_rec = scipy.stats.ttest_1samp(sub_rec_diff, 0, nan_policy='omit', alternative='greater')

        #print(pval_prim_rec, os_pval_prim, os_pval_rec)

        h1 = np.nanmax(np.concatenate((sub_prim_diff, sub_rec_diff))) + 3
        #h1 = np.nanmax([mean_prim_diff+1.96*se_prim_diff, mean_rec_diff+1.96*se_rec_diff]) + 0.3
        #h2 = h1 + 0.1

        #if pval_prim_rec < 0.05:
        #    ax[i%2, i//2].plot([1, 2], [h1, h1], color='midnightblue')

        if os_pval_prim < 0.05:
            ax[i%2, i//2].plot([1], [h1], marker='*', color='orange')
        if os_pval_rec < 0.05:
            ax[i%2, i//2].plot([2], [h1], marker='*', color='purple')

        ax[i%2, i//2].bar([1, 2], [mean_prim_diff, mean_rec_diff], yerr=[1.96*se_prim_diff, 1.96*se_rec_diff], width=0.65, 
                         color=['orange', 'purple'], alpha=0.6)
        ax[i%2, i//2].scatter(np.full(len(sub_prim_diff), 1), sub_prim_diff, s=5, color='orange', alpha=0.9)
        ax[i%2, i//2].scatter(np.full(len(sub_rec_diff), 2), sub_rec_diff, s=5, color='purple', alpha=0.9)
        ax[i%2, i//2].set(xticks=[1,2], xticklabels=['Primacy', 'Recency'], title=c)
        ax[i%2, i//2].spines['right'].set_visible(False)
        ax[i%2, i//2].spines['top'].set_visible(False)

    fig.supylabel('Difference in Mean Words Recalled: Stragey - Other')
    plt.tight_layout()
    plt.show()
    
def MWR_sess(df):
    mwr_dict = {}                  # dictionary mapping condition to mean words recalled for each session
    se_dict = {}                   # dictionary mapping condition to standard error for each session
    sub_dict = {}                  # dictionary mapping condition to subject scores for each session
    
    # iterate over conditions
    for c in df.condition.unique():
        cond_data, list_length, presentation_rate, cond_str = condition_metadata(df, c)
        
        mwr_sess_mat = np.full((len(cond_data.worker_id.unique()), len(cond_data.session.unique())), np.nan, dtype = float)    # matrix for individual subject scores for each session
        # iterate over subjects within condition
        for n, subj in enumerate(cond_data.worker_id.unique()):
            sub_data = cond_data[cond_data['worker_id'] == subj]
            
            # iterate over subject's sessions
            for s in sub_data.session.unique():
                sess_data = sub_data[sub_data['session'] == s]
                rec_evs = sess_data[sess_data['type'] == 'REC_WORD']
                list_recalls = []
                
                for i in sess_data.list.unique():
                    sp = rec_evs[rec_evs['list'] == i].serial_position.to_numpy()     # current list serial positions
                    list_recalls.append(len(np.unique(sp[(sp!=99) & (sp!=88)])))      # number of correct recalls on list
                    
                mwr_sess_mat[n, int(s)-1] = np.mean(list_recalls)              # average over lists, store in matrix
        
        # calculate mean and standard error across subjects
        mwr_dict.update({cond_str: np.nanmean(mwr_sess_mat, axis=0)})
        se_dict.update({cond_str: np.nanstd(mwr_sess_mat, axis=0, ddof=1) / np.sqrt(np.count_nonzero(~np.isnan(mwr_sess_mat), axis=0))})
        sub_dict.update({cond_str: mwr_sess_mat})
        
    return mwr_dict, se_dict, sub_dict

def plot_MWR_sess(mwrS_prim_dict, mwrS_prim_se, mwrS_prim_sub, mwrS_rec_dict, mwrS_rec_se, mwrS_rec_sub):
    conds = ['10-2', '20-1', '15-2', '30-1', '20-2', '40-1']
    fig, ax = plt.subplots(2, 3, figsize=(9, 6), sharex=True, sharey=True)
    x = np.linspace(1, 4, 4)
    for i, c in enumerate(conds):
        mwrS_prim = mwrS_prim_dict.get(c)
        se_prim = mwrS_prim_se.get(c)
        
        mwrS_rec = mwrS_rec_dict.get(c)
        se_rec = mwrS_rec_se.get(c)
        
        #ax[i%2, i//2].plot(x, mwrS_prim, color='orange', label='primacy')
        #ax[i%2, i//2].fill_between(x, mwrS_prim-1.96*se_prim, mwrS_prim+1.96*se_prim, color='orange', alpha=0.1)
        #ax[i%2, i//2].plot(x, mwrS_rec, color='purple', label='recency')
        #ax[i%2, i//2].fill_between(x, mwrS_rec-1.96*se_rec, mwrS_rec+1.96*se_rec, color='purple', alpha=0.1)
        ax[i%2, i//2].plot(x, mwrS_prim, marker='o', color='orange', label='primacy', alpha=0.8)
        ax[i%2, i//2].errorbar(x, mwrS_prim, ls='none', yerr=1.96*se_prim, color='orange', alpha=0.5)
        ax[i%2, i//2].plot(x, mwrS_rec, marker='o', color='purple', label='recency', alpha=0.8)
        ax[i%2, i//2].errorbar(x, mwrS_rec, ls='none', yerr=1.96*se_rec, color='purple', alpha=0.5)
        ax[i%2, i//2].set(xticks=x, yticks=np.linspace(5, 14, 10), title=c)
        ax[i%2, i//2].spines['right'].set_visible(False); ax[i%2, i//2].spines['top'].set_visible(False)
        
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=2, shadow=True, bbox_to_anchor=(0.65, 1.06))
    #fig.supxlabel('Serial Position')
    ax[1, 1].set_xlabel('Session')
    fig.supylabel('Mean Words Recalled')
    plt.tight_layout()
    plt.show()
    
# ---------- Serial Position Curve ---------- 

def SPC(df):
    spc_dict = {}                           # dictionary mapping condition to serial position curve
    se_dict = {}                            # dictionary mapping condition to standard error
    
    # iterate over conditions
    for c in df.condition.unique():
        cond_data, list_length, presentation_rate, cond_str = condition_metadata(df, c)
        
        spc_mat = np.zeros((len(cond_data.worker_id.unique()), list_length), dtype = float)     # matrix for individual subject's SPCs
        # iterate over subjects within condition
        for n, subj in enumerate(cond_data.worker_id.unique()):
            sub_data = cond_data[cond_data['worker_id'] == subj]
            
            sub_mat = np.zeros((len(sub_data.session.unique()), list_length), dtype = float)
            # iterate over subject's sessions
            for s_idx, s in enumerate(sub_data.session.unique()):
                sess_data = sub_data[sub_data['session'] == s]
                rec_evs = sess_data[sess_data['type'] == 'REC_WORD']
                
                for i in sess_data.list.unique():
                    sp = rec_evs[rec_evs['list'] == i].serial_position.to_numpy()        # current list serial positions
                    srpos = np.unique(sp[(sp != 99) & (sp != 88)])                       # remove intrusions and repetitions
                    
                    for j in range(1, list_length + 1):
                        sub_mat[s_idx, j-1] += np.count_nonzero(srpos == j)
                        
                sub_mat[s_idx, :] /= float(len(sess_data.list.unique()))                     # divide by lists in session
                        
            spc_mat[n, :] = np.mean(sub_mat, axis=0)
            
        # calculate mean and standard error across subjects
        spc_dict.update({cond_str: np.nanmean(spc_mat, axis=0)})
        se_dict.update({cond_str: np.nanstd(spc_mat, axis=0, ddof=1) / np.sqrt(spc_mat.shape[0])})
        
    return spc_dict, se_dict

def plot_SPC(spc_dict, se_dict):
    conds = ['10-2', '20-1', '15-2', '30-1', '20-2', '40-1']
    colors = ['firebrick', 'darkorange', 'forestgreen', 'dodgerblue', 'midnightblue', 'darkviolet']
    fig, ax = plt.subplots(figsize=(7,5))
    for i, c in enumerate(conds):
        spc = spc_dict.get(c)
        se = se_dict.get(c)
        x = np.linspace(1, len(spc), len(spc))
        ax.plot(x, spc, color=colors[i], label=c)
        ax.fill_between(x, spc-1.96*se, spc+1.96*se, color=colors[i], alpha=0.1)
        
    ax.set(xlabel='Serial Position', ylabel='Recall Probability', xticks=np.linspace(5, 40, 8), yticks=np.linspace(0, 1, 11))
    ax.legend()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()
    
def compare_SPC(spc_prim_dict, spc_prim_se, spc_ns_dict, spc_ns_se, spc_rec_dict, spc_rec_se):
    conds = ['10-2', '20-1', '15-2', '30-1', '20-2', '40-1']
    colors = ['firebrick', 'darkorange', 'forestgreen', 'dodgerblue', 'midnightblue', 'darkviolet']
    fig, ax = plt.subplots(1, 3, figsize=(20,6), sharey=True)
    for i, c in enumerate(conds):
        spc_prim = spc_prim_dict.get(c)
        se_prim = spc_prim_se.get(c)
        spc_ns = spc_ns_dict.get(c)
        se_ns = spc_ns_se.get(c)
        spc_rec = spc_rec_dict.get(c)
        se_rec = spc_rec_se.get(c)

        x = np.linspace(1, len(spc_prim), len(spc_prim))
        ax[0].plot(x, spc_prim, color=colors[i], label=c)
        ax[0].fill_between(x, spc_prim-1.96*se_prim, spc_prim+1.96*se_prim, color=colors[i], alpha=0.1)
        ax[1].plot(x, spc_ns, color=colors[i])
        ax[1].fill_between(x, spc_ns-1.96*se_ns, spc_ns+1.96*se_ns, color=colors[i], alpha=0.1)
        ax[2].plot(x, spc_rec, color=colors[i])
        ax[2].fill_between(x, spc_rec-1.96*se_rec, spc_rec+1.96*se_rec, color=colors[i], alpha=0.1)
        ax[0].set(ylabel='Recall Probability', xticks=np.linspace(5, 40, 8), yticks=np.linspace(0, 1, 11), title='Primacy Group')
        ax[1].set(xlabel='Serial Position', xticks=np.linspace(5, 40, 8), yticks=np.linspace(0, 1, 11), title='No Strategy Group')
        ax[2].set(xticks=np.linspace(5, 40, 8), yticks=np.linspace(0, 1, 11), title='Recency Group')
        ax[0].spines['right'].set_visible(False); ax[0].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False); ax[1].spines['top'].set_visible(False)
        ax[2].spines['right'].set_visible(False); ax[2].spines['top'].set_visible(False)
        
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=6, shadow=True, bbox_to_anchor=(0.67, 1.05))
    #fig.supxlabel('Serial Position')
    #fig.supylabel('Recall Probability')
    plt.tight_layout()
    plt.show()
    
def compare_SPC_cond(spc_prim_dict, spc_prim_se, spc_ns_dict, spc_ns_se, spc_rec_dict, spc_rec_se):
    conds = ['10-2', '20-1', '15-2', '30-1', '20-2', '40-1']
    fig, ax = plt.subplots(2, 3, figsize=(14,10), sharey=True, gridspec_kw={'width_ratios': [1, 1.5, 2]})
    for i, c in enumerate(conds):
        spc_prim = spc_prim_dict.get(c)
        se_prim = spc_prim_se.get(c)
        spc_ns = spc_ns_dict.get(c)
        se_ns = spc_ns_se.get(c)
        spc_rec = spc_rec_dict.get(c)
        se_rec = spc_rec_se.get(c)
        
        xax = np.prod(np.array(c.split('-')).astype(int))

        x = np.linspace(1, len(spc_prim), len(spc_prim))
        ax[i%2, i//2].plot(x, spc_prim, color='orange', label='primacy')
        ax[i%2, i//2].fill_between(x, spc_prim-1.96*se_prim, spc_prim+1.96*se_prim, color='orange', alpha=0.1)
        ax[i%2, i//2].plot(x, spc_ns, color='darkgray', label='no strategy')
        ax[i%2, i//2].fill_between(x, spc_ns-1.96*se_ns, spc_ns+1.96*se_ns, color='darkgray', alpha=0.1)
        ax[i%2, i//2].plot(x, spc_rec, color='purple', label='recency')
        ax[i%2, i//2].fill_between(x, spc_rec-1.96*se_rec, spc_rec+1.96*se_rec, color='purple', alpha=0.1)
        #xax = 40
        ax[i%2, i//2].set(xlabel='Serial Position', ylabel='Recall Probability', xticks=np.linspace(5, xax, xax//5), yticks=np.linspace(0, 1, 11), title=conds[i])
        ax[i%2, i//2].spines['right'].set_visible(False); ax[i%2, i//2].spines['top'].set_visible(False)
        
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=3, shadow=True, bbox_to_anchor=(0.65, 1.06))
    #fig.supxlabel('Serial Position')
    #fig.supylabel('Recall Probability')
    plt.tight_layout()
    plt.show()


### SPC conditioned on position of 1st recall
def SPC_fst(df, bufferStart, bufferEnd):
    spc_start_dict = {}; spc_end_dict = {}              # dictionaries mapping condition to conditional SPC
    se_start_dict = {}; se_end_dict = {}              # dictionaries mapping condition to standard error
    
    # iterate over conditions
    for c in df.condition.unique():
        cond_data, list_length, presentation_rate, cond_str = condition_metadata(df, c)
        
        spc_start_mat = np.zeros((len(cond_data.worker_id.unique()), list_length), dtype=float)    # matrices for subjects' start and end SPCs
        spc_end_mat = np.zeros((len(cond_data.worker_id.unique()), list_length), dtype=float)
        for n, subj in enumerate(cond_data.worker_id.unique()):
            sub_data = cond_data[cond_data['worker_id'] == subj]
            
            sub_start_mat = np.zeros((len(sub_data.session.unique()), list_length), dtype=float)
            sub_end_mat = np.zeros((len(sub_data.session.unique()), list_length), dtype=float)
            # iterate over subject's sessions
            for s_idx, s in enumerate(sub_data.session.unique()):
                sess_data = sub_data[sub_data['session'] == s]
                rec_evs = sess_data[sess_data['type'] == 'REC_WORD']
                start_lists = 0; end_lists = 0
                
                for i in sess_data.list.unique():
                    sp = rec_evs[rec_evs['list']== i].serial_position.to_numpy()         # current list serial positions
                    srpos = np.unique(sp[(sp != 99) & (sp != 88)])                       # remove intrusions and repetitions
                    
                    if sp[0] >=1 and sp[0] <= bufferStart:       # primacy item
                        start_lists += 1
                        for j in range(1, list_length + 1):
                            sub_start_mat[s_idx, j-1] += np.count_nonzero(srpos == j)
                        
                    if sp[0] >= list_length - bufferEnd + 1 and sp[0] <= list_length:     # recency item
                        end_lists += 1
                        for k in range(1, list_length + 1):
                            sub_end_mat[s_idx, k-1] += np.count_nonzero(srpos == k)
                            
                sub_start_mat[s_idx, :] /= float(start_lists)
                sub_end_mat[s_idx, :] /= float(end_lists)
                
            spc_start_mat[n, :] = np.mean(sub_start_mat, axis=0)
            spc_end_mat[n, :] = np.mean(sub_end_mat, axis=0)
            
        # calculate mean and standard error across subjects
        spc_start_dict.update({cond_str: np.nanmean(spc_start_mat, axis=0)})
        spc_end_dict.update({cond_str: np.nanmean(spc_end_mat, axis=0)})
        se_start_dict.update({cond_str: np.nanstd(spc_start_mat, axis=0, ddof=1) / np.sqrt(spc_start_mat.shape[0])})
        se_end_dict.update({cond_str: np.nanstd(spc_end_mat, axis=0, ddof=1) / np.sqrt(spc_end_mat.shape[0])})
        
    return spc_start_dict, se_start_dict, spc_end_dict, se_end_dict

def plot_SPC_fst(spc_start_dict, se_start_dict, spc_end_dict, se_end_dict, bufferStart, bufferEnd):
    conds = ['10-2', '20-1', '15-2', '30-1', '20-2', '40-1']
    colors = ['firebrick', 'darkorange', 'forestgreen', 'dodgerblue', 'midnightblue', 'darkviolet']
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    
    for i, c in enumerate(conds):
        spc_start = spc_start_dict.get(c)
        se_start = se_start_dict.get(c)
        spc_end = spc_end_dict.get(c)
        se_end = se_end_dict.get(c)
        x = np.linspace(1, len(spc_start), len(spc_start))
        ax[0].plot(x, spc_start, color=colors[i], label=c)
        ax[0].fill_between(x, spc_start-1.96*se_start, spc_start+1.96*se_start, color=colors[i], alpha=0.1)
        ax[1].plot(x, spc_end, color=colors[i], label=c)
        ax[1].fill_between(x, spc_end-1.96*se_end, spc_end+1.96*se_end, color=colors[i], alpha=0.1)
        
    ax[0].set(xlabel='Serial Position', ylabel='Recall Probability', title='Recall Initiated at Start of List', 
             xticks=np.linspace(5, 40, 8), yticks=np.linspace(0, 1, 11))
    ax[0].legend()
    ax[1].set(xlabel='Serial Position', ylabel='Recall Probability', title='Recall Initiated at End of List', 
             xticks=np.linspace(5, 40, 8), yticks=np.linspace(0, 1, 11))
    ax[1].legend()
    plt.tight_layout()
    plt.show()
    

# ---------- Probability of First Recall ----------

def PFR(df):
    pfr_dict = {}                          # dictionary mapping condition to probability of first recall
    se_dict = {}                           # dictionary mapping condition to standard error
    
    # iterate over conditions
    for c in df.condition.unique():
        cond_data, list_length, presentation_rate, cond_str = condition_metadata(df, c)
        
        pfr_mat = np.zeros((len(cond_data.worker_id.unique()), list_length), dtype = float)     # matrix for individual subject's PFRs
        # iterate over subjects within condition
        for n, subj in enumerate(cond_data.worker_id.unique()):
            sub_data = cond_data[cond_data['worker_id'] == subj]
            
            sub_mat = np.zeros((len(sub_data.session.unique()), list_length), dtype = float)
            # iterate over subject's sessions
            for s_idx, s in enumerate(sub_data.session.unique()):
                sess_data = sub_data[sub_data['session'] == s]
                rec_evs = sess_data[sess_data['type'] == 'REC_WORD']
                
                subtract = 0
                for i in sess_data.list.unique():
                    sp = rec_evs[rec_evs['list'] == i].serial_position.to_numpy()            # current list serial positions
                    if len(sp) > 1:       # always null recall at end of list
                        if sp[0] > 0 and sp[0] <= list_length:
                            sub_mat[s_idx, int(sp[0])-1] += 1
                    else:     # no recalls, subtract lists
                        subtract += 1
                
                sub_mat[s_idx, :] /= float(len(sess_data.list.unique() - subtract))           # divide by lists in session
                
            pfr_mat[n, :] = np.mean(sub_mat, axis=0)
            
        # calculate mean and standard error across subjects
        pfr_dict.update({cond_str: np.nanmean(pfr_mat, axis=0)})
        se_dict.update({cond_str: np.nanstd(pfr_mat, axis=0, ddof=1) / np.sqrt(pfr_mat.shape[0])})
        
    return pfr_dict, se_dict

def plot_PFR(pfr_dict, se_dict):
    conds = ['10-2', '20-1', '15-2', '30-1', '20-2', '40-1']
    colors = ['firebrick', 'darkorange', 'forestgreen', 'dodgerblue', 'midnightblue', 'darkviolet']
    fig, ax = plt.subplots(figsize=(7,5))
    for i, c in enumerate(conds):
        pfr = pfr_dict.get(c)
        se = se_dict.get(c)
        x = np.linspace(1, len(pfr), len(pfr))
        ax.plot(x, pfr, color=colors[i], label=c)
        ax.fill_between(x, pfr-1.96*se, pfr+1.96*se, color=colors[i], alpha=0.1)
        
    ax.set(xlabel='Serial Position', ylabel='Probability of First Recall')
    ax.legend()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()
    
def compare_PFR(pfr_prim_dict, pfr_prim_se, pfr_ns_dict, pfr_ns_se, pfr_rec_dict, pfr_rec_se):
    conds = ['10-2', '20-1', '15-2', '30-1', '20-2', '40-1']
    colors = ['firebrick', 'darkorange', 'forestgreen', 'dodgerblue', 'midnightblue', 'darkviolet']
    fig, ax = plt.subplots(1, 3, figsize=(18,5), sharey=True)
    for i, c in enumerate(conds):
        pfr_prim = pfr_prim_dict.get(c)
        se_prim = pfr_prim_se.get(c)
        pfr_ns = pfr_ns_dict.get(c)
        se_ns = pfr_ns_se.get(c)
        pfr_rec = pfr_rec_dict.get(c)
        se_rec = pfr_rec_se.get(c)
        
        x = np.linspace(1, len(pfr_prim), len(pfr_prim))
        ax[0].plot(x, pfr_prim, color=colors[i], label=c)
        ax[0].fill_between(x, pfr_prim-1.96*se_prim, pfr_prim+1.96*se_prim, color=colors[i], alpha=0.1)
        ax[1].plot(x, pfr_ns, color=colors[i], label=c)
        ax[1].fill_between(x, pfr_ns-1.96*se_ns, pfr_ns+1.96*se_ns, color=colors[i], alpha=0.1)
        ax[2].plot(x, pfr_rec, color=colors[i])
        ax[2].fill_between(x, pfr_rec-1.96*se_rec, pfr_rec+1.96*se_rec, color=colors[i], alpha=0.1)
        ax[0].set(ylabel='Recall Probability', xticks=np.linspace(5, 40, 8), yticks=np.linspace(0, 0.7, 8), title='Primacy Group')
        ax[1].set(xlabel='Serial Position', xticks=np.linspace(5, 40, 8), yticks=np.linspace(0, 0.7, 8), title='No Strategy Group')
        ax[2].set(xticks=np.linspace(5, 40, 8), yticks=np.linspace(0, 0.7, 8), title='Recency Group')
        ax[0].spines['right'].set_visible(False); ax[0].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False); ax[1].spines['top'].set_visible(False)
        ax[2].spines['right'].set_visible(False); ax[2].spines['top'].set_visible(False)
        
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=6, shadow=True, bbox_to_anchor=(0.67, 1.05))
    plt.tight_layout()
    plt.show()
    
def compare_PFR_cond(pfr_prim_dict, pfr_prim_se, pfr_ns_dict, pfr_ns_se, pfr_rec_dict, pfr_rec_se):
    conds = ['10-2', '20-1', '15-2', '30-1', '20-2', '40-1']
    colors = ['firebrick', 'darkorange', 'forestgreen', 'dodgerblue', 'midnightblue', 'darkviolet']
    fig, ax = plt.subplots(2, 3, figsize=(14,10), sharey=True, gridspec_kw={'width_ratios': [1, 1.5, 2]})
    for i, c in enumerate(conds):
        pfr_prim = pfr_prim_dict.get(c)
        se_prim = pfr_prim_se.get(c)
        pfr_ns = pfr_ns_dict.get(c)
        se_ns = pfr_ns_se.get(c)
        pfr_rec = pfr_rec_dict.get(c)
        se_rec = pfr_rec_se.get(c)
        
        xax = np.prod(np.array(c.split('-')).astype(int))
        
        x = np.linspace(1, len(pfr_prim), len(pfr_prim))
        ax[i%2, i//2].plot(x, pfr_prim, color='orange', label='primacy')
        ax[i%2, i//2].fill_between(x, pfr_prim-1.96*se_prim, pfr_prim+1.96*se_prim, color='orange', alpha=0.1)
        ax[i%2, i//2].plot(x, pfr_ns, color='darkgray', label='no strategy')
        ax[i%2, i//2].fill_between(x, pfr_ns-1.96*se_ns, pfr_ns+1.96*se_ns, color='darkgray', alpha=0.1)
        ax[i%2, i//2].plot(x, pfr_rec, color='purple', label='recency')
        ax[i%2, i//2].fill_between(x, pfr_rec-1.96*se_rec, pfr_rec+1.96*se_rec, color='purple', alpha=0.1)
        
        ax[i%2, i//2].set(xlabel='Serial Position', ylabel='Recall Probability', xticks=np.linspace(5, xax, xax//5), yticks=np.linspace(0, 0.7, 8), title=c)
        ax[i%2, i//2].spines['right'].set_visible(False); ax[i%2, i//2].spines['top'].set_visible(False)
        
    handles, labels = ax[i%2, i//2].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=3, shadow=True, bbox_to_anchor=(0.65, 1.06))
    plt.tight_layout()
    plt.show()


# ---------- Lag-CRP ----------

def lag_CRP(df, buffer):
    lcrp_dict = {}                               # dictionary mapping condition to lag-CRP
    se_dict = {}                                 # dictionary mapping condition to standard error
    
    # iterate over conditions
    for c in df.condition.unique():
        cond_data, list_length, presentation_rate, cond_str = condition_metadata(df, c)
        
        lcrp_mat = np.zeros((len(cond_data.worker_id.unique()), 2*list_length-1), dtype=float)      # matrix for individual subject's lag-CRPs
        # iterate over subjects within condition
        for n, subj in enumerate(cond_data.worker_id.unique()):
            sub_data = cond_data[cond_data['worker_id'] == subj]
            
            sub_ac_mat = np.zeros((len(sub_data.session.unique()), 2*list_length-1), dtype=float)
            sub_poss_mat = np.zeros((len(sub_data.session.unique()), 2*list_length-1), dtype=float)
            # iterate over subject's sessions
            for s_idx, s in enumerate(sub_data.session.unique()):
                sess_data = sub_data[sub_data['session'] == s]
                word_evs = sess_data[sess_data['type'] == 'WORD']; rec_evs = sess_data[sess_data['type'] == 'REC_WORD']
                
                for i in sess_data.list.unique():
                    sp = rec_evs[rec_evs['list'] == i].serial_position.to_numpy()          # current list serial positions
                    sp = sp[sp != 99]                                                      # remove last null recall
                    
                    sp = mark_repetitions(sp)
                    
                    # exclude first 'buffer' output positions
                    excludeOutputs = sp[:buffer]
                    sp = sp[buffer:]
                    sps_left = [x for x in range(1, list_length+1)]    # array with remaining serial positions, subtract from
                    for exOut in excludeOutputs:
                        try:
                            sps_left.remove(exOut)                     # remove first outputs from possible transitions
                        except:                                        # item already removed
                            pass
                        
                    for j in range(len(sp) - 1):
                        if sp[j]!=88 and sp[j]!=77:                    # correct recall
                            sps_left.remove(sp[j])                     # can't transition to already recalled serial position
                            if sp[j+1]!=88 and sp[j+1]!=77:            # transition between correct recalls
                                lag = sp[j+1] - sp[j]                  # actual transition
                                sub_ac_mat[s_idx, int(lag)+list_length-1] += 1
                                for l in sps_left:                     # find all possible transitions
                                    poss = l - sp[j]
                                    sub_poss_mat[s_idx, int(poss)+list_length-1] += 1
                                    
            lcrp_mat[n, :] = np.mean((sub_ac_mat / sub_poss_mat), axis=0)     # we actually want the NaNs from zero division
            
        # calculate mean and standard error across subjects
        lcrp_dict.update({cond_str: np.nanmean(lcrp_mat, axis=0)})
        se_dict.update({cond_str: np.nanstd(lcrp_mat, axis=0, ddof=1) / np.sqrt(lcrp_mat.shape[0])})
        
    return lcrp_dict, se_dict

def plot_lag_CRP(lcrp_dict, se_dict):
    conds = ['10-2', '20-1', '15-2', '30-1', '20-2', '40-1']
    colors = ['firebrick', 'darkorange', 'forestgreen', 'dodgerblue', 'midnightblue', 'darkviolet']
    fig, ax = plt.subplots(figsize=(7,5))
    x_bwd = np.linspace(-7, -1, 7); x_fwd = np.linspace(1, 7, 7)
    for i, c in enumerate(conds):
        list_length = int(c[:2])
        lcrp = lcrp_dict.get(c)
        lcrp_bwd = lcrp[(list_length - 1) - 7: list_length - 1]; lcrp_fwd = lcrp[list_length: list_length + 7]
        se = se_dict.get(c)
        se_bwd = se[(list_length - 1) - 7: list_length - 1]; se_fwd = se[list_length: list_length + 7]
        ax.plot(x_bwd, lcrp_bwd, color=colors[i])
        ax.fill_between(x_bwd, lcrp_bwd-1.96*se_bwd, lcrp_bwd+1.96*se_bwd, color=colors[i], alpha=0.1)
        ax.plot(x_fwd, lcrp_fwd, color=colors[i], label=c)
        ax.fill_between(x_fwd, lcrp_fwd-1.96*se_fwd, lcrp_fwd+1.96*se_fwd, color=colors[i], alpha=0.1)
        
    ax.set(xlabel='Lag', ylabel='Conditional Response Probability')
    ax.legend()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()
    
def compare_lag_CRP(lcrp_prim_dict, lcrp_prim_se, lcrp_ns_dict, lcrp_ns_se, lcrp_rec_dict, lcrp_rec_se):
    conds = ['10-2', '20-1', '15-2', '30-1', '20-2', '40-1']
    fig, ax = plt.subplots(2, 3, figsize=(9,6), sharey=True, sharex=True)
    x_bwd = np.linspace(-7, -1, 7); x_fwd = np.linspace(1, 7, 7)
    for i, c in enumerate(conds):
        list_length = int(c[:2])
        lcrp_prim = lcrp_prim_dict.get(c)
        lcrp_prim_bwd = lcrp_prim[(list_length - 1) - 7: list_length - 1]; lcrp_prim_fwd = lcrp_prim[list_length: list_length + 7]
        se_prim = lcrp_prim_se.get(c)
        se_prim_bwd = se_prim[(list_length - 1) - 7: list_length - 1]; se_prim_fwd = se_prim[list_length: list_length + 7]
        
        lcrp_ns = lcrp_ns_dict.get(c)
        lcrp_ns_bwd = lcrp_ns[(list_length - 1) - 7: list_length - 1]; lcrp_ns_fwd = lcrp_ns[list_length: list_length + 7]
        se_ns = lcrp_ns_se.get(c)
        se_ns_bwd = se_ns[(list_length - 1) - 7: list_length - 1]; se_ns_fwd = se_ns[list_length: list_length + 7]
        
        lcrp_rec = lcrp_rec_dict.get(c)
        lcrp_rec_bwd = lcrp_rec[(list_length - 1) - 7: list_length - 1]; lcrp_rec_fwd = lcrp_rec[list_length: list_length + 7]
        se_rec = lcrp_rec_se.get(c)
        se_rec_bwd = se_rec[(list_length - 1) - 7: list_length - 1]; se_rec_fwd = se_rec[list_length: list_length + 7]
        
        ax[i%2, i//2].plot(x_bwd, lcrp_prim_bwd, color='orange', label='primacy')
        ax[i%2, i//2].fill_between(x_bwd, lcrp_prim_bwd-1.96*se_prim_bwd, lcrp_prim_bwd+1.96*se_prim_bwd, color='orange', alpha=0.1)
        ax[i%2, i//2].plot(x_fwd, lcrp_prim_fwd, color='orange')
        ax[i%2, i//2].fill_between(x_fwd, lcrp_prim_fwd-1.6*se_prim_fwd, lcrp_prim_fwd+1.96*se_prim_fwd, color='orange', alpha=0.1)
        
        ax[i%2, i//2].plot(x_bwd, lcrp_ns_bwd, color='darkgray', label='no strategy')
        ax[i%2, i//2].fill_between(x_bwd, lcrp_ns_bwd-1.96*se_ns_bwd, lcrp_ns_bwd+1.96*se_ns_bwd, color='darkgray', alpha=0.1)
        ax[i%2, i//2].plot(x_fwd, lcrp_ns_fwd, color='darkgray')
        ax[i%2, i//2].fill_between(x_fwd, lcrp_ns_fwd-1.6*se_ns_fwd, lcrp_ns_fwd+1.96*se_ns_fwd, color='darkgray', alpha=0.1)
        
        ax[i%2, i//2].plot(x_bwd, lcrp_rec_bwd, color='purple', label='recency')
        ax[i%2, i//2].fill_between(x_bwd, lcrp_rec_bwd-1.96*se_rec_bwd, lcrp_rec_bwd+1.96*se_rec_bwd, color='purple', alpha=0.1)
        ax[i%2, i//2].plot(x_fwd, lcrp_rec_fwd, color='purple')
        ax[i%2, i//2].fill_between(x_fwd, lcrp_rec_fwd-1.6*se_rec_fwd, lcrp_rec_fwd+1.96*se_rec_fwd, color='purple', alpha=0.1)
        
        ax[i%2, i//2].set(yticks=np.linspace(0, 0.8, 5), title=conds[i])
        ax[i%2, i//2].spines['right'].set_visible(False)
        ax[i%2, i//2].spines['top'].set_visible(False)
        
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=3, shadow=True, bbox_to_anchor=(0.76, 1.06))
    ax[1, 1].set(xlabel='Lag')
    fig.supylabel('Conditional Response Probability')
    plt.tight_layout()
    plt.show()


# ---------- Semantic-CRP ----------

def semantic_CRP(df, wordpool, w2v_scores):
    scrp_dict = {}                             # dictionary mapping condition to semantic-CRP
    se_dict = {}                               # dictionary mapping condition to standard error
    
    # iterate over conditions
    for c in df.condition.unique():
        cond_data, list_length, presentation_rate, cond_str = condition_metadata(df, c)
        
        scrp_mat = np.zeros((len(cond_data.worker_id.unique()), 6), dtype=float)               # matrix for individual subject's semantic-CRPs
        # iterate over subjects within condition
        for n, subj in enumerate(cond_data.worker_id.unique()):
            sub_data = cond_data[cond_data['worker_id'] == subj]
            
            sub_ac_mat = np.zeros((len(sub_data.session.unique()), 6), dtype=float)
            sub_poss_mat = np.zeros((len(sub_data.session.unique()), 6), dtype=float)
            # iterate over subject's sessions
            for s_idx, s in enumerate(sub_data.session.unique()):
                sess_data = sub_data[sub_data['session'] == s]
                word_evs = sess_data[sess_data['type'] == 'WORD']; rec_evs = sess_data[sess_data['type'] == 'REC_WORD']
                ac_list = []                      # semantic similarities for actual transitions
                poss_list = []                    # semantic similarities for possible transitions
                
                for i in sess_data.list.unique():
                    sp = rec_evs[rec_evs['list'] == i].serial_position.to_numpy()          # current list serial positions
                    sp = sp[sp != 99]                                                      # remove last null recall
                    w = word_evs[word_evs['list'] == i].word.tolist()                      # current list words
                    r = rec_evs[rec_evs['list'] == i].rec_word.tolist()                    # current list recalls
                    r = r[:len(sp)]
                    
                    sp = mark_repetitions(sp)
                    
                    words_left = [x.upper() for x in w]                       # array with remaining words, subtract from
                    for j in range(len(sp) - 1):
                        if sp[j]!=88 and sp[j]!=77:                         # correct recall
                            words_left.remove(r[j].upper())                 # can't transition to already recalled serial position
                            if sp[j+1]!=88 and sp[j+1]!=77:                 # transitions between correct recalls
                                if r[j].upper() in wordpool and r[j+1].upper() in wordpool:
                                    wv1 = wordpool.index(r[j].upper())
                                    wv2 = wordpool.index(r[j+1].upper())
                                    ss = w2v_scores[wv1][wv2]               # actual transition semantic similarity
                                    ac_list.append(ss)
                                    for wl in words_left:                   # loop over words not yet recalled
                                        wv3 = wordpool.index(wl.upper())
                                        _ss = w2v_scores[wv1][wv3]          # possible transition semantic similarity
                                        poss_list.append(_ss)
                
                sub_ac_mat[s_idx, :], _ = np.histogram(ac_list, bins=[min(ac_list), 0.1, 0.2, 0.3, 0.4, 0.5, 1])
                sub_poss_mat[s_idx, :], _ = np.histogram(poss_list, bins=[min(poss_list), 0.1, 0.2, 0.3, 0.4, 0.5, 1])
                
            scrp_mat[n, :] = np.mean((sub_ac_mat / sub_poss_mat), axis=0)   # we want the NaNs from zero division
            
        # calculate mean and standard error across subjects
        scrp_dict.update({cond_str: np.nanmean(scrp_mat, axis=0)})
        se_dict.update({cond_str: np.nanstd(scrp_mat, axis=0, ddof=1) / np.sqrt(scrp_mat.shape[0])})
    
    return scrp_dict, se_dict

def plot_semantic_CRP(scrp_dict, se_dict):
    conds = ['10-2', '20-1', '15-2', '30-1', '20-2', '40-1']
    colors = ['firebrick', 'darkorange', 'forestgreen', 'dodgerblue', 'midnightblue', 'darkviolet']
    fig, ax = plt.subplots(figsize=(5,5))
    x = np.linspace(1, 6, 6)
    for i, c in enumerate(conds):
        scrp = scrp_dict.get(c)
        se = se_dict.get(c)
        ax.plot(x, scrp, color=colors[i], label=c)
        ax.fill_between(x, scrp-1.96*se, scrp+1.96*se, color=colors[i], alpha=0.1)
       
    ax.set(xlabel='Semantic Similarity', ylabel='Conditional Response Probability', 
          xticks = x, xticklabels=['< 0.1', '0.1 - 0.2', '0.2 - 0.3', '0.3 - 0.4', '0.4 - 0.5', '0.5 - 1'])
    ax.legend()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()
    
def compare_semantic_CRP(scrp_prim_dict, scrp_prim_se, scrp_ns_dict, scrp_ns_se, scrp_rec_dict, scrp_rec_se):
    conds = ['10-2', '20-1', '15-2', '30-1', '20-2', '40-1']
    fig, ax = plt.subplots(2, 3, figsize=(12,6), sharex=True)
    x = np.linspace(1, 6, 6)
    for i, c in enumerate(conds):
        scrp_prim = scrp_prim_dict.get(c)
        se_prim = scrp_prim_se.get(c)
        scrp_ns = scrp_ns_dict.get(c)
        se_ns = scrp_ns_se.get(c)
        scrp_rec = scrp_rec_dict.get(c)
        se_rec = scrp_rec_se.get(c)
        
        ax[i%2, i//2].plot(x, scrp_prim, color='orange', label='primacy')
        ax[i%2, i//2].fill_between(x, scrp_prim-1.96*se_prim, scrp_prim+1.96*se_prim, color='orange', alpha=0.1)
        ax[i%2, i//2].plot(x, scrp_ns, color='darkgray', label='no strategy')
        ax[i%2, i//2].fill_between(x, scrp_ns-1.96*se_ns, scrp_ns+1.96*se_ns, color='darkgray', alpha=0.1)
        ax[i%2, i//2].plot(x, scrp_rec, color='purple', label='recency')
        ax[i%2, i//2].fill_between(x, scrp_rec-1.96*se_rec, scrp_rec+1.96*se_rec, color='purple', alpha=0.1)
        ax[i%2, i//2].set(xticks = x, xticklabels=['< 0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-1'], title=c)
        ax[i%2, i//2].spines['right'].set_visible(False)
        ax[i%2, i//2].spines['top'].set_visible(False)
        
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=3, shadow=True, bbox_to_anchor=(0.76, 1.06))
    ax[1, 1].set(xlabel='Semantic Similarity')
    fig.supylabel('Conditional Response Probability')
    plt.tight_layout()
    plt.show()
    

# ---------- Temporal Clustering ----------

def temporal_cl(df):
    tcl_dict = {}                           # dictionary mapping condition to temporal clustering score
    se_dict = {}                            # dictionary mapping condition to standard error
    sub_dict = {}                           # dictionary mapping condition to subject scores
    
    # iterate over conditions
    for c in df.condition.unique():
        cond_data, list_length, presentation_rate, cond_str = condition_metadata(df, c)
        
        tcl_mat = np.zeros(len(cond_data.worker_id.unique()), dtype=float)   # matrix for individual subject's temporal clustering scores
        # iterate over subjects within condition
        for n, subj in enumerate(cond_data.worker_id.unique()):
            sub_data = cond_data[cond_data['worker_id'] == subj]
            
            sub_tot_mat = np.zeros(len(sub_data.session.unique()), dtype=float)
            sub_cnt_mat = np.zeros(len(sub_data.session.unique()), dtype=float)
            # iterate over subject's sessions
            for s_idx, s in enumerate(sub_data.session.unique()):
                sess_data = sub_data[sub_data['session'] == s]
                word_evs = sess_data[sess_data['type'] == 'WORD']; rec_evs = sess_data[sess_data['type'] == 'REC_WORD']
                
                for i in sess_data.list.unique():
                    sp = rec_evs[rec_evs['list'] == i].serial_position.to_numpy()         # current list serial positions
                    sp = sp[sp != 99]                                                     # remove last null recall
                    
                    sp = mark_repetitions(sp)
                    
                    sps_left = [x for x in range(1, list_length+1)]
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
                                    sub_tot_mat[s_idx] += ptile_rank
                                    sub_cnt_mat[s_idx] += 1
                                    
            tcl_mat[n] = np.mean(sub_tot_mat / sub_cnt_mat)            # subject average across sessions
                
        # calculate mean and standard error across subjects
        tcl_dict.update({cond_str: np.nanmean(tcl_mat)})
        se_dict.update({cond_str: np.nanstd(tcl_mat, ddof=1) / np.sqrt(len(tcl_mat))})
        sub_dict.update({cond_str: tcl_mat})
        
    return tcl_dict, se_dict, sub_dict

def plot_temporal_cl(tcl_dict, se_dict):
    conds = ['10-2', '20-1', '15-2', '30-1', '20-2', '40-1']
    colors = ['firebrick', 'darkorange', 'forestgreen', 'dodgerblue', 'midnightblue', 'darkviolet']
    tcl = []
    se = []
    for c in conds:
        tcl.append(tcl_dict.get(c))
        se.append(se_dict.get(c))
    
    fig, ax = plt.subplots(figsize=(7,5))
    ax.bar(conds, tcl, yerr=se, width=0.65, color=colors)
    ax.set(xlabel='Condition', ylabel='Temporal Clutering Score', ylim=(0.65, 0.95))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()
    
def compare_temporal_cl(tcl_prim_dict, tcl_prim_se, tcl_prim_sub, tcl_ns_dict, tcl_ns_se, tcl_ns_sub, tcl_rec_dict, tcl_rec_se, tcl_rec_sub):
    conds = ['10-2', '20-1', '15-2', '30-1', '20-2', '40-1']
    fig, ax = plt.subplots(2, 3, figsize=(9, 6), sharex=True, sharey=True)
    for i, c in enumerate(conds):
        tcl_prim = tcl_prim_dict.get(c)
        se_prim = tcl_prim_se.get(c)
        sub_prim = tcl_prim_sub.get(c)
        
        tcl_ns = tcl_ns_dict.get(c)
        se_ns = tcl_ns_se.get(c)
        sub_ns = tcl_ns_sub.get(c)

        tcl_rec = tcl_rec_dict.get(c)
        se_rec = tcl_rec_se.get(c)
        sub_rec = tcl_rec_sub.get(c)

        #ts, pvals = scipy.stats.ttest_ind(sub_prim, sub_rec, equal_var=False, alternative='two-sided')

        ax[i%2, i//2].bar([1,2,3], [tcl_prim, tcl_ns, tcl_rec], yerr=[se_prim, se_ns, se_rec], width=0.65, color=['orange', 'darkgray', 'purple'])
        ax[i%2, i//2].set(xticks=[1,2,3], xticklabels=['Primacy', 'No Strategy', 'Recency'], title=c, ylim=(0.5, 1))
        ax[i%2, i//2].spines['right'].set_visible(False)
        ax[i%2, i//2].spines['top'].set_visible(False)
    
    fig.supylabel('Temporal Clustering Score')
    ax[1, 1].set(xlabel='Group')
    plt.tight_layout()
    plt.show()
    
def compare_temporal_cl_stats(tcl_prim_dict, tcl_prim_se, tcl_prim_sub, tcl_ns_dict, tcl_ns_se, tcl_ns_sub, tcl_rec_dict, tcl_rec_se, tcl_rec_sub):
    conds = ['10-2', '20-1', '15-2', '30-1', '20-2', '40-1']
    fig, ax = plt.subplots(2, 3, figsize=(9, 6), sharex=True, sharey=True)
    for i, c in enumerate(conds):
        tcl_prim = tcl_prim_dict.get(c)
        se_prim = tcl_prim_se.get(c)
        sub_prim = tcl_prim_sub.get(c)
        
        tcl_ns = tcl_ns_dict.get(c)
        se_ns = tcl_ns_se.get(c)
        sub_ns = tcl_ns_sub.get(c)

        tcl_rec = tcl_rec_dict.get(c)
        se_rec = tcl_rec_se.get(c)
        sub_rec = tcl_rec_sub.get(c)

        ts_prim_rec, pval_prim_rec = scipy.stats.ttest_ind(sub_prim, sub_rec, equal_var=False, alternative='two-sided')
        ts_prim_ns, pval_prim_ns = scipy.stats.ttest_ind(sub_prim, sub_ns, equal_var=False, alternative='two-sided')
        ts_ns_rec, pval_ns_rec = scipy.stats.ttest_ind(sub_ns, sub_rec, equal_var=False, alternative='two-sided')
        
        h1 = np.max(np.concatenate((sub_prim, sub_ns, sub_rec))) + 0.04
        h2 = h1 + 0.04
        
        if pval_prim_rec < 0.05:
            ax[i%2, i//2].plot([1,3], [h2, h2], color='midnightblue')
        if pval_prim_ns < 0.05:
            ax[i%2, i//2].plot([1.1,1.9], [h1, h1], color='saddlebrown')
        if pval_ns_rec < 0.05:
            ax[i%2, i//2].plot([2.1,2.9], [h1, h1], color='saddlebrown')

        ax[i%2, i//2].bar([1,2,3], [tcl_prim, tcl_ns, tcl_rec], yerr=[se_prim, se_ns, se_rec], width=0.65, color=['orange', 'darkgray', 'purple'], alpha=0.6)
        ax[i%2, i//2].scatter(np.full(len(sub_prim), 1), sub_prim, s=5, color='orange', alpha=0.9)
        ax[i%2, i//2].scatter(np.full(len(sub_ns), 2), sub_ns, s=5, color='darkgray', alpha=0.9)
        ax[i%2, i//2].scatter(np.full(len(sub_rec), 3), sub_rec, s=5, color='purple', alpha=0.9)
        ax[i%2, i//2].set(xticks=[1,2,3], yticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0], xticklabels=['Primacy', 'No Strategy', 'Recency'], title=c, ylim=(0.4, 1.1))
        ax[i%2, i//2].spines['right'].set_visible(False)
        ax[i%2, i//2].spines['top'].set_visible(False)
    
    fig.supylabel('Temporal Clustering Score')
    ax[1, 1].set(xlabel='Group')
    plt.tight_layout()
    plt.show()
    

# ---------- Semantic Clustering ----------

def semantic_cl(df, wordpool, w2v_scores):
    scl_dict = {}                            # dictionary mapping condition to temporal clustering score
    se_dict = {}                             # dictionary mapping condition to standard error
    sub_dict = {}                            # dictionary mapping condition to subject scores
    
    # iterate over conditions
    for c in df.condition.unique():
        cond_data, list_length, presentation_rate, cond_str = condition_metadata(df, c)
        
        scl_mat = np.zeros(len(cond_data.worker_id.unique()), dtype=float)       # matrix for individual subject's semantic clustering scores
        # iterate over subjects within condition
        for n, subj in enumerate(cond_data.worker_id.unique()):
            sub_data = cond_data[cond_data['worker_id'] == subj]
            
            sub_tot_mat = np.zeros(len(sub_data.session.unique()), dtype=float)
            sub_cnt_mat = np.zeros(len(sub_data.session.unique()), dtype=float)
            # iterate over subject's sessions
            for s_idx, s in enumerate(sub_data.session.unique()):
                sess_data = sub_data[sub_data['session'] == s]
                word_evs = sess_data[sess_data['type'] == 'WORD']; rec_evs = sess_data[sess_data['type'] == 'REC_WORD']
                
                for i in sess_data.list.unique():
                    sp = rec_evs[rec_evs['list'] == i].serial_position.to_numpy()      # current list serial positions
                    sp = sp[sp != 99]                                                  # remove last null recall
                    w = word_evs[word_evs['list'] == i].word.tolist()                  # current list words
                    r = rec_evs[rec_evs['list'] == i].rec_word.tolist()                # current list recalls
                    r = r[:len(sp)]
                    
                    sp = mark_repetitions(sp)
                    
                    words_left = [x.upper() for x in w]                       # array with remaining words, subtract from
                    for j in range(len(sp) - 1):
                        if sp[j]!=88 and sp[j]!=77:                           # correct recall
                            words_left.remove(r[j].upper())                   # can't transition to already recalled serial position
                            if sp[j+1]!=88 and sp[j+1]!=77:                   # transition between correct recalls
                                if r[j].upper() in wordpool and r[j+1].upper() in wordpool:
                                    possList = []
                                    wv1 = wordpool.index(r[j].upper())
                                    wv2 = wordpool.index(r[j+1].upper())
                                    ss = w2v_scores[wv1][wv2]                 # actual transition semantic similarity
                                    for l in range(len(words_left)):
                                        wv3 = wordpool.index(words_left[l].upper())
                                        poss_ss = w2v_scores[wv1][wv3]
                                        possList.append(poss_ss)
                                        
                                    ptile_rank = percentile_rank_S(ss, possList)
                                    if ptile_rank is not None:
                                        sub_tot_mat[s_idx] += ptile_rank
                                        sub_cnt_mat[s_idx] += 1
                                        
            scl_mat[n] = np.mean(sub_tot_mat / sub_cnt_mat)          # subject average across sessions
            
        # calculate mean and standard error across subjects
        scl_dict.update({cond_str: np.nanmean(scl_mat)})
        se_dict.update({cond_str: np.nanstd(scl_mat) / np.sqrt(len(scl_mat))})
        sub_dict.update({cond_str: scl_mat})
    
    return scl_dict, se_dict, sub_dict

def plot_semantic_cl(scl_dict, se_dict):
    conds = ['10-2', '20-1', '15-2', '30-1', '20-2', '40-1']
    colors = ['firebrick', 'darkorange', 'forestgreen', 'dodgerblue', 'midnightblue', 'darkviolet']
    scl = []
    se = []
    for c in conds:
        scl.append(scl_dict.get(c))
        se.append(se_dict.get(c))
    
    fig, ax = plt.subplots(figsize=(7,5))
    ax.bar(conds, scl, yerr=se, width=0.65, color=colors)
    ax.set(xlabel='Condition', ylabel='Temporal Clutering Score', ylim=(0.4, 0.6))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()
    
def compare_semantic_cl(scl_prim_dict, scl_prim_se, scl_prim_sub, scl_ns_dict, scl_ns_se, scl_ns_sub, scl_rec_dict, scl_rec_se, scl_rec_sub):
    conds = ['10-2', '20-1', '15-2', '30-1', '20-2', '40-1']
    fig, ax = plt.subplots(2, 3, figsize=(9, 6), sharex=True, sharey=True)
    for i, c in enumerate(conds):
        scl_prim = scl_prim_dict.get(c)
        se_prim = scl_prim_se.get(c)
        sub_prim = scl_prim_sub.get(c)
        
        scl_ns = scl_ns_dict.get(c)
        se_ns = scl_ns_se.get(c)
        sub_ns = scl_ns_sub.get(c)
        
        scl_rec = scl_rec_dict.get(c)
        se_rec = scl_rec_se.get(c)
        sub_rec = scl_rec_sub.get(c)
        
        #ts, pvals = scipy.stats.ttest_ind(sub_prim, sub_rec, equal_var=False, alternative='two-sided')
        #os_t, os_p = scipy.stats.ttest_1samp(sub_prim, 0.5)
        
        ax[i%2, i//2].bar([1,2,3], [scl_prim, scl_ns, scl_rec], yerr=[se_prim, se_ns, se_rec], width=0.65, color=['orange', 'darkgray', 'purple'])
        ax[i%2, i//2].set(xticks=[1,2,3], xticklabels=['Primacy', 'No Strategy', 'Recency'], title=c, ylim=(0.48, 0.56))
        ax[i%2, i//2].spines['right'].set_visible(False)
        ax[i%2, i//2].spines['top'].set_visible(False)
        
    fig.supylabel('Semantic Clustering Score')
    ax[1, 1].set(xlabel='Group')
    plt.tight_layout()
    plt.show()
    
def compare_semantic_cl_stats(scl_prim_dict, scl_prim_se, scl_prim_sub, scl_ns_dict, scl_ns_se, scl_ns_sub, scl_rec_dict, scl_rec_se, scl_rec_sub):
    conds = ['10-2', '20-1', '15-2', '30-1', '20-2', '40-1']
    fig, ax = plt.subplots(2, 3, figsize=(9, 6), sharex=True, sharey=True)
    for i, c in enumerate(conds):
        scl_prim = scl_prim_dict.get(c)
        se_prim = scl_prim_se.get(c)
        sub_prim = scl_prim_sub.get(c)
        
        scl_ns = scl_ns_dict.get(c)
        se_ns = scl_ns_se.get(c)
        sub_ns = scl_ns_sub.get(c)
        
        scl_rec = scl_rec_dict.get(c)
        se_rec = scl_rec_se.get(c)
        sub_rec = scl_rec_sub.get(c)
        
        # two-sample t-tests between strategy groups
        ts_prim_rec, pval_prim_rec = scipy.stats.ttest_ind(sub_prim, sub_rec, equal_var=False, alternative='two-sided')
        ts_prim_ns, pval_prim_ns = scipy.stats.ttest_ind(sub_prim, sub_ns, equal_var=False, alternative='two-sided')
        ts_ns_rec, pval_ns_rec = scipy.stats.ttest_ind(sub_ns, sub_rec, equal_var=False, alternative='two-sided')
        
        # one-sample t-test against chance
        os_ts_prim, os_pval_prim = scipy.stats.ttest_1samp(sub_prim, 0.5, alternative='greater')
        os_ts_ns, os_pval_ns = scipy.stats.ttest_1samp(sub_ns, 0.5, alternative='greater')
        os_ts_rec, os_pval_rec = scipy.stats.ttest_1samp(sub_rec, 0.5, alternative='greater')
        
        h1 = np.max(np.concatenate((sub_prim, sub_ns, sub_rec))) + 0.02
        h2 = h1 + 0.015
        h3 = h2 + 0.015
        
        #print(np.max(np.concatenate((sub_prim, sub_ns, sub_rec))))
        #print(np.min(np.concatenate((sub_prim, sub_ns, sub_rec))))
        
        if pval_prim_rec < 0.05:
            ax[i%2, i//2].plot([1,3], [h2, h2], color='midnightblue')
        if pval_prim_ns < 0.05:
            ax[i%2, i//2].plot([1.1,1.9], [h1, h1], color='saddlebrown')
        if pval_ns_rec < 0.05:
            ax[i%2, i//2].plot([2.1,2.9], [h1, h1], color='saddlebrown')
            
        if os_pval_prim < 0.05:
            ax[i%2, i//2].plot([1], [h3], marker='*', color='orange')
        if os_pval_ns < 0.05:
            ax[i%2, i//2].plot([2], [h3], marker='*', color='darkgray')
        if os_pval_rec < 0.05:
            ax[i%2, i//2].plot([3], [h3], marker='*', color='purple')
        
        #ts, pvals = scipy.stats.ttest_ind(sub_prim, sub_rec, equal_var=False, alternative='two-sided')
        #os_t, os_p = scipy.stats.ttest_1samp(sub_prim, 0.5)
        
        ax[i%2, i//2].bar([1,2,3], [scl_prim, scl_ns, scl_rec], yerr=[1.96*se_prim, 1.96*se_ns, 1.96*se_rec], width=0.65, color=['orange', 'darkgray', 'purple'], alpha=0.6)
        ax[i%2, i//2].scatter(np.full(len(sub_prim), 1), sub_prim, s=5, color='orange', alpha=0.9)
        ax[i%2, i//2].scatter(np.full(len(sub_ns), 2), sub_ns, s=5, color='darkgray', alpha=0.9)
        ax[i%2, i//2].scatter(np.full(len(sub_rec), 3), sub_rec, s=5, color='purple', alpha=0.9)
        ax[i%2, i//2].set(xticks=[1,2,3], yticks=[0.4, 0.45, 0.5, 0.55, 0.6], xticklabels=['Primacy', 'No Strategy', 'Recency'], title=c, ylim=(0.39, 0.67))
        ax[i%2, i//2].spines['right'].set_visible(False)
        ax[i%2, i//2].spines['top'].set_visible(False)
        
    fig.supylabel('Semantic Clustering Score')
    ax[1, 1].set(xlabel='Group')
    plt.tight_layout()
    plt.show()
    

# ---------- Temporal and Semantic Clustering Correlation ----------

def plot_cl_corr(tcl_prim_sub, tcl_ns_sub, tcl_rec_sub, scl_prim_sub, scl_ns_sub, scl_rec_sub):
    conds = ['10-2', '20-1', '15-2', '30-1', '20-2', '40-1']
    fig, ax = plt.subplots(2, 3, figsize=(11, 7), sharey=True, sharex=True)
    tcl_prim_all = np.array([]); scl_prim_all = np.array([])
    tcl_ns_all = np.array([]); scl_ns_all = np.array([])
    tcl_rec_all = np.array([]); scl_rec_all = np.array([])
    for i, c in enumerate(conds):
        tcl_prim = tcl_prim_sub.get(c)
        scl_prim = scl_prim_sub.get(c)
        corr_prim, pval_prim = scipy.stats.pearsonr(scl_prim, tcl_prim)
        print(corr_prim, pval_prim)
        
        tcl_ns = tcl_ns_sub.get(c)
        scl_ns = scl_ns_sub.get(c)
        corr_ns, pval_ns = scipy.stats.pearsonr(scl_ns, tcl_ns)
        print(corr_ns, pval_ns)
        
        tcl_rec = tcl_rec_sub.get(c)
        scl_rec = scl_rec_sub.get(c)
        corr_rec, pval_rec = scipy.stats.pearsonr(scl_rec, tcl_rec)
        print(corr_rec, pval_rec)
        
        # all subjects
        tcl = np.concatenate((tcl_prim, tcl_ns, tcl_rec))
        scl = np.concatenate((scl_prim, scl_ns, scl_rec))
        corr, pval = scipy.stats.pearsonr(scl, tcl)
        print(corr, pval)
        
        # pool conditions
        tcl_prim_all = np.concatenate((tcl_prim_all, tcl_prim))
        scl_prim_all = np.concatenate((scl_prim_all, scl_prim))
        
        tcl_ns_all = np.concatenate((tcl_ns_all, tcl_ns))
        scl_ns_all = np.concatenate((scl_ns_all, scl_ns))
        
        tcl_rec_all = np.concatenate((tcl_rec_all, tcl_rec))
        scl_rec_all = np.concatenate((scl_rec_all, scl_rec))
        
        ax[i%2, i//2].scatter(scl_prim, tcl_prim, color='orange', alpha=0.6, label='primacy')
        ax[i%2, i//2].scatter(scl_ns, tcl_ns, color='darkgray', alpha=0.6, label='no strategy')
        ax[i%2, i//2].scatter(scl_rec, tcl_rec, color='purple', alpha=0.6, label='recency')
        ax[i%2, i//2].set(xlabel='Semantic Clustering Score', ylabel='Temporal Clustering Score', title=c)
        ax[i%2, i//2].spines['right'].set_visible(False)
        ax[i%2, i//2].spines['top'].set_visible(False)
        
    plt.tight_layout()
    plt.show()
    
    return tcl_prim_all, scl_prim_all, tcl_ns_all, scl_ns_all, tcl_rec_all, scl_rec_all

def plot_cl_corr_all(scl_prim_all, tcl_prim_all, scl_ns_all, tcl_ns_all, scl_rec_all, tcl_rec_all, alt):
    # calculate correlations
    r_prim, p_prim = scipy.stats.pearsonr(scl_prim_all, tcl_prim_all, alternative=alt)
    r_ns, p_ns = scipy.stats.pearsonr(scl_ns_all, tcl_ns_all, alternative=alt)
    r_rec, p_rec = scipy.stats.pearsonr(scl_rec_all, tcl_rec_all, alternative=alt)
    
    # boostrap -- resample and calculate correlations to get distribution of correlations for SE/95% confidence
    n_boots = 1000
    corrs_prim = []; corrs_ns = []; corrs_rec = []
    #pvals_prim = []; pvals_ns = []; pvals_rec = []
    for n in range(n_boots):     # bootstrap 100 times
        idx_prim = np.random.randint(0, len(scl_prim_all), len(scl_prim_all))
        r_prim_, p_prim_ = scipy.stats.pearsonr(scl_prim_all[idx_prim], tcl_prim_all[idx_prim], alternative=alt)
        corrs_prim.append(r_prim_)
        #pvals_prim.append(p_prim_)
        
        idx_ns = np.random.randint(0, len(scl_ns_all), len(scl_ns_all))
        r_ns_, p_ns_ = scipy.stats.pearsonr(scl_ns_all[idx_ns], tcl_ns_all[idx_ns], alternative=alt)
        corrs_ns.append(r_ns_)
        #pvals_ns.append(p_ns_)
        
        idx_rec = np.random.randint(0, len(scl_rec_all), len(scl_rec_all))
        r_rec_, p_rec_ = scipy.stats.pearsonr(scl_rec_all[idx_rec], tcl_rec_all[idx_rec], alternative=alt)
        corrs_rec.append(r_rec_)
        #pvals_rec.append(p_rec_)
    
    # sort and find 95% confidence interval
    corrs_prim.sort(); corrs_ns.sort(); corrs_rec.sort()
    ci_prim = np.array([[r_prim - corrs_prim[24]], 
                        [corrs_prim[974] - r_prim]])
    ci_ns = np.array([[r_ns - corrs_ns[24]], 
                      [corrs_ns[974] - r_ns]])
    ci_rec = np.array([[r_rec - corrs_rec[24]], 
                       [corrs_rec[974] - r_rec]])
    
    # plotting
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [2.3, 1]})
    ax[0].scatter(scl_prim_all, tcl_prim_all, c='orange', alpha=0.6)
    ax[0].scatter(scl_ns_all, tcl_ns_all, c='darkgray', alpha=0.6)
    ax[0].scatter(scl_rec_all, tcl_rec_all, c='purple', alpha=0.6)
    ax[0].set(xlabel='Semantic Clustering Score', ylabel='Temporal Clustering Score')
    ax[0].spines['right'].set_visible(False); ax[0].spines['top'].set_visible(False)
    
    ax[1].scatter([1], [r_prim], color='orange')
    ax[1].errorbar([1], [r_prim], yerr = ci_prim, color='orange')
    ax[1].scatter([2], [r_ns], color='darkgray')
    ax[1].errorbar([2], [r_ns], yerr = ci_ns, color='darkgray')
    ax[1].scatter([3], [r_rec], color='purple')
    ax[1].errorbar([3], [r_rec], yerr = ci_rec, color='purple')
    
    # statistical significance
    h = np.max([corrs_prim[974], corrs_ns[974], corrs_rec[974]]) + 0.05
    if p_prim < 0.05:
        ax[1].plot([1], [h], marker='*', color='orange')
    if p_ns < 0.05:
        ax[1].plot([2], [h], marker='*', color='darkgray')
    if p_rec < 0.05:
        ax[1].plot([3], [h], marker='*', color='purple')
    
    ax[1].set(xlabel='Strategy Group', xticks = [1,2,3], xticklabels=['Primacy', 'No Strategy', 'Recency'], ylabel='Correlation')
    ax[1].spines['right'].set_visible(False); ax[1].spines['top'].set_visible(False)
    
    plt.tight_layout()
    plt.show()

# ---------- PLI Recency ----------

def pli_recency(df):
    plirec_dict = {}                  # dictionary mapping condition to pli recency
    se_dict = {}                      # dictionary mapping condition to standard error
    
    # iterate over conditions
    for c in df.condition.unique():
        cond_data, list_length, presentation_rate, cond_str = condition_metadata(df, c)
        
        plirec_mat = np.zeros((len(cond_data.worker_id.unique()), 5), dtype=float)     # matrix for subject level pli recency proportions
        # iterate over subjects within condition
        for n, subj in enumerate(cond_data.worker_id.unique()):
            sub_data = cond_data[cond_data['worker_id'] == subj]
            
            sub_mat = np.zeros((len(sub_data.session.unique()), 5), dtype = float)
            # iterate over subject's sessions
            for s_idx, s in enumerate(sub_data.session.unique()):
                sess_data = sub_data[sub_data['session'] == s]
                word_evs = sess_data[sess_data['type'] == 'WORD']; rec_evs = sess_data[sess_data['type'] == 'REC_WORD']
                wl_dict = dict(zip([w.upper() for w in word_evs.word], word_evs.list))                     # dictionary mapping words to list
                intrusions = rec_evs[(rec_evs['list'] >= 5) & (rec_evs['serial_position'] == 88)]     # intrusions to analyze (after 1st 5 lists)
                intrusions_rec = intrusions['rec_word'].to_numpy()                                    # intrusion recalls
                intrusions_lst = intrusions['list'].to_numpy()                                        # intrusion lists
                sess_tot_PLI = 0                                                                      # total PLIs for session
                
                # iterate over intrusions, find PLIs
                for i, intr in enumerate(intrusions_rec):
                    if intr.upper() in wl_dict.keys():         # presented word
                        l = wl_dict.get(intr.upper())          # list of word presentation
                        if l < intrusions_lst[i]:              # PLI
                            sess_tot_PLI += 1
                            lr = int(intrusions_lst[i] - l)   # list recency
                            if lr > 0 and lr <= 5:
                                sub_mat[s_idx, lr-1] += 1
                                
                sub_mat[s_idx, :] /= sess_tot_PLI                         # divide by total PLIs for proportion, want NaNs from zero division
                
            plirec_mat[n, :] = np.nanmean(sub_mat, axis=0)                # subject average
            
        plirec_dict.update({cond_str: np.nanmean(plirec_mat, axis=0)})
        se_dict.update({cond_str: np.nanstd(plirec_mat, axis=0, ddof=1) / np.sqrt(plirec_mat.shape[0])})
        
    return plirec_dict, se_dict

def plot_pli_recency(plirec_dict, se_dict):
    conds = ['10-2', '20-1', '15-2', '30-1', '20-2', '40-1']
    colors = ['firebrick', 'darkorange', 'forestgreen', 'dodgerblue', 'midnightblue', 'darkviolet']
    fig, ax = plt.subplots(figsize=(5,5))
    x = np.linspace(1, 5, 5)
    for i, c in enumerate(conds):
        plirec = plirec_dict.get(c)
        se = se_dict.get(c)
        ax.plot(x, plirec, color=colors[i], label=c)
        ax.fill_between(x, plirec-1.96*se, plirec+1.96*se, color=colors[i], alpha=0.1)
        
    ax.set(xlabel='List Recency', ylabel='Proportion of Prior List Intrusions', xticks=x)
    ax.legend()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()
    
def compare_pli_recency(plirec_prim_dict, plirec_prim_se, plirec_ns_dict, plirec_ns_se, plirec_rec_dict, plirec_rec_se):
    conds = ['10-2', '20-1', '15-2', '30-1', '20-2', '40-1']
    fig, ax = plt.subplots(2, 3, figsize=(9,6), sharey=True, sharex=True)
    x = np.linspace(1, 5, 5)
    for i, c in enumerate(conds):
        plirec_prim = plirec_prim_dict.get(c)
        se_prim = plirec_prim_se.get(c)
        
        plirec_ns = plirec_ns_dict.get(c)
        se_ns = plirec_ns_se.get(c)
        
        plirec_rec = plirec_rec_dict.get(c)
        se_rec = plirec_rec_se.get(c)
        
        ax[i%2, i//2].plot(x, plirec_prim, color='orange', label='primacy')
        ax[i%2, i//2].fill_between(x, plirec_prim-1.96*se_prim, plirec_prim+1.96*se_prim, color='orange', alpha=0.1)
        ax[i%2, i//2].plot(x, plirec_ns, color='darkgray', label='no strategy')
        ax[i%2, i//2].fill_between(x, plirec_ns-1.96*se_ns, plirec_ns+1.96*se_ns, color='darkgray', alpha=0.1)
        ax[i%2, i//2].plot(x, plirec_rec, color='purple', label='recency')
        ax[i%2, i//2].fill_between(x, plirec_rec-1.96*se_rec, plirec_rec+1.96*se_rec, color='purple', alpha=0.1)
        ax[i%2, i//2].set(xticks = x, title=c)
        ax[i%2, i//2].spines['right'].set_visible(False)
        ax[i%2, i//2].spines['top'].set_visible(False)
        
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=3, shadow=True, bbox_to_anchor=(0.76, 1.06))
    ax[1, 1].set(xlabel='List Recency')
    fig.supylabel('Proportion of PLIs')
    plt.tight_layout()
    plt.show()
                        
        
# ---------- Intrusion Rates ----------

def intrusion_rates(df):
    ir_dict = {}                         # dictionary maping condition to intrusion rates
    se_dict = {}                         # dictionary mapping condition to standard error
    sub_dict = {}                        # dictionary mapping condition to subject scores
    
    # iterate over conditions
    for c in df.condition.unique():
        cond_data, list_length, presentation_rate, cond_str = condition_metadata(df, c)

        ir_mat = np.zeros((len(cond_data.worker_id.unique()), 3), dtype=float)        # matrix for subject level intrusion rates
        # iterate over subjects within condition
        for n, subj in enumerate(cond_data.worker_id.unique()):
            sub_data = cond_data[cond_data['worker_id'] == subj]

            sub_mat = np.zeros((len(sub_data.session.unique()), 3), dtype=float)
            # iterate over subject's sessions
            for s_idx, s in enumerate(sub_data.session.unique()):
                sess_data = sub_data[sub_data['session'] == s]
                word_evs = sess_data[sess_data['type'] == 'WORD']; rec_evs = sess_data[sess_data['type'] == 'REC_WORD']
                wl_dict = dict(zip([w.upper() for w in word_evs.word], word_evs.list))                       # dictionary mapping words to list

                sess_tot_ELI = 0; sess_tot_PLI = 0; sess_tot_rep = 0                    # total ELIs, PLIs, repetitions for session

                for i in sess_data.list.unique():
                    if i >= 5:                                                          # only look at lists 5-11 (omit first 5 lists for PLIs)
                        sp = rec_evs[rec_evs['list'] == i].serial_position.to_numpy()       # current list serial positions
                        sp = sp[sp != 99]                                                   # remove last null recall
                        r = rec_evs[rec_evs['list'] == i].rec_word.tolist()                 # current list recalls
                        r = r[:len(sp)]

                        sp = mark_repetitions(sp)

                        for j in range(len(sp)):
                            if sp[j] == 77:                             # repetition
                                sess_tot_rep += 1
                            elif sp[j] == 88:
                                if r[j].upper() in wl_dict.keys():      # presented word
                                    l = wl_dict.get(r[j].upper())       # list of word presentation
                                    if l < i:                           # PLI
                                        sess_tot_PLI += 1
                                    elif l > i:
                                        sess_tot_ELI += 1               # ELI (presented on future list)
                                else:
                                    sess_tot_ELI += 1                   # ELI (not presented)

                sub_mat[s_idx, :] = np.array([sess_tot_ELI, sess_tot_PLI, sess_tot_rep]) / (len(sess_data.list.unique()) - 5)

            ir_mat[n, :] = np.mean(sub_mat, axis=0)                 # subject average

        ir_dict.update({cond_str: np.nanmean(ir_mat, axis=0)})
        se_dict.update({cond_str: np.nanstd(ir_mat, axis=0, ddof=1) / np.sqrt(ir_mat.shape[0])})
        sub_dict.update({cond_str: ir_mat})
        
    return ir_dict, se_dict, sub_dict

def plot_intrusion_rates(ir_dict, se_dict):
    conds = ['10-2', '20-1', '15-2', '30-1', '20-2', '40-1']
    colors = ['firebrick', 'darkorange', 'forestgreen', 'dodgerblue', 'midnightblue', 'darkviolet']
    fig, ax = plt.subplots(2, 3, figsize=(12,8), sharey=True)
    for i, c in enumerate(conds):
        ir = ir_dict.get(c)
        se = se_dict.get(c)
        ax[i%2, i//2].bar([1,2,3], ir, yerr=se, width=0.65, color=colors[i], hatch=['-', 'X', 'oo'])
        ax[i%2, i//2].set(xlabel='Intrusion Type', ylabel='Number Per List', xticks=[1,2,3], xticklabels=['ELI', 'PLI', 'Repetition'], title=conds[i])
        ax[i%2, i//2].spines['right'].set_visible(False)
        ax[i%2, i//2].spines['top'].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
def compare_intrusion_rates(ir_prim_dict, ir_prim_se, ir_prim_sub, ir_ns_dict, ir_ns_se, ir_ns_sub, ir_rec_dict, ir_rec_se, ir_rec_sub, typ):
    conds = ['10-2', '20-1', '15-2', '30-1', '20-2', '40-1']
    fig, ax = plt.subplots(2, 3, figsize=(10, 6), sharey=True, sharex=True)
    for i, c in enumerate(conds):
        ir_prim = ir_prim_dict.get(c)
        se_prim = ir_prim_se.get(c)
        sub_prim = ir_prim_sub.get(c)
        
        ir_ns = ir_ns_dict.get(c)
        se_ns = ir_ns_se.get(c)
        sub_ns = ir_ns_sub.get(c)
        
        ir_rec = ir_rec_dict.get(c)
        se_rec = ir_rec_se.get(c)
        sub_rec = ir_rec_sub.get(c)
        
        ts, pvals = scipy.stats.ttest_ind(sub_prim, sub_rec, axis=0, equal_var=False, alternative='two-sided')
        
        if typ=='ELI':
            ax[i%2, i//2].bar([1,2,3], [ir_prim[0], ir_ns[0], ir_rec[0]], yerr=[1.96*se_prim[0], 1.96*se_ns[0], 1.96*se_rec[0]], width=0.65, 
                              color=['orange', 'darkgray', 'purple'])
            ax[i%2, i//2].set(xticks=[1,2,3], xticklabels=['Primacy', 'No Strategy', 'Recency'], title=c)   
        elif typ=='PLI':
            ax[i%2, i//2].bar([1,2,3], [ir_prim[1], ir_ns[1], ir_rec[1]], yerr=[1.96*se_prim[1], 1.96*se_ns[1], 1.96*se_rec[1]], width=0.65, 
                              color=['orange', 'darkgray', 'purple'])
            ax[i%2, i//2].set(xticks=[1,2,3], xticklabels=['Primacy', 'No Strategy', 'Recency'], title=c)
        elif typ=='Rep':
            ax[i%2, i//2].bar([1,2,3], [ir_prim[2], ir_ns[2], ir_rec[2]], yerr=[1.96*se_prim[2], 1.96*se_ns[2], 1.96*se_rec[2]], width=0.65, 
                              color=['orange', 'darkgray', 'purple'])
            ax[i%2, i//2].set(xticks=[1,2,3], xticklabels=['Primacy', 'No Strategy', 'Recency'], title=c)
           
        ax[i%2, i//2].spines['right'].set_visible(False)
        ax[i%2, i//2].spines['top'].set_visible(False)
        
    if typ=='ELI':
        fig.supylabel('ELIs per List')
    elif typ=='PLI':
        fig.supylabel('PLIs per List')
    elif typ=='Rep':
        fig.supylabel('Repetitions per List')
    
    ax[1, 1].set(xlabel='Group')
    plt.tight_layout()
    plt.show()
    
def compare_intrusion_rates_stats(ir_prim_dict, ir_prim_se, ir_prim_sub, ir_ns_dict, ir_ns_se, ir_ns_sub, ir_rec_dict, ir_rec_se, ir_rec_sub, typ):
    conds = ['10-2', '20-1', '15-2', '30-1', '20-2', '40-1']
    fig, ax = plt.subplots(2, 3, figsize=(10, 6), sharey=True, sharex=True)
    for i, c in enumerate(conds):
        ir_prim = ir_prim_dict.get(c)
        se_prim = ir_prim_se.get(c)
        sub_prim = ir_prim_sub.get(c)
        
        ir_ns = ir_ns_dict.get(c)
        se_ns = ir_ns_se.get(c)
        sub_ns = ir_ns_sub.get(c)
        
        ir_rec = ir_rec_dict.get(c)
        se_rec = ir_rec_se.get(c)
        sub_rec = ir_rec_sub.get(c)
        
        ts_prim_rec, pval_prim_rec = scipy.stats.ttest_ind(sub_prim, sub_rec, axis=0, equal_var=False, alternative='two-sided')
        ts_prim_ns, pval_prim_ns = scipy.stats.ttest_ind(sub_prim, sub_ns, axis=0, equal_var=False, alternative='two-sided')
        ts_ns_rec, pval_ns_rec = scipy.stats.ttest_ind(sub_ns, sub_rec, axis=0, equal_var=False, alternative='two-sided')    
        
        if typ=='ELI':
            lim = 5.5
            #h1 = np.max(np.concatenate((sub_prim, sub_ns, sub_rec), axis=0)) + 0.5
            h1 = np.max(np.array([ir_prim[0]+1.96*se_prim[0], ir_ns[0]+1.96*se_ns[0], ir_rec[0]+1.96*se_rec[0]])) + 1
            h2 = h1 + 0.5
            if pval_prim_rec[0] < 0.05:
                ax[i%2, i//2].plot([1,3], [h2, h2], color='midnightblue')
            if pval_prim_ns[0] < 0.05:
                ax[i%2, i//2].plot([1.1,1.9], [h1, h1], color='saddlebrown')
            if pval_ns_rec[0] < 0.05:
                ax[i%2, i//2].plot([2.1,2.9], [h1, h1], color='saddlebrown')
            ax[i%2, i//2].bar([1,2,3], [ir_prim[0], ir_ns[0], ir_rec[0]], yerr=[1.96*se_prim[0], 1.96*se_ns[0], 1.96*se_rec[0]], width=0.65, 
                              color=['orange', 'darkgray', 'purple'], alpha=0.6)
            ax[i%2, i//2].scatter(np.full(len(sub_prim[:, 0]), 1), sub_prim[:, 0], s=5, color='orange', alpha=0.9)
            ax[i%2, i//2].scatter(np.full(len(sub_ns[:, 0]), 2), sub_ns[:, 0], s=5, color='darkgray', alpha=0.9)
            ax[i%2, i//2].scatter(np.full(len(sub_rec[:, 0]), 3), sub_rec[:, 0], s=5, color='purple', alpha=0.9)
            ax[i%2, i//2].set(xticks=[1,2,3], yticks=np.linspace(0,5,6), xticklabels=['Primacy', 'No Strategy', 'Recency'], title=c, ylim=[0,lim])
            if np.max(sub_prim[:,0]) > lim:
                ax[i%2, i//2].plot([1], [lim-0.1], color='orange', marker='P')
            if np.max(sub_ns[:,0]) > lim:
                ax[i%2, i//2].plot([2], [lim-0.1], color='darkgray', marker='P')
            if np.max(sub_rec[:,0]) > lim:
                ax[i%2, i//2].plot([3], [lim-0.1], color='purple', marker='P')
        elif typ=='PLI':
            lim = 1.5
            #h1 = np.max(np.concatenate((sub_prim, sub_ns, sub_rec), axis=0)) + 0.5
            h1 = np.max(np.array([ir_prim[1]+1.96*se_prim[1], ir_ns[1]+1.96*se_ns[1], ir_rec[1]+1.96*se_rec[1]])) + 0.25
            h2 = h1 + 0.125
            if pval_prim_rec[1] < 0.05:
                ax[i%2, i//2].plot([1,3], [h2, h2], color='midnightblue')
            if pval_prim_ns[1] < 0.05:
                ax[i%2, i//2].plot([1.1,1.9], [h1, h1], color='saddlebrown')
            if pval_ns_rec[1] < 0.05:
                ax[i%2, i//2].plot([2.1,2.9], [h1, h1], color='saddlebrown')
            ax[i%2, i//2].bar([1,2,3], [ir_prim[1], ir_ns[1], ir_rec[1]], yerr=[1.96*se_prim[1], 1.96*se_ns[1], 1.96*se_rec[1]], width=0.65, 
                              color=['orange', 'darkgray', 'purple'], alpha=0.6)
            ax[i%2, i//2].scatter(np.full(len(sub_prim[:, 1]), 1), sub_prim[:, 1], s=5, color='orange', alpha=0.9)
            ax[i%2, i//2].scatter(np.full(len(sub_ns[:, 1]), 2), sub_ns[:, 1], s=5, color='darkgray', alpha=0.9)
            ax[i%2, i//2].scatter(np.full(len(sub_rec[:, 1]), 3), sub_rec[:, 1], s=5, color='purple', alpha=0.9)
            ax[i%2, i//2].set(xticks=[1,2,3], yticks=np.linspace(0,1.4,8), xticklabels=['Primacy', 'No Strategy', 'Recency'], title=c, ylim=[0,lim])
            if np.max(sub_prim[:,1]) > lim:
                ax[i%2, i//2].plot([1], [lim-0.1], color='orange', marker='P')
            if np.max(sub_ns[:,1]) > lim:
                ax[i%2, i//2].plot([2], [lim-0.1], color='darkgray', marker='P')
            if np.max(sub_rec[:,1]) > lim:
                ax[i%2, i//2].plot([3], [lim-0.1], color='purple', marker='P')
        elif typ=='Rep':
            lim = 1.5
            #h1 = np.max(np.concatenate((sub_prim, sub_ns, sub_rec), axis=0)) + 0.5
            maxval = np.max(np.concatenate((sub_prim[:,2], sub_ns[:,2], sub_rec[:,2])))
            h1 = np.max(np.array([ir_prim[2]+1.96*se_prim[2], ir_ns[2]+1.96*se_ns[2], ir_rec[2]+1.96*se_rec[2]])) + 0.3
            h2 = h1 + 0.15
            if pval_prim_rec[2] < 0.05:
                ax[i%2, i//2].plot([1,3], [h2, h2], color='midnightblue')
            if pval_prim_ns[2] < 0.05:
                ax[i%2, i//2].plot([1.1,1.9], [h1, h1], color='saddlebrown')
            if pval_ns_rec[2] < 0.05:
                ax[i%2, i//2].plot([2.1,2.9], [h1, h1], color='saddlebrown')
            ax[i%2, i//2].bar([1,2,3], [ir_prim[2], ir_ns[2], ir_rec[2]], yerr=[1.96*se_prim[2], 1.96*se_ns[2], 1.96*se_rec[2]], width=0.65, 
                              color=['orange', 'darkgray', 'purple'], alpha=0.6)
            ax[i%2, i//2].scatter(np.full(len(sub_prim[:, 2]), 1), sub_prim[:, 1], s=5, color='orange', alpha=0.9)
            ax[i%2, i//2].scatter(np.full(len(sub_ns[:, 2]), 2), sub_ns[:, 1], s=5, color='darkgray', alpha=0.9)
            ax[i%2, i//2].scatter(np.full(len(sub_rec[:, 2]), 3), sub_rec[:, 1], s=5, color='purple', alpha=0.9)
            ax[i%2, i//2].set(xticks=[1,2,3], xticklabels=['Primacy', 'No Strategy', 'Recency'], title=c, ylim=[0,lim])
            if np.max(sub_prim[:,2]) > lim:
                ax[i%2, i//2].plot([1], [lim-0.1], color='orange', marker='P')
            if np.max(sub_ns[:,2]) > lim:
                ax[i%2, i//2].plot([2], [lim-0.1], color='darkgray', marker='P')
            if np.max(sub_rec[:,2]) > lim:
                ax[i%2, i//2].plot([3], [lim-0.1], color='purple', marker='P')
           
        ax[i%2, i//2].spines['right'].set_visible(False)
        ax[i%2, i//2].spines['top'].set_visible(False)
        
    if typ=='ELI':
        fig.supylabel('ELIs per List')
    elif typ=='PLI':
        fig.supylabel('PLIs per List')
    elif typ=='Rep':
        fig.supylabel('Repetitions per List')
    
    ax[1, 1].set(xlabel='Group')
    plt.tight_layout()
    plt.show() 
    

# ---------- Semantic Relatedness of Intrusions ----------

def sem_intrusions(df, wordpool, w2v_scores):
    semintr_dict = {}               # dictionary mapping condition to [possible, correct response, PLI, repetition]
    se_dict = {}                    # dictionary mapping condition to standard error
    sub_dict = {}                   # dictionary mapping condition to subject results

    # iterate over conditions
    for c in df.condition.unique():
        cond_data, list_length, presentation_rate, cond_str = condition_metadata(df, c)

        si_mat = np.zeros((len(cond_data.worker_id.unique()), 4), dtype=float)           # subject level matrix
        # iterate over subjects within condition
        for n, subj in enumerate(cond_data.worker_id.unique()):
            sub_data = cond_data[cond_data['worker_id'] == subj]

            sub_tot_mat = np.zeros((len(sub_data.session.unique()), 4), dtype=float)
            sub_cnt_mat = np.zeros((len(sub_data.session.unique()), 4), dtype=float)
            # iterate over subject's sessions
            for s_idx, s in enumerate(sub_data.session.unique()):
                sess_data = sub_data[sub_data['session'] == s]
                word_evs = sess_data[sess_data['type'] == 'WORD']; rec_evs = sess_data[sess_data['type'] == 'REC_WORD']
                wl_dict = dict(zip([w.upper() for w in word_evs.word], word_evs.list))   # dictionary mapping words to list
                
                for i in sess_data.list.unique():
                    sp = rec_evs[rec_evs['list'] == i].serial_position.to_numpy()        # current list serial positions
                    sp = sp[sp != 99]                                                    # remove last null recall
                    w = word_evs[word_evs['list'] == i].word.tolist()                      # current list words
                    r = rec_evs[rec_evs['list'] == i].rec_word.tolist()                  # current list recalls
                    r = r[:len(sp)]

                    sp = mark_repetitions(sp)

                    for j in range(len(sp) - 1):
                        if sp[j]!=88 and sp[j]!=77:                    # current recall is correct response
                            if sp[j+1]!=88 and sp[j+1]!=77:            # transition to correct response
                                if r[j].upper() in wordpool and r[j+1].upper() in wordpool:
                                    wv1 = wordpool.index(r[j].upper())
                                    wv2 = wordpool.index(r[j+1].upper())
                                    ss = w2v_scores[wv1][wv2]
                                    sub_tot_mat[s_idx, 1] += ss
                                    sub_cnt_mat[s_idx, 1] += 1
                            elif sp[j+1]==88:                          # transition to intrusion
                                if r[j+1].upper() in wl_dict.keys():   # presented word
                                    l = wl_dict.get(r[j+1].upper())    # list of word presentation
                                    if l < i and r[j].upper() in wordpool and r[j+1].upper() in wordpool:       # PLI
                                        wv1 = wordpool.index(r[j].upper())
                                        wv2 = wordpool.index(r[j+1].upper())
                                        ss = w2v_scores[wv1][wv2]
                                        sub_tot_mat[s_idx, 2] += ss
                                        sub_cnt_mat[s_idx, 2] += 1
                            elif sp[j+1]==77:                          # transition to repetition
                                if r[j].upper() in wordpool and r[j+1].upper() in wordpool:
                                    wv1 = wordpool.index(r[j].upper())
                                    wv2 = wordpool.index(r[j+1].upper())
                                    if wv1 != wv2:                     # don't include immediate repititions
                                        ss = w2v_scores[wv1][wv2]
                                        sub_tot_mat[s_idx, 3] += ss
                                        sub_cnt_mat[s_idx, 3] += 1

                    # possible transitions
                    words_left = [x.upper() for x in w]
                    for k in range(len(sp) -1):
                        if sp[k]!=88 and sp[k]!=77 and r[k].upper() in wordpool:
                            words_left.remove(r[k].upper())
                            for l in words_left:
                                wv1 = wordpool.index(r[k].upper())
                                wv2 = wordpool.index(l.upper())
                                poss_ss = w2v_scores[wv1][wv2]
                                sub_tot_mat[s_idx, 0] += poss_ss
                                sub_cnt_mat[s_idx, 0] += 1

            si_mat[n, :] = np.mean((sub_tot_mat / sub_cnt_mat), axis=0)

        semintr_dict.update({cond_str: np.nanmean(si_mat, axis=0)})
        se_dict.update({cond_str: np.nanstd(si_mat, axis=0, ddof=1) / np.sqrt(si_mat.shape[0])})
        sub_dict.update({cond_str: si_mat})
        
    return semintr_dict, se_dict, sub_dict

def plot_sem_intrusions(semintr_dict, se_dict):
    conds = ['10-2', '20-1', '15-2', '30-1', '20-2', '40-1']
    colors = ['firebrick', 'darkorange', 'forestgreen', 'dodgerblue', 'midnightblue', 'darkviolet']
    fig, ax = plt.subplots(2, 3, figsize=(12,8), sharey=True)
    for i, c in enumerate(conds):
        semintr = semintr_dict.get(c)
        se = se_dict.get(c)
        ax[i%2, i//2].bar([1,2,3,4], semintr, yerr=se, width=0.65, color=colors[i], hatch=['-', 'X', 'oo', '++'])
        ax[i%2, i//2].set(xlabel='Transition Type', ylabel='Number Per List', xticks=[1,2,3,4], xticklabels=['Possible', 'Correct', 'PLI', 'Repetition'], title=conds[i], ylim=(0.08, 0.16))
        ax[i%2, i//2].spines['right'].set_visible(False)
        ax[i%2, i//2].spines['top'].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
def compare_sem_intrusions_stats(semintr_prim_sub, semintr_ns_sub, semintr_rec_sub):
    conds = ['10-2', '20-1', '15-2', '30-1', '20-2', '40-1']
    fig, ax = plt.subplots(2, 3, figsize=(10, 6), sharey=True, sharex=True)
    for i, c in enumerate(conds):
        sub_prim = semintr_prim_sub.get(c)
        sub_prim_diff = sub_prim[:, 2] - sub_prim[:, 1]         # subtract PLI from correct
        #print(sub_prim_diff)
        mean_prim_diff = np.nanmean(sub_prim_diff)
        se_prim_diff = np.nanstd(sub_prim_diff, ddof=1) / np.sqrt(np.count_nonzero(~np.isnan(sub_prim_diff)))

        sub_ns = semintr_ns_sub.get(c)
        sub_ns_diff = sub_ns[:, 2] - sub_ns[:, 1]
        #print(sub_ns_diff)
        mean_ns_diff = np.nanmean(sub_ns_diff)
        se_ns_diff = np.nanstd(sub_ns_diff, ddof=1) / np.sqrt(np.count_nonzero(~np.isnan(sub_ns_diff)))

        sub_rec = semintr_rec_sub.get(c)
        sub_rec_diff = sub_rec[:, 2] - sub_rec[:, 1]
        #print(sub_rec_diff)
        mean_rec_diff = np.nanmean(sub_rec_diff)
        se_rec_diff = np.nanstd(sub_rec_diff, ddof=1) / np.sqrt(np.count_nonzero(~np.isnan(sub_rec_diff)))
        
        #print(np.nanmax(np.concatenate((sub_prim_diff, sub_ns_diff, sub_rec_diff))))
        #print(np.nanmin(np.concatenate((sub_prim_diff, sub_ns_diff, sub_rec_diff))))

        # two-sample t-tests between strategy groups
        ts_prim_rec, pval_prim_rec = scipy.stats.ttest_ind(sub_prim_diff, sub_rec_diff, nan_policy='omit', equal_var=False, alternative='two-sided')
        ts_prim_ns, pval_prim_ns = scipy.stats.ttest_ind(sub_prim_diff, sub_ns_diff, nan_policy='omit', equal_var=False, alternative='two-sided')
        ts_ns_rec, pval_ns_rec = scipy.stats.ttest_ind(sub_ns_diff, sub_rec_diff, nan_policy='omit', equal_var=False, alternative='two-sided')

        # one-sample t-test against no difference
        os_ts_prim, os_pval_prim = scipy.stats.ttest_1samp(sub_prim_diff, 0, nan_policy='omit', alternative='greater')
        os_ts_ns, os_pval_ns = scipy.stats.ttest_1samp(sub_ns_diff, 0, nan_policy='omit', alternative='greater')
        os_ts_rec, os_pval_rec = scipy.stats.ttest_1samp(sub_rec_diff, 0, nan_policy='omit', alternative='greater')
        
        #print(pval_prim_rec, pval_prim_ns, pval_ns_rec)
        #print(os_pval_prim, os_pval_ns, os_pval_rec)

        h1 = np.nanmax(np.concatenate((sub_prim_diff, sub_ns_diff, sub_rec_diff))) + 0.003
        h2 = h1 + 0.004
        h3 = h2 + 0.004
        
        if pval_prim_rec < 0.05:
            ax[i%2, i//2].plot([1, 3], [h2, h2], color='midnightblue')
        if pval_prim_ns < 0.05:
            ax[i%2, i//2].plot([1.1, 1.9], [h1, h1], color='saddlebrown')
        if pval_ns_rec < 0.05:
            ax[i%2, i//2].plot([2.1, 2.9], [h1, h1], color='saddlebrown')

        if os_pval_prim < 0.05:
            ax[i%2, i//2].plot([1], [h3], marker='*', color='orange')
        if os_pval_ns < 0.05:
            ax[i%2, i//2].plot([2], [h3], marker='*', color='darkgray')
        if os_pval_rec < 0.05:
            ax[i%2, i//2].plot([3], [h3], marker='*', color='purple')

        ax[i%2, i//2].bar([1,2,3], [mean_prim_diff, mean_ns_diff, mean_rec_diff], yerr=[1.96*se_prim_diff, 1.96*se_ns_diff, 1.96*se_rec_diff], 
                          width=0.65, color=['orange', 'darkgray', 'purple'], alpha=0.6)
        ax[i%2, i//2].scatter(np.full(len(sub_prim_diff), 1), sub_prim_diff, s=5, color='orange', alpha=0.9)
        ax[i%2, i//2].scatter(np.full(len(sub_ns_diff), 2), sub_ns_diff, s=5, color='darkgray', alpha=0.9)
        ax[i%2, i//2].scatter(np.full(len(sub_rec_diff), 3), sub_rec_diff, s=5, color='purple', alpha=0.9)
        ax[i%2, i//2].set(xticks=[1,2,3], xticklabels=['Primacy', 'No Strategy', 'Recency'], title=c)
        ax[i%2, i//2].spines['right'].set_visible(False)
        ax[i%2, i//2].spines['top'].set_visible(False)

    fig.supylabel('Difference in Semantic Relatedness: PLI - Correct')
    plt.tight_layout()
    plt.show()
    
# ---------- Recall Initiation ----------

def prim_rec_sub(df, bufferStart, bufferEnd):
    prim_rec_dict = {}
    prim_rec_se = {}
    prim_rec_sub = {}
    
    for c in df.condition.unique():
        cond_data, list_length, presentation_rate, cond_str = condition_metadata(df, c)
        sub_prim_rec_dict = {}                                          # dictionary mapping subject to # primacy/recency lists
        sub_prim_rec_mat = np.zeros((len(cond_data.worker_id.unique()), 4), dtype=float)
        
        for n, subj in enumerate(cond_data.worker_id.unique()):
            sub_data = cond_data[cond_data['worker_id'] == subj]
            primacy = 0; other = 0; recency = 0; wrong = 0; null = 0
            
            for s_idx, s in enumerate(sub_data.session.unique()):
                sess_data = sub_data[sub_data['session'] == s]
                rec_evs = sess_data[sess_data['type'] == 'REC_WORD']
                
                for i in sess_data.list.unique():
                    sp = rec_evs[rec_evs['list'] == i].serial_position.to_numpy()
                    
                    if sp[0] >= 1 and sp[0] <= bufferStart:
                        primacy += 1
                        
                    if sp[0] >= list_length - bufferEnd + 1 and sp[0] <= list_length:
                        recency += 1
                        
                    if sp[0] > bufferStart and sp[0] < list_length - bufferEnd + 1:
                        other += 1
                        
                    if sp[0] == 88:
                        wrong += 1
                        
                    if sp[0] == 99:
                        null += 1
                        
                        
            sub_prim_rec_dict.update({subj: [primacy, other, recency, wrong, null]})
            sub_prim_rec_mat[n, :] = np.array([primacy, other, recency, wrong]) / (primacy + other + recency + wrong)
            
        prim_rec_dict.update({cond_str: np.mean(sub_prim_rec_mat, axis=0)})
        prim_rec_se.update({cond_str: np.std(sub_prim_rec_mat, axis=0, ddof=1) / np.sqrt(sub_prim_rec_mat.shape[0])})
        prim_rec_sub.update({cond_str: sub_prim_rec_mat})
        
    return prim_rec_dict, prim_rec_se, prim_rec_sub

def plot_prim_rec(prim_rec_dict, prim_rec_se):
    conds = ['10-2', '20-1', '15-2', '30-1', '20-2', '40-1']
    colors = ['firebrick', 'darkorange', 'forestgreen', 'dodgerblue', 'midnightblue', 'darkviolet']
    fig, ax = plt.subplots(2, 3, figsize=(11, 6), sharey=True, sharex=True)
    for i, c in enumerate(conds):
        prop_prim, prop_other, prop_rec, prop_wrong = prim_rec_dict.get(c)
        se_prim, se_other, se_rec, se_wrong = prim_rec_se.get(c)


        ax[i%2, i//2].bar([1,2,3,4], [prop_prim, prop_other, prop_rec, prop_wrong], yerr=[se_prim, se_other, se_rec, se_wrong], width=0.65, color=['orange', 'darkgray', 'purple', 'royalblue'])
        ax[i%2, i//2].set(xticks=[1,2,3,4], xticklabels=['Primacy', 'Other', 'Recency', 'Wrong'], title=c)
        ax[i%2, i//2].spines['right'].set_visible(False)
        ax[i%2, i//2].spines['top'].set_visible(False)

    fig.supylabel('Proportion of Lists')
    ax[1, 1].set(xlabel='Recall Initiation')
    plt.tight_layout()
    plt.show()
    
def plot_prim_rec_stats(prim_rec_dict, prim_rec_se, prim_rec_sub):
    conds = ['10-2', '20-1', '15-2', '30-1', '20-2', '40-1']
    colors = ['firebrick', 'darkorange', 'forestgreen', 'dodgerblue', 'midnightblue', 'darkviolet']
    fig, ax = plt.subplots(2, 3, figsize=(11, 6), sharey=True, sharex=True)
    for i, c in enumerate(conds):
        prop_prim, prop_other, prop_rec, prop_wrong = prim_rec_dict.get(c)
        se_prim, se_other, se_rec, se_wrong = prim_rec_se.get(c)
        sub_prim, sub_other, sub_rec, sub_wrong = prim_rec_sub.get(c)[:,0], prim_rec_sub.get(c)[:,1], prim_rec_sub.get(c)[:,2], prim_rec_sub.get(c)[:,3]
        
        ts_prim_rec, pval_prim_rec = scipy.stats.ttest_rel(sub_prim, sub_rec, alternative='two-sided')
        #print(c, ts_prim_rec, pval_prim_rec)

        ax[i%2, i//2].bar([1,2,3,4], [prop_prim, prop_other, prop_rec, prop_wrong], yerr=[1.96*se_prim, 1.96*se_other, 1.96*se_rec, 1.96*se_wrong], width=0.65, 
                          color=['orange', 'darkgray', 'purple', 'royalblue'], alpha=0.6)
        ax[i%2, i//2].scatter(np.full(len(sub_prim),1), sub_prim, s=5, color='orange', alpha=0.9)
        ax[i%2, i//2].scatter(np.full(len(sub_other),2), sub_other, s=5, color='darkgray', alpha=0.9)
        ax[i%2, i//2].scatter(np.full(len(sub_rec),3), sub_rec, s=5, color='purple', alpha=0.9)
        ax[i%2, i//2].scatter(np.full(len(sub_wrong),4), sub_wrong, s=5, color='royalblue', alpha=0.9)
        ax[i%2, i//2].set(xticks=[1,2,3,4], xticklabels=['Primacy', 'Other', 'Recency', 'Wrong'], title=c)
        ax[i%2, i//2].spines['right'].set_visible(False)
        ax[i%2, i//2].spines['top'].set_visible(False)

    fig.supylabel('Proportion of Lists')
    ax[1, 1].set(xlabel='Recall Initiation')
    plt.tight_layout()
    plt.show()
    
def plot_wrong_stats(pr_prim_dict, pr_prim_se, pr_prim_sub, pr_ns_dict, pr_ns_se, pr_ns_sub, pr_rec_dict, pr_rec_se, pr_rec_sub):
    conds = ['10-2', '20-1', '15-2', '30-1', '20-2', '40-1']
    fig, ax = plt.subplots(2, 3, figsize=(9,6), sharex=True, sharey=True)
    for i, c in enumerate(conds):
        wrong_prim = pr_prim_dict.get(c)[3]
        se_prim = pr_prim_se.get(c)[3]
        sub_prim = pr_prim_sub.get(c)[:,3]
        
        wrong_ns = pr_ns_dict.get(c)[3]
        se_ns = pr_ns_se.get(c)[3]
        sub_ns = pr_ns_sub.get(c)[:,3]
        
        wrong_rec = pr_rec_dict.get(c)[3]
        se_rec = pr_rec_se.get(c)[3]
        sub_rec = pr_rec_sub.get(c)[:,3]
        
        # two-sample t-tests between strategy groups
        ts_prim_rec, pval_prim_rec = scipy.stats.ttest_ind(sub_prim, sub_rec, equal_var=False, alternative='two-sided')
        ts_prim_ns, pval_prim_ns = scipy.stats.ttest_ind(sub_prim, sub_ns, equal_var=False, alternative='two-sided')
        ts_ns_rec, pval_ns_rec = scipy.stats.ttest_ind(sub_ns, sub_rec, equal_var=False, alternative='two-sided')
        
        h1 = np.max(np.concatenate((sub_prim, sub_ns, sub_rec))) + 0.05
        h2 = h1 + 0.05
        
        if pval_prim_rec < 0.05:
            ax[i%2, i//2].plot([1,3], [h2, h2], color='midnightblue')
        if pval_prim_ns < 0.05:
            ax[i%2, i//2].plot([1.1,1.9], [h1, h1], color='saddlebrown')
        if pval_ns_rec < 0.05:
            ax[i%2, i//2].plot([2.1,2.9], [h1, h1], color='saddlebrown')
            
        ax[i%2, i//2].bar([1,2,3], [wrong_prim, wrong_ns, wrong_rec], yerr=[1.96*se_prim, 1.96*se_ns, 1.96*se_rec], width=0.65, color=['orange', 'darkgray', 'purple'], alpha=0.6)
        ax[i%2, i//2].scatter(np.full(len(sub_prim), 1), sub_prim, s=5, color='orange', alpha=0.9)
        ax[i%2, i//2].scatter(np.full(len(sub_ns), 2), sub_ns, s=5, color='darkgray', alpha=0.9)
        ax[i%2, i//2].scatter(np.full(len(sub_rec), 3), sub_rec, s=5, color='purple', alpha=0.9)
        ax[i%2, i//2].set(xticks=[1,2,3], xticklabels=['Primacy', 'No Strategy', 'Recency'], title=c)
        ax[i%2, i//2].spines['right'].set_visible(False)
        ax[i%2, i//2].spines['top'].set_visible(False)
        
    fig.supylabel('Proportion of Lists Initiated with Intrusion')
    ax[1,1].set(xlabel='Group')
    plt.tight_layout()
    plt.show()
    
    
# ---------- Learning Effects ----------

def init_learn(df):
    il_dict = {}                         # dictionary mapping condition to 2D arrrays of trial and r1
    # iterate over conditions
    for c in df.condition.unique():
        cond_data, list_length, presentation_rate, cond_str = condition_metadata(df, c)
        sub_list = []                    # list holding 2D array for each subject
        
        # iterate over subjects within condition
        for n, subj in enumerate(cond_data.worker_id.unique()):
            sub_data = cond_data[cond_data['worker_id'] == subj]
            trial_list = []
            r1_list = []
            
            # iterate of subject's sessions
            for s_idx, s in enumerate(sub_data.session.unique()):
                sess_data = sub_data[sub_data['session'] == s]
                rec_evs = sess_data[sess_data['type'] == 'REC_WORD']
                
                for i in sess_data.list.unique():
                    sp = rec_evs[rec_evs['list'] == i].serial_position.to_numpy()
                    trial_list.append(12*(s-1) + (i+1))
                    r1_list.append(sp[0])
                    
            # only keep lists initiated with correct recall
            trial_list = np.array(trial_list)
            r1_list = np.array(r1_list)
            idx_mask = np.argwhere(r1_list <= list_length).T[0]
            
            trial_arr = trial_list[idx_mask]
            r1_arr = r1_list[idx_mask]
            # normalize to deciles
            r1_arr /= (list_length/10)
            
            sub_list.append(np.vstack((trial_arr, r1_arr)))    # normalize to deciles
            
        # update dictionary
        il_dict.update({cond_str: sub_list})
    
    return il_dict

def decile_trial(il_dict):
    conds = ['10-2', '20-1', '15-2', '30-1', '20-2', '40-1']
    # calculate total number of subjects across conditions
    tot_subs = 0
    for k in il_dict.keys():
        tot_subs += len(il_dict.get(k))
        
    sub_mat = np.zeros((tot_subs, 48), dtype = float)
    idx = 0

    for i, c in enumerate(conds):
        il = il_dict.get(c)
        for n in range(len(il)):
            trial = il[n][0, :].astype(int) - 1
            r1 = il[n][1, :]

            sub_mat[idx, trial] = r1
            idx += 1
            
    mean = np.mean(sub_mat, axis = 0)
    se = np.std(sub_mat, axis = 0, ddof = 1) / np.sqrt(sub_mat.shape[0])
    
    # plot sessions seperately
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(1, 49)
    # session 1
    ax.plot(x[:12], mean[:12])
    ax.fill_between(x[:12], mean[:12]-1.96*se[:12], mean[:12]+1.96*se[:12], alpha=0.1)
    # session 2
    ax.plot(x[12:24], mean[12:24])
    ax.fill_between(x[12:24], mean[12:24]-1.96*se[12:24], mean[12:24]+1.96*se[12:24], alpha=0.1)
    # session 3
    ax.plot(x[24:36], mean[24:36])
    ax.fill_between(x[24:36], mean[24:36]-1.96*se[24:36], mean[24:36]+1.96*se[24:36], alpha=0.1)
    # session 4
    ax.plot(x[36:], mean[36:])
    ax.fill_between(x[36:], mean[36:]-1.96*se[36:], mean[36:]+1.96*se[36:], alpha=0.1)
    ax.set(xlabel='Trial', xticks=np.linspace(0, 48, 9), ylabel='Decile of First Recall Serial Position', yticks=np.arange(0, 11))
    plt.show()
    
def plot_init_learn(il_dict):
    beta_dict = {}            # dictionary mapping condition to slopes
    conds = ['10-2', '20-1', '15-2', '30-1', '20-2', '40-1']
    fig, ax = plt.subplots(2, 3, figsize=(15, 10), sharex=True)
    for i, c in enumerate(conds):
        beta_list = []
        il = il_dict.get(c)
        for n in range(len(il)):
            trial = il[n][0, :]
            r1 = il[n][1, :]
            
            # calculate linear regression
            m, b = np.polyfit(trial, r1, deg=1)
            beta_list.append(m)
            
            ax[i%2, i//2].scatter(trial, r1, color='lightblue', alpha=0.1)
        
        beta_dict.update({c: beta_list})
        ax[i%2, i//2].set(xlabel = 'Trial', ylabel='Decile of First Recall Serial Position', xticks=np.linspace(0, 48, 9), title=c)
        ax[i%2, i//2].spines['right'].set_visible(False); ax[i%2, i//2].spines['top'].set_visible(False)
        
    plt.tight_layout()
    plt.show()
    
    return beta_dict

def plot_slopes(beta_dict):
    conds = ['10-2', '20-1', '15-2', '30-1', '20-2', '40-1']
    colors = ['firebrick', 'darkorange', 'forestgreen', 'dodgerblue', 'midnightblue', 'darkviolet']
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    for i, c in enumerate(conds):
        betas = beta_dict.get(c)
        os_ts, os_pval = scipy.stats.ttest_1samp(betas, 0, alternative='two-sided')
        print(c, np.median(betas), np.mean(betas), os_ts, os_pval)
        if os_pval < 0.05:
            ax[i%2, i//2].plot(np.max(betas)-0.1, 15, marker='*', markersize=15, color='darkgoldenrod')
        edge = np.max([abs(np.min(betas)-0.05), abs(np.max(betas)+0.05)])
        bins = np.arange(-edge, edge, 2*edge/15)
        ax[i%2, i//2].hist(betas, bins=bins, color=colors[i], alpha=0.8)
        ax[i%2, i//2].axvline(np.mean(betas), color='k', linestyle='dotted')
        ax[i%2, i//2].grid(True)
        ax[i%2, i//2].spines['right'].set_visible(False); ax[i%2, i//2].spines['top'].set_visible(False)
        ax[i%2, i//2].set(xlabel='Slope', ylabel='Subjects', title=c)
        
    plt.tight_layout()
    plt.show()
    
def plot_slopes_all_conds(beta_dict):
    conds = ['10-2', '20-1', '15-2', '30-1', '20-2', '40-1']
    fig, ax = plt.subplots(figsize=(7, 5))
    all_betas = []
    for i, c in enumerate(conds):        # concatenate all conditions
        betas = beta_dict.get(c)
        all_betas.extend(betas)
        
    os_ts, os_pval = scipy.stats.ttest_1samp(all_betas, 0, alternative='two-sided')
    print(len(all_betas), np.median(all_betas), np.mean(all_betas), os_ts, os_pval)
    if os_pval < 0.05:
        ax.plot(np.max(all_betas)-0.1, 40, marker='*', markersize=15, color='darkgoldenrod')
    
    edge = np.max([abs(np.min(all_betas)-0.05), abs(np.max(all_betas)+0.05)])
    bins = np.arange(-edge, edge, 2*edge/20)
    ax.hist(all_betas, bins=bins, color='plum', alpha=0.8)
    ax.axvline(np.mean(all_betas), color='k', linestyle='dotted')
    ax.grid(True)
    ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
    ax.set(xlabel='Slope', ylabel='Subjects')
    plt.show()