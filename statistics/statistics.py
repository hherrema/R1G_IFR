### Statistics code
### FDR correction with Benjamini-Yekutieli method

# imports
import pandas as pd
import numpy as np
import scipy.stats


# primacy and recency effect
# 1-sample t-test against mean of 0
def prim_rec_slopes_statistics(spc_prim_rec_lr_all):
    stats = []
    for (c, ll, pr), cond_data in spc_prim_rec_lr_all.groupby(['condition', 'l_length', 'pres_rate']):
        prim_res = scipy.stats.ttest_1samp(cond_data.prim_slope, popmean=0, alternative='two-sided')
        rec_res = scipy.stats.ttest_1samp(cond_data.rec_slope, popmean=0, alternative='two-sided')
        stats.append((c, ll, pr, prim_res.statistic, prim_res.pvalue, prim_res.df, rec_res.statistic, rec_res.pvalue, rec_res.df))
        
    stats = pd.DataFrame(stats, columns=['condition', 'l_length', 'pres_rate', 
                                         'prim_t_stat', 'prim_p_val', 'prim_dof', 
                                         'rec_t_stat', 'rec_p_val', 'rec_dof'])
    
    # FDR correction
    all_pvals = list(stats.prim_p_val) + list(stats.rec_p_val)
    fdr_pvals = scipy.stats.false_discovery_control(all_pvals, method='by')
    fdr_prim, fdr_rec = np.split(fdr_pvals, 2)
    
    stats['prim_p_val_fdr'] = fdr_prim
    stats['rec_p_val_fdr'] = fdr_rec
    
    return stats


# primacy and recency initiation bias
# 1-sample t-test against mean of 0
def prim_rec_pfr_statistics(prim_rec_pfr):
    stats = []
    for (c, ll, pr), cond_data in prim_rec_pfr.groupby(['condition', 'l_length', 'pres_rate']):
        res = scipy.stats.ttest_1samp(cond_data.rec_prim_bias, popmean=0, alternative='two-sided')
        stats.append((c, ll, pr, res.statistic, res.pvalue, res.df))
        
    stats = pd.DataFrame(stats, columns=['condition', 'l_length', 'pres_rate', 't_stat', 'p_val', 'dof'])
    
    # FDR correction
    fdr_pvals = scipy.stats.false_discovery_control(stats.p_val, method='by')
    stats['p_val_fdr'] = fdr_pvals
    
    return stats


# correlation of PFR and SPC (at each serial position)
### DETERMINE WHICH CORRECTION TO USE (I THINK LL COMPARISONS)
def pfr_spc_correlation_statistics(pfr_spc_corrs):
    stats = pfr_spc_corrs.copy()
    
    # all comparisons
    fdr_pvals = scipy.stats.false_discovery_control(pfr_spc_corrs.p_value)
    stats['p_val_fdr'] = fdr_pvals
    
    # within condition
    ps = []
    for c, cond_data in pfr_spc_corrs.groupby('condition', sort=False):
        cond_fdr_pvals = scipy.stats.false_discovery_control(cond_data.p_value)
        ps.extend(cond_fdr_pvals)
    
    stats['p_val_cond_fdr'] = ps
    
    return stats


# within session recall initiation variance
# paired t-test
def r1_variance_statistics(r1_var_data_bsa):
    stats = []
    for (c, ll, pr), cond_data in r1_var_data_bsa.groupby(['condition', 'l_length', 'pres_rate']):
        res = scipy.stats.ttest_rel(cond_data.sp_sem, cond_data.permutation_sp_sem, nan_policy='omit', alternative='two-sided')
        stats.append((c, ll, pr, res.statistic, res.pvalue, res.df))
        
    stats = pd.DataFrame(stats, columns=['condition', 'l_length', 'pres_rate', 't_stat', 'p_val', 'dof'])
    
    # FDR correction
    fdr_pvals = scipy.stats.false_discovery_control(stats.p_val, method='by')
    stats['p_val_fdr'] = fdr_pvals
    
    return stats


# change in recall initiation serial position across sessions
# 1-sample t-test against mean of 0
def r1_sp_statistics(r1_sp_dec_data_bsa):
    stats = []
    for (c, ll, pr), cond_data in r1_sp_dec_data_bsa.groupby(['condition', 'l_length', 'pres_rate']):
        res = scipy.stats.ttest_1samp(cond_data.r1_sp_slope, popmean=0, alternative='two-sided')
        stats.append((c, ll, pr, res.statistic, res.pvalue, res.df))
        
    stats = pd.DataFrame(stats, columns=['condition', 'l_length', 'pres_rate', 't_stat', 'p_val', 'dof'])
    
    # FDR corrections
    fdr_pvals = scipy.stats.false_discovery_control(stats.p_val, method='by')
    stats['p_val_fdr'] = fdr_pvals
    
    return stats

# semantic clustering score
# 1-sample t-test against mean of 0.5
def scl_statistics(scl_data_bsa):
    stats = []
    for (strat, c, ll, pr), data in scl_data_bsa.groupby(['strategy', 'condition', 'l_length', 'pres_rate']):
        res = scipy.stats.ttest_1samp(data.scl, popmean=0.5, nan_policy='omit', alternative='two-sided')
        stats.append((strat, c, ll, pr, res.statistic, res.pvalue, res.df))
        
    stats = pd.DataFrame(stats, columns=['strategy', 'condition', 'l_length', 'pres_rate', 't_stat', 'p_val', 'dof'])
    
    # FDR correction
    fdr_pvals = scipy.stats.false_discovery_control(stats.p_val, method='by')
    stats['p_val_fdr'] = fdr_pvals
    
    return stats



