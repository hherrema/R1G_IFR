### Plotting code
### Formatting for publication (labels, axis, sizes)

# imports
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# formatting parameters
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# ---------- All Data Together ----------

# serial position curve
def plot_spc_all(spc_data_bsa_all):
    # re-organize dataframe
    dfm = pd.melt(spc_data_bsa_all, id_vars=['subject', 'condition', 'l_length', 'pres_rate'], value_vars=[f'sp_{x}' for x in range(1, 41)],
                  var_name='serial_position', value_name='recall_probability')
    dfm['serial_position'] = [int(x.split('_')[-1]) for x in dfm.serial_position]         # change serial positions to ints

    fig, ax = plt.subplots(figsize=(7, 5))
    conds_tt = ['10-2', '20-1', '15-2', '30-1', '20-2', '40-1']
    colors = ['rosybrown', 'maroon', 'limegreen', 'forestgreen', 'lightskyblue', 'midnightblue']

    sns.lineplot(dfm, x='serial_position', y='recall_probability', hue='condition', hue_order=conds_tt, palette=colors, errorbar=('se', 1.96), ax=ax)

    ax.set(xlabel='Serial Position', xticks=np.arange(5, 41, 5), ylabel='Recall Probability', yticks=np.linspace(0, 1, 11))
    ax.spines[['right', 'top']].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='Condition', shadow=True, ncols=3, loc='upper right')

    plt.savefig('figures/gallery/spc_all.pdf', bbox_inches='tight')
    plt.show()


# primacy and recency effect
def plot_prim_rec_slopes(spc_prim_rec_lr_all):
    # re-orgnaize dataframe
    dfm = pd.melt(spc_prim_rec_lr_all, id_vars=['subject', 'condition', 'l_length', 'pres_rate'], value_vars=['prim_slope', 'rec_slope'],
                  var_name='sp_region', value_name='slope')

    fig, ax = plt.subplots(figsize=(7, 5))
    conds_ll_pr = ['10-2', '15-2', '20-2', '20-1', '30-1', '40-1']

    sns.stripplot(dfm, x='condition', y='slope', order=conds_ll_pr, hue='sp_region', hue_order=['prim_slope', 'rec_slope'],
                  dodge=True, palette=['palevioletred', 'steelblue'], alpha=0.3, ax=ax, legend=False)
    sns.barplot(dfm, x='condition', y='slope', order=conds_ll_pr, hue='sp_region', hue_order=['prim_slope', 'rec_slope'],
                dodge=True, palette=['palevioletred', 'steelblue'], errorbar=('se', 1.96), alpha=0.6, ax=ax, legend=True)

    ax.set(xlabel='Condition', ylabel='Slope', yticks=np.linspace(-0.3, 0.3, 7))
    ax.spines[['right', 'top']].set_visible(False)

    labels = ['Primacy Effect', 'Recency Effect']
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, labels, shadow=True, ncols=2, loc='upper right', bbox_to_anchor=(1, 1.05))
    
    plt.savefig('figures/gallery/prim_rec_slopes.pdf', bbox_inches='tight')
    plt.show()


# probability of first recall
def plot_pfr_all(pfr_data_bsa_all):
    # re-organize dataframe
    dfm = pd.melt(pfr_data_bsa_all, id_vars=['subject', 'condition', 'l_length', 'pres_rate'], value_vars = [f'sp_{x}' for x in range(1, 41)],
                      var_name='serial_position', value_name='recall_probability')
    dfm['serial_position'] = [int(x.split('_')[-1]) for x in dfm.serial_position]         # change serial positions to ints

    fig, ax = plt.subplots(figsize=(7, 5))
    conds_tt = ['10-2', '20-1', '15-2', '30-1', '20-2', '40-1']
    colors = ['rosybrown', 'maroon', 'limegreen', 'forestgreen', 'lightskyblue', 'midnightblue']

    sns.lineplot(dfm, x='serial_position', y='recall_probability', hue='condition', hue_order=conds_tt, palette=colors, errorbar=('se', 1.96), ax=ax)
    
    ax.set(xlabel='Serial Position', xticks=np.arange(5, 41, 5), ylabel='Probability of First Recall', yticks=np.linspace(0, 0.5, 6))
    ax.spines[['right', 'top']].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='Condition', shadow=True, ncols=3, loc='upper right')

    plt.savefig('figures/gallery/pfr_all.pdf', bbox_inches='tight')
    plt.show()


# primacy and recency initiation bias
def plot_prim_rec_pfr(prim_rec_pfr):
    fig, ax = plt.subplots(figsize=(7, 5))
    conds_ll_pr = ['10-2', '15-2', '20-2', '20-1', '30-1', '40-1']

    sns.stripplot(prim_rec_pfr, x='condition', y='rec_prim_bias', order=conds_ll_pr, hue='rec_prim_bias_bool', hue_order=[False, True],
                  palette=['palevioletred', 'steelblue'], alpha=0.3, ax=ax, legend=True)
    sns.barplot(prim_rec_pfr, x='condition', y='rec_prim_bias', order=conds_ll_pr, hue='condition_mu', hue_order=['prim', 'rec'], 
                palette=['palevioletred', 'steelblue'], alpha=0.6, errorbar=('se', 1.96), ax=ax, legend=False)

    ax.set(xlabel='Condition', ylabel='Probability of First Recall (Recency - Primacy)')
    ax.spines[['right', 'top']].set_visible(False)

    labels = ['Primacy Bias', 'Recency Bias']
    handles, _ = ax.get_legend_handles_labels()
    leg = ax.legend(handles, labels, shadow=True, ncols=2, loc='upper right', bbox_to_anchor=(1, 1.1))
    for lh in leg.legend_handles:
        lh.set_alpha(1)

    plt.savefig('figures/gallery/pfr_prim_rec_bias.pdf', bbox_inches='tight')
    plt.show()


# correlation of PFR and SPC
def plot_pfr_spc_correlation(pfr_spc_corrs, pfr_spc_corrs_stats):
    fig, ax = plt.subplots(2, 3, figsize=(10, 6), sharey=True, gridspec_kw={'width_ratios': [1, 1.5, 2]})
    conds_tt = ['10-2', '20-1', '15-2', '30-1', '20-2', '40-1']

    for i, c in enumerate(conds_tt):
        cond_data = pfr_spc_corrs.query("condition == @c")
        cond_stats = pfr_spc_corrs_stats.query("condition == @c")
        ll = cond_data.l_length.unique()[0]

        sns.lineplot(cond_data, x='serial_position', y='pearson_r', color='silver', ax=ax[i%2, i//2])
        sns.scatterplot(cond_stats.query("p_val_fdr < 0.05 & pearson_r > 0"), x='serial_position', y=0.9, color='black', ax=ax[i%2, i//2], marker='*', s=70)
        sns.scatterplot(cond_stats.query("p_val_fdr < 0.05 & pearson_r < 0"), x='serial_position', y=-0.3, color='black', ax=ax[i%2, i//2], marker='*', s=70)

        b = 0.25   # border for shading
        ax[i%2, i//2].axvspan(1-b, 4+b, color='palevioletred', alpha=0.1)       # primacy region
        ax[i%2, i//2].axvspan(ll-3-b, ll+b, color='steelblue', alpha=0.1)   # recency region

        xax = np.prod(np.array(c.split('-')).astype(int))
        ax[i%2, i//2].set(title=c, xlabel='', xticks=np.arange(5, xax+1, 5), xlim=(0, xax+1), ylabel='')
        ax[i%2, i//2].spines[['right', 'top']].set_visible(False)

    fig.supxlabel('Serial Position')
    fig.supylabel('Pearson r Correlation (PFR, SPC)')
    
    plt.tight_layout()
    plt.savefig('figures/gallery/pfr_spc_correlation.pdf', bbox_inches='tight')
    plt.show()


# within session recall initiation variance
def plot_r1_variance(r1_var_data_bsa):
    # re-organize dataframe
    dfm = pd.melt(r1_var_data_bsa, id_vars=['subject', 'condition', 'l_length', 'pres_rate'], value_vars=['sp_sem', 'permutation_sp_sem'],
                  var_name='type', value_name='standard_error')

    fig, ax = plt.subplots(figsize=(6, 4))
    conds_ll_pr = ['10-2', '15-2', '20-2', '20-1', '30-1', '40-1']

    sns.barplot(dfm, x='condition', y='standard_error', hue='type', order=conds_ll_pr, palette=['hotpink', 'silver'],
                gap=0.1, errorbar=('se', 1.96), alpha=0.9)

    ax.set(xlabel='Condition', ylabel='Standard Error (Serial Positions)')
    ax.spines[['right', 'top']].set_visible(False)

    labels = ['Data', 'PFR Simulation']
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, labels, shadow=True, ncols=2, loc='upper left', bbox_to_anchor=(0.05, 1.05))
    
    plt.savefig('figures/gallery/r1_var.pdf', bbox_inches='tight')
    plt.show()


# change in recall initiation serial position across sessions
def plot_r1_sp_slopes(r1_sp_dec_data_bsa, r1_sp_stats):
    fig, ax = plt.subplots(figsize=(6, 4))
    conds_ll_pr = ['10-2', '15-2', '20-2', '20-1', '30-1', '40-1']

    sns.pointplot(r1_sp_dec_data_bsa, x='condition', y='r1_sp_slope', order=conds_ll_pr, color='saddlebrown', 
                  linestyle='none', markers='o', errorbar=('se', 1.96), ax=ax, alpha=0.9)
    ax.axhline(0, color='black', linestyle='dotted')

    # statistical significance
    sns.scatterplot(r1_sp_stats.query("p_val_fdr < 0.05"), x='condition', y=2, color='black', marker='*', s=200)

    ax.set(xlabel='Condition', ylabel='R1 Serial Position Slope', ylim=(-1.5, 2.1))
    ax.spines[['right', 'top']].set_visible(False)
    
    plt.savefig('figures/gallery/r1_sp_slopes.pdf', bbox_inches='tight')
    plt.show()


# ---------- Recall Initiation Groups ----------

# serial position curve
def plot_spc(spc_data_bsa):
    # re-organize dataframe
    dfm = pd.melt(spc_data_bsa, id_vars=['subject', 'strategy', 'condition', 'l_length', 'pres_rate'],
                  value_vars=[f'sp_{x}' for x in range(1, 41)], var_name='serial_position', value_name='recall_probability')
    dfm['serial_position'] = [int(x.split('_')[-1]) for x in dfm.serial_position]          # change serial positions to ints

    fig, ax = plt.subplots(2, 3, figsize=(9, 6), sharey=True, gridspec_kw={'width_ratios': [1, 1.5, 2]})
    conds_tt = ['10-2', '20-1', '15-2', '30-1', '20-2', '40-1']

    for i, c in enumerate(conds_tt):
        spc_cond = dfm.query("condition == @c")

        sns.lineplot(spc_cond, x='serial_position', y='recall_probability', hue='strategy', hue_order=['prim', 'ns', 'rec'], 
                         palette=['orange', 'darkgray', 'purple'], errorbar=('se', 1.96), ax=ax[i%2, i//2], legend=True)
        
        handles, _ = ax[i%2, i//2].get_legend_handles_labels()
        ax[i%2, i//2].get_legend().remove()

        xax = np.prod(np.array(c.split('-')).astype(int))
        ax[i%2, i//2].set(title=c, xlabel='', xticks=np.arange(5, xax+1, 5), xlim=(0, xax+1), ylabel='', yticks=np.linspace(0, 1, 6))
        if i%2 == 0:
            ax[i%2, i//2].set_xticklabels([])
        ax[i%2, i//2].spines[['right', 'top']].set_visible(False)

    fig.supxlabel('Serial Position')
    fig.supylabel('Recall Probability')
    labels = ['Primacy', 'Other', 'Recency']
    fig.legend(handles, labels, shadow=True, ncols=3, loc='upper center', bbox_to_anchor=(0.5, 1.1))
    
    plt.tight_layout()
    plt.savefig('figures/gallery/spc.pdf', bbox_inches='tight')
    plt.show()


# probability of first recall
def plot_pfr(pfr_data_bsa):
    # re-organize dataframe
    dfm = pd.melt(pfr_data_bsa, id_vars=['subject', 'strategy', 'condition', 'l_length', 'pres_rate'],
                  value_vars = [f'sp_{x}' for x in range(1, 41)], var_name='serial_position', value_name='recall_probability')
    dfm['serial_position'] = [int(x.split('_')[-1]) for x in dfm.serial_position]       # change serial positions to ints

    fig, ax = plt.subplots(2, 3, figsize=(9, 6), sharey=True, gridspec_kw={'width_ratios': [1, 1.5, 2]})
    conds_tt = ['10-2', '20-1', '15-2', '30-1', '20-2', '40-1']

    for i, c in enumerate(conds_tt):
        pfr_cond = dfm.query("condition == @c")

        sns.lineplot(pfr_cond, x='serial_position', y='recall_probability', hue='strategy', hue_order=['prim', 'ns', 'rec'], 
                     palette=['orange', 'darkgray', 'purple'], errorbar=('se', 1.96), ax=ax[i%2, i//2], legend=True)
        
        handles, _ = ax[i%2, i//2].get_legend_handles_labels()
        ax[i%2, i//2].get_legend().remove()

        xax = np.prod(np.array(c.split('-')).astype(int))
        ax[i%2, i//2].set(title=c, xlabel='', xticks=np.arange(0, xax+1, 5), xlim=(0, xax+1), ylabel='', yticks=np.linspace(0, 0.7, 8))
        if i%2 == 0:
            ax[i%2, i//2].set_xticklabels([])
        ax[i%2, i//2].spines[['right', 'top']].set_visible(False)


    fig.supxlabel('Serial Position')
    fig.supylabel('Probability of First Recall')
    labels = ['Primacy', 'Other', 'Recency']
    fig.legend(handles, labels, shadow=True, ncols=3, loc='upper center', bbox_to_anchor=(0.5, 1.1))

    plt.tight_layout()
    plt.savefig('figures/gallery/pfr.pdf', bbox_inches='tight')
    plt.show()


# mean words recalled
def plot_mwr(mwr_data_bsa):
    fig, ax = plt.subplots(2, 3, figsize=(9, 6), sharey=True, sharex=True)
    conds_tt = ['10-2', '20-1', '15-2', '30-1', '20-2', '40-1']

    for i, c in enumerate(conds_tt):
        mwr_cond = mwr_data_bsa.query("condition == @c")

        sns.barplot(mwr_cond, x='strategy', y='mwr', order=['prim', 'ns', 'rec'], hue='strategy', hue_order=['prim', 'ns', 'rec'], 
                    palette=['orange', 'darkgray', 'purple'], alpha=0.7, errorbar=('se', 1.96), ax=ax[i%2, i//2])
        sns.scatterplot(mwr_cond, x='strategy', y='mwr', hue='strategy', hue_order=['prim', 'ns', 'rec'],
                        palette=['orange', 'darkgray', 'purple'], alpha=0.6, ax=ax[i%2, i//2], legend=False)


        ax[i%2, i//2].set(xlabel='', ylabel='', xticks=[0, 1, 2], xticklabels=['Primacy', 'Other', 'Recency'], title=c, ylim=(0, 23))
        ax[i%2, i//2].spines[['right', 'top']].set_visible(False)

    fig.supxlabel('Recall Initiation Group', x=0.53)
    fig.supylabel('Mean Words Recalled')

    plt.tight_layout()
    plt.savefig('figures/gallery/mwr.pdf', bbox_inches='tight')
    plt.show()


# proportion of lists initiated with an intrusion
def plot_r1_intr(r1_intr_data_bsa):
    fig, ax = plt.subplots(figsize=(8, 5))
    conds_ll_pr = ['10-2', '15-2', '20-2', '20-1', '30-1', '40-1']

    sns.barplot(r1_intr_data_bsa, x='condition', y='prop_wrong', order=conds_ll_pr,
                hue='strategy', hue_order=['prim', 'ns', 'rec'], palette=['orange', 'darkgray', 'purple'], 
                alpha=0.7, errorbar=('se', 1.96), gap=0.1)
    
    ax.set(xlabel='Condition', ylabel='R1 Intrusion Probability')
    ax.spines[['right', 'top']].set_visible(False)

    # legend
    labels = ['Primacy', 'Other', 'Recency']
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, labels, shadow=True, ncols=3, loc='upper right', bbox_to_anchor=(1, 1.08))
    
    plt.savefig('figures/gallery/r1_intr.pdf', bbox_inches='tight')
    plt.show()


# initial response times
def plot_rti(rti_at_data, rti_data_bsa):
    fig, ax = plt.subplots(2, 3, figsize=(9, 6), sharey=True, sharex=True)
    conds_tt = ['10-2', '20-1', '15-2', '30-1', '20-2', '40-1']

    for i, c in enumerate(conds_tt):
        cond_data = rti_at_data.query("condition == @c and min_rt == False")
        bins = np.arange(0, max(cond_data.rt)+250, 250)

        cond_prim = cond_data.query("strategy == 'prim'")
        cond_ns = cond_data.query("strategy == 'ns'")
        cond_rec = cond_data.query("strategy == 'rec'")

        # trial histograms
        _, _, _ = ax[i%2, i//2].hist(cond_prim.rt, bins=bins, density=True, color='orange', alpha=0.4, label='Primacy')
        _, _, _ = ax[i%2, i//2].hist(cond_ns.rt, bins=bins, density=True, color='darkgray', alpha=0.3, label='Other')
        _, _, _ = ax[i%2, i//2].hist(cond_rec.rt, bins=bins, density=True, color='purple', alpha=0.4, label='Recency')

        # between-subject averages
        ax[i%2, i//2].axvline(rti_data_bsa.query("condition == @c and strategy == 'prim'").rt.mean(), color='orange', linestyle='dashed')
        ax[i%2, i//2].axvline(rti_data_bsa.query("condition == @c and strategy == 'ns'").rt.mean(), color='darkgray', linestyle='dashed')
        ax[i%2, i//2].axvline(rti_data_bsa.query("condition == @c and strategy == 'rec'").rt.mean(), color='purple', linestyle='dashed')

        ax[i%2, i//2].set(title=c, xlabel='', xlim=(0, 20000), ylabel='')
        ax[i%2, i//2].grid(True)
        ax[i%2, i//2].spines[['right', 'top']].set_visible(False)

    fig.supxlabel("Response Time (ms)", x=0.53)
    fig.supylabel("Proportion of Trials")
    handles, labels = ax[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, shadow=True, ncols=3, loc='upper center', bbox_to_anchor=(0.53, 1.08))

    plt.tight_layout()
    plt.savefig('figures/gallery/rti.pdf', bbox_inches='tight')
    plt.show()


# intrusion rates (ELI, PLI)
# typ = 'eli_rate' or 'pli_rate'
def plot_intr_data(intr_data_bsa, typ):
    fig, ax = plt.subplots(figsize=(8, 5))
    conds_ll_pr = ['10-2', '15-2', '20-2', '20-1', '30-1', '40-1']
    
    sns.barplot(intr_data_bsa, x='condition', y=typ, order=conds_ll_pr, 
                hue='strategy', hue_order=['prim', 'ns', 'rec'], palette=['orange', 'darkgray','purple'], alpha=0.7, 
                errorbar=('se', 1.96), gap=0.1, ax=ax)
       
    ax.set(xlabel='Condition')
    ax.spines[['right', 'top']].set_visible(False)
    if typ == 'eli_rate':
        ax.set_ylabel('ELIs per Trial')
    elif typ == 'pli_rate':
        ax.set_ylabel('PLIs per Trial')
    
        
    # legend
    new_labels = ['Primacy', 'Other', 'Recency']
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, new_labels, ncols=3, loc=(0.45, 1))
    
    if typ == 'eli_rate':
        ax.set(ylabel='ELIs per Trial', ylim=(0, 2.45))
        plt.savefig('figures/gallery/eli.pdf', bbox_inches='tight')
    elif typ == 'pli_rate':
        ax.set(ylabel='PLIs per Trial', ylim=(0, 0.68))
        plt.savefig('figures/gallery/pli.pdf', bbox_inches='tight')
    plt.show()


# inter-response times
# joint function of number of correct recalls and output position
# trials with mu +/- sig correct recalls
def plot_irt(irt_data_bsa, mwr_md):
    # select trials mu +/- sig correct recalls
    data = []
    for c, dat in irt_data_bsa.groupby('condition'):
        lb = mwr_md.query("condition == @c").iloc[0].lb
        ub = mwr_md.query("condition == @c").iloc[0].ub
        data.append(dat.query("ncr > @lb and ncr <= @ub"))

    data = pd.concat(data, ignore_index=True)

    # re-organize dataframe
    dfm = pd.melt(data, id_vars=['subject', 'strategy', 'condition', 'l_length', 'pres_rate', 'ncr'],
                  value_vars=[f'tr_{x}' for x in range(1, 28)], var_name='output_position', value_name='irt')
    dfm['output_position'] = [int(x.split('_')[-1]) for x in dfm.output_position]       # change output position to ints
    
    fig, ax = plt.subplots(2, 3, figsize=(9, 6), sharex=True, sharey=True)
    conds_tt = ['10-2', '20-1', '15-2', '30-1', '20-2', '40-1']

    for i, c in enumerate(conds_tt):
        cond_data = dfm.query("condition == @c")
        for n in cond_data.ncr.unique():
            sns.lineplot(cond_data.query("ncr == @n"), x='output_position', y='irt', hue='strategy', hue_order=['prim', 'ns', 'rec'],
                         palette=['orange', 'darkgray', 'purple'], err_style='bars', errorbar=('se', 1.96), marker='o', alpha=0.6, ax=ax[i%2, i//2])
            handles, _ = ax[i%2, i//2].get_legend_handles_labels()
            ax[i%2, i//2].get_legend().remove()

        ax[i%2, i//2].set(title=c, xlabel='', ylabel='')
        ax[i%2, i//2].spines[['right', 'top']].set_visible(False)

    fig.supxlabel('Output Position', x=0.53)
    fig.supylabel("Inter-Response Time (ms)")
    labels = ['Primacy', 'Other', 'Recency']
    fig.legend(handles, labels, shadow=True, ncols=3, loc='upper center', bbox_to_anchor=(0.53, 1.06))

    plt.tight_layout()
    plt.savefig('figures/gallery/irt.pdf', bbox_inches='tight')
    plt.show()

# final N recalls
# trials with mu +/- sig correct recalls
def plot_irt_final(irt_final_data_bsa):
    # re-organize dataframe
    dfm = pd.melt(irt_final_data_bsa, id_vars=['subject', 'strategy', 'condition', 'l_length', 'pres_rate'],
                  value_vars=[f'rot_{x}' for x in np.arange(7, -1, -1)], var_name='relative_output_transition', value_name='irt')
    dfm['relative_output_transition'] = [-int(x.split('_')[-1]) for x in dfm.relative_output_transition]

    fig, ax = plt.subplots(2, 3, figsize=(9, 6), sharey=True, sharex=True)
    conds_tt = ['10-2', '20-1', '15-2', '30-1', '20-2', '40-1']

    for i, c in enumerate(conds_tt):
        cond_data = dfm.query("condition == @c")

        sns.lineplot(cond_data, x='relative_output_transition', y='irt', hue='strategy', hue_order=['prim', 'ns', 'rec'],
                     palette=['orange', 'darkgray', 'purple'], err_style='bars', errorbar=('se', 1.96), marker='o', ax=ax[i%2, i//2])
        handles, _ = ax[i%2, i//2].get_legend_handles_labels()
        ax[i%2, i//2].get_legend().remove()

        ax[i%2, i//2].set(title=c, xlabel='', ylabel='')
        ax[i%2, i//2].spines[['right', 'top']].set_visible(False)

    fig.supxlabel('Relative Output Position', x=0.53)
    fig.supylabel('Inter-Response Time (ms)')
    labels = ['Primacy', 'Other', 'Recency']
    fig.legend(handles, labels, shadow=True, ncols=3, loc='upper center', bbox_to_anchor=(0.53, 1.08))

    plt.tight_layout()
    plt.savefig('figures/gallery/irt_final.pdf', bbox_inches='tight')
    plt.show()

# total time (each half of recall)
def plot_irt_total_h(irt_tot_data_bsa):
    fig, ax = plt.subplots(2, 3, figsize=(9, 6), sharey=True, sharex=True)
    conds_tt = ['10-2', '20-1', '15-2', '30-1', '20-2', '40-1']

    for i, c in enumerate(conds_tt):
        cond_data = irt_tot_data_bsa.query("condition == @c")

        sns.lineplot(cond_data, x='ncr_bin', y='irt_delta', hue='strategy', hue_order=['prim', 'ns', 'rec'],
                     palette=['orange', 'darkgray', 'purple'], err_style='bars', errorbar=('se', 1.96), marker='o', ax=ax[i%2, i//2])
        ax[i%2, i//2].axhline(0, color='black', linestyle='dotted')
        handles, _ = ax[i%2, i//2].get_legend_handles_labels()
        ax[i%2, i//2].get_legend().remove()


        ax[i%2, i//2].set(title=c, xlabel='', xticks=np.arange(2, 15, 2), ylabel='', ylim=(-20000, 25000))
        ax[i%2, i//2].spines[['right', 'top']].set_visible(False)

    fig.supxlabel('Correct Recalls (each half)', x=0.53)
    fig.supylabel('Difference in Total Inter-Response Time (ms)')
    labels = ['Primacy', 'Other', 'Recency']
    fig.legend(handles, labels, shadow=True, ncols=3, loc='upper center', bbox_to_anchor=(0.53, 1.08))

    plt.tight_layout()
    plt.savefig('figures/gallery/irt_total_h.pdf', bbox_inches='tight')
    plt.show()


# temporal clustering score
# total
def plot_tcl(tcl_data_bsa):
    fig, ax = plt.subplots(figsize=(8,5))
    conds_ll_pr = ['10-2', '15-2', '20-2', '20-1', '30-1', '40-1']

    sns.barplot(tcl_data_bsa, x='condition', y='tcl', order=conds_ll_pr, hue='strategy', hue_order=['prim', 'ns', 'rec'],
                palette=['orange', 'darkgray', 'purple'], errorbar=('se', 1.96), alpha=0.7, gap=0.1)

    ax.set(xlabel='Condition', ylabel='Temporal Clustering Score', ylim=(0.45, 1))
    ax.spines[['right', 'top']].set_visible(False)

    labels = ['Primacy', 'Other', 'Recency']
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, labels, shadow=True, ncols=3, loc='upper right', bbox_to_anchor=(1, 1.05))
    
    plt.savefig('figures/gallery/tcl.pdf', bbox_inches='tight')
    plt.show()

# each half of recall
def plot_tcl_h(tcl_h_data_bsa):
    # re-organize dataframe
    dfm = pd.melt(tcl_h_data_bsa, id_vars=['subject', 'strategy', 'condition', 'l_length', 'pres_rate', 'tcl_delta'],
                  value_vars=['tcl_h1', 'tcl_h2'], var_name='half_label', value_name='tcl_h')

    fig, ax = plt.subplots(2, 3, figsize=(7, 5), sharex=True, sharey=True)
    conds_tt = ['10-2', '20-1', '15-2', '30-1', '20-2', '40-1']

    for i, c in enumerate(conds_tt):
        cond_data = dfm.query("condition == @c")
        sns.pointplot(cond_data, x='half_label', y='tcl_h', order=['tcl_h1', 'tcl_h2'], hue='strategy', hue_order=['prim', 'ns', 'rec'], 
                      palette=['orange', 'darkgray', 'purple'], errorbar=('se', 1.96), dodge=0.3, alpha=0.85, ax=ax[i%2, i//2])
        handles, _ = ax[i%2, i//2].get_legend_handles_labels()
        ax[i%2, i//2].get_legend().remove()

        ax[i%2, i//2].set(title=c, xlabel='', ylabel='', ylim=(0.45, 1))
        ax[i%2, i//2].set_xticks([0, 1], labels=['Half 1', 'Half 2'])
        ax[i%2, i//2].spines[['right', 'top']].set_visible(False)

    fig.supxlabel('Position in Recall Sequence', x=0.53)
    fig.supylabel('Temporal Clustering Score')
    labels = ['Primacy', 'Other', 'Recency']
    fig.legend(handles, labels, shadow=True, ncols=3, loc='upper center', bbox_to_anchor=(0.53, 1.1))

    plt.tight_layout()
    plt.savefig('figures/gallery/tcl_h.pdf', bbox_inches='tight')
    plt.show()


# lag-CRP
# full
def plot_lcrp(lcrp_data_bsa):
    # re-structure dataframe
    dfm = pd.melt(lcrp_data_bsa, id_vars=['subject', 'strategy', 'condition', 'l_length', 'pres_rate'], 
              value_vars=[f'ln_{x}' for x in np.arange(7, 0, -1)] + [f'lp_{x}' for x in range(1, 8)], 
              var_name='lag', value_name='crp')
    dfm['lag'] = [-int(x.split('_')[-1]) if x.split('_')[0] == 'ln' else int(x.split('_')[-1]) for x in dfm.lag]    # change lag to ints

    fig, ax = plt.subplots(2, 3, figsize=(9, 6), sharey=True, sharex=True)
    conds_tt = ['10-2', '20-1', '15-2', '30-1', '20-2', '40-1']

    for i, c in enumerate(conds_tt):
        lcrp_cond = dfm.query("condition == @c")

        sns.lineplot(lcrp_cond.query("lag < 0"), x='lag', y='crp', hue='strategy',
                     hue_order=['prim', 'ns', 'rec'], palette=['orange', 'darkgray', 'purple'], 
                     errorbar=('se', 1.96), ax=ax[i%2, i//2])
        sns.lineplot(lcrp_cond.query("lag > 0"), x='lag', y='crp', hue='strategy',
                     hue_order=['prim', 'ns', 'rec'], palette=['orange', 'darkgray', 'purple'], 
                     errorbar=('se', 1.96), ax=ax[i%2, i//2], legend=False)

        handles, _ = ax[i%2, i//2].get_legend_handles_labels()
        ax[i%2, i//2].get_legend().remove()

        ax[i%2, i//2].set(title=c, xlabel='', xticks=np.linspace(-7, 7, 8), ylabel='')
        ax[i%2, i//2].spines[['right', 'top']].set_visible(False)


    fig.supxlabel('Lag', x=0.53)
    fig.supylabel('Conditional Response Probability')
    labels = ['Primacy', 'Other', 'Recency']
    fig.legend(handles, labels, shadow=True, ncols=3, loc='upper center', bbox_to_anchor=(0.53, 1.08))

    plt.tight_layout()
    plt.savefig('figures/gallery/lcrp.pdf', bbox_inches='tight')
    plt.show()


# semantic clustering score
def plot_scl(scl_data_bsa, scl_stats):
    fig, ax = plt.subplots(figsize=(8,5))
    conds_ll_pr = ['10-2', '15-2', '20-2', '20-1', '30-1', '40-1']

    sns.barplot(scl_data_bsa, x='condition', y='scl', order=conds_ll_pr, hue='strategy', hue_order=['prim', 'ns', 'rec'], 
                palette=['orange', 'darkgray', 'purple'], alpha=0.7, errorbar=('se', 1.96), gap=0.1)

    h = 0.25; v = 0.58
    # statistics
    for i, c in enumerate(conds_ll_pr):
        prim_stats = scl_stats[(scl_stats.condition == c) & (scl_stats.strategy == 'prim')].iloc[0]
        ns_stats = scl_stats[(scl_stats.condition == c) & (scl_stats.strategy == 'ns')].iloc[0]
        rec_stats = scl_stats[(scl_stats.condition == c) & (scl_stats.strategy == 'rec')].iloc[0]

        if prim_stats.p_val_fdr < 0.05:
            ax.scatter([i-h], [v], color='orange', marker='*', s=50, alpha=0.7)
        if ns_stats.p_val_fdr < 0.05:
             ax.scatter([i], [v], color='darkgray', marker='*', s=50, alpha=0.7)
        if rec_stats.p_val_fdr < 0.05:
             ax.scatter([i+h], [v], color='purple', marker='*', s=50, alpha=0.7)

    ax.set(xlabel='Condition', ylabel='Semantic Clustering Score', ylim=(0.45, 0.6))
    ax.spines[['right', 'top']].set_visible(False)

    labels = ['Primacy', 'Other', 'Recency']
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, labels, shadow=True, ncols=3, loc='upper right', bbox_to_anchor=(1, 1.05))
    
    plt.savefig('figures/gallery/scl.pdf', bbox_inches='tight')
    plt.show()

    
# ---------- Other Group Analyses ----------

# mean words recalled
def plot_mwr_ns(mwr_ns_data_bsa):
    fig, ax = plt.subplots(2, 3, figsize=(6, 6), sharey=True, sharex=True)
    conds_tt = ['10-2', '20-1', '15-2', '30-1', '20-2', '40-1']

    for i, c in enumerate(conds_tt):
        mwr_cond = mwr_ns_data_bsa.query("condition == @c")

        sns.pointplot(mwr_cond, x='r1_label', y='mwr', hue='r1_label', palette=['palevioletred', 'steelblue'], 
                      errorbar=('se', 1.96), ax=ax[i%2, i//2], alpha=0.9)

        ax[i%2, i//2].spines[['right', 'top']].set_visible(False)
        ax[i%2, i//2].set(xlabel='', ylabel='', title=c, ylim=(0, 23), yticks=np.linspace(0, 20, 5))
        ax[i%2, i//2].set_xticks([0, 1], labels=['Primacy', 'Recency'])

    fig.supxlabel('Recall Initiation Serial Position', x=0.53)
    fig.supylabel('Mean Words Recalled')
    
    plt.savefig('figures/gallery/mwr_ns.pdf', bbox_inches='tight')
    plt.show()
    
    
# initial response times
def plot_rti_ns(rti_at_ns_data, rti_ns_data_bsa):
    fig, ax = plt.subplots(2, 3, figsize=(9, 6), sharey=True, sharex=True)
    conds_tt = ['10-2', '20-1', '15-2', '30-1', '20-2', '40-1']

    for i, c in enumerate(conds_tt):
        ll = int(c.split('-')[0])
        prim_data = rti_at_ns_data[(rti_at_ns_data.condition == c) & (rti_at_ns_data.min_rt == False) &
                                   (rti_at_ns_data.serial_position >= 1) & (rti_at_ns_data.serial_position <= 4)]
        rec_data = rti_at_ns_data[(rti_at_ns_data.condition == c) & (rti_at_ns_data.min_rt == False) &
                                  (rti_at_ns_data.serial_position >= ll-3) & (rti_at_ns_data.serial_position <= ll)]
        bins = np.arange(0, max(np.concatenate([prim_data.rt, rec_data.rt]))+500, 500)

        # trial histograms
        _, _, _ = ax[i%2, i//2].hist(prim_data.rt, bins=bins, density=True, color='palevioletred', alpha=0.4, label='R1 Primacy')
        _, _, _ = ax[i%2, i//2].hist(rec_data.rt, bins=bins, density=True, color='steelblue', alpha=0.4, label='R1 Recency')

        # between-subject averages
        ax[i%2, i//2].axvline(rti_ns_data_bsa.query("condition == @c and r1_label == 'rt_prim'").rt.mean(), color='palevioletred', linestyle='dashed')
        ax[i%2, i//2].axvline(rti_ns_data_bsa.query("condition == @c and r1_label == 'rt_rec'").rt.mean(), color='steelblue', linestyle='dashed')

        ax[i%2, i//2].grid(True)
        ax[i%2, i//2].spines[['right', 'top']].set_visible(False)
        ax[i%2, i//2].set(title=c, xlabel='', xlim=(0, 20000), ylabel='')

    fig.supxlabel('Response Time (ms)', x=0.53)
    fig.supylabel('Proportion of Trials')
    handles, labels = ax[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, shadow=True, ncols=3, loc='upper center', bbox_to_anchor=(0.53, 1.08))

    plt.tight_layout()
    plt.savefig('figures/gallery/rti_ns.pdf', bbox_inches='tight')
    plt.show()
    
    
# temporal clustering score
def plot_tcl_ns(tcl_ns_data_bsa):
    fig, ax = plt.subplots(figsize=(7, 5))
    conds_ll_pr = ['10-2', '15-2', '20-2', '20-1', '30-1', '40-1']

    sns.barplot(tcl_ns_data_bsa, x='condition', y='tcl', order=conds_ll_pr, hue='r1_label', hue_order=['tcl_prim', 'tcl_rec'],
                palette=['palevioletred', 'steelblue'], gap=0.1, alpha=0.9, errorbar=('se', 1.96), ax=ax)

    ax.set(xlabel='Condition', ylabel='Temporal Clustering Score', ylim=(0.45, 1))
    ax.spines[['right', 'top']].set_visible(False)

    labels = ['R1 Primacy', 'R1 Recency']
    handles, _ = ax.get_legend_handles_labels()
    ax.get_legend().remove()
    fig.legend(handles, labels, shadow=True, ncols=2, loc='upper right', bbox_to_anchor=(0.9, 0.9))
    
    plt.savefig('figures/gallery/tcl_ns.pdf', bbox_inches='tight')
    plt.show()