### Plotting code (figures not in paper)

# imports
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# formatting parameters
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# probability of second recall intrusion following intrusion
def plot_p2r_intr(p2r_intr_data_bsa):
    fig, ax = plt.subplots(figsize=(8, 5))
    conds_ll_pr = ['10-2', '15-2', '20-2', '20-1', '30-1', '40-1']


    sns.barplot(p2r_intr_data_bsa, x='condition', y='intrusion', order=conds_ll_pr,
                hue='strategy', hue_order=['prim', 'ns', 'rec'], palette=['orange', 'darkgray', 'purple'], 
                alpha=0.7, errorbar=('se', 1.96), gap=0.1)
    
    ax.set(xlabel='Condition', ylabel='Probability of 2nd Recall Intruion\nFollowing R1 Intrusion', ylim=(0, 0.41))
    ax.spines[['right', 'top']].set_visible(False)
    
    # legend
    labels = ['Primacy', 'Other', 'Recency']
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, labels, shadow=True, ncols=3, loc='upper right', bbox_to_anchor=(1, 1.08))

    plt.savefig('figures/gallery/p2r_intr.pdf', bbox_inches='tight')
    plt.show()


# probability of second recall following intrusion
def plot_p2r(p2r_intr_data_bsa):
    # re-organize dataframe
    dfm = pd.melt(p2r_intr_data_bsa, id_vars=['subject', 'strategy', 'condition', 'l_length', 'pres_rate'],
                  value_vars=[f'sp_{x}' for x in range(1, 41)], var_name='serial_position', value_name='recall_probability')
    dfm['serial_position'] = [int(x.split('_')[-1]) for x in dfm.serial_position]         # change serial positions to ints

    fig, ax = plt.subplots(2, 3, figsize=(9, 6), sharey=True, gridspec_kw={'width_ratios': [1, 1.5, 2]})
    conds_tt = ['10-2', '20-1', '15-2', '30-1', '20-2', '40-1']

    for i, c in enumerate(conds_tt):
        p2r_cond = dfm.query('condition == @c')
        ll = p2r_cond.l_length.unique()[0]

        sns.lineplot(p2r_cond, x='serial_position', y='recall_probability', hue='strategy', hue_order=['prim', 'ns', 'rec'],
                     palette=['orange', 'darkgray', 'purple'], errorbar=('se', 1.96), err_style='bars', marker='o', ax=ax[i%2, i//2])
        handles, _ = ax[i%2, i//2].get_legend_handles_labels()
        ax[i%2, i//2].get_legend().remove()

        xax = np.prod(np.array(c.split('-')).astype(int))
        ax[i%2, i//2].set(title=c, xlabel='', xticks=np.arange(5, xax+1, 5), xlim=(0, xax+1), ylabel='', yticks=np.linspace(0, 0.7, 8))
        if i%2 == 0:
            ax[i%2, i//2].set_xticklabels([])
        ax[i%2, i//2].spines[['right', 'top']].set_visible(False)

    fig.supxlabel('Serial Position')
    fig.supylabel('Probability of 2nd Recall Following R1 Intrusion')
    labels = ['Primacy', 'Other', 'Recency']
    fig.legend(handles, labels, shadow=True, ncols=3, loc='upper center', bbox_to_anchor=(0.5, 1.08))

    plt.tight_layout()
    plt.savefig('figures/gallery/p2r.pdf', bbox_inches='tight')
    plt.show()
    
    
# conditioned on serial position
def plot_lcrp_sp(lcrp_sp_data_bsa, lag):
    # re-structure dataframe
    dfm = pd.melt(lcrp_sp_data_bsa, id_vars=['subject', 'strategy', 'condition', 'l_length', 'pres_rate', 'serial_position'], 
                  value_vars=[f'ln_{x}' for x in np.arange(7, 0, -1)] + [f'lp_{x}' for x in range(1, 8)], var_name='lag', value_name='crp')
    dfm['lag'] = [-int(x.split('_')[-1]) if x.split('_')[0] == 'ln' else int(x.split('_')[-1]) for x in dfm.lag]    # change lag to ints

    fig, ax = plt.subplots(2, 3, figsize=(9, 6), sharey=True, gridspec_kw={'width_ratios': [1, 1.5, 2]})


    conds_tt = ['10-2', '20-1', '15-2', '30-1', '20-2', '40-1']
    for i, c in enumerate(conds_tt):
        ll = int(c.split('-')[0])

        lcrp_cond = dfm.query("condition == @c and lag == @lag and serial_position <= @ll")

        sns.lineplot(lcrp_cond, x='serial_position', y='crp', hue='strategy', hue_order=['prim', 'ns', 'rec'],
                 palette=['orange', 'darkgray', 'purple'], ax=ax[i%2, i//2])

        handles, _ = ax[i%2, i//2].get_legend_handles_labels()
        ax[i%2, i//2].get_legend().remove()

        xax = np.prod(np.array(c.split('-')).astype(int))
        ax[i%2, i//2].set(xlabel='', xticks=np.arange(5, xax+1, 5), xlim=(0, xax+1), ylabel='', title=c)
        if i%2 == 0:
            ax[i%2, i//2].set_xticklabels([])
        ax[i%2, i//2].spines[['right', 'top']].set_visible(False)

    fig.supxlabel('Serial Position')
    fig.supylabel('Conditional Response Probability')
    labels = ['Primacy', 'Other', 'Recency']
    fig.legend(handles, labels, shadow=True, ncols=3, loc='upper center', bbox_to_anchor=(0.5, 1.08))

    plt.tight_layout()
    plt.savefig('figures/gallery/lcrp_sp.pdf', bbox_inches='tight')
    plt.show()


# lag +1 v. -1 asymmetry (both transitions available)
def plot_lcrp_l1(lcrp_l1_data_bsa):
    # re-organize dataframe
    dfm = pd.melt(lcrp_l1_data_bsa, id_vars=['subject', 'strategy', 'condition', 'l_length', 'pres_rate', 'crp_delta'],
              value_vars=['crp_p1', 'crp_n1'], var_name='lag', value_name='crp')
    dfm['lag'] = [-int(x[-1]) if x[-2] == 'n' else int(x[-1]) if x[-2] == 'p' else np.nan for x in dfm.lag]    # convert lags to ints
    
    fig, ax = plt.subplots(2, 3, figsize=(7, 5), sharex=True, sharey=True)
    conds_tt = ['10-2', '20-1', '15-2', '30-1', '20-2', '40-1']

    for i, c in enumerate(conds_tt):
        cond_data = dfm.query("condition == @c")
        sns.pointplot(cond_data, x='lag', y='crp', hue='strategy', hue_order=['prim', 'ns', 'rec'],
                      palette=['orange', 'darkgray', 'purple'], errorbar=('se', 1.96), dodge=0.3, alpha=0.85, ax=ax[i%2, i//2])
        handles, _ = ax[i%2, i//2].get_legend_handles_labels()
        ax[i%2, i//2].get_legend().remove()

        ax[i%2, i//2].set(title=c, xlabel='', ylabel='')
        ax[i%2, i//2].spines[['right', 'top']].set_visible(False)

    fig.supxlabel('Lag', x=0.53)
    fig.supylabel('Conditional Response Probability')
    labels = ['Primacy', 'Other', 'Recency']
    fig.legend(handles, labels, shadow=True, ncols=3, loc='upper center', bbox_to_anchor=(0.53, 1.1))

    plt.tight_layout()
    plt.savefig('figures/gallery/lcrp_l1.pdf', bbox_inches='tight')
    plt.show()