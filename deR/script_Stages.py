import matplotlib.pyplot as plt
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import minmax_scale
import matplotlib.patches as mpatches
from scipy.stats import rankdata

from networks.library_regulons import ClusterAnalyser, NeighbourCalculator, make_tsne
from networks.functionsDENet import loadPickle
from deR.stages_library import *

path = '/home/karin/Documents/timeTrajectories/data/deTime/de_time_impulse/'
lab = True
if lab:
    dataPath = '/home/karin/Documents/timeTrajectories/data/RPKUM/combined/'
    pathSelGenes = '/home/karin/Documents/timeTrajectories/data/regulons/selected_genes/'

else:
    dataPath = '/home/karin/Documents/DDiscoideum/data/RPKUM/'

genes = pd.read_csv(dataPath + 'mergedGenes_RPKUM.tsv', sep='\t', index_col=0)
conditions = pd.read_csv(dataPath + 'conditions_mergedGenes.tsv', sep='\t', index_col=None)

# ******  How variable are  strains between replicates - plot variability as boxplots
# Stds of genes  in strains+timepoints (std of replicates for a gene) divided by mean
genes_conditions = ClusterAnalyser.merge_genes_conditions(genes=genes, conditions=conditions, matching='Measurment')
groups = genes_conditions.groupby(['Strain', 'Time'])
variation = {strain: [] for strain in conditions['Strain'].unique()}
for group in groups:
    name = group[0][0]
    data = group[1].drop(list(conditions.columns), axis=1)
    std = (data.std() / data.mean()).dropna()
    variation[name].extend(std)
plt.boxplot(list(variation.values()))
plt.gca().set_xticklabels(list(variation.keys()), rotation=90)

# ******** tSNE of measurments
# Remove all 0 genes
data = genes[(genes != 0).any(axis=1)]
# Normalise data
names = data.columns
gene_names = data.index
data = pd.DataFrame(NeighbourCalculator.get_index_query(genes=data, inverse=False, scale='mean0std1', log=True
                                                        )[0].T, index=names, columns=gene_names)
# tSNE
tsne = make_tsne(data=data, perplexities_range=[50, 160], exaggerations=[1, 1],
                 momentums=[0.6, 0.9], random_state=0)
# Data for plotting
plot_data = pd.DataFrame(tsne, index=data.index, columns=['x', 'y'])
conditions_plot = conditions[['Replicate', 'Time', 'Group']]
conditions_plot.index = conditions['Measurment']
plot_data = pd.concat([plot_data, conditions_plot], axis=1)

# Plot tSNE with temporal info
colours = {'1Ag-': '#d40808', '2LAg': '#e68209', '3TA': '#d1b30a', '4CD': '#4eb314', '5WT': '#0fa3ab',
           '6SFB': '#525252', '7PD': '#7010b0'}
fig, ax = plt.subplots()
ax.scatter(plot_data['x'], plot_data['y'], s=minmax_scale(plot_data['Time'], (3, 30)),
           c=[colours[name] for name in plot_data['Group']], alpha=0.5)
for name, data_rep in plot_data.groupby('Replicate'):
    data_rep = data_rep.sort_values('Time')
    group = data_rep['Group'].values[0]
    ax.plot(data_rep['x'], data_rep['y'], color=colours[group], alpha=0.5, linewidth=0.5)
    ax.text(data_rep['x'][-1], data_rep['y'][-1], data_rep['Replicate'][0], fontsize=6)
ax.axis('off')
patchList = []
for name, colour in colours.items():
    data_key = mpatches.Patch(color=colour, label=name, alpha=0.5)
    patchList.append(data_key)
ax.legend(handles=patchList, title="Group")
fig.suptitle("t-SNE of measurements. Size denotes time; replicate's progression is market with a line.")

# **************************
# ***** Find genes that are more corelated with close neighbours in some strains but not the others
sims_dict_sim = loadPickle(
    pathSelGenes + 'newGenes_noAll-removeSelf_simsDict_scalemean0std1_logTrue_kN' + str(6) + '_splitStrain.pkl')
# Genes expressed in all strains:
sel_genes = set(genes.index.values)
for strain_data in sims_dict_sim.values():
    genes_strain = set(strain_data.index.values)
    sel_genes = sel_genes & genes_strain

# Quantile normalise similarities in strains (to get them to the same scale) based on avg across strains
# https://stackoverflow.com/a/41078786/11521462
# Get means of genes present in all strains
similarity_means = pd.DataFrame(index=sel_genes, columns=sims_dict_sim.keys())
for strain, data in sims_dict_sim.items():
    means = data.loc[similarity_means.index, :].mean(axis=1)
    similarity_means[strain]=means
# Get overall rank means
# Groupby groups all of the same rank and then averages values for rank
rank_mean = similarity_means.stack().groupby(similarity_means.rank(method='first').stack().astype(int)).mean()
# Normalise values
# Find (average) rank and map values to rank-specific values. If between 2 ranks uses their average
rank_df=similarity_means.rank(method='average')
quantile_normalised=np.empty(rank_df.shape)
quantile_normalised[:]=np.nan
for i in range(rank_df.shape[0]):
    for j in range(rank_df.shape[1]):
        rank=rank_df.iloc[i,j]
        if rank % 1 == 0:
            new_value=rank_mean[rank]
        else:
            rank_low=rank//1
            rank_high=rank_low+1
            new_value=(rank_mean[rank_low]+rank_mean[rank_high])/2
        quantile_normalised[i,j]=new_value
quantile_normalised=pd.DataFrame(quantile_normalised,index=rank_df.index,columns=rank_df.columns)


# Fit sigmoids
group_x_dict = {'1Ag-': 1, '2LAg': 2, '3TA': 3, '4CD': 4,'6SFB':5, '5WT': 6, '7PD': 7}

fit_data = []
could_not_fit = 0
for gene in quantile_normalised.index:
    x = []
    y = []
    for strain in quantile_normalised.columns:
        group = GROUPS[strain]
        if group in group_x_dict.keys():
            x_pos = group_x_dict[group]
            avg_similarity = quantile_normalised.loc[gene, strain]
            x.append(x_pos)
            y.append(avg_similarity)
    try:
        # Sometimes params can not be estimated
        params = sigmoid_fit(x=x, y=y)[0]
        cost = relative_cost(x=x, y=y, function=sigmoid, params=params)
        fit_data.append({'Gene': gene, 'Cost': cost, 'L': params[0], 'x0': params[1], 'k': params[2], 'b': params[3]})
    except:
        could_not_fit = could_not_fit + 1
fit_data = pd.DataFrame(fit_data)
