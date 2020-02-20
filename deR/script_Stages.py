import matplotlib.pyplot as plt
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import minmax_scale
import matplotlib.patches as mpatches
from scipy.stats import rankdata, mannwhitneyu
from statsmodels.stats.multitest import multipletests

from networks.library_regulons import ClusterAnalyser, NeighbourCalculator, make_tsne
from networks.functionsDENet import loadPickle, savePickle
from deR.stages_library import *
from correlation_enrichment.library_correlation_enrichment import SimilarityCalculator

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

# ************************************************************************
# ***** Count in how many replicates per strain the gene is consistently 0
genes_rep = ClusterAnalyser.merge_genes_conditions(genes=genes, conditions=conditions[['Replicate', 'Measurment']],
                                                   matching='Measurment')
genes_rep = ClusterAnalyser.split_data(genes_rep, 'Replicate')
genes_zero_count = pd.DataFrame(np.zeros((genes.shape[0], conditions['Strain'].unique().shape[0])),
                                index=genes.index, columns=conditions['Strain'].unique())

for rep, data in genes_rep.items():
    data = data.drop(['Measurment', 'Replicate'], axis=1).T
    strain = conditions[conditions['Replicate'] == rep]['Strain'].values[0]
    data = (data == 0).all(axis=1)
    genes_zero_count[strain] = genes_zero_count[strain] + data
genes_zero_count.to_csv(dataPath + 'zero_replicates_count.tsv', sep='\t')
# ***************************
# ** Similarity to closest neighbours in strains
# Put 'avg similarity' of all 0 genes to lowest similarity of that strain (many genes should be all 0 but are not -
# erroneous mapping (not consistent shape between replicates). The low similarities may be results of unexpressed genes.
# Also do that for genes with at least one replicate all 0.
# All these genes are removed in extraction of closest neighbours
NEIGHBOURS = 6
sims_dict = dict()

# Split data by strain, scaling and zero filtering is done in neighbours
merged = ClusterAnalyser.merge_genes_conditions(genes=genes, conditions=conditions[['Measurment', 'Strain']],
                                                matching='Measurment')
splitted = ClusterAnalyser.split_data(data=merged, split_by='Strain')
for strain, data in splitted.items():
    splitted[strain] = data.drop(["Strain", 'Measurment'], axis=1).T

# Calculate neighbours - sims_dict has similarity matrices from samples
for strain, data in splitted.items():
    print(strain)
    nonzero = genes_zero_count[genes_zero_count[strain] == 0].index
    data = data.loc[nonzero, :]
    neighbour_calculator = NeighbourCalculator(genes=data)
    neigh, sims_dict[strain] = neighbour_calculator.neighbours(n_neighbours=NEIGHBOURS, inverse=False,
                                                               scale='mean0std1',
                                                               log=True,
                                                               return_neigh_dist=True, remove_self=True)

savePickle(pathSelGenes + 'newGenes_noAll-removeSelf-removeZeroRep_simsDict_scalemean0std1_logTrue_kN' +
           str(NEIGHBOURS) + '_splitStrain.pkl', sims_dict)
# ***** Similarity to closest neighbours of AX4 across strains - do neighbourhoods change
NEIGHBOURS = 11
sims_dict_WT = dict()
# Closest neighbours in AX4
strain = 'AX4'
nonzero = genes_zero_count[genes_zero_count[strain] == 0].index
data = splitted[strain]
data = data.loc[nonzero, :]
neighbour_calculator = NeighbourCalculator(genes=data)
neigh_WT, sims_dict_WT[strain] = neighbour_calculator.neighbours(n_neighbours=NEIGHBOURS, inverse=False,
                                                                 scale='mean0std1',
                                                                 log=True,
                                                                 return_neigh_dist=True, remove_self=True)
for strain, data in splitted.items():
    if strain != 'AX4':
        print(strain)
        nonzero = set(genes_zero_count[genes_zero_count[strain] == 0].index)
        genes_WT = set(neigh_WT.index)
        nonzero = list(nonzero & genes_WT)
        data = data.loc[nonzero, :]
        data = pd.DataFrame(
            NeighbourCalculator.get_index_query(genes=data, inverse=False, scale='mean0std1', log=True)[0],
            index=data.index, columns=data.columns)
        n_genes = len(nonzero)
        similarities = np.empty((n_genes, NEIGHBOURS - 1))
        similarities[:] = np.nan
        for idx_gene in range(n_genes):
            gene = nonzero[idx_gene]
            neighbours_WT = neigh_WT.loc[gene, :].values
            for idx_neigh in range(NEIGHBOURS - 1):
                neigh = neighbours_WT[idx_neigh]
                if neigh in data.index:
                    similarities[idx_gene][idx_neigh] = SimilarityCalculator.calc_cosine(data.loc[gene, :],
                                                                                         data.loc[neigh, :])
        similarities = pd.DataFrame(similarities, index=nonzero, columns=range(NEIGHBOURS - 1))
        sims_dict_WT[strain] = similarities

savePickle(
    pathSelGenes + 'AX4basedNeigh_newGenes-removeZeroRep_neighSimsDict_scalemean0std1_logTrue_kN' + str(NEIGHBOURS) +
    '_splitStrain.pkl', (neigh_WT, sims_dict_WT))

# **************************
# ***** Find genes that are more correlated with close neighbours in some strains but not the others
sims_dict = loadPickle(
    pathSelGenes + 'newGenes_noAll-removeSelf-removeZeroRep_simsDict_scalemean0std1_logTrue_kN' + str(6) +
    '_splitStrain.pkl')
# Genes expressed in all strains:
# sel_genes = set(genes.index.values)
# for strain_data in sims_dict.values():
#    genes_strain = set(strain_data.index.values)
#    sel_genes = sel_genes & genes_strain

# Quantile normalise similarities in strains (to get them to the same scale) based on avg across strains
# https://stackoverflow.com/a/41078786/11521462
# Get means of gene similarities. If similarity was not calculated for a gene (because it had 0 expression), set it to
# min avg similarity
similarity_means = similarity_mean_df(sims_dict=sims_dict, index=genes.index, replace_na_sims=None,
                                      replace_na_mean='min')
# Get overall rank means
quantile_normalised = quantile_normalise(similarity_means=similarity_means)

quantile_normalised.to_csv(
    pathSelGenes + 'simsQuantileNormalised_newGenes_noAll-removeSelf-removeZeroRep_simsDict_scalemean0std1_logTrue_kN6_splitStrain.tsv',
    sep='\t')

# # Fit sigmoids to find if similarities follow the shape (two plateaus) - not producing ok results
#
# fit_data = []
# could_not_fit = 0
# for gene in quantile_normalised.index:
#     x = []
#     y = []
#     for strain in quantile_normalised.columns:
#         group = GROUPS[strain]
#         if group in GROUP_X.keys():
#             x_pos = GROUP_X[group]
#             avg_similarity = quantile_normalised.loc[gene, strain]
#             x.append(x_pos)
#             y.append(avg_similarity)
#     try:
#         # Sometimes params can not be estimated
#         params = sigmoid_fit(x=x, y=y)[0]
#         cost = relative_cost(x=x, y=y, function=sigmoid, params=params)
#         fit_data.append({'Gene': gene, 'Cost': cost, 'L': params[0], 'x0': params[1], 'k': params[2], 'b': params[3]})
#     except:
#         could_not_fit = could_not_fit + 1
# fit_data = pd.DataFrame(fit_data)

# *** Mann–Whitney U for omparing similarities distribution between strain groups
# How to decide if strain groes into group 1 vs 2 based on group_x_dict
# Tells which comparison on the strain developmental timeline to make
# First element group1, second group2, third comparison name
group_splits = [
    ([1], [2, 3, 4, 6, 7], 1),
    ([1, 2], [3, 4, 6, 7], 2),
    ([1, 2, 3], [4, 6, 7], 3),
    ([1, 2, 3, 4], [6, 7], 4)
]

results = compare_sims_groups(quantile_normalised=quantile_normalised, group_splits=group_splits,
                              test_params={'alternative': 'two-sided'})

results.to_csv(
    pathSelGenes + 'comparisonsAvgSims_newGenes_noAll-removeSelf-removeZeroRep_simsDict_scalemean0std1_logTrue_kN6_splitStrain.tsv',
    sep='\t', index=False)

# ************ Find genes that change neighbourhood compared to AX4
sims_dict_WT=loadPickle(
    pathSelGenes + 'AX4basedNeigh_newGenes-removeZeroRep_neighSimsDict_scalemean0std1_logTrue_kN11_splitStrain.pkl')[1]

similarity_means_WT = similarity_mean_df(sims_dict=sims_dict_WT, index=genes.index, replace_na_sims=0,
                                      replace_na_mean='zero')
quantile_normalised_WT = quantile_normalise(similarity_means=similarity_means_WT)

quantile_normalised_WT.to_csv(
    pathSelGenes + 'simsQuantileNormalised_AX4basedNeigh_newGenes_noAll-removeZeroRep_simsDict_scalemean0std1_logTrue_kN11_splitStrain.tsv',
    sep='\t')

results_WT = compare_sims_groups(quantile_normalised=quantile_normalised_WT, group_splits=group_splits,
                              test_params={'alternative': 'less'})

results_WT.to_csv(
    pathSelGenes + 'comparisonsAvgSims_AX4basedNeigh_newGenes_noAll-removeZeroRep_simsDict_scalemean0std1_logTrue_kN11_splitStrain.tsv',
    sep='\t', index=False)
