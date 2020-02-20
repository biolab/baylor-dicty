import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hc
import pandas as pd
from statistics import mean
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

from orangecontrib.bioinformatics.utils.statistics import Hypergeometric
from Orange.clustering.louvain import jaccard
from scipy.optimize import curve_fit

GROUPS = {'amiB': '1Ag-', 'mybB': '1Ag-', 'acaA': '1Ag-', 'gtaC': '1Ag-',
          'gbfA': '2LAg', 'tgrC1': '2LAg', 'tgrB1': '2LAg', 'tgrB1C1': '2LAg',
          'tagB': '3TA', 'comH': '3TA',
          'ecmARm': '4CD', 'gtaI': '4CD', 'cudA': '4CD', 'dgcA': '4CD', 'gtaG': '4CD',
          'AX4': '5WT', 'MybBGFP': '5WT',
          'acaAPkaCoe': '6SFB', 'ac3PkaCoe': '6SFB',
          'pkaR': '7PD', 'PkaCoe': '7PD'}

GROUP_X = {'1Ag-': 1, '2LAg': 2, '3TA': 3, '4CD': 4, '6SFB': 5, '5WT': 6, '7PD': 7}

GROUP_DF = []
for strain, group in GROUPS.items():
    GROUP_DF.append({'Strain': strain, 'Group': group, 'X': GROUP_X[group]})
GROUP_DF = pd.DataFrame(GROUP_DF)


# GROUP_DF.to_csv('/home/karin/Documents/timeTrajectories/data/regulons/selected_genes/group_df.tsv',sep='\t',
# index=False)

def plot_genegroup_similarity(retained_genes_dict, splitby='Strain', jaccard_or_p=True, n_all_genes: int = None,
                              group_colours=None, add_title=''):
    """
    Makes hc dendrogram of gene group similarity.
    :param retained_genes_dict: retained genes (values), group names (keys)
    :param splitby: Use in hc title, what are the groups
    :param n_all_genes: Used for p value calculation if jaccard_or_p=False, what to use for N in hypergeometric test
    :param jaccard_or_p: Use jaccard similarity or hypergeometric p value for distances.
    P values are -log10 transformed and resulting similarities are converted into distances by subtraction.
    :param group_colours: Colour labels based on this colour dict. Key as in retained_genes_dict, value a colour
    :param add_title: add at ned of title
    """
    replicates = list(retained_genes_dict.keys())
    # Calculates similarities between retained genes of different samples
    dist_arr = []
    if not jaccard_or_p:
        hypergeom_test = Hypergeometric(n_all_genes)
        min_p = 10 ** -323.6
        max_sim = -np.log10(min_p)
    for idx, rep1 in enumerate(replicates[:-1]):
        for rep2 in replicates[idx + 1:]:
            genes1 = set(retained_genes_dict[rep1])
            genes2 = set(retained_genes_dict[rep2])
            if jaccard_or_p:
                jaccard_index = jaccard(genes1, genes2)
                dist_arr.append(1 - jaccard_index)
            else:
                intersection = len(genes1 & genes2)
                p = hypergeom_test.p_value(k=intersection, N=n_all_genes, m=len(genes1), n=len(genes2))
                if p < min_p:
                    p = min_p
                sim = -np.log10(p)
                dist_arr.append(max_sim - sim)

    # Plot similarity
    fig, ax = plt.subplots(figsize=(10, 5))
    hc.dendrogram(hc.ward(dist_arr), labels=replicates, color_threshold=0, leaf_rotation=90)
    if jaccard_or_p:
        test = 'jaccard distance'
    else:
        test = 'hypergeometric p value'
    plt.title('Clustered ' + splitby + 's based on ' + test + ' of selected genes' + add_title)
    # Colour by strain  group
    xlbs = ax.get_xmajorticklabels()
    for xlb in xlbs:
        if xlb.get_text() in group_colours.keys():
            xlb.set_color(group_colours[xlb.get_text()])


def gene_heatmap(genes, genes_dict):
    """
    Make DF
    """
    n_genes = len(genes.index)
    n_strains = len(genes_dict.keys())
    genes_order = dict(zip(list(genes.index), range(n_genes)))
    strain_order = dict(zip(list(genes_dict.keys()), range(n_strains)))
    strain_genes = np.zeros((n_strains, n_genes))

    for strain, selected_genes in genes_dict.items():
        row = strain_order[strain]
        for gene in selected_genes:
            strain_genes[row, genes_order[gene]] = 1
    strain_genes = pd.DataFrame(strain_genes, index=strain_order.keys(), columns=genes_order.keys())
    return strain_genes


def group_gene_heatmap(genes_df, groups: dict = GROUPS, mode: str = 'sum'):
    if mode == 'sum':
        by_group = genes_df.groupby(by=groups, axis=0).sum()
    elif mode == 'average':
        by_group = genes_df.groupby(by=groups, axis=0).mean()
    return by_group


def sigmoid_fit(x, y):
    x = np.array(x)
    y = np.array(y)
    x_diff = (x.max() - x.min()) * 0.1
    # y_diff_01=(y.max()-y.min())*0.1
    bounds = ([y.min(), x.min() + x_diff, -np.inf, y.min()], [y.max(), x.max() - x_diff, np.inf, y.max()])
    p0 = [max(y), np.median(x), 1, min(y)]  # initial guess
    return curve_fit(sigmoid, x, y, p0, bounds=bounds)


def sigmoid(x, L, x0, k, b):
    y = L / (1 + np.exp(-k * (x - x0))) + b
    return (y)


def relative_cost(x, y, function, params):
    yp = []
    for xi in x:
        yp.append(function(xi, *params))
    return mean([((ypi - yi) / yi) ** 2 for ypi, yi in zip(yp, y)])


def similarity_mean_df(sims_dict: dict, index: list, replace_na_sims: float = None, replace_na_mean: str = None):
    similarity_means = pd.DataFrame(index=index, columns=sims_dict.keys())
    for strain, data in sims_dict.items():
        if replace_na_sims is not None:
            data = data.replace(np.nan, replace_na_sims)
        means = data.reindex(similarity_means.index).mean(axis=1)
        if replace_na_mean == 'min':
            means = means.replace(np.nan, means.min())
        elif replace_na_mean == 'zero':
            means = means.replace(np.nan, 0)
        similarity_means[strain] = means
    return similarity_means


def quantile_normalise(similarity_means):
    # Groupby groups all of the same rank and then averages values for rank
    rank_mean = similarity_means.stack().groupby(similarity_means.rank(method='first').stack().astype(int)).mean()
    # Normalise values
    # Find (average) rank and map values to rank-specific values. If between 2 ranks uses their average
    rank_df = similarity_means.rank(method='average')
    quantile_normalised = np.empty(rank_df.shape)
    quantile_normalised[:] = np.nan
    for i in range(rank_df.shape[0]):
        for j in range(rank_df.shape[1]):
            rank = rank_df.iloc[i, j]
            if rank % 1 == 0:
                new_value = rank_mean[rank]
            else:
                rank_low = rank // 1
                rank_high = rank_low + 1
                new_value = (rank_mean[rank_low] + rank_mean[rank_high]) / 2
            quantile_normalised[i, j] = new_value
    return pd.DataFrame(quantile_normalised, index=rank_df.index, columns=rank_df.columns)


def compare_sims_groups(quantile_normalised: pd.DataFrame, group_splits: list, test_params=None):
    results = []
    for gene in quantile_normalised.index:
        for comparison in group_splits:
            strains1 = GROUP_DF[GROUP_DF['X'].isin(comparison[0])]['Strain']
            strains2 = GROUP_DF[GROUP_DF['X'].isin(comparison[1])]['Strain']
            values1 = quantile_normalised.loc[gene, strains1].values
            values2 = quantile_normalised.loc[gene, strains2].values
            result = mannwhitneyu(values1, values2, **test_params)

            m1 = values1.mean()
            m2 = values2.mean()

            strains_last1 = GROUP_DF[GROUP_DF['X'] == comparison[0][-1]]['Strain']
            values_last1 = quantile_normalised.loc[gene, strains_last1].values
            mean_last1 = values_last1.mean()
            dif_1 = m2 - mean_last1
            strains_first2 = GROUP_DF[GROUP_DF['X'] == comparison[1][0]]['Strain']
            values_first2 = quantile_normalised.loc[gene, strains_first2].values
            mean_first2 = values_first2.mean()
            dif_2 = mean_first2 - m1
            separation = min([dif_1, dif_2], key=abs)

            results.append({'Gene': gene, 'Comparison': comparison[2], 'U': result[0], 'p': result[1],
                            'mean1': m1, 'mean2': m2, 'difference': m2 - m1, 'separation': separation})

    results = pd.DataFrame(results)
    # Adjust all pvals to FDR
    results['FDR'] = multipletests(pvals=results['p'], method='fdr_bh', is_sorted=False)[1]
    return results
