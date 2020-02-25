import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hc
import pandas as pd
from statistics import mean
from scipy.stats import mannwhitneyu, ttest_ind
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
    Make DF with selected genes of each strain.
    :param genes: DF with genes in index - uses index to make columns of resulting DF.
    :param genes_dict: Keys are strains (used for index in resulting DF) and values are lists of selected genes.
    :return: DF with strains in rows, genes in columns. Filled with 0 if not selected and 1 if selected.
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
    """
    Group DF of selected genes from gene_heatmap by (strain) groups.
    :param genes_df: Result from gene_heatmap, strains in rows, genes in columns.
    :param groups: Dict with indices as keys and values representing groups
    :param mode: Merge with summing or averaging
    :return: Selected genes DF with groups in rows and genes in columns.
    """
    if mode == 'sum':
        by_group = genes_df.groupby(by=groups, axis=0).sum()
    elif mode == 'average':
        by_group = genes_df.groupby(by=groups, axis=0).mean()
    return by_group


def sigmoid_fit(x, y):
    """
    Fit sigmoid function to x and y.
    :param x , y: Sequences of x and y values that can be transformed into np.array
    :return: Result of curve_fit with parameters as in sigmoid (L, x0, k, b) function
    """
    x = np.array(x)
    y = np.array(y)
    x_diff = (x.max() - x.min()) * 0.1
    # y_diff_01=(y.max()-y.min())*0.1
    bounds = ([y.min(), x.min() + x_diff, -np.inf, y.min()], [y.max(), x.max() - x_diff, np.inf, y.max()])
    p0 = [max(y), np.median(x), 1, min(y)]  # initial guess
    return curve_fit(sigmoid, x, y, p0, bounds=bounds)


def sigmoid(x: float, L: float, x0: float, k: float, b: float):
    """
    Calculate value (y) of sigmoid function based on x and parameters
    :param x: Single X value
    :param L, x0, k, b: used to fit sigmoid y = L / (1 + np.exp(-k * (x - x0))) + b
        L,b - regulate y span, k - steepness of the step, x0 - regulates x position of y=0.5
    :return: y value at that x.
    """
    y = L / (1 + np.exp(-k * (x - x0))) + b
    return (y)


def relative_cost(x: list, y: list, function, params: list):
    """
    Relative MSE of function fit.
    :param x,y: x and y values.
    :param function: Function used to predict y based on x position and parameters.
        Accepts one x at a time and returns the y. f(x,params) -> y
    :param params: Params in order as accepted by the function, excluding x param of function
    :return: Relative MSE, each error (difference between predicted and true y)  is divided by true y before squaring
    """
    yp = []
    for xi in x:
        yp.append(function(xi, *params))
    return mean([((ypi - yi) / yi) ** 2 for ypi, yi in zip(yp, y)])


def similarity_mean_df(sims_dict: dict, index: list, replace_na_sims: float = None, replace_na_mean: str = None):
    """
    Merges gene neighbour similarities of multiple strains into single DF. Transform similarities dict
        (strains as keys and values named similarity matrices as from pynndescent)
        into a table where similarities to all neighbours of a gene are averaged for each gene within a strain.
    :param sims_dict: Dict with strains qas keys and values similarity DFs as in pynndescent but named - index is
        gene names, values are similarities to closest N neighbours (0 to N-1 in columns)
    :param index: All genes to be included into the DF (as index).
    :param replace_na_sims: If np.nan are present in similariti DFs form sins_dict replace them with replace_na_sims
        before averaging. If none does not use replacement.
    :param replace_na_mean: If np.nan are present in means obtained for each gene replace them (e. g. because more genes
        are in index than in individual sims_dict). Options: 'min' replace with min mean of that strain/key,
        'zero' replace with zero.
    :return: DF with genes in row and sims_dict keys in columns. For each gene,key pair the neighbours in each sims_dict
        key  were averaged, so that each gene has one averaged value per sims_dict key.
    """
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


def quantile_normalise(similarity_means,return_ranks:bool=False):
    """
    Quantile normalise DF with samples in columns and values in rows, normalise columns to have same distribution.
    # https://stackoverflow.com/a/41078786/11521462
    Final mapping of sample ranks to rank means is done based on average rank.
    :param similarity_means: DF with samples in columns and values in rows.
    :return: DF of same shape as similarity_means, but with quantile normalised values for each columns.
    """

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
    normalised=pd.DataFrame(quantile_normalised, index=rank_df.index, columns=rank_df.columns)
    if not return_ranks:
        return normalised
    else:
        return normalised, rank_mean


def compare_gene_scores(quantile_normalised: pd.DataFrame, group_splits: list, test: str, alternative: str):
    """
    Compare gene scores across strain groups. For each gene there is a score (e.g. avg similarity to neighbours) in each
    strain that belongs to a strain group. Compare scores between groups to find genes that have lower/higher score in
    some set of groups.
    :param quantile_normalised: DF with strains in columns and genes in rows. For each gene in each strain there is a
        score. Strain groups are used as specified in GROUP_DF.
    :param group_splits: List of which comparisons between groups to perform. Each comparison is specified as a tupple.
        First tow elements are list of strain groups for comparison (e.g. strain groups in element1 vs strain groups in
        element 2). The strain groups must be given as in X column of GROUP_DF. The third element is name of the
        comparison that will be reported.
    :param test: 'u' for mannwhitneyu or 't' for t-test.
    :param alternative: less (first group has lesser values than the second), greater, two-sided
    :return: DF with columns: Gene, Comparison (as named in group_splits), Statistic (test statistic), p (p value), FDR
        (across whole results DF), Mean1, Mean2 (mean of each of two groups from group_splits), Difference
        (mean2-mean1), Separation (How well does the comparison separate the two groups based on means of border
        elements - min of two differences: mean2-mean of the last strain group in comparison group1 and  mean of first
        strain group in comparison group2 -mean1; last and first used as specified in group_splits).
    """
    results = []
    for gene in quantile_normalised.index:
        for comparison in group_splits:
            strains1 = GROUP_DF[GROUP_DF['X'].isin(comparison[0])]['Strain']
            strains2 = GROUP_DF[GROUP_DF['X'].isin(comparison[1])]['Strain']
            values1 = quantile_normalised.loc[gene, strains1].values
            values2 = quantile_normalised.loc[gene, strains2].values

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

            if test == 'u':
                result = mannwhitneyu(values1, values2, alternative=alternative)
                p = result[1]
                statistic = result[0]
            elif test == 't':
                result = ttest_ind(values1, values2)
                statistic = result[0]
                p = result[1]
                if alternative == 'less':
                    if statistic <= 0:
                        p = p / 2
                    else:
                        p = 1 - p / 2
                elif alternative == 'greater':
                    if statistic >= 0:
                        p = p / 2
                    else:
                        p = 1 - p / 2

            results.append({'Gene': gene, 'Comparison': comparison[2], 'Statistic': statistic, 'p': p,
                            'Mean1': m1, 'Mean2': m2, 'Difference': m2 - m1, 'Separation': separation})

    results = pd.DataFrame(results)
    # Adjust all pvals to FDR
    results['FDR'] = multipletests(pvals=results['p'], method='fdr_bh', is_sorted=False)[1]
    return results
