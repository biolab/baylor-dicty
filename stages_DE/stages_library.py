import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hc
import pandas as pd
from statistics import mean
from scipy.stats import mannwhitneyu, ttest_ind
from statsmodels.stats.multitest import multipletests

from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

from orangecontrib.bioinformatics.utils.statistics import Hypergeometric
from Orange.clustering.louvain import jaccard
from scipy.optimize import curve_fit

GROUPS = {'amiB': 'agg-', 'mybB': 'agg-', 'acaA': 'agg-', 'gtaC': 'agg-',
          'gbfA': 'lag_dis', 'tgrC1': 'lag_dis', 'tgrB1': 'tag_dis', 'tgrB1C1': 'tag_dis',
          'tagB': 'tag', 'comH': 'tag',
          'ecmARm': 'cud', 'gtaI': 'cud', 'cudA': 'cud', 'dgcA': 'cud', 'gtaG': 'cud',
          'AX4': 'WT', 'MybBGFP': 'WT',
          'acaAPkaCoe': 'sFB', 'ac3PkaCoe': 'sFB',
          'pkaR': 'Prec', 'PkaCoe': 'Prec'}

GROUP_X = {'agg-': 1, 'lag_dis': 2, 'tag_dis': 3, 'tag': 4, 'cud': 5, 'sFB': 6, 'WT': 7, 'Prec': 8}

GROUP_DF = []
for strain, group in GROUPS.items():
    GROUP_DF.append({'Strain': strain, 'Group': group, 'X': GROUP_X[group]})
GROUP_DF = pd.DataFrame(GROUP_DF)

# GROUP_DF.to_csv('/home/karin/Documents/timeTrajectories/data/regulons/selected_genes/group_df.tsv',sep='\t',
# index=False)

PHENOTYPES = ['no_agg', 'stream', 'lag', 'tag', 'tip', 'slug', 'mhat', 'cul', 'FB', 'disappear', 'tag_spore']


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


def quantile_normalise(similarity_means, return_ranks: bool = False):
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
    normalised = pd.DataFrame(quantile_normalised, index=rank_df.index, columns=rank_df.columns)
    if not return_ranks:
        return normalised
    else:
        return normalised, rank_mean


def compare_gene_scores(quantile_normalised: pd.DataFrame, test: str, alternative: str, group_splits: list = None,
                        select_single_comparsion: list = None, comparison_selection: str = 'gaussian_mixture'):
    """
    Compare gene scores across strain groups. For each gene there is a score (e.g. avg similarity to neighbours) in each
    strain that belongs to a strain group. Compare scores between groups to find genes that have lower/higher score in
    some set of groups.
    :param quantile_normalised: DF with strains in columns and genes in rows. For each gene in each strain there is a
        score. Strain groups are used as specified in GROUP_DF.
    :param group_splits: List of which comparisons between groups to perform. Each comparison is specified as a tupple.
        First tow elements are list of strain groups for comparison (e.g. strain groups in element1 vs strain groups in
        element 2). The strain groups must be given as in X column of GROUP_DF. The third element is name of the
        comparison that will be reported. If this is None the select_single_comparsion must be specified.
    :param select_single_comparsion: Order of strain groups to select single separation threshold to split groups in
        two parts (low and high neighbour similarity). This is a list where first element is a list of all strain groups
        as in X column of GROPUP_DF, groups should be ordered (e.g. by development). Second and third elements are lists
         of groups that should not be split up (e.g. references, can be just first and last group from the list of
         all groups. The splitting is performed between any two groups in the specified order so that the reference
         groups are not split.
    :param test: 'u' for mannwhitneyu or 't' for t-test.
    :param alternative: less (first group has lesser values than the second), greater, two-sided
    :param comparison_selection: Used if comparisons are selected based onv select_single_comparsion. How to separate
        strain groups into two groups.
    :return: DF with columns: Gene, Comparison (as named in group_splits), Statistic (test statistic), p (p value), FDR
        (across whole results DF), Mean1, Mean2 (mean of each of two groups from group_splits), Difference
        (mean2-mean1), Separation (How well does the comparison separate the two groups based on means of border
        elements - min of two differences: mean2-mean of the last strain group in comparison group1 and  mean of first
        strain group in comparison group2 -mean1; last and first used as specified in group_splits).
    """
    results = []
    if select_single_comparsion is None and group_splits is None:
        raise ValueError('select_single_comparsion or group_splits must be specified')
    if select_single_comparsion is not None:
        unsplit1 = select_single_comparsion[1]
        unsplit2 = select_single_comparsion[2]
        select_single_comparsion = select_single_comparsion[0]

        groups = []
        g1 = []
        for group in select_single_comparsion:
            g1.append(group)
            g2 = [x for x in select_single_comparsion if x not in g1]
            # print(g1,g2)
            if set(unsplit1).issubset(g1) and set(unsplit2).issubset(g2):
                # print(high,low)
                groups.append([g1.copy(), g2.copy()])
                # print(groups)

    for gene in quantile_normalised.index:
        # print(gene)
        if select_single_comparsion is not None:
            group1=None
            group2=None
            unsplit_m1 = group_statistic(groups=unsplit1, quantile_normalised=quantile_normalised, gene=gene)
            unsplit_m2 = group_statistic(groups=unsplit2, quantile_normalised=quantile_normalised, gene=gene)
            # TODO comments, documentation, test
            if unsplit_m1 > unsplit_m2:
                order = 1
                # high_groups = unsplit1.copy()
            else:
                order = -1
                # high_groups = unsplit2.copy()
            # This was used to select the best group split when first groups starts to deviate too much from the higher
            # (similarity) set of groups - either based on the difference to the mean of low/high or too many high's
            # standard deviations away from the high
            if comparison_selection in ['closest', 'std']:
                stop = False
                for groups12 in groups[::order]:
                    groups12_ordered = groups12[::order]
                    high_groups = groups12_ordered[0]
                    low_groups = groups12_ordered[1][::order*-1][:-1][::order*-1]
                    group = groups12_ordered[1][::order*-1][-1]
                    #print( low_groups, high_groups,group)
                    if group in unsplit1 or group in unsplit2:
                        stop=True
                    else:
                        m_high = group_statistic(groups=high_groups, quantile_normalised=quantile_normalised, gene=gene)
                        m = group_statistic(groups=[group], quantile_normalised=quantile_normalised, gene=gene)
                        if comparison_selection == 'closest':
                            m_low = group_statistic(groups=low_groups, quantile_normalised=quantile_normalised, gene=gene)
                            #print(m_high,m_low,m)
                            diff_high = abs(m - m_high)
                            diff_low = abs(m - m_low)
                            stop = diff_high > diff_low
                        elif comparison_selection == 'std':
                            std_high = group_statistic(groups=high_groups, quantile_normalised=quantile_normalised,
                                                       gene=gene,mode='std')
                            #print(m_high, std_high, m)
                            stop = m < (m_high - 2 * std_high)

                    if stop:
                        group1=groups12[0]
                        group2=groups12[1]
                        break
            # This was used to find best split based on separation that results in max differences between the
            # means of two groups
            elif comparison_selection == 'max_diff':
                diffs = list()
                for group_pair in groups:
                    high_groups = group_pair[::order][0]
                    low_groups = group_pair[::order][1]

                    m_high = group_statistic(groups=high_groups, quantile_normalised=quantile_normalised, gene=gene)
                    m_low = group_statistic(groups=low_groups, quantile_normalised=quantile_normalised, gene=gene)
                    #print( low_groups,high_groups, m_high - m_low)
                    diffs.append((m_high - m_low, low_groups, high_groups))
                best_sep = max(diffs, key=lambda item: item[0])
                group1 = best_sep[1]
                group2 = best_sep[2]

            # This splits groups so that they match the most Gaussian mixture clustering into two groups
            elif comparison_selection =='gaussian_mixture':
                std_m1 = group_statistic(groups=unsplit1, quantile_normalised=quantile_normalised, gene=gene,mode='std')
                std_m2 = group_statistic(groups=unsplit2, quantile_normalised=quantile_normalised, gene=gene,mode='std')
                clusters = GaussianMixture(n_components=2,
                                           #means_init=np.array([[unsplit_m1],[unsplit_m2]]),
                                           #precisions_init=np.array([[[std_m1**(-2)]],[[std_m2**(-2)]]])
                                           ).fit_predict(
                    quantile_normalised.loc[gene, :].values.reshape(-1, 1))
                scores = []
                for group_pair in groups:
                    strains1 = GROUP_DF[GROUP_DF['X'].isin(group_pair[0])]['Strain']
                    comparison_clusters = []
                    for strain in quantile_normalised.columns:
                        if strain in strains1.values:
                            comparison_clusters.append(0)
                        else:
                            comparison_clusters.append(1)
                    #print((adjusted_rand_score(clusters,comparison_clusters),group_pair))
                    scores.append((adjusted_rand_score(clusters, comparison_clusters), group_pair))
                best_sep = max(scores, key=lambda item: item[0])
                group1 = best_sep[1][0]
                group2 = best_sep[1][1]

            compariosn_name = group1[-1]
            compariosn_name=GROUP_DF.query('X == '+str(compariosn_name)).Group.unique()[0]
            group_splits = [(group1, group2, compariosn_name)]
            #print(group_splits)
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


def group_statistic(groups, quantile_normalised, gene, mode: str = 'mean'):
    """
    Calculates statistic (mean/std) of group of strain groups data (quantile normalised) for a single gene.
    :param groups: List of strain groups as in X column of GROUP_DF
    :param quantile_normalised: DF with strains in columns and genes in rows. For each gene in each strain there is a
        score (e.g. average similarity to neighbours).
    :param gene: Gene ID for which to calculate the statistic
    :param mode: mean or std
    """
    strains = GROUP_DF[GROUP_DF['X'].isin(groups)]['Strain']
    values = quantile_normalised.loc[gene, strains].values
    if mode == 'mean':
        return values.mean()
    elif mode == 'std':
        return values.std()