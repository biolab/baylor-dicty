import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hc
import pandas as pd
import random
from statistics import mean
from scipy.stats import mannwhitneyu, ttest_ind
from statsmodels.stats.multitest import multipletests

from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
import altair as alt
from sklearn import preprocessing as pp
import matplotlib.patches as mpatches

from orangecontrib.bioinformatics.utils.statistics import Hypergeometric
from Orange.clustering.louvain import jaccard
from scipy.optimize import curve_fit

GROUPS = {'amiB': 'agg-', 'mybB': 'agg-', 'acaA': 'agg-', 'gtaC': 'agg-',
          'gbfA': 'lag_dis', 'tgrC1': 'lag_dis', 'tgrB1': 'tag_dis', 'tgrB1C1': 'tag_dis',
          'tagB': 'tag', 'comH': 'tag',
          'ecmARm': 'cud', 'gtaI': 'cud', 'cudA': 'cud', 'dgcA': 'cud', 'gtaG': 'cud',
          'AX4': 'WT', 'MybBGFP': 'WT',
          'acaAPkaCoe': 'sFB', 'ac3PkaCoe': 'sFB',
          'pkaR': 'prec', 'PkaCoe': 'prec'}

GROUP_X = {'agg-': 1, 'lag_dis': 2, 'tag_dis': 3, 'tag': 4, 'cud': 5, 'sFB': 6, 'WT': 7, 'prec': 8}

GROUP_DF = []
for strain, group in GROUPS.items():
    GROUP_DF.append({'Strain': strain, 'Group': group, 'X': GROUP_X[group]})
GROUP_DF = pd.DataFrame(GROUP_DF)

# GROUP_DF.to_csv('/home/karin/Documents/timeTrajectories/data/regulons/selected_genes/group_df.tsv',sep='\t',
# index=False)

PHENOTYPES = ['no_agg', 'stream', 'lag', 'tag', 'tip', 'slug', 'mhat', 'cul',  'FB','yem']
PHENOTYPES_X = {'no_agg': 0, 'stream': 1, 'lag': 2, 'tag': 3, 'tip': 4, 'slug': 5, 'mhat': 6, 'cul': 7, 'yem': 9,
                'FB': 8}

COLOURS_GROUP = {'agg-': '#d40808', 'lag_dis': '#e68209', 'tag_dis': '#ffb13d', 'tag': '#d1b30a', 'cud': '#4eb314',
                 'WT': '#0fa3ab', 'sFB': '#525252', 'prec': '#7010b0'}
COLOURS_STAGE = {'NA': '#d9d9d9', 'no_agg': '#ed1c24', 'stream': '#985006',
                 'lag': '#f97402', 'tag': '#d9d800', 'tip': '#66cf00', 'slug': '#008629', 'mhat': '#00c58f',
                 'cul': '#0ff2ff', 'FB': '#00b2ff', 'yem': '#666666'}


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
            group1 = None
            group2 = None
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
                    low_groups = groups12_ordered[1][::order * -1][:-1][::order * -1]
                    group = groups12_ordered[1][::order * -1][-1]
                    # print( low_groups, high_groups,group)
                    if group in unsplit1 or group in unsplit2:
                        stop = True
                    else:
                        m_high = group_statistic(groups=high_groups, quantile_normalised=quantile_normalised, gene=gene)
                        m = group_statistic(groups=[group], quantile_normalised=quantile_normalised, gene=gene)
                        if comparison_selection == 'closest':
                            m_low = group_statistic(groups=low_groups, quantile_normalised=quantile_normalised,
                                                    gene=gene)
                            # print(m_high,m_low,m)
                            diff_high = abs(m - m_high)
                            diff_low = abs(m - m_low)
                            stop = diff_high > diff_low
                        elif comparison_selection == 'std':
                            std_high = group_statistic(groups=high_groups, quantile_normalised=quantile_normalised,
                                                       gene=gene, mode='std')
                            # print(m_high, std_high, m)
                            stop = m < (m_high - 2 * std_high)

                    if stop:
                        group1 = groups12[0]
                        group2 = groups12[1]
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
                    # print( low_groups,high_groups, m_high - m_low)
                    diffs.append((m_high - m_low, low_groups, high_groups))
                best_sep = max(diffs, key=lambda item: item[0])
                group1 = best_sep[1]
                group2 = best_sep[2]

            # This splits groups so that they match the most Gaussian mixture clustering into two groups
            elif comparison_selection == 'gaussian_mixture':
                std_m1 = group_statistic(groups=unsplit1, quantile_normalised=quantile_normalised, gene=gene,
                                         mode='std')
                std_m2 = group_statistic(groups=unsplit2, quantile_normalised=quantile_normalised, gene=gene,
                                         mode='std')
                clusters = GaussianMixture(n_components=2,
                                           # means_init=np.array([[unsplit_m1],[unsplit_m2]]),
                                           # precisions_init=np.array([[[std_m1**(-2)]],[[std_m2**(-2)]]])
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
                    # print((adjusted_rand_score(clusters,comparison_clusters),group_pair))
                    scores.append((adjusted_rand_score(clusters, comparison_clusters), group_pair))
                best_sep = max(scores, key=lambda item: item[0])
                group1 = best_sep[1][0]
                group2 = best_sep[1][1]

            compariosn_name = group1[-1]
            compariosn_name = GROUP_DF.query('X == ' + str(compariosn_name)).Group.unique()[0]
            group_splits = [(group1, group2, compariosn_name)]
            # print(group_splits)
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


def summary_classification(df: pd.DataFrame, statistic, split, macro_list: list = None, print_df=True):
    """
    Calculate mean and standard error (SE) of scores from cross validation.
    :param df: Data Frame with cross validation results in rows and quality metrics and metric descriptions
    in columns.
    :param statistic: Coloumn of df for which mean and SE are calculated.
    :param split: Column of df used for wsplitting cross validation results into categories within
    which mean and SE are calculate separately.
    :param macro_list: If not None calculate macro mean and SE using the rows whose split column value 
    is within macro_list. This calculates the mean and SE over the df subseted with macro_list in split column.
    :param print_df: If True prints the result, else returns df with results.
    """
    if print_df:
        print(statistic, 'mean and standard error for each group')
    groups = df[[statistic, split]].groupby(split)
    summary_df = []
    for group_name in groups.groups.keys():
        data = groups.get_group(group_name)
        summary_df.append({'Group': group_name, 'Mean': data.mean()[0], 'SE': data.sem()[0]})
        if print_df:
            print('%-12s%-6.2f%-3s%-3.2f' % (group_name, data.mean()[0], '+-', data.sem()[0]))
    if macro_list is not None:
        data = df[df[split].isin(macro_list)][statistic]
        summary_df.append({'Group': 'macro', 'Mean': data.mean(), 'SE': data.sem()})
        if print_df:
            print('%-12s%-6.2f%-3s%-3.2f' % ('macro', data.mean(), '+-', data.sem()))
    if not print_df:
        return pd.DataFrame(summary_df)


def summary_classification_print_sort(summary, statistic, averages, groups, sort: bool = 'score'):
    """
    :param sort: Sort classes by 'score' or by 'groups' order
    """
    print('Mean cross validation ' + statistic + ' averaged across all phenotypes and standard error')
    averages_summary = summary[summary.Group.isin(averages)].copy()
    averages_summary['Group'] = pd.Categorical(averages_summary['Group'], averages)
    for row in averages_summary.sort_values('Group').iterrows():
        row = row[1]
        print('%-12s%-6.2f%-3s%-3.2f' % (row['Group'], row['Mean'], '+-', row['SE']))
    print('Mean cross validation ' + statistic + ' of individual phenotypes and standard error')
    if sort == 'score':
        sorted_summary = summary[summary.Group.isin(groups)].sort_values('Mean', ascending=False)
    elif sort == 'groups':
        summary_groups = summary[summary.Group.isin(groups)].copy()
        summary_groups.Group = pd.Categorical(summary_groups.Group, categories=groups, ordered=True)
        sorted_summary = summary_groups.sort_values('Group')
    for row in sorted_summary.iterrows():
        row = row[1]
        print('%-12s%-6.2f%-3s%-3.2f' % (row['Group'], row['Mean'], '+-', row['SE']))


# From https://datavizpyr.com/stripplot-with-altair-in-python/
def scatter_catgory(df: pd.DataFrame, Y, categories=None, colour=None, shape=None, title: str = '', width=120):
    """
    Make scatter plot with categories on X axis and X jittering to reduce the overlap between 
    data points of the same category.
    :param df: Data Frame with points in rows and data for plotting in columns.
    :param categories: Column from df used for splitting the data on  X axis categories.
    :param Y: Column of df whose values are ploted on Y axis. 
    :param colour: Optional, column of df based on which the points are coloured.
    :param shape: Optional, column of df based on which the points are shaped.
    :param title: Optional, a title for the plot.
    """
    params_dict = {}
    if colour is not None:
        params_dict['color'] = alt.Color(colour)
    if shape is not None:
        params_dict['shape'] = alt.Shape(shape)
    if categories is not None:
        params_dict['column'] = alt.Column(str(categories) + ':O', sort=list(df[categories].unique()),
                                           header=alt.Header(
                                               labelAngle=0, titleOrient='bottom', labelOrient='bottom',
                                               labelAlign='center', labelPadding=10))
    return alt.Chart(df, width=width, title=title).mark_point(size=20).encode(
        x=alt.X('jitter:Q', title=None, axis=alt.Axis(values=[0], ticks=True, grid=False, labels=False),
                scale=alt.Scale(), ),
        y=alt.Y(Y, axis=alt.Axis(grid=False)),
        **params_dict
    ).transform_calculate(jitter='sqrt(-2*log(random()))*cos(2*PI*random())'
                          ).configure_facet(spacing=0
                                            # ).configure_view( stroke=None
                                            )


def FPR(y_true, y_pred, average):
    """
    False positive rate for true and predicted labels
    :param average: None,'micro','macro' (as in sklearn)
    """
    if average == 'micro':
        y_true = y_true.ravel()
        y_pred = y_pred.ravel()
    predicted_positive = y_pred == 1
    true_negative = y_true == 0
    false_positive_n = (predicted_positive & true_negative).sum(axis=0)
    true_negative_n = true_negative.sum(axis=0)
    fpr = false_positive_n / true_negative_n
    if average == 'macro':
        fpr = fpr.mean()
    return fpr


def accuracy(y_true, y_pred, average):
    """
    Accuracy for  true and predicted labels
    :param average: None,'micro','macro' (as in sklearn)
    """
    if average == 'micro':
        y_true = y_true.ravel()
        y_pred = y_pred.ravel()
    correct = (y_true == y_pred).sum(axis=0)
    acc = correct / y_true.shape[0]
    if average == 'macro':
        acc = acc.mean()
    return acc


def get_dimredplot_param(data, col, default, to_mode=False):
    """
    Based on data datatable (information for single/multiple points) find mode of the parameter or use default
    if there is no column for the parameter.
    :param data:
    :param col:
    :param default:
    :param to_mode:
    :return:
    """
    if isinstance(data, pd.DataFrame):
        if col in data.columns:
            result = data[col]
            if to_mode:
                result = result.mode()[0]
            return result
        else:
            return default
    elif isinstance(data, pd.Series):
        if col in data.index:
            return data[col]
        else:
            return default
    else:
        return default


# Jitter function
def rand_jitter(n, min, max, strength=0.005):
    """
    Number is jittered based on: n + randomN * (max-min), where -1 <= randomN <= 1
    :param n: Number to jitter
    :param min: Used to determine size of jittering
    :param max: Used to determine size of jittering
    :param strength: Larger strength leads to stronger jitter. Makes sense to be below 1 (adds random number scaled by
    max-min and strength.
    :return:
    """
    dev = (max - min) * strength
    return n + random.uniform(-1, 1) * dev


def dim_reduction_plot(plot_data: pd.DataFrame(), plot_by: str, fig_ax: tuple, order_column, colour_by_phenotype=False,
                       add_name=True, colours: dict = COLOURS_GROUP, colours_stage: dict = COLOURS_STAGE,
                       legend_groups='lower left', legend_phenotypes='upper right', fontsize=6,
                       plot_order: list = None, plot_points: bool = True, add_avg: bool = False, add_sem: bool = False,
                       sem_alpha: float = 0.1, alternative_lines: dict = None, sep_text: tuple = (30, 30),
                       phenotypes_list: list = PHENOTYPES, plot_lines: bool = True, jitter_all: bool = False,
                       point_alpha=0.5, point_size=5, jitter_strength: tuple = (0.005, 0.005)):
    """
    Plots PC1 vs time (or tSNE - not tested averages and SEM) of strains, phenotype groups and developmental stages.
    For plotting parameters that are not for individual points (e.g. line width, alpha)
    uses mode when plotting lines and legend.
    Adds all colours to legend.
    :param plot_data: Data of individual 'points'. Must have columns: 'x','y', 'Group' (for colouring),
        order_column (how to order points in line),
        and a column matching the split_by parameter (for line plotting and names).
        Can have 'size' (point size - default param Point_size),  'width' (line width),
        'alpha' (for plotting - default for points is point_alpha), 'linestyle', 'shape' (point shape),
        and phenotypes columns (matching phenotypes_list, valued 0 (absent) or 1 (present)).
    :param plot_by: Plot lines and text annotation based on this column.
    :param fig_ax: Tupple  (fig,ax) with plt elements used for plotting
    :param order_column: Order of plotting of groups from split_by, first is pliotted first.
    :param colour_by_phenotype: Whether to colours samples by phenotype. If false colour by 'Group' colour.
    :param add_name: Add text with name from plot_by groups.
    :param colours: Colours for 'Group'. Key: 'Group' value, value: colour.
    :param colours_stage: Colours for plotting stages, used if colour_by_phenotype=True. Dict with key: from
        phenotypes_list and value: colour.
    :param legend_groups: Position for Group legend, if None do not plot
    :param legend_phenotypes: Position for stages legend, if None do not plot
    :param  fontsize: Fontsize for annotation.
    :param plot_order: Plot lines and SEMs in this order. Matching groups from split_by.
    :param plot_points: Whether to plot points.
    :param add_avg: Average points with same x value (used for plotting lines and text positioning).
    :param add_sem: Plot SEM zones.
    :param sem_alpha: Alpha for SEM zone.
    :param phenotypes_list: List of phenotypes used to find stage columns in pplot_data
    :param alternative_lines: Plot different lines than based on data points from plot_data.
        Dict with keys being groups obtained by plot_by and values tuple of lists: ([xs],[ys]). Use this also for
        text annotation.
    :param plot_lines: Whether to plot lines. Lines are plotted between points (rows) in plot_data, ordered by
        order_column
    :param jitter_all: Jitter all points. Else jitter only when multiple stages are annotated to same point, not
        jittering the first stage.
    :param point_alpha:Default alpha for points used if alpha column absent
    :param point_size:Default size for points used if size column absent
    :param jitter_strength: Tuple (strength_x, strength_y) used to jitter points. Use floats << 1 - based on data range.
        Higher -> more jittering.
    :param sep_text: Smaller number increases the separation of text annotations. Tuple with (x,y), where x,y
        denote values for x and y axis
    """
    if plot_order is not None:
        plot_data = plot_data.loc[
            plot_data[plot_by].map(dict(zip(plot_order, range(len(plot_order))))).sort_values().index]
    else:
        plot_order = plot_data[plot_by].unique()

    fig, ax = fig_ax

    if plot_points:
        # Either add one point per measurment (coloured by group) or multiple jitter points coloured by phenotypes
        if not colour_by_phenotype:
            for row_name, point in plot_data.iterrows():
                ax.scatter(point['x'], point['y'], s=get_dimredplot_param(point, 'size', point_size),
                           c=colours[point['Group']], alpha=get_dimredplot_param(point, 'alpha', point_alpha, True),
                           marker=get_dimredplot_param(point, 'shape', 'o', True))
        # By phenotypes
        else:
            min_x = plot_data['x'].min()
            min_y = plot_data['y'].min()
            max_x = plot_data['x'].max()
            max_y = plot_data['x'].max()
            for point in plot_data.iterrows():
                point = point[1]
                phenotypes = point[phenotypes_list]

                x = point['x']
                y = point['y']
                if jitter_all:
                    x = rand_jitter(n=x, min=min_x, max=max_x, strength=jitter_strength[0])
                    y = rand_jitter(n=y, min=min_y, max=max_y, strength=jitter_strength[1])

                if phenotypes.sum() < 1:
                    ax.scatter(x, y, s=get_dimredplot_param(point, 'size', point_size),
                               c=colours_stage['NA'],
                               alpha=get_dimredplot_param(point, 'alpha', point_alpha, True),
                               marker=get_dimredplot_param(point, 'shape', 'o', True))
                elif phenotypes.sum() == 1:
                    phenotype = phenotypes[phenotypes > 0].index[0]
                    ax.scatter(x, y, s=get_dimredplot_param(point, 'size', point_size),
                               c=colours_stage[phenotype],
                               alpha=get_dimredplot_param(point, 'alpha', point_alpha, True),
                               marker=get_dimredplot_param(point, 'shape', 'o', True))
                else:
                    first = True
                    for phenotype in phenotypes_list:
                        if phenotypes[phenotype] == 1:
                            if not first:
                                x = rand_jitter(n=x, min=min_x, max=max_x, strength=jitter_strength[0])
                                y = rand_jitter(n=y, min=min_y, max=max_y, strength=jitter_strength[1])
                            ax.scatter(x, y, s=get_dimredplot_param(point, 'size', point_size),
                                       c=colours_stage[phenotype],
                                       alpha=get_dimredplot_param(point, 'alpha', point_alpha, True),
                                       marker=get_dimredplot_param(point, 'shape', 'o', True))
                            first = False

    # Group for lines/names
    grouped = plot_data.groupby(plot_by)

    # Add SEM zones
    if add_sem:
        for name in plot_order:
            data_rep = grouped.get_group(name).sort_values(order_column)
            group = data_rep['Group'].values[0]
            grouped_x = data_rep.groupby(['x'])
            x_line = grouped_x.mean().index
            y_line = grouped_x.mean()['y']
            sem = grouped_x.sem()['y']
            ax.fill_between(x_line, y_line - sem, y_line + sem, alpha=sem_alpha, color=colours[group])

    # Add line between replicates' measurments
    if plot_lines:
        for name in plot_order:
            data_rep = grouped.get_group(name).sort_values(order_column)
            group = data_rep['Group'].values[0]

            if alternative_lines is None:
                if not add_avg:
                    x_line = data_rep['x']
                    y_line = data_rep['y']
                else:
                    grouped_x = data_rep.groupby(['x'])
                    x_line = grouped_x.mean().index
                    y_line = grouped_x.mean()['y']
            else:
                x_line, y_line = alternative_lines[data_rep[plot_by].values[0]]

            ax.plot(x_line, y_line, color=colours[group],
                    alpha=get_dimredplot_param(data_rep, 'alpha', 0.5, True),
                    linewidth=get_dimredplot_param(data_rep, 'width', 0.5, True),
                    linestyle=get_dimredplot_param(data_rep, 'linestyle', 'solid', True))

    # Add replicate name
    if add_name:
        used_text_positions = pd.DataFrame(columns=['x', 'y'])
        x_span = plot_data['x'].max() - plot_data['x'].min()
        y_span = plot_data['y'].max() - plot_data['y'].min()
        for name in plot_order:
            data_rep = grouped.get_group(name).sort_values(order_column)
            group = data_rep['Group'].values[0]
            idx = -1
            if alternative_lines is None:
                if not add_avg:
                    x_values = data_rep['x'].values
                    y_values = data_rep['y'].values
                else:
                    grouped_x = data_rep.groupby(['x'])
                    x_values = grouped_x.mean().index.values
                    y_values = grouped_x.mean()['y'].values
            else:
                x_values, y_values = alternative_lines[data_rep[plot_by].values[0]]
            x = float(x_values[idx]) + x_span / 500
            y = float(y_values[idx]) + y_span / 500
            while ((abs(used_text_positions['x'] - x) < (x_span / sep_text[0])).values &
                   (abs(used_text_positions['y'] - y) < (y_span / sep_text[1])).values).any():
                idx -= 1
                x = float(x_values[idx]) + x_span / 500
                y = float(y_values[idx]) + y_span / 500
            used_text_positions = used_text_positions.append({'x': x, 'y': y}, ignore_index=True)
            ax.text(x, y, data_rep[plot_by].values[0], fontsize=fontsize, color=colours[group])

    # Legends for groups and phenotypes
    alpha_legend = get_dimredplot_param(plot_data, 'alpha', 0.5)
    if isinstance(alpha_legend, pd.Series):
        alpha_legend = alpha_legend.median()
    if legend_groups is not None:
        patchList = []
        for name, colour in colours.items():
            # if name in plot_data['Group'].values:
            data_key = mpatches.Patch(color=colour, label=name, alpha=alpha_legend)
            patchList.append(data_key)
        title = 'Phenotype'
        if colour_by_phenotype:
            title = title + ' (line)'
        legend_groups = ax.legend(handles=patchList, title=title, loc=legend_groups)

    if colour_by_phenotype and legend_phenotypes is not None:
        patchList = []
        for name, colour in colours_stage.items():
            # if name in plot_data.columns:
            data_key = mpatches.Patch(color=colour, label=name, alpha=alpha_legend)
            patchList.append(data_key)
        legend_stages = ax.legend(handles=patchList, title="Stage (point)", loc=legend_phenotypes)

    if legend_groups is not None:
        ax.add_artist(legend_groups)


class CustomScaler:

    def __init__(self, reference):
        """
        :param reference:
        """
        if isinstance(reference, pd.DataFrame):
            reference = reference.values
        self.reference = reference
        self.scalers = {}

    def transform(self, data, log, scale: str):
        """

        :param data:
        :param log: log2(data+1)
        :param scale: 'minmax','m0s1','divide_mean'
        :return:
        """
        if not (log, scale) in self.scalers.keys():
            scaler = None
            ref = self.reference.copy()
            if log:
                ref = np.log2(ref + 1)
            if scale == 'minmax':
                scaler = pp.MinMaxScaler()
                scaler.fit(ref)
            elif scale == 'm0s1':
                scaler = pp.StandardScaler()
                scaler.fit(ref)
            elif scale == 'divide_mean':
                scaler = ref.mean(axis=0)
            self.scalers[(log, scale)] = scaler
            # print(id(scaler))

        scaler = self.scalers[(log, scale)]
        # print(id(scaler))
        scaled = None
        if isinstance(data, pd.DataFrame):
            data = data.values

        if log:
            data = np.log2(data + 1)
        if scale in ['minmax', 'm0s1']:
            scaled = scaler.transform(data)
        elif scale == 'divide_mean':
            scaled = data / scaler
        return scaled
