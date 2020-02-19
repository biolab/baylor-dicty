import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hc
import pandas as pd
from statistics import mean

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

GROUP_X = {'1Ag-': 1, '2LAg': 2, '3TA': 3, '4CD': 4, '6SFB': 5,  '5WT': 6, '7PD': 7}

GROUP_DF=[]
for strain,group in GROUPS.items():
    GROUP_DF.append({'Strain':strain, 'Group':group,'X':GROUP_X[group]})
GROUP_DF=pd.DataFrame(GROUP_DF)


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


def sigmoid_fit(x,y):
    x=np.array(x)
    y=np.array(y)
    x_diff=(x.max()-x.min())*0.1
    #y_diff_01=(y.max()-y.min())*0.1
    bounds=([y.min(),x.min()+x_diff,-np.inf,y.min()],[y.max(),x.max()-x_diff,np.inf,y.max()])
    p0 = [max(y), np.median(x), 1, min(y)]  # initial guess
    return curve_fit(sigmoid, x, y, p0,bounds=bounds)

def sigmoid(x, L, x0, k, b):
    y = L / (1 + np.exp(-k * (x - x0))) + b
    return (y)

def relative_cost(x,y,function,params):
    yp=[]
    for xi in x:
        yp.append(function(xi,*params))
    return mean([((ypi-yi)/yi)**2 for ypi,yi in zip(yp,y)])
