import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt


def variation_statistic(gene_data: pd.DataFrame):
    statistic = gene_data.std(axis=1) / gene_data.mean(axis=1)
    # statistic = gene_data.std(axis=1)
    # TODO How to deal with 0 expressed genes? Are they informative?????
    return statistic.replace(np.nan, 0)


def plot_variation_distn(gene_vars: pd.DataFrame):
    plt.hist(gene_vars.median(axis=1), bins=100, alpha=0.4, label='median')
    plt.hist(gene_vars.mean(axis=1), bins=100, alpha=0.4, label='mean')
    plt.legend()


def remove_zero_median(gene_vars: pd.DataFrame):
    var_medians = gene_vars.median(axis=1)
    remove_genes = var_medians[var_medians == 0].index
    return gene_vars.drop(remove_genes)


def diff_vars_all(gene_vars: pd.DataFrame, gene_vars_all: pd.Series):
    # This does not take into account that there is different N between all and image group (less measurements).
    # However, this difference is the same for all image groups/genes
    genes_both = list(set(gene_vars.index) & set(gene_vars_all.index))
    return (gene_vars.loc[genes_both, :].T / gene_vars_all.loc[genes_both]).T
