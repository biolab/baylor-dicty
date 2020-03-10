import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt


def variation_statistic(gene_data: pd.DataFrame) -> pd.Series:
    """
    Calculate std/mean for each gene and replace nan with 0
    :gene_data: Expression DF with genes in rows. Calculations are performed for each row across features.
    :return: Series with statistic for each row
    """
    statistic = gene_data.std(axis=1) / gene_data.mean(axis=1)
    # statistic = gene_data.std(axis=1)
    # TODO How to deal with 0 expressed genes? Are they informative?????
    return statistic.replace(np.nan, 0)


def plot_variation_distn(gene_vars: pd.DataFrame):
    """
    Plot distn of variation mean and medians across genes
    :param gene_vars: DF with genes in rows and genes' variations as values across columns.
    """
    plt.hist(gene_vars.median(axis=1), bins=100, alpha=0.4, label='median')
    plt.hist(gene_vars.mean(axis=1), bins=100, alpha=0.4, label='mean')
    plt.legend()


def remove_zero_median(gene_vars: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with 0 median.
    :param gene_vars: DF with genes in rows and genes' variations as values across columns.
    :return: DF without rows with 0 median
    """
    var_medians = gene_vars.median(axis=1)
    remove_genes = var_medians[var_medians == 0].index
    return gene_vars.drop(remove_genes)


def diff_vars_all(gene_vars: pd.DataFrame, gene_vars_all: pd.Series):
    """
    Divide gene variations in different conditions with variations across all conditions.
    :param gene_vars: DF with genes in rows and genes' variations as values across columns.
    :param gene_vars_all: Series with genes in rows and genes' variations across all conditions as values.
    :return: gene_vars divided by gene_vars_all. This is performed only for genes present in both gene_vars and
        gene_vars_all.
    """
    # This does not take into account that there is different N between all and image group (less measurements).
    # However, this difference is the same for all image groups/genes
    genes_both = list(set(gene_vars.index) & set(gene_vars_all.index))
    return (gene_vars.loc[genes_both, :].T / gene_vars_all.loc[genes_both]).T
