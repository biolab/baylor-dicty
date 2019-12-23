from abc import ABC, abstractmethod
from collections import OrderedDict
import datetime

import pandas as pd
import numpy as np
from community import best_partition as louvain
import sklearn.preprocessing as pp

import Orange.clustering.louvain as orange_louvain_graph
from Orange.data import Table, Domain, DiscreteVariable, StringVariable

# Wait for "*** FINISHED ***" to be printed in the command line

# ********** To be set
# Clustering parameters
neighbours_louvain = 30
resolution = 1

dataPath = '/home/karin/Documents/timeTrajectories/data/RPKUM/combined/'

# *******************************************************************************
SCALING = 'mean0sd1'
LOG = True
# Remove in actual script
in_data = None


def merge_genes_conditions(genes: pd.DataFrame, conditions: pd.DataFrame, matching) -> pd.DataFrame:
    """
    Merge dataframes with genes and conditions
    :param genes: Expression data, genes in rows, measurements in columns, dimensions G*M
    :param conditions: Description (columns) of each measurements (rows), dimensions M*D
    :param matching: Which column in conditions matches column names in genes
    :return: Data frame with merged genes and conditions
    """
    conditions = conditions.copy()
    conditions.index = conditions[matching]
    return pd.concat([genes.T, conditions], axis=1)


def split_data(data: pd.DataFrame, split_by: str) -> dict:
    """
    Split data by column
    :param data: Data to be split by values of a column
    :param split_by: Column name for splitting
    :return: Key: split_by column value, value: data of this split_by column value
    """
    data_splitted = {}
    groupped = data.groupby(by=split_by)
    for group in groupped.groups.keys():
        data_splitted[group] = (groupped.get_group(group))
    return data_splitted


def get_index_query(genes: pd.DataFrame, scale: str = SCALING, log: bool = LOG) -> tuple:
    """
    Get gene data scaled to be index or query for neighbour search.
    :param genes: Gene data for index and query.
    :param inverse: Inverse query to compute neighbours with opposite profile. True if use inverse.
    :param scale: Scale expression by gene with 'minmax' (min=0, max=1) or 'mean0std1' (mean=0, std=1).
    :param log: Should expression data be log2 transformed.
    :return: genes for index (1st element) and genes for query (2nd element)
    """
    if log:
        genes = np.log2(genes + 1)
    if scale == 'minmax':
        genes = minmax_scale(genes)
    elif scale == 'mean0std1':
        genes = meanstd_scale(genes)
    genes_index = genes
    genes_query = genes
    return genes_index, genes_query


def minmax_scale(genes: pd.DataFrame) -> np.ndarray:
    """
    Scale each row from 0 to 1.
    :param genes: data
    :return: scaled data
    """
    return pp.minmax_scale(genes, axis=1)


def meanstd_scale(genes) -> np.ndarray:
    """
    Scale each row to mean 0 and std 1.
    :param genes: data
    :return: scaled data
    """
    return pp.scale(genes, axis=1)


def get_orange_scaled(genes: pd.DataFrame, scale: str = SCALING, log: bool = LOG) -> pd.DataFrame:
    """
    Get preprocessed expression data for orange - scale and log transform.
    :param genes: Expression data, genes in rows, measurments in columns
    :param scale: How to scale data  for neighbour calculation and returning
    :param log: Log transform data   for neighbour calculation and returning
    :return: Preprocessed genes with close neighbours
    """
    gene_names = list(genes.index)
    index, query = get_index_query(genes=genes, scale=scale, log=log)
    return pd.DataFrame(index, index=gene_names, columns=genes.columns)


class Clustering(ABC):
    """
    Abstract class for clustering.
    """

    def __init__(self, distance_matrix: np.array, gene_names: list, data: np.ndarray):
        self._gene_names_ordered = gene_names
        self._n_genes = len(gene_names)
        self._distance_matrix = distance_matrix
        self._data = data

    @abstractmethod
    def get_clusters(self, **splitting) -> np.ndarray:
        """
        Cluster memberships
        :param splitting: how to create clusters
        :return: List of cluster memberships over genes (cluster numbers in same order as gene names)
        """
        pass

    def get_clusters_by_genes(self, splitting: dict = None, filter_genes: iter = None, clusters=None) -> dict:
        """
        Get clusters with corresponding members.
        :param splitting: how to create clusters, needed if clusters is not give
        :param filter_genes: Report only genes (leafs) which are contained in filter_genes. If None use all genes in
            creation of membership dictionary.
        :param clusters: Predefined clusters, as returned by get_clusters
        :return: Dict keys: cluster, values: list of genes/members
        """
        if clusters is None:
            clusters = self.get_clusters(**splitting)
        gene_dict = dict()
        for gene, cluster in zip(self._gene_names_ordered, clusters):
            if (filter_genes is None) or (gene in filter_genes):
                gene_dict[gene] = cluster
        return gene_dict


class LouvainClustering(Clustering):

    def __init__(self, gene_names: list, data: np.ndarray, neighbours: int = 30):
        """
        Prepare graph for Louvain clustering. Graph construction can be perfomed in different ways, as described below.
        If orange_graph=False use 1-distances as similarity for graph weights, else use Jaccard index.
        :param distance_matrix: Cosine Distances between samples, square matrix
        :param gene_names: List of names specifying what each row/column in distance matrix represents
        :param data: Preprocessed data
        :param orange_graph: Should graph be constructed with orange graph constructor
        :param trimm: When orange graph is not used, if closest is None the graph will be constructed based on all
            edges that would have 1-distance (weight) at least as big as trimm
        :param closest: If orange_graph=False use that many closest neighbours from similarity transformed distance
            matrix to construct the graph (weighs are similarities), if orange_graph=True use that many neighbours to
            compute Jaccard index between nodes, which is used as a weight
        """
        super().__init__(distance_matrix=None, gene_names=gene_names, data=data)
        self._graph = orange_louvain_graph.matrix_to_knn_graph(data=data, k_neighbors=neighbours, metric='cosine')

    def get_clusters(self, **splitting) -> np.ndarray:
        """
        Cluster memberships
        :param splitting: Parameters passed to community.best_partition clustering
        :return: List of cluster memberships over genes
        """
        clusters = louvain(graph=self._graph, **splitting)
        return np.array(list(OrderedDict(sorted(clusters.items())).values()))


# *********** Script
# print('*** STARTED at ',datetime.datetime.now().strftime('%H:%M:%S'),'***')
metas = in_data.domain.metas
for meta, idx in zip(metas, range(len(metas))):
    if meta.name == 'Gene':
        gene_idx = idx
gene_names = in_data.metas[:, gene_idx]
genes = pd.read_csv(dataPath + 'mergedGenes_RPKUM.tsv', sep='\t', index_col=0)
genes_selected = genes.loc[gene_names, :]

conditions = pd.read_csv(dataPath + 'conditions_mergedGenes.tsv', sep='\t', index_col=None)

# Split data by strain
merged = merge_genes_conditions(genes=genes_selected, conditions=conditions[['Measurment', 'Strain']],
                                matching='Measurment')
splitted = split_data(data=merged, split_by='Strain')
splitted2 = dict()
for strain, data in splitted.items():
    data = data.drop(['Strain', 'Measurment'], axis=1).T
    data = get_orange_scaled(genes=data, scale='mean0std1', log=True)
    splitted2[strain] = data
splitted = splitted2
splitted['all'] = get_orange_scaled(genes=genes_selected, scale='mean0std1', log=True)

# Perform clustering
clustered = OrderedDict()
for strain, data in splitted.items():
    # resolution = 0.7 if strain == 'all' else 2
    clustered[strain + '_clusters'] = LouvainClustering(data=data.to_numpy(), gene_names=data.index,
                                                        neighbours=neighbours_louvain).get_clusters_by_genes(
        splitting={'resolution': resolution})

# Make matrix of clusters by gene for each strain/all
clusters_df = pd.DataFrame(list(clustered.values()), index=clustered.keys()).T.astype(int)
domain_columns = []
for col in clusters_df.columns:
    values = list(set(clusters_df[col]))
    values.sort()
    values = [str(value) for value in values]
    domain_columns.append(DiscreteVariable(name=col, values=values, ordered=True))

meta_columns = [StringVariable(name='Gene')]
out_data = Table.from_numpy(domain=Domain(domain_columns, metas=meta_columns), X=clusters_df.to_numpy(),
                            metas=pd.DataFrame(clusters_df.index).to_numpy())
print('*** FINISHED at', datetime.datetime.now().strftime('%H:%M:%S'), '***')
