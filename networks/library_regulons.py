from abc import ABC, abstractmethod

import pandas as pd
import sklearn.preprocessing as pp
from pynndescent import NNDescent
import numpy as np
from statistics import mean, median
import networkx as nx
import warnings
import scipy.cluster.hierarchy as hc
from sklearn.metrics import silhouette_score
from collections import Counter
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

from orangecontrib.bioinformatics.ncbi.gene import GeneMatcher
from orangecontrib.bioinformatics.geneset.__init__ import (list_all, load_gene_sets)

from correlation_enrichment.library import GeneExpression, SimilarityCalculator

SCALING = 'minmax'
LOG = True


class NeighbourCalculator:
    """
    Obtains best neighbours of genes based on their expression profile.
    This can be done for all conditions at once or by condition groups, later on merging the results into
    single neighbour data.
    """

    MINMAX = 'minmax'
    MEANSTD = 'mean0std1'
    SCALES = [MINMAX, MEANSTD]

    def __init__(self, genes: pd.DataFrame, remove_zero: bool = True, conditions: pd.DataFrame = None,
                 conditions_names_column=None):
        """
        :param genes: Data frame of genes in rows and conditions in columns. Index is treated as gene names
        :param remove_zero: Remove genes that have all expression values 0.
            If batches are latter specified there may be all 0 rows within individual batches.
        :param conditions: data frame with conditions for genes subseting, measurements (M) in rows;
            conditions table dimensions are M*D (D are description types).
            Rows should have same order  as genes table  columns (specified upon initialisation, dimension G(genes)*M) -
            column names of gene table and specified column in conditions table should match.
        :param conditions_names_column: conditions table column that matches genes index - tests that have same order
        """
        GeneExpression.check_numeric(genes)
        if remove_zero:
            genes = genes[(genes != 0).any(axis=1)]
        self._genes = genes
        if conditions is not None:
            if list(conditions.loc[:, conditions_names_column]) != list(self._genes.columns):
                raise ValueError('Conditions table row order must match genes table column order.')
        self.conditions = conditions

    def neighbours(self, n_neighbours: int, inverse: bool, scale: str = SCALING, log: bool = LOG,
                   batches: list = None) -> dict:
        """
        Calculates neighbours of genes on whole gene data or its subset by column.
        :param n_neighbours: Number of neighbours to obtain for each gene
        :param inverse: Calculate most similar neighbours (False) or neighbours with inverse profile (True)
        :param scale: Scale expression by gene with 'minmax' (min=0, max=1) or 'mean0std1' (mean=0, std=1)
        :param log: Should expression data be log2 transformed
        :param batches: Should comparisons be made for each batch separately.
            Batches should be a list of batch group names for each column (eg. length of batches is n columns of genes).
        :return: Dict with gene names as tupple keys (smaller by alphabet is first tuple value) and
            values representing cosine similarity. If batches are used such dicts are returned for each batch
            in form of dict with batch names as keys and above mentioned dicts as values.
        """
        if scale not in self.SCALES:
            raise ValueError('Scale must be:', self.SCALES)
        genes = self._genes
        if batches is None:
            return self.calculate_neighbours(genes=genes, n_neighbours=n_neighbours, inverse=inverse, scale=scale,
                                             log=log)
        else:
            batch_groups = set(batches)
            batches = np.array(batches)
            results = dict()
            for batch in batch_groups:
                genes_sub = genes.T[batches == batch].T
                result = self.calculate_neighbours(genes=genes_sub, n_neighbours=n_neighbours, inverse=inverse,
                                                   scale=scale, log=log)
                results[batch] = result
            return results

    def calculate_neighbours(self, genes, n_neighbours: int, inverse: bool, scale: str, log: bool) -> dict:
        """
        Calculate neighbours of genes.
        :param genes: Data frame as in init
        :param n_neighbours: Number of neighbours to obtain for each gene
        :param inverse: Calculate most similar neighbours (False) or neighbours with inverse profile (True)
        :param scale: Scale expression by gene with 'minmax' (min=0, max=1) or 'mean0std1' (mean=0, std=1)
        :param log: Should expression data be log2 transformed
        :return: Dict with gene names as tuple keys (smaller by alphabet is first tuple value) and
            values representing cosine similarity
        """
        genes_index, genes_query = self.get_index_query(genes=genes, inverse=inverse, scale=scale, log=log)
        # Can set speed-quality trade-off, default is ok
        index = NNDescent(genes_index, metric='cosine', n_jobs=4)
        neighbours, distances = index.query(genes_query.tolist(), k=n_neighbours)
        return self.parse_neighbours(neighbours=neighbours, distances=distances)

    @classmethod
    def get_index_query(cls, genes: pd.DataFrame, inverse: bool, scale: str, log: bool) -> tuple:
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
        if inverse:
            genes_query = genes * -1
            genes_index = genes
            if scale == cls.MINMAX:
                genes_query = cls.minmax_scale(genes_query)
                genes_index = cls.minmax_scale(genes_index)
            elif scale == cls.MEANSTD:
                genes_query = cls.meanstd_scale(genes_query)
                genes_index = cls.meanstd_scale(genes_index)
        else:
            if scale == cls.MINMAX:
                genes = cls.minmax_scale(genes)
            elif scale == cls.MEANSTD:
                genes = cls.meanstd_scale(genes)
            genes_index = genes
            genes_query = genes
        return genes_index, genes_query

    @staticmethod
    def minmax_scale(genes: pd.DataFrame) -> np.ndarray:
        """
        Scale each row from 0 to 1.
        :param genes: data
        :return: scaled data
        """
        return pp.minmax_scale(genes, axis=1)

    @staticmethod
    def meanstd_scale(genes) -> np.ndarray:
        """
        Scale each row to mean 0 and std 1.
        :param genes: data
        :return: scaled data
        """
        return pp.scale(genes, axis=1)

    def parse_neighbours(self, neighbours: np.ndarray, distances: np.ndarray) -> dict:
        """
        Transform lists of neighbours and distances into dictionary with neighbours as keys and values as similarities.
        If pair of neighbours is given more than once it is overwritten the second time it is added to dictionary.
        For cosine similarity the above should be always the same.
        The neighbours (in form of index) are named based on gene data (index) stored in NeighbourCalculator instance.
        :param neighbours: Array of shape: genes*neighbours, where for each gene there are specified neighbours
        :param distances: Array of shape: genes*neighbours, where for each gene there are specified distances
            for each neighbour, as they are given above
        :return: Dict with gene names as tuple keys (smaller by alphabet is first tuple value) and
            values representing similarity: 1-distance
        """
        parsed = dict()
        for gene in range(distances.shape[0]):
            for neighbour in range(distances.shape[1]):
                distance = distances[gene, neighbour]
                # Because of rounding the similarity may be slightly above one and distance slightly below 0
                if distance < 0 or distance > 1:
                    if round(distance, 4) != 0 or distance > 1:
                        warnings.warn(
                            'Odd cosine distance at ' + str(gene) + ' ' + str(neighbour) + ' :' + str(distance),
                            Warning)
                    distance = 0
                similarity = 1 - distance
                gene2 = neighbours[gene, neighbour]
                gene_name1 = self._genes.index[gene]
                gene_name2 = self._genes.index[gene2]
                if gene_name1 != gene_name2:
                    if gene_name2 > gene_name1:
                        add_name1 = gene_name1
                        add_name2 = gene_name2
                    else:
                        add_name1 = gene_name2
                        add_name2 = gene_name1
                    if (add_name1, add_name2) in parsed.keys():
                        # Can do average directly as there will not be more than 2 pairs with same elements
                        # (eg. both possible positions: a,b and b,a for index,query)
                        # Similarities may be different when inverse is used with minmax
                        parsed[(add_name1, add_name2)] = (parsed[(add_name1, add_name2)] + similarity) / 2
                    else:
                        parsed[(add_name1, add_name2)] = similarity
        return parsed

    @staticmethod
    def merge_results(results: list, similarity_threshold: float = 0, min_present: int = 1) -> dict:
        """
        Merges results of different batches form calculate_neighbours and retains only associations present in
            min_present batches (dictionaries) at given threshold. Merges by mean of retained similarities.
        :param results: List of dictionaries with results from neighbours method: key is tuple neighbour pair,
            value is similarity.
        :param similarity_threshold: Ignore similarity if it is below threshold
        :param min_present: Filter out if not present in at least that many results with specified threshold
        :return:  Dict with gene names as tuple keys (smaller by alphabet is first tuple value) and
            values representing average similarity of retained pairs
        """
        if len(results) < min_present:
            raise ValueError('Threshold min_present can be at most length of results list.')
        pairs = set()
        merged = dict()
        for result in results:
            pairs |= set(result.keys())
        for pair in pairs:
            pair_values = []
            for result in results:
                if pair in result.keys():
                    similarity = result[pair]
                    if similarity >= similarity_threshold:
                        pair_values.append(similarity)
            if len(pair_values) >= min_present:
                merged[pair] = mean(pair_values)
        return merged

    @staticmethod
    def filter_similarities(results: dict, similarity_threshold: float) -> dict:
        """
        Filter out pairs that have similarity below threshold.
        :param results: Dict with values as similarities.
        :param similarity_threshold: Remove all keys with value below threshold.
        :return: Filtered dictionary, retaining only items with values above or equal to threshold.
        """
        return dict(filter(lambda elem: elem[1] >= similarity_threshold, results.items()))

    def compare_conditions(self, neighbours_n: int, inverse: bool,
                           scale: str, use_log: bool, thresholds: list, filter_column, filter_column_values_sub: list,
                           filter_column_values_test: list, batch_column=None):
        """
        Evaluates pattern similarity calculation preprocessing and parameters based on difference between subset and test set.
        Computes MSE from differences between similarities of subset gene pairs and corresponding test gene pairs.
        :param neighbours_n: N of calculated neighbours for each gene
        :param inverse: find neighbours with opposite profile
        :param scale: 'minmax' (from 0 to 1) or 'mean0std1' (to mean 0 and std 1)
        :param use_log: Log transform expression values before scaling
        :param thresholds: filter out any result with similarity below threshold, do for each threshold
        :param filter_column: On which column of conditions should genes be subset for separation in subset and test set
        :param filter_column_values_sub: Values of filter_column to use for subset genes
        :param filter_column_values_test:Values of filter_column to use for test genes
        :param batch_column: Should batches be used based on some column of conditions
        :return: Dictionary with parameters and results, description as key, result/parameter setting as value
        """
        # Prepare data
        if not batch_column:
            batches = None
        else:
            batches = list(
                (self.conditions[self.conditions[filter_column].isin(filter_column_values_sub)].loc[:, batch_column]))
        genes_sub = self._genes.T[list(self.conditions[filter_column].isin(filter_column_values_sub))].T
        genes_test = self._genes.T[list(self.conditions[filter_column].isin(filter_column_values_test))].T

        neighbour_calculator = NeighbourCalculator(genes_sub)
        test_index, test_query = NeighbourCalculator.get_index_query(genes_test, inverse=inverse, scale=scale,
                                                                     log=use_log)
        gene_names = list(genes_test.index)

        # Is similarity matrix expected to be simetric or not
        both_directions = False
        if inverse and (scale == 'minmax'):
            both_directions = True

        # Calculate neighbours
        result = neighbour_calculator.neighbours(neighbours_n, inverse=inverse, scale=scale, log=use_log,
                                                 batches=batches)

        # Filter neighbours on similarity
        data_summary = []
        for threshold in thresholds:
            if batches is not None:
                result_filtered = neighbour_calculator.merge_results(list(result.values()), threshold,
                                                                     len(set(batches)))
            else:
                result_filtered = NeighbourCalculator.filter_similarities(result, threshold)

            # Calculate MSE for each gene pair -
            # compare similarity from gene subset to similarity of the gene pair in gene test set
            sq_errors = []
            for pair, similarity in result_filtered.items():
                gene1 = pair[0]
                gene2 = pair[1]

                index1 = gene_names.index(gene1)
                index2 = gene_names.index(gene2)
                similarity_test = calc_cosine(test_index, test_query, index1, index2, sim_dist=True,
                                              both_directions=both_directions)

                se = (similarity - similarity_test) ** 2
                # Happens if at least one vector has all 0 values
                if not np.isnan(se):
                    sq_errors.append(se)
            if len(sq_errors) > 0:
                mse = round(mean(sq_errors), 5)
            else:
                mse = float('NaN')
            n_genes = len(set(gene for pair in result_filtered.keys() for gene in pair))
            data_summary.append({'N neighbours': neighbours_n, 'inverse': inverse, 'use_log': use_log, 'scale': scale,
                                 'threshold': threshold, 'batches': batch_column, 'MSE': mse,
                                 'N pairs': len(result_filtered), 'N genes': n_genes})
        return data_summary

    def plot_select_threshold(self, thresholds: list, filter_column,
                              filter_column_values_sub: list,
                              filter_column_values_test: list, neighbours_n: int = 100, inverse: bool = False,
                              scale: str = SCALING, use_log: bool = LOG, batch_column=None):
        """
        Plots number of retained genes and MSE based on filtering threshold.
        Parameters are the same as in compare_conditions.
        """
        filtering_summary = self.compare_conditions(neighbours_n=neighbours_n, inverse=inverse,
                                                    scale=scale, thresholds=thresholds, filter_column=filter_column,
                                                    filter_column_values_sub=filter_column_values_sub,
                                                    filter_column_values_test=filter_column_values_test,
                                                    batch_column=batch_column, use_log=use_log)
        pandas_multi_y_plot(pd.DataFrame(filtering_summary), 'threshold', ['N genes', 'MSE'])


# Source: https://matplotlib.org/3.1.1/gallery/ticks_and_spines/multiple_yaxis_with_spines.html
def pandas_multi_y_plot(data: pd.DataFrame, x_col, y_cols: list = None, adjust_right_border=None):
    """
    Plot line plot with scatter points with multiple y axes
    :param data:
    :param x_col: Col names from DF for x
    :param y_cols: Col names from DF for y, if None plot all except x
    :param adjust_right_border: Move plotting area to left to allow more space for y axes
    :return:
    """
    # Get default color style from pandas - can be changed to any other color list
    if y_cols is None:
        y_cols = list(data.columns)
        y_cols.remove(x_col)
    if len(y_cols) == 0:
        return

    colors = getattr(getattr(pd.plotting, '_matplotlib').style, '_get_standard_colors')(num_colors=len(y_cols))
    fig, host = plt.subplots()
    if adjust_right_border is None:
        adjust_right_border = 0.2 * (len(y_cols) - 2) + 0.05
    fig.subplots_adjust(right=1 - adjust_right_border)
    x = data.loc[:, x_col]
    host.set_xlim(min(x), max(x))
    host.set_xlabel(x_col)

    # First axis
    y = data.loc[:, y_cols[0]]
    color = colors[0]
    host.scatter(x, y, color=color)
    host.plot(x, y, color=color)
    host.set_ylim(min(y), max(y))
    host.set_ylabel(y_cols[0])
    host.yaxis.label.set_color(color)
    host.tick_params(axis='y', colors=color)

    for n in range(1, len(y_cols)):
        # Multiple y-axes
        y = data.loc[:, y_cols[n]]
        color = colors[n % len(colors)]
        ax_new = host.twinx()
        ax_new.spines["right"].set_position(("axes", 1 + 0.2 * (n - 1)))
        ax_new.scatter(x, y, color=color)
        ax_new.plot(x, y, color=color)
        ax_new.set_ylim(min(y), max(y))
        ax_new.set_ylabel(y_cols[n])
        ax_new.yaxis.label.set_color(color)
        ax_new.tick_params(axis='y', colors=color)

    return host


def build_graph(similarities: dict) -> nx.Graph:
    """
    Build graph from dictionary.
    :param similarities: Keys are value pairs and values are similarities to be used as weights.
    :return: graph
    """
    graph = nx.Graph()
    for pair, similarity in similarities.items():
        graph.add_edge(pair[0], pair[1], weight=similarity)
    return graph


class Clustering(ABC):
    """
    Abstract class for clustering.
    """

    def __init__(self, result: dict, genes: pd.DataFrame, threshold: float, inverse: bool, scale: str, log: bool):
        """
        Prepare distances (cosine) and gene information data. Use only genes with at least one close neighbour.
        :param result: Closest neighbours result.
        :param genes: Expression data
        :param threshold: Retain only genes with at least one neighbour with similarity at least as big as threshold.
        :param inverse: Distances calculated based on profiole of gene1 and inverse profile of gene2 for each gene pair.
        :param scale: Scale expression data to common scale: 'minmax' from 0 to 1, 'mean0std1' to mean 0 and std 1
        :param log: Log transform data before scaling.
        """
        index, query, gene_names = self.get_genes(result=result, genes=genes, threshold=threshold, inverse=inverse,
                                                  scale=scale, log=log)
        self._gene_names_ordered = gene_names
        self._n_genes = len(gene_names)
        self._distance_matrix = [[0] * self._n_genes for count in range(self._n_genes)]
        self.get_distances_cosine(index=index, query=query, inverse=inverse, scale=scale)

    def get_genes(self, result: dict, genes: pd.DataFrame, threshold: float, inverse: bool, scale: str,
                  log: bool) -> tuple:
        """
        Prepare gene data for distance calculation.
        :param result: Closest neighbours result.
        :param genes: Expression data
        :param threshold: Retain only genes with at least one neighbour with similarity at least as big as threshold.
        :param inverse: Distances calculated based on profiole of gene1 and inverse profile of gene2 for each gene pair.
        :param scale: Scale expression data to common scale: 'minmax' from 0 to 1, 'mean0std1' to mean 0 and std 1
        :param log: Log transform data before scaling.
        :return: index,query - processed expression data. If inverse is False both of them are the same, else query is
            inverse profiles data.
        """
        result_filtered = NeighbourCalculator.filter_similarities(result, threshold)
        if len(result_filtered) == 0:
            raise ValueError('All genes were filtered out. Choose lower threshold.')
        genes_filtered = set((gene for pair in result_filtered.keys() for gene in pair))
        genes_data = genes.loc[genes_filtered, :]
        gene_names = list(genes_data.index)
        index, query = NeighbourCalculator.get_index_query(genes=genes_data, inverse=inverse, scale=scale, log=log)
        return index, query, gene_names

    def get_distances_cosine(self, index: np.ndarray, query: np.ndarray, inverse: bool, scale: str):
        """
        Calculate distances (cosine) between expression profiles for each gene pair.
        If inverse was used on minmax scaling distances matrix would not be symmetrical, thus for each unique gene pair
        distance is calculated in both directions (eg. either gene in index/query) and averaged into final distance.
        :param index: Expression data for gene1 from pair
        :param query: Expression data for gene2 from pair
        :param inverse: Was index data inversed
        :param scale: Which scaling was used to prepare the data
        :return:
        """
        both_directions = False
        if inverse & (scale == 'minmax'):
            both_directions = True
        for i in range(0, self._n_genes - 1):
            for q in range(i + 1, self._n_genes):
                # TODO This might be quicker if pdist from scipy was used when inverse was not needed
                distance = calc_cosine(data1=index, data2=query, index1=i, index2=q, sim_dist=False,
                                       both_directions=both_directions)
                self._distance_matrix[i][q] = distance
                self._distance_matrix[q][i] = distance

    def get_distance_matrix(self) -> list:
        """
        :return: Distance matrix
        """
        return self._distance_matrix.copy()

    @abstractmethod
    def cluster_sizes(self, splitting: float) -> list:
        """
        Size of each cluster
        :param splitting: how to create clusters
        :return: sizes
        """
        pass

    @abstractmethod
    def get_clusters(self, splitting: float) -> np.ndarray:
        """
        Cluster memberships
        :param splitting: how to create clusters
        :return: List of cluster memberships over genes
        """
        pass

    @abstractmethod
    def get_genes_by_clusters(self, splitting: float, filter_genes: iter = None) -> dict:
        """
        Get clusters with corresponding members.
        :param splitting: how to create clusters
        :param filter_genes: Report only genes (leafs) which are contained in filter_genes. If None use all genes in
            creation of membership dictionary.
        :return: Dict keys: cluster, values: list of genes/members
        """
        pass

    def plot_cluster_sizes(self, splitting: float):
        """
        Distribution of cluster sizes
        :param splitting: how to create clusters
         """
        plt.hist(self.cluster_sizes(splitting=splitting),bins=100)
        plt.xlabel('Cluster size')
        plt.ylabel('Cluster count')


class HierarchicalClustering(Clustering):
    """
    Performs hierarchical clustering.
    """

    def __init__(self, result: dict, genes: pd.DataFrame, threshold: float, inverse: bool, scale: str = SCALING,
                 log: bool = LOG):
        """
        Prepare distances (cosine)  and gene information data. Use only genes with at least one close neighbour.
        Performs clustering (Ward).
        :param result: Closest neighbours result.
        :param genes: Expression data
        :param threshold: Retain only genes with at least one neighbour with similarity at least as big as threshold.
        :param inverse: Distances calculated based on profiole of gene1 and inverse profile of gene2 for each gene pair.
        :param scale: Scale expression data to common scale: 'minmax' from 0 to 1, 'mean0std1' to mean 0 and std 1
        :param log: Log transform data before scaling.
        """
        super().__init__(result=result, genes=genes, threshold=threshold, inverse=inverse, scale=scale, log=log)
        upper_index = np.triu_indices(self._n_genes, 1)
        distances = np.array(self._distance_matrix)[upper_index]
        self._hcl = hc.ward(np.array(distances))

    def get_genes_by_clusters(self, splitting: int, filter_genes: iter = None) -> dict:
        """
        Get clusters with corresponding members.
        :param splitting: Height of cutting
        :param filter_genes: Report only genes (leafs) which are contained in filter_genes. If None use all genes in
            creation of membership dictionary.
        :return: Dict keys: cluster, values: list of genes/members
        """
        clusters = self.get_clusters(splitting=splitting)
        cluster_dict = dict((gene_set, []) for gene_set in range(1, len(clusters)))
        for gene, cluster in zip(self._gene_names_ordered, clusters):
            if (filter_genes is None) or (gene in filter_genes):
                cluster_dict[cluster].append(gene)
        if filter_genes is not None:
            for cluster, gene_list in zip(list(cluster_dict.keys()), list(cluster_dict.values())):
                if len(gene_list) == 0:
                    del cluster_dict[cluster]
        return cluster_dict

    def get_clusters(self, splitting: int) -> np.ndarray:
        """
        Cluster memberships
        :param splitting: Height of cutting
        :return: List of cluster memberships over genes
        """
        return hc.fcluster(self._hcl, t=splitting, criterion='distance')

    def cluster_sizes(self, splitting: int) -> list:
        """
        Size of each cluster
        :param splitting: Height of cutting
        :return: sizes
        """
        clusters = list(self.get_clusters(splitting=splitting))
        return list(Counter(clusters).values())

    def plot_clustering(self, cutting_distance: float):
        dendrogram(Z=self._hcl, color_threshold=cutting_distance, no_labels=True)
        plt.ylabel('Distance')


class ClusterAnalyser:
    """
    Analyse clustering result.
    """

    def __init__(self, gene_names: list, organism: int = 44689, max_set_size: int = 100, min_set_size: int = 2):
        """
        :param gene_names: list of gene names (leafs in clustering) for which GO/KEGG annotations will be computed
        :param organism: Organism ID
        :param max_set_size: Max GO/KEGG set size to include in evaluation of clusters
        :param min_set_size: Min GO/KEGG set size to include in evaluation of clusters
        """
        self._organism = organism
        self._entrez_names = dict()
        self._max_set_size = max_set_size
        self._min_set_size = min_set_size
        self._annotation_dict = dict((gene_set, None) for gene_set in list_all(organism=str(self._organism)))

        self.name_genes_entrez(gene_names=gene_names)

    def name_genes_entrez(self, gene_names):
        """
        Add entrez id to each gene name
        :param gene_names: Gene names (eg. from dictyBase)
        """
        matcher = GeneMatcher(self._organism)
        matcher.genes = gene_names
        for gene in matcher.genes:
            name = gene.input_identifier
            entrez = gene.gene_id
            if entrez is not None:
                self._entrez_names[entrez] = name

    def available_annotations(self) -> list:
        """
        Which gene set types are available for this organism (e.g. GO/KEGG gene sets)
        :return: Available gene sets
        """
        return list(self._annotation_dict.keys())

    @staticmethod
    def silhouette(clustering: Clustering, splitting: float) -> float:
        """
        Calculate mean Silhouette Coefficient over clusters  (e.g. sklearn.metrics.silhouette_score)
        :param clustering: The clustering object
        :param splitting: Parameter to use on clustering object's method get_clusters to obtain clusters
            (e.g. N of clusters for hierarchical clustering)
        :return: mean Silhouette Coefficient
        """
        matrix = clustering.get_distance_matrix()
        clusters = clustering.get_clusters(splitting=splitting)
        return silhouette_score(matrix, clusters, metric='precomputed')

    # Probably better to use annotation_ratio
    # Sources: https://stackoverflow.com/questions/35709562/how-to-calculate-clustering-entropy-a-working-example-or-software-code
    # compute_domain_entropy_by_domain_source: https://github.com/DRL/kinfin/blob/master/src/kinfin.py
    #
    # Use only genes (leafs) with annotations
    # H for cluster:
    # p=n_in_set/n_all_annotated_sets ; n_all_annotated_sets - n of all annotations for this cluster,
    #   can be greater than number of all genes wit6h annotations
    # H for single cluster: -sum_over_sets(p*log2(p))
    # H for clustering: sum(entropy*member_genes/all_genes)
    def annotation_entropy(self, clustering: Clustering, splitting: int, ontology: tuple) -> float:
        annotation_dict, clusters = self.init_annotation_evaluation(clustering=clustering, splitting=splitting,
                                                                    ontology=ontology)
        sizes = []
        entropies = []
        for members in clusters.values():
            sizes.append(len(members))
            annotations = []
            for gene in members:
                # if gene in annotation_dict.keys(): # Not needed as filtering is performed above
                annotations += annotation_dict[gene]
            n_annotations = len(annotations)
            cluster_entropy = -sum([count_anno / n_annotations * np.log2(count_anno / n_annotations) for count_anno in
                                    Counter(annotations).values()])
            if str(cluster_entropy) == "-0.0":
                cluster_entropy = 0
            entropies.append(cluster_entropy)

        n_genes = sum(sizes)
        return sum(
            [cluster_entropy * size / n_genes for cluster_entropy, size in zip(entropies, sizes)])

    def annotation_ratio(self, clustering: Clustering, splitting: int, ontology: tuple) -> float:
        """
        For each cluster calculate ratio of genes/members that have the most common annotation (GO/KEGG) in this cluster.
        Compute average over clusters weighted by cluster size. Use only genes/leafs that have annotations.
        For individual cluster: for each annotation term compute ratio of genes that have the term.
            Select the highest ratio.
        For clustering: sum_over_clusters(max_ratio_of_cluster*n_cluster_members/n_all_leafs)
        :param clustering: Clustering object
        :param splitting: Parameter to use on clustering object's method get_clusters to obtain clusters
            (e.g. N of clusters for hierarchical clustering)
        :param ontology: Name of ontology to use for annotations
        :return: Average weighted max ratio.
        """
        annotation_dict, clusters = self.init_annotation_evaluation(clustering, splitting, ontology)
        sizes = []
        ratios = []
        for members in clusters.values():
            n_members = len(members)
            sizes.append(n_members)
            annotations = []
            for gene in members:
                # if gene in annotation_dict.keys(): # Not needed as filtering is performed above
                annotations += annotation_dict[gene]
            max_ration = max([count_anno / n_members for count_anno in
                              Counter(annotations).values()])
            ratios.append(max_ration)
        n_genes = sum(sizes)
        return sum(
            [ratio * size / n_genes for ratio, size in zip(ratios, sizes)])

    def init_annotation_evaluation(self, clustering: Clustering, splitting: int, ontology: tuple):
        """
        Prepare data for annotation based clustering evaluation.
        :param clustering: Clustering object
        :param splitting: Parameter to use on clustering object's method get_clusters to obtain clusters
            (e.g. N of clusters for hierarchical clustering)
        :param ontology: Name of ontology to use for annotations
        :return: annotation_dictionary, clusters
            annotation_dictionary - keys are gene names, values are list of gene sets (from ontology) to
                which this gene belongs
            clusters - dict with keys being cluster numbers and values list of gene names/members
        """
        if self._annotation_dict[ontology] is None:
            self.add_gene_annotations(ontology)
        annotation_dict = self._annotation_dict[ontology]
        clusters = clustering.get_genes_by_clusters(splitting=splitting, filter_genes=annotation_dict.keys())
        return annotation_dict, clusters

    def add_gene_annotations(self, ontology: tuple):
        """
        Add per gene annotations for specified ontology to the self._annotation_dict
        :param  ontology: from available_annotations
        """
        gene_dict = dict((gene_name, []) for gene_name in self._entrez_names.values())
        gene_sets = load_gene_sets(ontology, str(self._organism))
        for gene_set in gene_sets:
            set_size = len(gene_set.genes)
            if self._max_set_size >= set_size >= self._min_set_size:
                for gene_EID in gene_set.genes:
                    if gene_EID in self._entrez_names.keys():
                        gene_dict[self._entrez_names[gene_EID]].append(gene_set.name)
        for gene, sets in zip(list(gene_dict.keys()), list(gene_dict.values())):
            if sets == []:
                del gene_dict[gene]
        self._annotation_dict[ontology] = gene_dict

    def plot_clustering_metrics(self, clustering: Clustering, splittings: list, ontology: tuple = ('KEGG', 'Pathways')):
        """
        Analyse different values of splitting parameter used on clustering.
        Plot silhouette values, median cluster size and average cluster-size weighted maximal annotation ratio.
        :param clustering: Clustering object
        Parameter to use on clustering object's method get_clusters to obtain clusters
            (e.g. N of clusters for hierarchical clustering)
        :param ontology: Name of ontology to use for annotations
        """
        silhouettes = []
        median_sizes = []
        ratios = []
        n_clusters = []
        for splitting in splittings:
            cluster_sizes = clustering.cluster_sizes(splitting=splitting)
            median_sizes.append(median(cluster_sizes))
            n_clusters.append(len(cluster_sizes))
            silhouettes.append(ClusterAnalyser.silhouette(clustering=clustering, splitting=splitting))
            ratios.append(self.annotation_ratio(clustering=clustering, splitting=splitting, ontology=ontology))
        data = pd.DataFrame({'Splitting parameter': splittings, 'Mean silhouette values': silhouettes,
                             'Mean max annotation ratio': ratios, 'Median cluster size': median_sizes,
                             'N clusters': n_clusters})

        pandas_multi_y_plot(data=data, x_col='Splitting parameter', y_cols=None, adjust_right_border=0.35)


def calc_cosine(data1: np.ndarray, data2: np.ndarray, index1: int, index2: int, sim_dist: bool,
                both_directions: bool) -> float:
    """
    Calculate cosine distance or similarity between 2 elements.
    :param data1: Matrix for 1st element's data
    :param data2: Matrix for 2nd element's data
    :param index1: Matrix index to obtain data from matrix for 1st element
    :param index2: Matrix index to obtain data from matrix for 2nd element
    :param sim_dist: True: similarity, False: distance (1-similarity)
    :param both_directions: If true calculate the metric (sim/dist) as average of metric when using index1 (i1) on
    matrix1 (m1) and i2 on m2 and when using vice versa; e.g. avr(metric(m1[i1],m2[i2]), metric(m1[i2],m2[i1]))
    :return: Similarity or distance
    """
    similarity = SimilarityCalculator.calc_cosine(data1[index1], data2[index2])
    if both_directions:
        similarity2 = SimilarityCalculator.calc_cosine(data1[index2], data2[index1])
        similarity = (similarity + similarity2) / 2
    if sim_dist:
        return similarity
    else:
        return 1 - similarity
