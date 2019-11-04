import pandas as pd
import sklearn.preprocessing as pp
from pynndescent import NNDescent
import numpy as np
from statistics import mean
import networkx as nx
import warnings
import scipy.cluster.hierarchy as hc
import scipy.spatial.distance as sp_dist

from correlation_enrichment.library import GeneExpression, SimilarityCalculator


class NeighbourCalculator:
    """
    Obtains best neighbours of genes based on their expression profile.
    This can be done for all conditions at once or by condition groups, later on merging the results into
    single neighbour data.
    """

    MINMAX = 'minmax'
    MEANSTD = 'mean0std1'
    SCALES = [MINMAX, MEANSTD]

    def __init__(self, genes: pd.DataFrame, remove_zero: bool = True):
        """
        :param genes: Data frame of genes in rows and conditions in columns. Index is treated as gene names
        :param remove_zero: Remove genes that have all expression values 0.
            If batches are latter specified there may be all 0 rows within individual batches.
        """
        GeneExpression.check_numeric(genes)
        if remove_zero:
            genes = genes[(genes != 0).any(axis=1)]
        self._genes = genes

    def neighbours(self, n_neighbours: int, inverse: bool, scale: str, log: bool, batches: list = None):
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
    def get_index_query(cls, genes: pd.DataFrame, inverse: bool, scale: str, log: bool):
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
    def minmax_scale(genes: pd.DataFrame):
        """
        Scale each row from 0 to 1.
        :param genes: data
        :return: scaled data
        """
        return pp.minmax_scale(genes, axis=1)

    @staticmethod
    def meanstd_scale(genes):
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


class HierarchicalCluster:

    @staticmethod
    def hclust_correlated(result: dict, genes: pd.DataFrame, threshold: int, inverse: bool, scale: str, log: bool):
        result_filtered = NeighbourCalculator.filter_similarities(result, threshold)
        genes_filtered = set((gene for pair in result_filtered.keys() for gene in pair))
        genes_data = genes.loc[genes_filtered, :]
        gene_names=list(genes_data.index)
        index, query = NeighbourCalculator.get_index_query(genes=genes_data, inverse=inverse, scale=scale, log=log)
        n_genes = genes_data.shape[0]
        distances = []
        both_directions = False
        if inverse & (scale == 'minmax'):
            both_directions = True
        for i in range(0, n_genes - 1):
            for q in range(i + 1, n_genes):
                # TODO This might be quicker if pdist from scipy was used when inverse was not needed
                distances.append(calc_cosine(data1=index, data2=query, index1=i, index2=q, sim_dist=False,
                                             both_directions=both_directions))
        return hc.ward(np.array(distances)),gene_names

    @staticmethod
    def get_clusters(clustering:np.ndarray,gene_names:list,groups:int):
        clusters=hc.fcluster(clustering, t=groups, criterion='maxclust')
        return dict(zip(gene_names,clusters))

class ClusterAnalyser:

    def c

def calc_cosine(data1, data2, index1, index2, sim_dist, both_directions):
    similarity = SimilarityCalculator.calc_cosine(data1[index1], data2[index2])
    if both_directions:
        similarity2 = SimilarityCalculator.calc_cosine(data1[index2], data2[index1])
        similarity = (similarity + similarity2) / 2
    if sim_dist:
        return similarity
    else:
        return 1 - similarity


# TODO
n=4
matrix=[[0] * n for i in range(n)]
position=0
for i in range(0, n - 1):
        for j in range(0, n - 1 - i):
            element = distances[position]
            if element<0:
                element=0
            position += 1
            matrix[i][i + j + 1] = element # fill in the upper triangle
            matrix[i + j + 1][i] = element # fill in the lower triangle
silhouette_score(matrix,classes,metric='precomputed')
