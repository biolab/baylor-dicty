from abc import ABC, abstractmethod

import pandas as pd
import sklearn.preprocessing as pp
from pynndescent import NNDescent
import numpy as np
from statistics import mean
import networkx as nx
import warnings
import scipy.cluster.hierarchy as hc
from sklearn.metrics import silhouette_score
from collections import Counter
from math import log

from orangecontrib.bioinformatics.ncbi.gene import GeneMatcher
from orangecontrib.bioinformatics.geneset.__init__ import (list_all, load_gene_sets)

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


class Clustering(ABC):
    def __init(self):
        self._n_genes = None
        self._distances = None

    @abstractmethod
    def get_clusters(self, n_clusters: int):
        pass

    @abstractmethod
    def get_distance_matrix(self):
        pass

    @abstractmethod
    def get_genes_by_clusters(self, n_clusters: int, filter_genes: iter = None):
        pass

    def distances_to_matrix(self):
        self._matrix = [[0] * self._n_genes for i in range(self._n_genes)]
        position = 0
        for i in range(0, self._n_genes - 1):
            for j in range(0, self._n_genes - 1 - i):
                element = self._distances[position]
                if element < 0:
                    element = 0
                position += 1
                self._matrix[i][i + j + 1] = element  # fill in the upper triangle
                self._matrix[i + j + 1][i] = element  # fill in the lower triangle


class HierarchicalClustering(Clustering):

    def __init__(self, result: dict, genes: pd.DataFrame, threshold: float, inverse: bool, scale: str, log: bool):
        result_filtered = NeighbourCalculator.filter_similarities(result, threshold)
        genes_filtered = set((gene for pair in result_filtered.keys() for gene in pair))
        genes_data = genes.loc[genes_filtered, :]
        self._gene_names = list(genes_data.index)
        index, query = NeighbourCalculator.get_index_query(genes=genes_data, inverse=inverse, scale=scale, log=log)
        self._n_genes = genes_data.shape[0]
        self._distances = []
        self._matrix = None
        both_directions = False
        if inverse & (scale == 'minmax'):
            both_directions = True
        for i in range(0, self._n_genes - 1):
            for q in range(i + 1, self._n_genes):
                # TODO This might be quicker if pdist from scipy was used when inverse was not needed
                self._distances.append(calc_cosine(data1=index, data2=query, index1=i, index2=q, sim_dist=False,
                                                   both_directions=both_directions))
        self._hcl = hc.ward(np.array(self._distances))

    def get_genes_by_clusters(self, n_clusters: int, filter_genes: iter = None):
        clusters = hc.fcluster(self._hcl, t=n_clusters, criterion='maxclust')
        cluster_dict = dict((gene_set, []) for gene_set in range(1, n_clusters + 1))
        for gene, cluster in zip(self._gene_names, clusters):
            if (filter_genes is None) or (gene in filter_genes):
                cluster_dict[cluster].append(gene)
        if filter_genes is not None:
            for cluster, gene_list in zip(list(cluster_dict.keys()), list(cluster_dict.values())):
                if len(gene_list) == 0:
                    del cluster_dict[cluster]
        return cluster_dict

    def get_clusters(self, n_clusters: int):
        return hc.fcluster(self._hcl, t=n_clusters, criterion='maxclust')

    def get_distance_matrix(self):
        # TODO maybe store only distance vector or matrix, as they can be converted to each other
        if self._matrix is None:
            self.distances_to_matrix()
        return self._matrix.copy()

    def cluster_sizes(self, n_clusters: int):
        clusters = list(hc.fcluster(self._hcl, t=n_clusters, criterion='maxclust'))
        return Counter(clusters).values()


class ClusterAnalyser:

    def __init__(self, gene_names: list, organism: int = '44689', max_set_size: int = 100, min_set_size: int = 2):
        self._organism = organism
        self._entrez_names = dict()
        self._max_set_size = max_set_size
        self._min_set_size = min_set_size
        self._annotation_dict = dict((gene_set, None) for gene_set in list_all(organism=str(self._organism)))

        self.name_genes_entrez(gene_names=gene_names)

    def name_genes_entrez(self, gene_names):
        matcher = GeneMatcher(self._organism)
        matcher.genes = gene_names
        for gene in matcher.genes:
            name = gene.input_identifier
            entrez = gene.gene_id
            if entrez is not None:
                self._entrez_names[entrez] = name

    def available_annotations(self):
        return list(self._annotation_dict.keys())

    @staticmethod
    def silhouette(clustering: Clustering, n_clusters: int):
        matrix = clustering.get_distance_matrix()
        clusters = clustering.get_clusters(n_clusters)
        return silhouette_score(matrix, clusters, metric='precomputed')

    # Sources: https://stackoverflow.com/questions/35709562/how-to-calculate-clustering-entropy-a-working-example-or-software-code
    # compute_domain_entropy_by_domain_source: https://github.com/DRL/kinfin/blob/master/src/kinfin.py
    # For cluster:
    # p=n_in_set/n_all_annotated_sets ; n_all_annotated_sets - n of all annotations for this cluster
    # for single cluster: -sum_over_sets(p*log2(p))
    # For clustering:
    # sum(entropy*member_genes/all_genes) : including only genes with annotations
    def annotation_entropy(self, clustering: Clustering, n_clusters: int, ontology: tuple):
        annotation_dict, clusters = self.init_annotation_evaluation(clustering, n_clusters, ontology)
        sizes = []
        entropies = []
        for members in clusters.values():
            sizes.append(len(members))
            annotations = []
            for gene in members:
                # if gene in annotation_dict.keys(): # Not needed as filtering is performed above
                annotations += annotation_dict[gene]
            n_annotations = len(annotations)
            cluster_entropy = -sum([count_anno / n_annotations * log(count_anno / n_annotations, 2) for count_anno in
                                    Counter(annotations).values()])
            if str(cluster_entropy) == "-0.0":
                cluster_entropy = 0
            entropies.append(cluster_entropy)

        n_genes = sum(sizes)
        return sum(
            [cluster_entropy * size / n_genes for cluster_entropy, size in zip(entropies, sizes)])

    def annotation_ratio(self, clustering: Clustering, n_clusters: int, ontology: tuple):
        annotation_dict, clusters = self.init_annotation_evaluation(clustering, n_clusters, ontology)
        sizes = []
        ratios = []
        for members in clusters.values():
            n_genes=len(members)
            sizes.append(n_genes)
            annotations = []
            for gene in members:
                # if gene in annotation_dict.keys(): # Not needed as filtering is performed above
                annotations += annotation_dict[gene]
            max([count_anno/n_genes for count_anno in
                                    Counter(annotations).values()])
        #TODO

    def init_annotation_evaluation(self, clustering: Clustering, n_clusters: int, ontology: tuple):
        if self._annotation_dict[ontology] is None:
            self.add_gene_annotations(ontology)
        annotation_dict = self._annotation_dict[ontology]
        clusters = clustering.get_genes_by_clusters(n_clusters, annotation_dict.keys())
        return annotation_dict, clusters

    def add_gene_annotations(self, ontology: tuple):
        """
        :param  ontology: from available_annotations
        """
        gene_dict = dict((gene_name, []) for gene_name in self._entrez_names.values())
        gene_sets = load_gene_sets(ontology, str(self._organism))
        for gene_set in gene_sets:
            for gene_EID in gene_set.genes:
                if gene_EID in self._entrez_names.keys():
                    gene_dict[self._entrez_names[gene_EID]].append(gene_set.name)
        for gene, sets in zip(list(gene_dict.keys()), list(gene_dict.values())):
            if sets == []:
                del gene_dict[gene]
        self._annotation_dict[ontology] = gene_dict


def calc_cosine(data1, data2, index1, index2, sim_dist, both_directions):
    similarity = SimilarityCalculator.calc_cosine(data1[index1], data2[index2])
    if both_directions:
        similarity2 = SimilarityCalculator.calc_cosine(data1[index2], data2[index1])
        similarity = (similarity + similarity2) / 2
    if sim_dist:
        return similarity
    else:
        return 1 - similarity
