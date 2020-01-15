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
import openTSNE as ot
from sklearn.cluster import DBSCAN
from community import best_partition as louvain
import random
from collections import OrderedDict
from sklearn.mixture import GaussianMixture
from scipy.integrate import cumtrapz

import Orange.clustering.louvain as orange_louvain_graph
from orangecontrib.bioinformatics.ncbi.gene import GeneMatcher
from orangecontrib.bioinformatics.geneset.__init__ import (list_all, load_gene_sets)
import orangecontrib.bioinformatics.go as go
from Orange.clustering.louvain import jaccard

from correlation_enrichment.library_correlation_enrichment import GeneExpression, SimilarityCalculator
from deR.enrichment_library import GO_enrichment, name_genes_entrez

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
                   batches: list = None, remove_batch_zero: bool = True, return_neigh_dist: bool = False,
                   genes_query_names: list = None, remove_self: bool = False):
        """
        Calculates neighbours of genes on whole gene data or its subset by column.
        :param n_neighbours: Number of neighbours to obtain for each gene
        :param inverse: Calculate most similar neighbours (False) or neighbours with inverse profile (True)
        :param scale: Scale expression by gene with 'minmax' (min=0, max=1) or 'mean0std1' (mean=0, std=1)
        :param log: Should expression data be log2 transformed
        :param batches: Should comparisons be made for each batch separately.
            Batches should be a list of batch group names for each column (eg. length of batches is n columns of genes).
        :param remove_batch_zero: Remove genes that have all expression values 0 for each batch individually.
        :param return_neigh_dist: Instead of parsed dictionary return tuple with NN matrix and distance matrix,
        as returned by pynndescent but named with gene names in data frame.
        :param genes_query_names: Use only the specified genes as query.
        :param remove_self: Used only if return_neigh_dist is true. Whether to remove sample from its closest
        neighbours or not. If retunr_neigh_dist is False this is done automatically. This also removes last
        column/neighbours is self is not present - should not be used with inverse.
        :return: Dict with gene names as tupple keys (smaller by alphabet is first tuple value) and
            values representing cosine similarity. If batches are used such dicts are returned for each batch
            in form of dict with batch names as keys and above mentioned dicts as values. Or see return_neigh_dist.
        """
        if scale not in self.SCALES:
            raise ValueError('Scale must be:', self.SCALES)

        genes = self._genes

        if batches is None:
            if genes_query_names is not None:
                genes_query = genes.loc[genes_query_names, :]
            else:
                genes_query = None
            return NeighbourCalculator.calculate_neighbours(genes=genes, n_neighbours=n_neighbours, inverse=inverse,
                                                            scale=scale,
                                                            log=log, return_neigh_dist=return_neigh_dist,
                                                            genes_query_data=genes_query, remove_self=remove_self)
        else:
            batch_groups = set(batches)
            batches = np.array(batches)
            results = dict()
            for batch in batch_groups:
                genes_sub = genes.T[batches == batch].T
                if remove_batch_zero:
                    genes_sub = genes_sub[(genes_sub != 0).any(axis=1)]

                if genes_query_names is not None:
                    genes_query_sub = genes_sub.loc[genes_query_names, :]
                else:
                    genes_query_sub = None

                result = NeighbourCalculator.calculate_neighbours(genes=genes_sub, n_neighbours=n_neighbours,
                                                                  inverse=inverse,
                                                                  scale=scale, log=log, description=batch,
                                                                  return_neigh_dist=return_neigh_dist,
                                                                  genes_query_data=genes_query_sub,
                                                                  remove_self=remove_self)
                results[batch] = result
            return results

    @staticmethod
    def calculate_neighbours(genes, n_neighbours: int, inverse: bool, scale: str, log: bool,
                             description: str = '', return_neigh_dist: bool = False,
                             genes_query_data: pd.DataFrame = None, remove_self: bool = False):
        """
        Calculate neighbours of genes.
        :param genes: Data frame as in init, gene names (rows) should match the one in init
        :param n_neighbours: Number of neighbours to obtain for each gene
        :param inverse: Calculate most similar neighbours (False) or neighbours with inverse profile (True)
        :param scale: Scale expression by gene with 'minmax' (min=0, max=1) or 'mean0std1' (mean=0, std=1)
        :param log: Should expression data be log2 transformed
        :param description: If an error occurs in KNN index formation report this with error
        :param return_neigh_dist: Instead of parsed dictionary return tuple with NN matrix and distance matrix,
        as returned by pynndescent but named with gene names  in data frame.
        :param genes_query_data: Use this as query. If None use genes.
        :param remove_self: Used only if return_neigh_dist is true. Whether to remove sample from its closest
        neighbours or not. If retunr_neigh_dist is False this is done automatically. This also removes last
        column/neighbours is self is not present - should not be used with inverse.
        :return: Dict with gene names as tuple keys (smaller by alphabet is first tuple value) and
            values representing cosine similarity. Or see return_neigh_dist
        """
        genes_index, genes_query = NeighbourCalculator.get_index_query(genes=genes, inverse=inverse, scale=scale,
                                                                       log=log,
                                                                       genes_query_data=genes_query_data)
        # Can set speed-quality trade-off, default is ok
        try:
            index = NNDescent(genes_index, metric='cosine', n_jobs=4)
        except ValueError:
            try:
                index = NNDescent(genes_index, metric='cosine', tree_init=False, n_jobs=4)
                warnings.warn(
                    'Dataset ' + description + ' index computed without tree initialisation',
                    Warning)
            except ValueError:
                raise ValueError('Dataset ' + description + ' can not be processed by pydescent')
        neighbours, distances = index.query(genes_query.tolist(), k=n_neighbours)

        if genes_query_data is None:
            genes_query_data = genes
        if return_neigh_dist:
            neighbours = NeighbourCalculator.parse_neighbours_matrix(neighbours=neighbours,
                                                                     genes_query=genes_query_data,
                                                                     genes_idx=genes)
            similarities = pd.DataFrame(NeighbourCalculator.parse_distances_matrix(distances),
                                        index=genes_query_data.index)
            if remove_self:
                neighbours, similarities = NeighbourCalculator.remove_self_pynn_matrix(neighbours=neighbours,
                                                                                       similarities=similarities)
            return neighbours, similarities
        else:
            return NeighbourCalculator.parse_neighbours(neighbours=neighbours, distances=distances,
                                                        genes_query=genes_query_data, genes_idx=genes)

    @classmethod
    def get_index_query(cls, genes: pd.DataFrame, inverse: bool, scale: str = SCALING, log: bool = LOG,
                        genes_query_data: pd.DataFrame = None) -> tuple:
        """
        Get gene data scaled to be index or query for neighbour search.
        :param genes: Gene data for index and query.
        :param inverse: Inverse query to compute neighbours with opposite profile. True if use inverse.
        :param scale: Scale expression by gene with 'minmax' (min=0, max=1) or 'mean0std1' (mean=0, std=1).
        :param log: Should expression data be log2 transformed.
        :param genes_query_data: Genes data for query, if None uses genes
        :return: genes for index (1st element) and genes for query (2nd element)
        """
        if log:
            genes = np.log2(genes + 1)
            if genes_query_data is not None:
                genes_query_data = np.log2(genes_query_data + 1)
        if inverse:
            if genes_query_data is None:
                genes_query_data = genes
            genes_query = genes_query_data * -1
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
                if genes_query_data is not None:
                    genes_query_data = cls.minmax_scale(genes_query_data)
            elif scale == cls.MEANSTD:
                genes = cls.meanstd_scale(genes)
                if genes_query_data is not None:
                    genes_query_data = cls.meanstd_scale(genes_query_data)
            genes_index = genes
            if genes_query_data is None:
                genes_query_data = genes
            genes_query = genes_query_data
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

    @staticmethod
    def parse_neighbours(neighbours: np.ndarray, distances: np.ndarray, genes_idx: pd.DataFrame,
                         genes_query: pd.DataFrame) -> dict:
        """
        Transform lists of neighbours and distances into dictionary with neighbours as keys and values as similarities.
        If pair of neighbours is given more than once it is overwritten the second time it is added to dictionary.
        For cosine similarity the above should be always the same.
        The neighbours (in form of index) are named based on gene data (index) stored in NeighbourCalculator instance.
        :param neighbours: Array of shape: genes*neighbours, where for each gene there are specified neighbours
        :param distances: Array of shape: genes*neighbours, where for each gene there are specified distances
            for each neighbour, as they are given above
        :param genes_query: DataFrame used to name genes in neighbours & distanmces rows (querry),
         gene names in rows.
        :param genes_idx: DataFrame used to name genes in neighbours & distanmces table entries,
         gene names in rows.
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
                similarity = NeighbourCalculator.cosine_dist_to_sim(distance)
                gene2 = neighbours[gene, neighbour]
                gene_name1 = genes_query.index[gene]
                gene_name2 = genes_idx.index[gene2]
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
    def parse_neighbours_matrix(neighbours: np.ndarray, genes_idx: pd.DataFrame,
                                genes_query: pd.DataFrame) -> pd.DataFrame:
        """
        Names pynndescent neighbours table (values and rows).
        :param neighbours: As returned by pynndescent
        :param genes_idx: Data frame with rownames matching neighbours values (used for pynndescent index)
        :param genes_query: Data frame with rownames matching neighbours rows (used for pynndescent query)
        :return: Named neighbours table.
        """
        parsed = pd.DataFrame(columns=range(neighbours.shape[1]))
        for gene1 in range(neighbours.shape[0]):
            for gene2_col in range(neighbours.shape[1]):
                gene2 = neighbours[gene1, gene2_col]
                gene_name1 = genes_query.index[gene1]
                gene_name2 = genes_idx.index[gene2]
                parsed.loc[gene_name1, gene2_col] = gene_name2
        return parsed

    @staticmethod
    def parse_distances_matrix(distances: np.ndarray):
        """
        Transform cosine distances to similarities
        :param distances: pynndescent cosine distance matrix
        """
        if (np.around(distances, 4).any() < 0):
            warnings.warn(
                'Odd cosine distance in the matrix', Warning)
        return NeighbourCalculator.cosine_dist_to_sim(distances)

    @staticmethod
    def remove_self_pynn_matrix(neighbours: pd.DataFrame, similarities: pd.DataFrame):
        """
        Parse similarities and neighbours data frames. Remove entries for each gene that represent itself being
        the closest neighbour. If row name not in neghours row removes last element from neighbours and similarities.
        :param similarities: Similarities matrix from neighbours function (row names with genes)
        :param neighbours: Neighbours matrix from neighbours function (row names and values named with genes)
        """
        similarities_parsed = pd.DataFrame(index=similarities.index, columns=similarities.columns[:-1])
        neighbours_parsed = pd.DataFrame(index=neighbours.index, columns=neighbours.columns[:-1])
        for gene in neighbours.index:
            neigh_row = neighbours.loc[gene, :]
            sims_row = similarities.loc[gene, :]
            if gene not in neigh_row.values:
                similarities_parsed.loc[gene, :] = sims_row.iloc[:-1]
                neighbours_parsed.loc[gene, :] = neigh_row.iloc[:-1]
            else:
                self_idx = neigh_row[neigh_row == gene].index[0]
                similarities_parsed.loc[gene, :] = sims_row.drop(similarities.columns[self_idx]).values
                neighbours_parsed.loc[gene, :] = neigh_row.drop(neighbours.columns[self_idx]).values
        return neighbours_parsed, similarities_parsed

    @staticmethod
    def cosine_dist_to_sim(dist):
        return 1 - dist

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
    def filter_similarities_batched(results: dict, similarity_threshold: float = 0, min_present: int = 1,
                                    merge_function=mean) -> dict:
        """
        Filter and merge results from batched neighbour calculation where results have been previously
        added to single dictionary.
        :param results: Dict with keys (gene1,gene2) and values list of similarities across batches
        :param similarity_threshold: Remove similarities below threshold.
        :param min_present: Remove gene pairs that have less than min_present similarities left after similarity
            threshold filtering.
        :param merge_function: Apply to list of retained similarities to get single value to save in result dict
        :return: Dict with keys (gene1,gene2) and values being results of merge function applied to
            list of filtered similarities.
        """
        retained = {}
        for pair, similarities in results.items():
            retained_similarities = [item for item in similarities if item >= similarity_threshold]
            if len(retained_similarities) >= min_present:
                retained[pair] = merge_function(retained_similarities)
        return retained

    @staticmethod
    def filter_similarities(results: dict, similarity_threshold: float) -> dict:
        """
        Filter out pairs that have similarity below threshold.
        :param results: Dict with values as similarities.
        :param similarity_threshold: Remove all keys with value below threshold.
        :return: Filtered dictionary, retaining only items with values above or equal to threshold.
        """
        return dict(filter(lambda elem: elem[1] >= similarity_threshold, results.items()))

    @staticmethod
    def filter_distances_matrix(similarities: pd.DataFrame, similarity_threshold: float,
                                min_neighbours: int = 2) -> list:
        """
        Returns list of genes that have at least min_neighbours close neighbours. If used on non-inverse profiles
        min-neighbours must be set to required+1 to account for a gene being its closest neighbour.
        :param similarities: Distances to nearest neighbours, queries/genes in rows, distances in columns.
        :param similarity_threshold: Retain only similarities equal or above threshold.
        :param min_neighbours: Retain only genes/rows with at least that many neighbours above similarity_threshold.
        :return: List of row names that passed filtering.
        """
        similarities_retained = similarities.mask(similarities < similarity_threshold
                                                  ).count(axis=1).iloc[:] >= min_neighbours
        return list(similarities_retained[similarities_retained].index)

    @staticmethod
    def hub_neighbours(neighbours: pd.DataFrame, similarities: pd.DataFrame, similarity_threshold: float) -> dict:
        """
        For each query gene return genes that are its neighbours above threshold.
        Removes the query itself if it is among its own neighbours.
        :param neighbours: Neighbours  queries/genes in rows, neighbours in columns.
        :param similarities: Similarities to neighbours, queries/genes in rows, distances in columns.
        :param similarity_threshold: Retain only neighbours with similarities equal or above threshold.
        :return: Dictionary of row names/queries (keys) and neighbours (values)  that passed filtering.
        """
        parsed = {}
        for gene in neighbours.index:
            neighbour_similarities = similarities.loc[gene, :] >= similarity_threshold
            retained_neighbours = list(neighbours.loc[gene, neighbour_similarities].values)
            if gene in retained_neighbours:
                retained_neighbours.remove(gene)
            parsed[gene] = retained_neighbours
        return parsed

    @staticmethod
    def find_hubs(similarities: pd.DataFrame, n_hubs: int) -> list:
        """
        Retain rows that have highest average similarity in the similarities data
        (Genes in rows, similarity of closest neighbours in columns)
        :param similarities: Distances to nearest neighbours, queries/genes in rows, distances in columns.
        :param n_hubs: N rows to retain
        :return: Rownames of selected rows
        """
        averages = similarities.mean(axis=1)
        averages = averages.sort_values(ascending=False)
        if n_hubs > averages.shape[0]:
            raise ValueError('n_hubs is greater than N data points')
        return list(averages.iloc[:n_hubs].index)

    def compare_conditions(self, neighbours_n: int, inverse: bool,
                           scale: str, use_log: bool, thresholds: list, filter_column, filter_column_values_sub: list,
                           filter_column_values_test: list, retained: list = None, batch_column=None,
                           do_mse: bool = True):
        """
        Evaluates pattern similarity calculation preprocessing and parameters based on difference between subset and
        test set. Computes MSE from differences between similarities of subset gene pairs and corresponding test gene
        pairs.
        :param neighbours_n: N of calculated neighbours for each gene
        :param inverse: find neighbours with opposite profile
        :param scale: 'minmax' (from 0 to 1) or 'mean0std1' (to mean 0 and std 1)
        :param use_log: Log transform expression values before scaling
        :param thresholds: filter out any result with similarity below threshold, do for each threshold
        :param filter_column: On which column of conditions should genes be subset for separation in subset and test set
        :param filter_column_values_sub: Values of filter_column to use for subset genes
        :param filter_column_values_test:Values of filter_column to use for test genes
        :param batch_column: Should batches be used based on some column of conditions
        :param do_mse: Should MSE calculation be performed
        :param retained: Two element list - how many genes must be retained. If not within range
         ignore and use next threshold from list. First element min, second max; inclusive.
          If None calculates on all thresholds.
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
        neighbour_calculator_test = NeighbourCalculator(genes_test)
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
        result_test = neighbour_calculator_test.neighbours(neighbours_n, inverse=inverse, scale=scale, log=use_log,
                                                           batches=batches)

        # Filter neighbours on similarity
        data_summary = []
        for threshold in thresholds:
            if batches is not None:
                result_filtered = NeighbourCalculator.merge_results(list(result.values()), threshold,
                                                                    len(set(batches)))
                result_filtered_test = NeighbourCalculator.merge_results(list(result_test.values()), threshold,
                                                                         len(set(batches)))
            else:
                result_filtered = NeighbourCalculator.filter_similarities(result, threshold)
                result_filtered_test = NeighbourCalculator.filter_similarities(result_test, threshold)

            # Find genes retained in each result
            gene_names_sub = {gene for pair in result_filtered for gene in pair}
            if retained is not None:
                if not (retained[0] <= len(gene_names_sub) <= retained[1]):
                    continue
            gene_names_test = {gene for pair in result_filtered_test for gene in pair}
            f_val = NeighbourCalculator.f_value(set1=gene_names_sub, set2=gene_names_test)
            # Calculate MSE for each gene pair -
            # compare similarity from gene subset to similarity of the gene pair in gene test set
            sq_errors = []
            if do_mse:
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
                                 'N pairs': len(result_filtered), 'N genes': n_genes, 'F value': f_val})
            if retained is not None:
                break
        return data_summary

    def compare_thresholds(self, neighbours_n: int, inverse: bool,
                           scale: str, use_log: bool, thresholds: list, filter_column, filter_column_values1: list,
                           filter_column_values2: list, genes_query_names: list = None) -> pd.DataFrame:
        """
        Compare retained genes and gene pairs at different similarity thresholds.
        Computes table with F values of retained genes/pairs from two different data subgroups.
         :param neighbours_n: N of calculated neighbours for each gene
        :param inverse: find neighbours with opposite profile
        :param scale: 'minmax' (from 0 to 1) or 'mean0std1' (to mean 0 and std 1)
        :param use_log: Log transform expression values before scaling
        :param thresholds: filter out any result with similarity below threshold, do for each threshold
        :param filter_column: On which column of conditions should genes be subset for separation in subsets
        :param filter_column_values1: Values of filter_column to use for subset1 genes
        :param filter_column_values2:Values of filter_column to use for subset2 genes
        :param genes_query_names: Use only these genes for query.
        :return: Data Frame with columns: paramteres (log, inverse,scaling, threshold, N neighbours)
        and number of retained genes/pairs for each subset and their agreement (F value).
        """
        genes1 = self._genes.T[list(self.conditions[filter_column].isin(filter_column_values1))].T
        genes2 = self._genes.T[list(self.conditions[filter_column].isin(filter_column_values2))].T

        neighbour_calculator1 = NeighbourCalculator(genes1)
        neighbour_calculator2 = NeighbourCalculator(genes2)

        # Calculate neighbours
        result1 = neighbour_calculator1.neighbours(neighbours_n, inverse=inverse, scale=scale, log=use_log,
                                                   genes_query_names=genes_query_names)
        result2 = neighbour_calculator2.neighbours(neighbours_n, inverse=inverse, scale=scale, log=use_log,
                                                   genes_query_names=genes_query_names)

        # Filter neighbours on similarity
        data_summary = []
        for threshold in thresholds:
            result_filtered1 = NeighbourCalculator.filter_similarities(result1, threshold)
            result_filtered2 = NeighbourCalculator.filter_similarities(result2, threshold)
            gene_names1 = {gene for pair in result_filtered1 for gene in pair}
            gene_names2 = {gene for pair in result_filtered2 for gene in pair}
            f_val_genes = NeighbourCalculator.f_value(set1=gene_names1, set2=gene_names2)
            f_val_pairs = NeighbourCalculator.f_value(set1=set(result_filtered1.keys()),
                                                      set2=set(result_filtered2.keys()))
            data_summary.append({'N neighbours': neighbours_n, 'inverse': inverse, 'use_log': use_log, 'scale': scale,
                                 'threshold': threshold,
                                 'N pairs1': len(result_filtered1), 'N pairs2': len(result_filtered2),
                                 'N genes1': len(gene_names1), 'N genes2': len(gene_names2),
                                 'F value genes': f_val_genes, 'F value pairs': f_val_pairs})
        return pd.DataFrame(data_summary)

    @staticmethod
    def f_value(set1: set, set2: set):
        """
        Compute mock F value. n_both = genes selected in both subsets, FN1 = genes selected only in subset2,
        FN2 = genes selected only in subset1. r1=n_both/(n_both+FN1) and r2=n_both/(n_both+FN2). F=2 * r1 * r2/(r1+r2)
        :param set1: elements 1
        :param set2:elements 2
        :return: F value
        """
        match = set1 & set2
        n_both = len(match)
        n_only1 = len(set1 ^ match)
        n_only2 = len(set2 ^ match)
        if n_both + n_only2 < 1:
            recall1 = float('NaN')
        else:
            recall1 = n_both / (n_both + n_only2)
        if n_both + n_only1 < 1:
            recall2 = float('NaN')
        else:
            recall2 = n_both / (n_both + n_only1)
        if recall1 == recall2 == float('NaN') or recall1 + recall2 == 0:
            f_val = float('NaN')
        else:
            f_val = 2 * recall1 * recall2 / (recall1 + recall2)
        return f_val

    def plot_select_threshold(self, thresholds: list, filter_column,
                              filter_column_values_sub: list,
                              filter_column_values_test: list, neighbours_n: int = 2, inverse: bool = False,
                              scale: str = SCALING, use_log: bool = LOG, batch_column=None, retained: list = None):
        """
        Plots number of retained genes and MSE based on filtering threshold.
        Parameters are the same as in compare_conditions.
        """
        filtering_summary = pd.DataFrame(self.compare_conditions(neighbours_n=neighbours_n, inverse=inverse,
                                                                 scale=scale, thresholds=thresholds,
                                                                 filter_column=filter_column,
                                                                 filter_column_values_sub=filter_column_values_sub,
                                                                 filter_column_values_test=filter_column_values_test,
                                                                 batch_column=batch_column, use_log=use_log,
                                                                 do_mse=False, retained=retained))
        pandas_multi_y_plot(filtering_summary, 'threshold', ['N genes', 'F value'])
        return filtering_summary

    @staticmethod
    def compare_threshold_batched(sample1: dict, sample2: dict, similarity_thresholds: list,
                                  min_present_thresholds: list, min_present_threshold2: int = None):
        """
        Compare thresholds in batched based on 2 subsets
        :param sample1: Dict key (gene1,gene2), value similarities list
        :param sample2: as for sample2
        :param similarity_thresholds: Apply these thresholds for sample1 and 2
        :param min_present_thresholds: Apply these thresholds for sample1 and 2
        :param min_present_threshold2: If specified apply this single threshold to sample2 instead of
        min_present_thresholds
        :return: Data frame with mock F values (as specified in f_value function) if min_present_threshold2=None, else
            return precision of sample1 against sample2 (as ground truth).
            Also includes parameters and n of retained genes and pairs.
        """
        summary = []
        for threshold in similarity_thresholds:
            if min_present_threshold2 is not None:
                filtered2 = NeighbourCalculator.filter_similarities_batched(results=sample2,
                                                                            similarity_threshold=threshold,
                                                                            min_present=min_present_threshold2)
                genes2 = set(gene for pair in filtered2.keys() for gene in pair)
            for min_present in min_present_thresholds:
                filtered1 = NeighbourCalculator.filter_similarities_batched(results=sample1,
                                                                            similarity_threshold=threshold,
                                                                            min_present=min_present)
                n_pairs1 = len(filtered1)
                genes1 = set(gene for pair in filtered1.keys() for gene in pair)
                n_genes1 = len(genes1)
                if min_present_threshold2 is None:
                    filtered2 = NeighbourCalculator.filter_similarities_batched(results=sample2,
                                                                                similarity_threshold=threshold,
                                                                                min_present=min_present)
                    genes2 = set(gene for pair in filtered2.keys() for gene in pair)
                    n_pairs2 = len(filtered2)
                    n_genes2 = len(genes2)

                    f_val_genes = NeighbourCalculator.f_value(set1=genes1, set2=genes2)
                    f_val_pairs = NeighbourCalculator.f_value(set(filtered1.keys()), set(filtered2.keys()))
                    summary.append(
                        {'threshold': threshold, 'min_present': min_present, 'n_pairs1': n_pairs1, 'n_genes1': n_genes1,
                         'n_pairs2': n_pairs2, 'n_genes2': n_genes2, 'f_val_genes': f_val_genes,
                         'f_val_pairs': f_val_pairs})
                else:
                    n_genes1 = len(genes1)
                    if n_genes1 < 1:
                        precision = float('NaN')
                    else:
                        precision = len(genes1 & genes2) / n_genes1
                    summary.append(
                        {'threshold': threshold, 'min_present': min_present, 'n_pairs1': n_pairs1, 'n_genes1': n_genes1,
                         'precision': precision})
        return pd.DataFrame(summary)

    @staticmethod
    def plot_threshold_batched(summary: pd.DataFrame = None, sample1: dict = None, sample2: dict = None,
                               similarity_thresholds: list = None,
                               min_present_thresholds: list = None):
        """
        Select similarity and min present threshold for batched.
        :param summary: Prespecifiy summary from compare_threshold_batched
        :param sample1: Dict key (gene1,gene2), value similarities list
        :param sample2: as for sample2
        :param similarity_thresholds: Apply these thresholds for sample1 and 2
        :param min_present_thresholds: Apply these thresholds for sample1 and 2
        :return: summary (newly obtained or specified)
        """
        if summary is None and (sample1 is None or sample2 is None or similarity_thresholds is
                                None or min_present_thresholds is None):
            raise ValueError('Either summary or all other arguments must be given.')
        if summary is None:
            summary = NeighbourCalculator.compare_threshold_batched(sample1=sample1, sample2=sample2,
                                                                    similarity_thresholds=similarity_thresholds,
                                                                    min_present_thresholds=min_present_thresholds,
                                                                    min_present_threshold2=None)
        font = 9
        fig = plt.figure(
            figsize=(10, 10)
        )
        fig.tight_layout()
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.set_ylabel('F value genes/ Proportion of max retained genes', fontdict={'fontsize': font})
        ax1.set_xlabel('Min similarity', fontdict={'fontsize': font})
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.set_xlabel('Present in N', fontdict={'fontsize': font})
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.set_ylabel('F value pairs/ Proportion of max retained pairs', fontdict={'fontsize': font})
        ax3.set_xlabel('Min similarity', fontdict={'fontsize': font})
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.set_xlabel('Present in N', fontdict={'fontsize': font})
        ax1.tick_params(axis="both", labelsize=font)
        ax2.tick_params(axis="both", labelsize=font)
        ax3.tick_params(axis="both", labelsize=font)
        ax4.tick_params(axis="both", labelsize=font)

        max_genes_sum = max(summary['n_genes1']) + max(summary['n_genes2'])
        max_pairs_sum = max(summary['n_pairs1']) + max(summary['n_pairs2'])

        groupmp = summary.groupby('min_present')
        for group in groupmp:
            x_genes, y_genes = NeighbourCalculator.x_y_fval_plot(group=group, x_col='threshold',
                                                                 y_col_nom='f_val_genes',
                                                                 y_col_denom1='n_genes1', y_col_denom2='n_genes2',
                                                                 y_col_denom_denom=max_genes_sum)
            x_pairs, y_pairs = NeighbourCalculator.x_y_fval_plot(group=group, x_col='threshold',
                                                                 y_col_nom='f_val_pairs',
                                                                 y_col_denom1='n_pairs1', y_col_denom2='n_pairs2',
                                                                 y_col_denom_denom=max_pairs_sum)
            label = group[0]
            plot_line_scatter(x=x_genes, y=y_genes, ax=ax1, label=label)
            plot_line_scatter(x=x_pairs, y=y_pairs, ax=ax3, label=label)
        legend1 = ax1.legend(title='Present in N', prop={'size': font})
        legend3 = ax3.legend(title='Present in N', prop={'size': font})
        legend1.get_title().set_fontsize(font)
        legend3.get_title().set_fontsize(font)

        groupt = summary.groupby('threshold')
        for group in groupt:
            x_genes, y_genes = NeighbourCalculator.x_y_fval_plot(group=group, x_col='min_present',
                                                                 y_col_nom='f_val_genes',
                                                                 y_col_denom1='n_genes1', y_col_denom2='n_genes2',
                                                                 y_col_denom_denom=max_genes_sum)
            x_pairs, y_pairs = NeighbourCalculator.x_y_fval_plot(group=group, x_col='min_present',
                                                                 y_col_nom='f_val_pairs',
                                                                 y_col_denom1='n_pairs1', y_col_denom2='n_pairs2',
                                                                 y_col_denom_denom=max_pairs_sum)
            label = group[0]
            plot_line_scatter(x=x_genes, y=y_genes, ax=ax2, label=label)
            plot_line_scatter(x=x_pairs, y=y_pairs, ax=ax4, label=label)
        legend2 = ax2.legend(title='Min similarity', fontsize=font)
        legend4 = ax4.legend(title='Min similarity', fontsize=font)
        legend2.get_title().set_fontsize(font)
        legend4.get_title().set_fontsize(font)

        return summary

    @staticmethod
    def x_y_fval_plot(group: tuple, x_col, y_col_nom, y_col_denom1, y_col_denom2, y_col_denom_denom):
        """
        Helper for F val plotting in plot_threshold_batched
        Plot on y: y_col_nom / ((y_col_denom1 + y_col_denom2) / y_col_denom_denom)
        :param group: pd Grop as obtained by df.groupby()
        :param x_col: Plot on x, column name from group
        :param y_col_nom: For y , column name from group
        :param y_col_denom1: For y , column name from group
        :param y_col_denom2: For y , column name from group
        :param y_col_denom_denom: For y, float
        :return: values for x and y
        """
        x = group[1][x_col]
        y = group[1][y_col_nom] / ((group[1][y_col_denom1] + group[1][y_col_denom2]) / y_col_denom_denom)
        return x, y

    @staticmethod
    def plot_threshold_batched_sub_vs_all(summary, result_all, similarity_threshold, selected_min_present_sub):
        """
        Plot retained genes in batched subsets vs on all results. Plot for all min_present specified in summary and for
            2times this number for result_all
        :param summary: Summary from compare_threshold_batched
        :param result_all: Result from all replicates, key (gene1,gene2) value list of similarities
        :param similarity_threshold: Use this similarity from summary and apply it to result_all
        :param selected_min_present_sub: Compute N of retained genes at this threshold in both sub sets from summary,
            average it and plot a line at this N of genes
        """
        if similarity_threshold not in list(summary['threshold']):
            raise ValueError('Similarity threshold is not in summary')
        summary_at_sim_threshold = summary[summary['threshold'] == similarity_threshold]
        summary_selected = summary_at_sim_threshold[summary_at_sim_threshold['min_present'] == selected_min_present_sub]
        genes_subs = (summary_selected['n_genes1'].item() + summary_selected['n_genes2'].item()) / 2
        fig, ax = plt.subplots()
        ax.set_ylabel('Sub samples min present thresholds')
        ax.set_xlabel('N retained genes')
        ax.axvline(x=genes_subs)
        line1 = plot_line_scatter(summary_at_sim_threshold['n_genes1'], summary_at_sim_threshold['min_present'], ax,
                                  label='Subsample1', c='g')
        line2 = plot_line_scatter(summary_at_sim_threshold['n_genes2'], summary_at_sim_threshold['min_present'], ax,
                                  label='Subsample2', c='b')
        ax1 = ax.twinx()
        ax1.set_ylabel('Complete result min present thresholds')
        min_present_thresholds = list(summary_at_sim_threshold['min_present'] * 2)
        n_genes_all = []
        for min_present in min_present_thresholds:
            retained = NeighbourCalculator.filter_similarities_batched(result_all, similarity_threshold, min_present)
            n_genes_all.append(len(set(gene for pair in retained.keys() for gene in pair)))
        line3 = plot_line_scatter(n_genes_all, min_present_thresholds, ax1, label='Complete results', c='r')
        ax.set_xscale('log')
        fig.legend(handles=[line1, line2, line3])


def plot_line_scatter(x, y, ax: plt.axes, **linplot_kwargs):
    """
    Plot line with points (ax.plot and ax.scatter)
    :param x: x data
    :param y: y data
    :param ax: plot to
    :param linplot_kwargs: add to ax.plot
    :return: as returned from ax.plot
    """
    plotted = ax.plot(x, y, **linplot_kwargs)
    color = plotted[0].get_color()
    ax.scatter(x, y, color=color)
    return plotted[0]


# Source: https://matplotlib.org/3.1.1/gallery/ticks_and_spines/multiple_yaxis_with_spines.html
def pandas_multi_y_plot(data: pd.DataFrame, x_col, y_cols: list = None, adjust_right_border=None,
                        no_line: bool = False):
    """
    Plot line plot with scatter points with multiple y axes
    :param data:
    :param x_col: Col names from DF for x
    :param y_cols: Col names from DF for y, if None plot all except x
    :param adjust_right_border: Move plotting area to left to allow more space for y axes
    :param no_line: Plot only scatterplot pointsjupyter-notebook

    """
    # TODO Mean Not working properly: If no_line is false use mean of each x to plot a line. Removes na from mean.
    plot_mean = False
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
    groupped = data.groupby(x_col).mean()
    x_mean = groupped.index.values
    x = data[x_col]
    host.set_xlim(min(x), max(x))
    host.set_xlabel(x_col)

    # First axis
    y = data[y_cols[0]]
    color = colors[0]
    host.scatter(x, y, color=color)
    if not no_line:
        if plot_mean:
            x_line = x_mean
            y_line = groupped[y_cols[0]].values
        else:
            x_line = x
            y_line = y
        host.plot(x_line, y_line, color=color)
    host.set_ylim(min(y), max(y))
    host.set_ylabel(y_cols[0])
    host.yaxis.label.set_color(color)
    host.tick_params(axis='y', colors=color)

    for n in range(1, len(y_cols)):
        # Multiple y-axes
        y = data[y_cols[n]]
        color = colors[n % len(colors)]
        ax_new = host.twinx()
        ax_new.spines["right"].set_position(("axes", 1 + 0.2 * (n - 1)))
        ax_new.scatter(x, y, color=color)
        if not no_line:
            if plot_mean:
                x_line = x_mean
                y_line = groupped[y_cols[n]].values
            else:
                x_line = x
                y_line = y
            host.plot(x_line, y_line, color=color)
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

    def __init__(self, distance_matrix: np.array, gene_names: list, data: np.ndarray):
        self._gene_names_ordered = gene_names
        self._n_genes = len(gene_names)
        self._distance_matrix = distance_matrix
        self._data = data

    @classmethod
    def from_knn_result(cls, result: dict, genes: pd.DataFrame, threshold: float, inverse: bool, scale: str, log: bool):
        """
        Initialize class from  knn result - prepare data.
        :param result: Closest neighbours result.
        :param genes: Expression data
        :param threshold: Retain only genes with at least one neighbour with similarity at least as big as threshold.
        :param inverse: Distances calculated based on profile of gene1 and inverse profile of gene2 for each gene pair.
        :param scale: Scale expression data to common scale: 'minmax' from 0 to 1, 'mean0std1' to mean 0 and std 1
        :param log: Log transform data before scaling.
        :return:
        """
        distance_matrix, gene_names, data = Clustering.get_clustering_data(result=result, genes=genes,
                                                                           threshold=threshold, inverse=inverse,
                                                                           scale=scale, log=log)
        return cls(distance_matrix=distance_matrix, gene_names=gene_names, data=data)

    @staticmethod
    def get_clustering_data(result: dict, genes: pd.DataFrame, threshold: float, inverse: bool, scale: str = SCALING,
                            log: bool = LOG) -> tuple:
        """
        Prepare data for clustering
        :param result: Closest neighbours result.
        :param genes: Expression data
        :param threshold: Retain only genes with at least one neighbour with similarity at least as big as threshold.
        :param inverse: Distances calculated based on profile of gene1 and inverse profile of gene2 for each gene pair.
        :param scale: Scale expression data to common scale: 'minmax' from 0 to 1, 'mean0std1' to mean 0 and std 1
        :param log: Log transform data before scaling.
        :return: Distance matrix, gene names specifiing rows/columns in distance matrix, preprocessed expression data
        """
        index, query, gene_names = Clustering.get_genes(result=result, genes=genes, threshold=threshold,
                                                        inverse=inverse,
                                                        scale=scale, log=log)
        distance_matrix = Clustering.get_distances_cosine(index=index, query=query, inverse=inverse, scale=scale)
        return distance_matrix, gene_names, np.array(index)

    @staticmethod
    def get_genes(genes: pd.DataFrame, inverse: bool, result: dict = None, threshold: float = None,
                  scale: str = SCALING,
                  log: bool = LOG, return_query: bool = True) -> tuple:
        """
        Prepare gene data for distance calculation. Retain only specified genes based on result and its
        filtering params.
        :param result: Closest neighbours result. Used to select genes from all genes. If none use all genes from genes
            Data Frame.
        :param genes: Expression data
        :param threshold: Retain only genes with at least one neighbour with similarity at least as big as threshold.
            If none use the raw result.
        :param inverse: Distances calculated based on profiole of gene1 and inverse profile of gene2 for each gene pair.
        :param scale: Scale expression data to common scale: 'minmax' from 0 to 1, 'mean0std1' to mean 0 and std 1
        :param log: Log transform data before scaling.
        :param return_query: Should query be returned (or only index)
        :return: index,query - processed expression data. If inverse is False both of them are the same, else query is
            inverse profiles data.
        """
        if result is not None:
            if threshold is not None:
                result_filtered = NeighbourCalculator.filter_similarities(result, threshold)
            else:
                result_filtered = result
            if len(result_filtered) == 0:
                raise ValueError('All genes were filtered out. Choose lower threshold.')
            genes_filtered = set((gene for pair in result_filtered.keys() for gene in pair))
            genes_data = genes.loc[genes_filtered, :]
        else:
            genes_data = genes
        gene_names = list(genes_data.index)
        index, query = NeighbourCalculator.get_index_query(genes=genes_data, inverse=inverse, scale=scale, log=log)
        if return_query:
            return index, query, gene_names
        else:
            return index, gene_names

    @staticmethod
    def get_distances_cosine(index: np.ndarray, query: np.ndarray, inverse: bool, scale: str) -> np.ndarray:
        """
        Calculate distances (cosine) between expression profiles for each gene pair.
        If inverse was used on minmax scaling distances matrix would not be symmetrical, thus for each unique gene pair
        distance is calculated in both directions (eg. either gene in index/query) and averaged into final distance.
        :param index: Expression data for gene1 from pair
        :param query: Expression data for gene2 from pair
        :param inverse: Was index data inversed
        :param scale: Which scaling was used to prepare the data
        :return: Distance matrix
        """
        if index.shape != query.shape:
            raise ValueError('Index and query must be of same dimensions')
        n_genes = index.shape[0]
        distance_matrix = [[0] * n_genes for count in range(n_genes)]
        both_directions = False
        if inverse & (scale == 'minmax'):
            both_directions = True
        for i in range(0, n_genes - 1):
            for q in range(i + 1, n_genes):
                # TODO This might be quicker if pdist from scipy was used when inverse was not needed
                distance = calc_cosine(data1=index, data2=query, index1=i, index2=q, sim_dist=False,
                                       both_directions=both_directions)
                distance_matrix[i][q] = distance
                distance_matrix[q][i] = distance
        return np.array(distance_matrix)

    def get_distance_matrix(self) -> np.array:
        """
        :return: Distance matrix
        """
        return self._distance_matrix.copy()

    @abstractmethod
    def get_clusters(self, **splitting) -> np.ndarray:
        """
        Cluster memberships
        :param splitting: how to create clusters
        :return: List of cluster memberships over genes (cluster numbers in same order as gene names)
        """
        pass

    def cluster_sizes(self, splitting: dict = None, clusters=None) -> list:
        """
        Get sizes of clusters
        :param splitting: how to create clusters, needed if clusters is not give
        :param clusters: Predefined clusters, as returned by get_clusters
        :return: sizes
        """
        if clusters is None:
            clusters = list(self.get_clusters(**splitting))
        return list(Counter(clusters).values())

    def get_genes_by_clusters(self, splitting: dict = None, filter_genes: iter = None, clusters=None) -> dict:
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
        cluster_dict = dict((gene_set, []) for gene_set in set(clusters))
        for gene, cluster in zip(self._gene_names_ordered, clusters):
            if (filter_genes is None) or (gene in filter_genes):
                cluster_dict[cluster].append(gene)
        if filter_genes is not None:
            for cluster, gene_list in zip(list(cluster_dict.keys()), list(cluster_dict.values())):
                if len(gene_list) == 0:
                    del cluster_dict[cluster]
        return cluster_dict

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

    def plot_cluster_sizes(self, splitting: float = None, clusters=None):
        """
        Distribution of cluster sizes
        :param splitting: how to create clusters, needed if clusters is not give
        :param clusters: Predefined clusters, as returned by get_clusters
         """
        if clusters is None:
            clusters = self.get_clusters(**splitting)
        sizes = self.cluster_sizes(clusters=clusters)
        plt.hist(sizes, bins=100)
        plt.xlabel('Cluster size')
        plt.ylabel('Cluster count')

    def save_clusters(self, file: str, splitting=None, clusters=None):
        """
        Save gene names with corresponding cluster number in tsv.
        :param splitting: how to create clusters, needed if clusters is not give
        :param file: File name
        :param clusters: Predefined clusters, as returned by get_clusters
        """
        if clusters is None:
            clusters_named = self.get_clusters_by_genes(splitting=splitting)
        else:
            clusters_named = self.get_clusters_by_genes(clusters=clusters)
        clusters_named = OrderedDict(clusters_named)
        data = pd.DataFrame({'genes': list(clusters_named.keys()), 'cluster': list(clusters_named.values())})
        data.to_csv(file, sep='\t', index=False)


class HierarchicalClustering(Clustering):
    """
    Performs hierarchical clustering.
    """

    def __init__(self, distance_matrix: np.ndarray, gene_names: list, data: np.ndarray, ):
        """
        Performs clustering (Ward).
        """
        super().__init__(distance_matrix=distance_matrix, gene_names=gene_names, data=data)
        upper_index = np.triu_indices(self._n_genes, 1)
        distances = self._distance_matrix[upper_index]
        self._hcl = hc.ward(np.array(distances))

    def get_clusters(self, **splitting) -> np.ndarray:
        """
        Cluster memberships
        :param splitting: Tree cutting parameters from scipy clustering fcluster
        :return: List of cluster memberships over genes
        """
        return hc.fcluster(self._hcl, criterion='distance', **splitting)

    def plot_clustering(self, cutting_distance: float):
        """
        Plot dendrogram with coloured clusters
        :param cutting_distance: Distance to cut at
        """
        dendrogram(Z=self._hcl, color_threshold=cutting_distance, no_labels=True)
        plt.ylabel('Distance')


class DBSCANClustering(Clustering):

    def get_clusters(self, splitting) -> np.ndarray:
        """
        Cluster memberships
        :param splitting: sklearn DBSCAN clustering parameters, excluding metric and n_jobs
        :return: List of cluster memberships over genes
        """
        return DBSCAN(metric='precomputed', n_jobs=4, **splitting).fit_predict(self._distance_matrix)


class LouvainClustering(Clustering):

    def __init__(self, distance_matrix: np.ndarray, gene_names: list, data: np.ndarray, orange_graph: bool,
                 trimm: float = 0, closest: int = None):
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
        super().__init__(distance_matrix=distance_matrix, gene_names=gene_names, data=data)
        if orange_graph:
            self._graph = orange_louvain_graph.matrix_to_knn_graph(data=data, k_neighbors=closest, metric='cosine')
        else:
            if closest is None:
                if trimm < 0:
                    trimm = 0
                    warnings.warn('trimm must be non-negative. trimm was set to 0.', Warning)
                adjacency_matrix = 1 - self._distance_matrix
                adjacency_matrix[adjacency_matrix < trimm] = 0
                np.fill_diagonal(adjacency_matrix, 0)
            else:
                if closest < 1 or not isinstance(closest, int):
                    raise ValueError('Closest must be a positive int')
                adjacency_matrix = np.zeros((self._n_genes, self._n_genes))
                for row_idx in range(self._n_genes):
                    sorted_neighbours = np.argsort(self._distance_matrix[row_idx])
                    top_added = 0
                    idx_in_top = 0
                    while top_added < closest:
                        if idx_in_top > self._n_genes - 1:
                            raise ValueError('Too many specified neighbours to retain')
                        idx = sorted_neighbours[idx_in_top]
                        idx_in_top += 1
                        if idx != row_idx:
                            adjacency_matrix[row_idx][idx] = self._distance_matrix[row_idx][idx]
                            top_added += 1
            self._graph = nx.from_numpy_matrix(adjacency_matrix)

    @classmethod
    def from_orange_graph(cls, data: np.ndarray, gene_names: list, neighbours: int = 40):
        """
        Initialize graph with KNN calculation and Jaccard index from Orange.
        (Distance matrix does not need to be computed).
        :param data: Data used for KNN computation (e.g. already preprocessed), samples in rows, feature in columns
        :param gene_names: Names specifying what is in each row of data
        :param neighbours: How many neighbours to use
        """
        return cls(distance_matrix=None, gene_names=gene_names, data=data, orange_graph=True, closest=neighbours)

    def get_clusters(self, **splitting) -> np.ndarray:
        """
        Cluster memberships
        :param splitting: Parameters passed to community.best_partition clustering
        :return: List of cluster memberships over genes
        """
        clusters = louvain(graph=self._graph, **splitting)
        return np.array(list(OrderedDict(sorted(clusters.items())).values()))


class GaussianMixtureClustering(Clustering):
    """
    Can not be performed with inverse (oposite profile neighbours) - uses only self._data in fitting
    """

    def get_clusters(self, **splitting) -> np.ndarray:
        """
        Cluster memberships
        :param splitting: sklearn.mixture.GaussianMixture parameters
        :return: List of cluster memberships over genes
        """
        return GaussianMixture(**splitting).fit_predict(self._data)


class ClusterAnalyser:
    """
    Analyses individual clusters/gene lists
    """
    PATTERN_SAMPLE = 'Gene'
    PATTERN_PEAK = 'Peak'
    PATTERM_MASS_CENTRE = 'Mass_centre'
    PATTERN_N_ATLEAST = 'N_atleast'

    def __init__(self, genes: pd.DataFrame, conditions: pd.DataFrame, average_data_by,
                 split_data_by, matching, control: str, organism: int = 44689):
        """
        Data required to analyse clusters
        :param genes: Expression data, genes in rows, measurements in columns, dimensions G*M
        :param conditions: Description (columns) of each measurements (rows), dimensions M*D
        :param average_data_by: Conditions column name. When multiple replicates are present, average their data.
            This is done after splitting. Name of column in conditions DataFrame that has identical values for all
            replicates and will be used for splitting. Eg if conditions contains Strain, Replicate and Time and data is
            to be split by Strain and then average data across replicates in individual time points average_data_by
            would be Time
        :param split_data_by: Name of column in conditions DataFrame used for splitting the data
        :param matching: Which column in conditions matches column names in genes
        :param control: Which name from names loacted in split_data_by column will be used for pattern characteristics
        :param organism: Organism ID
        """
        self._names_entrez = name_genes_entrez(gene_names=genes.index, organism=organism, key_entrez=False)
        self._organism = organism
        self._average_by = average_data_by
        self._split_by = split_data_by
        self._data = self.preprocess_data_by_groups(genes=genes, conditions=conditions, split_by=split_data_by,
                                                    average_by=average_data_by, matching=matching)
        self._control = control
        self._pattern_characteristics = self.pattern_characteristics(data=self._data[control])

    @staticmethod
    def preprocess_data_by_groups(genes: pd.DataFrame, conditions: pd.DataFrame, split_by, average_by,
                                  matching) -> dict:
        """
        Split and then average data by groups
        :param genes: Expression data, genes in rows, measurements in columns, dimensions G*M
        :param conditions: Description (columns) of each measurements (rows), dimensions M*D
        :param split_by:
        :param average_by: Conditions column name. When multiple replicates are present, average their data.
            This is done after splitting. Name of column in conditions DataFrame that has identical values for all
            replicates and will be used for splitting. Eg if conditions contains Strain, Replicate and Time and data is
            to be split by Strain and then average data across replicates in individual time points average_data_by
            would be Time
        :param matching: Which column in conditions matches column names in genes
        :return: Dict with keys being values from split_by column and values being data, averaged by average_by
        """
        merged = ClusterAnalyser.merge_genes_conditions(genes=genes, conditions=conditions, matching=matching)
        splitted = ClusterAnalyser.split_data(data=merged, split_by=split_by)

        data_processed = {}
        for split, data in splitted.items():
            averaged = ClusterAnalyser.average_data(data=data, average_by=average_by)
            data_processed[split] = averaged.T
        return data_processed

    @staticmethod
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
        return pd.concat([genes.T, conditions], axis=1, sort=True)

    @staticmethod
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

    # Eg. by time if want to average for all replicates as applied after splitting
    @staticmethod
    def average_data(data: pd.DataFrame, average_by: str) -> pd.DataFrame:
        """
        Average data by column
        :param data: Data to be averaged by values of a column (eg. multiple rows are averaged)
        :param average_by: Column name for averaging
        :return: Data Frame with index being average_by column values and retaining only  features that can be averaged
            (numeric)
        """
        return data.groupby(average_by).mean()

    @staticmethod
    def pattern_characteristics(data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute expression pattern characteristics for each row across all features  (e.g. pattern, must be numeric).
        Characteristics are: peak (which column has max value), mass center (at which column would be mass centre if
        values are mass measurements from continuous distribution), N features (that have value equal to or greater
        than value at the middle of min and max value across features).
        :param data: genes in rows, pattern through columns, values must be numeric,
            column names must be numeric, used for mass_centre and peak calculation.
            Values are sorted by column names (ascending) to get pattern for mass center.
        :return: Data Frame with rows being samples and columns specifying sample, mass_centre, peak, and N features
            with value at least as big as above described threshold
        """
        pattern_data = []
        for row in data.iterrows():
            gene = row[1]
            mass_centre = ClusterAnalyser.mass_centre(gene)
            peak = ClusterAnalyser.peak(gene)
            n_atleast = ClusterAnalyser.n_atleast(data=gene, ratio_from_below=0.5)
            pattern_data.append({ClusterAnalyser.PATTERN_SAMPLE: row[0],
                                 ClusterAnalyser.PATTERM_MASS_CENTRE: mass_centre,
                                 ClusterAnalyser.PATTERN_PEAK: peak,
                                 ClusterAnalyser.PATTERN_N_ATLEAST: n_atleast})
        return pd.DataFrame(pattern_data)

    @staticmethod
    def mass_centre(data: pd.Series) -> float:
        """
        Mass center of series which represents samples from continuous distribution.
        Masses between two x values (indcies) are computed as trapezoids and converted to cumulative mass.
        Half of total mass lies on an interval x1, x2 of cumulative mass function. Mass centre is obtained by
        determining  on which proportion of lower and upper cumulative mass (y1, y2) of the interval lies value equal to
        half of total mass. This proportion is used to obtain centre from x1, x2.
        :param data: Series with indices being x values of continuous function and values being y values/masses.
            Values and indices must be numeric
            Data is sorted ascendingly by index before calculation of mass center
        :return: Mass center
        """
        data = data.sort_index()
        x = list(data.index)
        y = data.values
        cumulative_masses = cumtrapz(y=y, x=x, initial=0)
        index2 = 0
        half = cumulative_masses.max() / 2
        center = None
        for mass in cumulative_masses:
            if mass > half:
                break
            elif mass == half:
                center = x[index2]
                break
            else:
                index2 += 1
        if center is None:
            mass1 = cumulative_masses[index2 - 1]
            mass2 = cumulative_masses[index2]
            x1 = x[index2 - 1]
            x2 = x[index2]
            proportion = (half - mass1) / (mass2 - mass1)
            center = (x2 - x1) * proportion + x1
        return center

    @staticmethod
    def peak(data: pd.Series):
        """
        Where (index) lies max value of series
        :param data: Series with numerical values
        :return: peak location (index name)
        """
        return data.sort_values().index[data.shape[0] - 1]

    @staticmethod
    def n_atleast(data: pd.Series, ratio_from_below: float) -> int:
        """
        N of values greater or equal to threshold. Threshold is between min and max value based on ratio_from_bellow:
        threshold  = (max - min) * ratio_from_below + min
        :param data: numerical series
        :param ratio_from_below: Where between min and max to put threshold, as specified above
        :return: N of values equal or greater than threshold
        """
        threshold = (data.max() - data.min()) * ratio_from_below + data.min()
        return data[data >= threshold].shape[0]

    def cluster_enrichment(self, gene_names: list, **enrichment_args):
        """
        Is cluster enriched for ontology terms
        :param gene_names: Gene names
        :param enrichment_args: Passed to GO_enrichment
        :return: As returned by GO_enrichment
        """
        entrez_ids = self.get_entrez_from_storage(gene_names=gene_names)
        enriched = GO_enrichment(entrez_ids=entrez_ids, organism=self._organism, **enrichment_args)
        return enriched

    def get_entrez_from_storage(self, gene_names: list) -> list:
        """
        Convert gene names to Entrez IDs as stored in ClusterAnalyser instance
        (based on gene names and organism specified in init).
        If EntrezID is not present ignore the name
        :param gene_names: Gene names
        :return: Entrez IDs
        """
        entrez_ids = []
        for name in gene_names:
            if name in self._names_entrez.keys():
                entrez_ids.append(self._names_entrez[name])
        return entrez_ids

    def plot_profiles(self, gene_names: list, fig=None, rows: int = 1, row: int = 1, row_label: str = None):
        """
        Plot profiles of genes. Done separately for each value of split_data_by column, as specified on init.
        Plots the graphs in one row. Plots profiles in gray and adds mean profile in black.
        :param gene_names: To obtain expression profiles from data given on init (genes in rows).
        :param fig: Plots can be added to existing figure. If None a new figure is created.
        :param rows: Specify N of subplot rows in figure
        :param row: Which row to use for plotting
        :param row_label: How to name the row of subplots. If None no name is added.
        :return:
        """
        if fig is None:
            fig = plt.figure()
        n_subplots = len(self._data)
        fig.set_size_inches(n_subplots * 2, rows * 4)
        subplots = []
        subplot_counter = 1
        for data_name, data_sub in self._data.items():
            ax = fig.add_subplot(rows, n_subplots, n_subplots * (row - 1) + subplot_counter)
            if row_label is not None and subplot_counter == 1:
                ax.set_ylabel(row_label, rotation=0)
            subplot_counter += 1
            subplots.append(ax)
            ax.set_title(data_name, fontdict={'fontsize': 8})
            ax.tick_params(axis="x", labelsize=8)
            ax.tick_params(axis="y", labelsize=6)
            # If y axis ticks are to be removed
            # ax.tick_params(axis='y', which='both', length=0)
            # plt.setp(ax.get_yticklabels(), visible=False)
            data_genes = data_sub.loc[gene_names, :]
            x = data_genes.columns
            ax.set_xlim([min(x), max(x)])
            y_mean = log_transform_series(data_genes.mean())
            for gene_data in data_genes.iterrows():
                y = log_transform_series(gene_data[1])
                ax.plot(x, y, alpha=0.3, c='gray')
            ax.plot(x, y_mean, c='black')
        return fig, subplots


def log_transform_series(series) -> np.ndarray:
    """
    Transform series with log10(value+1)
    :param series: data
    :return: Log transformed
    """
    return np.log10(series.to_numpy() + 1)


class ClusteringAnalyser:
    """
    Analyse clustering result.
    """

    def __init__(self, gene_names: list, organism: int = 44689, max_set_size: int = 100, min_set_size: int = 2,
                 cluster_analyser: ClusterAnalyser = None):
        """
        :param gene_names: list of gene names (leafs in clustering) for which GO/KEGG annotations will be computed
        :param organism: Organism ID
        :param max_set_size: Max GO/KEGG set size to include in evaluation of clusters
        :param min_set_size: Min GO/KEGG set size to include in evaluation of clusters
        :param cluster_analyser: ClusterAnalyser to use for analysis of individual clusters
        """
        self._organism = organism
        self._max_set_size = max_set_size
        self._min_set_size = min_set_size
        self._annotation_dict = dict((gene_set, None) for gene_set in list_all(organism=str(self._organism)))

        self._entrez_names = name_genes_entrez(gene_names=gene_names, organism=self._organism, key_entrez=True)
        self.cluster_analyser = cluster_analyser

    def available_annotations(self) -> list:
        """
        Which gene set types are available for this organism (e.g. GO/KEGG gene sets)
        :return: Available gene sets
        """
        return list(self._annotation_dict.keys())

    @staticmethod
    def silhouette(clustering: Clustering, splitting: dict) -> float:
        """
        Calculate mean Silhouette Coefficient over clusters  (e.g. sklearn.metrics.silhouette_score)
        :param clustering: The clustering object
        :param splitting: Parameters to use on clustering object's method get_clusters to obtain clusters
        :return: mean Silhouette Coefficient. If there is a single cluster or each
         element is in separate cluster None is returned
        """
        matrix = clustering.get_distance_matrix()
        clusters = clustering.get_clusters(splitting=splitting)
        if 1 < len(set(clusters)) < len(clusters):
            return silhouette_score(matrix, clusters, metric='precomputed')
        else:
            return None

    # Sources: https://stackoverflow.com/questions/35709562/how-to-calculate-clustering-entropy-a-working-example-or-software-code
    # compute_domain_entropy_by_domain_source: https://github.com/DRL/kinfin/blob/master/src/kinfin.py
    #
    # Use only genes (leafs) with annotations
    # H for cluster:
    # p=n_in_set/n_all_annotated_sets ; n_all_annotated_sets - n of all annotations for this cluster,
    #   can be greater than number of all genes wit6h annotations
    # H for single cluster: -sum_over_sets(p*log2(p))
    # H for clustering: sum(entropy*member_genes/all_genes)
    # Probably better to use annotation_ratio
    def annotation_entropy(self, clustering: Clustering, splitting: dict, ontology: tuple) -> float:
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

    def annotation_ratio(self, clustering: Clustering, splitting: dict, ontology: tuple) -> float:
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
        annotation_dict, clusters = self.init_annotation_evaluation(clustering=clustering, splitting=splitting,
                                                                    ontology=ontology)
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

    def init_annotation_evaluation(self, clustering: Clustering, splitting: dict, ontology: tuple):
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

    def plot_clustering_metrics(self, clustering: Clustering, splittings: list, x_labs: list = None,
                                ontology: tuple = ('KEGG', 'Pathways')):
        """
        Analyse different values of splitting parameter used on clustering.
        Plot silhouette values, median cluster size and average cluster-size weighted maximal annotation ratio.
        :param clustering: Clustering object
        Parameter to use on clustering object's method get_clusters to obtain clusters
            (e.g. N of clusters for hierarchical clustering)
        :param splittings: list of parameter dictionaries for clustering
        :param x_labs: Labels to put on plot x axis, if None uses first value from each splittings dict
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
            silhouettes.append(ClusteringAnalyser.silhouette(clustering=clustering, splitting=splitting))
            ratios.append(self.annotation_ratio(clustering=clustering, splitting=splitting, ontology=ontology))
        if x_labs is None:
            x_labs = [list(splitting.values())[0] for splitting in splittings]
        data = pd.DataFrame({'Splitting parameter': x_labs, 'Mean silhouette values': silhouettes,
                             'Mean max annotation ratio': ratios, 'Median cluster size': median_sizes,
                             'N clusters': n_clusters})

        pandas_multi_y_plot(data=data, x_col='Splitting parameter', y_cols=None, adjust_right_border=0.35)
        return data

    def analyse_clustering(self, clustering: Clustering, tsne_data: dict, tsne_plot: dict = None,
                           clusters: np.ndarray = None, splitting: dict = None) -> tuple:
        """
        Analyse specific clustering result. Plots tsne with coloured clusters, plots expression profiles of genes in
        each cluster and returns DataFrame with cluster size and gene ontology enrichment and
        cluster number for each gene
        :param clustering: Clustering object
        :param tsne_data: tsne data as returned by make_tsne_data
        :param tsne_plot: tsne plotting params
        :param clusters: Pre-specified clusters as returned from Clustering.get_clusters
        :param splitting: If clusters=None use this to obtain clusters from Clustering
        :return: tuple of: DataFrame with cluster size and gene onthology enrichment; dict with samples as keys and
            cluster numbers as values
        """
        if splitting is None and clusters is None:
            raise ValueError('Either splitting or clusters must be given')
        if clusters is None:
            clusters = clustering.get_clusters(splitting=splitting)
        genes_by_cluster = clustering.get_genes_by_clusters(clusters=clusters)
        clusters_by_genes = clustering.get_clusters_by_genes(clusters=clusters)
        tsne_clusters = {'classes': clusters_by_genes}
        plot_tsne(**{**tsne_data, 'legend': True, 'plotting_params': tsne_plot, **tsne_clusters})
        data = []
        fig = plt.figure()
        subsets = len(genes_by_cluster)
        subset_count = 1
        for cluster, members in genes_by_cluster.items():
            data_cluster = dict()
            data_cluster['cluster'] = cluster
            data_cluster['size'] = len(members)
            enriched = self.cluster_analyser.cluster_enrichment(gene_names=members)
            enriched = self.parse_enrichment_dict(dictionary=enriched)
            data_cluster['enriched FDR'] = enriched
            data.append(data_cluster)
            self.cluster_analyser.plot_profiles(gene_names=members, fig=fig, rows=subsets, row=subset_count,
                                                row_label=str(cluster))
            subset_count += 1
        return pd.DataFrame(data), clusters_by_genes

    @staticmethod
    def parse_enrichment_dict(dictionary, sep_item: str = '\n'):
        """
        Convert enrichment dictionary into string. Each item is added to the string, followed by sep_item.
        Key and value are separated by ": ".  Value is rounded to 3 decimal places in scientific notation.
        :param dictionary: Enrichment dictionary
        :param sep_item: How to separate items
        :return: String of items
        """
        parsed = ''
        for k, v in dictionary.items():
            parsed += k + ': ' + "{:.3E}".format(v) + sep_item
        return parsed


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


# For scaled data: perplexities_range=[5,100],exaggerations=[17,1.6],momentums=[0.6,0.97]
# For unscaled data: perplexities_range: list = [30, 100], exaggerations: list = [12, 1.5], momentums: list = [0.6, 0.9]
def make_tsne(data: pd.DataFrame, perplexities_range: list = [5, 100], exaggerations: list = [17, 1.6],
              momentums: list = [0.6, 0.97], random_state=0) -> ot.TSNEEmbedding:
    """
    Make tsne embedding. Uses openTSNE Multiscale followed by optimizations. Each optimization has exaggeration and
    momentum parameter - these are used sequentially from exaggerations and momenutms lists, which must be of same
    lengths. There are as many optimizations as are lengths of optimization parameter lists.
    :param data: Samples in rows,features in columns. Must be numeric
    :param perplexities_range: Used for openTSNE.affinity.Multiscale
    :param exaggerations: List of exaggeration parameters for sequential optimizations
    :param momentums: List of momentum parameters for sequential optimizations
    :param random_state: random state
    :return: Embedding: functions as list of lists, where 1st object in nested list is x position and 2nd is y.
        There is one nested list for each sample
    """
    if len(exaggerations) != len(momentums):
        raise ValueError('Exagerrations and momenutms list lengths must match')
    affinities_multiscale_mixture = ot.affinity.Multiscale(data, perplexities=perplexities_range,
                                                           metric="cosine", n_jobs=8, random_state=random_state)
    init = ot.initialization.pca(data)
    embedding = ot.TSNEEmbedding(init, affinities_multiscale_mixture, negative_gradient_method="fft", n_jobs=8)
    for exaggeration, momentum in zip(exaggerations, momentums):
        embedding = embedding.optimize(n_iter=250, exaggeration=exaggeration, momentum=momentum)
    return embedding


def plot_tsne(tsne: ot.TSNEEmbedding, classes: dict = None, names: list = None, legend: bool = False,
              plotting_params: dict = {'s': 1}):
    """
    Plot tsne embedding
    :param tsne: Embedding, as returned by make_tsne
    :param classes: If not None colour each item in tSNE embedding by class.
    Keys: names matching names of tSNE embedding, values: class
    :param names: List of names for items in tSNE embedding
    :param legend: Should legend be added
    :param plotting_params: plt.scatter parameters
    :return:
    """
    x = [x[0] for x in tsne]
    y = [x[1] for x in tsne]
    if classes is None:
        plt.scatter(x, y, alpha=0.5, **plotting_params)
    else:
        if names is not None and isinstance(classes, dict):
            classes_extended = []
            for name in names:
                if name in classes.keys():
                    classes_extended.append(classes[name])
                else:
                    classes_extended.append('NaN')
            classes = classes_extended
        if len(classes) != len(tsne):
            raise ValueError('Len classes must match len tsne or classes must be dict and names must be provided')
        class_names = set(classes)
        all_colours = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                       '#bcf60c',
                       '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000',
                       '#ffd8b1',
                       '#000075', '#808080', '#000000']
        all_colours = all_colours * (len(class_names) // len(all_colours) + 1)
        selected_colours = random.sample(all_colours, len(class_names))
        # colour_idx = range(len(class_names))
        colour_dict = dict(zip(class_names, selected_colours))
        class_dict = dict((class_name, {'x': [], 'y': [], 'c': []}) for class_name in class_names)
        for x_pos, y_pos, class_name in zip(x, y, classes):
            class_dict[class_name]['x'].append(x_pos)
            class_dict[class_name]['y'].append(y_pos)
            class_dict[class_name]['c'].append(colour_dict[class_name])

        fig = plt.figure()
        ax = plt.subplot(111)

        for class_name, data in class_dict.items():
            if isinstance(list(plotting_params.values())[0], dict):
                plotting_param = plotting_params[class_name]
            else:
                plotting_param = plotting_params
            ax.scatter(data['x'], data['y'], c=data['c'], label=class_name, **plotting_param)
        if legend:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            legend = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            for handle in legend.legendHandles:
                handle._sizes = [10]
                handle.set_alpha(1)


def make_tsne_data(tsne, names):
    """
    Put together tsne embedding and corresponding item names
    :param tsne: embedding
    :param names: item names, ordered as embedding items
    :return: Dict with tsne and names items
    """
    return {'tsne': tsne, 'names': names}


def get_orange_result(result: pd.DataFrame = None, threshold: float = None, genes: pd.DataFrame = None, scale=SCALING,
                      log=LOG):
    """
    Convert result from Neighbour calculator to df of genes for orange - DF specifiing genes from retained pairs.
    Obtain result if not specified.
    :param result: Neighbour calculator result (keys (gene1, gene2), values similarity). If None obtain new result
        from NeighbourCalculator.neighbours with n_neighbours=2, inverse=False, batches=None and specified params (below)
    :param threshold: Remove pairs with similarity below threshold
    :param genes: Expression data, genes in rows, measurements in columns, G*M
    :param scale: Used in NeighbourCalculator.neighbours, scale data before neighbour calculation
    :param log: Used in NeighbourCalculator.neighbours, log transform data before neighbour calculation
    :return: DF with single column 'Gene' specifying set of all genes from retained pairs
    """
    if result is None:
        if genes is None:
            raise ValueError('If result is none genes must be specified.')
        neighbour_calculator = NeighbourCalculator(genes)
        result = neighbour_calculator.neighbours(n_neighbours=2, inverse=False, scale=scale, log=log, batches=None)
    if threshold is not None:
        result = NeighbourCalculator.filter_similarities(result, similarity_threshold=threshold)
    genes = set([gene for pair in result.keys() for gene in pair])
    return pd.DataFrame(genes, columns=['Gene'])


def preprocess_for_orange(genes: pd.DataFrame, conditions: pd.DataFrame, split_by, average_by,
                          matching, group: str, scale: str = SCALING, log: bool = LOG) -> tuple:
    """
    Get preprocessed data (scaled, transformed), data splitted and averaged, and expression pattern characteristics
    :param genes: Expression data, genes in rows, measurements in columns, dimensions G*M
    :param conditions: Specifies what each measurment name means, dismensions M*D, where D are metadata columns
    :param split_by: By which column from conditions should expression data be split
    :param average_by: By which column from conditions should expression data be averaged
    :param matching: Which column from conditions matches genes column names
    :param group: Which name from split_by column to use for pattern calculation, done on split averaged data
    :param scale: How to scale data  for  returning of preprocessed data
    :param log: Use log transformation  for returning of preprocessed data
    :return: preprocessed data, splitted and averaged data, expression pattern characteristics data
    """
    genes_scaled = get_orange_scaled(genes=genes, scale=scale, log=log)
    genes_averaged = get_orange_averaged(genes=genes, conditions=conditions,
                                         split_by=split_by, average_by=average_by, matching=matching)
    patterns = get_orange_pattern(genes_averaged=genes_averaged, group=group)
    return genes_scaled, genes_averaged, patterns


def get_orange_scaled(genes: pd.DataFrame, scale: str = SCALING, log: bool = LOG) -> pd.DataFrame:
    """
    Get preprocessed expression data for orange - scale and log transform.
    :param genes: Expression data, genes in rows, measurments in columns
    :param scale: How to scale data  for neighbour calculation and returning
    :param log: Log transform data   for neighbour calculation and returning
    :return: Preprocessed genes with close neighbours
    """
    gene_names = list(genes.index)
    index, query = NeighbourCalculator.get_index_query(genes=genes, inverse=False, scale=scale, log=log)
    return pd.DataFrame(index, index=gene_names, columns=genes.columns)


def get_orange_averaged(genes: pd.DataFrame, conditions: pd.DataFrame, split_by, average_by, matching):
    """
    Split expression data by groups and average it by replicates within groups.
    :param genes: Expression data, genes in rows, measurements in columns, dimensions G*M
    :param conditions: Specifies what each measurment name means, dismensions M*D, where D are metadata columns
    :param split_by: By which column from conditions should expression data be split
    :param average_by: By which column from conditions should expression data be averaged
    :param matching: Which column from conditions matches genes column names
    :return: Data frame with rows as genes and column names being values of average_by column and additional row named
    Group, denoting from which splitting group the averaged data comes.
    """
    by_strain = ClusterAnalyser.preprocess_data_by_groups(genes=genes, conditions=conditions, split_by=split_by,
                                                          average_by=average_by, matching=matching)
    strains_data = []
    for strain, data in by_strain.items():
        strain_data = pd.DataFrame({'Group': [strain] * data.shape[1]}, index=data.columns).T.append(data)
        strains_data.append(strain_data)
    return pd.concat(strains_data, axis=1)


def get_orange_pattern(genes_averaged: pd.DataFrame, group: str) -> pd.DataFrame:
    """

    :param genes_averaged: Data as returned from  get_orange_averaged
    :param group: Which group from Group row in genes_averaged to use for determination of expression pattern
    :return: Data frame as returned from ClusterAnalyser.pattern_characteristics
    """
    genes_strain = genes_averaged.T[genes_averaged.T['Group'] == group].T.drop('Group')
    return ClusterAnalyser.pattern_characteristics(data=genes_strain)


class NeighbourhoodParser:

    # Set of methods for parsing neighbourhoods

    @staticmethod
    def add_hub_to_neigh(neighbourhoods: dict):
        """
        From dict where hub is key and neighbour are values make neighbourhods with hub+neighbours as list of sets.
        Returns new object.
        :param neighbourhoods: Dict key is hub, values are neighbours
        :return: list of sets
        """
        neighbourhoods_merged = list()
        for hub, neighbourhood in neighbourhoods.items():
            neighbourhood_new = neighbourhood.copy()
            neighbourhood_new.append(hub)
            neighbourhoods_merged.append(set(neighbourhood_new))
        return neighbourhoods_merged

    @staticmethod
    def remove_repeated(neighbourhoods: list):
        """
        Remove repeated neighbourhoods (sets) from list of sets. Returns new object.
        :param neighbourhoods: list of sets
        """
        neighbourhoods_merged = []
        for neighbourhood in neighbourhoods:
            if not isinstance(neighbourhood, set):
                neighbourhood = set(neighbourhood)
            if neighbourhood not in neighbourhoods_merged:
                neighbourhoods_merged.append(neighbourhood)
        return neighbourhoods_merged

    @staticmethod
    def remove_contained(neighbourhoods: list):
        """
        Remove neighbourhoods contained in other neighbourhoods. Return new object
        :param neighbourhoods:  list of sets (neighbourhoods)
        """
        neighbourhoods_merged = neighbourhoods.copy()
        to_remove = set()
        for idx1 in range(len(neighbourhoods_merged) - 1):
            for idx2 in range(idx1 + 1, len(neighbourhoods_merged)):
                genes1 = neighbourhoods_merged[idx1]
                genes2 = neighbourhoods_merged[idx2]
                if genes1.issubset(genes2) and genes2.issubset(genes1):
                    raise ValueError('neighbourhoods are not unique')
                elif genes1.issubset(genes2):
                    to_remove.add(tuple(neighbourhoods_merged[idx1]))
                elif genes2.issubset(genes1):
                    to_remove.add(tuple(neighbourhoods_merged[idx2]))
        for sub in to_remove:
            neighbourhoods_merged.remove(set(sub))
        return neighbourhoods_merged

    @staticmethod
    def neighbourhood_distances(neighbourhoods, measure: str, genes_dist: pd.DataFrame = None):
        """
        Calculate distance between neighbourhoods.
        :param neighbourhoods: List of sets, gene names in sets.
        Can be dict with keys as neighbourhood names and values neighbourhoods
        :param measure:Distance : jaccard, percent_shared_smaller, avg_dist.
        To get distance from jaccard and percent_shared_smaller uses 1 - sim_metric (as max is 1).
        :param genes_dist: Must be distance (e.g. for cosine 1-genes_cosine). Index must match gene names in
        neighbourhoods
        :return: 1st element is Upper triangular matrix without diagonal in shape of 1D array. 2nd dict with node_id as K
        and neighbourhood as V. 3th (optional) If neighbourhoods was dict also returns
         node id (K) -original group name (V) mapping in a dict
        """
        if genes_dist is None and measure == 'avg_dist':
            raise ValueError('If measure is avg_dist genes_dist must be given')
        dist_arr = []
        neigh_ids = range(len(neighbourhoods))
        neighbourhood_names = None
        if isinstance(neighbourhoods, dict):
            neighbourhood_names = list(neighbourhoods.keys())
            neighbourhoods = list(neighbourhoods.values())
        for idx1 in neigh_ids[:-1]:
            for idx2 in neigh_ids[idx1 + 1:]:
                genes1 = set(neighbourhoods[idx1])
                genes2 = set(neighbourhoods[idx2])
                if measure == 'jaccard':
                    dist = 1 - jaccard(genes1, genes2)
                elif measure == 'percent_shared_smaller':
                    intersection = len(genes1 & genes2)
                    dist = 1 - intersection / min(len(genes1), len(genes2))
                elif measure == 'avg_dist':
                    dist = genes_dist.loc[genes1, genes2].values.flatten().mean()
                dist_arr.append(dist)
        group_name_mapping = None
        if neighbourhood_names is not None:
            group_name_mapping = dict(zip(neigh_ids, neighbourhood_names))
        if group_name_mapping is None:
            return np.array(dist_arr), dict(zip(neigh_ids, neighbourhoods))
        else:
            return np.array(dist_arr), dict(zip(neigh_ids, neighbourhoods)), group_name_mapping

    @staticmethod
    def plot_size_distn_neighbourhoods(neighbourhoods: list):
        """
        Plot size of each neighbourhood in histogram
        :param neighbourhoods: List of sets
        """
        sizes = [len(neigh) for neigh in neighbourhoods]
        plt.hist(sizes, bins=max(sizes))
        plt.xlabel('Size of neighbourhood')
        plt.ylabel('N of neighbourhoods')
        # Min size = min neighbours+1 because itself/hub in neighbourhood. Max includes itself/hub as well.
        print('Smallest neighbourhood:', min(sizes), ', largest:', max(sizes))

    @staticmethod
    def merge_by_hc(hc_result, node_neighbourhoods: dict, genes_sims: pd.DataFrame, min_group_sim: float) -> dict:
        """
        Merge neighbourhoods based on hc and minimal similarity within merged group
        :param hc_result: HC to use for merging order
        :param node_neighbourhoods: Dict with keys matchin hc_result labels and values containing sets of genes
        :param genes_sims: DF with similarities for gene pairs, N*N for N genes, index and columns contain gene names
        :param min_group_sim: Minimal similarity within prospective merged group to still merge two groups
        :return: New set of groups. Keys: group names, values: gene sets
        """
        node_neighbourhoods = node_neighbourhoods.copy()

        tree = hc.to_tree(hc_result, rd=True)[1]

        for node in tree[len(node_neighbourhoods):]:
            id = node.get_id()
            id1 = node.get_left().get_id()
            id2 = node.get_right().get_id()
            # If not one of previous merges was not performed so similarity will definitely be too low
            if id1 in node_neighbourhoods.keys() and id2 in node_neighbourhoods.keys():
                genes1 = node_neighbourhoods[id1]
                genes2 = node_neighbourhoods[id2]
                min_sim = genes_sims.loc[genes1, genes2].min().min()
                if min_sim >= min_group_sim:
                    genes_new = genes1.union(genes2)
                    node_neighbourhoods[id] = genes_new
                    # Remove merged nodes
                    del node_neighbourhoods[id1]
                    del node_neighbourhoods[id2]
        return node_neighbourhoods

    @staticmethod
    def min_neighbourhood_similarity(neighbourhoods: list, genes_sims: pd.DataFrame) -> float:
        """
        Minimal similarity between any two genes within any of the groups
        :param neighbourhoods: list of sets with gene names
        :param genes_sims: DF with similarities for gene pairs, N*N for N genes, index and columns contain gene names
        :return: overal min similarity within any of the groups
        """
        min_group_sim = 1
        for group in neighbourhoods.values():
            min_sim = genes_sims.loc[group, group].values.flatten().min()
            if min_group_sim > min_sim:
                min_group_sim = min_sim
        return min_group_sim

    @staticmethod
    def parse_orange(neighbourhoods):
        orange_groups = []
        for id, group in neighbourhoods.items():
            for gene in group:
                orange_groups.append({'Gene': gene, 'Group': id})
        return pd.DataFrame(orange_groups)

    @staticmethod
    def count_same_group(clusters: pd.DataFrame) -> pd.DataFrame:
        """
        Count under how many conditions two genes are in the same group
        :param clusters: DF with rows being genes and columns different cluster conditions.
        :return: NrowDF*NrowDF data frame with counts for each pair of genes
        """
        similarity = pd.DataFrame(index=clusters.index, columns=clusters.index)
        for idx, name1 in enumerate(clusters.index[:-1]):
            for name2 in clusters.index[idx + 1:]:
                clusters1 = clusters.loc[name1, :]
                clusters2 = clusters.loc[name2, :]
                # This already ignores NaNs
                shared = sum(clusters1 == clusters2)
                similarity.loc[name1, name2] = shared
                similarity.loc[name2, name1] = shared
        return similarity

