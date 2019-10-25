import pandas as pd
import sklearn.preprocessing as pp
from pynndescent import NNDescent
import numpy as np
from statistics import mean


class NeighbourCalculator:

    MINMAX = 'minmax'
    MEANSTD = 'mean0std1'
    SCALES = [MINMAX, MEANSTD]

    def __init__(self, genes: pd.DataFrame, remove_zero: bool = True):
        """
        :param genes: Data frame of genes in rows and conditions in columns. Index is treated as gene names
        :param remove_zero: Remove genes that have all expression values 0.
            If batches are latter specified there may be all 0 within batch.
        """
        if remove_zero:
            genes = genes[(genes != 0).any(axis=1)]
        self._genes = genes

    def neighbours(self, n_neighbours: int, inverse: bool, scale: str, log: bool, batches: list = None) -> dict:
        """
        :param n_neighbours:
        :param inverse: Calculate most similar neighbours (False) or neighbours with inverse profile (True)
        :param scale: Scale expression by gene with 'minmax' (min=0, max=1) or 'mean0std1' (mean=0, std=1)
        :param log: Should expression data be log2 transformed
        :param batches: Should comparisons be made for each batch separately and the averaged.
            Batches should be a list of cluster group for each column (eg. length of batches is n columns of genes).
        :return: Dict with gene names as tupple keys (smaller by alphabet first) and
            values representing cosine similarity
        """
        if scale not in self.SCALES:
            raise ValueError('Scale must be:', self.SCALES)
        genes = self._genes
        if log:
            genes = np.log2(genes + 1)
        if batches is None:
            return self.calculate_neighbours(genes, n_neighbours, inverse, scale)
        else:
            batch_groups = set(batches)
            batches = np.array(batches)
            results = []
            for batch in batch_groups:
                genes_sub = genes.T[batches == batch].T
                result = self.calculate_neighbours(genes_sub, n_neighbours, inverse, scale)
                results.append(result)
            return results

    @classmethod
    def calculate_neighbours(cls, genes, n_neighbours: int, inverse: bool, scale: str) -> dict:
        """
        :param genes: Data frame as in init
        :param n_neighbours:
        :param inverse: Calculate most similar neighbours (False) or neighbours with inverse profile (True)
        :param scale: Scale expression by gene with 'minmax' (min=0, max=1) or 'mean0std1' (mean=0, std=1)
        :param log: Should expression data be log2 transformed
        :return: Dict with gene names as tupple keys (smaller by alphabet first) and
            values representing cosine similarity
        """
        genes_index, genes_query = cls.get_index_query(genes, inverse, scale)
        # Can set speed-quality trade-off, default is ok
        index = NNDescent(genes_index, metric='cosine', random_state=0)
        neighbours, distances = index.query(genes_query.tolist(), k=n_neighbours)
        return cls.parse_neighbours(neighbours, distances)

    @classmethod
    def get_index_query(cls, genes, inverse: bool, scale: str):
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
    def minmax_scale(genes):
        return pp.minmax_scale(genes, axis=1)

    @staticmethod
    def meanstd_scale(genes):
        return pp.scale(genes, axis=1)

    def parse_neighbours(self, neighbours, distances) -> dict:
        parsed = dict()
        for gene in range(distances.shape[0]):
            for neighbour in range(distances.shape[1]):
                distance = distances[gene, neighbour]
                # Because of rounding the similarity may be slightly above one and distance slightly below 0
                if distance < 0:
                    if round(distance, 4) != 0:
                        print('Odd cosine distance at', gene, neighbour, ':', distance)
                    distance = 0
                similarity = 1 - distance
                gene2 = neighbours[gene, neighbour]
                gene_name1 = self._genes.index[gene]
                gene_name2 = self._genes.index[gene2]
                if gene_name1 != gene_name2:
                    if gene_name2 > gene_name1:
                        parsed[(gene_name1, gene_name2)] = similarity
                    else:
                        parsed[(gene_name2, gene_name1)] = similarity
        return parsed

    @staticmethod
    # NO: (as filtering out): Merge by median as pair may not be present - add 0 similarity which could more strongly afect mean.
    def merge_results(results: list, similarity_threshold: float = 0, min_present: int = 1) -> dict:
        """
        Merges by mean of present.
        Adds 0 if the pair is absent (affects median).
        :param results: List of dictionaries with results from neighbours method
        :param similarity_threshold: Ignore similarity if it is below threshold
        :param min_present: Filter out if not present in at least that many results
        :return:
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
        return dict(filter(lambda elem: elem[1] >= similarity_threshold, results.items()))
