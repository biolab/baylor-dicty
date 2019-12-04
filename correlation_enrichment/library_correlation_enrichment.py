import random
from math import factorial, sqrt
from collections import OrderedDict
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from statistics import median
from statsmodels.stats.multitest import multipletests
from sklearn import preprocessing as pp
from scipy.stats import (spearmanr, pearsonr, norm)
import mpmath

mpmath.mp.dps = 500

from orangecontrib.bioinformatics.geneset.utils import (GeneSet, GeneSets)

import deR.enrichment_library as el

MAX_RANDOM_PAIRS = 10000

MEAN = 'mean'
MEDIAN = 'median'

SE_SCALING_POINTS = [3, 5, 7, 10, 20, 30, 50, 80]
PERMUTATIONS = 8000


class GeneSetDataCorrelation(el.GeneSetData):
    """
    Stores gene set with enrichment statistics
    """

    def __init__(self, gene_set: GeneSet):
        """
        :param gene_set: gene set for which data should be stored
        """
        super().__init__(gene_set=gene_set)
        self.mean = None
        self.median = None
        self.most_similar = None
        self.pattern_stdevs = None


class GeneSetPairData:
    """
    Stores pair of gene sets and their similarities
    """

    def __init__(self, gene_set_data1: GeneSetDataCorrelation, gene_set_data2: GeneSetDataCorrelation):
        """
        :param gene_set1: gene set 1 from the pair
        :param gene_set2: gene set 2 from the pair
        """
        self.gene_set_data1 = gene_set_data1
        self.gene_set_data2 = gene_set_data2
        self.mean_profile_similarity = None
        self.median_profile_similarity = None
        self.profile_changeability_similarity = None


class GeneExpression:
    """
    Stores a data frame with gene expression data.
    """

    def __init__(self, expression: pd.DataFrame, use_log: bool = True):
        """
        Automatically removes genes with all 0 expression
        :param expression: Expression data frame with genes as rows and sampling points as columns.
        Gene names (indices) must be Entrez IDs (integers in string format).
        Genes may have multiple measurement vectors (eg. from different replicates) - thus multiple
        rows may have same index.
        :param use_log: log2 transform data (e.g. log2(data+1))
        """
        self.check_numeric(expression)
        self.are_geneIDs(expression.index)

        # TODO add log and 0 filtering to tests
        expression = expression[(expression != 0).any(axis=1)]
        if use_log:
            expression = np.log2(expression + 1)
        self.expression = expression
        # If measurements for each gene for each replicate represent one point in space this represents number of points
        self.n_points = expression.shape[0]

    @staticmethod
    def check_numeric(expression: pd.DataFrame):
        """
        Is content of data frame numeric (pandas float/int).
        If data is not numeric raises exception.
        :param expression: data frame to be analysed
        """
        if not ((expression.dtypes == 'float64').all() or (expression.dtypes == 'int64').all()):
            raise ValueError('Expression data is not numeric.')

    @staticmethod
    def are_geneIDs(ids: iter):
        """
        Are elements strings that can be transformed into integers.
        If not raises exception.
        :param ids: Iterable object of elements
        """
        for i in ids:
            if not isinstance(i, str):
                raise ValueError('Gene names are not IDs. Gene IDs are of type str and can be converted to int.')
            else:
                try:
                    int(i)
                except ValueError:
                    raise ValueError(
                        'Gene names are not IDs. Gene IDs are of type str and can be converted to int.')

    def get_genes(self) -> set:
        """
        :return: all gene names
        """
        return set(self.expression.index)

    def get_genes_data_IDs(self, geneIDs: list) -> pd.DataFrame:
        """
        Data frame with only specified genes.
        All query IDs must be present in expression data.
        :param geneIDs: Specified genes, same as data index names.
        :return: Subset of original data.
        """
        if not all(gene in self.expression.index for gene in geneIDs):
            raise KeyError('Not all gene IDs are in expression data.')
        else:
            return self.expression.loc[geneIDs, :]

    def get_genes_data_index(self, index: int) -> np.ndarray:
        """
        Get data of single gene by numerical (row) index.
        :param index: The index of the row.
        :return: Expression data for that gene.
        """
        if self.n_points - 1 >= index:
            return np.array(self.expression.iloc[index, :])
        else:
            raise IndexError('Index must be below', self.n_points)


class SimilarityCalculator:
    """
    Calculates similarity between two vectors.
    """

    # Similarity method
    PEARSON = 'correlation_pearson'
    SPEARMAN = 'correlation_spearman'
    COSINE = 'cosine'
    SIMILARITIES = [PEARSON, SPEARMAN, COSINE]
    CORRELATIONS = [PEARSON, SPEARMAN]

    # Normalisation method
    M0SD1 = 'mean0std1'
    NORMALISATION = [M0SD1, None]

    def __init__(self, bidirectional: bool = True, similarity_type: str = COSINE, normalisation_type: str = None):
        """

        :param bidirectional: Return better of two possible similarities between profiles -
            eg. profile1 vs profile2 or profile1 vs inverse  profile2 (abs(correlation/cosine); as multiplied by -1)
        :param similarity_type: Similarity method options: 'correlation_pearson', 'correlation_spearman', 'cosine'
        :param normalisation_type: Normalisation method options, used only for cosine similarity:'mean0std1'
        or None.

        These types are used automatically when calling similarity method
        """
        if similarity_type not in self.SIMILARITIES:
            raise ValueError('Possible similarity_type:', self.SIMILARITIES)
        if normalisation_type not in self.NORMALISATION:
            raise ValueError('Possible normalisation_type:', self.NORMALISATION)

        self.bidirectional = bidirectional
        self.similarity_type = similarity_type
        self.normalisation_type = normalisation_type

    def similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Calculate similarity between two vectors
        :param vector1:
        :param vector2:
        :return: similarity
        """
        if self.similarity_type in self.CORRELATIONS:
            return self.correlation(vector1, vector2)
        elif self.similarity_type == self.COSINE:
            return self.cosine(vector1, vector2)

    def cosine(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Navigate cosine similarity calculation and reporting.
        :param vector1:
        :param vector2:
        :return: similarity
        """
        if self.normalisation_type is not None:
            vector1 = self.normalise(vector1)
            vector2 = self.normalise(vector2)
        similarity = self.calc_cosine(vector1, vector2)

        if self.bidirectional:
            similarity = abs(similarity)
        return similarity

    def correlation(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Navigate correlation (spearman or pearson) calculation.
        :param vector1:
        :param vector2:
        :return: similarity
        """
        if self.similarity_type == self.SPEARMAN:
            similarity = self.calc_spearman(vector1, vector2)
        elif self.similarity_type == self.PEARSON:
            similarity = self.calc_pearson(vector1, vector2)
        else:
            raise NotImplementedError('Valid similarity types are', self.SPEARMAN, 'and', self.PEARSON)
        if self.bidirectional:
            similarity = abs(similarity)
        return similarity

    def normalise(self, vector: np.ndarray) -> np.ndarray:
        """
        Normalise a vector
        :param vector:
        :return: normalised vector
        """
        if self.normalisation_type == self.M0SD1:
            return pp.scale(vector)

    @staticmethod
    def calc_cosine(vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Calculate cosine similarity
        :param vector1:
        :param vector2:
        :return: similarity
        """
        return np.inner(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

    @staticmethod
    def calc_spearman(vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
            Calculate spearman correlation
            :param vector1:
            :param vector2:
            :return: correlation
        """
        return spearmanr(vector1, vector2)[0]

    @staticmethod
    def calc_pearson(vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
            Calculate pearson correlation
            :param vector1:
            :param vector2:
            :return: correlation
        """
        return pearsonr(vector1, vector2)[0]


class SimilarityCalculatorNavigator:
    """
    Base class for navigation of similarity calculation between specified genes.
    """

    def __init__(self, expression_data: GeneExpression, calculator: SimilarityCalculator, rm_outliers: bool = True):
        """
        :param expression_data:  Data for all genes
        :param calculator: SimilarityCalculator used in all calculations
        :param rm_outliers: should outliers be removed before similarity statistics calculation
        """
        self.expression_data = expression_data
        self.calculator = calculator
        self.rm_outliers = rm_outliers

    @staticmethod
    def remove_outliers(measurments):
        """
        Remove outliers by quantile method (1.5*IQR above or below Q3 or Q1, respectively).
        :param measurments: similarities in list or in dict with similarities with values
        :return:filtered measurments without outliers
        """
        data = None
        if isinstance(measurments, dict):
            data = measurments
            measurments = list(measurments.values())
        measurments = np.array(measurments)
        upper_quartile = np.percentile(measurments, 75)
        lower_quartile = np.percentile(measurments, 25)
        IQR_scaled = (upper_quartile - lower_quartile) * 1.5
        quartile_set = (lower_quartile - IQR_scaled, upper_quartile + IQR_scaled)
        if isinstance(data, dict):
            retained = dict()
            for key, point in data.items():
                if quartile_set[0] <= point <= quartile_set[1]:
                    retained[key] = point
        else:
            retained = []
            for point in measurments:
                if quartile_set[0] <= point <= quartile_set[1]:
                    retained.append(point)
        return retained


class RandomSimilarityCalculatorNavigator(SimilarityCalculatorNavigator):
    """
    Navigate similarity calculation between random points.
    """

    def __init__(self, expression_data: GeneExpression, calculator: SimilarityCalculator, random_seed: int = None,
                 rm_outliers: bool = True):
        """
        :param expression_data: Data for all genes
        :param calculator: SimilarityCalculator used for all calculations
        :param random_seed: seed to be used for random number generator,
        used to determine which pairs will be used for distance calculations
        None sets the default random library seed
        :param rm_outliers: Should outliers be removed before summary statistics calculation
        """
        if random_seed is not None:
            self.random_generator = random.Random(random_seed)
        else:
            self.random_generator = random.Random()
        super().__init__(expression_data=expression_data, calculator=calculator, rm_outliers=rm_outliers)

    def similarities(self, n_pairs: int, adjust_n: bool = False) -> list:
        """
        Calculate similarities between random pairs.
        :param n_pairs: Number of pairs.
        :param adjust_n: Should number of pairs be adjusted to max possible number of pairs
        if n_pairs is larger than possible number of pairs based on n rows in expression data.
        :return: list of similarities
        """
        n_genes = self.expression_data.n_points
        pairs = self.get_random_pairs(n_pairs, n_genes, adjust_n)
        similarities_list = list()
        for pair in pairs:
            index1 = pair[0]
            index2 = pair[1]
            gene1 = self.expression_data.get_genes_data_index(index1)
            gene2 = self.expression_data.get_genes_data_index(index2)
            similarity = self.calculator.similarity(gene1, gene2)
            similarities_list.append(similarity)
        if self.rm_outliers:
            similarities_list = self.remove_outliers(similarities_list)
        return similarities_list

    def get_random_pairs(self, n_pairs, n_points, adjust_n: bool = False, n_points2: int = None) -> set:
        # TODO update test for 2 data sets
        """
        Get random index pairs from n_points. If n_points2 is none draws pairs of two different elements from
        n_points-1; else draws one point from n_points-1 and the other from n_points2-1.
        Indices can be between 0 and n_points-1 (or n_points2-1).
        No pairs are repeated.
        :param n_pairs: n of index pairs
        :param n_points: n of  points/indices
        :param adjust_n: As in similarities method.
        :param n_points2: n of points for the second element of a pair, draws points from two datasets
            (with n_points and n_points2 elements), if None draws pairs from a single dataset with n_points
        :return: index pairs
        """
        if n_pairs < 1:
            raise ValueError('Number of pairs must be at least 1.')
        if n_points2 is None:
            n_possible_pairs = possible_pairs(n_points)
        else:
            n_possible_pairs = n_points * n_points2
        if n_pairs > n_possible_pairs:
            if adjust_n:
                n_pairs = n_possible_pairs
            else:
                raise ValueError('Number of pairs is greater than number of possible unique pairs: ',
                                 n_possible_pairs)
        pairs = set()
        while len(pairs) < n_pairs:
            max2 = None if n_points2 is None else n_points2 - 1
            pair = self.generate_index_pair(min_index=0, max_index=n_points - 1, max_index2=max2)
            pairs.add(pair)
        return pairs

    def generate_index_pair(self, max_index: int, min_index: int = 0, max_index2: int = None) -> tuple:
        # TODO update test for 2 data sets
        """
        Make a pair of indices. Pair of indices contains 2 different indices if max_index2 is None -
        draws from elements between min and max_index; else draws pairs with first element between min and max_index and
        second element between min and max_index2 elements.
        :param max_index: largest possible index, inclusive
        :param min_index: smallest possible index,  inclusive
        :param max_index2: Largest possible index for second element of a pair, if None uses max_index for both elements
        :return: index pair, left index is always smaller if max_index2 is None
        """
        index1 = self.random_generator.randint(min_index, max_index)
        max2 = max_index if max_index2 is None else max_index2
        index2 = self.random_generator.randint(min_index, max2)
        if max_index2 is None:
            while index1 == index2:
                index2 = self.random_generator.randint(min_index, max2)
            if index1 > index2:
                index3 = index2
                index2 = index1
                index1 = index3
        return index1, index2


class GeneSetSimilarityCalculatorNavigator(SimilarityCalculatorNavigator):
    """
    Calculates similarities between genes from a gene set.
    """

    def __init__(self, expression_data: GeneExpression, calculator: SimilarityCalculator,
                 random_generator: RandomSimilarityCalculatorNavigator, rm_outliers: bool = True):
        """
        :param expression_data: Data of all genes
        :param calculator: Similarity Calculator used for all calculations
        :param random_generator: Used to generate random pairs samples when there is a lot of genes in gene set.
        :param rm_outliers: Should outliers be removed before summary statistics calculation
        """
        self.random_generator = random_generator
        super().__init__(expression_data=expression_data, calculator=calculator, rm_outliers=rm_outliers)

    def similarities(self, geneIDs: list, max_n_similarities: int = None, as_list=True):
        """
        Caluclate similarity between all genes from a gene set
        Treats replicates as different genes
        :param geneIDs: Entrez IDs of genes in the gene set
        :param max_n_similarities: Should be number of similarity calculations limited,
        eg. a random sample of similarities between specified genes
        :param as_list: Return list of similarities (True) or dict with keys being gene pair tuples and
            values being similarities
        :return: keys: gene names of two compared genes as tuple, values: similarity
        """
        geneIDs = filter_geneIDs(self.expression_data, geneIDs)
        if len(geneIDs) < 2:
            raise EnrichmentError('Min number of genes in gene set for enrichment calculaton is 2.')
        data = self.expression_data.get_genes_data_IDs(geneIDs)
        n = len(geneIDs)
        similarities_data = dict()
        n_possible_pairs = possible_pairs(n)
        if max_n_similarities is None or n_possible_pairs <= max_n_similarities:
            for i in range(0, n - 1):
                for j in range(i + 1, n):
                    self.calculate_similarity(data, i, j, similarities_data)
        else:
            pairs = self.random_generator.get_random_pairs(max_n_similarities, n, True)
            for pair in pairs:
                i = pair[0]
                j = pair[1]
                self.calculate_similarity(data, i, j, similarities_data)
        if self.rm_outliers:
            similarities_data = self.remove_outliers(similarities_data)
        if as_list:
            similarities_data = list(similarities_data.values())
        return similarities_data

    def similarities_pair(self, geneIDs1: list, geneIDs2: list, max_n_similarities: int = None, as_list=True):
        """
        Caluclate similarity between genes from two different sets.
        :param geneIDs: Entrez IDs of genes in the gene set 1 or 2. If elements are repeated in either geneIDs they are
        used only once (e.g. as if they were previously converted to a set).
        :param max_n_similarities: Should be number of similarity calculations limited,
        eg. a random sample of similarities between specified genes
        :param as_list: Return list of similarities (True) or dict with keys being gene pair tuples and
            values being similarities
        :return: keys: gene names of two compared genes as tuple, values: similarity
        """
        geneIDs1 = filter_geneIDs(self.expression_data, geneIDs1)
        geneIDs2 = filter_geneIDs(self.expression_data, geneIDs2)
        if len(geneIDs1) < 1 or len(geneIDs2) < 1:
            raise EnrichmentError('Both gene sets must contain at least one gene.')
        data1 = self.expression_data.get_genes_data_IDs(geneIDs1)
        data2 = self.expression_data.get_genes_data_IDs(geneIDs2)
        similarities_data = dict()
        n1 = len(geneIDs1)
        n2 = len(geneIDs2)
        n_possible_pairs = n1 * n2
        if max_n_similarities is None or n_possible_pairs <= max_n_similarities:
            for i in range(0, n1):
                for j in range(0, n2):
                    self.calculate_similarity(data1, i, j, similarities_data, data2)
        else:
            pairs = self.random_generator.get_random_pairs(n_pairs=max_n_similarities, n_points=n1, adjust_n=True,
                                                           n_points2=n2)
            for pair in pairs:
                i = pair[0]
                j = pair[1]
                self.calculate_similarity(data1, i, j, similarities_data, data2)
        if self.rm_outliers:
            similarities_data = self.remove_outliers(similarities_data)
        if as_list:
            similarities_data = list(similarities_data.values())
        return similarities_data

    @staticmethod
    def get_gene_vector(data: pd.DataFrame, index: int) -> np.array:
        """
        Get data of single gene of single replicate from expression data frame
        :param data: expression data
        :param index: gene index position in data frame
        :return: data for a gene
        """
        return np.array(data.iloc[index, :])

    @staticmethod
    def get_gene_name(data: pd.DataFrame, index: int):
        """
        Get index name from data frame
        :param data: expression data
        :param index: gene index position in data frame
        :return: index name
        """
        return data.index[index]

    def calculate_similarity(self, data: pd.DataFrame, index1: int, index2: int, add_to: dict,
                             data2: pd.DataFrame = None):
        """
        Calculate similarity between 2 genes
        :param data: expression data for index1 and index2 if data2 is not present
        :param data2: expression data from where second index will be retrieved, if present
        :param index1: of gene1 in expression data
        :param index2: of gene2 in expression data
        :param add_to: list to add similarity to
        """
        if data2 is None:
            data2 = data
        gene1 = self.get_gene_vector(data, index1)
        name1 = self.get_gene_name(data, index1)
        gene2 = self.get_gene_vector(data2, index2)
        name2 = self.get_gene_name(data2, index2)
        similarity = self.calculator.similarity(gene1, gene2)
        add_to[(name1, name2)] = similarity


def possible_pairs(n_points: int):
    """
    Max number of possible index pairs with unique indices if there are n_points genes
    :param n_points: number of rows/genes
    :return: max number of pairs
    """
    return int(factorial(n_points) / (2 * factorial(n_points - 2)))


class EnrichmentError(Exception):
    pass


class RandomStatisticBundleStorage(ABC):
    """
    Abstract class for storage of precomputed similarities between random points and providing their mean and
    standard deviation.
    Calculates standard error based on sample size.
    """

    def __init__(self, similarities: list):
        """
        :param similarities: precomputed similarities to store
        """
        self._similarities = similarities
        self._sd = std_list(self._similarities)
        self._summary_type = None
        self._center = None

    @abstractmethod
    def get_se(self, n_pairs) -> float:
        """
        Return standard error for sample size
        :param n_pairs: sample size
        """
        pass


class RandomMeanStorage(RandomStatisticBundleStorage):
    """
    Store precomputed similarities between random points and provides their mean and standard deviation.
    Calculates standard error based on sample size.
    """

    def __init__(self, similarities: list):
        """
        :param similarities: Set of precomputed random similarities
        """
        super().__init__(similarities=similarities)
        self._center = mean_list(self._similarities)
        self._summary_type = MEAN

    @classmethod
    def from_calculator(cls, calculator: RandomSimilarityCalculatorNavigator, max_similarities: int = MAX_RANDOM_PAIRS,
                        adjust_n: bool = True):
        """
        :param calculator: Used  to calculate not yet calculated random similarities
        :param max_similarities: Number of similarities to compute for summary statistics estimation
        :param adjust_n: If there are less possible similarities than specified in max_similarities automatically use
            number of max possible similarities.
        """
        similarities = calculator.similarities(max_similarities, adjust_n=adjust_n)
        return cls(similarities)

    def get_se(self, n_pairs) -> float:
        """
        Return standard error for sample size
        :param n_pairs: sample size
        :return: se
        """
        return self._sd / sqrt(n_pairs)


class RandomMedianStorage(RandomStatisticBundleStorage):
    """
    Store precomputed similarities between random points and provides their median and standard deviation.
    Calculates standard error based on sample size.
    Standard errors scalings are estimated based on precomputed se obtained by sampling
    samples of specified size and approximating scaling factor based on their distribution
    """

    def __init__(self, similarities: list, se_scaling_points: list = SE_SCALING_POINTS.copy(),
                 permutations: int = PERMUTATIONS):
        """
        :param similarities: precomputed random similarities
        :param se_scaling_points: at whihc sample sizes to estimate scaling for standard error  computation
        :param permutations: how many permutations to use for standard error  calculation
            (approximated based on normal distribution of N=perumtations medians)
        """
        super().__init__(similarities=similarities)
        se_scaling_points.sort()
        self._se_scaling_points = se_scaling_points
        self._permutations = permutations
        self._center = median(self._similarities)
        self._se_scalings = self.se_scalings_estimation()
        self._se_scaling_pairs = list(self._se_scalings.keys())
        self._min_scaling_pair = min(self._se_scaling_pairs)
        self._max_scaling_pair = max(self._se_scaling_pairs)
        self._num_scaling_pairs = len(self._se_scaling_pairs)
        self._summary_type = MEDIAN

    @classmethod
    def from_calculator(cls, calculator: RandomSimilarityCalculatorNavigator, max_similarities: int = 200000,
                        adjust_n: bool = True,
                        se_scaling_points: list = SE_SCALING_POINTS, permutations: int = PERMUTATIONS):
        """
        :param calculator: Used  to calculate not yet calculated random similarities
        :param max_similarities: Number of similarities to compute for summary statistics estimation
        :param adjust_n: If there are less possible similarities than specified in max_similarities automatically use
            number of max possible similarities.
        :param se_scaling_points: at whihc sample sizes to estimate scaling for standard error  computation
        :param permutations: how many permutations to use for standard error  calculation
            (approximated based on normal distribution of N=perumtations medians)
        """
        similarities = calculator.similarities(max_similarities, adjust_n=adjust_n)
        return cls(similarities=similarities, se_scaling_points=se_scaling_points, permutations=permutations)

    def se_scalings_estimation(self) -> OrderedDict:
        """
        Get scaling factors for sample sizes specified in self.se_scaling_points
        :return: list of scaling factors ordered by sample sizes
        """
        scalings = {}
        for n_point in self._se_scaling_points:
            n_pairs = possible_pairs(n_point)
            if len(self._similarities) >= n_pairs:
                scalings[n_pairs] = self.se_scaling_estimation(n_pairs)
        return OrderedDict(sorted(scalings.items()))

    def se_scaling_estimation(self, n_pairs: int) -> float:
        """
        Estimates se scaling factor for distribution of medians based on n_pairs
        :param n_pairs:  Number of elements in each sample.
        :return: scaling factor
        """
        medians = self.random_sample_medians(n_pairs)
        center_fit, se_fit = norm.fit(medians)
        observed_se_scaling = se_fit * sqrt(n_pairs) / self._sd
        return observed_se_scaling

    def random_sample_medians(self, sample_size: int) -> list:
        """
        Compute sample of sample medians.
        :param sample_size: Number of elements in each sample.
        :return: list of medians
        """
        medians = []
        for i in range(self._permutations):
            sample = random.sample(self._similarities, sample_size)
            sample_median = median(sample)
            medians.append(sample_median)
        return medians

    def get_se_scaling(self, n_pairs: int) -> float:
        """
        Estimate standard error scaling factor based on scaling factors of samples with neighbouring sizes.
        Returns min or max if the sample sizes is below or above precomputed, average if sample size is
        between two precomputed points and if sample size matches precomputed one returns its scaling factor
        :param n_pairs: sample size (eg. n of similarities)
        :return: scaling factor
        """
        # self.se_scalings and associated data must be sorted based on n pairs and can not have repeated values
        if n_pairs in self._se_scaling_pairs:
            return self._se_scalings[n_pairs]
        elif n_pairs < self._min_scaling_pair:
            return self._se_scalings[self._min_scaling_pair]
        elif n_pairs > self._max_scaling_pair:
            return self._se_scalings[self._max_scaling_pair]
        else:
            for i in range(self._num_scaling_pairs - 1):
                if self._se_scaling_pairs[i] < n_pairs < self._se_scaling_pairs[i + 1]:
                    y = (self._se_scalings[self._se_scaling_pairs[i]] + self._se_scalings[
                        self._se_scaling_pairs[i + 1]]) / 2
                    return y

    def get_se(self, n_pairs):
        """
        Return standard error for sample size
        :param n_pairs: sample size
        :return: se
        """
        scaling = self.get_se_scaling(n_pairs)
        return scaling * self._sd / sqrt(n_pairs)


def mean_list(data: list) -> float:
    """
    Speeds up mean calculation from list
    :param data: Data for mean calculation
    :return: mean
    """
    return np.array(data).mean()


def std_list(data: list) -> float:
    """
    Speeds up standard deviation (std) calculation from list
    :param data: Data for std calculation
    :return: std
    """
    return np.array(data).std()


class EnrichmentCalculator:
    """
    Determine whether gene set is enriched or not
    Genes are more closely located than expected for random points.
    """

    def __init__(self, random_storage: RandomStatisticBundleStorage,
                 gene_set_calculator: GeneSetSimilarityCalculatorNavigator):
        """
        :param random_storage: used for retrieval of random similarities statistics
        :param gene_set_calculator: used for retrieval of similarities for genes specified in gene sets
        """
        self.calculator = gene_set_calculator
        self.storage = random_storage

    @classmethod
    def quick_init(cls, expression_data: GeneExpression, calculator: SimilarityCalculator,
                   rm_outliers: bool = True, random_seed: int = None, max_random_pairs: int = MAX_RANDOM_PAIRS,
                   summary_type: str = MEAN, se_scaling_points: list = SE_SCALING_POINTS.copy(),
                   permutations: int = PERMUTATIONS):
        """
        :param expression_data: data of all genes
        :param calculator: used to calculate similarities between gene pairs
        :param rm_outliers: Should similarities outliers be removed before reporting/statistical testing
        :param random_seed: Seed used for random number generator used for random pairs
        :param max_random_pairs: Maximum number of random similarities calculated used for random points distribution
            estimation
        :param summary_type: compute p values based on mean or median of similarities
        :param se_scaling_points: if summary_type is 'median' the scaling factor for standard errors of
         medians distribution are estimated based on these points
        :param permutations: if summary_type is 'median' the scaling factor for standard errors of
         medians distribution are estimated based on so many medians for each sample size
        :return: the EnrichmentCalculator object
        """
        for_random = RandomSimilarityCalculatorNavigator(expression_data=expression_data, calculator=calculator,
                                                         rm_outliers=rm_outliers, random_seed=random_seed)
        calculator_gs = GeneSetSimilarityCalculatorNavigator(expression_data=expression_data,
                                                             calculator=calculator, random_generator=for_random,
                                                             rm_outliers=rm_outliers)
        if summary_type == MEAN:
            storage = RandomMeanStorage.from_calculator(calculator=for_random, max_similarities=max_random_pairs,
                                                        adjust_n=True)
        elif summary_type == MEDIAN:
            storage = RandomMedianStorage.from_calculator(calculator=for_random, max_similarities=max_random_pairs,
                                                          adjust_n=True, se_scaling_points=se_scaling_points,
                                                          permutations=permutations)
        return cls(random_storage=storage, gene_set_calculator=calculator_gs)

    def calculate_pval(self, gene_set: GeneSet, max_pairs: int = None) -> GeneSetDataCorrelation:
        """
        Calculate p-val for a single gene-set. Are genes closer in space than expected.
        Compares gene set similarities to similarities between random pairs.
        :param gene_set:
        :param max_pairs: Should number of calculated similarities be limited
        :return: data with gene set pointer and pval, median and mean of similarity
        """
        geneIDs = gene_set.genes
        try:
            set_similarities_data = self.calculator.similarities(geneIDs, max_n_similarities=max_pairs,
                                                                 as_list=False)
        except EnrichmentError:
            raise

        set_similarities = list(set_similarities_data.values())
        mean_set = mean_list(set_similarities)
        median_set = median(set_similarities)
        n = len(set_similarities)
        if self.storage._summary_type == MEAN:
            center_set = mean_set
        elif self.storage._summary_type == MEDIAN:
            center_set = mean_set
        else:
            raise ValueError('Possible summary types are', MEAN, 'and', MEDIAN)
        se = self.storage.get_se(n)
        center_random = self.storage._center
        p = float(1 - mpmath.ncdf(center_set, mu=center_random, sigma=se))

        gene_set_data = GeneSetDataCorrelation(gene_set)
        gene_set_data.mean = mean_set
        gene_set_data.median = median_set
        gene_set_data.pval = p
        gene_set_data.most_similar = self.retain_most_similar(set_similarities_data, 10)
        return gene_set_data

    def calculate_enrichment(self, gene_sets: GeneSets, max_pairs: int = None, max_set_size: int = None) -> list:
        """
        claculate enrichment for multiple gene sets, adjust p val (Benjamini Hoechber)
        :param gene_sets:
        :param max_pairs: for within gene set similarity calculation, as described for calculate_pval
        :param max_set_size: remove gene sets with size above threshold
        :return: gene set data with p and adjusted p values
        """
        if max_set_size is not None:
            gene_sets = [gene_set for gene_set in gene_sets if len(gene_set.genes) <= max_set_size]
        pvals = []
        data = []
        for gene_set in gene_sets:
            try:
                # This line may throw exception if not enough genes are present
                gene_set_data = self.calculate_pval(gene_set=gene_set, max_pairs=max_pairs)
                data.append(gene_set_data)
                pvals.append(gene_set_data.pval)
            except EnrichmentError:
                pass
        padjvals = multipletests(pvals=pvals, method='fdr_bh', is_sorted=False)[1]
        set_counter = 0
        for gene_set_data in data:
            gene_set_data.padj = padjvals[set_counter]
            set_counter += 1
        return data

    @staticmethod
    def retain_most_similar(similarities: dict, best: int) -> set:
        """
        Retain genes from pairs with highest similarities
        :param similarities: dict of similarities as values and keys as tuples of two compared genes' names
        :param best: How many best genes to retain. Always adds both genes from the pair and
            all pairs with similarity equal to last retained similarity,
            thus the number of retained genes may be longer than best.
        :return: retained genes
        """
        retained = set()
        last_retained = 0
        finish = False
        for key in sorted(similarities, key=similarities.get, reverse=True):
            similarity = similarities[key]
            if len(retained) >= best:
                finish = True
            if finish and similarity < last_retained:
                break
            retained.add(key[0])
            retained.add(key[1])
            last_retained = similarity
        return retained


def sort_gscordata_mean(data: list) -> list:
    """
    Sort GeneSetDataCorrelation list based on similarities mean
    :param data: list of GeneSetDataCorrelation
    :return: sorted list of GeneSetDataCorrelation from largest to smallest
    """
    return sorted(data, key=lambda x: x.mean, reverse=True)


def sort_gscordata_median(data: list) -> list:
    """
    Sort GeneSetDataCorrelation list based on similarities median
    :param data: list of GeneSetDataCorrelation
    :return: sorted list of GeneSetDataCorrelation from largest to smallest
    """
    return sorted(data, key=lambda x: x.median, reverse=True)


def filter_enrichment_data_top(data: list, best: int = 10, metric='padj') -> list:
    """
    Return top gene sets based on padj
    :param data: list of GeneSetDataCorrelation
    :param best: Number of retained sets. Will retain more if multiple sets have same, last retained, padj
    :param metric: Filter based on: padj, mean
    :return: list of top gene sets
    """
    MEAN = 'mean'
    PADJ = 'padj'
    retained = []
    terminate = False
    last_retained = 1
    if metric == PADJ:
        data = el.sort_gsdata_padj(data)
    elif metric == MEAN:
        data = sort_gscordata_mean(data)
    else:
        raise ValueError('Metric can be:', PADJ, MEAN)
    for gene_set in data:
        if len(retained) >= best:
            terminate = True
        if terminate and gene_set.padj > last_retained:
            break
        retained.append(gene_set)
        last_retained = gene_set.padj
    return retained


class GeneSetComparator:
    # TODO: write docs, finish testing, decide if changeability_similarity is even sensible

    """
    Compares GeneSets to each other.
    """

    def __init__(self, similarity_calculator: GeneSetSimilarityCalculatorNavigator):
        self.similarity_calculator = similarity_calculator
        self.expression_data = self.similarity_calculator.expression_data

    def between_set_similarities(self, set_pairs_data: list, mode: str, max_pairs: int = None):
        SAMPLING = 'sampling'
        REPRESENTATIVES = 'representatives'
        MODES = [REPRESENTATIVES, SAMPLING]
        """
        Calculates similariti between two data sets based on genes declared to be most similar within the sets
        :param set_pairs_data: list of  GeneSetPairData object to be compared to each other
        :return: Adds data to GeneSetPairData elements
        """
        if mode not in MODES:
            raise ValueError('Mode must be one of: ', MODES)
        for pair in set_pairs_data:
            set1 = pair.gene_set_data1
            set2 = pair.gene_set_data2
            if mode == REPRESENTATIVES:
                genes1=set1.most_similar
                genes2=set2.most_similar
            elif mode == SAMPLING:
                genes1=set1.gene_set.genes
                genes2=set2.gene_set.genes
            similarities = self.similarity_calculator.similarities_pair(geneIDs1=genes1,
                                                                        geneIDs2=genes2,
                                                                        max_n_similarities=max_pairs)
            pair.mean_profile_similarity = mean_list(similarities)
            pair.median_profile_similarity = median(similarities)

    def changeability_similarity(self, set_pairs_data: list):
        gene_sets_data = set()
        for pair in set_pairs_data:
            gene_sets_data.add(pair.gene_set_data1)
            gene_sets_data.add(pair.gene_set_data2)
        for gene_set_data in gene_sets_data:
            self.set_changeability_deviation(gene_set_data)
        for pair in set_pairs_data:
            set1 = pair.gene_set_data1
            set2 = pair.gene_set_data2
            stdevs1 = set1.pattern_stdevs
            stdevs2 = set2.pattern_stdevs
            stdevs_similarity = SimilarityCalculator.calc_cosine(stdevs1, stdevs2)
            pair.profile_changeability_similarity = stdevs_similarity

    def set_changeability_deviation(self, gene_set_data: GeneSetDataCorrelation):
        genes = gene_set_data.gene_set.genes
        genes = filter_geneIDs(self.expression_data, genes)
        data = self.expression_data.get_genes_data_IDs(genes)
        differences = []
        time_points = data.shape[1]
        # Compute absolute differences between time points
        for comparison_point in range(time_points - 1):
            difference = data.iloc[:, comparison_point + 1].subtract(data.iloc[:, comparison_point])
            differences.append(abs(difference))
        # Calculate deviation of differences between time points among genes
        differences = pd.concat(differences, axis=1)
        differences = pp.minmax_scale(differences, axis=1)
        gene_set_data.pattern_stdevs = differences.std(axis=0)

    @staticmethod
    def make_set_pairs(gene_set_data: GeneSetDataCorrelation, include_identical=False) -> list:
        # TODO: enable option to remove pairs that have more than N% shared genes
        set_pairs = []
        n_sets = len(gene_set_data)
        start_j_add = 1
        end_i_miss = 1
        if include_identical:
            start_j_add = 0
            end_i_miss = 0
        for i in range(0, n_sets - end_i_miss):
            for j in range(i + start_j_add, n_sets):
                data1 = gene_set_data[i]
                data2 = gene_set_data[j]
                pair = GeneSetPairData(data1, data2)
                set_pairs.append(pair)
        return set_pairs


def filter_geneIDs(expression_data, geneIDs: list):
    """
    Retain only gene IDs saved in expression_data
    :param geneIDs: gene IDs to filter
    :return: only gene IDs also present in expression_data
    """
    genes_saved = expression_data.get_genes()
    return [gene for gene in geneIDs if gene in genes_saved]
