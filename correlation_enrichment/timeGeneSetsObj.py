
from orangecontrib.bioinformatics.geneset.__init__ import (list_all,load_gene_sets)
from orangecontrib.bioinformatics.geneset.utils import (GeneSet,GeneSets)
import pandas as pd
from sklearn import preprocessing as pp
import numpy as np
from scipy.stats import spearmanr
from scipy.stats import pearsonr
import random
from scipy.stats import ks_2samp
from statsmodels.stats.multitest import multipletests
from math import factorial
from statistics import mean

MAX_PAIRS=5000


class GeneSetData():

    """
    Stores gene set with p value and adj p value enrichment data
    """

    def __init__(self, gene_set:GeneSet):
        """
        :param gene_set: gene set for which data should be stored
        """
        self.gene_set=gene_set

    def set_pval(self,pval:float):
        """
        Add p value to the data
        :param pval: p value
        """
        self.pval=pval

    def set_padj(self,padj:float):
        """
        Add adjusted p value to the data
        :param padj: adjusted p value
        """
        self.padj=padj

    def set_average_similarity_direction(self,greater:bool):
        """
        Specify if average similarity is gretaer or not than random similarities average
        :param greater: true if greater than random
        """
        self.avg_sim_direction=greater


class GeneExpression:
    """
    Stores a data frame with gene expression data.
    """

    def __init__(self,expression:pd.DataFrame):
        """
        :param expression: Expression data frame with genes as rows and sampling points as columns.
        Gene names (indices) must be Entrez IDs (integers in string format).
        """
        self.check_numeric(expression)
        self.are_geneIDs(expression.index)

        self.expression = expression
        self.n_points=expression.shape[0]

    @staticmethod
    def check_numeric(expression:pd.DataFrame):
        """
        Is content of data frame numeric (pandas float/int).
        If data is not numeric raises exception.
        :param expression: data frame to be analysed
        """
        if not ((expression.dtypes=='float64').all() or (expression.dtypes=='int64').all()):
            raise GeneExpressionExc('Expression data is not numeric.')

    @staticmethod
    def are_geneIDs(ids:iter):
        """
        Are elements strings that can be transformed into integers.
        If not raises exception.
        :param ids: Iterable object of elements
        """
        for i in ids:
            if not isinstance(i,str):
                raise GeneExpressionExc('Gene names are not IDs. Gene IDs are of type str and can be converted to int.')
            else:
                try:
                    int(i)
                except ValueError:
                    raise GeneExpressionExc('Gene names are not IDs. Gene IDs are of type str and can be converted to int.')

    def get_genes(self) -> list:
        """
        :return: all gene names
        """
        return list(self.expression.index)

    def get_num_points(self)->int:
        """
        :return: number of genes in expression data
        """
        return self.n_points

    def get_genes_data_IDs(self,geneIDs:list) -> pd.DataFrame:
        """
        Data frame with only specified genes.
        All query IDs must be present in expression data.
        :param geneIDs: Specified genes, same as data index names.
        :return: Subset of original data.
        """
        if not all(gene in self.expression.index for gene in geneIDs):
            raise KeyError('Not all gene IDs are in expression data.')
        else:
            return self.expression.loc[geneIDs,:]

    def get_genes_data_index(self,index:int)->np.ndarray:
        """
        Get data of single gene bynumerical (row) index.
        :param index: The index of the row.
        :return: Expression data for that gene.
        """
        if self.n_points-1 >= index:
            return np.array(self.expression.iloc[index,:])
        else:
            raise IndexError('Index must be below',self.n_points)


class GeneExpressionRep(GeneExpression):
    """
     Stores a data frame with gene expression data as GeneExpression.
     However, more than one measurment per gene can be present. This could be used with replicates.
    """
    def __init__(self,expression:pd.DataFrame,expression_index:dict):
        """
        :param expression: A numeric data frame with expression data with unique row names
        (eg. GeneA_1, GeneA_2,... for replicates).
        :param expression_index: A dictionary storing which row names in data frame belong to single EntrezID.
        For x+example: {EID_A:[GeneA_1,GeneA_2]}. Enterz IDs must be integr strings.
        """
        self.check_numeric(expression)
        self.are_geneIDs(expression_index.keys())
        self.match_index(expression, expression_index)

        self.expression = expression
        self.n_points = expression.shape[0]
        self.expression_index=expression_index

    @staticmethod
    def match_index(expression: pd.DataFrame, expression_index: dict):
        """
        Check if indices match between data frame (rown names) and dictionary (values) used for annotation.
        If not raise an exception.
        :param expression: data frame of expression data
        :param expression_index: annotation dict
        """
        index = expression.index
        indices_dict = {x for v in expression_index.values() for x in v}
        if not set(index) == indices_dict:
            raise GeneExpressionExc('DataFrame and dict gene names do not match.')

    def get_genes(self) -> list:
        """
        :return: all EntrezIDs
        """
        return list(self.expression_index.keys())

    def get_genes_data_IDs(self,geneIDs:list) -> pd.DataFrame:
        """
        Get all rows that belong to specified Entrez IDs.
        All query IDs must be present in expression data.
        :param geneIDs: Enterz IDs, integer strings
        :return: subset of expression data frame
        """
        if not all(gene in self.expression_index.keys() for gene in geneIDs):
            raise KeyError('Not all gene IDs are in expression data.')
        else:
            geneNames=[self.expression_index.get(id) for id in geneIDs]
            geneNames=[gene for geneList in geneNames for gene in geneList]
            return self.expression.loc[geneNames,:]



class GeneExpressionExc(Exception):
    pass

class SimilarityCalculator:
    """
    Calculates similarity between two vectors.
"""

    #Similarity method
    PEARSON='correlation_pearson'
    SPEARMAN='correlation_spearman'
    COSINE='cosine'
    SIMILARITIES=[PEARSON,SPEARMAN,COSINE]
    CORRELATIONS=[PEARSON,SPEARMAN]

    #Normalisation method
    M0SD1='mean0std1'
    TO_ONE='to_one'
    NORMALISATION=[M0SD1,TO_ONE,None]

    def __init__(self, bidirectional:bool=True, similarity_type:str=SPEARMAN, normalisation_type:str=None):
        """

        :param bidirectional: Return better of two possible similarities between profiles -
            eg. profile1 vs profile2 or profile1 vs inverse  profile2 (abs(corrlation/cosine); as multiplied by -1)
        :param similarity_type: Similarity method options: 'correlation_pearson', 'correlation_spearman', 'cosine'
        :param normalisation_type: Normalisation method options, used only for cosine similarity:'mean0std1','to_one' or None.
        'to_one' divides each value with max value of the vector

        These types are used automatically when calling similarity method
        """
        if not similarity_type in self.SIMILARITIES:
            raise ArgumentException('Possible similarity_type:',self.SIMILARITIES)
        if not normalisation_type in self.NORMALISATION:
            raise ArgumentException('Possible normalisation_type:',self.NORMALISATION)

        self.bidirectional=bidirectional
        self.similarity_type=similarity_type
        self.normalisation_type=normalisation_type

    def similarity(self,vector1:np.ndarray,vector2:np.ndarray)->float:
        """
        Calculate similaity between two vectors
        :param vector1:
        :param vector2:
        :return: similarity
        """
        if self.similarity_type in self.CORRELATIONS:
            return self.correlation(vector1,vector2)
        elif self.similarity_type == self.COSINE:
            return self.cosine( vector1, vector2)

    def cosine(self,vector1:np.ndarray,vector2:np.ndarray) -> float:
        """
        Navigate cosine similarity calculation and reporting.
        :param vector1:
        :param vector2:
        :return: similarity
        """
        if self.normalisation_type!=None:
            vector1=self.normalise(vector1)
            vector2 = self.normalise(vector2)
        similarity=self.calc_cosine(vector1,vector2)

        if self.bidirectional:
            similarity=abs(similarity)
        return similarity

    def correlation(self,vector1:np.ndarray,vector2:np.ndarray)->float:
        """
        Navigate correlation (spearman or pearson) calculation.
        :param vector1:
        :param vector2:
        :return: similarity
        """
        if self.similarity_type==self.SPEARMAN:
            similarity=self.calc_spearman(vector1,vector2)
        elif self.similarity_type==self.PEARSON:
            similarity=self.calc_pearson(vector1,vector2)

        if self.bidirectional:
            similarity=abs(similarity)
        return similarity

    def normalise(self,vector:np.ndarray) ->np.ndarray:
        """
        Normalise a vector
        :param vector:
        :return: normalised vector
        """
        if isinstance(vector, list):
            vector = np.array(vector)
        if self.normalisation_type==self.M0SD1:
            return pp.scale(vector)
        elif self.normalisation_type==self.TO_ONE:
            return vector/max(vector)

    @staticmethod
    def calc_cosine(vector1:np.ndarray,vector2:np.ndarray) -> float:
        """
        Calculate cosine similarity
        :param vector1:
        :param vector2:
        :return: similarity
        """
        return np.inner(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

    @staticmethod
    def calc_spearman(vector1:np.ndarray,vector2:np.ndarray) -> float:
        """
            Calculate spearman correlation
            :param vector1:
            :param vector2:
            :return: correlation
        """
        return spearmanr(vector1,vector2)[0]

    @staticmethod
    def calc_pearson(vector1:np.ndarray,vector2:np.ndarray) -> float:
        """
            Calculate pearson correlation
            :param vector1:
            :param vector2:
            :return: correlation
        """
        return pearsonr(vector1,vector2)[0]


class ArgumentException(Exception):
    pass

class SimilarityCalculatorNavigator:
    """
    Navigates similarity calculation between specified genes and reports similarity statistics
    """
    def __init__(self, expression_data:GeneExpression, calculator:SimilarityCalculator,rm_outliers:bool=True):
        """
        :param expression_data:  Data for all genes
        :param calculator: SimilarityCalculator used in all calculations
        :param rm_outliers: should outliers be removed before similarity statistics calculation
        """
        self.expression_data=expression_data
        self.calculator=calculator
        self.rm_outliers=rm_outliers

    @staticmethod
    def remove_outliers(measurments:list)->list:
        """
        Remove outliers by quantile method (1.5*IQR above or below Q3 or Q1, respectively).
        :param measurments: similarities
        :return:filtered measurments without outliers
        """
        measurments = np.array(measurments)
        upper_quartile = np.percentile(measurments, 75)
        lower_quartile = np.percentile(measurments, 25)
        IQR_scaled = (upper_quartile - lower_quartile) * 1.5
        quartile_set = (lower_quartile - IQR_scaled, upper_quartile + IQR_scaled)
        retained = []
        for distance in measurments:
            if quartile_set[0] <= distance <= quartile_set[1]:
                retained.append(distance)
        return retained

class RandomSimilarityCalculatorNavigator(SimilarityCalculatorNavigator):

    """
    Navigate similarity calculation between random points.
    """

    def __init__(self, expression_data:GeneExpression, calculator:SimilarityCalculator,random_seed:int=None,
                 rm_outliers:bool=True):
        """
        :param expression_data: Data for all genes
        :param calculator: SimilarityCalculator used for all calculations
        :param random_seed: seed to be used for random number generator,
        used to determine which pairs will be used for distance calculations
        None sets the default random library seed
        :param rm_outliers: Should outliers be removed before summary statistics calculation
        """
        if random_seed !=None:
            self.random_generator = random.Random(random_seed)
        else:
            self.random_generator = random.Random()
        super().__init__(expression_data=expression_data,calculator=calculator,rm_outliers=rm_outliers)

    def similarities(self,n_pairs:int,adjust_n:bool=False )->list:
        """
        Calculate similarities between random pairs.
        :param n_pairs: Number of pairs.
        :param adjust_n: Should number of pairs be adjusted to max possible number of pairs
        if n_pairs is larger than possible number of pairs based on n rows in expression data.
        :return: list of similarities
        """
        n_genes = self.expression_data.get_num_points()
        pairs=self.get_random_pairs(n_pairs,n_genes,adjust_n)
        similarities_list=list()
        for pair in pairs:
            index1=pair[0]
            index2=pair[1]
            gene1=self.expression_data.get_genes_data_index(index1)
            gene2=self.expression_data.get_genes_data_index(index2)
            similarity=self.calculator.similarity(gene1,gene2)
            similarities_list.append(similarity)
        if self.rm_outliers:
            similarities_list=self.remove_outliers(similarities_list)
        return similarities_list

    def get_random_pairs(self,n_pairs,n_points,adjust_n:bool=False)->set:
        """
        Get random index pairs from n_points. Indices can be between 0 and n_points-1.
        No pairs are repeated. Pair consists of 2 different indices.
        :param n_pairs: n of index pairs
        :param n_points: n of  points/indices
        :param adjust_n: As in similarities method.
        :return: index pairs
        """
        if n_pairs<1:
            raise ArgumentException('Number of pairs must be at least 1.')
        n_possible_pairs=possible_pairs(n_points)
        if n_pairs>n_possible_pairs:
            if adjust_n:
                n_pairs=n_possible_pairs
            else:
                raise ArgumentException('Number of pairs is greater than number of possible unique pairs: ',n_possible_pairs)
        pairs=set()
        while len(pairs)<n_pairs:
            pair = self.generate_index_pair(min_index=0,max_index= n_points - 1)
            pairs.add(pair)
        return pairs


    def generate_index_pair(self,max_index:int,min_index:int=0,)->tuple:
        """
        Make a pair of indices. Pair of indices contains 2 different indices.
        :param max_index: largest possible index, inclusive
        :param min_index: smallest possible index,  inclusive
        :return: index pair, left index is always smaller
        """
        index1 = self.random_generator.randint(min_index,max_index)
        index2 = self.random_generator.randint(min_index, max_index)
        while index1==index2:
            index2 = self.random_generator.randint(min_index, max_index)
        if index1 > index2:
            index3 = index2
            index2 = index1
            index1 = index3
        return index1,index2


class GeneSetSimilarityCalculatorNavigator(SimilarityCalculatorNavigator):
    """
    Calculates similarities between genes from a gene set.
    """

    def __init__(self, expression_data:GeneExpression, calculator:SimilarityCalculator,
                 random_generator:RandomSimilarityCalculatorNavigator,rm_outliers:bool=True):
        """
        :param expression_data: Data of all genes
        :param calculator: Similarity Calculator used for all calculations
        :param random_generator: Used to generate random pairs samples when there is a lot of genes in gene set.
        :param rm_outliers: Should outliers be removed before summary statistics calculation
        """
        self.random_generator = random_generator
        super().__init__(expression_data=expression_data,calculator=calculator,rm_outliers=rm_outliers)

    def similarities(self,geneIDs:list ,max_n_similarities:int=None) ->list:
        """
        Caluclate similarity between all genes from a gene set
        :param geneIDs: Entrez IDs of genes in the gene set
        :param max_n_similarities: Should be number of similarity calculations limited,
        eg. a random sample of similarities between specified genes
        :return: list of similarities
        """
        genes_saved=self.expression_data.get_genes()
        geneIDs=[gene for gene in geneIDs if gene in genes_saved]
        #Cant calculate variance otherwise
        if len(geneIDs)<3:
            raise EnrichmentError('Min number of genes in gene set for enrichment calculaton is 3.')
        data=self.expression_data.get_genes_data_IDs(geneIDs)
        n=len(geneIDs)
        similarities_list=[]
        n_possible_pairs=possible_pairs(n)
        if max_n_similarities==None or n_possible_pairs<=max_n_similarities:
            for i in range(0,n-1):
                for j in range(i+1,n):
                    self.calculate_similarity(data,i,j,similarities_list)
        else:
            pairs=self.random_generator.get_random_pairs(max_n_similarities,n,True)
            for pair in pairs:
                i = pair[0]
                j = pair[1]
                self.calculate_similarity(data, i, j, similarities_list)
        if self.rm_outliers:
            similarities_list=self.remove_outliers(similarities_list)
        return similarities_list

    def get_gene_vector(self,data:pd.DataFrame,index:int)->np.array:
        """
        Get data of single gene from expression data frame
        :param data: expression data
        :param index: gene index in data frame
        :return: data for a gene
        """
        return np.array(data.iloc[index, :])

    def calculate_similarity(self,data:pd.DataFrame,index1:int,index2:int,add_to:list):
        """
        Calculate similarity between 2 genes
        :param data: expression data of all genes
        :param index1: of gene1 in expression data
        :param index2: of gene2 in expression data
        :param add_to: list to add similarity to
        """
        gene1 = self.get_gene_vector(data, index1)
        gene2 = self.get_gene_vector(data, index2)
        similarity = self.calculator.similarity(gene1, gene2)
        add_to.append(similarity)

def possible_pairs(n_points:int):
    """
    Max number of possible index pairs with unique indices if there are n_points genes
    :param n_points: number of rows/genes
    :return: max number of pairs
    """
    return int(factorial(n_points)/(2*factorial(n_points-2)))


class EnrichmentError(Exception):
    pass


class RandomSimilarityStorage:
    """
    Supply similarities and store already calculated similarity statistics for similarities between random points
    """
    def __init__(self, calculator:RandomSimilarityCalculatorNavigator):
        """
        :param calculator: Used  to calculate not yet calculated random similarities
        """
        self.calculator=calculator
        self.storage=dict()

#Solwed elsewhere: TODO if number of genes is very small in some groups the distances may not be representative
    def get(self,number:int,adjust_n:bool=False)->list:
        """
        Get similarity between specified number of points.
        If similarity has not yet been calculated calculate it a new.
        :param number: n of points/rows/genes
        :param adjust_n: Should the number be reduced to max possible number of pairs
        (as in RandomSimilarityCalculatorNavigator.similarities())
        :return: similarity summary statistics
        """
        if number in self.storage.keys():
            return self.storage.get(number)
        else:
            stats=self.calculator.similarities(number,adjust_n=adjust_n)
            self.storage[number]=stats
            return stats

class EnrichmentCalculator:
    """
    Determine whether gene set is enriched or not
    Genes are more closely located than expected for random points.
    """
    def __init__(self,similarity_calculator:GeneSetSimilarityCalculatorNavigator,
                 similarity_storage:RandomSimilarityStorage):
        """
        :param similarity_calculator: Used to calculate similarities within gene sets
        :param similarity_storage: Supplies similarities between random points
        """
        self.calculator=similarity_calculator
        self.storage=similarity_storage

    def calculate_pval(self,gene_set:GeneSet,max_pairs:int=None)->GeneSetData:
        """
        Calculate p-val for a single gene-set. Are genes closer in space than expected.
        Compares gene set similarities to similarities between MAX_PAIRS random pairs
         (less if not so much possible pairs are present).
        :param gene_set:
        :param max_pairs: Should number of calculated similarities be limited
        :return: data with gene set pointer and pval
        """
        geneIDs=gene_set.genes
        try:
            if max_pairs!=None:
                set_similarities=self.calculator.similarities(geneIDs,max_pairs)
            else:
                set_similarities = self.calculator.similarities(geneIDs)
        except EnrichmentError:
            raise
        #This option was not ok as random distances varied heavily with n of calculated distances
        #random_stats=self.storage.get(set_stats.n)
        random_similarities = self.storage.get(MAX_PAIRS,True)
        d,p2=ks_2samp(set_similarities,random_similarities)
        #p=p2/2
        gene_set_data=GeneSetData(gene_set)
        gene_set_data.set_pval(p2)
        greater_mean=False
        if mean(set_similarities) > mean(random_similarities):
            greater_mean=True
        gene_set_data.set_average_similarity_direction(greater_mean)
        return gene_set_data

    def calculate_enrichment(self,gene_sets:GeneSets,max_pairs:int=None)->list:
        """
        claculate enrichment for multiple gene sets, adjust p val (Benjamini Hoechber)
        :param gene_sets:
        :param max_pairs: for within gene set similarity calculation, as described for  calculate_pval
        :return: gene set data with p and adjusted p values
        """
        pvals=[]
        data=[]
        for gene_set in gene_sets:
            try:
                #This line may throw exception if not enough genes are present
                gene_set_data=self.calculate_pval(gene_set,max_pairs)
                if gene_set_data.avg_sim_direction:
                    data.append(gene_set_data)
                    pvals.append(gene_set_data.pval)
            except EnrichmentError:
                pass
        padjvals=multipletests(pvals=pvals,method='fdr_bh',is_sorted=False)[1]
        set_counter=0
        for gene_set_data in data:
            gene_set_data.set_padj(padjvals[set_counter])
            set_counter+=1
        return data

    @staticmethod
    def filter_enrichment_data(data:list, max_padj:float)->list:
        """
        Return only enriched data with padj bellow threshold
        :param data: list of GeneSetData
        :param max_padj: max padj of gene set to be returned
        :return: list of gene sets with padj below threshold
        """
        enriched=[]
        for gene_set in data:
            if gene_set.padj <=max_padj:
                enriched.append(gene_set)
        return enriched


class EnrichmentCalculatorFactory:
    """
    Make objects necesary for enrichment calculation
    """

    @staticmethod
    def make_enrichment_calculator( expression_data:GeneExpression, calculator:SimilarityCalculator,
                                    rm_outliers:bool=True, random_seed:int=None)->EnrichmentCalculator:
        """
        Return initialised EnrichmentCalculator
        :param expression_data: data of all genes
        :param calculator: used to calculate similarities between gene pairs
        :param rm_outliers: Should similarities outliers be romoved befor reporting/statistical testing
        :param random_seed: Seed used for random number generator used for random pairs
        :return:
        """
        for_random = RandomSimilarityCalculatorNavigator(expression_data=expression_data, calculator=calculator,
                                                         rm_outliers=rm_outliers,random_seed=random_seed)
        for_GO=GeneSetSimilarityCalculatorNavigator(expression_data=expression_data,
                                                    calculator=calculator, random_generator=for_random,
                                                    rm_outliers=rm_outliers)
        random_storage=RandomSimilarityStorage(for_random)
        return EnrichmentCalculator(for_GO,random_storage)







