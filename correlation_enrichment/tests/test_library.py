import unittest

from scipy.spatial.distance import cosine

from correlation_enrichment.library import *

DF = pd.DataFrame({1: [1, 2, 3, 4, 5, 6, 7, -10], 2: [12, 22, 23, 24, 25, 26, 27, -100], 3: [1, 2, 3, 4, 5, 6, 7, 100]},
                  index=['1', '2', '3', '4', '5', '6', '7', '8'])
GE = GeneExpression(DF)

n_genes = 10
GENES = []
for i in range(n_genes):
    GENES.append(str(i))
DF2 = pd.DataFrame(
    {1: list(np.random.random_sample(n_genes)), 2: list(np.random.random_sample(n_genes)),
     3: list(np.random.random_sample(n_genes))}, index=GENES)
GE2 = GeneExpression(DF2)
SC = SimilarityCalculator(bidirectional=True)
GENE_SET = GeneSet(genes=random.sample(GENES, 2))
GENE_SET2 = GeneSet(genes=random.sample(GENES, 5))
GENE_SET3 = GeneSet(genes=random.sample(GENES, 4))
GENE_SET4 = GeneSet(genes=random.sample(GENES, 3))
GENE_SET5 = GeneSet(genes=random.sample(GENES, 5))
GENE_SET6 = GeneSet(genes=['1090', '198923'])
GENE_SETS = GeneSets([GENE_SET, GENE_SET2, GENE_SET3, GENE_SET4, GENE_SET5, GENE_SET6])


class TestGeneExpression(unittest.TestCase):
    ge = GeneExpression(pd.DataFrame({1: [11, 21, 31, 41], 1: [12, 22, 32, 43]}, index=['1', '2', '2', '3']))

    def test_init(self):
        # Correct data type
        GeneExpression(pd.DataFrame({1: [1, 1.2]}, index=['1', '2']))
        # Wrong DF value type
        with self.assertRaises(ValueError):
            GeneExpression(pd.DataFrame({1: ['a']}, index=['1']))
        with self.assertRaises(ValueError):
            GeneExpression(pd.DataFrame({1: [None]}, index=['1']))
        # Wrong index type
        with self.assertRaises(ValueError):
            GeneExpression(pd.DataFrame({1: [1]}, index=['a']))

    def test_check_numeric(self):
        # Correct numeric data
        GeneExpression.check_numeric(pd.DataFrame({1: [1, 1.2]}))
        # Wrong
        with self.assertRaises(ValueError):
            GeneExpression.check_numeric(pd.DataFrame({1: ['a']}))
        with self.assertRaises(ValueError):
            GeneExpression.check_numeric(pd.DataFrame({1: [None]}))

    def test_are_gene_IDs(self):
        # Correctly formatted gene IDs eg. integer string
        GeneExpression.are_geneIDs(['1'])
        # Incorrect gene IDs
        with self.assertRaises(ValueError):
            GeneExpression.are_geneIDs(['a'])
        with self.assertRaises(ValueError):
            GeneExpression.are_geneIDs([1])

    def test_get_genes(self):
        # Retrieves set of unique genes
        self.assertEqual(self.ge.get_genes(), {'1', '2', '3'})

    def test_get_genes_data_IDs(self):
        # Is data correctly retrieved by gene name
        if not self.ge.get_genes_data_IDs(['1', '2']).equals(
                pd.DataFrame({1: [11, 21, 31], 1: [12, 22, 32]}, index=['1', '2', '2'])):
            raise AssertionError('Results do not match')
        # Is error raised if gene not present in data is sought
        with self.assertRaises(KeyError):
            self.ge.get_genes_data_IDs(['4'])

    def test_get_genes_data_index(self):
        # Is data correctly retrieved by index position
        self.assertEqual(self.ge.get_genes_data_index(1),
                         np.array(pd.DataFrame({1: [21], 1: [22]}, index=['2'])))
        # Is error raised if too large index is used
        with self.assertRaises(IndexError):
            self.ge.get_genes_data_index(4)


class TestSimilarityCalculator(unittest.TestCase):
    vector = np.array([1, 2, 3, 4])

    def test_init(self):
        # Are errors raised if wrong arguments are passed
        with self.assertRaises(ValueError):
            SimilarityCalculator(similarity_type='a')
        with self.assertRaises(ValueError):
            SimilarityCalculator(normalisation_type='a')

    def test_similarity(self):
        # Is high level result of statistical test same as the one directly from the test
        # The calculator is thus set not to use any intermediate processing
        sc = SimilarityCalculator(similarity_type='correlation_pearson', bidirectional=False)
        self.assertAlmostEqual(sc.similarity(self.vector, self.vector), sc.calc_pearson(self.vector, self.vector))
        sc = SimilarityCalculator(similarity_type='correlation_spearman', bidirectional=False)
        self.assertAlmostEqual(sc.similarity(self.vector, self.vector), sc.calc_spearman(self.vector, self.vector))
        sc = SimilarityCalculator(similarity_type='cosine', bidirectional=False, normalisation_type=None)
        self.assertAlmostEqual(sc.similarity(self.vector, self.vector), sc.calc_cosine(self.vector, self.vector))

    def test_cosine(self):
        # Is high level result of statistical test same as the one directly from the test
        # The calculator is thus set not to use any intermediate processing
        sc = SimilarityCalculator(similarity_type='cosine', normalisation_type=None, bidirectional=False)
        self.assertAlmostEqual(sc.cosine(self.vector, self.vector + 10), sc.calc_cosine(self.vector, self.vector + 10))
        # Test if processing parameters change similarity result compared to raw one
        self.assertNotEqual(round(sc.cosine(self.vector, self.vector + 10), 7),
                            round(sc.cosine(self.vector, self.vector), 7))
        self.assertNotEqual(round(sc.cosine(self.vector, self.vector * -1), 7),
                            round(sc.cosine(self.vector, self.vector), 7))
        sc = SimilarityCalculator(similarity_type='cosine', normalisation_type='mean0std1', bidirectional=False)
        self.assertEqual(round(sc.cosine(self.vector, self.vector + 10), 7),
                         round(sc.cosine(self.vector, self.vector), 7))
        sc = SimilarityCalculator(similarity_type='cosine', normalisation_type=None, bidirectional=True)
        self.assertEqual(round(sc.cosine(self.vector, self.vector * -1), 7),
                         round(sc.cosine(self.vector, self.vector), 7))

    def test_correlation(self):
        # Is high level result of statistical test same as the one directly from the test
        # The calculator is thus set not to use any intermediate processing
        sc_p = SimilarityCalculator(similarity_type='correlation_pearson', bidirectional=False)
        self.assertAlmostEqual(sc_p.correlation(self.vector, self.vector), sc_p.calc_pearson(self.vector, self.vector))
        sc_s = SimilarityCalculator(similarity_type='correlation_spearman', bidirectional=False)
        self.assertAlmostEqual(sc_s.correlation(self.vector, self.vector), sc_s.calc_spearman(self.vector, self.vector))
        # Do processing parameters work (eg. is similarity the same for negative and positive correlation
        # with bidirectional=True)
        sc_p_bidirect = SimilarityCalculator(similarity_type='correlation_pearson', bidirectional=True)
        self.assertNotEqual(round(sc_p.correlation(self.vector, self.vector), 7),
                            round(sc_p.correlation(self.vector, self.vector * -1), 7))
        self.assertAlmostEqual(round(sc_p_bidirect.correlation(self.vector, self.vector), 7),
                               round(sc_p_bidirect.correlation(self.vector, self.vector * -1), 7))

    def test_normalise(self):
        # Test if data is not normalised if normalisation parameter is not set
        sc = SimilarityCalculator(similarity_type='cosine', normalisation_type=None)
        self.assertEqual(sc.normalise(self.vector), None)
        # Test if normalisation to mean 0 and std 1 results in vector with these parameters
        sc = SimilarityCalculator(similarity_type='cosine', normalisation_type='mean0std1')
        normalised = sc.normalise(self.vector)
        self.assertEqual(round(mean_list(normalised), 1), 0.0)
        self.assertEqual(round(std_list(normalised), 1), 1.0)

    def test_calc_cosine(self):
        # Test if 'hand made' cosine similarity method returns same result as scipy method
        sc = SimilarityCalculator(similarity_type='cosine')
        self.assertAlmostEqual(sc.calc_cosine(self.vector, self.vector + 10), 1 - cosine(self.vector, self.vector + 10))


class TestSimilarityCalculatorNavigator(unittest.TestCase):

    def test_remove_outliers(self):
        # Is vector without outliers left intact
        vector = [1, 2, 3, 4]
        self.assertListEqual(SimilarityCalculatorNavigator.remove_outliers(vector), vector)
        # Are outliers removed
        self.assertListEqual(SimilarityCalculatorNavigator.remove_outliers(vector + [max(np.array(vector) * 100)]),
                             vector)


class TestRandomSimilarityCalculatorNavigator(unittest.TestCase):
    sc = SimilarityCalculator(bidirectional=False)
    rscn = RandomSimilarityCalculatorNavigator(expression_data=GE, calculator=sc, rm_outliers=False)
    rscn_rm_outliers = RandomSimilarityCalculatorNavigator(expression_data=GE,
                                                           calculator=sc, rm_outliers=True)

    def test_similarities(self):
        # Are outliers removed
        # This requires GE to have one differently distributed row
        self.assertGreater(len(self.rscn.similarities(100, adjust_n=True)),
                           len(self.rscn_rm_outliers.similarities(100, adjust_n=True)))
        # Raise error if too many similarities are required (beyond possible similarities with this number of points)
        with self.assertRaises(ValueError):
            self.rscn.similarities(100, adjust_n=False)
        # Check if automatic adjustment of n computed similarities to max possible works
        self.assertEqual(possible_pairs(GE.n_points), len(self.rscn.similarities(100, adjust_n=True)))

    def test_get_random_pairs(self):
        n_points = 5
        n_pairs = 10
        # Is right number of pairs is computed
        pairs = self.rscn.get_random_pairs(n_pairs=n_pairs, n_points=n_points, adjust_n=False)
        self.assertEqual(len(pairs), n_pairs)
        # Do pairs contain only indices in allowed range, use n_points and n_pairs so that n_pairs is max possible
        # pairs for this n_points, enforcing that all possible pairs are used, ensuring that the point indices of pairs
        # are not by chance in the specified range
        points = list(sum(pairs, ()))
        self.assertLess(max(points), n_points)
        # Are errors raised if too big (more than points) or small number of pairs is specified
        with self.assertRaises(ValueError):
            self.rscn.get_random_pairs(n_pairs=10, n_points=3, adjust_n=False)
        with self.assertRaises(ValueError):
            self.rscn.get_random_pairs(n_pairs=0, n_points=3, adjust_n=False)
        # Check if auto adjusting for max possible number of pairs works
        self.rscn.get_random_pairs(n_pairs=10, n_points=3, adjust_n=True)

    def test_generate_index_pair(self):
        # Test if index pairs are correctly generated (if min=0 and max=1 only the pair 0,1 is possible)
        # Run test multiple times to check if orientation of the single possible pair is  not random - (smaller,larger)
        for rep in range(5):
            self.assertEqual(self.rscn.generate_index_pair(1, 0), (0, 1))


class TestGeneSetSimilarityCalculatorNavigator(unittest.TestCase):
    sc = SimilarityCalculator(bidirectional=False)
    rscn = RandomSimilarityCalculatorNavigator(expression_data=GE, calculator=sc)
    gsscn = GeneSetSimilarityCalculatorNavigator(expression_data=GE, calculator=sc, random_generator=rscn,
                                                 rm_outliers=False)
    gsscn_rm_outliers = GeneSetSimilarityCalculatorNavigator(expression_data=GE, random_generator=rscn,
                                                             calculator=sc, rm_outliers=True)

    def test_similarities(self):
        # Check if outliers are removed, for this GE has to have obvious outlier in shape and processing parameters
        # of gsscn must not eliminate it
        self.assertGreater(len(self.gsscn.similarities(['1', '2', '3', '4', '5', '6', '7', '8'])),
                           len(self.gsscn_rm_outliers.similarities(['1', '2', '3', '4', '5', '6', '7', '8'])))
        # Check if setting maximal number of similarities to be computed works
        max_pairs = 3
        self.assertEqual(max_pairs, len(self.gsscn.similarities(['1', '2', '3', '4', '5', '6', '7', '8'],
                                                                max_n_similarities=max_pairs)))
        # Does too small number of specified genes for similarities calculation result in correct type of error
        with self.assertRaises(EnrichmentError):
            self.gsscn.similarities(['1'])
        # Are all possible similarities computed
        self.assertEqual(possible_pairs(GE.n_points),
                         len(self.gsscn.similarities(['1', '2', '3', '4', '5', '6', '7', '8'])))

    def test_calculate_similarity(self):
        # Is a single computed similarity added to result list
        # Uses specified rows in GE so that specified similarity will be 1  - the GE structure must be correct
        result = dict()
        self.gsscn.calculate_similarity(DF, 0, 1, result)
        self.assertDictEqual(result, {('1','2'):1})


class RandomMeanStorage(unittest.TestCase):
    sc = SimilarityCalculator(bidirectional=True)
    rscn = RandomSimilarityCalculatorNavigator(expression_data=GE, calculator=sc)
    rms = RandomMeanStorage.from_calculator(rscn)

    def test_init(self):
        # Is center of similarities set to mean
        self.rms._center = mean_list(self.rms._similarities)


class RandomMedianStorage(unittest.TestCase):
    rscn = RandomSimilarityCalculatorNavigator(expression_data=GE2, calculator=SC)
    permutations = 10
    rms = RandomMedianStorage.from_calculator(rscn, se_scaling_points=[2, 3, 4, 300], permutations=permutations)

    def test_init(self):
        # Is center of similarities set to median
        self.rms._center = median(self.rms._similarities)

    def test_se_scalings_estimation(self):
        # Test if scaling factors are computed for specified sample sizes (n pairs), eliminating too big sample sizes
        # N points=2 ->n possible pairs =1
        # N points=3 ->n possible pairs =3
        # N points=4 ->n possible pairs =6
        self.assertListEqual(list(self.rms.se_scalings_estimation().keys()), [1, 3, 6])

    def test_random_sample_medians(self):
        # Is the number of computed medians correct
        self.assertEqual(len(self.rms.random_sample_medians(3)), self.permutations)

    def test_get_se_scaling(self):
        # Is retrieved se scaling for precomputed sample size (pairs) the same as the precomputed one
        self.assertEqual(self.rms.get_se_scaling(1), self.rms._se_scalings[1])
        # Is retrieved se scaling for sample size (pairs) smaller/larger than minimal/maximal precomputed sample size
        # equal to smallest/largest precomputed one
        self.assertEqual(self.rms.get_se_scaling(0), self.rms._se_scalings[1])
        self.assertEqual(self.rms.get_se_scaling(100), self.rms._se_scalings[6])
        # Is retrieved scaling for sample size between two precomputed sizes equal to their average
        self.assertEqual(self.rms.get_se_scaling(4), (self.rms._se_scalings[3] + self.rms._se_scalings[6]) / 2)


class TestEnrichmentCalculator(unittest.TestCase):
    ec_m = EnrichmentCalculator.quick_init(GE2, SC, summary_type='mean')
    ec_me = EnrichmentCalculator.quick_init(GE2, SC, summary_type='median')
    data = ec_m.calculate_enrichment(GENE_SETS)
    # Retrieve lists of enrichment statistics and sort them
    padj = []
    pval = []
    means = []
    medians = []
    num_genes = []
    for gene_set in data:
        padj.append(gene_set.padj)
        pval.append(gene_set.pval)
        means.append(gene_set.mean)
        medians.append(gene_set.median)
        num_genes.append(len(gene_set.gene_set.genes))
    padj.sort()
    pval.sort()
    means.sort(reverse=True)
    medians.sort(reverse=True)
    num_genes.sort()

    def test_init(self):
        # Was correct type of RandomStatisticBundleStorage used based on specified summary_type parameter
        self.assertTrue(self.ec_m.storage._summary_type, 'mean')
        self.assertTrue(self.ec_me.storage._summary_type, 'median')

    def test_calculate_pval(self):
        # Is pvalue in expected interval [0,1]
        pval_m = self.ec_m.calculate_pval(GENE_SET).pval
        pval_me = self.ec_me.calculate_pval(GENE_SET).pval
        self.assertGreaterEqual(pval_m, 0)
        self.assertGreaterEqual(pval_me, 0)
        self.assertLessEqual(pval_m, 1)
        self.assertLessEqual(pval_me, 1)

    def test_calculate_enrichment(self):
        # Is there expected number of results based on GENE_SETS - 5 out of 6 tested as
        # GENE_SET6 does not have enough genes specified in GE
        self.assertEqual(len(self.data), 5)
        # Is padj in expected interval [0,1] and smaller than pvalue
        for gene_set in self.data:
            self.assertLessEqual(gene_set.pval, gene_set.padj)
            self.assertLessEqual(gene_set.padj, 1)
            self.assertGreaterEqual(gene_set.padj, 0)

    def test_filter_enrichment_data_padj(self):
        # Are only padj values below or equal to filtering p_value present in filtered data.
        # Use min padj for filtering to ensure that also padjs equal to filtering padj are retained
        padj_filter = min(self.padj)
        filtered = self.ec_m.filter_enrichment_data_padj(self.data, padj_filter)
        for filtered_set in filtered:
            self.assertLessEqual(filtered_set.padj, padj_filter)

    def test_sort_padj(self):
        # Is list of sorted padj equal to list of padj obtained from sorted GeneSetData
        data_sorted = EnrichmentCalculator.sort_padj(self.data)
        padj2 = []
        for set_sorted in data_sorted:
            padj2.append(set_sorted.padj)
        self.assertListEqual(self.padj, padj2)

    def test_sort_pval(self):
        # Is list of sorted pvalues equal to list of pvalues obtained from sorted GeneSetData
        data_sorted = EnrichmentCalculator.sort_pval(self.data)
        pval2 = []
        for set_sorted in data_sorted:
            pval2.append(set_sorted.pval)
        self.assertListEqual(self.pval, pval2)

    def test_sort_mean(self):
        # Is list of sorted similarity means equal to list of means obtained from sorted GeneSetData,
        # largest to smallest
        data_sorted = EnrichmentCalculator.sort_mean(self.data)
        means2 = []
        for set_sorted in data_sorted:
            means2.append(set_sorted.mean)
        self.assertListEqual(self.means, means2)

    def test_sort_median(self):
        # Is list of sorted similarity medians equal to list of medians obtained from sorted GeneSetData,
        # largest to smallest
        data_sorted = EnrichmentCalculator.sort_median(self.data)
        medians2 = []
        for set_sorted in data_sorted:
            medians2.append(set_sorted.median)
        self.assertListEqual(self.medians, medians2)

    def test_sort_n_genes(self):
        # Is list of sorted gene set sizes equal to list of gene set sizes obtained from sorted GeneSetData
        data_sorted = EnrichmentCalculator.sort_n_genes(self.data)
        num_genes2 = []
        for set_sorted in data_sorted:
            num_genes2.append(len(set_sorted.gene_set.genes))
        self.assertListEqual(self.num_genes, num_genes2)


if __name__ == '__main__':
    unittest.main()
