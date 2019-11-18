import unittest
from unittest import mock
from unittest.mock import patch

from orangecontrib.bioinformatics.geneset.utils import GeneSet

from networks.library_regulons import *

GENES = pd.DataFrame(np.random.rand(100, 20))

# TODO change tests from changed methods
# TODO add missing tests

class TestNeighbourCalculator(unittest.TestCase):
    nc = NeighbourCalculator(GENES)
    params = {'n_neighbours': 2, 'inverse': True, 'scale': 'minmax', 'log': False}

    def test_init(self):
        # Checks that non-numeric data raises error
        with self.assertRaises(ValueError):
            NeighbourCalculator(pd.DataFrame({1: ['a']}, index=['1']))
        with self.assertRaises(ValueError):
            NeighbourCalculator(pd.DataFrame({1: [None]}, index=['1']))
        # Checks that all 0 rows are removed, but all 0 columns are retained
        pd.testing.assert_frame_equal(pd.DataFrame([[0, 1]]),
                                      NeighbourCalculator(pd.DataFrame([[0, 1], [0, 0], [0, 0]]))._genes)

    def test_neighbours(self):

        # Raises error when wrong scale is given
        with self.assertRaises(ValueError):
            self.nc.neighbours(scale='a', n_neighbours=2, inverse=True, log=False)

        # Arguments are passed correctly
        self.calculate_neighbours = self.nc.calculate_neighbours
        self.nc.calculate_neighbours = mock.MagicMock()
        # Without batches
        self.nc.neighbours(batches=None, **self.params)
        # Mock comparison does not work on DataFrames that are not same objects
        self.nc.calculate_neighbours.assert_called_once_with(genes=self.nc._genes, **self.params)
        # With batches (2 - each half of GENES)
        self.nc.calculate_neighbours.reset_mock()
        self.nc.neighbours(batches=['a'] * 10 + ['b'] * 10, **self.params)
        calls = self.nc.calculate_neighbours.call_args_list
        gene_batches = []
        for call in calls:
            args = call[1]
            # Compare all arguments but pd.dataFrame (genes)
            for key in self.params.keys():
                self.assertEqual(args[key], self.params[key])
            gene_batches.append(args['genes'])
        # Assert data frames used in calls were correct
        self.assertTrue(len(gene_batches) == 2)
        self.assertTrue(
            self.nc._genes.iloc[:, 10:].equals(gene_batches[0]) or self.nc._genes.iloc[:, :10].equals(gene_batches[0]))
        self.assertTrue(
            self.nc._genes.iloc[:, 10:].equals(gene_batches[1]) or self.nc._genes.iloc[:, :10].equals(gene_batches[1]))
        self.assertFalse(gene_batches[0].equals(gene_batches[1]))

        self.nc.calculate_neighbours = self.calculate_neighbours

        # TODO possibly check what is returned

    def test_calculate_neighbours(self):
        self.get_index_query = self.nc.get_index_query
        self.parse_neighbours = self.nc.parse_neighbours
        self.nc.get_index_query = mock.MagicMock(return_value=(GENES.values, GENES.values))
        self.nc.parse_neighbours = mock.MagicMock()
        self.nc.calculate_neighbours(GENES, **self.params)
        # Test passing of params to get_index_query
        params_index_querry = self.params.copy()
        del params_index_querry['n_neighbours']
        self.nc.get_index_query.assert_called_once_with(genes=GENES, **params_index_querry)
        self.nc.parse_neighbours.assert_called_once()
        # Implicitly test passing of params to parse_neighbours and knn search based on data shapes.
        args = self.nc.parse_neighbours.call_args[1]
        neighbours = args['neighbours']
        distances = args['distances']
        self.assertEqual(neighbours.shape, (GENES.shape[0], self.params['n_neighbours']))
        self.assertEqual(distances.shape, (GENES.shape[0], self.params['n_neighbours']))
        self.nc.get_index_query = self.get_index_query
        self.nc.parse_neighbours = self.parse_neighbours

    def test_get_index_query(self):
        # Check that scaling is done correctly
        df = pd.DataFrame([[0, 1, 2]])
        index, query = NeighbourCalculator.get_index_query(df, inverse=False, log=False, scale='minmax')
        self.assertListEqual([0, 0.5, 1], index.flatten().tolist())
        self.assertListEqual([0, 0.5, 1], query.flatten().tolist())
        index, query = NeighbourCalculator.get_index_query(df, inverse=True, log=False, scale='minmax')
        self.assertListEqual([0, 0.5, 1], index.flatten().tolist())
        self.assertListEqual([1, 0.5, 0], query.flatten().tolist())
        df = pd.DataFrame([[0, 1, 3]])
        index, query = NeighbourCalculator.get_index_query(df, inverse=False, log=True, scale='minmax')
        self.assertListEqual([0, 0.5, 1], index.flatten().tolist())
        self.assertListEqual([0, 0.5, 1], query.flatten().tolist())
        index, query = NeighbourCalculator.get_index_query(df, inverse=True, log=True, scale='minmax')
        self.assertListEqual([0, 0.5, 1], index.flatten().tolist())
        self.assertListEqual([1, 0.5, 0], query.flatten().tolist())
        df = pd.DataFrame([[0, 1, 2]])
        index, query = NeighbourCalculator.get_index_query(df, inverse=False, log=False, scale='mean0std1')
        self.assertTrue((np.array([-1.22474487, 0, 1.22474487]).round(5) == index.flatten().round(5)).all())
        self.assertTrue((np.array([-1.22474487, 0, 1.22474487]).round(5) == query.flatten().round(5)).all())
        index, query = NeighbourCalculator.get_index_query(df, inverse=True, log=False, scale='mean0std1')
        self.assertTrue((np.array([-1.22474487, 0, 1.22474487]).round(5) == index.flatten().round(5)).all())
        self.assertTrue((np.array([1.22474487, 0, -1.22474487]).round(5) == query.flatten().round(5)).all())
        df = pd.DataFrame([[0, 1, 3]])
        index, query = NeighbourCalculator.get_index_query(df, inverse=False, log=True, scale='mean0std1')
        self.assertTrue((np.array([-1.22474487, 0, 1.22474487]).round(5) == index.flatten().round(5)).all())
        self.assertTrue((np.array([-1.22474487, 0, 1.22474487]).round(5) == query.flatten().round(5)).all())
        index, query = NeighbourCalculator.get_index_query(df, inverse=True, log=True, scale='mean0std1')
        self.assertTrue((np.array([-1.22474487, 0, 1.22474487]).round(5) == index.flatten().round(5)).all())
        self.assertTrue((np.array([1.22474487, 0, -1.22474487]).round(5) == query.flatten().round(5)).all())

    def test_minmax_scale(self):
        # Test that scaling was done by row
        df = pd.DataFrame([[0, 1, 2]])
        scaled = NeighbourCalculator.minmax_scale(df)
        self.assertListEqual([0, 0.5, 1], scaled.flatten().tolist())

    def test_meanstd_scale(self):
        # Test that scaling was done by row
        df = pd.DataFrame([[0, 1, 2]])
        scaled = NeighbourCalculator.meanstd_scale(df)
        self.assertTrue((np.array([-1.22474487, 0, 1.22474487]).round(5) == scaled.flatten().round(5)).all())

    def test_parse_neighbours(self):
        # Parsing neighbours warns about odd cosine distances
        with warnings.catch_warnings(record=True) as w:
            self.nc.parse_neighbours(np.array([[1, 2], [2, 1]]), np.array([[-1, 0], [0, 0]]))
            print(len(w), w)
            assert len(w) == 1
            assert "Odd cosine" in str(w[-1].message)
        with warnings.catch_warnings(record=True) as w:
            self.nc.parse_neighbours(np.array([[1, 2], [2, 1]]), np.array([[2, 0], [0, 0]]))
            assert len(w) == 1
            assert "Odd cosine" in str(w[-1].message)

        # Parsing of neighbours' distances to similarity dict is correct
        df = pd.DataFrame([1, 2, 3], index=['a', 'b', 'A'])
        nc = NeighbourCalculator(df)
        parsed = nc.parse_neighbours(np.array([[0, 1], [1, 0], [2, 0]]), np.array([[0, 0], [0, 0], [0, 0.4]]))
        self.assertDictEqual(parsed, {('a', 'b'): 1, ('A', 'a'): 0.6})

    def test_merge_results(self):
        # Merges by mean and filters on threshold (inclusively) first and then min_present
        results = [{1: 1, 2: 2, 3: 3}, {1: 1, 2: 1.5}, {1: 1}]
        self.assertDictEqual(
            NeighbourCalculator.merge_results(results=results, similarity_threshold=1.5, min_present=2),
            {2: 1.75})
        self.assertDictEqual(NeighbourCalculator.merge_results(results=results, similarity_threshold=2, min_present=2),
                             {})
        self.assertDictEqual(NeighbourCalculator.merge_results(results=results, similarity_threshold=1, min_present=2),
                             {2: 1.75, 1: 1})
        self.assertDictEqual(NeighbourCalculator.merge_results(results=results, similarity_threshold=3, min_present=1),
                             {3: 3})

    def test_filter_similarities(self):
        # Filters inclusively above threshold
        self.assertDictEqual(NeighbourCalculator.filter_similarities({1: 1, 2: 2, 3: 3}, 2), {2: 2, 3: 3})


class TestClustering(unittest.TestCase):
    genes = pd.DataFrame([[1, 2, 3], [100, 200, 300], [30, 20, 10], [1, 4, 5]], index=['a', 'b', 'c', 'd'])
    result = {('a', 'b'): 1, ('c', 'b'): 1, ('c', 'd'): 0}
    threshold = 1

    # To test abstract class
    class TestClusteringClass(Clustering):
        def __init__(self, result: dict, genes: pd.DataFrame, threshold: float, inverse: bool, scale: str, log: bool):
            super().__init__(result=result, genes=genes, threshold=threshold, inverse=inverse, scale=scale, log=log)

        def cluster_sizes(self, splitting: float):
            return None

        def get_clusters(self, splitting: float):
            return None

        def get_genes_by_clusters(self, splitting: float, filter_genes: iter = None):
            return None

    tcc = TestClusteringClass(result=result, genes=genes, threshold=threshold, inverse=False,
                              scale='minmax', log=False)

    def test_init(self):
        # TODO change this to test only assignment of values returned from methods called in init (mock them)

        # Test that distance matrix is correct
        # Original gene names (indices) in test data were given alphabetically sorted
        indices_original_order = np.argsort(self.tcc._gene_names_ordered)
        matrix = np.array(self.tcc._distance_matrix).round(7)[indices_original_order].transpose()[
            indices_original_order].transpose()
        np.testing.assert_array_equal(matrix, np.array([[0, 0, 0.8], [0, 0, 0.8], [0.8, 0.8, 0]]))

        # Test that correct genes were retained
        retained = list(self.tcc._gene_names_ordered)
        retained.sort()
        self.assertListEqual(retained, ['a', 'b', 'c'])
        self.assertEqual(self.tcc._n_genes, 3)

    def test_get_genes(self):
        index, query, gene_names = self.tcc.get_genes(result=self.result, genes=self.genes, threshold=1,
                                                      inverse=True, scale='minmax', log=False)
        # TODO Test that right arguments were passed
        # Index and query are returned correctly
        indices_original_order = np.argsort(gene_names)
        index_ordered = np.array(index).round(7)[indices_original_order]
        query_ordered = np.array(query).round(7)[indices_original_order]
        np.testing.assert_array_equal(index_ordered, np.array([[0, 0.5, 1], [0, 0.5, 1], [1, 0.5, 0]]))
        np.testing.assert_array_equal(query_ordered, np.array([[1, 0.5, 0], [1, 0.5, 0], [0, 0.5, 1]]))

        # Correct genes were retained
        retained = list(gene_names)
        retained.sort()
        self.assertListEqual(retained, ['a', 'b', 'c'])

        # Fails if all genes are filtered out
        with self.assertRaises(ValueError):
            self.tcc.get_genes(result=self.result, genes=self.genes, threshold=2,
                               inverse=True, scale='minmax', log=False)

    def test_distances_cosine(self):
        tcc = self.TestClusteringClass(result=self.result, genes=self.genes, threshold=self.threshold, inverse=False,
                                       scale='minmax', log=False)
        # Test that matrix is filled in correctly and that both_directions parameter is used correctly
        tcc._distance_matrix = [[0] * 3 for counter in range(3)]
        tcc.get_distances_cosine(index=np.array([[-1, 0, 1], [1, 0, -1], [1, 0, -1]]),
                                 query=np.array([[-1, 0, 1], [-1, 0, 1], [1, 0, -1]]),
                                 inverse=True, scale='minmax')
        np.testing.assert_array_almost_equal(tcc._distance_matrix, np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]]))
        tcc._distance_matrix = [[0] * 3 for counter in range(3)]
        tcc.get_distances_cosine(index=np.array([[-1, 0, 1], [1, 0, -1], [1, 0, -1]]),
                                 query=np.array([[-1, 0, 1], [-1, 0, 1], [1, 0, -1]]),
                                 inverse=False, scale='minmax')
        np.testing.assert_array_almost_equal(tcc._distance_matrix, np.array([[0, 0, 2], [0, 0, 0], [2, 0, 0]]))
        tcc._distance_matrix = [[0] * 3 for counter in range(3)]
        tcc.get_distances_cosine(index=np.array([[-1, 0, 1], [1, 0, -1], [1, 0, -1]]),
                                 query=np.array([[-1, 0, 1], [-1, 0, 1], [1, 0, -1]]),
                                 inverse=True, scale='a')
        np.testing.assert_array_almost_equal(tcc._distance_matrix, np.array([[0, 0, 2], [0, 0, 0], [2, 0, 0]]))


class TestHierarchicalClustering(unittest.TestCase):
    genes = pd.DataFrame([[1, 2, 3], [100, 200, 300], [30, 20, 10], [1, 4, 5]], index=['a', 'b', 'c', 'd'])
    result = {('a', 'b'): 1, ('c', 'b'): 1, ('c', 'd'): 0}
    threshold = 0

    hcl = HierarchicalClustering(result=result, genes=genes, threshold=threshold, inverse=False,
                                 scale='minmax', log=False)

    def test_init(self):
        # Clusters are as expected
        indices_original_order = np.argsort(self.hcl._gene_names_ordered)
        clusters = np.array(hc.fcluster(self.hcl._hcl, t=2, criterion='maxclust'))[indices_original_order]
        np.testing.assert_array_equal(clusters, [1, 1, 2, 1])

    def test_get_genes_by_clusters(self):
        # Correctly assigns gene names to clusters
        result = self.hcl.get_genes_by_clusters(2)
        self.assertTrue('a' in result[1] and 'b' in result[1] and 'd' in result[1] and 'c' in result[2])
        # Retains only specified gene names in result
        result = self.hcl.get_genes_by_clusters(2, ['a', 'c'])
        self.assertTrue('a' in result[1] and 'b' not in result[1] and 'd' not in result[1] and 'c' in result[2])

    def test_cluster_sizes(self):
        # Cluster sizes are determined correctly
        self.assertSetEqual(set(self.hcl.cluster_sizes(2)), set([3, 1]))


class TestClusterAnalyser(unittest.TestCase):
    gene_names = ['DDB_G0267178', 'DDB_G0267180', 'DDB_G0267182', 'DDB_G0267184']
    genes = pd.DataFrame([[1, 2, 3], [100, 200, 300], [30, 20, 10], [1, 4, 5]], index=gene_names)
    result = {('DDB_G0267178', 'DDB_G0267180'): 1, ('DDB_G0267182', 'DDB_G0267180'): 1,
              ('DDB_G0267182', 'DDB_G0267184'): 0}
    threshold = 0
    max_set_size = 10
    min_set_size = 4

    hcl = HierarchicalClustering(result=result, genes=genes, threshold=threshold, inverse=False,
                                 scale='minmax', log=False)
    ca = ClusteringAnalyser(gene_names=gene_names, organism=44689, max_set_size=max_set_size, min_set_size=min_set_size)

    # TODO test init for correct passing/saving of arguments

    def test_gene_names_entrez(self):
        # Correct EIDs are added to the gene names
        self.assertDictEqual(self.ca._entrez_names, {'8617794': 'DDB_G0267178', '8615783': 'DDB_G0267180',
                                                     '8615779': 'DDB_G0267182', '8615780': 'DDB_G0267184'})

    @patch('networks.library_regulons.load_gene_sets')
    def test_add_gene_annotations(self, mock_load_gene_sets):
        # GO annotation is added to genes and gene sets of wrong size are not used
        gs1 = GeneSet(name='a', genes=['8617794'] + ['.'] * self.max_set_size)
        gs2 = GeneSet(name='b', genes=['8617794', '8615783'] + ['.'] * (self.max_set_size - 2))
        gs3 = GeneSet(name='c', genes=['8617794'] + ['.'] * (self.min_set_size - 1))
        gs4 = GeneSet(name='d', genes=['8617794'] + ['.'] * (self.min_set_size - 2))
        gene_sets = [gs1, gs2, gs3, gs4]
        mock_load_gene_sets.return_value = gene_sets

        self.ca.add_gene_annotations(('KEGG', 'Pathways'))
        ontology = ('X', 'Y')
        self.ca.add_gene_annotations(ontology)
        self.assertDictEqual(self.ca._annotation_dict[ontology], {'DDB_G0267178': ['b', 'c'], 'DDB_G0267180': ['b']})

    def test_init_annotation_evaluation(self):
        # Returns correct annotation dictionary and annotated gene names filtered clustering
        expected_dict = {'DDB_G0267178': ['a'], 'DDB_G0267182': ['b']}
        self.ca._annotation_dict[('mock', 'mock')] = expected_dict
        annotation_dict, clusters = self.ca.init_annotation_evaluation(self.hcl, 2, ('mock', 'mock'))
        self.assertDictEqual(annotation_dict, expected_dict)
        self.assertDictEqual(clusters, {1: ['DDB_G0267178'], 2: ['DDB_G0267182']})

    def test_annotation_ratio(self):
        # Maximal ratio is selected and weighted by cluster size (containing only annotated genes)
        self.ca._annotation_dict[('mock', 'mock')] = {'DDB_G0267178': ['a', 'b'], 'DDB_G0267180': ['a'],
                                                      'DDB_G0267182': ['b']}
        self.assertAlmostEqual(self.ca.annotation_ratio(self.hcl,2,('mock', 'mock')),1)
        self.ca._annotation_dict[('mock', 'mock')] = {'DDB_G0267178': ['b'], 'DDB_G0267180': ['a'],
                                                      'DDB_G0267182': ['b']}
        self.assertAlmostEqual(self.ca.annotation_ratio(self.hcl, 2, ('mock', 'mock')), (0.5*2/3 + 1/3))


class TestOther(unittest.TestCase):

    def test_build_graph(self):
        # Edges and weights are correctly added
        d = {(1, 2): 1, (2, 3): 2}
        g = build_graph(d)
        self.assertListEqual(list(g.edges(data=True)), [(1, 2, {'weight': 1}), (2, 3, {'weight': 2})])

    def test_calc_cosine(self):
        # Right metric (sim/dist) is used and results is based only on m1[i1],m2[i2] or
        # on average of m1[i1],m2[i2] and m1[i2],m2[i1]
        data1 = [-1, 0, 1], [1, 0, -1], [1, 0, -1]
        data2 = [-1, 0, 1], [-1, 0, 1], [1, 0, -1]
        self.assertAlmostEqual(calc_cosine(data1=data1, data2=data2, index1=1, index2=2, sim_dist=True,
                                           both_directions=False), 1)
        self.assertAlmostEqual(calc_cosine(data1=data1, data2=data2, index1=1, index2=2, sim_dist=True,
                                           both_directions=True), 0)
        self.assertAlmostEqual(calc_cosine(data1=data1, data2=data2, index1=1, index2=2, sim_dist=False,
                                           both_directions=False), 0)
        self.assertAlmostEqual(calc_cosine(data1=data1, data2=data2, index1=1, index2=2, sim_dist=False,
                                           both_directions=True), 1)


if __name__ == '__main__':
    unittest.main()
