import unittest
from unittest import mock

from networks.library_regulons import *

GENES = pd.DataFrame(np.random.rand(100, 20))


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
        self.calculate_neighbours=self.nc.calculate_neighbours
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

        self.nc.calculate_neighbours=self.calculate_neighbours

        # TODO possibly check what is returned

    def test_calculate_neighbours(self):
        self.get_index_query=self.nc.get_index_query
        self.parse_neighbours = self.nc.parse_neighbours
        self.nc.get_index_query = mock.MagicMock(return_value=(GENES.values,GENES.values))
        self.nc.parse_neighbours= mock.MagicMock()
        self.nc.calculate_neighbours(GENES,**self.params)
        # Test passing of params to get_index_query
        params_index_querry=self.params.copy()
        del params_index_querry['n_neighbours']
        self.nc.get_index_query.assert_called_once_with(genes=GENES,**params_index_querry)
        self.nc.parse_neighbours.assert_called_once()
        # Implicitly test passing of params to parse_neighbours and knn search based on data shapes.
        args=self.nc.parse_neighbours.call_args[1]
        neighbours=args['neighbours']
        distances = args['distances']
        self.assertEqual(neighbours.shape,(GENES.shape[0],self.params['n_neighbours']))
        self.assertEqual(distances.shape,(GENES.shape[0],self.params['n_neighbours']))
        self.nc.get_index_query=self.get_index_query
        self.nc.parse_neighbours = self.parse_neighbours

    def test_get_index_query(self):
        # Check that scaling is done correctly
        df=pd.DataFrame([[0,1,2]])
        index,query=NeighbourCalculator.get_index_query(df,inverse=False,log=False,scale='minmax')
        self.assertListEqual([0,0.5,1],index.flatten().tolist())
        self.assertListEqual([0, 0.5, 1], query.flatten().tolist())
        index, query = NeighbourCalculator.get_index_query(df, inverse=True, log=False, scale='minmax')
        self.assertListEqual([0, 0.5, 1], index.flatten().tolist())
        self.assertListEqual([1, 0.5, 0], query.flatten().tolist())
        df = pd.DataFrame([[0, 1,3]])
        index, query = NeighbourCalculator.get_index_query(df, inverse=False, log=True, scale='minmax')
        self.assertListEqual([0,0.5,1], index.flatten().tolist())
        self.assertListEqual([0,0.5,1], query.flatten().tolist())
        index, query = NeighbourCalculator.get_index_query(df, inverse=True, log=True, scale='minmax')
        self.assertListEqual([0,0.5,1], index.flatten().tolist())
        self.assertListEqual([1,0.5,0], query.flatten().tolist())
        df=pd.DataFrame([[0,1,2]])
        index,query=NeighbourCalculator.get_index_query(df,inverse=False,log=False,scale='mean0std1')
        self.assertTrue((np.array([-1.22474487,  0 ,  1.22474487]).round(5)== index.flatten().round(5)).all())
        self.assertTrue((np.array([-1.22474487,  0 ,  1.22474487]).round(5)== query.flatten().round(5)).all())
        index, query = NeighbourCalculator.get_index_query(df, inverse=True, log=False, scale='mean0std1')
        self.assertTrue((np.array([-1.22474487,  0 ,  1.22474487]).round(5) == index.flatten().round(5)).all())
        self.assertTrue((np.array([1.22474487,  0 ,  -1.22474487]).round(5) == query.flatten().round(5)).all())
        df = pd.DataFrame([[0, 1,3]])
        index, query = NeighbourCalculator.get_index_query(df, inverse=False, log=True, scale='mean0std1')
        self.assertTrue((np.array([-1.22474487,  0 ,  1.22474487]).round(5)== index.flatten().round(5)).all())
        self.assertTrue((np.array([-1.22474487,  0 ,  1.22474487]).round(5)== query.flatten().round(5)).all())
        index, query = NeighbourCalculator.get_index_query(df, inverse=True, log=True, scale='mean0std1')
        self.assertTrue((np.array([-1.22474487,  0 ,  1.22474487]).round(5) == index.flatten().round(5)).all())
        self.assertTrue((np.array([1.22474487,  0 ,  -1.22474487]).round(5) == query.flatten().round(5)).all())

    def test_minmax_scale(self):
        # Test that scaling was done by row
        df = pd.DataFrame([[0, 1, 2]])
        scaled= NeighbourCalculator.minmax_scale(df)
        self.assertListEqual([0, 0.5, 1], scaled.flatten().tolist())

    def test_meanstd_scale(self):
        # Test that scaling was done by row
        df = pd.DataFrame([[0, 1, 2]])
        scaled= NeighbourCalculator.meanstd_scale(df)
        self.assertTrue((np.array([-1.22474487,  0 ,  1.22474487]).round(5)== scaled.flatten().round(5)).all())

    def test_parse_neighbours(self):
        # Parsing neighbours warns about odd cosine distances
        with warnings.catch_warnings(record=True) as w:
            self.nc.parse_neighbours(np.array([[1,2],[2,1]]),np.array([[-1,0],[0,0]]))
            print(len(w),w)
            assert len(w) == 1
            assert "Odd cosine" in str(w[-1].message)
        with warnings.catch_warnings(record=True) as w:
            self.nc.parse_neighbours(np.array([[1,2],[2,1]]),np.array([[2,0],[0,0]]))
            assert len(w) == 1
            assert "Odd cosine" in str(w[-1].message)

        # Parsing of neighbours' distances to similarity dict is correct
        df=pd.DataFrame([1,2,3],index=['a','b','A'])
        nc=NeighbourCalculator(df)
        parsed=nc.parse_neighbours(np.array([[0,1],[1,0],[2,0]]) , np.array([[0,0],[0,0],[0,0.4]]))
        self.assertDictEqual(parsed,{('a','b'):1,('A','a'):0.6})

    def test_merge_results(self):
        # Merges by mean and filters on threshold (inclusively) first and then min_present
        results=[{1:1,2:2,3:3},{1:1,2:1.5},{1:1}]
        self.assertDictEqual( NeighbourCalculator.merge_results(results=results,similarity_threshold=1.5,min_present=2),
                              {2:1.75})
        self.assertDictEqual( NeighbourCalculator.merge_results(results=results,similarity_threshold=2,min_present=2),
                              {})
        self.assertDictEqual(NeighbourCalculator.merge_results(results=results, similarity_threshold=1, min_present=2),
                             {2: 1.75,1:1})
        self.assertDictEqual(NeighbourCalculator.merge_results(results=results, similarity_threshold=3, min_present=1),
                             {3:3})

    def test_filter_similarities(self):
        # Filters inclusively above threshold
        self.assertDictEqual(NeighbourCalculator.filter_similarities({1:1,2:2,3:3},2),{2:2,3:3})


class TestOther(unittest.TestCase):

    def test_build_graph(self):
        # Edges and weights are correctly added
        d = {(1, 2): 1, (2, 3): 2}
        g = build_graph(d)
        self.assertListEqual(list(g.edges(data=True)),[(1, 2, {'weight': 1}), (2, 3, {'weight': 2})])

if __name__ == '__main__':
    unittest.main()
