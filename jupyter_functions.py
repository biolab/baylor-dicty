import random
from networks.library_regulons import *

def sample_from_list(available,sample1:int):
    """
    Separate list in 2 pairts, one of them being of size sample1
    :param availiable: list
    :param sample1: sample1 size
    :return: sample1, sample2
    """
    if sample1>len(available):
        raise ValueError('Sample size is greater than length of input, max: ',len(available))
    sub1=random.sample(available,sample1)
    sub2=available.copy()
    for sub in sub1:
        sub2.remove(sub)
    return sub1,sub2


def obtained_genes_by_n_neighbours(neighbour_calculator:NeighbourCalculator, neighbours_size_larger:int,
                                   neighbours_sizes_smaller:list, thresholds:list)->pd.DataFrame:
    """
    Compare N of obtained genes based on threshold and N computed neighbours
    :param neighbour_calculator:
    :param neighbours_size_larger: N neighbours to use as test, larger number
    :param neighbours_sizes_smaller: N neighbours to compare against the larger number
    :param thresholds: similarity threshold for gene retantion
    :return: parameters, genes retained in both, only in smaller, only in larger
    """
    inverse = False
    neighbours_larger_all = neighbour_calculator.neighbours(n_neighbours=neighbours_size_larger, inverse=inverse)
    results = []
    for n_neighbours in neighbours_sizes_smaller:
        neighbours_smaller_all = neighbour_calculator.neighbours(n_neighbours, inverse=inverse)
        for threshold in thresholds:
            neighbours_smaller = NeighbourCalculator.filter_similarities(neighbours_smaller_all, threshold)
            neighbours_smaller = set((gene for pair in neighbours_smaller.keys() for gene in pair))
            neighbours_larger = NeighbourCalculator.filter_similarities(neighbours_larger_all, threshold)
            neighbours_larger = set((gene for pair in neighbours_larger.keys() for gene in pair))
            match = neighbours_smaller & neighbours_larger
            in_both = len(match)
            in_smaller = len(neighbours_smaller ^ match)
            in_larger = len(neighbours_larger ^ match)
            results.append(
                {'n_neighbours': n_neighbours, 'threshold': threshold, 'intersect': in_both, 'unique_smaller': in_smaller,
                 'unique_larger': in_larger})
    return pd.DataFrame(results)