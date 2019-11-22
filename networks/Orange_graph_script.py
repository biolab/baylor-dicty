import os
import pickle
import networkx as nx
from statistics import mean, median

path = '/home/karin/Documents/timeTrajectories/data/regulons/inverseReplicate/'
# Min 0.9
threshold = 0.97
# Min 10
min_present = 25
# Merge by mean (average retained similarities over replicates), sum (sum retained similarities over replicates) or len (N of replicates where pair was present at specified threshold)
merge_by = len


def build_graph(similarities: dict) -> nx.Graph:
    graph = nx.Graph()
    for pair, similarity in similarities.items():
        graph.add_edge(pair[0], pair[1], weight=similarity)
    return graph


def filter_similarities_batched(results: dict, similarity_threshold: float = 0, min_present: int = 1,
                                merge_function=mean) -> dict:
    retained = {}
    for pair, similarities in results.items():
        retained_similarities = [item for item in similarities if item >= similarity_threshold]
        if len(retained_similarities) >= min_present:
            retained[pair] = merge_function(retained_similarities)
    return retained


def loadPickle(file):
    pkl_file = open(file, 'rb')
    result = pickle.load(pkl_file)
    pkl_file.close()
    return result


results_all = loadPickle(path + 'merged_T0_9_min10.pkl')

filtered = filter_similarities_batched(results=results_all, similarity_threshold=threshold,
                                       min_present=min_present,
                                       merge_function=merge_by)
graph = build_graph(filtered)
print('Nodes', graph.number_of_nodes(), ', edges:', graph.number_of_edges())
nx.write_pajek(graph, path + 'kNN200_inverse_threshold' + str(threshold) + '_minPresentReplicates' + str(
    min_present) + '_mergeBy' + merge_by.__name__ + '_Orange.net')
