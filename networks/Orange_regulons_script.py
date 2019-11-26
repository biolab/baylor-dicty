import os
import pickle
import networkx as nx
from statistics import mean, median
import pandas as pd

# Where input and output data is saved (kNN2_m0s1_log_dict.pkl)
path = '/home/karin/Documents/timeTrajectories/Orange_workflows/regulons/'
# File to be filtered
file = 'dict_kNN2_m0s1log.pkl'
# Cosine similarity threshold
threshold = 0.95


# Saves to kNN2_m0s1_log_threshold_Orange.tsv in path directory

def filter_similarities(results: dict, similarity_threshold: float) -> dict:
    result = dict(filter(lambda elem: elem[1] >= similarity_threshold, results.items()))
    genes = set([gene for pair in result.keys() for gene in pair])
    return pd.DataFrame(genes, columns=['Gene'])


def loadPickle(file):
    pkl_file = open(file, 'rb')
    result = pickle.load(pkl_file)
    pkl_file.close()
    return result


splited = file.split('_')
preprocess = splited[(len(splited)) - 1].split('.')[0]

results = loadPickle(path + file)

filtered = filter_similarities(results=results, similarity_threshold=threshold)
print('Retained genes: ', filtered.shape[0])
filtered.to_csv(path + 'kNN2_threshold' + str(threshold) + '_' + preprocess + '_Orange.tsv', sep='\t', index=False)