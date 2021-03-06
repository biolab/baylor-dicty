import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
from statistics import median, mean
from scipy.cluster.hierarchy import dendrogram
import pickle as pkl
import glob
import seaborn as sb
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

from Orange.clustering.louvain import jaccard

from networks.library_regulons import *
import jupyter_functions as jf
from networks.functionsDENet import loadPickle, savePickle

# Script parts are to be run separately as needed
lab = True
if lab:
    # dataPath = '/home/karin/Documents/timeTrajectories/data/RPKUM/combined_mt/combined/'
    dataPath = '/home/karin/Documents/timeTrajectories/data/RPKUM/combined/'
    dataPathSaved = '/home/karin/Documents/timeTrajectories/data/regulons/'
    path_inverse = '/home/karin/Documents/timeTrajectories/data/regulons/inverseReplicate_m0s1log/'
    pathSelGenes = dataPathSaved + 'selected_genes/'
    pathMergSim = dataPathSaved + 'clusters/merge_similarity/'
    pathByStrain = '/home/karin/Documents/timeTrajectories/data/regulons/by_strain/'
else:
    dataPath = '/home/karin/Documents/DDiscoideum/data/RPKUM/'
    dataPathSaved = '/home/karin/Documents/DDiscoideum/data/regulons/'
    path_inverse = '/home/karin/Documents/DDiscoideum/data/regulons/inverse/'

genes = pd.read_csv(dataPath + 'mergedGenes_RPKUM.tsv', sep='\t', index_col=0)
conditions = pd.read_csv(dataPath + 'conditions_mergedGenes.tsv', sep='\t', index_col=None)

# ********** Average AX4 samples form new data
# Split to AX4 and rest of data
genes_conditions = ClusterAnalyser.merge_genes_conditions(genes=genes, conditions=conditions, matching='Measurment')
conditions_AX4 = conditions.loc[conditions['Strain'] == 'AX4', :]
conditions_rest = conditions.loc[conditions['Strain'] != 'AX4', :]
# Select AX4/rest genes by conditions rows so that they are both ordered the same
genes_AX4 = genes_conditions.loc[conditions_AX4['Measurment'], :]
genes_rest = genes_conditions.loc[conditions_rest['Measurment'], :]

# This is dropped in AX4 while averaging
genes_rest = genes_rest.drop(['Time', 'Strain', 'Replicate', 'Measurment'], axis=1)

# Group and average AX4, add averaged to rest
groupped_AX4 = genes_AX4.set_index('Replicate').groupby(
    {'AX4_FD_r1': 'AX4_FD', 'AX4_FD_r2': 'AX4_FD', 'AX4_PE_r3': 'AX4_PE',
     'AX4_PE_r4': 'AX4_PE', 'AX4_SE_r5': 'AX4_SE', 'AX4_SE_r6': 'AX4_SE',
     'AX4_SE_r7': 'AX4_SE'})
for name, group in groupped_AX4:
    group = group.groupby('Time').mean()
    times = list(group.index)
    group.index = [str(name) + '_' + str(idx) for idx in times]
    conditions_rest = conditions_rest.append(
        pd.DataFrame({'Measurment': group.index, 'Strain': ['AX4'] * group.shape[0],
                      'Replicate': [name] * group.shape[0], 'Time': times}), ignore_index=True, sort=True)
    genes_rest = genes_rest.append(group, sort=True)

genes_rest = genes_rest.T
# Check that ordered the same
if not (genes_rest.columns.values == conditions_rest['Measurment'].values).all():
    print('Genes and conditions are not ordered the same')

genes = genes_rest
conditions = conditions_rest
# ************************************

sub = genes.shape[1]
genes_sub = genes.iloc[:, :sub]
neighbour_calculator = NeighbourCalculator(genes)

neighbours_minmax = neighbour_calculator.neighbours(10, False, scale='minmax', log=True)
neighbours_meanstd = neighbour_calculator.neighbours(10, False, scale='mean0std1', log=True)
# Plot how it looks to have similarity above threshold:
neighbours = neighbours_minmax
sim_threshold = 0.90
step = 0.01
max_plot = 1
for gene_pair, sim in neighbours.items():
    if max_plot <= 9:
        if sim_threshold <= sim < sim_threshold + step:
            sim_minmax = 'NA'
            if gene_pair in neighbours_minmax.keys():
                sim_minmax = round(neighbours_minmax[gene_pair], 3)
            sim_meanstd = 'NA'
            # if gene_pair in neighbours_meanstd.keys():
            #    sim_meanstd = round(neighbours_meanstd[gene_pair],3)
            plt.subplot(3, 3, max_plot)
            plt.title('minmax ' + str(sim_minmax)
                      # +', mean0std1 '+str(sim_meanstd)
                      , fontdict={'size': 8})
            print(gene_pair)
            gene1 = pp.minmax_scale(list(genes.loc[gene_pair[0], :].iloc[:30]))
            gene2 = pp.minmax_scale(list(genes.loc[gene_pair[1], :].iloc[:30]))
            plt.plot(list(range(30)), gene1, color='g', linewidth=1)
            plt.plot(list(range(30)), gene2, color='m', linewidth=1)
            max_plot += 1

# Based on plots with minmax, log, all genes, decided that will start comparing thresholds at 0.9

# Plot distribution of similarities
plt.hist(neighbours.values())

# Compare difference in N of detected pairs at threshold (min) depending on N of computed neighbours
neighbour_calculator = NeighbourCalculator(genes)
inverse = False
scale = 'minmax'
use_log = True
result0 = np.array(list(neighbour_calculator.neighbours(30, inverse, scale=scale, log=use_log).values()))
result1 = np.array(list(neighbour_calculator.neighbours(310, inverse, scale=scale, log=use_log).values()))
result2 = np.array(list(neighbour_calculator.neighbours(400, inverse, scale=scale, log=use_log).values()))

# What percent of shorter result's values is missing
threshold = 0.95
resultA = result0
resultB = result1
(sum(resultB >= threshold) - sum(resultA >= threshold)) / len(resultA)

thresholds = [0.8, 0.85, 0.9, 0.92, 0.94, 0.95, 0.97, 0.98, 0.99]
differences = []
for threshold in thresholds:
    differences.append((sum(result1 >= threshold) - sum(result0 >= threshold)) / len(result0))
plt.scatter(thresholds, differences)

# On all genes:
# For neighbours (result0) 20,(result1) 30, (result2) 40, scale='minmax, log=False, inverse=False
# result0 vs result1 0.0102, result0 vs result01=1.004e-05, result1 vs result2: 0.00299, result1 vs result11: -3.320e-05
# For :neighbours 90, 100, scale='minmax, log=True, inverse=False
# thresholds and differences (knee at 0.95): {0.8: 0.0919429801303846,
# 0.85: 0.08757603169156868,
# 0.9: 0.0797643927342296,
# 0.92: 0.0747995882591472,
# 0.94: 0.06614783160215021,
# 0.95: 0.05897147966769602,
# 0.97: 0.030408517628954947,
# 0.98: 0.011565135115464196,
# 0.99: 0.0003566341225033012}
# When neighbours 250 and 300:
# {0.8: 0.15846030821583645,
# 0.85: 0.14988588269014977,
# 0.9: 0.13441715555935016,
# 0.92: 0.12286287322535006,
# 0.94: 0.10321746780016587,
# 0.95: 0.08689571949553593,
# 0.97: 0.03140208018246573,
# 0.98: 0.0075952425964775335,
# 0.99: 0.0}
# For 300 and 310:
# {0.8: 0.026306523784654787,
# 0.85: 0.02486818697868655,
# 0.9: 0.022217109835209428,
# 0.92: 0.020215100497172556,
# 0.94: 0.01687098335525039,
# 0.95: 0.01402936004208585,
# 0.97: 0.004912777331506044,
# 0.98: 0.0010464107378820762,
# 0.99: 0.0}
# For 300 and 400:
# {0.8: 0.26040938111687456,
# 0.85: 0.2459931932319785,
# 0.9: 0.21868104451164896,
# 0.92: 0.19832669873662792,
# 0.94: 0.16383019468529134,
# 0.95: 0.1344312786317128,
# 0.97: 0.04437049185446988,
# 0.98: 0.007848717812275061,
# 0.99: 0.0}

# Check how often is closest (non self) neighbour different if different N of sought neighbours
neighbour_calculator = NeighbourCalculator(genes)
inverse = False
scale = 'minmax'
use_log = True
index, query = neighbour_calculator.get_index_query(genes, inverse, scale=scale, log=use_log)
index = NNDescent(index, metric='cosine', n_jobs=4)
neighbours2, distances2 = index.query(query.tolist(), k=30)
for n_neighbours in [2, 3, 5, 10]:
    neighbours1, distances1 = index.query(query.tolist(), k=5)
    match = 0
    differ = 0
    for gene in range(neighbours1.shape[0]):
        closest1 = neighbours1[gene][0]
        if closest1 == gene:
            closest1 = neighbours1[gene][1]
        closest2 = neighbours2[gene][0]
        if closest2 == gene:
            closest2 = neighbours2[gene][1]
        if closest2 == closest1:
            match += 1
        else:
            differ += 1
    print(n_neighbours, round(differ / (match + differ), 5))


# Test Setting of parameters

def compare_conditions(genes, conditions, neighbours_n, inverse, scale, use_log, thresholds, filter_column,
                       filter_column_values_sub,
                       filter_column_values_test, batch_column=None):
    """
    Evaluates pattern similarity calculation preprocessing and parameters based on difference between subset and test set.
    Computes MSE from differences between similarities of subset gene pairs and corresponding test gene pairs.
    :param genes: data frame with gene expression data, genes in rows, measurments in columns, dimension G*M
    :param conditions: data frame with conditions for genes subseting, measurments in rows,rows should have same order as
        genes table columns, dimensions M*D (D are description types)
    :param neighbours_n: N of calculated neighbours for each gene
    :param inverse: find neighbours with opposite profile
    :param scale: 'minmax' (from 0 to 1) or 'mean0std1' (to mean 0 and std 1)
    :param use_log: Log transform expression values before scaling
    :param thresholds: filter out any result with similarity below threshold, do for each threshold
    :param filter_column: On which column of conditions should genes be subset for separation in subset and test set
    :param filter_column_values_sub: Values of filter_column to use for subset genes
    :param filter_column_values_test:Values of filter_column to use for test genes
    :param batch_column: Should batches be used based on some column of conditions
    :return: Dictionary with parameters and results, description as key, result/parameter setting as value
    """
    # Prepare data
    if not batch_column:
        batches = None
    else:
        list((conditions[conditions[filter_column].isin(filter_column_values_sub)].loc[:, batch_column]))
    genes_sub = genes.T[list(conditions[filter_column].isin(filter_column_values_sub))].T
    genes_test = genes.T[list(conditions[filter_column].isin(filter_column_values_test))].T

    neighbour_calculator = NeighbourCalculator(genes_sub)
    test_index, test_query = NeighbourCalculator.get_index_query(genes_test, inverse=inverse, scale=scale, log=use_log)
    gene_names = list(genes_test.index)

    # Is similarity matrix expected to be simetric or not
    both_directions = False
    if inverse and (scale == 'minmax'):
        both_directions = True

    # Calculate neighbours
    result = neighbour_calculator.neighbours(neighbours_n, inverse=inverse, scale=scale, log=use_log,
                                             batches=batches)

    # Filter neighbours on similarity
    data_summary = []
    for threshold in thresholds:
        if batches != None:
            result_filtered = neighbour_calculator.merge_results(result.values(), threshold, len(set(batches)))
        else:
            result_filtered = NeighbourCalculator.filter_similarities(result, threshold)

        # Calculate MSE for each gene pair -
        # compare similarity from gene subset to similarity of the gene pair in gene test set
        sq_errors = []
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
                             'N pairs': len(result_filtered), 'N genes': n_genes})
    return data_summary


# Result:
# Sample: comH_r1, cudA_r2, gbfA_r1,mybBGFP_bio1,pkaR_bio1 and test sample: AX4_bio1; not inverse: threshold 0.95; neighbours 30
# minmax, no log: MSE 0.01615798890022193, filtered N: 57921
# minmax, log: MSE 0.008855992300398607, filtered: 191347
# mean0std1, no log: MSE 0.10903301209390322, filtered: 14100
# mean0std1, log: MSE 0.008840844803079855, filtered: 6217
# mean0std1, log, threshold 0.935: MSE 0.008617332426242763, filtered: 14239
# Log on mean0std1 does not seem to perform better because of lesser number of retained pairs (eg. possibly more correlated ones,
# as shown when lowering filtering threshold for log based calculations

# minmax, log, batches: MSE 0.0005413249414331597, filtered: 243
# Try to lower threshold for merging to obtain similar number of results: However, even when lowering the threshold
# substantially (eg. to max of lowest similarity of individual strains - as pair must be present in all strains)
# the number of filtered pairs remained at 270

# Compare result by using batches on samples or strains
# Sample strains: comH, cudA, gbfA,mybBGFP,pkaR and test sample: AX4_bio1; not inverse: threshold 0.95; neighbours 30
# minmax, log ,not inverse, 30 neighbours, threshold 0.95
# By strain: 0.0006956959575191145 559, when lowering merging threshold to max of any result's min similarity there is 762 pairs
# By replicate: 6.592793901930816e-05 41, when lowering the merging threshold as above got to 54 pairs

# Use the 5 above defined replicates and compare MSE at different thresholds, minmax, log, not inverse, 300 neighbours, no batches
# threshold: 0.99 MSE: 0.004401159883607221 N: 27422
# threshold: 0.98 MSE: 0.008960403278511933 N: 235707
# threshold: 0.97 MSE: 0.010837772836915213 N: 554577
# threshold: 0.95 MSE: 0.012256426645819427 N: 1246669
# threshold: 0.93 MSE: 0.012841511138263213 N: 1788086
# threshold: 0.91 MSE: 0.013112523538440932 N: 2091354

# Make table of results for different condition settings
results = []
# Which parameters to try
thresholds = [0.95]
for scale in ['minmax', 'mean0std1']:
    for use_log in [True, False]:
        for inverse in [False, True]:
            for n_neighbours in [30]:
                result = compare_conditions(genes, conditions,  # Data
                                            # Parameters
                                            neighbours_n=n_neighbours, inverse=inverse, scale=scale, use_log=use_log,
                                            thresholds=thresholds,
                                            filter_column='Replicate',
                                            # For subset genes
                                            filter_column_values_sub=['comH_r1', 'cudA_r2', 'gbfA_r1', 'mybBGFP_bio1',
                                                                      'pkaR_bio1'],
                                            # For test genes
                                            filter_column_values_test=['AX4_bio1'],

                                            batch_column=None)
                for res in result:
                    results.append(res)
data_summary = pd.DataFrame(results)
alt.Chart(data_summary).mark_circle().encode(alt.X('threshold', scale=alt.Scale(zero=False)),
                                             alt.Y('MSE', scale=alt.Scale(zero=False))
                                             ).configure_circle(size=50).interactive().serve()

# Check if set of genes with highly correlated neighbours changes upon changing number of obtained neighbours.
# Also check if this changes at different retention thresholds
# Parameters
inverse = False
scale = 'minmax'
use_log = True
# Different relatively small numbers of neighbours used for calculation (neighbours_check) compared to results
# from larger number of neighbours (neighbours_reference)
neighbours_check = [2]
neighbours_reference = 100
thresholds = [0.98, 0.99]

neighbour_calculator = NeighbourCalculator(genes)
neighbours2_all = neighbour_calculator.neighbours(100, inverse, scale=scale, log=use_log)
results = []
for n_neighbours in neighbours_check:
    neighbours1_all = neighbour_calculator.neighbours(n_neighbours, inverse, scale=scale, log=use_log)
    for threshold in [0.98, 0.99]:
        neighbours1 = NeighbourCalculator.filter_similarities(neighbours1_all, threshold)
        neighbours1 = set((gene for pair in neighbours1.keys() for gene in pair))
        neighbours2 = NeighbourCalculator.filter_similarities(neighbours2_all, threshold)
        neighbours2 = set((gene for pair in neighbours2.keys() for gene in pair))
        match = neighbours1 & neighbours2
        in12 = len(match)
        in1 = len(neighbours1 ^ match)
        in2 = len(neighbours2 ^ match)
        results.append(
            {'n_neighbours': n_neighbours, 'threshold': threshold, 'intersect': in12, 'unique1': in1, 'unique2': in2})
pd.DataFrame(results)

# For inverse=False, scale='minmax' and use_log=True:
# Table n_neighbours, threshold, in1&2, only in 1, only in 2
# 2 0.98 5565 0 0
# 2 0.99 1967 0 0
# 3 0.98 5564 0 1
# 3 0.99 1967 0 0
# 5 0.98 5565 0 0
# 5 0.99 1967 0 0
# 10 0.98 5564 0 1
# 10 0.99 1967 0 0
# Thus: Calculation of 2 closest neighbours (for non inverse, as closest may be self) is sufficient
# **************************
# Make regulons

# Set parameters
neighbours_n = 200
threshold = 0.99
scale = 'minmax'
use_log = True
batches = None

# Calculate neighbours and make graphs
neighbour_calculator = NeighbourCalculator(genes)
result = neighbour_calculator.neighbours(neighbours_n, inverse=False, scale=scale, log=use_log, batches=batches)
result_inv = neighbour_calculator.neighbours(neighbours_n, inverse=True, scale=scale, log=use_log, batches=batches)
result_filtered = NeighbourCalculator.filter_similarities(result, threshold)
result_filtered_inv = NeighbourCalculator.filter_similarities(result_inv, threshold)
graph = build_graph(result_filtered)
graph_inv = build_graph(result_filtered_inv)
nx.write_pajek(graph, dataPathSaved + 'kN200_t0.99_scaleMinmax_log.net')
nx.write_pajek(graph_inv, dataPathSaved + 'kN200_t0.99_scaleMinmax_log_inv.net')

# ***********************************************8
# Find clusters based on best connected genes
# Set parameters
# Enough smalle N of neighbours as only interested if there is at least one highly connected neighbour
neighbours_n = 2
threshold = 0.99
scale = 'minmax'
use_log = True
batches = None

# Calculate neighbours and make hierarchical clustering
neighbour_calculator = NeighbourCalculator(genes)
result = neighbour_calculator.neighbours(neighbours_n, inverse=False, scale=scale, log=use_log, batches=batches)
result_inv = neighbour_calculator.neighbours(neighbours_n, inverse=True, scale=scale, log=use_log, batches=batches)
hcl = HierarchicalClustering.from_knn_result(result, genes, threshold, inverse=False, scale=scale, log=use_log)
hcl_inv = HierarchicalClustering.from_knn_result(result_inv, genes, threshold, inverse=True, scale=scale, log=use_log)

hcl_gene_data = genes.loc[hcl._gene_names_ordered, :]

dendrogram(hcl_inv._hcl)

ca = ClusteringAnalyser(genes.index)
silhouettes = []
median_sizes = []
entropy = []
ratios = []
n_clusters = list(range(3, 60, 6))
for n in n_clusters:
    median_sizes.append(median(hcl.cluster_sizes(n)))
    silhouettes.append(ClusteringAnalyser.silhouette(hcl, n))
    entropy.append(ca.annotation_entropy(hcl, n, ('KEGG', 'Pathways')))
    ratios.append(ca.annotation_ratio(hcl, n, ('KEGG', 'Pathways')))

# Taken from: https://matplotlib.org/3.1.1/gallery/ticks_and_spines/multiple_yaxis_with_spines.html
fig, host = plt.subplots()
fig.subplots_adjust(right=0.75)

par1 = host.twinx()
par2 = host.twinx()
par2.spines["right"].set_position(("axes", 1.2))

p11 = host.scatter(n_clusters, silhouettes, c="b")
p1, = host.plot(n_clusters, silhouettes, "b-", label="Silhouette")
p22 = par1.scatter(n_clusters, median_sizes, c="r")
p2, = par1.plot(n_clusters, median_sizes, "r-", label="Median size")
p33 = par2.scatter(n_clusters, ratios, c="g")
p3, = par2.plot(n_clusters, ratios, "g-", label="Annotation ratio")

host.set_xlim(min(n_clusters), max(n_clusters))
host.set_ylim(min(silhouettes), max(silhouettes))
par1.set_ylim(min(median_sizes), max(median_sizes))
par2.set_ylim(min(ratios), max(ratios))

host.set_xlabel('N clusters')
host.set_ylabel('Silhouette values')
par1.set_ylabel('Median cluster size')
par2.set_ylabel('Max annotation ratio')

host.yaxis.label.set_color(p1.get_color())
par1.yaxis.label.set_color(p2.get_color())
par2.yaxis.label.set_color(p3.get_color())

tkw = dict(size=4, width=1.5)
host.tick_params(axis='y', colors=p1.get_color())
par1.tick_params(axis='y', colors=p2.get_color())
par2.tick_params(axis='y', colors=p3.get_color())
host.tick_params(axis='x')

# lines = [p1, p2,p3]

# host.legend(lines, [l.get_label() for l in lines])

# *********************************

# Get genes for orange:
result = get_orange_result(result=None, threshold=0.99, genes=genes)

genes_orange_scaled, genes_orange_avg, patterns = preprocess_for_orange(genes=genes,
                                                                        conditions=conditions,
                                                                        split_by='Strain', average_by='Time',
                                                                        matching='Measurment', group='AX4')
result.to_csv(dataPathSaved + 'genes_selected_orange_T0_99.tsv', sep='\t', index=False)
genes_orange_scaled.to_csv(dataPathSaved + 'genes_scaled_orange.tsv', sep='\t')

zero_replicates=pd.read_table('/home/karin/Documents/timeTrajectories/data/RPKUM/combined/zero_replicates_count.tsv',
                              index_col=0)
zero_AX4=zero_replicates.query('AX4>0').index
patterns.index=patterns['Gene']
patterns=patterns.drop(zero_AX4)
patterns.to_csv(dataPathSaved + 'gene_patternsRemoveRep0_orange.tsv', sep='\t', index=False)

# Transpose so that column names unique (else Orange problems)
genes_orange_avg = genes_orange_avg.T
genes_orange_avg['Time'] = genes_orange_avg.index
genes_orange_avg.index = [strain + '_' + str(time) for strain, time in
                          zip(genes_orange_avg['Group'], genes_orange_avg['Time'])]
genes_orange_avg.to_csv(dataPathSaved + 'genes_averaged_orange.tsv', sep='\t')

# **** Add AX4 PE,SE, FD averaged to genes_orange_avg
# Split to AX4 and rest of data
genes_orange_avg_AX4 = genes_orange_avg.copy()
genes_conditions = ClusterAnalyser.merge_genes_conditions(genes=genes, conditions=conditions, matching='Measurment')
conditions_AX4 = conditions.loc[conditions['Strain'] == 'AX4', :]
# Select AX4/rest genes by conditions rows so that they are both ordered the same
genes_AX4 = genes_conditions.loc[conditions_AX4['Measurment'], :]

# Group and average AX4, add averaged to rest
groupped_AX4 = genes_AX4.set_index('Replicate').groupby(
    {'AX4_FD_r1': 'AX4_FD', 'AX4_FD_r2': 'AX4_FD', 'AX4_PE_r3': 'AX4_PE',
     'AX4_PE_r4': 'AX4_PE', 'AX4_SE_r5': 'AX4_SE', 'AX4_SE_r6': 'AX4_SE',
     'AX4_SE_r7': 'AX4_SE'})
for name, group in groupped_AX4:
    group = group.groupby('Time').mean()
    times = list(group.index)
    group.index = [str(name) + '_' + str(idx) for idx in times]
    group['Time'] = times
    group['Group'] = [name] * group.shape[0]
    genes_orange_avg_AX4 = genes_orange_avg_AX4.append(group, sort=True)

genes_orange_avg_AX4.to_csv(dataPathSaved + 'genes_averaged_orange_AX4groups.tsv', sep='\t')

# **** Scale averaged genes with AX4 max
# For each gene: 1.) calculate its max in AX4, 2.) For each value: value-AX4max,
# 3.) If AX4max is 0 substitute it with 1 and then divide each value with AX4max
genes_orange_avg_scaled = genes_orange_avg.copy()
genes_avg_AX4 = genes_orange_avg_scaled.loc[genes_orange_avg_scaled['Group'] == 'AX4', :]
genes_orange_avg_scaled = genes_orange_avg_scaled.drop(['Group', 'Time'], axis=1)
genes_avg_AX4_max = genes_avg_AX4.drop(['Group', 'Time'], axis=1).max(axis=0)
genes_orange_avg_scaled = genes_orange_avg_scaled - genes_avg_AX4_max
genes_avg_AX4_max = genes_avg_AX4_max.replace(0, 1)
genes_orange_avg_scaled = genes_orange_avg_scaled / genes_avg_AX4_max
genes_orange_avg_scaled[['Time', 'Group']] = genes_orange_avg[['Time', 'Group']]
genes_orange_avg_scaled.to_csv(dataPathSaved + 'genes_averaged_orange_maxScaledToAX4.tsv', sep='\t')

# Scale genes with Xth (eg a high one so that outliers can be latter clipped and most variability in
# lower expressed genes is still observable) percentile
# Subtract percentile from value and then divide it with the percentile (before division replace 0 with 1)
percentile = 0.99
genes_orange_avg_scaled2 = genes_orange_avg.copy()
genes_orange_avg_scaled2 = genes_orange_avg_scaled2.drop(['Group', 'Time'], axis=1)
genes_avg_percentile = genes_orange_avg.drop(['Group', 'Time'], axis=1).quantile(q=percentile, axis=0)
genes_orange_avg_scaled2 = genes_orange_avg_scaled2 - genes_avg_percentile
genes_avg_percentile = genes_avg_percentile.replace(0, 1)
genes_orange_avg_scaled2 = genes_orange_avg_scaled2 / genes_avg_percentile
genes_orange_avg_scaled2[['Time', 'Group']] = genes_orange_avg[['Time', 'Group']]
genes_orange_avg_scaled2.to_csv(dataPathSaved + 'genes_averaged_orange_scale' + str(percentile)[2:] + 'percentile.tsv',
                                sep='\t')
# Bound at max 0.1 (to remove outliers)
max_val = 0.1
genes_orange_avg_scaled2_bounded = genes_orange_avg_scaled2.drop(['Time', 'Strain'], axis=1)
genes_orange_avg_scaled2_bounded[genes_orange_avg_scaled2_bounded > max_val] = max_val
genes_orange_avg_scaled2_bounded[['Time', 'Strain']] = genes_orange_avg_scaled2[['Time', 'Strain']]
# Add strain group info
groups = {'amiB': '1Ag-', 'mybB': '1Ag-', 'acaA': '1Ag-', 'gtaC': '1Ag-',
          'gbfA': '2LAg', 'tgrC1': '2LAg', 'tgrB1': '2LAg', 'tgrB1C1': '2LAg',
          'tagB': '3TA', 'comH': '3TA',
          'ecmARm': '4CD', 'gtaI': '4CD', 'cudA': '4CD', 'dgcA': '4CD', 'gtaG': '4CD',
          'AX4': '5WT', 'MybBGFP': '5WT',
          'acaAPkaCoe': '6SFB', 'ac3PkaCoe': '6SFB',
          'pkaR': '7PD', 'PkaCoe': '7PD'}
genes_orange_avg_scaled2_bounded['Group'] = [groups[strain] for strain in genes_orange_avg_scaled2_bounded['Strain']]

genes_orange_avg_scaled2_bounded.to_csv(dataPathSaved + 'genes_averaged_orange_scale' + str(percentile)[2:] +
                                        'percentileMax' + str(max_val) + '.tsv',
                                        sep='\t')

# Calculate log2FC=log2(val/max) compared to max WT
genes_orange_avg_fc = genes_orange_avg.copy()
genes_avg_AX4 = genes_orange_avg_fc.loc[genes_orange_avg_fc['Group'] == 'AX4', :]
genes_orange_avg_fc = genes_orange_avg_fc.drop(['Group', 'Time'], axis=1)
genes_avg_AX4_max = genes_avg_AX4.drop(['Group', 'Time'], axis=1).max(axis=0)
# Replace zeros to prevent infs - in AX4 scaler and after scaling in expressions to get 0 after log
genes_avg_AX4_max = genes_avg_AX4_max.replace(0, 1)
genes_orange_avg_fc = genes_orange_avg_fc / genes_avg_AX4_max
# Not ok as min now 0 instead of below 0
genes_orange_avg_fc = genes_orange_avg_fc.replace(0, 1)
genes_orange_avg_fc = np.log2(genes_orange_avg_fc)
genes_orange_avg_fc[['Time', 'Group']] = genes_orange_avg[['Time', 'Group']]
genes_orange_avg_fc.to_csv(dataPathSaved + 'genes_averaged_orange_log2FC.tsv', sep='\t')

#*** Sort strains in averaged data
averaged_data=pd.read_table(dataPathSaved + 'genes_averaged_orange_scale99percentileMax0.1.tsv',index_col=0)
strain_order=pd.read_table('/home/karin/Documents/timeTrajectories/data/strain_order.tsv',header=None)
averaged_data['Strain'] = pd.Categorical(averaged_data['Strain'], strain_order[0].values)
averaged_data=averaged_data.sort_values(['Strain','Time'])
averaged_data.to_csv(dataPathSaved + 'genes_averaged_orange_scale99percentileMax0.1.tsv', sep='\t')

# **** Make expression data for single replicate per strain
merged = ClusterAnalyser.merge_genes_conditions(genes=genes,
                                                conditions=conditions[['Measurment', 'Replicate', 'Time', 'Strain']],
                                                matching='Measurment')
splitted = ClusterAnalyser.split_data(data=merged, split_by='Replicate')
for rep, data in splitted.items():
    data = data.drop(["Replicate", 'Measurment'], axis=1)
    data = data.sort_values('Time')
    data.index = [strain + '_' + str(time) for strain, time in zip(data['Strain'], data['Time'])]
    data['Group'] = [groups[data['Strain'][0]]] * data.shape[0]
    splitted[rep] = data

data = []
for strain in conditions['Strain'].unique():
    rep = conditions.query('Strain == "' + strain + '"')['Replicate'].unique()[0]
    data.append(splitted[rep])
data = pd.concat(data)

data.to_csv(dataPathSaved + 'genes_oneRep_orange.tsv', sep='\t')

# Scale one-rep data
percentile = 0.99
max_val = 0.1
data2 = data.drop(['Group', 'Time', 'Strain'], axis=1)
data2_percentile = data2.quantile(q=percentile, axis=0)
data2 = data2 - data2_percentile
data2_percentile = data2_percentile.replace(0, 1)
data2 = data2 / data2_percentile
data2[data2 > max_val] = max_val
data2[['Time', 'Group', 'Strain']] = data[['Time', 'Group', 'Strain']]

data2.to_csv(dataPathSaved + 'genes_oneRep_orange_scale' + str(percentile)[2:] +
             'percentileMax' + str(max_val) + '.tsv', sep='\t')
# ********************
# Check how many hypothetical and pseudogenes are in onthologies
# Get gene descriptions and EIDs
entrez_descriptions = dict()
matcher = GeneMatcher(44689)
matcher.genes = genes.index
for gene in matcher.genes:
    description = gene.description
    entrez = gene.gene_id
    if entrez is not None:
        # Key: EID, val: list [description, n GOs]
        entrez_descriptions[entrez] = [description, 0]
# For each gene set
for ontology in list_all(organism=str(44689)):
    gene_sets = load_gene_sets(ontology, '44689')
    for gene_set in gene_sets:
        for gene_EID in gene_set.genes:
            if gene_EID in entrez_descriptions.keys():
                entrez_descriptions[gene_EID][1] += 1

hypothetical = []
pseudo = []
annotated = []
for description, n_terms in entrez_descriptions.values():
    if 'hypothetical' in description:
        hypothetical.append(n_terms)
    elif 'pseudo' in description:
        pseudo.append(n_terms)
    else:
        annotated.append(n_terms)

# Result (from 12740 entrez terms there were 12740 entries in each of 3 lists - no duplicates/no genes in more than 1 lists):
# annotated: all: 4721, without GO term 1237
# pseudo: all: 159, without GO term: 157
# hypothetical: all: 7860, without GO term: 6434

# ****************************************
# *********************************
# Inverse regulons

# ***********Data from batches by replicate
batches = list(conditions['Replicate'])
batches = np.array(batches)

# Calculate neighbours
for batch in set(batches):
    print(batch)
    genes_sub = genes.T[batches == batch].T
    neighbour_calculator = NeighbourCalculator(genes_sub)
    result_inv = neighbour_calculator.neighbours(200, inverse=True, scale='mean0std1', log=True)
    output = open(path_inverse + 'raw/' + batch + '.pkl', 'wb')
    pkl.dump(result_inv, output)
    output.close()

# Combine neighbours

files = [f for f in glob.glob(path_inverse + 'raw/' + "*.pkl")]


# result_inv=[]
# for f in files:
#     pkl_file = open(f, 'rb')
#     result_inv.append(pkl.load(pkl_file))
#     pkl_file.close()
#
#
#
# summary=[]
# for threshold in [0.6,0.8, 0.85, 0.9, 0.95]:
#     for present in [5, 10, 15, 20]:
#         result_filtered = NeighbourCalculator.merge_results(result_inv, threshold, present)
#         pairs=len(result_filtered)
#         print(threshold,present,pairs)
#         summary.append({'threshold':threshold,'present':present,'N_pairs':pairs})
#

def merge_from_file(files: list, similarity_threshold: float):
    """
    Merge gene pair similaritioes dict from multiple pickled files
    :param files: List of files
    :param similarity_threshold: Remove similarities below
    :return: Dict: key (gene1,gene2), value: list of retained similarities
    """
    merged_results = {}
    for f in files:
        # print(f)
        result = loadPickle(f)
        for pair, similarity in result.items():
            if similarity >= similarity_threshold:
                if pair not in merged_results.keys():
                    merged_results[pair] = []
                merged_results[pair].append(similarity)
    return merged_results


def filter_merged_N_present(merged: dict, min_present: int):
    """
    Filter result of  merge_from_file based on len of values list
    :param merged: result of  merge_from_file
    :param min_present: remove if less than min_present similarities in values list
    :return: Dict: key (gene1,gene2) retained pairs, value: list of  similarities
    """
    filter_merged = {}
    for pair, similarities in merged.items():
        if len(similarities) >= min_present:
            filter_merged[pair] = similarities
    return filter_merged


def process_results_files(files, threshold, min_present, save: str = None):
    """
     Merge gene pair similaritioes dict from multiple pickled files. Filter based on threshold and min_present
    :param files: List of files
    :param threshold: Remove similarities below before min_present filtering
    :param min_present: remove if less than min_present similarities in values list
    :param save: save to a file, if None return instead of save
    :return: Dict: key (gene1,gene2) retained pairs, value: list of  retained similarities
    """
    merged_results = merge_from_file(files=files, similarity_threshold=threshold)
    filtered_present = filter_merged_N_present(merged=merged_results, min_present=min_present)
    if save is None:
        return filtered_present
    else:
        savePickle(save, filtered_present)


# merged_results = merge_from_file(files, similarity_threshold=0.6)
# Plot dist in how many reps present
# distn_present=[]
# for similarities in merged_results.values():
#     distn_present.append(len(similarities))
# plt.hist(distn_present,bins=49)

# Remove results present in less than 10 replicates:
# filter_merged = filter_merged_N_present(merged_results, min_present=10)
# Save filtered

# savePickle(path_inverse + 'merged_T0_6_min10.pkl', filter_merged)
process_results_files(files, threshold=0.8, min_present=10, save=path_inverse + 'merged_kNN200_T0_8_min10.pkl')
# ****************************
# ***********Parameters
# Load filtered
result = loadPickle(path_inverse + 'merged_T0_6_min10.pkl')
# Check how many connections is retained at different thresholds
summary = []
for threshold in [0.6, 0.8, 0.9, 0.95]:
    for present in [10, 20, 30, 40, 45]:

        pairs = 0
        genes_retained = set()
        for pair, similarities in result.items():
            retained_similarities = [item for item in similarities if item >= threshold]
            if len(retained_similarities) >= present:
                pairs += 1
                genes_retained.add(pair[0])
                genes_retained.add(pair[1])
        n_genes = len(genes_retained)
        print(threshold, present, pairs, n_genes)
        summary.append({'threshold': threshold, 'present': present, 'N_pairs': pairs, 'N_genes': n_genes})
summary = pd.DataFrame(summary)
# Result: By replicate.
#     threshold  present  N_pairs  N_genes
# 0        0.60       10   711795    11216
# 1        0.60       20    29922     4731
# 2        0.60       30      656      507
# 3        0.60       40        0        0
# 4        0.60       45        0        0
# 5        0.80       10   711070    11213
# 6        0.80       20    29903     4729
# 7        0.80       30      655      505
# 8        0.80       40        0        0
# 9        0.80       45        0        0
# 10       0.90       10   573170    10589
# 11       0.90       20    24803     3944
# 12       0.90       30      517      411
# 13       0.90       40        0        0
# 14       0.90       45        0        0
# 15       0.95       10   224402     7054
# 16       0.95       20     7530     1673
# 17       0.95       30       90      102
# 18       0.95       40        0        0
# 19       0.95       45        0        0

# *******************************
# ************ Compare between replicate samples

# Make results from randomly selected reps, if similar sized samples
sample1_files, sample2_files = jf.sample_from_list(files, 25)
count_sample = 1
threshold = 0.8
min_present = 5
for sample in [sample1_files[:24], sample2_files]:
    process_results_files(files=sample, threshold=threshold, min_present=min_present, save=path_inverse + 'merged_T' +
                                                                                           str(threshold).replace('.',
                                                                                                                  '_') + '_min' + str(
        min_present)
                                                                                           + '_sample' + str(
        count_sample) + '_Nsample'
                                                                                           + str(len(sample)) + '.pkl')
    count_sample += 1
# Sample1: ['acaA_bio2.pkl', 'cudA_r3.pkl', 'gtaG_r2.pkl', 'ecmA_bio1.pkl', 'acaA_bio1.pkl', 'tagB_bio1.pkl', 'tgrB1C1_r2.pkl', 'tgrB1_r2.pkl', 'comH_r1.pkl', 'AX4_r1.pkl', 'pkaCoeAX4_r2.pkl', 'pkaR_bio2.pkl', 'amiB_r1.pkl', 'gtaC_r41A.pkl', 'mybB_bio1.pkl', 'tgrC1_r2.pkl', 'AX4_FDpool02.pkl', 'AX4_pool21.pkl', 'pkaCoeAX4_r3.pkl', 'pkaR_bio1.pkl', 'gtaG_r1.pkl', 'AX4_bio2.pkl', 'gtaC_r21A.pkl', 'tagB_bio2.pkl', 'mybBGFP_bio1.pkl']
# Sample2: ['AX4_bio1.pkl', 'mybB_bio2.pkl', 'gbfA_r3.pkl', 'tgrC1_r1.pkl', 'gtaC_r47B.pkl', 'ac3pkaCoe_r2.pkl', 'ecmA_bio2.pkl', 'gbfA_r1.pkl', 'AX4_r2.pkl', 'dgcA_r2.pkl', 'ac3pkaCoe_r1.pkl', 'tgrB1C1_r1.pkl', 'gtaI_bio1.pkl', 'AX4_FDpool01.pkl', 'mybBGFP_bio2.pkl', 'AX4_bio3.pkl', 'tgrB1_r1.pkl', 'AX4_pool19.pkl', 'gtaC_r27B.pkl', 'gtaI_bio2.pkl', 'comH_r2.pkl', 'dgcA_r1.pkl', 'cudA_r2.pkl', 'amiB_r2.pkl']
# Sample3?
# Sample4?

# Make results from randomly selected reps, if differently sized samples:
process_results_files(files=sample1_files, threshold=0.6, min_present=20, save=path_inverse +
                                                                               'merged_T0_6_min20_sample1_Nsample'
                                                                               + str(len(sample1_files)) + '.pkl')
process_results_files(files=sample2_files, threshold=0.6, min_present=1, save=path_inverse +
                                                                              'merged_T0_6_min1_sample2_Nsample' +
                                                                              str(len(sample2_files)) + '.pkl')

# Compare which genes and pairs are retained in each of the samples, more in notebook
sample1 = loadPickle(path_inverse + 'merged_T0_8_min5_sample1_Nsample24.pkl')
sample2 = loadPickle(path_inverse + 'merged_T0_8_min5_sample2_Nsample24.pkl')
summary = NeighbourCalculator.plot_threshold_batched(sample1=sample1, sample2=sample2,
                                                     similarity_thresholds=[0.85, 0.9, 0.925, 0.94, 0.95, 0.97, 0.99],
                                                     min_present_thresholds=[5, 10, 12, 14, 15, 16, 17])

# For differently sized samples
# summary2=NeighbourCalculator.compare_threshold_batched(sample1,sample2,[0.8,0.9,0.925,0.94,0.95,0.97,0.99],[20,25,28,29,30,32],2)

# Testing with half-half samples can use F value, recall. However, hard to estimate where to put N Present in whole population.
# If testing on split with one sample almost as large as population then recall and F value dont make much sense as
# most genes retained in smaller sample if N present=1 - better if it must be present in more than 1 sample even in smaller population.


# ********************************
# **********Inverse regulons construction

# Get data of regulons
threshold = 0.96
min_present = 25
results_all = loadPickle(path_inverse + 'merged_kNN200_T0_8_min10.pkl')
filtered = NeighbourCalculator.filter_similarities_batched(results=results_all, similarity_threshold=threshold,
                                                           min_present=min_present)
# Get data for clustering of retained genes
genes_pp, inverse_pp, gene_names = Clustering.get_genes(result=filtered, genes=genes, threshold=None,
                                                        inverse=True, return_query=True)
# Make tsne data
# embedding1 = make_tsne(genes_pp, perplexities_range=[5, 100], exaggerations=[5, 1.6], momentums=[0.6, 0.8])
# embedding2 = make_tsne(genes_pp, perplexities_range=[5, 50], exaggerations=[10, 1.6], momentums=[0.6, 0.97])
embedding = make_tsne(genes_pp, perplexities_range=[5, 50], exaggerations=[15, 1.6], momentums=[0.6, 0.8])
tsne_data = make_tsne_data(tsne=embedding, names=gene_names)

# Cluster
louvain_cl = LouvainClustering.from_orange_graph(data=genes_pp, gene_names=gene_names, neighbours=50)
clusters = louvain_cl.get_clusters(resolution=0.8, random_state=0)

# Analyse clusters
cluster_analyser = ClusterAnalyser(genes=genes, conditions=conditions, organism=44689, average_data_by='Time',
                                   split_data_by='Strain', matching='Measurment', control='AX4')
clustering_analyser = ClusteringAnalyser(gene_names=genes.index, cluster_analyser=cluster_analyser)
cluster_data, membership_selected = clustering_analyser.analyse_clustering(clustering=louvain_cl,
                                                                           clusters=clusters,
                                                                           tsne_data=tsne_data,
                                                                           tsne_plot={'s': 2, 'alpha': 0.4})

# Make graph
graph = build_graph(filtered)
threshold_str = str(threshold).replace('.', '_')
nx.write_pajek(graph, path_inverse + 'kN200_t' + threshold_str + '_min' + str(min_present) + 'Rep_inv.net')
# graph=nx.read_pajek( path_inverse + 'kN200_t'+threshold_str+'_min'+str(min_present)+'Rep_inv.net')

# Get clusters for each node of graph
named_clusters = louvain_cl.get_clusters_by_genes(clusters=clusters)
# cluster_df = pd.DataFrame(named_clusters, index=['cluster']).T
# cluster_df.to_csv(path_inverse + 'kN200_t'+threshold_str+'_min'+str(min_present)+'Rep_inv_clusters.tsv', sep='\t')

# Draw graph
colours = []
for node in graph.nodes:
    colours.append(named_clusters[node])
nx.draw_spring(graph, with_labels=False, node_size=4, width=0.3, node_color=colours, cmap=plt.cm.Set1)

# ***************************
# *********** Unused

# Make means for each cluster, Do it after data preprocessing.
genes_sub = pd.DataFrame(genes_pp, index=gene_names)
genes_sub_inv = pd.DataFrame(inverse_pp, index=gene_names)
# genes_sub1=genes.loc[list(named_clusters.keys()),:] # Decided for first scaling and then mean
cluster_df = pd.DataFrame(named_clusters, index=['cluster']).T
genes_sub_cl = pd.concat([genes_sub, cluster_df], axis=1)
mean_cl_genes = genes_sub_cl.groupby('cluster').mean()
genes_sub_cl_inv = pd.concat([genes_sub_inv, cluster_df], axis=1)
mean_cl_genes_inv = genes_sub_cl_inv.groupby('cluster').mean()

# Calculate similarities between means of clusters.
# Use -0.5 in profiles to get opposite profiles below 0 (for heatmap colouring)
sims = OrderedDict()
cl_names = mean_cl_genes.index
for c1 in range(0, mean_cl_genes.shape[0] - 1):
    for c2 in range(c1 + 1, mean_cl_genes.shape[0]):
        sim = calc_cosine(mean_cl_genes.values - 0.5, mean_cl_genes_inv.values - 0.5, c1, c2, sim_dist=True,
                          both_directions=True)
        sims[(cl_names[c1], cl_names[c2])] = sim

# Convert similarities dict into matrix for heatmap
matrix = np.zeros((len(cl_names - 1), len(cl_names - 1)))
for k, v in sims.items():
    matrix[k[0], k[1]] = v
    matrix[k[1], k[0]] = v

# Heatmap
sb.clustermap(matrix, center=0, cmap='cooldwarm')

# ******************************
# *** Select genes for each replicate/strain separately
SCALE = 'mean0std1'
LOG = True
NHUBS = 1000
NEIGHBOURS = 6
SPLITBY = 'Strain'

# Check what happens when using different replicates/numbers of points
conditions_all = conditions.copy()
genes_all = genes.copy()
# Both with non averaged AX4
# Data without mybBGFP and only AX4_PE
conditions = conditions.loc[(conditions['Strain'] != 'MybBGFP') & (~conditions['Replicate'].str.contains('AX4_SE')) &
                            (~conditions['Replicate'].str.contains('AX4_FD')), :]
genes = genes.loc[:, conditions['Measurment']]

# Data with only 7 (or 5 for gtaC) measurments per strain
conditions_gtaC = conditions.loc[conditions['Strain'] == 'gtaC', :]
conditions = conditions.loc[conditions['Strain'] != 'gtaC', :]
conditions = conditions.loc[conditions['Time'].isin([0, 4, 8, 12, 16, 20, 24]), :]
conditions = conditions.append(conditions_gtaC)
genes = genes.loc[:, conditions['Measurment']]

# Data with only 7 measurments and no MybBGFP and only 1 AX4 (PE)
conditions = conditions.loc[(conditions['Strain'] != 'MybBGFP') & (~conditions['Replicate'].str.contains('AX4_SE')) &
                            (~conditions['Replicate'].str.contains('AX4_FD')), :]
conditions_gtaC = conditions.loc[conditions['Strain'] == 'gtaC', :]
conditions = conditions.loc[conditions['Strain'] != 'gtaC', :]
conditions = conditions.loc[conditions['Time'].isin([0, 4, 8, 12, 16, 20, 24]), :]
conditions = conditions.append(conditions_gtaC)
genes = genes.loc[:, conditions['Measurment']]

# *** Check if retaining genes by hubs works - compare on all data and previously selected by one close neighbour
neighbour_calculator_all = NeighbourCalculator(genes=genes)
neigh_all, sims_all = neighbour_calculator_all.neighbours(n_neighbours=NEIGHBOURS, inverse=False, scale=SCALE, log=LOG,
                                                          return_neigh_dist=True)
# hubs_all = NeighbourCalculator.find_hubs(similarities=sims_all, n_hubs=NHUBS)
# selected_genes_names = list(pd.read_table('/home/karin/Documents/timeTrajectories/Orange_workflows/regulons/' +
#                                         'kNN2_threshold0.95_m0s1log_Orange.tsv')['Gene'])
# jaccard(set(hubs_all), set(selected_genes_names))
# 0.647 - lower as hubs has more genes selected
# len(set(hubs_all) & set(selected_genes_names))
# Shared genes with neighbours=11 (=10+1): 700, out of 1000 in hubs and 782 selectd by 1 close neighbour for Orange
# if neighbours=6, shared genes=714, jaccard=0.669
# If 782 hubs are selected with KNN=6 there are 683 shared genes with jaccard index 0.775 (as there is less of hubs selected)
# If 782 hubs are selected with kNN=11 there are 662 shared genes with jaccard index 0.734

# *****
# *** Workflow on replicates
# Split data by replicate, scaling and zero filtering is done in neighbours
merged = ClusterAnalyser.merge_genes_conditions(genes=genes, conditions=conditions[['Measurment', SPLITBY]],
                                                matching='Measurment')
splitted = ClusterAnalyser.split_data(data=merged, split_by=SPLITBY)
for rep, data in splitted.items():
    splitted[rep] = data.drop([SPLITBY, 'Measurment'], axis=1).T

# Claulculate neighbours - sims_dict has similarity matrices from samples
sims_dict = dict()
for rep, data in splitted.items():
    print(rep)
    neighbour_calculator = NeighbourCalculator(genes=data)
    neigh, sims_dict[rep] = neighbour_calculator.neighbours(n_neighbours=NEIGHBOURS, inverse=False, scale=SCALE,
                                                            log=LOG,
                                                            return_neigh_dist=True)
sims_dict['all'] = sims_all
# Descriptions of files
# newGenes: Has mybBGFP and AX4 averaged, after mt were removed, polyA.
# noMybBGFP-AX4onlyPE: no MybBGFP and only AX4_PE non averaged, AX4 not averaged
# 7points: measurements at 0,4,8,12,16,20,24 and gtaC all points (5*4), AX4 not averaged
file_prefix = 'newGenes_'
savePickle(
    pathSelGenes + file_prefix + 'simsDict_scale' + SCALE + '_log' + str(LOG) + '_kN' + str(
        NEIGHBOURS) + '_split' + SPLITBY + '.pkl',
    sims_dict)
# sims_dict = loadPickle(
#    pathSelGenes + file_prefix+'simsDict_scale' + SCALE + '_log' + str(LOG) + '_kN' + str(NEIGHBOURS) + '_split' + SPLITBY + '.pkl')

# Select genes with highest average similarity to the neighbours
retained_genes_dict = dict()
for rep, sims in sims_dict.items():
    retained_genes_dict[rep] = NeighbourCalculator.find_hubs(similarities=sims_dict[rep], n_hubs=NHUBS)

# Save for R
pd.DataFrame(retained_genes_dict).to_csv(pathSelGenes + file_prefix + 'selectedGenes' + str(NHUBS) + '_scale' +
                                         SCALE + '_log' + str(LOG) + '_kN' + str(NEIGHBOURS) + '_split' + SPLITBY
                                         + '.tsv', sep='\t', index=False)

replicates = list(retained_genes_dict.keys())
# Calculates similarities between retained genes of different samples
retained_genes_jaccard = pd.DataFrame()
retained_genes_shared = pd.DataFrame(index=replicates, columns=replicates)
dist_arr = []
for idx, rep1 in enumerate(replicates[:-1]):
    for rep2 in replicates[idx + 1:]:
        genes1 = set(retained_genes_dict[rep1])
        genes2 = set(retained_genes_dict[rep2])
        jaccard_index = jaccard(genes1, genes2)
        retained_genes_jaccard.loc[rep1, rep2] = jaccard_index
        retained_genes_jaccard.loc[rep2, rep1] = jaccard_index
        shared = len(genes1 & genes2)
        retained_genes_shared.loc[rep1, rep2] = shared
        retained_genes_shared.loc[rep2, rep1] = shared
        dist_arr.append(1 - jaccard_index)

retained_genes_jaccard = retained_genes_jaccard.fillna(1)
retained_genes_shared = retained_genes_shared.fillna(NHUBS)

# Plot similarity
hc.dendrogram(hc.ward(dist_arr), labels=replicates, color_threshold=0)
plt.title('Clustered ' + SPLITBY + 's based on jaccard distance of selected genes')
sb.clustermap(retained_genes_shared, yticklabels=True, xticklabels=True)
plt.title('Heatmap of N shared selected genes in ' + SPLITBY + 's')
plt.figure()
sb.heatmap(pd.DataFrame(retained_genes_shared['all'].sort_values()), yticklabels=True, xticklabels=True)
plt.title('Heatmap of N shared selected genes in ' + SPLITBY + 's compared to data on all measurements')

# How many genes are shared by all groups/samples
genes_in_all = set()
first = True
retained_genes_dict_reps = retained_genes_dict.copy()
del retained_genes_dict_reps['all']
for rep, retained_genes in retained_genes_dict_reps.items():
    if first:
        genes_in_all = set(retained_genes)
        first = False
    else:
        genes_in_all = genes_in_all & set(retained_genes)
    print(rep, len(genes_in_all))

# Sample random genes
splitted['all'] = genes
retained_genes_dict = dict()
for rep, data in splitted.items():
    rep_genes = list(NeighbourCalculator(genes=data)._genes.index)
    retained_genes_dict[rep] = random.sample(rep_genes, NHUBS)

# ******************* How are similarities to 5 closest neighbours distributed in AX4 vs mybB (non averaged data)
# For both use only some points so that they have same number of measurments -
# Using AX4_SE 2 replicates results in same number of measurements as mybB
genes_AX4 = genes.loc[:, (conditions['Replicate'].isin(['AX4_SE_r6', 'AX4_SE_r7'])).values]
genes_mybB = genes.loc[:, (conditions['Strain'] == 'mybB').values]

neighbour_calculator_AX4 = NeighbourCalculator(genes=genes_AX4)
neigh_AX4, sims_AX4 = neighbour_calculator_AX4.neighbours(n_neighbours=NEIGHBOURS, inverse=False, scale=SCALE, log=LOG,
                                                          return_neigh_dist=True, remove_self=True)
neighbour_calculator_mybB = NeighbourCalculator(genes=genes_mybB)
neigh_mybB, sims_mybB = neighbour_calculator_mybB.neighbours(n_neighbours=NEIGHBOURS, inverse=False, scale=SCALE,
                                                             log=LOG,
                                                             return_neigh_dist=True, remove_self=True)

plt.hist(sims_AX4.mean(axis=1), bins=100, label='AX4_SE_r6,7', alpha=0.5, )
plt.hist(sims_mybB.mean(axis=1), bins=100, label='mybB', alpha=0.5)
plt.title('Average similarity to the closest 5 neighbours')
plt.legend()
plt.xlabel('Average cosine similarity')
plt.ylabel('N genes')
# ******** Merge and parse clusters form R tightclust based on selected genes
NHUBS = str(1000)
CLUST = str(25)
clusters = pd.read_table(dataPathSaved +
                         '/clusters/tight_clust/selectedGenes' + NHUBS +
                         '_scalemean0std1_logTrue_kN6_splitStraintightclust' + CLUST + '.tsv',
                         index_col=0)
# clusters = clusters.drop('all', axis=1)

# Remove all NaN (not selected) or all unclustered genes (-1)
remove = []
for row in clusters.iterrows():
    if np.array([a or b for a, b in zip(row[1].isna(), row[1] == -1)]).all():
        remove.append(row[0])
clusters = clusters.drop(remove)

# Split data by replicate, scaling is done latter
SPLITBY = 'Strain'
merged = ClusterAnalyser.merge_genes_conditions(genes=genes, conditions=conditions[['Measurment', SPLITBY]],
                                                matching='Measurment')
splitted = ClusterAnalyser.split_data(data=merged, split_by=SPLITBY)
for rep, data in splitted.items():
    splitted[rep] = data.drop([SPLITBY, 'Measurment'], axis=1).T
splitted['all'] = genes

# Merge clusters for each cluster group
SCALE = 'mean0std1'
LOG = True
shared_count = dict()
clusters_merged = pd.DataFrame(index=clusters.index, columns=clusters.columns)
for group in clusters.columns:
    # print(group)
    # Neighbourhoods
    group_data = clusters[group]
    group_data = group_data.loc[group_data > -1]
    neighbourhoods = dict()
    for cluster in pd.DataFrame(group_data).groupby(group):
        neighbourhoods[cluster[0]] = set(cluster[1].index)
    # Group expression data
    genes_normalised = \
        NeighbourCalculator.get_index_query(genes=splitted[group].loc[group_data.index, :], inverse=False,
                                            scale=SCALE, log=LOG)[0]
    genes_cosine = pd.DataFrame(cosine_similarity(genes_normalised), index=group_data.index, columns=group_data.index)
    # Distances
    dist_arr, node_neighbourhoods, group_name_mapping = NeighbourhoodParser.neighbourhood_distances(
        neighbourhoods=neighbourhoods,
        measure='avg_dist',
        genes_dist=1 - genes_cosine)
    # Merge based on hc
    neigh_hc = hc.ward(dist_arr)
    min_neighbourhood_similarity = NeighbourhoodParser.min_neighbourhood_similarity(neighbourhoods,
                                                                                    genes_sims=genes_cosine)
    node_neighbourhoods = NeighbourhoodParser.merge_by_hc(hc_result=neigh_hc, node_neighbourhoods=node_neighbourhoods,
                                                          genes_sims=genes_cosine,
                                                          min_group_sim=min_neighbourhood_similarity)
    # Dont include all group in count
    if group != 'all':
        # print('For count',group)
        for neighbourhood in node_neighbourhoods.values():
            neighbourhood = list(neighbourhood)
            for idx, gene1 in enumerate(neighbourhood[:-1]):
                for gene2 in neighbourhood[idx + 1:]:
                    if gene1 < gene2:
                        add1 = gene1
                        add2 = gene2
                    else:
                        add1 = gene2
                        add2 = gene1
                    pair = (add1, add2)
                    if pair in shared_count.keys():
                        shared_count[pair] += 1
                    else:
                        shared_count[pair] = 1

    # Add to df of strain clusters
    for neighbourhood, genes in node_neighbourhoods.items():
        for gene in genes:
            clusters_merged.loc[gene, group] = neighbourhood

# Put counts of shared group memeberships in DF
# Too slow - thus use the dict (as many 0 elements anyways)
# clusters_groups = clusters_merged.drop('all', axis=1)
# shared_count_df=NeighbourhoodParser.count_same_group(clusters_groups)
shared_count_df = pd.DataFrame(np.zeros((clusters_merged.shape[0], clusters_merged.shape[0])),
                               index=clusters_merged.index, columns=clusters_merged.index)
for pair, count in shared_count.items():
    shared_count_df.loc[pair[0], pair[1]] = count
    shared_count_df.loc[pair[1], pair[0]] = count

shared_count_df.to_csv(dataPathSaved +
                       '/clusters/tight_clust/selectedGenes' + NHUBS +
                       '_scalemean0std1_logTrue_kN6_splitStraintightclust' + CLUST + '_sharedCount.tsv', sep='\t')
clusters_merged.to_csv(dataPathSaved +
                       '/clusters/tight_clust/selectedGenes' + NHUBS +
                       '_scalemean0std1_logTrue_kN6_splitStraintightclust' + CLUST + '_clustersGroups.tsv', sep='\t')

# Distn of max scores per genes
# Some genes have score 0 because table includes 'all' genes which do not have count
plt.hist(shared_count_df.max(), bins=shared_count_df.max().max() + 1)
plt.xlabel('Highest sharing score per gene')
plt.ylabel('N genes')
plt.title('Per gene highest score for being shared in clusters across strains with another gene')

# For easier viewing filter genes so that only those that have at least one gene similarity to another gene at
# least equal to min_shared
min_shared = 5
genemax = shared_count_df.max()
remove_genes = genemax.loc[genemax < min_shared].index
shared_count_df_filtered = shared_count_df.drop(index=remove_genes, columns=remove_genes)
shared_count_df_filtered.to_csv(dataPathSaved +
                                '/clusters/tight_clust/selectedGenes' + NHUBS +
                                '_scalemean0std1_logTrue_kN6_splitStraintightclust' + CLUST +
                                '_sharedCount_filtered' + str(min_shared) + '.tsv', sep='\t')

sb.clustermap(shared_count_df_filtered, yticklabels=False, xticklabels=False)
# ******************************************************************************************************************
# *********** Find hub/seed genes (have strong closest connections) and their neighbourhoods, merge if needed
# *********************************************************************************************************************
# Jupyter notebook selection of parameters
SCALE = 'mean0std1'
LOG = True
NHUBS = 100
SIM = 0.95

# Get 5 (=5+1) closest neighbours for each gene
neighbour_calculator_all = NeighbourCalculator(genes=genes)
# Returns 5 neighbours as it removes self (or last neighbour) from neighbour list
neigh_all, sims_all = neighbour_calculator_all.neighbours(n_neighbours=6, inverse=False, scale=SCALE, log=LOG,
                                                          return_neigh_dist=True, remove_self=True)
neigh_all, sims_all = loadPickle(pathMergSim + 'newGenes_kN6_m0s1log_neighbours_sims.pkl')
# Remove self from neighbours and similarities

# Cheking how many are do not have self for neighbour or do not have self as first neighbour:
# not_in = 0
# not_first = 0
# for e in neigh_all.iterrows():
#     e1 = e[1]
#     el = e1[e1 == e[0]]
#     if len(el) is 0:
#         not_in+=1
#     elif int(el.index[0]) is not 0:
#         not_first+=1

# Compare avg similarity distn to first five neighbours on expression data and expression data permuted by column -
# also in jupyter notebook

# Permute the data
# Take forn NeighbourCalculator to match the data used for non-permuted neighbours - e.g. removes some null genes
# genes_permuted = neighbour_calculator_all._genes.apply(lambda x: x.sample(frac=1).values)
# neighbour_calculator_permuted = NeighbourCalculator(genes=genes_permuted)
# neigh_permuted, sims_permuted = neighbour_calculator_permuted.neighbours(n_neighbours=6, inverse=False, scale=SCALE, log=LOG,
#                                                          return_neigh_dist=True,remove_self=True)

# Plot comparison of permuted and non permuted average similarities of 5 closest neighbours
# plt.hist(sims_all.mean(axis=1), bins=100,label='original')
# plt.hist(sims_permuted.mean(axis=1), bins = 100, alpha=0.5,label='feature permuted')
# plt.title('Average similarity to the closest 5 neighbours')
# plt.legend()

# Find seed genes. First find genes that have at least 5 close neighbours ,
#  then select those that have highest avg similarity.
hubs_all_candidate = NeighbourCalculator.filter_distances_matrix(similarities=sims_all, similarity_threshold=SIM,
                                                                 min_neighbours=5)
# Candidates at different thresholds:
# similarity_threhold, min neighbours, N candidates
# 0.96, 5, 166
# 0.96, 3, 240
# 0.95, 5, 338
# 0.95, 3, 465

# Test if same N of genes with 6 or 11 KNN and filter 6 neigh above 0.96 sim (m0s1, log,noninverse):
# Got same number of genes

# Select top hubs from the ones that were not filtered out
# hubs_all = NeighbourCalculator.find_hubs(similarities=sims_all.loc[hubs_all_candidate, :], n_hubs=NHUBS)
# Or select all from above
hubs_all = hubs_all_candidate

# Get neighbours of hub genes
neigh_hubs, sims_hubs = neighbour_calculator_all.neighbours(n_neighbours=100, inverse=False, scale=SCALE,
                                                            log=LOG,
                                                            return_neigh_dist=True, genes_query_names=hubs_all)
neighbourhoods = NeighbourCalculator.hub_neighbours(neighbours=neigh_hubs, similarities=sims_hubs,
                                                    similarity_threshold=SIM)
# ***** Merge neighborhoods
# Add hub to neighbourhood
neighbourhoods_merged = list()
for hub, neighbourhood in neighbourhoods.items():
    neighbourhood_new = neighbourhood.copy()
    neighbourhood_new.append(hub)
    neighbourhood_new.sort()
    neighbourhoods_merged.append(tuple(neighbourhood_new))

# Remove repeated neighborhoods
neighbourhoods_merged = set(neighbourhoods_merged)
neighbourhoods_merged = list(neighbourhoods_merged)

# Remove contained neighbourhoods. Assumes neighbourhoods are unique.
to_remove = set()
for idx1 in range(len(neighbourhoods_merged) - 1):
    for idx2 in range(idx1 + 1, len(neighbourhoods_merged)):
        genes1 = set(neighbourhoods_merged[idx1])
        genes2 = set(neighbourhoods_merged[idx2])
        if genes1.issubset(genes2) and genes2.issubset(genes1):
            raise ValueError('neighbourhoods are not unique')
        elif genes1.issubset(genes2):
            to_remove.add(neighbourhoods_merged[idx1])
        elif genes2.issubset(genes1):
            to_remove.add(neighbourhoods_merged[idx2])
for sub in to_remove:
    neighbourhoods_merged.remove(sub)

# Calculate cosine similarity between genes for later
neighbourhood_genes = list({gene for neigh_list in neighbourhoods_merged for gene in neigh_list})
genes_normalised = NeighbourCalculator.get_index_query(genes=genes.loc[neighbourhood_genes, :], inverse=False,
                                                       scale=SCALE, log=LOG)[0]
genes_cosine = pd.DataFrame(cosine_similarity(genes_normalised), index=neighbourhood_genes, columns=neighbourhood_genes)

# Find order for merging - dist between neighbourhoods.
# Distance : jaccard, percent_shared_smaller, avg_sim (=  average simimarity). To get distance use 1 - sim_metric.
distance = 'jaccard'
dist_arr = []
neigh_ids = range(len(neighbourhoods_merged))
for idx1 in neigh_ids[:-1]:
    for idx2 in neigh_ids[idx1 + 1:]:
        genes1 = set(neighbourhoods_merged[idx1])
        genes2 = set(neighbourhoods_merged[idx2])
        if distance == 'jaccard':
            dist = 1 - jaccard(genes1, genes2)
        elif distance == 'percent_shared_smaller':
            intersection = len(genes1 & genes2)
            dist = 1 - intersection / min(len(genes1), len(genes2))
        elif distance == 'avg_sim':
            dist = 1 - genes_cosine.loc[genes1, genes2].values.flatten().mean()
        dist_arr.append(dist)
# TODO Think about other distances for merging that would be less affected by N of genes in each group -
# eg. intersection of the smaller, average similarity


neigh_hc = hc.ward(dist_arr)
# hc.dendrogram(neigh_hc,color_threshold=0)

tree = hc.to_tree(neigh_hc, rd=True)[1]

# Find minimal similarity within any of the groups - use this as later threshold for merging
min_group_sim = 1
for group in neighbourhoods_merged:
    min_sim = genes_cosine.loc[group, group].values.flatten().min()
    if min_group_sim > min_sim:
        min_group_sim = min_sim

node_neighbourhoods = dict(zip(neigh_ids, neighbourhoods_merged))
for node in tree[len(neighbourhoods_merged):]:
    id = node.get_id()
    id1 = node.get_left().get_id()
    id2 = node.get_right().get_id()
    # If not one of previous merges was not performed so similarity will definitely be too low
    if id1 in node_neighbourhoods.keys() and id2 in node_neighbourhoods.keys():
        genes1 = node_neighbourhoods[id1]
        genes2 = node_neighbourhoods[id2]
        min_sim = genes_cosine.loc[genes1, genes2].min().min()
        if min_sim >= min_group_sim:
            genes_new = tuple(set(genes1 + genes2))
            node_neighbourhoods[id] = genes_new
            # Remove merged nodes
            del node_neighbourhoods[id1]
            del node_neighbourhoods[id2]

# Make table for Orange:
orange_groups = []
for id, group in node_neighbourhoods.items():
    for gene in group:
        orange_groups.append({'Gene': gene, 'Group': id})
orange_groups = pd.DataFrame(orange_groups)

# *********************
# Compare average per gene expression in individual replicates
avg_expression = pd.DataFrame()
SPLITBY = 'Replicate'
merged = ClusterAnalyser.merge_genes_conditions(genes=genes, conditions=conditions[['Measurment', SPLITBY]],
                                                matching='Measurment')
splitted = ClusterAnalyser.split_data(data=merged, split_by=SPLITBY)
for rep, data in splitted.items():
    data = data.drop([SPLITBY, 'Measurment'], axis=1).T
    avg_expression[rep] = data.mean(axis=1)
avg_expression.to_csv(
    '/home/karin/Documents/timeTrajectories/Orange_workflows/regulons/avgPerGeneExpression_Replicate.tsv', sep='\t')

# **************************************************
# ***************** Regulons by strain, matrix of how many strains had connection
batches = list(conditions['Strain'])
batches = np.array(batches)

# Calculate neighbours
for batch in set(batches):
    print(batch)
    genes_sub = genes.T[batches == batch].T
    neighbour_calculator = NeighbourCalculator(genes_sub)
    result_inv = neighbour_calculator.neighbours(300, inverse=False, scale='mean0std1', log=True)
    savePickle(pathByStrain + 'kN300_mean0std1_log/' + batch + '.pkl', result_inv)

# Extract genes with close neighbours
# Similarity thresholds
# threshold_dict = {14: 0.93, 16: 0.91, 18: 0.9, 20: 0.9, 24: 0.89, 26: 0.86, 88: 0.83}
# threshold_dict_strain = dict()
# splitted = conditions.groupby('Strain')
# for group in splitted:
#    threshold_dict_strain[group[0]] = threshold_dict[group[1].shape[0]]
threshold_dict_strain = pd.read_table(
    '/home/karin/Documents/timeTrajectories/data/regulons/selected_genes/thresholds/strainThresholds_k2_m0s1log_best0.7.tsv',
    sep='\t')
# Here is another round around threshold as otherwise gets converted to longer float
threshold_dict_strain = {data['Strain']: np.round(data['Threshold'], 3) for row, data in
                         threshold_dict_strain.iterrows()}

# Extract genes from strains and merge into a single matrix
files = [f for f in glob.glob(pathByStrain + 'kN300_mean0std1_log/' + "*.pkl")]
# To select only FB strains:
# files = [f for f in glob.glob(pathByStrain + 'kN300_mean0std1_log/' + "*.pkl")
#         if any(strain in f for strain in ['/AX4','/MybBGFP','/PkaCoe','/pkaR']) ]
n_genes = genes.shape[0]
genes_dict = dict(zip(genes.index, range(n_genes)))
merged_results = np.zeros((n_genes, n_genes))
for f in files:
    strain = f.split('/')[-1].replace('.pkl', '')
    result = loadPickle(f)
    similarity_threshold = threshold_dict_strain[strain]
    print(strain, similarity_threshold)
    for pair, similarity in result.items():
        if similarity >= similarity_threshold:
            gene1 = genes_dict[pair[0]]
            gene2 = genes_dict[pair[1]]
            merged_results[gene1, gene2] += 1
            merged_results[gene2, gene1] += 1
merged_results = pd.DataFrame(merged_results, index=genes.index, columns=genes.index)

merged_results.to_csv(pathByStrain + 'kN300_mean0std1_log/' + 'mergedGenes.tsv', sep='\t')
merged_results = pd.read_table(pathByStrain + 'kN300_mean0std1_log/' + 'mergedGenes.tsv', sep='\t', index_col=0)

# Filter to best gene candidates (have a close neighbour across many strains)
genemax = merged_results.max()
remove_genes = genemax.loc[genemax < 18].index
merged_results_filtered = merged_results.drop(index=remove_genes, columns=remove_genes)

merged_results_filtered.to_csv(pathByStrain + 'kN300_mean0std1_log/' + 'mergedGenes_min18.tsv', sep='\t')
sb.clustermap(merged_results_filtered, yticklabels=False, xticklabels=False)

# Filter based on how many strains reach 'high' expression (50% of 99th percentile)
ration_max = 0.99
proportion = 0.5
threshold = genes.quantile(q=ration_max, axis=1) * proportion
# Count how many strains reach expression threshold
SPLITBY = 'Strain'
merged = ClusterAnalyser.merge_genes_conditions(genes=genes, conditions=conditions[['Measurment', SPLITBY]],
                                                matching='Measurment')
splitted = ClusterAnalyser.split_data(data=merged, split_by=SPLITBY)
for rep, data in splitted.items():
    splitted[rep] = data.drop([SPLITBY, 'Measurment'], axis=1).T

strain_expressed = pd.DataFrame()
for strain, data in splitted.items():
    strain_expressed[strain] = (data.T >= threshold).any()
# strain_expressed.to_csv(pathByStrain +'expressedGenes'+str(ration_max)+str(proportion)+'.tsv',sep='\t')
n_strains = strain_expressed.sum(axis=1)

# Require that min is 1 (e.g. at least one strain)
min_strains=1
n_strains[n_strains < min_strains] = min_strains
# Require that max is below actual N strains as genes may be not found as co-expressed due to an error
max_strains=18
n_strains[n_strains > max_strains] = max_strains

ratio_n_expressed = 1
#min_neighbours = 1
genemax = merged_results.max()
remove_genes = genemax.loc[genemax < (n_strains * ratio_n_expressed)].index
# Find neighbours present in at least N strains for each gene, using gene specific N strains (columnwise). ->
# Sum n of such neighbours. -> Check if the N of neighbours is big enough. -> Select columns of genes that do not
# satisfy the conditions.
# TODO Problem that filters out genes's neighbours if neighbours have low N neighbours
#remove_genes = merged_results.loc[:, ~((merged_results >= n_strains).sum() >= min_neighbours)].columns
merged_results_filtered = merged_results.drop(index=remove_genes, columns=remove_genes)
merged_results_filtered.to_csv(pathByStrain + 'kN300_mean0std1_log/' + 'mergedGenes_minExpressed' +
                               str(ration_max) + str(proportion) + 'StrainsProportion' +
                               str(ratio_n_expressed)+'Min'+str(min_strains)+'Max'+str(max_strains)+
                               #'minNeigh'+str(min_neighbours) +
                               '.tsv', sep='\t')

# !!! Fill the diagonal with N strains ?
N_STRAINS = 21
# merged_results_filtered=pd.read_table(pathByStrain + 'kN300_mean0std1_log/' + 'mergedGenes_min18.tsv', index_col=0)
for i in range(merged_results_filtered.shape[0]):
    merged_results_filtered.iloc[i, i] = N_STRAINS
merged_results_filtered.to_csv(pathByStrain + 'kN300_mean0std1_log/' + 'mergedGenes_min18_filledDiagonal.tsv', sep='\t')

# Adjust RPKUM so that it has 0 expression in strains where it is supposed to be unexpressed
genes_adjusted = []
for strain, data in splitted.items():
    expressed = strain_expressed[strain]
    data = data.copy()
    data.loc[~expressed, :] = 0
    genes_adjusted.append(data)
genes_adjusted = pd.concat(genes_adjusted, axis=1)
genes_adjusted.to_csv(pathByStrain + 'genesAdjustUnexpressed' + str(ration_max) + str(proportion) + '.tsv', sep='\t')
#******************************
#**** Compare regulon clusters
files = [f for f in glob.glob(pathByStrain + 'kN300_mean0std1_log/clusters/' + "*.tab")]
clusters_df=[]
for file in files:
    clusters=pd.read_table(file,index_col=0)
    clusters.columns=[file.split('/')[-1].replace('.tab','').replace('mergedGenes_','')]
    clusters_df.append(clusters)
clusters_df=pd.concat(clusters_df,axis=1,sort=True)
#clusters_df=clusters_df.replace(np.nan,'CNA')
clusters_df.to_csv(pathByStrain + 'kN300_mean0std1_log/clusters/cluster_summary.tsv',sep='\t')
# *************
# **** Find genes co-expressed with TFs in strains that progress to FB
# TODO

# *******************************
# ********* Interaction based similarity threshold
# Find similarity threshold based on known interactions

# Rename the gene names in the String file - this collapses multiple proteins to single genes
interactions = pd.read_table('/home/karin/Documents/timeTrajectories/data/44689.protein.links.detailed.v11.0.txt',
                             sep=' ')
protein_names = pd.read_table('/home/karin/Documents/timeTrajectories/data/DDB-GeneID-UniProt.txt')
protein_gene_names = dict(zip(protein_names['DDB ID'], protein_names['DDB_G ID']))


def rename_stringdb(protein_names):
    gene_names = []
    for name in protein_names:
        name = name.replace('44689.', '')
        if name in protein_gene_names.keys():
            gene_names.append(protein_gene_names[name])
        else:
            gene_names.append(np.nan)
    return gene_names


interactions['protein1'] = rename_stringdb(interactions['protein1'])
interactions['protein2'] = rename_stringdb(interactions['protein2'])

# Remove any rows with unnamed genes
interactions = interactions.dropna()

# Remove any rows that have same gene as protein1 and protein2 due to conversion of names from gene to protein
interactions = interactions[interactions['protein1'] != interactions['protein2']]

# Remove repeated rows - same protein pair in multiple rows; retain row with maximal combined_score
pairs = dict()
remove = set()
for idx, row in interactions.iterrows():
    pair = [row['protein1'], row['protein2']]
    pair.sort()
    pair = tuple(pair)
    score = row['combined_score']
    if pair not in pairs.keys():
        pairs[pair] = (score, idx)
    else:
        max_score, idx_max = pairs[pair]
        if max_score < score:
            remove.add(idx_max)
            pairs[pair] = (score, idx)
        else:
            remove.add(idx)

interactions = interactions.drop(remove)

# Save renamed interactions
interactions.to_csv('/home/karin/Documents/timeTrajectories/data/44689.protein.links.detailed.v11.0.genenames.txt',
                    sep='\t', index=False)
# Load renamed
interactions = pd.read_table(
    '/home/karin/Documents/timeTrajectories/data/44689.protein.links.detailed.v11.0.genenames.txt')

# Extract high confidence interactions
confident = interactions[interactions['combined_score'] > 700]
confident.to_csv(
    '/home/karin/Documents/timeTrajectories/data/44689.protein.links.detailed.v11.0.genenames.highconfidence.txt',
    sep='\t', index=False)

# Remove pairs where a member is not in genes
confident_present = confident[confident[['protein1', 'protein2']].isin(genes.index.values).all(axis=1)]
confident_present.to_csv(
    '/home/karin/Documents/timeTrajectories/data/44689.protein.links.detailed.v11.0.genenames_present.highconfidence.txt',
    sep='\t', index=False)

# *****************************
# ********** Make tSNE of all genes to plot regulons on it
genes_all_pp = NeighbourCalculator.get_index_query(genes=genes, inverse=False, scale='mean0std1', log=True)[0]
tsne = make_tsne(data=genes_all_pp)
tsne = pd.DataFrame(tsne, index=genes.index)
tsne.to_csv(pathByStrain + 'tsne.tsv', sep='\t')

classes = dict(zip(genes.index, ['other'] * genes.shape[0]))
regulon_genes = pd.read_table(
    pathByStrain + 'kN300_mean0std1_log/' + 'mergedGenes_minExpressed0.990.5Strains1_clustersLouvain0.4minmaxNologPCA30kN30.tab',
    index_col=0)
classes.update(dict(zip(regulon_genes.index, ['regulon'] * regulon_genes.shape[0])))

plot_tsne_colours([tsne.values], classes=[classes], names=[tsne.index], legend=True,
                  plotting_params={'regulon': {'s': 1, 'alpha': 0.6}, 'other': {'s': 0.5, 'alpha': 0.3}},
                  colour_dict={'regulon': 'red', 'other': 'black'})
