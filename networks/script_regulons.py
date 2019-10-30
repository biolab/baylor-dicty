import pandas as pd
import matplotlib.pyplot as plt

from correlation_enrichment.library import SimilarityCalculator
from networks.library_regulons import *

# Script parts are to be run separately as needed

dataPath = '/home/karin/Documents/timeTrajectories/data/RPKUM/combined/'
dataPathSaved = '/home/karin/Documents/timeTrajectories/data/correlations/replicates/'

genes = pd.read_csv(dataPath + 'mergedGenes_RPKUM.tsv', sep='\t', index_col=0)
conditions = pd.read_csv(dataPath + 'conditions_mergedGenes.tsv', sep='\t', index_col=None)

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


# Test Setting of parameters

# Cosine similarity threshold
threshold = 0.95
# Set if want to compute at multiple thresholds
thresholds = [threshold]
# thresholds=[0.85,0.9,0.92,0.94,0.96,0.98,0.99]
# Number of neighbours to obtain for each gene
neighbours_n = 3
# Scaling of genes
# scale: 'minmax' (from 0 to 1) or 'mean0std1' (to mean0 and std1)
scale = 'minmax'
# Log transform expression values before scaling
use_log = True
# Calculate inverse profiles
inverse = False
# Batches: 'replicate', 'strain', 'none'
# replicate,strain - computes neighbours for each replicate/strain separately
# and retains neighbours present at some threshold in all replicates/strains
# none - uses all samples at the same time
batches = 'none'

# Prepare data
if batches == 'none':
    batches = None
if batches == 'strain':
    batches = list(conditions[list(
        (conditions["Strain"] == 'comH') | (conditions["Strain"] == 'cudA') | (conditions["Strain"] == 'gbfA') | (
                    conditions["Strain"] == 'mybBGFP') | (conditions["Strain"] == 'pkaR'))]
                   .loc[:, 'Strain'])
if batches == 'replicate':
    batches = list(conditions[list(
        (conditions["Strain"] == 'comH') | (conditions["Strain"] == 'cudA') | (conditions["Strain"] == 'gbfA') | (
                    conditions["Strain"] == 'mybBGFP') | (conditions["Strain"] == 'pkaR'))]
                   .loc[:, 'Replicate'])
genes_sub = genes.T[list((conditions["Replicate"] == 'comH_r1')
                         | (conditions["Replicate"] == 'cudA_r2')
                         | (conditions["Replicate"] == 'gbfA_r1')
                         | (conditions["Replicate"] == 'mybBGFP_bio1')
                         | (conditions["Replicate"] == 'pkaR_bio1'))].T

#
# genes_sub=genes.T[list((conditions["Strain"]=='comH')
#                         | (conditions["Strain"]=='cudA')
#                         | (conditions["Strain"]=='gbfA')
#                         | (conditions["Strain"]=='mybBGFP')
#                         | (conditions["Strain"]=='pkaR'))].T

genes_test = genes.T[list((conditions["Replicate"] == 'AX4_bio1'))].T
neighbour_calculator = NeighbourCalculator(genes_sub)
test_index, test_query = NeighbourCalculator.get_index_query(genes_test, inverse=inverse, scale=scale, log=use_log)
gene_names = np.array(genes_test.index)
# Calculate neighbours
if batches != None:
    results = neighbour_calculator.neighbours(neighbours_n, inverse=inverse, scale=scale, log=use_log, batches=batches)
    result = neighbour_calculator.merge_results(results.values(), threshold, len(set(batches)))
else:
    result = neighbour_calculator.neighbours(neighbours_n, inverse=inverse, scale=scale, log=use_log, batches=batches)

# Filter neighbours on similarity
for threshold in thresholds:
    result_filtered = NeighbourCalculator.filter_similarities(result, threshold)
    # Calculate MSE for each gene pair -
    # compare similarity from gene subset to similarity of the gene pair in gene test set
    sq_errors = []
    for pair, similarity in result_filtered.items():
        gene1 = pair[0]
        gene2 = pair[1]
        similarity_test = SimilarityCalculator.calc_cosine(test_index[gene_names == gene1].flatten(),
                                                           test_query[gene_names == gene2].flatten())
        se = (similarity - similarity_test) ** 2
        # Happens if at least one vector has all 0 values
        if not np.isnan(se):
            sq_errors.append(se)
    print('threshold:', threshold, 'MSE:', mean(sq_errors), 'N:', len(result_filtered))

# Result:
# Sample: comH_r1, cudA_r2, gbfA_r1,mybBGBF_bio1,pkaR_bio1 and test sample: AX4_bio1; not inverse: threshold 0.95; neighbours 30
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
# Sample strains: comH, cudA, gbfA,mybBGBF,pkaR and test sample: AX4_bio1; not inverse: threshold 0.95; neighbours 30
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


# **************************
# Make regulons

# Set parameters
neighbour_n = 200
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
nx.write_pajek(graph,dataPathSaved+'kN200_t0.99_scaleMinmax_log.net')
nx.write_pajek(graph_inv,dataPathSaved+'kN200_t0.99_scaleMinmax_log_inv.net')