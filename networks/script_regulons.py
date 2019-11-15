import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
from statistics import median
from scipy.cluster.hierarchy import dendrogram

from networks.library_regulons import *

# Script parts are to be run separately as needed

dataPath = '/home/karin/Documents/timeTrajectories/data/RPKUM/combined/'
dataPathSaved = '/home/karin/Documents/timeTrajectories/data/correlations/replicates/'

genes = pd.read_csv(dataPath + 'mergedGenes_RPKUM.tsv', sep='\t', index_col=0)
conditions = pd.read_csv(dataPath + 'conditions_mergedGenes.tsv', sep='\t', index_col=None)

#************************************

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
hcl_inv=HierarchicalClustering.from_knn_result(result_inv,genes,threshold,inverse=True,scale=scale,log=use_log)

hcl_gene_data=genes.loc[hcl._gene_names_ordered,:]

dendrogram(hcl_inv._hcl)

ca=ClusteringAnalyser(genes.index)
silhouettes = []
median_sizes = []
entropy=[]
ratios=[]
n_clusters = list(range(3, 60, 6))
for n in n_clusters:
    median_sizes.append(median(hcl.cluster_sizes(n)))
    silhouettes.append(ClusteringAnalyser.silhouette(hcl, n))
    entropy.append(ca.annotation_entropy(hcl,n,('KEGG', 'Pathways')))
    ratios.append(ca.annotation_ratio(hcl, n, ('KEGG', 'Pathways')))

#Taken from: https://matplotlib.org/3.1.1/gallery/ticks_and_spines/multiple_yaxis_with_spines.html
fig, host = plt.subplots()
fig.subplots_adjust(right=0.75)

par1 = host.twinx()
par2 = host.twinx()
par2.spines["right"].set_position(("axes", 1.2))

p11 = host.scatter(n_clusters, silhouettes, c="b")
p1, = host.plot(n_clusters, silhouettes, "b-",label="Silhouette")
p22 = par1.scatter(n_clusters, median_sizes, c="r")
p2, = par1.plot(n_clusters, median_sizes, "r-", label="Median size")
p33 = par2.scatter(n_clusters, ratios, c="g")
p3, = par2.plot(n_clusters, ratios, "g-",label="Annotation ratio")

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
host.tick_params(axis='y', colors=p1.get_color() )
par1.tick_params(axis='y', colors=p2.get_color())
par2.tick_params(axis='y', colors=p3.get_color())
host.tick_params(axis='x' )

#lines = [p1, p2,p3]

#host.legend(lines, [l.get_label() for l in lines])

#*********************************

# Get genes for orange:
genes_orange_scaled,genes_orange_avg,patterns=preprocess_for_Orange(genes=genes, threshold=0.99,conditions=conditions,
                                                           split_by='Strain',average_by='Time',matching='Measurment',
                                                                    strain_pattern='AX4')
genes_orange_scaled.to_csv('/home/karin/Documents/timeTrajectories/data/regulons/genes_scaled_orange.tsv',sep='\t')
# Transpose so that column names unique (else Orange problems)
genes_orange_avg.T.to_csv('/home/karin/Documents/timeTrajectories/data/regulons/genes_averaged_orange.tsv',sep='\t')
patterns.to_csv('/home/karin/Documents/timeTrajectories/data/regulons/gene_patterns_orange.tsv',sep='\t',index=False)

#********************
# Check how many hypothetical and pseudogenes are in onthologies
#Get gene descriptions and EIDs
entrez_descriptions = dict()
matcher = GeneMatcher(44689)
matcher.genes = genes.index
for gene in matcher.genes:
    description = gene.description
    entrez = gene.gene_id
    if entrez is not None:
        #Key: EID, val: list [description, n GOs]
        entrez_descriptions[entrez]=[description,0]
#For each gene set
for ontology in list_all(organism=str(44689)):
    gene_sets = load_gene_sets(ontology, '44689')
    for gene_set in gene_sets:
        for gene_EID in gene_set.genes:
            if gene_EID in entrez_descriptions.keys():
                entrez_descriptions[gene_EID][1]+=1

hypothetical=[]
pseudo=[]
annotated=[]
for description,n_terms in entrez_descriptions.values():
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