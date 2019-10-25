import pandas as pd
import matplotlib.pyplot as plt

from correlation_enrichment.library import SimilarityCalculator
from networks.library_regulons import *

# Script parts are to be run separately as needed

dataPath = '/home/karin/Documents/timeTrajectories/data/RPKUM/combined/'
dataPathSaved = '/home/karin/Documents/timeTrajectories/data/correlations/replicates/'

genes = pd.read_csv(dataPath + 'mergedGenes.tsv', sep='\t', index_col=0)

sub=genes.shape[1]
genes_sub=genes.iloc[:,:sub]
neighbour_calculator = NeighbourCalculator(genes_sub)

neighbours_minmax = neighbour_calculator.neighbours(10, False, scale='minmax')
neighbours_meanstd = neighbour_calculator.neighbours(10, False, scale='mean0std1')
# Plot how it looks to have similarity above threshold:
neighbours=neighbours_minmax
sim_threshold = 0.96
step = 0.01
max_plot = 1
for gene_pair, sim in neighbours.items():
    if max_plot <= 9:
        if sim_threshold <= sim < sim_threshold + step:
            sim_minmax='NA'
            if gene_pair in neighbours_minmax.keys():
                sim_minmax=round(neighbours_minmax[gene_pair],3)
            sim_meanstd = 'NA'
            if gene_pair in neighbours_meanstd.keys():
                sim_meanstd = round(neighbours_meanstd[gene_pair],3)
            plt.subplot(3, 3, max_plot)
            plt.title('minmax '+str(sim_minmax)+', mean0std1 '+str(sim_meanstd),fontdict = {'size': 8 })
            print(gene_pair)
            gene1 = pp.minmax_scale(list(genes_sub.loc[gene_pair[0],:]))
            gene2 = pp.minmax_scale(list(genes_sub.loc[gene_pair[1],:]))
            plt.plot(list(range(sub)), gene1, color='g', linewidth=1)
            plt.plot(list(range(sub)), gene2, color='m', linewidth=1)
            max_plot += 1

# Plot distribution of similarities
plt.hist(neighbours_meanstd.values())

# Compare difference in N of detected pairs at threshold depending on n neighbours

inverse=True
result0=np.array(list(neighbour_calculator.neighbours(20,inverse,scale='minmax').values()))
result1=np.array(list(neighbour_calculator.neighbours(30,inverse,scale='minmax').values()))
result2=np.array(list(neighbour_calculator.neighbours(40,inverse,scale='minmax').values()))
# What percent of shorter result's values is missing
threshold=0.95
resultA=result0
resultB=result1
(sum(resultB>=threshold)-sum(resultA>=threshold))/len(resultA)

# Calculate MSE of left out sample
sub=20
threshold=0.95
genes_sub=genes.iloc[:,sub:]
genes_test=genes.iloc[:,:sub]
neighbour_calculator=NeighbourCalculator(genes_sub)
result= neighbour_calculator.neighbours(10, False, scale='minmax')
result_filtered=NeighbourCalculator.filter_similarities(result,threshold)
test_index,test_query=NeighbourCalculator.get_index_query(genes_test,False,'minmax')
gene_names=np.array(genes_test.index)
sq_errors=[]
for pair,similarity in result_filtered.items():
    gene1=pair[0]
    gene2=pair[1]
    similarity_test=SimilarityCalculator.calc_cosine(test_index[gene_names==gene1].flatten(),
                                                     test_query[gene_names==gene2].flatten())
    se=(similarity-similarity_test)**2
    # Happens if at least one vector all 0
    if not np.isnan(se):
        sq_errors.append(se)
print(mean(sq_errors))