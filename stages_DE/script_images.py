import matplotlib.pyplot as plt
import glob
import pandas as pd
import numpy as np
import re

from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sb
from statistics import mean
import sklearn.preprocessing as pp

from Orange.clustering.louvain import jaccard

from networks.functionsDENet import loadPickle, savePickle
from networks.library_regulons import NeighbourCalculator
from stages_DE.stages_library import GROUP_DF, GROUPS, PHENOTYPES
from stages_DE.library_images import *

path = '/home/karin/Documents/timeTrajectories/data/deTime/de_time_impulse/'
lab = True
if lab:
    dataPath = '/home/karin/Documents/timeTrajectories/data/RPKUM/combined/'
    images_path = '/home/karin/Documents/timeTrajectories/data/from_huston/phenotypes/Jpeg_development/RNA-seq sample jpeg/'
    path_embedding = '/home/karin/Documents/timeTrajectories/data/stages/images/'

else:
    dataPath = '/home/karin/Documents/DDiscoideum/data/RPKUM/'

genes = pd.read_csv(dataPath + 'mergedGenes_RPKUM.tsv', sep='\t', index_col=0)
conditions = pd.read_csv(dataPath + 'conditions_mergedGenes.tsv', sep='\t', index_col=None)
conditions_img = pd.read_csv(dataPath + 'conditions_mergedGenes_images.tsv', sep='\t', index_col=None)
# ***************************************
# Add image names to conditions
files = glob.glob(images_path + '**/**/*.JPG')
conditions_images = pd.DataFrame()
conditions_images["Image"] = np.nan
rep_change = {
    'PkaCoe_r2': 'PkaCoe_r1',
    'PkaCoe_r3': 'PkaCoe_r2',
    'gbfA_r2': 'gbfA_r1',
    'gbfA_r3': 'gbfA_r2',
    'acaAPkaCoe#1_r1': 'acaAPkaCoe_r1',
    'acaAPkaCoe#3_r2': 'acaAPkaCoe_r2',
    'AX4s_SE_r7': 'AX4_SE_r7'
}
absent = 0
for file in files:
    file_name = file.replace(images_path, '')
    file = file.split('/')
    rep = file[-2].split(' ')[0]
    if rep in rep_change.keys():
        rep = rep_change[rep]
    strain = file[-3]

    name_data = re.split('-|_|\.| ', file[-1].lower())
    times = []
    for data in name_data:
        if 'h' in data:
            times.append(data)
    if len(times) != 1:
        print(times, name_data)
    else:
        time = int(times[0].replace('h', '').replace('r', ''))
    conditions_line = conditions.query('Replicate == "' + rep + '" & Time == ' + str(time)).copy()
    if conditions_line.shape[0] == 0:
        print(name_data, rep, time)
        absent += 1
    else:
        conditions_line['Image'] = file_name
        conditions_images = conditions_images.append(conditions_line, ignore_index=True)

conditions_images.to_csv(dataPath + 'conditions_mergedGenes_images.tsv', sep='\t', index=False)

# How similar are neighbours with different embeddings: embedder_comparison.ipynb

# *************************
# ******* Variance of genes within similar images
# Get image neighbours
image_group_size=4

embedding = pd.read_table(path_embedding + 'EmbeddedImages_inception-v3.tab', index_col='image')

# Some genes may be naturally more variable - compare variability on all measurements vs in image groups.
# E.g. genes almost unexpressed in all conditions will be constantly near 0
# Deviation in all conditions (even those without images)
gene_vars_all = variation_statistic(genes)
# Remove genes with 0 deviation/mean in whole population
remove_genes = gene_vars_all[gene_vars_all == 0].index
gene_vars_all = gene_vars_all.drop(remove_genes)

# Approximate best neighbours - not needed as all can be calculated!
# nc = NeighbourCalculator(genes=embedding, remove_zero=False)
# Can not select 1000 images as the samples would be repeated
# neigh, sims = nc.neighbours(n_neighbours=4, inverse=False, scale='none', log=False,
#                            return_neigh_dist=True, remove_self=True)

# Calculate all cosine similarities between genes
measurement_images = [conditions_img[conditions_img['Image'] == image]['Measurment'].values[0]
                      for image in embedding.index]
cosine_sims = pd.DataFrame(cosine_similarity(embedding), index=measurement_images, columns=measurement_images)

# Calculate gene deviances within image groups of closest neighbours
gene_vars_similar = pd.DataFrame()
# Not needed as used for approximated neighbours
# for query,neighs in neigh.iterrows():
#    images=[query]+list(neighs.values)
for query, data in cosine_sims.iterrows():
    data = data.sort_values()
    # Select 4 closest neighbours (including itself)
    # This already includes itself - as all simialrities were calculated, so itself is top
    measurements_best = data.index[-image_group_size:]
    # measurements=conditions_images[conditions_images['Image'].isin(images_best)]['Measurment'].unique()
    gene_data = genes[measurements_best]
    statistic = variation_statistic(gene_data)
    gene_vars_similar = pd.concat([gene_vars_similar, statistic], axis=1, sort=True)

# Plot mean and median var distn
# plot_variation_distn(gene_vars_similar)

# Remove all genes whose median deviation is 0 (mostly unexpressed, hard to reason about them)
# However, this could remove genes not expressed only in some stages
# (Below scaling by deviation on the whole profile is not enough to remove them from top genes)
gene_vars_similar = remove_zero_median(gene_vars_similar)

# Compare with deviations on all measurements
gene_vars_diff = diff_vars_all(gene_vars=gene_vars_similar, gene_vars_all=gene_vars_all)

# Plot distn of deviations for random genes
chosen = np.random.choice(gene_vars_similar.index, 30, replace=False)
plt.violinplot(gene_vars_similar.loc[chosen, :].values.T, showmeans=True, showmedians=True)
plt.violinplot(gene_vars_diff.loc[chosen, :].values.T, showmeans=True, showmedians=True)
# Compare expression profiles based on ranking from median deviations within images and median devisation after scaling
# based on deviation on whole profile - select top genes
genes_avg = pd.read_table(
    '/home/karin/Documents/timeTrajectories/data/regulons/genes_averaged_orange_scale99percentileMax0.1.tsv',
    index_col=0).T
best = 2000
best_sim = gene_vars_similar.median(axis=1).sort_values().index[:best]
best_diff = gene_vars_diff.median(axis=1).sort_values().index[:best]
# Overlap Jaccard index
jaccard(set(best_sim), set(best_diff))  # 0.0354
# expression of both sets of selected genes
sb.clustermap(genes_avg.loc[best_sim, :].astype('float'), col_cluster=False, xticklabels=False, yticklabels=False)
sb.clustermap(genes_avg.loc[best_diff, :].astype('float'), col_cluster=False, xticklabels=False, yticklabels=False)
# The selection based on diff is better

# Calculate deviations on random sets of 4 images
gene_vars_random = pd.DataFrame()
for i in range(gene_vars_similar.shape[1]):
    selected = np.random.choice(measurement_images, image_group_size, replace=False)
    gene_data = genes[selected]
    statistic = variation_statistic(gene_data)
    gene_vars_random = pd.concat([gene_vars_random, statistic], axis=1, sort=True)

gene_vars_random = remove_zero_median(gene_vars_random)
gene_vars_diff_random = diff_vars_all(gene_vars=gene_vars_random, gene_vars_all=gene_vars_all)

# Plot distributions of medians in randomly selected images and close neighbour images
plt.hist(gene_vars_diff.median(axis=1), bins=100, alpha=0.4, label='Close neighbours (median)')
# plt.hist(gene_vars_diff.mean(axis=1), bins=100, alpha=0.4, label='Close neighbours (mean)')
plt.hist(gene_vars_diff_random.median(axis=1), bins=100, alpha=0.4, label='Random (median)')
# plt.hist(gene_vars_diff_random.mean(axis=1), bins=100, alpha=0.4, label='Random (mean)')
plt.legend()

# Select genes with median unlikely to be randomly so low - one sided probability that below threshold
# (nonadjusted for multiple comparison)
p_threshold=0.05
threshold = gene_vars_diff_random.median(axis=1).quantile(p_threshold)
best_diff = gene_vars_diff[gene_vars_diff.median(axis=1) <= threshold].index
# Plot expression of these genes
sb.clustermap(genes_avg.loc[best_diff, :].astype('float'), col_cluster=False, xticklabels=False, yticklabels=False)

# Average expression of the selected genes in each stage
mean_expression_data = pd.DataFrame(columns=PHENOTYPES)
for gene in best_diff:
    stage_expression = defaultdict(list)
    for i, data in conditions_img.iterrows():
        # Enable this to find out stage expression only in WT
        #if data['Group'] =='WT':
            expression = genes.at[gene, data['Measurment']]
            #print(expression,data[PHENOTYPES])
            for stage in PHENOTYPES:
                if data[stage] == 1:
                    stage_expression[stage].append(expression)
    #print(stage_expression)
    stage_expression_mean = dict()
    for stage,data in stage_expression.items():
        if len(data) >0:
            stage_expression_mean[stage] = mean(stage_expression[stage])
    #print(stage_expression_mean)
    mean_expression_data=mean_expression_data.append(pd.Series(stage_expression_mean,name=gene))
# Scale based on largest avg expression in any stage
mean_expression_data=(mean_expression_data.T/mean_expression_data.max(axis=1)).T
mean_expression_data.to_csv('/home/karin/Documents/timeTrajectories/data/stages/images/stageExpression_lowVariabilityGenes_img'+
                            str(image_group_size)+'_p'+str(p_threshold)+'.tsv',sep='\t')
#Plot
sb.clustermap(pd.DataFrame(pp.minmax_scale(mean_expression_data,axis=1),columns=mean_expression_data.columns),col_cluster=False)