from networks.library_regulons import *

lab = True
if lab:
    dataPath = '/home/karin/Documents/timeTrajectories/data/RPKUM/combined/'
    dataPathSaved = '/home/karin/Documents/timeTrajectories/data/regulons/clusters/'

else:
    dataPath = '/home/karin/Documents/DDiscoideum/data/RPKUM/'
    dataPathSaved = '/home/karin/Documents/DDiscoideum/data/regulons/'

genes = pd.read_csv(dataPath + 'mergedGenes_RPKUM.tsv', sep='\t', index_col=0)
conditions = pd.read_csv(dataPath + 'conditions_mergedGenes.tsv', sep='\t', index_col=None)

selected_genes_names = list(pd.read_table('/home/karin/Documents/timeTrajectories/Orange_workflows/regulons/' +
                                          'kNN2_threshold0.95_m0s1log_Orange.tsv')['Gene'])
genes_selected = genes.loc[selected_genes_names, :]

# Split data by strain
merged = ClusterAnalyser.merge_genes_conditions(genes=genes_selected, conditions=conditions[['Measurment', 'Strain']],
                                                matching='Measurment')
splitted = ClusterAnalyser.split_data(data=merged, split_by='Strain')
for strain, data in splitted.items():
    data = data.drop(['Strain', 'Measurment'], axis=1).T
    data = get_orange_scaled(genes=data, scale='mean0std1', log=True)
    splitted[strain] = data
splitted['all'] = get_orange_scaled(genes=genes_selected, scale='mean0std1', log=True)

# Louvain clustering
# Select params for by strain based on AX4
data = splitted['AX4']
tsneAX4 = make_tsne(data=data)
clustering = LouvainClustering.from_orange_graph(data=data.to_numpy(), gene_names=data.index, neighbours=30)
clusters = clustering.get_clusters_by_genes(splitting={'resolution': 1.8})
plot_tsne(tsne=tsneAX4, classes=clusters, names=data.index, legend=True)

# Perform clustering
clustered = dict()
resolution=0.7
for strain, data in splitted.items():
    #resolution = 0.7 if strain == 'all' else 2
    clustered[strain] = LouvainClustering.from_orange_graph(data=data.to_numpy(), gene_names=data.index, neighbours=30
                                                            ).get_clusters_by_genes(
        splitting={'resolution': resolution})

# Make matrix of clusters by gene for each strain/all
clusters_df = pd.DataFrame(list(clustered.values()), index=clustered.keys()).T
clusters_df.to_csv(dataPathSaved+'kNN2_threshold0.95_m0s1log_clusters_LouvainRes'+str(resolution)+'.tsv',sep='\t')