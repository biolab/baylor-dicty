import time
import seaborn as sb

from Orange.widgets.tests.base import datasets
from orangecontrib.bioinformatics.geneset.__init__ import (list_all, load_gene_sets)
import matplotlib.pyplot as plt

from correlation_enrichment.library_correlation_enrichment import *
import networks.functionsDENet as f
import deR.enrichment_library as el

lab = True
# dataPath='/home/karin/Documents/git/baylor-dicty/data_expression/'
if lab:
    dataPath = '/home/karin/Documents/timeTrajectories/data/RPKUM/combined/'
    dataPathSaved = '/home/karin/Documents/timeTrajectories/data/timeGeneSets_correlation_enrichment/'
else:
    dataPathSaved = '/home/karin/Documents/DDiscoideum/data/enrichmentMultiD/'

samples = {
    'AX4': [1, 2, 5, 6, 7, 8, 9],
    'tagB': [1, 2],
    'comH': [1, 2],
    'tgrC1': [1, 2],
    'tgrB1': [1, 2],
    'tgrB1C1': [1, 2],
    'gbfA': [1, 2],
}

genesFromRow = 2
tableEID = f.loadPickle(dataPathSaved + 'trans_9repAX4_6strains_2rep_avr_T12_EID.tab')
genesEID, genesNotNullEID = f.extractGenesFromTable(tableEID, genesFromRow, 12734)

gene_sets = load_gene_sets(('GO', 'biological_process'), '44689')

# Enrichment for each strain, merging replicates horizontally
sc = SimilarityCalculator(similarity_type='cosine', normalisation_type='mean0std1')
for strain in samples.keys():
    genesStrainEID, genesStrainNNEID = f.genesByKeyword(genesNotNullEID, tableEID, 12735, strain + '_r', genesFromRow)
    start = time.time()
    ge_strain = GeneExpression(genesStrainNNEID)
    ec_strain = EnrichmentCalculator.quick_init(ge_strain, sc)
    result = ec_strain.calculate_enrichment(gene_sets, 5000)
    print(time.time() - start)
    f.savePickle(dataPathSaved + 'enrichment_cosine_5000_pheno_' + strain + '.pkl', result)
    resDict = dict()
    for r in result:
        resDict[r.gene_set.name] = r.padj
    f.savePickle(dataPathSaved + 'enrichment_cosine_5000_pheno_' + strain + '_dict.pkl', resDict)

# Make enrichment table
strains = []
datas = []
for strain in samples.keys():
    data = f.loadPickle(dataPathSaved + 'enrichment_cosine_5000_pheno_' + strain + '_dict.pkl')
    strains.append(strain)
    datas.append(data)
table_enriched = pd.DataFrame(datas)
table_enriched.index = strains
table_enriched.to_csv(dataPathSaved + 'enriched_cosine_5000_pheno.tsv', sep='\t')

table_enriched = pd.read_csv(dataPathSaved + 'enriched_cosine_5000_pheno.tsv', sep='\t', index_col=0)

# Retain only best pvals, replace others with 1
table_filtered = table_enriched.copy()
padjs = table_enriched.values
index = 0
for row in padjs:
    boundary = np.nanpercentile(row, 5)
    table_filtered.iloc[index, :][table_filtered.iloc[index, :] > boundary] = -1
    index += 1
table_filtered[table_filtered < 0] = 'NaN'
table_filtered = table_filtered.loc[:, (table_filtered != 'NaN').any(axis=0)]
table_filtered.to_csv(dataPathSaved + 'enriched_cosine_5000_pheno_top5.tsv', sep='\t')

# Log transform
table_log = np.log10(table_enriched)
table_log = table_log * -1
table_log[table_log == float('inf')] = 500
table_log.to_csv(dataPathSaved + 'enriched_cosine_5000_pheno_-log10.tsv', sep='\t')

# ******************************
# Compare gene sets:
strain = 'AX4'
# genesStrainEID, genesStrainNNEID = f.genesByKeyword(genesNotNullEID, tableEID, 12735, strain + '_r', genesFromRow)
genesEID = f.loadPickle(dataPath + 'mergedGenes_RPKUM_EID.pkl')
conditions = pd.read_csv(dataPath + 'conditions_mergedGenes.tsv', sep='\t', index_col=None)
genesStrainEID = genesEID.loc[:, np.array(conditions['Strain'] == 'AX4')]

# TODO
ge_strain = GeneExpression(genesStrainEID)
ec_strain = EnrichmentCalculator.quick_init(ge_strain, sc)
gsc = GeneSetComparator(ec_strain.calculator)
# Max pairs not really needed as max for 100<5000
# enrichment_data = ec_strain.calculate_enrichment(gene_sets=gene_sets, max_pairs=5000, max_set_size=100)
# f.savePickle(dataPathSaved + 'enrichment_cosine_5000_GOprocess_maxSize100_' + strain + '.pkl',enrichment_data)
enrichment_data=f.loadPickle(dataPathSaved + 'enrichment_cosine_5000_GOprocess_maxSize100_' + strain + '.pkl')
top_padj = filter_enrichment_data_top(data=enrichment_data, metric='padj',best=5)
top_mean = filter_enrichment_data_top(data=enrichment_data, metric='mean',best=5)
top=top_padj
set_pairs = gsc.make_set_pairs(top, include_identical=True)
gsc.between_set_similarities(set_pairs,mode='sampling',max_pairs=100)
n_sets = len(top)
matrix = np.ones((n_sets, n_sets))
matrix = pd.DataFrame(matrix)
set_names = []
for gene_set in top:
    set_names.append(gene_set.gene_set.name)
matrix.index = set_names
matrix.columns = set_names
for pair in set_pairs:
    name1 = pair.gene_set_data1.gene_set.name
    name2 = pair.gene_set_data2.gene_set.name
    mean_sim = pair.mean_profile_similarity
    matrix.loc[name1, name2] = mean_sim
    matrix.loc[name2, name1] = mean_sim
#matrix.to_csv(dataPathSaved + 'enrichment_cosine_5000_pheno_AX4_topSetSimilarity.tsv', sep='\t')

sb.clustermap(matrix)


