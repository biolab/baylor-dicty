import time

from statistics import (mean, stdev,median)
from orangecontrib.bioinformatics.geneset.__init__ import (list_all,load_gene_sets)
import matplotlib.pyplot as plt

from correlation_enrichment.library import *
import networks.functionsDENet as f


dataPath='/home/karin/Documents/git/baylor-dicty/data_expression/'
dataPathSaved='/home/karin/Documents/timeTrajectories/data/timeGeneSets/'

samples={
'AX4':[1,2,5,6,7,8,9],
'tagB':[1,2],
'comH':[1,2],
'tgrC1':[1,2],
'tgrB1':[1,2],
'tgrB1C1':[1,2],
'gbfA':[1,2],
}

genesFromRow=2
tableEID=f.loadPickle(dataPathSaved+'trans_9repAX4_6strains_2rep_avr_T12_EID.tab')
genesEID,genesNotNullEID=f.extractGenesFromTable(tableEID,genesFromRow,12734)

gene_sets=load_gene_sets(('Dictybase', 'Phenotypes'),'44689')

#Enrichment for each strain, merging replicates horizontally
sc=SimilarityCalculator(similarity_type='cosine',normalisation_type='mean0std1')
for strain in samples.keys():
    genesStrainEID, genesStrainNNEID = f.genesByKeyword(genesNotNullEID, tableEID, 12735, strain + '_r', genesFromRow)
    start = time.time()
    ge_strain = GeneExpression(genesStrainNNEID)
    ec_strain = EnrichmentCalculator.quick_init(ge_strain, sc)
    result = ec_strain.calculate_enrichment(gene_sets, 5000)
    print(time.time() - start)
    resDict = dict()
    for r in result:
        resDict[r.gene_set.name] = r.padj
    f.savePickle(dataPathSaved + 'enrichment_cosine_5000_pheno_' + strain + '_dict.pkl', resDict)

#Make enrichment table
strains=[]
datas=[]
for strain in samples.keys():
    data=f.loadPickle(dataPathSaved + 'enrichment_cosine_5000_pheno_' + strain + '_dict.pkl')
    strains.append(strain)
    datas.append(data)
table_enriched=pd.DataFrame(datas)
table_enriched.index=strains
table_enriched.to_csv(dataPathSaved+'enriched_cosine_5000_pheno.tsv',sep='\t')


#Retain only best pvals, replace others with 1
table_filtered=table_enriched.copy()
padjs=table_enriched.values
index=0
for row in padjs:
    boundary = np.nanpercentile(row, 5)
    table_filtered.iloc[index,:][table_filtered.iloc[index,:]>boundary]=-1
    index+=1
table_filtered[table_filtered<0]='NaN'
table_filtered.to_csv(dataPathSaved+'enriched_pearson_3000_top5.tsv',sep='\t')

