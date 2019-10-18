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

list_of_genesets = list_all(organism='44689')
gene_sets=load_gene_sets(list_of_genesets[1],'44689')

#OLD!!!!!
#Try enrichment for each strain and replicate:
sc=SimilarityCalculator(similarity_type='cosine',normalisation_type='mean0std1')
for strain in samples.keys():
    for rep in samples[strain]:
        start=time.time()
        print(strain,rep)
        genesStrainEID, genesStrainNNEID=f.genesByStrain(genesNotNullEID,tableEID,12735,strain+'_r'+str(rep),genesFromRow)
        ge=GeneExpression(genesStrainNNEID)
        ec=EnrichmentCalculator(ge,sc)
        result=ec.calculate_enrichment(gene_sets,3000)
        resDict=dict()
        for r in result:
            resDict[r.gene_set.name]=r.padj
        f.savePickle(dataPathSaved+'enrichment_pearson_3000_pheno'+strain+str(rep)+'.pkl',resDict)
        print(time.time()-start)

#Retain sets that are present in at least min_replicates replicates at level of max padj, average the padj
#Reps data - list of enrichment data, where list elements are dicts (resDict) of gene set names (key) and padj (value)
def merge_from_replicates(reps_data:list, max_padj,min_reps:int=2)->dict:
    merged=dict()
    for data in reps_data:
        for gene_set,padj in data.items():
            if padj<=max_padj:
                if gene_set in merged.keys():
                    merged[gene_set].append(padj)
                else:
                    merged[gene_set]=[padj]
    filtered=dict()
    for gene_set,padjs in merged.items():
        if len(padjs)>=min_reps:
            filtered[gene_set]=mean(padjs)
    return filtered


#Merge replicates data:
strain_data=dict()
for strain in samples.keys():
    reps=[]
    for rep in samples[strain]:
        rep_data=f.loadPickle(dataPathSaved+'enrichment_pearson_3000_pheno'+strain+str(rep)+'.pkl')
        reps.append(rep_data)
        #Decided not to filter on pval as Ornage cant plot heatmap NaN well
    merged=merge_from_replicates(reps,1)
    strain_data[strain]=merged
#Make enrichment table
strains=[]
datas=[]
for strain,data in strain_data.items():
    strains.append(strain)
    datas.append(data)
table_enriched=pd.DataFrame(datas)
table_enriched.index=strains
table_enriched.to_csv(dataPathSaved+'enriched_pearson_3000_pheno.tsv',sep='\t')
#Make table 0 or 1 enriched
te_filter=table_enriched.copy()
te_filter[te_filter<=0.05]=-1
te_filter[te_filter>0.05]=0
te_filter[te_filter==-1]=1
te_filter.to_csv(dataPathSaved+'enriched_pearson_3000_pheno_yn.tsv',sep='\t')

#Plot adjp distn
padjs=table_enriched.values.flatten()
padjs_filter=padjs[padjs<=0.05]
plt.hist(padjs_filter,bins=10000)
plt.xscale('log')
plt.ylabel('count')
plt.xlabel('padj')

#Retain only best pvals, replace others with 1
table_filtered=table_enriched.copy()
boundary=np.nanpercentile(padjs,5)
table_filtered[table_filtered>boundary]=1
table_filtered.to_csv(dataPathSaved+'enriched_pearson_3000_top5.tsv',sep='\t')

#Log transform -10log10(val)
table_log=np.log10(table_enriched)
table_log=table_log*-1
table_log.to_csv(dataPathSaved+'enriched_pearson_3000_-log10.tsv',sep='\t')
table_log=table_log*10
table_log.to_csv(dataPathSaved+'enriched_pearson_3000_-10log10.tsv',sep='\t')