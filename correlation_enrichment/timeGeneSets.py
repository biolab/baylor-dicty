import time
from statistics import (mean, stdev,median)
import networks.functionsDENet as f
from orangecontrib.bioinformatics.geneset.__init__ import (list_all,load_gene_sets)
import matplotlib.pyplot as plt
from correlation_enrichment.timeGeneSetsObj import *
from scipy.stats import (mannwhitneyu,ks_2samp,ttest_ind)
import random
import math
#Load data
#dataPath='/home/karin/Documents/DDiscoideum/'
#dataPathSaved='/home/karin/Documents/DDiscoideum/timeTrajectoriesNet/data/enrichmentMultiD/'
dataPath='/home/karin/Documents/git/baylor-dicty/data_expression/'
dataPathSaved='/home/karin/Documents/timeTrajectories/data/timeGeneSets/'
tableLayout=dict()
tableLayout['rep']=dict()
tableLayout['single']=dict()
tableLayout['single']['lastGene']=12734
tableLayout['single']['Time']=12737
tableLayout['single']['Strain']=12736
tableLayout['rep']['lastGene']=12868
tableLayout['rep']['Time']=12870
tableLayout['rep']['Strain']=12869
#data=tableLayout['single']
data=tableLayout['rep']
genesFromRow=2

samples={
'AX4':[1,2,5,6,7,8,9],
'tagB':[1,2],
'comH':[1,2],
'tgrC1':[1,2],
'tgrB1':[1,2],
'tgrB1C1':[1,2],
'gbfA':[1,2],
}

#geneNames=f.loadPickle(dataPath+'Genes.pkl')

#table=f.importTable(dataPath+'trans_9repAX4_6strains_2rep_avr_T12.tab')

#Name table with entrez IDs, if EID is not know remove the gene column
#tableEID=table.copy()
#columns=tableEID.columns
#columnsEID=[]
#geneNamesDDB=geneNames.keys()
#for colname in columns:
#    if colname in geneNamesDDB:
#        eid=geneNames[colname][1]
#        if eid != '':
#            columnsEID.append(eid)
#        else:
#            del tableEID[colname]
#    else:
#        columnsEID.append(colname)
#tableEID.columns=columnsEID

#f.savePickle(dataPathSaved+'trans_9repAX4_6strains_2rep_avr_T12_EID.tab',tableEID)
tableEID=f.loadPickle(dataPathSaved+'trans_9repAX4_6strains_2rep_avr_T12_EID.tab')

genesEID,genesNotNullEID=f.extractGenesFromTable(tableEID,genesFromRow,12734)

#strain='AX4_avr'
#repN=1

#genesStrainEID, genesStrainNNEID=f.genesByStrain(genesNotNullEID,tableEID,12735,strain+'_r'+str(repN),genesFromRow)

#Get gene set
list_of_genesets = list_all(organism='44689')
gene_sets=load_gene_sets(list_of_genesets[3],'44689')

#Objects for testing:
genesStrainEID, genesStrainNNEID=f.genesByStrain(genesNotNullEID,tableEID,12735,'AX4_avr',genesFromRow)
sc=SimilarityCalculator()
ge=GeneExpression(genesStrainNNEID)
rscn=RandomSimilarityCalculatorNavigator(ge,sc)
gsscn=GeneSetSimilarityCalculatorNavigator(ge,sc,rscn)


#Calculate mean and stdev distn
genesStrainEID, genesStrainNNEID=f.genesByStrain(genesNotNullEID,tableEID,12735,'AX4_avr',genesFromRow)
sc=SimilarityCalculator()
ge=GeneExpression(genesStrainNNEID)

sims=[]
pairN=[3,4,5,10,20,40,60,100,300,500,1000,5000,10000,50000,100000]
rscn=RandomSimilarityCalculatorNavigator(ge,sc)
for pN in pairN:
    print(pN)
    sims.append(rscn.similarities(pN,True))
sds=[]
means=[]
for s in sims:
    means.append(s.mean_val)
    sds.append(s.std)
plt.scatter(pairN,means)
plt.plot(pairN,means)
plt.xscale('log')
plt.scatter(pairN,sds)
plt.plot(pairN,sds)
plt.xlabel('n random pairs')
plt.ylabel('mean (blue) and stdev (orange) similarity')

#Calculate MSE (Used: pearson,Ax4_avg, random max pairs set to 5000,GO mollecular function?)
#Select gene sets with more genes than pairs
sc=SimilarityCalculator(similarity_type='correlation_pearson')
ec=EnrichmentCalculatorFactory.make_enrichment_calculator(ge,sc)
large_sets=[]
for s in gene_sets:
    if len(s.genes)>300:
        large_sets.append(s)
#Reduce number of sets:
max_sets=20
if len(large_sets)>max_sets:
    large_sets=large_sets[:max_sets]

pairN=[500,800,1000,2000,3500,5000,10000,20000,7500]
results=[]
res=ec.calculate_enrichment(large_sets)
results.append(res)
for n in pairN:
    print(n)
    res=ec.calculate_enrichment(large_sets,max_pairs=n)
    results.append(res)
#MSE
mse=[]
max_se=[]
for n in range(1,len(results)):
    errorsSquared=[]
    for s in range(max_sets):
        padjOriginal=results[0][s].padj
        padjShortened=results[n][s].padj
        errSq=(padjOriginal-padjShortened)**2
        errorsSquared.append(errSq)
    mse.append(mean(errorsSquared))
    max_se.append(max(errorsSquared))
#Sort for plotting
mse=[x for _, x in sorted(zip(pairN,mse), key=lambda pair: pair[0])]
max_se=[x for _, x in sorted(zip(pairN,max_se), key=lambda pair: pair[0])]
pairN=sorted(pairN)
plt.plot(pairN,mse)
plt.scatter(pairN,mse)
plt.plot(pairN,max_se)
plt.ylabel('MSE (blue) and max SE (orange) padj')
plt.xlabel('n pairs')


#Try enrichment for each strain and replicate:
sc=SimilarityCalculator(similarity_type='correlation_pearson')
for strain in samples.keys():
    for rep in samples[strain]:
        start=time.time()
        print(strain,rep)
        genesStrainEID, genesStrainNNEID=f.genesByStrain(genesNotNullEID,tableEID,12735,strain+'_r'+str(rep),genesFromRow)
        ge=GeneExpression(genesStrainNNEID)
        ec=EnrichmentCalculatorFactory.make_enrichment_calculator(ge,sc)
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

#********************Not normal!

#Plot hitogram of similarities for random sample:
sims=rscn.similarities(10000)
plt.hist(sims,bins=1000)
plt.xlabel('similarity')
plt.ylabel('count')

#Plot point based histogram (points of bin heights connected with line)
#Description - for legend, width - line width,
# rm_zeros - remove zero count points from line ploting (if graph will be later ploted with log scale),
# annotation - add annotation to beginning of the line
# pdfORcdf - plot proportion of counts in a bin or cumulative of the former
def point_hist(data,ax,description='',width=1,bins=10,rm_zeros=True,annotation="",pdfORcdf=True,proportion=False):
    #Bins
    counts, bin_edges = np.histogram(data, bins)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.
    if proportion:
        counts=counts/len(data)
    x=bin_centres
    y=counts
    if not pdfORcdf:
        cdf=[]
        position=0
        for count in counts:
            if position==0:
                cdf.append(count)
            else:
                cdf.append(count+cdf[position-1])
            position+=1
        y=cdf
    #Plot points
    ax.scatter(x, y, s=8, label=description)
    #Annotate
    if annotation !='':
        ax.annotate(annotation,(bin_centres[0],counts[0]))
    #Plot line
    count_counts = 0
    if rm_zeros:
        y_NN = []
        x_NN = []
        for count in y:
            if count != 0:
                y_NN.append(y[count_counts])
                x_NN.append(x[count_counts])
            count_counts += 1
        x=x_NN
        y=y_NN
    ax.plot(x, y,linewidth=width)


#Plot similarities distribution for random pairs (bold) and gene sets,
# add  Kolmogorov Smernov two-sided p values to line starts (compares medians) and whether mean is greater than random mean (T/F)
#Print out gene set name and number of genes

#Should log y scale be used
use_log=False
#Plot  histogram or cumulative  histogram
pdfORcdf=True
# Convert counts into porportions (count in bin/total N similarities)
proportion=True
#Which test to use: Mann-Whitney: 'MW', Kolmogorov Smernov: 'KS' or permutation mean based 'pm', Welsch t test: 'Wt'
test='pm'
fig = plt.figure()
ax = fig.add_subplot(111)
#max_sims - Max number of similarities to calculate for random pairs/any gene set
max_sims=1000
max_sims_random=5000
#Add random
simsRandom=rscn.similarities(max_sims_random)
mRandom=mean(simsRandom)
point_hist(simsRandom,ax,'random'+' m: '+str(round(mRandom,2)),width=3,rm_zeros=use_log,annotation='rand'+str(round(mRandom,2)),pdfORcdf=pdfORcdf,proportion=proportion)
#print(median(simsRandom))
#Add sets similarities
count_sets=0
set_num=0
#How many sets to plot
while count_sets< 8:
    gene_set=list(gene_sets)[set_num]
    set_num+=1
    try:
        sims=gsscn.similarities(gene_set.genes,max_sims)
        #print(median(sims))
        count_sets+=1
        p=None
        m=mean(sims)
        if test=='KS':
            p=ks_2samp(sims,simsRandom)[1]
        elif test == 'MW':
            p = mannwhitneyu(sims, simsRandom,alternative='greater')[1]
        elif test=='Wt':
            t, p2 = ttest_ind(sims,simsRandom,equal_var=False)
            p = p2 / 2
            if t < 0:
                p = 1 - p
        elif test=='pm':
            size_sims=len(sims)
            randomMeans=[]
            bootstraps=1000
            for i in range(bootstraps):
                #randomMeanSample = mean(rscn.similarities(size_sims))
                #Faster - uses only precomputed correlations from smaller population
                randomMeanSample = mean(random.sample(simsRandom,size_sims))
                randomMeans.append(randomMeanSample)
            greater=sum(randomMean >= m for randomMean in randomMeans)
            p=greater/bootstraps
        point_hist(sims,ax,gene_set.name +' N: '+ str(len(gene_set.genes))+' m: '+str(round(m,2))+' p: '+str(round(p,4)),rm_zeros=use_log,annotation=str(round(p,3))+' '+str(round(m,3)),pdfORcdf=pdfORcdf,proportion=proportion)
        print(gene_set.name , str(len(gene_set.genes)))
    except EnrichmentError:
        pass
plt.xlabel('Similarity bins peaks')
if proportion:
    plt.ylabel('Proportion of pairs')
else:
    plt.ylabel('Count')
if use_log:
    plt.yscale('log')
#plt.legend(loc='upper left')


