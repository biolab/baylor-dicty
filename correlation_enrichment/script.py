import time
import random
from math import sqrt

from statistics import (mean, stdev,median)
import matplotlib.pyplot as plt
from scipy.stats import (mannwhitneyu,ks_2samp,ttest_ind,norm)
import altair as alt

from orangecontrib.bioinformatics.geneset.__init__ import (list_all,load_gene_sets)

from correlation_enrichment.library import *
import networks.functionsDENet as f

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
#Example of random similarities
simsRandom=f.loadPickle(dataPathSaved+'random500000.pkl')
#strain='AX4_avr'
#repN=1

#genesStrainEID, genesStrainNNEID=f.genesByStrain(genesNotNullEID,tableEID,12735,strain+'_r'+str(repN),genesFromRow)

#Get gene set
list_of_genesets = list_all(organism='44689')
gene_sets=load_gene_sets(list_of_genesets[1],'44689')

#Objects for testing:
genesStrainEID, genesStrainNNEID=f.genesByStrain(genesNotNullEID,tableEID,12735,'AX4_avr',genesFromRow)
sc=SimilarityCalculator()
ge=GeneExpression(genesStrainNNEID)
rscn=RandomSimilarityCalculatorNavigator(ge,sc)
gsscn=GeneSetSimilarityCalculatorNavigator(ge,sc,rscn)
rss=RandomMeanStorage(simsRandom)
ec=EnrichmentCalculator(random_storage=rss,gene_set_calculator=gsscn)

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

#Calculate MSE (Used: spearman,Ax4_avg, random max pairs set to 500000,GO mollecular function)
#Select gene sets specified size
large_sets=[]
min_points=300
max_points=400
for s in gene_sets:
    n_genes=len(s.genes)
    if n_genes>=min_points and n_genes<=max_points:
        large_sets.append(s)
#Reduce number of sets:
max_sets=20
if len(large_sets)>max_sets:
    large_sets=large_sets[:max_sets]

#Decide on number of similarities  used for calculaion
min_possible_pair=possible_pairs(min_points)
pairN=[]
for i in range(1,11):
    pairN.append(round(min_possible_pair/20*i,0))
#Calculate p values
#TODO remember padj scale (eg. was it small/large)
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
    for s in range(len(large_sets)):
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
#plt.plot(pairN,max_se)
#plt.ylabel('MSE (blue) and max SE (orange) padj')
plt.ylabel('MSE padj')
avg_points=0
for i in large_sets:
    avg_points+=len(i.genes)
avg_pairs=possible_pairs(round(avg_points/len(large_sets),0))
plt.xlabel('n pairs (out of at least '+str(possible_pairs(min_points))+' possible, average '+str(avg_pairs)+')')


#Try enrichment for each strain and replicate:
sc=SimilarityCalculator(similarity_type='correlation_pearson')
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
def point_hist(data,ax,legend_description='',width=1,bins=10,rm_zeros=False,annotation="",pdfORcdf=True,proportion=False):
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
    ax.scatter(x, y, s=8, label=legend_description)
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


#Calculates summary statistics of random samples
#random_similarities - list of random similarities to take samples from
#sample_size - individual sample size
#n_samples - number of samples to take
#summary_stats - summary statistics  to use on each sample , 'mean' or 'median'
#Returns list of summary statistics for individual samples
def random_samples(random_similarities:list,sample_size:int,n_samples:int,summary_stats) ->list:
    randomStats = []
    for i in range(n_samples):
        # randomSampleStat = summary_stats_function(rscn.similarities(sample_size))
        # Faster - uses only precomputed correlations from smaller population
        #Np choice without replacement is slower than list version (but not if used with replacement)
        #randomSampleStat = summary_stats_function(npran.choice(random_similarities, sample_size,replace=False))
        sample=random.sample(random_similarities, sample_size)
        #Np mean quicker than list mean, even when included converting to np.array; but list median quicker than if converted to np array
        randomSampleStat=calculate_summary_stats(sample,summary_stats)
        randomStats.append(randomSampleStat)
    return randomStats

#Calculates p value for sample summary statistics vs summary statistic of samples from random population
#similarities - sample similarities
#n_samples - initial number of permutations
#Other args as in random_samples
def calculate_permutation_p(similarities,random_similarities,summary_stats,n_samples=1000):
    sample_stat=calculate_summary_stats(similarities,summary_stats)
    sample_size=len(similarities)
    #random_similarities=np.array(random_similarities)
    p,randomMeans=permutation_p(sample_stat,random_similarities, sample_size, n_samples, summary_stats)
    repeats=0
    while p<=(1/n_samples)*10:
        print(p,(1/n_samples)*10)
        if repeats>=2:
            break
        repeats+=1
        n_samples=n_samples*10
        p,randomMeans=permutation_p(sample_stat,random_similarities, sample_size, n_samples, summary_stats)
    return (p,randomMeans)

#Calculate summary statistics
#data - the sample for calculation
#summary_stats:'mean','median'
#Calculation of mean and stdev is quicker when list is first transformed to np array
def calculate_summary_stats(data:list,summary_stats:str='mean'):
    if summary_stats == 'mean':
        stat = np.array(data).mean()
    if summary_stats == 'median':
        stat = median(data)
    return stat

#Helper function for calculate_bootstrap_p
#sample_stat - statistics on the sample
#Other arguments as in random_samples
def permutation_p(sample_stat,random_similarities, sample_size, n_samples, summary_stats='mean'):
    randomMeans = np.array(random_samples(random_similarities, sample_size, n_samples, summary_stats))
    #Quicker than list comprehention
    greater = (randomMeans>=sample_stat).sum()
    p = greater / n_samples
    return (p,randomMeans)


#Plot similarities distribution for random pairs (bold) and gene sets,
# add  Kolmogorov Smernov two-sided p values to line starts (compares medians) and whether mean is greater than random mean (T/F)
#Print out gene set name and number of genes

#Should log y scale be used
use_log=False
#Plot  histogram or cumulative  histogram
pdfORcdf=True
# Convert counts into proportions (count in bin/total N similarities)
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
            p,randomMeans=calculate_permutation_p(sims,simsRandom,'mean')
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

#Distribution of random pair statistics based on random pair sample sizes

#Plot distribution of random means/medians and lines for gene sets statistics

#Calculate subset of random similarities
#print('random sims')
#max_sims_random=500000
#simsRandom=np.array(rscn.similarities(max_sims_random))
simsRandom=f.loadPickle(dataPathSaved+'random500000.pkl')

#Make histograms of summary statistics for different random pairs sample sizes (based on number of similarities, not points)
#sample_sizes=[6,10,15,21,40,100,500,1000,5000,10000] # N similarities
sample_sizes=[] # N similarities
n_points=[3,6,15,20,100,200]
for point_n in n_points:
    sample_sizes.append(possible_pairs(point_n))
n_samples=1000
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
#Uncomment selected function
summary_function='mean'
#summary_function='median'
index=0
for sample_size in sample_sizes:
    print(sample_size)
    means=random_samples(simsRandom,sample_size,n_samples,summary_function)
    point_hist(means,ax,'points: '+str(n_points[index])+' sims: '+str(sample_size),rm_zeros=False)
    index+=1

plt.xlabel('Random similarity sample '+summary_function)
plt.ylabel('Count')
plt.legend(title='Similarities sample sizes')

#Plot gene set summary statistic vs distribution of random summary statistics
#Left plot - distribution of random summary statistics with marked sample summary statistics and permutation test p value
#right plot - distribution of similarities for random sample and gene set
count_sets=0
set_num=0
n_sets=5
max_sims=1000
summary_function='mean'
#How many sets to plot
plt.figure(figsize=(12, 12))
while count_sets< n_sets:
    # Add int to set_num to try other gene sets
    gene_set=list(gene_sets)[set_num+20]
    set_num+=1
    try:
        sims=gsscn.similarities(gene_set.genes,max_sims)
        #print(median(sims))
        p=None
        m=calculate_summary_stats(sims,summary_function)
        size_sims=len(sims)
        p,randomMeans=calculate_permutation_p(sims,simsRandom,'mean')
        ax1 =plt.subplot(n_sets , 2, count_sets*2+1)
        ax1.hist(randomMeans, weights=np.ones(len(randomMeans)) / len(randomMeans),bins=20)
        ax1.vlines(m,0,0.33)
        plt.ylabel('Proportion')
        plt.xlabel("Sample's "+summary_function)
        ax1.annotate('p='+str(p), (m+m/100, 0.15))

        ax2 = plt.subplot(n_sets, 2, count_sets*2+2)
        point_hist(sims,ax2,legend_description=gene_set.name +" N: "+str(len(gene_set.genes)),proportion=True)
        point_hist(simsRandom, ax2, legend_description='random',proportion=True)
        ax2.legend(prop={'size': 6})
        plt.ylabel('Proportion')
        plt.xlabel('Similarities')
        print(gene_set.name , str(len(gene_set.genes)))
        count_sets+=1
    except EnrichmentError:
        pass


#Time permutation test based on number of permutations/smallest possible p value
#This is not the same as whole proces of p value calculation, as there permutation tests are repeated with more permutation if p value is low
p_decimals=5 # Number of decimals for p value: the p value is not necessarily reliable to this decimal
bootstraps=10**p_decimals # Number of permutations required to obtain such a decimal (any digits from 0 to 9 on it)
sample_size=500 # Each sample (number of similarities,not points) will be of this size
random_population_size=500000  # Samples drawn from this population, in this example this are similarities not points
repeat_test=2 # How many times should the test be repeated for time estimation
summary_stats='mean'  # Or median
sample_stat=1 # Summary statistics for the sample, here chosen at random as it should not affect the timing
population=list(np.random.sample(random_population_size))

def time_permutation(bootstraps,sample_size,population,summary_stats,sample_stat,repeat_test):
    start=time.time()
    for i in range(repeat_test):
        permutation_p(sample_stat,population,sample_size,bootstraps,summary_stats)
    elapsed=time.time()-start
    return elapsed/repeat_test

# Calculate once
#print('Average time (sec)',time_permutation(bootstraps,sample_size,population,summary_stats,sample_stat,repeat_test))

# Plot elapsed time for different bootstraps
times=[]
decimals=[1,2,3,4,5]
for p_decimals in decimals:
    bootstraps=10**p_decimals
    print('Calculating ',bootstraps,' bootstraps for decimals',p_decimals)
    t=time_permutation(bootstraps, sample_size, population, summary_stats, sample_stat, repeat_test=1)
    times.append(t)
plt.plot(decimals,times)
plt.yscale('log')
plt.xlabel('p value decimals precision')
plt.ylabel('time (sec)')

#Ddistribution of permutation means follows normal distribution
# mean is approximately normally distributed with mean μ and standard error sqrt(population_var/sample_size)

#Estimate mean and stdev of whole population:
arr_random=np.array(simsRandom)
mu=arr_random.mean()
sd=arr_random.std()
me=median(simsRandom)

#Plot distributions of sample means/medians
#statistic='mean'
statistic='median'
index=1
plt.rcParams.update({'font.size': 8})
sizes=[3,10,50,100,500,1000]
plt.figure(figsize=(12,12))
for n in sizes:
    plt.subplot(int(round(len(sizes)/2,0)),2,index)
    index+=1
    stats=random_samples(simsRandom, n, 5000, statistic)
    x = np.linspace(min(stats), max(stats), 100)
    plt.hist(stats,weights=np.ones(len(stats)) / len(stats),density=True,bins=15)
    #For means use predefined se and mean
    if statistic == 'mean':
        se = sd / sqrt(n)
        normal=norm.pdf(x, mu, se)
        plt.plot(x, normal)
        add_title=''
    #For median fit normal distribution to the data
    elif statistic == 'median':
        centerFit,seFit=norm.fit(stats)
        normalFit=norm.pdf(x, centerFit, seFit)
        plt.plot(x, normalFit)
        #Check how se, as used for means distribution, could be scaled to obtain se for medians
        observed_se_scaling=seFit*sqrt(n)/sd
        add_title=', observed se scaling: '+str(observed_se_scaling)
    plt.title('n='+str(n)+add_title,fontsize=8)
plt.suptitle('Distribution of random sample '+statistic+'s, n=number of similarities',fontsize=10)

#Plot distribution of scaling factors for se of sample medians distribution

#Estimates se scaling for medians distribution based on n_pairs (sample size) and n permutations (n samples)
def se_scaling_estimation(n_pair:int,permutations:int):
    stats=random_samples(simsRandom, n_pair, permutations, 'median')
    centerFit, seFit = norm.fit(stats)
    observed_se_scaling = seFit * sqrt(n_pair) / sd
    return observed_se_scaling

#Get number of pairs from points and scaling factor for that number of pairs
#n_points - eg. n genes
#permutations - as in se_scaling_estimation
#Return tupple with list of n_similarities for each n_points as first element and list of scaling factors as second element
def se_scalings_estimation(n_points:list,permutations:int):
    n_pairs = []
    scalings = []
    for n_point in n_points:
        n = possible_pairs(n_point)
        n_pairs.append(n)
        scalings.append(se_scaling_estimation(n,permutations))
    return(n_pairs,scalings)

def log_fit_to_function(fit):
    b1=fit[0]
    b0=fit[1]
    def F(x):
        return b1*np.log(x)+b0
    return F

#Converts poly fit with degree up to 5 to function
def poly_fit_to_function(fit,is_log=False):
    degree=len(fit)-1
    if degree<=5:
        b0=fit[degree]
        coeffs = list(fit[:degree])
        while len(coeffs) < 5:
            coeffs.insert(0, 0)
        b5=coeffs[0]
        b4=coeffs[1]
        b3 = coeffs[2]
        b2 = coeffs[3]
        b1 = coeffs[4]
        if not is_log:
            def F(x):
                y=b5*x**5+b4*x**4+b3*x**3+b2*x**2+b1*x+b0
                return y
        else:
            def F(x):
                y=b5*np.log(x)**5+b4*np.log(x)**4+b3*np.log(x)**3+b2*np.log(x)**2+b1*np.log(x)+b0
                return y
        return F

#Find y based on y-s of (2) neighbouring x-s. If x is in data_x return data_y value
#Data_x must be sorted from smallest to largest, it can not have repeated values
def fit_neighbours(data_x:list,data_y:list,x:int):
    if x in data_x:
        index=data_x.index(x)
        return data_y[index]
    elif x<min(data_x):
        return data_y[0]
    elif x>max(data_x):
        return data_y[len(data_y)-1]
    else:
        for i in range(len(data_x)):
            if data_x[i]<x and data_x[i+1]>x:
                y=(data_y[i]+data_y[i+1])/2
                return y

#Plot distribution of scaling factors for se of sample medians distribution
#N points (eg. genes)
n_points=[3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,33,35,37,40,42,45,47,50,55,60,70,80]
#Sample sizes - n similarities
permutations=1000
n_pairs,scalings=se_scalings_estimation(n_points,permutations)
plt.plot(n_pairs,scalings)
plt.xlabel('n similarities')
plt.ylabel('se scaling factor')

#Try to estimate se scaling factors
#Compare how densiti of n_points sampling and n permutations affects the estimation of se
#This might need to be latter re adjusted when it becomes evident how the final run times of the two compare
#Data
#Many sample sizes, less permutations
n_points_many_points=[3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,33,35,37,40,42,45,47,50,55,60,70,80]
n_pairs_many_point ,scalings_many_point =se_scalings_estimation(n_points_many_points,1000)
#Less sample sizes, more permutations
n_points_many_permutations=[3,5,7,10,20,30,50,80]
n_pairs_many_permutations ,scalings_many_permutations =se_scalings_estimation(n_points_many_permutations,5000)
#Test set
#Add some points before random addition of points to also ensure inclusion of extreme points
#The points could be in fact sampled from distribution of gene set lengths (adjusting long sets to have less points,as would be used in enrichment calculations)
n_points_test=[3,4,5,6,10,20,50,80]
for i in range(20):
    n=random.randint(3,81)
    n_points_test.append(n)
n_pairs_test ,scalings_test =se_scalings_estimation(n_points_test,2500)
scalings_test=[x for _,x in sorted(zip(n_pairs_test,scalings_test))]
n_pairs_test.sort()

#Fits to data points
#Test
plt.figure(figsize=(10,10))
plt.scatter(n_pairs_test,scalings_test)
#Polinomial
fit_many_points1= poly_fit_to_function(np.polyfit(n_pairs_many_point , scalings_many_point, 3))
fits=[]
for i in n_pairs_test:
    fits.append(fit_many_points1(i))
plt.plot(n_pairs_test,fits,label='many points poly')
#Log polinomial
fit_many_points2= poly_fit_to_function(np.polyfit(np.log(n_pairs_many_point) , scalings_many_point, 2),True)
fits=[]
for i in n_pairs_test:
    fits.append(fit_many_points2(i))
plt.plot(n_pairs_test,fits,label='many points log poly')
#Based on neighbouring points
fits=[]
for i in n_pairs_test:
    fits.append(fit_neighbours(n_pairs_many_point,scalings_many_point,i))
plt.plot(n_pairs_test,fits,label='many points neighbours')

#Fit for more permutations
#Polinomial
fit_many_permutations1=poly_fit_to_function(np.polyfit(n_pairs_many_permutations , scalings_many_permutations,3))
fits=[]
for i in n_pairs_test:
    fits.append(fit_many_permutations1(i))
plt.plot(n_pairs_test,fits,label='many permutations  poly')
#Log polinomial
fit_many_permutations2= poly_fit_to_function(np.polyfit(np.log(n_pairs_many_permutations) , scalings_many_permutations, 2),True)
fits=[]
for i in n_pairs_test:
    fits.append(fit_many_permutations2(i))
plt.plot(n_pairs_test,fits,label='many permutations log poly')
#Based on neighbouring points
fits=[]
for i in n_pairs_test:
    fits.append(fit_neighbours(n_pairs_many_permutations,scalings_many_permutations,i))
plt.plot(n_pairs_test,fits,label='many permutations neighbours')

plt.legend()
#MSE was not calculated as it could be influenced by random distribution of test points -
# for MSE estimation they should be sampled for gene set lengths

#For medians and means compare normal fit results and permutation test results
#se scaling factor was estimated with neighbours based on smaller number of points with more permutations
#Median normal distribution fit: from above

#Left Plot - distribution of random means with marked gene set mean,
#Middle plot - distribution of random medians with marked gene set median,
# permutation test p value (pp) and median normal distribution based p value (pn)
#Histograms are permutation results, line fit is the normal fit (based on whole random population, not observed histogram)
#right plot - distribution of similarities for random sample and gene set
count_sets=0
set_num=0
n_sets=5
max_sims=1000
#How many sets to plot
plt.figure(figsize=(12, 12))
while count_sets< n_sets:
    # Add int to set_num to try other gene sets
    gene_set=list(gene_sets)[set_num+20]
    set_num+=1
    try:
        sims=gsscn.similarities(gene_set.genes,max_sims)
        #print(median(sims))
        size_sims=len(sims)

        #Mean based
        #Permutations
        m=calculate_summary_stats(sims,'mean')
        pp, randomMeans = calculate_permutation_p(sims, simsRandom, 'mean')
        ax1 = plt.subplot(n_sets,3, count_sets * 3 + 1)
        ax1.hist(randomMeans, density=True, bins=20)
        plt.ylabel('Density')
        plt.xlabel("Sample's means")

        # Add normal distribution to sampling means
        se = sd / sqrt(size_sims)
        x = np.linspace(min(randomMeans), max(randomMeans), 100)
        normalFit = norm.pdf(x, mu, se)
        plt.plot(x, normalFit)
        pn = 1 - norm(mu, se).cdf(m)

        # Plot p values
        ax1.vlines(m, 0, max(normalFit))
        ax1.annotate('pp=' + str(pp) + '\npn=' + str(pn), (m + m / 100, max(normalFit) / 3))

        #Median based
        #Permutations
        m = calculate_summary_stats(sims, 'median')
        pp,randomMeans=calculate_permutation_p(sims,simsRandom,'median')
        ax1 =plt.subplot(n_sets , 3, count_sets*3+2)
        ax1.hist(randomMeans,density=True ,bins=20)
        plt.ylabel('Density')
        plt.xlabel("Sample's medians")

        #Add normal distribution to sampling medians
        scaling_fit=fit_neighbours(n_pairs_many_permutations, scalings_many_permutations, size_sims)
        se=scaling_fit*sd/sqrt(size_sims)
        x = np.linspace(min(randomMeans), max(randomMeans), 100)
        normalFit=norm.pdf(x, me, se)
        plt.plot(x, normalFit)
        pn=1-norm(me, se).cdf(m)

        #Plot p values
        ax1.vlines(m, 0, max(normalFit))
        ax1.annotate('pp='+str(pp)+'\npn='+str(pn), (m+m/100, max(normalFit)/3))

        #Distributions of similarities
        ax2 = plt.subplot(n_sets, 3, count_sets*3+3)
        point_hist(sims,ax2,legend_description=gene_set.name +" N: "+str(len(gene_set.genes)),proportion=True)
        point_hist(simsRandom, ax2, legend_description='random',proportion=True)
        ax2.legend(prop={'size': 6})
        plt.ylabel('Proportion')
        plt.xlabel('Similarities')
        print(gene_set.name , str(len(gene_set.genes)))
        count_sets+=1
    except EnrichmentError:
        pass

#Compare means and medians p values at lower p values (eg. mean based p value <0.1)
#data for median and mean based fitting is from above

#Compare with similarities graphs
count_sets=0
set_num=0
n_sets=20
max_sims=1000
#How many sets to plot
plt.figure(figsize=(12, 12))
while count_sets< n_sets:
    # Add int to set_num to try other gene sets
    gene_set=list(gene_sets)[set_num+20]
    set_num+=1
    try:
        sims=gsscn.similarities(gene_set.genes,max_sims)
        size_sims=len(sims)

        # mean based p value
        m = calculate_summary_stats(sims, 'mean')
        se = sd / sqrt(size_sims)
        pm = 1 - norm(mu, se).cdf(m)
        if pm<=0.1:

            # median based p value
            m = calculate_summary_stats(sims, 'median')
            scaling_fit = fit_neighbours(n_pairs_many_permutations, scalings_many_permutations, size_sims)
            se = scaling_fit * sd / sqrt(size_sims)
            pme = 1 - norm(me, se).cdf(m)

            ax = plt.subplot(5, 4, count_sets+1)
            point_hist(sims, ax, legend_description=gene_set.name + " N: " + str(len(gene_set.genes)), proportion=True)
            point_hist(simsRandom, ax, legend_description='random', proportion=True)
            plt.ylabel('Proportion')
            plt.xlabel('Similarities')
            ax.annotate('pm=' + str(pm) + '\npme=' + str(pme)+'\nNgen='+str(len(gene_set.genes))+' Nsims='+str(size_sims), (0.05, 0.05))

            print(gene_set.name, str(len(gene_set.genes)))
            count_sets += 1

    except EnrichmentError:
        pass


#Compare difference between p mean and median based on sample size:
count_sets=0
set_num=0
n_sets=100
max_sims=1000
#How many sets to plot
n_sims=[]
p_diffs=[]
p_avgs=[]
pms=[]
pmes=[]
while count_sets< n_sets:
    # Add int to set_num to try other gene sets
    gene_set=list(gene_sets)[set_num]
    set_num+=1
    try:
        sims=gsscn.similarities(gene_set.genes,max_sims)
        size_sims=len(sims)

        # mean based p value
        m = calculate_summary_stats(sims, 'mean')
        se = sd / sqrt(size_sims)
        pm = 1 - norm(mu, se).cdf(m)
        if pm<=0.1:
            count_sets += 1
            # median based p value
            m = calculate_summary_stats(sims, 'median')
            scaling_fit = fit_neighbours(n_pairs_many_permutations, scalings_many_permutations, size_sims)
            se = scaling_fit * sd / sqrt(size_sims)
            pme = 1 - norm(me, se).cdf(m)

            pms.append(pm)
            pmes.append(pme)
            p_diff=pm-pme
            p_diffs.append(p_diff)
            p_avg=(pm+pme)/2
            p_avgs.append(p_avg)
            n_sims.append(size_sims)
    except EnrichmentError:
        pass
df = pd.DataFrame(list(zip(n_sims, p_diffs,p_avgs,pms,pmes)),
               columns =['N_similarities', 'p_difference','p_aerage','p_mean','p_median'])
#Use in notebook
alt.Chart(df).mark_circle().encode(x='N_similarities',y='p_difference',
                                   color=alt.Color('p_mean',scale=alt.Scale(range=['darkviolet','yellow']))
                                  ).configure_circle(size=20)
alt.Chart(df).mark_circle().encode(x='N_similarities',y='p_difference',
                                   color=alt.Color('p_median',
                                                   scale=alt.Scale(range=['darkviolet','yellow','yellowgreen','green']))
                                  ).configure_circle(size=20)


#Plot distribution of expression values:
values=genesEID.values.tolist()
values=[x for sub in values for x in sub]
plt.hist(values,bins=10000)

#Compare different parameters
#Compare difference between spearman and normalised cosine based results:
sc_spearman=SimilarityCalculator(similarity_type='correlation_spearman')
sc_cosine=SimilarityCalculator(similarity_type='cosine',normalisation_type='mean0std1')
ec_cosine=EnrichmentCalculator.quick_init(ge,sc_spearman)
ec_spearman=EnrichmentCalculator.quick_init(ge,sc_cosine)

max_pairs=10000
gene_sets_sub=list(gene_sets)[:50]
result_cosine=ec_cosine.calculate_enrichment(gene_sets_sub,max_pairs=max_pairs)
print('spearman')
result_spearman=ec_spearman.calculate_enrichment(gene_sets_sub,max_pairs=max_pairs)


result_1=result_spearman
result_2=result_cosine

def result_df(result_1,result_2,max_pairs):
    padj_diff=[]
    padjs_1=[]
    padjs_2=[]
    n_genes=[]
    means_1=[]
    means_2=[]
    medians_1=[]
    medians_2=[]
    n_sims=[]
    matching=True
    for index in range(len(result_1)):
        data_1=result_1[index]
        data_2=result_2[index]
        if data_1.gene_set is not data_2.gene_set:
            matching=False
            break

    for index in range(len(result_1)):
        data_1=result_1[index]
        data_2=result_2[index]
        if data_1.gene_set is data_2.gene_set:
            padj_1=data_1.padj
            padj_2=data_2.padj
            padj_diff.append(padj_1-padj_2)
            padjs_1.append(padj_1)
            padjs_2.append(padj_2)
            n=len(data_1.gene_set.genes)
            n_genes.append(n)
            means_1.append(data_1.mean)
            means_2.append(data_2.mean)
            medians_1.append(data_1.median)
            medians_2.append(data_2.median)
            n_sim=possible_pairs(n)
            if n_sim>max_pairs:
                n_sim=max_pairs
            n_sims.append(n_sim)
    abs_padj_diff = [abs(x) for x in padj_diff]

    return pd.DataFrame(list(zip(n_sims, n_genes, padj_diff, abs_padj_diff, padjs_1, padjs_2, means_1, means_2)),
                      columns=['N_similarities', 'N_genes', 'padj_difference', 'abs_padj_difference', 'padj_1',
                               'padj_2','mean_1', 'mean_2'])


#Compare distribution of similarities of gene sets with divergent p values between cosine and spearman similarity
n = 0
plt.figure(figsize=(12,12))
for index in range(len(result_cosine)):
    data_1 = result_cosine[index]
    data_2 = result_spearman[index]
    if data_1.gene_set is data_2.gene_set:
        padj_1 = data_1.padj
        padj_2 = data_2.padj
        if (padj_1 <= 0.05 and padj_2 > 0.05) or (padj_2 <= 0.05 and padj_1 > 0.05):
            n += 1
            sims_cosine = ec_cosine.calculator.similarities(data_1.gene_set.genes, 10000)
            sims_spearman = ec_spearman.calculator.similarities(data_2.gene_set.genes, 10000)
            sims_random_cosine = ec_cosine.storage._similarities
            sims_random_spearman = ec_spearman.storage._similarities
            point_hist(sims_cosine, plt.subplot(3, 3, n), 'cosine ' + str(padj_1),proportion=True)
            point_hist(sims_spearman, plt.subplot(3, 3, n), 'spearman ' + str(padj_2),proportion=True)
            point_hist(sims_random_cosine, plt.subplot(3, 3, n), 'cosine random',proportion=True)
            point_hist(sims_random_spearman, plt.subplot(3, 3, n), 'spearman random',proportion=True)
            plt.legend()

