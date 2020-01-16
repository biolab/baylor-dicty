library(pvclust)
library(dplyr)
library(ComplexHeatmap)
library(RColorBrewer)
library(tightClust)
source('/home/karin/Documents/timeTrajectories/deTime/TightCluster_Large.R')

data_path = '/home/karin/Documents/timeTrajectories/data/regulons/clusters/'
path_selGenes = '/home/karin/Documents/timeTrajectories/data/regulons/selected_genes/'

  
#*******Heatmap
#Louvain
cl_louvain<-read.table(paste(data_path,'kNN2_threshold0.95_m0s1log_clusters_LouvainRes1.8.tsv',sep=''),sep='\t',row.names=1,header=TRUE)
cl_louvain<-cl_louvain[order(cl_louvain$all),]
cl_louvain<-as.matrix(cl_louvain)

#Heatmap
n_clusters=max(cl_louvain)
palette=c(brewer.pal(n=9,name='Set1'),brewer.pal(n=8,name='Pastel1'))
colours=structure(palette, names = as.character(seq(0,16)))
Heatmap(cl_louvain,cluster_rows =  FALSE,cluster_columns = FALSE,
        show_row_names = FALSE,col=colours,name='cluster')

#***********pvclust
# Load data
data_Orange=read.table('/home/karin/Documents/timeTrajectories/Orange_workflows/regulons/kNN2_threshold0.95_RPKUM+pattern_m0s1log_Orange.tab',header=TRUE,row.names='Gene')
data_Orange <- data_Orange[!rownames(data_Orange) %in% c('string','meta'),!colnames(data_Orange) %in% c('Mass_centre','Peak','N_atleast')] 
# Not needed 
#data_Orange1<-mutate_all(data_Orange, function(x) as.numeric(as.character(x)))
#rownames(data_Orange1)<-rownames(data_Orange)
#data_Orange<-t(data_Orange1)
#data_Orange1<-NULL
genes<-rownames(data_Orange)
n_all=length(genes)

# Distance metric, taken from manual
cosine <- function(x) {
   x <- as.matrix(x)
   y <- t(x) %*% x
   res <- 1 - y / (sqrt(diag(y)) %*% t(sqrt(diag(y))))
   res <- as.dist(res)
   attr(res, "method") <- "cosine"
   return(res)
}

#*** Clustering try
# Data must be in columns, features in rows
clust<-pvclust(data,method.hclust='ward.D2',method.dist=cosine,parallel=TRUE,iseed=0)

#Find clusters
pv='au'
alpha=0.95

# Option 1: Assume each gene will be in one cluster (has at least one close neighbour)
alphas=sort(unique(clust$edges[,pv]),decreasing = TRUE)
alphas=alphas[alphas<alpha]
alphas<-append(alpha,alphas)
n_included=0
index=0
clusters=NA
proportion_retained=1
while(n_included<n_all*proportion_retained ){
  index=index+1
  alpha_used=alphas[index]
  clusters<-pvpick(clust,alpha=alpha_used,pv=pv,max.only=TRUE)
  n_included=0
  for (cluster in clusters$clusters){
    n_included=n_included+length(cluster)
  }
}
#OR 
#Option 2: Asign clusters only above a pv threshold
pv='au'
alpha=0.95
clusters<-pvpick(clust,alpha=alpha,pv=pv,max.only=TRUE)

# Analyse cluster size distribution   
cluster_sizes=c()
for (cluster in clusters$clusters){
  cluster_sizes<-append(cluster_sizes,length(cluster))
}
hist(cluster_sizes,main=paste('N clustered genes',sum(cluster_sizes)),breaks=max,log='x')

#Make cluster DF
genes_cl<-data.frame(row.names=genes)
genes_cl[,paste('all','cluster',sep='_')]=rep(NA,length(genes))
clust_counter=0
for (cluster in clusters$clusters){
  for (gene in cluster){
    genes_cl[gene,paste('all','cluster',sep='_')]=clust_counter
  }
  clust_counter=clust_counter+1
}


#********* Make clustering for each strain and for all strains
# Load RPKUM  data and preprocess
dataRPKUM=read.table('/home/karin/Documents/timeTrajectories/data/RPKUM/combined/mergedGenes_RPKUM.tsv',header=TRUE)
dataRPKUM<-dataRPKUM[genes,]
conditions=read.table('/home/karin/Documents/timeTrajectories/data/RPKUM/combined/conditions_mergedGenes.tsv',header=TRUE)
conditions$Measurment=make.names(conditions$Measurment)
if (!all(colnames(dataRPKUM)==conditions['Measurment'])) print('Err in sample name matching ')
dataRPKUM=log2(dataRPKUM+1)

not_all_na <- function(x) any(x!=0)

dataRPKUM_scaled<-as.data.frame(t(dataRPKUM)) %>% select_if(not_all_na)
dataRPKUM_scaled=scale(dataRPKUM_scaled)


#Make clustering
for (strain in c(levels(unique(conditions$Strain)),'all')){
  if (strain!='all') data=dataRPKUM[,conditions[conditions$Strain==strain,'Measurment']]
  else  data=dataRPKUM
  print(paste(strain,'samples:',dim(data)[2]))
  data=as.data.frame(t(data)) %>% select_if(not_all_na)
  print(paste('Not null genes',dim(data)[2]))
  data=scale(data)
  clust<-pvclust(data,method.hclust='ward.D2',method.dist=cosine,parallel=TRUE,iseed=0)
  saveRDS(clust,paste(data_path,'pvclust_cosine_kNN2_threshold0.95_m0s1log/',strain,'_pvclust_cosine_kNN2_threshold0.95_m0s1log.rds',sep=''))
}

#Get clusters into df and save their sizes into another df
files<-list.files(paste(data_path,'pvclust_cosine_kNN2_threshold0.95_m0s1log/',sep=''),pattern='*.rds')
samples<-c()
for (file in files){
  sample<-strsplit(file, '_')[[1]][1]
  samples<-append(samples,sample)
}
pv='au'
alpha=0.95
min_size=7
genes_cl<-NULL
sizes_cl<-NULL
genes_cl<-data.frame(row.names=genes)
sizes_cl<-data.frame(row.names=samples)
for( sample in samples){
  clust<-readRDS(paste(data_path,'pvclust_cosine_kNN2_threshold0.95_m0s1log/',sample,'_pvclust_cosine_kNN2_threshold0.95_m0s1log.rds',sep=''))
  clusters<-pvpick(clust,alpha=alpha,pv=pv,max.only=TRUE)
  sample_col<-paste(sample,'cluster',sep='_')
  genes_cl[,sample_col]=rep(NA,length(genes))
  clust_counter=0
  for (cluster in clusters$clusters){
    if (length(cluster)>=min_size){
      sizes_cl[sample,as.character(clust_counter)]<-length(cluster)
      for (gene in cluster){
        genes_cl[gene,sample_col]=clust_counter
      }
      clust_counter=clust_counter+1
    }
  }
  
}

write.table(genes_cl,paste(data_path,'pvclust_cosine_kNN2_threshold0.95_m0s1log/','clusters_min7.tsv',sep=''),sep='\t',col.names=NA)
write.table(sizes_cl,paste(data_path,'pvclust_cosine_kNN2_threshold0.95_m0s1log/','cluster_sizes_min7.tsv',sep=''),sep='\t',col.names=NA)

#******************** Tight clustering

# Load RPKUM  data and preprocess
dataRPKUM=read.table('/home/karin/Documents/timeTrajectories/data/RPKUM/combined_mt/combined/mergedGenes_RPKUM.tsv',header=TRUE)
conditions=read.table('/home/karin/Documents/timeTrajectories/data/RPKUM/combined_mt/combined/conditions_mergedGenes.tsv',header=TRUE)
conditions$Measurment=make.names(conditions$Measurment)
if (!all(colnames(dataRPKUM)==conditions['Measurment'])) print('Err in sample name matching ')
dataRPKUM=log2(dataRPKUM+1)

not_all_na <- function(x) any(x!=0)
#Selected genes
sel_genes_file='selectedGenes1000_scalemean0std1_logTrue_kN6_splitStrain'
selected_genes<-read.table(paste(path_selGenes,sel_genes_file,'.tsv',sep=''),sep='\t',header=TRUE)

#Make clustering
cluster_df<-data.frame(row.names=as.vector(unique(unlist(selected_genes))))
n_clust_main=25
for (strain in c(levels(unique(conditions$Strain)),'all')){
  #Process data after log (above) - select strain, genes and scale
  if (strain!='all') data=dataRPKUM[,conditions[conditions$Strain==strain,'Measurment']]
  else  data=dataRPKUM
  data<-data[as.vector(selected_genes[[strain]]),]
  print(paste(strain,'samples:',dim(data)[2]))
  data=as.data.frame(t(data)) %>% select_if(not_all_na)
  print(paste('Not null genes',dim(data)[2]))
  data=t(scale(data))
  # Try clustering with K groups, reduce if error
  # tries - try same K more than once (N+1 to try N times, N for remainder '%%' below)
  #Set this to one more if without tries because subtracts the first time
  n_clust=n_clust_main+1
  #tries=0
  failed=TRUE
  while (failed){
    #Tries - to try same number of clusters a few times before reducing it
    #tries=tries+1
    #if (tries %%6 ==0){
    #tries=0
    n_clust=n_clust-1
    #}
    tryCatch({
      clust<-tight.clust (x=data,target=n_clust,k.min=n_clust+5)
      failed=FALSE
      print(paste('**** OK ',n_clust))
    }, error=function(cond) {
      print(paste('**** NOT ',n_clust))
  }
  )
  }
  #Add to df
  gene_names<-rownames(data)
  clusters<-clust$cluster
  for (idx in 1:length(gene_names)){
    cluster_df[gene_names[idx],strain]<-clusters[idx]
  }
}
write.table(cluster_df,paste(data_path,'tight_clust/',sel_genes_file,'tightclust',n_clust_main,'.tsv',sep=''),sep='\t',col.names=NA)

#Count N unclustered genes per strain
unclust<-function(x){sum(x==-1 & !is.na(x))}
unclustdf <- function(data) sapply(data, unclust)
unclustdf(cluster_df)
#N Clusters 
colMax <- function(data) sapply(data, max, na.rm = TRUE)
colMax(cluster_df)




#Order strain df for ploting -TODO
conditions_strain=conditions[conditions$Strain==strain,]
conditions_strain=conditions_strain[order(conditions_strain$Replicate,conditions_strain$Time),]
data=dataRPKUM[,conditions[conditions$Strain==strain,'Measurment']]
data=data[,order(match(colnames(data),conditions_strain$Measurment))]
data=t(as.data.frame(t(data)) %>% select_if(not_all_na))





