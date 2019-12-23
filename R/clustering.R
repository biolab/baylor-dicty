library(pvclust)
library(dplyr)
library(ComplexHeatmap)
library(RColorBrewer)

  data_path = '/home/karin/Documents/timeTrajectories/data/regulons/clusters/'

  
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
data=read.table('/home/karin/Documents/timeTrajectories/Orange_workflows/regulons/kNN2_threshold0.95_RPKUM+pattern_m0s1log_Orange.tab',header=TRUE,row.names='Gene')
data <- data[!rownames(data) %in% c('string','meta'),!colnames(data) %in% c('Mass_centre','Peak','N_atleast')] 
data1<-mutate_all(data, function(x) as.numeric(as.character(x)))
rownames(data1)<-rownames(data)
data<-data1
  
# Clustering
# Distance metric, taken from manual
cosine <- function(x) {
   x <- as.matrix(x)
   y <- t(x) %*% x
   res <- 1 - y / (sqrt(diag(y)) %*% t(sqrt(diag(y))))
   res <- as.dist(res)
   attr(res, "method") <- "cosine"
   return(res)
}

# Data must be in columns, features in rows
n_all=dim(data)[1]
clust<-pvclust(t(data),method.hclust='ward.D2',method.dist=cosine,parallel=TRUE,iseed=0)

#Find clusters, assume each gene will be in one cluster (has at least one close neighbour)
pv='au'
alpha=0.95
alphas=sort(unique(clust$edges[,pv]),decreasing = TRUE)
alphas=alphas[alphas<alpha]
alphas<-append(alpha,alphas)
n_included=0
index=0
clusters=NA
while(n_included<n_all ){
  index=index+1
  alpha_used=alphas[index]
  clusters<-pvpick(clust,alpha=alpha_used,pv=pv,max.only=TRUE)
  n_included=0
  for (cluster in clusters$clusters){
    n_included=n_included+length(cluster)
  }
}

