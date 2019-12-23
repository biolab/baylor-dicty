#Variables
lab=TRUE
if (lab){
  dataPathSaved='/home/karin/Documents/timeTrajectories/data/deTime/before_mutated_expression/'
  dataPathCode='/home/karin/Documents/timeTrajectories/deTime/'
  dataPath='/home/karin/Documents/timeTrajectories/data/countsRaw/combined/'
  dataPathNormalised='/home/karin/Documents/timeTrajectories/data/RPKUM/combined/'
}else {
  dataPathSaved='/home/karin/Documents/DDiscoideum/data/timeDE/transitions/'
  dataPath='/home/karin/Documents/DDiscoideum/data/countsRaw/'
  dataPathNormalised='/home/karin/Documents/DDiscoideum/'
}

library("DESeq2")
library('dplyr')
library(ggplot2)
library(RColorBrewer)
library("ImpulseDE2")
source(paste(dataPathCode,'libraryDE.R',sep=''))
library(mclust)
library(lmms)
library("BiocParallel")
register(MulticoreParam(4))
library(ComplexHeatmap)
library(circlize)
library(viridis)

#Load data counts
genes<-read.table(paste(dataPath,"mergedGenes_counts.tsv",sep=''), header=TRUE,row.names=1, sep="\t")
conditions<-read.table(paste(dataPathNormalised,"conditions_mergedGenes.tsv",sep=''), header=TRUE,row.names='Measurment', sep="\t")
#R imported colnames of genes with changes but gene IDs remained ok
rownames(conditions)<-make.names(rownames(conditions))
#Load data normalised
genesNorm<-read.table(paste(dataPathNormalised,"mergedGenes_RPKUM.tsv",sep=''), header=TRUE,row.names=1, sep="\t")

# Which strains have almost 0 expression in AX4 and to which hour
unchanged<-list('ecmA'=8,'comH'=5,'tagB'=4)

# DE for these strains
for (mutant in names(unchanged)){
  time_max<-unchanged[mutant]
  times=unique(conditions[conditions$Strain %in% c(mutant,'AX4'),'Time'])
  times<-times[times<=time_max]
  runImpulseCustomDispersion(conditions=conditions,genes=genes,times=times,case=mutant,control='AX4',main_lvl='Strain',
                           nested_dispersion=c('Time'),confounder_impulse=NULL,fdr=0.05,path=paste(dataPathSaved,'unchanged_',sep=''))

}

unchanged_points<-list('cudA'=4,'ecmA'=8,'comH'=5,'tagB'=4,'tgrC1'=3,'tgrB1'=3,'tgrB1C1'=3)
times_AX4=unique(conditions[conditions$Strain =='AX4','Time'])
# DE for these strains
for (mutant in names(unchanged_points)){
  time_max<-unchanged_points[mutant]
  times_mutant<-unique(conditions[conditions$Strain ==mutant,'Time'])
  times<-intersect(times_AX4,times_mutant)
  times<-times[times<=time_max]
  for(time in times){
    print(mutant)
    print(time)
    runDeSeq2(conditions=conditions,genes=genes,time=time,case=mutant,control='AX4',design=~Strain,main_lvl='Strain',
              padj=0.05,logFC=1,path=paste(dataPathSaved,'unchangedDESeq_',sep=''))
  }
  
}







