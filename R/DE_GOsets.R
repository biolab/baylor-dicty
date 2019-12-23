#Variables
lab=TRUE
if (lab){
  dataPathSaved='/home/karin/Documents/timeTrajectories/data/deTime/impulse_strains/'
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

mutant='tgrB1'
times=unique(conditions[conditions$Strain %in% c(mutant,'AX4'),'Time'])
runImpulseCustomDispersion(conditions=conditions,genes=genes,times=times,case=mutant,control='AX4',main_lvl='Strain',
                           nested_dispersion=c('Time'),confounder_impulse=NULL,fdr=0.05,path=paste(dataPathSaved,'try_',sep=''))