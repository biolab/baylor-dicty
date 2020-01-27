
#Variables
dataPath='/home/khrovatin/data/countsRaw/combined/'
dataPathNormalised='/home/khrovatin/data/RPKUM/combined/'
dataPathCode='/home/khrovatin/git/baylor-dicty/R/'
dataPathImpulse='/home/khrovatin/data/deTime/de_time_impulse/'

library("DESeq2")
library("ImpulseDE2")
source(paste(dataPathCode,'libraryDE.R',sep=''))
library("BiocParallel")
register(MulticoreParam(20))

#Load data counts
genes<-read.table(paste(dataPath,"mergedGenes_counts.tsv",sep=''), header=TRUE,row.names=1, sep="\t")
conditions<-read.table(paste(dataPathNormalised,"conditions_mergedGenes.tsv",sep=''), header=TRUE,row.names='Measurment', sep="\t")
#R imported colnames of genes with changes but gene IDs remained ok
rownames(conditions)<-make.names(rownames(conditions))

#List mutants
strains=as.vector(unlist(unique(conditions$Strain),use.names=FALSE))
mutant_list<-strains[strains!='AX4']

#*************** Compare DE over whole profile with Impulse

for( mutant in mutant_list){
  print(mutant)
  runImpulseCustomDispersion(conditions=conditions,genes=genes,times=NULL,case=mutant,control='AX4',main_lvl='Strain',
                           nested_dispersion=c('Time'),confounder_impulse=NULL,fdr=0.05,path=dataPathImpulse)
}
