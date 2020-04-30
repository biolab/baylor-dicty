
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
# Do it for mutant vs WT or between WT (TRUE=mutant vs WT)
run_mutant_or_wt=FALSE
if (run_mutant_or_wt){
    # DE between mutant and WT
    for( mutant in mutant_list){
    print(mutant)
    runImpulseCustomDispersion(conditions=conditions,genes=genes,times=NULL,case=mutant,control='AX4',main_lvl='Strain',
               nested_dispersion=c('Time'),confounder_impulse=NULL,fdr=0.05,path=dataPathImpulse)
    }
}else{
    # Compare DE between AX4 replicates to find where to use FDR cutoff: FD, PE, SE 1vs1
    conditions_AX4=data.frame(conditions[conditions$Strain=='AX4',])
    genes_AX4=genes[,rownames(conditions_AX4)]
    conditions_AX4['Group']=unlist(lapply(conditions_AX4$Replicate,function(x) substring(x,1,6)),use.names = FALSE)

    AX4_reps<-c('AX4_PE','AX4_SE','AX4_FD')
    for (rep1 in 1:(length(AX4_reps)-1)){
      for (rep2 in (rep1+1):length(AX4_reps)){
        print(paste(AX4_reps[rep1],AX4_reps[rep2]))
            runImpulseCustomDispersion(conditions=conditions_AX4,genes=genes_AX4,times=NULL,case=AX4_reps[rep1], 
                                       control=AX4_reps[rep2], main_lvl='Group',nested_dispersion=c('Time'), 
                                       confounder_impulse=NULL, fdr=0.05,path=dataPathImpulse)
      }
    }
}