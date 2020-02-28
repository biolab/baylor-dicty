
#Variables
lab=TRUE
if (lab){
  dataPathSaved='/home/karin/Documents/timeTrajectories/data/deTime/diffTimeTry/'
  dataPathCode='/home/karin/Documents/timeTrajectories/deTime/'
  dataPath='/home/karin/Documents/timeTrajectories/data/countsRaw/combined/'
  dataPathNormalised='/home/karin/Documents/timeTrajectories/data/RPKUM/combined/'
  pathDETime='/home/karin/Documents/timeTrajectories/deTime/de_time_impulse_strain/'
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

#For testing - add developmental column to conditions, use AX4,comH,tagB replicates and subset genes to 500:'
repsTest<-c('AX4_FD_r1','AX4_FD_r2','AX4_PE_r3','comH_r1','comH_r2','tagB_r1','tagB_r2','tgrC1_r1','tgrC1_r2')
genesTest<-genes[1:200,(conditions$Replicate %in% repsTest)]
conditionsTest<-conditions[(conditions$Replicate %in% repsTest),]
conditionsTest <- conditionsTest[order(conditionsTest$Replicate, conditionsTest$Time),]
genesTest<-t(t(genesTest)[rownames(conditionsTest),])
phenos<-c(rep(c(1,1,2,2,2,3,3,4,4,5,5,6,6,6,6,6,6,6,6),2),c(1,2,3,4,5,6,6),rep(c(1,2,2,3,4,4,4,4,4,4),4),rep(c(1,2,2,2,2,2,2,2,2,2),2))
conditionsTest$Stage<-phenos

# ******************************************
#********* DE when mutants start to differ in phenotype compared to WT
# WT data
timesWT<-unique(conditionsTest[conditionsTest$Strain=='AX4','Time'])
conditionsWT<-conditionsTest[conditionsTest$Strain=='AX4',]

# Prepare data of mutants - when difference occurs
strains=as.vector(unlist(unique(conditionsTest$Strain),use.names=FALSE))
mutants<-c()
last_same<-c()
diff_points<-c()
for (mutant in strains[strains!='AX4']){
  times_mutant<-sort(unique(conditionsTest[conditionsTest$Strain==mutant,'Time']))
  times_both<-sort(intersect(timesWT,times_mutant))
  max_stage<-max(conditionsTest[conditionsTest$Strain==mutant,'Stage'])
  max_stage_WT_times<-sort(unique(conditionsWT[conditionsWT$Stage==max_stage,'Time']),decreasing=TRUE)
  last_time=0
  for (time_WT in max_stage_WT_times){
    if (time_WT %in% times_mutant){
      last_time=time_WT
      break
    }
  }
  diff_time=0
  for (t in times_both){
    if (t>last_time){
      diff_time=t
      break
    }
  }
  mutants<-append(mutants,mutant)
  last_same<-append(last_same,last_time)
  diff_points<-append(diff_points,diff_time)
}

diff_df<-DataFrame(mutant=mutants,last_same=last_same,diff_point=diff_points)

# Test DEs and save
for (rowN in sequence(nrow(diff_df))){
  mutant<-diff_df[rowN,'mutant']
  last_same<-diff_df[rowN,'last_same']
  diff_point<-diff_df[rowN,'diff_point']
  times_mutant<-unique(conditionsTest[conditionsTest$Strain==mutant,'Time'])
  times_both<-union(timesWT,times_mutant)
  times_before<-times_both[times_both<=diff_point]
  times_after<-times_both[times_both>=diff_point]
  # Can Stage / Replicate be converted to nested factors?????
  runImpulseCustomDispersion(conditions=conditionsTest,genes=genesTest,times=times_after,case=mutant,control='AX4',main_lvl='Strain',
                            nested_dispersion=c('Time'),confounder_impulse=NULL,fdr=0.05,path=paste(dataPathSaved,'after_',sep=''))
  runImpulseCustomDispersion(conditions=conditionsTest,genes=genesTest,times=times_before,case=mutant,control='AX4',main_lvl='Strain',
                             nested_dispersion=c('Time'),confounder_impulse=NULL,fdr=0.05,path=paste(dataPathSaved,'before_',sep=''))
  runDeSeq2(conditions=conditionsTest,genes=genesTest,time=diff_point,case=mutant,control='AX4',design=~Strain,main_lvl='Strain',
            padj=0.05,logFC=1,path=paste(dataPathSaved,'firstDiff_',sep=''))
  runDeSeq2(conditions=conditionsTest,genes=genesTest,time=last_same,case=mutant,control='AX4',design=~Strain,main_lvl='Strain',
            padj=0.05,logFC=1,path=paste(dataPathSaved,'lastSame_',sep=''))
}

#*************** Plot the result

#Import summary made in python and plot it
summary=read.csv(paste(dataPathSaved,'summary.tab',sep=''),sep='\t',stringsAsFactors = FALSE)
summary_data<-summary[,3:ncol(summary)]
#Order genes based on similarity across all features (prioritizes the ones with 2 measurments??) - hclust order 
#Scale each row 0 to 1, excluding metadata
scaled<-apply(summary_data, 1, function(x)(x-min(x))/(max(x)-min(x)))
ordered<-hclust(dist(scaled,method='cosine'),method='ward.D2')$order
summary_data<-t(summary_data[,ordered])
Heatmap(summary_data,cluster_rows = FALSE,cluster_columns = FALSE)

# Split/prepare data for spliting based on summary type 
fc_padj=unlist(summary['Summary'],use.names=FALSE)
stage=unlist(summary['Stage'],use.names=FALSE)
padj_idx=grep('padj',fc_padj)
fc_idx=grep('FC',fc_padj)
padj<-summary_data[,padj_idx]
fc<-summary_data[,fc_idx]

# Create colours for legends
stage_cols=c('0'='blueviolet','1'='slateblue4','2'='blue','3'='royalblue','4'='cyan','5'='olivedrab3','6'='chartreuse3',
                                              '8'='darkolivegreen2','10'='yellow','12'='orange', '14'='brown','16'='red','18'='black','20'='pink','24'='gray')
summary_cols=c('before_padj'='darkseagreen2','after_padj'='burlywood2','lastSame_padj'='olivedrab3',
                        'lastSame_lFC'='darkolivegreen4','firstDiff_padj'='orange', 'firstDiff_lFC'='darkorange3')
padj_col= colorRamp2( seq(min(padj), max(padj),length=12), viridis(12))
fc_col=colorRamp2( c(min(fc), 0, max(fc)),c("blue", "#EEEEEE", "red"))


summarySubMap<-function(summary_type,colour){
  #Depends on outerscope: fc_padj, stage, summary_data, stage_cols, summary_cols
  #summary_type - finds in summary$Summary - uses data belonging to it
  #colour - heatmap colour function
  idx=grep(summary_type,fc_padj)
  anno = HeatmapAnnotation(Summary = fc_padj[idx],Stage=as.character(stage[idx]),
                           col = list(Stage=stage_cols,Summary=summary_cols),show_legend = FALSE)
  heatmap=Heatmap(summary_data[,idx],cluster_rows =  FALSE,cluster_columns = FALSE,
                  show_row_names = FALSE,show_heatmap_legend = FALSE,
                  show_column_names = FALSE,top_annotation = anno,
                  col=colour)
  return(heatmap)
}

#Make heatmaps
ha_before_padj<-summarySubMap(summary_type='before_padj',colour=padj_col)
ha_after_padj<-summarySubMap(summary_type='after_padj',colour=padj_col)
ha_lastSame_padj<-summarySubMap(summary_type='lastSame_padj',colour=padj_col)
ha_firstDiff_padj<-summarySubMap(summary_type='firstDiff_padj',colour=padj_col)
ha_lastSame_FC<-summarySubMap(summary_type='lastSame_lFC',colour=fc_col)
ha_firstDiff_FC<-summarySubMap(summary_type='firstDiff_lFC',colour=fc_col)

heatmap_list=ha_before_padj+ha_lastSame_padj+ha_lastSame_FC+ha_after_padj+ha_firstDiff_padj+ha_firstDiff_FC

#Make legends
summary_labels<-c('before_padj','lastSame_padj','lastSame_lFC','after_padj','firstDiff_padj','firstDiff_lFC')
legends<-c(Legend(labels = summary_labels, 
                         title = "Summary", 
                         legend_gp = gpar(fill = as.character(summary_cols[summary_labels],use.names = FALSE) )),
                  Legend(labels = as.character(unique(stage)), title = "Stage", 
                         legend_gp = gpar(fill = as.character(stage_cols[as.character(stage)],use.names = FALSE) )),
                  Legend(col_fun = padj_col, title = "-log10(padj)"),
                  Legend(col_fun = fc_col, title = "log2FC")
                  )

#Plot
draw(heatmap_list, ht_gap = unit(0.5, "mm"),  annotation_legend_list = legends)

#********************************
#****** Stage specific genes
# Compare all unique pairs of stages for DE, use only replicates that have samples in both stages
# Test adjusted for batch effect or replicates
stages<-unique(conditionsTest$Stage)
for (i in (1:(length(stages)-1))){
  for (j in ((i+1):length(stages))){
    print(paste('Stages:',i,j))
    stage1=stages[i]
    stage2=stages[j]
    # Find replicates with samples in both stages and their data
    replicates_stage1<-conditionsTest[conditionsTest$Stage ==stage1,'Replicate']
    replicates_stage2<-conditionsTest[conditionsTest$Stage ==stage2,'Replicate']
    replicates_both<-intersect(replicates_stage1,replicates_stage2)
    print(replicates_both)
    conditions_sub<-conditionsTest[conditionsTest$Replicate %in% replicates_both & 
                                     conditionsTest$Stage %in% c(stage1,stage2),]
    genes_sub<-genesTest[,rownames(conditions_sub)]
    # DE analysis, saves to file
    res<-runDeSeq2(conditions_sub,genes_sub,case=stage2,control=stage1,design=~Replicate+Stage,main_lvl='Stage',padj=0.05,logFC=1,path=dataPathSaved)
  }
}
# **********************************************
# **** DE genes in time in each strain
padj=1
for (strain in unique(conditions$Strain)){
  genesSub<-as.matrix(genes[,conditions$Strain==strain])
  conditionsSub<-conditions[conditions$Strain==strain,]
  anno<-data.frame(Sample=rownames(conditionsSub), Condition=rep("case",ncol(genesSub)), Time=conditionsSub$Time, Batch=conditionsSub$Replicate)
  de <- runImpulseDE2(matCountData=genesSub,dfAnnotation=anno,boolCaseCtrl= FALSE,
                      vecConfounders= c("Batch"),scaNProc=4, boolIdentifyTransients = TRUE)
  de_filter<-de$dfImpulseDE2Results[de$dfImpulseDE2Results$padj<=padj & ! is.na(de$dfImpulseDE2Results$padj),]
  write.table(de_filter,file=paste(pathDETime,'DE_',strain, '_padj',padj,'.tsv',sep=''), sep='\t',row.names = FALSE)
}


