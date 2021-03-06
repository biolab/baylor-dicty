
#Variables
lab=TRUE
if (lab){
  dataPathSaved='/home/karin/Documents/timeTrajectories/data/deTime/diffTimeTry/'
  dataPathCode='/home/karin/Documents/git/baylor-dicty/R/'
  dataPath='/home/karin/Documents/timeTrajectories/data/countsRaw/combined/'
  dataPathNormalised='/home/karin/Documents/timeTrajectories/data/RPKUM/combined/'
  pathDETime='/home/karin/Documents/timeTrajectories/data/deTime/de_time_impulse_strain/'
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

STAGES<-c('no_agg','stream','lag','tag','tip','slug','mhat','cul','FB','yem')
PHENOTYPES_ORDERED=c('no_agg', 'stream', 'lag', 'tag', 'tip', 'slug', 'mhat', 'cul', 'FB','yem')
PHENOTYPES_X=data.frame(Phenotype=PHENOTYPES_ORDERED,X=c(1:length(PHENOTYPES_ORDERED)))

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

# *** Find stage specific genes with design stage vs stage adjusting for replicates - Not ok as not enough data for this!!! 
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


#*** Find stage specific genes 1 vs all adjusting for replicates (Do in WT or all)
#Uses only replicates present in both stage and rest group
# For stage A use timepoints with phenotypes A and A/B for test and everything without A as control (B, B/C)

#Use replicate as confounder - if FALSE use all data (not only those reps that in stage of interest and in other)
adjust_rep=FALSE
#WT or all - also change the path for saving below
conditions_test=conditions
#conditions_test=conditions[conditions$Group=='WT',]
conditions_test=conditions_test[rowSums(conditions_test[, PHENOTYPES_ORDERED])>0, ]
for (stage in STAGES){
  print(stage)
  if (sum(conditions_test[stage])>0){
    test<-conditions_test[conditions_test[stage]==1,]
    #Select only measurments where 'stage' is the only phenotype e.g. test only A (not A/B)
    #single_stage1<-rowSums(test[,STAGES])==1
    #test<-test[single_stage1,]
    control<-conditions_test[conditions_test[stage]!=1,]
    if (adjust_rep){
      replicates<-intersect(control$Replicate,test$Replicate)
      test<-test[test$Replicate %in% replicates,]
      control<-control[control$Replicate %in% replicates,]
    }
    test$Comparison<-rep(stage,dim(test)[1])
    control$Comparison<-rep('other',dim(control)[1])
    conditions_sub<-rbind(test,control)
    genes_sub<-genes[,rownames(conditions_sub)]
    design_formula=~Replicate+Comparison
    if (!adjust_rep) design_formula=~Comparison
    print(design_formula)
    res<-runDeSeq2(conditions_sub,genes_sub,case=stage,control='other',design=design_formula,main_lvl='Comparison',padj=0.05,logFC=1,
                   path='/home/karin/Documents/timeTrajectories/data/deTime/stage_vs_other/')
  } 
} 


#*** Find stage specific genes 1 vs 1 stage NOT adjusting for replicates
# For stage A use timepoints with phenotype A for test and B as control (not using mixed phenotypes)
for (i in (1:(length(STAGES)-1))){
  for (j in ((i+1):length(STAGES))){
    stage1=STAGES[i]
    stage2=STAGES[j]
    print(paste('Stages:',stage1,stage2))
    
    conditions1<-conditions[conditions[stage1]==1,]
    single_stage1<-rowSums(conditions1[,STAGES])==1
    conditions1<-conditions1[single_stage1,]
    conditions1$Comparison<-rep(stage1,dim(conditions1)[1])
    conditions2<-conditions[conditions[stage2]==1,]
    single_stage2<-rowSums(conditions2[,STAGES])==1
    conditions2<-conditions2[single_stage2,]
    conditions2$Comparison<-rep(stage2,dim(conditions2)[1])
    if (dim(conditions1)[1]>0 & dim(conditions2)[1]>0){
      conditions_sub<-rbind(conditions1,conditions2)
    
      genes_sub<-genes[,rownames(conditions_sub)]
      res<-runDeSeq2(conditions_sub,genes_sub,case=stage1,control=stage2,design=~Comparison,main_lvl='Comparison',padj=0.05,logFC=1,
                   path='/home/karin/Documents/timeTrajectories/data/deTime/stage_vs_stage/')
    }
  }
}
    
# Process this in python to retain only genes overexpressed against all other stages

# Find genes overexpressed in WT (AX4, MybBGFP) stage compared to rest of WT and strains that do not reach the stage, excluding PD
# Other strains that reach the stage are excluded because they can be different from WT despite the similar phenotype
# PD is excluded as it shows WT expression in non WT phenotypes
# Comparison: test: WT A and A/B, control: WT non A, all measurments from all replicates (excluding PD) that do not reach the satge
stage_order<-data.frame(Stage=c('no_agg', 'stream', 'lag', 'tag', 'tip', 'slug', 'mhat', 'cul', 'FB'), Order=1:9)
max_stages<-c()
replicates<-as.vector(unique(conditions$Replicate))
for(replicate in replicates){
  data_strain<-conditions[conditions$Replicate == replicate,as.vector(stage_order$Stage)]
  max_stage=1
  for (stage in stage_order$Stage){
    if (any(data_strain[stage]==1)){
      order<-stage_order[stage_order$Stage==stage,'Order'][1]
      if (order > max_stage) max_stage=order
    }
  }
  max_stages<-c(max_stages,max_stage)
}
max_stages<-data.frame(Replicate=replicates,Max_stage=max_stages)
#Replace max stages of strains with no data with 'NA'
for( strain in c('ac3PkaCoe','gtaC','gtaI')){
  for (replicate in unique(conditions[conditions$Strain ==strain, 'Replicate'])){
    max_stages[max_stages$Replicate==replicate,'Max_stage']<-'NA'
  }
}
#Add tip to tagB_r2, so that it is not used in comparisons for not-tip as it may have a tip (image was not taken)
max_stages[max_stages$Replicate=='tagB_r2','Max_stage']<-5

for (stage in stage_order$Stage[2:9]){
  print(stage)
  stage_n<-stage_order[stage_order$Stage==stage,'Order'][1]
  test<-conditions[conditions[stage]==1 & conditions['Group']=='WT',]
  control_WT<-conditions[conditions[stage]!=1 & conditions['Group']=='WT',]
  other_reps<-as.vector(max_stages[max_stages$Max_stage < stage_n,'Replicate'])
  control_other<-conditions[conditions$Replicate %in% other_reps & conditions$Group !='PD',]
  print(unique(control_other$Replicate))
  control<-rbind(control_WT,control_other)
  print(paste(dim(test),dim(control_WT),dim(control_other),dim(control),sep=','))
  
  test$Comparison<-rep(stage,dim(test)[1])
  control$Comparison<-rep('other',dim(control)[1])
  conditions_sub<-rbind(test,control)
  genes_sub<-genes[,rownames(conditions_sub)]
  res<-runDeSeq2(conditions_sub,genes_sub,case=stage,control='other',design=~Comparison,main_lvl='Comparison',padj=0.05,logFC=1,
                 path='/home/karin/Documents/timeTrajectories/data/deTime/stageWT_vs_other/')
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

# **********************************************
# ********* Genes DE across stages - use impulse to find DE genes while using ordered stages as time

#Prepare data - add sample to each stage it has annotated
conditions_annotated=conditions[rowSums(conditions[, PHENOTYPES_ORDERED])>0, ]
X=data.frame(row.names=rownames(genes))
Y=data.frame()
for (row_idx in 1:dim(conditions_annotated)[1]){
  y=conditions_annotated[row_idx,]
  x=genes[,rownames(y)]
  repeated_sample=0
  for (phenotype in PHENOTYPES_ORDERED){
    if (y[phenotype]==1){
      sample_name=paste(rownames(y),repeated_sample,sep='_')
      X[,sample_name]=x
      Y[sample_name,c('Sample','Condition','Batch','Time')]=c(sample_name,'case',y['Replicate'],PHENOTYPES_X[PHENOTYPES_X['Phenotype']==phenotype,'X'])
      repeated_sample=repeated_sample+1
    }
  }
}
Y<-Y[order(Y$Time),]
#Run ImpulseDE2
objectImpulseDE2 <- runImpulseDE2(matCountData = as.matrix(X), dfAnnotation = Y,boolCaseCtrl = FALSE,vecConfounders = c("Batch"),scaNProc = 4 )
saveRDS(object=objectImpulseDE2,file=paste(pathSaveStagesImpulse,'DEacrossStages.rds',sep=''))
saveRDS(object=PHENOTYPES_X,file=paste(pathSave,'DEacrossStages_phenotypesOrder.rds',sep=''))
fdr=0.05
result<-objectImpulseDE2$dfImpulseDE2Results
result<-result[result$padj<=fdr & ! is.na(result$padj),]
result<-result[order(result$padj),]
write.table(result,file=paste(pathSaveStagesImpulse,'DEacrossStages_padj',fdr,'.tsv',sep='') ,sep='\t',row.names = FALSE)

#**** Find DE genes across stages in individual strains
for (strain in unique(conditions$Strain)){
  conditions_annotated=conditions[rowSums(conditions[, PHENOTYPES_ORDERED])>0 &conditions$Strain==strain, ]
  #Drop phenotypes not annoated in this strain
  phenotypes_count<-colSums(conditions_annotated[, PHENOTYPES_ORDERED])
  if (length(phenotypes_present)>1){
    phenotypes_present<-names(phenotypes_count[phenotypes_count>0])
    phenotypes_x=data.frame(Phenotype=phenotypes_present,X=c(1:length(phenotypes_present)))
    drop<-names(phenotypes_count[phenotypes_count==0])
    drop<-which(colnames(conditions_annotated) %in% drop)
    conditions_annotated=conditions_annotated[,-drop]
    X=data.frame(row.names=rownames(genes))
    Y=data.frame()
    for (row_idx in 1:dim(conditions_annotated)[1]){
      y=conditions_annotated[row_idx,]
      x=genes[,rownames(y)]
      repeated_sample=0
      for (phenotype in phenotypes_present){
        if (y[phenotype]==1){
          sample_name=paste(rownames(y),repeated_sample,sep='_')
          X[,sample_name]=x
          Y[sample_name,c('Sample','Condition','Batch','Time')]=c(sample_name,'case',y['Replicate'],phenotypes_x[phenotypes_x['Phenotype']==phenotype,'X'])
          repeated_sample=repeated_sample+1
        }
      }
    } 
    Y<-Y[order(Y$Time),]
    objectImpulseDE2 <- runImpulseDE2(matCountData = as.matrix(X), dfAnnotation = Y,boolCaseCtrl = FALSE,vecConfounders = c("Batch"),
                                      boolIdentifyTransients = TRUE,scaNProc = 4 )
    saveRDS(object=objectImpulseDE2,file=paste(pathSaveStagesImpulse,strain,'_DEacrossStages.rds',sep=''))
    saveRDS(object=phenotypes_x,file=paste(pathSave,strain,'_DEacrossStages_phenotypesOrder.rds',sep=''))
  }
}














