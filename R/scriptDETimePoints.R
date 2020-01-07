

#Variables
lab=TRUE
if (lab){
  dataPathCode='/home/karin/Documents/timeTrajectories/deTime/'
  dataPathSaved='/home/karin/Documents/timeTrajectories/data/deTime/new/'
  dataPath='/home/karin/Documents/timeTrajectories/data/countsRaw/combined/'
  dataPathNormalised='/home/karin/Documents/timeTrajectories/data/'
  dataPathNormalisedNew='/home/karin/Documents/timeTrajectories/data/RPKUM/combined/'
}else {
  dataPathSaved='/home/karin/Documents/DDiscoideum/data/timeDE/'
  dataPath='/home/karin/Documents/DDiscoideum/data/countsRaw/'
  dataPathNormalised='/home/karin/Documents/DDiscoideum/'
}

library("DESeq2")
library('dplyr')
library(ggplot2)
library(RColorBrewer)
library("ImpulseDE2")
source(paste(dataPathCode,'functionsDETime.R',sep=''))
source(paste(dataPathCode,'functionsDETime.R',sep=''))
library(mclust)
library(lmms)
library("BiocParallel")
register(MulticoreParam(4))
library(ComplexHeatmap)
library(circlize)
library(gdata)

#Load data counts
#genes<-read.table(paste(dataPath,"mergedGenes_counts.tsv",sep=''), header=TRUE,row.names='Gene', sep="\t")
genes<-read.table(paste(dataPath,"mergedGenes_counts.tsv",sep=''), header=TRUE,row.names=1, sep="\t")
conditions<-read.table(paste(dataPathNormalisedNew,"conditions_mergedGenes.tsv",sep=''), header=TRUE,row.names='Measurment', sep="\t")
#rownames(conditions)=gsub('-','.',rownames(conditions))
#R imported colnames of genes with changes but gene IDs remained ok
#Load data normalised
dataNorm<-loadDataOrangeTable(dataPathNormalised)
genesNotNullNorm<-dataNorm$gNN
genesNorm<-dataNorm$genes
genesTNNNorm<-dataNorm$gTNN
conditionsNorm<-dataNorm$condit
genesNormNew<-read.table(paste(dataPathNormalisedNew,"mergedGenes_RPKUM.tsv",sep=''), header=TRUE,row.names=1, sep="\t")

#*************************************************************
#*****Calculate DE of mutant vs AX4 for each time point

#Remove unneeded 'samples' (averages, AX4_r3,4: pool_19, pool_21) (Pruned genes, conditions)
#genesP<-genes[,conditions$Include]
#conditionsP<-conditions[conditions$Include,]
conditionsP<-conditions
rownames(conditionsP)<-make.names(rownames(conditionsP))
genesP<-genes

#Make times and associated samples data
timepoints<-c(0,1,2,3,4,5,6,8,10,12,14,16,18,20,24)
samples<-list()
for(t in timepoints){
  measurmentsT=conditionsP[conditionsP$Time==t,]
  strainsT=unique(measurmentsT$Strain)
  mutantsT=strainsT[strainsT!='AX4']
  samples[[as.character(t)]]<-as.vector(mutantsT)
}

#Make dds for specific time point
#If error in samplenames matching return NaN
buildDDS<-function(t,conditionsP,genesP){
  conditionsPT=conditionsP[conditionsP$Time==t,]
  genesPT=genesP[,conditionsP$Time==t]

  #Make coldata table for DeSeq2 
  coldata<-data.frame(condition=paste(conditionsPT$Strain,'_',conditionsPT$Time,sep=''))
  rownames(coldata)<-rownames(conditionsPT)
  #Check that sample descriptions match betweeen expression  data and description
  if (all(rownames(coldata)==colnames(genesPT))){
    #Make DeSeq2 data object
    dds <- DESeqDataSetFromMatrix(countData = genesPT,
                                colData = coldata,
                                design = ~ condition)
    dds$condition <- relevel(dds$condition, ref = paste("AX4_",t,sep=''))
    return(dds)
  }
  else {
    stop("Sample names in conditions and genes do not match")
  }
}


#Get results from dds analysed with DESeq of sample1 against AX4 (at timepoint of dds) - get specific contrast
testDE<-function(dds,sample1,padjSave,logFCSave,t){
  res <- results(dds,contrast=c("condition", paste(sample1,'_',t,sep=''), paste("AX4_",t,sep='')),parallel = TRUE)
  resNN <- na.omit(res)
  resOrder<-resNN[order(resNN$padj),]
  #plot(resNN$log2FoldChange,-log10(resNN$padj))
  resFilter<-resOrder[resOrder$padj<=padjSave & abs(resOrder$log2FoldChange)>=logFCSave,]
  #R reads this format fine, but Libre offic shifts colnames!!!
  write.table(x = resFilter,file =paste( dataPathSaved,'DE_',sample1,'_t',t,'.tsv',sep=''),sep='\t')
}


#Make all the comparisons

for(t in timepoints){
  print(paste('Building for',t))
  dds=buildDDS(t,conditionsP,genesP)
  dds <- DESeq(dds,parallel = TRUE)
  #Get contrasts
  mutantsT=samples[[as.character(t)]]
  for (mutant in mutantsT){
    print(paste('Analysing',mutant))
    testDE(dds,mutant,padjSave=0.1,logFCSave=0.5,t=t)
  }
}
    
#Make a data table of DE genes for each strain in each time point, keep only padj<=0.05 and abs(logFC)>=1
#Matrix to specify which genes are De in which sample grouop, 0 if not DE under the condition, in matrix are log2FC
#make conditions matrix
first=TRUE
deDF=NaN
#sampleStrains<-c()
#sampleTimes<-c()
#sampleNames<-c()
for (t in timepoints){
  for (mutant in samples[[as.character(t)]]){
    sampleName=paste(mutant,'_',t,sep='')
    #sampleStrains<-append(sampleStrains,mutant)
    #sampleTimes<-append(sampleTimes,t)
    #sampleNames<-append(sampleNames,sampleName)
    de<-read.table(file =paste( dataPathSaved,'DE_',mutant,'_t',t,'.tsv',sep=''),sep='\t',header=TRUE)
    deFilter<-de[de$padj<=0.05 & abs(de$log2FoldChange)>=1,]
    filterDF<-data.frame(a=rownames(deFilter),b= deFilter$log2FoldChange)
    colnames(filterDF)<-c('Gene',sampleName)
    if(first){
      deDF=filterDF
      first=FALSE
    }
    else{
      deDF=merge(deDF,filterDF,by='Gene',all=TRUE)
      deDF[is.na(deDF)] <- 0
    }
  }
}
write.table(x = deDF,file =paste( dataPathSaved,'DE_p0.05_abslog2FC1.tsv',sep=''),sep='\t')
#conditionsDE<-data.frame(sampleName=sampleNames,strain=sampleStrains,time=sampleTimes)

#******************Plot N DE
#Plot N of DE (up/down) in timepoints
#Extract data about N up/down per strain and t point
strains<-c()
times<-c()
directions<-c()
deN<-c()
for (sampleName in tail(colnames(deDF),-1)){
  data=strsplit(sampleName,'_')
  time=as.numeric(data[[1]][2])
  strain=data[[1]][1]
  strains<-append(strains,rep(strain,2))
  times<-append(times,rep(time,2))
  upN<-sum(deDF[,sampleName]>0)
  downN<-sum(deDF[,sampleName]<0)
  directions<-append(directions,c('up','down'))
  deN<-append(deN,c(upN,downN))
  
}
deNumbers<-data.frame(strain=strains,time=times,direction=directions,N=deN)

#Plot how many genes De up or down
ggplot(deNumbers, aes(x=time, y=N, shape=direction, color=strain,linetype=direction)) +
  geom_point()+geom_line(size=0.5)+
  scale_shape_manual(values=c(6,17))+
  scale_linetype_manual(values=c('dashed','solid'))+
  scale_color_manual( values=c("purple", "yellow", "violet",'skyblue','darkblue','green'))

#How many shared genes DE between strains
comparisons<-c()
deNums<-c()
times<-c()
directions<-c()
for(t in timepoints){
  mutants=samples[[as.character(t)]]
  mutantN=length(mutants)
  mutantSort<-sort(mutants)
  for(i in seq(1,mutantN-1)){
    for(j in seq(i+1,mutantN)){
      #print(paste(i,j))
      strain1=mutantSort[i]
      strain2=mutantSort[j]
      sample1=paste(strain1,'_',t,sep='')
      sample2=paste(strain2,'_',t,sep='')
      print(paste(sample1,sample2))
      comp=paste(strain1,'_',strain2,sep='')
      comparisons<-append(comparisons,rep(comp,3))
      times<-append(times,rep(t,3))
      directions<-append(directions,c('up','down','oposite'))
      df<-deDF[,c(sample1,sample2)]
      upN=sum(rowSums(df>0)==2)
      downN=sum(rowSums(df<0)==2)
      deN=sum(rowSums(df!=0)==2)
      opositeN=deN-(upN+downN)
      deNums<-append(deNums,c(upN,downN,opositeN))
    }
    
  }
}
deNumbersPairs<-data.frame(comparison=comparisons,time=times,direction=directions,N=deNums)

#Plot
col = brewer.pal(n = 9, name = "Set1")
col<-append(col,brewer.pal(n = 8, name = "Pastel2"))
ggplot(deNumbers, aes(x=time, y=N, shape=direction, color=comparison,linetype=direction)) +
  geom_point()+geom_line(size=0.5)+
  scale_shape_manual(values=c(6,1,2))+
  scale_color_manual(values=col)+
  scale_linetype_manual(values=c('dashed','dotted','solid'))+
  theme_bw()


#How many shared genes compared to all genes for each strain pair
strains<-unique(conditions$Strain)
mutants<-strains[strains!='AX4']
mutants<-as.vector(sort(mutants))
mutantN<-length(mutants)
for(i in seq(1,mutantN-1)){
  for(j in seq(i+1,mutantN)){
    mutant1=mutants[i]
    mutant2=mutants[j]
    deIndividual=deNumbers[deNumbers$strain==mutant1 | deNumbers$strain==mutant2,]
    dePair=deNumbersPairs[deNumbersPairs$comparison==paste(mutant1,'_',mutant2,sep=''),]
    colnames(dePair)<-colnames(deIndividual)
    bind<-rbind(deIndividual,dePair)
    ggplot(bind, aes(x=time, y=N, shape=direction, color=strain,linetype=direction)) +
      geom_point()+geom_line(size=0.5)+
      scale_shape_manual(values=c(6,2,1))+
      scale_color_manual(values=c('skyblue','red','purple'))+
      scale_linetype_manual(values=c('dashed','solid','dotted'))+
      theme_bw()
   ggsave(paste(dataPathSaved,'DE_',mutant1,'_',mutant2,'.png',sep=''))
    
  }
}

#*******************Heatmap

#Plot DE expression patterns
deDF<-read.csv(paste(dataPathSaved,'DE_p0.05_abslog2FC1.tsv',sep=''),sep='\t')
#Genes in DE and normalised data (DE also has nc genes)
#genes_plot=intersect(colnames(genesNotNullNorm),deDF[,1])
genes_plot=intersect(rownames(genesNormNew),deDF[,1])
#If times are to be coloured by different colours
#time_cols=list(Time=c('0'='blueviolet','1'='slateblue4','2'='blue','3'='royalblue','4'='cyan','5'='olivedrab3','6'='chartreuse3',
#                       '8'='darkolivegreen2','10'='yellow','12'='orange', '14'='brown','16'='red','18'='black','20'='pink','24'='gray'))
time_cols = colorRamp2(c(min(timepoints),max(timepoints)), c("white", "blue"))
#Make matrix from df
dataDE<-deDF[,2:ncol(deDF)]
rownames(dataDE)<-deDF[,1]
#dataDE<-dataDE
m<-data.matrix(dataDE)
m<-t(m)
m<-m[,genes_plot]
#Make conditions df
strains_dataDE<-c()
times_dataDE<-c()
for(measurment in rownames(m)){
  data_parts=strsplit(measurment,'_')
  strains_dataDE<-append(strains_dataDE,data_parts[[1]][1])
  times_dataDE<-append(times_dataDE,data_parts[[1]][2])
}
conditions_dataDE=data.frame(Strain=strains_dataDE,Time=as.numeric(times_dataDE))
rownames(conditions_dataDE)=rownames(m)
conditions_dataDE <- conditions_dataDE[order(conditions_dataDE$Strain, conditions_dataDE$Time),]

strains_PCA<-c('gtaI','mybBGFP','cudA','gtaG','pkaCoeAX4','pkaR',
              'comH','tagB','dgcA','ac3pkaCoe','gtaC','tgrB1','tgrB1C1','tgrC1',
              'ecmA','gbfA','acaA','amiB','mybB')
conditions_dataDE$Strain<- reorder.factor(conditions_dataDE$Strain, new.order=strains_PCA)
conditions_dataDE<-conditions_dataDE %>% arrange(Strain)
rownames(conditions_dataDE)<-paste(conditions_dataDE$Strain,conditions_dataDE$Time,sep='_')
m<-m[rownames(conditions_dataDE),]

#col_time=as.list(c(brewer.pal(8,'Dark2'),brewer.pal(7,'Pastel2'))) #Must be named list
#time_anno = rowAnnotation(Time = as.character(conditions_dataDE$Time),col=time_cols)
time_anno = rowAnnotation(Time = conditions_dataDE$Time,col=list(Time = time_cols),annotation_legend_param = list(Time = list(at = c(0, 6,12, 18,24))))
n_clusters=8
heatmap_de=Heatmap(m,column_km = n_clusters,row_order = rownames(conditions_dataDE),cluster_rows =  FALSE,
        show_row_names = FALSE,
        show_column_names = FALSE,
       row_split = conditions_dataDE$Strain,left_annotation = time_anno,name='log2FC',row_title_gp=gpar(fontsize=6))

#AX4 heatmap 
#MAke AX4 data
#genes_AX4=genesNorm[conditionsNorm$Strain=='AX4_avr',genes_plot]
genes_AX4=genesNormNew[genes_plot,conditions$Strain=='AX4']
times_AX4=conditions[conditions$Strain=='AX4','Time']
timepoints_AX4=unique(times_AX4)
first=TRUE
genes_AX4_avr=NaN
for (time in timepoints_AX4){
  genes_time=genes_AX4[,times_AX4==time]
  df=as.data.frame(rowMeans(genes_time))
  colnames(df)=time
  if (first){
    genes_AX4_avr=df
    first=FALSE
  }else{
    genes_AX4_avr=merge(genes_AX4_avr,df,by=0, all=TRUE)
    rownames(genes_AX4_avr)=genes_AX4_avr$Row.names
    genes_AX4_avr=genes_AX4_avr[,2: dim(genes_AX4_avr)[2]]
  }
}

#Normalise by max of each gene
#genes_AX4_T=t(genes_AX4)
genes_AX4_T=genes_AX4_avr
row_max<-apply(genes_AX4_T, 1, max)
row_max[row_max==0]=1
genes_AX4_T<-genes_AX4_T[ , order(as.numeric(names(genes_AX4_T)))]
genes_AX4<-t(genes_AX4_T/row_max)
#TO DO figure out what pseudocount to add no to disturb log
#genes_AX4[genes_AX4<1]=1
###genes_AX4<-data.matrix(log10(genes_AX4+0.001))
#genes_AX4<-data.matrix(log2(genes_AX4))
#times_AX4=conditionsNorm[conditionsNorm$Strain=='AX4_avr','Time']
times_AX4=rownames(genes_AX4)
#time_anno_AX4 = rowAnnotation(Time = as.character(times_AX4),col=time_cols)
time_anno_AX4 = rowAnnotation(Time = as.numeric(times_AX4),col=list(Time = time_cols))
col_AX4= colorRamp2(c(min(genes_AX4),max(genes_AX4)), c( "white", "brown"))
heatmap_wild=Heatmap(genes_AX4,cluster_rows = FALSE,cluster_columns=FALSE,
                     left_annotation = time_anno_AX4,row_order = rownames(genes_AX4),show_row_names = FALSE,
                     show_column_names = FALSE,name='Max scaled expression',col = col_AX4)

#Plot heatmaps
if( sum(colnames(genes_AX4)!=colnames(m))==0){
  ht_list =heatmap_de %v% heatmap_wild
  heatmap=draw(ht_list)
}

#Extract clusters
genes_clusters<-c()
clusters<-c()
for(i in seq(n_clusters)){
  genes_cluster<-colnames(m[,column_order(heatmap)[[i]]])
  cluster=names(column_order(heatmap))[i]
  genes_clusters<-append(genes_clusters,genes_cluster)
  clusters<-append(clusters,rep(cluster,length(genes_cluster)))
}
df_cluster<-data.frame(Genes=genes_clusters,Clusters=as.numeric(clusters))

#Save DE file for Orange:
de_Orange<-t(dataDE)
de_Orange <- merge(de_Orange, conditions_dataDE, by=0, all=TRUE)  
#colnames(de_Orange)<-c('Row.names',colnames(de_Orange)[2:ncol(de_Orange)])
rownames(de_Orange)<-de_Orange$Row.names
de_Orange<-de_Orange[,2:ncol(de_Orange)]
de_Orange <- t(merge(t(de_Orange), df_cluster, by.x=0, by.y='Genes',all=TRUE))
colnames(de_Orange)<-de_Orange[1,]
de_Orange<-de_Orange[2:nrow(de_Orange),]
de_Orange[nrow(de_Orange),colnames(de_Orange)=='Row.names']='Cluster'
de_Orange<-as.data.frame(de_Orange)
de_Orange<-de_Orange[order(de_Orange$Strain,de_Orange$Time),]
columns=colnames(de_Orange)
de_Orange<-cbind(de_Orange,rownames(de_Orange))
colnames(de_Orange)<-c(columns,'row_names')
write.table(de_Orange,paste(dataPathSaved,'DE_p0.05_abslog2FC1_Orange.tsv',sep=''),sep='\t',row.names = FALSE,col.names = TRUE)


#********************************************************
#*********Find genes that are DE expressed in time

#Perform lmms DE of single genes in time
not_all_na <- function(x) any(x!=0)
filterLmms<- function(genes,conditions,strain,useLog=TRUE,doPlot=FALSE,subsetGenes=NaN){
  if (is.nan(subsetGenes)){
    subsetGenes=ncol(genes)
  }
  genesSub<-genes[conditions$Strain==strain & conditions$Include,1:subsetGenes]
  genesSNN<-data.matrix(as.data.frame(genesSub) %>% select_if(not_all_na))
  #Add pseudocount before log2
  if (useLog) genesLog2<-log2(genesSNN+0.01)
  conditionsSub=conditions[conditions$Strain==strain & conditions$Include,]
  timeSub<-conditionsSub$Time
  sampleSub<-conditionsSub$Sample
  if (!useLog){ noiseTest <-investNoise(data=genesSNN,time=timeSub,sampleID=sampleSub,log=FALSE)}
  else {noiseTest <-investNoise(data=genesLog2,time=timeSub,sampleID=sampleSub,log=TRUE)}
  clusterFilter <- Mclust(cbind(noiseTest@RT,noiseTest@RI),G=2)
  if(doPlot) plot(clusterFilter,what = "classification")
  meanRTCluster <-tapply(noiseTest@RT,clusterFilter$classification,mean)
  bestCluster <- names(meanRTCluster[which.min(meanRTCluster)])
  filterData=NaN
  if(!useLog) {filterData <- genesSNN[,clusterFilter$classification==bestCluster]}
  else {filterData <- genesLog2[,clusterFilter$classification==bestCluster]}
  return(filterData)
}


#Extract DE genes for each strain
#lmms
strains=unique(conditionsNorm[conditionsNorm$Include,]$Strain)
for (strain in strains){
  print(strain)
  lmmsL<-list()
  reps=5
  for(i in seq(1,reps)){
    lmmsR<-filterLmms(genes=genesNotNullNorm,conditions=conditionsNorm,strain=strain,useLog = FALSE,doPlot=TRUE)
    lmmsL[[i]]=colnames(lmmsR)
  }
  unionG<-lmmsL[[1]]
  #print(length(lmmsL[[1]]))install.packages("devtools", dependencies=TRUE)
  for (i in seq(2,reps)){
    unionG<-union(unionG,lmmsL[[i]])
    #print('len added and union')
    #print(length(lmmsL[[i]]))
    #print(length(unionG))
  }
  print(length(unionG))
  write(unionG,file=paste(dataPathSaved,'DEinTime_',strain,'.txt',sep=''))
}

#Plot of expression classification was strange for AX4, tgrC1 and gbfA

#ImpulseDE2: For genesP and conditionsP
filterImpulse <- function(strain,padj=0.05,threads=4,subsetGenes=NaN,genes,conditions) {
  if (is.nan(subsetGenes)) subsetGenes=nrow(genes)
  genesSample<-as.matrix(genes[1:subsetGenes,conditions$Strain==strain & conditions$Include])
  conditionsSub<-conditions[conditions$Strain==strain & conditions$Include,]
  anno<-data.frame(Sample=rownames(conditionsSub), Condition=rep("case",ncol(genesSample)), Time=as.numeric(conditionsSub$Time), Batch=as.character(conditionsSub$Replicate),stringsAsFactors = FALSE)
  #Remove this rounding somehow
  de <- runImpulseDE2(matCountData=round(genesSample),dfAnnotation=anno,boolCaseCtrl= FALSE,vecConfounders=c("Batch"),scaNProc= threads ,boolIdentifyTransients = TRUE,)
  deorder<-de$dfImpulseDE2Results[order(de$dfImpulseDE2Results$padj),]
  deImpulse<-deorder[deorder$padj<=padj,]
  return(deImpulse)
}
#Extract de genes in time impulseDE2
strains=unique(conditionsNorm[conditionsNorm$Include,]$Strain)
for (strain in strains){
  print(strain)
  impulseR<-filterImpulse(genes=genesP,conditions=conditionsP,strain=strain)
  write.table(impulseR,file=paste(dataPathSaved,'DEinTime_ImpulseDe2_Transient_',strain,'.tsv',sep=''),sep='\t',row.names = FALSE)
}

#Plot genes declared DE in time:

outersect <- function(x, y) {
  sort(c(x[!x%in%y],
         y[!y%in%x]))
}
#Load data
impulseORlmms=TRUE
strain='tagB'
if (impulseORlmms){
impulse_res<-read.csv(paste(dataPathSaved,'DEinTime_ImpulseDe2_Transient_',strain,'.tsv',sep=''),sep='\t',header = TRUE)
#Filter ImpulseDE2 result
impulse_y<-impulse_res[impulse_res$padj<=0.05,]$Gene
}else {lmms_y<-scan(paste(dataPathSaved,'DEinTime_',strain,'.txt',sep=''),what = "character")}

#Get gene names for plotting (de or not)
if (impulseORlmms){names_y=intersect(colnames(genesNorm),impulse_y)
}else{names_y=intersect(colnames(genesNorm),lmms_y)}

names_n=outersect(names_y,colnames(genesNorm))

#Prepare data for heatmap
genes_strain=genesNorm[conditionsNorm$Strain==paste(strain,'_avr',sep=''),]
#Normalise by max of each gene
genes_strain_T=t(genes_strain)
row_max<-apply(genes_strain_T, 1, max)
row_max[row_max==0]=1
genes_strain<-t(genes_strain_T/row_max)
genes_strain_y=genes_strain[,names_y]
genes_strain_n=genes_strain[,names_n]

if (impulseORlmms){
transient=c()
monotonous=c()
for (gene in colnames(genes_strain_y)){
  data=filter(impulse_res, Gene == gene)
  transient=as.character(append(transient,data$isTransient))
  monotonous=as.character(append(monotonous,data$isMonotonous))
}
color_type=c("FALSE" = "red", "TRUE" = "green")
ha_type = HeatmapAnnotation(
  Monotonous=monotonous,
  Transient=transient,
  col = list(Monotonous = color_type,Transient =color_type),annotation_name_side = "left"
)
}else{
  ha_type=NULL
}
times_strain=conditionsNorm[conditionsNorm$Strain==paste(strain,'_avr',sep=''),'Time']
time_anno_strain = rowAnnotation(Times = as.character(times_strain) )
col_strain= colorRamp2(c(0,1), c( "blue", "yellow"))
heatmap_y=Heatmap(genes_strain_y,cluster_rows = FALSE,cluster_columns=TRUE,
                     left_annotation = time_anno_strain,row_order = rownames(genes_strain),show_row_names = FALSE,
                     show_column_names = FALSE,name='Max scaled expression',col = col_strain, column_title ='DE',top_annotation = ha_type)
heatmap_n=Heatmap(genes_strain_n,cluster_rows = FALSE,cluster_columns=TRUE,
                  left_annotation = time_anno_strain,row_order = rownames(genes_strain),show_row_names = FALSE,
                  show_column_names = FALSE,name='Max scaled expression',col = col_strain, column_title ='Not DE')

heatmap_y+heatmap_n


#***********************************************
#Genes DE between strains at the beginning of developmental differences

#times_DE<-c(6,8,10,12,14,16,18,20,24)
times_DE<-c(8,12,16,20,24)
#Genes DE in between strains, impulseDE2, together for replicates of each strain:
#With strains
#strain1='AX4'    
strain1='comH'
strain2='tagB'
#For genesP and conditionsP
#strain1-control, strain2-case
deImpulse <- function(strain1='AX4',strain2,padj=0.05,threads=4,genes,conditions,timesDE) {
  genesSub<-as.matrix(genes[,(conditions$Strain==strain1|conditions$Strain==strain2) & conditions$Include & 
                               (conditions$Time %in% times_DE)])
  conditionsSub<-conditions[(conditions$Strain==strain1|conditions$Strain==strain2) & conditions$Include & 
                               (conditions$Time %in% times_DE ),]
  #OR With replicates:
  # strain1='AX4'
  # strain2='tagB'
  # replicates1=c('AX4_E','AX4_A')
  # replicates2=c('tagB_M','tagB_L')
  # genesSub<-as.matrix(genesP[,(conditionsP$Replicate %in% replicates1|conditionsP$Replicate %in% replicates2) & conditionsP$Include & 
  #                              (conditionsP$Time %in% times_DE)])
  # conditionsSub<-conditionsP[(conditionsP$Replicate %in% replicates1|conditionsP$Replicate %in% replicates2) & conditionsP$Include & 
  #                              (conditionsP$Time %in% times_DE ),]
  # 
  
  #Annotate as case/control
  case<-c()
  for(s in conditionsSub$Strain){
    if (s==strain1){
      case<-c(case,c("control"))
    }
    else {
      case<-c(case,c("case"))
    }
  }
  #time - double; condition,sample,batch - character
  anno<-data.frame(Sample=rownames(conditionsSub), Condition=case,Time=as.numeric(conditionsSub$Time),Batch=as.character(conditionsSub$Replicate),stringsAsFactors = FALSE)
  rownames(anno)<-anno$Sample
  #Does not work with batches if different number of replicates. In that case works without batches, but drops then when correcting for batches.?
  #To correct this estimate dispersion and scaling factors with DESeq2 as described in “Model matrix not full rank” and 
  #use processData to make Impulse object.Then use fitModels and runDEAnalysis
  #Thus for now done without batches
  de <- runImpulseDE2(matCountData=genesSub,dfAnnotation=anno,boolCaseCtrl= TRUE,scaNProc= 4
                     # ,vecConfounders=c("Batch") 
                      ) 
  de_filter<-de$dfImpulseDE2Results[de$dfImpulseDE2Results$padj<=padj & ! is.na(de$dfImpulseDE2Results$padj),]
  return(de_filter)
}


#Extract de genes in between strains with impulse

for (strain in mustantsT){
  print(strain)
  impulseR<-deImpulse(strain1='AX4',strain2=strain,threads=4,genes=genesP,conditions=conditionsP,timesDE=timesDE)
  write.table(impulseR,file=paste(dataPathSaved,'DEagainstAX4_noBatch_',strain,'.tsv',sep=''),sep='\t',row.names = FALSE)
}

  
#Plot DE between strains  
#Max possible 0.05 based on pre-filtering
padj_cutoff=0.05
#Plot DE/not_DE/both together
do_plot='both'
#Load data
impulse_res<-read.csv(paste(dataPathSaved, 'EXAMPLE_DE_ImpulseDe2_t',paste(times_DE,collapse = "."),'_',strain1,strain2,'.tsv',sep=''),
                      sep='\t',header = TRUE)
#DE/not DE genes 
names_y=intersect(colnames(genesNorm),impulse_res[impulse_res$padj<=padj_cutoff  ,'Gene'])
names_n=outersect(colnames(genesNorm),names_y)

#Gene data of each strain
if (do_plot=='both'){
  genes_strain1=genesNorm[conditionsNorm$Strain==paste(strain1,'_avr',sep='')  &
                            (conditionsNorm$Time %in% times_DE), ]
  genes_strain2=genesNorm[conditionsNorm$Strain==paste(strain2,'_avr',sep='')  &
                            (conditionsNorm$Time %in% times_DE),]
  
}else{
  if(do_plot=='DE'){
    names=names_y
  }else{
    names=names_n
  }
  genes_strain1=genesNorm[conditionsNorm$Strain==paste(strain1,'_avr',sep='')  &
                            (conditionsNorm$Time %in% times_DE),names ]
  genes_strain2=genesNorm[conditionsNorm$Strain==paste(strain2,'_avr',sep='')  &
                            (conditionsNorm$Time %in% times_DE),names ]
  
}

time_cols=list(Times=c('0'='blueviolet','2'='blue','4'='cyan','6'='chartreuse3','8'='darkolivegreen2','10'='yellow','12'='orange',
                       '14'='brown','16'='red','18'='black','20'='pink','24'='gray'))
#Normalise by max of each gene
genes_strain_T1=t(genes_strain1)
row_max1<-apply(genes_strain_T1, 1, max)
row_max1[row_max1==0]=1
genes_strain1<-t(genes_strain_T1/row_max1)
times_strain1=conditionsNorm[conditionsNorm$Strain==paste(strain1,'_avr',sep='')  &
                              (conditionsNorm$Time %in% times_DE),'Time']
time_anno_strain1 = rowAnnotation(Times = as.character(times_strain1),col=time_cols)

genes_strain_T2=t(genes_strain2)
row_max2<-apply(genes_strain_T2, 1, max)
row_max2[row_max2==0]=1
genes_strain2<-t(genes_strain_T2/row_max2)
times_strain2=conditionsNorm[conditionsNorm$Strain==paste(strain2,'_avr',sep='')  &
                               (conditionsNorm$Tim %in% times_DE),'Time']
time_anno_strain2 = rowAnnotation(Times = as.character(times_strain2),col=time_cols)
#Make DE/not DE annotation
de_names=c()
for(gene in colnames(genes_strain1)){
  if (gene  %in% names_y){
    de_names=append(de_names,'y')
  }else{
    de_names=append(de_names,'n')
  }
}

ha = HeatmapAnnotation(DE =de_names,col = list(DE = c("y" = "green", "n" = "red")))

col_strain= colorRamp2(c(0,1), c( "blue", "yellow"))
heatmap_1=Heatmap(genes_strain1,cluster_rows = FALSE,cluster_columns=TRUE,
                  left_annotation = time_anno_strain1,row_order = rownames(genes_strain1),show_row_names = TRUE,
                  show_column_names = FALSE,name='Max scaled expression',col = col_strain, top_annotation = ha  
                  )
heatmap_2=Heatmap(genes_strain2,cluster_rows = FALSE,cluster_columns=FALSE,
                  left_annotation = time_anno_strain2,row_order = rownames(genes_strain2),show_row_names = TRUE,
                  show_column_names = FALSE,name='Max scaled expression',col = col_strain)

if( sum(colnames(genes_strain1)!=colnames(genes_strain2))==0){
  ht_list =heatmap_1 %v% heatmap_2
  heatmap=draw(ht_list)
}

