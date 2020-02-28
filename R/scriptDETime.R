#R
library("ImpulseDE2")
library('dplyr')
library("splineTimeR")
library("Biobase")
library(mclust)
library(lmms)
library(data.table)
source('functionsDETime.R')

#Variables
dataPathSaved='/home/karin/Documents/timeTrajectories/data/deTime/'
dataPath='/home/karin/Documents/timeTrajectories/data/'

#load data
data<-loadDataOrangeTable(dataPath)
genesNotNull<-data$gNN
genesTNN<-data$gTNN
conditions<-data$condit
#*****
#Genes DE in time, impulseDE2, together for replicates of each strain:
strain='comH'
filterImpulse <- function(strain,padj=0.05,threads=4,subsetGenes=nrow(genesTNN)) {
  genesSample<-genesTNN[1:subsetGenes,conditions$Strain==strain & conditions$Include]
  conditionsSub<-conditions[conditions$Strain==strain & conditions$Include,]
  anno<-data.frame(Sample=conditionsSub$Measurment, Condition=rep("case",ncol(genesSample)), Time=conditionsSub$Time, Batch=rep("B_NULL",ncol(genesSample)))
  #Remove this rounding  - should be done on counts!!!!
  de <- runImpulseDE2(matCountData=round(genesSample),dfAnnotation=anno,boolCaseCtrl= FALSE,vecConfounders= NULL,scaNProc= threads )
  deorder<-de$dfImpulseDE2Results[order(de$dfImpulseDE2Results$padj),]
  deImpulse<-deorder[deorder$padj<=padj,]
  return(deImpulse)
}
#Genes DE in time between strains, SplineTimeR -> No because little enough time points when compare strains, however impulseDE2 accepts different time points?

#Genes DE in between strains, impulseDE2, together for replicates of each strain:
strain1='AX4'
strain2='tagB'
#g1<-genesTNN[,conditions$Strain==strain1 & conditions$Include]
#g2<-genesTNN[,conditions$Strain==strain2 & conditions$Include]
#genesSub<-cbind(g1,g2)
genesSub<-genesTNN[,(conditions$Strain==strain1|conditions$Strain==strain2) & conditions$Include]
#c1<-conditions[conditions$Strain==strain1 & conditions$Include,]
#c2<-conditions[conditions$Strain==strain2 & conditions$Include,]
#conditionsSub<-rbind(c1,c2)
conditionsSub<-conditions[(conditions$Strain==strain1|conditions$Strain==strain2) & conditions$Include,]
#From before the samples in genes and info should be in same order
case<-c()
batch<-c()
for(s in conditionsSub$Strain){
  if (s==strain1){
    case<-c(case,c("control"))
    batch<-c(batch,c("B1_NULL"))
  }
  else {
    case<-c(case,c("case"))
    batch<-c(batch,c("B2_NULL"))
  }
}

anno<-data.frame(Sample=conditionsSub$Measurment, Condition=case,Time=conditionsSub$Time,Batch=batch)
de <- runImpulseDE2(matCountData=round(genesSub),dfAnnotation=anno,boolCaseCtrl= TRUE,vecConfounders= NULL,boolIdentifyTransients = TRUE,scaNProc= 1 )

#******
  #lmms
  #Filter
  strain<-'comH'
filterLmms<- function(strain,useLog=TRUE,doPlot=FALSE,subsetGenes=ncol(genesNotNull)){
  genesSub<-genesNotNull[conditions$Strain==strain & conditions$Include,1:subsetGenes]
  genesSNN<-data.matrix(as.data.frame(genesSub) %>% select_if(not_all_na))
  #Add pseudocount before log2
  if (useLog) genesLog2<-log2(genesSNN+1)
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

#NOT OK - impulse need count data !!!!!
#Compare impulse and lmms filtering
strains=unique(conditions[conditions$Include,]$Strain)
title='strain500\timpulse0.05\tlmms\tlmmsLog2\timpulse_lmms\timpulse_lmmsLog\tlmms_lmmsLog'
write(title,file=paste(dataPathSaved,'CompareImpulseLmms.tsv',sep=''))
for (strain in strains){
  lmmsL<-c()
  impulseL<-c()
  lmmsLogL<-c()
  impulselmms<-c()
  impulselmmsLog<-c()
  lmmslmmsLog<-c()
  subset=500
  for(i in seq(1,3)){
    lmmsR<-filterLmms(strain,useLog=FALSE,subsetGenes=subset)
    lmmsL<-append(lmmsL,dim(lmmsR)[2])
    lmmsLogR<-filterLmms(strain,useLog=TRUE,subsetGenes=subset)
    lmmsLogL<-append(lmmsLogL,dim(lmmsLogR)[2])
    impulseR<-filterImpulse(strain,padj=0.05,subsetGenes=subset)
    impulseL<-append(impulseL,dim(impulseR)[1])
    lmmsG<-colnames(lmmsR)
    lmmsLogG<-colnames(lmmsLogR)
    impulseG<-rownames(impulseR)
    impulselmms<-append(impulselmms,length(intersect(impulseG,lmmsG)))
    impulselmmsLog<-append(impulselmmsLog,length(intersect(impulseG,lmmsLogG)))
    lmmslmmsLog<-append(lmmslmmsLog,length(intersect(lmmsG,lmmsLogG)))
  }
  lmmsAvg<-mean(lmmsL)
  lmmsLogAvg<-mean(lmmsLogL)
  impulseAvg<-mean(impulseL)
  impulselmmsAvg<-mean(impulselmms)
  impulselmmsLogAvg<-mean(impulselmmsLog)
  lmmslmmsLogAvg<-mean(lmmslmmsLog)
  line=paste(strain,impulseAvg,lmmsAvg,lmmsLogAvg,impulselmmsAvg,impulselmmsLogAvg,lmmslmmsLogAvg,sep='\t')
  write(line,file=paste(dataPathSaved,'CompareImpulseLmms.tsv',sep=''),append=TRUE)
}


#Extract DE genes for each strain - !!NOT FINISHED
strains=unique(conditions[conditions$Include,]$Strain)
for (strain in strains){
  lmmsL<-list()
  reps=3
  for(i in seq(1,reps)){
    lmmsR<-filterLmms(strain,useLog=FALSE,subsetGenes=subset)
    lmmsL[[i]]=rownames(lmmsR)
  }
  unionG<-lmmsL[[1]]
  print(length(lmmsL[[1]]))
  for (i in seq(2,reps)){
    unionG<-union(unionG,lmmsL[[i]])
    print(length(lmmsL[[i]]))
    print(length(unionG))
  }
  #TO-DO
  write(line,file=paste(dataPathSaved,'CompareImpulseLmms.tsv',sep=''),append=TRUE)
}



#Lmms DE groups - NOT FINISHED!!!!
#Do on log data
sample1='comH'
sample2='tgrB1'
basisF='p-spline'
genesSub<-genesTNN[(conditions$Strain==strain1|conditions$Strain==strain2) & conditions$Include,]
conditionsSub<-conditions[(conditions$Strain==strain1|conditions$Strain==strain2) & conditions$Include,]
case<-c()
for(s in conditionsSub$Strain){
  if (s==strain1){
    case<-c(case,c("g1"))
  }
  else {
    case<-c(case,c("g2"))
  }
}

lmmsDEtest <-lmmsDE(data=genesSub,time=conditionsSub$Time, sampleID=conditionsSub$Sample, group=case,type='time*group', experiment='all',basis=basisF)
summary(lmmsDEtest)






















