library("DESeq2")
library('dplyr')
source('functionsDETime.R')
library(ggplot2)
library(RColorBrewer)
library("ImpulseDE2")
library(mclust)
library(lmms)
library("BiocParallel")
register(MulticoreParam(4))

#************Calculate DE of mutant vs AX4 for each time point

#Works on count data
#genesP -data frame with expression data with sample descriptions as colnames and genes as rownames; counts
#conditionsP - data frame with sample descriptions as rownames and necesary collumns Strain, Time 
#dataPathSaved - where the data will be saved
#samples - which mutants are present at each time point, list with timepoint names and mutants as elements of each list element
#Example: $`0` [1] "comH"    "tgrB1"   "tgrC1"   "gbfA"    "tgrB1C1" "tagB"   $`2` [1] "tagB" "comH"

#Make DeSeq2 data object for specific time point
#If error in samplenames matching return NaN
buildDDS<-function(t){
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
    return(NaN)
  }
}

#Get results from DeSeq2 data object already analysed with DESeq of sample1 against AX4 (at timepoint of dds) - get specific contrast
#It assumes that sample1 will be compared against AX4, where the Strain name is 'AX4_t', t being time point
testDE<-function(dds,sample1,padjSave,logFCSave,t){
  res <- results(dds,contrast=c("condition", paste(sample1,'_',t,sep=''), paste("AX4_",t,sep='')),parallel = TRUE)
  resNN <- na.omit(res)
  resOrder<-resNN[order(resNN$padj),]
  #plot(resNN$log2FoldChange,-log10(resNN$padj))
  resFilter<-resOrder[resOrder$padj<=padjSave & abs(resOrder$log2FoldChange)>=logFCSave,]
  #R reads this format fine, but Libre offic shifts colnames!!!
  write.table(x = resFilter,file =paste( dataPathSaved,'DE_',sample1,'_t',t,'.tsv',sep=''),sep='\t')
}

#Make the comparisons in all timepoints
#Compare each mutant from vector mutantsT (mutants at timepoint t form samples) to AX4 as specified in testDE
#Saves all DE genes with padj max 0.1 and abs(logFC) min 0.5
#Timepoints is vector of timepoints 
for(t in timepoints){
  print(paste('Building for',t))
  dds=buildDDS(t)
  dds <- DESeq(dds,parallel = TRUE)
  #Get contrasts
  mutantsT=samples[[as.character(t)]]
  for (mutant in mutantsT){
    print(paste('Analysing',sample1))
    testDE(dds,mutant,padjSave=0.1,logFCSave=0.5,t=t)
  }
}

#Make a data table of DE genes for each strain in each time point, keep only padj<=0.05 and abs(logFC)>=1
#Matrix to specify which genes are De in which sample grouop, 0 if not DE under the condition, in matrix are log2FC
#resultsin df is deDF
first=TRUE
deDF=NaN
for (t in timepoints){
  for (mutant in samples[[as.character(t)]]){
    sampleName=paste(mutant,'_',t,sep='')
    print(sampleName)
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


#How many shared genes compared to all DE genes in each strain for each strain pair
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

#****************DE in time, for timeseries with >8 points

#Works on normalised data
#conditionsNorm - data frame with necesary collumns:
#Sample (eg. one sample measured through time), Strain, Time, Measurment (unique name for each time point of Sample, same as genesNotNullNorm rownames)
#and Include (should Sample be included, boolean)
#genesNotNullNorm - expression data, genes in columns, samples descriptions in rows; normalised 

#Perform lmms DE of genes from single strain in time
#Removes genes that have 0 in all time points
#subsetGenes - for testing, should only frist n rows be used
#Plot decision of which genes are DE
#useLog - log transform data
#Returns data of DE genes, if log was used returns log
not_all_na <- function(x) any(x!=0)
filterLmms<- function(genes,conditions,strain,useLog=TRUE,doPlot=FALSE,subsetGenes=NaN){
  if (is.nan(subsetGenes)){
    subsetGenes=ncol(genes)
  }
  genesSub<-genes[conditions$Strain==strain & conditions$Include,1:subsetGenes]
  genesSNN<-data.matrix(as.data.frame(genesSub) %>% select_if(not_all_na))
  #Add pseudocount before log2, small beseuse already normalised (not counts)
  if (useLog) genesLog2<-log2(genesSNN+10^-100)
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
#Repeat 5 times and save union of DE (prints its length at the end)
strains=unique(conditionsNorm[conditionsNorm$Include,]$Strain)
for (strain in strains){
  print(strain)
  lmmsL<-list()
  reps=5
  for(i in seq(1,reps)){
    lmmsR<-filterLmms(genes=genesNotNullNorm,conditions=conditionsNorm,strain=strain)
    lmmsL[[i]]=colnames(lmmsR)
  }
  unionG<-lmmsL[[1]]
  for (i in seq(2,reps)){
    unionG<-union(unionG,lmmsL[[i]])
  }
  print(length(unionG))
  write(unionG,file=paste(dataPathSaved,'DEinTime_',strain,'.txt',sep=''),append=TRUE)
}


