library('dplyr')

loadDataOrangeTable<-function(dataPath){
  genesTo=12869
file<-read.csv(paste(dataPath,"trans_9repAX4_6strains_2rep_avr_T12.tab",sep=''), header=TRUE, sep="\t")
file<-file[3:nrow(file),]
fileM<-matrix( unlist(file), ncol=length(file) )
genes<-fileM[,1:genesTo]
class(genes) <- "numeric"
colnames(genes)<-colnames(file[,1:genesTo])
not_all_na <- function(x) any(x!=0)
genesNotNull<-as.data.frame(genes) %>% select_if(not_all_na)
genesTNN<-t(genesNotNull)
Samples<-file[,12870]
Strains<-c()
Include<-c()
for (sample in Samples){
  if (grepl("_avr",sample, fixed=TRUE)){
    Strains<-append(Strains, sample)
    Include<-append(Include,FALSE)
  }
  else{
    strain<-strsplit(sample,'_')[[1]][1]
    Strains<-append(Strains, strain)
    if(sample %in% c('AX4_r3','AX4_r4')){
      Include<-append(Include,FALSE)
    }
    else{
      Include<-append(Include,TRUE)
    }
  }
}
Time<-as.numeric(as.character(file[,12871]))
Measurments<-file[,12872]
#Make DFs for DE analysis
conditions<-data.frame(Sample=Samples, Strain=Strains, Time=Time,Measurment=Measurments, Include=Include)
colnames(genesTNN)<-conditions$Measurment
rownames(genesNotNull)<-conditions$Measurment
return(list(gNN=genesNotNull,gTNN=genesTNN,condit=conditions,genes=genes))
}

