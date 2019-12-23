#R code for example image
library('ggplot2')
library('dplyr')
library(directlabels)
not_all_na <- function(x) any(x!=0)
pca_path='/home/karin/Documents/timeTrajectories/data/replicate_image/'

#******OLD
file<-read.csv("trans_erasure_7strains_for_orange.tab", header=TRUE, sep="\t")
#First 3 lines headers, in R only the first one, so two lines at the beginning unused
#DDB_G0304961
#continuous
#Entrez\ ID=8627219 ddb_g=DDB_G0304961
#collumns:[1,12735] genes, 12736 Feature name string meta AX4_00h_avr7, 12737 Source ID discrete class AX4, 12738 Time c meta 0
#Transfrom without unneded lines
file<-file[3:nrow(file),]
fileM<-matrix( unlist(file), ncol=length(file) )
genes<-fileM[,1:12735]
class(genes) <- "numeric"
colnames(genes)<-colnames(file[,1:12735])
genesNotNull<-as.data.frame(genes) %>% select_if(not_all_na)
#PCA
pca<-prcomp(genesNotNull,scale=TRUE)
#Make DF for plot
pcaFeats<-cbind.data.frame(as.data.frame(pca$x),file[,12737])
pcaFeats<-cbind.data.frame(pcaFeats, as.numeric(as.character(file[,12738])))
names<-c(colnames(pca$x),"cell","time")
colnames(pcaFeats)<-names
#Plot
ggplot(pcaFeats, aes(x=PC1, y=time, group=cell)) + geom_path(aes(color=cell))+geom_point(aes(color=cell))+scale_color_manual(values=c("blue", "red", "green", "orange", "violet", "brown", "pink")) + geom_text(aes(label=time),hjust=-0.5, vjust=0)
#Scree plot
pvar<-pca$sdev^2
pve<-pvar/sum(pvar)
plot(pve)
#First 10 PCA vs time
for (p in 1:10){
ggplot(pcaFeats, aes(x=pcaFeats[,p], y=time, group=cell)) + geom_path(aes(color=cell))+geom_point(aes(color=cell))+scale_color_manual(values=c("blue", "red", "green", "orange", "violet", "brown", "pink")) + geom_text(aes(label=time),hjust=-0.5, vjust=0)+xlab(paste("PC",p,sep=""))
ggsave(paste("PC",p,"vsTime.png",sep=""))
}

#******************************************************************
#New genes
genesNormAvg<-read.table('/home/karin/Documents/timeTrajectories/Orange_workflows/regulons/genes_averaged_orange.tsv', header=TRUE, sep="\t")
genesM<-genesNormAvg[,3:ncol(genesNormAvg)]
genesNotNull<-as.data.frame(genesM) %>% select_if(not_all_na)
#PCA
pca<-prcomp(genesNotNull,scale=TRUE)
saveRDS(pca, file = paste(pca_path,'pca.rds',sep=''))
write.table(pca$rotation,paste(pca_path,'loadings.tsv',sep=''),sep='\t',col.names=NA)
#write.table(pca$x,paste(pca_path,'positions.tsv',sep=''),sep='\t',col.names=NA)
#write.table(pca$sdev,paste(pca_path,'sdev.tsv',sep=''),sep='\t',col.names=NA)

#Make DF for plot
pcaFeats<-cbind.data.frame(as.data.frame(pca$x),genesNormAvg$Group)
pcaFeats<-cbind.data.frame(pcaFeats, as.numeric(genesNormAvg$Time))
names<-c(colnames(pca$x),"Strain","Time")
colnames(pcaFeats)<-names
#pcaFeats<-pcaFeats[order(pcaFeats$Strain,pcaFeats$Time),]
#Plot
pl<-ggplot(pcaFeats, aes(x=Time, y=PC1, group=Strain,colour=Strain)) + geom_path()+geom_point()
#geom_dl(aes(label=Strain,color=Strain),method="last.bumpup")
direct.label(pl, list("last.bumpup", cex=0.8))


#+geom_dl(aes(label = Strain), method = list(dl.trans(x = x + 0.2), "last.points", cex = 0.8))
#direct.label(pl,"last.qp")
# geom_text(aes(label=Time),hjust=-0.5, vjust=0)+
# + scale_color_manual(values=c("blue", "red", "green", "orange", "violet", "brown", "pink"))
#Scree plot
pvar<-pca$sdev^2
pve<-pvar/sum(pvar)
plot(pve,ylab = 'Proportion of explained variance',xlab='PCA component')
#First 10 PCA vs time
for (p in 1:10){
ggplot(pcaFeats, aes(x=pcaFeats[,p], y=time, group=cell)) + geom_path(aes(color=cell))+geom_point(aes(color=cell))+scale_color_manual(values=c("blue", "red", "green", "orange", "violet", "brown", "pink")) + geom_text(aes(label=time),hjust=-0.5, vjust=0)+xlab(paste("PC",p,sep=""))
ggsave(paste("PC",p,"vsTime.png",sep=""))
}

#Distribution of loadings across genes for PC1 - are there some genes with very high importance
loadings1<-pca$rotation[,'PC1']
boxplot(loadings1)
hist(loadings1,breaks=100)
