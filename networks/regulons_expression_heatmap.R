#*** Draws expression of multiple gene clusters across strains
#* Comments starting with '**' denote changes that can be easily made to the heatmap. 
#* Comments starting with '**!' denote required changes (e.g. file names, ...).
#* For Regulons_by_strain heatmap the images can be saved with resolution 2500w*1500h (specified in R studio Export -> 
#* Save as image). This resolution ensures that the text on the heatmap does not overlap anymore. 
dataPathCode='/home/karin/Documents/git/baylor-dicty/R/'

library(ComplexHeatmap)
library(circlize)
library(viridis)
library(proxy)
#library(cba)
library(seriation)
library(dendextend)
source(paste(dataPathCode,'heatmap_annotation.R',sep=''))

#**! Paths where expression data (average expression, expression patterns, expression height), strain order,
#** regulons clusters, and phenotipic data are saved
path_clusters='/home/karin/Documents/timeTrajectories/data/regulons/by_strain/kN300_mean0std1_log/'
path_expression='/home/karin/Documents/timeTrajectories/data/regulons/'
path_expression_height='/home/karin/Documents/timeTrajectories/data/regulons/by_strain/'
#path_strain_order='/home/karin/Documents/timeTrajectories/data/'
#path_phenotypes = '/home/karin/Documents/timeTrajectories/data/stages/'

#**! Specify file names for regulons and expression
#** Expression tab file: Genes in columns (already scaled), averaged strain data in rows, 
#** three additional comlumns: Time, Strain, and Group (meaning strain group)
avg_expression=read.table(paste(path_expression,"genes_averaged_orange_scale99percentileMax0.1.tsv",sep=''),
                          header=TRUE,row.names=1, sep="\t")
#** Expression patterns file: Expression pattern in AX4. 
#** Requirements: genes in the first column (used as row names); a column Mass_centre or Peak with mass 
#** centre/peak time of expression in AX4 - used for sorting the regulons, instead another column can be 
#** used as specified below in cluster sorting
expression_patterns=read.table(paste(path_expression,"gene_patterns_orange.tsv",sep=''),
                               header=TRUE,row.names=1, sep="\t")
#**! Specify file names for phenotipic data
#** Phenotypes tab file: Short averaged sample names in rows (as in avg_expression) and columns with phenotypes.
#** Phenotypes should have values: yes, no, no data
#avg_phenotype=read.table(paste(path_phenotypes,"averageStages.tsv",sep=''),
#                         header=TRUE,row.names=1, sep="\t", stringsAsFactors=FALSE)
#Change avg_phenotypes data so that each phenotype can be coloured differently
#avg_phenotype[avg_phenotype=='no']=NA
#for(col in colnames(avg_phenotype)){
#  new_col=avg_phenotype[col]
#  new_col[new_col=='yes']=col
#  avg_phenotype[col]=new_col
#}

#**! Data about gene having low/high expression in AX4. 
#** First coulmn gene names, another coulumn named AX4 with True/False denoting if gene has high expression in AX4 relative to other strains.
expression_height=read.table(paste(path_expression_height,"expressedGenes0.990.5.tsv",sep=''),
                               header=TRUE,row.names=1, sep="\t")
expression_height=data.frame(row.names=rownames(expression_height),'AX4'=expression_height$AX4)
# Convert python True/False to R TRUE/FALSE
expression_height<-expression_height=='True'

#** Strain order - single column with ordered strain names
#strain_order<-as.vector(read.table(paste(path_strain_order,"strain_order.tsv",sep=''))[,1])

#** Regulon groups tab file: First column lists genes and 
#** a column named Cluster specifying cluster/regulon of each gene 
regulons=read.table(paste(path_clusters,
                          #"clusters/mergedGenes_minExpressed0.990.1Strains1Min1Max18_clustersAX4Louvain0.4m0s1log.tab"
                          "clusters/mergedGenes_minExpressed0.990.1Strains1Min1Max18_clustersLouvain0.4minmaxNologPCA30kN30.tab"
                          ,sep=''),header=TRUE, sep="\t")
#Name the first column (should contain genes
colnames(regulons)[1]<-'Gene'

#** Regulon2 groups tab file (used for side annotation): First column lists genes and 
#** a column named Cluster specifying cluster/regulon of each gene
regulons2=read.table(paste(path_clusters,"clusters/mergedGenes_minExpressed0.990.1Strains1Min1Max18_clustersAX4Louvain0.4m0s1log.tab",sep=''),
                    header=TRUE, sep="\t")
#Name the first column (should contain genes)
rownames(regulons2)<-regulons2[,1]
regulons2<-regulons2[,'Cluster', drop=F]
  
# Get clusters - list unique and sort
clusters=unique(regulons$Cluster)
vals <- as.numeric(gsub("C","", clusters))
clusters=clusters[order(vals)]
    
#** Some plotting parameters
#legend_font=12
phenotypes_font=10
legened_height=1.5
legend_width=0.7
top_annotation_height=0.6
phenotype_annotation_height=3
cluster_font=15
#Imports made in heatmao_annotation.r
fontfamily='Arial'

ht_list<-make_annotation(phenotypes_font,legend_height,legend_width,top_annotation_height,phenotype_annotation_height,cluster_font)
#Expression heatmap

#Sort clusters based on average peak time in AX4
sort_clusters <- function(regulons,expression_height=expression_height,expression_patterns=expression_patterns,pattern_type='Peak',
                          use_height=FALSE) {
  cluster_patterns_mean<-c()
  cluster_patterns_median<-c()
  #Sort clusters by name (for clusters that have too low AX4 expression)
  #These clusters are all given max time, but added to df based on their name, so they will stay in 
  #name-order after sorting by time 
  clusters<-unique(regulons$Cluster)
  vals <- as.numeric(gsub("C","", clusters))
  clusters<-clusters[order(vals)]
  
  for (cluster in clusters){
    genes=as.character(regulons[regulons$Cluster==cluster,'Gene'])
    # Check how many genes were termed unexpressed
    expressed<-mean(expression_height[genes,])
    if (expressed > 0.5 || !use_height){
    #** Select column from expression_patterns to sort clusters on
      pattern_mean<-mean(expression_patterns[genes,pattern_type])
      pattern_median<-median(expression_patterns[genes,pattern_type])
      cluster_patterns_mean<-c(cluster_patterns_mean,pattern_mean)
      cluster_patterns_median<-c(cluster_patterns_median,pattern_median)
    }else{
      #If manly unexpressed in AX4 put it at the end
      cluster_patterns_mean<-c(cluster_patterns,max(times))
      cluster_patterns_median<-c(cluster_patterns,max(times))
    }
  }
  #Sort
  cluster_order<-data.frame('Cluster'=clusters,'Pattern_mean'=cluster_patterns_mean,'Pattern_median'=cluster_patterns_median)
  cluster_order<-cluster_order[order(cluster_order$Pattern_median,cluster_order$Pattern_mean),]
  return(cluster_order)
}
#UNUSED
#Sort clusters based on peak time in  average of cluster (on averaged scaled data) in AX4
sort_clusters_expression <- function(regulons,expression){
  
  cluster_max<-c()
  
  expression<-expression[expression$Strain=='AX4',]
  times_AX4<-expression$Time
  #Sort by name so that if overlaping pattern they are still sorted
  clusters<-unique(regulons$Cluster)
  vals <- as.numeric(gsub("C","", clusters))
  clusters<-clusters[order(vals)]
  for (cluster in clusters){
    genes=as.character(regulons[regulons$Cluster==cluster,'Gene'])
    expression_cluster=t(expression[,genes])
    mean_expression<-colMeans(expression_cluster)
    max_time<-times_AX4[mean_expression==max(mean_expression)][1]
    cluster_max<-append(cluster_max,max_time)
  }
  cluster_order<-data.frame('Cluster'=clusters,'Max'=cluster_max)
  cluster_order<-cluster_order[order(cluster_order$Max),]
  return(cluster_order)
  
}

cluster_order<-sort_clusters(regulons=regulons,expression_height=expression_height,expression_patterns=expression_patterns,pattern_type='Peak')
regulons2_temp<-data.frame(regulons2)
regulons2_temp$Gene<-row.names(regulons2_temp)
cluster_order2<-sort_clusters(regulons=regulons2_temp,expression_height=expression_height,expression_patterns=expression_patterns,pattern_type='Peak')

# Perform clustering in AX4 within reference regulons, rename reference regulons to letters.
AX4_ordered<-c()
for (cluster in cluster_order2$Cluster){
  genes=row.names(regulons2)[regulons2$Cluster==cluster]
  #print(paste(cluster,length(genes)))
  if (length(genes)>1){
    expression=t(avg_expression[avg_expression$Strain=='AX4',genes])
    #expression=t(avg_expression[,genes])
    distances<-dist(expression, method="cosine")
    hc<-hclust(d=distances, method = "ward.D2" )
    # Not good ordering-Wrong extraction of labels?
    #ordering <- order.optimal(distances, hc$merge)$order
    #ordering<-as.data.frame(ordering)
    #genes<-row.names(ordering)[order(ordering$ordering)]
    #Another reordering library
    hc_ordered<-reorder(x=hc,dist = distances)
    #Wrong extraction of labels?
    #ordering<-data.frame(Gene=hc_ordered$labels,Order=hc_ordered$order)
    #genes<-as.character(ordering[order(ordering$Order),'Gene'])
    genes<- as.dendrogram(hc_ordered) %>% labels
  }
  AX4_ordered<-append(AX4_ordered,genes)
}

# Expression range for legend
expressions<-within(avg_expression, rm('Time', 'Strain','Group'))
min_expression<-min(expressions[,regulons$Gene])
max_expression<-max(expressions[,regulons$Gene])
#** Expression colours - can be used instead of viridis(256) below
#col = colorRamp2(c(min_expression,mean(c(min_expression,max_expression)),max_expression), c( "#440154FF", "#1F968BFF",'#FDE725FF'))

#Expression heatmaps
colours_regulons2=c('#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
                    '#008080', '#000000', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000',  '#000075', '#808080',
                    '#80a2ff','#e6beff')

colours_regulons2_map=colours_regulons2[1:length(unique(regulons2$Cluster))]
#Unique is already ordered as first unique elements
names(colours_regulons2_map)<-unique(regulons2$Cluster)
first=TRUE
for (cluster in cluster_order$Cluster){
  print(cluster)
  
  genes=as.character(regulons[regulons$Cluster==cluster,'Gene'])
  genes<-as.character(genes[order(match(genes,AX4_ordered))])
  regulons2_annotation=rowAnnotation(AX4_clusters = regulons2[genes,],col = list(AX4_clusters = colours_regulons2_map),
                     show_legend = FALSE,annotation_name_side = "top",show_annotation_name = first,
                     annotation_name_gp=gpar(fontsize = cluster_font,fontfamily=fontfamily))
  
  # Remove 'C' from cluster name
  # The as.character ensures that the code works with numeric clusters
  cluster_anno=gsub('C','',as.character(cluster))
  # Rename cluster number to a letter
  #cluster_anno=LETTERS[as.integer(cluster_anno)]
  
  heatmap=Heatmap(t(avg_expression[,genes]),cluster_columns = FALSE,cluster_rows = FALSE,show_column_names = FALSE,
                  show_row_names = FALSE, col=viridis(256),column_title=NULL, 
                  row_title=cluster_anno,
                  show_heatmap_legend = first,heatmap_legend_param = list(
                  title = "\nRelative \nexpression\n",
                  at = c(min_expression, round(mean(c(min_expression,max_expression)),1),max_expression),
                  grid_width= unit(legend_width, "cm"),grid_height= unit(legened_height, "cm") ,
                  labels_gp = gpar(fontsize = cluster_font,fontfamily=fontfamily),title_gp = gpar(fontsize = cluster_font,fontfamily=fontfamily)),
                  #** Cluster name fontsize
                  row_title_gp=gpar(fontsize=cluster_font,fontfamily=fontfamily),
                  left_annotation = regulons2_annotation)
  first=FALSE
  ht_list=ht_list %v% heatmap
}

#Plots the combined heatmap 
ht_list

#***************************
#**** Update regulons numbers anf ile based on order in heatmap
library(purrr)
cluster_map<-c(paste('C',c(1:nrow(cluster_order)),sep=''))
names(cluster_map)<-as.vector(cluster_order$Cluster)
remap_cluster<-function(x){return(cluster_map[[x]])}
regulons['Cluster']<-unlist(map(as.character(regulons$Cluster),remap_cluster))
write.table(regulons,paste(path_clusters,
                          "clusters/mergedGenes_minExpressed0.990.1Strains1Min1Max18_clustersAX4Louvain0.4m0s1log.tab"
                          #"clusters/mergedGenes_minExpressed0.990.1Strains1Min1Max18_clustersLouvain0.4minmaxNologPCA30kN30.tab"
                          ,sep=''),row.names=FALSE, sep="\t")

#*************
#*** Add letters to AX4 clusters
cluster_anno=gsub('C','',as.character(regulons$Cluster))
cluster_anno=LETTERS[as.integer(cluster_anno)]
regulons$Letter<-cluster_anno
write.table(regulons,paste(path_clusters,
                           "clusters/mergedGenes_minExpressed0.990.1Strains1Min1Max18_clustersAX4Louvain0.4m0s1log.tab"
                           ,sep=''),row.names=FALSE, sep="\t")
  