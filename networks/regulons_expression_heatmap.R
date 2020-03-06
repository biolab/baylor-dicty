#*** Draws expression of multiple gene clusters across strains
#* Comments starting with '**' denote changes that can be easily made to the heatmap. 
#* Comments starting with '**!' denote required changes (e.g. file names, ...).
#* For Regulons_by_strain heatmap the images can be saved with resolution 2500w*1500h (specified in R studio Export -> 
#* Save as image). This resolution ensures that the text on the heatmap does not overlap anymore. 

library(ComplexHeatmap)
library(circlize)
library(viridis)

#**! Paths where expression data (average expression, expression patterns, expression height),strain order,
#** and regulons clusters are saved
path_clusters='/home/karin/Documents/timeTrajectories/data/regulons/by_strain/kN300_mean0std1_log/'
path_expression='/home/karin/Documents/timeTrajectories/data/regulons/'
path_expression_height='/home/karin/Documents/timeTrajectories/data/regulons/by_strain/'
path_strain_order='/home/karin/Documents/timeTrajectories/data/'

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

#**! Data about gene having low/high expression in AX4. 
#** First coulmn gene names, another coulumn named AX4 with True/False denoting if gene has high expression in AX4 relative to other strains.
expression_height=read.table(paste(path_expression_height,"expressedGenes0.990.5.tsv",sep=''),
                               header=TRUE,row.names=1, sep="\t")
expression_height=data.frame(row.names=rownames(expression_height),'AX4'=expression_height$AX4)
# Convert python True/False to R TRUE/FALSE
expression_height<-expression_height=='True'

#** Strain order - single column with ordered strain names
strain_order<-as.vector(read.table(paste(path_strain_order,"strain_order.tsv",sep=''))[,1])

#** Regulon groups tab file: First column lists genes and 
#** a column named Cluster specifying cluster/regulon of each gene
regulons=read.table(paste(path_clusters,"clusters/mergedGenes_minExpressed0.990.3Strains1Min1Max18_clustersLouvain0.4minmaxNologPCA30kN30.tab",sep=''),
                    header=TRUE, sep="\t")
#Name the first column (should contain genes
colnames(regulons)[1]<-'Gene'

  
# Get clusters - list unique and sort
clusters=unique(regulons$Cluster)
vals <- as.numeric(gsub("C","", clusters))
clusters=clusters[order(vals)]
    
#** Some plotting parameters
legend_font=12
legened_height=1.5
legend_width=0.7
top_annotation_height=0.6

#Strain groups annotation
#** Colours of strain groups
group_cols=c('1Ag-'= '#d40808', '2LAg'= '#e68209', '3TA'='#d1b30a', '4CD'= '#4eb314', '5WT'= '#0fa3ab',
                        '6SFB'= '#525252', '7PD'='#7010b0' )
ht_list=Heatmap(t(avg_expression['Group']),show_column_names = FALSE, height = unit(top_annotation_height, "cm"),
                column_split=factor(avg_expression$Strain,
                                    #** Ordering of the strains in the heatmap (a vector of strain names)
                                    #levels=unique(avg_expression$Strain)
                                    levels=strain_order
                ),
                cluster_columns=FALSE,name='Group',
                #** Strain name font size
                column_title_gp=gpar(fontsize=12),
                col=group_cols, heatmap_legend_param = list( 
                grid_width= unit(legend_width, "cm"),grid_height= unit(legened_height, "cm") ,
                labels_gp = gpar(fontsize = legend_font),title_gp = gpar(fontsize = legend_font)))

#Time annotation
times=unique(avg_expression$Time)
#** Time colours
col_time = colorRamp2( c(min(times),max(times)),c( "white", "#440154FF"))
ht_time=Heatmap(t(avg_expression['Time']), height = unit(top_annotation_height, "cm"),
                cluster_columns=FALSE, show_column_names = FALSE,name='Time',col=col_time,
                heatmap_legend_param = list( at = c(min(times),as.integer(mean(c(min(times),max(times)))),max(times)),
                grid_width= unit(legend_width, "cm"),grid_height= unit(legened_height, "cm") ,
                labels_gp = gpar(fontsize = legend_font),title_gp = gpar(fontsize = legend_font)))
ht_list=ht_list %v% ht_time

#Expression heatmap

#Sort clusters based on average peak time in AX4
cluster_patterns<-c()
for (cluster in clusters){
  genes=as.character(regulons[regulons$Cluster==cluster,'Gene'])
  # Check how many genes were termed unexpressed
  expressed<-mean(expression_height[genes,])
  if (expressed > 0.5){
  #** Select column from expression_patterns to sort clusters on
    pattern<-mean(expression_patterns[genes,'Peak'])
    cluster_patterns<-c(cluster_patterns,pattern)
  }else{
    #If manly unexpressed in AX4 put it at the end
    cluster_patterns<-c(cluster_patterns,max(times))
  }
}
cluster_order<-data.frame('Cluster'=clusters,'Pattern'=cluster_patterns)
cluster_order<-cluster_order[order(cluster_order$Pattern),]

#Expression range for legend
expressions<-within(avg_expression, rm('Time', 'Strain','Group'))
min_expression<-min(expressions[,regulons$Gene])
max_expression<-max(expressions[,regulons$Gene])
#** Expression colours - can be used instead of viridis(256) below
#col = colorRamp2(c(min_expression,mean(c(min_expression,max_expression)),max_expression), c( "#440154FF", "#1F968BFF",'#FDE725FF'))

#Expression heatmaps
first=TRUE
for (cluster in cluster_order$Cluster){
  print(cluster)
  genes=as.character(regulons[regulons$Cluster==cluster,'Gene'])
  heatmap=Heatmap(t(avg_expression[,genes]),cluster_columns = FALSE,show_column_names = FALSE,
                  show_row_names = FALSE, col=viridis(256),column_title=NULL, 
                  #The as.character ensures that the code works with numeric clusters
                  row_title=gsub('C','',as.character(cluster)),
                  show_heatmap_legend = first,heatmap_legend_param = list(
                  title = "Relative expression",
                  at = c(min_expression, round(mean(c(min_expression,max_expression)),1),max_expression),
                  grid_width= unit(legend_width, "cm"),grid_height= unit(legened_height, "cm") ,
                  labels_gp = gpar(fontsize = legend_font),title_gp = gpar(fontsize = legend_font)),
                  #** Cluster name fontsize
                  row_title_gp=gpar(fontsize=8))
  first=FALSE
  ht_list=ht_list %v% heatmap
}

#Plots the combined heatmap 
ht_list



  