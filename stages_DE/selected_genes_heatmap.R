dataPathCode='/home/karin/Documents/git/baylor-dicty/R/'

library(ComplexHeatmap)
library(circlize)
library(plyr)
library(viridis)
library(proxy)
library(seriation)
library(dendextend)
source(paste(dataPathCode,'heatmap_annotation.R',sep=''))

path_expression='/home/karin/Documents/timeTrajectories/data/regulons/'
path_comparisons='/home/karin/Documents/timeTrajectories/data/regulons/selected_genes/'

avg_expression=read.table(paste(path_expression,"genes_averaged_orange_scale99percentileMax0.1.tsv",sep=''),
                          header=TRUE,row.names=1, sep="\t")

# Expression range for legend
expressions<-within(avg_expression, rm('Time', 'Strain','Group'))
min_expression<-min(expressions)
max_expression<-max(expressions)
#*******************************************************
#** OLD selected abberant genes heatmaps
#**
comparisons=read.table(paste(path_comparisons,"summary_comparisonsAvgSimsSingle_AX4basedNeigh_u-less_newGenes_noAll-removeZeroRep_simsDict_scalemean0std1_logTrue_kN11_splitStrain.tsv",
                             sep=''), header=TRUE,row.names=1, sep="\t")
comparisons=as.matrix(data.frame(lapply(comparisons,as.character),row.names = row.names(comparisons)))
comparisons<-mapvalues(comparisons, from = c("1", "0"), to = c("Difference", "Not difference"))

path_strain_order='/home/karin/Documents/timeTrajectories/data/'
strain_order<-as.vector(read.table(paste(path_strain_order,"strain_order.tsv",sep=''))[,1])

legend_font=12
legened_height=1.5
legend_width=0.7
top_annotation_height=0.6

#Strain groups annotation
#** Colours of strain groups
group_cols=c('1Ag-'= '#d40808', '2LAg'= '#e68209', '3TA'='#d1b30a', '4CD'= '#4eb314', '5WT'= '#0fa3ab',
             '6SFB'= '#525252', '7PD'='#7010b0' )

#Time annotation
times=unique(avg_expression$Time)
#** Time colours
col_time = colorRamp2( c(min(times),max(times)),c( "white", "#440154FF"))

ha = HeatmapAnnotation(
  Group = avg_expression$Group, 
  Time = avg_expression$Time,
  col = list(Group = group_cols,
             Time = col_time),
  annotation_legend_param = list(
    Group = list(grid_width= unit(legend_width, "cm"),grid_height= unit(legened_height, "cm") ,
                 labels_gp = gpar(fontsize = legend_font),title_gp = gpar(fontsize = legend_font)
    ),
    Time = list(at = c(min(times),as.integer(mean(c(min(times),max(times)))),max(times)),
                grid_width= unit(legend_width, "cm"),grid_height= unit(legened_height, "cm") ,
                labels_gp = gpar(fontsize = legend_font),title_gp = gpar(fontsize = legend_font)
    )
  )
)

#Expression heatmap
expressions<-within(avg_expression, rm('Time', 'Strain','Group'))
min_expression<-min(expressions)
max_expression<-max(expressions)
#** Expression colours
#col = colorRamp2(c(min_expression,mean(c(min_expression,max_expression)),max_expression), c( "#440154FF", "#1F968BFF",'#FDE725FF'))

genes=as.character(row.names(comparisons))
heatmap=Heatmap(t(avg_expression[,genes]),cluster_columns = FALSE,show_column_names = FALSE,
                column_title_gp=gpar(fontsize=12),
                column_split=factor(avg_expression$Strain,levels=strain_order),
                  show_row_names = FALSE, col=viridis(256),
                  show_heatmap_legend = TRUE,heatmap_legend_param = list(
                    title = "Relative expression",
                    at = c(min_expression, round(mean(c(min_expression,max_expression)),1),max_expression),
                    grid_width= unit(legend_width, "cm"),grid_height= unit(legened_height, "cm") ,
                    labels_gp = gpar(fontsize = legend_font),title_gp = gpar(fontsize = legend_font)),top_annotation=ha
               )
heatmap_diff=Heatmap(comparisons,width = unit(4, "cm"),cluster_columns = FALSE,show_row_names=FALSE,
                     col=c('Difference'='darkred','Not difference'='white'),name='Comparison',
                     heatmap_legend_param = list(
                       title = "Comparison",grid_width= unit(legend_width, "cm"),grid_height= unit(legened_height, "cm") ,
                     labels_gp = gpar(fontsize = legend_font),title_gp = gpar(fontsize = legend_font), border = "grey"), border = "grey")

#Plots the combined heatmap 
heatmap+heatmap_diff

#********* Heatmap of avg similarity per strain group
similarities=read.table(paste(path_comparisons,"simsQuantileNormalised_AX4basedNeigh_newGenes_noAll-removeZeroRep_simsDict_scalemean0std1_logTrue_kN11_splitStrain.tsv",
                             sep=''), header=TRUE,row.names=1, sep="\t")

mapping=unique(avg_expression[,c('Strain','Group')])
groups<-mapvalues(colnames(similarities), from = as.vector(mapping$Strain), to = as.vector(mapping$Group))
  
ha = HeatmapAnnotation(
  Group = groups, 
  col = list(Group = group_cols),
  annotation_legend_param = list(
    Group = list(grid_width= unit(legend_width, "cm"),grid_height= unit(legened_height, "cm") ,
                 labels_gp = gpar(fontsize = legend_font),title_gp = gpar(fontsize = legend_font)
    )
))
similarities_plot=as.matrix(similarities[genes,])
min_sims=round(min(similarities_plot),1)
max_sims=round(max(similarities_plot),1)
heatmap=Heatmap(similarities_plot,cluster_columns = FALSE,show_column_names = FALSE,
                column_title_gp=gpar(fontsize=12),
                column_split=factor(colnames(similarities),levels=strain_order),
                show_row_names = FALSE, col=magma(256),
                show_heatmap_legend = TRUE,heatmap_legend_param = list(
                  title = "Average similarity",
                  at = c(min_sims, round(mean(c(min_sims,max_sims)),1),max_sims),
                  grid_width= unit(legend_width, "cm"),grid_height= unit(legened_height, "cm") ,
                  labels_gp = gpar(fontsize = legend_font),title_gp = gpar(fontsize = legend_font)),top_annotation=ha
)
heatmap+heatmap_diff

#*****************************
#NEW heatmaps, gene assigned to single comparison/strain group split

comparisons<-read.table(paste(path_comparisons,
                              "comparisonsAvgSimsSingle2STDAny0.2_lessComparisons2_AX4basedNeigh_u-less_newGenes_noAll-removeZeroRep_simsDict_scalemean0std1_logTrue_kN11_splitStrain.tsv",
                              sep=''), header=TRUE,row.names=1, sep="\t")
#Filter comparisons on general terms (FDR, diff between group means)
comparisons<-comparisons[comparisons$FDR<=0.05 & comparisons$Difference.mean>=0.3,]
comparisons_order<-c('agg-','dis','tag','cud')
#*** Expression heatmap


#Peak data AX4
expression_patterns=read.table(paste(path_expression,"gene_patterns_orange.tsv",sep=''),
                               header=TRUE,row.names=1, sep="\t")
#Expression type - up or down during development - when peak
for (gene in row.names(comparisons)){
  pattern_t<-expression_patterns[gene,'Peak']
  type=NULL
  if(pattern_t>5){
    type='up'
  }else{
    type='down'
  }
  comparisons[gene,'Type']=type
}

type_col=c('up'='#ed1c24','down'='#00b2ff')

ht_list<-make_annotation()
first=TRUE
for (comparison in comparisons_order){
  comparisons_sub<-comparisons[comparisons$Comparison==comparison,]
  genes=row.names(comparisons_sub)
  #rowblock_title=paste(comparison,'N', length(genes))
  rowblock_title=comparison
  print(paste(comparison,length(genes)))
  
  
  #Order genes visually
  AX4_ordered<-c()
    #print(paste(cluster,length(genes)))
  if (length(genes)>1){
    expression=t(avg_expression[avg_expression$Strain=='AX4',genes])
    distances<-dist(expression, method="cosine")
    hc<-hclust(d=distances, method = "ward.D2" )
    hc_ordered<-reorder(x=hc,dist = distances)
    genes<- as.dendrogram(hc_ordered) %>% labels
  }
  
  #pattern_type_anno=rowAnnotation(Type =comparisons_sub[genes,]$Type,col=list(Type = type_col),
  #                                show_legend = first,annotation_name_side = "top",show_annotation_name = first,
   #                               annotation_name_gp=gpar(fontsize = cluster_font))
  
  heatmap=Heatmap(t(avg_expression[,genes]),cluster_columns = FALSE,cluster_rows =FALSE,show_column_names = FALSE,
                  show_row_names = FALSE, col=viridis(256),column_title=NULL, 
                  row_title=rowblock_title,
                  show_heatmap_legend = first,heatmap_legend_param = list(
                    title = "\nRelative \nexpression\n",
                    at = c(min_expression, round(mean(c(min_expression,max_expression)),1),max_expression),
                    grid_width= unit(legend_width, "cm"),grid_height= unit(legened_height, "cm") ,
                    labels_gp = gpar(fontsize = cluster_font),title_gp = gpar(fontsize = cluster_font)),
                  #** Cluster name fontsize
                  row_title_gp=gpar(fontsize=cluster_font)
                  # ,left_annotation=pattern_type_anno
                  ,border = TRUE
                )
  first=FALSE
  ht_list=ht_list %v% heatmap
}
#Save dim 35*20
#******* Similarities heatmap
similarities<-read.table(paste(path_comparisons,
                              "simsQuantileNormalised_AX4basedNeigh_newGenes_noAll-removeZeroRep_simsDict_scalemean0std1_logTrue_kN11_splitStrain.tsv",
                              sep=''), header=TRUE,row.names=1, sep="\t")
#Reorder strains
similarities<-similarities[,strain_order]

#Scale similarities by row
similarities<-t(apply(similarities, 1, function(x)(x-min(x))/(max(x)-min(x))))

strain_cols=c()
groups<-c()
for (strain in colnames(similarities)){
  group=as.character(avg_expression[avg_expression$Strain==strain,'Group'][1])
  col<- as.character(group_cols[group])
  strain_cols<-append(strain_cols,col)
  if (grepl('dis', group, fixed = TRUE)) group<-'dis'
  groups<-append(groups,group)
}

group_order<-c()
for (strain in strain_order){
  group=as.character(avg_expression[avg_expression$Strain==strain,'Group'][1])
  if (grepl('dis', group, fixed = TRUE)) group<-'dis'
  if(!group %in% group_order){
    group_order<-append(group_order,group)
  }
}

ht_list=NULL
first=TRUE
for (comparison in comparisons_order){
  comparisons_sub<-comparisons[comparisons$Comparison==comparison,]
  genes=row.names(comparisons_sub)
  rowblock_title=comparison
  print(paste(comparison,length(genes)))

  heatmap=Heatmap(similarities[genes,],cluster_columns = FALSE,cluster_rows =TRUE,show_column_names = first,
                  show_row_names = FALSE, col=inferno(256),column_title=NULL, 
                  row_title=rowblock_title, column_split=factor(groups,levels=group_order),
                  show_heatmap_legend = first,heatmap_legend_param = list(
                    title = "\nRelative \nsimilarities\n",
                    at = c(0,0.5,1),
                    grid_width= unit(legend_width, "cm"),grid_height= unit(legened_height, "cm") ,
                    labels_gp = gpar(fontsize = cluster_font),title_gp = gpar(fontsize = cluster_font)),
                  #** Row block name fontsize
                  row_title_gp=gpar(fontsize=cluster_font),
                  column_names_gp = gpar(fontsize = cluster_font, col=strain_cols),
                  column_names_side = 'top'
  )
  first=FALSE
  ht_list=ht_list %v% heatmap
}

