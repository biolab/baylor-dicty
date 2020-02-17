lab=TRUE
if (lab){
  path_clusters='/home/karin/Documents/timeTrajectories/data/regulons/by_strain/kN300_mean0std1_log/'
  path_regulons='/home/karin/Documents/timeTrajectories/data/regulons/'
}else {
}

library(RColorBrewer)
library(ComplexHeatmap)
library(circlize)

avg_expression=read.table(paste(path_regulons,"genes_averaged_orange_scale99percentileMax0.1.tsv",sep=''), header=TRUE,row.names=1, sep="\t")
regulons=read.table(paste(path_clusters,"mergedGenes_min18_clusters_larger.tab",sep=''), header=TRUE,row.names=1, sep="\t")

# Get clusters
clusters=unique(regulons$Cluster)
vals <- as.numeric(gsub("C","", clusters))
clusters=clusters[order(vals)]

group_cols=c('1Ag-'= '#d40808', '2LAg'= '#e68209', '3TA'='#d1b30a', '4CD'= '#4eb314', '5WT'= '#0fa3ab',
                        '6SFB'= '#525252', '7PD'='#7010b0' )
ht_list=Heatmap(t(avg_expression['Group']),show_column_names = FALSE,
                column_split=factor(avg_expression$Strain,levels=unique(avg_expression$Strain)),
                cluster_columns=FALSE,name='Group',
                column_title_gp=gpar(fontsize=9),col=group_cols)
times=unique(avg_expression$Time)
col_time = colorRamp2( c(min(times),max(times)),c( "white", "blue"))
ht_time=Heatmap(t(avg_expression['Time']),
                cluster_columns=FALSE, show_row_names = FALSE,name='Time',col=col_time)
ht_list=ht_list %v% ht_time
col = colorRamp2(c(-1,0.1), c( "yellow", "blue"))
first=TRUE
for (cluster in clusters){
  print(cluster)
  genes=rownames(regulons)[regulons$Cluster==cluster]
  heatmap=Heatmap(t(avg_expression[,genes]),cluster_columns = FALSE,show_column_names = FALSE,
                  show_row_names = FALSE, col=col,column_title=NULL, row_title=gsub('C','',cluster),
                show_heatmap_legend = first,heatmap_legend_param = list(
                  title = "Relative expression", at = c(-1, -0.5,0.1)),
                  row_title_gp=gpar(fontsize=8))
  first=FALSE
  ht_list=ht_list %v% heatmap
}
ht_list