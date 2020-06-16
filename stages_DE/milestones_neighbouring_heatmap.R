dataPathCode='/home/karin/Documents/git/baylor-dicty/R/'

library(ComplexHeatmap)
library(circlize)
library(viridis)
library(proxy)
#library(cba)
library(seriation)
library(dendextend)
source(paste(dataPathCode,'heatmap_annotation.R',sep=''))

path_expression='/home/karin/Documents/timeTrajectories/data/stages/'
pathImpulse='/home/karin/Documents/timeTrajectories/data/stages/DE_across_stages/'
pathDeseq='/home/karin/Documents/timeTrajectories/data/deTime/neighbouring/'

# Load data
avg_expression=read.table(paste(path_expression,"genes_averaged_orange_mainStage_scale99percentileMax0.1.tsv",sep=''),
                          header=TRUE,row.names=1, sep="\t")

data_impulse<-read.table(paste(pathImpulse,'DEacrossStages_summary_mainstage_AX4_0.001.tsv',sep=''),
                    header=TRUE,sep='\t',row.names=1)
data_deseq<-read.table(paste(pathDeseq,'AX4_keepNA/','combined.tsv',sep=''),
                    header=TRUE,sep='\t',row.names=1)
data<-merge(data_deseq,data_impulse,all=TRUE,by="row.names")
row.names(data)<-data$Row.names
data<-data[,colnames(data)!='Row.names']

# Expression scale range
expressions<-within(avg_expression, rm( 'Strain','Group','main_stage'))
min_expression<-min(expressions)
max_expression<-max(expressions)

# List computed comparisons between stages
comparisons<-c()
for (col in colnames(data)){
    if(grepl('_FDR_overall', col,  fixed = TRUE)){
        comparison<-gsub('_FDR_overall','', col,  fixed = TRUE)
        comparisons<-append(comparisons,comparison)
    }
}

optically_order_genes <- function(genes,avg_expression=parent.frame()$avg_expression){
  if (length(genes)>1){
    expression<-t(avg_expression[avg_expression$Strain=='AX4',genes])
    #expression=t(avg_expression[,genes])
    # Looks same as 1-simil(expression, method="cosine")
    distances<-dist(expression, method="cosine")
    hc<-hclust(d=distances, method = "ward.D2" )
    hc_ordered<-reorder(x=hc,dist = distances)
    genes<- as.dendrogram(hc_ordered) %>% labels
  }
  return(genes)
}

# Make lists of genes for each comparison and up/down regulation. Order with optical clustering.
milestone_lists<-list()
for (comparison in comparisons){
  fc_col=paste(comparison,'_log2FoldChange',sep='')
  fdr_col=paste(comparison,'_FDR_overall',sep='')
  data_defined=data[!is.na(data[fc_col]) & !is.na(data[fdr_col]) & !is.na(data[comparison]),]
  genes_up<-rownames(data_defined[data_defined[fc_col]>=2 & data_defined[fdr_col]<=0.01 &
                        data_defined[comparison]==1,])
  genes_down<-rownames(data_defined[data_defined[fc_col] <= -2 & data_defined[fdr_col]<=0.01 &
                        data_defined[comparison]==1,])
  genes_up<-optically_order_genes(genes=genes_up)
  genes_down<-optically_order_genes(genes=genes_down)
  milestone_lists[[comparison]][['up']]<-genes_up
  milestone_lists[[comparison]][['down']]<-genes_down
}

legend_height=1.5
legend_width=0.7
top_annotation_height=0.6
cluster_font=15
#Imports made in heatmao_annotation.R
fontfamily='Arial'
strain_gap=1
group_gap=3.5
gap_units='mm'

ht_list<-make_anno_mainstage(legend_height=legend_height,legend_width=legend_width,
                             top_annotation_height=top_annotation_height,cluster_font=cluster_font,
                             strain_gap=1,group_gap=group_gap,gap_units=gap_units)

first<-TRUE
for (comparison in comparisons){
    genes_up<-milestone_lists[[comparison]][['up']]
    genes_down<-milestone_lists[[comparison]][['down']]
    split<-c(rep('down',length(genes_down)),rep('up',length(genes_up)))
    genes<-c(genes_down,genes_up)

    data_anno<-comparison
    data_anno<-gsub('no_agg','no agg',data_anno)
    data_anno<-gsub('_',' to ',data_anno)
    data_anno<-gsub('no agg','no_agg',data_anno)

    direction_annotation<-rowAnnotation(Direction = split,col = list(Direction = c('down'='#5673e0','up'='#d44e41')),
                     show_legend = first,annotation_name_side = "top",show_annotation_name = first,
                     annotation_name_gp=gpar(fontsize = cluster_font,fontfamily=fontfamily),
                     annotation_legend_param = list(Direction = list(title='\nDirection\n',
                     grid_width= unit(legend_width, "cm"),grid_height= unit(legened_height, "cm") ,
                     labels_gp = gpar(fontsize = cluster_font,fontfamily=fontfamily),
                     title_gp = gpar(fontsize = cluster_font,fontfamily=fontfamily))))

    heatmap<-Heatmap(t(avg_expression[,genes]),cluster_columns = FALSE,cluster_rows = FALSE,
                    show_column_names = FALSE,row_split = split,
                      show_row_names = FALSE, col=viridis(256),column_title=NULL,
                      row_title=data_anno,
                      show_heatmap_legend = first,heatmap_legend_param = list(
                      title = "\nRelative \nexpression\n",
                      at = c(min_expression, round(mean(c(min_expression,max_expression)),1),max_expression),
                      grid_width= unit(legend_width, "cm"),grid_height= unit(legened_height, "cm") ,
                      labels_gp = gpar(fontsize = cluster_font,fontfamily=fontfamily),
                      title_gp = gpar(fontsize = cluster_font,fontfamily=fontfamily)),
                      #** Cluster name fontsize
                      row_title_gp=gpar(fontsize=cluster_font,fontfamily=fontfamily),
                      left_annotation = direction_annotation,row_gap=unit(strain_gap, gap_units))
    first<-FALSE
    ht_list <- ht_list %v% heatmap

}
draw(ht_list, ht_gap = unit(group_gap, gap_units))
# Save w35, h30