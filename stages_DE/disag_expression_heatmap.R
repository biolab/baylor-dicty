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
path_disag='/home/karin/Documents/timeTrajectories/data/stages/disagg/deSeq/'
path_expression='/home/karin/Documents/timeTrajectories/data/regulons/'

#**! Specify file names for regulons and expression
#** Expression tab file: Genes in columns (already scaled), averaged strain data in rows,
#** three additional comlumns: Time, Strain, and Group (meaning strain group)
avg_expression=read.table(paste(path_expression,"genes_averaged_orange_scale99percentileMax0.1.tsv",sep=''),
                          header=TRUE,row.names=1, sep="\t")


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

optically_order_genes <- function(genes,avg_expression=parent.frame()$avg_expression,ax4=TRUE){
  if (length(genes)>1){
    if(ax4){    expression<-t(avg_expression[avg_expression$Strain=='AX4',genes])
    }else{
    expression=t(avg_expression[,genes])
    }
    # Looks same as 1-simil(expression, method="cosine")
    distances<-dist(expression, method="Euclidean")
    hc<-hclust(d=distances, method = "ward.D2" )
    hc_ordered<-reorder(x=hc,dist = distances)
    genes<- as.dendrogram(hc_ordered) %>% labels
  }
  return(genes)
}

# Expression range for legend
expressions<-within(avg_expression, rm('Time', 'Strain','Group'))
min_expression<-min(expressions)
max_expression<-max(expressions)

#Expression heatmap
ht_list<-make_annotation(phenotypes_font,legend_height,legend_width,top_annotation_height,phenotype_annotation_height,cluster_font)

first=TRUE
for (times in list(c(4,6),c(6,8),c(8,12))){
  print(times)

  data_AX4<-read.table(paste0(path_disag,'timepoints_within_strain/AX4_confoundRep_FDRoptim0.01_DE_hr',times[2],
                              '_ref_hr',times[1],'_padj_lFC.tsv'),
                         sep='\t',header=TRUE,row.names=1)
  data_comH<-read.table(paste0(path_disag,'timepoints_within_strain/comH_confoundRep_FDRoptim0.01_DE_hr',times[2],
                               '_ref_hr',times[1],'_padj_lFC.tsv'),
                         sep='\t',header=TRUE,row.names=1)
  data_tagB<-read.table(paste0(path_disag,'timepoints_within_strain/tagB_confoundRep_FDRoptim0.01_DE_hr',times[2],
                               '_ref_hr',times[1],'_padj_lFC.tsv'),
                         sep='\t',header=TRUE,row.names=1)
  data_tgrB1<-read.table(paste0(path_disag,'timepoints_within_strain/tgrB1_confoundRep_FDRoptim0.01_DE_hr',times[2],
                                '_ref_hr',times[1],'_padj_lFC.tsv'),
                         sep='\t',header=TRUE,row.names=1)
  data_tgrB1C1<-read.table(paste0(path_disag,'timepoints_within_strain/tgrB1C1_confoundRep_FDRoptim0.01_DE_hr',times[2],
                                  '_ref_hr',times[1],'_padj_lFC.tsv'),
                         sep='\t',header=TRUE,row.names=1)
  # Cant do intersect of not AX4/tagB/comH and yes tgrB1/tgrB1C1 as this misses genes not calculated in
  # AX4/tagB/comH
  padj=0.01
  lfc=1.32
  filter<-Reduce(union,list(
               rownames(data_AX4)[data_AX4$padj <= padj & data_AX4$log2FoldChange >= lfc],
               rownames(data_tagB)[data_tagB$padj <= padj & data_tagB$log2FoldChange >= lfc],
               rownames(data_comH)[data_comH$padj <= padj & data_comH$log2FoldChange >= lfc]))
  genes<-Reduce(intersect,list(
               rownames(data_tgrB1)[data_tgrB1$padj <= padj & data_tgrB1$log2FoldChange >= lfc],
               rownames(data_tgrB1C1)[data_tgrB1C1$padj <= padj & data_tgrB1C1$log2FoldChange >= lfc]
               ))
  genes<-genes[!(genes %in% filter)]
  print(length(genes))
  genes<-optically_order_genes(genes=genes)

  heatmap=Heatmap(t(avg_expression[,genes]),cluster_columns = FALSE,cluster_rows = FALSE,show_column_names = FALSE,
                  show_row_names = FALSE, col=viridis(256),column_title=NULL,
                  row_title=paste0(times[2],'hr vs ',times[1],'hr'),
                  show_heatmap_legend = first,heatmap_legend_param = list(
                  title = "\nRelative \nexpression\n",
                  at = c(min_expression, round(mean(c(min_expression,max_expression)),1),max_expression),
                  grid_width= unit(legend_width, "cm"),grid_height= unit(legened_height, "cm") ,
                  labels_gp = gpar(fontsize = cluster_font,fontfamily=fontfamily),title_gp = gpar(fontsize = cluster_font,fontfamily=fontfamily)),
                  #** Cluster name fontsize
                  row_title_gp=gpar(fontsize=cluster_font,fontfamily=fontfamily),)
  first=FALSE
  ht_list=ht_list %v% heatmap
}

#Plots the combined heatmap
pdf(paste0(path_disag,
           'timepoints_within_strain/expressionHeatmap_tgrB1tgrB1C1vsAX4tagBcomH_confoundRep_FDRoptim0.01_DEpadj',padj,
           'lfc',lfc,'.pdf'), width = 35, height = 20)
draw(ht_list)
graphics.off()

# *** Expression heatmap - disag tag_dic vs AX4 all
ht_list<-make_annotation(phenotypes_font,legend_height,legend_width,top_annotation_height,phenotype_annotation_height,cluster_font)

data<-read.table(paste0(path_disag,
                        'disagVSAX4all/disagVSAX4all_alternativegreater_FDRoptim0.01DE_tgrB1hr8hr10hr12andtgrB1C1hr8hr10hr12_ref_AX4all_padj_lFC.tsv'),
                       sep='\t',header=TRUE,row.names=1)
padj=0.01
lfc=2
genes<-rownames(data)[data$padj <= padj & data$log2FoldChange >= lfc]
print(length(genes))
genes<-optically_order_genes(genes=genes)

heatmap=Heatmap(t(avg_expression[,genes]),cluster_columns = FALSE,cluster_rows = FALSE,show_column_names = FALSE,
                show_row_names = FALSE, col=viridis(256),column_title=NULL,
                show_heatmap_legend = TRUE,heatmap_legend_param = list(
                title = "\nRelative \nexpression\n",
                at = c(min_expression, round(mean(c(min_expression,max_expression)),1),max_expression),
                grid_width= unit(legend_width, "cm"),grid_height= unit(legened_height, "cm") ,
                labels_gp = gpar(fontsize = cluster_font,fontfamily=fontfamily),
                title_gp = gpar(fontsize = cluster_font,fontfamily=fontfamily)))
ht_list=ht_list %v% heatmap

# Plots the combined heatmap
pdf(paste0(path_disag,
           'disagVSAX4all/expressionHeatmap_disagVSAX4all_alternativegreater_FDRoptim0.01DE_tgrB1hr8hr10hr12andtgrB1C1hr8hr10hr12_ref_AX4all_DEpadj',padj,
           'lfc',lfc,'.pdf'), width = 35, height = 8)
draw(ht_list)
graphics.off()



# *** Expression heatmap - chubs
ht_list<-make_annotation(phenotypes_font,legend_height,legend_width,top_annotation_height,phenotype_annotation_height,cluster_font)

data<-read.table(paste0(path_disag,
                        'media_buffer/mediaVSbufferAll_alternativegreater_FDRoptim0.01_DE_media0.5hr1hr2hr_ref_bufferAll_padj_lFC.tsv'),
                       sep='\t',header=TRUE,row.names=1)
padj=0.01
lfc=2
genes<-rownames(data)[data$padj <= padj & data$log2FoldChange >= lfc]
print(length(genes))
genes<-optically_order_genes(genes=genes)
# On huston data
heatmap=Heatmap(t(avg_expression[,genes]),cluster_columns = FALSE,cluster_rows = FALSE,show_column_names = FALSE,
                show_row_names = FALSE, col=viridis(256),column_title=NULL,
                show_heatmap_legend = TRUE,heatmap_legend_param = list(
                title = "\nRelative \nexpression\n",
                at = c(min_expression, round(mean(c(min_expression,max_expression)),1),max_expression),
                grid_width= unit(legend_width, "cm"),grid_height= unit(legened_height, "cm") ,
                labels_gp = gpar(fontsize = cluster_font,fontfamily=fontfamily),
                title_gp = gpar(fontsize = cluster_font,fontfamily=fontfamily)))
ht_list=ht_list %v% heatmap

#Plots the combined heatmap
pdf(paste0(path_disag,
           'media_buffer/expressionHeatmap_mediaVSbufferAll_alternativegreater_FDRoptim0.01_DE_media0.5hr1hr2hr_ref_bufferAll_padj',padj,
           'lfc',lfc,'.pdf'), width = 35, height = 8)
draw(ht_list)
graphics.off()

# On medium buffer data
avg_expression_mb<-read.table(paste("/home/karin/Documents/timeTrajectories/data/from_huston/averaged_scale99percentileMax0.1mediaBuffer.tsv",sep=''),
                          header=TRUE,row.names=1, sep="\t")
genes<-optically_order_genes(genes=genes,avg_expression=avg_expression_mb,ax4=FALSE)
times_mb<-unique(avg_expression_mb['Time'])
col_time_mb = colorRamp2( c(min(times_mb),max(times_mb)),c( "white", "#440154FF"))
 ht_list=Heatmap(t(avg_expression_mb['Time']), height = unit(top_annotation_height, "cm"),
                  column_split=factor(avg_expression_mb$Group),
                  column_title =NULL,
                  cluster_columns=FALSE, show_column_names = FALSE,name='\nTime\n',col=col_time_mb,
                  heatmap_legend_param = list( at = c(min(times_mb),as.integer(mean(c(min(times_mb),max(times_mb)))),max(times_mb)),
                                               grid_width= unit(legend_width, "cm"),grid_height= unit(legened_height, "cm") ,
                                               labels_gp = gpar(fontsize = cluster_font,fontfamily=fontfamily),title_gp = gpar(fontsize = cluster_font,fontfamily=fontfamily)),
                  row_names_gp = gpar(fontsize = cluster_font,fontfamily=fontfamily),
                  #column_title_gp=gpar(border =group_cols_ordered,fontsize=cluster_font,fontfamily=fontfamily,col =text_cols_ordered,fill=group_cols_ordered,
                  #                     fontface='bold'),
                  #Annotation for Phenotype group
                  top_annotation = HeatmapAnnotation(
                    Group=anno_block(gp =
                                           gpar(fill=c('#00b2ff','#ed1c24'),col=c('#00b2ff','#ed1c24'),lwd =2,linejoin='mitre'),
                                         labels = c('buff','media') , labels_gp = gpar(col ='black',
                                                                                    fontsize = cluster_font,fontfamily=fontfamily
                                                                                    #,fontface='bold'
                                         ),show_name = TRUE),
                    annotation_name_gp=gpar(fontsize = cluster_font,fontfamily=fontfamily)
                  )
  )

heatmap=Heatmap(t(avg_expression_mb[,genes]),cluster_columns = FALSE,cluster_rows = FALSE,show_column_names = FALSE,
                show_row_names = FALSE, col=viridis(256),column_title=NULL,
                show_heatmap_legend = TRUE,heatmap_legend_param = list(
                title = "\nRelative \nexpression\n",
                at = c(min_expression, round(mean(c(min_expression,max_expression)),1),max_expression),
                grid_width= unit(legend_width, "cm"),grid_height= unit(legened_height, "cm") ,
                labels_gp = gpar(fontsize = cluster_font,fontfamily=fontfamily),
                title_gp = gpar(fontsize = cluster_font,fontfamily=fontfamily)))
ht_list=ht_list %v% heatmap

#Plots the combined heatmap
pdf(paste0(path_disag,
           'media_buffer/expressionMediaBufferHeatmap_mediaVSbufferAll_alternativegreater_FDRoptim0.01_DE_media0.5hr1hr2hr_ref_bufferAll_padj',padj,
           'lfc',lfc,'.pdf'), width = 10, height = 8)
draw(ht_list)
graphics.off()

