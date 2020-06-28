library(ComplexHeatmap)
dataPathCode='/home/karin/Documents/git/baylor-dicty/R/'

library(ComplexHeatmap)
library(circlize)
library(viridis)
library(proxy)
library(seriation)
library(dendextend)
source(paste(dataPathCode,'heatmap_annotation.R',sep=''))

#**! Paths where expression data (average expression, expression patterns, expression height), strain order,
#** regulons clusters, and phenotipic data are saved
path_aberrant='/home/karin/Documents/timeTrajectories/data/regulons/selected_genes/'
path_expression='/home/karin/Documents/timeTrajectories/data/regulons/'

#**! Specify file names for regulons and expression
#** Expression tab file: Genes in columns (already scaled), averaged strain data in rows,
#** three additional comlumns: Time, Strain, and Group (meaning strain group)
avg_expression <- read.table(paste(path_expression, "genes_averaged_orange_scale99percentileMax0.1.tsv", sep=''),
                             header=TRUE, row.names=1, sep="\t")

# Median similarities
median_sims<-read.table(paste0(path_aberrant,'simsMedian_AX4basedNeigh_newGenes-removeZeroRep_neighSimsDict_scalemean0std1_logTrue_kN11_splitStrain_samples10resample10.tsv'),
                    header=TRUE,sep=',',row.names=1)
median_sims<-median_sims[,strain_order]

# Selecte genes
FDR<-0.001
MEDIFF<-0.3
group_abberant<-list()
#Ordered strain groups
strain_groups<-c('prec','sFB','cud','tag','tag_dis','lag_dis','agg-')
for (group in strain_groups){
    data <- read.table(paste0(path_aberrant, 'comparisonsSims_', group,
                              '_AX4basedNeigh_u-less_removeZeroRep_scalemean0std1_logTrue_kN11_samples10resample10.tsv'),
                       header=TRUE, sep='\t')
    genes<-as.vector(data[data$FDR<=FDR & data['Difference.median']>=MEDIFF,'Gene'])
    n_genes <- length(genes)
    group_abberant[[group]]<-genes
}
# Put selected genes in a single df - 1 if selected for a group (col) and 0 if not
all_abberant<-NULL
for(group in strain_groups){
    genes<-group_abberant[[group]]
    df<-data.frame('Gene'=genes,group=rep(1,(length(genes))))
    colnames(df)<-c('Gene',group)
    if(is.null(all_abberant)){
        all_abberant<-df
    }else{
        all_abberant<-merge(all_abberant,df,by='Gene',all=TRUE)
    }
}
all_abberant[is.na(all_abberant)]<-0
rownames(all_abberant)<-all_abberant$Gene
all_abberant <- all_abberant[, strain_groups]


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

optically_order_genes <- function(genes,avg_expression=parent.frame()$avg_expression){
  if (length(genes)>1){
    expression<-t(avg_expression[avg_expression$Strain=='AX4',genes])
    #expression=t(avg_expression[,genes])
    # Looks same as 1-simil(expression, method="cosine")
    distances<-dist(expression, method="Euclidean")
    hc<-hclust(d=distances, method = "ward.D2" )
    hc_ordered<-reorder(x=hc,dist = distances)
    genes<- as.dendrogram(hc_ordered) %>% labels
  }
  return(genes)
}

# Expression range for legend
expressions<-within(avg_expression[], rm('Time', 'Strain','Group'))
min_expression<-min(expressions[,rownames(all_abberant)])
max_expression<-max(expressions[,rownames(all_abberant)])

selected_cols<-list()
for(group in strain_groups){
  selected_cols[[group]]<-c('0'='white','1'=group_cols[[group]])
}

group_cols_list<-as.vector(unlist(lapply(strain_groups, function(x) group_cols[[x]])))
if(FALSE){
for(group in strain_groups){
  print(group)

  genes<-group_abberant[[group]]
  print(length(genes))
  genes<-optically_order_genes(genes=genes)

  row_annotation<-rowAnnotation(df = all_abberant[genes,],col = selected_cols,
                                gp = gpar(col=NA),
                                show_legend=FALSE,
                                annotation_name_side = "top",show_annotation_name = TRUE,
                                annotation_name_gp=gpar(
                                  fontsize = cluster_font,fontfamily=fontfamily
                                )

  )

  ht_list<-make_annotation(phenotypes_font,legend_height,legend_width,top_annotation_height,phenotype_annotation_height,cluster_font)

  heatmap <- Heatmap(t(avg_expression[, genes]), cluster_columns = FALSE, cluster_rows = FALSE, show_column_names = FALSE,
                     show_row_names = FALSE, col=viridis(256), column_title=NULL,
                     show_heatmap_legend = TRUE, heatmap_legend_param = list(
                  title = "\nRelative \nexpression\n",
                  at = c(min_expression, round(mean(c(min_expression,max_expression)),1),max_expression),
                  grid_width= unit(legend_width, "cm"),grid_height= unit(legened_height, "cm") ,
                  labels_gp = gpar(fontsize = cluster_font,fontfamily=fontfamily),
                  title_gp = gpar(fontsize = cluster_font,fontfamily=fontfamily)),
                     left_annotation = row_annotation)
  ht_list <- ht_list %v% heatmap

  lgd_list <- list(
    Legend(labels = strain_groups, title = "Aberrant\nneighborhood\n",
        legend_gp = gpar(col = group_cols_list,
                         fill=group_cols_list,
           fontsize = cluster_font,fontfamily=fontfamily
        ),grid_width= unit(legend_width, "cm"),grid_height= unit(legened_height, "cm"),
           labels_gp = gpar(fontsize = cluster_font,fontfamily=fontfamily),
                  title_gp = gpar(fontsize = cluster_font,fontfamily=fontfamily)
    )
)

  #Plots the combined heatmap
  pdf(paste0(path_aberrant,group,'_padj',FDR, 'MEdiff',MEDIFF,
             '_AX4basedNeigh_newGenes-removeZeroRep_neighSimsDict_scalemean0std1_logTrue_kN11_splitStrain_samples10resample10.pdf'), width = 40, height = 25)
  draw(ht_list, annotation_legend_list = lgd_list)
  graphics.off()
}
}

# *** Heatmap of similarities and selected genes
distances<-dist(all_abberant, method="binary")
hc<-hclust(d=distances, method = "ward.D2" )
hc_ordered<-reorder(x=hc,dist = distances)
genes<- as.dendrogram(hc_ordered) %>% labels

abberant_colours<-all_abberant[genes,]
for(col in colnames(abberant_colours)){
  new_col=abberant_colours[col]
  new_col[new_col=='1']=col
  new_col[new_col=='0']=NA
  abberant_colours[col]=new_col
}

heatmap_selected <- Heatmap(as.matrix(abberant_colours),
                            cluster_columns = FALSE, cluster_rows = FALSE,
                            col = group_cols, show_row_names = FALSE,na_col='white',
                            heatmap_legend_param = list(title = "\nAberrant\nneighbourhood\n",
                                                        grid_width= unit(legend_width, "cm"),grid_height= unit(legened_height, "cm") ,
                                                        labels_gp = gpar(fontsize = cluster_font,fontfamily=fontfamily),
                  title_gp = gpar(fontsize = cluster_font,fontfamily=fontfamily)
                            ),column_title_gp=(gpar(fontsize = cluster_font,fontfamily=fontfamily)),
                            width=3, column_names_side = "top"
)

group_cols_ordered=c()
groups_ordered=c()
background_cols_ordered=c()
text_cols_ordered=c()
gaps=c()
previous_group=NULL
for(strain in strain_order){
  group=as.character(avg_expression[avg_expression$Strain==strain,'Group'][1])
  #print(paste(strain,group,group_cols[group]))
  groups_ordered<-append(groups_ordered,group)
  group_cols_ordered<-append(group_cols_ordered,group_cols[group])
  background_cols_ordered<-append(background_cols_ordered,group_cols_background[group])
  text_cols_ordered<-append(text_cols_ordered,group_cols_text[group])
  #Gaps - if previous group was different add larger gap; (N gaps = N-1 columns)
  if (!is.null(previous_group)){
    if (previous_group==group){
      gaps=append(gaps,1)
    }else{
      gaps=append(gaps,2.5)
    }
  }
  previous_group=group
}
gaps=unit(gaps,'mm')


heatmap_sims <- Heatmap(as.matrix(median_sims[genes,strain_order]), col=rev(viridis(256)),
                        column_split=factor(
                          colnames(median_sims),levels=strain_order
                          ),
                        column_gap=unit(gsub(gap_units,'',gsub(unit(strain_gap,gap_units),unit(0,gap_units),gaps, fixed=TRUE)),gap_units),
                        column_title =NULL,
                        cluster_columns = FALSE, cluster_rows = FALSE, show_row_names = FALSE,show_column_names=FALSE,
                        heatmap_legend_param = list(title = "\nMedian\nsimilarity\n",
                                                    grid_width= unit(legend_width, "cm"),grid_height= unit(legened_height, "cm") ,
                                                        labels_gp = gpar(fontsize = cluster_font,fontfamily=fontfamily),
                                                    title_gp = gpar(fontsize = cluster_font,fontfamily=fontfamily)),
                        top_annotation = HeatmapAnnotation(
                    Phenotype=anno_block(gp =
                                           gpar(fill=group_cols_ordered,col=group_cols_ordered,lwd =2,linejoin='mitre'),
                                         labels = groups_ordered , labels_gp = gpar(col =
                                                                                      text_cols_ordered,
                                                                                    fontsize = cluster_font,fontfamily=fontfamily
                                         ),show_name = TRUE),
                    Strain = anno_block(gp =
                                          gpar(fill='white',col=group_cols_ordered,lwd =2,linejoin='mitre'),
                                        labels = strain_order , labels_gp = gpar(col =
                                                                                   # Text colour
                                                                                   'black',
                                                                                 fontsize = cluster_font,fontfamily=fontfamily
                                        ), show_name = TRUE),
                    annotation_name_gp=gpar(fontsize = cluster_font,fontfamily=fontfamily)
                  )
)
pdf(paste0(path_aberrant,'selectedSimilarities_padj',FDR, 'MEdiff',MEDIFF,
             '_AX4basedNeigh_newGenes-removeZeroRep_neighSimsDict_scalemean0std1_logTrue_kN11_splitStrain_samples10resample10.pdf'), width =33, height = 25)
  draw(heatmap_selected+heatmap_sims)
  graphics.off()

