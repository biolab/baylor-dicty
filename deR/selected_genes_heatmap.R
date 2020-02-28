
library(ComplexHeatmap)
library(circlize)
library(plyr)
library(viridis)

path_regulons='/home/karin/Documents/timeTrajectories/data/regulons/'
path_comparisons='/home/karin/Documents/timeTrajectories/data/regulons/selected_genes/'

avg_expression=read.table(paste(path_regulons,"genes_averaged_orange_scale99percentileMax0.1.tsv",sep=''),
                          header=TRUE,row.names=1, sep="\t")
comparisons=read.table(paste(path_comparisons,"summary_comparisonsAvgSims_AX4basedNeigh_u-less_newGenes_noAll-removeZeroRep_simsDict_scalemean0std1_logTrue_kN11_splitStrain.tsv",
                             sep=''), header=TRUE,row.names=1, sep="\t")
comparisons=as.matrix(data.frame(lapply(comparisons,as.character),row.names = row.names(comparisons)))
comparisons<-mapvalues(comparisons, from = c("1", "0"), to = c("Difference", "Not difference"))

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
                column_split=factor(avg_expression$Strain,levels=unique(avg_expression$Strain)),
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
                column_split=factor(colnames(similarities),levels=unique(avg_expression$Strain)),
                show_row_names = FALSE, col=magma(256),
                show_heatmap_legend = TRUE,heatmap_legend_param = list(
                  title = "Average similarity",
                  at = c(min_sims, round(mean(c(min_sims,max_sims)),1),max_sims),
                  grid_width= unit(legend_width, "cm"),grid_height= unit(legened_height, "cm") ,
                  labels_gp = gpar(fontsize = legend_font),title_gp = gpar(fontsize = legend_font)),top_annotation=ha
)
heatmap+heatmap_diff

#ImpulseDE2 test
#geneA=c(0,0,0,5,10,5,0,0,0,0,0,5,9,7,0,0,0,5,10,5,0,0,0,0,0,5,9,7,0,0,0,0)
#geneB=c(0,0,0,5,10,5,0,0,0,0,0,5,9,7,0,0,0,0,0,5,10,5,0,0,0,0,0,5,9,7,0,0)
#geneC=c(0,0,0,5,10,5,0,0,0,0,0,5,9,7,0,0, 10,10,10,20,30,20,10,10,10,10,10,20,28,22,10,10)