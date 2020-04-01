#*** Draws expression of multiple gene clusters across strains
#* Comments starting with '**' denote changes that can be easily made to the heatmap. 
#* Comments starting with '**!' denote required changes (e.g. file names, ...).
#* For Regulons_by_strain heatmap the images can be saved with resolution 2500w*1500h (specified in R studio Export -> 
#* Save as image). This resolution ensures that the text on the heatmap does not overlap anymore. 

library(ComplexHeatmap)
library(circlize)
library(viridis)

#**! Paths where expression data (average expression, expression patterns, expression height), strain order,
#** regulons clusters, and phenotipic data are saved
path_clusters='/home/karin/Documents/timeTrajectories/data/regulons/by_strain/kN300_mean0std1_log/'
path_expression='/home/karin/Documents/timeTrajectories/data/regulons/'
path_expression_height='/home/karin/Documents/timeTrajectories/data/regulons/by_strain/'
path_strain_order='/home/karin/Documents/timeTrajectories/data/'
path_phenotypes = '/home/karin/Documents/timeTrajectories/data/stages/'

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
avg_phenotype=read.table(paste(path_phenotypes,"averageStages.tsv",sep=''),
                          header=TRUE,row.names=1, sep="\t", stringsAsFactors=FALSE)
#Change avg_phenotypes data so that each phenotype can be coloured differently
avg_phenotype[avg_phenotype=='no']=NA
for(col in colnames(avg_phenotype)){
  new_col=avg_phenotype[col]
  new_col[new_col=='yes']=col
  avg_phenotype[col]=new_col
}

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
regulons=read.table(paste(path_clusters,"clusters/mergedGenes_minExpressed0.990.1Strains1Min1Max18_clustersAX4Louvain0.4m0s1log.tab",sep=''),
                    header=TRUE, sep="\t")
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
legend_font=12
phenotypes_font=10
legened_height=1.5
legend_width=0.7
top_annotation_height=0.6
phenotype_annotation_height=3
cluster_font=15

#Strain groups annotation
#** Colours of strain groups
group_cols=c('agg-'= '#ed1c24', 'lag_dis'= '#f58220','tag_dis'='#fefd44', 'tag'='#d3c300', 'cud'= '#008629', 'WT'= '#00b2ff',
                        'sFB'= '#1925ae', 'prec'='#a400d4' )
#group_cols_background=c('agg-'= '#cccccc', 'lag_dis'= '#666666','tag_dis'='#666666', 'tag'='#666666', 'cud'= '#cccccc', 
#                        'WT'= '#cccccc','sFB'= '#cccccc', 'prec'='#cccccc' )
group_cols_background=c('agg-'= 'white', 'lag_dis'= 'white','tag_dis'='#666666', 'tag'='#666666', 'cud'= 'white', 
                        'WT'= 'white','sFB'= 'white', 'prec'='white' )
group_cols_text=c('agg-'= 'black', 'lag_dis'= 'black','tag_dis'='black', 'tag'='black', 'cud'= '#eeeeee', 
                        'WT'= 'black','sFB'= '#eeeeee', 'prec'='#eeeeee' )

group_data=t(avg_expression['Group'])
rownames(group_data)<-c('Phenotypic group')            
# ht_list=Heatmap(group_data,show_column_names = FALSE, 
#                 height = unit(top_annotation_height, "cm"),
#                 column_split=factor(avg_expression$Strain,
#                                     #** Ordering of the strains in the heatmap (a vector of strain names)
#                                     #levels=unique(avg_expression$Strain)
#                                     levels=strain_order
#                 ),
#                 cluster_columns=FALSE,name='\nPhenotypic \ngroup\n',
#                 #** Strain name font size
#                 column_title_gp=gpar(fontsize=legend_font),
#                 col=group_cols, heatmap_legend_param = list( 
#                 grid_width= unit(legend_width, "cm"),grid_height= unit(legened_height, "cm") ,
#                 labels_gp = gpar(fontsize = cluster_font),title_gp = gpar(fontsize = cluster_font)),
#                 row_names_gp = gpar(fontsize = cluster_font))

#Time annotation
times=unique(avg_expression$Time)
#** Time colours
group_cols_ordered=c()
groups_ordered=c()
background_cols_ordered=c()
text_cols_ordered=c()
for(strain in strain_order){
  group=as.character(avg_expression[avg_expression$Strain==strain,'Group'][1])
  #print(paste(strain,group,group_cols[group]))
  groups_ordered<-append(groups_ordered,group)
  group_cols_ordered<-append(group_cols_ordered,group_cols[group])
  background_cols_ordered<-append(background_cols_ordered,group_cols_background[group])
  text_cols_ordered<-append(text_cols_ordered,group_cols_text[group])
}
col_time = colorRamp2( c(min(times),max(times)),c( "white", "#440154FF"))
ht_list=Heatmap(t(avg_expression['Time']), height = unit(top_annotation_height, "cm"),
                column_split=factor(avg_expression$Strain,
                #** Ordering of the strains in the heatmap (a vector of strain names)
                levels=strain_order ),
                cluster_columns=FALSE, show_column_names = FALSE,name='\nTime\n',col=col_time,
                heatmap_legend_param = list( at = c(min(times),as.integer(mean(c(min(times),max(times)))),max(times)),
                grid_width= unit(legend_width, "cm"),grid_height= unit(legened_height, "cm") ,
                labels_gp = gpar(fontsize = cluster_font),title_gp = gpar(fontsize = cluster_font)),
                row_names_gp = gpar(fontsize = cluster_font),
                top_annotation = HeatmapAnnotation(Phenotype = anno_block(gp = 
                                            #gpar(fill = '#949494',col='#949494'),
                                            gpar(fill = 'white',col='white'),
                                            #gpar(fill = background_cols_ordered,col=background_cols_ordered),
                                            #gpar(fill = group_cols_ordered,col=group_cols_ordered),
                        labels = groups_ordered, labels_gp = gpar(col = 
                                                                    group_cols_ordered, 
                                                                    #text_cols_ordered,
                                                                  fontsize = cluster_font
                                                                  #,fontface='bold'
                                                                  ))
                        )
                )
#ht_list=ht_list %v% ht_time

#Phenotype annotation
#** Colours of phenotype annotations
phenotype_cols=c('unknown'= '#d9d9d9', 'no_agg'= '#ed1c24', 'stream'= '#985006', 'lag'= '#f58220', 'tag'= '#d3c300', 'tip'= '#66cf00',
  'slug'= '#008629', 'mhat'= '#00c58f', 'cul'= '#0ff2ff', 'FB'= '#00b2ff', 'yem'='#666666')
#phenotype_cols=c('no data'= '#d9d9d9', 'yes'= '#74cf19', 'no'='#b54c4c')
ht_phenotype=Heatmap(t(avg_phenotype)[,rownames(avg_expression)], height = unit(phenotype_annotation_height, "cm"),
                cluster_columns=FALSE,cluster_rows=FALSE, show_column_names = FALSE,name='\nMorphological \nstage\n',col=phenotype_cols,
                row_names_gp = gpar(fontsize = phenotypes_font), na_col = "white",
                row_title ='Morphological stage',row_title_side ='right',row_title_gp=gpar(fontsize = cluster_font),
                heatmap_legend_param = list( grid_width= unit(legend_width, "cm"),grid_height= unit(legened_height, "cm") ,
                                             labels_gp = gpar(fontsize = cluster_font),title_gp = gpar(fontsize = cluster_font)))
ht_list=ht_list %v% ht_phenotype


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
colours_regulons2=c('#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
                    '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000',  '#000075', '#808080', '#ffffff',
                    '#80a2ff','#5c3c00')

colours_regulons2_map=colours_regulons2[1:length(unique(regulons2$Cluster))]
names(colours_regulons2_map)<-unique(regulons2$Cluster)
for (cluster in cluster_order$Cluster){
  print(cluster)
  genes=as.character(regulons[regulons$Cluster==cluster,'Gene'])

  regulons2_annotation=rowAnnotation(AX4_clusters = regulons2[genes,],col = list(AX4_clusters = colours_regulons2_map),
                     show_legend = FALSE,annotation_name_side = "top",show_annotation_name = first,
                     annotation_name_gp=gpar(fontsize = cluster_font))
  
  heatmap=Heatmap(t(avg_expression[,genes]),cluster_columns = FALSE,show_column_names = FALSE,
                  show_row_names = FALSE, col=viridis(256),column_title=NULL, 
                  #The as.character ensures that the code works with numeric clusters
                  row_title=gsub('C','',as.character(cluster)),
                  show_heatmap_legend = first,heatmap_legend_param = list(
                  title = "\nRelative \nexpression\n",
                  at = c(min_expression, round(mean(c(min_expression,max_expression)),1),max_expression),
                  grid_width= unit(legend_width, "cm"),grid_height= unit(legened_height, "cm") ,
                  labels_gp = gpar(fontsize = cluster_font),title_gp = gpar(fontsize = cluster_font)),
                  #** Cluster name fontsize
                  row_title_gp=gpar(fontsize=cluster_font),
                  left_annotation = regulons2_annotation)
  first=FALSE
  ht_list=ht_list %v% heatmap
}

#Plots the combined heatmap 
ht_list



  