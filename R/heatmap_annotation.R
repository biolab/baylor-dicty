

library(ComplexHeatmap)
library(circlize)
library(viridis)
library(extrafont)


path_strain_order='/home/karin/Documents/timeTrajectories/data/'
path_phenotypes = '/home/karin/Documents/timeTrajectories/data/stages/'
path_expression='/home/karin/Documents/timeTrajectories/data/regulons/'

#Import fonts and press y to continue
font_import(prompt=FALSE)
loadfonts(device = "postscript")
loadfonts(device = "pdf")

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

# Only for finding strain groups
avg_expression=read.table(paste(path_expression,"genes_averaged_orange_scale99percentileMax0.1.tsv",sep=''),
                          header=TRUE,row.names=1, sep="\t")

#** Strain order - single column with ordered strain names
strain_order<-as.vector(read.table(paste(path_strain_order,"strain_order.tsv",sep=''))[,1])


#** Some plotting parameters
#legend_font=12
phenotypes_font=10
legened_height=1.5
legend_width=0.7
top_annotation_height=0.6
phenotype_annotation_height=3
cluster_font=15
fontfamily='Arial'

#Strain groups annotation
#** Colours of strain groups
group_cols=c('agg-'= '#ed1c24', 'lag_dis'= '#f97402','tag_dis'='#ffb100', 'tag'='#d9d800', 'cud'= '#008629', 'WT'= '#00b2ff',
             'sFB'= '#1925ae', 'prec'='#a400d4' )
#group_cols_background=c('agg-'= '#cccccc', 'lag_dis'= '#666666','tag_dis'='#666666', 'tag'='#666666', 'cud'= '#cccccc', 
#                        'WT'= '#cccccc','sFB'= '#cccccc', 'prec'='#cccccc' )
group_cols_background=c('agg-'= 'white', 'lag_dis'= 'white','tag_dis'='#666666', 'tag'='#666666', 'cud'= 'white', 
                        'WT'= 'white','sFB'= 'white', 'prec'='white' )
group_cols_text=c('agg-'= 'black', 'lag_dis'= 'black','tag_dis'='black', 'tag'='black', 'cud'= '#eeeeee', 
                  'WT'= 'black','sFB'= '#eeeeee', 'prec'='#eeeeee' )

make_annotation<-function(phenotypes_font=parent.frame()$phenotypes_font,legend_height=parent.frame()$legend_height,
                          legend_width=parent.frame()$legend_width,top_annotation_height=parent.frame()$top_annotation_height,
                          phenotype_annotation_height=parent.frame()$phenotype_annotation_height,cluster_font=parent.frame()$cluster_font){
  
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
  #                 column_title_gp=gpar(fontsize=legend_font,fontfamily=fontfamily),
  #                 col=group_cols, heatmap_legend_param = list( 
  #                 grid_width= unit(legend_width, "cm"),grid_height= unit(legened_height, "cm") ,
  #                 labels_gp = gpar(fontsize = cluster_font,fontfamily=fontfamily),title_gp = gpar(fontsize = cluster_font,fontfamily=fontfamily)),
  #                 row_names_gp = gpar(fontsize = cluster_font,fontfamily=fontfamily))
  
  #Time annotation
  times=unique(avg_expression$Time)
  #** Time colours, group colours, and gaps
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
  
  col_time = colorRamp2( c(min(times),max(times)),c( "white", "#440154FF"))
  
  ht_list=Heatmap(t(avg_expression['Time']), height = unit(top_annotation_height, "cm"),
                  column_split=factor(avg_expression$Strain,
                                      #** Ordering of the strains in the heatmap (a vector of strain names)
                                      levels=strain_order ),
                  column_title =NULL,column_gap=gaps,
                  cluster_columns=FALSE, show_column_names = FALSE,name='\nTime\n',col=col_time,
                  heatmap_legend_param = list( at = c(min(times),as.integer(mean(c(min(times),max(times)))),max(times)),
                                               grid_width= unit(legend_width, "cm"),grid_height= unit(legened_height, "cm") ,
                                               labels_gp = gpar(fontsize = cluster_font,fontfamily=fontfamily),title_gp = gpar(fontsize = cluster_font,fontfamily=fontfamily)),
                  row_names_gp = gpar(fontsize = cluster_font,fontfamily=fontfamily),
                  #column_title_gp=gpar(border =group_cols_ordered,fontsize=cluster_font,fontfamily=fontfamily,col =text_cols_ordered,fill=group_cols_ordered,
                  #                     fontface='bold'),
                  #Annotation for Phenotype group
                  top_annotation = HeatmapAnnotation(
                    Phenotype=anno_block(gp = 
                                           # Background colour; fill: color, col: border                               
                                           #gpar(fill = '#949494',col='transparent'),
                                           #gpar(fill = 'white',col='transparent'),
                                           #gpar(fill = background_cols_ordered,col='transparent'),
                                           #gpar(fill = group_cols_ordered,col=group_cols_ordered),
                                           gpar(fill=group_cols_ordered,col=group_cols_ordered,lwd =2,linejoin='mitre'),
                                         labels = groups_ordered , labels_gp = gpar(col = 
                                                                                      # Text colour
                                                                                      # 'black',
                                                                                      #group_cols_ordered, 
                                                                                      text_cols_ordered,
                                                                                    fontsize = cluster_font,fontfamily=fontfamily
                                                                                    #,fontface='bold'
                                         ),show_name = TRUE),
                    Strain = anno_block(gp = 
                                          # Background colour; fill: color, col: border                               
                                          #gpar(fill = '#949494',col='transparent'),
                                          #gpar(fill = 'white',col='transparent'),
                                          #gpar(fill = background_cols_ordered,col='transparent'),
                                          #gpar(fill = group_cols_ordered,col=group_cols_ordered),
                                          gpar(fill='white',col=group_cols_ordered,lwd =2,linejoin='mitre'),
                                        labels = strain_order , labels_gp = gpar(col = 
                                                                                   # Text colour
                                                                                   'black',
                                                                                 #group_cols_ordered, 
                                                                                 #text_cols_ordered,
                                                                                 fontsize = cluster_font,fontfamily=fontfamily
                                                                                 #,fontface='bold'
                                        ), show_name = TRUE),
                    annotation_name_gp=gpar(fontsize = cluster_font,fontfamily=fontfamily)
                  )
  )
  #ht_list=ht_list %v% ht_time
  
  #Phenotype annotation
  #** Colours of phenotype annotations
  phenotype_cols=c('no image'= '#d9d9d9', 'no_agg'= '#ed1c24', 'stream'= '#985006', 'lag'= '#f97402', 'tag'= '#d9d800', 'tip'= '#66cf00',
                   'slug'= '#008629', 'mhat'= '#00c58f', 'cul'= '#0ff2ff', 'FB'= '#00b2ff', 'yem'='#666666')
  #phenotype_cols=c('no data'= '#d9d9d9', 'yes'= '#74cf19', 'no'='#b54c4c')
  ht_phenotype=Heatmap(t(avg_phenotype)[,rownames(avg_expression)], height = unit(phenotype_annotation_height, "cm"),
                       cluster_columns=FALSE,cluster_rows=FALSE, show_column_names = FALSE,name='\nMorphological \nstage\n',col=phenotype_cols,
                       row_names_gp = gpar(fontsize = phenotypes_font,fontfamily=fontfamily), na_col = "white",
                       row_title ='Morphological stage',row_title_side ='right',row_title_gp=gpar(fontsize = cluster_font,fontfamily=fontfamily),
                       heatmap_legend_param = list( grid_width= unit(legend_width, "cm"),grid_height= unit(legened_height, "cm") ,
                                                    labels_gp = gpar(fontsize = cluster_font,fontfamily=fontfamily),title_gp = gpar(fontsize = cluster_font,fontfamily=fontfamily)))
  ht_list=ht_list %v% ht_phenotype
  
  return(ht_list)
}










