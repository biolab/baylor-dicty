# Build data for DeSeq2

THREADS=20

buildDDS<-function(conditions,genes,t=NULL,case=NULL,ref,design,main_lvl=NULL,coldata=NULL,filter=1,set_main_lvl=FALSE){

  # BUilds a DDS for DESeq2
  # Conditions (M*D), genes (G*M) - dataframes
  # t - subset Time, vector
  # case, ref - retain only these from main_lvl, case (case can be vector)
  # ref - Set as reference if main_level is provided
  # design - DeSeq2 design
  # coldata -Prespecify coldata for DeSeq2
  # Filter - remove genes rows with less than filter row count
  # set_main_lvl - set ref as main lvl
  # Returns dds and coldata in list
  if (!is.null(t)){
    genes=genes[,conditions$Time %in% t]
    conditions=conditions[conditions$Time %in% t,]
  }
  if (!is.null(case)){
    genes=genes[,unlist(conditions[main_lvl] , use.names=FALSE) %in% c(case,ref)]
    conditions=conditions[unlist(conditions[main_lvl]  , use.names=FALSE) %in% c(case,ref),]
  }
  #Make coldata table for DeSeq2 
  if (is.null(coldata)){
    coldata<-data.frame(lapply(conditions, as.factor))
    coldata<-data.frame(lapply(coldata,droplevels))
    rownames(coldata)<-rownames(conditions)
  }
  #Check that sample descriptions match betweeen expression  data and description
  if (all(rownames(coldata)==colnames(genes))){
    #Make DeSeq2 data object
    dds <- DESeqDataSetFromMatrix(countData = genes,
                                  colData = coldata,
                                  design = design)
    if (set_main_lvl & !is.null(main_lvl) ) dds[[main_lvl]]<- relevel(dds[[main_lvl]], ref = ref)
    if (! is.null(filter)){
      keep <- rowSums(counts(dds)) >= filter
      dds <- dds[keep,]
    }
    return(list(dds=dds,coldata=coldata))
  }
  else {
    stop("Sample names in conditions and genes do not match")
  }
}

# Testd DE with DeSeq2
testDE<-function(dds,sample,ref,padjSave,logFCSave,path=NULL,time=NULL,main_lvl='Strain'){
  res <- results(dds,contrast=c(main_lvl, sample, ref),parallel = TRUE)
  resNN <- na.omit(res)
  resOrder<-resNN[order(resNN$padj),]
  resFilter<-resOrder[resOrder$padj<=padjSave & abs(resOrder$log2FoldChange)>=logFCSave,]
  #R reads this format fine, but Libre offic shifts colnames!!!
  if (is.null(path)){
    return(resFilter)
  }else {
    time_str=''
    if (!is.null(time)){
      time_str=paste('_t',time,'h',sep='')
    }
    write.table(x = resFilter,file =paste( path,'DE_',paste(sample,collapse = ''),'_ref_',ref,time_str,'_padj',padjSave,'_lFC',logFCSave,'.tsv',sep=''),sep='\t')
  }
}

# Run DeSeq2 from raw data. Removes genes with all 0
runDeSeq2<-function(conditions,genes,time=NULL,case,control='AX4',design=~Strain,main_lvl='Strain',padj=0.05,logFC=1,path=NULL){
  # Conditions (M*D), genes (G*M) - dataframes
  # time - subset Time, vector
  # case, ref - retain only these from main_lvl, case (case can be vector)
  # control- Set as reference in main_level design
  # design - DeSeq2 design
  # main_lvl -where DE will be analysed
  # padj, logFC - filter results, remove below logFC or above padj
  # path - save, adds file name
  dds<-buildDDS(conditions=conditions,genes=genes,t=time,case=case,ref=control,design=design,main_lvl=main_lvl,filter=1,set_main_lvl=TRUE)$dds
  dds <- DESeq(dds,parallel = TRUE)
  if (is.null(path)) return(testDE(dds=dds,sample=case,ref=control,padjSave=padj,logFCSave=logFC,path=path,main_lvl=main_lvl))
  else testDE(dds=dds,sample=case,ref=control,padjSave=padj,logFCSave=logFC,path=path,time=time,main_lvl=main_lvl)
}

# Make design matrix for nested levels as described in DESeq2 manual
nestedModelDDSData<-function(coldata,main_lvl,nested){
  # coldata - data, use to build model
  # main level - where DE will be analysed
  # nested - if no intersection between main_level values use from 1 to X for each main_level level, else use as factors specified in coldata
  # returns model matrix
  data_parts<-split(coldata,f=coldata[,main_lvl])
  model_df<-data.frame(coldata[,main_lvl])
  names(model_df)<-c(main_lvl)
  for (nested_lvl in nested){
    intersect_by_main<-NULL
    for (data_part in data_parts){
      if (is.null(intersect_by_main)) intersect_by_main<-data_part[,nested_lvl]
      else intersect_by_main<-intersect(data_part[,nested_lvl],intersect_by_main)
    }
    if (length(intersect_by_main)>0) model_df[,nested_lvl]<-coldata[,nested_lvl]
    else{
      changed<-c()
      maps<-c()
      for (data_part_name in names(data_parts)){
        data_part<-data_parts[[data_part_name]]
        factor_map<-c()
        counter=1
        for(factor_val in unique(data_part[,nested_lvl])){
          dict<-makeDict(k=as.character(factor_val),v=as.character(counter))
          factor_map<-append(factor_map,dict)
          counter<-counter+1
        }
        dict<-makeDict(k=data_part_name,v=factor_map)
        maps<-append(maps,dict)
      }
      for (rowN in  sequence(nrow(coldata))){
        row<-coldata[rowN,]
        main_element<-row[,main_lvl]
        nested_element<-row[,nested_lvl]
        factor_map<-maps[[main_element]]
        changed<-append(changed,factor_map[[as.character(nested_element)]])
      }
      model_df[,nested_lvl]<-as.factor(changed)
    }
  }
  
  main_lvl_confounder=paste(' + ',main_lvl,':',sep='')
  formula<-as.formula(paste('~ ',main_lvl,main_lvl_confounder,paste(nested,collapse = main_lvl_confounder),sep=''))
  
  model<-model.matrix(formula, model_df)
  idx <- which(apply(model, 2, function(x) all(x==0)))
  if (length(idx)>0) model <- model[,-idx]
  return(model)
}

# Make named list from keys and values
makeDict<-function(k,v){
  dict<-list(v)
  names(dict)<-as.character(k)
  return(dict)
}

# estimate DeSeq2 dispersion factors
estimateDDSScaling<-function(dds,model=NULL){
  # model - model matrix, if NULL uses the one from dds
  dds<-estimateSizeFactors(dds)
  dds <- estimateDispersions(dds,modelMatrix=model)
  return(dds)
}

#Run impulse with custom dispersion, if provided in dds
impulseCustomDispersion<-function(dds,confounders=NULL,condition='Strain',fdr=0.05,control='AX4',path=NULL,threads=THREADS){
  # dds - DeSeq object
  # confounders
  # condition - for case/control determination, if not control is case
  # fdr - retain only genes with padj >=fdr
  # control - use as control
  # path - save result to this path (creates file name)
  coldata<-colData(dds)
  coldata<-coldata[ , (names(coldata) != 'sizeFactor')]
  coldata$Sample<-rownames(coldata)
  conditions<-c()
  for(s in coldata[,condition]){
    if (s==control){
      conditions<-c(conditions,c("control"))
    }
    else {
      conditions<-c(conditions,c("case"))
    }
  }
  coldata$Condition<-conditions
  coldata$Time<-as.numeric(as.character(coldata$Time))
  if (!is.null(dispersions(dds))) dispersion<-impulseDispersion(dds)
  coldata<-as.data.frame(coldata)
  result<-runImpulseDE2(matCountData = counts(dds), dfAnnotation = coldata,
                boolCaseCtrl = TRUE, vecConfounders = confounders, scaNProc = threads,
                scaQThres = fdr, vecDispersionsExternal = dispersion,
                # Computes size factors itself (despite specifiing them) but they match to the DeSeq2 ones
                #vecSizeFactorsExternal = sizeFactors(dds), 
                boolVerbose = TRUE)
  result<-result$dfImpulseDE2Results
  de_filter<-result[result$padj<=fdr & ! is.na(result$padj),]
  if (is.null(path)) return(de_filter)
  else {
    conditions<-unique(coldata[,condition])
    cases<-conditions[conditions!=control]
    write.table(de_filter,file=paste(path,'DE_',paste(cases,collapse=''),'_ref_',control,'_t',paste(unique(coldata$Time),collapse='h'),'h',
                                     '_padj',fdr,'.tsv',sep='') ,sep='\t',row.names = FALSE)
  }
}

impulseDispersion<-function(dds){
  # Taken from ImpulseDe2 library function runDESeq2
  #************************************************************
  vecDispersionsInv <- mcols(dds)$dispersion
  # Catch dispersion trend outliers at the upper boundary
  # (alpha = 20 ->large variance)
  # which contain zero measurements: 
  # The zeros throw off the dispersion estimation 
  # in DESeq2 which may converge to the upper bound 
  # even though the obesrved variance is small.
  # Avoid outlier handling and replace estimates by 
  # MAP which is more stable in these cases.
  # Note: the upper bound 20 is not exactly reached - 
  # there is numeric uncertainty here - use > rather than ==
  vecindDESeq2HighOutliesFailure <- !is.na(mcols(dds)$dispOutlier) & 
    mcols(dds)$dispOutlier==TRUE &
    mcols(dds)$dispersion>(20-10^(-5)) & 
    apply(counts(dds), 1, function(gene) any(gene==0) )
  vecDispersionsInv[vecindDESeq2HighOutliesFailure] <- 
    mcols(dds)$dispMAP[vecindDESeq2HighOutliesFailure]
  if(sum(vecindDESeq2HighOutliesFailure)>0){
    print(paste0("Corrected ", sum(vecindDESeq2HighOutliesFailure),
                 " DESEq2 dispersion estimates which ",
                 "to avoid variance overestimation and loss of ",
                 "discriminatory power for model selection."))
  }
  # DESeq2 uses alpha=1/phi as dispersion
  vecDispersions <- 1/vecDispersionsInv
  names(vecDispersions) <- rownames(dds)
  return (vecDispersions)
}

# Run Impulse with dispersion factors from modified design matrix

runImpulseCustomDispersion<-function(conditions,genes,times,case,control='AX4',main_lvl='Strain',nested_dispersion=c('Time'),
                           confounder_impulse=NULL,fdr=0.05,path=NULL,threads=THREADS){

  # Conditions (M*D), genes (G*M) - dataframes
  # times - use only these timepoints
  # case - from column Strain, which to use/subset
  # control - from column Strain, which to use/subset, also use as control in Impulse
  # main_lvl - for dispersion and impulse designs, column with cases and controls
  # nested_dispersion - use as nested confounders in dispersion estimation
  # confounder_impulse - use in impulse
  # fdr - filter in impulse
  # path - if not null save  to path (adds file name). if null returns result
  conditions<-conditions[order(conditions$Time),]
  genes<-genes[,rownames(conditions)]
  deseq_data<-buildDDS(conditions=conditions,genes=genes,t=times,case=case,ref=control,design=~1,main_lvl=main_lvl,coldata=NULL,filter = 1,set_main_lvl=FALSE)
  coldata<-deseq_data$coldata
  dds<-deseq_data$dds
  # Can Stage/Replicate be converted to nested factors
  model<-nestedModelDDSData(coldata=coldata,main_lvl=main_lvl,nested=nested_dispersion)
  dds<-estimateDDSScaling(dds,model)
  return(impulseCustomDispersion(dds,confounders=confounder_impulse,condition=main_lvl,fdr=fdr,control=control,path=path,
                                 threads=threads))
}

# Test DE with Impulse without batches, with Impulse provided dispersion
deImpulse <- function(control='AX4',case,padj=0.05,threads=THREADS,genes,conditions,times_DE,path=NaN) {
  genesSub<-as.matrix(genes[,(conditions$Strain %in% c(case,control)) & 
                              (conditions$Time %in% times_DE)])
  conditionsSub<-conditions[(conditions$Strain %in% c(case,control)) & 
                              (conditions$Time %in% times_DE ),]
  
  #Annotate as case/control
  case<-c()
  for(s in conditionsSub$Strain){
    if (s==control){
      case<-c(case,c("control"))
    }
    else {
      case<-c(case,c("case"))
    }
  }
  #time - double; condition,sample
  anno<-data.frame(Sample=rownames(conditionsSub), Condition=case,Time=as.numeric(as.character(conditionsSub$Time)),stringsAsFactors = FALSE)
  rownames(anno)<-anno$Sample
  de <- runImpulseDE2(matCountData=genesSub,dfAnnotation=anno,boolCaseCtrl= TRUE,scaNProc= threads) 
  de_filter<-de$dfImpulseDE2Results[de$dfImpulseDE2Results$padj<=padj & ! is.na(de$dfImpulseDE2Results$padj),]
  if (is.nan(path)) return(de_filter)
  else  {
    write.table(de_filter,file=paste(path,'DE_',paste(case,collapse = ''),'_ref_',control,'_t',paste(timesDE,collapse='h'),'h',
                                     '_padj',padj,'.tsv',sep=''), sep='\t',row.names = FALSE)
  }
}


