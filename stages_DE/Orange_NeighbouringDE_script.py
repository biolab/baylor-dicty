min_abs_lFC=2
max_FDR=0.01

#************************

import pandas as pd

from Orange.data import Table, Domain, ContinuousVariable, StringVariable


data=in_data.X
index=in_data.metas.ravel()
columns=[col.name for col in in_data.domain.variables]
data=pd.DataFrame(data,index=index,columns=columns)

stages=['no_agg', 'stream', 'lag', 'tag', 'tip', 'slug', 'mhat', 'cul', 'FB', 'yem']
for gene in data.index:
    data_gene=data.loc[gene,:]
    significant=0
    for idx in range(len(stages)-1):
        stage1=stages[idx]
        stage2=stages[idx+1]
        comparison=stage1+'_'+stage2
        # Stages were compared
        if data_gene.index.str.contains(comparison).any():
            if abs(data_gene[comparison+'_log2FoldChange']) >= min_abs_lFC and data_gene[comparison+'_padj'] <=max_FDR:
                significant+=1
    data.loc[gene,'Significant_N']=significant

domain_columns=[]
for col in data.columns:
    domain_columns.append(ContinuousVariable(name=col))

meta_columns = [StringVariable(name='Gene')]
out_data = Table.from_numpy(domain=Domain(domain_columns, metas=meta_columns), X=data.to_numpy(),
                            metas=pd.DataFrame(data.index).to_numpy())

