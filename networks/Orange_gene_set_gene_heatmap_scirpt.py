import pandas as pd

from Orange.data import Table, Domain, ContinuousVariable, StringVariable

# Select FDR for gene sets
FDR=0.25

# Get data from Orange table and put it in a new tqable of gene sets (rows) and genes (columns)
# If gene is in gene set put 1 in table
data=in_data.metas
columns=in_data.domain.metas
columns=[column.name for column in columns]
data=pd.DataFrame(data,columns=columns)
data=data.loc[data['FDR']<=FDR,:]
gene_enrichment=pd.DataFrame()
for gene_set_data in data.iterrows():
    gene_set_data=gene_set_data[1]
    gene_set=gene_set_data['GO Term Name']
    for gene in gene_set_data['Genes'].split(','):
        gene_enrichment.loc[gene_set,gene]=1

#Replace NA with 0
gene_enrichment=gene_enrichment.fillna(0)

#Orange table
domain_columns=[]
for col in gene_enrichment.columns:
    domain_columns.append(ContinuousVariable(name=col))

meta_columns = [StringVariable(name='Gene set')]
out_data = Table.from_numpy(domain=Domain(domain_columns, metas=meta_columns), X=gene_enrichment.to_numpy(),
                            metas=pd.DataFrame(gene_enrichment.index).to_numpy())
