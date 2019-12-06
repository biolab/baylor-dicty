import pandas as pd

data_path='/home/karin/Documents/retinal/data/'

gtf=pd.read_table(data_path+'mito_SRZ189920_MacFas_TruSeq.gtf',sep='\t',header=None)
mt_genes=set()
for row in gtf.iterrows():
    row=row[1]
    if row.iloc[0] == 'chrM':
        gene=row.iloc[8].split('gene_name "')[1].split('"; ')[0]
        mt_genes.add(gene)
pd.DataFrame({'Gene':list(mt_genes)}).to_csv(data_path+'macaque_anno_MT.tsv',sep='\t')

