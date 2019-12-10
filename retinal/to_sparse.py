import glob
import pandas as pd
import scipy as sp
from scipy.io import mmwrite
import matplotlib.pyplot as plt
import numpy as np

data_path = '/home/karin/Documents/retinal/data/counts/'

# ********** Make sparse data
summary = []
files = [f for f in glob.glob(data_path + "*.csv.gz", recursive=True)]
for file in files:
    print('*****', file)
    matrix = pd.read_csv(file, delimiter=',', index_col=0)
    print('loaded')
    rownames = matrix.index
    colnames = matrix.columns
    cells_start = len(colnames)
    matrix = sp.sparse.csc_matrix(matrix.values)
    col_ok = []
    for col_n in range(matrix.shape[1]):
        if matrix.getcol(col_n).count_nonzero() > 500:
            col_ok.append(col_n)
    matrix = matrix[:, col_ok]
    colnames = colnames[col_ok]
    cells_end = len(colnames)
    summary.append({'file': file.split('/')[-1], 'N_cells': cells_start, 'N_cells_above500': cells_end})
    print('processed')
    mmwrite(file + '_above500_sparse', matrix)
    pd.DataFrame({'rownames': rownames}).to_csv(file + '_above500_sparseRows.tsv', sep='\t', index=False)
    pd.DataFrame({'colnames': colnames}).to_csv(file + '_above500_sparseCols.tsv', sep='\t', index=False)
pd.DataFrame(summary)

# **************Merge data
# Check that all genes are in the same order
files = [f for f in glob.glob(data_path + "*_sparseRows.tsv", recursive=True)]
order = None
for file in files:
    genes = pd.read_table(file)
    if order is None:
        order = genes
    else:
        if not order.equals(genes):
            print('not equal')
            break
        else:
            print('ok')
# Result: Was ok

# Check that cells are uniquely named in files
files = [f for f in glob.glob(data_path + "*_sparseCols.tsv", recursive=True)]
count = 0
names = set()
for file in files:
    cells = pd.read_table(file)
    count += cells.shape[0]
    names.update(list(cells['colnames']))
if count == len(names):
    print('ok')
# Result: Was ok

# Merge matrices and colnames
files = [f for f in glob.glob(data_path + "*.mtx", recursive=True)]
datas = []
names = []
for file in files:
    print(file)
    datas.append(sp.io.mmread(file))
    file_col = file.strip('.mtx') + 'Cols.tsv'
    names.extend(list(pd.read_table(file_col)['colnames']))
merged = sp.sparse.hstack(datas)
mmwrite(data_path + 'merged_above500_sparse', merged)
pd.DataFrame({'colnames': names}).to_csv(data_path + 'merged_above500_sparseCols.tsv', sep='\t', index=False)

# *********** Split by cell type
files = [f for f in glob.glob(data_path + 'broadinstitute/' + "*cells.csv", recursive=True)]
merged = sp.sparse.csc_matrix(sp.io.mmread(data_path + 'merged_above500_sparse.mtx'))
names_original = list(pd.read_table(data_path + 'merged_above500_sparseCols.tsv')['colnames'])
# Original prefixes:
# {'PerCd90S4', 'M1Fovea8', 'M2Fovea5', 'M1Fovea7', 'M2Fovea6', 'PerCd90S3', 'M1Fovea6', 'PerCd90S8', 'PerCd90PNAS1',
# 'Fovea4S3', 'PerCd90S6', 'M1Fovea2', 'PerMixedS1', 'M2Fovea2', 'M1Fovea3', 'M1Fovea4', 'M2Fovea3', 'Fovea4S1',
# 'MacaqueCD73DP2S1', 'PerCd73S1', 'M3Fovea1', 'PerCd90S7', 'Fovea4S2', 'M2Fovea8', 'M1Fovea1', 'M2Fovea4', 'M1Fovea5',
# 'M2Fovea1', 'PerCd73S2', 'PerCd90S1', 'PerCd90S5', 'PerCd73S4', 'PerCd73S3', 'PerCd90S9', 'M3Fovea3', 'M3Fovea2',
# 'MacaqueCD73DP2S2', 'PerCd90S2', 'M2Fovea7'}
# Broadinst prefixes:
# {'M4PerCD73S1', 'M1PerCD90S4', 'M1Fovea8', 'M2Fovea5', 'M1Fovea7', 'M2Fovea6', 'M1Fovea6', 'M1PerCD90S1',
# 'M2PerCD73S2', 'M1Fovea2', 'M2Fovea2', 'M4Fovea2', 'M1CD90PNA_S1', 'M1PerCD73S2', 'M1Fovea4', 'M1Fovea3', 'M4Fovea3',
# 'M2Fovea3', 'M3PerCD90S1', 'M3Fovea1', 'M4Fovea1', 'M3PerCD90S2', 'M2Fovea8', 'M2PerCD90S2', 'M1PerCD90S3', 'M1Fovea1',
# 'M2PerCD90S1', 'M2PerMixedS1', 'M2Fovea4', 'M1Fovea5', 'M2Fovea1', 'M2PerCD73S1', 'M3Fovea3', 'M1PerCD90S2',
# 'M3PerCD90S3', 'M4PerCD73S2', 'M3Fovea2', 'M1PerCD73S1', 'M2Fovea7'}
change_dict={
'PerCd90S4':'M1PerCD90S4',
'M1Fovea8':'M1Fovea8',
'M2Fovea5':'M2Fovea5',
'M1Fovea7':'M1Fovea7',
'M2Fovea6':'M2Fovea6',
'PerCd90S3':'M1PerCD90S3',
'M1Fovea6':'M1Fovea6',
'PerCd90S8':'M3PerCD90S2',
'PerCd90PNAS1':'M1CD90PNA_S1',
'Fovea4S3':'M4Fovea3',
'PerCd90S6': 'M2PerCD90S2',
'M1Fovea2':'M1Fovea2',
'PerMixedS1':'M2PerMixedS1',
'M2Fovea2':'M2Fovea2',
'M1Fovea3':'M1Fovea3',
'M1Fovea4':'M1Fovea4',
'M2Fovea3':'M2Fovea3',
'Fovea4S1':'M4Fovea1',
'MacaqueCD73DP2S1': 'M4PerCD73S1',
'PerCd73S1':'M1PerCD73S1',
'M3Fovea1':'M3Fovea1',
'PerCd90S7':'M3PerCD90S1',
'Fovea4S2': 'M4Fovea2',
'M2Fovea8':'M2Fovea8',
'M1Fovea1':'M1Fovea1',
'M2Fovea4':'M2Fovea4',
'M1Fovea5': 'M1Fovea5',
'M2Fovea1':'M2Fovea1',
'PerCd73S2':'M1PerCD73S2',
'PerCd90S1':'M1PerCD90S1',
'PerCd90S5':'M2PerCD90S1',
'PerCd73S4':'M2PerCD73S2',
'PerCd73S3':'M2PerCD73S1',
'PerCd90S9':'M3PerCD90S3',
'M3Fovea3':'M3Fovea3',
'M3Fovea2': 'M3Fovea2',
'MacaqueCD73DP2S2':'M4PerCD73S2',
'PerCd90S2':'M1PerCD90S2',
'M2Fovea7':'M2Fovea7'
}
names=[]
for name in names_original:
    name1=name.split('_')[0]
    name2=name.split('_')[1].replace('-','.')
    names.append(change_dict[name1]+'_'+name2)
for file in files:
    print(file)
    subset = list(pd.read_csv(file).columns)
    subset_idx = [names.index(cell.replace('-','.')) for cell in subset]
    subset_mtx=merged[:,subset_idx]
    file_save=file.split('/')[-1].split('.')[0]
    mmwrite(data_path +file_save+ '_above500_sparse', subset_mtx)

# **************************************************
#************ Retain cells that passed QC from the merged
# As above
merged = sp.sparse.csc_matrix(sp.io.mmread(data_path + 'merged_above500_sparse.mtx'))
names_original = list(pd.read_table(data_path + 'merged_above500_sparseCols.tsv')['colnames'])
genes=pd.read_table(data_path+'counts_RowNames.tsv')['rownames']
names=[]
for name in names_original:
    name1=name.split('_')[0]
    name2=name.split('_')[1].replace('-','.')
    names.append(change_dict[name1]+'_'+name2)

#Passed QC
files = [f for f in glob.glob(data_path  + "*namesPassedQC.csv", recursive=True)]
passed=[]
region=[]
cell_type=[]
for file in files:
    passed_sub=list(pd.read_csv(file,header=None)[0])
    passed.extend(passed_sub)
    file_parts=file.split('/')[-1].split('_')
    n_passed=len(passed_sub)
    region.extend([file_parts[1]]*n_passed)
    cell_type.extend([file_parts[2]]*n_passed)

print('Unique cells? ',len(passed)==len(set(passed)))
passed_idx = [names.index(cell.replace('-','.')) for cell in passed]
subset_mtx=sp.sparse.csr_matrix(merged[:,passed_idx])
count_genes=[]
for row_n in range(subset_mtx.shape[0]):
    count_genes.append(subset_mtx.getrow(row_n).count_nonzero())

plt.clf()
plt.hist(np.log10([count for count in count_genes if count>0]),bins=100)
plt.xlabel('Gene in log10(N) cells')
plt.ylabel('Count')
plt.savefig(data_path+'Gene_presence_distribution.png')


mean_expression=subset_mtx.mean(axis=1).flatten().tolist()[0]
plt.clf()
plt.hist(np.log10(np.array([mean for mean,count in zip(mean_expression,count_genes) if count>0])),bins=100)
plt.xlabel('log10(mean) Mean expression')
plt.ylabel('Count')
plt.savefig(data_path+'Gene_mean_distribution.png')


gene_idx=[idx for count,mean,idx in zip(count_genes,mean_expression,range(len(count_genes)))
          if count>15 and mean>0.0001]
subset_mtx2=subset_mtx[gene_idx,:]
genes_passed=genes[gene_idx]
mmwrite(data_path + 'passedQC', subset_mtx2)
pd.DataFrame(genes_passed).to_csv(data_path+'passedQC_genes.tsv', sep='\t', index=False)
pd.DataFrame({'region':region,'cell_type':cell_type},index=passed).to_csv(data_path+'passedQC_cellData.tsv',
                                                                               sep='\t', index=True)
