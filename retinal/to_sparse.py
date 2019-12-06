import glob
import pandas as pd
import scipy as sp
from scipy.io import mmwrite

data_path='/home/karin/Documents/retinal/data/counts/try/'


files = [f for f in glob.glob(data_path + "*.csv.gz", recursive=True)]
for file in files:
    matrix=pd.read_csv(file,  delimiter=',',index_col=0)
    rownames=matrix.index
    colnames=matrix.columns
    matrix=sp.sparse.csr_matrix(matrix.values)
    mmwrite(file + '_sparse', matrix)
    pd.DataFrame({'rownames':rownames}).to_csv(file+'_sparseRows.tsv',sep='\t',index=False)
    pd.DataFrame({'colnames': colnames}).to_csv(file + '_sparseCols.tsv', sep='\t', index=False)

