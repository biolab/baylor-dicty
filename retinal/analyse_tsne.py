
import pandas as pd
import scipy as sp
from scipy.io import mmwrite
import matplotlib.pyplot as plt
import numpy as np

from networks.library_regulons import make_tsne,plot_tsne

data_path = '/home/karin/Documents/retinal/data/counts/'

data = sp.sparse.csc_matrix(sp.io.mmread(data_path + 'passedQC.mtx'))
genes=pd.read_table(data_path+'passedQC_genes.tsv')
cell_data=pd.read_table(data_path+'passedQC_cellData.tsv',index_col=0)

data1=data[1:500,1:100]
genes1=genes[1:500]
cell_data1=cell_data[1:100]