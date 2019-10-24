import networks.functionsDENet as f
import pandas as pd
# Script parts are to be run separately as needed

dataPath = '/home/karin/Documents/timeTrajectories/data/RPKUM/combined/'
dataPathSaved = '/home/karin/Documents/timeTrajectories/data/correlations/replicates/'

genes=pd.read_csv(dataPath+'mergedGenes.tsv',sep='\t',index_col=0)
f.genesKNN(5,genes.iloc[:,:5],scaleByAxis=1,save=False)