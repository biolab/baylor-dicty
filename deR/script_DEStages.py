import matplotlib.pyplot as plt
import glob
import pandas as pd
import numpy as np

path = '/home/karin/Documents/timeTrajectories/data/deTime/de_time_impulse/'

FDR=10**-5

files = [f for f in glob.glob(path + "DE_*tsv")]
files.sort()
padj_dict = dict()
for f in files:
    data = pd.read_table(f)
    name = f.split('DE_')[1].split('_t')[0]
    padjs = data['padj'].values
    padjs[padjs<=FDR]
    padj_dict[name] = data['padj'].values

# How variable are  strains between replicates - plot variability as boxplots
# Stds of genes  in strains+timepoints (std of replicates for a gene) divided by mean
genes_conditions = ClusterAnalyser.merge_genes_conditions(genes=genes, conditions=conditions, matching='Measurment')
groups=genes_conditions.groupby(['Strain','Time'])
variation={strain:[] for strain in conditions['Strain'].unique()}
for group in groups:
    name=group[0][0]
    data=group[1].drop(list(conditions.columns),axis=1)
    std=(data.std()/data.mean()).dropna()
    variation[name].extend(std)
plt.boxplot(list(variation.values()))
plt.gca().set_xticklabels(list(variation.keys()),rotation=90)