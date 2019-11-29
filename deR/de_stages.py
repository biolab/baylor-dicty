import glob

from collections import defaultdict
from statistics import mean
import numpy as np
import pandas as pd

dataPathSaved = '/home/karin/Documents/timeTrajectories/data/deTime/diffTimeTry/'

BP = 'before_padj'
AP = 'after_padj'
SFC = 'lastSame_lFC'
DFC = 'firstDiff_lFC'
SP = 'lastSame_padj'
DP = 'firstDiff_padj'
B = '/before_'
A = '/after_'
S = '/lastSame_'
D = '/firstDiff_'

# Read data and make summary DF shape/backbone
conditions = pd.read_table(dataPathSaved + 'conditionsTest.tsv', index_col=0)
mutant_conditions = conditions[conditions.Strain != 'AX4']
by_stage = pd.DataFrame(mutant_conditions.groupby('Strain').max()['Stage']).groupby('Stage')
last_stages = dict()
for stage, strains in by_stage:
    last_stages[stage] = list(strains.index)
stages = list(last_stages.keys())
stages.sort()
stages_df_list = stages * 6
n_stages = len(stages)
summary_df_list = [BP] * n_stages + [AP] * n_stages + [SP] * n_stages + [DP] * n_stages+ [SFC] * n_stages + [
    DFC] * n_stages
summary_df = pd.DataFrame({'Summary': summary_df_list, 'Stage': stages_df_list})


def add_to_summary_dict(summary: defaultdict, files: list, filter_files: str, key_col, value_col):
    """
    Adds to specified dict from table file, adds each row
    :param summary: list valued dict
    :param files: list of files
    :param filter_files: filter string that will result in single file being extracted from files and used, must be tsv
    :param key_col: add element from this column of the row to key, if name adds index
    :param value_col: add element from this column of the row to value list
    """
    file = pd.read_table([f for f in files if filter_files in f][0])
    for row in file.iterrows():
        row = row[1]
        if key_col == 'name':
            summary[row.name].append(row[value_col])
        else:
            summary[row[key_col]].append(row[value_col])


def add_to_summary_df(summary_dict:dict, summary,stage):
    """
    Adds to summary_df from dictionary, key=column, value=adds to df
    :param summary_dict: Dict with column and value data
    :param summary: Uses to select row, finds row in column Summary that matches this
    :param stage: Uses to select row, finds row in column Stage that matches this
    """
    for gene, stat in summary_dict.items():
        row = summary_df[(summary_df['Summary'] == summary) & (summary_df['Stage'] == stage)].index[0]
        if gene not in summary_df.columns:
            summary_df[gene] = np.zeros(summary_df.shape[0])
        summary_df.loc[row, gene] = stat


# For each stage parse the data, retaining only genes present in all strains that finish developing at the same stage.
# Averages logFC and padj over these strains. Convert padj to -log10(padj)
files = [f for f in glob.glob(dataPathSaved + "*.tsv", recursive=True)]
for stage, strains in last_stages.items():
    n_strains = len(strains)

    # Dictionaries for different data types
    before_padj = defaultdict(list)
    after_padj = defaultdict(list)
    lastSame_FC = defaultdict(list)
    firstDiff_FC = defaultdict(list)
    lastSame_padj = defaultdict(list)
    firstDiff_padj = defaultdict(list)

    # Add data from each strain
    for strain in strains:
        strain_files = [f for f in files if strain in f]
        add_to_summary_dict(summary=before_padj, files=strain_files, filter_files=B, key_col='Gene', value_col='padj')
        add_to_summary_dict(summary=after_padj, files=strain_files, filter_files=A, key_col='Gene', value_col='padj')
        add_to_summary_dict(summary=lastSame_FC, files=strain_files, filter_files=S, key_col='name',
                            value_col='log2FoldChange')
        add_to_summary_dict(summary=firstDiff_FC, files=strain_files, filter_files=D, key_col='name',
                            value_col='log2FoldChange')
        add_to_summary_dict(summary=lastSame_padj, files=strain_files, filter_files=S, key_col='name',
                            value_col='padj')
        add_to_summary_dict(summary=firstDiff_padj, files=strain_files, filter_files=D, key_col='name',
                            value_col='padj')

    # Combine strains data for this stage - average if present in all strains
    before_padj = {key: -np.log10(mean(value)) for (key, value) in before_padj.items() if len(value) == n_strains}
    after_padj = {key: -np.log10(mean(value)) for (key, value) in after_padj.items() if len(value) == n_strains}
    lastSame_FC = {key: mean(value) for (key, value) in lastSame_FC.items() if len(value) == n_strains}
    firstDiff_FC = {key: mean(value) for (key, value) in firstDiff_FC.items() if len(value) == n_strains}
    lastSame_padj = {key: -np.log10(mean(value)) for (key, value) in lastSame_padj.items() if len(value) == n_strains}
    firstDiff_padj = {key: -np.log10(mean(value)) for (key, value) in firstDiff_padj.items() if len(value) == n_strains}

    # Add combined data to summary dataframe
    add_to_summary_df(summary_dict=before_padj,summary=BP,stage=stage)
    add_to_summary_df(summary_dict=after_padj, summary=AP, stage=stage)
    add_to_summary_df(summary_dict=lastSame_FC, summary=SFC, stage=stage)
    add_to_summary_df(summary_dict=firstDiff_FC, summary=DFC, stage=stage)
    add_to_summary_df(summary_dict=lastSame_padj, summary=SP, stage=stage)
    add_to_summary_df(summary_dict=firstDiff_padj, summary=DP, stage=stage)

summary_df.to_csv(dataPathSaved+'summary.tab',index=False,sep='\t')

#******************************************************************
#*** Stages heatmap
n_strains=3
n_stages=3
n_times=4
matrix=pd.DataFrame(np.zeros(n_strains*n_stages,n_times))
