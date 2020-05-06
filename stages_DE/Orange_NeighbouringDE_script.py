# Count between how many neighbouring stages is the gene DE

# Set parameters displayed above the row of starts.

# Filter genes for strong enough DE between nighbouring stages with log fold change (min_abs_lFC) and FDR (max_FDR)
min_abs_lFC = 2
max_FDR = 0.01

# ************************

import pandas as pd

from Orange.data import Table, Domain, ContinuousVariable, StringVariable

data = in_data.X
index = in_data.metas.ravel()
columns = [col.name for col in in_data.domain.variables]
data = pd.DataFrame(data, index=index, columns=columns)

stages = ['no_agg', 'stream', 'lag', 'tag', 'tip', 'slug', 'mhat', 'cul', 'FB', 'yem']
for gene in data.index:
    data_gene = data.loc[gene, :]
    significant = 0
    for idx in range(len(stages) - 1):
        stage1 = stages[idx]
        stage2 = stages[idx + 1]
        comparison = stage1 + '_' + stage2
        # Stages were compared
        if data_gene.index.str.contains(comparison).any():
            if abs(data_gene[comparison + '_log2FoldChange']) >= min_abs_lFC and data_gene[
                comparison + '_FDR_overall'] <= max_FDR:
                significant += 1
    data.loc[gene, 'Significant_N'] = significant

domain_columns = []
for col in data.columns:
    domain_columns.append(ContinuousVariable(name=col))

meta_columns = [StringVariable(name='Gene')]
out_data = Table.from_numpy(domain=Domain(domain_columns, metas=meta_columns), X=data.to_numpy(),
                            metas=pd.DataFrame(data.index).to_numpy())

# **************************
# ** For filtering DE after individual stages
#*****************

# Fiter genes that are DE between the specified neighbouring stages and are not DE between too many stages

# Set parameters displayed above the row of starts.

# One of the stages, after which the DE will be analysed
# 'no_agg', 'stream', 'lag', 'tag', 'tip', 'slug', 'mhat', 'cul',
after_stage = 'stream'
# Type o filtering: 'N_DE' (gene is not DE between too many stages) or 'impulse' (impulse transition was identified after the satge)
filter = 'impulse'

# Filter genes for strong enough DE between nighbouring stages with log fold change (min_abs_lFC) and FDR (max_FDR)
min_abs_lFC = 2
max_FDR = 0.01
# If filter is 'N_DE' select only genes that are not DE in more than max_significant_n stage neighbours.
# The count of stage neighbours where a gene is DE. This is obtained from the previous Python Script,
# so the same min_abs_lFC and max_FDR should probably be used here and there.
max_significant_n = 2

# ************************

import pandas as pd
import numpy as np

from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable

stages = ['no_agg', 'stream', 'lag', 'tag', 'tip', 'slug', 'mhat', 'cul', 'FB', 'yem']

data = in_data.X
# Assumes there is a single meta variable
index = in_data.metas.ravel()
columns = [col.name for col in in_data.domain.variables]
data = pd.DataFrame(data, index=index, columns=columns)
discrete_cols = []
for var in in_data.domain.variables:
    if isinstance(var, DiscreteVariable):
        discrete_cols.append(var.name)
        data[var.name] = data[var.name].map(dict(enumerate(var.values)))

stage_idx = stages.index(after_stage)
comparison = stages[stage_idx] + '_' + stages[stage_idx + 1]
if filter == 'N_DE':
    data = data.query('abs(' + comparison + '_log2FoldChange) >= ' + str(min_abs_lFC) +
                      ' & ' + comparison + '_FDR_overall <=' + str(max_FDR) +
                      '& Significant_N <=' + str(max_significant_n))
elif filter == 'impulse':
    data = data.query('abs(' + comparison + '_log2FoldChange) >= ' + str(min_abs_lFC) +
                      ' & ' + comparison + '_FDR_overall <=' + str(max_FDR) +
                      # Discrete variables in Orange are str
                      '& ' + comparison + ' == "1"')
else:
    data = pd.DataFrame()
    print('No such filter type option.')

print('N selected genes: ', data.shape[0])

domain_columns = []
for col in data.columns:
    if col in discrete_cols:
        map_values = {v: k for k, v in enumerate(data[col].unique())}
        data[col] = data[col].map(map_values)
        domain_columns.append(
            DiscreteVariable(name=col, values=[k for k, v in sorted(map_values.items(), key=lambda item: item[1]) if
                                               not pd.isnull(k)]))
    else:
        domain_columns.append(ContinuousVariable(name=col))

meta_columns = [StringVariable(name='Gene')]
out_data = Table.from_numpy(domain=Domain(domain_columns, metas=meta_columns), X=data.to_numpy(),
                            metas=pd.DataFrame(data.index).to_numpy())
