import matplotlib.pyplot as plt
import glob
import pandas as pd
import numpy as np

path = '/home/karin/Documents/timeTrajectories/data/deTime/de_time_impulse/'

MINP = 10 ** -323.6

# Replace low p values with MINP so that they are not -inf after log transform for plotting
files = [f for f in glob.glob(path + "DE_*tsv")]
files.sort()
padj_dict = dict()
for f in files:
    data = pd.read_table(f)
    name = f.split('DE_')[1].split('_t')[0]
    padjs = data['padj'].values
    padjs[padjs < MINP] = MINP
    padj_dict[name] = data['padj'].values

refs = ['AX4_PE_ref_AX4_FD', 'AX4_PE_ref_AX4_SE', 'AX4_SE_ref_AX4_FD']

names = list(padj_dict.keys())
padjs = [np.log10(padj_list) for padj_list in padj_dict.values()]
fig, ax = plt.subplots()
plt.boxplot(padjs)
ax.set_xticklabels(names, rotation=90, fontsize=6)
plt.ylabel('log10(padj)')

fig1, ax1 = plt.subplots()
for name, padjs in padj_dict.items():
    padjs = np.log10(padjs)
    padjs[padjs > -np.inf]
    if name in refs:
        colour = 'r'
        name='WT'
    else:
        colour = 'b'
        name='mutant'
    ax1.hist(padjs, bins=300, histtype='step', fill=None, alpha=0.5,  label=name, color=colour)
ax1.legend()
handles, labels = fig1.gca().get_legend_handles_labels()
by_label =dict(zip(labels, handles))
ax1.legend(by_label.values(), by_label.keys())
ax1.set_xlabel('log10(padj)')
ax1.set_ylabel('N DE genes')
ax1.axvline(-5,color='yellow')
ax1.set_xlim(left=-60)
