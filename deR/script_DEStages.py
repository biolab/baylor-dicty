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

