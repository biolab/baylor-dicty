#*********Sample on server to get smaller file for testing the code
import h5py
import random
import pandas as pd
import numpy as np
file=h5py.File('scaledata.h5','r')
matrix_h5=file.get('integrated/matrix')
n=matrix_h5.shape[0]
sample=random.sample(range(0,n),500)
sample.sort()
# Turned around as R saves it turned compared to python
data=pd.DataFrame(matrix_h5[sample,:])
# As turned
data.columns=[name.decode() for name in file.get('integrated/rownames')]
data.index=[name.decode() for name in file.get('integrated/colnames')[sample]]
data.to_csv('scaledata_sample.tsv',sep='\t')
#**********************
import pandas as pd
import scipy as sp
from scipy.io import mmwrite
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from collections import OrderedDict
from sklearn.metrics import precision_recall_fscore_support as score

from networks.library_regulons import make_tsne

data_path = '/home/karin/Documents/retinal/data/counts/'

data=pd.read_table(data_path+'scaledata_sample.tsv',index_col=0)
col_data=pd.read_table(data_path+'passedQC_cellData.tsv',index_col=0)

data1=data.iloc[:300,:]
tsne1=make_tsne(data1,perplexities_range=[50,500],exaggerations=[7,1.2],momentums=[0.7,0.94])
#plot_tsne([tsne])
col_data1=col_data.loc[data1.index,:]

data2=data.iloc[300:,:]
tsne2 = tsne1.prepare_partial(data2,initialization="median",k=25)
col_data2=col_data.loc[data2.index,:]

def plot_tsne(tsnes: list, classes: list = None, names: list = None, legend: bool = False,
              plotting_params: dict = {'s': 1}):
    """
    Plot tsne embedding
    :param tsne: List of embeddings, as returned by make_tsne
    :param classes: List of class annotations (dict), one per tsne.
    If not None colour each item in tSNE embedding by class.
    Keys: names matching names of tSNE embedding, values: class
    :param names: List of lists. Each list contains names for items in corresponding tSNE embedding.
    :param legend: Should legend be added
    :param plotting_params: plt.scatter parameters. Can be: 1.) List with dicts (single or nested, as below) for
    each tsne,  2.) dict with class names as keys and parameters dicts as values, 3.) dict of parameters.
    :return:
    """
    if classes is None:
        data = pd.DataFrame()
        for tsne in tsnes:
            x = [x[0] for x in tsne]
            y = [x[1] for x in tsne]
            data = data.append(pd.DataFrame({'x': x, 'y': y}))
        plt.scatter(data['x'],data['y'] , alpha=0.5, **plotting_params)
    else:
        if len(tsnes) != len(names):
            raise ValueError('N of tSNEs must match N of their name lists')
        data = pd.DataFrame()
        for tsne, name, group in zip(tsnes, names, range(len(tsnes))):
            x = [x[0] for x in tsne]
            y = [x[1] for x in tsne]
            data = data.append(pd.DataFrame({'x': x, 'y': y, 'name': name, 'group': [group] * len(x)}))
        if names is not None and classes is not None:
            classes_extended = []
            for row in data.iterrows():
                row=row[1]
                group_classes=classes[int(row['group'])]
                name=row['name']
                if name in group_classes.keys():
                    classes_extended.append(group_classes[name])
                else:
                    classes_extended.append('NaN')
            data['class']=classes_extended
        class_names = set(data['class'])
        all_colours = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                       '#bcf60c',
                       '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000',
                       '#ffd8b1',
                       '#000075', '#808080', '#000000']
        all_colours = all_colours * (len(class_names) // len(all_colours) + 1)
        selected_colours = random.sample(all_colours, len(class_names))
        # colour_idx = range(len(class_names))
        colour_dict = dict(zip(class_names, selected_colours))

        fig = plt.figure()
        ax = plt.subplot(111)
        for group_name,group_data in data.groupby('group'):
            for class_name,class_data in group_data.groupby('class'):
                plotting_params_point=[]
                if isinstance(plotting_params,list):
                    plotting_params_group=plotting_params[int(group_name)]
                    plotting_params_class=plotting_params_group
                else:
                    plotting_params_class = plotting_params
                if isinstance(list(plotting_params_class.values())[0], dict):
                    plotting_params_class = plotting_params_class[class_name]
                else:
                    plotting_params_class = plotting_params_class
                ax.scatter(class_data['x'],class_data['y'],
                       c=[colour_dict[class_name] for class_name in class_data['class']],
                       label=class_name, **plotting_params_class)
        if legend:
            handles, labels = fig.gca().get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))

            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            legend = plt.legend(by_label.values(), by_label.keys(),loc='center left', bbox_to_anchor=(1, 0.5))
            for handle in legend.legendHandles:
                handle._sizes = [10]
                handle.set_alpha(1)

plot_tsne([tsne1,tsne2], classes=[dict(zip(col_data1.index,col_data1['cell_type'])),
                                  dict(zip(col_data2.index,col_data2['cell_type']))
                                  ], names=[data1.index,data2.index], legend=True,
              plotting_params = [{'alpha': 0.2,'s':1},{'alpha': 1,'s':1}])

#***KNN clasifier
classifier = KNeighborsClassifier(n_neighbors=10,n_jobs=4)
classifier.fit(tsne1,col_data1['cell_type'])
prediction=classifier.predict(tsne2)
truth=col_data2['cell_type']
labels=list(set(col_data2['cell_type']))
labels.sort()
precision, recall, fscore, support = score(y_true=truth, y_pred=prediction,labels=labels)
print('labels: {}'.format(labels))
print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))