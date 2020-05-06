import pandas as pd
import stages_DE.enrichment_library as enr

path_sets = '/home/karin/Documents/timeTrajectories/data/from_huston/gene_sets/'
#################
### Change Mariko's gene sets to gmt
set_files = {'cAMP': ['cAMP_gene_inf.xlsx'],
             'chemo': ['chemo_list.xlsx'],
             'cell cycle': ['cell_cycle.txt'],
             'hssA': ['hssA_sigN_57aa.txt'],
             #'psp': ['psp_pst_20200426.txt'],
             #'pst': ['psp_pst_20200426.txt'],
             'psp': ['pst_495_psp_394_FC.xlsx','psp_394_FC>2.4'],
             'pst': ['pst_495_psp_394_FC.xlsx','pst_495_FC<-2.4'],
             'TF': ['TF-list.xlsx',0],
             'histone/histone variant':[ 'TF-list.xlsx',0],
             'chromatin/centromere': ['TF-list.xlsx',0],
             'chromatin remodeling/histone modification': ['TF-list.xlsx',0],
             'TF (regulation)': ['TF-list.xlsx',0],
             'TF (general/pol)': ['TF-list.xlsx',0],
             'TF (mediator)': ['TF-list.xlsx',0]
             }
set_name = 'hssA'
file = pd.read_table(path_sets + set_files[set_name][0], encoding="ISO-8859-1")
# OR
#file = pd.read_excel(path_sets + set_files[set_name][0],sheet_name=set_files[set_name][1])

# Use this in   TF subsets (and old/full psp-pst)
#file = file.query('subset =="' + set_name + '"')
organism = '44689'
ontology = 'Custom-Baylor'
genes = file['ddb_g'].unique()
genes_eid_name = enr.name_genes_entrez(gene_names=genes, key_entrez=True, organism=int(organism))
genes_eid = genes_eid_name.keys()
print(set_name, ': all genes', len(genes), '(with repeated', file['ddb_g'].shape[0], '),', 'with EID', len(genes_eid))
print('No EID:', set(genes) - set(genes_eid_name.values()))
print('Repeated:\n', pd.DataFrame(file['ddb_g'].value_counts()).query('ddb_g > 1'))
with open(path_sets + ontology + '-' + organism + '.gmt', 'a') as fw:
    fw.write(set_name + '\t' + set_name + ',' + ontology + ',' + organism + ',' + set_name + ',_,_,_')
    for eid in genes_eid:
        fw.write('\t' + eid)
    fw.write('\n')

# Find path for Orange data and add /serverfiles/2020_04_05/gene_sets/
from orangecontrib.bioinformatics.utils import local_cache
print(local_cache)
# in unix: cp /home/karin/Documents/timeTrajectories/data/from_huston/gene_sets/Custom-Baylor-44689.gmt* /home/karin/.local/share/Orange/3.26.0.dev/bioinformatics/serverfiles/2020_04_05/gene_sets/