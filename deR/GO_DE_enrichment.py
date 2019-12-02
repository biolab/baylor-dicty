import pandas as pd
import matplotlib.pyplot as plt

from networks.library_regulons import ClusterAnalyser,name_genes_entrez
from orangecontrib.bioinformatics.geneset.utils import GeneSet
from orangecontrib.bioinformatics.geneset.__init__ import (list_all, load_gene_sets)
from deR.enrichment_library import *

organism=44689
dataPathDE = "/home/karin/Documents/timeTrajectories/data/deTime/impulse_strains/"
dataPath = '/home/karin/Documents/timeTrajectories/data/RPKUM/combined/'

genes_de=pd.read_table(dataPathDE+'try_DE_tgrB1_ref_AX4_t24h14h4h0h8h6h12h16h5h2h3h20h10h1h18h_padj0.05.tsv',sep='\t')
genes = pd.read_table(dataPath + 'mergedGenes_RPKUM.tsv', sep='\t', index_col=0)

entrez_dict=name_genes_entrez(gene_names=genes.index,organism=organism,key_entrez=False)
reference_EID=convert_EID(genes=genes.index,name_EID=entrez_dict)
de_EID=convert_EID(genes_de['Gene'],name_EID=entrez_dict)

result=enrichment_in_gene_set(set_EID=de_EID, reference_EID=reference_EID, gene_set_names=[('GO','biological_process')])