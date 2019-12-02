import pandas as pd
import matplotlib.pyplot as plt

from networks.library_regulons import ClusterAnalyser,name_genes_entrez
from orangecontrib.bioinformatics.geneset.utils import GeneSet
from orangecontrib.bioinformatics.geneset.__init__ import (list_all, load_gene_sets)

from deR.enrichment_library import *

organism=44689
dataPath = "/home/karin/Documents/timeTrajectories/data/replicate_image/"

# Read pca loadings
loadings = pd.read_table(dataPath + 'loadings.tsv', sep='\t')
# Extract genes and top loading genes
top=abs(loadings['PC1']).sort_values(ascending=False).head(int(loadings.shape[0]*0.01)).index
entrez_dict=name_genes_entrez(gene_names=loadings.index,organism=organism,key_entrez=False)
referenceEID=convert_EID(genes=loadings.index,name_EID=entrez_dict)
topEID=convert_EID(genes=top,name_EID=entrez_dict)

# Enrichment - gene sets or go
# Gene sets enrichment
gene_sets=load_gene_sets(('GO','biological_process'),str(organism))
# TODO add p-value adjustment
for gs in gene_sets:
    res=gs.set_enrichment(reference=referenceEID,query=topEID)
    if res.p_value<0.01:
        print(gs.name,res.p_value)


# GO enrichment
enriched=ClusterAnalyser.enrichment(entrez_ids=topEID,organism=organism,slims=False)
# pd.DataFrame(top,columns=['Gene']).to_csv(
#    "/home/karin/Documents/timeTrajectories/data/replicate_image/top1percentPC1.tsv",sep='\t',index=False)
