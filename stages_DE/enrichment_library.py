from collections import OrderedDict
from statsmodels.stats.multitest import multipletests
import pandas as pd
import numpy as np
import networkx as nx
from Orange.widgets.utils.colorpalettes import ContinuousPalettes
import sklearn.preprocessing as pp
from collections import defaultdict
import warnings

from orangecontrib.bioinformatics.ncbi.gene import GeneMatcher
from orangecontrib.bioinformatics.geneset.__init__ import (list_all, load_gene_sets)
import orangecontrib.bioinformatics.go as go
from orangecontrib.bioinformatics.geneset.utils import (GeneSet, GeneSets)
from orangecontrib.bioinformatics.utils.statistics import Hypergeometric
from Orange.clustering.louvain import jaccard

ORGANISM = 44689
HYPERGEOMETRIC = Hypergeometric()


class GeneSetData:
    """
    Stores gene set with enrichment statistics
    """

    def __init__(self, gene_set: GeneSet, ontology: tuple = None, in_query: int = None):
        """
        :param gene_set: gene set for which data should be stored
        :param ontology: to which ontology the gene set belongs
        :param in_query: N of genes from set present in query
        """
        self.gene_set = gene_set
        self.ontology = ontology
        self.pval = None
        self.padj = None
        self.in_query = in_query
        self.in_reference = None


def name_genes_entrez(gene_names: list, key_entrez: bool, organism: int = ORGANISM) -> dict:
    """
    Add entrez id to each gene name
    :param gene_names: Gene names (eg. from dictyBase)
    :param organism: organism ID
    :param key_entrez: True: Entrez IDs as keys and names as values, False: vice versa
    :return: Dict of gene names and matching Entres IDs for genes that have Entrez ID
    """
    entrez_names = dict()
    matcher = GeneMatcher(organism)
    matcher.genes = gene_names
    for gene in matcher.genes:
        name = gene.input_identifier
        entrez = gene.gene_id
        if entrez is not None:
            if key_entrez:
                entrez_names[entrez] = name
            else:
                entrez_names[name] = entrez
    return entrez_names


def GO_enrichment(entrez_ids: list, organism: int = ORGANISM, fdr=0.25, slims: bool = True,
                  aspect: str = None) -> OrderedDict:
    """
    Calulate onthology enrichment for list of genes
    :param entrez_ids: entrez IDs of gene group to be analysed for enrichemnt
    :param organism: organism ID
    :param fdr: For retention of enriched gene sets
    :param slims: From Orange Annotations
    :param aspect: Which GO aspect to use. From Orange Annotations: None: all, 'Process', 'Function', 'Component'
    :return: Dict: key ontology term, value FDR. Sorted by FDR ascendingly.
    """
    anno = go.Annotations(organism)
    enrichment = anno.get_enriched_terms(entrez_ids, slims_only=slims, aspect=aspect)
    filtered = go.filter_by_p_value(enrichment, fdr)
    enriched_data = dict()
    for go_id, data in filtered.items():
        terms = anno.get_annotations_by_go_id(go_id)
        for term in terms:
            if term.go_id == go_id:
                padj = data[1]
                enriched_data[term.go_term] = padj
                break
    enriched_data = OrderedDict(sorted(enriched_data.items(), key=lambda x: x[1]))
    return enriched_data


def gene_set_enrichment(query_EID: set, reference_EID: set, gene_set_names: list = None, organism=ORGANISM,
                        padj_threshold: float = None, output: str = None, go_slims: bool = False,
                        gene_sets_ontology: dict = None, set_sizes: tuple = (-np.inf, np.inf), min_overlap: int = None):
    """
    Calculate enrichment for specified gene set ontologies. Padj is performed on combined results of all ontologies.
    Prior to p value calculation removes gene sets that do not overlap with query.
    :param query_EID: Query Ensembl IDs
    :param reference_EID: Reference Ensembl IDs
    :param gene_set_names: Onthologies for which to calculaten enrichment
    :param organism: Organism ID
    :param padj_threshold: Remove all gene sets with padj below threshold
    :param output: By default returns DataSetData object, if 'name': a dict with gene set name as key and padj as value,
    'name_ontology': a dict with (gene set name, ontology) tuple as key and padj as value
    :param go_slims: For Go use only generic slims if gene_sets_ontology not given
    :param gene_sets_ontology: Dict with keys ontology names and values gene sets. Use this instead of obtaining gene
    sets based on gene_set_names, go_slims and organism
    :param set_sizes: Use only gene sets with size greater or equal to 1st element of set_sizes and smaller or equal to
    2nd element of set_sizes, default is -inf,inf
    :param min_overlap: After padj calculation filter out gene sets that have ovarlap with query below min_overlap
    :return: List of GeneSetData objects
    """
    if gene_sets_ontology is None and gene_set_names is None:
        raise ValueError('Either gene_sets or gene_set_names must be specified')
    enriched = []
    if gene_sets_ontology is None:
        gene_sets_ontology = get_gene_sets(gene_set_names=gene_set_names, organism=organism, go_slims=go_slims)
    for gene_set_name, gene_sets in gene_sets_ontology.items():
        for gene_set in gene_sets:
            gene_set_size = len(gene_set.genes)
            if set_sizes[0] <= gene_set_size <= set_sizes[1]:
                intersect_query = len(gene_set.genes.intersection(query_EID))
                intersect_reference = len(gene_set.genes.intersection(reference_EID))
                if intersect_query > 0:
                    result = gene_set.set_enrichment(reference=reference_EID, query=query_EID)
                    data = GeneSetData(gene_set=gene_set, ontology=gene_set_name)
                    data.pval = result.p_value
                    data.in_query = intersect_query
                    data.in_reference = intersect_reference
                    enriched.append(data)
    if len(enriched) > 0:
        compute_padj(enriched)
        if padj_threshold is not None:
            enriched = [data for data in enriched if data.padj <= padj_threshold]
        if min_overlap is not None:
            enriched = [data for data in enriched if data.in_query >= min_overlap]
        if output == 'name':
            enriched = dict([(data.gene_set.name, data.padj) for data in enriched])
        elif output == 'name_ontology':
            enriched = dict([((data.gene_set.name, data.ontology), data.padj) for data in enriched])
    return enriched


def get_gene_sets(gene_set_names: list, organism: str = ORGANISM, go_slims: bool = False,
                  set_sizes: tuple = (-np.inf, np.inf)) -> dict:
    """
    Get all gene sets.
    :param gene_set_names: Names of ontologies for which to get gene sets (as returned by list_gene_sets)
    :param organism: organism id
    :param go_slims: If ontology type (first element from tuples retured by list_gene_sets) is GO then output
    only gene sets that are in 'goslim_generic'
    :param set_sizes: Use only gene sets with size greater or equal to 1st element of set_sizes and smaller or equal to
    2nd element of set_sizes, default is -inf,inf
    :return: Dict with key is ontology name and values are its GeneSet objects
    """
    gene_set_ontology = dict()
    if go_slims:
        anno = go.Annotations(organism)
        anno._ensure_ontology()
        anno._ontology.set_slims_subset('goslim_generic')
        slims = anno._ontology.slims_subset
    for gene_set_name in gene_set_names:
        gene_sets = load_gene_sets(gene_set_name, str(organism))
        gene_sets = [gene_set for gene_set in gene_sets if set_sizes[0] <= len(gene_set.genes) <= set_sizes[1]]
        if go_slims and gene_set_name[0] == 'GO':
            gene_sets = [gene_set for gene_set in gene_sets if gene_set.gs_id in slims]
        gene_set_ontology[gene_set_name] = gene_sets
    return gene_set_ontology


def rgb_hex(rgb):
    return '#%02x%02x%02x' % (rgb[0], rgb[1], rgb[2])

def overlap_coefficient(set1,set2):
    overlap=len(set1 & set2)
    min_size=min(len(set1),len(set2))
    return overlap/min_size

def enrichment_map(enriched: list, ax: tuple,query_size,
                   max_col=-np.log10(10**(-10)),min_col=-np.log10(0.25),
                   node_sizes=(50,2000),fontsize=10, min_overlap=0):
    """
    Plot enrichment map for enriched gene sets
    :param enriched: List of enriched GeneSetData objects
    :param ax: plt axis
    """
    # Colour palette
    palette = ContinuousPalettes['linear_viridis']
    padjs = [-np.log10(data.padj) for data in enriched]
    # Smallest palette colour is smallest -log10 padj
    if min_col is None:
        min_col = min(padjs)
    if max_col is None:
        max_col = max(padjs)

    # Size palette
    size_palette = pp.MinMaxScaler(feature_range=node_sizes)
    size_palette.fit(np.array([0,1]).reshape(-1,1))

    # Make nodes
    nodes=[]
    for data in enriched:
        name= data.gene_set.name
        colour= rgb_hex(palette.value_to_color(x=-np.log10(data.padj), low=min_col, high=max_col))
        ratio_query=data.in_query/query_size
        if ratio_query>1:
            warnings.warn('Query size was set too small (in_query>query). Setting ratio to 1.')
            ratio_query=1
        size= size_palette.transform(np.array([ratio_query]).reshape(-1,1))[0][0]
        nodes.append((name,{'color':colour,'size':size}))

    # Make graph (gs overlaps) and directed graph (specify subset)
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph_directed = nx.DiGraph()
    graph_directed.add_nodes_from(nodes)

    # Compare enriched gene sets
    for i in range(0, len(enriched) - 1):
        for j in range(i + 1, len(enriched)):
            set1, set2 = enriched[i], enriched[j]
            overlap = len(set1.gene_set.genes & set2.gene_set.genes)
            #overlap_weight=jaccard(set1.gene_set.genes,set2.gene_set.genes)
            overlap_weight = overlap_coefficient(set1.gene_set.genes, set2.gene_set.genes)

            if overlap_weight > min_overlap:
                graph.add_edge(set1.gene_set.name, set2.gene_set.name, weight=overlap_weight)
                size1=len(set1.gene_set.genes)
                size2=len(set2.gene_set.genes)
                min_size = min([size1, size2])
                #None if equal sizes
                smaller_order=None
                if size1<size2:
                    smaller_order=1
                elif size1>size2:
                    smaller_order=-1
                if overlap == min_size and smaller_order is not None:
                    node1,node2=[set1,set2][::smaller_order]
                    graph_directed.add_edge(node1.gene_set.name,node2.gene_set.name,weight=overlap_weight)

    node_attrs = nx_attr_list_node(graph, ['color', 'size'])
    node_attrs['label']=[node.replace(' ','\n') for node in node_attrs['node']]
    edge_attrs = nx_attr_list_edge(graph, ['weight'])
    edge_attrs_directed = nx_attr_list_edge(graph_directed, ['weight'])

    layout=nx.drawing.nx_agraph.graphviz_layout(graph,prog='fdp')
    node_attrs['layout']=[layout[node] for node in node_attrs['node']]

    nx.draw_networkx_edges(graph_directed, pos=layout, nodelist=node_attrs['node'],
                           edgelist=edge_attrs_directed['edge'], width=edge_attrs_directed['weight'],
                           arrowsize=15, alpha=0.3, min_target_margin=30,ax=ax)
    nx.draw_networkx_edges(graph, pos=layout,nodelist=node_attrs['node'],
                     edgelist=edge_attrs['edge'],width=edge_attrs['weight'],alpha=0.5,ax=ax)
    nx.draw_networkx_nodes(graph, nodelist=node_attrs['node'], pos=layout,
                     node_color=node_attrs['color'], node_size=node_attrs['size'], alpha=0.6,ax=ax)
    nx.draw_networkx_labels(graph, nodelist=node_attrs['node'], pos=layout,
                            labels=dict(zip(node_attrs['node'],node_attrs['label'])),font_size=fontsize,ax=ax)
    ax.axis('off')
    ax.margins(0.2, 0.1)


def nx_attr_list_node(graph,attrs_node):
    attrs=defaultdict(list)
    for node,node_attrs in graph.nodes(data=True):
        attrs['node'].append(node)
        for attr in attrs_node:
            if attr in node_attrs.keys():
                attrs[attr].append(node_attrs[attr])
            else:
                attrs[attr].append(None)
    return attrs

def nx_attr_list_edge(graph, attrs_edge):
    attrs = defaultdict(list)
    for node1,node2,edge_attrs in graph.edges(data=True):
        attrs['edge'].append((node1,node2))
        for attr in attrs_edge:
            if attr in edge_attrs.keys():
                attrs[attr].append(edge_attrs[attr])
            else:
                attrs[attr].append(None)
    return attrs





# Not useful as same to normal enrichment due to simetry (Combinatorial identities)
# https://en.wikipedia.org/wiki/Hypergeometric_distribution
# def enrichment_in_gene_set(set_EID: set, reference_EID: set, gene_set_names: list, organism=ORGANISM):
# # Set EID - e.g. DE genes
#     enriched = []
#     for gene_set_name in gene_set_names:
#         gene_sets = load_gene_sets(gene_set_name, str(organism))
#         for gene_set in gene_sets:
#             intersect_query = len(gene_set.genes.intersection(set_EID))
#             if intersect_query > 0:
#                 intersect_reference=len(reference_EID.intersection(set_EID))
#                 pval = HYPERGEOMETRIC.p_value(k=intersect_query, N=len(reference_EID), m=intersect_reference,
#                                               n=len(gene_set.genes))
#                 data = GeneSetData(gene_set=gene_set, ontology=gene_set_name)
#                 data.pval = pval
#                 data.in_query = intersect_query
#                 enriched.append(data)
#     compute_padj(enriched)
#     return enriched


def compute_padj(data):
    """
    Add padj (FDR Benjamini-Hochberg) values to list of GeneSetData objects, based on their p values
    :param data: list of GeneSetData objects
    """
    pvals = []
    for set_data in data:
        pvals.append(set_data.pval)
    padjvals = multipletests(pvals=pvals, method='fdr_bh', is_sorted=False)[1]
    for data, padj in zip(data, padjvals):
        data.padj = padj


def filter_enrichment_data_top(data: list, max_padj: float) -> list:
    """
    Return only enriched data with padj bellow threshold
    :param data: list of GeneSetData
    :param max_padj: max padj of gene set to be returned
    :return: list of gene sets with padj below threshold
    """
    enriched = []
    for gene_set in data:
        if gene_set.padj <= max_padj:
            enriched.append(gene_set)
    return enriched


def sort_gsdata_pval(data: list) -> list:
    """
    Sort GeneSetDataCorrelation list based on pval
    :param data: list of GeneSetData
    :return: sorted list of GeneSetData
    """
    return sorted(data, key=lambda x: x.pval)


def sort_gsdata_padj(data: list) -> list:
    """
    Sort GeneSetDataCorrelation list based on padj
    :param data: list of GeneSetData
    :return: sorted list of GeneSetData
    """
    return sorted(data, key=lambda x: x.padj)


def sort_gsdata_n_genes(data: list) -> list:
    """
    Sort GeneSetDataCorrelation list based on number orf genes in GeneSets
    :param data: list of GeneSetData
    :return: sorted list of GeneSetData
    """
    return sorted(data, key=lambda x: len(x.gene_set.genes))


def sort_gsdata_n_query(data: list) -> list:
    """
    Sort GeneSetDataCorrelation list based on N of genes in query that belong to gene set
    :param data: list of GeneSetData
    :return: sorted list of GeneSetData
    """
    return sorted(data, key=lambda x: x.in_query)


def gene_set_data_to_df(data):
    """
    Convert list of GeneSetData objects to DataFrame
    :param data: list of GeneSetData objects
    :return: DataFrame with columns: name (of gene set), pval, padj, n_genes_set (n genes in gene set),
    n_set_query (n genes from gene set present in query), ontology
    """
    summary = []
    for set_data in data:
        summary.append({'name': set_data.gene_set.name, 'pval': set_data.pval, 'padj': set_data.padj,
                        'n_genes_set': len(set_data.gene_set.genes), 'n_set_query': set_data.in_query,
                        'ontology': set_data.ontology})
    return pd.DataFrame(summary)


def convert_EID(genes: iter, name_EID: dict) -> set:
    """
    Convert gene names to EID based on name dict
    :param genes: gene names
    :param name_EID: dict where keys are names and values are EIDs
    :return: set of gene EIDs
    """
    return set(name_EID[gene] for gene in genes if gene in name_EID.keys())
