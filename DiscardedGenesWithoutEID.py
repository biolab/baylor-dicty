from orangecontrib.bioinformatics.dicty import phenotypes
from orangecontrib.bioinformatics.ncbi.gene import GeneMatcher

# Count of dictyBase genes and genes with EID (involved in Orange gene sets)
dicty_annotations = 0
dicty_genes = set()
orange_annotations = 0
orange_genes = set()
empty_sets = 0

gene_matcher = GeneMatcher('44689')
for phenotype, mutants in phenotypes.phenotype_mutants().items():
    gene_symbols = set(phenotypes.mutant_genes(mutant)[0] for mutant in mutants)
    dicty_annotations += len(gene_symbols)
    dicty_genes.update(gene_symbols)
    gene_matcher.genes = gene_symbols
    N_genes_set_Orange = 0
    N_genes_set_dicty = len(gene_symbols)
    for gene in gene_matcher.genes:
        if gene.gene_id is not None:
            orange_genes.add(gene.gene_id)
            N_genes_set_Orange += 1
    orange_annotations += N_genes_set_Orange
    if N_genes_set_Orange < 1 and N_genes_set_dicty > 0:
        empty_sets += 1

print('N genes with phenotype annotations in dictyBase:', len(dicty_genes),
      'and in Orange Dictybase Phenotypes:', len(orange_genes))
print('N of genes across gene sets (with genes being involved in multiple gene sets): dictyBase',
      dicty_annotations, ', Orange', orange_annotations)
print('N of gene sets that have no genes left in Orange:', empty_sets, ', out of',
      len(phenotypes.phenotype_mutants()), 'gene sets in dictyBase')
