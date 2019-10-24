
import pandas as pd
import sklearn.preprocessing as pp
from pynndescent import NNDescent


class NeighbourCalculator:

    def __init__(self, genes: pd.DataFrame):
        """
        :param genes: Data frame of genes in rows and conditions in columns. Index is treated as gene names
        """
        self._genes=genes
        self._genes_scaled = pp.scale(self._genes, axis=1)
        # Does not work on less than 16 instances (eg. 16 rows)
        self._index=NNDescent(self._genes_scaled, metric='cosine', random_state=0)

    def neighbours(self,n_neighbours:int,inverse:bool) ->dict:
        """
        :param n_neighbours:
        :param inverse: Calculate most similar neighbours (False) or neighbours with inverse profile (True)
        :return: Dict with gene names as tupple keys (smaller by alphabet first) and
            values representing cosine similarity
        """
        if inverse:
            genes = self._genes_scaled * -1
        else:
            genes=self._genes_scaled
        # Can set speed-quality trade-off
        neighbours,distances = self._index.query(genes.tolist(), k=n_neighbours)
        return self.parse_neighbours(neighbours,distances)

    def parse_neighbours(self, neighbours, distances):
        parsed = dict()
        for gene in range(distances.shape[0]):
            for neighbour in range(distances.shape[1]):
                distance = distances[gene, neighbour]
                # Because of rounding the similarity may be slightly above one and distance slightly below 0
                if distance < 0:
                    if round(distance, 4) != 0:
                        print('Odd cosine distance at', gene, neighbour, ':', distance)
                    distance = 0
                similarity=1-distance
                gene2 = neighbours[gene, neighbour]
                gene_name1=self._genes.index[gene]
                gene_name2=self._genes.index[gene2]
                if gene_name1 != gene_name2:
                    if gene_name2 > gene_name1:
                        parsed[(gene_name1, gene_name2)] = similarity
                    else:
                        parsed[(gene_name2, gene_name1)] = similarity
        return parsed

