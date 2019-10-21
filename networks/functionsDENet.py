import pandas as pd
from pynndescent import NNDescent
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.stats import spearmanr
import networkx as nx
import csv
from sklearn import preprocessing as pp
from sklearn.manifold import TSNE
from statistics import mean
from openTSNE import TSNEEmbedding
from openTSNE.affinity import PerplexityBasedNN
from openTSNE import initialization
import random as rand
from orangecontrib.bioinformatics.go import *
from orangecontrib.bioinformatics.ncbi.gene import GeneMatcher
import time


def savePickle(file, object):
    f = open(file, 'wb')
    pickle.dump(object, f)
    f.close()


def loadPickle(file):
    return pickle.load(open(file, 'rb'))


def formatFilePrefix(prefix):
    return prefix.replace('.', '_')


# Import table into pandas
def importTable(table):
    return pd.read_table(table, low_memory=False)


# Extract genes from tab Orange file with genes in collumns
# table - pandas table
# fromRow - in which row do genes data start
# toCollumnInclusive - to which collumn inclusively is gene expression data
def extractGenesFromTable(table, fromRow, toCollumnInclusive):
    genes = table.iloc[fromRow:, :toCollumnInclusive + 1].astype(float)
    genesT = pd.DataFrame.transpose(genes)
    genesNotNull = notNull(genesT, 1)
    return genes, genesNotNull


# Extract genes by strain based on gene table (genes in rows, 'genes') and original Orange formated 'table'
# Collumn in 'table' with strain info
# In which row does expression/strain data start in table
def genesByStrain(genes, table, collumn, strain, fromRow):
    genesWT = genes.T[(table.iloc[:, collumn:collumn + 1] == strain).values[fromRow:]].T
    genesWTN = notNull(genesWT, 1)
    return genesWT, genesWTN


# Extract genes by keyword based on gene table (genes in rows, 'genes') and original Orange formated 'table'
# Collumn in 'table' with strain info - retains if the keyword (strain) is present
# In which row does expression/strain data start in table
def genesByKeyword(genes, table, collumn, strain, fromRow):
    genesWT = genes.T[(table.iloc[:, collumn].str.contains(strain)).values[fromRow:]].T
    genesWTN = notNull(genesWT, 1)
    return genesWT, genesWTN


# Return portion of table with non zero rows/collumns
def notNull(table, axis):
    return table[(table != 0).any(axis=1)]


# Scale by rows or collumns to have mean 0 and stdev 1
# axisN = 1 if genes in rows
def normaliseGenes(genes, axisN):
    return pp.scale(genes, axis=axisN)


# Get gene neighbours with KNN and save, kN-neighbours
# Genes table on which the calculations will be done, genes in rows
# scaleByAxis - scale axis to have stdev 1 and mean 0
# filePrefix - file prefix and path to save the results
# Returns distances and neighbours data for original data and inversed profiles, respectively and number of used genes

# The first calling of index.query is slower than subsequent callings,
# although it does not seem to strongly affect neighbours
def genesKNN(kN, genes, scaleByAxis, filePrefix='', save=True, timing=False,adjustForSelf=False):
    scaled = normaliseGenes(genes, scaleByAxis)
    if timing:
        start = time.time()
    index = NNDescent(scaled, metric='cosine', random_state=0)
    if timing:
        end = str(time.time() - start)
        print("Index sec: " + end)
    if adjustForSelf:
        kN_similar = kN + 1
    else:
        kN_similar = kN
    if timing:
        start = time.time()
    resultKNN = index.query(scaled.tolist(), k=kN_similar)
    if timing:
        end = str(time.time() - start)
        print("KNN " + str(kN) + " sec: " + end)
    dist = resultKNN[1]
    neigh = resultKNN[0]
    # Inverse result
    if timing:
        start = time.time()
    inverse = scaled * -1
    resultKNNInv = index.query(inverse.tolist(), k=kN)
    if timing:
        end = str(time.time() - start)
        print("KNN " + str(kN) + " inverse sec: " + end)
    distInv = resultKNNInv[1]
    neighInv = resultKNNInv[0]
    if save:
        # Save
        filePrefix = formatFilePrefix(filePrefix)
        savePickle(filePrefix + "knnIndex.pkl", index)
        savePickle(filePrefix + 'knnResult_k' + str(kN) + '.pkl', resultKNN)
        savePickle(filePrefix + 'knnResultInverse_k' + str(kN) + '.pkl', resultKNNInv)
    nGenes = nGenesKnn(resultKNN, resultKNNInv)
    return dist, neigh, distInv, neighInv, nGenes


# Import results of KNN based on file prefix and kN (number of neighbours) - based on file names specified in genesKNN
# Return as genesKNN
def importKnnResults(filePrefix, kN):
    resultKNN = pickle.load(open(filePrefix + 'knnResult_k' + str(kN) + '.pkl', 'rb'))
    dist = resultKNN[1]
    neigh = resultKNN[0]
    resultKNNInv = pickle.load(open(filePrefix + 'knnResultInverse_k' + str(kN) + '.pkl', 'rb'))
    distInv = resultKNNInv[1]
    neighInv = resultKNNInv[0]
    nGenes = nGenesKnn(resultKNN, resultKNNInv)
    return dist, neigh, distInv, neighInv, nGenes


# Get n Genes involved in KNN, None if error
def nGenesKnn(resultKNN, resultKNNInv):
    nGenesSub = resultKNN[0].shape[0]
    nGenesInvSub = resultKNNInv[0].shape[0]
    nGenes = None
    if nGenesSub == nGenesInvSub:
        nGenes = nGenesSub
    return nGenes


def plotKnnDist(dist, fileName='', save=True):
    fig = plt.hist(dist.flatten(), bins=2000)
    plt.ylabel('Count')
    plt.xlabel('Distance')
    if save:
        plt.savefig(fileName)
    # plt.show()


# For lower diagonal matrix without diagonal
def position(j, i):
    if i == j:
        return None
    elif j > i:
        row = j
        col = i
    else:
        row = i
        col = j
    return int((row * (row + 1) / 2) + col - row)


# NO-changed: Based on KNN result make lower triangular matrix w/o diagonal (vector), denoting which elemnts are similar
# Result: 1 if not similar, less if similar
# use Dict - returns dict where value is similarity (calculated as 1-distance) and keys are gene pairs
def chooseGenePairsFromKnn(nGenesKnn, knnNeighbours, threshold, dist, neigh, distInv, neighInv,useDict=False):
    if not useDict:
        knnChosen = set()
    else:
        knnChosen=dict()
    for gene in range(nGenesKnn):
        for k in range(dist.shape[1]):
            d = dist[gene, k]
            if 0 <= d <= threshold:
                gene2 = neigh[gene, k]
                if not useDict:
                    addToknnDMatrix(gene, gene2, knnChosen)
                else:
                    addToknnDDict(gene, gene2, 1-d,knnChosen)
        for k in range(distInv.shape[1]):
            di = distInv[gene, k]
            if 0 <= di <= threshold:
                gene2i = neighInv[gene, k]
                if not useDict:
                    addToknnDMatrix(gene, gene2i, knnChosen)
                else:
                    addToknnDDict(gene, gene2i, 1 - di, knnChosen)
    return knnChosen


# Add distance to diagonal matrix, excluding diagonal elements, prioritise lower distances
def addToknnDMatrix(j, i, matrix):
    if i != j:
        if j > i:
            matrix.add((j, i))
        else:
            matrix.add((i, j))

# Add distance to diagonal matrix, excluding diagonal elements, prioritise lower distances
def addToknnDDict(j, i,value, dictionary):
    if i != j:
        if j > i:
            dictionary[(j, i)]=value
        else:
            dictionary[(i,j)]=value


# Correlations
# Singlethread  All (subset first few rows) or from knn - based on filter
# rows-only a subset for testing or nGenes included in KNN if all
# gen-genes
# knnDMatrix - matrix with distances between pairs, uses everything below 1
# Results for p and r are vectors with None, except when calculation was performed
def geneCorrelations(gen, knnChosen, filePrefix):
    # start=time.time()
    # Calculate correlation
    pScoresS = dict()
    rScoresS = dict()
    # start=time.time()
    print('All ' + str(len(knnChosen)))
    count = 1
    for pair in knnChosen:
        j = pair[0]
        i = pair[1]
        if count % 10000 == 0:
            print(count)
        count += 1
        r, p = spearmanr(gen.iloc[j], gen.iloc[i])
        pScoresS[pair] = p
        rScoresS[pair] = r
    # print(time.time()-start)

    # Save
    filePrefix = formatFilePrefix(filePrefix)
    savePickle(filePrefix + "_pScores.pkl", pScoresS)
    savePickle(filePrefix + "_rScores.pkl", rScoresS)

    return pScoresS, rScoresS


def loadCorrelationData(filePrefix):
    filePrefix = formatFilePrefix(filePrefix)
    pScores = pickle.load(open(filePrefix + '_pScores.pkl', 'rb'))
    rScores = pickle.load(open(filePrefix + '_rScores.pkl', 'rb'))
    return pScores, rScores


def loadRScores(filePrefix, named=False):
    filePrefix = formatFilePrefix(filePrefix)
    if not named:
        rScores = pickle.load(open(filePrefix + '_rScores.pkl', 'rb'))
    else:
        rScores = pickle.load(open(filePrefix + '_rScoresNamed.pkl', 'rb'))
    return rScores


# Plot corelation r with vector with Nones
def plotCorrelationR(rScoresS, fileName='', save=True):
    toPlot = []
    for r in rScoresS.values():
        toPlot.append(r)
    fig = plt.hist(toPlot, bins=200)
    plt.xlabel('Correlation coefficient')
    plt.ylabel('Count')
    plt.savefig(fileName)
    # plt.show()
    if save:
        plt.savefig(fileName)


# Find top correlations (by p value),
# add pairs with same correlation as max p in top
def topIdxs(top, scores):
    ps = list(scores.values())
    pArr = np.array(ps)
    if len(ps) > top:
        idxs = (pArr).argsort()[:top]
        maxP = pArr[idxs[len(idxs) - 1]]
    else:
        maxP = max(ps)
    idxs = dict(filter(lambda elem: elem[1] <= maxP, scores.items()))
    return idxs


# Find last still included value from top
# minMax - seek lower or upper top
def lastInTop(top, scoresValues, minMax):
    pArr = np.array(scoresValues)
    if (minMax):
        if len(scoresValues) > top:
            idxs = (pArr).argsort()[:top]
            max = pArr[idxs[len(idxs) - 1]]
        else:
            max = max(pArr)
        return max
    else:
        if len(scoresValues) > top:
            idxs = (pArr).argsort()[len(pArr) - top:]
            min = pArr[idxs[0]]
        else:
            min = min(pArr)
        return min


# Top by p value to space separated file
def saveTopCorrelPairsByP(filePrefix, pScores, rScores, gen, top):
    f = open(formatFilePrefix(filePrefix) + "_TopByP.txt", "w")
    for idx in topIdxs(top, pScores):
        j, i = idx
        f.write(
            gen.iloc[i].name + ' ' + gen.iloc[j].name + ' p ' + str(pScores[idx]) + ' r ' + str(rScores[idx]) + '\n')
        # print(gen.iloc[i].name,gen.iloc[j].name, "p", pScoresFull[idx], "r",rScoresFull[idx])
    f.close()


# Merge (sum) r scores of replicates into singe DF, use abs(r)
def mergeRScoresRep(table, genesNotNull, collumnStrains, repDict, fromRow, path, knnNumber, thresholdKnn,
                    thresholdRScores):
    rows = genesNotNull.shape[0]
    master = pd.DataFrame(np.zeros((rows, rows)))
    names = genesNotNull.index
    master.index = names
    master.columns = names
    suffix = formatFilePrefix('kN' + str(knnNumber) + 'T' + str(thresholdKnn))
    for repName, repNum in repDict.items():
        genesWT, genesWTN = genesByStrain(genesNotNull, table, collumnStrains, repName, fromRow)
        pScores, rScores = loadCorrelationData(path + 'rep' + str(repNum) + suffix)
        # rMatrix=matrixFromR(rScores,thresholdRScores,genesWTN.shape[0])
        # rNamed=nameMatrix(rMatrix,genesWTN)
        # master=addToDFFromDF(master,rNamed)
        namesDict = dictFromIndex(genesWTN)
        master = addToDFFromDict(master, rScores, namesDict, thresholdRScores)
    return master


def matrixFromR(rScores, threshold, rows):
    rThresholded = dict(filter(lambda elem: abs(elem[1]) >= threshold, rScores.items()))
    nMatrix = rows
    matrix = np.zeros((nMatrix, nMatrix)).astype(float)
    for k, v in rThresholded.items():
        j, i = k
        matrix[i, j] = abs(v)
        matrix[j, i] = abs(v)
    return matrix


def nameMatrix(matrix, genes):
    rows = genes.shape[0]
    if matrix.shape == (rows, rows):
        df = pd.DataFrame(matrix)
        df.index = genes.index
        df.columns = genes.index
        return df


def dictFromIndex(dataFrame):
    values = dataFrame.index
    keys = range(len(values))
    return dict(zip(keys, values))


# Add toAdd (m*m) to dataFrame (n*n), associations/distances if indices of toAdd match dataFrame
# Add absolute value
# Sum added to previous value in dataFrame
def addToDFFromDF(dataFrame, toAdd):
    if all(x in dataFrame.index for x in toAdd.index):
        for j in toAdd.index:
            for i in toAdd.columns:
                start = dataFrame.loc[j][i]
                dataFrame.loc[j][i] = start + abs(toAdd.loc[j][i])
        return dataFrame


# Add values from toAdd to dataFrame (n*n),
# where toAdd is dict with keys specifying position (add to both upper and lower matrix dataFrame)
# and names dict specifying names in dataFrame
# Thresholding is based on absolute values
# Adds absolute value
def addToDFFromDict(dataFrame, toAdd, names, threshold):
    toAddNames = [item for sublist in toAdd for item in sublist]
    if all(x in dataFrame.index for x in names.values()) and all(x in names.keys() for x in toAddNames):
        for pair, val in toAdd.items():
            valAbs = abs(val)
            if valAbs >= threshold:
                j = names[pair[0]]
                i = names[pair[1]]
                start1 = dataFrame.loc[j][i]
                dataFrame.loc[j][i] = start1 + valAbs
                if i != j:
                    start2 = dataFrame.loc[i][j]
                    dataFrame.loc[i][j] = start2 + valAbs
        return dataFrame


#
def prunedDF(dataFrame, saveAsGraph, file=''):
    if isinstance(dataFrame, pd.DataFrame):
        graph = nx.from_pandas_adjacency(dataFrame)
        graph.remove_nodes_from(list(nx.isolates(graph)))
        if saveAsGraph:
            saveGraph(graph, file)
        return nx.to_pandas_adjacency(graph)


# Add to graph (nx.MultiGraph()) from rScores
# Inclusive threshold
# Add abs(r) to weight, strain to strain, replicateNumber to replicate
def buildGraph(graph, strain, replicateNumber, rScores, threshold, genesForNames, threshRorP=True, pScores=''):
    names = genesForNames.index
    for pair, r in rScores.items():
        w = abs(r)
        if threshRorP:
            if w >= threshold:
                graph.add_edge(names[pair[0]], names[pair[1]], weight=w, strain=strain, replicate=replicateNumber)
        else:
            p = pScores[pair]
            if p <= threshold:
                graph.add_edge(names[pair[0]], names[pair[1]], weight=w, strain=strain, replicate=replicateNumber)


def makeMultiGraph():
    return nx.MultiGraph()


# Make Graph from dict of type:{(node1,node2):weight}
def graphFromWeightDict(dict):
    graph = nx.Graph()
    for (node1, node2), w in dict.items():
        graph.add_edge(node1, node2, weight=w)
    return graph


# This works only for remouving edges that are single  (not a multiedge)-removes 1 copy
def removeSingleSampleEdges(graph):
    edges = set(graph.edges())
    for edge in edges:
        multi = len(graph.get_edge_data(edge[0], edge[1]))
        if multi < 2:
            graph.remove_edge(edge[0], edge[1])


# Remove edge of a sample if the edge is only for 1 replicate
def removeSingleReplicateEdges(graph):
    edges = set(graph.edges())
    for edge in edges:
        subEdges = dict()
        for key, subEdge in graph.get_edge_data(edge[0], edge[1]).items():
            strain = subEdge['strain']
            if strain in subEdges.keys():
                subEdges[strain].append(key)
            else:
                subEdges[strain] = [key]
        for strain in subEdges.keys():
            if len(subEdges[strain]) < 2:
                graph.remove_edge(edge[0], edge[1], subEdges[strain][0])


# Remove all edges between two points if they belong to less than thresholdMin strains
def removeEdgeLessThanStrains(graph, min):
    edges = set(graph.edges())
    for edge in edges:
        strains = set()
        for subEdge in graph.get_edge_data(edge[0], edge[1]).values():
            strain = subEdge['strain']
            strains.add(strain)
        if len(strains) < min:
            while graph.get_edge_data(edge[0], edge[1]) != None:
                graph.remove_edge(edge[0], edge[1])


# Remove isolates and small sub nets afterwards
def removeEdgeLessThanStrainsAndPrune(graph, minStrain, subNetsMin):
    graphPrun = graph.copy()
    removeEdgeLessThanStrains(graphPrun, minStrain)
    removeIsolates(graphPrun)
    graphPrun = removeSubNetsBelow(graphPrun, subNetsMin)
    return graphPrun


def removeIsolates(graph):
    isolates = list(nx.isolates(graph))
    graph.remove_nodes_from(isolates)


# Merge edges of replicates with average weight
def mergeReplicateEdges(graph):
    edges = set(graph.edges())
    for edge in edges:
        node1 = edge[0]
        node2 = edge[1]
        strainWeights = dict()
        for subEdge in graph.get_edge_data(node1, node2).values():
            strain = subEdge['strain']
            weight = subEdge['weight']
            if strain in strainWeights.keys():
                strainWeights[strain].append(weight)
            else:
                strainWeights[strain] = [weight]
        subEdgeKeys = list(graph.get_edge_data(node1, node2).keys())
        for subEdgeKey in subEdgeKeys:
            graph.remove_edge(node1, node2, subEdgeKey)
        for strain in strainWeights.keys():
            graph.add_edge(node1, node2, strain=strain, weight=mean(strainWeights[strain]))


# Merge edges from graphs on single graph, retain separate edges, add attributes
def mergeGraphs(graphList):
    graph = nx.MultiGraph()
    for graphSub in graphList:
        for edgeData in graphSub.edges(data=True):
            node1 = edgeData[0]
            node2 = edgeData[1]
            attr = edgeData[2]
            key = graph.add_edge(node1, node2)
            for attrKey, attrVal in attr.items():
                graph[node1][node2][key][attrKey] = attrVal
    return graph


# Merge edges by summing the weights, does not preserve other attributes
# Makes Graph type
def mergeStrainEdges(graphMulti):
    graph = nx.Graph()
    for edgeData in graphMulti.edges(data=True):
        node1 = edgeData[0]
        node2 = edgeData[1]
        attr = edgeData[2]
        weight = attr['weight']
        if ((node1, node2)) in graph.edges:
            weightOld = graph.get_edge_data(node1, node2)['weight']
            weightNew = weight + weightOld
            graph[node1][node2]['weight'] = weightNew
        else:
            graph.add_edge(node1, node2, weight=weight)
    return graph


# Remove edges so that each node has only maxDegree best edges
# If more edges have same weight, that would be used for pruning, randomly select one edge
# For Graph Type
def pruneMaxDegree(graph, maxDegree, randomState=0):
    rand.seed(randomState)
    for node in list(graph.nodes):
        degree = len(graph[node])
        if degree > maxDegree:
            toRemove = degree - maxDegree
            edgeWeights = dict()
            nodeAtlas = graph[node]
            for nodeDestination, attr in nodeAtlas.items():
                weight = attr['weight']
                if weight in edgeWeights.keys():
                    edgeWeights[weight].append(nodeDestination)
                else:
                    edgeWeights[weight] = [nodeDestination]
            sortedWeights = list(edgeWeights.keys())
            sortedWeights.sort()
            for weight in sortedWeights:
                if toRemove == 0:
                    break
                elif toRemove < 0:
                    print('ERROR: Too many edges were removed for node ' + node)
                else:
                    destinations = edgeWeights[weight]
                    if len(destinations) <= toRemove:
                        for destination in destinations:
                            graph.remove_edge(node, destination)
                            toRemove -= 1
                    else:
                        sampledToRemove = rand.sample(range(0, len(destinations)), toRemove)
                        for destinationIndex in sampledToRemove:
                            destination = destinations[destinationIndex]
                            graph.remove_edge(node, destination)
                            toRemove -= 1
            if maxDegree != len(graph[node]):
                print('ERROR: wrong number of edges removed for ' + node)


# Name dict of scores "(1,2)=score" with genes
# Indices in dict keys correspond to indices of gene row names
def nameScoresDict(scores, genes):
    names = genes.index
    named = dict()
    for pair, score in scores.items():
        gene1 = names[pair[0]]
        gene2 = names[pair[1]]
        named[(gene1, gene2)] = score
    return named


# Make dict with numerical values intersect by averaging values
# Include only items that are present in presenInMin dicts with value at least minVal
# If useAbs the minVal is applied to abs(value) and values ar averaged as abs
def dictIntersectAvgVal(dictList, presentInMin, minVal=0, useAbs=True):
    union = dict()
    for d in dictList:
        for k, v in d.items():
            if useAbs:
                v = abs(v)
            if v >= minVal:
                if k in union.keys():
                    union[k].append(v)
                else:
                    union[k] = [v]
    intersect = dict()
    for k, v in union.items():
        if len(v) >= presentInMin:
            intersect[k] = mean(v)
    return intersect


# Label nodes with info in dict - find info by matching node name to dict key
# Dict values of form: (a,b,..) where dictInfo specifies what each value means:(A,B,..)
# ai belongs to data name A and so forth
def labelNodesFromTuppleDict(graph, dict, dictInfo):
    for node in graph.nodes():
        if node in dict.keys():
            info = dict[node]
            for i in range(len(dictInfo)):
                graph.node[node][dictInfo[i]] = info[i]
        else:
            print('No info for: ' + node)


# Save graph and dist file
def saveNet(rScores, threshold, doGraph, doDst, rows, gen, filePrefix):
    filePrefix = formatFilePrefix(filePrefix)

    matrix = matrixFromR(rScores, threshold, rows)

    if doGraph:
        # Graph
        graph = nx.from_numpy_matrix(matrix, create_using=nx.Graph)
        # Name nodes (change graph)
        mapNames = dict(zip(graph, list(gen.index[0:rows])))
        graphNamed = nx.relabel_nodes(graph, mapNames)

        nx.write_pajek(graphNamed, filePrefix + ".net")
        nx.write_gml(graphNamed, filePrefix + ".gml")
        # Without isolates
        graphNoIsolNamed = graphNamed.copy()
        graphNoIsolNamed.remove_nodes_from(list(nx.isolates(graphNoIsolNamed)))
        nx.write_pajek(graphNoIsolNamed, filePrefix + "_NoIsolates.net")
        nx.write_gml(graphNoIsolNamed, filePrefix + "_NoIsolates.gml")

    if doDst:
        # Write to file dst
        matrixD = 1 - matrix
        f = open(filePrefix + '.dst', "wt")
        tsvWriter = csv.writer(f, delimiter='\t')
        tsvWriter.writerow([rows, "labelled"])
        for row in range(0, rows):
            writeArr = [[gen.index[row]], matrixD[row, :row]]
            flattened = [val for sublist in writeArr for val in sublist]
            tsvWriter.writerow(flattened)
        f.close()


def saveGraph(graph, filePrefix, saveNet=True):
    filePrefix = formatFilePrefix(filePrefix)
    nx.write_gml(graph, filePrefix + ".gml")
    if saveNet:
        graphGML = nx.read_gml(filePrefix + ".gml")
        nx.write_pajek(graphGML, filePrefix + ".net")


def loadNetGraph(file):
    return nx.read_pajek(file)


def loadGMLGraph(file):
    return nx.read_gml(file)


# Returns string between before and after, '' means end/start
def splitBy(str, before, after):
    if before == '' and after == '':
        return str
    elif before == '':
        return str.split(after)[0]
    elif after == '':
        return str.split(before)[1]
    else:
        return str.split(after)[0].split(before)[1]


# Save genes file with genes in rows
# genes - file with genes in collumns
def saveGenesForNet(graph, genes, table, file, dataFirstRow, hasEntrez):
    # Extract genes
    geneNames = list(graph.nodes)
    geneSub = genes[geneNames].T
    geneSub.columns = table['Feature name'].tolist()[dataFirstRow:]
    featCols = len(geneSub.columns)

    # Add cols
    geneSub['Gene'] = geneSub.index

    if hasEntrez:
        entrezIDs = []
        for gene in geneSub.index:
            dictData = table.iloc[1, :].loc[gene]
            id = splitBy(dictData, 'Entrez\ ID=', ' ddb_g')
            entrezIDs.append(id)

        geneSub['Entrez ID'] = entrezIDs

    # Make attributes
    if hasEntrez:
        addAttrType = 2
        addAttrFlag = ['class', 'meta']
    else:
        addAttrType = 1
        addAttrFlag = ['class']
    attrFlagTime = table['Time'].tolist()[dataFirstRow:]
    attrFlagTime = ['Time=' + t for t in attrFlagTime]
    attrFlagStrain = table['Source ID'].tolist()[dataFirstRow:]
    attrFlagStrain = ['Source\ ID=' + t for t in attrFlagStrain]
    attrFlag = [i + ' ' + j for i, j in zip(attrFlagTime, attrFlagStrain)]
    attrType = tuple(['continuous'] * featCols + ['string'] * addAttrType)
    attrFlag = tuple(attrFlag + addAttrFlag)
    attr = [attrType, attrFlag]
    attributesRows = pd.DataFrame(attr, columns=geneSub.columns)

    # Make DF
    subsetDF = attributesRows.append(geneSub)
    subsetDF.to_csv(path_or_buf=file, sep='\t', index=False)


#
# table and rows/cols from table
# Genes - geneIDs are row names
def plotNetOfGene(table, timeRowFirst, timeRowLastInclusive, timeCol, graph, geneID, genes, fileName='', save=True):
    timeAX4 = list(table.iloc[timeRowFirst:timeRowLastInclusive + 1, timeCol])
    edges = list(graph.edges(geneID))
    nodes = [geneID]
    for e in edges:
        nodes.append(e[1])
    fig, ax = plt.subplots()
    for n in nodes:
        ax.scatter(timeAX4, list(genes.loc[n, :]), label=n)
    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.01), frameon=False)
    ax.legend(loc='center left', bbox_to_anchor=(0.7, 0.98))
    plt.yscale('log')
    plt.xlabel('Time')
    plt.ylabel('Expression')
    # plt.show()
    if save:
        plt.savefig(fileName)


def maxDegree(graph, fileName):
    degrees = dict(graph.degree())
    fig = plt.hist(degrees.values(), bins=max(degrees.values()))
    plt.savefig(fileName)
    plt.show()
    # Max edges:
    maxDeg = max(degrees, key=degrees.get)
    return degrees[maxDeg]


# Get nodes with degree at least
def nodesDegree(graph, threshold):
    nodes = []
    for gene, deg in dict(graph.degree()).items():
        if deg >= threshold:
            nodes.append((gene, deg))
    return nodes


# Plot subgraphs sizes
def plotSubgraphSize(graph, fileName='', save=True):
    # if needed: matplotlib.use('TkAgg')
    subLen = [len(c) for c in sorted(nx.connected_components(graph), key=len, reverse=True)]
    fig = plt.hist(subLen, bins=max(subLen))
    # plt.yscale('log')
    plt.xscale('log')
    plt.ylabel('Count')
    plt.xlabel('Number of nodes')
    if save:
        plt.savefig(fileName)
    # plt.show()


# Plot how many edges would remain after pruning of edges so that no node has greater degree
# than individual degree from pruneList
# Plot remaining edges vs prune threshold
# tested on Grpah Type
def plotEdgeNAfterDegreePruning(graph, pruneList, fileName='', save=True):
    # Make matrix for  measurmenst, add in measuremnt pairs where index 0=prune, 1=nEdges
    nEdgesM = np.zeros((len(pruneList), 2))
    indexComputed = 0
    for prune in pruneList:
        graphCopy = graph.copy()
        pruneMaxDegree(graphCopy, prune)
        nEdgesM[indexComputed, 0] = prune
        nEdgesM[indexComputed, 1] = graphCopy.number_of_edges()
        indexComputed += 1
    fig = plt.scatter(nEdgesM[:, 0], nEdgesM[:, 1])
    plt.xlabel('Max degree')
    plt.ylabel('N edges')
    if save:
        plt.savefig(fileName)


# Plot distribution of node degrees
def plotDegreeDist(graph, fileName='', save=True):
    degrees = list(dict(graph.degree).values())
    fig = plt.hist(degrees, bins=max(degrees))
    plt.ylabel('Count')
    plt.xlabel('Degree')
    # plt.yscale('log')
    plt.xscale('log')
    if save:
        plt.savefig(fileName)


def extractSubGraphs(graph):
    subGraphs = []
    subGraphNodes = [c for c in sorted(nx.connected_components(graph), key=len, reverse=True)]
    for sub in subGraphNodes:
        subGraph = graph.subgraph(sub)
        subGraphs.append(subGraph)
    return subGraphs


def plotEdgeWeigths(graph, file='', save=True):
    weigths = []
    for e in list(graph.edges):
        for k, v in graph.get_edge_data(e[0], e[1]).items():
            if type(v) == dict:
                if 'weight' in v.keys():
                    weigths.append(v['weight'])
            elif (type(v) == float or type(v) == int) and k == 'weight':
                weigths.append(v)
    fig = plt.hist(weigths, bins=100)
    plt.yscale('log')
    plt.xlabel('Weigth')
    plt.ylabel('Count')
    if save:
        plt.savefig(file)
    # plt.show()


# Remouve subgraphs with size below minSizeNet
def removeSubNetsBelow(graph, minSizeNet):
    removeNodes = set()
    subgraphs = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]
    for sg in subgraphs:
        if len(sg) < minSizeNet:
            removeNodes.update(list(sg.nodes))
    graphPruned = graph.copy()
    for n in removeNodes:
        graphPruned.remove_node(n)
    return graphPruned


# Remove below minWeigth, NOT for MULTI graph (if it has more than 1 edge between 2 nodes), can work on multigraph class
def removeEdgesWeigth(graph, minWeigth):
    toRemove = []
    for e in list(graph.edges):
        for k, v in graph.get_edge_data(e[0], e[1]).items():
            if type(v) == dict:
                if 'weight' in v.keys():
                    weigth = v['weight']
                    if weigth < minWeigth:
                        toRemove.append((e[0], e[1]))
            elif (type(v) == float or type(v) == int) and k == 'weight':
                weigth = v
                if weigth < minWeigth:
                    toRemove.append((e[0], e[1]))
    pruned = graph.copy()
    pruned.remove_edges_from(toRemove)
    return pruned


def multiGraphSumEdge(graph):
    G = nx.Graph()
    for u, v, data in graph.edges(data=True):
        w = data['weight']
        if G.has_edge(u, v):
            G[u][v]['weight'] += w
        else:
            G.add_edge(u, v, weight=w)
    return G


# weigthMin - prune anythuing below to 0
def tsneScatter(associations, perplexity, weigthMin=0, random=0, openTsne=True):
    if isinstance(associations, nx.Graph):
        matrix = nx.to_numpy_matrix(associations)
    elif isinstance(associations, pd.DataFrame):
        matrix = associations.values
    matrix[matrix < weigthMin] = 0
    max = np.amax(matrix)
    matrixD = max - matrix
    np.fill_diagonal(matrixD, 0)
    if not openTsne:
        tsne = TSNE(metric='precomputed', perplexity=perplexity, random_state=random).fit_transform(matrixD)
        x = tsne[:, 0]
        y = tsne[:, 1]
    else:
        n = matrixD.shape[0]
        narrayD = np.zeros((n, n))
        for j in range(n):
            narrayD[j, :] = matrixD[j, :]
        affinities_train = PerplexityBasedNN(narrayD, perplexity=perplexity, method='predefined', metric="predefined",
                                             random_state=random)
        init_train = initialization.random(narrayD, random_state=random)
        embedding_train = TSNEEmbedding(init_train, affinities=affinities_train)
        embedding = embedding_train.optimize(n_iter=250, exaggeration=12, momentum=0.5)
        x = [x[0] for x in embedding]
        y = [x[1] for x in embedding]

        for i in range(len(matrixD) - 1):
            for j in range(i + 1, len(matrixD)):
                if matrixD[i, j] < max:
                    plt.plot([x[i], x[j]], [y[i], y[j]], 'bo-', linewidth=0.5, markersize=1)

    # plt.scatter(x,y, s=1)


# Match DDB ID list to gene symbol, EntrezID and description
# Return dict k=DDB ID, val=(symbol, EntrezID,description)
def matchDDBids(genesDDB):
    matcher = GeneMatcher(44689)
    matcher.genes = genesDDB
    geneNames = matcher.genes
    geneInfo = dict()
    for gene in geneNames:
        ddb = gene.input_identifier
        symbol = parseNoneStr(gene.symbol)
        entrez = parseNoneStr(gene.gene_id)
        description = parseNoneStr(gene.description)
        geneInfo[ddb] = (symbol, entrez, description)
    return geneInfo


# Return '' instead of None object, if object is not None return object
def parseNoneStr(data):
    if data == None:
        return ''
    else:
        return data


# Functions for GO annotation

# Return list of entrez ids from nodes annotated with Entrez ID
def getEntrezIDFromNodes(graphAnnotated):
    ids = []
    for i in graphAnnotated.nodes(data=True):
        ids.append(i[1]['EntrezID'])
    return ids


# NOT USED!
# Extract graph (must be annotated) sub graphs and ther Entrez IDs
# Return collection of nodes and correponding EntrezIDs groups (not guarante same order)
# If ID is absent ('') remove it from list and remove sub sets if they have no Entrez IDs
# Return [([nodeNames],[EntrezIDs]),..] for each sub graph
def extractSubGraphNodeIDs(graphAnnotated):
    annoSubs = []
    subs = extractSubGraphs(graphAnnotated)
    for sub in subs:
        ids = getEntrezIDFromNodes(sub)
        ids = [x for x in ids if x != '']
        if len(ids) != 0:
            annoSubs.append((list(sub.nodes), ids))
    return annoSubs


# Add GO Process terms to all nodes of a sub graph used for enrichment
# Graph must ne annotated wit hEntrez IDs
def annSubsWithGO(graphAnnotated, fdrCutoff=0.25, slims=True, aspect='Process'):
    subs = extractSubGraphs(graphAnnotated)
    for sub in subs:
        ids = getEntrezIDFromNodes(sub)
        ids = [x for x in ids if x != '']
        if len(ids) != 0:
            enriched = enrichment(ids, fdrCutoff=fdrCutoff, slims=slims, aspect=aspect)
            if len(enriched) != 0:
                print(sub.nodes, enriched)
                for node in sub.nodes:
                    graphAnnotated.node[node]["GO"] = str(enriched)
            else:
                print('No enrichment:', sub.nodes)


# Calculate enrichemnt from Entrez ID list (as strings)
# Return dict k= GO term string, v=FDR
# slims, aspect -> look at get_enriched_terms() documentation
def enrichment(entrezIDList, fdrCutoff=0.25, slims=True, aspect='Process'):
    anno = Annotations(44689)
    enrichment = anno.get_enriched_terms(entrezIDList, slims_only=slims, aspect=aspect)
    filtered = filter_by_p_value(enrichment, fdrCutoff)
    enrichData = dict()
    for go, data in filtered.items():
        terms = anno.get_annotations_by_go_id(go)
        for term in terms:
            if term.go_id == go:
                padj = data[1]
                enrichData[term.go_term] = padj
                break
    return enrichData
