import networks.functionsDENet as f

# Script parts are to be run separately as needed

dataPath = '/home/karin/Documents/timeTrajectories/data/'
dataPathSaved = '/home/karin/Documents/timeTrajectories/data/correlations/replicates/'

tableLayout = dict()
tableLayout['rep'] = dict()
tableLayout['single'] = dict()
tableLayout['single']['lastGene'] = 12734
tableLayout['single']['Time'] = 12737
tableLayout['single']['Strain'] = 12736
tableLayout['rep']['lastGene'] = 12868
tableLayout['rep']['Time'] = 12870
tableLayout['rep']['Strain'] = 12869
# data=tableLayout['single']
data = tableLayout['rep']

samples = {
    'AX4': [1, 2, 5, 6, 7, 8, 9],
    'tagB': [1, 2],
    'comH': [1, 2],
    'tgrC1': [1, 2],
    'tgrB1': [1, 2],
    'tgrB1C1': [1, 2],
    'gbfA': [1, 2],
}

geneDictInfo = ('symbol', 'EntrezID', 'description')

repN = 2
rep = 'rep' + str(repN)
# rep=''
repDict = {'AX4_r1': 1, 'AX4_r2': 2, 'AX4_r5': 5, 'AX4_r6': 6, 'AX4_r7': 7, 'AX4_r8': 8, 'AX4_r9': 9}

strain = 'gbfA'
knnNeighbours = 300
thresholdKNND = 0.3
# For correlations
threshold = 0.80
conditions = strain + '_' + rep + 'kN' + str(knnNeighbours) + 'T' + str(thresholdKNND)
thresholdRStr = 'Tr' + str(threshold).replace('.', '_')

genesFromRow = 2

# table=f.importTable(dataPath+'trans_erasure_7strains_for_orange.tab')
table = f.importTable(dataPath + 'trans_9repAX4_6strains_2rep_avr_T12.tab')

genes, genesNotNull = f.extractGenesFromTable(table, genesFromRow, data['lastGene'])

genesWT, genesWTN = f.genesByStrain(genesNotNull, table, data['Strain'], strain + '_r' + str(repN), genesFromRow)

# Load gene info
geneNames = f.loadPickle(dataPath + 'Genes.pkl')

# Perform KNN or import results
# dist,neigh,distInv,neighInv,nGenesKnn=f.genesKNN(knnNeighbours,genesWTN,1,dataPathSaved+strain+'_'+rep)
dist, neigh, distInv, neighInv, nGenesKnn = f.importKnnResults(dataPathSaved + strain + '_' + rep, knnNeighbours)
# f.plotKnnDist(dist,dataPathSaved+strain+'_'+rep+'KNN'+str(knnNeighbours)+'Dist.png')
# f.plotKnnDist(distInv,dataPathSaved+strain+'_'+rep+'KNN'+str(knnNeighbours)+'DistInverse.png')

rows = nGenesKnn
# Decided for 0.15 for trans_erasure_7strains_for_orange.tab, k=200 (due to time+RAM constrain used 0.05)
# Select pairs with distances max the specified distance
knnChosen = f.chooseGenePairsFromKnn(rows, knnNeighbours, thresholdKNND, dist, neigh, distInv, neighInv)

# Make corrrelation
pScoresS, rScoresS = f.geneCorrelations(genesWTN, knnChosen, dataPathSaved + conditions)
# Load correlationResults
pScoresS, rScoresS = f.loadCorrelationData(dataPathSaved + conditions)
# f.plotCorrelationR(rScoresS,dataPathSaved+conditions+'CorrelatR.png')


# Save corelations top, graph
# f.saveTopCorrelPairsByP(dataPathSaved+conditions,pScoresS,rScoresS,genesWTN,100)
# threshold=f.lastInTop(500, [abs(x) for x in rScoresS.values()],False)
thresholdRStr = 'Tr' + str(threshold).replace('.', '_')
# f.saveNet(rScoresS,threshold,True,False,rows,genesWTN,dataPathSaved+conditions+thresholdRStr)

# Merge replicates correlations:
# merged=f.mergeRScoresRep(table,genesNotNull,data['Strain'],repDict,genesFromRow,dataPathSaved,knnNeighbours,thresholdKNND,threshold)
# mergedPruned=f.prunedDF(merged,True,dataPathSaved+conditions+thresholdRStr+'MergedRep')

# NO!!! Merge correlations from strains and replicates -TOO SLOW
graph = f.makeMultiGraph()
# graph=f.loadNetGraph(dataPathSaved+'gbfA_rep2kN300T0_3Tr0_8_tagBrep2.gml')
for sampleStrain, reps in samples.items():
    for replicate in reps:
        print(sampleStrain, replicate)
        repN = replicate
        rep = 'rep' + str(repN)
        conditionsRep = sampleStrain + '_' + rep + 'kN' + str(knnNeighbours) + 'T' + str(thresholdKNND)
        rScoresS = f.loadRScores(dataPathSaved + conditionsRep)
        genesWT, genesWTN = f.genesByStrain(genesNotNull, table, data['Strain'], sampleStrain + '_r' + str(repN),
                                            genesFromRow)
        f.buildGraph(graph, sampleStrain, replicate, rScoresS, threshold, genesWTN, threshRorP=True, pScores='')
        f.saveGraph(graph, dataPathSaved + conditions.replace('.', '_') + thresholdRStr + "_" + sampleStrain + rep,
                    False)
f.saveGraph(graph, dataPathSaved + conditions.replace('.', '_') + thresholdRStr + "_MergedRep", False)
# trimm
f.removeSingleSampleEdges(graph)
f.removeSingleReplicateEdges(graph)
graph = graph.to_undirected()

# Merge correlations from replicates of strain
for sampleStrain, reps in samples.items():
    graph = f.makeMultiGraph()
    # Add connections from replicates in MultiGraph
    for replicate in reps:
        print(sampleStrain, replicate)
        repN = replicate
        rep = 'rep' + str(repN)
        conditionsRep = sampleStrain + '_' + rep + 'kN' + str(knnNeighbours) + 'T' + str(thresholdKNND)
        rScoresS = f.loadRScores(dataPathSaved + conditionsRep)
        genesWT, genesWTN = f.genesByStrain(genesNotNull, table, data['Strain'], sampleStrain + '_r' + str(repN),
                                            genesFromRow)
        f.buildGraph(graph, sampleStrain, replicate, rScoresS, threshold, genesWTN, threshRorP=True, pScores='')
    print("saving " + sampleStrain)
    conditionsStrain = sampleStrain + '_' + 'kN' + str(knnNeighbours) + 'T' + str(thresholdKNND) + thresholdRStr
    f.saveGraph(graph, dataPathSaved + conditionsStrain.replace('.', '_'), False)
    # Remove single  edges (not multi-edge, only 1 sample/replicate) from graph
    f.removeSingleSampleEdges(graph)
    f.removeIsolates(graph)
    # Merge remaining multi-edges to retain average weight
    f.mergeReplicateEdges(graph)
    # Prune
    f.plotEdgeWeigths(graph, dataPathSaved + conditionsStrain + '_weights.png')
    newTr = 0.95
    thresholdRStr = 'Tr' + str(newTr).replace('.', '_')
    conditionsStrain = sampleStrain + '_' + 'kN' + str(knnNeighbours) + 'T' + str(thresholdKNND) + thresholdRStr
    graph = f.removeEdgesWeigth(graph, newTr)
    f.removeIsolates(graph)
    f.plotSubgraphSize(graph, dataPathSaved + conditionsStrain + '_subGraphs.png')
    graph = f.removeSubNetsBelow(graph, 4)
    f.saveGraph(graph, dataPathSaved + conditionsStrain + '_Trimmed', False)

# Merge strain graphs:
newTr = 0.95
thresholdRStr = 'Tr' + str(newTr).replace('.', '_')
sampleStrain = 'AX4'
fileName = (sampleStrain + '_' + 'kN' + str(knnNeighbours) + 'T' + str(thresholdKNND) + thresholdRStr).replace('.',
                                                                                                               '_') + '_Trimmed.gml'
gAX4 = f.loadGMLGraph(dataPathSaved + fileName)
sampleStrain = 'tagB'
fileName = (sampleStrain + '_' + 'kN' + str(knnNeighbours) + 'T' + str(thresholdKNND) + thresholdRStr).replace('.',
                                                                                                               '_') + '_Trimmed.gml'
gtagB = f.loadGMLGraph(dataPathSaved + fileName)
sampleStrain = 'comH'
fileName = (sampleStrain + '_' + 'kN' + str(knnNeighbours) + 'T' + str(thresholdKNND) + thresholdRStr).replace('.',
                                                                                                               '_') + '_Trimmed.gml'
gcomH = f.loadGMLGraph(dataPathSaved + fileName)
sampleStrain = 'tgrC1'
fileName = (sampleStrain + '_' + 'kN' + str(knnNeighbours) + 'T' + str(thresholdKNND) + thresholdRStr).replace('.',
                                                                                                               '_') + '_Trimmed.gml'
gtgrC1 = f.loadGMLGraph(dataPathSaved + fileName)
sampleStrain = 'tgrB1'
fileName = (sampleStrain + '_' + 'kN' + str(knnNeighbours) + 'T' + str(thresholdKNND) + thresholdRStr).replace('.',
                                                                                                               '_') + '_Trimmed.gml'
gtgrB1 = f.loadGMLGraph(dataPathSaved + fileName)
sampleStrain = 'tgrB1C1'
fileName = (sampleStrain + '_' + 'kN' + str(knnNeighbours) + 'T' + str(thresholdKNND) + thresholdRStr).replace('.',
                                                                                                               '_') + '_Trimmed.gml'
gtgrB1C1 = f.loadGMLGraph(dataPathSaved + fileName)
sampleStrain = 'gbfA'
fileName = (sampleStrain + '_' + 'kN' + str(knnNeighbours) + 'T' + str(thresholdKNND) + thresholdRStr).replace('.',
                                                                                                               '_') + '_Trimmed.gml'
ggbfA = f.loadGMLGraph(dataPathSaved + fileName)

graph = f.mergeGraphs([gAX4, gtagB, gcomH, gtgrC1, gtgrB1, gtgrB1C1, ggbfA])
conditionsMerg = ('kN' + str(knnNeighbours) + 'T' + str(thresholdKNND) + thresholdRStr).replace('.',
                                                                                                '_') + '_MergedStrains'
f.saveGraph(graph, dataPathSaved + conditionsMerg, False)
f.plotEdgeWeigths(graph, dataPathSaved + conditionsMerg + '_weights.png')

# Remove edges not in at least x strains
minStrain = 2
minStrainStr = 'minStrain' + str(minStrain)
graph2 = f.removeEdgeLessThanStrainsAndPrune(graph, minStrain, 4)
f.saveGraph(graph2, dataPathSaved + conditionsMerg + minStrainStr, False)
minStrain = 7
minStrainStr = 'minStrain' + str(minStrain)
graph7 = f.removeEdgeLessThanStrainsAndPrune(graph, minStrain, 4)
f.saveGraph(graph7, dataPathSaved + conditionsMerg + minStrainStr, False)
minStrain = 3
minStrainStr = 'minStrain' + str(minStrain)
graph3 = f.removeEdgeLessThanStrainsAndPrune(graph, minStrain, 4)
f.saveGraph(graph3, dataPathSaved + conditionsMerg + minStrainStr, False)

# Remove weaker connections
newTr = 0.97
thresholdRStr = 'Tr' + str(newTr).replace('.', '_')
conditionsMerg = ('kN' + str(knnNeighbours) + 'T' + str(thresholdKNND) + thresholdRStr).replace('.',
                                                                                                '_') + '_MergedStrains'
graph97 = f.removeEdgesWeigth(graph, newTr)
f.removeIsolates(graph97)
graph97 = f.removeSubNetsBelow(graph97, 4)
f.saveGraph(graph97, dataPathSaved + conditionsMerg, False)

newTr = 0.99
thresholdRStr = 'Tr' + str(newTr).replace('.', '_')
conditionsMerg = ('kN' + str(knnNeighbours) + 'T' + str(thresholdKNND) + thresholdRStr).replace('.',
                                                                                                '_') + '_MergedStrains'
graph99 = f.removeEdgesWeigth(graph, newTr)
f.removeIsolates(graph99)
graph99 = f.removeSubNetsBelow(graph99, 4)
f.saveGraph(graph99, dataPathSaved + conditionsMerg, False)

graph297 = f.removeEdgesWeigth(graph2, newTr)
f.removeIsolates(graph297)
graph297 = f.removeSubNetsBelow(graph297, 4)
minStrain = 2
minStrainStr = 'minStrain' + str(minStrain)
f.saveGraph(graph297, dataPathSaved + conditionsMerg + minStrainStr, False)

# Save for g297 and g99:
f.saveGenesForNet(graph297, genes, table, dataPathSaved + 'kN300T0_3Tr0_97_MergedStrainsminStrain2.tsv', genesFromRow,
                  False)

# Load graph
# graph = f.loadNetGraph(dataPathSaved + conditions.replace('.','_')+thresholdRStr+"_MergedRep.net")

# Analyse graph:
# acaA plot
f.plotNetOfGene(table, 2, 9, data['Time'], graph, 'DDB_G0281545', genesWTN)
f.maxDegree(graph, dataPathSaved + conditions + thresholdRStr + 'MergedRep' + 'Degrees.png')
f.plotSubgraphSize(graph, dataPathSaved + conditions + thresholdRStr + 'MergedRep' + 'SubGraphs.png')
pruneMin = 4
pruned = f.removeSubNetsBelow(graph, pruneMin)
f.plotSubgraphSize(pruned)

# Save for pruned:
prunStr = 'PrunedMin' + str(pruneMin)
f.saveGraph(pruned, dataPathSaved + conditions + thresholdRStr + prunStr)
f.saveGenesForNet(pruned, genes, table,
                  dataPathSaved + conditions + thresholdRStr + prunStr + 'subsetForNetworkNoIsolates.tsv', genesFromRow)
pruned = f.loadNetGraph(dataPathSaved + conditions.replace('.', '_') + thresholdRStr + prunStr + '.net')

# Save genes from graph in tab format (complimentary data)
f.saveGenesForNet(graph, genes, table, dataPathSaved + conditions + thresholdRStr + 'subsetForNetworkNoIsolates.tsv',
                  genesFromRow)

# Make graph for min(abs(r))=0.8 and present in all strains
# Name r score lists
for sampleStrain, reps in samples.items():
    for replicate in reps:
        conditionR = (sampleStrain + '_rep' + str(replicate) + 'kN' + str(knnNeighbours) + 'T' + str(
            thresholdKNND)).replace('.', '_')
        rScores = f.loadRScores(dataPathSaved + conditionR)
        genesWT, genesWTN = f.genesByStrain(genesNotNull, table, data['Strain'], sampleStrain + '_r' + str(replicate),
                                            genesFromRow)
        print(sampleStrain + '_r' + str(replicate))
        named = f.nameScoresDict(rScores, genesWTN)
        f.savePickle(dataPathSaved + conditionR + '_rScoresNamed.pkl', named)

# Request that association is present in 2 replicates and minStrains strains:
strainIntersectDicts = []
threshold = 0.8
minStrains = 4
for sampleStrain, reps in samples.items():
    strainRdicts = []
    for replicate in reps:
        print(sampleStrain, replicate)
        conditionR = (sampleStrain + '_rep' + str(replicate) + 'kN' + str(knnNeighbours) + 'T' + str(
            thresholdKNND)).replace('.', '_')
        rScoresNamed = f.loadRScores(dataPathSaved + conditionR, True)
        strainRdicts.append(rScoresNamed)
    print('Merging reps')
    intersect = f.dictIntersectAvgVal(strainRdicts, 2, threshold, True)
    strainIntersectDicts.append(intersect)
print('Merging strains')
allStrains = f.dictIntersectAvgVal(strainIntersectDicts, minStrains, threshold, True)
conditionMerged = ('kN' + str(knnNeighbours) + 'T' + str(thresholdKNND) + 'Tr' + str(threshold)).replace('.',
                                                                                                         '_') + '_Present' + str(
    minStrains)
f.savePickle(dataPathSaved + conditionMerged + '.pkl', allStrains)
graph = f.graphFromWeightDict(allStrains)
f.saveGraph(graph, dataPathSaved + conditionMerged, True)
f.saveGenesForNet(graph, genes, table, dataPathSaved + conditionMerged + '.tsv', genesFromRow, False)
f.plotSubgraphSize(graph, dataPathSaved + conditionMerged + '_subGraphs.png')

# Make graph with summed strain edges
graphMerged = f.mergeStrainEdges(graph)

# Prune graph to have max g degree
f.pruneMaxDegree(graphMerged, 4)

# Convert gene IDs to symbols
geneInfo = f.matchDDBids(list(genes.columns))
f.savePickle(dataPath + 'Genes.pkl', geneInfo)

# label graph nodes with gene name dict:
f.labelNodesFromTuppleDict(graph, geneNames, geneDictInfo)

# Get annotation for EntrezIDs
f.annSubsWithGO(graph, slims=False)  # Annotates the existing graph

# ******************************************
# Graph with 5 neighbours from knn package

knnNeighbours=5
graphs=[]
for strain in samples.keys():
    genesWT, genesWTN = f.genesByKeyword(genesNotNull, table, data['Strain'], strain + '_r' , genesFromRow)
    dist,neigh,distInv,neighInv,nGenesKnn=f.genesKNN(knnNeighbours,genesWTN,1,dataPathSaved+strain,adjustForSelf=True,
                                                     save=False)
    knnChosen = f.chooseGenePairsFromKnn(nGenesKnn, knnNeighbours, 1, dist, neigh, distInv, neighInv,useDict=True)
    graph=f.makeMultiGraph()
    f.buildGraph(graph,strain,'merged',knnChosen,0,genesWTN,True)
    graphs.append(graph)
graph=f.mergeGraphs(graphs)
f.saveGraph(graph,dataPathSaved+'kN6-5_mergedRep_mergedStrain',False)

graph=f.loadGMLGraph(dataPathSaved+'kN6-5_mergedRep_mergedStrain.gml')
pruned=f.removeEdgesWeigth(graph,0.9)
f.removeIsolates(pruned)
pruned=f.removeSubNetsBelow(pruned,4)
f.removeEdgeLessThanStrains(pruned,2)
merged=f.mergeStrainEdges(pruned)