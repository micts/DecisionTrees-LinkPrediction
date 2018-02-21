import numpy as np
import networkx as nx
import linkpred
import collections
import pickle
import os

# load the data
data = np.loadtxt('./datasets//DiggNetwork//DiggNetwork.txt', dtype = int)

data = data[np.where(data[:, 0] != data[:, 1])] # remove self loops
data = data[:,[0, 1, 3]] 
data = data[data[:, 2].argsort()] # sort by timestamp

# Train period from rows 0:51721. Rest is the test period.
trainPeriod = data[:51721,:]  
testPeriod = data[51721:,:]

# Convert the periods to undirected graphs.
trainPeriodGraph = nx.Graph(trainPeriod[:,[0,1]].tolist())
testPeriodGraph = nx.Graph(testPeriod[:,[0,1]].tolist())

"""
Calculate metrics for each node and its 2-size neighborhood. The functions exclude nodes
that belong in the train period ('excluded' argument).

Some of those measures will be very slow to compute due to the size of the network.
"""

# Adamic Adar
adamicAdar = linkpred.predictors.AdamicAdar(trainPeriodGraph, 
                                            excluded = trainPeriodGraph.edges())
adamicAdar_results = adamicAdar.predict()

# commonNeighbors measure
commonNeighbors = linkpred.predictors.CommonNeighbours(trainPeriodGraph, 
                                                       excluded = trainPeriodGraph.edges())
commonNeighbors_results = commonNeighbors.predict(alpha = 0)

# Rooted PageRank
rootedPageRank = linkpred.predictors.RootedPageRank(trainPeriodGraph, 
                                                    excluded = trainPeriodGraph.edges())
rootedPageRank_results = rootedPageRank.predict(weight = None, k = 2)

# Jaccard coefficient
jaccard = linkpred.predictors.Jaccard(trainPeriodGraph, 
                                      excluded = trainPeriodGraph.edges())
jaccard_results = jaccard.predict()

# NMeasure
nmeasure = linkpred.predictors.NMeasure(trainPeriodGraph, 
                                        excluded=trainPeriodGraph.edges())
nmeasure_results = nmeasure.predict()

# Min Overlap
minOverlap = linkpred.predictors.MinOverlap(trainPeriodGraph, 
                                            excluded=trainPeriodGraph.edges())
minOverlap_results = minOverlap.predict()

# Resource Allocation
resAllocation = linkpred.predictors.ResourceAllocation(trainPeriodGraph, 
                                                       excluded = trainPeriodGraph.edges())
resAllocation_results = resAllocation.predict()

# Association Strength
assocStrength = linkpred.predictors.AssociationStrength(trainPeriodGraph, 
                                                        excluded = trainPeriodGraph.edges())
assocStrength_results = assocStrength.predict()

# converting the metrics into lists. We will feed them into pandas data frames later
adamicAdarList = list(adamicAdar_results.values())
commonNeighborsList = list(commonNeighbors_results.values())
rootedPageRankList = list(rootedPageRank_results.values())
jaccardList = list(jaccard_results.values())
resAllocationList = list(resAllocation_results.values())
assocStrengthList = list(assocStrength_results.values())
nmeasureList = list(nmeasure_results.values())
minOverlapList = list(minOverlap_results.values())

# save the metrics
if not os.path.isdir('Metrics'):
    os.mkdir('Metrics')

if not os.path.isdir('Metrics/DiggNetwork'):
    os.mkdir('Metrics/DiggNetwork')

with open('Metrics/DiggNetwork/adamicAdar_DiggNetwork.pkl', 'wb') as y:
    pickle.dump(adamicAdarList, y)

with open('Metrics/DiggNetwork/commonNeighbors_DiggNetwork.pkl', 'wb') as y:
    pickle.dump(commonNeighborsList, y)

with open('Metrics/DiggNetwork/rootedPageRankList_DiggNetwork.pkl', 'wb') as y:
    pickle.dump(rootedPageRankList, y)

with open('Metrics/DiggNetwork/jaccard_DiggNetwork.pkl', 'wb') as y:
    pickle.dump(jaccardList, y)

with open('Metrics/DiggNetwork/resAllocation_DiggNetwork.pkl', 'wb') as y:
    pickle.dump(resAllocationList, y)

with open('Metrics/DiggNetwork/assocStrength_DiggNetwork.pkl', 'wb') as y:
    pickle.dump(assocStrengthList, y)
    
with open('Metrics/DiggNetwork/nmeasure_DiggNetwork.pkl', 'wb') as y:
    pickle.dump(nmeasureList, y)
        
with open('Metrics/DiggNetwork/minOverlap_DiggNetwork.pkl', 'wb') as y:
    pickle.dump(minOverlapList, y)    


# Create a dictionary that represents the testPeriodGraph
testPeriodDict = collections.defaultdict(list)
for node1, node2 in testPeriodGraph.edges():
    testPeriodDict[node1].append(node2)

"""Creating the labels (0 or 1). If a pair for which we calculated a metric does not exist in 
the testing period, it takes a 0, otherwise an 1. """
    
labels = []
datasetPairs = []
for u, v in jaccard_results.keys():   
    datasetPairs.append([u,v])
    if (v in testPeriodDict[u]) or (u in testPeriodDict[v]):
        labels.append(1)
    else:
        labels.append(0)

# Also save the labels
with open('DiggNetwork/labels_DiggNetwork.pkl', 'wb') as y:
    pickle.dump(labels, y)
























