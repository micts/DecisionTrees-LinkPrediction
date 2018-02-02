import pickle
import pandas as pd

with open('DiggNetwork/adamicAdar_DiggNetwork.pkl', 'rb') as y:
    adamicAdarList = pickle.load(y)

with open('DiggNetwork/commonNeighbors_DiggNetwork.pkl', 'rb') as y:
    commonNeighborsList = pickle.load(y)

with open('DiggNetwork/rootedPageRank_DiggNetwork.pkl', 'rb') as y:
    rootedPageRankList = pickle.load(y)

with open('DiggNetwork/jaccard_DiggNetwork.pkl', 'rb') as y:
    jaccardList = pickle.load(y)

with open('DiggNetwork/resAllocation_DiggNetwork.pkl', 'rb') as y:
    resAllocationList = pickle.load(y)

with open('DiggNetwork/assocStrength_DiggNetwork.pkl', 'rb') as y:
    assocStrengthList = pickle.load(y)
    
with open('DiggNetwork/labels_DiggNetwork.pkl', 'rb') as y:
    labels = pickle.load(y)    
    
with open('DiggNetwork/nmeasure_DiggNetwork.pkl', 'rb') as y:
    nmeasureList = pickle.load(y)
    
with open('DiggNetwork/minOverlap_DiggNetwork.pkl', 'rb') as y:
    minOverlapList = pickle.load(y)    
    
labels2 = []
for el in labels:
    if el == 1:
        labels2.append("Form")
    else:
        labels2.append("Not Form")
    
dataset = pd.DataFrame(
    {'Adamic Adar': adamicAdarList,
     'Common Neighbors': commonNeighborsList,
     'Rooted PageRank': rootedPageRankList,
     'Jaccard': jaccardList,
     'Resource Allocation': resAllocationList,
     'Association Strength': assocStrengthList,
     'Labels': labels2,
     'NMeasure': nmeasureList,
     'Min Overlap': minOverlapList,
    })

dataset.to_csv('DiggNetwork_Data.csv',index=False)

