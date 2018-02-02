import pickle
import pandas as pd

with open('SlashDot/adamicAdar_SlashDot.pkl', 'rb') as y:
    adamicAdarList = pickle.load(y)

with open('SlashDot/commonNeighbors_SlashDot.pkl', 'rb') as y:
    commonNeighborsList = pickle.load(y)

with open('SlashDot/rootedPageRank_SlashDot.pkl', 'rb') as y:
    rootedPageRankList = pickle.load(y)

with open('SlashDot/jaccard_SlashDot.pkl', 'rb') as y:
    jaccardList = pickle.load(y)

with open('SlashDot/resAllocation_SlashDot.pkl', 'rb') as y:
    resAllocationList = pickle.load(y)

with open('SlashDot/assocStrength_SlashDot.pkl', 'rb') as y:
    assocStrengthList = pickle.load(y)
    
with open('SlashDot/labels_SlashDot.pkl', 'rb') as y:
    labels = pickle.load(y)    
    
with open('SlashDot/nmeasure_SlashDot.pkl', 'rb') as y:
    nmeasureList = pickle.load(y)
    
with open('SlashDot/minOverlap_SlashDot.pkl', 'rb') as y:
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
     
dataset.to_csv('SlashDot_Data.csv',index=False)

