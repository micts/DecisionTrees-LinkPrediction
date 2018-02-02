import pickle
import pandas as pd

with open('MathOverflow/adamicAdar_MathOverflow.pkl', 'rb') as y:
    adamicAdarList = pickle.load(y)

with open('MathOverflow/commonNeighbors_MathOverflow.pkl', 'rb') as y:
    commonNeighborsList = pickle.load(y)

with open('MathOverflow/rootedPageRank_MathOverflow.pkl', 'rb') as y:
    rootedPageRankList = pickle.load(y)

with open('MathOverflow/jaccard_MathOverflow.pkl', 'rb') as y:
    jaccardList = pickle.load(y)

with open('MathOverflow/resAllocation_MathOverflow.pkl', 'rb') as y:
    resAllocationList = pickle.load(y)

with open('MathOverflow/assocStrength_MathOverflow.pkl', 'rb') as y:
    assocStrengthList = pickle.load(y)
    
with open('MathOverflow/labels_MathOverflow.pkl', 'rb') as y:
    labels = pickle.load(y)    
    
with open('MathOverflow/nmeasure_MathOverflow.pkl', 'rb') as y:
    nmeasureList = pickle.load(y)
    
with open('MathOverflow/minOverlap_MathOverflow.pkl', 'rb') as y:
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

dataset.to_csv('MathOverflow_Data.csv',index=False)