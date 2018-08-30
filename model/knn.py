
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd

import pickle as pkl
import h5py

from math import sqrt
from sklearn.metrics import roc_auc_score



f=open('../data/supervised_data/adj.pkl','rb')
adj_coo=pkl.load(f)
f.close()

f=open('..data/supervised_data/mask.pkl','rb')
train_mask,val_mask,test_mask=pkl.load(f)
f.close()

f=open('..data/supervised_data/sample_id.pkl','rb')
sample=pkl.load(f)
f.close()

n=len(sample)


f=h5py.File("./pLocNet/data/preprocessed/localization.h5", "r")
localization=pd.DataFrame(f["mat"].value,
                          index=f["protein_id"].value,
                          columns=f["label"].value)
f.close()

localization.index=[str(s, encoding='utf-8') for s in localization.index]
localization.columns=[str(s, encoding='utf-8') for s in localization.columns]
localization_rv = localization.loc[sample]


neighborhood=dict()
for i in range(n):
    neighborhood[i]=[]

for i in range(adj_coo.row.shape[0]):
    neighborhood[adj_coo.row[i]].append(adj_coo.col[i])

neighborhood_test=dict()
for i in neighborhood:
    if test_mask[i]:
        neighborhood_test[i]=[]
        for j in neighborhood[i]:
            if train_mask[j]:
                neighborhood_test[i].append(j)

def knn_label(loc):
    labels = np.zeros([n])
    for i in range(n):
        if test_mask[i]:
            if len(neighborhood_test[i])>0:
                labels[i] = len([j for j in neighborhood_test[i] if localization_rv.loc[sample[j],loc]])/len(neighborhood_test[i])
            else:
                labels[i] = sum(localization_rv.loc[train_mask,loc]/sum(train_mask)
    return(labels)


def evaluate(labels,preds):
    auc=roc_auc_score(labels,preds)
    
    TP=sum(np.round(preds,0)*labels)
    FP=sum(np.round(preds,0)*(1-labels))
    TN=sum((1-np.round(preds,0))*(1-labels))
    FN=sum((1-np.round(preds,0))*labels)
    
    acc=(TP+TN)/(TP+FP+TN+FN)
    se=TP/(TP+FN)
    sp=TN/(TN+FP)
    pre=TP/(TP+FP)
    mcc=(TP*TN-FP*FN)/sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    
    print('AUC = ',auc,' ACC = ',acc,' SE = ',se,' SP = ',sp,' PRE = ',pre, ' MCC = ', mcc)


for loc in localization_rv.columns:
    print(loc+' results')
    evaluate(localization_rv[loc].iloc[test_mask] ,knn_label(loc)[test_mask])