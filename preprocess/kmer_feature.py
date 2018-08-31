
# coding: utf-8

import numpy as np
import pandas as pd

import h5py
import pickle as pkl
import gzip


f = open( '../data/preprocessed/sample_id.pkl','rb')
sample = pkl.load(f)
f.close()

sequence90 = dict()
with gzip.open('../data/preprocessed/sequence90.fasta.gz','rb') as f:
    lines=f.readlines()
    for i in range(int(len(lines)/2)):
        sequence90[str(lines[2*i], encoding='utf-8').replace('>','').replace('\n','')]=str(lines[2*i+1], encoding='utf-8').replace('\n','')
    del(lines)

mer=[['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y'],[],[]]

for i in aa:
    for j in aa:
        mer[1].append(i+j)

for i in aa:
    for j in aa:
        for k in aa:
            mer[2].append(i+j+k)


n=len(sample)

def generative_kmer_feature(k):
    feature = pd.DataFrame(np.zeros([n,len(mer[k])]),index=sample,columns=mer[k-1])
    for i in range(n):
        for j in range(len(sequence90[sample[i]])-k+1):
            feature.loc[sample[i],sequence90[sample[i]][j:j+k]] = feature.loc[sample[i],sequence90[sample[i]][j:j+k]]+1
        feature.loc[sample[i]] = feature.loc[sample[i]] / (len(sequence90[sample[i]]) - k+1)
    return(feature)

def save_feature_kmer(k):
    feature = generative_kmer_feature(k)
    with h5py.File('../data/preprocessed/feature_%dmer.h5' % k, 'w') as f:
        opts = {
            "compression": "gzip",
            "compression_opts": 9
        }
        f.create_dataset("mat", data = feature.values, **opts)
        f.create_dataset("mer", data = np.array([s.encode('UTF-8') for s in feature.columns]), **opts)
        f.create_dataset("protein_id", data = np.array([s.encode('UTF-8') for s in feature.index]), **opts)

for k in range(1,4):
    save_feature_kmer(k)
