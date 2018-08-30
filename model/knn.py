#!/usr/bin/env python
# coding: utf-8

import os
import argparse
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import h5py

from run_gcn import read_data


parser = argparse.ArgumentParser()
parser.add_argument("-g", dest="g", type=str, required=True)
parser.add_argument("-y", dest="y", type=str, required=True)
parser.add_argument("--split", dest="split", type=str, required=True)
parser.add_argument("-o", dest="output_path", type=str, required=True)
cmd_args = parser.parse_args()

if not os.path.exists(cmd_args.output_path):
    os.makedirs(cmd_args.output_path)

print("Reading data...")
data, adj_coo, train_mask, val_mask, test_mask = \
    read_data(None, cmd_args.y, cmd_args.g, cmd_args.split)
adj_coo = coo_matrix(adj_coo)
n = data.size()
localization = pd.DataFrame(
    data["y"], index=data["protein_id"])

print("")
neighborhood = dict()
for i in range(n):
    neighborhood[i] = []

for i in range(adj_coo.row.shape[0]):
    neighborhood[adj_coo.row[i]].append(adj_coo.col[i])

neighborhood_train = dict()
for i in neighborhood:
    neighborhood_train[i] = []
    for j in neighborhood[i]:
        if train_mask[j]:
            neighborhood_train[i].append(j)

def knn_label(loc, mask):
    labels = np.zeros([n])
    for i in range(n):
        if not mask[i]:
            continue
        if neighborhood_train[i]:
            labels[i] = len([
                j for j in neighborhood_train[i] if
                localization.iloc[j, loc]
            ]) / len(neighborhood_train[i])
        else:
            labels[i] = sum(
                localization.iloc[train_mask, loc]
            ) / sum(train_mask)
    return labels


# Training set
print("Predicting training set...")
train_pred = []
for loc in localization.columns:
    train_pred.append(knn_label(loc, train_mask)[test_mask])
train_pred = np.stack(train_pred, axis=0).T

# Test set
print("Predicting test set...")
train_pred = []
test_pred = []
for loc in localization.columns:
    test_pred.append(knn_label(loc, test_mask)[test_mask])
test_pred = np.stack(test_pred, axis=0).T


with h5py.File(os.path.join(cmd_args.output_path, "result.h5")) as f:
    g = f.create_group("train")
    g.create_dataset("true", data=data["y"][train_mask])
    g.create_dataset("pred", data=train_pred)
    g = f.create_group("test")
    g.create_dataset("true", data=data["y"][test_mask])
    g.create_dataset("pred", data=test_pred)
