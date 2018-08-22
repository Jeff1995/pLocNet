#!/usr/bin/env python

import numpy as np
import pandas as pd
import h5py
import utils


print("Reading data...")
df = pd.read_table("../data/preprocessed/localization.tsv.gz", header=None)
df.columns = ["protein_id", "localization", "evidence"]
used_labels = utils.decode(np.loadtxt(
    "../data/preprocessed/used_labels.txt",
    dtype=bytes, delimiter="\n"
))

print("Encoding data...")
df = df.loc[
    np.logical_and(
        df["evidence"] != "None",
        np.in1d(df["localization"], used_labels)
    ), :
].drop_duplicates()
df.index = np.arange(df.shape[0])
protein_id = np.unique(df["protein_id"])
protein_idx = {
    protein_id[i]: i for i in range(len(protein_id))
}
label_idx = {
    used_labels[i]: i for i in range(len(used_labels))
}
label_mat = np.zeros((len(protein_id), len(used_labels)), dtype=np.int8)
for i in range(df.shape[0]):
    label_mat[
        protein_idx[df["protein_id"][i]],
        label_idx[df["localization"][i]]
    ] = 1

print("Saving results...")
with h5py.File("../data/preprocessed/localization.h5", "w") as f:
    opts = {
        "compression": "gzip",
        "compression_opts": 9
    }
    f.create_dataset("mat", data=label_mat, **opts)
    f.create_dataset("protein_id", data=utils.encode(protein_id), **opts)
    f.create_dataset("label", data=utils.encode(used_labels), **opts)
