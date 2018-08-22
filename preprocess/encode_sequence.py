#!/usr/bin/env python

import sys
import gzip
import numpy as np
import h5py
from tqdm import tqdm
import utils


alphabet = "ARNDCQEGHILKMFPSTWYV"
alphabet_idx = {
    alphabet[i]: i for i in range(len(alphabet))
}

len_cutoff = 5000

print("Reading data...")
protein_id_list, sequence_list = [], []
with gzip.open("../data/preprocessed/%s.fasta.gz" % sys.argv[1], "rt") as f:
    for line in f:
        if line.startswith(">"):
            protein_id = line.strip()[1:]
        else:
            line = line.strip()
            if len(line) <= len_cutoff and line.find("U") == -1:
                protein_id_list.append(protein_id)
                sequence_list.append(line)

print("Encoding data...")
seq_mat = np.zeros((len(sequence_list), len_cutoff, len(alphabet)),
                   dtype=np.int8)
for i in tqdm(range(len(sequence_list)), unit="sequences"):
    for j in range(len(sequence_list[i])):
        seq_mat[i, j, alphabet_idx[sequence_list[i][j]]] = 1

print("Saving result...")
with h5py.File("../data/preprocessed/%s.h5" % sys.argv[1], "w") as f:
    opts = {
        "compression": "gzip",
        "compression_opts": 9
    }
    f.create_dataset("protein_id", data=utils.encode(protein_id_list), **opts)
    f.create_dataset("mat", data=seq_mat, **opts)
    f.create_dataset("aa", data=utils.encode(list(alphabet)))
