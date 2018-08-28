#!/usr/bin/env python

import os
import argparse
import functools
import numpy as np
import h5py
from sklearn.model_selection import KFold
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", type=str, nargs="+")
    parser.add_argument("-k", "--k-fold", dest="k_fold", type=int, default=5)
    parser.add_argument("-v", "--val-frac", dest="val_frac",
                        type=float, default=0.1)
    parser.add_argument("-s", "--seed", dest="seed", type=int, default=None)
    parser.add_argument("-o", "--output-path", dest="output_path",
                        type=str, required=True)
    return parser.parse_args()


def main():
    cmd_args = parse_args()
    idx_list = []
    for input in cmd_args.input:
        with h5py.File(input, "r") as f:
            idx = f["protein_id"][...]
            idx_list.append(utils.unique(idx))
    common_idx = functools.reduce(np.intersect1d, idx_list)
    if cmd_args.seed is not None:
        np.random.seed(cmd_args.seed)
    kf = KFold(n_splits=cmd_args.k_fold, shuffle=True,
               random_state=cmd_args.seed)
    if not os.path.exists(cmd_args.output_path):
        os.makedirs(cmd_args.output_path)
    current_fold = 0
    for train_idx, test_idx in kf.split(common_idx):
        with h5py.File(os.path.join(
            cmd_args.output_path, "fold%d.h5" % current_fold
        ), "w") as f:
            val_size = np.round(
                len(train_idx) * cmd_args.val_frac
            ).astype(np.int)
            val_idx = np.random.choice(train_idx, val_size, replace=False)
            train_idx = np.setdiff1d(train_idx, val_idx)
            f.create_dataset("train", data=common_idx[train_idx])
            f.create_dataset("val", data=common_idx[val_idx])
            f.create_dataset("test", data=common_idx[test_idx])
        current_fold += 1


if __name__ == "__main__":
    main()
    print("Done!")
