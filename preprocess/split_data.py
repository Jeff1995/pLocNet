#!/usr/bin/env python

import argparse
import functools
import numpy as np
import h5py
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", type=str, nargs="+")
    parser.add_argument("--train", dest="train", type=float, default=0.7)
    parser.add_argument("--val", dest="val", type=float, default=0.1)
    parser.add_argument("--test", dest="test", type=float, default=0.2)
    parser.add_argument("-s", "--seed", dest="seed", type=int, default=None)
    parser.add_argument("-o", "--output", dest="output",
                        type=str, required=True)
    return parser.parse_args()


def main():
    cmd_args = parse_args()
    idx_list = []
    for input in cmd_args.input:
        with h5py.File(input, "r") as f:
            idx = f["protein_id"].value
            idx_list.append(utils.unique(idx))
    common_idx = functools.reduce(np.intersect1d, idx_list)
    if cmd_args.seed is not None:
        np.random.seed(cmd_args.seed)
    common_idx = np.random.permutation(common_idx)
    normalizer = cmd_args.train + cmd_args.val + cmd_args.test
    cmd_args.train, cmd_args.val, cmd_args.test = \
        cmd_args.train / normalizer, \
        cmd_args.val / normalizer, \
        cmd_args.test / normalizer
    train_size = np.round(len(common_idx) * cmd_args.train).astype(np.int)
    val_size = np.round(len(common_idx) * cmd_args.val).astype(np.int)
    train_idx = common_idx[:train_size]
    val_idx = common_idx[train_size:(train_size + val_size)]
    test_idx = common_idx[(train_size + val_size):]
    with h5py.File(cmd_args.output, "w") as f:
        f.create_dataset("train", data=train_idx)
        f.create_dataset("val", data=val_idx)
        f.create_dataset("test", data=test_idx)


if __name__ == "__main__":
    main()
    print("Done!")
