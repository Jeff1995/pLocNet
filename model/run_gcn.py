#!/usr/bin/env python

import os
import random
import argparse
import numpy as np
from scipy.sparse import coo_matrix
import tensorflow as tf
import h5py
import utils
from gcn import GCNPredictor
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-x", dest="x", type=str, required=True)
    parser.add_argument("-y", dest="y", type=str, required=True)
    parser.add_argument("-g", dest="g", type=str, required=True)
    parser.add_argument("--split", dest="split", type=str, required=True)
    parser.add_argument("-o", "--output-path", dest="output_path",
                        type=str, required=True)
    parser.add_argument("-s", "--seed", dest="seed", type=int, default=None)
    parser.add_argument("-d", "--device", dest="device", type=str, default="")
    parser.add_argument("-n", "--no-fit", dest="no_fit",
                        default=False, action="store_true")
    return parser.parse_args()


def read_data(x, y, g, split):
    with h5py.File(split, "r") as f:
        train_idx = utils.decode(f["train"][...])
        val_idx = utils.decode(f["val"][...])
        test_idx = utils.decode(f["test"][...])
        combined_idx = np.concatenate([
            train_idx, val_idx, test_idx
        ], axis=0)
        train_mask = np.concatenate([
            np.ones(len(train_idx), dtype=np.bool_),
            np.zeros(len(val_idx), dtype=np.bool_),
            np.zeros(len(test_idx), dtype=np.bool_)
        ], axis=0)
        val_mask = np.concatenate([
            np.zeros(len(train_idx), dtype=np.bool_),
            np.ones(len(val_idx), dtype=np.bool_),
            np.zeros(len(test_idx), dtype=np.bool_)
        ], axis=0)
        test_mask = np.concatenate([
            np.zeros(len(train_idx), dtype=np.bool_),
            np.zeros(len(val_idx), dtype=np.bool_),
            np.ones(len(test_idx), dtype=np.bool_)
        ], axis=0)

    with h5py.File(x, "r") as f:
        idx = utils.decode(f["protein_id"][...])
        idx_dict = {idx[i]: i for i in range(len(idx))}

        @np.vectorize
        def _idx_map(item):
            return idx_dict[item]

        x = f["protein_vec"][...][_idx_map(combined_idx)]

    with h5py.File(y, "r") as f:
        idx = utils.decode(f["protein_id"][...])
        idx_dict = {idx[i]: i for i in range(len(idx))}

        @np.vectorize
        def _idx_map(item):
            return idx_dict[item]

        y = f["mat"][:, 0:1][_idx_map(combined_idx)]

    with h5py.File(g, "r") as f:
        idx = utils.decode(f["protein_id"][...])
        idx_dict = {idx[i]: i for i in range(len(idx))}

        @np.vectorize
        def _idx_map(item):
            return idx_dict[item]

        g = f["mat_bool"][...][
            _idx_map(combined_idx)[:, None],
            _idx_map(combined_idx)
        ]

    return utils.DataDict([
        ("x", x), ("y", y),
        ("protein_id", combined_idx),
    ]), g, train_mask, val_mask, test_mask


def main():
    cmd_args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = cmd_args.device
    if cmd_args.seed is not None:
        random.seed(cmd_args.seed)
        np.random.seed(cmd_args.seed)
        tf.set_random_seed(cmd_args.seed)

    print("Reading data...")
    data, graph, train_mask, val_mask, test_mask = \
        read_data(cmd_args.x, cmd_args.y, cmd_args.g, cmd_args.split)

    print("Building model...")
    train_val_mask = np.logical_or(train_mask, val_mask)
    model = GCNPredictor(
        path=cmd_args.output_path,
        input_dim=data["x"].shape[1], graph=graph,
        fc_depth=2, fc_dim=500,
        class_num=data["y"].shape[1],
        class_weights=[(
            0.5 * train_val_mask.sum() / (
                train_val_mask.sum() - data["y"][train_val_mask, i].sum()),
            0.5 * train_val_mask.sum() / data["y"][train_val_mask, i].sum()
        ) for i in range(data["y"].shape[1])]
    )
    model.compile(lr=1e-3)
    if os.path.exists(os.path.join(cmd_args.output_path, "final")):
        print("Loading existing weights...")
        model.load(os.path.join(cmd_args.output_path, "final"))

    if not cmd_args.no_fit:
        print("Fitting model...")
        model.fit(data, train_mask, val_mask,
                  epoch=1000, patience=10)
        model.save(os.path.join(cmd_args.output_path, "final"))

    print("Evaluating result...")
    print("#### Training set ####")
    utils.evaluate(data["y"][train_val_mask],
                   model.predict(data, train_val_mask),
                   cutoff=0)
    print("#### Testing set ####")
    utils.evaluate(data["y"][test_mask],
                   model.predict(data, test_mask),
                   cutoff=0)


if __name__ == "__main__":
    main()
    print("Done!")
