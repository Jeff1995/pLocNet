#!/usr/bin/env python

import os
import random
import argparse
import numpy as np
import tensorflow as tf
import h5py
import utils
from gcn import ConvGCNPredictor
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
    with h5py.File(x, "r") as x_file, \
         h5py.File(y, "r") as y_file, \
         h5py.File(g, "r") as g_file:
        x_idx, x = utils.unique(
            utils.decode(x_file["protein_id"][...]),
            x_file["mat"])
        y_idx, y = utils.unique(
            utils.decode(y_file["protein_id"][...]),
            y_file["mat"])
        g_idx, g = utils.decode(g_file["protein_id"][...]), \
            g_file["mat_bool"][...]  # Confident that g is unique
        xg_idx = np.intersect1d(x_idx, g_idx)

        x = x[utils.get_idx_mapper(x_idx)(xg_idx)]
        g_extract = utils.get_idx_mapper(g_idx)(xg_idx)
        g = g[g_extract[:, None], g_extract]
        y = y[:, 0:1]
        y = np.concatenate([y, np.zeros(
            (1,) + y.shape[1:], dtype=y.dtype.type
        )], axis=0)  # Fill zeros if not existing
        y = y[utils.get_idx_mapper(y_idx)(xg_idx)]

    with h5py.File(split, "r") as f:
        train_idx = utils.decode(f["train"][...])
        val_idx = utils.decode(f["val"][...])
        test_idx = utils.decode(f["test"][...])
        xyg_idx = np.intersect1d(xg_idx, y_idx)
        assert np.all(np.in1d(train_idx, xyg_idx)) \
            and np.all(np.in1d(val_idx, xyg_idx)) \
            and np.all(np.in1d(test_idx, xyg_idx))
        train_mask = np.in1d(xg_idx, train_idx)
        val_mask = np.in1d(xg_idx, val_idx)
        test_mask = np.in1d(xg_idx, test_idx)

    return utils.DataDict([
        ("x", x), ("y", y),
        ("protein_id", xg_idx),
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
    model = ConvGCNPredictor(
        path=cmd_args.output_path,
        input_len=data["x"].shape[1],
        input_channel=data["x"].shape[2],
        graph=graph,
        kernel_num=500, kernel_len=10, pool_size=10,
        gc_depth=1, gc_dim=500,
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
                  epoch=1000, patience=50)
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
