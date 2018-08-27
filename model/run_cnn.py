#!/usr/bin/env python

import os
import random
import argparse
import numpy as np
import tensorflow as tf
import h5py
import utils
from cnn import CNNPredictor
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-x", dest="x", type=str, required=True)
    parser.add_argument("-y", dest="y", type=str, required=True)
    parser.add_argument("--split", dest="split", type=str, required=True)
    parser.add_argument("-o", "--output-path", dest="output_path",
                        type=str, required=True)
    parser.add_argument("-s", "--seed", dest="seed", type=int, default=None)
    parser.add_argument("-d", "--device", dest="device", type=str, default="")
    parser.add_argument("-n", "--no-fit", dest="no_fit",
                        default=False, action="store_true")
    return parser.parse_args()


def read_data(x, y, split):
    with h5py.File(split, "r") as f:
        train_idx = utils.decode(f["train"][...])
        val_idx = utils.decode(f["val"][...])
        test_idx = utils.decode(f["test"][...])

    with h5py.File(x, "r") as f:
        idx = utils.decode(f["protein_id"][...])
        idx_dict = {idx[i]: i for i in range(len(idx))}

        @np.vectorize
        def _idx_map(item):
            return idx_dict[item]

        x_train = f["mat"][...][_idx_map(train_idx)]
        x_val = f["mat"][...][_idx_map(val_idx)]
        x_test = f["mat"][...][_idx_map(test_idx)]

    with h5py.File(y, "r") as f:
        idx = utils.decode(f["protein_id"][...])
        idx_dict = {idx[i]: i for i in range(len(idx))}

        @np.vectorize
        def _idx_map(item):
            return idx_dict[item]

        y_train = f["mat"][:, 0:1][_idx_map(train_idx)]
        y_val = f["mat"][:, 0:1][_idx_map(val_idx)]
        y_test = f["mat"][:, 0:1][_idx_map(test_idx)]

    return utils.DataDict([
        ("x", x_train), ("y", y_train), ("protein_id", train_idx)
    ]), utils.DataDict([
        ("x", x_val), ("y", y_val), ("protein_id", val_idx)
    ]), utils.DataDict([
        ("x", x_test), ("y", y_test), ("protein_id", test_idx)
    ])


def main():
    cmd_args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = cmd_args.device
    if cmd_args.seed is not None:
        random.seed(cmd_args.seed)
        np.random.seed(cmd_args.seed)
        tf.set_random_seed(cmd_args.seed)

    print("Reading data...")
    train_data, val_data, test_data = read_data(
        cmd_args.x, cmd_args.y, cmd_args.split)
    train_val_data = train_data + val_data

    print("Building model...")
    model = CNNPredictor(
        path=cmd_args.output_path,
        input_len=train_val_data["x"].shape[1],
        input_channel=train_val_data["x"].shape[2],
        kernel_num=500, kernel_len=10, fc_depth=2, fc_dim=500,
        class_num=train_val_data["y"].shape[1],
        class_weights=[(
            0.5 * train_val_data.size() / (
                train_val_data.size() - train_val_data["y"][:, i].sum()),
            0.5 * train_val_data.size() / train_val_data["y"][:, i].sum()
        ) for i in range(train_val_data["y"].shape[1])]
    )
    model.compile(lr=1e-3)
    if os.path.exists(os.path.join(cmd_args.output_path, "final")):
        print("Loading existing weights...")
        model.load(os.path.join(cmd_args.output_path, "final"))

    if not cmd_args.no_fit:
        print("Fitting model...")
        model.fit(train_data, val_data, batch_size=128,
                  epoch=1000, patience=10)
        model.save(os.path.join(cmd_args.output_path, "final"))

    print("Evaluating result...")
    print("#### Training set ####")
    utils.evaluate(train_val_data["y"],
                   model.predict(train_val_data),
                   cutoff=0)
    print("#### Testing set ####")
    utils.evaluate(test_data["y"],
                   model.predict(test_data),
                   cutoff=0)


if __name__ == "__main__":
    main()
    print("Done!")
