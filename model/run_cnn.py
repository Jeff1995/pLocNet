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
    parser.add_argument("-o", "--output-path", dest="output_path",
                        type=str, required=True)
    parser.add_argument("-s", "--seed", dest="seed", type=int, default=None)
    parser.add_argument("-d", "--device", dest="device", type=str, default="")
    parser.add_argument("-n", "--no-fit", dest="no_fit",
                        default=False, action="store_true")
    return parser.parse_args()


def read_data(x, y):
    with h5py.File(x, "r") as f:
        seq_idx, seq = utils.unique(
            utils.decode(f["protein_id"].value),
            f["mat"].value
        )
        seq_map = {seq_idx[i]: i for i in range(len(seq_idx))}
    with h5py.File(y, "r") as f:
        loc_idx, loc = utils.unique(
            utils.decode(f["protein_id"].value),
            f["mat"].value[:, 0:3]
        )
        loc_map = {loc_idx[i]: i for i in range(len(loc_idx))}
    common_names = np.intersect1d(seq_idx, loc_idx)
    return utils.DataDict([
        ("x", seq[np.vectorize(
            lambda x, seq_map=seq_map: seq_map[x]
        )(common_names)]),
        ("y", loc[np.vectorize(
            lambda x, loc_map=loc_map: loc_map[x]
        )(common_names)]),
        ("protein_id", common_names)
    ])


def main():
    cmd_args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = cmd_args.device
    if cmd_args.seed is not None:
        random.seed(cmd_args.seed)
        np.random.seed(cmd_args.seed)
        tf.set_random_seed(cmd_args.seed)

    print("Reading data...")
    data_dict = read_data(cmd_args.x, cmd_args.y)
    data_dict = data_dict.shuffle()
    train_size = np.floor(float(data_dict.size()) * 0.8).astype(int)
    train_data_dict = data_dict[:train_size]
    test_data_dict = data_dict[train_size:]

    print("Building model...")
    model = CNNPredictor(
        path=cmd_args.output_path,
        input_len=train_data_dict["x"].shape[1],
        input_channel=train_data_dict["x"].shape[2],
        kernel_num=500, kernel_len=10,
        fc_depth=2, fc_dim=500, class_num=data_dict["y"].shape[1]
    )
    model.compile(lr=1e-4)
    if os.path.exists(os.path.join(cmd_args.output_path, "final")):
        print("Loading existing weights...")
        model.load(os.path.join(cmd_args.output_path, "final"))

    if not cmd_args.no_fit:
        print("Fitting model...")
        model.fit(train_data_dict, val_split=0.1, batch_size=128,
                  epoch=100, patience=20)
        model.save(os.path.join(cmd_args.output_path, "final"))

    print("Evaluating result...")
    print("#### Training set ####")
    utils.evaluate(model, train_data_dict)
    print("#### Testing set ####")
    utils.evaluate(model, test_data_dict)


if __name__ == "__main__":
    main()
    print("Done!")
