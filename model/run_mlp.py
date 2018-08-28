#!/usr/bin/env python

import os
import random
import argparse
import numpy as np
import tensorflow as tf
import h5py
import utils
from mlp import MLPPredictor
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
        all_idx = np.concatenate([train_idx, val_idx, test_idx], axis=0)

    with h5py.File(x, "r") as f:
        idx, mat = utils.unique(utils.decode(f["protein_id"][...]), f["mat"])
        assert np.all(np.in1d(all_idx, idx))
        idx_mapper = utils.get_idx_mapper(idx)
        x_train = mat[idx_mapper(train_idx)]
        x_val = mat[idx_mapper(val_idx)]
        x_test = mat[idx_mapper(test_idx)]

    with h5py.File(y, "r") as f:
        idx, mat = utils.unique(utils.decode(f["protein_id"][...]), f["mat"])
        assert np.all(np.in1d(all_idx, idx))
        idx_mapper = utils.get_idx_mapper(idx)
        y_train = mat[idx_mapper(train_idx)]
        y_val = mat[idx_mapper(val_idx)]
        y_test = mat[idx_mapper(test_idx)]

    return utils.DataDict([
        ("x", x_train), ("y", y_train), ("protein_id", train_idx)
    ]), utils.DataDict([
        ("x", x_val), ("y", y_val), ("protein_id", val_idx)
    ]), utils.DataDict([
        ("x", x_test), ("y", y_test), ("protein_id", test_idx)
    ])


def evaluate(model, train_data, test_data, output_path):

    print("#### Training set ####")
    train_pred = model.predict(train_data)
    utils.evaluate(train_data["y"], train_pred, cutoff=0)
    print("#### Testing set ####")
    test_pred = model.predict(test_data)
    utils.evaluate(test_data["y"], test_pred, cutoff=0)

    with h5py.File(os.path.join(output_path, "result.h5"), "w") as f:
        g = f.create_group("train")
        g.create_dataset("true", data=train_data["y"])
        g.create_dataset("pred", data=train_pred)
        g = f.create_group("test")
        g.create_dataset("true", data=test_data["y"])
        g.create_dataset("pred", data=test_pred)


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
    model = MLPPredictor(
        path=cmd_args.output_path,
        input_dim=train_val_data["x"].shape[1], fc_depth=2, fc_dim=500,
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
    evaluate(model, train_val_data, test_data, cmd_args.output_path)


if __name__ == "__main__":
    main()
    print("Done!")
