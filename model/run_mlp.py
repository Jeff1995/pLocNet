#!/usr/bin/env python

import os
import random
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import h5py
import utils
from sklearn.metrics import roc_auc_score
from mlp import MLPPredictor
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


encode = np.vectorize(lambda x: str(x).encode("utf-8"))
decode = np.vectorize(lambda x: x.decode("utf-8"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-x", dest="x", type=str, required=True)
    parser.add_argument("-y", dest="y", type=str, required=True)
    parser.add_argument("-o", "--output-path", dest="output_path",
                        type=str, required=True)
    parser.add_argument("-s", "--seed", dest="seed", type=int, default=None)
    parser.add_argument("-d", "--device", dest="device", type=str, default="")
    return parser.parse_args()


def read_data(x, y):
    with h5py.File(x, "r") as f:
        embedding = pd.DataFrame(
            data=f["protein_vec"].value,
            index=decode(f["protein_id"].value)
        )
    with h5py.File(y, "r") as f:
        localization = pd.DataFrame(
            data=f["mat"].value,
            index=decode(f["protein_id"].value),
            columns=decode(f["label"].value)
        )
        localization = localization.iloc[:, 0:3]
    merged = pd.merge(embedding, localization,
                      left_index=True, right_index=True)
    return utils.DataDict([
        ("x", merged.values[:, :embedding.shape[1]]),
        ("y", merged.values[:, embedding.shape[1]:]),
        ("protein_id", merged.index.values)
    ])


def evaluate(model, data_dict):
    true = data_dict["y"]
    pred = model.predict(data_dict)
    auc = np.vectorize(
        lambda i, true=true, pred=pred: roc_auc_score(true[:, i], pred[:, i])
    )(np.arange(true.shape[1]))
    print("AUC: ", end="")
    print(auc)


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
    model = MLPPredictor(
        path=cmd_args.output_path,
        input_dim=train_data_dict["x"].shape[1],
        fc_depth=2, fc_dim=500, class_num=data_dict["y"].shape[1]
    )
    model.compile(lr=1e-4)

    print("Fitting model...")
    model.fit(train_data_dict, val_split=0.1, batch_size=128,
              epoch=100, patience=20)

    print("Evaluating result...")
    print("#### Training set ####")
    evaluate(model, train_data_dict)
    print("#### Testing set ####")
    evaluate(model, test_data_dict)

if __name__ == "__main__":
    main()
    print("Done!")
