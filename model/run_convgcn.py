#!/usr/bin/env python

import os
import random
import argparse
import numpy as np
import tensorflow as tf
import h5py
import utils
from gcn import ConvGCNPredictor
from run_gcn import parse_args, read_data, evaluate
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


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
    evaluate(model, data, train_val_mask, test_mask, cmd_args.output_path)


if __name__ == "__main__":
    main()
    print("Done!")
