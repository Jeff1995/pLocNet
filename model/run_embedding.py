#!/usr/bin/env python

import os
import random
import argparse
import numpy as np
import tensorflow as tf
import h5py
import utils
from embedding import ProteinEmbedding
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", type=str, required=True)
    parser.add_argument("-o", "--output-path", dest="output_path",
                        type=str, required=True)
    parser.add_argument("-s", "--seed", dest="seed", type=int, default=None)
    parser.add_argument("-d", "--device", dest="device", type=str, default="")
    return parser.parse_args()


def read_data(file_name):
    with h5py.File(file_name, "r") as f:
        data_dict = utils.DataDict()
        data_dict["sequence"] = f["mat"][:, 1:, :]  # Drop first 'M'
        data_dict["protein_id"] = f["protein_id"].value
    return data_dict


def main():
    cmd_args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = cmd_args.device
    if cmd_args.seed is not None:
        random.seed(cmd_args.seed)
        np.random.seed(cmd_args.seed)
        tf.set_random_seed(cmd_args.seed)

    print("Reading data...")
    data_dict = read_data(cmd_args.input)
    aa_freq = data_dict["sequence"].sum(axis=(0, 1)).astype(np.float32)
    aa_freq /= aa_freq.sum()

    print("Building model...")
    model = ProteinEmbedding(
        path=cmd_args.output_path,
        input_len=data_dict["sequence"].shape[1],
        input_channel=data_dict["sequence"].shape[2],
        kernel_num=1000, kernel_len=5,
        fc_depth=1, fc_dim=1000, latent_dim=100,
        noise_distribution=aa_freq
    )
    model.compile(lr=1e-3)

    print("Fitting CNN...")
    model.fit(data_dict, val_split=0.1, batch_size=128,
              epoch=50, patience=3, stage="CNN")
    print("Fitting VAE...")
    model.fit(data_dict, val_split=0.1, batch_size=128,
              epoch=100, patience=5, stage="VAE")

    print("Saving result...")
    protein_vec = model.inference(data_dict)
    with h5py.File(os.path.join(cmd_args.output_path, "result.h5"), "w") as f:
        f.create_dataset("protein_vec", data=protein_vec)
        f.create_dataset("protein_id", data=data_dict["protein_id"])


if __name__ == "__main__":
    main()
    print("Done!")
