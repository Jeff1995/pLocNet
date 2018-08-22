import numpy as np


@np.vectorize
def encode(x):
    return str(x).encode("utf-8")


@np.vectorize
def decode(x):
    return x.decode("utf-8")
