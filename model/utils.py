import functools
from collections import OrderedDict
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


encode = np.vectorize(lambda x: str(x).encode("utf-8"))
decode = np.vectorize(lambda x: x.decode("utf-8"))


def minibatch(batch_size, desc, use_last=False):
    def minibatch_wrapper(func):
        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            total_size = args[0].shape[0]
            if use_last:
                n_batch = np.ceil(
                    total_size / float(batch_size)
                ).astype(np.int)
            else:
                n_batch = max(1, np.floor(
                    total_size / float(batch_size)
                ).astype(np.int))
            for batch_idx in tqdm(range(n_batch), desc=desc,
                                  unit="batches", leave=False):
                start = batch_idx * batch_size
                end = min((batch_idx + 1) * batch_size, total_size)
                this_args = (item[start:end] for item in args)
                func(*this_args, **kwargs)
        return wrapped_func
    return minibatch_wrapper


def unique(idx, mat):
    assert len(idx) == mat.shape[0]
    unique_idx, count = np.unique(idx, return_counts=True)
    unique_idx = unique_idx[count == 1]
    mask = np.in1d(idx, unique_idx)
    return idx[mask], mat[mask]


def valid_kmaxpooling(ptr, mask, k=10, parallel_iterations=32):

    def _valid_kmaxpooling(init, packed):
        mask = tf.cast(packed[:, 0], tf.bool)
        x = packed[:, 1:]
        return tf.reduce_sum(tf.transpose(tf.nn.top_k(tf.transpose(
            tf.boolean_mask(x, mask)
        ), k=k, sorted=False)[0]), axis=0)

    return tf.scan(
        _valid_kmaxpooling,
        tf.concat([tf.expand_dims(
            tf.cast(mask, dtype=tf.float32), axis=2
        ), ptr], axis=2),
        initializer=tf.constant(
            0, shape=(ptr.get_shape().as_list()[2], ),
            dtype=tf.float32
        ), parallel_iterations=parallel_iterations,
        back_prop=True, swap_memory=False
    )


def evaluate(model, data_dict):
    true = data_dict["y"]
    pred = model.predict(data_dict)
    auc = np.vectorize(
        lambda i, true=true, pred=pred: roc_auc_score(true[:, i], pred[:, i])
    )(np.arange(true.shape[1]))
    print("AUC: ", end="")
    print(auc)


class DataDict(OrderedDict):

    def shuffle(self):
        shuffled, shuffle_idx = DataDict(), None
        for item in self:
            shuffle_idx = np.random.permutation(self[item].shape[0]) \
                if shuffle_idx is None else shuffle_idx
            shuffled[item] = self[item][shuffle_idx]
        return shuffled

    def size(self):
        data_size = set([item.shape[0] for item in self.values()])
        assert len(data_size) == 1
        return data_size.pop()

    @property
    def shape(self):  # Compatibility with numpy arrays
        return [self.size()]

    def __getitem__(self, fetch):
        if isinstance(fetch, slice) or isinstance(fetch, np.ndarray):
            return DataDict([
                (item, self[item][fetch]) for item in self
            ])
        return super(DataDict, self).__getitem__(fetch)
