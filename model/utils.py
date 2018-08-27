import functools
from collections import OrderedDict
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve


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


def unique(idx, mat=None):  # Only idx with single occurrence will be retained
    if mat is not None:
        assert len(idx) == mat.shape[0]
    unique_idx, count = np.unique(idx, return_counts=True)
    unique_idx = unique_idx[count == 1]
    mask = np.in1d(idx, unique_idx)
    if mat is not None:
        return idx[mask], mat[mask]
    return idx[mask]


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


def graph_conv(input, graph, units, activation=None,
               use_bias=True, name="graph_conv"):
    with tf.variable_scope(name):
        kernel = tf.get_variable("kernel", shape=(
            input.get_shape().as_list()[1], units
        ), dtype=tf.float32)
        if use_bias:
            bias = tf.get_variable(
                "bias", shape=(units, ), dtype=tf.float32)
    ptr = tf.sparse_tensor_dense_matmul(
        graph, tf.matmul(input, kernel)
    ) + bias
    return activation(ptr) if activation is not None else ptr


def evaluate(true, pred, cutoff=None):

    @np.vectorize
    def _auc(i):
        return roc_auc_score(true[:, i], pred[:, i])

    @np.vectorize
    def _cutoff(i):
        fpr, tpr, thresholds = roc_curve(true[:, i], pred[:, i])
        return thresholds[(tpr - fpr).argmax()]

    auc = _auc(np.arange(true.shape[1]))
    print("AUC: ", end="")
    print(auc)
    if cutoff is None:
        cutoff = _cutoff(np.arange(true.shape[1]))
    pred = pred > cutoff
    accuracy = (pred == true).sum(axis=0) / pred.shape[0]
    print("Accuracy: ", end="")
    print(accuracy)
    precision = np.logical_and(pred, true).sum(axis=0) / pred.sum(axis=0)
    print("Precision: ", end="")
    print(precision)
    recall = np.logical_and(pred, true).sum(axis=0) / true.sum(axis=0)
    print("Recall: ", end="")
    print(recall)


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

    def __add__(self, another):
        added = DataDict()
        for item in self:
            assert item in another
            added[item] = np.concatenate(
                [self[item], another[item]], axis=0)
        return added
