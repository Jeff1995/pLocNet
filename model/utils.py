import functools
from collections import OrderedDict
import numpy as np
from tqdm import tqdm


# Wraps a batch function into minibatch version
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
