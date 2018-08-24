from builtins import input
import os
import time
import traceback
import numpy as np
import tensorflow as tf
import ipdb
import utils


class Model(object):

    """
    Abstract `Model` class, providing a common framework for initialization and
    training.
    """

    def __init__(self, path, **kwargs):
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        self._init_session(**kwargs)
        self._init_graph(**kwargs)

    def compile(self, **kwargs):
        self._compile(**kwargs)
        self.saver = tf.train.Saver(max_to_keep=1)
        self.sess.run(tf.global_variables_initializer())
        self.summarizer = tf.summary.FileWriter(
            os.path.join(self.path, "summary"),
            graph=self.sess.graph, flush_secs=10
        )

    def _init_session(self, **kwargs):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

    def _init_graph(self, **kwargs):
        raise NotImplementedError(
            "Calling virtual `_init_graph` from `Model`!")

    def _compile(self, **kwargs):
        raise NotImplementedError(
            "Calling virtual `_compile` from `Model`!")

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.saver.save(self.sess, os.path.join(path, "save.ckpt"))

    def load(self, path):
        self.saver.restore(self.sess, os.path.join(path, "save.ckpt"))

    def close(self):
        self.sess.close()

    def fit(self, data_dict, val_split=0.1, epoch=100,
            patience=np.inf, **kwargs):

        """
        This function wraps an epoch-by-epoch update function into
        complete training process that supports data splitting, shuffling
        and early stop.
        """

        data_dict = utils.DataDict(data_dict)

        # Leave out validation set
        data_dict = data_dict.shuffle()
        data_size = data_dict.size()
        train_data_dict = data_dict[int(val_split * data_size):]
        val_data_dict = data_dict[:int(val_split * data_size)]

        # Fit prep
        loss_record = np.inf
        patience_countdown = patience
        saver = tf.train.Saver(max_to_keep=1)
        with tf.variable_scope("epoch", reuse=tf.AUTO_REUSE):
            self.epoch = tf.get_variable(
                "epoch", shape=(), dtype=tf.float32, trainable=False)
        ckpt_file = os.path.join(self.path, "checkpoint")

        # Fit loop
        for epoch_idx in range(epoch):
            try:
                print("[epoch %d] " % epoch_idx, end="")
                self.sess.run(self.epoch.assign(epoch_idx))

                try:
                    t_start = time.time()
                    self._fit_epoch(train_data_dict.shuffle(), **kwargs)
                    loss = self._val_epoch(val_data_dict, **kwargs)
                    print("time elapsed = %.1fs" % (
                        time.time() - t_start
                    ), end="")
                except Exception:
                    print("\n==== Oops! Model has crashed... ====\n")
                    traceback.print_exc()
                    print("\n====================================\n")
                    break

                # Early stop
                if loss < loss_record:
                    print(" Best save...", end="")
                    latest_checkpoint = saver.save(
                        self.sess, ckpt_file, global_step=epoch_idx)
                    patience_countdown = patience
                    loss_record = loss
                else:
                    patience_countdown -= 1
                print()
                if patience_countdown == 0:
                    break

            except KeyboardInterrupt:
                print("\n\n==== Caught keyboard interruption! ====\n")
                success_flag = False
                break_flag = False
                while not success_flag:
                    choice = input("Stop model training? (y/d/c) ")
                    if choice == "y":
                        break_flag = True
                        break
                    if choice == "d":
                        ipdb.set_trace()
                    elif choice == "c":
                        success_flag = True
                if break_flag:
                    break

        # Fit finish
        if "latest_checkpoint" in locals():
            print("Restoring best model...")
            saver.restore(self.sess, latest_checkpoint)
        else:
            print("No checkpoint can be restored...\nTraining failed!")

    def _fit_epoch(self, data_dict, **kwargs):
        raise NotImplementedError(
            "Calling virtual `_fit_epoch` from `Model`!")

    def _val_epoch(self, data_dict, **kwargs):
        raise NotImplementedError(
            "Calling virtual `_val_epoch` from `Model`!")
