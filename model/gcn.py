import os
import time
import traceback
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
import ipdb
import model
import utils


class GCNPredictor(model.Model):

    def _init_graph(
        self, input_dim, graph,
        gc_depth=1, gc_dim=100,
        class_num=1, class_weights=None,
        dropout_rate=0.5, **kwargs
    ):
        assert class_weights is None or len(class_weights) == class_num
        assert np.all(graph == graph.T)

        # Normalize graph matirx
        graph = sp.coo_matrix(graph).astype(np.float32)
        graph += sp.eye(graph.shape[0], dtype=np.float32, format="coo")
        normalizer = np.power(graph.sum(axis=0), -1 / 2)
        graph = graph.multiply(normalizer)
        graph = graph.multiply(normalizer.T)
        self.graph = tf.SparseTensor(
            indices=np.stack([graph.row, graph.col], axis=0).T,
            values=graph.data,
            dense_shape=graph.shape
        )

        with tf.name_scope("placeholder"):
            ptr = self.x = tf.placeholder(
                dtype=tf.float32, shape=(graph.shape[0], input_dim),
                name="input"
            )
            self.y = tf.placeholder(
                dtype=tf.float32, shape=(graph.shape[0], class_num),
                name="label"
            )
            self.mask = tf.placeholder(
                dtype=tf.bool, shape=(graph.shape[0]), name="mask"
            )
            self.training_flag = tf.placeholder(
                dtype=tf.bool, shape=(), name="training_flag"
            )

        with tf.variable_scope("gc"):
            for l in range(gc_depth):
                ptr = utils.graph_conv(
                    ptr, graph=self.graph, units=gc_dim,
                    activation=tf.nn.leaky_relu, name="layer_%d" % l
                )
                ptr = tf.layers.dropout(
                    ptr, rate=dropout_rate, training=self.training_flag
                )
            self.pred = tf.boolean_mask(utils.graph_conv(
                ptr, graph=self.graph, units=class_num
            ), self.mask)

        with tf.name_scope("loss"):
            self.loss, masked_y = 0, tf.boolean_mask(self.y, self.mask)
            for i in range(class_num):
                loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=masked_y[:, i], logits=self.pred[:, i])
                weight = masked_y[:, i] * (
                    class_weights[i][1] - class_weights[i][0]
                ) + class_weights[i][0] if class_weights is not None else 1
                self.loss += tf.reduce_mean(weight * loss)

    def _compile(self, lr, **kwargs):
        with tf.name_scope("optimize"):
            self.step = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def fit(self, data, train_mask, val_mask, epoch=100,
            patience=np.inf, **kwargs):

        # Fit prep
        loss_record = np.inf
        patience_countdown = patience
        saver = tf.train.Saver(max_to_keep=1)
        with tf.variable_scope("epoch", reuse=tf.AUTO_REUSE):
            self.epoch = tf.get_variable(
                "epoch", shape=(), dtype=tf.int32, trainable=False)
        ckpt_file = os.path.join(self.path, "checkpoint")

        # Fit loop
        for epoch_idx in range(epoch):
            try:
                print("[epoch %d] " % epoch_idx, end="")
                self.sess.run(self.epoch.assign(epoch_idx))

                try:
                    t_start = time.time()
                    self._fit_epoch(data, train_mask, **kwargs)
                    loss = self._val_epoch(data, val_mask, **kwargs)
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

    def _fit_epoch(self, data, mask, **kwargs):
        feed_dict = {
            self.x: data["x"],
            self.y: data["y"],
            self.mask: mask,
            self.training_flag: True
        }
        _, loss = self.sess.run(
            [self.step, self.loss],
            feed_dict=feed_dict
        )
        print("train=%.3f, " % loss, end="")

        manual_summary = tf.Summary(value=[
            tf.Summary.Value(tag="loss (train)", simple_value=loss)
        ])
        self.summarizer.add_summary(manual_summary, self.sess.run(self.epoch))

    def _val_epoch(self, data, mask, **kwargs):
        feed_dict = {
            self.x: data["x"],
            self.y: data["y"],
            self.mask: mask,
            self.training_flag: False
        }
        loss = self.sess.run(self.loss, feed_dict=feed_dict)
        print("val=%.3f, " % loss, end="")

        manual_summary = tf.Summary(value=[
            tf.Summary.Value(tag="loss (val)", simple_value=loss)
        ])
        self.summarizer.add_summary(manual_summary, self.sess.run(self.epoch))
        return loss

    def fetch(self, tensor, data, mask):
        feed_dict = {
            self.x: data["x"],
            self.mask: mask,
            self.training_flag: False
        }
        return self.sess.run(tensor, feed_dict=feed_dict)

    def __getitem__(self, key):
        if key in self.__dict__.keys():
            return self.__dict__[key]
        return self.sess.graph.get_tensor_by_name(key + ":0")

    def predict(self, data, mask):
        return self.fetch(self.pred, data, mask)


class ConvGCNPredictor(GCNPredictor):

    def _init_graph(
        self, input_len, input_channel, graph,
        kernel_num=100, kernel_len=3, pool_size=10,
        gc_depth=1, gc_dim=100,
        class_num=1, class_weights=None,
        dropout_rate=0.5, **kwargs
    ):
        assert class_weights is None or len(class_weights) == class_num
        assert np.all(graph == graph.T)

        # Normalize graph matirx
        graph = sp.coo_matrix(graph).astype(np.float32)
        graph += sp.eye(graph.shape[0], dtype=np.float32, format="coo")
        normalizer = np.power(graph.sum(axis=0), -1 / 2)
        graph = graph.multiply(normalizer)
        graph = graph.multiply(normalizer.T)
        self.graph = tf.SparseTensor(
            indices=np.stack([graph.row, graph.col], axis=0).T,
            values=graph.data,
            dense_shape=graph.shape
        )

        with tf.name_scope("placeholder"):
            ptr = self.x = tf.placeholder(
                dtype=tf.float32,
                shape=(graph.shape[0], input_len, input_channel),
                name="input"
            )
            self.y = tf.placeholder(
                dtype=tf.float32, shape=(graph.shape[0], class_num),
                name="label"
            )
            self.mask = tf.placeholder(
                dtype=tf.bool, shape=(graph.shape[0]), name="mask"
            )
            self.training_flag = tf.placeholder(
                dtype=tf.bool, shape=(), name="training_flag"
            )

        with tf.variable_scope("conv"):
            ptr = tf.nn.relu(tf.layers.conv1d(
                self.x, filters=kernel_num, kernel_size=kernel_len,
                padding="valid", use_bias=False, name="conv1d"
            ))

        with tf.name_scope("pool"):  # Global valid pooling
            seq_len = tf.reduce_sum(self.x, axis=(1, 2), name="seq_len")
            valid_mask = tf.sequence_mask(
                lengths=seq_len - (kernel_len - 1),
                maxlen=input_len - (kernel_len - 1)
            )
            ptr = self.conv = utils.valid_kmaxpooling(
                ptr, valid_mask, k=pool_size)

        with tf.variable_scope("gc"):
            for l in range(gc_depth):
                ptr = utils.graph_conv(
                    ptr, graph=self.graph, units=gc_dim,
                    activation=tf.nn.leaky_relu, name="layer_%d" % l
                )
                ptr = tf.layers.dropout(
                    ptr, rate=dropout_rate, training=self.training_flag
                )
            self.pred = tf.boolean_mask(utils.graph_conv(
                ptr, graph=self.graph, units=class_num
            ), self.mask)

        with tf.name_scope("loss"):
            self.loss, masked_y = 0, tf.boolean_mask(self.y, self.mask)
            for i in range(class_num):
                loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=masked_y[:, i], logits=self.pred[:, i])
                weight = masked_y[:, i] * (
                    class_weights[i][1] - class_weights[i][0]
                ) + class_weights[i][0] if class_weights is not None else 1
                self.loss += tf.reduce_mean(weight * loss)
