import numpy as np
import tensorflow as tf
import model
import utils


class CNNPredictor(model.Model):

    def _init_graph(
        self, input_len, input_channel,
        kernel_num=100, kernel_len=3,
        fc_depth=1, fc_dim=100,
        class_num=1, **kwargs
    ):

        with tf.name_scope("placeholder"):
            ptr = self.x = tf.placeholder(
                dtype=tf.float32, shape=(None, input_len, input_channel),
                name="sequence"
            )
            self.y = tf.placeholder(
                dtype=tf.float32, shape=(None, class_num),
                name="label"
            )

        with tf.variable_scope("conv", reuse=tf.AUTO_REUSE):
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
            ptr = self.conv = tf.scan(
                self._valid_pooling,
                tf.concat([tf.expand_dims(
                    tf.cast(valid_mask, dtype=tf.float32), axis=2
                ), ptr], axis=2),
                initializer=tf.constant(
                    0, shape=(kernel_num, ), dtype=tf.float32
                ), parallel_iterations=32, back_prop=True, swap_memory=False
            )

        with tf.variable_scope("fc"):
            for l in range(fc_depth):
                ptr = tf.layers.dense(
                    ptr, units=fc_dim, activation=tf.nn.leaky_relu,
                    name="layer_%d" % l
                )
            self.pred = tf.layers.dense(ptr, units=class_num)

        with tf.name_scope("loss"):
            self.loss = 0
            for i in range(class_num):
                self.loss += tf.losses.sigmoid_cross_entropy(
                    self.y[:, i], self.pred[:, i]
                )

    def _compile(self, lr, **kwargs):
        with tf.name_scope("optimize"):
            self.step = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def _fit_epoch(self, data_dict, batch_size, **kwargs):
        loss = 0

        @utils.minibatch(batch_size, desc="training")
        def _train(data_dict):
            nonlocal loss
            feed_dict = {
                self.x: data_dict["x"],
                self.y: data_dict["y"]
            }
            _, batch_loss = self.sess.run(
                [self.step, self.loss],
                feed_dict=feed_dict
            )
            loss += batch_loss * data_dict.size()

        _train(data_dict)
        loss /= data_dict.size()
        print("train=%.3f, " % loss, end="")

        manual_summary = tf.Summary(value=[
            tf.Summary.Value(tag="loss (train)", simple_value=loss)
        ])
        self.summarizer.add_summary(manual_summary, self.sess.run(self.epoch))

    def _val_epoch(self, data_dict, batch_size, **kwargs):
        loss = 0

        @utils.minibatch(batch_size, desc="validation")
        def _val(data_dict):
            nonlocal loss
            feed_dict = {
                self.x: data_dict["x"],
                self.y: data_dict["y"]
            }
            batch_loss = self.sess.run(
                self.loss,
                feed_dict=feed_dict
            )
            loss += batch_loss * data_dict.size()

        _val(data_dict)
        loss /= data_dict.size()
        print("val=%.3f, " % loss, end="")

        manual_summary = tf.Summary(value=[
            tf.Summary.Value(tag="loss (val)", simple_value=loss)
        ])
        self.summarizer.add_summary(manual_summary, self.sess.run(self.epoch))
        return loss

    def fetch(self, tensor, data_dict, batch_size=128):
        tensor_shape = tuple(
            item for item in tensor.get_shape().as_list() if item is not None)
        result = np.empty((data_dict.size(),) + tuple(tensor_shape))

        @utils.minibatch(batch_size, desc="fetch", use_last=True)
        def _fetch(data_dict, result):
            feed_dict = {
                self.x: data_dict["x"]
            }
            result[:] = self.sess.run(tensor, feed_dict=feed_dict)
        _fetch(data_dict, result)
        return result

    def __getitem__(self, key):
        if key in self.__dict__.keys():
            return self.__dict__[key]
        return self.sess.graph.get_tensor_by_name(key + ":0")

    def predict(self, data_dict, batch_size=128):
        return self.fetch(self.pred, data_dict, batch_size)

    @staticmethod
    def _valid_pooling(init, packed):
        mask = tf.cast(packed[:, 0], tf.bool)
        x = packed[:, 1:]
        return tf.reduce_sum(tf.boolean_mask(x, mask), axis=0)
