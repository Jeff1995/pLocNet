import numpy as np
import tensorflow as tf
import model
import utils


class MLPPredictor(model.Model):

    def _init_graph(
        self, input_dim=50, fc_depth=1, fc_dim=100,
        class_num=1, class_weights=None,
        dropout_rate=0.5, **kwargs
    ):
        assert class_weights is None or len(class_weights) == class_num

        with tf.name_scope("placeholder"):
            ptr = self.x = tf.placeholder(
                dtype=tf.float32, shape=(None, input_dim),
                name="input"
            )
            self.y = tf.placeholder(
                dtype=tf.float32, shape=(None, class_num),
                name="label"
            )
            self.training_flag = tf.placeholder(
                dtype=tf.bool, shape=(), name="training_flag"
            )

        with tf.variable_scope("fc"):
            for l in range(fc_depth):
                ptr = tf.layers.dense(
                    ptr, units=fc_dim, activation=tf.nn.leaky_relu,
                    name="layer_%d" % l
                )
                ptr = tf.layers.batch_normalization(
                    ptr, center=True, scale=True,
                    training=self.training_flag
                )
                ptr = tf.layers.dropout(
                    ptr, rate=dropout_rate, training=self.training_flag
                )
            self.pred = tf.layers.dense(ptr, units=class_num)

        with tf.name_scope("loss"):
            self.loss = 0
            for i in range(class_num):
                self.loss += tf.losses.sigmoid_cross_entropy(
                    self.y[:, i], self.pred[:, i]
                ) if class_weights is None else \
                tf.losses.sigmoid_cross_entropy(
                    self.y[:, i], self.pred[:, i],
                    weights=self.y[:, i] * (
                        class_weights[i][1] - class_weights[i][0]
                    ) + class_weights[i][0]
                )

    def _compile(self, lr, **kwargs):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.name_scope("optimize"):
                self.step = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def _fit_epoch(self, data, batch_size, **kwargs):
        loss = 0

        @utils.minibatch(batch_size, desc="training")
        def _train(data):
            nonlocal loss
            feed_dict = {
                self.x: data["x"],
                self.y: data["y"],
                self.training_flag: True
            }
            _, batch_loss = self.sess.run(
                [self.step, self.loss],
                feed_dict=feed_dict
            )
            loss += batch_loss * data.size()

        _train(data)
        loss /= data.size()
        print("train=%.3f, " % loss, end="")

        manual_summary = tf.Summary(value=[
            tf.Summary.Value(tag="loss (train)", simple_value=loss)
        ])
        self.summarizer.add_summary(manual_summary, self.sess.run(self.epoch))

    def _val_epoch(self, data, batch_size, **kwargs):
        loss = 0

        @utils.minibatch(batch_size, desc="validation")
        def _val(data):
            nonlocal loss
            feed_dict = {
                self.x: data["x"],
                self.y: data["y"],
                self.training_flag: False
            }
            batch_loss = self.sess.run(
                self.loss,
                feed_dict=feed_dict
            )
            loss += batch_loss * data.size()

        _val(data)
        loss /= data.size()
        print("val=%.3f, " % loss, end="")

        manual_summary = tf.Summary(value=[
            tf.Summary.Value(tag="loss (val)", simple_value=loss)
        ])
        self.summarizer.add_summary(manual_summary, self.sess.run(self.epoch))
        return loss

    def fetch(self, tensor, data, batch_size=128):
        tensor_shape = tuple(
            item for item in tensor.get_shape().as_list() if item is not None)
        result = np.empty((data.size(),) + tuple(tensor_shape))

        @utils.minibatch(batch_size, desc="fetch", use_last=True)
        def _fetch(data, result):
            feed_dict = {
                self.x: data["x"],
                self.training_flag: False
            }
            result[:] = self.sess.run(tensor, feed_dict=feed_dict)
        _fetch(data, result)
        return result

    def __getitem__(self, key):
        if key in self.__dict__.keys():
            return self.__dict__[key]
        return self.sess.graph.get_tensor_by_name(key + ":0")

    def predict(self, data, batch_size=128):
        return self.fetch(self.pred, data, batch_size)
