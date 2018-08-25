import numpy as np
import tensorflow as tf
import model
import utils


class ProteinEmbedding(model.Model):

    def _init_graph(
        self, input_len, input_channel,
        kernel_num=100, kernel_len=3, pool_size=10,
        fc_depth=1, fc_dim=100, latent_dim=10,
        noise_distribution=None, dropout_rate=0.5,
        eps=1e-8, **kwargs
    ):

        # CNN
        with tf.name_scope("placeholder"):
            self.sequence = tf.placeholder(
                dtype=tf.float32, shape=(None, input_len, input_channel),
                name="sequence"
            )
            self.training_flag = tf.placeholder(
                dtype=tf.bool, shape=(), name="training_flag")

        with tf.name_scope("cnn/noise"):
            noise_distribution = tf.constant(noise_distribution) \
                if noise_distribution is not None \
                else tf.ones(input_channel) / input_channel
            noise = tf.distributions.Categorical(
                probs=noise_distribution
            ).sample(tf.shape(self.sequence)[0:2])
            noise = tf.one_hot(noise, depth=input_channel)

        with tf.variable_scope("cnn/conv", reuse=tf.AUTO_REUSE):
            ptr = tf.nn.relu(tf.layers.conv1d(
                self.sequence, filters=kernel_num, kernel_size=kernel_len,
                padding="valid", use_bias=False, name="conv1d"
            ))
            noise = tf.nn.relu(tf.layers.conv1d(
                noise, filters=kernel_num, kernel_size=kernel_len,
                padding="valid", use_bias=False, name="conv1d"
            ))  # Weights are shared

        with tf.name_scope("cnn/pool"):  # Global valid pooling
            seq_len = tf.reduce_sum(self.sequence, axis=(1, 2), name="seq_len")
            valid_mask = tf.sequence_mask(
                lengths=seq_len - (kernel_len - 1),
                maxlen=input_len - (kernel_len - 1)
            )
            ptr = self.conv = utils.valid_kmaxpooling(
                ptr, valid_mask, k=pool_size)
            noise = utils.valid_kmaxpooling(noise, valid_mask, k=pool_size)
            ptr = tf.concat([ptr, noise], axis=0)

        with tf.variable_scope("cnn/fc"):
            ptr = tf.layers.dropout(
                ptr, rate=dropout_rate, training=self.training_flag)
            for l in range(fc_depth):
                ptr = tf.layers.dense(
                    ptr, units=fc_dim, activation=tf.nn.leaky_relu,
                    name="layer_%d" % l
                )
                ptr = tf.layers.dropout(
                    ptr, rate=dropout_rate, training=self.training_flag)
            ptr = tf.layers.dense(ptr, units=1)

        with tf.name_scope("cnn/label"):
            label = tf.reshape(tf.concat([
                tf.ones(tf.shape(self.sequence)[0]),
                tf.zeros(tf.shape(noise)[0])
            ], axis=0), (-1, 1))

        with tf.name_scope("cnn/loss"):
            self.cnn_loss = tf.losses.sigmoid_cross_entropy(label, ptr)

        # VAE
        ptr = self.conv
        with tf.variable_scope("vae/encoder"):
            for l in range(fc_depth):
                ptr = tf.layers.dense(
                    ptr, units=fc_dim, activation=tf.nn.leaky_relu,
                    name="layer_%d" % l
                )
                ptr = tf.layers.batch_normalization(
                    ptr, center=True, scale=True,
                    training=self.training_flag
                )
            self.zm = tf.layers.dense(ptr, units=latent_dim, name="zm")
            self.log_zv = tf.layers.dense(ptr, units=latent_dim, name="log_zv")

        with tf.name_scope("vae/stochastic"):
            ptr = tf.random_normal(
                tf.shape(self.zm),
                mean=self.zm, stddev=tf.sqrt(tf.exp(self.log_zv))
            )

        with tf.variable_scope("vae/decoder"):
            for l in range(fc_depth):
                ptr = tf.layers.dense(
                    ptr, units=fc_dim, activation=tf.nn.leaky_relu,
                    name="layer_%d" % l
                )
            xa = tf.layers.dense(ptr, units=kernel_num,
                                 activation=tf.nn.softplus, name="xa")
            log_xb = tf.layers.dense(ptr, units=kernel_num, name="log_xb")
            # log_xb = tf.ones_like(xa, name="log_xb")

        with tf.name_scope("vae/loss"):
            log_likelihood = tf.reduce_sum(
                xa * log_xb - tf.lgamma(xa) +
                (xa - 1) * tf.log(eps + self.conv) -
                tf.exp(log_xb) * self.conv,
                axis=1, name="log_likelihood"
            )  # Gamma distribution
            kl = 0.5 * tf.reduce_sum(
                tf.square(self.zm) + tf.exp(self.log_zv) - self.log_zv - 1,
                axis=1
            )
            self.vae_loss = -tf.reduce_mean(log_likelihood - kl)

    def _compile(self, **kwargs):
        with tf.name_scope("optimize_cnn"):
            self.cnn_step = tf.train.AdamOptimizer(1e-3).minimize(
                self.cnn_loss, var_list=tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, "cnn"
                )
            )
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            with tf.name_scope("optimize_vae"):
                self.vae_step = tf.train.AdamOptimizer(1e-3).minimize(
                    self.vae_loss, var_list=tf.get_collection(
                        tf.GraphKeys.TRAINABLE_VARIABLES, "vae"
                    )
                )

    def _fit_epoch(self, data_dict, batch_size, stage=None, **kwargs):
        if stage == "CNN":
            self._fit_epoch_CNN(data_dict, batch_size, **kwargs)
        elif stage == "VAE":
            self._fit_epoch_VAE(data_dict, batch_size, **kwargs)
        else:
            raise ValueError("Unknown stage!")

    def _val_epoch(self, data_dict, batch_size, stage=None, **kwargs):
        if stage == "CNN":
            return self._val_epoch_CNN(data_dict, batch_size, **kwargs)
        elif stage == "VAE":
            return self._val_epoch_VAE(data_dict, batch_size, **kwargs)
        else:
            raise ValueError("Unknown stage!")

    def _fit_epoch_CNN(self, data_dict, batch_size, **kwargs):
        cnn_loss = 0

        @utils.minibatch(batch_size, desc="training")
        def _train(data_dict):
            nonlocal cnn_loss
            feed_dict = {
                self.sequence: data_dict["sequence"],
                self.training_flag: True
            }
            _, batch_loss = self.sess.run(
                [self.cnn_step, self.cnn_loss],
                feed_dict=feed_dict
            )
            cnn_loss += batch_loss * data_dict.size()

        _train(data_dict)
        cnn_loss /= data_dict.size()
        print("train=%.3f, " % cnn_loss, end="")

        manual_summary = tf.Summary(value=[
            tf.Summary.Value(tag="CNN loss (train)", simple_value=cnn_loss)
        ])
        self.summarizer.add_summary(manual_summary, self.sess.run(self.epoch))

    def _fit_epoch_VAE(self, data_dict, batch_size, **kwargs):
        vae_loss = 0

        @utils.minibatch(batch_size, desc="training")
        def _train(data_dict):
            nonlocal vae_loss
            feed_dict = {
                self.sequence: data_dict["sequence"],
                self.training_flag: True
            }
            _, batch_loss = self.sess.run(
                [self.vae_step, self.vae_loss],
                feed_dict=feed_dict
            )
            vae_loss += batch_loss * data_dict.size()

        _train(data_dict)
        vae_loss /= data_dict.size()
        print("train=%.3f, " % vae_loss, end="")

        manual_summary = tf.Summary(value=[
            tf.Summary.Value(tag="VAE loss (train)", simple_value=vae_loss)
        ])
        self.summarizer.add_summary(manual_summary, self.sess.run(self.epoch))

    def _val_epoch_CNN(self, data_dict, batch_size, **kwargs):
        cnn_loss = 0

        @utils.minibatch(batch_size, desc="validation")
        def _val(data_dict):
            nonlocal cnn_loss
            feed_dict = {
                self.sequence: data_dict["sequence"],
                self.training_flag: False
            }
            batch_loss = self.sess.run(
                self.cnn_loss,
                feed_dict=feed_dict
            )
            cnn_loss += batch_loss * data_dict.size()

        _val(data_dict)
        cnn_loss /= data_dict.size()
        print("val=%.3f, " % cnn_loss, end="")

        manual_summary = tf.Summary(value=[
            tf.Summary.Value(tag="CNN loss (val)", simple_value=cnn_loss)
        ])
        self.summarizer.add_summary(manual_summary, self.sess.run(self.epoch))
        return cnn_loss

    def _val_epoch_VAE(self, data_dict, batch_size, **kwargs):
        vae_loss = 0

        @utils.minibatch(batch_size, desc="validation")
        def _val(data_dict):
            nonlocal vae_loss
            feed_dict = {
                self.sequence: data_dict["sequence"],
                self.training_flag: False
            }
            batch_loss = self.sess.run(
                self.vae_loss,
                feed_dict=feed_dict
            )
            vae_loss += batch_loss * data_dict.size()

        _val(data_dict)
        vae_loss /= data_dict.size()
        print("val=%.3f, " % vae_loss, end="")

        manual_summary = tf.Summary(value=[
            tf.Summary.Value(tag="VAE loss (val)", simple_value=vae_loss)
        ])
        self.summarizer.add_summary(manual_summary, self.sess.run(self.epoch))
        return vae_loss

    def fetch(self, tensor, data_dict, batch_size=128):
        tensor_shape = tuple(
            item for item in tensor.get_shape().as_list() if item is not None)
        result = np.empty((data_dict.size(),) + tuple(tensor_shape))

        @utils.minibatch(batch_size, desc="fetch", use_last=True)
        def _fetch(data_dict, result):
            feed_dict = {
                self.sequence: data_dict["sequence"],
                self.training_flag: False
            }
            result[:] = self.sess.run(tensor, feed_dict=feed_dict)
        _fetch(data_dict, result)
        return result

    def __getitem__(self, key):
        if key in self.__dict__.keys():
            return self.__dict__[key]
        return self.sess.graph.get_tensor_by_name(key + ":0")

    def inference(self, data_dict, batch_size=128):
        return self.fetch(self.zm, data_dict, batch_size)
