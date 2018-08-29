from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import numpy as np

from gcn.utils import *
from gcn.models import GCN, MLP

from load_data_rv import load_data_rv

from metrics import *

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
# flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'dense', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 1000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 100, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 100, 'Number of units in hidden layer 2.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 1000, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data_rv('../../supervised_data/')

# Some preprocessing
features = preprocess_features(features)
if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
model = model_func(placeholders, input_dim=features[2][1], logging=True)

# Initialize session
sess = tf.Session()


# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)

def evaluate2(features, support, labels, mask, placeholders):
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy, model.predict()], feed_dict=feed_dict_val)
    precision = masked_precision(outs_val[2], labels, mask)
    recall = masked_recall(outs_val[2], labels, mask)
    AUC = masked_AUC(outs_val[2], labels, mask)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], precision, recall, AUC

# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

    # Validation
    cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
    cost_val.append(cost)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

# Testing

train_cost, train_acc, train_pre, train_rec, train_auc = evaluate2(features, support, y_train, train_mask, placeholders)
print("Train set results:",
      "cost=", "{:.5f}".format(train_cost),
      "accuracy=", "{:.5f}".format(train_acc),
      "precision=", "{:.5f}".format(train_pre),
      "recall=", "{:.5f}".format(train_rec),
      "AUC=", "{:.5f}".format(train_auc))

val_cost, val_acc, val_pre, val_rec, val_auc = evaluate2(features, support, y_val, val_mask, placeholders)
print("Val set results:",
      "cost=", "{:.5f}".format(val_cost),
      "accuracy=", "{:.5f}".format(val_acc),
      "precision=", "{:.5f}".format(val_pre),
      "recall=", "{:.5f}".format(val_rec),
      "AUC=", "{:.5f}".format(val_auc))

test_cost, test_acc, test_pre, test_rec, test_auc = evaluate2(features, support, y_test, test_mask, placeholders)
print("Test set results:",
      "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc),
      "precision=", "{:.5f}".format(test_pre),
      "recall=", "{:.5f}".format(test_rec),
      "AUC=", "{:.5f}".format(test_auc))