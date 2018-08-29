import tensorflow as tf
from sklearn.metrics import roc_auc_score
import numpy as np


def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)

def masked_AUC(preds,labels,mask):
    preds_mask=preds[mask,1]
    labels_mask=labels[mask,1]
    AUC=roc_auc_score(labels_mask,preds_mask)
    return AUC

def masked_precision(preds,labels,mask):
    labels_mask=labels[mask,1]
    preds_mask=preds[mask,1]>0.5
    precision = np.logical_and(preds_mask, labels_mask).sum(axis=0) / preds_mask.sum(axis=0)
    return precision

def masked_recall(preds,labels,mask):
    labels_mask = labels[mask, 1]
    preds_mask = preds[mask, 1] > 0.5
    recall = np.logical_and(preds_mask, labels_mask).sum(axis=0) / labels_mask.sum(axis=0)
    return recall