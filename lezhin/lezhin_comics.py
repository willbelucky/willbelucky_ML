# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2017. 11. 11.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

# The mnist dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 2

# The mnist units are always 28x28 pixels.
UNIT_NUMBER = 166


def inference(units, hidden_units, dropout=None):
    """Build the mnist model up to where it may be used for inference.

    Args:
      units: Units placeholder, from inputs().
      hidden_units: Size of the hidden layers.
      dropout: The dropout rate, between 0 and 1. E.g. "rate=0.1" would drop out 10% of input units.

    Returns:
      softmax_linear: Output tensor with the computed logits.
    """
    pre_unit = UNIT_NUMBER
    pre_layer = units
    for i, hidden_unit in enumerate(hidden_units):
        # Hidden n
        with tf.name_scope('hidden{}'.format(i + 1)):
            weights = tf.Variable(
                tf.truncated_normal([pre_unit, hidden_unit],
                                    stddev=1.0 / math.sqrt(float(pre_unit))),
                name='weights')
            biases = tf.Variable(tf.zeros([hidden_unit]),
                                 name='biases')
            hidden_layer = tf.nn.relu(tf.matmul(pre_layer, weights) + biases)
            if dropout is not None:
                hidden_layer = tf.layers.dropout(hidden_layer, rate=dropout, training=True)

            pre_unit = hidden_unit
            pre_layer = hidden_layer
    # Linear
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(
            tf.truncated_normal([pre_unit, NUM_CLASSES],
                                stddev=1.0 / math.sqrt(float(pre_unit))),
            name='weights')
        biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                             name='biases')
        logits = tf.matmul(hidden_layer, weights) + biases
    return logits


def loss(logits, labels):
    """Calculates the loss from the logits and the labels.

    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size].

    Returns:
      loss: Loss tensor of type float.
    """
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='xentropy')
    return tf.reduce_mean(cross_entropy, name='xentropy_mean')


def training(loss, learning_rate):
    """Sets up the training Ops.

    Creates a summarizer to track the loss over time in TensorBoard.

    Creates an optimizer and applies the gradients to all trainable variables.

    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.

    Args:
      loss: Loss tensor, from loss().
      learning_rate: The learning rate to use for gradient descent.

    Returns:
      train_op: The Op for training.
    """
    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar('loss', loss)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.

    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size], with values in the
        range [0, NUM_CLASSES).

    Returns:
      A scalar int32 tensor with the number of examples (out of batch_size)
      that were predicted correctly.
    """
    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label is in the top k (here k=1)
    # of all logits for that example.
    correct = tf.nn.in_top_k(logits, labels, 1)
    # Return the number of true entries.
    return tf.reduce_sum(tf.cast(correct, tf.int32))
