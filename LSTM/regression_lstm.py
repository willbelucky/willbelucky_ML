# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2017. 11. 29.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from data.data_reader import read_data, to_recurrent_data
from util import logging


def lstm_cell(hidden_unit, dropout):
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_unit, state_is_tuple=True, activation=tf.tanh)
    if dropout is not None:
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=(1 - dropout))
    return cell


def inference(units, hidden_units, column_number, class_number, batch_size, dropout=None):
    """Build the mnist_example model up to where it may be used for inference.

    Args:
      units: Units placeholder, from inputs().
      hidden_units: Size of the hidden layers.
      column_number: Size of the input columns.
      class_number: Size of the output classes.
      batch_size: Size of a batch
      dropout: The dropout rate, between 0 and 1. E.g. "rate=0.1" would drop out 10% of input units.

    Returns:
      softmax_linear: Output tensor with the computed logits.
    """
    stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell(hidden_unit, dropout) for hidden_unit in hidden_units])
    outputs, _states = tf.nn.dynamic_rnn(stacked_lstm, units, dtype=tf.float32)
    dense = tf.layers.dense(outputs[:, -1], batch_size)
    predictions = tf.layers.dense(dense, column_number)

    # Linear
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(
            tf.truncated_normal([column_number, class_number],
                                stddev=1.0 / math.sqrt(float(column_number))),
            name='weights')
        biases = tf.Variable(tf.zeros([class_number]),
                             name='biases')
        logits = tf.matmul(predictions, weights) + biases
    return logits


def do_loss(logits, labels):
    """Calculates the loss from the logits and the labels.

    Args:
      logits: Logits tensor, float32 - [batch_size, NUM_CLASSES].
      labels: Labels tensor, float32 - [batch_size].

    Returns:
      loss: Loss tensor of type float.
    """
    # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='xentropy')
    return tf.reduce_sum(tf.square(logits - labels))


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
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.

    Args:
      logits: Logits tensor, float32 - [batch_size, 1].
      labels: Labels tensor, float32 - [batch_size], with values.

    Returns:
      A scalar float32 tensor with the number of examples (out of batch_size)
      that were predicted correctly.
    """
    return tf.reduce_sum(tf.square(logits - labels))


def placeholder_inputs(batch_size, time_step, column_number):
    """Generate placeholder variables to represent the input tensors.
    These placeholders are used as inputs by the rest of the model building
    code and will be fed from the downloaded data in the .run() loop, below.
    Args:
      batch_size: The batch size will be baked into both placeholders.
      time_step: The number of time steps.
      column_number: Size of the input columns.
    Returns:
      units_placeholder: Units placeholder.
      labels_placeholder: Labels placeholder.
    """
    # Note that the shapes of the placeholders match the shapes of the full
    # unit and label tensors, except the first dimension is now batch_size
    # rather than the full size of the train or test data sets.
    units_placeholder = tf.placeholder(tf.float32, [batch_size, time_step, column_number])
    labels_placeholder = tf.placeholder(tf.float32, [batch_size])
    return units_placeholder, labels_placeholder


def fill_feed_dict(data_set, units_pl, labels_pl, flags):
    """Fills the feed_dict for training the given step.
    A feed_dict takes the form of:
    feed_dict = {
        <placeholder>: <tensor of values to be passed for placeholder>,
        ....
    }
    Args:
      data_set: The set of units and labels, from data.read_data_sets()
      units_pl: The units placeholder, from placeholder_inputs().
      labels_pl: The labels placeholder, from placeholder_inputs().
      flags: The given options.
    Returns:
      feed_dict: The feed dictionary mapping from placeholders to values.
    """
    # Create the feed_dict for the placeholders filled with the next
    # `batch size` examples.
    units_feed, labels_feed, dates_feed = data_set.next_batch(fake_data=flags.fake_data,
                                                              shuffle=False)
    feed_dict = {
        units_pl: units_feed,
        labels_pl: labels_feed,
    }
    return feed_dict, labels_feed, dates_feed


def do_eval(sess,
            eval_correct,
            units_placeholder,
            labels_placeholder,
            data_set,
            flags):
    """Runs one evaluation against the full epoch of data.
    Args:
      sess: The session in which the model has been trained.
      eval_correct: The Tensor that returns the number of correct predictions.
      units_placeholder: The units placeholder.
      labels_placeholder: The labels placeholder.
      data_set: The set of units and labels to evaluate, from
        data.read_data_sets().
      flags: Given options.
    """
    # And run one epoch of eval.
    total_squared_error = 0.0  # Counts the number of correct predictions.
    steps_per_epoch = data_set.num_examples // data_set.batch_size
    num_examples = steps_per_epoch * data_set.batch_size
    for step in range(steps_per_epoch):
        feed_dict, labels_feed, dates_feed = fill_feed_dict(data_set,
                                                            units_placeholder,
                                                            labels_placeholder,
                                                            flags)
        total_squared_error += sess.run(eval_correct, feed_dict=feed_dict)
    average_squared_error = total_squared_error / num_examples
    return average_squared_error


def default_label_managing_function(label):
    return label


def run_training(flags, class_number=1, label_managing_function=default_label_managing_function, is_profit=False):
    """Train mnist_example for a number of steps."""

    logger = logging.get_logger('evaluationLogger', flags.log_dir, 'evaluation', logging.INFO)

    # Get the sets of units and labels for training, validation, and
    # test on mnist_example.
    # data_sets = read_data(file_name=flags.file_name, company=flags.company, label_name=flags.label_name,
    #                       columns=flags.columns,class_number=class_number, label_profit=label_profit,
    #                       test_rate=flags.test_rate, validation_rate=flags.validation_rate, shuffle=False)
    data_sets = read_data(file_name=flags.file_name, company=flags.company, label_name=flags.label_name,
                          columns=flags.columns, class_number=class_number, label_profit=label_managing_function,
                          test_rate=flags.test_rate, shuffle=False)
    data_sets = to_recurrent_data(data_sets, flags.time_step)
    logger.info("train\ttest")
    logger.info("{:d}\t{:d}".format(len(data_sets.train.units),
                                    len(data_sets.test.units)))
    logger.info("")

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        # Generate placeholders for the units and labels.
        units_placeholder, labels_placeholder = placeholder_inputs(data_sets.batch_size, flags.time_step,
                                                                   data_sets.column_number)

        # Build a Graph that computes predictions from the inference model.
        logits = inference(units_placeholder,
                           flags.hidden_units,
                           data_sets.column_number,
                           data_sets.class_number,
                           data_sets.batch_size,
                           flags.dropout)

        # Add to the Graph the Ops for loss calculation.
        loss = do_loss(logits, labels_placeholder)

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = training(loss, flags.learning_rate)

        # Add the Op to compare the logits to the labels during evaluation.
        eval_correct = evaluation(logits, labels_placeholder)

        # Add the variable initializer Op.
        init = tf.global_variables_initializer()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        # Create a session for running Ops on the Graph.
        sess = tf.Session()

        # Run the Op to initialize the variables.
        sess.run(init)

        logger.info("\t".join(['learning_rate', 'max_steps', 'hidden_units']))
        logger.info("{:f}\t{:d}\t{}".format(flags.learning_rate, flags.max_steps, flags.hidden_units))
        logger.info("")
        logger.info(" ".join(['step', 'loss_value', 'training_precision', 'test_precision']))
        # Start the training loop.
        for step in range(flags.max_steps + 1):

            # Fill a feed dictionary with the actual set of units and labels
            # for this particular training step.
            feed_dict, labels_feed, dates_feed = fill_feed_dict(data_sets.train,
                                                                units_placeholder,
                                                                labels_placeholder,
                                                                flags)

            # Run one step of the model.  The return values are the activations
            # from the `train_op` (which is discarded) and the `loss` Op.  To
            # inspect the values of your Ops or variables, you may include them
            # in the list passed to sess.run() and the value tensors will be
            # returned in the tuple from the call.
            _, loss_value = sess.run([train_op, loss],
                                     feed_dict=feed_dict)

            # Save a checkpoint and evaluate the model periodically.
            if step % (flags.max_steps / 2) == 0:
                checkpoint_file = os.path.join(flags.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)
                # Evaluate against the training set.
                training_average_squared_error = do_eval(sess,
                                                         eval_correct,
                                                         units_placeholder,
                                                         labels_placeholder,
                                                         data_sets.train,
                                                         flags)
                # Evaluate against the test set.
                test_average_squared_error = do_eval(sess,
                                                     eval_correct,
                                                     units_placeholder,
                                                     labels_placeholder,
                                                     data_sets.test,
                                                     flags)
                logger.info("{:d}\t{:f}\t{:f}\t{:f}".format(step, loss_value, training_average_squared_error,
                                                            test_average_squared_error))

        # Compare labels(targets) with predictions.
        feed_dict, labels_feed, dates_feed = fill_feed_dict(data_sets.test,
                                                            units_placeholder,
                                                            labels_placeholder,
                                                            flags)
        test_predictions = sess.run(logits, feed_dict)
        average_squared_error = sess.run(eval_correct, feed_dict=feed_dict) / data_sets.batch_size
        y_label = 'Result'

        if is_profit:
            labels_feed = np.array(labels_feed) + 1
            labels_feed = labels_feed.cumprod()
            labels_feed = labels_feed - 1
            test_predictions = np.array(test_predictions) + 1
            test_predictions = test_predictions.cumprod()
            test_predictions = test_predictions - 1
            y_label = 'Profit'

        # draw a test graph
        matplotlib.rc('font', family='NanumBarunGothicOTF')
        fig, ax = plt.subplots()
        ax.plot(dates_feed, labels_feed, 'r', label='target')
        ax.plot(dates_feed, test_predictions, 'b', label='prediction, {:4.3f}'.format(average_squared_error))
        ax.legend()

        title = flags.file_name
        if flags.company is not None:
            title = title + ", " + flags.company
        plt.title(title)
        plt.xlabel('Time Period')
        plt.ylabel(y_label)
        plt.show()
