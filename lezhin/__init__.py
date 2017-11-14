# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2017. 11. 11.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=missing-docstring
import argparse
import os
import sys
import time

import tensorflow as tf

from lezhin import lezhin_comics, input_data
from util import logging

# Basic model parameters as external flags.
FLAGS = None


def placeholder_inputs(batch_size):
    """Generate placeholder variables to represent the input tensors.
    These placeholders are used as inputs by the rest of the model building
    code and will be fed from the downloaded data in the .run() loop, below.
    Args:
      batch_size: The batch size will be baked into both placeholders.
    Returns:
      units_placeholder: Units placeholder.
      labels_placeholder: Labels placeholder.
    """
    # Note that the shapes of the placeholders match the shapes of the full
    # unit and label tensors, except the first dimension is now batch_size
    # rather than the full size of the train or test data sets.
    units_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                          lezhin_comics.UNIT_NUMBER))
    labels_placeholder = tf.placeholder(tf.int32, shape=batch_size)
    return units_placeholder, labels_placeholder


def fill_feed_dict(data_set, units_pl, labels_pl):
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
    Returns:
      feed_dict: The feed dictionary mapping from placeholders to values.
    """
    # Create the feed_dict for the placeholders filled with the next
    # `batch size` examples.
    units_feed, labels_feed = data_set.next_batch(FLAGS.batch_size,
                                                  FLAGS.fake_data)
    feed_dict = {
        units_pl: units_feed,
        labels_pl: labels_feed,
    }
    return feed_dict


def do_eval(sess,
            eval_correct,
            units_placeholder,
            labels_placeholder,
            data_set):
    """Runs one evaluation against the full epoch of data.
    Args:
      sess: The session in which the model has been trained.
      eval_correct: The Tensor that returns the number of correct predictions.
      units_placeholder: The units placeholder.
      labels_placeholder: The labels placeholder.
      data_set: The set of units and labels to evaluate, from
        data.read_data_sets().
    """
    # And run one epoch of eval.
    true_count = 0  # Counts the number of correct predictions.
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size
    num_examples = steps_per_epoch * FLAGS.batch_size
    for step in range(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set,
                                   units_placeholder,
                                   labels_placeholder)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = float(true_count) / num_examples
    # print('Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
    #       (num_examples, true_count, precision))
    return num_examples, true_count, precision


def run_training():
    """Train mnist for a number of steps."""

    logger = logging.get_logger('evaluationLogger', FLAGS.log_dir, 'evaluation', logging.INFO)

    # Get the sets of units and labels for training, validation, and
    # test on mnist.
    data_sets = input_data.read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data)

    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        # Generate placeholders for the units and labels.
        units_placeholder, labels_placeholder = placeholder_inputs(
            FLAGS.batch_size)

        # Build a Graph that computes predictions from the inference model.
        logits = lezhin_comics.inference(units_placeholder,
                                         FLAGS.hidden_units,
                                         FLAGS.dropout)

        # Add to the Graph the Ops for loss calculation.
        loss = lezhin_comics.loss(logits, labels_placeholder)

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = lezhin_comics.training(loss, FLAGS.learning_rate)

        # Add the Op to compare the logits to the labels during evaluation.
        eval_correct = lezhin_comics.evaluation(logits, labels_placeholder)

        # Build the summary Tensor based on the TF collection of Summaries.
        # summary = tf.summary.merge_all()

        # Add the variable initializer Op.
        init = tf.global_variables_initializer()

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        # Create a session for running Ops on the Graph.
        sess = tf.Session()

        # Instantiate a SummaryWriter to output summaries and the Graph.
        # summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

        # And then after everything is built:

        # Run the Op to initialize the variables.
        sess.run(init)

        logger.info(
            " ".join(['step', 'loss_value', 'training_num_examples', 'training_true_count', 'training_precision',
                      'validation_num_examples', 'validation_true_count', 'validation_precision', 'test_num_examples',
                      'test_true_count', 'test_precision']))
        # Start the training loop.
        for step in range(FLAGS.max_steps):
            # start_time = time.time()

            # Fill a feed dictionary with the actual set of units and labels
            # for this particular training step.
            feed_dict = fill_feed_dict(data_sets.train,
                                       units_placeholder,
                                       labels_placeholder)

            # Run one step of the model.  The return values are the activations
            # from the `train_op` (which is discarded) and the `loss` Op.  To
            # inspect the values of your Ops or variables, you may include them
            # in the list passed to sess.run() and the value tensors will be
            # returned in the tuple from the call.
            _, loss_value = sess.run([train_op, loss],
                                     feed_dict=feed_dict)

            # duration = time.time() - start_time

            # Save a checkpoint and evaluate the model periodically.
            if step % 10000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)
                # Evaluate against the training set.
                # print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                # print('Training Data Eval:')
                training_num_examples, training_true_count, training_precision = do_eval(sess,
                                                                                         eval_correct,
                                                                                         units_placeholder,
                                                                                         labels_placeholder,
                                                                                         data_sets.train)
                # Evaluate against the validation set.
                # print('Validation Data Eval:')
                validation_num_examples, validation_true_count, validation_precision = do_eval(sess,
                                                                                               eval_correct,
                                                                                               units_placeholder,
                                                                                               labels_placeholder,
                                                                                               data_sets.validation)
                # Evaluate against the test set.
                # print('Test Data Eval:')
                test_num_examples, test_true_count, test_precision = do_eval(sess,
                                                                             eval_correct,
                                                                             units_placeholder,
                                                                             labels_placeholder,
                                                                             data_sets.test)
                logger.info("{:d} {:f} {:d} {:d} {:f} {:d} {:d} {:f} {:d} {:d} {:f}".format(step, loss_value,
                                                                                            training_num_examples,
                                                                                            training_true_count,
                                                                                            training_precision,
                                                                                            validation_num_examples,
                                                                                            validation_true_count,
                                                                                            validation_precision,
                                                                                            test_num_examples,
                                                                                            test_true_count,
                                                                                            test_precision))


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    run_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.0001,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=None,
        help='The dropout rate, between 0 and 1. E.g. "rate=0.1" would drop out 10% of input units.'
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=100000,
        help='Number of steps to run trainer.'
    )
    parser.add_argument(
        '--hidden_units',
        nargs='+',
        type=int,
        default=[166, 332, 332, 166],
        help='Number of units in hidden layers.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='Batch size.  Must divide evenly into the dataset sizes.'
    )
    parser.add_argument(
        '--input_data_dir',
        type=str,
        default=os.path.join(os.getenv('TEST_TMPDIR', os.getcwd()),
                             'data'),
        help='Directory to put the input data.'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default=os.path.join(os.getenv('TEST_TMPDIR', os.getcwd()),
                             'logs/fully_connected_feed'),
        help='Directory to put the log data.'
    )
    parser.add_argument(
        '--fake_data',
        default=False,
        help='If true, uses fake data for unit testing.',
        action='store_true'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
