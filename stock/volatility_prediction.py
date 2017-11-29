# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2017. 11. 29.
"""
import argparse
import os
import sys

import tensorflow as tf

from LSTM.regression_lstm import run_training

FLAGS = None


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    run_training(flags=FLAGS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--file_name',
        type=str,
        default='volatility_prediction',
        help='Name of input csv file without `.csv`.'
    )
    parser.add_argument(
        '--company',
        type=str,
        default='현대차',
        help='Name of company.'
    )
    parser.add_argument(
        '--label_name',
        type=str,
        default='volatility_D+1',
        help='Label name.'
    )
    parser.add_argument(
        '--columns',
        nargs='+',
        type=str,
        default=['pre_profit', 'volume', 'short_sell', 'institution_net_buy', 'private_net_buy', 'foreign_net_buy',
                 'volatility', 'PER', 'PBR', 'PSR'],
        help='Names of input columns.'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.001,
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
        default=20000,
        help='Number of steps to run trainer.'
    )
    parser.add_argument(
        '--hidden_units',
        nargs='+',
        type=int,
        default=[16],
        help='Number of units in hidden layers.'
    )
    parser.add_argument(
        '--time_step',
        type=int,
        default=1,
        help='The number of time steps.'
    )
    parser.add_argument(
        '--test_rate',
        type=float,
        default=0.25,
        help='Test rate. The portion of test set.'
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
