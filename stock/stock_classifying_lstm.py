# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2017. 11. 27.
"""
import argparse
import os
import sys

import tensorflow as tf

from LSTM.classifying_lstm import run_training

FLAGS = None

# number of category.
class_number = 2


def label_profit(profit):
    if profit < 0.0:
        label = 0
    else:
        label = 1
    return label


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    run_training(flags=FLAGS, class_number=class_number, label_profit=label_profit)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--file_name',
        type=str,
        default='FE539',
        help='Name of input csv file without `.csv`.'
    )
    parser.add_argument(
        '--company',
        type=str,
        default=None,
        help='Name of company.'
    )
    parser.add_argument(
        '--label_name',
        type=str,
        default='PROFIT',
        help='Label name.'
    )
    parser.add_argument(
        '--columns',
        nargs='+',
        type=str,
        default=['INTEREST', 'GAS', 'ROE', 'EPS', 'BPS', 'PER', 'PBR', 'PSR', 'DR', 'ASSET', 'SALES',
                 'OPERATION_PROFIT', 'CASH_FLOW'],
        help='Names of input columns.'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.000001,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.1,
        help='The dropout rate, between 0 and 1. E.g. "rate=0.1" would drop out 10% of input units.'
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=2000,
        help='Number of steps to run trainer.'
    )
    parser.add_argument(
        '--hidden_units',
        nargs='+',
        type=int,
        default=[13, 15],
        help='Number of units in hidden layers.'
    )
    parser.add_argument(
        '--time_step',
        type=int,
        default=7,
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
    # len(>0.0) = 2475, len(<=0.0) = 2385, total = 4860
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
