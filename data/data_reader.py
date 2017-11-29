# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2017. 11. 27.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed

DATASETS = collections.namedtuple('Datasets', ['train', 'test', 'column_number', 'class_number', 'batch_size'])


class DataSet(object):
    def __init__(self,
                 units,
                 labels,
                 dates,
                 column_number,
                 class_number,
                 batch_size=100,
                 fake_data=False,
                 one_hot=False,
                 dtype=dtypes.float32,
                 seed=None):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.  Seed arg provides for convenient deterministic testing.
        """
        seed1, seed2 = random_seed.get_seed(seed)
        # If op level seed is not set, use whatever graph level seed is returned
        np.random.seed(seed1 if seed is None else seed2)
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid unit dtype %r, expected uint8 or float32' %
                            dtype)
        if fake_data:
            self._num_examples = 10000
            self.one_hot = one_hot
        else:
            assert units.shape[0] == labels.shape[0], (
                'units.shape: %s labels.shape: %s' % (units.shape, labels.shape))
            self._num_examples = units.shape[0]
        self._seed = seed
        self._dtype = dtype
        self._units = units
        self._labels = labels
        self._dates = dates
        self._batch_size = batch_size
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self.column_number = column_number
        self.class_number = class_number

    @property
    def units(self):
        return self._units

    @property
    def labels(self):
        return self._labels

    @property
    def dates(self):
        return self._dates

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def epochs_completed(self):
        return self._epochs_completed

    @property
    def dtype(self):
        return self._dtype

    @property
    def seed(self):
        return self._seed

    def next_batch(self, fake_data=False, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_unit = [1] * self.column_number
            if self.one_hot:
                fake_label = 1
            else:
                fake_label = 0
            fake_date = [i for i in range(self.column_number)]
            return [fake_unit for _ in range(self._batch_size)], [fake_label for _ in range(self._batch_size)], [
                fake_date for _ in range(self._batch_size)]
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._units = self.units.iloc[perm0]
            self._labels = self.labels[perm0]
            self._dates = self.dates[perm0]
        # Go to the next epoch
        if start + self._batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            units_rest_part = self._units[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            dates_rest_part = self._dates[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._units = self.units.iloc[perm]
                self._labels = self.labels[perm]
                self._dates = self.dates[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = self._batch_size - rest_num_examples
            end = self._index_in_epoch
            units_new_part = self._units[start:end]
            labels_new_part = self._labels[start:end]
            dates_new_part = self._dates[start:end]
            return np.concatenate((units_rest_part, units_new_part), axis=0), np.concatenate(
                (labels_rest_part, labels_new_part), axis=0), np.concatenate((dates_rest_part, dates_new_part), axis=0)
        else:
            self._index_in_epoch += self._batch_size
            end = self._index_in_epoch
            return self._units[start:end], self._labels[start:end], self._dates[start:end]


def read_csv(file_name, company=None):
    """
    Read data from file_name.csv and return it as DataFrame without null value.

    :return stock_data: (DataFrame)
    """
    try:
        stock_data = pd.read_csv(file_name + '.csv', parse_dates=['date']).dropna()
        stock_data = stock_data.sort_values(by=['date'])
    except ValueError:
        stock_data = pd.read_csv(file_name + '.csv').dropna()

    if company is not None:
        stock_data = stock_data.loc[stock_data['company'] == company]

    return stock_data


def to_recurrent_data(data_sets, time_step):
    options = dict(dtype=data_sets.train.dtype, seed=data_sets.train.seed, column_number=data_sets.column_number,
                   class_number=data_sets.class_number, batch_size=(data_sets.batch_size-time_step))

    train = DataSet(dataframe_to_recurrent_ndarray(data_sets.train.units, time_step),
                    data_sets.train.labels[time_step:], data_sets.train.dates[time_step:], **options)
    test = DataSet(dataframe_to_recurrent_ndarray(data_sets.test.units, time_step),
                   data_sets.test.labels[time_step:], data_sets.test.dates[time_step:], **options)

    return DATASETS(train=train, test=test, column_number=data_sets.column_number,
                    class_number=data_sets.class_number, batch_size=(data_sets.batch_size-time_step))


def dataframe_to_recurrent_ndarray(x, time_step):
    recurrent_panel = []
    for i in range(0, len(x) - time_step):
        recurrent_frame = []
        for index, values in x[i:i + time_step].iterrows():
            recurrent_frame.append(np.asarray(values))

        recurrent_panel.append(np.asarray(recurrent_frame))

    return np.asarray(recurrent_panel)


def read_data(file_name,
              company,
              label_name,
              columns,
              class_number,
              label_profit,
              fake_data=False,
              shuffle=True,
              test_rate=0.25,
              one_hot=False,
              dtype=dtypes.float32,
              seed=None):
    if fake_data:
        def fake():
            return DataSet(units=[], labels=[], dates=[], column_number=0, class_number=0,
                           fake_data=True, one_hot=one_hot, dtype=dtype, seed=seed)

        train = fake()
        test = fake()
        return DATASETS(train=train, test=test, column_number=0, class_number=0)

    stock_data = read_csv(file_name, company=company)

    assert len(stock_data) > 0

    if shuffle:
        stock_data = stock_data.sample(frac=1)

    units = stock_data[columns]
    units = pd.DataFrame(MinMaxScaler().fit_transform(units))
    labels = stock_data[label_name].apply(label_profit).values
    try:
        dates = stock_data['date']
    except KeyError:
        dates = range(len(units))

    test_size = int(len(stock_data) * test_rate)
    train_size = len(stock_data) - test_size

    assert test_size > 0

    train_units = units[:train_size]
    train_labels = labels[:train_size]
    train_dates = dates[:train_size]
    test_units = units[train_size:]
    test_labels = labels[train_size:]
    test_dates = dates[train_size:]

    options = dict(dtype=dtype, seed=seed, column_number=len(columns), class_number=class_number, batch_size=test_size)

    train = DataSet(train_units, train_labels, train_dates, **options)
    test = DataSet(test_units, test_labels, test_dates, **options)

    return DATASETS(train=train, test=test, column_number=len(columns),
                    class_number=class_number, batch_size=test_size)
