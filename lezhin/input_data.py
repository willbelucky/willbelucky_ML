# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2017. 11. 11.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed

from lezhin import lezhin_comics

SOURCE_URL = 'https://storage.googleapis.com/lz-insight/pycon17/dataset/'
TRAIN_FILE_NAME = 'lezhin_dataset_v2_training.tsv.gz'
TEST_FILE_NAME = 'lezhin_dataset_v2_test.tsv.gz'


class DataSet(object):
    def __init__(self,
                 units,
                 labels,
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
        self._units = units
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def units(self):
        return self._units

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_unit = [1] * lezhin_comics.UNIT_NUMBER
            if self.one_hot:
                fake_label = 1
            else:
                fake_label = 0
            return [fake_unit for _ in range(batch_size)], [
                fake_label for _ in range(batch_size)
            ]
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._units = self.units.iloc[perm0]
            self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            units_rest_part = self._units[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._units = self.units.iloc[perm]
                self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            units_new_part = self._units[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((units_rest_part, units_new_part), axis=0), np.concatenate(
                (labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._units[start:end], self._labels[start:end]


CATEGORICAL_UNITS = [6, 7, 9]
PRODUCT_UNIT = 7
PURCHASE_HISTORY_UNITS = [x for x in range(11, 111)]


def read_data_sets(train_dir,
                   fake_data=False,
                   validation_rate=0.1,
                   one_hot=False,
                   dtype=dtypes.float32,
                   seed=None):
    if fake_data:
        def fake():
            return DataSet(
                [], [], fake_data=True, one_hot=one_hot, dtype=dtype, seed=seed)

        train = fake()
        validation = fake()
        test = fake()
        return base.Datasets(train=train, validation=validation, test=test)

    train_units = pd.read_table(base.maybe_download(TRAIN_FILE_NAME, train_dir, SOURCE_URL + TRAIN_FILE_NAME),
                                header=None)
    train_units = train_units.sample(frac=1)
    train_units[CATEGORICAL_UNITS] = train_units[CATEGORICAL_UNITS].fillna(value='00000000')
    train_units = train_units.fillna(value=0.0)
    train_labels = train_units.pop(0).values

    test_units = pd.read_table(base.maybe_download(TEST_FILE_NAME, train_dir, SOURCE_URL + TEST_FILE_NAME), header=None)
    test_units = test_units.sample(frac=1)
    test_units[CATEGORICAL_UNITS] = test_units[CATEGORICAL_UNITS].fillna(value='00000000')
    test_units = test_units.fillna(value=0.0)
    test_labels = test_units.pop(0).values

    # content_labels = set(list(test_units[7].append(train_units[7])))
    # for content_label in content_labels:
    #     for unit in PURCHASE_HISTORY_UNITS:
    #         train_units[content_label + str(unit)] = train_units[unit]*(train_units[PRODUCT_UNIT] == content_label)
    #         test_units[content_label + str(unit)] = test_units[unit]*(test_units[PRODUCT_UNIT] == content_label)
    #         train_units = train_units.drop(PURCHASE_HISTORY_UNITS, axis=1)
    #         test_units = test_units.drop(PURCHASE_HISTORY_UNITS, axis=1)

    for var in CATEGORICAL_UNITS:
        le = LabelEncoder().fit(train_units[var].append(test_units[var]))
        train_units[var] = le.transform(train_units[var])
        test_units[var] = le.transform(test_units[var])

    validation_size = int(len(train_units) * validation_rate)

    validation_units = train_units[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_units = train_units[validation_size:]
    train_labels = train_labels[validation_size:]

    options = dict(dtype=dtype, seed=seed)

    train = DataSet(train_units, train_labels, **options)
    validation = DataSet(validation_units, validation_labels, **options)
    test = DataSet(test_units, test_labels, **options)

    return base.Datasets(train=train, validation=validation, test=test)
