# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2017. 11. 22.
"""
from unittest import TestCase
from stock.data_reader import *


class TestReadData(TestCase):
    def test_read_data(self):
        print('{} is started...'.format(self._testMethodName))

        stock_data = read_data()
        print(stock_data.head())

        self.assertEqual(1295, len(stock_data))

        print('{} is done!!'.format(self._testMethodName))
