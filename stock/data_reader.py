# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2017. 11. 22.
"""
import pandas as pd


def read_data():
    """
    Read data from stock_data_set.csv and return it as DataFrame.

    :return stock_data: (DataFrame)
        index       company | (string)
                    year    | (string)
        columns     per     | (float)
                    pbr     | (float)
                    psr     | (float)
                    profit  | (float)
    """
    stock_data = pd.read_csv('stock_data_set.csv').dropna()
    stock_data = stock_data.set_index(['company', 'year'])
    return stock_data
