# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2017. 11. 22.
"""
import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    iris_dataset = load_iris()

    feature_names = iris_dataset['feature_names']
    data = iris_dataset['data']
    target_names = iris_dataset['target_names']
    target = iris_dataset['target']

    x_train, x_test, y_train, y_test = train_test_split(data, target, random_state=0)

    iris_dataframe = pd.DataFrame(x_train, columns=feature_names)
