#!/usr/bin/env python
__author__ = "XXX XXX"
__email__ = "XXX"

import pytest
import numpy as np
import pandas as pd


@pytest.fixture()
def random_int_df():
    """ Create a random pandas dataframe of integers"""
    random_array = np.random.randint(low=0, high=2000000, size=(1000000, 2))
    random_df = pd.DataFrame(random_array)
    random_df.astype(int)
    return random_df


def test_remap_columns_consecutive(random_int_df):
    from utils.pandas_utils import remap_columns_consecutive

    remap_columns_consecutive(random_int_df, columns_names=[0, 1])

    col_1 = sorted(random_int_df[0].unique())
    col_2 = sorted(random_int_df[1].unique())

    consecutive_1 = np.arange(len(col_1))
    consecutive_2 = np.arange(len(col_2))

    comparison_1 = consecutive_1 == col_1
    assert comparison_1.all()

    comparison_2 = consecutive_2 == col_2
    assert comparison_2.all()


def test_remap_column_consecutive(random_int_df):
    from utils.pandas_utils import remap_column_consecutive

    remap_column_consecutive(random_int_df, 0)
    col_1 = sorted(random_int_df[0].unique())
    consecutive_1 = np.arange(len(col_1))
    comparison_1 = consecutive_1 == col_1
    assert comparison_1.all()
