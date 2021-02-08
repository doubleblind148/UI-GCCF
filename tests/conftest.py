#!/usr/bin/env python
__author__ = "XXX XXX"
__email__ = "XXX"

import pytest
import pandas as pd
from constants import *
import numpy as np




@pytest.fixture()
def dummy_train_data():
    return pd.DataFrame(
        {"userID": [0, 0, 1, 1, 2, 2, 3, 4, 5], "itemID": [0, 1, 0, 1, 0, 3, 4, 4, 2]}
    )


@pytest.fixture()
def dataset_stat():
    return {
        "number_of_rows": 1000,
        "number_of_items": 50,
        "number_of_users": 20,
    }


@pytest.fixture()
def dummy_dataset_pandas(dataset_stat):
    from utils.pandas_utils import remap_columns_consecutive
    python_dataset = pd.DataFrame(
        {
            DEFAULT_USER_COL: np.random.randint(
                0, dataset_stat["number_of_users"], dataset_stat["number_of_rows"]
            ),
            DEFAULT_ITEM_COL: np.random.randint(
                0, dataset_stat["number_of_items"], dataset_stat["number_of_rows"]
            ),
            DEFAULT_RATING_COL: np.random.randint(1, 6, dataset_stat["number_of_rows"]),
        }
    )
    remap_columns_consecutive(python_dataset, columns_names=[DEFAULT_USER_COL, DEFAULT_ITEM_COL])
    return python_dataset
