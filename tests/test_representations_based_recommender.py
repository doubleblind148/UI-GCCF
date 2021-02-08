#!/usr/bin/env python
__author__ = "XXX XXX"
__email__ = "XXX"

import numpy as np
import pandas as pd


def test_compute_items_scores(dummy_train_data):
    from models.tensorflow.matrix_factorization_bpr import MatrixFactorizationBPR

    # generate random scores
    unique_users = dummy_train_data["userID"].unique()
    unique_items = dummy_train_data["itemID"].unique()

    factors = 10

    dummy_rec = MatrixFactorizationBPR(dummy_train_data, embeddings_size=10)

    dummy_rec.users_repr_df = pd.DataFrame(
        np.random.random(size=(len(unique_users), factors)), index=unique_users
    )
    dummy_rec.items_repr_df = pd.DataFrame(
        np.random.random(size=(len(unique_items), factors)), index=unique_items
    )

    scores = dummy_rec.compute_items_scores({"interactions":dummy_train_data})

    assert scores is not None
    assert scores.shape[0] == len(unique_users)
    assert scores.shape[1] == len(unique_items)
