#!/usr/bin/env python
__author__ = "XXX XXX"
__email__ = "XXX"

import pytest
import pandas as pd
import numpy as np


def test_remove_seen_items(dummy_train_data):
    from models.tensorflow.matrix_factorization_bpr import MatrixFactorizationBPR

    # generate random scores
    unique_users = dummy_train_data["userID"].unique()
    unique_items = dummy_train_data["itemID"].unique()
    scores = pd.DataFrame(np.random.random(size=(len(unique_users), len(unique_items))))
    dummy_rec = MatrixFactorizationBPR(dummy_train_data, embeddings_size=10)

    filtered_scores = dummy_rec.remove_seen_items(
        scores, {"interactions": dummy_train_data}
    ).to_numpy()
    seen_score = list(dummy_train_data.to_records(index=False))
    for s in seen_score:
        assert filtered_scores[s[0], s[1]] == -np.inf

    # check it works also when indeces of scores are not contiguos user
    user_data = pd.DataFrame(
        {"userID": [0, 0, 1, 1, 2, 2], "itemID": [0, 1, 0, 1, 0, 2]}
    )
    unique_items = user_data["itemID"].unique()
    index = [2, 0, 1]
    scores = pd.DataFrame(np.random.random(size=(3, len(unique_items))), index=index)
    filtered_scores = dummy_rec.remove_seen_items(scores, {"interactions": user_data})
    score_df = pd.DataFrame(filtered_scores, index=index)
    seen_score = list(user_data.to_records(index=False))
    for s in seen_score:
        assert score_df.loc[s[0], s[1]] == -np.inf


@pytest.mark.parametrize("cutoff", [1, 2, 3])
def test_recommend(dummy_train_data, cutoff):
    from models.tensorflow.matrix_factorization_bpr import MatrixFactorizationBPR

    # generate random scores
    unique_users = dummy_train_data["userID"].unique()
    unique_items = dummy_train_data["itemID"].unique()
    scores = np.random.random(size=(len(unique_users), len(unique_items)))
    dummy_rec = MatrixFactorizationBPR(dummy_train_data, embeddings_size=10)
    user_data = {"interactions": dummy_train_data}
    dummy_rec.scores = pd.DataFrame(scores, index=unique_users)
    recs_df = dummy_rec.recommend(cutoff=cutoff, user_data=user_data)
    assert recs_df.shape[0] == len(unique_users) * cutoff
