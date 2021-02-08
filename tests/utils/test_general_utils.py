#!/usr/bin/env python
__author__ = "XXX XXX"
__email__ = "XXX"

import numpy as np
import pytest


@pytest.fixture(scope="module")
def scores():
    return np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [1, 5, 3, 4, 2]])


def test_get_top_k_scored_items(scores):
    from utils.general_utils import get_top_k_scored_items

    top_items, top_scores = get_top_k_scored_items(
        scores=scores, top_k=3, sort_top_k=True
    )

    assert np.array_equal(top_items, np.array([[4, 3, 2], [0, 1, 2], [1, 3, 2]]))
    assert np.array_equal(top_scores, np.array([[5, 4, 3], [5, 4, 3], [5, 4, 3]]))


def test_truncate_top_k():
    from utils.general_utils import truncate_top_k
    from utils.general_utils import truncate_top_k_2

    arr = np.array([[0, 1, 2], [3, 4, 5]])
    p_arr = truncate_top_k(arr, k=2)
    assert p_arr[0, 0] == p_arr[1, 0] == 0
    assert p_arr[0, 1] == 1
    assert p_arr[0, 2] == 2
    assert p_arr[1, 1] == 4
    assert p_arr[1, 2] == 5

    arr = np.array([[0, 1, 2], [3, 4, 5]])
    p_arr = truncate_top_k(arr, k=1)
    assert p_arr[0, 0] == p_arr[1, 0] == p_arr[0, 1] == p_arr[1, 1] == 0
    assert p_arr[0, 2] == 2
    assert p_arr[1, 2] == 5

    arr = np.array([[0, 1, 2], [3, 4, 5]])
    p_arr = truncate_top_k_2(arr, k=2)
    assert p_arr[0, 0] == p_arr[1, 0] == 0
    assert p_arr[0, 1] == 1
    assert p_arr[0, 2] == 2
    assert p_arr[1, 1] == 4
    assert p_arr[1, 2] == 5

    arr = np.array([[0, 1, 2], [3, 4, 5]])
    p_arr = truncate_top_k_2(arr, k=1)
    assert p_arr[0, 0] == p_arr[1, 0] == p_arr[0, 1] == p_arr[1, 1] == 0
    assert p_arr[0, 2] == 2
    assert p_arr[1, 2] == 5


@pytest.mark.parametrize(
    "m", [[[1, 2, 3], [0, 4, 5]], np.random.random(size=(1000, 2000))]
)
def test_normalize_csr_sparse_matrix(m):
    import scipy.sparse as sps
    from utils.general_utils import normalize_csr_sparse_matrix
    import numpy as np

    with pytest.raises(ValueError):
        normalize_csr_sparse_matrix(m)

    sp_m = sps.csr_matrix(m)
    norm_sp_m = normalize_csr_sparse_matrix(sp_m)
    norm_m = norm_sp_m.todense()
    assert pytest.approx(np.sum(norm_m, axis=1), 1e-6) == 1
