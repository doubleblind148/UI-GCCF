#!/usr/bin/env python
__author__ = "XXX XXX"
__email__ = "XXX"

import numpy as np



def test_tf_dropout_sparse():
    from utils.tensorflow_utils import tf_dropout_sparse
    import scipy.sparse as sps
    from utils.tensorflow_utils import to_tf_sparse_tensor

    rnd_sparse_matrix = sps.random(100, 100, density=0.1)
    tf_m = to_tf_sparse_tensor(rnd_sparse_matrix)
    drop_tf_m = tf_dropout_sparse(tf_m, keep_prob=0.5, n_nonzero_elems=len(rnd_sparse_matrix.data))
    #todo: assert
    assert True
