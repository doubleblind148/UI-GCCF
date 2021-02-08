#!/usr/bin/env python
__author__ = "XXX XXX"
__email__ = "XXX"

import pytest

from constants import *


@pytest.fixture()
def dummy_graph(dummy_dataset_pandas):
    from utils.graph_utils import nxgraph_from_user_item_interaction_df

    graph = nxgraph_from_user_item_interaction_df(
        dummy_dataset_pandas, DEFAULT_USER_COL, DEFAULT_ITEM_COL
    )
    return graph


def test_nxgraph_from_user_item_interaction_df(dummy_dataset_pandas):
    from utils.graph_utils import nxgraph_from_user_item_interaction_df

    user = len(dummy_dataset_pandas[DEFAULT_USER_COL].unique())
    item = len(dummy_dataset_pandas[DEFAULT_ITEM_COL].unique())
    user_item = user + item

    graph = nxgraph_from_user_item_interaction_df(
        dummy_dataset_pandas, DEFAULT_USER_COL, DEFAULT_ITEM_COL
    )

    # check kind attribute properly set on all nodes
    assert len(graph.nodes()) == user_item
    for _, data in graph.nodes(data=True):
        assert "kind" in data


def test_symmetric_normalized_laplacian_matrix(dummy_graph):
    from utils.graph_utils import symmetric_normalized_laplacian_matrix
    import networkx as nx
    import math

    # check number of entries of laplacian
    laplacian = symmetric_normalized_laplacian_matrix(dummy_graph, self_loop=False)
    assert len(laplacian.data) == len(dummy_graph.edges()) * 2

    laplacian = symmetric_normalized_laplacian_matrix(dummy_graph, self_loop=True)
    assert len(laplacian.data) == len(dummy_graph.edges()) * 2 + len(
        dummy_graph.nodes()
    )

    graph_2 = nx.from_edgelist([(0, 1), (0, 2)])
    laplacian = symmetric_normalized_laplacian_matrix(
        graph_2, self_loop=False
    ).todense()
    assert laplacian[0, 0] == 0
    assert laplacian[0, 1] == pytest.approx(1 / (math.sqrt(2)), 1e-8)
    assert laplacian[0, 2] == pytest.approx(1 / (math.sqrt(2)), 1e-8)
    assert laplacian[1, 0] == pytest.approx(1 / (math.sqrt(2)), 1e-8)


def test_urm_from_nxgraph(dummy_graph):
    from utils.graph_utils import urm_from_nxgraph

    users = sorted([x for x, y in dummy_graph.nodes(data=True) if y["kind"] == "user"])
    items = sorted([x for x, y in dummy_graph.nodes(data=True) if y["kind"] == "item"])
    urm = urm_from_nxgraph(dummy_graph)
    assert urm.shape[0] == len(users)
    assert urm.shape[1] == len(items)
    for e in sorted(dummy_graph.edges()):
        # check onlu user-item edges
        if e[0] < e[1]:
            assert urm[e[0], e[1]-max(users)-1] == 1
