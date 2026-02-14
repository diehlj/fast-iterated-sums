import pytest
import torch

from src.models.fis_layer_random_trees import FISLayerRandomTrees
from src.models.sums import TreeType


def test_fis_layer_random_trees_forward_shape_and_param_count():
    in_channels = 3
    out_channels = 5
    num_nodes = 4

    layer = FISLayerRandomTrees(
        in_channels=in_channels,
        semiring="real",
        out_channels=out_channels,
        num_nodes=num_nodes,
        tree_type=TreeType.RANDOM,
        seed=7,
    )

    x = torch.randn(2, in_channels, 8, 8)
    y = layer(x)

    assert y.shape == (2, out_channels, 8, 8)
    assert len(layer.forest) == out_channels
    assert len(layer.tree_params) == out_channels * num_nodes


def test_fis_layer_random_trees_valuation_and_normalization_forward():
    layer = FISLayerRandomTrees(
        in_channels=2,
        semiring="real",
        out_channels=3,
        num_nodes=3,
        tree_type=TreeType.LINEAR_NE,
        seed=0,
        use_valuation=True,
        maxplus_normalization=True,
    )

    x = torch.randn(1, 2, 6, 6)
    y = layer(x)
    assert y.shape == (1, 3, 6, 6)