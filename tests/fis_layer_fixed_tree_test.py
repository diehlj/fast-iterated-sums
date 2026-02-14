import pytest
import torch

from src.models.fis_resnet_v1 import FISLayerParameterSharingV2
from src.models.fis_layer_fixed_tree import FISLayerFixedTree, replace_parameters_in_tree_generic


@pytest.mark.parametrize("num_nodes", [2, 3, 4])
def test_fis_layer_fixed_tree_forward_shape(num_nodes: int):
    layer = FISLayerFixedTree(
        in_channels=3,
        out_channels=7,
        semiring="real",
        num_nodes=num_nodes,
        tree_index=0,
        device="cpu",
    )
    x = torch.randn(2, 3, 8, 8)
    y = layer(x)
    assert y.shape == (2, 7, 8, 8)


@pytest.mark.parametrize("num_nodes", [0, 1, 5, -3])
def test_fis_layer_fixed_tree_invalid_num_nodes(num_nodes: int):
    with pytest.raises(ValueError, match="supports num_nodes"):
        FISLayerFixedTree(
            in_channels=3,
            out_channels=7,
            semiring="real",
            num_nodes=num_nodes,
            tree_index=0,
            device="cpu",
        )


def test_fis_layer_fixed_tree_matches_legacy_for_3_nodes():
    kwargs = dict(
        in_channels=3,
        out_channels=5,
        semiring="real",
        num_nodes=3,
        sums_internal_activation="relu",
        tree_index=1,
        maxplus_linear=False,
        use_valuation=False,
        maxplus_normalization=False,
        use_discounted_sums=False,
        discount_init=0.95,
        use_complex=False,
        device="cpu",
    )

    torch.manual_seed(123)
    kwargs_copy = kwargs.copy()
    kwargs_copy['num_trees'] = kwargs_copy.pop('out_channels')
    kwargs_copy['seed'] = kwargs_copy.pop('tree_index')
    kwargs_copy.pop('device')
    legacy_layer = FISLayerParameterSharingV2(**kwargs_copy)
    torch.manual_seed(123)
    new_layer = FISLayerFixedTree(**kwargs)

    x = torch.randn(2, 3, 10, 10)
    y_legacy = legacy_layer(x)
    y_new = new_layer(x)

    assert y_legacy.shape == y_new.shape == (2, 5, 10, 10)
    torch.testing.assert_close(y_new, y_legacy)

def test_replace_parameters_in_tree_generic():
    tree_4 = [1, [("left", [2, [("left", [3, []])]]), ("right", [4, []])]]
    params = [10, 20, 30, 40]
    expected_tree = [10, [("left", [20, [("left", [30, []])]]), ("right", [40, []])]]
    assert replace_parameters_in_tree_generic(tree_4, params) == expected_tree

    tree_4_linear = [1, [("left", [2, [("left", [3, [("left", [4, []])]])]])]]
    expected_tree_linear = [10, [("left", [20, [("left", [30, [("left", [40, []])]])]])]]
    assert replace_parameters_in_tree_generic(tree_4_linear, params) == expected_tree_linear