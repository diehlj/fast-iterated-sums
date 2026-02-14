import torch
import torch.nn as nn

from src.models.discounted_sums import sum_tree as discounted_sum_tree
from src.models.sums import all_trees, n_vertices, pretty_str_tree_one_line, sum_tree
from enum import IntEnum, StrEnum, auto
from typing import Callable

class ParamInit(StrEnum):
    NORMAL = auto()
    UNIFORM = auto()
    ONES = auto()

def get_activation_function(activation_name: str) -> Callable:
    if activation_name == 'relu':
        return torch.relu
    elif activation_name == 'zero': # For debugging.
        return lambda x: 0. * x
    elif activation_name == 'gelu':
        return torch.nn.functional.gelu
    elif activation_name == 'swish' or activation_name == 'silu':
        return torch.nn.functional.silu
    elif activation_name == 'leaky_relu':
        return torch.nn.functional.leaky_relu
    elif activation_name == 'elu':
        return torch.nn.functional.elu
    elif activation_name == 'mish':
        return torch.nn.functional.mish
    else:
        raise ValueError(f"Invalid activation function: {activation_name}")

def clone_tree(tree):
    """Deep-clone the tree structure while preserving node values."""
    value, children = tree
    return [value, [(direction, clone_tree(subtree)) for direction, subtree in children]]


def _assign_parameters_preorder(tree, params, index):
    tree[0] = params[index]
    index += 1
    for _, subtree in tree[1]:
        index = _assign_parameters_preorder(subtree, params, index)
    return index


def replace_parameters_in_tree_generic(tree, params):
    """Assign params to tree nodes in preorder traversal."""
    expected = n_vertices(tree)
    if len(params) != expected:
        raise ValueError(
            f"Expected {expected} parameters for tree with {expected} nodes, got {len(params)}"
        )

    tree_copy = clone_tree(tree)
    consumed = _assign_parameters_preorder(tree_copy, params, 0)
    if consumed != expected:
        raise RuntimeError(
            f"Internal error while assigning tree parameters: consumed={consumed}, expected={expected}"
        )
    return tree_copy


class FISLayerFixedTree(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        semiring: str,
        num_nodes: int,
        sums_internal_activation: str | None = None,
        tree_index: int | None = None,
        maxplus_linear: bool = False,
        use_valuation: bool = False,
        maxplus_normalization: bool = False,
        use_discounted_sums: bool = False,
        discount_init: float = 0.99,
        use_complex: bool = False,
        complex_init_r_min: float = 0.0,
        complex_init_r_max: float = 1.0,
        complex_init_max_phase: float = 2 * 3.141592653589793,
        device: str = "cuda",
    ):
        super().__init__()
        if num_nodes not in {2, 3, 4}:
            raise ValueError(
                f"FISLayerFixedTree supports num_nodes in {{2, 3, 4}}, got {num_nodes}"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sums_internal_activation = sums_internal_activation
        self.semiring = semiring

        if use_complex:
            if use_valuation:
                print(
                    "[WARNING] use_valuation is True but use_complex is True. Setting use_complex to False."
                )
                use_complex = False
            if maxplus_normalization:
                print(
                    "[WARNING] maxplus_normalization is True but use_complex is True. Setting use_complex to False."
                )
                use_complex = False
            if semiring != "real":
                print(
                    f"[WARNING] semiring is not {semiring} but use_complex is True. Setting use_complex to False."
                )
                use_complex = False
        if use_complex:
            assert use_discounted_sums, "use_discounted_sums must be True with use_complex"
            self.go_to_complex = nn.Linear(1, 2)
            self.go_to_real = nn.Linear(2, 1)

        self.use_complex = use_complex
        self.num_nodes = num_nodes
        self.maxplus_linear = maxplus_linear
        self.use_valuation = use_valuation
        self.maxplus_normalization = maxplus_normalization
        self.use_discounted_sums = use_discounted_sums
        self.discount_init = discount_init

        if self.maxplus_normalization:
            if not self.use_valuation:
                print(
                    "Warning: maxplus_normalization is True but use_valuation is False. Setting use_valuation to True."
                )
                self.use_valuation = True
            self.maxplus_norm_param = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

        self.alphas = nn.ParameterList(
            [
                nn.Parameter(torch.randn(self.out_channels, self.in_channels))
                for _ in range(self.num_nodes)
            ]
        )
        # TODO Legacy compatibility; remove this later.
        for i, alpha in enumerate(self.alphas, start=1):
            setattr(self, f"alpha_{i}", alpha)

        if self.sums_internal_activation is not None:
            activation_fn = get_activation_function(self.sums_internal_activation)
            for alpha in self.alphas[1:]:
                alpha.activation_function_hack = activation_fn

        if self.use_discounted_sums:
            if self.use_complex:
                u1 = torch.rand(1, self.out_channels, 1, 1, device=device)
                self.discount_nu_log = nn.Parameter(
                    torch.log(
                        -0.5
                        * torch.log(
                            u1 * (complex_init_r_max**2 - complex_init_r_min**2)
                            + complex_init_r_min**2
                        )
                    )
                )
                u2 = torch.rand(1, self.out_channels, 1, 1, device=device)
                self.discount_theta_log = nn.Parameter(torch.log(complex_init_max_phase * u2))
            else:
                self.discount = nn.Parameter(
                    discount_init * torch.ones(1, self.out_channels, 1, 1)
                )

        ats = all_trees(self.num_nodes)
        n_trees = len(ats)
        self.tree_index = (
            tree_index if tree_index is not None else int(torch.randint(0, n_trees, (1,))[0])
        )
        self.tree = ats[self.tree_index % n_trees]
        self.tree = replace_parameters_in_tree_generic(self.tree, list(self.alphas))

        def _sum(x: torch.Tensor, discount: torch.Tensor | None = None) -> torch.Tensor:
            if self.use_discounted_sums:
                if self.use_complex:
                    batch_size, _, height, width, _ = x.shape
                    discount_expanded = discount.expand(
                        batch_size, self.out_channels, height, width, 2
                    )
                else:
                    batch_size, _, height, width = x.shape
                    discount_expanded = discount.expand(
                        batch_size, self.out_channels, height, width
                    )
                return discounted_sum_tree(
                    x,
                    self.tree,
                    discount_expanded,
                    semiring=self.semiring,
                    maxplus_linear=self.maxplus_linear,
                    channel_wise=False,
                )
            return sum_tree(
                x,
                self.tree,
                semiring=self.semiring,
                maxplus_linear=self.maxplus_linear,
                channel_wise=False,
            )

        self.sum = _sum

    def valuate(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_valuation:
            tmp = torch.log1p(torch.relu(x))
            if self.maxplus_normalization:
                tmp = tmp - self.maxplus_norm_param
        else:
            tmp = x
        return tmp

    def devaluate(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_valuation:
            return torch.expm1(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.valuate(x)
        if self.use_complex and self.use_discounted_sums:
            lambda_r = torch.exp(-torch.exp(self.discount_nu_log)) * torch.cos(
                torch.exp(self.discount_theta_log)
            )
            lambda_i = torch.exp(-torch.exp(self.discount_nu_log)) * torch.sin(
                torch.exp(self.discount_theta_log)
            )
            discount = torch.stack([lambda_r, lambda_i], dim=-1)
            x = x.unsqueeze(-1)
            x = self.go_to_complex(x)
            result = self.sum(x, discount)
            result = self.go_to_real(result).squeeze(-1)
        elif self.use_discounted_sums:
            result = self.sum(x, self.discount)
        else:
            result = self.sum(x)
        result = self.devaluate(result)
        return result

    def __str__(self):
        return (
            "FISLayerFixedTree("
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"num_nodes={self.num_nodes}, "
            f"semiring={self.semiring}, "
            f"use_discounted_sums={self.use_discounted_sums}, "
            f"tree={pretty_str_tree_one_line(self.tree)})"
        )

    def __repr__(self):
        return self.__str__()

def try_fis_layer_fixed_tree():
    torch.manual_seed(0)
    layer = FISLayerFixedTree(
        in_channels=3,
        out_channels=11,
        semiring='maxplus',
        tree_index=11,
        num_nodes=3,
    )
    x = torch.randn(1, 3, 32, 32)
    result = layer(x)
    assert result.shape == (1, 11, 32, 32)
    from src.models.sums import pretty_print_tree
    pretty_print_tree(layer.tree)
    return layer

if __name__ == "__main__":
    try_fis_layer_fixed_tree()