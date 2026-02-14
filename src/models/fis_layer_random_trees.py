import torch
from torch import nn

from src.models.fis_layer_fixed_tree import ParamInit, get_activation_function
from src.models.sums import (
    TreeType,
    create_forest,
    sum_trees,
)

class FISLayerRandomTrees(nn.Module):
    def __init__(
        self,
        in_channels: int,
        semiring: str,
        out_channels: int,
        num_nodes: int,
        tree_type: str = TreeType.RANDOM,
        seed: int | None = None,
        param_init: str = ParamInit.NORMAL,
        maxplus_linear: bool = False,
        use_valuation: bool = False,
        maxplus_normalization: bool = False,

        use_discounted_sums: bool = False, # catch kwargs for generic instantiation
        use_complex: bool = False,
        complex_init_r_min: float = 0.0,
        complex_init_r_max: float = 1.0,
        complex_init_max_phase: float = 2 * 3.141592653589793,
        discount_init: float = 0.99,
        sums_internal_activation: str | None = None,
    ):
        super().__init__()
        assert not use_discounted_sums, "use_discounted_sums not supported for FISLayerRandomTrees"
        assert not use_complex, "use_complex not supported for FISLayerRandomTrees"
        assert not sums_internal_activation, "sums_internal_activation not supported for FISLayerRandomTrees"

        self.in_channels = in_channels
        self.semiring = semiring
        self.out_channels = out_channels
        self.maxplus_linear = maxplus_linear
        self.use_valuation = use_valuation
        self.maxplus_normalization = maxplus_normalization
        self.num_nodes = num_nodes
        self.tree_type = tree_type
        self.seed = seed
        self.param_init = param_init

        self.tree_params = nn.ParameterList()

        def alpha_generator():
            while True:
                if param_init == ParamInit.NORMAL:
                    param = nn.Parameter(torch.randn(self.in_channels))
                elif param_init == ParamInit.UNIFORM:
                    param = nn.Parameter(torch.rand(self.in_channels))
                elif param_init == ParamInit.ONES:
                    param = nn.Parameter(torch.ones(self.in_channels))

                else:
                    raise ValueError(
                        f"param_init must be one of {', '.join(ParamInit)}"
                    )

                self.tree_params.append(param)
                yield param

        self.forest = create_forest(
            alpha_generator=alpha_generator(),
            num_trees=self.out_channels,
            num_nodes=self.num_nodes,
            seed=self.seed,
            tree_type=self.tree_type,
        )
        if self.maxplus_normalization:
            assert self.use_valuation, (
                "maxplus_normalization requires use_valuation to be True"
            )
            self.maxplus_norm_param = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

    def valuate(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_valuation:
            tmp = torch.log1p(torch.relu(x))  # XXX log(1+ReLU(x)) vs log(eps+ReLU(x))
            if self.maxplus_normalization:
                tmp = tmp - self.maxplus_norm_param
        else:
            tmp = x
        return tmp

    def devaluate(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_valuation:
            return torch.expm1(x)
        else:
            return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.valuate(x)
        result = sum_trees(
            x, self.forest, semiring=self.semiring, maxplus_linear=self.maxplus_linear
        )
        result = self.devaluate(result)
        return result