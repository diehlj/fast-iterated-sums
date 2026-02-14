##
# XXX Legacy file, to not have the experiment configs break.
# Use the FIS layer in fis_layer.py instead.
##
from enum import IntEnum, StrEnum, auto

import torch
from torch import nn
from torchvision import models
from src.models.sums import Semiring
from src.models.fis_layer_fixed_tree import ParamInit, get_activation_function

from src.models.discounted_sums import ds_cumsum_XX
from src.models.sums import (
    cumsum_XX,
    TreeDirection,
    TreeType,
    all_trees,
    create_forest,
    n_vertices,
    pretty_print_tree,
    pretty_str_tree_one_line,
    sum_tree,
    sum_trees,
)
from src.models.discounted_sums import sum_tree as discounted_sum_tree

def replace_parameters_in_tree(tree, params):
    # Manual HACK.
    n = n_vertices(tree)
    assert len(params) == n
    tree[0] = params[0]
    if n == 2:
        # linear
        tree[1][0][1][0] = params[1]
    elif n == 3:
        if len(tree[1]) == 2:
            # cherry
            tree[1][0][1][0] = params[1]
            tree[1][1][1][0] = params[2]
        else:
            # linear
            tree[1][0][1][0] = params[1]
            tree[1][0][1][1][0][1][0] = params[2]
    elif n == 4:
        if len(tree[1]) == 3:
            # three cherry
            # [0, [(<TreeDirection.N: 'N'>, [0, []]), (<TreeDirection.NW: 'NW'>, [0, []]), (<TreeDirection.NE: 'NE'>, [0, []])]]
            tree[1][0][1][0] = params[1]
            tree[1][1][1][0] = params[2]
            tree[1][2][1][0] = params[3]
        elif len(tree[1]) == 2:
            # mutated cherry
            # [0, [(<TreeDirection.E: 'E'>, [0, []]), (<TreeDirection.S: 'S'>, [0, [(<TreeDirection.SE: 'SE'>, [0, []])]])]]
            tree[1][0][1][0] = params[1]
            tree[1][1][1][0] = params[2]
            tree[1][1][1][1][0][1][0] = params[3]
        elif len(tree[1]) == 1:
            if len(tree[1][0][1][1]) == 1:
                # linear
                # [0, [(<TreeDirection.E: 'E'>, [0, [(<TreeDirection.E: 'E'>, [0, [(<TreeDirection.E: 'E'>, [0, []])]])]])]]
                tree[1][0][1][0] = params[1]
                tree[1][0][1][1][0][1][0] = params[2]
                tree[1][0][1][1][0][1][1][0][1][0] = params[3]
            else:
                # Y
                # [0, [(<TreeDirection.E: 'E'>, [0, [(<TreeDirection.W: 'W'>, [0, []]), (<TreeDirection.SE: 'SE'>, [0, []])]])]]
                tree[1][0][1][0] = params[1]
                tree[1][0][1][1][0][1][0] = params[2]
                tree[1][0][1][1][1][1][0] = params[3]
    else:
        raise NotImplementedError(
            "replace_parameters_in_tree only supports trees with 2, 3, or 4 vertices"
        )

    return tree

def print_tensor_stats(x, name):
    # print(f"{name}:", x, "max:", torch.max(x), "min:", torch.min(x))
    print(f"{name}: {x}, max: {torch.max(x)}, min: {torch.min(x)}, dtype: {x.dtype}, device: {x.device}, shape: {x.shape}, is_contiguous: {x.is_contiguous()}, stride: {x.stride()}, ndim: {x.ndim}")
    # print(f"{name} max: {torch.max(x)}, min: {torch.min(x)}, dtype: {x.dtype}, device: {x.device}, shape: {x.shape}, is_contiguous: {x.is_contiguous()}, stride: {x.stride()}, ndim: {x.ndim}")
    assert not torch.isnan(x).any() and not torch.isinf(x).any(), f"{name} has nan or inf"

class ConvNextSize(StrEnum):
    TINY = auto()
    SMALL = auto()
    LARGE = auto()


class CNModelMode(StrEnum):
    BASE = auto()
    L2 = auto()
    L4 = auto()


class ENModelMode(StrEnum):
    BASE = auto()
    S3 = auto()
    S4 = auto()


class ResnetMode(StrEnum):
    RESNET20 = auto()
    RESNET32 = auto()
    RESNET44 = auto()
    RESNET56 = auto()


class LayerInChannels(IntEnum):
    LAYER1 = 16
    LAYER2 = 16
    LAYER3 = 32
    DOWNSAMPLE2 = 16
    DOWNSAMPLE3 = 32


class LayerOutChannels(IntEnum):
    LAYER1 = 16
    LAYER2 = 32
    LAYER3 = 64
    DOWNSAMPLE2 = 32
    DOWNSAMPLE3 = 64


class DataName(StrEnum):
    CIFAR10 = auto()
    CIFAR100 = auto()


class ParamInit(StrEnum):
    NORMAL = auto()
    UNIFORM = auto()
    ONES = auto()


class AEBackbone(StrEnum):
    RESNET50 = auto()
    WIDE_RESNET50 = auto()
    RESNET101 = auto()
    WIDE_RESNET101 = auto()


class ModelMode(StrEnum):
    CA = auto()
    CA_FISBLOCK = auto()
    L2 = auto()
    L2_PARAMSHARE_V1 = auto()
    L23 = auto()
    L3 = auto()
    DOWNSAMPLE = auto()
    BASE = auto()


class PoolMode(StrEnum):
    AVG = auto()
    MAX = auto()


# TODO use chenyaofo_resnet
def load_resnet(
    data_name: str = DataName.CIFAR10,
    resnet_mode: str = ResnetMode.RESNET20,
    pretrained: bool = False,
) -> nn.Module:
    if data_name not in DataName:
        raise ValueError(
            f"Invalid data_name '{data_name}'. Must be one of [{', '.join(DataName)}]."
        )

    if resnet_mode not in ResnetMode:
        raise ValueError(
            f"Invalid resnet_mode '{resnet_mode}'. Must be one of [{', '.join(ResnetMode)}]."
        )

    model = torch.hub.load(
        repo_or_dir="chenyaofo/pytorch-cifar-models",
        model=f"{data_name}_{resnet_mode}",
        trust_repo=True,
        pretrained=pretrained,
        verbose=False,
    )
    return model


def load_efficientnet(phi: int = 0, pretrained: bool = False) -> nn.Module:
    """
    Load an EfficientNet model from torchvision based on the phi value.

    Args:
        phi (int): The EfficientNet variant number (0–7).
        pretrained (bool): Whether to load pretrained weights. Default to False.

    Returns:
        torch.nn.Module: The loaded EfficientNet model.
    """
    if not (0 <= phi <= 7):
        raise ValueError(f"phi must be between 0 and 7 (inclusive) but {phi} is given.")

    model_name = f"efficientnet_b{phi}"
    model_fn = getattr(models, model_name, None)
    if model_fn is None:
        raise ValueError(
            f"EfficientNet variant '{model_name}' not found in torchvision.models."
        )

    if pretrained:
        weights_enum = getattr(models, f"EfficientNet_B{phi}_Weights", None)
        if weights_enum is not None:
            weights = weights_enum.DEFAULT

            return model_fn(weights=weights)

    return model_fn(weights=None)


def load_convnext(size: str = ConvNextSize.TINY, pretrained: bool = False):
    """
    Load a ConvNeXt model of a specified size.

    Args:
        size (str, optional): The size of the ConvNeXt model. Must be one of
        ["tiny", "small", "large"]. Default to "tiny".
        pretrained (bool, optional): If True, loads pretrained weights. Defaults to False.

    Returns:
        torch.nn.Module: The ConvNeXt model instance.
    """
    if size not in ConvNextSize:
        raise ValueError(
            f"Invalid size '{size}'. Must be one of [{', '.join(ConvNextSize)}]."
        )

    model_fn_name = f"convnext_{size}"

    if not hasattr(models, model_fn_name):
        raise ValueError(f"torchvision.models has no model named '{model_fn_name}'.")

    model_fn = getattr(models, model_fn_name)

    if pretrained:
        weight_class_name = f"ConvNeXt_{size.capitalize()}_Weights"

        if not hasattr(models, weight_class_name):
            raise ValueError(f"No pretrained weights available for ConvNeXt-{size}.")

        weights_enum = getattr(models, weight_class_name).IMAGENET1K_V1
        model = model_fn(weights=weights_enum)

    else:
        model = model_fn(weights=None)

    return model


def create_pool_layer(pool_mode: str, output_size: tuple[int, int]) -> nn.Module:
    if pool_mode == PoolMode.AVG:
        return nn.AdaptiveAvgPool2d(output_size=output_size)

    elif pool_mode == PoolMode.MAX:
        return nn.AdaptiveMaxPool2d(output_size=output_size)

    else:
        raise ValueError(
            f"pool_mode must be either {PoolMode.AVG} or {PoolMode.MAX}; but {pool_mode} is given"
        )

def get_num_layer_for_convnext(var_name, variant='tiny'):
    """
    From: https://github.com/state-spaces/s4/blob/e757cef57d89e448c413de7325ed5601aceaac13/src/models/baselines/convnext_timm.py

    Divide [3, 3, 27, 3] layers into 12 groups; each group is three
    consecutive blocks, including possible neighboring downsample layers;
    adapted from https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py
    """
    # XXX UNTESTED This was for timm; not sure if it works for torchvision
    num_max_layer = 12
    if "stem" in var_name:
        return 0

    # note:  moved norm_layer outside of downsample module
    elif "downsample" in var_name or "norm_layer" in var_name:
        stage_id = int(var_name.split('.')[2])
        if stage_id == 0:
            layer_id = 0
        elif stage_id == 1 or stage_id == 2:
            layer_id = stage_id + 1
        elif stage_id == 3:
            layer_id = 12
        return layer_id

    elif "stages" in var_name:
        stage_id = int(var_name.split('.')[2])
        block_id = int(var_name.split('.')[4])
        if stage_id == 0 or stage_id == 1:
            layer_id = stage_id + 1
        elif stage_id == 2:
            if variant == 'tiny':
                layer_id = 3 + block_id
            else:
                layer_id = 3 + block_id // 3
        elif stage_id == 3:
            layer_id = 12
        return layer_id

    else:
        return num_max_layer + 1

def get_num_layer_for_convnext_tiny(var_name):
    return get_num_layer_for_convnext(var_name, 'tiny')

class FISLayerParameterSharingV0(nn.Module):
    """Uses one tree with 2 nodes; controlled by model.seed"""

    def __init__(
        self,
        in_channels: int,
        semiring: str,
        num_trees: int,
        num_nodes: int,
        sums_internal_activation: str | None = None,
        tree_type: str = TreeType.RANDOM,  # unused
        seed: int | None = None,
        param_init: str = ParamInit.NORMAL,
        maxplus_linear: bool = False,
        use_valuation: bool = False,
        maxplus_normalization: bool = False,
        use_discounted_sums: bool = False,
        discount_init: float = 0.99,
        fix_alpha_1: float | None = None,
        use_complex: bool = False,
        complex_use_only_real_part: bool = False,

        complex_init_r_min: float = 0.0,
        complex_init_r_max: float = 1.0,
        complex_init_max_phase: float = 2 * 3.141592653589793,
        # OPTIONS
        # - [ ] (128, 8, 28, 28) ->_zero (128, 8, 28, 28, 2)    ->_sum (128, 8, 28, 28, 2) ->_project (128, 8, 28, 28, 2)
    ):
        super().__init__()
        self.in_channels = in_channels
        self.sums_internal_activation = sums_internal_activation
        self.semiring = semiring
        self.num_trees = num_trees
        # Validate complex parameters
        if use_complex:
            if use_valuation:
                print("[WARNING] use_valuation is True but use_complex is True. Setting use_complex to False.")
                use_complex = False
            if maxplus_normalization:
                print("[WARNING] maxplus_normalization is True but use_complex is True. Setting use_complex to False.")
                use_complex = False
            if semiring != 'real':
                print(f"[WARNING] semiring is not {semiring} but use_complex is True. Setting use_complex to False.")
                use_complex = False
        if use_complex:
            assert use_discounted_sums, "use_discounted_sums must be True with use_complex"
            self.go_to_complex = nn.Linear(1, 2)
            self.go_to_real = nn.Linear(2, 1)
            if complex_use_only_real_part:
                # go_to_complex: embed real to complex by setting imaginary part to zero
                # Maps [x] -> [x, 0]
                with torch.no_grad():
                    self.go_to_complex.weight.data = torch.tensor([[1.0], [0.0]])
                    self.go_to_complex.bias.data = torch.tensor([0.0, 0.0])
                self.go_to_complex.weight.requires_grad = False
                self.go_to_complex.bias.requires_grad = False
                # go_to_real: pick the real part
                # Maps [x, y] -> [x]
                with torch.no_grad():
                    self.go_to_real.weight.data = torch.tensor([[1.0, 0.0]])
                    self.go_to_real.bias.data = torch.tensor([0.0])
                self.go_to_real.weight.requires_grad = False
                self.go_to_real.bias.requires_grad = False
        self.use_complex = use_complex
        self.complex_use_only_real_part = complex_use_only_real_part
        assert num_nodes == 2
        self.seed = seed
        self.param_init = param_init
        self.maxplus_linear = maxplus_linear
        self.use_valuation = use_valuation
        self.maxplus_normalization = maxplus_normalization
        self.use_discounted_sums = use_discounted_sums
        
        if self.maxplus_normalization:
            if not self.use_valuation:
                print(
                    "Warning: maxplus_normalization is True but use_valuation is False. Setting use_valuation to True."
                )
                self.use_valuation = True
            #assert self.use_valuation, (
                #"maxplus_normalization requires use_valuation to be True"
            #)
            self.maxplus_norm_param = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

        if fix_alpha_1 is not None:
            self.alpha_1 = torch.ones(self.num_trees, self.in_channels, device='cuda') * fix_alpha_1 # XXX hack
        else:
            self.alpha_1 = nn.Parameter(torch.randn(self.num_trees, self.in_channels))
        self.alpha_2 = nn.Parameter(torch.randn(self.num_trees, self.in_channels))
        if self.sums_internal_activation is not None:
            self.alpha_2.activation_function_hack = get_activation_function(self.sums_internal_activation)

        # Initialize discount parameter if using discounted sums
        if self.use_discounted_sums:
            if self.use_complex:
                u1 = torch.rand(1, self.num_trees, 1, 1, device='cuda')
                self.discount_nu_log = nn.Parameter(torch.log(-0.5 * torch.log(u1 * (complex_init_r_max **2 - complex_init_r_min**2) + complex_init_r_min**2)))
                u2 = torch.rand(1, self.num_trees, 1, 1, device='cuda')
                self.discount_theta_log = nn.Parameter(torch.log(complex_init_max_phase * u2))
            else:
                self.discount = nn.Parameter(discount_init * torch.ones(1, self.num_trees, 1, 1))
        
        ats = all_trees(2)
        n_trees = len(ats)
        # self.tree_linear = [self.alpha_1,  [(TreeDirection.NE, [self.alpha_2,  [(TreeDirection.SW, [self.alpha_3,  []])]])]]
        # self.tree_cherry = [self.alpha_1p, [(TreeDirection.SW, [self.alpha_3p, []]), (TreeDirection.SE, [self.alpha_2p, []])]]
        self.tree = ats[seed % n_trees]
        self.tree = replace_parameters_in_tree(self.tree, [self.alpha_1, self.alpha_2])

        def _sum(x: torch.Tensor, discount: torch.Tensor | None = None) -> torch.Tensor:
            if self.use_discounted_sums:
                # Expand discount to match the spatial dimensions of x
                # x shape: (batch_size, channels, height, width)
                # self.discount shape: (1, in_channels, 1, 1)
                # We need to expand to: (batch_size, channels, height, width)
                if self.use_complex:
                    batch_size, channels, height, width, _ = x.shape
                    discount_expanded = discount.expand(batch_size, self.num_trees, height, width, 2)
                else:
                    batch_size, channels, height, width = x.shape
                    # This should be cheap, since it does so by using 0 strides
                    discount_expanded = discount.expand(batch_size, self.num_trees, height, width)
                
                return discounted_sum_tree(
                    x, self.tree, discount_expanded, semiring=self.semiring, 
                    maxplus_linear=self.maxplus_linear, channel_wise=False
                )
            else:
                return sum_tree(
                    x, self.tree, semiring=self.semiring, maxplus_linear=self.maxplus_linear, 
                    channel_wise=False
                )

        self.sum = _sum

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
        if self.use_complex:
            # Compute derived parameters
            Lambda_r = torch.exp(-torch.exp(self.discount_nu_log)) * torch.cos(torch.exp(self.discount_theta_log))
            Lambda_i = torch.exp(-torch.exp(self.discount_nu_log)) * torch.sin(torch.exp(self.discount_theta_log))
            # TODO gamma = torch.sqrt(1 - torch.abs(Lambda)**2)
            discount = torch.stack([Lambda_r, Lambda_i], dim=-1)
            x = x.unsqueeze(-1)
            x = self.go_to_complex(x)
            result = self.sum(x, discount)
        elif self.use_discounted_sums:
            result = self.sum(x, self.discount)
        else:
            result = self.sum(x)
        if self.use_complex:
            result = self.go_to_real(result).squeeze(-1)
        result = self.devaluate(result)
        return result

    def __str__(self):
        return f"FISLayerParameterSharingV0(num_trees={self.num_trees}, in_channels={self.in_channels}, semiring={self.semiring}, use_discounted_sums={self.use_discounted_sums}, tree={pretty_str_tree_one_line(self.tree)})"

    def __repr__(self):
        return self.__str__()


class FISLayerParameterSharingV1(nn.Module):
    def __init__(
        self,
        in_channels: int,
        semiring: str,
        num_trees: int,
        num_nodes: int,
        tree_type: str = TreeType.RANDOM,  # unused
        seed: int | None = None,
        param_init: str = ParamInit.NORMAL,
        maxplus_linear: bool = False,
        use_valuation: bool = False,
        maxplus_normalization: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.semiring = semiring
        self.num_trees = num_trees
        assert num_nodes == 3
        assert num_trees % 2 == 0
        self.seed = seed
        self.param_init = param_init
        self.maxplus_linear = maxplus_linear
        if use_valuation or maxplus_normalization:
            raise NotImplementedError(
                "FISLayerParameterSharingV1 does not support use_valuation or maxplus_normalization yet"
            )

        self.alpha_1 = nn.Parameter(torch.randn(self.num_trees // 2, self.in_channels))
        self.alpha_2 = nn.Parameter(torch.randn(self.num_trees // 2, self.in_channels))
        self.alpha_3 = nn.Parameter(torch.randn(self.num_trees // 2, self.in_channels))
        self.alpha_1p = nn.Parameter(torch.randn(self.num_trees // 2, self.in_channels))
        self.alpha_2p = nn.Parameter(torch.randn(self.num_trees // 2, self.in_channels))
        self.alpha_3p = nn.Parameter(torch.randn(self.num_trees // 2, self.in_channels))
        # TODO sweep over tree topology + cardinal directions
        self.tree_linear = [
            self.alpha_1,
            [
                (
                    TreeDirection.NE,
                    [self.alpha_2, [(TreeDirection.SW, [self.alpha_3, []])]],
                )
            ],
        ]
        self.tree_cherry = [
            self.alpha_1p,
            [
                (TreeDirection.SW, [self.alpha_3p, []]),
                (TreeDirection.SE, [self.alpha_2p, []]),
            ],
        ]

        def _sum(x: torch.Tensor) -> torch.Tensor:
            return sum_trees(
                x,
                [self.tree_linear, self.tree_cherry],
                semiring=self.semiring,
                maxplus_linear=self.maxplus_linear,
                cat=True,
            )

        self.sum = _sum

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sum(x)



class FISLayerParameterSharingV2(nn.Module):
    def __init__(
        self,
        in_channels: int,
        semiring: str,
        num_trees: int,
        num_nodes: int,
        sums_internal_activation: str | None = None,
        tree_type: str = TreeType.RANDOM,  # unused
        seed: int | None = None,
        param_init: str = ParamInit.NORMAL, # XXX unused
        maxplus_linear: bool = False,
        use_valuation: bool = False,
        maxplus_normalization: bool = False,
        use_discounted_sums: bool = False,
        discount_init: float = 0.99,
        fix_alpha_1: float | None = None,
        use_complex: bool = False,
        complex_use_only_real_part: bool = False, # XXX unused

        complex_init_r_min: float = 0.0,
        complex_init_r_max: float = 1.0,
        complex_init_max_phase: float = 2 * 3.141592653589793,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.sums_internal_activation = sums_internal_activation
        self.semiring = semiring
        self.num_trees = num_trees
        # Validate complex parameters
        if use_complex:
            if use_valuation:
                print("[WARNING] use_valuation is True but use_complex is True. Setting use_complex to False.")
                use_complex = False
            if maxplus_normalization:
                print("[WARNING] maxplus_normalization is True but use_complex is True. Setting use_complex to False.")
                use_complex = False
            if semiring != 'real':
                print(f"[WARNING] semiring is not {semiring} but use_complex is True. Setting use_complex to False.")
                use_complex = False
        if use_complex:
            assert use_discounted_sums, "use_discounted_sums must be True with use_complex"
            self.go_to_complex = nn.Linear(1, 2)
            self.go_to_real = nn.Linear(2, 1)
        self.use_complex = use_complex
        assert num_nodes == 3
        self.num_nodes = num_nodes
        self.tree_type = tree_type # unused
        # assert num_trees % 2 == 0
        self.seed = seed
        self.param_init = param_init # XXX unused
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

        if fix_alpha_1 is not None:
            self.alpha_1 = torch.ones(self.num_trees, self.in_channels, device='cuda') * fix_alpha_1 # XXX hack
        else:
            self.alpha_1 = nn.Parameter(torch.randn(self.num_trees, self.in_channels))
        self.alpha_2 = nn.Parameter(torch.randn(self.num_trees, self.in_channels))
        self.alpha_3 = nn.Parameter(torch.randn(self.num_trees, self.in_channels))
        if self.sums_internal_activation is not None:
            self.alpha_2.activation_function_hack = get_activation_function(self.sums_internal_activation)
            self.alpha_3.activation_function_hack = get_activation_function(self.sums_internal_activation)

        # Initialize discount parameter if using discounted sums
        if self.use_discounted_sums:
            if self.use_complex:
                u1 = torch.rand(1, self.num_trees, 1, 1, device='cuda')
                self.discount_nu_log = nn.Parameter(torch.log(-0.5 * torch.log(u1 * (complex_init_r_max **2 - complex_init_r_min**2) + complex_init_r_min**2)))
                u2 = torch.rand(1, self.num_trees, 1, 1, device='cuda')
                self.discount_theta_log = nn.Parameter(torch.log(complex_init_max_phase * u2))
            else:
                self.discount = nn.Parameter(discount_init * torch.ones(1, self.num_trees, 1, 1))
        
        ats = all_trees(3)
        n_trees = len(ats)
        # self.tree_linear = [self.alpha_1,  [(TreeDirection.NE, [self.alpha_2,  [(TreeDirection.SW, [self.alpha_3,  []])]])]]
        # self.tree_cherry = [self.alpha_1p, [(TreeDirection.SW, [self.alpha_3p, []]), (TreeDirection.SE, [self.alpha_2p, []])]]
        self.tree = ats[seed % n_trees]

        self.tree = replace_parameters_in_tree(
            self.tree, [self.alpha_1, self.alpha_2, self.alpha_3]
        )

        def _sum(x: torch.Tensor, discount: torch.Tensor | None = None) -> torch.Tensor:
            if self.use_discounted_sums:
                # Expand discount to match the spatial dimensions of x
                # x shape: (batch_size, channels, height, width)
                # self.discount shape: (1, in_channels, 1, 1)
                # We need to expand to: (batch_size, channels, height, width)
                if self.use_complex:
                    batch_size, channels, height, width, _ = x.shape
                    discount_expanded = discount.expand(batch_size, self.num_trees, height, width, 2)
                else:
                    batch_size, channels, height, width = x.shape
                    # This should be cheap, since it does so by using 0 strides
                    discount_expanded = discount.expand(batch_size, self.num_trees, height, width)
                # print("[FISLayerParameterSharingV2] tree: ", self.tree)
                # print_tensor_stats(self.discount, "[FISLayerParameterSharingV2] discount")
                # print_tensor_stats(x, "[FISLayerParameterSharingV2] x")
                ret = discounted_sum_tree(
                    x, self.tree, discount_expanded, semiring=self.semiring, 
                    maxplus_linear=self.maxplus_linear, channel_wise=False
                )
                #if torch.isnan(ret).any() or torch.isinf(ret).any():
                #    # pickle input, alpha_1, alpha_2, alpha_3, discount, tree
                #    import pickle
                #    filename = "fis_resnet_v1_nan_inf.pkl"
                #    import os
                #    print(f"Saving to file in current directory: {os.getcwd()}/{filename}")
                #    with open(filename, "wb") as f:
                #        # pickle.dump((x, self, self.alpha_1, self.alpha_2, self.alpha_3, self.discount, self.tree), f)
                #        # all parameters as well as x
                #        pickle.dump((x, self.in_channels, self.semiring, self.num_trees, self.num_nodes, self.tree_type, self.seed, self.param_init, self.maxplus_linear, self.use_valuation, self.maxplus_normalization, self.use_discounted_sums, self.discount_init, self.alpha_1, self.alpha_2, self.alpha_3, self.tree, ret), f)



                # print_tensor_stats(ret, "ret")
                # assert not nans
                # assert not torch.isnan(ret).any()
                return ret
            else:
                return sum_tree(
                    x, self.tree, semiring=self.semiring, maxplus_linear=self.maxplus_linear, 
                    channel_wise=False
                )

        self.sum = _sum

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
        if self.use_complex:
            # Compute derived parameters
            Lambda_r = torch.exp(-torch.exp(self.discount_nu_log)) * torch.cos(torch.exp(self.discount_theta_log))
            Lambda_i = torch.exp(-torch.exp(self.discount_nu_log)) * torch.sin(torch.exp(self.discount_theta_log))
            # TODO gamma = torch.sqrt(1 - torch.abs(Lambda)**2)
            discount = torch.stack([Lambda_r, Lambda_i], dim=-1)
            x = x.unsqueeze(-1)
            x = self.go_to_complex(x)
            result = self.sum(x, discount)
        elif self.use_discounted_sums:
            result = self.sum(x, self.discount)
        else:
            result = self.sum(x)
        if self.use_complex:
            result = self.go_to_real(result).squeeze(-1)
        result = self.devaluate(result)
        return result

    def __str__(self):
        return f"FISLayerParameterSharingV2(num_trees={self.num_trees}, in_channels={self.in_channels}, semiring={self.semiring}, use_discounted_sums={self.use_discounted_sums}, tree={pretty_str_tree_one_line(self.tree)})"

    def __repr__(self):
        return self.__str__()


class FISLayerParameterSharingV3(nn.Module):
    def __init__(
        self,
        in_channels: int,
        semiring: str,
        num_trees: int,
        num_nodes: int,
        tree_type: str = TreeType.RANDOM,  # unused
        seed: int | None = None,
        param_init: str = ParamInit.NORMAL,
        maxplus_linear: bool = False,
        use_valuation: bool = False,
        maxplus_normalization: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.semiring = semiring
        self.num_trees = num_trees
        assert num_nodes == 4
        self.seed = seed
        self.param_init = param_init
        self.maxplus_linear = maxplus_linear
        if use_valuation or maxplus_normalization:
            raise NotImplementedError(
                "FISLayerParameterSharingV3 does not support use_valuation or maxplus_normalization yet"
            )

        self.alpha_1 = nn.Parameter(torch.randn(self.num_trees, self.in_channels))
        self.alpha_2 = nn.Parameter(torch.randn(self.num_trees, self.in_channels))
        self.alpha_3 = nn.Parameter(torch.randn(self.num_trees, self.in_channels))
        self.alpha_4 = nn.Parameter(torch.randn(self.num_trees, self.in_channels))
        ats = all_trees(4)
        n_trees = len(ats)
        self.tree = ats[seed % n_trees]

        # print("FISLayerParameterSharingV3: using tree:", self.tree)
        # pretty_print_tree(self.tree)
        self.tree = replace_parameters_in_tree(
            self.tree, [self.alpha_1, self.alpha_2, self.alpha_3, self.alpha_4]
        )
        # print("FISLayerParameterSharingV3: using tree:", self.tree)
        # pretty_print_tree(self.tree)

        def _sum(x: torch.Tensor) -> torch.Tensor:
            return sum_tree(
                x, self.tree, semiring=self.semiring, maxplus_linear=self.maxplus_linear
            )

        self.sum = _sum

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sum(x)

    def __str__(self):
        return f"FISLayerParameterSharingV2(num_trees={self.num_trees}, in_channels={self.in_channels}, semiring={self.semiring}, tree={pretty_str_tree_one_line(self.tree)})"

    def __repr__(self):
        return self.__str__()


class FISLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        semiring: str,
        num_trees: int,
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
        self.in_channels = in_channels
        self.semiring = semiring
        self.num_trees = num_trees
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
            num_trees=self.num_trees,
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


def get_fis_layer(tree_type: str):
    if tree_type == "parameter_sharing_v0":
        return FISLayerParameterSharingV0
    elif tree_type == "parameter_sharing_v1":
        return FISLayerParameterSharingV1
    elif tree_type == "parameter_sharing_v2":
        return FISLayerParameterSharingV2
    elif tree_type == "parameter_sharing_v3":
        return FISLayerParameterSharingV3
    else:
        return FISLayer


class FISBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        semiring: str,
        num_nodes: int,
        pool_mode: str,
        output_size: tuple[int, int],
        match_dim: bool,
        match_dim_cnn: bool = False,
        tree_type: str = TreeType.RANDOM,
        seed: int | None = None,
        add_identity: bool = True,
        maxplus_linear: bool = False,
        use_valuation: bool = False,
        maxplus_normalization: bool = False,

        use_discounted_sums: bool = False,
        use_complex: bool = False,
        complex_init_r_min: float = 0.0,
        complex_init_r_max: float = 1.0,
        complex_init_max_phase: float = 2 * 3.141592653589793,
        discount_init: float = 0.99,

        sums_internal_activation: str | None = None,
    ):
        super().__init__()

        if isinstance(seed, int):
            if match_dim_cnn:
                seeds = [seed] * 2
            else:
                seeds = [seed] * 3
        else:
            if match_dim_cnn:
                assert len(seed) == 2, "seed must be a list of 2 integers or a single integer"
            else:
                assert len(seed) == 3, "seed must be a list of 3 integers or a single integer"
            seeds = seed

        print(
            "fisl1 in_channels=",
            in_channels,
            " out_channels=",
            out_channels,
            " tree_type=",
            tree_type,
            repr(tree_type),
            type(tree_type),
            tree_type == "parameter_sharing_v1",
            " num_nodes=", num_nodes,
        )
        fislayer = get_fis_layer(tree_type)
        print("fislayer=", fislayer)
        self.fisl1 = fislayer(
            in_channels=in_channels,
            num_trees=out_channels,
            num_nodes=num_nodes,
            semiring=semiring,
            tree_type=tree_type,
            seed=seeds[0],
            maxplus_linear=maxplus_linear,
            use_valuation=use_valuation,
            maxplus_normalization=maxplus_normalization,
            use_discounted_sums=use_discounted_sums,
            use_complex=use_complex,
            complex_init_r_min=complex_init_r_min,
            complex_init_r_max=complex_init_r_max,
            complex_init_max_phase=complex_init_max_phase,
            discount_init=discount_init,
            sums_internal_activation=sums_internal_activation,
        )
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.fisl2 = fislayer(
            in_channels=out_channels,
            num_trees=out_channels,
            num_nodes=num_nodes,
            semiring=semiring,
            tree_type=tree_type,
            seed=seeds[1],
            maxplus_linear=maxplus_linear,
            use_valuation=use_valuation,
            maxplus_normalization=maxplus_normalization,
            use_discounted_sums=use_discounted_sums,
            use_complex=use_complex,
            complex_init_r_min=complex_init_r_min,
            complex_init_r_max=complex_init_r_max,
            complex_init_max_phase=complex_init_max_phase,
            discount_init=discount_init,
            sums_internal_activation=sums_internal_activation,
        )
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        self.match_dim = match_dim
        self.add_identity = add_identity
        if self.match_dim:
            if match_dim_cnn:
                self.match = nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(num_features=out_channels),
                )
            else:
                self.match = nn.Sequential( 
                    fislayer( # XXX This should probably be a CNN
                        in_channels=in_channels,
                        num_trees=out_channels,
                        num_nodes=num_nodes,
                        semiring=semiring,
                        tree_type=tree_type,
                        seed=seeds[2],
                        maxplus_linear=maxplus_linear,
                        use_valuation=use_valuation,
                        maxplus_normalization=maxplus_normalization,
                        use_discounted_sums=use_discounted_sums,
                        use_complex=use_complex,
                        complex_init_r_min=complex_init_r_min,
                        complex_init_r_max=complex_init_r_max,
                        complex_init_max_phase=complex_init_max_phase,
                        discount_init=discount_init,
                        sums_internal_activation=sums_internal_activation,
                    ),
                    nn.BatchNorm2d(num_features=out_channels),
                )

        self.pool = create_pool_layer(pool_mode=pool_mode, output_size=output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.fisl1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.fisl2(out)
        out = self.bn2(out)

        if self.add_identity:
            if self.match_dim:
                identity = self.match(x)

            out += identity

        out = self.relu(out)

        out = self.pool(out)

        return out


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class FISBlockV2(nn.Module):
    """Two 'seeds' to specify the two tree structures.
    No fis for the skip connection; just a conv1x1."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        semiring: str,
        num_nodes: int,
        pool_mode: str,
        output_size: tuple[int, int],
        match_dim: bool,
        tree_type: str = TreeType.RANDOM,
        seeds: list[int] | None = None,
        add_identity: bool = True,
        maxplus_linear: bool = False,
        use_valuation: bool = False,
        maxplus_normalization: bool = False,
    ):
        super().__init__()

        print(
            "fisl1 in_channels=",
            in_channels,
            " out_channels=",
            out_channels,
            " tree_type=",
            tree_type,
            repr(tree_type),
            type(tree_type),
            tree_type == "parameter_sharing_v1",
        )
        fislayer = get_fis_layer(tree_type)
        self.fisl1 = fislayer(
            in_channels=in_channels,
            num_trees=out_channels,
            num_nodes=num_nodes,
            semiring=semiring,
            tree_type=tree_type,
            seed=seeds[0],
            maxplus_linear=maxplus_linear,
            use_valuation=use_valuation,
            maxplus_normalization=maxplus_normalization,
        )
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.fisl2 = fislayer(
            in_channels=out_channels,
            num_trees=out_channels,
            num_nodes=num_nodes,
            semiring=semiring,
            tree_type=tree_type,
            seed=seeds[1],
            maxplus_linear=maxplus_linear,
            use_valuation=use_valuation,
            maxplus_normalization=maxplus_normalization,
        )
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        self.match_dim = match_dim
        self.add_identity = add_identity
        self.match = nn.Sequential(
            conv1x1(in_planes=in_channels, out_planes=out_channels),
            nn.BatchNorm2d(num_features=out_channels),
        )

        self.pool = create_pool_layer(pool_mode=pool_mode, output_size=output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.fisl1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.fisl2(out)
        out = self.bn2(out)

        if self.add_identity:
            if self.match_dim:
                identity = self.match(x)

            out += identity

        out = self.relu(out)

        out = self.pool(out)

        return out


class FISResNet(nn.Module):
    def __init__(
        self,
        model_mode: str,
        data_name: str,
        resnet_mode: str,
        pool_mode: str,
        num_nodes: int,
        semiring: str,
        seed: int | None = None,
        seeds: list[int] | None = None,  # for model_mode=='l2_block_v1'
        tree_type: str = TreeType.RANDOM,
        maxplus_linear: bool = False,
        use_valuation: bool = False,
        maxplus_normalization: bool = False,
        num_classes=None,  # catch kwarg for generic instantiation
    ):
        super().__init__()

        self.pool_mode = pool_mode

        self.base = load_resnet(
            data_name=data_name, resnet_mode=resnet_mode, pretrained=False
        )

        if data_name == DataName.CIFAR10:
            num_classes = 10
        elif data_name == DataName.CIFAR100:
            num_classes = 100
        else:
            raise ValueError(
                f"data_name must be one of {', '.join(DataName)} but {data_name} is given."
            )

        match model_mode:
            case ModelMode.L2:
                self.base.layer2 = FISBlock(
                    in_channels=LayerInChannels.LAYER2,
                    out_channels=LayerOutChannels.LAYER2,
                    semiring=semiring,
                    num_nodes=num_nodes,
                    pool_mode=pool_mode,
                    output_size=(16, 16),
                    add_identity=True,
                    match_dim=True,
                    tree_type=tree_type,
                    seed=seed,
                    maxplus_linear=maxplus_linear,
                    use_valuation=use_valuation,
                    maxplus_normalization=maxplus_normalization,
                )
                # self.log_to_wandb = [ [ 'FISResnet.base.layer2', str(self.base.layer2)] ]
            case "l2_block_v2":
                self.base.layer2 = FISBlockV2(
                    in_channels=LayerInChannels.LAYER2,
                    out_channels=LayerOutChannels.LAYER2,
                    semiring=semiring,
                    num_nodes=num_nodes,
                    pool_mode=pool_mode,
                    output_size=(16, 16),
                    add_identity=True,
                    match_dim=True,
                    tree_type=tree_type,
                    seeds=seeds,
                    maxplus_linear=maxplus_linear,
                    use_valuation=use_valuation,
                    maxplus_normalization=maxplus_normalization,
                )
                # self.log_to_wandb = [ [ 'FISResnet.base.layer2', str(self.base.layer2)] ]

            case ModelMode.CA_FISBLOCK:
                self.base.layer2 = FISBlock(
                    in_channels=LayerInChannels.LAYER2,
                    out_channels=LayerOutChannels.LAYER2,
                    semiring=semiring,
                    num_nodes=num_nodes,
                    pool_mode=pool_mode,
                    output_size=(16, 16),
                    add_identity=True,
                    match_dim=True,
                    tree_type=tree_type,
                    seed=seed,
                    maxplus_linear=maxplus_linear,
                    use_valuation=use_valuation,
                    maxplus_normalization=maxplus_normalization,
                )

                self.base.layer3 = nn.Identity()

                self.base.avgpool = nn.Sequential(
                    FISLayer(
                        in_channels=LayerOutChannels.LAYER2,
                        num_trees=LayerOutChannels.LAYER2,
                        num_nodes=num_nodes,
                        semiring=semiring,
                        seed=seed,
                        tree_type=tree_type,
                        maxplus_linear=maxplus_linear,
                        use_valuation=use_valuation,
                        maxplus_normalization=maxplus_normalization,
                    ),
                    nn.BatchNorm2d(num_features=LayerOutChannels.LAYER2),
                    create_pool_layer(pool_mode=pool_mode, output_size=(1, 1)),
                )
                # we need to match the fc layer with the output of layer 2
                self.base.fc = nn.Linear(
                    in_features=LayerOutChannels.LAYER2, out_features=num_classes
                )

            case ModelMode.CA:
                # remove layers 2 and 3
                self.base.layer2 = nn.Identity()
                self.base.layer3 = nn.Identity()

                self.base.avgpool = nn.AdaptiveAvgPool2d(
                    output_size=(1, 1)
                )  # the original Resnet architecture uses this

                # we need to match the fc layer with the output of layer 2
                self.base.fc = nn.Linear(
                    in_features=LayerOutChannels.LAYER1, out_features=num_classes
                )

            case ModelMode.L23:
                self.base.layer2 = FISBlock(
                    in_channels=LayerInChannels.LAYER2,
                    out_channels=LayerOutChannels.LAYER2,
                    semiring=semiring,
                    num_nodes=num_nodes,
                    pool_mode=pool_mode,
                    output_size=(16, 16),
                    add_identity=True,
                    match_dim=True,
                    seed=seed,
                    maxplus_linear=maxplus_linear,
                    use_valuation=use_valuation,
                    maxplus_normalization=maxplus_normalization,
                    tree_type=tree_type,
                )
                self.base.layer3 = FISBlock(
                    in_channels=LayerInChannels.LAYER3,
                    out_channels=LayerOutChannels.LAYER3,
                    semiring=semiring,
                    num_nodes=num_nodes,
                    pool_mode=pool_mode,
                    output_size=(8, 8),
                    add_identity=True,
                    match_dim=True,
                    seed=seed,
                    maxplus_linear=maxplus_linear,
                    use_valuation=use_valuation,
                    maxplus_normalization=maxplus_normalization,
                    tree_type=tree_type,
                )

            case ModelMode.L3:
                self.base.layer3 = FISBlock(
                    in_channels=LayerInChannels.LAYER3,
                    out_channels=LayerOutChannels.LAYER3,
                    semiring=semiring,
                    num_nodes=num_nodes,
                    pool_mode=pool_mode,
                    output_size=(8, 8),
                    add_identity=True,
                    match_dim=True,
                    seed=seed,
                    maxplus_linear=maxplus_linear,
                    use_valuation=use_valuation,
                    maxplus_normalization=maxplus_normalization,
                    tree_type=tree_type,
                )

            case ModelMode.DOWNSAMPLE:
                self.base.layer2[0].downsample = FISBlock(
                    in_channels=LayerInChannels.DOWNSAMPLE2,
                    out_channels=LayerOutChannels.DOWNSAMPLE2,
                    semiring=semiring,
                    num_nodes=num_nodes,
                    pool_mode=pool_mode,
                    output_size=(16, 16),
                    add_identity=False,
                    match_dim=False,
                    seed=seed,
                    maxplus_linear=maxplus_linear,
                    use_valuation=use_valuation,
                    maxplus_normalization=maxplus_normalization,
                    tree_type=tree_type,
                )

                self.base.layer3[0].downsample = FISBlock(
                    in_channels=LayerInChannels.DOWNSAMPLE3,
                    out_channels=LayerOutChannels.DOWNSAMPLE3,
                    semiring=semiring,
                    num_nodes=num_nodes,
                    pool_mode=pool_mode,
                    output_size=(8, 8),
                    add_identity=False,
                    match_dim=False,
                    seed=seed,
                    maxplus_linear=maxplus_linear,
                    use_valuation=use_valuation,
                    maxplus_normalization=maxplus_normalization,
                    tree_type=tree_type,
                )

            case ModelMode.BASE:
                pass

            case _:
                raise ValueError(
                    f"model_mode must be one of {', '.join([m for m in ModelMode])}; but {model_mode} is given"
                )

        if model_mode not in (
            ModelMode.BASE,
            ModelMode.CA,
            ModelMode.CA_FISBLOCK,
        ):
            fislayer = get_fis_layer(tree_type)
            self.base.avgpool = nn.Sequential(
                fislayer(
                    in_channels=LayerOutChannels.DOWNSAMPLE3,
                    num_trees=LayerOutChannels.DOWNSAMPLE3,
                    num_nodes=num_nodes,
                    semiring=semiring,
                    seed=seed,
                    tree_type=tree_type,
                    maxplus_linear=maxplus_linear,
                    use_valuation=use_valuation,
                    maxplus_normalization=maxplus_normalization,
                ),
                nn.BatchNorm2d(num_features=LayerOutChannels.DOWNSAMPLE3),
                create_pool_layer(pool_mode=pool_mode, output_size=(1, 1)),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x)


class FeatureExtractor(nn.Module):
    def __init__(self, backbone: str):
        super().__init__()

        match backbone:
            case AEBackbone.RESNET50:
                self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

            case AEBackbone.WIDE_RESNET50:
                self.model = models.wide_resnet50_2(
                    weights=models.Wide_ResNet50_2_Weights.DEFAULT
                )

            case AEBackbone.RESNET101:
                self.model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)

            case AEBackbone.WIDE_RESNET101:
                self.model = models.wide_resnet101_2(
                    weights=models.Wide_ResNet101_2_Weights.DEFAULT
                )

            case _:
                raise ValueError(
                    f"Invalid backbone: {backbone}; backbone must be one of {', '.join(AEBackbone)}"
                )

        self.OUT_CHANNELS = 1536  # number of channels in the concatenated features
        self.backbone = backbone

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        def hook(module, input, output) -> None:
            self.features.append(output)

        self.model.layer2[-1].register_forward_hook(hook)
        self.model.layer3[-1].register_forward_hook(hook)

    def forward(self, input):
        self.features = []
        with torch.no_grad():
            _ = self.model(input)

        resize_pool = nn.AdaptiveAvgPool2d(output_size=self.features[0].shape[-1])

        return torch.cat([self.features[0], resize_pool(self.features[1])], 1)


class FISEfficientNet(nn.Module):
    """
    Requires image of size C, H, W = 3, 224, 224
    """

    def __init__(
        self,
        model_mode: str,
        pool_mode: str,
        num_nodes: int,
        semiring: str,
        num_classes: int,
        seed: int | None = None,
        tree_type: str = TreeType.RANDOM,
    ):
        super().__init__()

        self.model_mode = model_mode
        self.pool_mode = pool_mode
        self.num_nodes = num_nodes
        self.semiring = semiring
        self.num_classes = num_classes
        self.seed = seed
        self.tree_type = tree_type

        PHI = 0  # for now we test for phi=0 as the input img resolution is minimal (224, 224)
        self.base = load_efficientnet(
            phi=PHI,
            pretrained=False,
        )

        def _create_fisblock(
            in_channels: int, out_channels: int, output_size: tuple[int, int]
        ) -> FISBlock:
            return FISBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                semiring=semiring,
                num_nodes=num_nodes,
                pool_mode=pool_mode,
                output_size=output_size,
                match_dim=True,
                tree_type=tree_type,
                seed=seed,
                add_identity=True,
            )

        match model_mode:
            case ENModelMode.BASE:
                pass

            case ENModelMode.S3:
                IN_C = 16
                OUT_C, OUT_R = 24, 56
                self.base.features[2] = _create_fisblock(
                    in_channels=IN_C, out_channels=OUT_C, output_size=(OUT_R, OUT_R)
                )
            case ENModelMode.S4:
                IN_C = 24
                OUT_C, OUT_R = 40, 28
                self.base.features[3] = _create_fisblock(
                    in_channels=IN_C, out_channels=OUT_C, output_size=(OUT_R, OUT_R)
                )

            case _:
                raise ValueError(
                    f"invalid model_mode={model_mode} provided. Value must be one of {', '.join(ENModelMode)}."
                )

        in_features = self.base.classifier[-1].in_features
        self.base.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x) -> torch.Tensor:
        return self.base(x)


class FISConvNext(nn.Module):
    """
    For images of shape 3, H, W.
    """

    def __init__(
        self,
        model_mode: str,
        pool_mode: str,
        num_nodes: int,
        semiring: str,
        num_classes: int,
        seed: int | None = None,
        tree_type: str = TreeType.RANDOM,

        use_complex: bool = False,
        complex_init_r_min: float = 0.0,
        complex_init_r_max: float = 1.0,
        complex_init_max_phase: float = 2 * 3.141592589793,
        use_discounted_sums: bool = False,
        discount_init: float = 0.99,
        sums_internal_activation: str | None = None,

        match_dim_cnn: bool = False, # otherwise, use fis
    ):
        super().__init__()

        self.model_mode = model_mode
        self.pool_mode = pool_mode
        self.num_nodes = num_nodes
        self.semiring = semiring
        self.num_classes = num_classes
        self.seed = seed
        self.tree_type = tree_type

        SIZE = "tiny"  # testing the tiny model for now as its params is around 28M
        self.base = load_convnext(
            size=SIZE,
            pretrained=False,
        )

        def _make_fisblock(
            in_channels: int, out_channels: int, output_size: tuple[int, int]
        ) -> FISBlock:
            return FISBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                semiring=semiring,
                num_nodes=num_nodes,
                pool_mode=pool_mode,
                output_size=output_size,
                match_dim=True,
                match_dim_cnn=match_dim_cnn,
                tree_type=tree_type,
                seed=seed,
                add_identity=True,
                use_discounted_sums=use_discounted_sums,
                use_complex=use_complex,
                complex_init_r_min=complex_init_r_min,
                complex_init_r_max=complex_init_r_max,
                complex_init_max_phase=complex_init_max_phase,
                discount_init=discount_init,
                sums_internal_activation=sums_internal_activation,
            )

        match model_mode:
            case CNModelMode.BASE:
                pass

            case CNModelMode.L2:
                IN_C = 96
                OUT_C, OUT_R = 96, 56
                self.base.features[1] = _make_fisblock(
                    in_channels=IN_C, out_channels=OUT_C, output_size=(OUT_R, OUT_R)
                )
            case CNModelMode.L4:
                IN_C = 192
                OUT_C, OUT_R = 192, 28
                self.base.features[3] = _make_fisblock(
                    in_channels=IN_C, out_channels=OUT_C, output_size=(OUT_R, OUT_R)
                )

            case 'just_gating':
                directions = list(TreeDirection)
                direction_1 = directions[ seed[0] % len(directions) ]
                direction_2 = directions[ seed[1] % len(directions) ]
                direction_3 = directions[ seed[2] % len(directions) ]
                print(f'[FISConvNext] direction_1={direction_1}, direction_2={direction_2}, direction_3={direction_3}')
                self.base.features[1].append(GatingLayer(in_channels=96,  direction=direction_1,  semiring=semiring, discounted_sums=use_discounted_sums))
                self.base.features[3].append(GatingLayer(in_channels=192, direction=direction_2, semiring=semiring, discounted_sums=use_discounted_sums))
                self.base.features[5].append(GatingLayer(in_channels=384, direction=direction_3, semiring=semiring, discounted_sums=use_discounted_sums))
                # self.base.features[7].append(GatingLayer(in_channels=768, direction=TreeDirection.N, semiring=semiring, discounted_sums=False))

            case _:
                raise ValueError(
                    f"invalid model_mode={model_mode} provided. Value must be one of {', '.join(CNModelMode)}."
                )

        in_features = self.base.classifier[-1].in_features
        self.base.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x) -> torch.Tensor:
        return self.base(x)


if __name__ == "__main__":
    # FISLayerParameterSharingV3(in_channels=16, semiring="plus_times", num_trees=32, num_nodes=4, seed=1)
    linear_3 = [0, [(TreeDirection.NE, [0, [(TreeDirection.SW, [0, []])]])]]
    assert n_vertices(linear_3) == 3
    linear_3 = replace_parameters_in_tree(linear_3, [11.0, 22.0, 33.0])
    assert linear_3 == [
        11.0,
        [(TreeDirection.NE, [22.0, [(TreeDirection.SW, [33.0, []])]])],
    ]

    cherry_3 = [0, [(TreeDirection.SW, [0, []]), (TreeDirection.SE, [0, []])]]
    assert n_vertices(cherry_3) == 3
    cherry_3 = replace_parameters_in_tree(cherry_3, [11.0, 22.0, 33.0])
    assert cherry_3 == [
        11.0,
        [(TreeDirection.SW, [22.0, []]), (TreeDirection.SE, [33.0, []])],
    ]

    cherry_4 = [
        0,
        [
            (TreeDirection.N, [0, []]),
            (TreeDirection.NW, [0, []]),
            (TreeDirection.NE, [0, []]),
        ],
    ]
    assert n_vertices(cherry_4) == 4
    cherry_4 = replace_parameters_in_tree(cherry_4, [11.0, 22.0, 33.0, 44.0])
    assert cherry_4 == [
        11.0,
        [
            (TreeDirection.N, [22.0, []]),
            (TreeDirection.NW, [33.0, []]),
            (TreeDirection.NE, [44.0, []]),
        ],
    ]
    mutated_cherry_4 = [
        0,
        [
            (TreeDirection.E, [0, []]),
            (TreeDirection.S, [0, [(TreeDirection.SE, [0, []])]]),
        ],
    ]
    assert n_vertices(mutated_cherry_4) == 4
    mutated_cherry_4 = replace_parameters_in_tree(
        mutated_cherry_4, [11.0, 22.0, 33.0, 44.0]
    )
    assert mutated_cherry_4 == [
        11.0,
        [
            (TreeDirection.E, [22.0, []]),
            (TreeDirection.S, [33.0, [(TreeDirection.SE, [44.0, []])]]),
        ],
    ]

    linear_4 = [
        0,
        [
            (
                TreeDirection.E,
                [0, [(TreeDirection.E, [0, [(TreeDirection.E, [0, []])]])]],
            )
        ],
    ]
    assert n_vertices(linear_4) == 4
    linear_4 = replace_parameters_in_tree(linear_4, [11.0, 22.0, 33.0, 44.0])
    assert linear_4 == [
        11.0,
        [
            (
                TreeDirection.E,
                [22.0, [(TreeDirection.E, [33.0, [(TreeDirection.E, [44.0, []])]])]],
            )
        ],
    ]

    Y_4 = [
        0,
        [
            (
                TreeDirection.E,
                [0, [(TreeDirection.W, [0, []]), (TreeDirection.SE, [0, []])]],
            )
        ],
    ]
    assert n_vertices(Y_4) == 4
    Y_4 = replace_parameters_in_tree(Y_4, [11.0, 22.0, 33.0, 44.0])
    assert Y_4 == [
        11.0,
        [
            (
                TreeDirection.E,
                [22.0, [(TreeDirection.W, [33.0, []]), (TreeDirection.SE, [44.0, []])]],
            )
        ],
    ]


class GatingLayer(nn.Module):
    def __init__(self, in_channels: int, direction: TreeDirection, semiring: str, discounted_sums: bool = False):
        super().__init__()
        self.in_channels = in_channels
        self.proj = nn.Linear(in_channels, in_channels * 2)
        self.direction = direction
        self.discounted_sums = discounted_sums
        self.semiring = semiring
        if discounted_sums:
            if semiring == Semiring.REAL:
                self.discount = nn.Parameter(torch.ones(in_channels))
            else:
                self.discount = nn.Parameter(torch.zeros(in_channels))
        else:
            self.cumsum = cumsum_XX
        self.gamma = nn.Parameter(torch.ones(1))
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x_perm = input.permute(0, 2, 3, 1)
        x_perm = self.proj(x_perm)
        x = x_perm.permute(0, 3, 1, 2)
        if self.discounted_sums:
            X_cum = ds_cumsum_XX(x[:, x.shape[1]//2:], self.direction, self.discount, self.semiring)
        else:
            X_cum = cumsum_XX(x[:, x.shape[1]//2:], self.direction, self.semiring)
        gated = x[:, :x.shape[1]//2] * torch.sigmoid(X_cum)
        output = input + self.gamma * gated
        return output

def test_gating_layer():
    x = torch.randn(1, 3, 32, 32)
    #gating_layer = GatingLayer(in_channels=16, direction=TreeDirection.E, semiring=Semiring.REAL, discounted_sums=False)
    #assert gating_layer(x).shape == (1, 16, 28, 28)

    tiny = load_convnext(size="tiny", pretrained=False)
    tiny.features[1].append(GatingLayer(in_channels=96, direction=TreeDirection.E, semiring=Semiring.REAL, discounted_sums=False))
    tiny.features[3].append(GatingLayer(in_channels=192, direction=TreeDirection.S, semiring=Semiring.REAL, discounted_sums=False))
    tiny.features[5].append(GatingLayer(in_channels=384, direction=TreeDirection.W, semiring=Semiring.REAL, discounted_sums=False))
    # tiny.features[7].append(GatingLayer(in_channels=768, direction=TreeDirection.N, semiring=Semiring.REAL, discounted_sums=False))

    print(tiny.features[1])

    y = tiny(x)
    print('y.shape=', y.shape)

if __name__ == "__main__":
    test_gating_layer()

# python train.py experiment=fis_resnet20_L2-cifar10 loader.batch_size=256 model=fis_resnet20_L2 model.num_nodes=2 model.seed=0 model.tree_type=parameter_sharing_v0 optimizer.lr=0.008925206235360979 optimizer.weight_decay=0.007610487457469339 train.seed=2222 trainer.gradient_clip_val=0 trainer.max_epochs=100 trainer.track_grad_norm=-1 trainer.watch_model=False
