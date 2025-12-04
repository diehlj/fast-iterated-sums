from typing import Callable

import torch
from torch import nn
from torchvision import models

from shared.defintions import (
    AEBackbone,
    DataName,
    LayerInChannels,
    LayerOutChannels,
    ModelMode,
    ParamInit,
    PoolMode,
    ResnetMode,
    TreeType,
)
from shared.discounted_sums import sum_tree as discounted_sum_tree
from shared.sums import (
    all_trees,
    create_forest,
    n_vertices,
    pretty_str_tree_one_line,
    sum_tree,
    sum_trees,
)
from shared.pe import PE


def load_resnet(
    data_name: str = DataName.CIFAR10,
    resnet_mode: str = ResnetMode.RESNET20,
    pretrained: bool = False,
) -> nn.Module:
    model = torch.hub.load(
        repo_or_dir="chenyaofo/pytorch-cifar-models",
        model=f"{data_name}_{resnet_mode}",
        trust_repo=True,
        pretrained=pretrained,
        verbose=False,
    )
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
    ):
        super().__init__()
        self.in_channels = in_channels
        self.semiring = semiring
        self.num_trees = num_trees
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return sum_trees(x, self.forest, semiring=self.semiring)


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
        tree_type: str = TreeType.RANDOM,
        seed: int | None = None,
        add_identity: bool = True,
    ):
        super().__init__()

        self.fisl1 = FISLayer(
            in_channels=in_channels,
            num_trees=out_channels,
            num_nodes=num_nodes,
            semiring=semiring,
            tree_type=tree_type,
            seed=seed,
        )
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.fisl2 = FISLayer(
            in_channels=out_channels,
            num_trees=out_channels,
            num_nodes=num_nodes,
            semiring=semiring,
            tree_type=tree_type,
            seed=seed,
        )
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        self.match_dim = match_dim
        self.add_identity = add_identity
        self.match = nn.Sequential(
            FISLayer(
                in_channels=in_channels,
                num_trees=out_channels,
                num_nodes=num_nodes,
                semiring=semiring,
                seed=seed,
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
        tree_type: str = TreeType.RANDOM,
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
                )

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
                    tree_type=tree_type,
                )

            case ModelMode.BASE:
                pass

            case _:
                raise ValueError(
                    f"model_mode must be one of {', '.join([m for m in ModelMode])}; but {self.pool_mode} is given"
                )

        if model_mode not in (
            ModelMode.BASE,
            ModelMode.CA,
            ModelMode.CA_FISBLOCK,
        ):
            self.base.avgpool = nn.Sequential(
                FISLayer(
                    in_channels=LayerOutChannels.DOWNSAMPLE3,
                    num_trees=LayerOutChannels.DOWNSAMPLE3,
                    num_nodes=num_nodes,
                    semiring=semiring,
                    seed=seed,
                    tree_type=tree_type,
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


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def get_activation_function(activation_name: str) -> Callable:
    if activation_name == "relu":
        return torch.relu
    elif activation_name == "zero":  # For debugging.
        return lambda x: 0.0 * x
    elif activation_name == "gelu":
        return torch.nn.functional.gelu
    elif activation_name == "swish" or activation_name == "silu":
        return torch.nn.functional.silu
    elif activation_name == "leaky_relu":
        return torch.nn.functional.leaky_relu
    elif activation_name == "elu":
        return torch.nn.functional.elu
    elif activation_name == "mish":
        return torch.nn.functional.mish
    else:
        raise ValueError(f"Invalid activation function: {activation_name}")


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
        param_init: str = ParamInit.NORMAL,  # XXX unused
        maxplus_linear: bool = False,
        use_valuation: bool = False,
        maxplus_normalization: bool = False,
        use_discounted_sums: bool = False,
        discount_init: float = 0.99,
        fix_alpha_1: float | None = None,
        use_complex: bool = False,
        complex_use_only_real_part: bool = False,  # XXX unused
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
            assert use_discounted_sums, (
                "use_discounted_sums must be True with use_complex"
            )
            self.go_to_complex = nn.Linear(1, 2)
            self.go_to_real = nn.Linear(2, 1)
        self.use_complex = use_complex
        assert num_nodes == 3
        self.num_nodes = num_nodes
        self.tree_type = tree_type  # unused
        assert num_trees % 2 == 0
        self.seed = seed
        self.param_init = param_init  # XXX unused
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
            self.alpha_1 = (
                torch.ones(self.num_trees, self.in_channels, device="cuda")
                * fix_alpha_1
            )  # XXX hack
        else:
            self.alpha_1 = nn.Parameter(torch.randn(self.num_trees, self.in_channels))
        self.alpha_2 = nn.Parameter(torch.randn(self.num_trees, self.in_channels))
        self.alpha_3 = nn.Parameter(torch.randn(self.num_trees, self.in_channels))
        if self.sums_internal_activation is not None:
            self.alpha_2.activation_function_hack = get_activation_function(
                self.sums_internal_activation
            )
            self.alpha_3.activation_function_hack = get_activation_function(
                self.sums_internal_activation
            )

        # Initialize discount parameter if using discounted sums
        if self.use_discounted_sums:
            if self.use_complex:
                u1 = torch.rand(1, self.num_trees, 1, 1, device="cuda")
                self.discount_nu_log = nn.Parameter(
                    torch.log(
                        -0.5
                        * torch.log(
                            u1 * (complex_init_r_max**2 - complex_init_r_min**2)
                            + complex_init_r_min**2
                        )
                    )
                )
                u2 = torch.rand(1, self.num_trees, 1, 1, device="cuda")
                self.discount_theta_log = nn.Parameter(
                    torch.log(complex_init_max_phase * u2)
                )
            else:
                self.discount = nn.Parameter(
                    discount_init * torch.ones(1, self.num_trees, 1, 1)
                )

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
                    discount_expanded = discount.expand(
                        batch_size, self.num_trees, height, width, 2
                    )
                else:
                    batch_size, channels, height, width = x.shape
                    # This should be cheap, since it does so by using 0 strides
                    discount_expanded = discount.expand(
                        batch_size, self.num_trees, height, width
                    )
                # print("[FISLayerParameterSharingV2] tree: ", self.tree)
                # print_tensor_stats(self.discount, "[FISLayerParameterSharingV2] discount")
                # print_tensor_stats(x, "[FISLayerParameterSharingV2] x")
                ret = discounted_sum_tree(
                    x,
                    self.tree,
                    discount_expanded,
                    semiring=self.semiring,
                    maxplus_linear=self.maxplus_linear,
                    channel_wise=False,
                )
                # if torch.isnan(ret).any() or torch.isinf(ret).any():
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
                    x,
                    self.tree,
                    semiring=self.semiring,
                    maxplus_linear=self.maxplus_linear,
                    channel_wise=False,
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
            Lambda_r = torch.exp(-torch.exp(self.discount_nu_log)) * torch.cos(
                torch.exp(self.discount_theta_log)
            )
            Lambda_i = torch.exp(-torch.exp(self.discount_nu_log)) * torch.sin(
                torch.exp(self.discount_theta_log)
            )
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


def get_fis_layer(tree_type: str):
    # if tree_type == "parameter_sharing_v0":
    #     return FISLayerParameterSharingV0
    # elif tree_type == "parameter_sharing_v1":
    #     return FISLayerParameterSharingV1
    if tree_type == "parameter_sharing_v2":
        return FISLayerParameterSharingV2
    # elif tree_type == "parameter_sharing_v3":
    #     return FISLayerParameterSharingV3
    else:
        return FISLayer


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
        tree_type: str = "parameter_sharing_v2",
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


class FISAutoEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        num_nodes: int,
        semiring: str,
        seed: int | None = None,
        tree_type: str = TreeType.RANDOM,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            FISLayerParameterSharingV2(
                in_channels=in_channels,
                num_trees=2 * latent_dim,
                num_nodes=num_nodes,
                semiring=semiring,
                seed=seed,
                tree_type="parameter_sharing_v2",
                use_complex=False,
                use_discounted_sums=True,
            ),
            nn.BatchNorm2d(num_features=2 * latent_dim),
            nn.ReLU(),
            FISLayerParameterSharingV2(
                in_channels=2 * latent_dim,
                num_trees=latent_dim,
                num_nodes=num_nodes,
                semiring=semiring,
                seed=seed,
                tree_type="parameter_sharing_v2",
                use_complex=False,
                use_discounted_sums=True,
            ),
            nn.BatchNorm2d(num_features=latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=latent_dim,
                out_channels=2 * latent_dim,
                kernel_size=1,
                stride=1,
            ),
            nn.BatchNorm2d(num_features=2 * latent_dim),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=2 * latent_dim,
                out_channels=in_channels,
                kernel_size=1,
                stride=1,
            ),
        )

    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)
        return x


class ConvAutoEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=2 * latent_dim,
                kernel_size=1,
                stride=1,
            ),
            nn.BatchNorm2d(num_features=2 * latent_dim),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=2 * latent_dim,
                out_channels=latent_dim,
                kernel_size=1,
                stride=1,
            ),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=latent_dim,
                out_channels=2 * latent_dim,
                kernel_size=1,
                stride=1,
            ),
            nn.BatchNorm2d(num_features=2 * latent_dim),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=2 * latent_dim,
                out_channels=in_channels,
                kernel_size=1,
                stride=1,
            ),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def build_model(
    model_mode: str,
    data_name: str,
    resnet_mode: str,
    pool_mode: str,
    num_nodes: int,
    semiring: str,
    seed: int | None = None,
    tree_type: str = TreeType.RANDOM,
) -> nn.Module:
    return FISResNet(
        model_mode=model_mode,
        data_name=data_name,
        resnet_mode=resnet_mode,
        pool_mode=pool_mode,
        num_nodes=num_nodes,
        seed=seed,
        semiring=semiring,
        tree_type=tree_type,
    )


def build_optimizer(
    model: nn.Module, num_epochs: int
) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.1,
        momentum=0.9,
        dampening=0,
        weight_decay=0.0005,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=num_epochs, eta_min=0
    )

    return optimizer, scheduler


def build_feature_extractor(
    backbone: str = AEBackbone.WIDE_RESNET50,
) -> FeatureExtractor:
    fe = FeatureExtractor(backbone=backbone)

    return fe


def build_autoencoder(
    in_channels: int,
    latent_dim: int,
    num_nodes: int,
    semiring: str,
    seed: int | None = None,
    tree_type: str = TreeType.RANDOM,
) -> FISAutoEncoder:
    ae = FISAutoEncoder(
        in_channels=in_channels,
        latent_dim=latent_dim,
        num_nodes=num_nodes,
        semiring=semiring,
        seed=seed,
        tree_type=tree_type,
    )

    return ae


def build_ablation_autoencoder(
    in_channels: int,
    latent_dim: int,
) -> ConvAutoEncoder:
    ae = ConvAutoEncoder(
        in_channels=in_channels,
        latent_dim=latent_dim,
    )

    return ae
