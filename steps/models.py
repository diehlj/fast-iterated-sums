import torch
from torch import nn
from torchvision import models

from shared.builders import create_forest
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
from shared.sums import sum_trees


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
            tree_type=tree_type,
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
            FISLayer(
                in_channels=in_channels,
                num_trees=2 * latent_dim,
                num_nodes=num_nodes,
                semiring=semiring,
                seed=seed,
                tree_type=tree_type,
            ),
            nn.BatchNorm2d(num_features=2 * latent_dim),
            nn.ReLU(),
            FISLayer(
                in_channels=2 * latent_dim,
                num_trees=latent_dim,
                num_nodes=num_nodes,
                semiring=semiring,
                seed=seed,
                tree_type=tree_type,
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
