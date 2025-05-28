from enum import IntEnum, StrEnum, auto
from types import MappingProxyType

import torch


class TreeType(StrEnum):
    RANDOM = auto()
    LINEAR = auto()
    LINEAR_NW = "linear_NW"


class ResultDir(StrEnum):
    PLOTS = "./plots"
    MODELS = "./models"
    HISTORY = "./history"
    WANDB = "./wandb"


class AEBackbone(StrEnum):
    RESNET50 = auto()
    WIDE_RESNET50 = auto()
    RESNET101 = auto()
    WIDE_RESNET101 = auto()


class Semiring(StrEnum):
    REAL = auto()
    MAXPLUS = auto()


class SelectMode(StrEnum):
    ALL_RANDOM = auto()
    ALL_TRAIN = auto()


class TreeDirection(StrEnum):
    NE = "NE"
    SE = "SE"
    NW = "NW"
    SW = "SW"
    N = "N"
    S = "S"
    W = "W"
    E = "E"


type Tree = list[float | int | torch.Tensor | list[tuple[str, list]]]


class DocType(StrEnum):
    THESIS = auto()
    BEAMER = auto()


class ParamInit(StrEnum):
    NORMAL = auto()
    UNIFORM = auto()
    ONES = auto()


class MVTECCategory(StrEnum):
    CARPET = auto()
    GRID = auto()
    LEATHER = auto()
    TILE = auto()
    WOOD = auto()


CIFAR_CATEGORY = MappingProxyType(
    dict(
        airplane=0,
        automobile=1,
        bird=2,
        cat=3,
        deer=4,
        dog=5,
        frog=6,
        horse=7,
        ship=8,
        truck=9,
    )
)


class ModelMode(StrEnum):
    CA = auto()
    CA_FISBLOCK = auto()
    L2 = auto()
    L23 = auto()
    L3 = auto()
    DOWNSAMPLE = auto()
    BASE = auto()


class PoolMode(StrEnum):
    AVG = auto()
    MAX = auto()


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


class Pipeline(StrEnum):
    TRAIN = auto()
    HPSEARCH = auto()
    CLEAN = auto()
    COMPARE = auto()
    RANDOM = auto()
    ANALYSE = auto()


class DataName(StrEnum):
    CIFAR10 = auto()
    CIFAR100 = auto()


COLOUR_MAP = MappingProxyType(
    dict(
        zip(
            ModelMode,
            [
                "black",
                "cyan",
                "fuchsia",
                "tab:purple",
            ],
        )
    )
)

LINE_MAP = MappingProxyType(
    dict(
        zip(
            ModelMode,
            [
                "-",
                ":",
                "-.",
                "--",
            ],
        )
    )
)

# see https://github.com/chenyaofo/pytorch-cifar-models
BASE_METRICS = MappingProxyType(
    {
        "cifar10": {
            "resnet20": {
                "accuracy@1": 92.60,
                "accuracy@5": 99.81,
                "total_params": 0.27,
                "total_mult_adds": 40.81,
            },
            "resnet32": {
                "accuracy@1": 93.53,
                "accuracy@5": 99.77,
                "total_params": 0.47,
                "total_mult_adds": 69.12,
            },
            "resnet44": {
                "accuracy@1": 94.01,
                "accuracy@5": 99.77,
                "total_params": 0.66,
                "total_mult_adds": 97.44,
            },
            "resnet56": {
                "accuracy@1": 94.37,
                "accuracy@5": 99.83,
                "total_params": 0.86,
                "total_mult_adds": 125.75,
            },
        },
        "cifar100": {
            "resnet20": {
                "accuracy@1": 68.83,
                "accuracy@5": 91.01,
                "total_params": 0.28,
                "total_mult_adds": 40.82,
            },
            "resnet32": {
                "accuracy@1": 70.16,
                "accuracy@5": 90.89,
                "total_params": 0.47,
                "total_mult_adds": 69.13,
            },
            "resnet44": {
                "accuracy@1": 71.63,
                "accuracy@5": 91.58,
                "total_params": 0.67,
                "total_mult_adds": 97.44,
            },
            "resnet56": {
                "accuracy@1": 72.63,
                "accuracy@5": 91.94,
                "total_params": 0.86,
                "total_mult_adds": 125.75,
            },
        },
    }
)
