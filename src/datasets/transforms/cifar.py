import torchvision.transforms as transforms

from .randaugment import CIFAR10Policy, Cutout, RandAugment

MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
DEFAULT_IMAGE_SIZE: int = 32


class JuliusTransform:
    def __init__(self, *args, image_size: int = DEFAULT_IMAGE_SIZE, **kwargs):
        super(JuliusTransform, self).__init__()
        self.image_size = image_size

    def __call__(self):
        return transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(DEFAULT_IMAGE_SIZE, padding=4),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
            ]
        )


class BaselineTransform:
    def __init__(self, *args, image_size: int = DEFAULT_IMAGE_SIZE, **kwargs):
        super(BaselineTransform, self).__init__()
        self.image_size = image_size

    def __call__(self):
        return transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.RandomCrop(self.image_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        )


class BaselineCutoutTransform:
    def __init__(
        self, cutout_size: int, *args, image_size: int = DEFAULT_IMAGE_SIZE, **kwargs
    ):
        super(BaselineCutoutTransform, self).__init__()
        self.cutout_size = cutout_size
        self.image_size = image_size

    def __call__(self):
        return transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.RandomCrop(self.image_size, padding=4),
                transforms.RandomHorizontalFlip(),
                Cutout(size=self.cutout_size),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        )


class AutoAugmentTransform:
    def __init__(
        self, cutout_size: int, *args, image_size: int = DEFAULT_IMAGE_SIZE, **kwargs
    ):
        super(AutoAugmentTransform, self).__init__()
        self.cutout_size = cutout_size
        self.image_size = image_size

    def __call__(self):
        return transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.RandomCrop(self.image_size, padding=4),
                transforms.RandomHorizontalFlip(),
                CIFAR10Policy(),
                Cutout(size=self.cutout_size),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
                # Random Erase with p=1 is an alternative to Cutout but worked slightly worse
                # transforms.RandomErasing(p=1,
                #                        scale=(0.125, 0.2), # range for how big the cutout should be compared to original img
                #                        ratio=(0.99, 1.0), # squares
                #                        value=0, inplace=False)
            ]
        )


class RandAugmentTransform:
    def __init__(
        self, cutout_size: int, *args, image_size: int = DEFAULT_IMAGE_SIZE, **kwargs
    ):
        super(RandAugmentTransform, self).__init__()
        self.cutout_size = cutout_size
        self.image_size = image_size

    def __call__(self):
        return transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.RandomCrop(self.image_size, padding=4),
                transforms.RandomHorizontalFlip(),
                RandAugment(),
                Cutout(size=self.cutout_size),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
                # Random Erase with p=1 is an alternative to Cutout but worked slightly worse
                # transforms.RandomErasing(p=1,
                #                        scale=(0.125, 0.2), # range for how big the cutout should be compared to original img
                #                        ratio=(0.99, 1.0), # squares
                #                        value=0, inplace=False)
            ]
        )


# FIXME: albumentation does not work for now with the torchvision dataset; thus commented out
# class AlbumAugmentTransform:
#     def __init__(self, *args, image_size: int = DEFAULT_IMAGE_SIZE, **kwargs):
#         super(AlbumAugmentTransform, self).__init__()
#         self.image_size = image_size
#
#     def __call__(self):
#         return A.Compose(
#             [
#                 A.Resize(height=self.image_size, width=self.image_size),
#                 A.InvertImg(always_apply=False, p=0.2),
#                 A.PadIfNeeded(
#                     always_apply=False,
#                     p=0.2,
#                     min_height=36,
#                     min_width=36,
#                     pad_height_divisor=None,
#                     pad_width_divisor=None,
#                     border_mode=4,
#                     value=None,
#                     mask_value=None,
#                 ),
#                 A.RandomCrop(
#                     always_apply=1, p=0.2, height=self.image_size, width=self.image_size
#                 ),
#                 A.HorizontalFlip(always_apply=False, p=0.2),
#                 A.RandomBrightnessContrast(always_apply=False, p=0.2),
#                 A.ShiftScaleRotate(
#                     always_apply=False,
#                     p=0.2,
#                     shift_limit_x=(-0.2, 0.2),
#                     shift_limit_y=(-0.2, 0.2),
#                     scale_limit=(0.0, 0.0),
#                     rotate_limit=(0, 0),
#                     interpolation=1,
#                     border_mode=4,
#                     value=None,
#                     mask_value=None,
#                 ),
#                 A.Equalize(always_apply=False, p=0.2, mode="cv", by_channels=True),
#                 A.Solarize(always_apply=False, p=0.2, threshold=(128, 128)),
#                 A.Normalize(MEAN, STD),
#                 ToTensorV2(),
#             ]
#         )
#


class TestTransform:
    def __init__(self, *args, image_size: int = DEFAULT_IMAGE_SIZE, **kwargs):
        super(TestTransform, self).__init__()
        self.image_size = image_size

    def __call__(self):
        return transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        )


class JuliusTestTransform:
    def __init__(self, *args, image_size: int = DEFAULT_IMAGE_SIZE, **kwargs):
        super(JuliusTestTransform, self).__init__()
        self.image_size = image_size

    def __call__(self):
        return transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
            ]
        )
