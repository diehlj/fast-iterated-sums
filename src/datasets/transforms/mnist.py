import torchvision.transforms as transforms

from .randaugment import Cutout, RandAugment

# MNIST normalization values (grayscale)
MEAN, STD = (0.1307,), (0.3081,)
DEFAULT_IMAGE_SIZE: int = 28


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
