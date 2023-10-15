from typing import List, Callable

from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase


class SequentialAugmentation(AugmentationBase):
    def __init__(self, augmentation_list: List[Callable]):
        self.augmentation_list = augmentation_list

    def __call__(self, data: Tensor) -> tuple[Tensor, list[str]]:
        x = data
        aug_names = []
        for augmentation in self.augmentation_list:
            x, names = augmentation(x)
            aug_names.extend(names)
        return x, aug_names
