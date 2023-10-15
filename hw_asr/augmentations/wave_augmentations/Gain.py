import torch_audiomentations
from torch import Tensor

from hw_asr.augmentations.base import AugmentationBase


class Gain(AugmentationBase):
    def __init__(self, p, *args, **kwargs):
        super().__init__(p)
        self._aug = torch_audiomentations.Gain(*args, **kwargs, p=1)

    def forward(self, data: Tensor):
        x = data.unsqueeze(1)
        return self._aug(x).squeeze(1), ['Gain']
