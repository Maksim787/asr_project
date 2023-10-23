import random

from torch import Tensor


class AugmentationBase:
    """
    Class for applying any augmentation randomly
    """

    def __init__(self, p: float) -> None:
        assert 0 <= p <= 1
        self.p = p

    def __call__(self, data: Tensor) -> tuple[Tensor, list[str]]:
        if random.random() < self.p:
            return self.forward(data)
        else:
            return data, []

    def forward(self, data: Tensor) -> tuple[Tensor, list[str]]:
        raise NotImplementedError()
